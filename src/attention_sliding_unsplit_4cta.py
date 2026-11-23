"""Sliding-window prefill attention, unsplit build at 4 CTAs per SM (kernels/attention_sliding_unsplit_4cta.cu).

The kernel of attention_sliding_unsplit.py compiled with no producer warp (thread 0 of the one warpgroup issues the TMA
loads), 128 threads per CTA and no shared memory alignment slack, so 4 CTAs fit on an SM. Same numerics: FP8 E4M3 q / K
/ V, exp-free FP8 softmax codes, FP16 P V accumulation, no row sum, o_norm_gate fused in the epilogue. layer.py calls
attn_ng for unsplit sliding calls of at least SA_X4_MIN tokens, on the K / V copies built by
attention_sliding_split.gather. This build has no q_lo term: attn_ng2 raises.
"""

import torch
from torch.utils.cpp_extension import load_inline
from cuda_source import cuda_source

SK_CPP = 'void sl_gather(torch::Tensor kc, torch::Tensor vc, torch::Tensor pt, torch::Tensor seqlens, torch::Tensor cu_q, int64_t win, torch::Tensor ko, torch::Tensor vt);\nvoid sl_attn(torch::Tensor q, torch::Tensor ko, torch::Tensor vt, torch::Tensor cache_seqlens, torch::Tensor cu_q, int64_t max_q, int64_t window, double softmax_scale, torch::Tensor out);\nvoid sl_ts(torch::Tensor buf);\nvoid sl_pvstat(torch::Tensor buf);\nvoid sl_gather2(torch::Tensor kc, torch::Tensor vc, torch::Tensor pt, torch::Tensor seqlens, torch::Tensor cu_q, int64_t win, int64_t part, torch::Tensor ko, torch::Tensor vt);\nvoid sl_attn_ng(torch::Tensor q, torch::Tensor ko, torch::Tensor vt, torch::Tensor cache_seqlens, torch::Tensor cu_q, int64_t max_q, int64_t window, double softmax_scale, torch::Tensor gate, double eps, torch::Tensor out);\nvoid sl_attn_ng2(torch::Tensor q, torch::Tensor qlo, torch::Tensor ko, torch::Tensor vt, torch::Tensor cache_seqlens, torch::Tensor cu_q, int64_t max_q, int64_t window, double softmax_scale, torch::Tensor gate, double eps, torch::Tensor out);\n'
SK_FLAGS = ['-O3', '--use_fast_math', '-DNDEBUG', '-U__CUDA_NO_HALF_OPERATORS__', '-U__CUDA_NO_HALF_CONVERSIONS__', '-U__CUDA_NO_BFLOAT16_CONVERSIONS__', '-U__CUDA_NO_HALF2_OPERATORS__', '-gencode=arch=compute_90a,code=sm_90a']
SK_SRC = cuda_source("attention_sliding_unsplit_4cta.cu")

sk_ext = load_inline(name="okc_prefill_p9033", cpp_sources=SK_CPP, cuda_sources=SK_SRC,
                     functions=["sl_gather", "sl_attn", "sl_ts", "sl_pvstat", "sl_gather2", "sl_attn_ng", "sl_attn_ng2"],
                     extra_cuda_cflags=SK_FLAGS, extra_ldflags=["-lcuda"])


def alloc(cu_seqlens_q, max_q, window, device):
    """Empty K / V^T copies for one call: S = max_q + window - 1 rows rounded up to 64."""
    B = cu_seqlens_q.shape[0] - 1
    S = (max_q + window - 1 + 63) // 64 * 64
    return (torch.empty((B, S, 2, 64), dtype=torch.uint8, device=device),
            torch.empty((B, 2, 64, S), dtype=torch.uint8, device=device))


def gather(k_cache, v_cache, page_table, cache_seqlens, cu_seqlens_q, window, part, ko, vt):
    """Copy each request's window rows from the paged cache into ko / vt (from alloc). part 0: only the blocks of
    rows cached before this call; part 1: the other blocks; part 2: all."""
    sk_ext.sl_gather2(k_cache.view(torch.uint8), v_cache.view(torch.uint8), page_table, cache_seqlens, cu_seqlens_q,
                      window - 1, part, ko, vt)


def attn(q8, ko, vt, cache_seqlens, cu_seqlens_q, max_q, window, softmax_scale, out):
    """Attention into out [T, 16, 64] BF16, unnormalized: each row is off by a positive factor (there is no softmax
    denominator), so this entry point is only correct ahead of a per-head RMSNorm. The layer uses attn_ng."""
    sk_ext.sl_attn(q8.view(torch.uint8), ko, vt, cache_seqlens, cu_seqlens_q, max_q, window, softmax_scale, out)


def attn_ng(q8, ko, vt, cache_seqlens, cu_seqlens_q, max_q, window, softmax_scale, gate, eps, out):
    """out [T, 1024] BF16 = o_norm_gate(attention, gate); gate is a [T, 1024] BF16 view with unit inner stride. The
    per-head RMSNorm of o_norm_gate cancels the missing softmax denominator."""
    sk_ext.sl_attn_ng(q8.view(torch.uint8), ko, vt, cache_seqlens, cu_seqlens_q, max_q, window, softmax_scale, gate,
                      eps, out)


def attn_ng2(q8, qlo, ko, vt, cache_seqlens, cu_seqlens_q, max_q, window, softmax_scale, gate, eps, out):
    """Not available in this build: the kernel binding raises (no q_lo term). Kept so the three sliding modules share
    one interface."""
    sk_ext.sl_attn_ng2(q8.view(torch.uint8), qlo.view(torch.uint8), ko, vt, cache_seqlens, cu_seqlens_q, max_q, window,
                       softmax_scale, gate, eps, out)
