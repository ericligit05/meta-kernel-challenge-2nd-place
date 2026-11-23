"""Full-attention layers: a FlashAttention-3 Hopper forward kernel built from source (kernels/attention_full.cu).

The kernel is FA3's SM90 forward for FP8 E4M3 q / K / V, head dim 64, 16 q heads on 2 KV heads (GQA packed), varlen
queries, causal, compiled inside the submission so its softmax could be replaced: S is accumulated in FP16, P is made
exp-free (one FMA turns two FP16 scores into two E4M3 codes), P V accumulates in FP16 and there is no row sum. The
output therefore carries an unknown positive factor per row, which the per-head RMSNorm of o_norm_gate cancels.

layer.py uses three entry points per full-attention call of at least AT_FULL_MIN tokens: front() (QK-norm, FP8 q / k
/ v, cache write), gather() (contiguous K and a prepared V image from the paged cache) and attn_g() (attention with
o_norm_gate fused into the epilogue or the split combine when the call allows it).
"""

import os

import torch
from torch.utils.cpp_extension import load_inline
from cuda_source import cuda_source

ATT_CPP = ("torch::Tensor att_fwd(torch::Tensor q, torch::Tensor k, torch::Tensor v, torch::Tensor page_table, "
           "torch::Tensor cache_seqlens, torch::Tensor cu_seqlens_q, int64_t max_seqlen_q, double softmax_scale, "
           "int64_t window_left, int64_t num_splits_req, torch::Tensor out, torch::Tensor gate, double gate_eps);\n"
           "void att_gather(torch::Tensor kc, torch::Tensor vc, torch::Tensor pt, torch::Tensor seqlens, "
           "torch::Tensor cu_q, int64_t max_q, int64_t part, torch::Tensor ko, torch::Tensor vo);\n"
           "void att_front(torch::Tensor qkv, torch::Tensor loc, torch::Tensor qo, torch::Tensor kc, torch::Tensor vc, double eps, double qmul);\n"
           "void att_gather_t(torch::Tensor kc, torch::Tensor vc, torch::Tensor pt, torch::Tensor seqlens, "
           "torch::Tensor cu_q, int64_t win, int64_t part, torch::Tensor ko, torch::Tensor vt);")
ATT_FLAGS = ["-O3", "--use_fast_math", "--expt-relaxed-constexpr", "--expt-extended-lambda", "-DNDEBUG",
             "-DCUTE_USE_PACKED_TUPLE=1", "-DCUTLASS_ENABLE_GDC_FOR_SM90", "-DCUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED",
             "-DCUTLASS_ENABLE_TENSOR_CORE_MMA=1", "-U__CUDA_NO_HALF_OPERATORS__", "-U__CUDA_NO_HALF_CONVERSIONS__",
             "-U__CUDA_NO_BFLOAT16_CONVERSIONS__", "-U__CUDA_NO_HALF2_OPERATORS__",
             "-gencode=arch=compute_90a,code=sm_90a"]
ATT_SRC = cuda_source("attention_full.cu")


def _cutlass_include():
    import flashinfer
    return os.path.join(os.path.dirname(flashinfer.__file__), "data", "cutlass", "include")


att_ext = load_inline(name="okc_att_z14643", cpp_sources=ATT_CPP, cuda_sources=ATT_SRC,
                      functions=["att_fwd", "att_gather", "att_gather_t", "att_front"], extra_include_paths=[_cutlass_include()],
                      extra_cuda_cflags=ATT_FLAGS)
EMPTY_PT = {}


def attn(q, k_cache, v_cache, page_table, cache_seqlens, cu_seqlens_q, max_seqlen_q, softmax_scale, window_left, out):
    """FA3 forward into out [T, 16, 64] BF16 (unnormalized rows, see the module docstring). page_table must be None:
    only the contiguous variants are compiled, and k_cache / v_cache must come from gather()."""
    if page_table is None:
        page_table = EMPTY_PT.get(q.device)
        if page_table is None:
            page_table = EMPTY_PT[q.device] = torch.empty(0, dtype=torch.int32, device=q.device)
    return att_ext.att_fwd(q, k_cache, v_cache, page_table, cache_seqlens, cu_seqlens_q, max_seqlen_q, softmax_scale,
                           window_left, 0, out, _empty_gate(q.device), 0.0)


EMPTY_GATE = {}


def _empty_gate(dev):
    g = EMPTY_GATE.get(dev)
    if g is None:
        g = EMPTY_GATE[dev] = torch.empty(0, dtype=torch.bfloat16, device=dev)
    return g


def attn_g(q, k_cache, v_cache, page_table, cache_seqlens, cu_seqlens_q, max_seqlen_q, softmax_scale, window_left, out,
           gate, eps):
    """attn() with o_norm_gate fused where the call allows it: in the attention epilogue when the call is not split,
    in the combine kernel when it is split by FA3's heuristic. Returns True when out already holds
    o_norm_gate(attention, gate), else False: the equal-work split (long-context calls with at least 4 waves of
    tiles and keys >= 2 x queries) merges in-kernel and leaves the unnormalized attention in out for the caller's
    o_norm_gate."""
    if page_table is None:
        page_table = EMPTY_PT.get(q.device)
        if page_table is None:
            page_table = EMPTY_PT[q.device] = torch.empty(0, dtype=torch.int32, device=q.device)
    r = att_ext.att_fwd(q, k_cache, v_cache, page_table, cache_seqlens, cu_seqlens_q, max_seqlen_q, softmax_scale,
                        window_left, 0, out, gate, eps)
    return r.numel() > 0


def gather(k_cache, v_cache, page_table, cache_seqlens, cu_seqlens_q, max_q, part, ko=None, vo=None):
    """Copies of the cached rows for attn(): K contiguous [B, S, hkv, d] and V as one prepared image per (request,
    KV head, 256-key block), 72 x 256 bytes each (V^T in the layout of FA3's PV operand, the last 8 rows padding),
    stored in a [B, S, hkv, 72] buffer. S = pages * page rounded up to 256; rows past a request's keys are zero.
    Allocated when ko / vo are None. part must be 2 (all rows) or a kernel select in 20..26; the kernel rejects
    other values."""
    if ko is None:
        B, pages = page_table.shape
        S = (pages * k_cache.shape[1] + 255) // 256 * 256
        shape = (B, S) + tuple(k_cache.shape[2:])
        ko = torch.empty(shape, dtype=k_cache.dtype, device=k_cache.device)
        vo = torch.empty((B, S, k_cache.shape[2], 72), dtype=v_cache.dtype, device=v_cache.device)
    att_ext.att_gather(k_cache.view(torch.uint8), v_cache.view(torch.uint8), page_table, cache_seqlens, cu_seqlens_q,
                       max_q, part, ko.view(torch.uint8), vo.view(torch.uint8))
    return ko, vo


def gather_window(k_cache, v_cache, page_table, cache_seqlens, cu_seqlens_q, max_q, window):
    """Window copy for a sliding layer (part 3). Not usable in this build: att_gather accepts only part 2 or 20..26
    and raises on part 3. layer.py runs sliding layers on attention_sliding*.py instead."""
    B = page_table.shape[0]
    shape = (B, max_q + window - 1) + tuple(k_cache.shape[2:])
    ko = torch.empty(shape, dtype=k_cache.dtype, device=k_cache.device)
    vo = torch.empty(shape, dtype=v_cache.dtype, device=v_cache.device)
    att_ext.att_gather(k_cache.view(torch.uint8), v_cache.view(torch.uint8), page_table, cache_seqlens, cu_seqlens_q,
                       window - 1, 3, ko.view(torch.uint8), vo.view(torch.uint8))
    seqlens = torch.minimum(cache_seqlens, cu_seqlens_q[1:] - cu_seqlens_q[:-1] + (window - 1)).to(torch.int32)
    return ko, vo, seqlens


def gather_t(k_cache, v_cache, page_table, cache_seqlens, cu_seqlens_q, max_q, window=0):
    """K contiguous [B, S, hkv, d] and V column-major (a [B, S, hkv, d] view of [B, hkv, d, S] storage), S padded to
    64 rows. window > 0: the last L_b + window - 1 rows of each request from row 0 and their sequence lengths;
    window 0: every row (seqlens unchanged). Not used by layer.py: attn() here reads V only in gather()'s image
    layout, so this output is not a valid input for it."""
    B, pages = page_table.shape
    S = max_q + window - 1 if window else pages * k_cache.shape[1]
    S = (S + 63) // 64 * 64
    hkv, d = k_cache.shape[2], k_cache.shape[3]
    ko = torch.empty((B, S, hkv, d), dtype=k_cache.dtype, device=k_cache.device)
    vt = torch.empty((B, hkv, d, S), dtype=v_cache.dtype, device=v_cache.device)
    att_ext.att_gather_t(k_cache.view(torch.uint8), v_cache.view(torch.uint8), page_table, cache_seqlens, cu_seqlens_q,
                         window - 1 if window else 0, 3 if window else 2, ko.view(torch.uint8), vt.view(torch.uint8))
    vo = vt.permute(0, 3, 1, 2)
    if not window:
        return ko, vo, cache_seqlens
    seqlens = torch.minimum(cache_seqlens, cu_seqlens_q[1:] - cu_seqlens_q[:-1] + (window - 1)).to(torch.int32)
    return ko, vo, seqlens


# The kernel's softmax (AT_SUB) wants S in log2 units: front() writes q prescaled by QMUL x softmax_scale, with
# QMUL = log2(e) (FP8 of the BF16-rounded normed q times that, one rounding), so the QK GEMM gives
# S = log2(e) softmax_scale q.k directly. attn / attn_g take the softmax scale divided by QMUL, which keeps FA3's own
# log2(e) bookkeeping consistent.
QMUL = 1.4426950408889634


def front(qkv, cache_loc, q8, k_cache_u8, v_cache_u8, eps, softmax_scale):
    """Full-layer front kernel: RMS-normalizes each q and k head (no weight, no RoPE: layer.py takes this path only
    when rope_theta is 0), writes q as FP8 of q x QMUL x softmax_scale into q8 and FP8 k / v into the cache rows at
    cache_loc. Returns 1 / QMUL, the factor the caller applies to the softmax scale it passes to attn / attn_g."""
    att_ext.att_front(qkv, cache_loc, q8, k_cache_u8, v_cache_u8, eps, QMUL * softmax_scale)
    return 1.0 / QMUL
