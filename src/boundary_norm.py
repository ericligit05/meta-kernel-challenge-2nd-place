"""Layer boundary kernel (nr2_k in kernels/boundary_norm.cu): FusedPostNormResidualPreNorm's math as one kernel.

The math is the post-norm of the branch, the gated residual add and the pre-norm of the new residual. nr_params(op)
turns a prepared FusedPostNormResidualPreNorm into the tensors the kernel reads; boundary() runs it. layer.py runs
boundary() at both boundaries for more than 8 tokens wherever the fused pair boundary_attn_remat.py /
boundary_ffn_remat.py does not apply (below 2048 tokens, or parameters it does not support), and passes nr_params'
tensors to that pair as well.
"""

import torch
from torch.utils.cpp_extension import load_inline
from cuda_source import cuda_source

_NR_SRC = cuda_source("boundary_norm.cu")
_nr = load_inline(name="okc_nr2_z06930", cuda_sources=_NR_SRC,
                  cpp_sources="void nr2_run(torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, "
                              "torch::Tensor, torch::Tensor, torch::Tensor, double, double, int64_t, int64_t);",
                  functions=["nr2_run"], extra_cuda_cflags=["-O3", "-gencode=arch=compute_90a,code=sm_90a"])


def nr_params(op):
    """The tensors and constants FusedPostNormResidualPreNorm's kernels read, or None where this kernel does not apply
    (a scalar residual gate or a pre-norm without weight)."""
    op._ensure_cached_weights(torch.bfloat16)
    if not (op._use_residual_gate and op._cc_alpha.numel() == 8192 and op._pre_norm.has_weight
            and op._hidden_size == 8192):
        return None
    return {"postw": op._cc_post_weight.contiguous(), "prew": op._cc_pre_weight.contiguous(),
            "alpha": op._cc_alpha.contiguous(), "beta": op._cc_beta.contiguous(),
            "post_eps": float(op._post_norm.eps), "pre_eps": float(op._pre_norm.eps),
            "scale": 1 if op._post_norm_is_input_scale else 0}


def boundary(p, resid, branch, mode=None, pdl=True):
    """(BF16 pre-normed output, new FP32 residual) for a [T, 8192] branch (BF16) and residual (FP32).

    mode overrides p["scale"]: 0 plain post-norm, 1 post-norm with input scale, 9 copy only (no math)."""
    out = torch.empty_like(branch)
    new = torch.empty_like(resid)
    _nr.nr2_run(branch, resid, out, new, p["postw"], p["prew"], p["alpha"], p["beta"], p["post_eps"], p["pre_eps"],
                p["scale"] if mode is None else mode, 1 if pdl else 0)
    return out, new
