"""Layer boundaries without the mid-layer FP32 residual (nr3_k in kernels/boundary_attn_remat.cu).

The attention boundary writes only the FFN input and each row's post-norm scale; the FFN boundary recomputes the
attention residual from the layer's input stream and the attention branch, bit-identical to writing it. layer.py
calls boundary_a at the attention boundary on the 2048+ token path, optionally with the token order's inverse map
(token t's outputs go to row inv[t]); the matching FFN boundary there is boundary_ffn_remat.py. boundary_b is the
plain FFN-side kernel of this file and is not used by layer.py.

boundary_a(pa, stream, attn_out) -> (ffn_input, rps); boundary_b(pa, pb, stream, attn_out, ffn_out, rps) ->
(next input, resid).
"""

import torch
from torch.utils.cpp_extension import load_inline
from cuda_source import cuda_source

_NR3_SRC = cuda_source("boundary_attn_remat.cu")
_nr3 = load_inline(name="okc_nr3_z07469", cuda_sources=_NR3_SRC,
                   cpp_sources="void nr3_run(torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, "
                               "torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, double, double, int64_t, "
                               "torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, double, double, int64_t, int64_t, torch::Tensor, int64_t, torch::Tensor);",
                   functions=["nr3_run"], extra_cuda_cflags=["-O3", "-gencode=arch=compute_90a,code=sm_90a"])


_NOPERM = {}


def _noperm(dev):
    if dev not in _NOPERM:
        _NOPERM[dev] = torch.empty(0, dtype=torch.int32, device=dev)
    return _NOPERM[dev]


def boundary_a(pa, resid, attn_out, pdl=True, perm=None):
    """(FFN input BF16, per-row post-norm scale of the attention branch) after the attention boundary; the residual
    itself is not written. perm (int32 [T] or None): here the INVERSE map, token t's outputs go to row perm[t]."""
    out = torch.empty_like(attn_out)
    rps = torch.empty(attn_out.shape[0], dtype=torch.float32, device=attn_out.device)
    _nr3.nr3_run(resid, attn_out, attn_out, out, resid, pa["postw"], pa["prew"], pa["alpha"], pa["beta"],
                 pa["post_eps"], pa["pre_eps"], pa["scale"], pa["postw"], pa["prew"], pa["alpha"], pa["beta"],
                 pa["post_eps"], pa["pre_eps"], pa["scale"], 0, rps, 1 if pdl else 0,
                 _noperm(attn_out.device) if perm is None else perm)
    return out, rps


def boundary_b(pa, pb, resid, attn_out, ffn_out, rps):
    """(next attention input BF16, new FP32 residual): the attention residual recomputed, then the FFN boundary."""
    out = torch.empty_like(ffn_out)
    new = torch.empty_like(resid)
    _nr3.nr3_run(resid, attn_out, ffn_out, out, new, pa["postw"], pa["prew"], pa["alpha"], pa["beta"],
                 pa["post_eps"], pa["pre_eps"], pa["scale"], pb["postw"], pb["prew"], pb["alpha"], pb["beta"],
                 pb["post_eps"], pb["pre_eps"], pb["scale"], 1, rps, 1, _noperm(ffn_out.device))
    return out, new
