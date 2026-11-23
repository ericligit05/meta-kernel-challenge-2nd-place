"""Post-expert RMSnorm and weighted sum of a token's routed expert outputs (wsum3_k in kernels/weighted_sum.cu).

layer.py passes wsum() into the MoE block (moe_block.py), which launches it as a programmatic dependent of the
expert kernel. One CTA per token: each CTA waits only for the experts its own routes use (per-expert counters of
stored GEMM2 tiles, cnt2), so a token whose experts finish early is summed while the expert kernel still runs.
The prologue loads (routes, slots, weights) are issued together, warp 0 checks every route's expert at once, the
rows of routes whose experts are already complete are loaded into registers before waiting for the rest, and after
the wait only the late rows are loaded. Up to WS_NB routes per token are held in registers; further routes (rare)
are read again from memory. The sum runs in ascending route order, so the output does not depend on timing.
"""

import torch
from torch.utils.cpp_extension import load_inline
from cuda_source import cuda_source

# The kernel reads these as -D flags (set by hand in extra_cuda_cflags below, keep them equal).
WS_MINB = 4  # CTAs per SM the register budget aims at (__launch_bounds__)
WS_NB = 4    # routes per token held in registers

_SRC = cuda_source("weighted_sum.cu")
_ws = load_inline(name="okc_prefill_wsum_z08492", cuda_sources=_SRC,
                  cpp_sources="void wsum_run(torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, "
                              "torch::Tensor, torch::Tensor, double, int64_t);",
                  functions=["wsum_run"],
                  extra_cuda_cflags=["-O3", "-gencode=arch=compute_90a,code=sm_90a", "-DWS_MINB=4", "-DWS_NB=4"])


def wsum(y, tw, slot, w, meta, cnt2, T, eps, pdl=True):
    """[T, 4096] BF16 weighted sum of the normalized local expert outputs y [T * 8, 4096] (slot ids < 0 skipped)."""
    out = torch.empty((T, 4096), dtype=torch.bfloat16, device=y.device)
    _ws.wsum_run(y, tw, slot, w, meta, cnt2, out, eps, 1 if pdl else 0)
    return out
