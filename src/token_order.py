"""Token similarity order (row map) from the residual stream's first KC columns, one CTA of 1024 threads.

order(stream) -> (perm, inv), int32 [T]: row i of the post-attention pipeline holds token perm[i]; token t sits in row
inv[t]. One kernel: KS evenly spaced sample rows give the column mean and the KC x KC covariance; subspace iteration
starting from a fixed start (Gram-Schmidt) gives 4 leading directions and their variances; every token's coordinates
z_j = (x - mean) . v_j get 8 levels u = Phi(z_j / sd_j) and a 12-bit Morton key (shared-memory histogram); an
exclusive scan of the 4096 counts and a shared-memory cursor per key place the tokens. Tokens with the same key land
in any order (atomics); every downstream kernel is row-independent, so outputs are identical per token for any order.
Nothing here depends on the benchmark: the directions come from the call's own stream.

layer.py runs the order on a side queue for 2048+ tokens on the rematerialized-boundary path; the attention boundary
(boundary_attn_remat.py) writes rows in this order and boundary_ffn_remat.py maps the results back to token order.
The kernel runs 3 subspace iterations: the stream's leading directions stand far above the rest, so 3 steps are
enough for an ordering key, and with more the kernel ended about 2 us after the qkv GEMM and delayed the sliding
attention that joins on it. The order only affects speed, never values.
"""
import torch
from torch.utils.cpp_extension import load_inline
from cuda_source import cuda_source

# The same values as the constexprs in kernels/token_order.cu.
KC = 32  # stream columns read per token
KS = 128  # sample rows for the directions
TMAX = 16384  # keys of up to TMAX tokens in shared memory (u16)

_SRC = cuda_source("token_order.cu")

_mod = load_inline(name="okc_order2_z02590", cuda_sources=_SRC,
                   cpp_sources="void order2_init(); void order2_run(torch::Tensor, torch::Tensor, torch::Tensor, int64_t);",
                   functions=["order2_init", "order2_run"], extra_cuda_cflags=["-O3", "-gencode=arch=compute_90a,code=sm_90a"],
                   verbose=False)
_init = []


def supported(T):
    return T <= TMAX


def order(resid, ident=False):
    """(perm, inv), int32 [T]: row i holds token perm[i], token t sits in row inv[t] (ident: same work, identity)."""
    if not _init:
        _mod.order2_init()
        _init.append(1)
    T = resid.shape[0]
    pi = torch.empty(2 * T, dtype=torch.int32, device=resid.device)
    perm, inv = pi[:T], pi[T:]
    _mod.order2_run(resid, perm, inv, 1 if ident else 0)
    return perm, inv
