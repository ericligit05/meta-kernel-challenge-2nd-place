# Reusing the code

The code is the submitted layer, kept token for token so that the rebuild from `src/` stays equal to the scored file.
We therefore document its limits here instead of adding argument checks, which would change the code. Read this before
calling any component outside the layer.

## Entry points

The layer's entry points are `prepare(weights)` and `kernel_fn(data, state)` in `src/layer.py`, with the tensors that
the `Weights`, `Config` and `Inputs` classes in `src/moe_block.py` describe. `prepare()` repacks the expert weights,
stacks the pooler and router weights and builds the RoPE table, so every component needs the `state` it returns.

The grouped NVFP4 expert GEMMs live in `src/kernels/moe_expert_gemm.cu`. The smallest call into them is
`_experts_moe(x, topk_weights, local_ids, state["expert_a"])` in `src/moe_block.py`, which sorts the routes by
expert, gathers their rows and runs both GEMMs and the weighted sum. Importing `src/moe_block.py` also imports SGLang,
Triton and QuACK and compiles the extensions, so it needs the full stack of [SETUP.md](SETUP.md).

## Input layout

`prepare()` takes the problem's NVFP4 weights for this GPU's experts:

| tensor | dtype and shape | content |
|---|---|---|
| `w13` | uint8 [32, 8192, 2048] | gate then up rows, two E2M1 codes per byte, element 2i in the low nibble (as `torch.float4_e2m1fn_x2` and the problem's `nvfp4.py`) |
| `w13_sf` | float8_e4m3fn [32, 8192, 256] | one block scale per 16 values along k |
| `w13_gs` | float32 [32] | one global scale per expert |
| `w2`, `w2_sf`, `w2_gs` | as above, with 4096 rows | the down projection |

`_pack_nvfp4` then repacks each expert losslessly into the 16-row by 64-k units that `decode4` reads, and the GEMMs
read only that repacked form.

## Fixed sizes

The host functions in `moe_expert_gemm.cu` check none of these, so other values read out of bounds instead of raising
an error.
- K is 4096 in both expert GEMMs, as a compile-time chunk count (`X63_C = 64` chunks of 64) and as the `* 4096`
  output strides.
- 32 local experts and top 8 routes per token, in the expert plan that the routing kernels of `routing_front.cu`
  write (`meta[0..31]` holds one route count per local expert, and the offsets after it assume 32).
- 132 persistent CTAs (`MOE_CTAS` and `ROUTE_CTAS` in `src/moe_block.py`, `GRID` in `src/boundary_ffn_remat.py`,
  `ST_SK_SLOTS = 3 x 132` in `src/layer.py`), one per SM of an H200 or an H100 SXM.
- sm_90a only: the kernels use wgmma, TMA and programmatic dependent launch.
- The boundary kernels in `src/boundary_ffn_remat.py` accept only width 8192 and the norm form that `supported()`
  tests, and with a token order at most 128 x grid tokens (`RMAX` in `boundary_ffn_remat.cu`).

## Code that does not run

`src/moe_block.py` also holds a second layer, `kernel_fn` and `_after_attn` with `_experts`, `_experts_grouped` and
the Marlin bank, written against SGLang's modules. `src/layer.py` never calls it, and it does not run, because its
`_moe_block` call passes 3 of the 5 arguments. `src/kernels/moe_grouped_fallback.cu` serves only that path but still
compiles at import. Both stay because they are part of the submitted file's code tokens.

## The cuBLAS workspace fault at 8 tokens or fewer

The submitted file can write freed memory when it runs with 8 tokens or fewer (README section 8). The fault lives in
the FP32 router `F.linear` of the 8-token path in `src/layer.py`, which the comment above that call marks, and no
scored shape reaches it. That call is the layer's only torch cuBLAS GEMM at 8 or fewer tokens that uses a cuBLAS
workspace, into which its split-K kernel stores partial sums by TMA. torch keeps one workspace per cuBLAS handle and
queue, and destroying any CUDA graph captured on that queue frees it, even while other live graphs still replay into
it. A process that captures many graphs, keeps some and destroys others then faults on a later replay. Any one of
these avoids it:
- keep every captured graph alive while other graphs captured on the same queue still replay;
- set `CUBLAS_WORKSPACE_CONFIG=:0:0`, so that cuBLAS gets no workspace;
- route the 8-token router and BF16 GEMMs through kernels with no cuBLAS workspace, as the layer already does above 8
  tokens (`_ltgemm` in `src/moe_block.py`).
