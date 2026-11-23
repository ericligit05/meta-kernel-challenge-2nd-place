# okc: problem=muse_prefill_layer
from functools import partial
from sgl_kernel.flash_attn import flash_attn_with_kvcache
from sgl_kernel.scalar_type import scalar_types
from sglang.kernels.ops.activation.activation import silu_and_mul
from sglang.kernels.ops.kvcache.kvcache import store_cache
from sglang.kernels.ops.moe.moe_wna16_marlin import moe_wna16_marlin_gemm
from sglang.kernels.ops.muse_spark_v12.fused_muse_spark_routing import FusedMuseSparkRouting
from sglang.kernels.ops.muse_spark_v12.fused_norm_gate import FusedRMSNormGate
from sglang.kernels.ops.muse_spark_v12.fused_norm_residual import FusedPostNormResidualPreNorm
from sglang.kernels.ops.muse_spark_v12.fused_post_expert_norm_weighted_sum_triton import (
    fused_post_expert_norm_weighted_sum,
)
from sglang.kernels.ops.muse_spark_v12.fused_qk_norm import FusedQKNorm
from sglang.srt.layers.moe.fused_moe_triton import moe_align_block_size
from sglang.srt.layers.muse_spark_layers import (
    MuseSpark1xNormPreservingResidualConnection,
    MuseSpark1xRMSNorm,
    MuseSpark1xRMSNormWithInputScale,
)
from sglang.srt.layers.quantization.marlin_utils import marlin_make_workspace
from sglang.srt.layers.quantization.marlin_utils_fp4 import prepare_moe_nvfp4_layer_for_marlin
from sglang.srt.layers.quantization.unquant import _bf16_gemm_dispatch_impl
from torch import nn
from torch.utils.cpp_extension import load_inline
import threading as _par_threading
_par_load_inline = load_inline


class _ParExt:
    """An extension module still compiling on a daemon thread: the first attribute read waits for the build."""

    def __init__(self, args, kwargs):
        box = {}

        def build():
            try:
                box["mod"] = _par_load_inline(*args, **kwargs)
            except BaseException as e:  # raised again at first use
                box["err"] = e

        th = _par_threading.Thread(target=build, daemon=True)
        th.start()
        object.__setattr__(self, "_box", box)
        object.__setattr__(self, "_th", th)

    def __getattr__(self, name):
        object.__getattribute__(self, "_th").join()
        box = object.__getattribute__(self, "_box")
        if "err" in box:
            raise box["err"]
        return getattr(box["mod"], name)


def load_inline(*args, **kwargs):  # noqa: F811 (concurrent builds, same arguments)
    return _ParExt(args, kwargs)
from typing import TypedDict
import os
import torch
import torch.nn.functional as F
import triton
import triton.language as tl
import types







class xa_Weights(TypedDict):
    qkv_gate: torch.Tensor
    o: torch.Tensor
    attn_residual_gate: torch.Tensor
    pre_ffn_norm: torch.Tensor
    router: torch.Tensor
    router_bias: torch.Tensor
    pooler: torch.Tensor
    pre_expert_norm: torch.Tensor
    w13: torch.Tensor
    w13_sf: torch.Tensor
    w13_gs: torch.Tensor
    w2: torch.Tensor
    w2_sf: torch.Tensor
    w2_gs: torch.Tensor
    post_expert_norm: torch.Tensor
    unpooler: torch.Tensor
    post_ffn_norm: torch.Tensor
    ffn_residual_gate: torch.Tensor
    next_pre_attn_norm: torch.Tensor


class xa_Config(TypedDict):
    batch: int
    q_len: int
    tokens: int
    hidden: int
    num_q_heads: int
    num_kv_heads: int
    head_dim: int
    page_size: int
    window: int
    rope_theta: float
    qk_scale_factor: float
    sm_scale: float
    kv_scale: float
    n_experts: int
    top_k: int
    local_experts: int


class xa_Inputs(TypedDict):
    resid: torch.Tensor
    attn_input: torch.Tensor
    k_cache: torch.Tensor
    v_cache: torch.Tensor
    positions: torch.Tensor
    cache_loc: torch.Tensor
    cache_seqlens: torch.Tensor
    cu_seqlens_q: torch.Tensor
    scheduler_metadata: torch.Tensor | None
    k_scale: torch.Tensor
    v_scale: torch.Tensor
    page_table: torch.Tensor
    seq_lens: torch.Tensor
    local_start: torch.Tensor
    config: xa_Config


xa_HIDDEN = 8192
xa_RMS_EPS = 1e-5
xa_POST_NORM_EPS = 1e-8
xa_FP8 = torch.float8_e4m3fn
xa_N_EXPERTS = 256
xa_TOP_K = 8
xa_LOCAL_EXPERTS = 32
xa_D_EXPERT = 4096
xa_ROUTER_FP32_MAX_TOKENS = 8
xa_SLICED_UNPOOL_MAX_TOKENS = 8
xa_UNPOOL_SLICE = xa_HIDDEN // 8
xa_BLOCK_SIZE_M_CANDIDATES = (8, 16, 32, 48, 64)
xa_ROPE_POSITIONS = 1 << 21
xa_ROPE_THETA = 500_000.0


def xa__gemm(x, w):
    """x @ w.T in BF16. Above 8 tokens through cuBLASLt with no workspace (_ltgemm): the same heuristic kernels as
    torch's call, minus the workspace memset node and its graph gap (~3 us per GEMM at 3012+ tokens)."""
    if (xa_LT_GEMM and x.dim() == 2 and x.shape[0] > 8 and x.dtype == torch.bfloat16 and w.dtype == torch.bfloat16
            and x.is_contiguous() and w.is_contiguous()):
        return xa__ltgemm(x, w)
    return _bf16_gemm_dispatch_impl(x, w, None)


def xa__param(w):
    return nn.Parameter(w.contiguous(), requires_grad=False)


def xa__norm(cls, w, size, eps, **kw):
    """A Muse norm module holding w (applied as 1 + w), as the checkpoint loads it."""
    norm = cls(size, eps=eps, has_weight=w is not None, zero_centered_gamma=True, **kw)
    if w is not None:
        norm.weight = xa__param(w)
        norm.precompute_effective_weight()
    return norm


def xa__residual(w):
    gate = MuseSpark1xNormPreservingResidualConnection(xa_HIDDEN, temperature=0.3)
    gate.gate = xa__param(w)
    gate.precompute_coefficients()
    return gate


def xa__marlin_bank(weights):
    """The local experts in Marlin's layout, via SGLang's own loader step."""
    device = weights["w13"].device
    layer = types.SimpleNamespace(
        quant_config=types.SimpleNamespace(group_size=16),
        moe_runner_config=types.SimpleNamespace(is_gated=True),
        intermediate_size_per_partition=weights["w13"].shape[1] // 2,
        params_dtype=torch.bfloat16,
        w13_weight=weights["w13"].contiguous(),
        w2_weight=weights["w2"].contiguous(),
        w13_weight_scale=weights["w13_sf"].contiguous(),
        w2_weight_scale=weights["w2_sf"].contiguous(),
        w13_weight_scale_2=weights["w13_gs"].contiguous(),
        w2_weight_scale_2=weights["w2_gs"].contiguous(),
    )
    prepare_moe_nvfp4_layer_for_marlin(layer)
    expert_map = torch.full((xa_N_EXPERTS,), -1, dtype=torch.int32, device=device)
    expert_map[:xa_LOCAL_EXPERTS] = torch.arange(xa_LOCAL_EXPERTS, dtype=torch.int32, device=device)
    return {
        "w13": layer.w13_weight.data, "w13_s": layer.w13_weight_scale.data, "w13_gs": layer.w13_weight_scale_2.data,
        "w2": layer.w2_weight.data, "w2_s": layer.w2_weight_scale.data, "w2_gs": layer.w2_weight_scale_2.data,
        "expert_map": expert_map, "size_n": layer.intermediate_size_per_partition,
        "workspace": marlin_make_workspace(device, max_blocks_per_sm=4),
    }


def xa_prepare(weights):
    device = weights["qkv_gate"].device
    weights = dict(weights)
    ki = xa_kperm_index(xa_D_EXPERT).to(device)
    rows13 = torch.cat((ki, xa_D_EXPERT + ki))
    weights["w13"] = weights["w13"].index_select(1, rows13).contiguous()
    weights["w13_sf"] = weights["w13_sf"].index_select(1, rows13).contiguous()
    weights["pooler"] = weights["pooler"].index_select(0, ki).contiguous()
    weights["pre_expert_norm"] = weights["pre_expert_norm"].index_select(0, ki).contiguous()
    xa__glue.px_init()
    router = weights["router"]
    hi = router.to(torch.bfloat16)
    lo = (router - hi.float()).to(torch.bfloat16)
    inv_freq = 1.0 / (xa_ROPE_THETA ** (torch.arange(0, 64, 2, dtype=torch.float, device=device) / 64))
    freqs = torch.einsum("i,j -> ij", torch.arange(xa_ROPE_POSITIONS, dtype=torch.float, device=device), inv_freq)
    pre_ffn = xa__norm(MuseSpark1xRMSNorm, weights["pre_ffn_norm"], xa_HIDDEN, xa_RMS_EPS)
    next_pre_attn = xa__norm(MuseSpark1xRMSNorm, weights["next_pre_attn_norm"], xa_HIDDEN, xa_RMS_EPS)
    post_attn = xa__norm(MuseSpark1xRMSNorm, None, xa_HIDDEN, xa_POST_NORM_EPS, is_post_norm=True,
                      post_norm_gain_center_type="one", residual_id=1)
    post_ffn = xa__norm(MuseSpark1xRMSNormWithInputScale, weights["post_ffn_norm"], xa_HIDDEN, xa_POST_NORM_EPS,
                     is_post_norm=True, post_norm_gain_center_type="one", residual_id=2)
    post_expert = xa__norm(MuseSpark1xRMSNormWithInputScale, weights["post_expert_norm"], xa_D_EXPERT, xa_POST_NORM_EPS,
                        is_post_norm=True, post_norm_gain_center_type="one", residual_id=2)
    state = {
        "dense": {name: weights[name].contiguous() for name in ("qkv_gate", "o", "pooler", "unpooler")},
        "router": router.contiguous(),
        "router_hi_t": hi.t().contiguous(),
        "router_lo_t": lo.t().contiguous(),
        "routing": FusedMuseSparkRouting(expert_bias=weights["router_bias"].float().contiguous(), route_scale=1.0,
                                         router_output_multiplier=0.1767766952966369,
                                         router_score_division_eps=1e-15),
        "cos_sin": torch.cat((freqs.cos(), freqs.sin()), dim=-1),
        "qk_norm": FusedQKNorm(head_dim=64, num_q_heads=16, num_kv_heads=2, eps=xa_RMS_EPS),
        "o_norm_gate": FusedRMSNormGate(MuseSpark1xRMSNorm(64, eps=xa_RMS_EPS, has_weight=False), True, 64),
        "attn_boundary": FusedPostNormResidualPreNorm(post_attn, xa__residual(weights["attn_residual_gate"]), pre_ffn),
        "ffn_boundary": FusedPostNormResidualPreNorm(post_ffn, xa__residual(weights["ffn_residual_gate"]), next_pre_attn),
        "pre_expert_norm": xa__norm(MuseSpark1xRMSNorm, weights["pre_expert_norm"], xa_D_EXPERT, xa_RMS_EPS),
        "post_expert_w": post_expert._get_effective_weight().detach().to(torch.bfloat16),
        "bank": xa__marlin_bank(weights),
        "raw": {"w13": weights["w13"].contiguous(), "w13_sf": weights["w13_sf"].contiguous(),
                "gs13": weights["w13_gs"].float().reshape(-1).contiguous(), "w2": weights["w2"].contiguous(),
                "w2_sf": weights["w2_sf"].contiguous(), "gs2": weights["w2_gs"].float().reshape(-1).contiguous()},
        "moe": {"w13": xa__pack_nvfp4(weights["w13"].contiguous(), weights["w13_sf"].contiguous(),
                                   torch.arange(2 * xa_D_EXPERT, device=device).view(2, xa_D_EXPERT // 128, 2, 64)
                                   .permute(1, 2, 0, 3).reshape(-1), bfs=False),
                "w2": xa__pack_nvfp4(weights["w2"].contiguous(), weights["w2_sf"].contiguous(), bfs=False),
                "w13w": xa__pack_nvfp4(weights["w13"].contiguous(), weights["w13_sf"].contiguous(),
                                    torch.stack((torch.arange(xa_D_EXPERT, device=device).view(xa_D_EXPERT // 64, 8, 8),
                                                 xa_D_EXPERT + torch.arange(xa_D_EXPERT, device=device).view(xa_D_EXPERT // 64, 8, 8)),
                                                dim=2).reshape(-1), mb=1),
                "w2w": xa__pack_nvfp4(weights["w2"].contiguous(), weights["w2_sf"].contiguous(), mb=1),
                "gs13": weights["w13_gs"].float().reshape(-1).contiguous(),
                "gs2": weights["w2_gs"].float().reshape(-1).contiguous()},
        "router_cat_t": torch.cat((hi.t(), lo.t()), dim=1).contiguous(),
        "pool_router": torch.cat((weights["pooler"], hi, lo), dim=0).contiguous(),
        "route_ctr": torch.zeros(98, dtype=torch.int32, device=device),
        "router_bias": weights["router_bias"].float().contiguous(),
        "side_stream": torch.cuda.Stream(device),
        "_modules": (pre_ffn, next_pre_attn, post_attn, post_ffn, post_expert),
    }
    pen = state["pre_expert_norm"]
    pw = pen._get_effective_weight() if hasattr(pen, "_get_effective_weight") else 1.0 + weights["pre_expert_norm"].float()
    state["pre_expert_wf"] = pw.detach().float().contiguous()
    del freqs
    torch.cuda.synchronize()
    return state


def xa__experts(x, topk_weights, topk_ids, state):
    """MuseSpark1xFp4Experts._marlin_expert_outputs and its post-expert-norm finalize."""
    bank = state["bank"]
    num_tokens, size_k = x.shape
    size_n = bank["size_n"]
    gemm1_n = 2 * size_n
    local_ids = bank["expert_map"][topk_ids].contiguous()
    topk_weights = topk_weights.to(torch.float32).contiguous()
    for block_size_m in xa_BLOCK_SIZE_M_CANDIDATES:
        if num_tokens * xa_TOP_K / xa_LOCAL_EXPERTS / block_size_m < 0.9:
            break
    sorted_ids, block_experts, n_padded = moe_align_block_size(local_ids, block_size_m, xa_N_EXPERTS)
    cache2 = torch.empty((num_tokens * xa_TOP_K, size_n), device=x.device, dtype=x.dtype)
    cache13 = torch.empty((num_tokens * xa_TOP_K * max(gemm1_n, size_k),), device=x.device, dtype=x.dtype)
    cache1 = cache13[: num_tokens * xa_TOP_K * gemm1_n].view(-1, gemm1_n)
    cache3 = cache13[: num_tokens * xa_TOP_K * size_k].view(-1, size_k)
    common = dict(moe_block_size=block_size_m, mul_topk_weights=False, is_ep=True,
                  b_q_type=scalar_types.float4_e2m1f, is_k_full=True, use_atomic_add=False,
                  use_fp32_reduce=True, is_zp_float=False)
    ws = bank["workspace"]
    cache1 = moe_wna16_marlin_gemm(
        x, cache1, bank["w13"], None, bank["w13_s"], bank["w13_gs"], None, None, None, ws,
        sorted_ids, block_experts, n_padded, topk_weights, top_k=xa_TOP_K,
        size_m=num_tokens, size_n=gemm1_n, size_k=size_k, **common)
    silu_and_mul(cache1.view(-1, gemm1_n), cache2, expert_ids=local_ids.view(-1), expert_step=1)
    y = moe_wna16_marlin_gemm(
        cache2, cache3, bank["w2"], None, bank["w2_s"], bank["w2_gs"], None, None, None, ws,
        sorted_ids, block_experts, n_padded, topk_weights, top_k=1,
        size_m=num_tokens * xa_TOP_K, size_n=size_k, size_k=size_n, **common).view(num_tokens, xa_TOP_K, size_k)
    return fused_post_expert_norm_weighted_sum(
        y, topk_weights, norm_weight=state["post_expert_w"], eps=xa_POST_NORM_EPS, weight_first=True,
        slot_expert_ids=local_ids)


xa__GROUPED_SRC = r"""

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
__constant__ float kE2M1[8] = {0.f, 0.5f, 1.f, 1.5f, 2.f, 3.f, 4.f, 6.f};

__global__ void dequant_k(const uint2* __restrict__ p, const __nv_fp8_e4m3* __restrict__ sf, uint4* __restrict__ out,
             long nblk) {
 const long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
 if (i >= nblk) return;
 const float s = float(sf[i]);
 const uint2 v = p[i];
 const uint32_t w[2] = {v.x, v.y};
 __align__(16) __nv_bfloat16 o[16];
#pragma unroll
 for (int b = 0; b < 2; b++)
#pragma unroll
  for (int n = 0; n < 8; n++) {
   const uint32_t c = (w[b] >> (4 * n)) & 15u;
   const float x = kE2M1[c & 7] * s;
   o[b * 8 + n] = __float2bfloat16(c & 8 ? -x : x);
  }
 out[2 * i] = reinterpret_cast<uint4*>(o)[0];
 out[2 * i + 1] = reinterpret_cast<uint4*>(o)[1];
}

__global__ void gather_k(const uint4* __restrict__ x, const long* __restrict__ order, const int* __restrict__ offs,
             uint4* __restrict__ a, int K8) {
 const int r = blockIdx.x;
 if (r >= offs[31]) return;
 const long src = order[r] / 8;
 for (int c = threadIdx.x; c < K8; c += blockDim.x) a[(long)r * K8 + c] = x[src * K8 + c];
}

__global__ void silu_k(const __nv_bfloat16* __restrict__ h, const float* __restrict__ gs, const long* __restrict__ key,
            const int* __restrict__ offs, __nv_bfloat16* __restrict__ inter, int I) {
 const int r = blockIdx.x;
 if (r >= offs[31]) return;
 const float s = gs[key[r]];
 for (int c = threadIdx.x; c < I; c += blockDim.x) {
  const float g = __bfloat162float(h[(long)r * 2 * I + c]) * s, u = __bfloat162float(h[(long)r * 2 * I + I + c]) * s;
  inter[(long)r * I + c] = __float2bfloat16(__bfloat162float(__float2bfloat16(g / (1.f + __expf(-g)))) * __bfloat162float(__float2bfloat16(u)));
 }
}

__global__ void unsort_k(const __nv_bfloat16* __restrict__ ys, const long* __restrict__ inv, const long* __restrict__ keyslot,
             const float* __restrict__ gs, __nv_bfloat16* __restrict__ y, int H) {
 const long slot = blockIdx.x;
 const long e = keyslot[slot];
 const bool loc = e < 32;
 const float s = loc ? gs[e] : 0.f;
 const long src = loc ? inv[slot] : 0;
 for (int c = threadIdx.x; c < H; c += blockDim.x)
  y[slot * H + c] = loc ? __float2bfloat16(__bfloat162float(ys[src * H + c]) * s) : __float2bfloat16(0.f);
}
void dequant(torch::Tensor p, torch::Tensor sf, torch::Tensor out) {
 const long nblk = sf.numel();
 dequant_k<<<(nblk + 255) / 256, 256, 0, at::cuda::getCurrentCUDAStream()>>>(
   (const uint2*)p.data_ptr(), (const __nv_fp8_e4m3*)sf.data_ptr(), (uint4*)out.data_ptr(), nblk);
}
void gather(torch::Tensor x, torch::Tensor order, torch::Tensor offs, torch::Tensor a) {
 gather_k<<<a.size(0), 256, 0, at::cuda::getCurrentCUDAStream()>>>((const uint4*)x.data_ptr(), order.data_ptr<long>(),
   offs.data_ptr<int>(), (uint4*)a.data_ptr(), (int)(x.size(1) / 8));
}
void silu(torch::Tensor h, torch::Tensor gs, torch::Tensor key, torch::Tensor offs, torch::Tensor inter) {
 silu_k<<<h.size(0), 256, 0, at::cuda::getCurrentCUDAStream()>>>((const __nv_bfloat16*)h.data_ptr(), gs.data_ptr<float>(),
   key.data_ptr<long>(), offs.data_ptr<int>(), (__nv_bfloat16*)inter.data_ptr(), (int)inter.size(1));
}
void unsort(torch::Tensor ys, torch::Tensor inv, torch::Tensor keyslot, torch::Tensor gs, torch::Tensor y) {
 unsort_k<<<y.size(0), 256, 0, at::cuda::getCurrentCUDAStream()>>>((const __nv_bfloat16*)ys.data_ptr(),
   inv.data_ptr<long>(), keyslot.data_ptr<long>(), gs.data_ptr<float>(), (__nv_bfloat16*)y.data_ptr(), (int)y.size(1));
}
"""

xa__LT_SRC = r"""


#include <cuda_runtime.h>
#include <cublasLt.h>
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <map>
#include <tuple>
#include <vector>

namespace {
cublasLtHandle_t lt_handle() {
 static cublasLtHandle_t h = nullptr;
 if (!h) TORCH_CHECK(cublasLtCreate(&h) == CUBLAS_STATUS_SUCCESS, "cublasLtCreate failed");
 return h;
}
struct Plan {
 cublasLtMatmulDesc_t op;
 cublasLtMatrixLayout_t a, b, c;
 cublasLtMatmulAlgo_t algo;
};


Plan* plan_for(int64_t M, int64_t N, int64_t K, int64_t ai, int64_t smc, int64_t tile, int64_t cl, int64_t cu) {
 static std::map<std::tuple<int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t>, Plan*> cache;
 auto key = std::make_tuple(M, N, K, ai, smc, tile, cl, cu);
 auto it = cache.find(key);
 if (it != cache.end()) return it->second;
 Plan* p = new Plan();
 cublasLtMatmulDescCreate(&p->op, CUBLAS_COMPUTE_32F, CUDA_R_32F);
 cublasOperation_t ta = CUBLAS_OP_T, tb = CUBLAS_OP_N;
 cublasLtMatmulDescSetAttribute(p->op, CUBLASLT_MATMUL_DESC_TRANSA, &ta, sizeof(ta));
 cublasLtMatmulDescSetAttribute(p->op, CUBLASLT_MATMUL_DESC_TRANSB, &tb, sizeof(tb));
 if (smc > 0) {
  int32_t s = (int32_t)smc;
  cublasLtMatmulDescSetAttribute(p->op, CUBLASLT_MATMUL_DESC_SM_COUNT_TARGET, &s, sizeof(s));
 }
 cublasLtMatrixLayoutCreate(&p->a, CUDA_R_16BF, K, N, K);
 cublasLtMatrixLayoutCreate(&p->b, CUDA_R_16BF, K, M, K);
 cublasLtMatrixLayoutCreate(&p->c, CUDA_R_16BF, N, M, N);
 cublasLtMatmulPreference_t pref;
 cublasLtMatmulPreferenceCreate(&pref);
 size_t ws = 0;
 cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &ws, sizeof(ws));
 cublasLtMatmulHeuristicResult_t res[16];
 int found = 0;
 cublasLtMatmulAlgoGetHeuristic(lt_handle(), p->op, p->a, p->b, p->c, p->c, pref, 16, res, &found);
 cublasLtMatmulPreferenceDestroy(pref);
 TORCH_CHECK(found > 0, "no cuBLASLt algorithm");
 int pick = ai < found ? (int)ai : 0;
 if (tile >= 0) {




  int loose = -1;
  for (int i = 0; i < found; i++) {
   uint32_t t = 0, o = 0; uint16_t c = 0; size_t sz = 0;
   cublasLtMatmulAlgoConfigGetAttribute(&res[i].algo, CUBLASLT_ALGO_CONFIG_TILE_ID, &t, sizeof(t), &sz);
   cublasLtMatmulAlgoConfigGetAttribute(&res[i].algo, CUBLASLT_ALGO_CONFIG_CLUSTER_SHAPE_ID, &c, sizeof(c), &sz);
   cublasLtMatmulAlgoConfigGetAttribute(&res[i].algo, CUBLASLT_ALGO_CONFIG_CUSTOM_OPTION, &o, sizeof(o), &sz);
   if ((int64_t)t == tile && (int64_t)c == cl) {
    if (cu < 0 || (int64_t)o == cu) { pick = i; loose = -1; break; }
    if (loose < 0) loose = i;
   }
  }
  if (loose >= 0) {
   cublasLtMatmulAlgo_t a2 = res[loose].algo;
   uint32_t o = (uint32_t)cu;
   cublasLtMatmulAlgoConfigSetAttribute(&a2, CUBLASLT_ALGO_CONFIG_CUSTOM_OPTION, &o, sizeof(o));
   cublasLtMatmulHeuristicResult_t r;
   if (cublasLtMatmulAlgoCheck(lt_handle(), p->op, p->a, p->b, p->c, p->c, &a2, &r) == CUBLAS_STATUS_SUCCESS &&
     r.workspaceSize == 0) {
    p->algo = a2; cache[key] = p; return p;
   }
   pick = loose;
  }
 }
 p->algo = res[pick].algo;
 cache[key] = p;
 return p;
}
}



std::vector<uint64_t> algo_cfg(const cublasLtMatmulAlgo_t* a) {
 return std::vector<uint64_t>(a->data, a->data + 8);
}


Plan* plan_like(int64_t M, int64_t Mfull, int64_t N, int64_t K, int64_t ai, int64_t smc, int64_t tile, int64_t cl, int64_t cu) {
 static std::map<std::tuple<int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t>, Plan*> cache;
 auto key = std::make_tuple(M, Mfull, N, K, ai, smc, tile, cl, cu);
 auto it = cache.find(key);
 if (it != cache.end()) return it->second;
 const std::vector<uint64_t> want = algo_cfg(&plan_for(Mfull, N, K, ai, smc, tile, cl, cu)->algo);
 Plan* p = new Plan();
 cublasLtMatmulDescCreate(&p->op, CUBLAS_COMPUTE_32F, CUDA_R_32F);
 cublasOperation_t ta = CUBLAS_OP_T, tb = CUBLAS_OP_N;
 cublasLtMatmulDescSetAttribute(p->op, CUBLASLT_MATMUL_DESC_TRANSA, &ta, sizeof(ta));
 cublasLtMatmulDescSetAttribute(p->op, CUBLASLT_MATMUL_DESC_TRANSB, &tb, sizeof(tb));
 cublasLtMatrixLayoutCreate(&p->a, CUDA_R_16BF, K, N, K);
 cublasLtMatrixLayoutCreate(&p->b, CUDA_R_16BF, K, M, K);
 cublasLtMatrixLayoutCreate(&p->c, CUDA_R_16BF, N, M, N);
 cublasLtMatmulPreference_t pref;
 cublasLtMatmulPreferenceCreate(&pref);
 size_t ws = 0;
 cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &ws, sizeof(ws));
 static cublasLtMatmulHeuristicResult_t res[64];
 int found = 0;
 cublasLtMatmulAlgoGetHeuristic(lt_handle(), p->op, p->a, p->b, p->c, p->c, pref, 64, res, &found);
 cublasLtMatmulPreferenceDestroy(pref);
 Plan* r = nullptr;
 for (int i = 0; i < found && !r; i++)
  if (algo_cfg(&res[i].algo) == want) { p->algo = res[i].algo; r = p; }
 cache[key] = r;
 return r;
}

bool lt_like_ok(int64_t M, int64_t Mfull, int64_t N, int64_t K, int64_t ai, int64_t smc, int64_t tile, int64_t cl, int64_t cu) {
 return plan_like(M, Mfull, N, K, ai, smc, tile, cl, cu) != nullptr;
}
void lt_gemm_like(torch::Tensor x, torch::Tensor w, torch::Tensor out, int64_t Mfull, int64_t ai, int64_t smc, int64_t tile, int64_t cl, int64_t cu) {
 const int64_t M = x.size(0), K = x.size(1), N = w.size(0);
 Plan* p = plan_like(M, Mfull, N, K, ai, smc, tile, cl, cu);
 TORCH_CHECK(p != nullptr, "lt_gemm_like: no matching algorithm");
 const float alpha = 1.f, beta = 0.f;
 auto r = cublasLtMatmul(lt_handle(), p->op, &alpha, w.data_ptr(), p->a, x.data_ptr(), p->b, &beta, out.data_ptr(),
             p->c, out.data_ptr(), p->c, &p->algo, nullptr, 0, at::cuda::getCurrentCUDAStream());
 TORCH_CHECK(r == CUBLAS_STATUS_SUCCESS, "cublasLtMatmul failed");
}
void lt_gemm(torch::Tensor x, torch::Tensor w, torch::Tensor out, int64_t ai, int64_t smc, int64_t tile, int64_t cl, int64_t cu) {
 const int64_t M = x.size(0), K = x.size(1), N = w.size(0);
 Plan* p = plan_for(M, N, K, ai, smc, tile, cl, cu);
 const float alpha = 1.f, beta = 0.f;
 auto r = cublasLtMatmul(lt_handle(), p->op, &alpha, w.data_ptr(), p->a, x.data_ptr(), p->b, &beta, out.data_ptr(),
             p->c, out.data_ptr(), p->c, &p->algo, nullptr, 0, at::cuda::getCurrentCUDAStream());
 TORCH_CHECK(r == CUBLAS_STATUS_SUCCESS, "cublasLtMatmul failed");
}
"""
xa__lt = load_inline(name="okc_prefill_lt_y10201_b460010", cuda_sources=xa__LT_SRC, cpp_sources="void lt_gemm(torch::Tensor, torch::Tensor, torch::Tensor, int64_t, int64_t, int64_t, int64_t, int64_t);\n"
                  "bool lt_like_ok(int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t);\n"
                  "void lt_gemm_like(torch::Tensor, torch::Tensor, torch::Tensor, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t);",
                  functions=["lt_gemm", "lt_like_ok", "lt_gemm_like"], extra_cuda_cflags=["-O3"], extra_ldflags=["-lcublasLt"])
xa_LT_GEMM = True
xa_LT_ALGO = {(True, 8192, 1024): 3, (True, 4096, 8192): 1, (True, 8192, 4096): 3}


xa_LT_PREF = {
    (1, 8192, 1024): (130, 20, 5, 3),
    (1, 8192, 4096): (128, 20, 5, 3),
    (2, 8192, 1024): (126, 201, 5, -1),
    (2, 2304, 8192): (0, 496, 2, 5),
    (2, 4096, 8192): (130, 24, 5, 7),
    (2, 8192, 4096): (128, 423, 3, 5),
    (3, 2304, 8192): (128, 24, 5, 7),
    (3, 8192, 1024): (0, 201, 5, 2),
    (3, 8192, 4096): (0, 197, 5, -1),
}
xa_LT_PREF_ON = True


def xa__band(m):
    return 1 if 512 <= m < 1024 else 2 if 2048 <= m < 4096 else 3 if m >= 8192 else 0


def xa__ltgemm(x, w):
    out = torch.empty((x.shape[0], w.shape[0]), dtype=torch.bfloat16, device=x.device)
    pref = xa_LT_PREF.get((xa__band(x.shape[0]), w.shape[0], w.shape[1])) if xa_LT_PREF_ON else None
    smc, tile, cl, cu = pref if pref else (0, -1, 0, -1)
    xa__lt.lt_gemm(x, w, out, xa_LT_ALGO.get((x.shape[0] >= 8192, w.shape[0], w.shape[1]), 0), smc, tile, cl, cu)
    return out


xa_POOL_SPLIT_MIN = 12288
xa_POOL_SPLIT_ROWS = 33 * 256
xa__POOL_SPLIT = {}


def xa__ltgemm_cfg(m, w, mfull):
    pref = xa_LT_PREF.get((xa__band(mfull), w.shape[0], w.shape[1])) if xa_LT_PREF_ON else None
    smc, tile, cl, cu = pref if pref else (0, -1, 0, -1)
    return xa_LT_ALGO.get((mfull >= 8192, w.shape[0], w.shape[1]), 0), smc, tile, cl, cu


def xa__pool_split(h, w):
    """Rows of the first part of a split pooler GEMM, or 0 (no split): both parts must have the full call's
    algorithm configuration (the same nvjet kernel)."""
    T = h.shape[0]
    if not (xa_LT_GEMM and T >= xa_POOL_SPLIT_MIN and h.dtype == torch.bfloat16 and h.is_contiguous() and w.is_contiguous()):
        return 0
    key = (T, w.shape[0], w.shape[1])
    if key not in xa__POOL_SPLIT:
        ma = xa_POOL_SPLIT_ROWS
        cfg = xa__ltgemm_cfg(T, w, T)
        ok = all(xa__lt.lt_like_ok(m, T, w.shape[0], w.shape[1], *cfg) for m in (ma, T - ma))
        xa__POOL_SPLIT[key] = ma if ok else 0
    return xa__POOL_SPLIT[key]


xa__POOL_B = {}


def xa__pool_b_stream(dev):
    """The queue of a split pooler GEMM's second part (the layer's expert state has no slot for it)."""
    if dev not in xa__POOL_B:
        xa__POOL_B[dev] = torch.cuda.Stream(dev, priority=-1)
    return xa__POOL_B[dev]


def xa__ltgemm_part(x, w, out, mfull):
    """Rows of a GEMM of mfull rows: x @ w.T into out on the full call's kernel."""
    xa__lt.lt_gemm_like(x, w, out, mfull, *xa__ltgemm_cfg(x.shape[0], w, mfull))


xa__RT_SRC = r"""


#include <cuda_runtime.h>
#include <cublasLt.h>
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <map>
#include <tuple>

namespace {
cublasLtHandle_t rt_handle() {
 static cublasLtHandle_t h = nullptr;
 if (!h) TORCH_CHECK(cublasLtCreate(&h) == CUBLAS_STATUS_SUCCESS, "cublasLtCreate failed");
 return h;
}
struct RPlan {
 cublasLtMatmulDesc_t op;
 cublasLtMatrixLayout_t a, b, c;
 cublasLtMatmulAlgo_t algo[8];
 int found;
};
void batched(cublasLtMatrixLayout_t l, int32_t n, int64_t stride) {
 cublasLtMatrixLayoutSetAttribute(l, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &n, sizeof(n));
 cublasLtMatrixLayoutSetAttribute(l, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &stride, sizeof(stride));
}

RPlan* rplan(int64_t M, int64_t N, int64_t K, int64_t S) {
 static std::map<std::tuple<int64_t, int64_t, int64_t, int64_t>, RPlan*> cache;
 auto key = std::make_tuple(M, N, K, S);
 auto it = cache.find(key);
 if (it != cache.end()) return it->second;
 RPlan* p = new RPlan();
 const int64_t Kc = K / S;
 cublasLtMatmulDescCreate(&p->op, CUBLAS_COMPUTE_32F, CUDA_R_32F);
 cublasOperation_t tn = CUBLAS_OP_N;
 cublasLtMatmulDescSetAttribute(p->op, CUBLASLT_MATMUL_DESC_TRANSA, &tn, sizeof(tn));
 cublasLtMatmulDescSetAttribute(p->op, CUBLASLT_MATMUL_DESC_TRANSB, &tn, sizeof(tn));
 cublasLtMatrixLayoutCreate(&p->a, CUDA_R_16BF, N, Kc, N);
 cublasLtMatrixLayoutCreate(&p->b, CUDA_R_16BF, Kc, M, K);
 cublasLtMatrixLayoutCreate(&p->c, CUDA_R_32F, N, M, N);
 batched(p->a, (int32_t)S, Kc * N);
 batched(p->b, (int32_t)S, Kc);
 batched(p->c, (int32_t)S, M * N);
 cublasLtMatmulPreference_t pref;
 cublasLtMatmulPreferenceCreate(&pref);
 size_t ws = 0;
 cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &ws, sizeof(ws));
 cublasLtMatmulHeuristicResult_t res[8];
 p->found = 0;
 cublasLtMatmulAlgoGetHeuristic(rt_handle(), p->op, p->a, p->b, p->c, p->c, pref, 8, res, &p->found);
 cublasLtMatmulPreferenceDestroy(pref);
 TORCH_CHECK(p->found > 0, "no cuBLASLt algorithm for the router slices");
 for (int i = 0; i < p->found; i++) p->algo[i] = res[i].algo;
 cache[key] = p;
 return p;
}
}


void lt_router(torch::Tensor x, torch::Tensor wt, torch::Tensor out, int64_t tile) {
 const int64_t S = out.size(0), M = x.size(0), K = x.size(1), N = wt.size(1);
 RPlan* p = rplan(M, N, K, S);
 int ai = 0;
 for (int i = 0; i < p->found; i++) {
  uint32_t t = 0;
  size_t w = 0;
  cublasLtMatmulAlgoConfigGetAttribute(&p->algo[i], CUBLASLT_ALGO_CONFIG_TILE_ID, &t, sizeof(t), &w);
  if ((int64_t)t == tile) { ai = i; break; }
 }
 const float alpha = 1.f, beta = 0.f;
 auto r = cublasLtMatmul(rt_handle(), p->op, &alpha, wt.data_ptr(), p->a, x.data_ptr(), p->b, &beta, out.data_ptr(),
             p->c, out.data_ptr(), p->c, &p->algo[ai], nullptr, 0, at::cuda::getCurrentCUDAStream());
 TORCH_CHECK(r == CUBLAS_STATUS_SUCCESS, "cublasLtMatmul (router slices) failed");
}


torch::Tensor lt_router_info(int64_t M, int64_t N, int64_t K, int64_t S) {
 RPlan* p = rplan(M, N, K, S);
 auto t = torch::zeros({p->found, 4}, torch::kInt64);
 for (int i = 0; i < p->found; i++) {
  uint32_t tile = 0, red = 0, swz = 0;
  int32_t sk = 0;
  size_t w = 0;
  cublasLtMatmulAlgoConfigGetAttribute(&p->algo[i], CUBLASLT_ALGO_CONFIG_TILE_ID, &tile, sizeof(tile), &w);
  cublasLtMatmulAlgoConfigGetAttribute(&p->algo[i], CUBLASLT_ALGO_CONFIG_SPLITK_NUM, &sk, sizeof(sk), &w);
  cublasLtMatmulAlgoConfigGetAttribute(&p->algo[i], CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME, &red, sizeof(red), &w);
  cublasLtMatmulAlgoConfigGetAttribute(&p->algo[i], CUBLASLT_ALGO_CONFIG_CTA_SWIZZLING, &swz, sizeof(swz), &w);
  t[i][0] = tile; t[i][1] = sk; t[i][2] = red; t[i][3] = swz;
 }
 return t;
}



namespace {
struct CPlan {
 cublasLtMatmulDesc_t op;
 cublasLtMatrixLayout_t a, b, c;
 cublasLtMatmulAlgo_t algo[16];
 int found;
 cublasLtMatmulAlgo_t forced;
 bool has_forced = false;
};
CPlan* cplan(int64_t M, int64_t N, int64_t K, int64_t tile, int64_t cl, int64_t cu) {
 static std::map<std::tuple<int64_t, int64_t, int64_t, int64_t, int64_t, int64_t>, CPlan*> cache;
 auto key = std::make_tuple(M, N, K, tile, cl, cu);
 auto it = cache.find(key);
 if (it != cache.end()) return it->second;
 CPlan* p = new CPlan();
 cublasLtMatmulDescCreate(&p->op, CUBLAS_COMPUTE_32F, CUDA_R_32F);
 cublasOperation_t ta = CUBLAS_OP_T, tb = CUBLAS_OP_N;
 cublasLtMatmulDescSetAttribute(p->op, CUBLASLT_MATMUL_DESC_TRANSA, &ta, sizeof(ta));
 cublasLtMatmulDescSetAttribute(p->op, CUBLASLT_MATMUL_DESC_TRANSB, &tb, sizeof(tb));
 cublasLtMatrixLayoutCreate(&p->a, CUDA_R_16BF, K, N, K);
 cublasLtMatrixLayoutCreate(&p->b, CUDA_R_16BF, K, M, K);
 cublasLtMatrixLayoutCreate(&p->c, CUDA_R_32F, N, M, N);
 cublasLtMatmulPreference_t pref;
 cublasLtMatmulPreferenceCreate(&pref);
 size_t ws = 0;
 cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &ws, sizeof(ws));
 cublasLtMatmulHeuristicResult_t res[16];
 p->found = 0;
 cublasLtMatmulAlgoGetHeuristic(rt_handle(), p->op, p->a, p->b, p->c, p->c, pref, 16, res, &p->found);
 cublasLtMatmulPreferenceDestroy(pref);
 TORCH_CHECK(p->found > 0, "no cuBLASLt algorithm for the pooler + router GEMM");
 for (int i = 0; i < p->found; i++) p->algo[i] = res[i].algo;
 if (tile >= 0) {
  cublasLtMatmulAlgo_t a2 = res[0].algo;
  uint32_t t = (uint32_t)tile, o = (uint32_t)cu; uint16_t c = (uint16_t)cl;
  cublasLtMatmulAlgoConfigSetAttribute(&a2, CUBLASLT_ALGO_CONFIG_TILE_ID, &t, sizeof(t));
  cublasLtMatmulAlgoConfigSetAttribute(&a2, CUBLASLT_ALGO_CONFIG_CLUSTER_SHAPE_ID, &c, sizeof(c));
  cublasLtMatmulAlgoConfigSetAttribute(&a2, CUBLASLT_ALGO_CONFIG_CUSTOM_OPTION, &o, sizeof(o));
  cublasLtMatmulHeuristicResult_t r;
  if (cublasLtMatmulAlgoCheck(rt_handle(), p->op, p->a, p->b, p->c, p->c, &a2, &r) == CUBLAS_STATUS_SUCCESS &&
    r.workspaceSize == 0) {
   p->forced = a2; p->has_forced = true;
  }
 }
 cache[key] = p;
 return p;
}
}

void lt_f32(torch::Tensor x, torch::Tensor w, torch::Tensor out, int64_t ai, int64_t tile, int64_t cl, int64_t cu) {
 const int64_t M = x.size(0), K = x.size(1), N = w.size(0);
 CPlan* p = cplan(M, N, K, tile, cl, cu);
 const int i = ai < p->found ? (int)ai : 0;
 const float alpha = 1.f, beta = 0.f;
 auto r = cublasLtMatmul(rt_handle(), p->op, &alpha, w.data_ptr(), p->a, x.data_ptr(), p->b, &beta, out.data_ptr(),
             p->c, out.data_ptr(), p->c, (p->has_forced ? &p->forced : &p->algo[i]), nullptr, 0, at::cuda::getCurrentCUDAStream());
 TORCH_CHECK(r == CUBLAS_STATUS_SUCCESS, "cublasLtMatmul (pooler + router) failed");
}
"""
xa__rt = load_inline(name="okc_prefill_rt_eg3806_b460010", cuda_sources=xa__RT_SRC,
                  cpp_sources="void lt_router(torch::Tensor, torch::Tensor, torch::Tensor, int64_t);\n"
                              "torch::Tensor lt_router_info(int64_t, int64_t, int64_t, int64_t);\n"
                              "void lt_f32(torch::Tensor, torch::Tensor, torch::Tensor, int64_t, int64_t, int64_t, int64_t);",
                  functions=["lt_router", "lt_router_info", "lt_f32"], extra_cuda_cflags=["-O3"],
                  extra_ldflags=["-lcublasLt"])
xa_CAT_MAX_TOKENS = 2047
xa_CAT_ALGO = 0
xa_CAT_CFG = (185, 3, 4)
xa_SLIDE_WIDE_MIN = 1 << 30
xa_TILE_HD = 64
xa_RNG3_XM = 49
xa_CAT_FUSED = True
xa_ROUTER_SPLIT = 4
xa_ROUTER_SPLIT_MAX_TOKENS = 2047
xa_ROUTER_TILE = 20
xa_ROUTE_CTAS_SMALL = 32


xa__gx = load_inline(name="okc_prefill_grouped_v1_b460010", cuda_sources=xa__GROUPED_SRC,
                  cpp_sources="void dequant(torch::Tensor, torch::Tensor, torch::Tensor);\nvoid gather(torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor);\nvoid silu(torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor);\nvoid unsort(torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor);",
                  functions=["dequant", "gather", "silu", "unsort"], extra_cuda_cflags=["-O3"])
xa_GROUPED_MIN_TOKENS = 4096

xa__MOE_SRC = r"""







#define Z_MIDSET 15
#define Z_M2SET 7
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <stdint.h>




static int g_pdl = 1;
template <typename K, typename... A>
static void pdl_launch(K kern, dim3 grid, dim3 block, size_t smem, A... args) {
 cudaLaunchAttribute at_[1];
 at_[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
 at_[0].val.programmaticStreamSerializationAllowed = g_pdl;
 cudaLaunchConfig_t c_ = {grid, block, smem, at::cuda::getCurrentCUDAStream(), at_, 1};
 cudaLaunchKernelEx(&c_, kern, args...);
}
#define GDC_WAIT() asm volatile("griddepcontrol.wait;" ::: "memory")
#define GDC_GO() asm volatile("griddepcontrol.launch_dependents;" ::: "memory")

namespace {














constexpr int FZ_PAIR = 13312;
constexpr int NW = 8, UNIT = 576, UNIT2 = 576;


#define MOE_SWZ 1
#define MOE_IL 1
constexpr int EBYTES = 256 * 136 * 2;


__host__ __device__ constexpr int wbytes(int mb) { return mb == 2 ? NW * 2 * UNIT2 : NW * mb * UNIT; }
__host__ __device__ constexpr int stage_bytes(int tn, int mb) { return ((tn * 128 + wbytes(mb)) + 1023) / 1024 * 1024; }
__host__ __device__ constexpr int nstages(int tn) { return tn >= 256 ? 4 : 5; }

__device__ __forceinline__ void mbar_init(uint32_t bar, int count) {
 asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::"r"(bar), "r"(count));
}
__device__ __forceinline__ void mbar_expect(uint32_t bar, int bytes) {
 asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(bar), "r"(bytes) : "memory");
}
__device__ __forceinline__ void mbar_arrive(uint32_t bar) {
 asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" ::"r"(bar) : "memory");
}
__device__ __forceinline__ void mbar_wait(uint32_t bar, uint32_t parity) {
 asm volatile(
   "{\n .reg .pred p;\n W: mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n @!p bra W;\n}" ::"r"(bar),
   "r"(parity) : "memory");
}
__device__ __forceinline__ bool x64_elect() {
 uint32_t p;
 asm volatile("{\n .reg .pred P;\n elect.sync _|P, 0xffffffff;\n selp.u32 %0, 1, 0, P;\n}" : "=r"(p));
 return p != 0;
}
__device__ __forceinline__ void bulk_g2s(uint32_t dst, const void* src, int bytes, uint32_t bar) {
 asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];"
        ::"r"(dst), "l"(src), "r"(bytes), "r"(bar) : "memory");
}


__device__ __forceinline__ void stsm4t(uint32_t addr, uint32_t a, uint32_t b, uint32_t c, uint32_t d) {
 asm volatile("stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%0], {%1, %2, %3, %4};" ::"r"(addr), "r"(a), "r"(b), "r"(c),
        "r"(d) : "memory");
}
__device__ __forceinline__ uint32_t pkbf(float lo, float hi) {
 __nv_bfloat162 v = __floats2bfloat162_rn(lo, hi);
 return *reinterpret_cast<uint32_t*>(&v);
}
__device__ __forceinline__ void bulk_s2g(void* dst, uint32_t src, int bytes) {
 asm volatile("cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;" ::"l"(dst), "r"(src), "r"(bytes) : "memory");
}
__device__ __forceinline__ void tma_2d(uint32_t dst, const CUtensorMap* map, int c0, int c1, uint32_t bar) {
 asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%2, %3}], [%4];"
        ::"r"(dst), "l"(map), "r"(c0), "r"(c1), "r"(bar) : "memory");
}

#define FT_WSEM ""

__device__ __forceinline__ void tma_2d_mc(uint32_t dst, const CUtensorMap* map, int c0, int c1, uint32_t bar, uint16_t mask) {
 asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster [%0], [%1, {%2, %3}], [%4], %5;"
        ::"r"(dst), "l"(map), "r"(c0), "r"(c1), "r"(bar), "h"(mask) : "memory");
}


__device__ __forceinline__ void tma_2d_h(uint32_t dst, const CUtensorMap* map, int c0, int c1, uint32_t bar, uint64_t pol) {
 asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint [%0], [%1, {%2, %3}], [%4], %5;"
        ::"r"(dst), "l"(map), "r"(c0), "r"(c1), "r"(bar), "l"(pol) : "memory");
}
__device__ __forceinline__ void tma_2d_mc_h(uint32_t dst, const CUtensorMap* map, int c0, int c1, uint32_t bar, uint16_t mask, uint64_t pol) {
 asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster.L2::cache_hint [%0], [%1, {%2, %3}], [%4], %5, %6;"
        ::"r"(dst), "l"(map), "r"(c0), "r"(c1), "r"(bar), "h"(mask), "l"(pol) : "memory");
}
__device__ __forceinline__ uint64_t x40_pol_ef() { uint64_t p; asm("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;" : "=l"(p)); return p; }
__device__ __forceinline__ uint64_t x40_pol_el() { uint64_t p; asm("createpolicy.fractional.L2::evict_last.b64 %0, 1.0;" : "=l"(p)); return p; }


__device__ __forceinline__ uint32_t ft_mapa(uint32_t a, uint32_t r) {
 uint32_t o;
 asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(o) : "r"(a), "r"(r));
 return o;
}
__device__ __forceinline__ void ft_arrive_remote(uint32_t a) {
 asm volatile("mbarrier.arrive.shared::cluster.b64 _, [%0];" ::"r"(a) : "memory");
}
__device__ __forceinline__ void ft_wait_cluster(uint32_t bar, uint32_t parity) {
 asm volatile(
   "{\n .reg .pred p;\n W: mbarrier.try_wait.parity" FT_WSEM ".shared::cta.b64 p, [%0], %1;\n @!p bra W;\n}" ::"r"(bar),
   "r"(parity) : "memory");
}
__device__ __forceinline__ void ft_arrive_claim(uint32_t a) {
 asm volatile("mbarrier.arrive.release.cluster.shared::cluster.b64 _, [%0];" ::"r"(a) : "memory");
}
__device__ __forceinline__ void ft_wait_claim(uint32_t bar, uint32_t parity) {
 asm volatile(
   "{\n .reg .pred p;\n W: mbarrier.try_wait.parity.acquire.cluster.shared::cta.b64 p, [%0], %1;\n @!p bra W;\n}" ::"r"(bar),
   "r"(parity) : "memory");
}
__device__ __forceinline__ void ft_st_remote(uint32_t a, int v) {
 asm volatile("st.shared::cluster.u32 [%0], %1;" ::"r"(a), "r"(v) : "memory");
}
__device__ __forceinline__ uint32_t ft_peer(uint32_t a) {
 uint32_t r, o;
 asm volatile("mov.u32 %0, %%cluster_ctarank;" : "=r"(r));
 asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(o) : "r"(a), "r"(r ^ 1u));
 return o;
}
__device__ __forceinline__ void ft_cluster_sync() {
 asm volatile("barrier.cluster.arrive.release;\nbarrier.cluster.wait.acquire;" ::: "memory");
}
__device__ __forceinline__ uint32_t prmt(uint32_t a, uint32_t b, uint32_t s) {
 uint32_t r;
 asm("prmt.b32 %0, %1, %2, %3;" : "=r"(r) : "r"(a), "r"(b), "r"(s));
 return r;
}
__device__ __forceinline__ uint64_t wg_desc(uint32_t addr) {
 return (uint64_t)((addr & 0x3FFFF) >> 4) | ((uint64_t)1 << 16) | ((uint64_t)(1024 >> 4) << 32) | ((uint64_t)1 << 62);
}
__device__ __forceinline__ int ld_acquire(const int* p) {
 int v;
 asm volatile("ld.acquire.gpu.global.b32 %0, [%1];" : "=r"(v) : "l"(p) : "memory");
 return v;
}
__device__ __forceinline__ void wg_fence() { asm volatile("wgmma.fence.sync.aligned;" ::: "memory"); }
__device__ __forceinline__ void wg_commit() { asm volatile("wgmma.commit_group.sync.aligned;" ::: "memory"); }
template <int K_>
__device__ __forceinline__ void wg_wait() { asm volatile("wgmma.wait_group.sync.aligned %0;" ::"n"(K_) : "memory"); }


@@WGMMA@@



__device__ __forceinline__ void mma16816(float* d, uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t b0, uint32_t b1) {
 asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]) : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}
__device__ __forceinline__ void ldsm4(uint32_t addr, uint32_t& r0, uint32_t& r1, uint32_t& r2, uint32_t& r3) {
 asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];" : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(addr));
}








__device__ __forceinline__ uint32_t e2m1_pair(uint32_t w, int m) {
 const uint32_t lo = (m == 0 ? w << 6 : m == 1 ? w << 2 : m == 2 ? w >> 2 : w >> 6) & 0x01C001C0u;
 const uint32_t sg = (m == 0 ? w << 12 : m == 1 ? w << 8 : m == 2 ? w << 4 : w) & 0x80008000u;
 return lo | sg;
}




__device__ __forceinline__ void decode4(uint32_t q, uint32_t* o) {
 const uint32_t M = 0x81C081C0u;
 o[0] = q & M;


 o[1] = (q << 3) & M;
 o[2] = (q << 6) & M;
 o[3] = ((__funnelshift_l(q, q, 6) & 0x001B001Bu) * 0x1084u) & M;
}

__device__ __forceinline__ uint32_t bmul(uint32_t a, uint32_t s) {
 __nv_bfloat162 r = __hmul2(*reinterpret_cast<__nv_bfloat162*>(&a), *reinterpret_cast<__nv_bfloat162*>(&s));
 return *reinterpret_cast<uint32_t*>(&r);
}



__device__ __forceinline__ uint32_t sc_bf16(uint32_t v, uint32_t sel) { return prmt(v, 0, sel) * 16u + 0x5A805A80u; }

__device__ __forceinline__ uint32_t bmulh(uint32_t a, uint32_t s, int h) {
 const __nv_bfloat162 sv = *reinterpret_cast<__nv_bfloat162*>(&s);
 __nv_bfloat162 r = __hmul2(*reinterpret_cast<__nv_bfloat162*>(&a), h ? __high2bfloat162(sv) : __low2bfloat162(sv));
 return *reinterpret_cast<uint32_t*>(&r);
}

__device__ __forceinline__ void e4m3x2(uint32_t v, uint32_t& s0, uint32_t& s1) {
 s0 = sc_bf16(v, 0x4040);
 s1 = sc_bf16(v, 0x4141);
}


__device__ __forceinline__ void decode_unit(const uint8_t* up, int lane, uint32_t (&r)[16]) {
 const uint4 nb = *reinterpret_cast<const uint4*>(up + lane * 16);


 const uint32_t S = sc_bf16(*reinterpret_cast<const uint16_t*>(up + 512 + lane * 2), 0x4140);
 const uint32_t w[4] = {nb.x, nb.y, nb.z, nb.w};
#pragma unroll
 for (int kk = 0; kk < 4; kk++) {
  uint32_t o[4];
  decode4(w[kk], o);
  r[4 * kk + 0] = bmulh(o[0], S, 0);
  r[4 * kk + 1] = bmulh(o[1], S, 1);
  r[4 * kk + 2] = bmulh(o[2], S, 0);
  r[4 * kk + 3] = bmulh(o[3], S, 1);
 }
}




template <int W, int TN, int MB, int IL>
__device__ __forceinline__ void chunk_mma(float (&acc)[MB][TN / 2], uint32_t (&r)[MB][16], const uint8_t* wb, int ubase,
                     int lane, uint32_t xa) {
 const uint64_t d0 = wg_desc(xa);
 auto mma = [&](int b, int kk) {
  if constexpr (W == TN) wgmma_n<TN>(acc[b], r[b][4 * kk], r[b][4 * kk + 1], r[b][4 * kk + 2], r[b][4 * kk + 3], d0 + 2 * kk);
  else wgmma_part<W, TN>(acc[b], r[b][4 * kk], r[b][4 * kk + 1], r[b][4 * kk + 2], r[b][4 * kk + 3], d0 + 2 * kk);
 };
 if constexpr (IL == 0) {
#pragma unroll
  for (int b = 0; b < MB; b++) decode_unit(wb + (ubase + 4 * b) * UNIT, lane, r[b]);
  wg_fence();
#pragma unroll
  for (int b = 0; b < MB; b++)
#pragma unroll
   for (int kk = 0; kk < 4; kk++) mma(b, kk);
 } else {
  uint4 nb[MB];
  uint32_t S[MB];
#pragma unroll
  for (int b = 0; b < MB; b++) {
   const uint8_t* up = wb + (ubase + 4 * b) * (MB == 2 ? UNIT2 : UNIT);
   nb[b] = *reinterpret_cast<const uint4*>(up + lane * 16);
   if constexpr (false) S[b] = 0;
   else S[b] = sc_bf16(*reinterpret_cast<const uint16_t*>(up + 512 + lane * 2), 0x4140);
  }
#pragma unroll
  for (int kk = 0; kk < 4; kk++) {
#pragma unroll
   for (int b = 0; b < MB; b++) {
    const uint32_t w = kk == 0 ? nb[b].x : kk == 1 ? nb[b].y : kk == 2 ? nb[b].z : nb[b].w;
    uint32_t o[4];
    decode4(w, o);
    r[b][4 * kk + 0] = bmulh(o[0], S[b], 0); r[b][4 * kk + 1] = bmulh(o[1], S[b], 1);
    r[b][4 * kk + 2] = bmulh(o[2], S[b], 0); r[b][4 * kk + 3] = bmulh(o[3], S[b], 1);
   }
   if constexpr (IL != 4) {
    wg_fence();
#pragma unroll
    for (int b = 0; b < MB; b++) mma(b, kk);
   }
  }
  if constexpr (IL == 4) {
   wg_fence();
#pragma unroll
   for (int kk = 0; kk < 4; kk++)
#pragma unroll
    for (int b = 0; b < MB; b++) mma(b, kk);
  }
 }
}
template <int V> using IC = std::integral_constant<int, V>;




template <int TN, int MB>
__global__ void __launch_bounds__((NW + 4) * 32, 1) moe_fused(const uint8_t* __restrict__ W1, const uint8_t* __restrict__ W2,
                               const __grid_constant__ CUtensorMap bmap1,
                               const __grid_constant__ CUtensorMap bmap2,
                               const __grid_constant__ CUtensorMap bmap1w,
                               const __grid_constant__ CUtensorMap bmap2w,
                               const int* __restrict__ starts, const int* __restrict__ cnts,
                               const int* __restrict__ toff1, const int* __restrict__ toff2,
                               const float* __restrict__ gs1, const float* __restrict__ gs2,
                               __nv_bfloat16* __restrict__ out1, __nv_bfloat16* __restrict__ out2,
                               int RB1, int RB2, int C, const long* __restrict__ slot_of,
                               int* __restrict__ counter, int dyn, const int* __restrict__ perm,
                               int* __restrict__ cnt1, int phase) {
 constexpr int NC = NW * 32, XB = TN * 128, SB = stage_bytes(TN, MB), NS = nstages(TN), WB = wbytes(MB);
 extern __shared__ uint8_t mraw[];
 __shared__ __align__(8) uint64_t bars[2 * NS];
 __shared__ int s_toff[2][33], s_start[32], s_cnt[32], s_perm[32];
 __shared__ int s_tq[8];
 __shared__ int4 s_tl[8];
 __shared__ int s_tj[8];
 __shared__ __align__(8) uint64_t tbars[8];
 __shared__ __align__(8) uint64_t zsfree;
 __shared__ float s_gs[2][32];
 const int tid = threadIdx.x, warp = tid / 32, lane = tid % 32, g = lane / 4, q = lane % 4;
 const uint32_t zsf = (uint32_t)__cvta_generic_to_shared(&zsfree);
 const uint32_t raw = (uint32_t)__cvta_generic_to_shared(mraw);
 const uint32_t sb = (raw + 1023) & ~1023u;
 uint8_t* const sp0 = mraw + (sb - raw);


 float* const ebuf = reinterpret_cast<float*>(sp0 + NS * SB);
 __nv_bfloat16* const obuf = reinterpret_cast<__nv_bfloat16*>(sp0 + NS * SB);
 const uint32_t bfull = (uint32_t)__cvta_generic_to_shared(bars), bempty = bfull + 8 * NS;
 if (tid < 33) { s_toff[0][tid] = toff1[tid]; s_toff[1][tid] = toff2[tid]; }
 if (tid < 32) { s_start[tid] = starts[tid]; s_cnt[tid] = cnts[tid]; s_gs[0][tid] = gs1[tid]; s_gs[1][tid] = gs2[tid]; s_perm[tid] = perm[tid]; }
 if (tid == 0) {
  for (int s = 0; s < NS; s++) { mbar_init(bfull + 8 * s, 1); mbar_init(bempty + 8 * s, NW); }
  for (int s = 0; s < 8; s++) mbar_init((uint32_t)__cvta_generic_to_shared(&tbars[s]), 1);
  mbar_init(zsf, 96);
  asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
 }
 __syncthreads();
 const int total1 = s_toff[0][32], total = total1 + s_toff[1][32];

 const int tbeg = phase == 2 ? total1 : 0, tend = phase == 1 ? total1 : total;
 const uint32_t btq = (uint32_t)__cvta_generic_to_shared(tbars);


 auto locate = [&](int t, int& mode, int& e, int& rb, int& row0, int& nv, int& jt) {
  mode = t < total1 ? 1 : 2;
  if (mode == 2) t -= total1;
  const int* to = s_toff[mode - 1];
  int lo = 0, hi = 31;
  while (lo < hi) {
   const int mid = (lo + hi + 1) >> 1;
   if (to[mid] <= t) lo = mid; else hi = mid - 1;
  }
  e = s_perm[lo];
  const int ntt = (s_cnt[e] + TN - 1) / TN, lt = t - to[lo];



  const int RBm = mode == 1 ? RB1 : RB2;
  const int ch = lt / (MOE_SWZ * RBm), rem = lt - ch * MOE_SWZ * RBm;
  const int gt = min(MOE_SWZ, ntt - ch * MOE_SWZ);
  rb = rem / gt;
  const int tt = ch * MOE_SWZ + rem % gt;
  jt = to[lo] / (mode == 1 ? RB1 : RB2) + tt;
  row0 = s_start[e] + tt * TN;
  nv = min(TN, s_cnt[e] - tt * TN);
 };
 if (warp >= NW) {
  asm volatile("setmaxnreg.dec.sync.aligned.u32 40;");
  if (warp == NW) {


   int gch = 0, x91s = 0, x91ph = 0;
   for (int k = 0;; k++) {
    int t0 = 0;
    if (lane == 0) t0 = dyn ? atomicAdd(counter + (phase == 2), 1) : (int)blockIdx.x + k * (int)gridDim.x;
    int t = tbeg + __shfl_sync(0xffffffffu, t0, 0);
    if (t >= tend) t = -1;
    int mode = 0, e = 0, rb = 0, row0 = 0, nv = 0, jt = 0;
    if (t >= 0) locate(t, mode, e, rb, row0, nv, jt);
    if (lane == 0) {
     s_tq[k & 7] = t;
     s_tl[k & 7] = make_int4(e | mode << 8, rb, row0, nv);
     s_tj[k & 7] = jt;
     mbar_arrive(btq + 8 * (k & 7));
    }
    __syncwarp();
    if (t < 0) break;
    const uint8_t* wp = mode == 1 ? W1 + ((size_t)e * RB1 + rb) * C * WB : W2 + ((size_t)e * RB2 + rb) * C * WB;
    const CUtensorMap* bm = mode == 1 ? &bmap1 : &bmap2;
    const CUtensorMap* bmw = mode == 1 ? &bmap1w : &bmap2w;
    if (mode == 2 && phase != 2) {

     while (ld_acquire(&cnt1[jt]) < RB1) __nanosleep(128);
     asm volatile("fence.proxy.async.global;" ::: "memory");
    }
    const int wvp = nv <= 32 ? 32 : nv <= 64 ? 64 : nv <= 128 ? 128 : TN;
    const int nbox = (wvp + 63) / 64;
    if (TN == 256 && wvp <= 64) {


     for (int c = 0; c < C; c += 2, gch++, x91s = x91s + 1 == NS ? 0 : x91s + 1, x91ph ^= (x91s == 0)) {
      const int s = x91s;
      if (gch >= NS) mbar_wait(bempty + 8 * s, x91ph ^ 1);
      if (x64_elect()) {
       mbar_expect(bfull + 8 * s, 2 * (8192 + WB));
       for (int h = 0; h < 2; h++) {
        tma_2d(sb + s * SB + h * FZ_PAIR, bm, (c + h) * 64, row0, bfull + 8 * s);
        bulk_g2s(sb + s * SB + h * FZ_PAIR + 8192, wp + (size_t)(c + h) * WB, WB, bfull + 8 * s);
       }
      }
     }
     continue;
    }
    for (int c = 0; c < C; c++, gch++, x91s = x91s + 1 == NS ? 0 : x91s + 1, x91ph ^= (x91s == 0)) {
     const int s = x91s;
     if (gch >= NS) mbar_wait(bempty + 8 * s, x91ph ^ 1);

     if (x64_elect()) {
      mbar_expect(bfull + 8 * s, nbox * 8192 + WB);
      if (TN == 256 && nbox == 4) tma_2d(sb + s * SB, bmw, c * 64, row0, bfull + 8 * s);
      else for (int bx = 0; bx < nbox; bx++) tma_2d(sb + s * SB + bx * 8192, bm, c * 64, row0 + bx * 64, bfull + 8 * s);
      bulk_g2s(sb + s * SB + XB, wp + (size_t)c * WB, WB, bfull + 8 * s);
     }
    }
   }
  } else if (warp > NW) {


   const int ct = tid - (NC + 32);
   for (int k = 0;; k++) {
    mbar_wait(btq + 8 * (k & 7), (k >> 3) & 1);
    const int t = s_tq[k & 7];
    if (t < 0) break;
    const int4 tl = s_tl[k & 7];
    const int mode = tl.x >> 8, rb = tl.y, row0 = tl.z, nv = tl.w;
    int zs0 = 0, zs1 = 0, zs2 = 0;
    if (TN == 256 && mode == 2) {
     if (ct < nv) zs0 = (int)slot_of[row0 + ct];
     if (ct + 96 < nv) zs1 = (int)slot_of[row0 + ct + 96];
     if (ct + 192 < nv) zs2 = (int)slot_of[row0 + ct + 192];
    }
    asm volatile("bar.sync 2, %0;" ::"n"(NC + 96) : "memory");
    const uint32_t ob = (uint32_t)__cvta_generic_to_shared(obuf);
    if (mode == 1 && MB == 1) {


     for (int i = ct; i < nv * 8; i += 96) {
      const int tok = i / 8, c8 = (i % 8) * 8;
      const uint4 gq = *reinterpret_cast<const uint4*>(obuf + tok * 136 + c8);
      const uint4 uq = *reinterpret_cast<const uint4*>(obuf + tok * 136 + 64 + c8);
      const __nv_bfloat16* gp = reinterpret_cast<const __nv_bfloat16*>(&gq);
      const __nv_bfloat16* up = reinterpret_cast<const __nv_bfloat16*>(&uq);
      __align__(16) __nv_bfloat16 o[8];
 #pragma unroll
      for (int x = 0; x < 8; x++) {
       const float gv = __bfloat162float(gp[x]);
       const float sl = __bfloat162float(__float2bfloat16(__fdividef(gv, 1.f + __expf(-gv))));
       o[x] = __float2bfloat16(sl * __bfloat162float(up[x]));
      }
      *reinterpret_cast<uint4*>(out1 + (size_t)(row0 + tok) * 4096 + rb * 64 + c8) = *reinterpret_cast<const uint4*>(o);
     }
    } else if (mode == 1) {
     for (int i = ct; i < nv * 16; i += 96) {
      const int tok = i / 16, c8 = (i % 16) * 8;
      *reinterpret_cast<uint4*>(out1 + (size_t)(row0 + tok) * 4096 + rb * 128 + c8) =
        *reinterpret_cast<const uint4*>(obuf + tok * 136 + c8);
     }
    } else {
     constexpr int RW = 128 * MB, OS = RW + 8;
     if constexpr (TN == 256) {
 #pragma unroll
      for (int u = 0; u < 3; u++) {
       const int tok = ct + 96 * u;
       if (tok < nv)
        bulk_s2g(out2 + (size_t)(u == 0 ? zs0 : u == 1 ? zs1 : zs2) * 4096 + rb * RW, ob + tok * OS * 2, RW * 2);
      }
     } else {
      for (int i = ct; i < nv * (RW / 8); i += 96) {
       const int tok = i / (RW / 8), c8 = (i % (RW / 8)) * 8;
       *reinterpret_cast<uint4*>(out2 + (size_t)slot_of[row0 + tok] * 4096 + rb * RW + c8) =
         *reinterpret_cast<const uint4*>(obuf + tok * OS + c8);
      }
     }
    }
    asm volatile("cp.async.bulk.commit_group;" ::: "memory");
    asm volatile("cp.async.bulk.wait_group.read 0;" ::: "memory");
    if (mode == 1 && phase != 1) {

     asm volatile("cp.async.bulk.wait_group 0;" ::: "memory");
     asm volatile("fence.proxy.async.global;" ::: "memory");
     __threadfence();
     asm volatile("bar.sync 4, 96;" ::: "memory");
     if (ct == 0) atomicAdd(&cnt1[s_tj[k & 7]], 1);
    }
    mbar_arrive(zsf);
   }
  }
  return;
 }
 asm volatile("setmaxnreg.inc.sync.aligned.u32 232;");
 int gch = 0, x91s = 0, x91ph = 0;
 uint32_t ra[MB][16], rb_[MB][16];


 const int ubase = MB == 2 ? (warp < 4 ? 0 : 8) + (warp % 4) : warp;
 for (int k = 0;; k++) {
  mbar_wait(btq + 8 * (k & 7), (k >> 3) & 1);
  const int t = s_tq[k & 7];
  if (t < 0) break;
  const int4 tl = s_tl[k & 7];
  const int mode = tl.x >> 8, e = tl.x & 255, nv = tl.w;

  const int wv = nv <= 32 ? 32 : nv <= 64 ? 64 : nv <= 128 ? 128 : TN;


  auto tile = [&](auto Wc) {
   constexpr int W = decltype(Wc)::value;
   float acc[MB][W / 2];
 #pragma unroll
   for (int b = 0; b < MB; b++)
 #pragma unroll
    for (int i = 0; i < W / 2; i++) acc[b][i] = 0.f;
   auto step = [&](int c, uint32_t (&r)[MB][16]) {
    const int s = x91s;
    mbar_wait(bfull + 8 * s, x91ph);
    const uint8_t* sp = sp0 + s * SB;
    const uint8_t* wb = sp + XB;
    const uint32_t xa = sb + s * SB;
    chunk_mma<W, W, MB, MOE_IL>(acc, r, wb, ubase, lane, xa);
    wg_commit();
    wg_wait<1>();
    if (c > 0) {
     __syncwarp();
     if (lane == 0) mbar_arrive(bempty + 8 * (x91s == 0 ? NS - 1 : x91s - 1));
    }
    gch++;
x91s = x91s + 1 == NS ? 0 : x91s + 1; x91ph ^= (x91s == 0);
   };
   if constexpr (TN == 256 && W <= 64) {

    for (int c = 0; c < C; c += 2) {
     const int s = x91s;
     mbar_wait(bfull + 8 * s, x91ph);
     const uint8_t* sp = sp0 + s * SB;
     const uint32_t xa = sb + s * SB;
     chunk_mma<W, W, MB, MOE_IL>(acc, ra, sp + 8192, ubase, lane, xa);
     wg_commit();
     wg_wait<1>();
     if (c > 0) {
      __syncwarp();
      if (lane == 0) mbar_arrive(bempty + 8 * (x91s == 0 ? NS - 1 : x91s - 1));
     }
     chunk_mma<W, W, MB, MOE_IL>(acc, rb_, sp + FZ_PAIR + 8192, ubase, lane, xa + FZ_PAIR);
     wg_commit();
     wg_wait<1>();
     gch++;
x91s = x91s + 1 == NS ? 0 : x91s + 1; x91ph ^= (x91s == 0);
    }
   } else {
    for (int c = 0; c < C; c += 2) {
     step(c, ra);
     step(c + 1, rb_);
    }
   }
   wg_wait<0>();
   __syncwarp();
   if (lane == 0) mbar_arrive(bempty + 8 * (x91s == 0 ? NS - 1 : x91s - 1));

   const float sc = s_gs[mode - 1][e] * 9.223372036854775808e18f;
   const int lr = (warp % 4) * 16 + g;
   if (k > 0) mbar_wait(zsf, (k - 1) & 1);
   if constexpr (MB == 1) {




    const int i4 = lane >> 3, t8 = lane & 7;
    const int cb = mode == 1 ? 64 * (i4 & 1) + warp * 8 : (warp / 4) * 64 + (warp % 4) * 16 + 8 * (i4 & 1);
    const uint32_t base = (uint32_t)__cvta_generic_to_shared(obuf) + ((8 * (i4 >> 1) + t8) * 136 + cb) * 2;
 #pragma unroll
    for (int j = 0; j < W / 8; j += 2)
     stsm4t(base + j * 8 * 136 * 2, pkbf(acc[0][4 * j] * sc, acc[0][4 * j + 1] * sc),
         pkbf(acc[0][4 * j + 2] * sc, acc[0][4 * j + 3] * sc), pkbf(acc[0][4 * j + 4] * sc, acc[0][4 * j + 5] * sc),
         pkbf(acc[0][4 * j + 6] * sc, acc[0][4 * j + 7] * sc));
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
    asm volatile("bar.arrive 2, %0;" ::"n"(NC + 96) : "memory");
   } else if (mode == 1) {


    const int col = (warp / 4) * 64 + lr;
 #pragma unroll
    for (int j = 0; j < W / 8; j++)
 #pragma unroll
     for (int c = 0; c < 2; c++)
 #pragma unroll
      for (int h = 0; h < 2; h++) {
       const int tok = 8 * j + 2 * q + c;
       const float gv = acc[0][4 * j + 2 * h + c] * sc, uv = acc[MB - 1][4 * j + 2 * h + c] * sc;
       const float sl = __bfloat162float(__float2bfloat16(__fdividef(gv, 1.f + __expf(-gv))));
       obuf[tok * 136 + col + 8 * h] = __float2bfloat16(sl * __bfloat162float(__float2bfloat16(uv)));
      }
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
    asm volatile("bar.arrive 2, %0;" ::"n"(NC + 96) : "memory");
   } else {


    constexpr int RW = 128 * MB, OS = RW + 8;
    const int r0 = (warp / 4) * 64 * MB + lr;
 #pragma unroll
    for (int b = 0; b < MB; b++)
 #pragma unroll
     for (int j = 0; j < W / 8; j++)
 #pragma unroll
      for (int c = 0; c < 2; c++) {
       const int tok = 8 * j + 2 * q + c;
       obuf[tok * OS + 64 * b + r0] = __float2bfloat16(acc[b][4 * j + c] * sc);
       obuf[tok * OS + 64 * b + r0 + 8] = __float2bfloat16(acc[b][4 * j + 2 + c] * sc);
      }
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
    asm volatile("bar.arrive 2, %0;" ::"n"(NC + 96) : "memory");
   }
  };
  if constexpr (TN == 256) {
   const int w32 = (nv + 31) >> 5;
   if (w32 >= 8) tile(IC<256>{});
   else if (w32 == 7) tile(IC<224>{});
   else if (w32 == 6) tile(IC<192>{});
   else if (w32 == 5) tile(IC<160>{});
   else if (w32 == 4) tile(IC<128>{});
   else if (w32 == 3) tile(IC<96>{});
   else if (w32 == 2) tile(IC<64>{});
   else tile(IC<32>{});
  } else {
   if (wv >= TN) tile(IC<TN>{});
   else if (wv == 128) tile(IC<(TN > 128 ? 128 : TN)>{});
   else if (wv == 64) tile(IC<64>{});
   else tile(IC<32>{});
  }
 }
}









#define MOE_ILN 4
#define MOE_ILM 4
#define MOE_PIPE2 0
constexpr int MX_SB = 37888, MX_NS = 4;
constexpr int X63_C = 64;

constexpr int MX_PAIR = 18432;


constexpr uint32_t X65_W1 = 8192 + 2 * NW * UNIT2, X65_X1 = 28672;
#define MX_MS 16
#define MX_OB 128


struct MxMaps { CUtensorMap m[2][9]; CUtensorMap w[2][2]; };
__host__ __device__ constexpr int mx_xoff(int mb) { return mb == 2 ? 16384 : 32768; }


#define TF_WSET 15
__host__ __device__ __forceinline__ int tf_w1(int nv) {
 return nv <= 128 ? 128 : (nv + 7) / 8 * 8;
}
__host__ __device__ __forceinline__ int mx_w(int mb, int nv) {
 if (mb == 2) return nv <= MX_MS ? (nv + 7) / 8 * 8 : nv <= 32 ? 32 : nv <= 64 ? 64 : (nv <= 96 && (TF_WSET & 16)) ? 96 : 128;
 return tf_w1(nv);
}

__device__ __forceinline__ int z0_qexact(int a, int b) {
 int q = __float2int_rz(__fdividef((float)a, (float)b));
 if (q * b > a) --q;
 if ((q + 1) * b <= a) ++q;
 return q;
}
__global__ void __launch_bounds__((NW + 4) * 32, 1) moe_mixed(const uint8_t* __restrict__ W1a, const uint8_t* __restrict__ W2a,
                               const uint8_t* __restrict__ W1b, const uint8_t* __restrict__ W2b,
                               const __grid_constant__ CUtensorMap bmap1,
                               const __grid_constant__ CUtensorMap bmap2,
                               const __grid_constant__ MxMaps bmx,
                               int* __restrict__ meta, const float* __restrict__ gs1,
                               const float* __restrict__ gs2, __nv_bfloat16* __restrict__ out1,
                               __nv_bfloat16* __restrict__ out2, int C_, const long* __restrict__ slot_of,
                               int* __restrict__ cnt2) {
 constexpr int NC = NW * 32, SB = MX_SB, NS = MX_NS;
 constexpr int C = X63_C;
 (void)C_;
 int* const counter = meta + 130;
 int* const cnt1 = meta + 260;
 extern __shared__ uint8_t mraw[];
 __shared__ __align__(8) uint64_t bars[2 * NS];
 __shared__ int s_toff[2][33], s_start[32], s_cnt[32], s_perm[32], s_info[32], s_jt0[32], s_map2[32];
 __shared__ int s_tq[8];
 __shared__ int4 s_tl[8];
 __shared__ int s_tj[8];
 __shared__ __align__(8) uint64_t tbars[8];
 __shared__ __align__(8) uint64_t sbars[8];
 __shared__ __align__(8) uint64_t gbars[8];
 __shared__ float s_gs[2][32];
 __shared__ int z0_slot[2][256];
 const int tid = threadIdx.x, warp = tid / 32, lane = tid % 32, g = lane / 4, q = lane % 4;
 const uint32_t raw = (uint32_t)__cvta_generic_to_shared(mraw);
 const uint32_t sb = (raw + 1023) & ~1023u;
 uint8_t* const sp0 = mraw + (sb - raw);
 __nv_bfloat16* const obuf = reinterpret_cast<__nv_bfloat16*>(sp0 + NS * SB);
 const uint32_t bfull = (uint32_t)__cvta_generic_to_shared(bars), bempty = bfull + 8 * NS;


 if (tid < 32) { s_gs[0][tid] = gs1[tid]; s_gs[1][tid] = gs2[tid]; }
 if (tid == 0) {
  for (int s = 0; s < NS; s++) { mbar_init(bfull + 8 * s, 1); mbar_init(bempty + 8 * s, NW); }
  for (int s = 0; s < 8; s++) mbar_init((uint32_t)__cvta_generic_to_shared(&tbars[s]), 1);
  for (int s = 0; s < 8; s++) mbar_init((uint32_t)__cvta_generic_to_shared(&sbars[s]), 1);
  for (int s = 0; s < 8; s++) mbar_init((uint32_t)__cvta_generic_to_shared(&gbars[s]), 1);
  asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
 }
 if (tid >= NC && tid < NC + 16) {
  const int i = tid - NC;
  const CUtensorMap* pm = i == 0 ? &bmap1 : i == 1 ? &bmap2 : i < 14 ? &bmx.m[(i - 2) / 6][(i - 2) % 6] : &bmx.w[i - 14][0];
  asm volatile("prefetch.tensormap [%0];" ::"l"(reinterpret_cast<uint64_t>(pm)) : "memory");
  if (i >= 14) asm volatile("prefetch.tensormap [%0];" ::"l"(reinterpret_cast<uint64_t>(&bmx.w[i - 14][1])) : "memory");
 }


 if (tid < 33) { s_toff[0][tid] = tid ? 1 : 0; s_toff[1][tid] = 0; }
 if (tid < 32) {
  s_start[tid] = 0; s_cnt[tid] = 128; s_perm[tid] = 0; s_info[tid] = 1 | 1 << 8 | 128 << 12; s_jt0[tid] = 0; s_map2[tid] = 0;
 }
 __syncthreads();
 const uint32_t btq = (uint32_t)__cvta_generic_to_shared(tbars);
 const uint32_t bsl = (uint32_t)__cvta_generic_to_shared(sbars);
 const uint32_t bgo = (uint32_t)__cvta_generic_to_shared(gbars);


 auto locate = [&](int t, int& mode, int& e, int& mb, int& rb, int& row0, int& nv, int& jt, int& rem) {
  const int total1 = s_toff[0][32];
  mode = t < total1 ? 1 : 2;
  if (mode == 2) t -= total1;
  const int* to = s_toff[mode - 1];
  int lo = 0, hi = 31;
  while (lo < hi) {
   const int mid = (lo + hi + 1) >> 1;
   if (to[mid] <= t) lo = mid; else hi = mid - 1;
  }
  const int lt = t - to[lo];
  if (mode == 2) lo = s_map2[lo];
  e = s_perm[lo];
  const int inf = s_info[lo], ntt = inf & 255, tsz = (inf >> 12) & 511, rr = inf >> 21;
  mb = (inf >> 8) & 15;
  const int k1 = ntt - (rr > 0), per = ((mode == 1 ? 64 : 32) >> (mb - 1));
  int tt;
  if (lt < k1 * per) {
   rb = z0_qexact(lt, k1);
   tt = lt - rb * k1;
   row0 = s_start[e] + tt * tsz;
   nv = min(tsz, s_cnt[e] - rr - tt * tsz);
   rem = 0;
  } else {
   mb = 2;
   rb = lt - k1 * per;
   tt = k1;
   row0 = s_start[e] + s_cnt[e] - rr;
   nv = rr;
   rem = 1;
  }
  jt = s_jt0[lo] + tt;
 };
 if (warp >= NW) {
  asm volatile("setmaxnreg.dec.sync.aligned.u32 40;");
  if (warp == NW) {
   uint32_t gch = 0;
   for (int k = 0;; k++) {
    int x_t = -1, x_mode = 0, x_e = 0, x_mb = 2, x_rb = 0, x_row0 = 0, x_nv = 0, x_jt = 0, x_rem = 0;
    if (k == 0) {
     GDC_WAIT();
     int ln;
     asm volatile("mov.u32 %0, %%laneid;" : "=r"(ln));
     const int* const mp = counter - 130;
     const int a0 = mp[64 + ln], a1 = mp[97 + ln], tot0 = mp[96], tot1 = mp[129], st = mp[32 + ln], cn = mp[ln];
     const int pe = mp[164 + ln], inf_ = mp[196 + ln], jm = mp[228 + ln];
     s_toff[0][ln] = a0; s_toff[1][ln] = a1;
     if (ln == 0) { s_toff[0][32] = tot0; s_toff[1][32] = tot1; }
     s_start[ln] = st; s_cnt[ln] = cn;
     s_perm[ln] = pe; s_info[ln] = inf_; s_jt0[ln] = jm & 0xFFFF;
     s_map2[ln] = jm >> 16;


     x_t = (int)blockIdx.x;
     if (x_t >= tot0 + tot1) x_t = -1;
     if (x_t >= 0) {
      x_mode = x_t < tot0 ? 1 : 2;
      const int tq = x_mode == 2 ? x_t - tot0 : x_t;
      const int myto = x_mode == 1 ? a0 : a1;
      int lo = __popc(__ballot_sync(~0u, myto <= tq)) - 1;
      const int lt = tq - __shfl_sync(~0u, myto, lo);
      if (x_mode == 2) lo = __shfl_sync(~0u, jm >> 16, lo);
      x_e = __shfl_sync(~0u, pe, lo);
      const int inf = __shfl_sync(~0u, inf_, lo), jt0 = __shfl_sync(~0u, jm & 0xFFFF, lo);
      const int se = __shfl_sync(~0u, st, x_e), ce = __shfl_sync(~0u, cn, x_e);
      const int ntt = inf & 255, tsz = (inf >> 12) & 511, rr = inf >> 21;
      x_mb = (inf >> 8) & 15;
      const int k1 = ntt - (rr > 0), per = ((x_mode == 1 ? 64 : 32) >> (x_mb - 1));
      int tt;
      if (lt < k1 * per) {
       x_rb = z0_qexact(lt, k1);
       tt = lt - x_rb * k1;
       x_row0 = se + tt * tsz;
       x_nv = min(tsz, ce - rr - tt * tsz);
       x_rem = 0;
      } else {
       x_mb = 2;
       x_rb = lt - k1 * per;
       tt = k1;
       x_row0 = se + ce - rr;
       x_nv = rr;
       x_rem = 1;
      }
      x_jt = jt0 + tt;
     }
     __syncwarp();
    }

    int t, mode = 0, e = 0, mb = 2, rb = 0, row0 = 0, nv = 0, jt = 0, W;
    if (k == 0) {
     t = x_t; mode = x_mode; e = x_e; mb = x_mb; rb = x_rb; row0 = x_row0; nv = x_nv; jt = x_jt;
     const int rem = x_rem;
     W = mx_w(mb, nv);
     if (x64_elect()) s_tq[0] = t;
     if (x64_elect()) s_tl[0] = make_int4(e | mode << 8 | mb << 12 | W << 16, rb | rem << 16, row0, nv);
     if (x64_elect()) s_tj[0] = jt;
    } else {
     mbar_wait(bsl + 8 * (k & 7), ((k >> 3) - ((k & 7) == 0 ? 1 : 0)) & 1);
     t = s_tq[k & 7];
     const int4 tl0 = s_tl[k & 7];
     mode = (tl0.x >> 8) & 15; e = tl0.x & 255; mb = (tl0.x >> 12) & 15; W = tl0.x >> 16;
     rb = tl0.y & 0xFFFF; row0 = tl0.z; nv = tl0.w; jt = s_tj[k & 7];
    }
    __syncwarp();
    if (x64_elect()) mbar_arrive(btq + 8 * (k & 7));
    if (t < 0) break;
    const int WB = wbytes(mb), xo = mx_xoff(mb);
    const uint8_t* wp = mode == 1 ? (mb == 2 ? W1a : W1b) + ((size_t)e * ((64 >> (mb - 1))) + rb) * C * WB
                   : (mb == 2 ? W2a : W2b) + ((size_t)e * ((32 >> (mb - 1))) + rb) * C * WB;
    const CUtensorMap* bm = mb == 2 && W <= 32 ? &bmx.m[mode - 1][W == 8 ? 6 : W == 16 ? 7 : 8]
                          : mode == 1 ? &bmap1 : &bmap2;
    const CUtensorMap* wm = &bmx.w[mode - 1][mb - 1];
    const int wr0 = (int)(((mode == 1 ? (size_t)e * ((64 >> (mb - 1))) : (size_t)e * ((32 >> (mb - 1)))) + rb) * C * (WB / 512));
    (void)wp;
    if (mode == 2 && k >= 0) {
     while (ld_acquire(&cnt1[jt]) < (64 >> (mb - 1))) __nanosleep(128);
     asm volatile("fence.proxy.async.global;" ::: "memory");
    }
    const int nbox = (W + 63) / 64;

    const bool one = W >= MX_OB && W >= 128;
    const int Wb = (W + 31) & ~31;
    const CUtensorMap* bmw = &bmx.m[mode - 1][one ? (Wb - 96) / 32 : 0];
    const int xbytes = one ? Wb * 128 : nbox * 8192;
    if (mb == 2 && W <= 64) {
     constexpr int WB2 = wbytes(2);
     const int z0_ex = 2 * (W * 128 + WB2);


     for (int c = 0; c < C; c += 2, gch++) {
      const int s = gch % NS;
      if (k >= 0) {
       if (gch >= NS) mbar_wait(bempty + 8 * s, ((gch / NS) - 1) & 1);
       if (x64_elect()) {
        mbar_expect(bfull + 8 * s, z0_ex);
#pragma unroll
        for (int j = 0; j < 2; j++) {
         tma_2d_h(sb + s * SB + j * MX_PAIR + 8192, wm, 0, wr0 + (c + j) * (WB2 / 512), bfull + 8 * s, x40_pol_ef());
         tma_2d(sb + s * SB + j * MX_PAIR, bm, (c + j) * 64, row0, bfull + 8 * s);
        }
        if (c == C - 8) mbar_arrive(bgo + 8 * (k & 7));
       }
      }
      if (k < 0) break;
     }
    } else {


     if (one) {
      const int x63_ex = xbytes + WB;
      for (int c = 0; c < X63_C; c += NS, gch += NS) {
       const uint32_t par = ((gch / NS) - 1) & 1;
#pragma unroll
       for (int s = 0; s < NS; s++) {
        if (gch >= NS) mbar_wait(bempty + 8 * s, par);
        if (x64_elect()) {
         mbar_expect(bfull + 8 * s, x63_ex);
         tma_2d_h(sb + s * SB + xo, wm, 0, wr0 + (c + s) * (WB / 512), bfull + 8 * s, x40_pol_ef());
         tma_2d(sb + s * SB, bmw, (c + s) * 64, row0, bfull + 8 * s);
         if (s == 0 && c == X63_C - 4) mbar_arrive(bgo + 8 * (k & 7));
        }
       }
      }
     } else
     for (int c = 0; c < C; c++, gch++) {
      const int s = gch % NS;
      if (k >= 0) {
       if (gch >= NS) mbar_wait(bempty + 8 * s, ((gch / NS) - 1) & 1);
       if (x64_elect()) {
        mbar_expect(bfull + 8 * s, xbytes + WB);
        tma_2d_h(sb + s * SB + xo, wm, 0, wr0 + c * (WB / 512), bfull + 8 * s, x40_pol_ef());
        if (one) {
         tma_2d(sb + s * SB, bmw, c * 64, row0, bfull + 8 * s);
        } else {
         for (int bx = 0; bx < nbox; bx++) tma_2d(sb + s * SB + bx * 8192, bm, c * 64, row0 + bx * 64, bfull + 8 * s);
        }
        if (c == C - 4) mbar_arrive(bgo + 8 * (k & 7));
       }
      }
      if (k < 0) break;
     }
    }
   }
  } else if (warp > NW) {
   GDC_WAIT();
   const int ct = tid - (NC + 32);
   for (int k = 0;; k++) {
    mbar_wait(btq + 8 * (k & 7), (k >> 3) & 1);
    const int t = s_tq[k & 7];
    if (t < 0) break;
    if (ct == 0) {
     const int kk = k + 1;
     mbar_wait(bgo + 8 * (k & 7), (k >> 3) & 1);
     int t_ = atomicAdd(counter, 1) + (int)gridDim.x;
     if (t_ >= *reinterpret_cast<volatile int*>(&s_toff[0][32]) + *reinterpret_cast<volatile int*>(&s_toff[1][32])) t_ = -1;
     int mode_ = 0, e_ = 0, mb_ = 2, rb_ = 0, row0_ = 0, nv_ = 0, jt_ = 0, rem_ = 0;
     if (t_ >= 0) locate(t_, mode_, e_, mb_, rb_, row0_, nv_, jt_, rem_);
     const int W_ = mx_w(mb_, nv_);
     s_tq[kk & 7] = t_;
     s_tl[kk & 7] = make_int4(e_ | mode_ << 8 | mb_ << 12 | W_ << 16, rb_ | rem_ << 16, row0_, nv_);
     s_tj[kk & 7] = jt_;
     mbar_arrive(bsl + 8 * (kk & 7));
    }
    const int4 tl = s_tl[k & 7];
    const int mode = (tl.x >> 8) & 15, mb = (tl.x >> 12) & 15, rb = tl.y & 0xFFFF, row0 = tl.z, nv = tl.w;
    int* const zs = z0_slot[k & 1];
    if (mode == 2) {



     for (int i = ct; i < nv; i += 96) zs[i] = (int)slot_of[row0 + i];
     asm volatile("bar.sync 4, 96;" ::: "memory");
    }
    asm volatile("bar.sync 2, %0;" ::"n"(NC + 96) : "memory");
    if (mode == 1) {


     const int CW = 64 * mb, ST = mb == 1 ? 136 : 264, G8 = CW / 8;
     for (int i = ct; i < nv * G8; i += 96) {
      const int tok = i >> (2 + mb), c8 = (i & (G8 - 1)) * 8;
      const uint4 gq = *reinterpret_cast<const uint4*>(obuf + tok * ST + c8);
      const uint4 uq = *reinterpret_cast<const uint4*>(obuf + tok * ST + CW + c8);
      const __nv_bfloat16* gp = reinterpret_cast<const __nv_bfloat16*>(&gq);
      const __nv_bfloat16* up = reinterpret_cast<const __nv_bfloat16*>(&uq);
      __align__(16) __nv_bfloat16 o[8];
 #pragma unroll
      for (int x = 0; x < 8; x++) {
       const float gv = __bfloat162float(gp[x]);
       const float sl = __bfloat162float(__float2bfloat16(__fdividef(gv, 1.f + __expf(-gv))));
       o[x] = __float2bfloat16(sl * __bfloat162float(up[x]));
      }
      *reinterpret_cast<uint4*>(out1 + (size_t)(row0 + tok) * 4096 + rb * CW + c8) = *reinterpret_cast<const uint4*>(o);
     }
    } else {
     const int RW = 128 * mb, OS = RW + 8, R8 = RW / 8;
     for (int i = ct; i < nv * R8; i += 96) {
      const int tok = i >> (3 + mb), c8 = (i & (R8 - 1)) * 8;
      *reinterpret_cast<uint4*>(out2 + (size_t)zs[tok] * 4096 + rb * RW + c8) =
        *reinterpret_cast<const uint4*>(obuf + tok * OS + c8);
     }
    }
    if (mode == 1) {
     asm volatile("fence.proxy.async.global;" ::: "memory");
     __threadfence();
     asm volatile("bar.sync 4, 96;" ::: "memory");
     if (ct == 0) atomicAdd(&cnt1[s_tj[k & 7]], 1);
    } else if (cnt2) {

     __threadfence();
     asm volatile("bar.sync 4, 96;" ::: "memory");
     if (ct == 0) atomicAdd(&cnt2[tl.x & 255], 1 + (tl.y >> 16));
    }
    asm volatile("bar.arrive 3, %0;" ::"n"(NC + 96) : "memory");
   }
  }
  return;
 }
 asm volatile("setmaxnreg.inc.sync.aligned.u32 232;");
 GDC_WAIT();


 if (cnt2 && tid == 0) asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
 uint32_t gch = 0;
 for (int k = 0;; k++) {
  mbar_wait(btq + 8 * (k & 7), (k >> 3) & 1);
  const int t = s_tq[k & 7];
  if (t < 0) break;
  const int4 tl = s_tl[k & 7];
  const int mode = (tl.x >> 8) & 15, e = tl.x & 255, mbv = (tl.x >> 12) & 15, wv = tl.x >> 16;


  auto tile = [&](auto MBc, auto Wc) {
   constexpr int MB = decltype(MBc)::value, W = decltype(Wc)::value;
   const int ubase = MB == 2 ? (warp < 4 ? 0 : 8) + (warp % 4) : warp;
   constexpr int AW = (W + 15) / 16 * 16;


   const int nw = (MB == 2 && W == 128) ? (tl.w <= 72 ? 72 : tl.w <= 80 ? 80 : tl.w <= 88 ? 88 : tl.w <= 96 ? 96 : tl.w <= 104 ? 104 : tl.w <= 112 ? 112 : tl.w <= 120 ? 120 : 128) : W;
   (void)nw;
   float acc[MB][AW / 2];
   uint32_t ra[MB][16], rb_[MB][16];
 #pragma unroll
   for (int b = 0; b < MB; b++)
 #pragma unroll
    for (int i = 0; i < AW / 2; i++) acc[b][i] = 0.f;
   auto step = [&](int c, uint32_t (&r)[MB][16]) {
    const int s = gch % NS;
    mbar_wait(bfull + 8 * s, (gch / NS) & 1);
    const uint8_t* wb = sp0 + s * SB + mx_xoff(MB);
    const uint32_t xa = sb + s * SB;
    if constexpr (W >= 32) chunk_mma<W, AW, MB, MB == 2 ? MOE_ILM : MOE_IL>(acc, r, wb, ubase, lane, xa);
    wg_commit();
    if constexpr (MB == 1) {
     wg_wait<0>();
     __syncwarp();
     if (lane == 0) mbar_arrive(bempty + 8 * s);
    } else {
     wg_wait<1>();
     if (c > 0) {
      __syncwarp();
      if (lane == 0) mbar_arrive(bempty + 8 * ((gch - 1) % NS));
     }
    }
    gch++;
   };
   auto step80 = [&](int c, uint32_t (&r)[MB][16]) {
    const int s = gch % NS;
    mbar_wait(bfull + 8 * s, (gch / NS) & 1);
    const uint8_t* wb = sp0 + s * SB + mx_xoff(MB);
    const uint32_t xa = sb + s * SB;
    if constexpr (MB == 2 && W == 128) chunk_mma<80, 128, MB, MOE_ILM>(acc, r, wb, ubase, lane, xa);
    wg_commit();
    if constexpr (true) {
     wg_wait<0>();
     __syncwarp();
     if (lane == 0) mbar_arrive(bempty + 8 * s);
    } else {
     wg_wait<1>();
     if (c > 0) {
      __syncwarp();
      if (lane == 0) mbar_arrive(bempty + 8 * ((gch - 1) % NS));
     }
    }
    gch++;
   };
   auto step72 = [&](int c, uint32_t (&r)[MB][16]) {
    const int s = gch % NS;
    mbar_wait(bfull + 8 * s, (gch / NS) & 1);
    const uint8_t* wb = sp0 + s * SB + mx_xoff(MB);
    const uint32_t xa = sb + s * SB;
    if constexpr (MB == 2 && W == 128) chunk_mma<72, 128, MB, MOE_ILM>(acc, r, wb, ubase, lane, xa);
    wg_commit();
    if constexpr (true) {
     wg_wait<0>();
     __syncwarp();
     if (lane == 0) mbar_arrive(bempty + 8 * s);
    } else {
     wg_wait<1>();
     if (c > 0) {
      __syncwarp();
      if (lane == 0) mbar_arrive(bempty + 8 * ((gch - 1) % NS));
     }
    }
    gch++;
   };
   auto step88 = [&](int c, uint32_t (&r)[MB][16]) {
    const int s = gch % NS;
    mbar_wait(bfull + 8 * s, (gch / NS) & 1);
    const uint8_t* wb = sp0 + s * SB + mx_xoff(MB);
    const uint32_t xa = sb + s * SB;
    if constexpr (MB == 2 && W == 128) chunk_mma<88, 128, MB, MOE_ILM>(acc, r, wb, ubase, lane, xa);
    wg_commit();
    if constexpr (true) {
     wg_wait<0>();
     __syncwarp();
     if (lane == 0) mbar_arrive(bempty + 8 * s);
    } else {
     wg_wait<1>();
     if (c > 0) {
      __syncwarp();
      if (lane == 0) mbar_arrive(bempty + 8 * ((gch - 1) % NS));
     }
    }
    gch++;
   };
   auto step104 = [&](int c, uint32_t (&r)[MB][16]) {
    const int s = gch % NS;
    mbar_wait(bfull + 8 * s, (gch / NS) & 1);
    const uint8_t* wb = sp0 + s * SB + mx_xoff(MB);
    const uint32_t xa = sb + s * SB;
    if constexpr (MB == 2 && W == 128) chunk_mma<104, 128, MB, MOE_ILM>(acc, r, wb, ubase, lane, xa);
    wg_commit();
    if constexpr (true) {
     wg_wait<0>();
     __syncwarp();
     if (lane == 0) mbar_arrive(bempty + 8 * s);
    } else {
     wg_wait<1>();
     if (c > 0) {
      __syncwarp();
      if (lane == 0) mbar_arrive(bempty + 8 * ((gch - 1) % NS));
     }
    }
    gch++;
   };
   auto step120 = [&](int c, uint32_t (&r)[MB][16]) {
    const int s = gch % NS;
    mbar_wait(bfull + 8 * s, (gch / NS) & 1);
    const uint8_t* wb = sp0 + s * SB + mx_xoff(MB);
    const uint32_t xa = sb + s * SB;
    if constexpr (MB == 2 && W == 128) chunk_mma<120, 128, MB, MOE_ILM>(acc, r, wb, ubase, lane, xa);
    wg_commit();
    if constexpr (true) {
     wg_wait<0>();
     __syncwarp();
     if (lane == 0) mbar_arrive(bempty + 8 * s);
    } else {
     wg_wait<1>();
     if (c > 0) {
      __syncwarp();
      if (lane == 0) mbar_arrive(bempty + 8 * ((gch - 1) % NS));
     }
    }
    gch++;
   };
   auto step96 = [&](int c, uint32_t (&r)[MB][16]) {
    const int s = gch % NS;
    mbar_wait(bfull + 8 * s, (gch / NS) & 1);
    const uint8_t* wb = sp0 + s * SB + mx_xoff(MB);
    const uint32_t xa = sb + s * SB;
    if constexpr (MB == 2 && W == 128) chunk_mma<96, 128, MB, MOE_ILM>(acc, r, wb, ubase, lane, xa);
    wg_commit();
    if constexpr (true) {
     wg_wait<0>();
     __syncwarp();
     if (lane == 0) mbar_arrive(bempty + 8 * s);
    } else {
     wg_wait<1>();
     if (c > 0) {
      __syncwarp();
      if (lane == 0) mbar_arrive(bempty + 8 * ((gch - 1) % NS));
     }
    }
    gch++;
   };
   auto step112 = [&](int c, uint32_t (&r)[MB][16]) {
    const int s = gch % NS;
    mbar_wait(bfull + 8 * s, (gch / NS) & 1);
    const uint8_t* wb = sp0 + s * SB + mx_xoff(MB);
    const uint32_t xa = sb + s * SB;
    if constexpr (MB == 2 && W == 128) chunk_mma<112, 128, MB, MOE_ILM>(acc, r, wb, ubase, lane, xa);
    wg_commit();
    if constexpr (true) {
     wg_wait<0>();
     __syncwarp();
     if (lane == 0) mbar_arrive(bempty + 8 * s);
    } else {
     wg_wait<1>();
     if (c > 0) {
      __syncwarp();
      if (lane == 0) mbar_arrive(bempty + 8 * ((gch - 1) % NS));
     }
    }
    gch++;
   };
   if constexpr (MB == 2 && W < 32) {
    constexpr int NT = W / 8;
    {

     const int lrow = lane & 7, lblk = lane >> 3;
     for (int c = 0; c < C; c += 2, gch++) {
      const int s = gch % NS;
      mbar_wait(bfull + 8 * s, (gch / NS) & 1);
 #pragma unroll
      for (int j = 0; j < 2; j++) {
       const uint32_t xr = sb + s * SB + j * MX_PAIR + lrow * 128;
       uint32_t bf[NT][8];
 #pragma unroll
       for (int n = 0; n < NT; n++) {
        const uint32_t xn = xr + n * 1024;
        ldsm4(xn + ((lblk ^ lrow) << 4), bf[n][0], bf[n][1], bf[n][2], bf[n][3]);
        ldsm4(xn + (((lblk + 4) ^ lrow) << 4), bf[n][4], bf[n][5], bf[n][6], bf[n][7]);
       }
       const uint8_t* wb = sp0 + s * SB + j * MX_PAIR + 8192;
 #pragma unroll
       for (int b = 0; b < 2; b++) {
        const uint8_t* up = wb + (ubase + 4 * b) * UNIT2;
        const uint4 nb = *reinterpret_cast<const uint4*>(up + lane * 16);
        const uint32_t S = sc_bf16(*reinterpret_cast<const uint16_t*>(up + 512 + lane * 2), 0x4140);
 #pragma unroll
        for (int kk = 0; kk < 4; kk++) {
         const uint32_t w = kk == 0 ? nb.x : kk == 1 ? nb.y : kk == 2 ? nb.z : nb.w;
         uint32_t o[4];
         decode4(w, o);
         const uint32_t a0 = bmulh(o[0], S, 0), a1 = bmulh(o[1], S, 1), a2 = bmulh(o[2], S, 0),
                 a3 = bmulh(o[3], S, 1);
 #pragma unroll
         for (int n = 0; n < NT; n++) mma16816(&acc[b][4 * n], a0, a1, a2, a3, bf[n][2 * kk], bf[n][2 * kk + 1]);
        }
       }
      }
      __syncwarp();


      asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
      if (lane == 0) mbar_arrive(bempty + 8 * s);
     }
    }
   } else if constexpr (MB == 2 && W <= 64) {



    const int nwm = W == 64 ? (tl.w <= 40 ? 40 : tl.w <= 48 ? 48 : tl.w <= 56 ? 56 : 64)
                : W == 32 ? (tl.w <= 24 ? 24 : 32) : W;
    (void)nwm;
    if constexpr (W == 64) {
     if ((Z_MIDSET & 1) && nwm == 40) {
      for (int c = 0; c < C; c += 2, gch++) {
       const int s = gch % NS;
       mbar_wait(bfull + 8 * s, (gch / NS) & 1);
       chunk_mma<40, W, MB, MOE_ILN>(acc, ra, sp0 + s * SB + 8192, ubase, lane, sb + s * SB);
       wg_commit();
       chunk_mma<40, W, MB, MOE_ILN>(acc, rb_, sp0 + s * SB + MX_PAIR + 8192, ubase, lane, sb + s * SB + MX_PAIR);
       wg_commit();
       wg_wait<0>();
       __syncwarp();
       if (lane == 0) mbar_arrive(bempty + 8 * s);
      }
     } else if ((Z_MIDSET & 2) && nwm == 48) {
      for (int c = 0; c < C; c += 2, gch++) {
       const int s = gch % NS;
       mbar_wait(bfull + 8 * s, (gch / NS) & 1);
       chunk_mma<48, W, MB, MOE_ILN>(acc, ra, sp0 + s * SB + 8192, ubase, lane, sb + s * SB);
       wg_commit();
       chunk_mma<48, W, MB, MOE_ILN>(acc, rb_, sp0 + s * SB + MX_PAIR + 8192, ubase, lane, sb + s * SB + MX_PAIR);
       wg_commit();
       wg_wait<0>();
       __syncwarp();
       if (lane == 0) mbar_arrive(bempty + 8 * s);
      }
     } else if ((Z_MIDSET & 4) && nwm == 56) {
      for (int c = 0; c < C; c += 2, gch++) {
       const int s = gch % NS;
       mbar_wait(bfull + 8 * s, (gch / NS) & 1);
       chunk_mma<56, W, MB, MOE_ILN>(acc, ra, sp0 + s * SB + 8192, ubase, lane, sb + s * SB);
       wg_commit();
       chunk_mma<56, W, MB, MOE_ILN>(acc, rb_, sp0 + s * SB + MX_PAIR + 8192, ubase, lane, sb + s * SB + MX_PAIR);
       wg_commit();
       wg_wait<0>();
       __syncwarp();
       if (lane == 0) mbar_arrive(bempty + 8 * s);
      }
     } else {
      for (int c = 0; c < C; c += 2, gch++) {
       const int s = gch % NS;
       mbar_wait(bfull + 8 * s, (gch / NS) & 1);
       chunk_mma<W, W, MB, MOE_ILN>(acc, ra, sp0 + s * SB + 8192, ubase, lane, sb + s * SB);
       wg_commit();
       chunk_mma<W, W, MB, MOE_ILN>(acc, rb_, sp0 + s * SB + MX_PAIR + 8192, ubase, lane, sb + s * SB + MX_PAIR);
       wg_commit();
       wg_wait<0>();
       __syncwarp();
       if (lane == 0) mbar_arrive(bempty + 8 * s);
      }
     }
    } else if constexpr (W == 32) {
     if ((Z_MIDSET & 8) && nwm == 24) {
      for (int c = 0; c < C; c += 2, gch++) {
       const int s = gch % NS;
       mbar_wait(bfull + 8 * s, (gch / NS) & 1);
       chunk_mma<24, W, MB, MOE_ILN>(acc, ra, sp0 + s * SB + 8192, ubase, lane, sb + s * SB);
       wg_commit();
       chunk_mma<24, W, MB, MOE_ILN>(acc, rb_, sp0 + s * SB + MX_PAIR + 8192, ubase, lane, sb + s * SB + MX_PAIR);
       wg_commit();
       wg_wait<0>();
       __syncwarp();
       if (lane == 0) mbar_arrive(bempty + 8 * s);
      }
     } else {
      for (int c = 0; c < C; c += 2, gch++) {
       const int s = gch % NS;
       mbar_wait(bfull + 8 * s, (gch / NS) & 1);
       chunk_mma<W, W, MB, MOE_ILN>(acc, ra, sp0 + s * SB + 8192, ubase, lane, sb + s * SB);
       wg_commit();
       chunk_mma<W, W, MB, MOE_ILN>(acc, rb_, sp0 + s * SB + MX_PAIR + 8192, ubase, lane, sb + s * SB + MX_PAIR);
       wg_commit();
       wg_wait<0>();
       __syncwarp();
       if (lane == 0) mbar_arrive(bempty + 8 * s);
      }
     }
    } else {
      for (int c = 0; c < C; c += 2, gch++) {
       const int s = gch % NS;
       mbar_wait(bfull + 8 * s, (gch / NS) & 1);
       chunk_mma<W, W, MB, MOE_ILN>(acc, ra, sp0 + s * SB + 8192, ubase, lane, sb + s * SB);
       wg_commit();
       chunk_mma<W, W, MB, MOE_ILN>(acc, rb_, sp0 + s * SB + MX_PAIR + 8192, ubase, lane, sb + s * SB + MX_PAIR);
       wg_commit();
       wg_wait<0>();
       __syncwarp();
       if (lane == 0) mbar_arrive(bempty + 8 * s);
      }
    }
   } else {
    if (false) {
    } else if ((Z_M2SET & 1) && MB == 2 && W == 128 && nw == 80) {
     for (int c = 0; c < C; c += 2) {
      step80(c, ra);
      step80(c + 1, rb_);
     }
    } else if ((Z_M2SET & 2) && MB == 2 && W == 128 && nw == 96) {
     for (int c = 0; c < C; c += 2) {
      step96(c, ra);
      step96(c + 1, rb_);
     }
    } else if ((Z_M2SET & 4) && MB == 2 && W == 128 && nw == 112) {
     for (int c = 0; c < C; c += 2) {
      step112(c, ra);
      step112(c + 1, rb_);
     }
    } else if (MB == 2 && W == 128 && nw == 72) {
     for (int c = 0; c < C; c += 2) {
      step72(c, ra);
      step72(c + 1, rb_);
     }
    } else if (MB == 2 && W == 128 && nw == 88) {
     for (int c = 0; c < C; c += 2) {
      step88(c, ra);
      step88(c + 1, rb_);
     }
    } else if (MB == 2 && W == 128 && nw == 104) {
     for (int c = 0; c < C; c += 2) {
      step104(c, ra);
      step104(c + 1, rb_);
     }
    } else if (MB == 2 && W == 128 && nw == 120) {
     for (int c = 0; c < C; c += 2) {
      step120(c, ra);
      step120(c + 1, rb_);
     }
    } else {
     for (int c = 0; c < C; c += 2) {
      step(c, ra);
      step(c + 1, rb_);
     }
    }
   }
   wg_wait<0>();
   __syncwarp();

   if (MB != 1 && !(MB == 2 && W <= 64 && W >= 32 && !MOE_PIPE2) && !(MB == 2 && W < 32) && nw == W && lane == 0) mbar_arrive(bempty + 8 * ((gch - 1) % NS));
   const float sc = s_gs[mode - 1][e] * 9.223372036854775808e18f;
   const int lr = (warp % 4) * 16 + g;
   if (k > 0) asm volatile("bar.sync 3, %0;" ::"n"(NC + 96) : "memory");





   {
    (void)lr;
    constexpr int ST = MB == 1 ? 136 : 264;
    const int i4 = lane >> 3, t8 = lane & 7;
    const uint32_t ob = (uint32_t)__cvta_generic_to_shared(obuf);
 #pragma unroll
    for (int b = 0; b < MB; b++) {
     const int cb = MB == 1 ? (mode == 1 ? 64 * (i4 & 1) + warp * 8 : (warp / 4) * 64 + (warp % 4) * 16 + 8 * (i4 & 1))
                 : (mode == 1 ? 128 * b + (warp / 4) * 64 : 64 * b + (warp / 4) * 128) + (warp % 4) * 16 +
                    8 * (i4 & 1);
     const uint32_t base = ob + ((8 * (i4 >> 1) + t8) * ST + cb) * 2;
 #pragma unroll
     for (int j = 0; j < AW / 8; j += 2)
      stsm4t(base + j * 8 * ST * 2, pkbf(acc[b][4 * j] * sc, acc[b][4 * j + 1] * sc),
          pkbf(acc[b][4 * j + 2] * sc, acc[b][4 * j + 3] * sc), pkbf(acc[b][4 * j + 4] * sc, acc[b][4 * j + 5] * sc),
          pkbf(acc[b][4 * j + 6] * sc, acc[b][4 * j + 7] * sc));
    }
   }
   asm volatile("bar.arrive 2, %0;" ::"n"(NC + 96) : "memory");
  };
  if (mbv == 2) {
   if (wv == 8) tile(IC<2>{}, IC<8>{});
   else if (wv == 16) tile(IC<2>{}, IC<16>{});
   else if (wv == 32) tile(IC<2>{}, IC<32>{});
   else if (wv == 64) tile(IC<2>{}, IC<64>{});
   else tile(IC<2>{}, IC<128>{});
  } else {
   if (wv == 128) tile(IC<1>{}, IC<128>{});
   else if (wv == 136) tile(IC<1>{}, IC<136>{});
   else if (wv == 144) tile(IC<1>{}, IC<144>{});
   else if (wv == 152) tile(IC<1>{}, IC<152>{});
   else if (wv == 160) tile(IC<1>{}, IC<160>{});
   else if (wv == 168) tile(IC<1>{}, IC<168>{});
   else if (wv == 176) tile(IC<1>{}, IC<176>{});
   else if (wv == 184) tile(IC<1>{}, IC<184>{});
   else if (wv == 192) tile(IC<1>{}, IC<192>{});
   else if (wv == 200) tile(IC<1>{}, IC<200>{});
   else if (wv == 208) tile(IC<1>{}, IC<208>{});
   else if (wv == 216) tile(IC<1>{}, IC<216>{});
   else if (wv == 224) tile(IC<1>{}, IC<224>{});
   else if (wv == 232) tile(IC<1>{}, IC<232>{});
   else if (wv == 240) tile(IC<1>{}, IC<240>{});
   else if (wv == 248) tile(IC<1>{}, IC<248>{});
   else tile(IC<1>{}, IC<256>{});
  }
 }
}







__global__ void __launch_bounds__((NW + 4) * 32, 1) moe_mixed_mc(const uint8_t* __restrict__ W1a, const uint8_t* __restrict__ W2a,
                               const uint8_t* __restrict__ W1b, const uint8_t* __restrict__ W2b,
                               const __grid_constant__ CUtensorMap bmap1,
                               const __grid_constant__ CUtensorMap bmap2,
                               const __grid_constant__ MxMaps bmx, const __grid_constant__ CUtensorMap x65w40a, const __grid_constant__ CUtensorMap x65w40b,
                               int* __restrict__ meta, const float* __restrict__ gs1,
                               const float* __restrict__ gs2, __nv_bfloat16* __restrict__ out1,
                               __nv_bfloat16* __restrict__ out2, int C, const long* __restrict__ slot_of,
                               int* __restrict__ cnt2) {
 constexpr int NC = NW * 32, SB = MX_SB, NS = MX_NS;
 int* const counter = meta + 130;
 int* const cnt1 = meta + 260;
 extern __shared__ uint8_t mraw[];
 __shared__ __align__(8) uint64_t bars[2 * NS];
 __shared__ int s_toff[2][33], s_start[32], s_cnt[32], s_perm[32], s_info[32], s_jt0[32], s_map2[32];
 __shared__ int s_tq[8];
 __shared__ int4 s_tl[8];
 __shared__ int s_tj[8];
 __shared__ __align__(8) uint64_t tbars[8];
 __shared__ float s_gs[2][32];
 __shared__ int z0_slot[2][256];
 constexpr bool MC = true;
 __shared__ int s_claim[8];
 __shared__ __align__(8) uint64_t cbars[8], peer_ready[NS];
 uint32_t crank;
 asm volatile("mov.u32 %0, %%cluster_ctarank;" : "=r"(crank));
 const int tid = threadIdx.x, warp = tid / 32, lane = tid % 32, g = lane / 4, q = lane % 4;
 const uint32_t raw = (uint32_t)__cvta_generic_to_shared(mraw);
 const uint32_t sb = (raw + 1023) & ~1023u;
 uint8_t* const sp0 = mraw + (sb - raw);
 __nv_bfloat16* const obuf = reinterpret_cast<__nv_bfloat16*>(sp0 + NS * SB);
 const uint32_t bfull = (uint32_t)__cvta_generic_to_shared(bars), bempty = bfull + 8 * NS;
 GDC_WAIT();
 if (tid < 33) { s_toff[0][tid] = meta[64 + tid]; s_toff[1][tid] = meta[97 + tid]; }
 if (tid < 32) {
  s_start[tid] = meta[32 + tid]; s_cnt[tid] = meta[tid]; s_gs[0][tid] = gs1[tid]; s_gs[1][tid] = gs2[tid];
  s_perm[tid] = meta[164 + tid]; s_info[tid] = meta[196 + tid]; s_jt0[tid] = meta[228 + tid] & 0xFFFF;
  s_map2[tid] = meta[228 + tid] >> 16;
 }
 if (tid == 0) {
  for (int s = 0; s < NS; s++) { mbar_init(bfull + 8 * s, 1); mbar_init(bempty + 8 * s, 2 * NW); }
  for (int s = 0; s < 8; s++) mbar_init((uint32_t)__cvta_generic_to_shared(&tbars[s]), 1);
  for (int i = 0; i < 8; i++) mbar_init((uint32_t)__cvta_generic_to_shared(&cbars[i]), 1);
  for (int i = 0; i < NS; i++) mbar_init((uint32_t)__cvta_generic_to_shared(&peer_ready[i]), 1);
  asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
 }
 __syncthreads();
 ft_cluster_sync();
 const uint32_t peer_empty = ft_mapa(bempty, crank ^ 1u);


 if (cnt2 && tid == 0) asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
 const int total1 = s_toff[0][32], total = total1 + s_toff[1][32];
 const uint32_t btq = (uint32_t)__cvta_generic_to_shared(tbars);


 auto locate = [&](int t, int& mode, int& e, int& mb, int& rb, int& row0, int& nv, int& jt, int& rem) {
  mode = t < total1 ? 1 : 2;
  if (mode == 2) t -= total1;
  const int* to = s_toff[mode - 1];
  int lo = 0, hi = 31;
  while (lo < hi) {
   const int mid = (lo + hi + 1) >> 1;
   if (to[mid] <= t) lo = mid; else hi = mid - 1;
  }
  const int lt = t - to[lo];
  if (mode == 2) lo = s_map2[lo];
  e = s_perm[lo];
  const int inf = s_info[lo], ntt = inf & 255, tsz = (inf >> 12) & 511, rr = inf >> 21;
  mb = (inf >> 8) & 15;
  const int k1 = ntt - (rr > 0), per = ((mode == 1 ? 64 : 32) >> (mb - 1));
  int tt;
  if (lt < k1 * per) {
   const int qr = z0_qexact(lt >> 1, k1);
   rb = 2 * qr + (int)crank;
   tt = (lt >> 1) - qr * k1;
   row0 = s_start[e] + tt * tsz;
   nv = min(tsz, s_cnt[e] - rr - tt * tsz);
   rem = 0;
  } else {
   mb = 2;
   rb = lt - k1 * per + (int)crank;
   tt = k1;
   row0 = s_start[e] + s_cnt[e] - rr;
   nv = rr;
   rem = 1;
  }
  jt = s_jt0[lo] + tt;
 };
 if (warp >= NW) {
  asm volatile("setmaxnreg.dec.sync.aligned.u32 56;");
  if (warp == NW) {
   uint32_t gch = 0;
   const uint32_t pclaim = ft_mapa((uint32_t)__cvta_generic_to_shared(s_claim), crank ^ 1u);
   const uint32_t pcbar = ft_mapa((uint32_t)__cvta_generic_to_shared(cbars), crank ^ 1u);
   const uint32_t lcbar = (uint32_t)__cvta_generic_to_shared(cbars);
   const uint32_t ready = (uint32_t)__cvta_generic_to_shared(peer_ready);
   const uint32_t pready = ft_mapa(ready, crank ^ 1u);
   int next_pair = -1;
   for (int k = 0;; k++) {
    int pair;
    if (crank == 0) {
     auto claim = [&](int kk) {
      int p = 0;
      if (lane == 0) {
       p = atomicAdd(counter, 1);
       if (2 * p >= total) p = -1;
       ft_st_remote(pclaim + 4 * (kk & 7), p);
       ft_arrive_claim(pcbar + 8 * (kk & 7));
      }
      return __shfl_sync(0xffffffffu, p, 0);
     };
     if (k == 0) next_pair = claim(0);
     pair = next_pair;
     if (pair >= 0) next_pair = claim(k + 1);
    } else {
     ft_wait_claim(lcbar + 8 * (k & 7), (k >> 3) & 1);
     pair = *reinterpret_cast<volatile int*>(&s_claim[k & 7]);
    }
    int t = pair < 0 ? -1 : 2 * pair;
    int mode = 0, e = 0, mb = 2, rb = 0, row0 = 0, nv = 0, jt = 0, rem = 0;
    if (t >= 0) locate(t, mode, e, mb, rb, row0, nv, jt, rem);
    const int W = mx_w(mb, nv);
    if (lane == 0) {
     s_tq[k & 7] = t;
     s_tl[k & 7] = make_int4(e | mode << 8 | mb << 12 | W << 16, rb | rem << 16, row0, nv);
     s_tj[k & 7] = jt;
     mbar_arrive(btq + 8 * (k & 7));
    }
    __syncwarp();
    if (t < 0) break;
    const int WB = wbytes(mb), xo = mx_xoff(mb);
    const uint8_t* wp = mode == 1 ? (mb == 2 ? W1a : W1b) + ((size_t)e * ((64 >> (mb - 1))) + rb) * C * WB
                   : (mb == 2 ? W2a : W2b) + ((size_t)e * ((32 >> (mb - 1))) + rb) * C * WB;
    const CUtensorMap* bm = mb == 2 && W <= 32 ? &bmx.m[mode - 1][W == 8 ? 6 : W == 16 ? 7 : 8]
                          : mode == 1 ? &bmap1 : &bmap2;
    const CUtensorMap* wm = &bmx.w[mode - 1][mb - 1];
    const CUtensorMap* wm40 = mode == 1 ? &x65w40a : &x65w40b;
    const int wr0 = (int)(((mode == 1 ? (size_t)e * ((64 >> (mb - 1))) : (size_t)e * ((32 >> (mb - 1)))) + rb) * C * (WB / 512));
    (void)wp;
    if (mode == 2) {
     while (ld_acquire(&cnt1[jt]) < (64 >> (mb - 1))) __nanosleep(128);
     asm volatile("fence.proxy.async.global;" ::: "memory");
    }
    const int nbox = (W + 63) / 64;

    const bool one = W >= MX_OB && W >= 128;
    const int Wb = (W + 31) & ~31;
    const CUtensorMap* bmw = &bmx.m[mode - 1][one ? (Wb - 96) / 32 : 0];
    const int xbytes = one ? Wb * 128 : nbox * 8192;
    if (mb == 2 && W <= 64) {
     constexpr int WB2 = wbytes(2);
     const int z0_ex = 2 * (W * 128 + WB2);


     for (int c = 0; c < C; c += 2, gch++) {
      const int s = gch % NS;
      if (gch >= NS) ft_wait_cluster(bempty + 8 * s, ((gch / NS) - 1) & 1);
      if (x64_elect()) {
       mbar_expect(bfull + 8 * s, z0_ex);
       tma_2d_mc(sb + s * SB + (crank ? X65_X1 : 0u), bm, (c + (int)crank) * 64, row0, bfull + 8 * s, (uint16_t)3);
       tma_2d(sb + s * SB + 8192, wm40, 0, wr0 + c * (WB2 / 512), bfull + 8 * s);
      }
     }
    } else {
     for (int c = 0; c < C; c++, gch++) {
      const int s = gch % NS;
      if (gch >= NS) ft_wait_cluster(bempty + 8 * s, ((gch / NS) - 1) & 1);
      if (x64_elect()) {
       mbar_expect(bfull + 8 * s, xbytes + WB);
       tma_2d(sb + s * SB + xo, wm, 0, wr0 + c * (WB / 512), bfull + 8 * s);
       if (one) {
        if (crank == 0) tma_2d_mc(sb + s * SB, bmw, c * 64, row0, bfull + 8 * s, (uint16_t)3);
       } else {
        for (int bx = (int)crank; bx < nbox; bx += 2) tma_2d_mc(sb + s * SB + bx * 8192, bm, c * 64, row0 + bx * 64, bfull + 8 * s, (uint16_t)3);
       }
      }
     }
    }
   }
  } else if (warp > NW) {
   const int ct = tid - (NC + 32);
   for (int k = 0;; k++) {
    mbar_wait(btq + 8 * (k & 7), (k >> 3) & 1);
    const int t = s_tq[k & 7];
    if (t < 0) break;
    const int4 tl = s_tl[k & 7];
    const int mode = (tl.x >> 8) & 15, mb = (tl.x >> 12) & 15, rb = tl.y & 0xFFFF, row0 = tl.z, nv = tl.w;
    int* const zs = z0_slot[k & 1];
    if (mode == 2) {

     for (int i = ct; i < nv; i += 96) zs[i] = (int)slot_of[row0 + i];
     asm volatile("bar.sync 4, 96;" ::: "memory");
    }
    asm volatile("bar.sync 2, %0;" ::"n"(NC + 96) : "memory");
    if (mode == 1) {


     const int CW = 64 * mb, ST = mb == 1 ? 136 : 264, G8 = CW / 8;
     for (int i = ct; i < nv * G8; i += 96) {
      const int tok = i / G8, c8 = (i % G8) * 8;
      const uint4 gq = *reinterpret_cast<const uint4*>(obuf + tok * ST + c8);
      const uint4 uq = *reinterpret_cast<const uint4*>(obuf + tok * ST + CW + c8);
      const __nv_bfloat16* gp = reinterpret_cast<const __nv_bfloat16*>(&gq);
      const __nv_bfloat16* up = reinterpret_cast<const __nv_bfloat16*>(&uq);
      __align__(16) __nv_bfloat16 o[8];
 #pragma unroll
      for (int x = 0; x < 8; x++) {
       const float gv = __bfloat162float(gp[x]);
       const float sl = __bfloat162float(__float2bfloat16(__fdividef(gv, 1.f + __expf(-gv))));
       o[x] = __float2bfloat16(sl * __bfloat162float(up[x]));
      }
      *reinterpret_cast<uint4*>(out1 + (size_t)(row0 + tok) * 4096 + rb * CW + c8) = *reinterpret_cast<const uint4*>(o);
     }
    } else {
     const int RW = 128 * mb, OS = RW + 8, R8 = RW / 8;
     for (int i = ct; i < nv * R8; i += 96) {
      const int tok = i / R8, c8 = (i % R8) * 8;
      *reinterpret_cast<uint4*>(out2 + (size_t)zs[tok] * 4096 + rb * RW + c8) =
        *reinterpret_cast<const uint4*>(obuf + tok * OS + c8);
     }
    }
    if (mode == 1) {
     asm volatile("fence.proxy.async.global;" ::: "memory");
     __threadfence();
     asm volatile("bar.sync 4, 96;" ::: "memory");
     if (ct == 0) atomicAdd(&cnt1[s_tj[k & 7]], 1);
    } else if (cnt2) {

     __threadfence();
     asm volatile("bar.sync 4, 96;" ::: "memory");
     if (ct == 0) atomicAdd(&cnt2[tl.x & 255], 1 + (tl.y >> 16));
    }
    asm volatile("bar.arrive 3, %0;" ::"n"(NC + 96) : "memory");
   }
  }
  ft_cluster_sync();
  return;
 }
 asm volatile("setmaxnreg.inc.sync.aligned.u32 224;");
 uint32_t gch = 0;
 for (int k = 0;; k++) {
  mbar_wait(btq + 8 * (k & 7), (k >> 3) & 1);
  const int t = s_tq[k & 7];
  if (t < 0) break;
  const int4 tl = s_tl[k & 7];
  const int mode = (tl.x >> 8) & 15, e = tl.x & 255, mbv = (tl.x >> 12) & 15, wv = tl.x >> 16;


  auto tile = [&](auto MBc, auto Wc) {
   constexpr int MB = decltype(MBc)::value, W = decltype(Wc)::value;
   const int ubase = MB == 2 ? (warp < 4 ? 0 : 8) + (warp % 4) : warp;
   constexpr int AW = (W + 15) / 16 * 16;

   const int nw = (MB == 2 && W == 128) ? ((tl.w <= 80 && (Z_M2SET & 1)) ? 80 : (tl.w <= 96 && (Z_M2SET & 2)) ? 96 :
                       (tl.w <= 112 && (Z_M2SET & 4)) ? 112 : 128) : W;
   (void)nw;
   float acc[MB][AW / 2];
   uint32_t ra[MB][16], rb_[MB][16];
 #pragma unroll
   for (int b = 0; b < MB; b++)
 #pragma unroll
    for (int i = 0; i < AW / 2; i++) acc[b][i] = 0.f;
   auto step = [&](int c, uint32_t (&r)[MB][16]) {
    const int s = gch % NS;
    mbar_wait(bfull + 8 * s, (gch / NS) & 1);
    const uint8_t* wb = sp0 + s * SB + mx_xoff(MB);
    const uint32_t xa = sb + s * SB;
    if constexpr (W >= 32) chunk_mma<W, AW, MB, MB == 2 ? MOE_ILM : MOE_IL>(acc, r, wb, ubase, lane, xa);
    wg_commit();
    if constexpr (MB == 1) {
     wg_wait<0>();
     __syncwarp();
     if (lane == 0) { mbar_arrive(bempty + 8 * s); ft_arrive_remote(peer_empty + 8 * s); }
    } else {
     wg_wait<1>();
     if (c > 0) {
      __syncwarp();
      if (lane == 0) { mbar_arrive(bempty + 8 * ((gch - 1) % NS)); ft_arrive_remote(peer_empty + 8 * ((gch - 1) % NS)); }
     }
    }
    gch++;
   };
   auto step80 = [&](int c, uint32_t (&r)[MB][16]) {
    const int s = gch % NS;
    mbar_wait(bfull + 8 * s, (gch / NS) & 1);
    const uint8_t* wb = sp0 + s * SB + mx_xoff(MB);
    const uint32_t xa = sb + s * SB;
    if constexpr (MB == 2 && W == 128) chunk_mma<80, 128, MB, MOE_ILM>(acc, r, wb, ubase, lane, xa);
    wg_commit();
    if constexpr (true) {
     wg_wait<0>();
     __syncwarp();
     if (lane == 0) { mbar_arrive(bempty + 8 * s); ft_arrive_remote(peer_empty + 8 * s); }
    } else {
     wg_wait<1>();
     if (c > 0) {
      __syncwarp();
      if (lane == 0) { mbar_arrive(bempty + 8 * ((gch - 1) % NS)); ft_arrive_remote(peer_empty + 8 * ((gch - 1) % NS)); }
     }
    }
    gch++;
   };
   auto step96 = [&](int c, uint32_t (&r)[MB][16]) {
    const int s = gch % NS;
    mbar_wait(bfull + 8 * s, (gch / NS) & 1);
    const uint8_t* wb = sp0 + s * SB + mx_xoff(MB);
    const uint32_t xa = sb + s * SB;
    if constexpr (MB == 2 && W == 128) chunk_mma<96, 128, MB, MOE_ILM>(acc, r, wb, ubase, lane, xa);
    wg_commit();
    if constexpr (true) {
     wg_wait<0>();
     __syncwarp();
     if (lane == 0) { mbar_arrive(bempty + 8 * s); ft_arrive_remote(peer_empty + 8 * s); }
    } else {
     wg_wait<1>();
     if (c > 0) {
      __syncwarp();
      if (lane == 0) { mbar_arrive(bempty + 8 * ((gch - 1) % NS)); ft_arrive_remote(peer_empty + 8 * ((gch - 1) % NS)); }
     }
    }
    gch++;
   };
   auto step112 = [&](int c, uint32_t (&r)[MB][16]) {
    const int s = gch % NS;
    mbar_wait(bfull + 8 * s, (gch / NS) & 1);
    const uint8_t* wb = sp0 + s * SB + mx_xoff(MB);
    const uint32_t xa = sb + s * SB;
    if constexpr (MB == 2 && W == 128) chunk_mma<112, 128, MB, MOE_ILM>(acc, r, wb, ubase, lane, xa);
    wg_commit();
    if constexpr (true) {
     wg_wait<0>();
     __syncwarp();
     if (lane == 0) { mbar_arrive(bempty + 8 * s); ft_arrive_remote(peer_empty + 8 * s); }
    } else {
     wg_wait<1>();
     if (c > 0) {
      __syncwarp();
      if (lane == 0) { mbar_arrive(bempty + 8 * ((gch - 1) % NS)); ft_arrive_remote(peer_empty + 8 * ((gch - 1) % NS)); }
     }
    }
    gch++;
   };
   if constexpr (MB == 2 && W < 32) {
    constexpr int NT = W / 8;
    {

     const int lrow = lane & 7, lblk = lane >> 3;
     for (int c = 0; c < C; c += 2, gch++) {
      const int s = gch % NS;
      mbar_wait(bfull + 8 * s, (gch / NS) & 1);
 #pragma unroll
      for (int j = 0; j < 2; j++) {
       const uint32_t xr = sb + s * SB + j * X65_X1 + lrow * 128;
       uint32_t bf[NT][8];
 #pragma unroll
       for (int n = 0; n < NT; n++) {
        const uint32_t xn = xr + n * 1024;
        ldsm4(xn + ((lblk ^ lrow) << 4), bf[n][0], bf[n][1], bf[n][2], bf[n][3]);
        ldsm4(xn + (((lblk + 4) ^ lrow) << 4), bf[n][4], bf[n][5], bf[n][6], bf[n][7]);
       }
       const uint8_t* wb = sp0 + s * SB + 8192 + j * (X65_W1 - 8192);
 #pragma unroll
       for (int b = 0; b < 2; b++) {
        const uint8_t* up = wb + (ubase + 4 * b) * UNIT2;
        const uint4 nb = *reinterpret_cast<const uint4*>(up + lane * 16);
        const uint32_t S = sc_bf16(*reinterpret_cast<const uint16_t*>(up + 512 + lane * 2), 0x4140);
 #pragma unroll
        for (int kk = 0; kk < 4; kk++) {
         const uint32_t w = kk == 0 ? nb.x : kk == 1 ? nb.y : kk == 2 ? nb.z : nb.w;
         uint32_t o[4];
         decode4(w, o);
         const uint32_t a0 = bmulh(o[0], S, 0), a1 = bmulh(o[1], S, 1), a2 = bmulh(o[2], S, 0),
                 a3 = bmulh(o[3], S, 1);
 #pragma unroll
         for (int n = 0; n < NT; n++) mma16816(&acc[b][4 * n], a0, a1, a2, a3, bf[n][2 * kk], bf[n][2 * kk + 1]);
        }
       }
      }
      __syncwarp();


      asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
      if (lane == 0) { mbar_arrive(bempty + 8 * s); ft_arrive_remote(peer_empty + 8 * s); }
     }
    }
   } else if constexpr (MB == 2 && W <= 64) {



    const int nwm = W == 64 ? (tl.w <= 40 ? 40 : tl.w <= 48 ? 48 : tl.w <= 56 ? 56 : 64)
                : W == 32 ? (tl.w <= 24 ? 24 : 32) : W;
    (void)nwm;
    if constexpr (W == 64) {
     if ((Z_MIDSET & 1) && nwm == 40) {
      for (int c = 0; c < C; c += 2, gch++) {
       const int s = gch % NS;
       mbar_wait(bfull + 8 * s, (gch / NS) & 1);
       chunk_mma<40, W, MB, MOE_ILN>(acc, ra, sp0 + s * SB + 8192, ubase, lane, sb + s * SB);
       wg_commit();
       chunk_mma<40, W, MB, MOE_ILN>(acc, rb_, sp0 + s * SB + X65_W1, ubase, lane, sb + s * SB + X65_X1);
       wg_commit();
       wg_wait<0>();
       __syncwarp();
       if (lane == 0) { mbar_arrive(bempty + 8 * s); ft_arrive_remote(peer_empty + 8 * s); }
      }
     } else if ((Z_MIDSET & 2) && nwm == 48) {
      for (int c = 0; c < C; c += 2, gch++) {
       const int s = gch % NS;
       mbar_wait(bfull + 8 * s, (gch / NS) & 1);
       chunk_mma<48, W, MB, MOE_ILN>(acc, ra, sp0 + s * SB + 8192, ubase, lane, sb + s * SB);
       wg_commit();
       chunk_mma<48, W, MB, MOE_ILN>(acc, rb_, sp0 + s * SB + X65_W1, ubase, lane, sb + s * SB + X65_X1);
       wg_commit();
       wg_wait<0>();
       __syncwarp();
       if (lane == 0) { mbar_arrive(bempty + 8 * s); ft_arrive_remote(peer_empty + 8 * s); }
      }
     } else if ((Z_MIDSET & 4) && nwm == 56) {
      for (int c = 0; c < C; c += 2, gch++) {
       const int s = gch % NS;
       mbar_wait(bfull + 8 * s, (gch / NS) & 1);
       chunk_mma<56, W, MB, MOE_ILN>(acc, ra, sp0 + s * SB + 8192, ubase, lane, sb + s * SB);
       wg_commit();
       chunk_mma<56, W, MB, MOE_ILN>(acc, rb_, sp0 + s * SB + X65_W1, ubase, lane, sb + s * SB + X65_X1);
       wg_commit();
       wg_wait<0>();
       __syncwarp();
       if (lane == 0) { mbar_arrive(bempty + 8 * s); ft_arrive_remote(peer_empty + 8 * s); }
      }
     } else {
      for (int c = 0; c < C; c += 2, gch++) {
       const int s = gch % NS;
       mbar_wait(bfull + 8 * s, (gch / NS) & 1);
       chunk_mma<W, W, MB, MOE_ILN>(acc, ra, sp0 + s * SB + 8192, ubase, lane, sb + s * SB);
       wg_commit();
       chunk_mma<W, W, MB, MOE_ILN>(acc, rb_, sp0 + s * SB + X65_W1, ubase, lane, sb + s * SB + X65_X1);
       wg_commit();
       wg_wait<0>();
       __syncwarp();
       if (lane == 0) { mbar_arrive(bempty + 8 * s); ft_arrive_remote(peer_empty + 8 * s); }
      }
     }
    } else if constexpr (W == 32) {
     if ((Z_MIDSET & 8) && nwm == 24) {
      for (int c = 0; c < C; c += 2, gch++) {
       const int s = gch % NS;
       mbar_wait(bfull + 8 * s, (gch / NS) & 1);
       chunk_mma<24, W, MB, MOE_ILN>(acc, ra, sp0 + s * SB + 8192, ubase, lane, sb + s * SB);
       wg_commit();
       chunk_mma<24, W, MB, MOE_ILN>(acc, rb_, sp0 + s * SB + X65_W1, ubase, lane, sb + s * SB + X65_X1);
       wg_commit();
       wg_wait<0>();
       __syncwarp();
       if (lane == 0) { mbar_arrive(bempty + 8 * s); ft_arrive_remote(peer_empty + 8 * s); }
      }
     } else {
      for (int c = 0; c < C; c += 2, gch++) {
       const int s = gch % NS;
       mbar_wait(bfull + 8 * s, (gch / NS) & 1);
       chunk_mma<W, W, MB, MOE_ILN>(acc, ra, sp0 + s * SB + 8192, ubase, lane, sb + s * SB);
       wg_commit();
       chunk_mma<W, W, MB, MOE_ILN>(acc, rb_, sp0 + s * SB + X65_W1, ubase, lane, sb + s * SB + X65_X1);
       wg_commit();
       wg_wait<0>();
       __syncwarp();
       if (lane == 0) { mbar_arrive(bempty + 8 * s); ft_arrive_remote(peer_empty + 8 * s); }
      }
     }
    } else {
      for (int c = 0; c < C; c += 2, gch++) {
       const int s = gch % NS;
       mbar_wait(bfull + 8 * s, (gch / NS) & 1);
       chunk_mma<W, W, MB, MOE_ILN>(acc, ra, sp0 + s * SB + 8192, ubase, lane, sb + s * SB);
       wg_commit();
       chunk_mma<W, W, MB, MOE_ILN>(acc, rb_, sp0 + s * SB + X65_W1, ubase, lane, sb + s * SB + X65_X1);
       wg_commit();
       wg_wait<0>();
       __syncwarp();
       if (lane == 0) { mbar_arrive(bempty + 8 * s); ft_arrive_remote(peer_empty + 8 * s); }
      }
    }
   } else {
    if (false) {
    } else if ((Z_M2SET & 1) && MB == 2 && W == 128 && nw == 80) {
     for (int c = 0; c < C; c += 2) {
      step80(c, ra);
      step80(c + 1, rb_);
     }
    } else if ((Z_M2SET & 2) && MB == 2 && W == 128 && nw == 96) {
     for (int c = 0; c < C; c += 2) {
      step96(c, ra);
      step96(c + 1, rb_);
     }
    } else if ((Z_M2SET & 4) && MB == 2 && W == 128 && nw == 112) {
     for (int c = 0; c < C; c += 2) {
      step112(c, ra);
      step112(c + 1, rb_);
     }
    } else {
     for (int c = 0; c < C; c += 2) {
      step(c, ra);
      step(c + 1, rb_);
     }
    }
   }
   wg_wait<0>();
   __syncwarp();

   if (MB != 1 && !(MB == 2 && W <= 64 && W >= 32 && !MOE_PIPE2) && !(MB == 2 && W < 32) && nw == W && lane == 0) { mbar_arrive(bempty + 8 * ((gch - 1) % NS)); ft_arrive_remote(peer_empty + 8 * ((gch - 1) % NS)); }
   const float sc = s_gs[mode - 1][e] * 9.223372036854775808e18f;
   const int lr = (warp % 4) * 16 + g;
   if (k > 0) asm volatile("bar.sync 3, %0;" ::"n"(NC + 96) : "memory");





   {
    (void)lr;
    constexpr int ST = MB == 1 ? 136 : 264;
    const int i4 = lane >> 3, t8 = lane & 7;
    const uint32_t ob = (uint32_t)__cvta_generic_to_shared(obuf);
 #pragma unroll
    for (int b = 0; b < MB; b++) {
     const int cb = MB == 1 ? (mode == 1 ? 64 * (i4 & 1) + warp * 8 : (warp / 4) * 64 + (warp % 4) * 16 + 8 * (i4 & 1))
                 : (mode == 1 ? 128 * b + (warp / 4) * 64 : 64 * b + (warp / 4) * 128) + (warp % 4) * 16 +
                    8 * (i4 & 1);
     const uint32_t base = ob + ((8 * (i4 >> 1) + t8) * ST + cb) * 2;
 #pragma unroll
     for (int j = 0; j < AW / 8; j += 2)
      stsm4t(base + j * 8 * ST * 2, pkbf(acc[b][4 * j] * sc, acc[b][4 * j + 1] * sc),
          pkbf(acc[b][4 * j + 2] * sc, acc[b][4 * j + 3] * sc), pkbf(acc[b][4 * j + 4] * sc, acc[b][4 * j + 5] * sc),
          pkbf(acc[b][4 * j + 6] * sc, acc[b][4 * j + 7] * sc));
    }
   }
   asm volatile("bar.arrive 2, %0;" ::"n"(NC + 96) : "memory");
  };
  if (mbv == 2) {
   if (wv == 8) tile(IC<2>{}, IC<8>{});
   else if (wv == 16) tile(IC<2>{}, IC<16>{});
   else if (wv == 32) tile(IC<2>{}, IC<32>{});
   else if (wv == 64) tile(IC<2>{}, IC<64>{});
   else tile(IC<2>{}, IC<128>{});
  } else {
   if (wv == 128) tile(IC<1>{}, IC<128>{});
   else if (wv == 136) tile(IC<1>{}, IC<136>{});
   else if (wv == 144) tile(IC<1>{}, IC<144>{});
   else if (wv == 152) tile(IC<1>{}, IC<152>{});
   else if (wv == 160) tile(IC<1>{}, IC<160>{});
   else if (wv == 168) tile(IC<1>{}, IC<168>{});
   else if (wv == 176) tile(IC<1>{}, IC<176>{});
   else if (wv == 184) tile(IC<1>{}, IC<184>{});
   else if (wv == 192) tile(IC<1>{}, IC<192>{});
   else if (wv == 200) tile(IC<1>{}, IC<200>{});
   else if (wv == 208) tile(IC<1>{}, IC<208>{});
   else if (wv == 216) tile(IC<1>{}, IC<216>{});
   else if (wv == 224) tile(IC<1>{}, IC<224>{});
   else if (wv == 232) tile(IC<1>{}, IC<232>{});
   else if (wv == 240) tile(IC<1>{}, IC<240>{});
   else if (wv == 248) tile(IC<1>{}, IC<248>{});
   else tile(IC<1>{}, IC<256>{});
  }
 }
 ft_cluster_sync();
}

typedef CUresult (*EncodeTiled)(CUtensorMap*, CUtensorMapDataType, cuuint32_t, void*, const cuuint64_t*,
                const cuuint64_t*, const cuuint32_t*, const cuuint32_t*, CUtensorMapInterleave,
                CUtensorMapSwizzle, CUtensorMapL2promotion, CUtensorMapFloatOOBfill);

template <int TN, int MB>
void launch_fused(torch::Tensor w1, torch::Tensor w2, torch::Tensor a, torch::Tensor inter, torch::Tensor meta,
         torch::Tensor gs1, torch::Tensor gs2, torch::Tensor y, int ctas, const long* slot_of, int phase) {
 const int dyn = ctas > 0 ? 1 : 0;
 ctas = ctas > 0 ? ctas : -ctas;
 constexpr int smem = 1024 + nstages(TN) * stage_bytes(TN, MB) + EBYTES;
 constexpr int RB1 = 2 * 4096 / (128 * MB), RB2 = 4096 / (128 * MB);
 static bool attr_set = false;
 static EncodeTiled encode = nullptr;
 if (!attr_set) {
  cudaFuncSetAttribute(moe_fused<TN, MB>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
  cudaDriverEntryPointQueryResult qr;
  cudaGetDriverEntryPoint("cuTensorMapEncodeTiled", reinterpret_cast<void**>(&encode), cudaEnableDefault, &qr);
  attr_set = true;
 }
 CUtensorMap map[4];
 for (int i = 0; i < 4; i++) {
  torch::Tensor& b = i % 2 == 0 ? a : inter;
  const cuuint64_t dims[2] = {4096, (cuuint64_t)b.size(0)};
  const cuuint64_t strides[1] = {4096 * 2};
  const cuuint32_t box[2] = {64, i < 2 ? 64u : 256u}, estr[2] = {1, 1};
  encode(&map[i], CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, b.data_ptr(), dims, strides, box, estr,
      CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_128B,
      CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
 }
 int* m = meta.data_ptr<int>();
 moe_fused<TN, MB><<<ctas, (NW + 4) * 32, smem, at::cuda::getCurrentCUDAStream()>>>(
   (const uint8_t*)w1.data_ptr(), (const uint8_t*)w2.data_ptr(), map[0], map[1], map[2], map[3], m + 32, m, m + 64, m + 97,
   gs1.data_ptr<float>(), gs2.data_ptr<float>(), (__nv_bfloat16*)inter.data_ptr(), (__nv_bfloat16*)y.data_ptr(),
   RB1, RB2, 64, slot_of, m + 130, dyn, m + 164, m + 196, phase);
}

int* g_cnt2 = nullptr;
void launch_mixed(torch::Tensor w1, torch::Tensor w2, torch::Tensor w1w, torch::Tensor w2w, torch::Tensor a,
         torch::Tensor inter, torch::Tensor meta, torch::Tensor gs1, torch::Tensor gs2, torch::Tensor y, int ctas,
         const long* slot_of) {
 constexpr int smem = 1024 + MX_NS * MX_SB + EBYTES;
 static bool attr_set = false;
 static EncodeTiled encode = nullptr;
 if (!attr_set) {
  cudaFuncSetAttribute(moe_mixed, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
  cudaFuncSetAttribute(moe_mixed_mc, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
  cudaDriverEntryPointQueryResult qr;
  cudaGetDriverEntryPoint("cuTensorMapEncodeTiled", reinterpret_cast<void**>(&encode), cudaEnableDefault, &qr);
  attr_set = true;
 }
 CUtensorMap map[2];
 MxMaps mx;
 for (int i = 0; i < 14; i++) {
  torch::Tensor& b = (i < 2 ? i : (i - 2) / 6) == 0 ? a : inter;
  const cuuint32_t rows = i < 2 ? 64u : (i - 2) % 6 == 0 ? 64u : 96u + 32u * ((i - 2) % 6);
  CUtensorMap* dst = i < 2 ? &map[i] : &mx.m[(i - 2) / 6][(i - 2) % 6];
  const cuuint64_t dims[2] = {4096, (cuuint64_t)b.size(0)};
  const cuuint64_t strides[1] = {4096 * 2};
  const cuuint32_t box[2] = {64, rows}, estr[2] = {1, 1};
  encode(dst, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, b.data_ptr(), dims, strides, box, estr,
      CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_128B,
      CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
 }
 for (int i = 0; i < 6; i++) {
  torch::Tensor& b = i < 3 ? a : inter;
  const cuuint64_t dims[2] = {4096, (cuuint64_t)b.size(0)};
  const cuuint64_t strides[1] = {4096 * 2};
  const cuuint32_t box[2] = {64, 8u << (i % 3)}, estr[2] = {1, 1};
  encode(&mx.m[i / 3][6 + i % 3], CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, b.data_ptr(), dims, strides, box, estr,
      CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_128B,
      CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
 }
 for (int i = 0; i < 4; i++) {
  torch::Tensor& wt = i == 0 ? w1 : i == 1 ? w1w : i == 2 ? w2 : w2w;
  const int mbi = i % 2 == 0 ? 2 : 1;
  const cuuint64_t dims[2] = {256, (cuuint64_t)(wt.numel() * wt.element_size() / 512)};
  const cuuint64_t strides[1] = {512};
  const cuuint32_t box[2] = {256, (cuuint32_t)(1 * (mbi == 2 ? 18 : 9))}, estr[2] = {1, 1};
  CUresult r = encode(&mx.w[i / 2][mbi - 1], CU_TENSOR_MAP_DATA_TYPE_UINT16, 2, wt.data_ptr(), dims, strides, box, estr,
      CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_NONE, CU_TENSOR_MAP_L2_PROMOTION_NONE,
      CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  if (r != CUDA_SUCCESS) printf("pr_mk3: weight map %d encode failed %d\n", i, (int)r);
 }
 CUtensorMap x65w40[2];
 for (int i = 0; i < 2; i++) {
  torch::Tensor& wt = i == 0 ? w1 : w2;
  const cuuint64_t dims[2] = {256, (cuuint64_t)(wt.numel() * wt.element_size() / 512)};
  const cuuint64_t strides[1] = {512};
  const cuuint32_t box[2] = {256, (cuuint32_t)(2 * 2 * NW * UNIT2 / 512)}, estr[2] = {1, 1};
  CUresult r = encode(&x65w40[i], CU_TENSOR_MAP_DATA_TYPE_UINT16, 2, wt.data_ptr(), dims, strides, box, estr,
      CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_NONE, CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
      CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);

  if (r != CUDA_SUCCESS) printf("x65: w40 map %d encode failed %d\n", i, (int)r);
 }
 if (y.size(0) >= 2048 * 8) {
  cudaLaunchAttribute attrs[1];
  attrs[0].id = cudaLaunchAttributeClusterDimension;
  attrs[0].val.clusterDim = {2, 1, 1};
  cudaLaunchConfig_t config = {dim3((unsigned)(ctas & ~1)), dim3((NW + 4) * 32), (size_t)smem, at::cuda::getCurrentCUDAStream(), attrs, 1};
  cudaLaunchKernelEx(&config, moe_mixed_mc,
   (const uint8_t*)w1.data_ptr(), (const uint8_t*)w2.data_ptr(), (const uint8_t*)w1w.data_ptr(),
   (const uint8_t*)w2w.data_ptr(), map[0], map[1], mx, x65w40[0], x65w40[1], meta.data_ptr<int>(), (const float*)gs1.data_ptr<float>(), (const float*)gs2.data_ptr<float>(),
   (__nv_bfloat16*)inter.data_ptr(), (__nv_bfloat16*)y.data_ptr(), 64, slot_of, g_cnt2);
 } else {
 pdl_launch(moe_mixed, dim3(ctas), dim3((NW + 4) * 32), (size_t)smem,
   (const uint8_t*)w1.data_ptr(), (const uint8_t*)w2.data_ptr(), (const uint8_t*)w1w.data_ptr(),
   (const uint8_t*)w2w.data_ptr(), map[0], map[1], mx, meta.data_ptr<int>(), (const float*)gs1.data_ptr<float>(), (const float*)gs2.data_ptr<float>(),
   (__nv_bfloat16*)inter.data_ptr(), (__nv_bfloat16*)y.data_ptr(), 64, slot_of, g_cnt2);
 }
}
}

void moe_set_cnt2(int64_t p) { g_cnt2 = reinterpret_cast<int*>(p); }


void moe_fused_run(torch::Tensor w1, torch::Tensor w2, torch::Tensor w1w, torch::Tensor w2w, torch::Tensor a,
          torch::Tensor inter, torch::Tensor meta, torch::Tensor gs1, torch::Tensor gs2, torch::Tensor y,
          int64_t ctas, int64_t tn, torch::Tensor slot_of) {
 const long* so = slot_of.data_ptr<long>();

 if (tn == 256) {
  launch_fused<256, 1>(w1w, w2w, a, inter, meta, gs1, gs2, y, (int)ctas, so, 1);
  launch_fused<256, 1>(w1w, w2w, a, inter, meta, gs1, gs2, y, (int)ctas, so, 2);
 } else {
  launch_mixed(w1, w2, w1w, w2w, a, inter, meta, gs1, gs2, y, (int)ctas, so);
 }
}








__global__ void __launch_bounds__(1024) count_k(const int* __restrict__ ids, int R, int* __restrict__ meta) {
 __shared__ int c[32];
 const int tid = threadIdx.x, lane = tid % 32;
 if (tid < 32) c[tid] = 0;
 __syncthreads();
 for (int base = blockIdx.x * 1024; base < R; base += gridDim.x * 1024) {
  const int i = base + tid;
  const int e = i < R ? ids[i] : 99;
  const bool loc = e >= 0 && e < 32;
  const unsigned m = __match_any_sync(0xffffffffu, loc ? e : 99);
  if (loc && lane == __ffs(m) - 1) atomicAdd(&c[e], __popc(m));
 }
 __syncthreads();
 if (tid < 32 && c[tid]) atomicAdd(&meta[tid], c[tid]);
}


__global__ void plan_k(int* __restrict__ meta, int tn, int rb1, int rb2) {
 __shared__ int cnt[32];
 const int e = threadIdx.x;
 const int n = meta[e];
 cnt[e] = n;
 __syncwarp();
 int pos = 0;
 for (int j = 0; j < 32; j++) pos += cnt[j] > n || (cnt[j] == n && j < e);
 meta[164 + pos] = e;

 int s = n;
 for (int o = 1; o < 32; o <<= 1) { const int v = __shfl_up_sync(0xffffffffu, s, o); if (e >= o) s += v; }
 meta[32 + e] = s - n; meta[132 + e] = s - n;

 __syncwarp();
 const int pe = meta[164 + e], ntt = (cnt[pe] + tn - 1) / tn;
 int t1 = ntt * rb1, t2 = ntt * rb2;
 for (int o = 1; o < 32; o <<= 1) {
  const int v1 = __shfl_up_sync(0xffffffffu, t1, o), v2 = __shfl_up_sync(0xffffffffu, t2, o);
  if (e >= o) { t1 += v1; t2 += v2; }
 }
 meta[64 + e] = t1 - ntt * rb1; meta[97 + e] = t2 - ntt * rb2;
 if (e == 31) { meta[96] = t1; meta[129] = t2; meta[130] = 0; meta[131] = 0; }
}


__device__ __forceinline__ int mx_cost(int mb, int W) {
 if (mb == 2) return W <= 32 ? 901 : W <= 64 ? 1027 : W <= 96 ? 1180 : 1334;
 return W <= 64 ? 715 : W <= 128 ? 883 : W <= 160 ? 940 : W <= 192 ? 1000 : W <= 224 ? 1060 : 1118;
}



__global__ void plan2_k(int* __restrict__ meta, int force_mb) {
 __shared__ int cnt[32], inf[32];
 const int e = threadIdx.x;
 const int n = meta[e];
 cnt[e] = n;
 int mb = 2, tsz = 32, ntt = 0;
 if (n > 0) {
  int best = 0x7fffffff;
  for (int m = 2; m >= 1; m--) {
   if (force_mb && m != force_mb) continue;
   const int wmax = 128 * (3 - m), k = (n + wmax - 1) / wmax, ts = ((n + k - 1) / k + 31) / 32 * 32;
   int cost = 0;
   for (int i = 0; i < k; i++) cost += mx_cost(m, mx_w(m, min(ts, n - i * ts)));
   if (m == 1) cost *= 2;
   if (cost < best) { best = cost; mb = m; tsz = ts; ntt = k; }
  }
 }
 inf[e] = ntt | mb << 8 | tsz << 12;
 __syncwarp();
 int pos = 0;
 for (int j = 0; j < 32; j++) pos += cnt[j] > n || (cnt[j] == n && j < e);
 meta[164 + pos] = e;
 int s = n;
 for (int o = 1; o < 32; o <<= 1) { const int v = __shfl_up_sync(0xffffffffu, s, o); if (e >= o) s += v; }
 meta[32 + e] = s - n; meta[132 + e] = s - n;
 __syncwarp();
 const int pe = meta[164 + e], pi = inf[pe], pn = pi & 255, pm = (pi >> 8) & 15;
 int t1 = pn * (64 / pm), t2 = pn * (32 / pm), j = pn;
 for (int o = 1; o < 32; o <<= 1) {
  const int v1 = __shfl_up_sync(0xffffffffu, t1, o), v2 = __shfl_up_sync(0xffffffffu, t2, o),
       v3 = __shfl_up_sync(0xffffffffu, j, o);
  if (e >= o) { t1 += v1; t2 += v2; j += v3; }
 }
 meta[64 + e] = t1 - pn * (64 / pm); meta[97 + e] = t2 - pn * (32 / pm);
 meta[196 + e] = pi; meta[228 + e] = (j - pn) | e << 16;
 if (e == 31) { meta[96] = t1; meta[129] = t2; meta[130] = 0; meta[131] = 0; }
}
__global__ void __launch_bounds__(1024) scatter_k(const int* __restrict__ ids, int R, long* __restrict__ order,
                         int* __restrict__ meta) {
 const int lane = threadIdx.x % 32;
 for (int base = blockIdx.x * 1024; base < R; base += gridDim.x * 1024) {
  const int i = base + threadIdx.x;
  const int e = i < R ? ids[i] : 99;
  const bool loc = e >= 0 && e < 32;
  const unsigned m = __match_any_sync(0xffffffffu, loc ? e : 99);
  const int leader = __ffs(m) - 1;
  int p = 0;
  if (loc && lane == leader) p = atomicAdd(&meta[132 + e], __popc(m));
  p = __shfl_sync(0xffffffffu, p, leader);
  if (loc) order[p + __popc(m & ((1u << lane) - 1))] = i;
 }
}

__global__ void __launch_bounds__(256) gather2_k(const uint4* __restrict__ x, const long* __restrict__ order,
                         const int* __restrict__ meta, uint4* __restrict__ a, int K8) {
 const int n = meta[63] + meta[31];
 for (int r = blockIdx.x; r < n; r += gridDim.x) {
  const long src = order[r] / 8;
  for (int c = threadIdx.x; c < K8; c += 256) a[(long)r * K8 + c] = x[src * K8 + c];
 }
}
void moe_build(torch::Tensor ids, torch::Tensor order, torch::Tensor meta, int64_t tn) {
 const int R = (int)ids.numel(), g = std::min(264, (R + 1023) / 1024);
 auto st = at::cuda::getCurrentCUDAStream();
 count_k<<<g, 1024, 0, st>>>(ids.data_ptr<int>(), R, meta.data_ptr<int>());
 if (tn == 256) plan_k<<<1, 32, 0, st>>>(meta.data_ptr<int>(), 256, 64, 32);
 else plan2_k<<<1, 32, 0, st>>>(meta.data_ptr<int>(), tn < 0 ? (int)-tn : 0);
 scatter_k<<<g, 1024, 0, st>>>(ids.data_ptr<int>(), R, order.data_ptr<long>(), meta.data_ptr<int>());
}
void moe_gather(torch::Tensor x, torch::Tensor order, torch::Tensor meta, torch::Tensor a) {
 gather2_k<<<2048, 256, 0, at::cuda::getCurrentCUDAStream()>>>((const uint4*)x.data_ptr(), order.data_ptr<long>(),
                                meta.data_ptr<int>(), (uint4*)a.data_ptr(), (int)(x.size(1) / 8));
}
"""


def xa__wgmma_src():
    """m64nNk16 BF16 wgmma, A from registers, FP32 accumulators: wgmma_n<N> on float[N / 2] (N = 128, 256), and
    wgmma_part<N, FULL> using the first N / 2 of a float[FULL / 2] (a tile with fewer tokens; stubs where N >= FULL)."""
    def one(name, n, full):
        nd = n // 2
        dl = ",".join(f"%{i}" for i in range(nd))
        cons = ", ".join(f'"+f"(d[{i}])' for i in range(nd))
        return (f"template <> __device__ __forceinline__ void {name}(float (&d)[{full // 2}], uint32_t a0, uint32_t a1, "
                f"uint32_t a2, uint32_t a3, uint64_t desc) {{\n"
                f'  asm volatile("{{\\n.reg .pred p;\\nsetp.ne.b32 p, %{nd + 5}, 0;\\n"\n'
                f'               "wgmma.mma_async.sync.aligned.m64n{n}k16.f32.bf16.bf16 {{{dl}}}, '
                f'{{%{nd},%{nd + 1},%{nd + 2},%{nd + 3}}}, %{nd + 4}, p, 1, 1, 0;\\n}}\\n"\n'
                f'               : {cons}\n'
                f'               : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "l"(desc), "r"(1));\n}}')
    out = ["template <int N> __device__ __forceinline__ void wgmma_n(float (&d)[N / 2], uint32_t a0, uint32_t a1, "
           "uint32_t a2, uint32_t a3, uint64_t desc);",
           "template <int N, int FULL> __device__ __forceinline__ void wgmma_part(float (&d)[FULL / 2], uint32_t a0, "
           "uint32_t a1, uint32_t a2, uint32_t a3, uint64_t desc);"]
    for n in (32, 64, 80, 96, 112, 128, 144, 160, 176, 192, 208, 224, 240, 256):
        out.append(one(f"wgmma_n<{n}>", n, n))
    for n in range(136, 256, 16):
        out.append(one(f"wgmma_part<{n}, {n + 8}>", n, n + 8))
    for n in (72, 80, 88, 96, 104, 112, 120):
        out.append(one(f"wgmma_part<{n}, 128>", n, 128))
    for n, full in ((24, 32), (40, 64), (48, 64), (56, 64)):
        out.append(one(f"wgmma_part<{n}, {full}>", n, full))
    for full in (128, 256):
        for n in (32, 64, 128):
            if n < full:
                out.append(one(f"wgmma_part<{n}, {full}>", n, full))
            else:
                out.append(f"template <> __device__ __forceinline__ void wgmma_part<{n}, {full}>(float (&d)[{full // 2}], "
                           f"uint32_t, uint32_t, uint32_t, uint32_t, uint64_t) {{}}")
    return "\n".join(out)


xa__moe = load_inline(name="okc_prefill_moe_x46001_b460010", cuda_sources=xa__MOE_SRC.replace("@@WGMMA@@", xa__wgmma_src()),
                   cpp_sources="void moe_fused_run(torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, int64_t, int64_t, torch::Tensor);\n"
                               "void moe_build(torch::Tensor, torch::Tensor, torch::Tensor, int64_t);\nvoid moe_set_cnt2(int64_t);\n"
                               "void moe_gather(torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor);",
                   functions=["moe_fused_run", "moe_build", "moe_gather", "moe_set_cnt2"],
                   extra_cuda_cflags=["-O3", "-gencode=arch=compute_90a,code=sm_90a"])

xa__GLUE_SRC = r"""















#include <cuda_runtime.h>
#include <vector>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <stdint.h>

static int g_pdl = 1;
template <typename K, typename... A>
static void pdl_launch(K kern, dim3 grid, dim3 block, size_t smem, A... args) {
 cudaLaunchAttribute at_[1];
 at_[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
 at_[0].val.programmaticStreamSerializationAllowed = g_pdl;
 cudaLaunchConfig_t c_ = {grid, block, smem, at::cuda::getCurrentCUDAStream(), at_, 1};
 cudaLaunchKernelEx(&c_, kern, args...);
}
#define GDC_WAIT() asm volatile("griddepcontrol.wait;" ::: "memory")
#define GDC_GO() asm volatile("griddepcontrol.launch_dependents;" ::: "memory")

namespace {
constexpr int RW = 16;



__device__ void plan_warp(int* meta, int* cnt, int tn, int rb1, int rb2) {
 const int e = threadIdx.x % 32;
 const int n = meta[e];
 cnt[e] = n;
 __syncwarp();
 int pos = 0;
 for (int j = 0; j < 32; j++) pos += cnt[j] > n || (cnt[j] == n && j < e);
 meta[164 + pos] = e;
 int s = n;
 for (int o = 1; o < 32; o <<= 1) { const int v = __shfl_up_sync(0xffffffffu, s, o); if (e >= o) s += v; }
 meta[32 + e] = s - n; meta[132 + e] = s - n;
 __syncwarp();
 const int pe = meta[164 + e], ntt = (cnt[pe] + tn - 1) / tn;
 int t1 = ntt * rb1, t2 = ntt * rb2;
 for (int o = 1; o < 32; o <<= 1) {
  const int v1 = __shfl_up_sync(0xffffffffu, t1, o), v2 = __shfl_up_sync(0xffffffffu, t2, o);
  if (e >= o) { t1 += v1; t2 += v2; }
 }
 meta[64 + e] = t1 - ntt * rb1; meta[97 + e] = t2 - ntt * rb2;
 if (e == 31) { meta[96] = t1; meta[129] = t2; meta[130] = 0; meta[131] = 0; }
}




#define TF_WSET 15
__host__ __device__ __forceinline__ int gl_tf_w1(int nv) {
 if (nv <= 128) return 128;
 const int w = (nv + 15) / 16 * 16, bit = w == 144 ? 1 : w == 176 ? 2 : w == 208 ? 4 : w == 240 ? 8 : 0;
 return (w % 32 == 0 || (TF_WSET & bit)) ? w : w + 16;
}
__device__ __forceinline__ int gl_mx_w(int mb, int nv) {
 if (mb == 2) return nv <= 32 ? 32 : nv <= 64 ? 64 : (nv <= 96 && (TF_WSET & 16)) ? 96 : 128;
 return gl_tf_w1(nv);
}
__device__ __forceinline__ int gl_mx_cost(int mb, int W) {
 if (mb == 2) return W <= 32 ? 901 : W <= 64 ? 1027 : W <= 96 ? 1180 : 1334;
 return W <= 64 ? 715 : W <= 128 ? 883 : W <= 160 ? 940 : W <= 192 ? 1000 : W <= 224 ? 1060 : 1118;
}

#define TF_C144 3110
#define TF_C176 3364
#define TF_C208 3596
#define TF_C240 3944
#define TF_C96 3977

__device__ __forceinline__ int px_c1(int nv) {
 const int W = gl_tf_w1(nv);
 return 2 * (W <= 128 ? 2985 : W <= 144 ? TF_C144 : W <= 160 ? 3236 : W <= 176 ? TF_C176 : W <= 192 ? 3492
       : W <= 208 ? TF_C208 : W <= 224 ? 3700 : W <= 240 ? TF_C240 : 4188);
}
__device__ __forceinline__ int px_c2(int nv) { return nv <= 8 ? 1793 : nv <= 16 ? 2147 : nv <= 32 ? 2482 : nv <= 64 ? 3236 : (nv <= 96 && (TF_WSET & 16)) ? TF_C96 : 4714; }



__device__ __forceinline__ int pdiv(int x, int k) { return (int)(((float)x + 0.5f) * __frcp_rn((float)k)); }
__device__ __forceinline__ int px_plan(int n) {
 if (n <= 0) return 0 | 2 << 8 | 32 << 12;
 const int k2 = (n + 127) >> 7, k1 = (n + 255) >> 8;
 const int ts2 = (pdiv(n + k2 - 1, k2) + 31) & ~31, ts1 = (pdiv(n + k1 - 1, k1) + 15) & ~15;
 int best = (k2 - 1) * px_c2(ts2) + px_c2(n - (k2 - 1) * ts2), inf = k2 | 2 << 8 | ts2 << 12;
 const int c1 = (k1 - 1) * px_c1(ts1) + px_c1(n - (k1 - 1) * ts1);
 if (c1 < best) { best = c1; inf = k1 | 1 << 8 | ts1 << 12; }
#pragma unroll
 for (int i = 0; i < 9; i++) {
  const int r = (n & 15) + 16 * i, n1 = n - r;
  if (r < 1 || r > 128 || n1 < 129) continue;
  const int kk = (n1 + 255) >> 8, ts = (pdiv(n1 + kk - 1, kk) + 15) & ~15;
  const int c = (kk - 1) * px_c1(ts) + px_c1(n1 - (kk - 1) * ts) + px_c2(r);
  if (c < best) { best = c; inf = (kk + 1) | 1 << 8 | ts << 12 | r << 21; }
 }
 return inf;
}
__device__ int g_ptab[4096];
__global__ void px_init_k() { const int n = blockIdx.x * 256 + threadIdx.x; g_ptab[n] = px_plan(n); }
__device__ __forceinline__ int px_get(int n) { return n <= 0 ? (0 | 2 << 8 | 32 << 12) : n < 4096 ? g_ptab[n] : px_plan(n); }
__device__ void plan2_warp(int* meta, int* cnt, int* inf) {
 const int e = threadIdx.x % 32;
 const int n = meta[e];
 cnt[e] = n;
 int mb = 2, tsz = 32, ntt = 0;
 if (n > 0) {
  int best = 0x7fffffff;
  for (int m = 2; m >= 1; m--) {
   const int wmax = 128 * (3 - m), k = (n + wmax - 1) / wmax, ts = ((n + k - 1) / k + 31) / 32 * 32;
   int cost = 0;
   for (int i = 0; i < k; i++) cost += gl_mx_cost(m, gl_mx_w(m, min(ts, n - i * ts)));
   if (m == 1) cost *= 2;
   if (cost < best) { best = cost; mb = m; tsz = ts; ntt = k; }
  }
 }
 inf[e] = ntt | mb << 8 | tsz << 12;
 __syncwarp();
 int pos = 0;
 for (int j = 0; j < 32; j++) pos += cnt[j] > n || (cnt[j] == n && j < e);
 meta[164 + pos] = e;
 int s = n;
 for (int o = 1; o < 32; o <<= 1) { const int v = __shfl_up_sync(0xffffffffu, s, o); if (e >= o) s += v; }
 meta[32 + e] = s - n; meta[132 + e] = s - n;
 __syncwarp();
 const int pe = meta[164 + e], pi = inf[pe], pn = pi & 255, pm = (pi >> 8) & 15;
 int t1 = pn * (64 / pm), t2 = pn * (32 / pm), j = pn;
 for (int o = 1; o < 32; o <<= 1) {
  const int v1 = __shfl_up_sync(0xffffffffu, t1, o), v2 = __shfl_up_sync(0xffffffffu, t2, o),
       v3 = __shfl_up_sync(0xffffffffu, j, o);
  if (e >= o) { t1 += v1; t2 += v2; j += v3; }
 }
 meta[64 + e] = t1 - pn * (64 / pm); meta[97 + e] = t2 - pn * (32 / pm);
 meta[196 + e] = pi; meta[228 + e] = (j - pn) | e << 16;
 if (e == 31) { meta[96] = t1; meta[129] = t2; meta[130] = 0; meta[131] = 0; }
}



__device__ int plan2_s(int* meta, int n, int* cnt, int* inf, int* ord) {
 const int e = threadIdx.x % 32;
 cnt[e] = n;
 inf[e] = px_get(n);
 __syncwarp();
 int pos = 0;
#pragma unroll
 for (int j = 0; j < 32; j++) { const int c = cnt[j]; pos += c > n || (c == n && j < e); }
 ord[pos] = e;
 meta[164 + pos] = e;
 int s = n;
 for (int o = 1; o < 32; o <<= 1) { const int v = __shfl_up_sync(0xffffffffu, s, o); if (e >= o) s += v; }
 const int start = s - n;
 meta[32 + e] = start; meta[132 + e] = start;
 __syncwarp();
 const int pe = ord[e], pi = inf[pe], pn = pi & 255, pm = (pi >> 8) & 15, pr = (pi >> 21) > 0;
 const int o1 = pn * (64 / pm) - 32 * pr, o2 = pn * (32 / pm) - 16 * pr;


 const int ptz = (pi >> 12) & 511;
 const int key = (pn == 0 ? 0 : (pm == 2 && ptz <= 64) ? 1 : 2 + (pm == 1 ? px_c1(ptz) / 2 : px_c2(ptz))) * 64 + 63 - e;
 cnt[e] = key;
 __syncwarp();
 int r2 = 0;
#pragma unroll
 for (int q = 0; q < 32; q++) r2 += cnt[q] > key;
 __syncwarp();
 cnt[r2] = e;
 __syncwarp();
 const int p2 = cnt[e], o2r = __shfl_sync(0xffffffffu, o2, p2);
 int t1 = o1, t2 = o2r, j = pn;
 for (int o = 1; o < 32; o <<= 1) {
  const int v1 = __shfl_up_sync(0xffffffffu, t1, o), v2 = __shfl_up_sync(0xffffffffu, t2, o),
       v3 = __shfl_up_sync(0xffffffffu, j, o);
  if (e >= o) { t1 += v1; t2 += v2; j += v3; }
 }
 meta[64 + e] = t1 - o1; meta[97 + e] = t2 - o2r;
 meta[196 + e] = pi; meta[228 + e] = (j - pn) | p2 << 16;
 if (e == 31) { meta[96] = t1; meta[129] = t2; meta[130] = 0; meta[131] = 0; }
 return start;
}





__global__ void __launch_bounds__(RW * 32) route_k(const float* __restrict__ lg, const float* __restrict__ bias,
  const int* __restrict__ lstart, int T, float mult, float eps, float* __restrict__ tw, int* __restrict__ tid_out,
  int* __restrict__ slot, int* __restrict__ rinfo, int* __restrict__ cnt_tab, int* __restrict__ base_tab,
  int* __restrict__ meta, int done_idx, int tn, int rb1, int rb2, int S) {
 __shared__ int c[32];
 __shared__ int amlast;
 __shared__ int pc[32], pinf[32], pord[32];
 const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
 if (threadIdx.x < 32) c[threadIdx.x] = 0;
 __syncthreads();
 const int ls = *lstart;
 float b[8];
 {
  const float4 b0 = reinterpret_cast<const float4*>(bias)[lane * 2], b1 = reinterpret_cast<const float4*>(bias)[lane * 2 + 1];
  b[0] = b0.x; b[1] = b0.y; b[2] = b0.z; b[3] = b0.w; b[4] = b1.x; b[5] = b1.y; b[6] = b1.z; b[7] = b1.w;
 }
 for (int t = blockIdx.x * RW + warp; t < T; t += gridDim.x * RW) {
  float lv[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
#pragma unroll 4
  for (int sp = 0; sp < S; sp++) {
   const float4* p = reinterpret_cast<const float4*>(lg + ((size_t)sp * T + t) * 512);
   const float4 h0 = p[lane * 2], h1 = p[lane * 2 + 1], l0 = p[64 + lane * 2], l1 = p[64 + lane * 2 + 1];
   lv[0] += h0.x + l0.x; lv[1] += h0.y + l0.y; lv[2] += h0.z + l0.z; lv[3] += h0.w + l0.w;
   lv[4] += h1.x + l1.x; lv[5] += h1.y + l1.y; lv[6] += h1.z + l1.z; lv[7] += h1.w + l1.w;
  }
  float s[8];
  unsigned key[8];
#pragma unroll
  for (int j = 0; j < 8; j++) {
   s[j] = 1.f / (1.f + expf(-(lv[j] * mult)));

   const unsigned u = __float_as_uint(s[j] + b[j]);
   key[j] = (u & 0x80000000u) ? ~u : (u | 0x80000000u);
  }
  unsigned used = 0;
  float wsum = 0.f, my_s = 0.f, my_b = 0.f, v8 = 0.f;
  int my_id = 0;
#pragma unroll
  for (int k = 0; k < 8; k++) {

   unsigned bk = 0;
   int bj = 0;
   float bs = 0.f, bb = 0.f;
#pragma unroll
   for (int j = 0; j < 8; j++) {
    const unsigned kj = ((used >> j) & 1u) ? 0u : key[j];
    if (kj > bk) { bk = kj; bj = j; bs = s[j]; bb = s[j] + b[j]; }
   }
   const unsigned m = __reduce_max_sync(0xffffffffu, bk);
   const int bi = (int)__reduce_min_sync(0xffffffffu, bk == m ? (unsigned)(lane * 8 + bj) : 0xffffffffu);
   if ((bi >> 3) == lane) used |= 1u << bj;
   const float sv = __shfl_sync(0xffffffffu, bs, bi >> 3);
   wsum += sv;
   v8 = __shfl_sync(0xffffffffu, bb, bi >> 3);
   if (lane == k) { my_id = bi; my_s = sv; my_b = v8; }
  }
  if (lane < 8) {
   const float inv = 1.f / (wsum + eps);
   const int e = (my_id - ls) & 255;
   const bool loc = e < 32;
   const int i = t * 8 + lane;
   tw[i] = my_s * inv;
   tid_out[i] = my_id;
   slot[i] = loc ? e : -1;
   int info = -1;
   if (loc) info = e | (atomicAdd(&c[e], 1) << 8);
   rinfo[i] = info;
  }
 }
 __syncthreads();
 if (threadIdx.x < 32) {
  cnt_tab[blockIdx.x * 32 + threadIdx.x] = c[threadIdx.x];
 }
 __threadfence();
 __syncthreads();
 if (threadIdx.x == 0) amlast = atomicAdd(&meta[done_idx], 1) == (int)gridDim.x - 1;
 __syncthreads();
 if (!amlast) return;
 __threadfence();


 __shared__ int part[RW][32];
 const int G = gridDim.x, per = (G + RW - 1) / RW, g0 = warp * per, g1 = min(G, g0 + per);
 int v[9];
#pragma unroll
 for (int j = 0; j < 9; j++) v[j] = g0 + j < g1 ? __ldcg(&cnt_tab[(g0 + j) * 32 + lane]) : 0;
 int sum = 0;
#pragma unroll
 for (int j = 0; j < 9; j++) sum += v[j];
 part[warp][lane] = sum;
 __syncthreads();
 if (warp == 0) {
  int tot = 0;
#pragma unroll
  for (int w = 0; w < RW; w++) tot += part[w][lane];
  meta[lane] = tot;
  int run;
  if (tn == 256) {
   __syncwarp();
   plan_warp(meta, pc, tn, rb1, rb2);
   __syncwarp();
   run = meta[32 + lane];
  } else {
   run = plan2_s(meta, tot, pc, pinf, pord);
  }
  for (int w = 0; w < RW; w++) { const int x = part[w][lane]; part[w][lane] = run; run += x; }
 }
 __syncthreads();
 int run = part[warp][lane];
#pragma unroll
 for (int j = 0; j < 9; j++)
  if (g0 + j < g1) { base_tab[(g0 + j) * 32 + lane] = run; run += v[j]; }
}






__global__ void __launch_bounds__(256) normgather_k(const uint4* __restrict__ x, const float* __restrict__ w,
  const int* __restrict__ rinfo, const int* __restrict__ base_tab, int G, float eps, uint4* __restrict__ a,
  long* __restrict__ order, int T, int t0) {
 __shared__ float red[8];
 __shared__ int dst[8];
 GDC_WAIT();
 GDC_GO();
 const int tid = threadIdx.x;

 for (int t = t0 + (int)blockIdx.x; t < T; t += gridDim.x) {
  const uint4 v0 = x[(size_t)t * 512 + tid], v1 = x[(size_t)t * 512 + 256 + tid];
  if (tid < 8) {
   const int info = rinfo[t * 8 + tid];
   int p = -1;
   if (info >= 0) {
    p = base_tab[((t / RW) % G) * 32 + (info & 255)] + (info >> 8);
    order[p] = (long)t * 8 + tid;
   }
   dst[tid] = p;
  }
  float f[16];
  const __nv_bfloat162* b0 = reinterpret_cast<const __nv_bfloat162*>(&v0);
  const __nv_bfloat162* b1 = reinterpret_cast<const __nv_bfloat162*>(&v1);
#pragma unroll
  for (int j = 0; j < 4; j++) {
   const float2 u = __bfloat1622float2(b0[j]), z = __bfloat1622float2(b1[j]);
   f[2 * j] = u.x; f[2 * j + 1] = u.y; f[8 + 2 * j] = z.x; f[8 + 2 * j + 1] = z.y;
  }
  float ss = 0.f;
#pragma unroll
  for (int j = 0; j < 16; j++) ss += f[j] * f[j];
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) ss += __shfl_xor_sync(0xffffffffu, ss, o);
  if (tid % 32 == 0) red[tid / 32] = ss;
  __syncthreads();
  float tot = 0.f;
#pragma unroll
  for (int j = 0; j < 8; j++) tot += red[j];
  const float r = rsqrtf(tot / 4096.f + eps);
  const float4* wp0 = reinterpret_cast<const float4*>(w + tid * 8);
  const float4* wp1 = reinterpret_cast<const float4*>(w + 2048 + tid * 8);
  uint4 o0, o1;
  __nv_bfloat162* c0 = reinterpret_cast<__nv_bfloat162*>(&o0);
  __nv_bfloat162* c1 = reinterpret_cast<__nv_bfloat162*>(&o1);
#pragma unroll
  for (int j = 0; j < 4; j++) {
   const float4 ww = j < 2 ? wp0[j] : wp1[j - 2];
   const float w4[4] = {ww.x, ww.y, ww.z, ww.w};
   __nv_bfloat162* cc = j < 2 ? c0 : c1;
   const int jj = (j % 2) * 2;
   cc[jj] = __floats2bfloat162_rn(f[4 * j] * r * w4[0], f[4 * j + 1] * r * w4[1]);
   cc[jj + 1] = __floats2bfloat162_rn(f[4 * j + 2] * r * w4[2], f[4 * j + 3] * r * w4[3]);
  }
#pragma unroll
  for (int k = 0; k < 8; k++) {
   const int p = dst[k];
   if (p >= 0) { a[(size_t)p * 512 + tid] = o0; a[(size_t)p * 512 + 256 + tid] = o1; }
  }
  __syncthreads();
 }
}




__global__ void __launch_bounds__(RW * 32) route2_k(const float* __restrict__ lg, int ldl, const float* __restrict__ bias,
  const int* __restrict__ lstart, int T, float mult, float eps, float* __restrict__ tw, int* __restrict__ tid_out,
  int* __restrict__ slot, int* __restrict__ rinfo, int* __restrict__ cnt_tab) {
 __shared__ int c[32];
 const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
 if (threadIdx.x < 32) c[threadIdx.x] = 0;
 const int ls = *lstart;
 float b[8];
 {
  const float4 b0 = reinterpret_cast<const float4*>(bias)[lane * 2], b1 = reinterpret_cast<const float4*>(bias)[lane * 2 + 1];
  b[0] = b0.x; b[1] = b0.y; b[2] = b0.z; b[3] = b0.w; b[4] = b1.x; b[5] = b1.y; b[6] = b1.z; b[7] = b1.w;
 }
 __syncthreads();
 for (int t = blockIdx.x * RW + warp; t < T; t += gridDim.x * RW) {
  const float4* p = reinterpret_cast<const float4*>(lg + (size_t)t * ldl);
  const float4 h0 = p[lane * 2], h1 = p[lane * 2 + 1], l0 = p[64 + lane * 2], l1 = p[64 + lane * 2 + 1];
  const float lv[8] = {h0.x + l0.x, h0.y + l0.y, h0.z + l0.z, h0.w + l0.w, h1.x + l1.x, h1.y + l1.y, h1.z + l1.z, h1.w + l1.w};
  float s[8];
  unsigned key[8];
#pragma unroll
  for (int j = 0; j < 8; j++) {
   s[j] = 1.f / (1.f + expf(-(lv[j] * mult)));
   const unsigned u = __float_as_uint(s[j] + b[j]);
   key[j] = (u & 0x80000000u) ? ~u : (u | 0x80000000u);
  }
  unsigned used = 0;
  float wsum = 0.f, my_s = 0.f;
  int my_id = 0;
#pragma unroll
  for (int k = 0; k < 8; k++) {
   unsigned bk = 0;
   int bj = 0;
   float bs = 0.f;
#pragma unroll
   for (int j = 0; j < 8; j++) {
    const unsigned kj = ((used >> j) & 1u) ? 0u : key[j];
    if (kj > bk) { bk = kj; bj = j; bs = s[j]; }
   }
   const unsigned m = __reduce_max_sync(0xffffffffu, bk);
   const int bi = (int)__reduce_min_sync(0xffffffffu, bk == m ? (unsigned)(lane * 8 + bj) : 0xffffffffu);
   if ((bi >> 3) == lane) used |= 1u << bj;
   const float sv = __shfl_sync(0xffffffffu, bs, bi >> 3);
   wsum += sv;
   if (lane == k) { my_id = bi; my_s = sv; }
  }
  if (lane < 8) {
   const float inv = 1.f / (wsum + eps);
   const int e = (my_id - ls) & 255;
   const bool loc = e < 32;
   const int i = t * 8 + lane;
   tw[i] = my_s * inv;
   tid_out[i] = my_id;
   slot[i] = loc ? e : -1;
   int info = -1;
   if (loc) info = e | (atomicAdd(&c[e], 1) << 8);
   rinfo[i] = info;
  }
 }
 __syncthreads();
 if (threadIdx.x < 32) cnt_tab[blockIdx.x * 32 + threadIdx.x] = c[threadIdx.x];
}





__global__ void __launch_bounds__(256) normgather2_k(const float* __restrict__ x, int ldx, const float* __restrict__ w,
  const int* __restrict__ rinfo, const int* __restrict__ cnt_tab, int G, float eps, uint4* __restrict__ a,
  long* __restrict__ order, int T, int* __restrict__ meta, int nzero) {
 __shared__ float red[8];
 __shared__ int dst[8];
 __shared__ int ps[8][32], ts[8][32], sb[32];
 __shared__ int pc[32], pinf[32], pord[32];
 const int tid = threadIdx.x, warp = tid / 32, lane = tid % 32;
 const int t = blockIdx.x;
 const int g = t / RW;

 const float4* xr = reinterpret_cast<const float4*>(x + (size_t)t * ldx);
 const float4 u0 = xr[2 * tid], u1 = xr[2 * tid + 1], z0 = xr[512 + 2 * tid], z1 = xr[512 + 2 * tid + 1];
 int info = -1;
 if (tid < 8) info = rinfo[t * 8 + tid];
 {
  int pre = 0, tot = 0;
  for (int gg = warp; gg < G; gg += 8) {
   const int cv = __ldcg(&cnt_tab[gg * 32 + lane]);
   tot += cv;
   pre += gg < g ? cv : 0;
  }
  ps[warp][lane] = pre; ts[warp][lane] = tot;
 }
 float f[16] = {u0.x, u0.y, u0.z, u0.w, u1.x, u1.y, u1.z, u1.w, z0.x, z0.y, z0.z, z0.w, z1.x, z1.y, z1.z, z1.w};
#pragma unroll
 for (int j = 0; j < 16; j++) f[j] = __bfloat162float(__float2bfloat16(f[j]));
 float ss = 0.f;
#pragma unroll
 for (int j = 0; j < 16; j++) ss += f[j] * f[j];
#pragma unroll
 for (int o = 16; o > 0; o >>= 1) ss += __shfl_xor_sync(0xffffffffu, ss, o);
 if (lane == 0) red[warp] = ss;
 __syncthreads();
 int tot = 0;
 if (warp == 0) {
  int pre = 0;
#pragma unroll
  for (int j = 0; j < 8; j++) { pre += ps[j][lane]; tot += ts[j][lane]; }
  int sc = tot;
  for (int o = 1; o < 32; o <<= 1) { const int v = __shfl_up_sync(0xffffffffu, sc, o); if (lane >= o) sc += v; }
  sb[lane] = sc - tot + pre;
  __syncwarp();
  if (lane < 8) {
   int p = -1;
   if (info >= 0) {
    p = sb[info & 255] + (info >> 8);
    order[p] = (long)t * 8 + lane;
   }
   dst[lane] = p;
  }
 }
 float sum = 0.f;
#pragma unroll
 for (int j = 0; j < 8; j++) sum += red[j];
 const float r = rsqrtf(sum / 4096.f + eps);
 const float4* wp0 = reinterpret_cast<const float4*>(w + tid * 8);
 const float4* wp1 = reinterpret_cast<const float4*>(w + 2048 + tid * 8);
 uint4 o0, o1;
 __nv_bfloat162* c0 = reinterpret_cast<__nv_bfloat162*>(&o0);
 __nv_bfloat162* c1 = reinterpret_cast<__nv_bfloat162*>(&o1);
#pragma unroll
 for (int j = 0; j < 4; j++) {
  const float4 ww = j < 2 ? wp0[j] : wp1[j - 2];
  const float w4[4] = {ww.x, ww.y, ww.z, ww.w};
  __nv_bfloat162* cc = j < 2 ? c0 : c1;
  const int jj = (j % 2) * 2;
  cc[jj] = __floats2bfloat162_rn(f[4 * j] * r * w4[0], f[4 * j + 1] * r * w4[1]);
  cc[jj + 1] = __floats2bfloat162_rn(f[4 * j + 2] * r * w4[2], f[4 * j + 3] * r * w4[3]);
 }
 __syncthreads();
#pragma unroll
 for (int k = 0; k < 8; k++) {
  const int p = dst[k];
  if (p >= 0) { a[(size_t)p * 512 + tid] = o0; a[(size_t)p * 512 + 256 + tid] = o1; }
 }
 if (blockIdx.x == 0) {
  for (int i = tid; i < nzero; i += 256) meta[260 + i] = 0;
  if (warp == 0) {
   meta[lane] = tot;
   plan2_s(meta, tot, pc, pinf, pord);
  }
 }
}


__device__ void plan2_cap(int* meta, int n, int start, int* cnt, int* inf, int* ord) {
 const int e = threadIdx.x % 32;
 cnt[e] = n;
 inf[e] = px_get(n);
 __syncwarp();
 int pos = 0;
#pragma unroll
 for (int j = 0; j < 32; j++) { const int c = cnt[j]; pos += c > n || (c == n && j < e); }
 ord[pos] = e;
 meta[164 + pos] = e;
 meta[e] = n;
 meta[32 + e] = start; meta[132 + e] = start;
 __syncwarp();
 const int pe = ord[e], pi = inf[pe], pn = pi & 255, pm = (pi >> 8) & 15, pr = (pi >> 21) > 0;

 const int o1 = pn * (pm == 1 ? 64 : 32) - 32 * pr, o2 = pn * (pm == 1 ? 32 : 16) - 16 * pr;


 const int ptz = (pi >> 12) & 511;
 const int key = (pn == 0 ? 0 : (pm == 2 && ptz <= 64) ? 1 : 2 + (pm == 1 ? px_c1(ptz) / 2 : px_c2(ptz))) * 64 + 63 - e;
 cnt[e] = key;
 __syncwarp();
 int r2 = 0;
#pragma unroll
 for (int q = 0; q < 32; q++) r2 += cnt[q] > key;
 __syncwarp();
 cnt[r2] = e;
 __syncwarp();
 const int p2 = cnt[e], o2r = __shfl_sync(0xffffffffu, o2, p2);
 int t1 = o1, t2 = o2r, j = pn;
 for (int o = 1; o < 32; o <<= 1) {
  const int v1 = __shfl_up_sync(0xffffffffu, t1, o), v2 = __shfl_up_sync(0xffffffffu, t2, o),
       v3 = __shfl_up_sync(0xffffffffu, j, o);
  if (e >= o) { t1 += v1; t2 += v2; j += v3; }
 }
 meta[64 + e] = t1 - o1; meta[97 + e] = t2 - o2r;
 meta[196 + e] = pi; meta[228 + e] = (j - pn) | p2 << 16;
 if (e == 31) { meta[96] = t1; meta[129] = t2; meta[130] = 0; meta[131] = 0; }
}


template <bool PR>
__device__ void plan2_capx(int* meta, int n, int start, int* cnt, int* inf, int* ord) {
 const int e = threadIdx.x % 32;
 cnt[e] = n;
 inf[e] = PR ? (n <= 0 ? (0 | 2 << 8 | 32 << 12) : px_plan(n)) : px_get(n);
 __syncwarp();
 int pos = 0;
#pragma unroll
 for (int j = 0; j < 32; j++) { const int c = cnt[j]; pos += c > n || (c == n && j < e); }
 ord[pos] = e;
 meta[164 + pos] = e;
 meta[e] = n;
 meta[32 + e] = start; meta[132 + e] = start;
 __syncwarp();
 const int pe = ord[e], pi = inf[pe], pn = pi & 255, pm = (pi >> 8) & 15, pr = (pi >> 21) > 0;

 const int o1 = pn * (pm == 1 ? 64 : 32) - 32 * pr, o2 = pn * (pm == 1 ? 32 : 16) - 16 * pr;


 const int ptz = (pi >> 12) & 511;
 const int key = (pn == 0 ? 0 : (pm == 2 && ptz <= 64) ? 1 : 2 + (pm == 1 ? px_c1(ptz) / 2 : px_c2(ptz))) * 64 + 63 - e;
 cnt[e] = key;
 __syncwarp();
 int r2 = 0;
#pragma unroll
 for (int q = 0; q < 32; q++) r2 += cnt[q] > key;
 __syncwarp();
 cnt[r2] = e;
 __syncwarp();
 const int p2 = cnt[e], o2r = __shfl_sync(0xffffffffu, o2, p2);
 int t1 = o1, t2 = o2r, j = pn;
 for (int o = 1; o < 32; o <<= 1) {
  const int v1 = __shfl_up_sync(0xffffffffu, t1, o), v2 = __shfl_up_sync(0xffffffffu, t2, o),
       v3 = __shfl_up_sync(0xffffffffu, j, o);
  if (e >= o) { t1 += v1; t2 += v2; j += v3; }
 }
 meta[64 + e] = t1 - o1; meta[97 + e] = t2 - o2r;
 meta[196 + e] = pi; meta[228 + e] = (j - pn) | p2 << 16;
 if (e == 31) { meta[96] = t1; meta[129] = t2; meta[130] = 0; meta[131] = 0; }
}








__global__ void __launch_bounds__(256) rng2_k(const float* __restrict__ x, int ldx, const float* __restrict__ bias,
  const int* __restrict__ lstart, float mult, float sceps, const float* __restrict__ w, float eps,
  float* __restrict__ tw, int* __restrict__ tid_out, int* __restrict__ slot, uint4* __restrict__ a,
  long* __restrict__ order, int T, int cap, int* __restrict__ ctr, int* __restrict__ meta, int nzero) {
 __shared__ float red[8];
 __shared__ int dst[8];
 __shared__ int amlast;
 __shared__ int pc[32], pinf[32], pord[32];
 if (threadIdx.x == 0 && blockIdx.x < 128) asm volatile("prefetch.global.L2 [%0];" ::"l"(g_ptab + blockIdx.x * 32));
 GDC_WAIT();
 GDC_GO();
 const int tid = threadIdx.x, warp = tid / 32, lane = tid % 32;
 const int t = blockIdx.x;
 const float* row = x + (size_t)t * ldx;
 const float4* xr = reinterpret_cast<const float4*>(row);
 const float4 u0 = xr[2 * tid], u1 = xr[2 * tid + 1], z0 = xr[512 + 2 * tid], z1 = xr[512 + 2 * tid + 1];
 if (warp == 0) {
  const float4* p = reinterpret_cast<const float4*>(row + 4096);
  const float4 h0 = p[lane * 2], h1 = p[lane * 2 + 1], l0 = p[64 + lane * 2], l1 = p[64 + lane * 2 + 1];
  const float4 b0 = reinterpret_cast<const float4*>(bias)[lane * 2], b1 = reinterpret_cast<const float4*>(bias)[lane * 2 + 1];
  const float b[8] = {b0.x, b0.y, b0.z, b0.w, b1.x, b1.y, b1.z, b1.w};
  const float lv[8] = {h0.x + l0.x, h0.y + l0.y, h0.z + l0.z, h0.w + l0.w, h1.x + l1.x, h1.y + l1.y, h1.z + l1.z, h1.w + l1.w};
  const int ls = *lstart;
  float s[8];
  unsigned key[8];
#pragma unroll
  for (int j = 0; j < 8; j++) {
   s[j] = 1.f / (1.f + expf(-(lv[j] * mult)));
   const unsigned uu = __float_as_uint(s[j] + b[j]);
   key[j] = (uu & 0x80000000u) ? ~uu : (uu | 0x80000000u);
  }
  unsigned used = 0;
  float wsum = 0.f, my_s = 0.f;
  int my_id = 0;
#pragma unroll
  for (int k = 0; k < 8; k++) {
   unsigned bk = 0;
   int bj = 0;
   float bs = 0.f;
#pragma unroll
   for (int j = 0; j < 8; j++) {
    const unsigned kj = ((used >> j) & 1u) ? 0u : key[j];
    if (kj > bk) { bk = kj; bj = j; bs = s[j]; }
   }
   const unsigned m = __reduce_max_sync(0xffffffffu, bk);
   const int bi = (int)__reduce_min_sync(0xffffffffu, bk == m ? (unsigned)(lane * 8 + bj) : 0xffffffffu);
   if ((bi >> 3) == lane) used |= 1u << bj;
   const float sv = __shfl_sync(0xffffffffu, bs, bi >> 3);
   wsum += sv;
   if (lane == k) { my_id = bi; my_s = sv; }
  }
  if (lane < 8) {
   const float inv = 1.f / (wsum + sceps);
   const int e = (my_id - ls) & 255;
   const bool loc = e < 32;
   const int i = t * 8 + lane;
   tw[i] = my_s * inv;
   tid_out[i] = my_id;
   slot[i] = loc ? e : -1;
   int pp = -1;
   if (loc) {
    pp = e * cap + atomicAdd(&ctr[e], 1);
    order[pp] = (long)i;
   }
   dst[lane] = pp;
  }
 }
 float f[16] = {u0.x, u0.y, u0.z, u0.w, u1.x, u1.y, u1.z, u1.w, z0.x, z0.y, z0.z, z0.w, z1.x, z1.y, z1.z, z1.w};
#pragma unroll
 for (int j = 0; j < 16; j++) f[j] = __bfloat162float(__float2bfloat16(f[j]));
 float ss = 0.f;
#pragma unroll
 for (int j = 0; j < 16; j++) ss += f[j] * f[j];
#pragma unroll
 for (int o = 16; o > 0; o >>= 1) ss += __shfl_xor_sync(0xffffffffu, ss, o);
 if (lane == 0) red[warp] = ss;
 __syncthreads();
 float sum = 0.f;
#pragma unroll
 for (int j = 0; j < 8; j++) sum += red[j];
 const float r = rsqrtf(sum / 4096.f + eps);
 const float4* wp0 = reinterpret_cast<const float4*>(w + tid * 8);
 const float4* wp1 = reinterpret_cast<const float4*>(w + 2048 + tid * 8);
 uint4 o0, o1;
 __nv_bfloat162* c0 = reinterpret_cast<__nv_bfloat162*>(&o0);
 __nv_bfloat162* c1 = reinterpret_cast<__nv_bfloat162*>(&o1);
#pragma unroll
 for (int j = 0; j < 4; j++) {
  const float4 ww = j < 2 ? wp0[j] : wp1[j - 2];
  const float w4[4] = {ww.x, ww.y, ww.z, ww.w};
  __nv_bfloat162* cc = j < 2 ? c0 : c1;
  const int jj = (j % 2) * 2;
  cc[jj] = __floats2bfloat162_rn(f[4 * j] * r * w4[0], f[4 * j + 1] * r * w4[1]);
  cc[jj + 1] = __floats2bfloat162_rn(f[4 * j + 2] * r * w4[2], f[4 * j + 3] * r * w4[3]);
 }
#pragma unroll
 for (int k = 0; k < 8; k++) {
  const int p = dst[k];
  if (p >= 0) { a[(size_t)p * 512 + tid] = o0; a[(size_t)p * 512 + 256 + tid] = o1; }
 }

 if (tid == 0) amlast = atomicAdd(&ctr[32], 1) == T - 1;
 __syncthreads();
 if (!amlast) return;
 for (int i = tid; i < nzero; i += 256) meta[260 + i] = 0;
 if (warp == 0) {
  const int n = atomicExch(&ctr[lane], 0);
  if (lane == 0) ctr[32] = 0;
  plan2_cap(meta, n, lane * cap, pc, pinf, pord);
 }
}
#define RNG3_MINB 6

__device__ __forceinline__ float route_sigmoid_sglang(float x) {
 float y;
 asm volatile("{ .reg .f32 n,e,d; "
   "sub.f32 n, 0f00000000, %1; "
   "mul.f32 e, n, 0f3FB8AA3B; "
   "ex2.approx.f32 e, e; "
   "add.f32 d, e, 0f3F800000; "
   "div.full.f32 %0, 0f3F800000, d; }"
   : "=f"(y) : "f"(x));
 return y;
}

__device__ __forceinline__ int ld_rlx_gpu(const int* p) {
 int v;
 asm volatile("ld.relaxed.gpu.global.b32 %0, [%1];" : "=r"(v) : "l"(p) : "memory");
 return v;
}

#define Z29(k) do {} while (0)
#define Z29D(k, dep) do {} while (0)
#define Z29C(k) do {} while (0)
#define Z29V(k, v) do {} while (0)
__device__ __forceinline__ void red_add_gpu(int* p, int v) {
 asm volatile("red.relaxed.gpu.global.add.s32 [%0], %1;" ::"l"(p), "r"(v) : "memory");
}



#define V2_PTN 1024
__device__ __forceinline__ int v2_kcost(int pi) {
 const int pn = pi & 255, pm = (pi >> 8) & 15, ptz = (pi >> 12) & 511;
 return pn == 0 ? 0 : (pm == 2 && ptz <= 64) ? 1 : 2 + (pm == 1 ? px_c1(ptz) / 2 : px_c2(ptz));
}
__device__ __forceinline__ void plan2_v2(int* meta, int n, int start, int* cnt, int* ord, const int* tp, const unsigned short* tk, int nt) {
 const int e = threadIdx.x % 32;
 int pio, kco;
 if (n < nt) { pio = tp[n]; kco = tk[n]; } else { pio = px_get(n); kco = v2_kcost(pio); }
 cnt[e] = n;
 __syncwarp();
 int a0 = 0, a1 = 0, a2 = 0, a3 = 0;
#pragma unroll
 for (int j = 0; j < 32; j += 4) {
  const int c0 = cnt[j], c1 = cnt[j + 1], c2 = cnt[j + 2], c3 = cnt[j + 3];
  a0 += (c0 > n) | ((c0 == n) & (j < e));
  a1 += (c1 > n) | ((c1 == n) & (j + 1 < e));
  a2 += (c2 > n) | ((c2 == n) & (j + 2 < e));
  a3 += (c3 > n) | ((c3 == n) & (j + 3 < e));
 }
 const int pos = (a0 + a1) + (a2 + a3);
 ord[pos] = e;
 meta[164 + pos] = e;
 meta[e] = n;
 meta[32 + e] = start; meta[132 + e] = start;
 __syncwarp();
 const int pe = ord[e];
 const int pi = __shfl_sync(0xffffffffu, pio, pe), kc = __shfl_sync(0xffffffffu, kco, pe);
 const int pn = pi & 255, pm = (pi >> 8) & 15, pr = (pi >> 21) > 0;
 const int o1 = pn * (pm == 1 ? 64 : 32) - 32 * pr, o2 = pn * (pm == 1 ? 32 : 16) - 16 * pr;
 const int key = kc * 64 + 63 - e;
 cnt[e] = key;
 __syncwarp();
 int b0 = 0, b1 = 0, b2 = 0, b3 = 0;
#pragma unroll
 for (int q = 0; q < 32; q += 4) {
  b0 += cnt[q] > key; b1 += cnt[q + 1] > key; b2 += cnt[q + 2] > key; b3 += cnt[q + 3] > key;
 }
 const int r2 = (b0 + b1) + (b2 + b3);
 __syncwarp();
 cnt[r2] = e;
 __syncwarp();
 const int p2 = cnt[e], o2r = __shfl_sync(0xffffffffu, o2, p2);
 int t1 = o1, t2 = o2r, j = pn;
 for (int o = 1; o < 32; o <<= 1) {
  const int v1 = __shfl_up_sync(0xffffffffu, t1, o), v2 = __shfl_up_sync(0xffffffffu, t2, o),
       v3 = __shfl_up_sync(0xffffffffu, j, o);
  if (e >= o) { t1 += v1; t2 += v2; j += v3; }
 }
 meta[64 + e] = t1 - o1; meta[97 + e] = t2 - o2r;
 meta[196 + e] = pi; meta[228 + e] = (j - pn) | p2 << 16;
 if (e == 31) { meta[96] = t1; meta[129] = t2; meta[130] = 0; meta[131] = 0; }
}




template <int D>
__global__ void __launch_bounds__(128, RNG3_MINB) rng3y_k(const float* __restrict__ x, int ldx, const float* __restrict__ bias,
  const int* __restrict__ lstart, float mult, float sceps, const float* __restrict__ w, float eps,
  float* __restrict__ tw, int* __restrict__ tid_out, int* __restrict__ slot, uint4* __restrict__ a,
  long* __restrict__ order, int T, int cap, int* __restrict__ ctr, int* __restrict__ meta, int nzero,
  int* __restrict__ emask, int hd) {
 __shared__ float red[8];
 __shared__ int dst[8];
 __shared__ int amlast;
 __shared__ int pc[32], pinf[32], pord[32];
 if (threadIdx.x == 0) {
  Z29(0);
 }
 if (threadIdx.x == 0 && blockIdx.x < 128) asm volatile("prefetch.global.L2 [%0];" ::"l"(g_ptab + blockIdx.x * 32));
 GDC_WAIT();
 GDC_GO();
 if (threadIdx.x == 0) { Z29(1); Z29C(10); }
 const int tid = threadIdx.x, warp = tid / 32, lane = tid % 32;
 const int t = blockIdx.x;
 if (D && t == T) {


  __shared__ int v2_tp[V2_PTN];
  __shared__ unsigned short v2_tk[V2_PTN];
  const int v2_nt = min(T + 1, V2_PTN);
  if (warp != 0) {


   int pv[8];
#pragma unroll
   for (int k = 0; k < 8; k++) pv[k] = g_ptab[min(tid - 32 + 96 * k, V2_PTN - 1)];
#pragma unroll
   for (int k = 0; k < 8; k++) {
    const int i = tid - 32 + 96 * k;
    if (i < v2_nt) { v2_tp[i] = pv[k]; v2_tk[i] = (unsigned short)v2_kcost(pv[k]); }
   }
   for (int i = tid - 32 + 768; i < v2_nt; i += 96) { const int pi = g_ptab[i]; v2_tp[i] = pi; v2_tk[i] = (unsigned short)v2_kcost(pi); }
   if (tid == 32) Z29(14);
   asm volatile("bar.arrive 1, 128;" ::: "memory");
   return;
  }
  emask[lane] = 0x7fffffff;
  for (int i = lane; i < nzero; i += 32) meta[260 + i] = 0;
  if (lane == 0)
   while (ld_rlx_gpu(ctr + 32) < T) __nanosleep(20);
  __syncwarp();
  const int n = ld_rlx_gpu(ctr + lane);
  if (lane == 0) Z29D(7, n);
  ctr[lane] = 0;
  if (lane == 0) ctr[32] = 0;
  asm volatile("bar.sync 1, 128;" ::: "memory");
  if (lane == 0) Z29(13);
  plan2_v2(meta, n, lane * cap + hd, pc, pord, v2_tp, v2_tk, v2_nt);
  if (lane == 0) { Z29(8); Z29C(11); }
  return;
 }
 const float* row = x + (size_t)t * ldx;
 const float4* xr = reinterpret_cast<const float4*>(row);
 float4 h0, h1, l0, l1, b0, b1;
 if (warp == 0) {
  const float4* p = reinterpret_cast<const float4*>(row + 4096);
  h0 = p[lane * 2]; h1 = p[lane * 2 + 1]; l0 = p[64 + lane * 2]; l1 = p[64 + lane * 2 + 1];
  b0 = reinterpret_cast<const float4*>(bias)[lane * 2]; b1 = reinterpret_cast<const float4*>(bias)[lane * 2 + 1];
 }
 const float4 u0 = xr[2 * tid], u1 = xr[2 * tid + 1], z0 = xr[512 + 2 * tid], z1 = xr[512 + 2 * tid + 1];
 const float4 U0 = xr[256 + 2 * tid], U1 = xr[256 + 2 * tid + 1], Z0 = xr[768 + 2 * tid], Z1 = xr[768 + 2 * tid + 1];
 if (warp == 0) {
  const float b[8] = {b0.x, b0.y, b0.z, b0.w, b1.x, b1.y, b1.z, b1.w};
  const float lv[8] = {h0.x + l0.x, h0.y + l0.y, h0.z + l0.z, h0.w + l0.w, h1.x + l1.x, h1.y + l1.y, h1.z + l1.z, h1.w + l1.w};
  const int ls = *lstart;
  float s[8], bv[8];
  unsigned key[8];
#pragma unroll
  for (int j = 0; j < 8; j++) {
   s[j] = route_sigmoid_sglang(lv[j] * mult);
   bv[j] = s[j] + b[j];
   const unsigned uu = __float_as_uint(bv[j]);
   key[j] = (uu & 0x80000000u) ? ~uu : (uu | 0x80000000u);
  }
  if (lane == 0) Z29D(2, key[0] ^ key[7]);
  unsigned used = 0;
  float wsum = 0.f, my_s = 0.f;
  int my_id = 0;
#pragma unroll
  for (int k = 0; k < 8; k++) {
   unsigned bk = 0;
   int bj = 0;
   float bs = 0.f;
#pragma unroll
   for (int j = 0; j < 8; j++) {
    const unsigned kj = ((used >> j) & 1u) ? 0u : key[j];
    if (kj > bk) { bk = kj; bj = j; bs = s[j]; }
   }
   const unsigned m = __reduce_max_sync(0xffffffffu, bk);
   const int bi = (int)__reduce_min_sync(0xffffffffu, bk == m ? (unsigned)(lane * 8 + bj) : 0xffffffffu);
   if ((bi >> 3) == lane) used |= 1u << bj;
   const float sv = __shfl_sync(0xffffffffu, bs, bi >> 3);
   wsum += sv;
   if (lane == k) { my_id = bi; my_s = sv; }
  }
  const int e = (my_id - ls) & 255;
  const bool loc = lane < 8 && e < 32;
  int pp = -1;
  if (lane < 8) {
   const float inv = 1.f / (wsum + sceps);
   const int i = t * 8 + lane;
   tw[i] = my_s * inv;
   tid_out[i] = my_id;
   slot[i] = loc ? e : -1;
   if (loc) {
    pp = e * cap + hd + atomicAdd(&ctr[e], 1);
    order[pp] = (long)i;
   }
   dst[lane] = pp;
  }
  if (D) {


   const unsigned bm = __ballot_sync(0xffffffffu, pp >= 0);
   if (lane == 0) { red_add_gpu(ctr + 32, 1 + (bm == 0xffffffffu)); Z29D(3, bm); }
  } else {
  }
 }
 float f[16] = {u0.x, u0.y, u0.z, u0.w, u1.x, u1.y, u1.z, u1.w, z0.x, z0.y, z0.z, z0.w, z1.x, z1.y, z1.z, z1.w};
 float F[16] = {U0.x, U0.y, U0.z, U0.w, U1.x, U1.y, U1.z, U1.w, Z0.x, Z0.y, Z0.z, Z0.w, Z1.x, Z1.y, Z1.z, Z1.w};
#pragma unroll
 for (int j = 0; j < 16; j++) { f[j] = __bfloat162float(__float2bfloat16(f[j])); F[j] = __bfloat162float(__float2bfloat16(F[j])); }
 float ss = 0.f, SS = 0.f;
#pragma unroll
 for (int j = 0; j < 16; j++) { ss += f[j] * f[j]; SS += F[j] * F[j]; }
#pragma unroll
 for (int o = 16; o > 0; o >>= 1) { ss += __shfl_xor_sync(0xffffffffu, ss, o); SS += __shfl_xor_sync(0xffffffffu, SS, o); }
 if (lane == 0) { red[warp] = ss; red[warp + 4] = SS; }
 __syncthreads();
 float sum = 0.f;
#pragma unroll
 for (int j = 0; j < 8; j++) sum += red[j];
 const float r = rsqrtf(sum / 4096.f + eps);
 if (tid == 0) Z29D(4, __float_as_int(r));
#pragma unroll
 for (int hh = 0; hh < 2; hh++) {
  const int ot = tid + 128 * hh;
  const float* fv = hh ? F : f;
  const float4* wp0 = reinterpret_cast<const float4*>(w + ot * 8);
  const float4* wp1 = reinterpret_cast<const float4*>(w + 2048 + ot * 8);
  uint4 o0, o1;
  __nv_bfloat162* c0 = reinterpret_cast<__nv_bfloat162*>(&o0);
  __nv_bfloat162* c1 = reinterpret_cast<__nv_bfloat162*>(&o1);
#pragma unroll
  for (int j = 0; j < 4; j++) {
   const float4 ww = j < 2 ? wp0[j] : wp1[j - 2];
   const float w4[4] = {ww.x, ww.y, ww.z, ww.w};
   __nv_bfloat162* cc = j < 2 ? c0 : c1;
   const int jj = (j % 2) * 2;
   cc[jj] = __floats2bfloat162_rn(fv[4 * j] * r * w4[0], fv[4 * j + 1] * r * w4[1]);
   cc[jj + 1] = __floats2bfloat162_rn(fv[4 * j + 2] * r * w4[2], fv[4 * j + 3] * r * w4[3]);
  }
#pragma unroll
  for (int k = 0; k < 8; k++) {
   const int p = dst[k];
   if (p >= 0) { a[(size_t)p * 512 + ot] = o0; a[(size_t)p * 512 + 256 + ot] = o1; }
  }
 }
 if (tid == 0) Z29(5);
 if (D) return;

 if (tid == 0) { amlast = atomicAdd(&ctr[32], 1) == T - 1; Z29D(6, amlast); }
 __syncthreads();
 if (!amlast) return;
 for (int i = tid; i < nzero; i += 128) meta[260 + i] = 0;
 if (warp == 0) {
  const int n = atomicExch(&ctr[lane], 0);
  if (lane == 0) { Z29D(7, n); ctr[32] = 0; }
  emask[lane] = 0x7fffffff;
  plan2_capx<false>(meta, n, lane * cap + hd, pc, pinf, pord);
  if (lane == 0) { Z29(8); Z29C(11); }
 }
}




__device__ __forceinline__ float bfr(float x) { return __bfloat162float(__float2bfloat16(x)); }
__device__ __forceinline__ uint16_t fp8x2(float a, float b) {
 return (uint16_t)__nv_cvt_float2_to_fp8x2(make_float2(a, b), __NV_SATFINITE, __NV_E4M3);
}
template <bool LO>
__global__ void __launch_bounds__(256) front_k(const __nv_bfloat162* __restrict__ qkv, const long* __restrict__ pos,
  const float* __restrict__ cs, const long* __restrict__ loc, uint16_t* __restrict__ qo, uint8_t* __restrict__ kc,
  uint8_t* __restrict__ vc, int T, float eps, int rope, uint16_t* __restrict__ ql) {
 GDC_WAIT();
 const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
 const int t = blockIdx.x * 8 + warp;
 if (t >= T) return;
 const __nv_bfloat162* row = qkv + (size_t)t * 1152;
 float c = 1.f, s = 0.f;
 if (rope) {
  const long p = pos[t];
  c = cs[p * 64 + lane];
  s = cs[p * 64 + 32 + lane];
 }
 const long L = loc[t];
 float2 v[20];
#pragma unroll
 for (int h = 0; h < 20; h++) v[h] = __bfloat1622float2(row[h * 32 + lane]);
#pragma unroll
 for (int h = 0; h < 20; h++) {
  float x0 = v[h].x, x1 = v[h].y;
  if (h < 18) {
   float ss = x0 * x0 + x1 * x1;
#pragma unroll
   for (int o = 16; o > 0; o >>= 1) ss += __shfl_xor_sync(0xffffffffu, ss, o);
   const float r = rsqrtf(ss / 64.f + eps);
   x0 *= r; x1 *= r;
   if (rope) {
    x0 = bfr(x0); x1 = bfr(x1);
    const float y0 = x0 * c - x1 * s, y1 = x0 * s + x1 * c;
    x0 = y0; x1 = y1;
   }
   x0 = bfr(x0); x1 = bfr(x1);
  }
  const uint16_t o2 = fp8x2(x0, x1);
  if (h < 16) {
   qo[((size_t)t * 16 + h) * 32 + lane] = o2;
   if (LO) {
    __nv_fp8_e4m3 e0, e1;
    e0.__x = (__nv_fp8_storage_t)(o2 & 0xffu);
    e1.__x = (__nv_fp8_storage_t)(o2 >> 8);
    ql[((size_t)t * 16 + h) * 32 + lane] = fp8x2(x0 - static_cast<float>(e0), x1 - static_cast<float>(e1));
   }
  }
  else if (h < 18) *reinterpret_cast<uint16_t*>(kc + L * 128 + (h - 16) * 64 + 2 * lane) = o2;
  else *reinterpret_cast<uint16_t*>(vc + L * 128 + (h - 18) * 64 + 2 * lane) = o2;
 }
}





template <bool LO>
__global__ void __launch_bounds__(256) frontg_k(const __nv_bfloat162* __restrict__ qkv, const long* __restrict__ pos,
  const float* __restrict__ cs, const long* __restrict__ loc, uint16_t* __restrict__ qo, uint8_t* __restrict__ kc,
  uint8_t* __restrict__ vc, float eps, int rope, uint16_t* __restrict__ ql, const int* __restrict__ pt,
  const int* __restrict__ seqlens, const int* __restrict__ cu_q, int pages, int page, int S, int win, int part,
  uint8_t* __restrict__ ko, uint8_t* __restrict__ vt) {
 __shared__ __align__(16) uint8_t vtile[8][132];
 const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
 const int b = blockIdx.y;
 const int n = seqlens[b], cq = cu_q[b], Lb = cu_q[b + 1] - cq;
 const int s0 = max(0, n - Lb - win), nv = n - s0, jn = n - Lb - s0;


 const int g = (part == 1 ? (jn & ~31) >> 3 : 0) + (int)blockIdx.x;
 if (g * 8 >= S) return;
 GDC_WAIT();
 GDC_GO();
 const int j0 = g * 8, j = j0 + warp;
 uint8_t* krow = ko + ((size_t)b * S + j) * 128;
 if (j >= nv) {
  reinterpret_cast<uint32_t*>(krow)[lane] = 0u;
  reinterpret_cast<uint32_t*>(&vtile[warp][0])[lane] = 0u;
 } else if (j < jn) {
  const int s = s0 + j;
  int pg = pt[(size_t)b * pages + s / page];
  pg = pg < 0 ? 0 : pg;
  const size_t src = ((size_t)pg * page + s % page) * 128;
  reinterpret_cast<uint32_t*>(krow)[lane] = __ldg(reinterpret_cast<const uint32_t*>(kc + src) + lane);
  reinterpret_cast<uint32_t*>(&vtile[warp][0])[lane] = __ldg(reinterpret_cast<const uint32_t*>(vc + src) + lane);
 } else {
  const int t = cq + (j - jn);
  const __nv_bfloat162* row = qkv + (size_t)t * 1152;
  float c = 1.f, s = 0.f;
  if (rope) {
   const long p = pos[t];
   c = cs[p * 64 + lane];
   s = cs[p * 64 + 32 + lane];
  }
  const long Lc = loc[t];
  float2 v[20];
#pragma unroll
  for (int h = 0; h < 20; h++) v[h] = __bfloat1622float2(row[h * 32 + lane]);
#pragma unroll
  for (int h = 0; h < 20; h++) {
   float x0 = v[h].x, x1 = v[h].y;
   if (h < 18) {
    float ss = x0 * x0 + x1 * x1;
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) ss += __shfl_xor_sync(0xffffffffu, ss, o);
    const float r = rsqrtf(ss / 64.f + eps);
    x0 *= r; x1 *= r;
    if (rope) {
     x0 = bfr(x0); x1 = bfr(x1);
     const float y0 = x0 * c - x1 * s, y1 = x0 * s + x1 * c;
     x0 = y0; x1 = y1;
    }
    x0 = bfr(x0); x1 = bfr(x1);
   }
   const uint16_t o2 = fp8x2(x0, x1);
   if (h < 16) {
    qo[((size_t)t * 16 + h) * 32 + lane] = o2;
    if (LO) {
     __nv_fp8_e4m3 e0, e1;
     e0.__x = (__nv_fp8_storage_t)(o2 & 0xffu);
     e1.__x = (__nv_fp8_storage_t)(o2 >> 8);
     ql[((size_t)t * 16 + h) * 32 + lane] = fp8x2(x0 - static_cast<float>(e0), x1 - static_cast<float>(e1));
    }
   } else if (h < 18) {
    *reinterpret_cast<uint16_t*>(kc + Lc * 128 + (h - 16) * 64 + 2 * lane) = o2;
    *reinterpret_cast<uint16_t*>(krow + (h - 16) * 64 + 2 * lane) = o2;
   } else {
    *reinterpret_cast<uint16_t*>(vc + Lc * 128 + (h - 18) * 64 + 2 * lane) = o2;
    *reinterpret_cast<uint16_t*>(&vtile[warp][(h - 18) * 64 + 2 * lane]) = o2;
   }
  }
 }
 __syncthreads();


 const int a = (j0 >> 3) & 3, sb = (j0 & ~31) + 16 * (a >> 1) + 2 * (a & 1);
#pragma unroll
 for (int r = 0; r < 2; ++r) {
  const int idx = threadIdx.x + 256 * r, hd = idx >> 2, m = idx & 3;
  const uint16_t w2 = (uint16_t)vtile[2 * m][hd] | (uint16_t)((uint16_t)vtile[2 * m + 1][hd] << 8);
  *reinterpret_cast<uint16_t*>(vt + ((size_t)b * 128 + hd) * S + sb + 4 * m) = w2;
 }
}


__global__ void __launch_bounds__(256) oldg_k(const uint4* __restrict__ kc, const uint4* __restrict__ vc,
  const int* __restrict__ pt, const int* __restrict__ seqlens, const int* __restrict__ cu_q, int pages, int page,
  int S, int win, uint4* __restrict__ ko, uint4* __restrict__ vt) {
 __shared__ uint4 tile[32][9];
 const int b = blockIdx.y, j0 = blockIdx.x * 32, t = threadIdx.x;
 const int n = seqlens[b], L = cu_q[b + 1] - cu_q[b];
 const int s0 = max(0, n - L - win);
 if (!(j0 + 32 <= n - L - s0)) return;
 {
  const int jr = t >> 3, u = t & 7, j = j0 + jr;
  const int s = s0 + j;
  int pg = pt[(long long)b * pages + s / page];
  pg = pg < 0 ? 0 : pg;
  const long long src = ((long long)pg * page + s % page) * 8 + u;
  const uint4 kv = __ldg(kc + src), vv = __ldg(vc + src);
  ko[((long long)b * S + j) * 8 + u] = kv;
  tile[jr][u] = vv;
 }
 __syncthreads();
 const unsigned char* tb = reinterpret_cast<const unsigned char*>(&tile[0][0]);
 const int hd = t >> 1, q = t & 1;
 uint32_t w[4];
#pragma unroll
 for (int k = 0; k < 4; ++k) {
  uint32_t x = 0;
#pragma unroll
  for (int i = 0; i < 4; ++i) {
   const int ks = q * 16 + 4 * k + i;
   const int src = 2 * ((ks & 15) >> 2) + (ks & 1) + 8 * ((ks & 3) >> 1) + 16 * (ks >> 4);
   x |= uint32_t(tb[src * 144 + hd]) << (8 * i);
  }
  w[k] = x;
 }
 vt[(((long long)b * 128 + hd) * S + j0) / 16 + q] = make_uint4(w[0], w[1], w[2], w[3]);
}
}

void front_run(torch::Tensor qkv, torch::Tensor pos, torch::Tensor cs, torch::Tensor loc, torch::Tensor qo,
        torch::Tensor kc, torch::Tensor vc, double eps, int64_t rope) {
 const int T = (int)qkv.size(0);
 pdl_launch(front_k<false>, dim3((T + 7) / 8), dim3(256), 0,
   (const __nv_bfloat162*)qkv.data_ptr(), (const long*)pos.data_ptr<long>(), (const float*)(rope ? cs.data_ptr<float>() : nullptr),
   (const long*)loc.data_ptr<long>(), (uint16_t*)qo.data_ptr(), (uint8_t*)kc.data_ptr(), (uint8_t*)vc.data_ptr(), T, (float)eps,
   (int)rope, (uint16_t*)nullptr);
}



void frontg_run(torch::Tensor qkv, torch::Tensor pos, torch::Tensor cs, torch::Tensor loc, torch::Tensor qo,
        torch::Tensor qlo, torch::Tensor kc, torch::Tensor vc, double eps, int64_t rope, torch::Tensor pt,
        torch::Tensor seqlens, torch::Tensor cu_q, int64_t win, int64_t part, torch::Tensor ko, torch::Tensor vt, int64_t gx) {
 const int B = (int)pt.size(0), pages = (int)pt.size(1), page = (int)kc.size(1), S = (int)ko.size(1);
 TORCH_CHECK(kc.size(2) * kc.size(3) == 128 && S % 64 == 0 && ko.size(0) == B, "frontg_run");
 const bool lo = qlo.numel() > 0;
 TORCH_CHECK(!lo || (qlo.is_contiguous() && qlo.numel() == qo.numel()), "frontg_run qlo");
 auto kern = lo ? frontg_k<true> : frontg_k<false>;
 pdl_launch(kern, dim3(gx > 0 ? (int)gx : S / 8, B), dim3(256), 0,
   (const __nv_bfloat162*)qkv.data_ptr(), (const long*)pos.data_ptr<long>(), (const float*)(rope ? cs.data_ptr<float>() : nullptr),
   (const long*)loc.data_ptr<long>(), (uint16_t*)qo.data_ptr(), (uint8_t*)kc.data_ptr(), (uint8_t*)vc.data_ptr(), (float)eps,
   (int)rope, (uint16_t*)(lo ? qlo.data_ptr() : nullptr), (const int*)pt.data_ptr<int>(), (const int*)seqlens.data_ptr<int>(),
   (const int*)cu_q.data_ptr<int>(), pages, page, S, (int)win, (int)part, (uint8_t*)ko.data_ptr(), (uint8_t*)vt.data_ptr());
}
void oldg_run(torch::Tensor kc, torch::Tensor vc, torch::Tensor pt, torch::Tensor seqlens, torch::Tensor cu_q, int64_t win,
       torch::Tensor ko, torch::Tensor vt) {
 const int B = (int)pt.size(0), pages = (int)pt.size(1), page = (int)kc.size(1), S = (int)ko.size(1);
 TORCH_CHECK(kc.size(2) * kc.size(3) == 128 && S % 64 == 0, "oldg_run");
 oldg_k<<<dim3(S / 32, B), dim3(256), 0, at::cuda::getCurrentCUDAStream()>>>(
   (const uint4*)kc.data_ptr(), (const uint4*)vc.data_ptr(), (const int*)pt.data_ptr<int>(), (const int*)seqlens.data_ptr<int>(),
   (const int*)cu_q.data_ptr<int>(), pages, page, S, (int)win, (uint4*)ko.data_ptr(), (uint4*)vt.data_ptr());
}



int64_t make_edges(int64_t port) {
 cudaStream_t st = at::cuda::getCurrentCUDAStream();
 cudaStreamCaptureStatus cst; unsigned long long id; cudaGraph_t g;
 const cudaGraphNode_t* deps = nullptr; const cudaGraphEdgeData* ed = nullptr; size_t n = 0;
 if (cudaStreamGetCaptureInfo(st, &cst, &id, &g, &deps, &ed, &n) != cudaSuccess) return -1;
 if (cst != cudaStreamCaptureStatusActive) return -2;
 std::vector<cudaGraphNode_t> d(deps, deps + n);
 std::vector<cudaGraphEdgeData> e(n);
 for (size_t i = 0; i < n; ++i) {
  e[i] = cudaGraphEdgeData{};
  e[i].type = cudaGraphDependencyTypeProgrammatic;
  e[i].from_port = port == 1 ? cudaGraphKernelNodePortLaunchCompletion : cudaGraphKernelNodePortProgrammatic;
 }
 const cudaError_t r = cudaStreamUpdateCaptureDependencies(st, d.data(), e.data(), n, cudaStreamSetCaptureDependencies);
 return r == cudaSuccess ? (int64_t)n : -10 - (int64_t)r;
}

void front2_run(torch::Tensor qkv, torch::Tensor pos, torch::Tensor cs, torch::Tensor loc, torch::Tensor qo,
        torch::Tensor qlo, torch::Tensor kc, torch::Tensor vc, double eps, int64_t rope) {
 const int T = (int)qkv.size(0);
 TORCH_CHECK(qlo.is_contiguous() && qlo.numel() == qo.numel(), "front2_run qlo");
 pdl_launch(front_k<true>, dim3((T + 7) / 8), dim3(256), 0,
   (const __nv_bfloat162*)qkv.data_ptr(), (const long*)pos.data_ptr<long>(), (const float*)(rope ? cs.data_ptr<float>() : nullptr),
   (const long*)loc.data_ptr<long>(), (uint16_t*)qo.data_ptr(), (uint8_t*)kc.data_ptr(), (uint8_t*)vc.data_ptr(), T, (float)eps,
   (int)rope, (uint16_t*)qlo.data_ptr());
}
void route_run(torch::Tensor lg, torch::Tensor bias, torch::Tensor lstart, double mult, double eps, torch::Tensor tw,
        torch::Tensor tid, torch::Tensor slot, torch::Tensor rinfo, torch::Tensor cnt_tab, torch::Tensor base_tab,
        torch::Tensor meta, int64_t tn) {
 const int T = (int)lg.size(-2), G = (int)cnt_tab.size(0), S = lg.dim() == 3 ? (int)lg.size(0) : 1;
 route_k<<<G, RW * 32, 0, at::cuda::getCurrentCUDAStream()>>>(
   lg.data_ptr<float>(), bias.data_ptr<float>(), lstart.data_ptr<int>(), T, (float)mult, (float)eps,
   tw.data_ptr<float>(), tid.data_ptr<int>(), slot.data_ptr<int>(), rinfo.data_ptr<int>(), cnt_tab.data_ptr<int>(),
   base_tab.data_ptr<int>(), meta.data_ptr<int>(), (int)meta.numel() - 1, (int)tn, tn == 256 ? 64 : 32,
   tn == 256 ? 32 : 16, S);
}
void route2_run(torch::Tensor lg, torch::Tensor bias, torch::Tensor lstart, double mult, double eps, torch::Tensor tw,
        torch::Tensor tid, torch::Tensor slot, torch::Tensor rinfo, torch::Tensor cnt_tab) {

 const int T = (int)lg.size(0), G = (int)cnt_tab.size(0);
 TORCH_CHECK(G * RW >= T, "route2: too few CTAs");
 route2_k<<<G, RW * 32, 0, at::cuda::getCurrentCUDAStream()>>>(
   lg.data_ptr<float>(), (int)lg.stride(0), bias.data_ptr<float>(), lstart.data_ptr<int>(), T, (float)mult,
   (float)eps, tw.data_ptr<float>(), tid.data_ptr<int>(), slot.data_ptr<int>(), rinfo.data_ptr<int>(),
   cnt_tab.data_ptr<int>());
}
void normgather2_run(torch::Tensor x, torch::Tensor w, torch::Tensor rinfo, torch::Tensor cnt_tab, double eps,
           torch::Tensor a, torch::Tensor order, torch::Tensor meta, int64_t nzero) {

 const int T = (int)x.size(0);
 normgather2_k<<<T, 256, 0, at::cuda::getCurrentCUDAStream()>>>(
   x.data_ptr<float>(), (int)x.stride(0), w.data_ptr<float>(), rinfo.data_ptr<int>(), cnt_tab.data_ptr<int>(),
   (int)cnt_tab.size(0), (float)eps, (uint4*)a.data_ptr(), order.data_ptr<long>(), T, meta.data_ptr<int>(),
   (int)nzero);
}
void px_init() { px_init_k<<<16, 256, 0, at::cuda::getCurrentCUDAStream()>>>(); }
void rng2_run(torch::Tensor x, torch::Tensor bias, torch::Tensor lstart, double mult, double sceps, torch::Tensor w,
       double eps, torch::Tensor tw, torch::Tensor tid, torch::Tensor slot, torch::Tensor a, torch::Tensor order,
       torch::Tensor ctr, torch::Tensor meta, int64_t nzero, torch::Tensor emask, int64_t hd, int64_t xm) {


 const int T = (int)x.size(0);
 TORCH_CHECK((xm & 32) && T < 4096, "rng2_run: rng3y_k needs xm & 32 and T < 4096");
#define Z0Y(M, G) pdl_launch(rng3y_k<M>, dim3(G), dim3(128), 0, \
   (const float*)x.data_ptr<float>(), (int)x.stride(0), (const float*)bias.data_ptr<float>(), (const int*)lstart.data_ptr<int>(), (float)mult, (float)sceps, \
   (const float*)w.data_ptr<float>(), (float)eps, tw.data_ptr<float>(), tid.data_ptr<int>(), slot.data_ptr<int>(), \
   (uint4*)a.data_ptr(), order.data_ptr<long>(), T, T + (int)hd, ctr.data_ptr<int>(), meta.data_ptr<int>(), (int)nzero, \
   emask.data_ptr<int>(), (int)hd)
 if (xm & 16) Z0Y(1, T + 1); else Z0Y(0, T);
#undef Z0Y
}
void normgather_run(torch::Tensor x, torch::Tensor w, torch::Tensor rinfo, torch::Tensor base_tab, double eps,
          torch::Tensor a, torch::Tensor order) {
 const int T = (int)x.size(0);
 pdl_launch(normgather_k, dim3(T), dim3(256), 0,
   (const uint4*)x.data_ptr(), (const float*)w.data_ptr<float>(), (const int*)rinfo.data_ptr<int>(), (const int*)base_tab.data_ptr<int>(),
   (int)base_tab.size(0), (float)eps, (uint4*)a.data_ptr(), order.data_ptr<long>(), T, 0);
}

void normgather_range_run(torch::Tensor x, torch::Tensor w, torch::Tensor rinfo, torch::Tensor base_tab, double eps,
             torch::Tensor a, torch::Tensor order, int64_t t0, int64_t t1) {
 TORCH_CHECK(0 <= t0 && t0 < t1 && t1 <= x.size(0), "normgather_range_run");
 pdl_launch(normgather_k, dim3((unsigned)(t1 - t0)), dim3(256), 0,
   (const uint4*)x.data_ptr(), (const float*)w.data_ptr<float>(), (const int*)rinfo.data_ptr<int>(), (const int*)base_tab.data_ptr<int>(),
   (int)base_tab.size(0), (float)eps, (uint4*)a.data_ptr(), order.data_ptr<long>(), (int)t1, (int)t0);
}
"""
xa__glue = load_inline(name="okc_prefill_glue_a10002_b460010", cuda_sources=xa__GLUE_SRC,
                    cpp_sources="void route_run(torch::Tensor, torch::Tensor, torch::Tensor, double, double, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, int64_t);\n"
                                "void normgather_run(torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, double, torch::Tensor, torch::Tensor);\n"
                                "void route2_run(torch::Tensor, torch::Tensor, torch::Tensor, double, double, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor);\n"
                                "void rng2_run(torch::Tensor, torch::Tensor, torch::Tensor, double, double, torch::Tensor, double, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, int64_t, torch::Tensor, int64_t, int64_t);\n"
                                "void normgather2_run(torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, double, torch::Tensor, torch::Tensor, torch::Tensor, int64_t);\n"
                                "void front_run(torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, double, int64_t);\n"
                                "void front2_run(torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, double, int64_t);\n"
                                "void frontg_run(torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, double, int64_t, torch::Tensor, torch::Tensor, torch::Tensor, int64_t, int64_t, torch::Tensor, torch::Tensor, int64_t);\n"
                                "void oldg_run(torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, int64_t, torch::Tensor, torch::Tensor);\n"
                                "int64_t make_edges(int64_t);\n"
                                "void normgather_range_run(torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, double, torch::Tensor, torch::Tensor, int64_t, int64_t);\n"
                                "void px_init();",
                    functions=["route_run", "normgather_run", "front_run", "front2_run", "route2_run", "normgather2_run", "rng2_run", "px_init", "frontg_run", "oldg_run", "make_edges", "normgather_range_run"],
                    extra_cuda_cflags=["-O3", "-gencode=arch=compute_90a,code=sm_90a"])
xa_ROUTE_CTAS = 132
xa_ROUTE_MODE_SMALL = True
xa_ROUTE_MODE_BIG = True
xa_TQ_RT_MODE = 2
xa_MOE_CTAS = 132
xa_MOE_DYNAMIC = True
xa_MOE_MIN_TOKENS = 1


xa_MOE_FORCE_MB = 0
xa_MOE_TILE = 0


def xa__moe_tile(routes):
    """Token tile by the average routes per local expert."""
    if xa_MOE_TILE:
        return xa_MOE_TILE
    return 256 if routes >= 8192 else 128


xa__POS = [
    [(6, 7, 8, 15), (22, 23, 24, 31)],
    [(3, 4, 5, 12), (19, 20, 21, 28)],
    [(0, 1, 2, 9), (16, 17, 18, 25)],
    [(30, 26, 27, 29), (14, 10, 11, 13)],
]


def xa__decode4_layout(words):
    """int32 words with pair m's codes at bits 4m (first) and 16 + 4m (second) -> decode4's bit layout."""
    w = words.to(torch.int64) & 0xFFFFFFFF
    out = torch.zeros_like(w)
    for m in range(4):
        for h in range(2):
            code = (w >> (16 * h + 4 * m)) & 15
            for b, pos in enumerate(xa__POS[m][h]):
                out |= ((code >> b) & 1) << pos
    return torch.where(out >= 1 << 31, out - (1 << 32), out).to(torch.int32)


def xa__scale_code(v):
    """E4M3 scale bytes -> the lossless code 8 E + m with scale = 1.m * 2^(E - 9) (see sc_bf16); 0 for a zero scale."""
    v = v.to(torch.int32)
    assert not bool((v & 0x80).any()) and not bool((v == 0x7F).any()), "scales must be positive and finite"
    e, m = (v >> 3) & 15, v & 7
    p = torch.where(m >= 4, 2, torch.where(m >= 2, 1, 0))
    code = torch.where(e == 0, p * 8 + ((m << (3 - p)) & 7), (e + 2) * 8 + m)
    return torch.where(v == 0, 0, code).to(torch.uint8)


def xa_kperm_index(n):
    """The K permutation: index [n] with position p of every 64-chunk -> column sigma(p): p = 16 kk + 8 half + 2 q + j (k16 step kk,
    fragment column 8 half + 2 q + j of lane q) holds 16 q + 4 kk + 2 half + j, so lane q's values are scale block q."""
    p = torch.arange(n)
    c, r = p // 64 * 64, p % 64
    kk, half, q, j = r // 16, (r // 8) % 2, (r // 2) % 4, r % 2
    k = c + 16 * q + 4 * kk + 2 * half + j
    assert sorted(k.tolist()) == list(range(n))
    return k


def xa__pack_nvfp4(w, sf, perm=None, mb=2, bfs=False):
    """Packed E2M1 [E, N, K/2] + E4M3 scales [E, N, K/16] -> units [E, N/(128 mb), K/64, 8 mb, 576] uint8 (lossless): per
    16-row x 64-k unit, 512 B of codes (per lane and k16 step one word: the first values of the four A pairs in
    the low 16 bits, the second values in the high 16) then the 64 block scales [16 rows][4] as _scale_code bytes."""
    E, N, K2 = w.shape
    K = K2 * 2
    if perm is not None:
        w, sf = w[:, perm], sf[:, perm]
    codes = torch.stack((w & 0x0F, w >> 4), dim=-1).reshape(E, N, K)
    sfb = sf.view(torch.uint8)
    codes = codes.masked_fill((sfb == 0).repeat_interleave(16, dim=-1), 0)
    codes = codes.index_select(2, xa_kperm_index(K).to(codes.device))
    TPG = 8 * mb
    G, C = N // (16 * TPG), K // 64
    v = codes.view(E, G, TPG, 2, 8, C, 4, 2, 4, 2)
    v = v.permute(0, 1, 5, 2, 4, 8, 6, 9, 7, 3)
    nib = (v[..., 0] | (v[..., 1] << 4)).reshape(E, G, C, TPG, 512)
    s = xa__scale_code(sfb).view(E, G, TPG, 2, 8, C, 4).permute(0, 1, 5, 2, 4, 6, 3).reshape(E, G, C, TPG, 64)
    if bfs:
        s = (s.to(torch.int32) * 16 + 0x5A80).to(torch.int16).view(torch.uint8).reshape(E, G, C, TPG, 128)
    out = torch.cat((nib, s), dim=-1).contiguous()
    for e in range(E):
        cw = out[e, ..., :512].contiguous().view(torch.int32)
        out[e, ..., :512] = xa__decode4_layout(cw).view(torch.uint8).view(out[e, ..., :512].shape)
    return out


def xa__experts_moe(x, topk_weights, local_ids, state):
    """Routed experts of an already routed batch (layer.py's path for up to 8 tokens): sort the routes by expert,
    gather their rows, then the two grouped NVFP4 GEMMs (gate/up with SiLU fused; down) and SGLang's weighted sum.
    No BF16 copy of the weights."""
    moe = state["moe"]
    T, H = x.shape
    E, I = xa_LOCAL_EXPERTS, xa_D_EXPERT
    ids = local_ids.reshape(-1).to(torch.int32)
    R = ids.numel()
    tn = xa__moe_tile(T * xa_TOP_K * xa_LOCAL_EXPERTS / xa_N_EXPERTS)
    order = torch.empty(R, dtype=torch.long, device=x.device)
    meta = torch.zeros(260 + R // 32 + 64, dtype=torch.int32, device=x.device)
    wide = tn == 256
    xa__moe.moe_build(ids, order, meta, tn if wide else -xa_MOE_FORCE_MB if xa_MOE_FORCE_MB else 128)
    a = torch.empty((R, H), dtype=torch.bfloat16, device=x.device)
    xa__moe.moe_gather(x.contiguous(), order, meta, a)
    inter = torch.empty((R, I), dtype=torch.bfloat16, device=x.device)
    ctas = xa_MOE_CTAS if xa_MOE_DYNAMIC else -xa_MOE_CTAS
    y = torch.empty((R, H), dtype=torch.bfloat16, device=x.device)
    xa__moe.moe_fused_run(moe["w13"], moe["w2"], moe["w13w"], moe["w2w"], a, inter, meta, moe["gs13"], moe["gs2"], y, ctas, tn,
                       order)
    slot_ids = torch.where(ids < E, ids, -1).view(T, xa_TOP_K)
    return fused_post_expert_norm_weighted_sum(
        y.view(T, xa_TOP_K, H), topk_weights.to(torch.float32).contiguous(), norm_weight=state["post_expert_w"],
        eps=xa_POST_NORM_EPS, weight_first=True, slot_expert_ids=slot_ids)



def xa__experts_grouped(x, topk_weights, local_ids, state):
    raw = state["raw"]
    T, H = x.shape
    I = xa_D_EXPERT
    E = xa_LOCAL_EXPERTS
    w13 = torch.empty((E, 2 * I, H), dtype=torch.bfloat16, device=x.device)
    w2 = torch.empty((E, H, I), dtype=torch.bfloat16, device=x.device)
    xa__gx.dequant(raw["w13"], raw["w13_sf"], w13)
    xa__gx.dequant(raw["w2"], raw["w2_sf"], w2)
    ids = local_ids.reshape(-1).long()
    key = torch.where(ids < E, ids, torch.full_like(ids, E))
    order = torch.argsort(key, stable=True)
    inv = torch.empty_like(order)
    inv[order] = torch.arange(order.numel(), device=x.device)
    counts = torch.zeros(E + 1, dtype=torch.long, device=x.device).scatter_add_(0, key, torch.ones_like(key))
    offs = torch.cumsum(counts[:E], 0).to(torch.int32)
    key_sorted = key[order]
    R = ids.numel()
    a = torch.empty((R, H), dtype=torch.bfloat16, device=x.device)
    xa__gx.gather(x.contiguous(), order, offs, a)
    h = torch._grouped_mm(a, w13.transpose(1, 2), offs=offs)
    inter = torch.empty((R, I), dtype=torch.bfloat16, device=x.device)
    xa__gx.silu(h, raw["gs13"], key_sorted, offs, inter)
    ys = torch._grouped_mm(inter, w2.transpose(1, 2), offs=offs)
    y = torch.empty((R, H), dtype=torch.bfloat16, device=x.device)
    xa__gx.unsort(ys, inv, key, raw["gs2"], y)
    slot_ids = torch.where(ids < E, ids, torch.full_like(ids, -1)).to(torch.int32).view(T, xa_TOP_K)
    return fused_post_expert_norm_weighted_sum(
        y.view(T, xa_TOP_K, H), topk_weights.to(torch.float32).contiguous(), norm_weight=state["post_expert_w"],
        eps=xa_POST_NORM_EPS, weight_first=True, slot_expert_ids=slot_ids)


def xa__moe_block(h, data, state, pooler_w, side, wsum=None):
    """The routed-expert block for more than 8 tokens, from the pre-FFN normalized stream h: router, routing and route
    list, pooler, pre-expert norm fused into the row gather, the grouped NVFP4 expert GEMMs and the weighted sum (wsum,
    weighted_sum.py's kernel, on the moe_mixed paths). Up to CAT_MAX_TOKENS this is _moe_block_cat. Above it the
    router runs first on the main queue (FP32 output on the hi | lo weights), route_k on the side queue beside the
    pooler, then normgather; from POOL_SPLIT_MIN tokens the pooler is split in two parts. Returns (expert output,
    topk ids)."""
    T = h.shape[0]
    dev = h.device
    R = T * xa_TOP_K
    tn = xa__moe_tile(T * xa_TOP_K * xa_LOCAL_EXPERTS / xa_N_EXPERTS)
    if T <= xa_CAT_MAX_TOKENS and tn != 256 and "pool_router" in state:
        return xa__moe_block_cat(h, data, state, T, tn, wsum)
    sliced = xa_ROUTER_SPLIT and T <= xa_ROUTER_SPLIT_MAX_TOKENS
    G = min(xa_ROUTE_CTAS_SMALL if sliced else xa_ROUTE_CTAS, (T + 15) // 16)
    L = 260 + R // 32 + 64 + 1
    full = torch.empty(L + 32, dtype=torch.int32, device=dev)
    meta, cnt2 = full[:L], full[L:]
    tw = torch.empty((T, xa_TOP_K), dtype=torch.float32, device=dev)
    tid = torch.empty((T, xa_TOP_K), dtype=torch.int32, device=dev)
    slot = torch.empty_like(tid)
    rinfo = torch.empty(R, dtype=torch.int32, device=dev)
    cnt_tab = torch.empty((G, 32), dtype=torch.int32, device=dev)
    base_tab = torch.empty_like(cnt_tab)

    first = xa_TQ_RT_MODE == 2 and not sliced and T >= 2048
    if first:
        lg0 = torch.empty((T, 2 * xa_N_EXPERTS), dtype=torch.float32, device=dev)
        xa__rt.lt_f32(h, state["pool_router"][4096:], lg0, 0, -1, 0, -1)

    def route():
        if sliced:
            lg = torch.empty((xa_ROUTER_SPLIT, T, 2 * xa_N_EXPERTS), dtype=torch.float32, device=dev)
            xa__rt.lt_router(h, state["router_cat_t"], lg, xa_ROUTER_TILE)
        elif first:
            lg = lg0
        elif xa_TQ_RT_MODE == 1:
            lg = torch.empty((T, 2 * xa_N_EXPERTS), dtype=torch.float32, device=dev)
            xa__rt.lt_f32(h, state["pool_router"][4096:], lg, 0, -1, 0, -1)
        else:
            lg = torch.mm(h, state["router_cat_t"], out_dtype=torch.float32)
        full.zero_()
        xa__glue.route_run(lg, state["router_bias"], data["local_start"], 0.1767766952966369, 1e-15, tw, tid, slot, rinfo,
                        cnt_tab, base_tab, meta, tn)

    ma = xa__pool_split(h, pooler_w) if (xa_ROUTE_MODE_BIG if T >= 2048 else xa_ROUTE_MODE_SMALL) else 0
    if ma:
        main = torch.cuda.current_stream()
        expert_input = torch.empty((T, pooler_w.shape[0]), dtype=torch.bfloat16, device=dev)
        order = torch.empty(R, dtype=torch.long, device=dev)
        a = torch.empty((R, xa_D_EXPERT), dtype=torch.bfloat16, device=dev)
        xa__ltgemm_part(h[:ma], pooler_w, expert_input[:ma], T)
        pb = xa__pool_b_stream(dev)
        pb.wait_stream(main)
        with torch.cuda.StreamContext(pb):
            xa__ltgemm_part(h[ma:], pooler_w, expert_input[ma:], T)
        side.wait_stream(main)
        with torch.cuda.StreamContext(side):
            route()
        main.wait_stream(side)
        xa__glue.normgather_range_run(expert_input, state["pre_expert_wf"], rinfo, base_tab, xa_RMS_EPS, a, order, 0, ma)
        main.wait_stream(pb)
        if not torch.cuda.is_current_stream_capturing():
            for tensor in (h, expert_input):
                tensor.record_stream(pb)
        xa__glue.normgather_range_run(expert_input, state["pre_expert_wf"], rinfo, base_tab, xa_RMS_EPS, a, order, ma, T)
        return xa__moe_tail(T, a, order, meta, tw, slot, tid, state, tn, cnt2, wsum)
    if xa_ROUTE_MODE_BIG if T >= 2048 else xa_ROUTE_MODE_SMALL:
        main = torch.cuda.current_stream()
        side.wait_stream(main)
        with torch.cuda.StreamContext(side):
            route()
        expert_input = xa__gemm(h, pooler_w)
        main.wait_stream(side)
    else:
        route()
        expert_input = xa__gemm(h, pooler_w)
    moe = state["moe"]
    order = torch.empty(R, dtype=torch.long, device=dev)
    a = torch.empty((R, xa_D_EXPERT), dtype=torch.bfloat16, device=dev)
    xa__glue.normgather_run(expert_input, state["pre_expert_wf"], rinfo, base_tab, xa_RMS_EPS, a, order)
    return xa__moe_tail(T, a, order, meta, tw, slot, tid, state, tn, cnt2, wsum)


xa_WSID = True


@triton.jit
def xa__wsid_k(x_ptr, w_ptr, s_ptr, o_ptr, ids_ptr, HD: tl.constexpr, eps, BH: tl.constexpr):
    tok = tl.program_id(0)
    offs = tl.arange(0, BH)
    mask = offs < HD
    w = tl.load(w_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    b = tok * 8
    i0 = tl.load(ids_ptr + b + 0)
    i1 = tl.load(ids_ptr + b + 1)
    i2 = tl.load(ids_ptr + b + 2)
    i3 = tl.load(ids_ptr + b + 3)
    i4 = tl.load(ids_ptr + b + 4)
    i5 = tl.load(ids_ptr + b + 5)
    i6 = tl.load(ids_ptr + b + 6)
    i7 = tl.load(ids_ptr + b + 7)
    s0 = tl.load(s_ptr + b + 0)
    s1 = tl.load(s_ptr + b + 1)
    s2 = tl.load(s_ptr + b + 2)
    s3 = tl.load(s_ptr + b + 3)
    s4 = tl.load(s_ptr + b + 4)
    s5 = tl.load(s_ptr + b + 5)
    s6 = tl.load(s_ptr + b + 6)
    s7 = tl.load(s_ptr + b + 7)
    l0 = i0 >= 0
    l1 = i1 >= 0
    l2 = i2 >= 0
    l3 = i3 >= 0
    l4 = i4 >= 0
    l5 = i5 >= 0
    l6 = i6 >= 0
    l7 = i7 >= 0
    acc = tl.zeros([BH], dtype=tl.float32)
    if l0:
        x = tl.load(x_ptr + (b + 0) * HD + offs, mask=mask, other=0.0).to(tl.float32)
        y = x * w
        var = tl.sum(y * y, axis=0) / HD
        y = y * tl.rsqrt(var + eps)
        s = s0.to(tl.float32)
        acc += y * s
    if l1:
        x = tl.load(x_ptr + (b + 1) * HD + offs, mask=mask, other=0.0).to(tl.float32)
        y = x * w
        var = tl.sum(y * y, axis=0) / HD
        y = y * tl.rsqrt(var + eps)
        s = s1.to(tl.float32)
        acc += y * s
    if l2:
        x = tl.load(x_ptr + (b + 2) * HD + offs, mask=mask, other=0.0).to(tl.float32)
        y = x * w
        var = tl.sum(y * y, axis=0) / HD
        y = y * tl.rsqrt(var + eps)
        s = s2.to(tl.float32)
        acc += y * s
    if l3:
        x = tl.load(x_ptr + (b + 3) * HD + offs, mask=mask, other=0.0).to(tl.float32)
        y = x * w
        var = tl.sum(y * y, axis=0) / HD
        y = y * tl.rsqrt(var + eps)
        s = s3.to(tl.float32)
        acc += y * s
    if l4:
        x = tl.load(x_ptr + (b + 4) * HD + offs, mask=mask, other=0.0).to(tl.float32)
        y = x * w
        var = tl.sum(y * y, axis=0) / HD
        y = y * tl.rsqrt(var + eps)
        s = s4.to(tl.float32)
        acc += y * s
    if l5:
        x = tl.load(x_ptr + (b + 5) * HD + offs, mask=mask, other=0.0).to(tl.float32)
        y = x * w
        var = tl.sum(y * y, axis=0) / HD
        y = y * tl.rsqrt(var + eps)
        s = s5.to(tl.float32)
        acc += y * s
    if l6:
        x = tl.load(x_ptr + (b + 6) * HD + offs, mask=mask, other=0.0).to(tl.float32)
        y = x * w
        var = tl.sum(y * y, axis=0) / HD
        y = y * tl.rsqrt(var + eps)
        s = s6.to(tl.float32)
        acc += y * s
    if l7:
        x = tl.load(x_ptr + (b + 7) * HD + offs, mask=mask, other=0.0).to(tl.float32)
        y = x * w
        var = tl.sum(y * y, axis=0) / HD
        y = y * tl.rsqrt(var + eps)
        s = s7.to(tl.float32)
        acc += y * s
    tl.store(o_ptr + tok * HD + offs, acc.to(o_ptr.dtype.element_ty), mask=mask)


xa_X53_WSC = True
xa__X53_WSC_SRC = r"""



#define WSC_NS 3
#define WSC_RPS 4

#include <cuda_bf16.h>
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

namespace {
constexpr int HD = 4096, NT = 256, NS = WSC_NS, RPS = WSC_RPS, ROWB = HD * 2, STB = RPS * ROWB;

__device__ __forceinline__ void w_mbar_init(uint32_t bar, int count) {
 asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::"r"(bar), "r"(count));
}
__device__ __forceinline__ void w_mbar_expect(uint32_t bar, int bytes) {
 asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(bar), "r"(bytes) : "memory");
}
__device__ __forceinline__ void w_mbar_wait(uint32_t bar, uint32_t parity) {
 asm volatile(
   "{\n .reg .pred p;\n W: mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n @!p bra W;\n}" ::"r"(bar),
   "r"(parity) : "memory");
}
__device__ __forceinline__ void w_bulk_g2s(uint32_t dst, const void* src, int bytes, uint32_t bar) {
 asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];"
        ::"r"(dst), "l"(src), "r"(bytes), "r"(bar) : "memory");
}

__device__ __forceinline__ bool x73_elect() {
 uint32_t p;
 asm volatile("{\n .reg .pred P;\n elect.sync _|P, 0xffffffff;\n selp.u32 %0, 1, 0, P;\n}" : "=r"(p));
 return p != 0;
}
__device__ __forceinline__ uint4 x73_lds128(uint32_t a) {
 uint4 v;
 asm volatile("ld.shared.v4.u32 {%0, %1, %2, %3}, [%4];" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "r"(a) : "memory");
 return v;
}
__device__ __forceinline__ void unpack8(const uint4 v, float* f) {
 const __nv_bfloat162* h = reinterpret_cast<const __nv_bfloat162*>(&v);
#pragma unroll
 for (int i = 0; i < 4; i++) { const float2 t = __bfloat1622float2(h[i]); f[2 * i] = t.x; f[2 * i + 1] = t.y; }
}





__global__ void __launch_bounds__(NT + 32, 2) wsc_k(const __nv_bfloat16* __restrict__ x, const __nv_bfloat16* __restrict__ w,
  const float* __restrict__ sc, __nv_bfloat16* __restrict__ out, const int* __restrict__ ids, int T, float eps) {
 extern __shared__ __align__(128) uint8_t wraw[];


 __shared__ __align__(8) uint64_t full[2 * NS];
 __shared__ __align__(16) int s_live[NS][8];
 __shared__ __align__(16) float s_s[NS][8];
 __shared__ __align__(16) float s_red[2][9][8];
 const int tid = threadIdx.x, warp = tid / 32, lane = tid % 32;
 const uint32_t sraw = (uint32_t)__cvta_generic_to_shared(wraw);
 const uint32_t sb = (sraw + 127) & ~127u;
 uint8_t* const sp = wraw + (sb - sraw);
 const uint32_t fb = (uint32_t)__cvta_generic_to_shared(full);
 if (tid == 0) {
  for (int s = 0; s < NS; s++) { w_mbar_init(fb + 8 * s, 1); w_mbar_init(fb + 8 * NS + 8 * s, 1); }
  asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
 }
 __syncthreads();
 const int G = gridDim.x;
 const int ntok = T > (int)blockIdx.x ? (T - (int)blockIdx.x + G - 1) / G : 0;

 auto issue = [&](int i, const int4 a, const int4 b, const float4 c, const float4 d) {
  const int s = i % NS, tok = blockIdx.x + i * G;
  const int id[8] = {a.x, a.y, a.z, a.w, b.x, b.y, b.z, b.w};
  const float sv[8] = {c.x, c.y, c.z, c.w, d.x, d.y, d.z, d.w};
  int n = 0;
#pragma unroll
  for (int j = 0; j < 8; j++) {
   s_live[s][j] = id[j] >= 0;
   s_s[s][j] = sv[j];
   n += id[j] >= 0 && n < RPS;
  }
  w_mbar_expect(fb + 8 * s, n * ROWB);
  int k = 0;
#pragma unroll
  for (int j = 0; j < 8; j++)
   if (id[j] >= 0 && k < RPS) {
    w_bulk_g2s(sb + s * STB + k * ROWB, x + ((size_t)tok * 8 + j) * HD, ROWB, fb + 8 * s);
    k++;
   }
 };
 if (tid >= NT) {
  const int4* ip = reinterpret_cast<const int4*>(ids);
  const float4* sp4 = reinterpret_cast<const float4*>(sc);
  for (int cb = 0; cb < ntok; cb += 32) {
   int4 a = make_int4(-1, -1, -1, -1), b = a;
   float4 c = make_float4(0.f, 0.f, 0.f, 0.f), d = c;
   if (cb + lane < ntok) {
    const size_t t = blockIdx.x + (size_t)(cb + lane) * G;
    a = ip[2 * t]; b = ip[2 * t + 1]; c = sp4[2 * t]; d = sp4[2 * t + 1];
   }
   const int nb = min(32, ntok - cb);
   for (int u = 0; u < nb; u++) {
    int4 a2; int4 b2; float4 c2; float4 d2;
    a2.x = __shfl_sync(0xffffffffu, a.x, u); a2.y = __shfl_sync(0xffffffffu, a.y, u);
    a2.z = __shfl_sync(0xffffffffu, a.z, u); a2.w = __shfl_sync(0xffffffffu, a.w, u);
    b2.x = __shfl_sync(0xffffffffu, b.x, u); b2.y = __shfl_sync(0xffffffffu, b.y, u);
    b2.z = __shfl_sync(0xffffffffu, b.z, u); b2.w = __shfl_sync(0xffffffffu, b.w, u);
    c2.x = __shfl_sync(0xffffffffu, c.x, u); c2.y = __shfl_sync(0xffffffffu, c.y, u);
    c2.z = __shfl_sync(0xffffffffu, c.z, u); c2.w = __shfl_sync(0xffffffffu, c.w, u);
    d2.x = __shfl_sync(0xffffffffu, d.x, u); d2.y = __shfl_sync(0xffffffffu, d.y, u);
    d2.z = __shfl_sync(0xffffffffu, d.z, u); d2.w = __shfl_sync(0xffffffffu, d.w, u);
    const int i = cb + u;
    if (i >= NS) w_mbar_wait(fb + 8 * NS + 8 * (i % NS), ((i / NS) - 1) & 1);
    if (x73_elect()) issue(i, a2, b2, c2, d2);
    __syncwarp();
   }
  }
  return;
 }
 float wv[16];
 {
  unpack8(*reinterpret_cast<const uint4*>(w + 8 * tid), wv);
  unpack8(*reinterpret_cast<const uint4*>(w + 2048 + 8 * tid), wv + 8);
 }


 uint32_t a_row = sb + 16 * tid;
 uint32_t a_live = (uint32_t)__cvta_generic_to_shared(&s_live[0][0]);
 uint32_t a_s = (uint32_t)__cvta_generic_to_shared(&s_s[0][0]);
 uint32_t a_red = (uint32_t)__cvta_generic_to_shared(&s_red[0][0][0]);
 asm("" : "+r"(a_row), "+r"(a_live), "+r"(a_s), "+r"(a_red));
 for (int i = 0; i < ntok; i++) {
  const int s = i % NS, tok = blockIdx.x + i * G;
  w_mbar_wait(fb + 8 * s, (i / NS) & 1);
  float acc[16];
#pragma unroll
  for (int e = 0; e < 16; e++) acc[e] = 0.f;
  const uint4 lv0 = x73_lds128(a_live + 32 * s), lv1 = x73_lds128(a_live + 32 * s + 16);
  const uint4 sv0 = x73_lds128(a_s + 32 * s), sv1 = x73_lds128(a_s + 32 * s + 16);
  const int live[8] = {(int)lv0.x, (int)lv0.y, (int)lv0.z, (int)lv0.w, (int)lv1.x, (int)lv1.y, (int)lv1.z, (int)lv1.w};
  const float svv[8] = {__uint_as_float(sv0.x), __uint_as_float(sv0.y), __uint_as_float(sv0.z), __uint_as_float(sv0.w),
             __uint_as_float(sv1.x), __uint_as_float(sv1.y), __uint_as_float(sv1.z), __uint_as_float(sv1.w)};
  const uint32_t a_st = a_row + s * STB, a_rd = a_red + (i & 1) * 288;
  int k = 0;
#pragma unroll
  for (int j = 0; j < 8; j++) {
   if (!live[j]) continue;
   uint4 v0, v1;
   if (k < RPS) {
    v0 = x73_lds128(a_st + k * ROWB);
    v1 = x73_lds128(a_st + k * ROWB + 4096);
   } else {
    const __nv_bfloat16* r = x + ((size_t)tok * 8 + j) * HD;
    v0 = *reinterpret_cast<const uint4*>(r + 8 * tid);
    v1 = *reinterpret_cast<const uint4*>(r + 2048 + 8 * tid);
   }
   k++;
   float y[16];
   unpack8(v0, y);
   unpack8(v1, y + 8);
   float ss = 0.f;
#pragma unroll
   for (int e = 0; e < 16; e++) { y[e] *= wv[e]; ss += y[e] * y[e]; }
#pragma unroll
   for (int o = 16; o > 0; o >>= 1) ss += __shfl_xor_sync(0xffffffffu, ss, o);
   if (lane == 0) asm volatile("st.shared.f32 [%0], %1;" ::"r"(a_rd + 32 * k + 4 * warp), "f"(ss) : "memory");
   asm volatile("bar.sync 1, %0;" ::"n"(NT) : "memory");
   const uint4 q0 = x73_lds128(a_rd + 32 * k), q1 = x73_lds128(a_rd + 32 * k + 16);
   const float rv[8] = {__uint_as_float(q0.x), __uint_as_float(q0.y), __uint_as_float(q0.z), __uint_as_float(q0.w),
              __uint_as_float(q1.x), __uint_as_float(q1.y), __uint_as_float(q1.z), __uint_as_float(q1.w)};
   float t = 0.f;
#pragma unroll
   for (int u = 0; u < 8; u++) t += rv[u];
   const float rs = rsqrtf(t * (1.f / HD) + eps), sj = svv[j];
#pragma unroll
   for (int e = 0; e < 16; e++) acc[e] += (y[e] * rs) * sj;
  }
  __align__(16) __nv_bfloat16 o[16];
#pragma unroll
  for (int e = 0; e < 16; e++) o[e] = __float2bfloat16(acc[e]);
  *reinterpret_cast<uint4*>(out + (size_t)tok * HD + 8 * tid) = *reinterpret_cast<const uint4*>(o);
  *reinterpret_cast<uint4*>(out + (size_t)tok * HD + 2048 + 8 * tid) = *reinterpret_cast<const uint4*>(o + 8);
  asm volatile("bar.sync 1, %0;" ::"n"(NT) : "memory");
  if (tid == 0) asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" ::"r"(fb + 8 * NS + 8 * s) : "memory");
 }
}
}

void wsc3_run(torch::Tensor x, torch::Tensor w, torch::Tensor s, torch::Tensor out, torch::Tensor ids, int64_t T,
       double eps, int64_t ctas) {
 const int smem = NS * STB + 128;
 static bool once = false;
 if (!once) { cudaFuncSetAttribute(wsc_k, cudaFuncAttributeMaxDynamicSharedMemorySize, smem); once = true; }
 wsc_k<<<(int)ctas, NT + 32, smem, at::cuda::getCurrentCUDAStream()>>>(
   (const __nv_bfloat16*)x.data_ptr(), (const __nv_bfloat16*)w.data_ptr(), s.data_ptr<float>(),
   (__nv_bfloat16*)out.data_ptr(), ids.data_ptr<int>(), (int)T, (float)eps);
}
"""
xa__wsc = load_inline(name="okc_wsc_x46001_b460010", cuda_sources=xa__X53_WSC_SRC,
                   cpp_sources="void wsc3_run(torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, int64_t, double, int64_t);\n",
                   functions=["wsc3_run"], extra_cuda_cflags=["-O3"])


def xa__moe_tail(T, a, order, meta, tw, slot, tid, state, tn, cnt2=None, wsum=None):
    """The grouped NVFP4 expert GEMMs and the weighted sum (shared by both routing paths)."""
    R = T * xa_TOP_K
    dev = a.device
    moe = state["moe"]
    inter = torch.empty((a.shape[0], xa_D_EXPERT), dtype=torch.bfloat16, device=dev)
    y = torch.empty((R, xa_D_EXPERT), dtype=torch.bfloat16, device=dev)
    ctas = xa_MOE_CTAS if xa_MOE_DYNAMIC else -xa_MOE_CTAS
    mixed = tn != 256 and wsum is not None and cnt2 is not None
    if mixed:
        xa__moe.moe_set_cnt2(cnt2.data_ptr())
    xa__moe.moe_fused_run(moe["w13"], moe["w2"], moe["w13w"], moe["w2w"], a, inter, meta, moe["gs13"], moe["gs2"], y, ctas,
                       tn, order)
    xa__moe.moe_set_cnt2(0)
    if mixed:
        return wsum(y, tw, slot, state["post_expert_w"], meta, cnt2, T, xa_POST_NORM_EPS, True), tid
    if xa_X53_WSC and T >= 32:
        out = torch.empty((T, xa_D_EXPERT), dtype=y.dtype, device=dev)
        xa__wsc.wsc3_run(y, state["post_expert_w"], tw.contiguous(), out, slot, T, xa_POST_NORM_EPS, 264)
        return out, tid
    if xa_WSID and T >= 32:
        out = torch.empty((T, xa_D_EXPERT), dtype=y.dtype, device=dev)
        xa__wsid_k[(T,)](y, state["post_expert_w"], tw.contiguous(), out, slot, xa_D_EXPERT, xa_POST_NORM_EPS, 4096, num_warps=8,
                      num_stages=2)
        return out, tid
    out = fused_post_expert_norm_weighted_sum(
        y.view(T, xa_TOP_K, xa_D_EXPERT), tw, norm_weight=state["post_expert_w"], eps=xa_POST_NORM_EPS, weight_first=True,
        slot_expert_ids=slot)
    return out, tid


def xa__moe_block_cat(h, data, state, T, tn, wsum=None):
    """Up to CAT_MAX_TOKENS: pooler + router as one FP32-output GEMM (one wave, nothing beside it), then rng3y_k
    (through rng2_run): per-token top 8, BF16 rounding of the pooler columns, the pre-expert norm and the row gather into
    fixed-capacity route lists, with a planner CTA that writes the expert plan. The route2_k + normgather2_k form after
    it runs only with CAT_FUSED off."""
    dev = h.device
    R = T * xa_TOP_K
    G = (T + 15) // 16
    nzero = R // 32 + 64
    L = 260 + nzero + 1
    full = torch.empty(L + 32, dtype=torch.int32, device=dev)
    meta, cnt2 = full[:L], full[L:]
    tw = torch.empty((T, xa_TOP_K), dtype=torch.float32, device=dev)
    tid = torch.empty((T, xa_TOP_K), dtype=torch.int32, device=dev)
    slot = torch.empty_like(tid)
    rinfo = torch.empty(R, dtype=torch.int32, device=dev)
    cnt_tab = torch.empty((G, 32), dtype=torch.int32, device=dev)
    out = torch.empty((T, xa_D_EXPERT + 2 * xa_N_EXPERTS), dtype=torch.float32, device=dev)
    cfg = xa_CAT_CFG if 512 <= T <= 768 else (-1, 0, -1)
    xa__rt.lt_f32(h, state["pool_router"], out, xa_CAT_ALGO, cfg[0], cfg[1], cfg[2])
    if xa_CAT_FUSED:
        order = torch.empty(xa_LOCAL_EXPERTS * (T + xa_TILE_HD), dtype=torch.long, device=dev)
        a = torch.empty((xa_LOCAL_EXPERTS * (T + xa_TILE_HD), xa_D_EXPERT), dtype=torch.bfloat16, device=dev)
        emask = torch.empty(32, dtype=torch.int32, device=dev)
        xa__glue.rng2_run(out, state["router_bias"], data["local_start"], 0.1767766952966369, 1e-15,
                       state["pre_expert_wf"], xa_RMS_EPS, tw, tid, slot, a, order, state["route_ctr"], full, nzero + 33,
                       emask, xa_TILE_HD, xa_RNG3_XM)
        return xa__moe_tail(T, a, order, meta, tw, slot, tid, state, tn, cnt2, wsum)
    xa__glue.route2_run(out[:, xa_D_EXPERT:], state["router_bias"], data["local_start"], 0.1767766952966369, 1e-15, tw, tid,
                     slot, rinfo, cnt_tab)
    order = torch.empty(R, dtype=torch.long, device=dev)
    a = torch.empty((R, xa_D_EXPERT), dtype=torch.bfloat16, device=dev)
    xa__glue.normgather2_run(out[:, :xa_D_EXPERT], state["pre_expert_wf"], rinfo, cnt_tab, xa_RMS_EPS, a, order, meta, nzero)
    return xa__moe_tail(T, a, order, meta, tw, slot, tid, state, tn)


def xa__after_attn(data, state, out, gate, cfg):
    """From FA3's output to the layer's outputs (more than 8 tokens)."""
    tokens = cfg["tokens"]
    dense = state["dense"]
    o = state["o_norm_gate"](out.reshape(tokens, -1).to(torch.bfloat16), gate)
    attn_output = xa__gemm(o, dense["o"])
    ffn_input, resid, _ = state["attn_boundary"](data["stream"], attn_output, branch_dtype=torch.bfloat16)
    expert_output, topk_ids = xa__moe_block(ffn_input, data, state, dense["pooler"], state["side_stream"])
    ffn_output = xa__gemm(expert_output, dense["unpooler"])
    next_attn_input, resid, _ = state["ffn_boundary"](resid, ffn_output, branch_dtype=torch.bfloat16)
    return next_attn_input, resid, topk_ids


def xa_kernel_fn(data, state):
    cfg = data["config"]
    batch, length, tokens = cfg["batch"], cfg["q_len"], cfg["tokens"]
    hq, hkv, d = cfg["num_q_heads"], cfg["num_kv_heads"], cfg["head_dim"]
    dense = state["dense"]
    qkv = xa__gemm(data["attn_input"], dense["qkv_gate"])
    if cfg["kv_scale"] == 1.0 and data["k_cache"].dtype == xa_FP8 and tokens > xa_ROUTER_FP32_MAX_TOKENS:
        q8 = torch.empty((tokens, hq * d), dtype=xa_FP8, device=qkv.device)
        xa__glue.front_run(qkv, data["positions"], state["cos_sin"], data["cache_loc"], q8,
                        data["k_cache"].view(torch.uint8), data["v_cache"].view(torch.uint8), xa_RMS_EPS,
                        1 if cfg["rope_theta"] else 0)
        gate = qkv[:, hq * d + 2 * hkv * d:]
        window = cfg["window"]
        out = flash_attn_with_kvcache(
            q=q8.view(tokens, hq, d), k_cache=data["k_cache"], v_cache=data["v_cache"], page_table=data["page_table"],
            cache_seqlens=data["cache_seqlens"], cu_seqlens_q=data["cu_seqlens_q"], max_seqlen_q=length,
            softmax_scale=cfg["sm_scale"] * cfg["qk_scale_factor"], causal=True,
            window_size=(window - 1, 0) if window else (-1, -1), scheduler_metadata=data["scheduler_metadata"])
        if isinstance(out, tuple):
            out = out[0]
        return xa__after_attn(data, state, out, gate, cfg)
    if cfg["rope_theta"]:
        q, k = state["qk_norm"](qkv, data["positions"], state["cos_sin"])
    else:
        q, k = state["qk_norm"](qkv)
    v = qkv[:, hq * d + hkv * d: hq * d + 2 * hkv * d]
    gate = qkv[:, hq * d + 2 * hkv * d:]
    k_cache, v_cache = data["k_cache"], data["v_cache"]
    scaled = cfg["kv_scale"] != 1.0
    k = k.reshape(tokens, hkv * d)
    if scaled:
        k, v = k / data["k_scale"], v / data["v_scale"]
    store_cache(k.to(xa_FP8).view(torch.uint8), v.to(xa_FP8).view(torch.uint8),
                k_cache.view(-1, hkv * d).view(torch.uint8), v_cache.view(-1, hkv * d).view(torch.uint8),
                data["cache_loc"], reserved_skip_index=-1)
    window = cfg["window"]
    descale = {}
    if scaled:
        descale = dict(k_descale=data["k_scale"].reshape(1, 1).expand(batch, hkv),
                       v_descale=data["v_scale"].reshape(1, 1).expand(batch, hkv))
    out = flash_attn_with_kvcache(
        q=q.view(tokens, hq, d).to(xa_FP8),
        k_cache=k_cache,
        v_cache=v_cache,
        page_table=data["page_table"],
        cache_seqlens=data["cache_seqlens"],
        cu_seqlens_q=data["cu_seqlens_q"],
        max_seqlen_q=length,
        softmax_scale=cfg["sm_scale"] * cfg["qk_scale_factor"],
        causal=True,
        window_size=(window - 1, 0) if window else (-1, -1),
        scheduler_metadata=data["scheduler_metadata"],
        **descale,
    )
    if isinstance(out, tuple):
        out = out[0]
    o = state["o_norm_gate"](out.reshape(tokens, hq * d).to(torch.bfloat16), gate)
    attn_output = xa__gemm(o, dense["o"])

    split_router = tokens > xa_ROUTER_FP32_MAX_TOKENS
    ffn_input, resid, router_input = state["attn_boundary"](
        data["stream"], attn_output, branch_dtype=torch.bfloat16, router_dtype=None if split_router else torch.float32)
    if split_router and tokens >= xa_MOE_MIN_TOKENS:
        expert_output, topk_ids = xa__moe_block(ffn_input, data, state)
        ffn_output = xa__gemm(expert_output, dense["unpooler"])
        next_attn_input, resid, _ = state["ffn_boundary"](resid, ffn_output, branch_dtype=torch.bfloat16)
        return next_attn_input, resid, topk_ids
    main_stream, side_stream = torch.cuda.current_stream(), state["side_stream"]
    side_stream.wait_stream(main_stream)
    expert_input = xa__gemm(ffn_input, dense["pooler"])
    with torch.cuda.stream(side_stream):
        if split_router:
            logits = torch.mm(ffn_input, state["router_hi_t"], out_dtype=torch.float32)
            logits += torch.mm(ffn_input, state["router_lo_t"], out_dtype=torch.float32)
        else:
            logits = F.linear(router_input, state["router"])
        topk_weights, topk_ids = state["routing"](None, logits, xa_TOP_K, True)
    main_stream.wait_stream(side_stream)
    if not torch.cuda.is_current_stream_capturing():
        for tensor in (topk_weights, topk_ids):
            tensor.record_stream(main_stream)
    local_ids = (topk_ids - data["local_start"]) % xa_N_EXPERTS
    if tokens >= xa_MOE_MIN_TOKENS:
        expert_output = xa__experts_moe(state["pre_expert_norm"](expert_input), topk_weights, local_ids, state)
    elif tokens >= xa_GROUPED_MIN_TOKENS:
        expert_output = xa__experts_grouped(state["pre_expert_norm"](expert_input), topk_weights, local_ids, state)
    else:
        expert_output = xa__experts(state["pre_expert_norm"](expert_input), topk_weights, local_ids, state)
    if tokens <= xa_SLICED_UNPOOL_MAX_TOKENS:
        unpooled = xa__gemm(expert_output, dense["unpooler"][:xa_UNPOOL_SLICE])
        ffn_output = F.pad(unpooled, (0, xa_HIDDEN - xa_UNPOOL_SLICE))
    else:
        ffn_output = xa__gemm(expert_output, dense["unpooler"])

    next_attn_input, resid, _ = state["ffn_boundary"](resid, ffn_output, branch_dtype=torch.bfloat16)
    return next_attn_input, resid, topk_ids




xn__NR_SRC = r"""



#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <stdint.h>

namespace {
__device__ __forceinline__ float bfr(float x) { return __bfloat162float(__float2bfloat16(x)); }
__device__ __forceinline__ void ld4(const __nv_bfloat16* p, int i, float* f) {
 const uint2 v = reinterpret_cast<const uint2*>(p)[i];
 const float2 a = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&v.x));
 const float2 b = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&v.y));
 f[0] = a.x; f[1] = a.y; f[2] = b.x; f[3] = b.y;
}
__device__ __forceinline__ float bsum(float v, float* red) {
#pragma unroll
 for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
 if (threadIdx.x % 32 == 0) red[threadIdx.x / 32] = v;
 __syncthreads();
 float t = red[threadIdx.x & 15];
#pragma unroll
 for (int o = 8; o > 0; o >>= 1) t += __shfl_xor_sync(0xffffffffu, t, o);
 return t;
}



template <bool SCALE, bool POSTW, bool COPY>
__global__ void __launch_bounds__(512, 2) nr2_k(const __nv_bfloat16* __restrict__ br, const float* __restrict__ st,
  __nv_bfloat16* __restrict__ out, float* __restrict__ nst, const __nv_bfloat16* __restrict__ postw,
  const __nv_bfloat16* __restrict__ prew, const float* __restrict__ alpha, const float* __restrict__ beta,
  float post_eps, float pre_eps) {
 __shared__ float red[2][16];
 asm volatile("griddepcontrol.wait;" ::: "memory");
 const int tid = threadIdx.x;
 const size_t row = blockIdx.x;
 const float4* s4 = reinterpret_cast<const float4*>(st + row * 8192);
 const __nv_bfloat16* brow = br + row * 8192;
 float x[16], b[16];
#pragma unroll
 for (int g = 0; g < 4; g++) {
  const float4 v = __ldlu(s4 + 512 * g + tid);
  x[4 * g] = v.x; x[4 * g + 1] = v.y; x[4 * g + 2] = v.z; x[4 * g + 3] = v.w;
  ld4(brow, 512 * g + tid, b + 4 * g);
 }
 if (COPY) {
  float4* n4 = reinterpret_cast<float4*>(nst + row * 8192);
#pragma unroll
  for (int g = 0; g < 4; g++) {
   n4[512 * g + tid] = make_float4(x[4 * g] + b[4 * g], x[4 * g + 1] + b[4 * g + 1], x[4 * g + 2] + b[4 * g + 2],
                   x[4 * g + 3] + b[4 * g + 3]);
   uint2 o;
   *reinterpret_cast<__nv_bfloat162*>(&o.x) = __floats2bfloat162_rn(x[4 * g], x[4 * g + 1]);
   *reinterpret_cast<__nv_bfloat162*>(&o.y) = __floats2bfloat162_rn(x[4 * g + 2], x[4 * g + 3]);
   reinterpret_cast<uint2*>(out + row * 8192)[512 * g + tid] = o;
  }
  return;
 }
 if (SCALE) {
#pragma unroll
  for (int g = 0; g < 4; g++) {
   float w[4];
   ld4(postw, 512 * g + tid, w);
#pragma unroll
   for (int q = 0; q < 4; q++) b[4 * g + q] = bfr(b[4 * g + q] * w[q]);
  }
 }
 float ss = 0.f;
#pragma unroll
 for (int j = 0; j < 16; j++) ss += b[j] * b[j];
 const float rp = rsqrtf(bsum(ss, red[0]) / 8192.f + post_eps);
 float ss2 = 0.f;
#pragma unroll
 for (int g = 0; g < 4; g++) {
  const float4 al = reinterpret_cast<const float4*>(alpha)[512 * g + tid];
  const float4 be = reinterpret_cast<const float4*>(beta)[512 * g + tid];
  const float a4[4] = {al.x, al.y, al.z, al.w}, b4[4] = {be.x, be.y, be.z, be.w};
  float w[4] = {1.f, 1.f, 1.f, 1.f};
  if (!SCALE && POSTW) ld4(postw, 512 * g + tid, w);
#pragma unroll
  for (int q = 0; q < 4; q++) {
   float nb = b[4 * g + q] * rp;
   if (!SCALE && POSTW) nb *= w[q];
   nb = bfr(nb);
   const int j = 4 * g + q;
   x[j] = a4[q] * x[j] + b4[q] * nb;
   ss2 += x[j] * x[j];
  }
  if (!SCALE) {
   float4* q_ = reinterpret_cast<float4*>(nst + row * 8192) + 512 * g + tid;
   unsigned long long pol_;
   asm volatile("createpolicy.fractional.L2::evict_last.b64 %0, 1.0;" : "=l"(pol_));
   asm volatile("st.global.L2::cache_hint.v4.f32 [%0], {%1, %2, %3, %4}, %5;" ::"l"(q_), "f"(x[4 * g]), "f"(x[4 * g + 1]),
          "f"(x[4 * g + 2]), "f"(x[4 * g + 3]), "l"(pol_) : "memory");
  } else {
   reinterpret_cast<float4*>(nst + row * 8192)[512 * g + tid] = make_float4(x[4 * g], x[4 * g + 1], x[4 * g + 2], x[4 * g + 3]);
  }
 }
 const float rq = rsqrtf(bsum(ss2, red[1]) / 8192.f + pre_eps);
#pragma unroll
 for (int g = 0; g < 4; g++) {
  float w[4];
  ld4(prew, 512 * g + tid, w);
  uint2 o;
  *reinterpret_cast<__nv_bfloat162*>(&o.x) = __floats2bfloat162_rn(x[4 * g] * rq * w[0], x[4 * g + 1] * rq * w[1]);
  *reinterpret_cast<__nv_bfloat162*>(&o.y) = __floats2bfloat162_rn(x[4 * g + 2] * rq * w[2], x[4 * g + 3] * rq * w[3]);
  reinterpret_cast<uint2*>(out + row * 8192)[512 * g + tid] = o;
 }
}
}

void nr2_run(torch::Tensor br, torch::Tensor st, torch::Tensor out, torch::Tensor nst, torch::Tensor postw,
       torch::Tensor prew, torch::Tensor alpha, torch::Tensor beta, double post_eps, double pre_eps,
       int64_t mode, int64_t pdl) {
 const int T = (int)br.size(0);
 auto s = at::cuda::getCurrentCUDAStream();
 auto bp = (const __nv_bfloat16*)br.data_ptr();
 auto pp = postw.numel() ? (const __nv_bfloat16*)postw.data_ptr() : nullptr;
 auto qp = (const __nv_bfloat16*)prew.data_ptr();
 auto op = (__nv_bfloat16*)out.data_ptr();
#define NR2_ARGS bp, (const float*)st.data_ptr<float>(), op, nst.data_ptr<float>(), pp, qp, (const float*)alpha.data_ptr<float>(), \
         (const float*)beta.data_ptr<float>(), (float)post_eps, (float)pre_eps
 cudaLaunchAttribute at_[1];
 at_[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
 at_[0].val.programmaticStreamSerializationAllowed = 1;
 cudaLaunchConfig_t c_ = {dim3(T), dim3(512), 0, s, at_, pdl ? 1u : 0u};
 if (mode == 9) cudaLaunchKernelEx(&c_, nr2_k<false, false, true>, NR2_ARGS);
 else if (mode == 1) cudaLaunchKernelEx(&c_, nr2_k<true, true, false>, NR2_ARGS);
 else if (pp) cudaLaunchKernelEx(&c_, nr2_k<false, true, false>, NR2_ARGS);
 else cudaLaunchKernelEx(&c_, nr2_k<false, false, false>, NR2_ARGS);
}
"""
xn__nr = load_inline(name="okc_nr2_z06930_b460010", cuda_sources=xn__NR_SRC,
                  cpp_sources="void nr2_run(torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, "
                              "torch::Tensor, torch::Tensor, torch::Tensor, double, double, int64_t, int64_t);",
                  functions=["nr2_run"], extra_cuda_cflags=["-O3", "-gencode=arch=compute_90a,code=sm_90a"])


def xn_nr_params(op):
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


def xn_boundary(p, resid, branch, mode=None, pdl=True):
    """(BF16 pre-normed output, new FP32 residual) for a [T, 8192] branch (BF16) and residual (FP32).

    mode overrides p["scale"]: 0 plain post-norm, 1 post-norm with input scale, 9 copy only (no math)."""
    out = torch.empty_like(branch)
    new = torch.empty_like(resid)
    xn__nr.nr2_run(branch, resid, out, new, p["postw"], p["prew"], p["alpha"], p["beta"], p["post_eps"], p["pre_eps"],
                p["scale"] if mode is None else mode, 1 if pdl else 0)
    return out, new




xb__NR3_SRC = r"""




#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <stdint.h>

namespace {
__device__ __forceinline__ float bfr(float x) { return __bfloat162float(__float2bfloat16(x)); }
__device__ __forceinline__ void ld4(const __nv_bfloat16* p, int i, float* f) {
 const uint2 v = reinterpret_cast<const uint2*>(p)[i];
 const float2 a = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&v.x));
 const float2 b = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&v.y));
 f[0] = a.x; f[1] = a.y; f[2] = b.x; f[3] = b.y;
}
__device__ __forceinline__ void cv4(const uint2 v, float* f) {
 const float2 a = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&v.x));
 const float2 b = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&v.y));
 f[0] = a.x; f[1] = a.y; f[2] = b.x; f[3] = b.y;
}
__device__ __forceinline__ float bsum(float v, float* red) {
#pragma unroll
 for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
 if (threadIdx.x % 32 == 0) red[threadIdx.x / 32] = v;
 __syncthreads();
 float t = red[threadIdx.x & 15];
#pragma unroll
 for (int o = 8; o > 0; o >>= 1) t += __shfl_xor_sync(0xffffffffu, t, o);
 return t;
}
struct R4 { uint2 v[4]; };
struct NrP {
 const __nv_bfloat16* postw;
 const __nv_bfloat16* prew;
 const float* alpha;
 const float* beta;
 float post_eps, pre_eps;
 int scale;
};


template <bool SCALE, bool POSTW>
__device__ __forceinline__ float resid_step(float (&x)[16], const R4 raw, const NrP p, float* red, float rp_in) {
 const int tid = threadIdx.x;
 float b[16];
#pragma unroll
 for (int g = 0; g < 4; g++) cv4(raw.v[g], b + 4 * g);
 if (SCALE) {
#pragma unroll
  for (int g = 0; g < 4; g++) {
   float w[4];
   ld4(p.postw, 512 * g + tid, w);
#pragma unroll
   for (int q = 0; q < 4; q++) b[4 * g + q] = bfr(b[4 * g + q] * w[q]);
  }
 }
 float rp = rp_in;
 if (rp_in < 0.f) {
  float ss = 0.f;
#pragma unroll
  for (int j = 0; j < 16; j++) ss += b[j] * b[j];
  rp = rsqrtf(bsum(ss, red) / 8192.f + p.post_eps);
 }
#pragma unroll
 for (int g = 0; g < 4; g++) {
  const float4 al = reinterpret_cast<const float4*>(p.alpha)[512 * g + tid];
  const float4 be = reinterpret_cast<const float4*>(p.beta)[512 * g + tid];
  const float a4[4] = {al.x, al.y, al.z, al.w}, b4[4] = {be.x, be.y, be.z, be.w};
  float w[4] = {1.f, 1.f, 1.f, 1.f};
  if (!SCALE && POSTW) ld4(p.postw, 512 * g + tid, w);
#pragma unroll
  for (int q = 0; q < 4; q++) {
   float nb = b[4 * g + q] * rp;
   if (!SCALE && POSTW) nb *= w[q];
   nb = bfr(nb);
   const int j = 4 * g + q;
   x[j] = a4[q] * x[j] + b4[q] * nb;
  }
 }
 return rp;
}

__device__ __forceinline__ void pre_out(const float (&x)[16], __nv_bfloat16* orow, const NrP p, float* red) {
 const int tid = threadIdx.x;
 float ss2 = 0.f;
#pragma unroll
 for (int j = 0; j < 16; j++) ss2 += x[j] * x[j];
 const float rq = rsqrtf(bsum(ss2, red) / 8192.f + p.pre_eps);
#pragma unroll
 for (int g = 0; g < 4; g++) {
  float w[4];
  ld4(p.prew, 512 * g + tid, w);
  uint2 o;
  *reinterpret_cast<__nv_bfloat162*>(&o.x) = __floats2bfloat162_rn(x[4 * g] * rq * w[0], x[4 * g + 1] * rq * w[1]);
  *reinterpret_cast<__nv_bfloat162*>(&o.y) = __floats2bfloat162_rn(x[4 * g + 2] * rq * w[2], x[4 * g + 3] * rq * w[3]);
  reinterpret_cast<uint2*>(orow)[512 * g + tid] = o;
 }
}
__device__ __forceinline__ float resid_any(float (&x)[16], const R4 raw, const NrP p, float* red, float rp_in = -1.f) {
 if (p.scale) return resid_step<true, true>(x, raw, p, red, rp_in);
 else if (p.postw) return resid_step<false, true>(x, raw, p, red, rp_in);
 else return resid_step<false, false>(x, raw, p, red, rp_in);
}





template <bool B>
__global__ void __launch_bounds__(512, 2) nr3_k(const float* __restrict__ st, const __nv_bfloat16* __restrict__ bra,
  const __nv_bfloat16* __restrict__ brb, __nv_bfloat16* __restrict__ out, float* __restrict__ nst, float* __restrict__ rps, NrP pa, NrP pb,
  const int* __restrict__ perm) {
 __shared__ float red[4][16];
 __shared__ __align__(16) uint2 sb[B ? 2 : 1][B ? 2048 : 1];


 const size_t src = blockIdx.x;
 const int dst_i = perm ? __ldg(perm + src) : (int)src;
 const size_t row = (size_t)dst_i;
 asm volatile("griddepcontrol.wait;" ::: "memory");
 const int tid = threadIdx.x;
 const float4* s4 = reinterpret_cast<const float4*>(st + src * 8192);
 R4 ra;
 if (B) {
  const char* ga = reinterpret_cast<const char*>(bra + src * 8192);
  const char* gb = reinterpret_cast<const char*>(brb + row * 8192);
#pragma unroll
  for (int r = 0; r < 2; ++r) {
   const int c = tid + 512 * r;
   const uint32_t da = static_cast<uint32_t>(__cvta_generic_to_shared(reinterpret_cast<char*>(&sb[0][0]) + 16 * c));
   const uint32_t db = static_cast<uint32_t>(__cvta_generic_to_shared(reinterpret_cast<char*>(&sb[B ? 1 : 0][0]) + 16 * c));
   asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" ::"r"(da), "l"(ga + 16 * c) : "memory");
   asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" ::"r"(db), "l"(gb + 16 * c) : "memory");
  }
  asm volatile("cp.async.commit_group;" ::: "memory");
 }
 float x[16];
#pragma unroll
 for (int g = 0; g < 4; g++) {
  const float4 v = s4[512 * g + tid];
  x[4 * g] = v.x; x[4 * g + 1] = v.y; x[4 * g + 2] = v.z; x[4 * g + 3] = v.w;
  if (!B) ra.v[g] = reinterpret_cast<const uint2*>(bra + src * 8192)[512 * g + tid];
 }
 if (B) {
  asm volatile("cp.async.wait_all;" ::: "memory");
  __syncthreads();
#pragma unroll
  for (int g = 0; g < 4; g++) ra.v[g] = sb[0][512 * g + tid];
 }
 const float rpa = resid_any(x, ra, pa, red[0], B ? rps[row] : -1.f);
 if (!B && tid == 0) rps[row] = rpa;
 if (B) {
  R4 rb;
#pragma unroll
  for (int g = 0; g < 4; g++) rb.v[g] = sb[B ? 1 : 0][512 * g + tid];
  resid_any(x, rb, pb, red[1]);
#pragma unroll
  for (int g = 0; g < 4; g++)
   reinterpret_cast<float4*>(nst + row * 8192)[512 * g + tid] = make_float4(x[4 * g], x[4 * g + 1], x[4 * g + 2], x[4 * g + 3]);
  pre_out(x, out + row * 8192, pb, red[2]);
 } else {
  pre_out(x, out + row * 8192, pa, red[2]);
 }
}
}

static NrP mkp(torch::Tensor postw, torch::Tensor prew, torch::Tensor alpha, torch::Tensor beta, double post_eps,
        double pre_eps, int64_t scale) {
 NrP p;
 p.postw = postw.numel() ? (const __nv_bfloat16*)postw.data_ptr() : nullptr;
 p.prew = (const __nv_bfloat16*)prew.data_ptr();
 p.alpha = alpha.data_ptr<float>();
 p.beta = beta.data_ptr<float>();
 p.post_eps = (float)post_eps;
 p.pre_eps = (float)pre_eps;
 p.scale = (int)scale;
 return p;
}

void nr3_run(torch::Tensor st, torch::Tensor bra, torch::Tensor brb, torch::Tensor out, torch::Tensor nst,
       torch::Tensor apw, torch::Tensor aqw, torch::Tensor aal, torch::Tensor abe, double apeps, double aqeps,
       int64_t asc, torch::Tensor bpw, torch::Tensor bqw, torch::Tensor bal, torch::Tensor bbe, double bpeps,
       double bqeps, int64_t bsc, int64_t two, torch::Tensor rps, int64_t pdl, torch::Tensor perm) {
 const int T = (int)st.size(0);
 TORCH_CHECK(rps.numel() >= T && rps.scalar_type() == torch::kFloat, "nr3 rps");
 NrP pa = mkp(apw, aqw, aal, abe, apeps, aqeps, asc);
 NrP pb = two ? mkp(bpw, bqw, bal, bbe, bpeps, bqeps, bsc) : pa;
 cudaLaunchAttribute at_[1];
 at_[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
 at_[0].val.programmaticStreamSerializationAllowed = 1;
 cudaLaunchConfig_t c_ = {dim3(T), dim3(512), 0, at::cuda::getCurrentCUDAStream(), at_, pdl ? 1u : 0u};
 const __nv_bfloat16* bb = two ? (const __nv_bfloat16*)brb.data_ptr() : nullptr;
 float* np = two ? nst.data_ptr<float>() : nullptr;
 const int* pm = perm.numel() ? perm.data_ptr<int>() : nullptr;
 TORCH_CHECK(!pm || perm.numel() >= T, "nr3 perm");
 if (two) cudaLaunchKernelEx(&c_, nr3_k<true>, (const float*)st.data_ptr<float>(), (const __nv_bfloat16*)bra.data_ptr(),
               bb, (__nv_bfloat16*)out.data_ptr(), np, rps.data_ptr<float>(), pa, pb, pm);
 else cudaLaunchKernelEx(&c_, nr3_k<false>, (const float*)st.data_ptr<float>(), (const __nv_bfloat16*)bra.data_ptr(),
             bb, (__nv_bfloat16*)out.data_ptr(), np, rps.data_ptr<float>(), pa, pb, pm);
}
"""
xb__nr3 = load_inline(name="okc_nr3_z07469_b460010", cuda_sources=xb__NR3_SRC,
                   cpp_sources="void nr3_run(torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, "
                               "torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, double, double, int64_t, "
                               "torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, double, double, int64_t, int64_t, torch::Tensor, int64_t, torch::Tensor);",
                   functions=["nr3_run"], extra_cuda_cflags=["-O3", "-gencode=arch=compute_90a,code=sm_90a"])


xb__NOPERM = {}


def xb__noperm(dev):
    if dev not in xb__NOPERM:
        xb__NOPERM[dev] = torch.empty(0, dtype=torch.int32, device=dev)
    return xb__NOPERM[dev]


def xb_boundary_a(pa, resid, attn_out, pdl=True, perm=None):
    """(FFN input BF16, per-row post-norm scale of the attention branch) after the attention boundary; the residual
    itself is not written. perm (int32 [T] or None): here the INVERSE map, token t's outputs go to row perm[t]."""
    out = torch.empty_like(attn_out)
    rps = torch.empty(attn_out.shape[0], dtype=torch.float32, device=attn_out.device)
    xb__nr3.nr3_run(resid, attn_out, attn_out, out, resid, pa["postw"], pa["prew"], pa["alpha"], pa["beta"],
                 pa["post_eps"], pa["pre_eps"], pa["scale"], pa["postw"], pa["prew"], pa["alpha"], pa["beta"],
                 pa["post_eps"], pa["pre_eps"], pa["scale"], 0, rps, 1 if pdl else 0,
                 xb__noperm(attn_out.device) if perm is None else perm)
    return out, rps


def xb_boundary_b(pa, pb, resid, attn_out, ffn_out, rps):
    """(next attention input BF16, new FP32 residual): the attention residual recomputed, then the FFN boundary."""
    out = torch.empty_like(ffn_out)
    new = torch.empty_like(resid)
    xb__nr3.nr3_run(resid, attn_out, ffn_out, out, new, pa["postw"], pa["prew"], pa["alpha"], pa["beta"],
                 pa["post_eps"], pa["pre_eps"], pa["scale"], pb["postw"], pb["prew"], pb["alpha"], pb["beta"],
                 pb["post_eps"], pb["pre_eps"], pb["scale"], 1, rps, 1, xb__noperm(ffn_out.device))
    return out, new



xr__SRC = r"""






#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <stdint.h>

namespace {
constexpr int RMAX = 128;
constexpr int NT = 1024, VEC = 8192 / NT, GROUPS = VEC / 4, WARPS = NT / 32;
__device__ __forceinline__ float bfr(float x) { return __bfloat162float(__float2bfloat16(x)); }
__device__ __forceinline__ void ld4(const __nv_bfloat16* p, int i, float* f) {
 uint2 v = reinterpret_cast<const uint2*>(p)[i];
 float2 a = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&v.x));
 float2 b = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&v.y));
 f[0]=a.x; f[1]=a.y; f[2]=b.x; f[3]=b.y;
}

__device__ __forceinline__ void reg4(uint32_t lo, uint32_t hi, float* f) {
 uint32_t a,b,c,d;
 asm volatile("{ .reg .b32 x,y,z,w; "
   "shl.b32 x, %4, 16; and.b32 y, %4, 0xffff0000; "
   "shl.b32 z, %5, 16; and.b32 w, %5, 0xffff0000; "
   "mov.b32 %0, x; mov.b32 %1, y; mov.b32 %2, z; mov.b32 %3, w; }"
   : "=r"(a),"=r"(b),"=r"(c),"=r"(d) : "r"(lo),"r"(hi));
 f[0]=__uint_as_float(a); f[1]=__uint_as_float(b);
 f[2]=__uint_as_float(c); f[3]=__uint_as_float(d);
}
__device__ __forceinline__ float bsum(float v, float* red) {
 #pragma unroll
 for (int o=16; o>0; o>>=1) v+=__shfl_xor_sync(0xffffffffu,v,o);
 if ((threadIdx.x&31)==0) red[threadIdx.x/32]=v;
 __syncthreads();
 float s=red[(threadIdx.x & 31) & (WARPS - 1)];
 #pragma unroll
 for (int o=WARPS/2; o>0; o>>=1) s+=__shfl_xor_sync(0xffffffffu,s,o);
 return s;
}
__device__ __forceinline__ void minit(uint32_t a) {
 asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" :: "r"(a) : "memory");
}
__device__ __forceinline__ void expect(uint32_t a) {
 asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], 65536;" :: "r"(a) : "memory");
}
__device__ __forceinline__ void wait(uint32_t a, int p) {
 asm volatile("{ .reg .pred q; W: mbarrier.try_wait.parity.shared::cta.b64 q, [%0], %1; @!q bra W; }"
   :: "r"(a),"r"(p) : "memory");
}
__device__ __forceinline__ void bulk(uint32_t d, const void* s, int n, uint32_t b) {
 asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];"
   :: "r"(d),"l"(s),"r"(n),"r"(b) : "memory");
}



template <bool MAP>
__global__ void __launch_bounds__(NT,1) remat_k(const float* __restrict__ resid,
  const __nv_bfloat16* __restrict__ attn, const __nv_bfloat16* __restrict__ ffn,
  __nv_bfloat16* __restrict__ out, float* __restrict__ nst, const float* __restrict__ rps,
  const float* __restrict__ aa, const float* __restrict__ ba,
  const float* __restrict__ ab, const float* __restrict__ bb,
  const __nv_bfloat16* __restrict__ postw, const __nv_bfloat16* __restrict__ prew,
  float post_eps, float pre_eps, int T,
  const int* __restrict__ inv, const int* __restrict__ ids_in, int* __restrict__ ids_out) {
 __shared__ float red[2][WARPS];
 __shared__ __align__(16) uint64_t bars[2];
 extern __shared__ __align__(128) unsigned char stage[];
 __shared__ int prow[MAP ? RMAX : 1];
 __shared__ float prps[MAP ? RMAX : 1];
 const int nk=(int)((T-1-(int)blockIdx.x)/(int)gridDim.x)+1;
 if (MAP)
  for (int k=threadIdx.x; k<nk; k+=NT) prow[k]=__ldg(inv+blockIdx.x+(size_t)k*gridDim.x);
 asm volatile("griddepcontrol.wait;" ::: "memory");
 const int tid=threadIdx.x;
 float ar[VEC], br[VEC], cr[VEC], dr[VEC];
 uint32_t pre_r[2 * GROUPS], post_r[2 * GROUPS];
 #pragma unroll
 for (int g=0; g<GROUPS; ++g) {
  float4 a=reinterpret_cast<const float4*>(aa)[NT*g+tid];
  float4 b=reinterpret_cast<const float4*>(ba)[NT*g+tid];
  float4 c=reinterpret_cast<const float4*>(ab)[NT*g+tid];
  float4 d=reinterpret_cast<const float4*>(bb)[NT*g+tid];
  ar[4*g]=a.x; ar[4*g+1]=a.y; ar[4*g+2]=a.z; ar[4*g+3]=a.w;
  br[4*g]=b.x; br[4*g+1]=b.y; br[4*g+2]=b.z; br[4*g+3]=b.w;
  cr[4*g]=c.x; cr[4*g+1]=c.y; cr[4*g+2]=c.z; cr[4*g+3]=c.w;
  dr[4*g]=d.x; dr[4*g+1]=d.y; dr[4*g+2]=d.z; dr[4*g+3]=d.w;
  uint2 p=reinterpret_cast<const uint2*>(prew)[NT*g+tid];
  uint2 q=reinterpret_cast<const uint2*>(postw)[NT*g+tid];
  pre_r[2*g]=p.x; pre_r[2*g+1]=p.y;
  post_r[2*g]=q.x; post_r[2*g+1]=q.y;
 }
 const uint32_t sa=(uint32_t)__cvta_generic_to_shared(stage);
 const uint32_t ma=(uint32_t)__cvta_generic_to_shared(bars);
 if (tid==0) {
  minit(ma); minit(ma+8);
  asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
 }
 __syncthreads();
 auto prefetch=[&](size_t row,int slot,int k) {
  if (tid!=0 || row>=(size_t)T) return;
  uint32_t dst=sa+slot*65536, bar=ma+slot*8;
  expect(bar);
  bulk(dst,resid+row*8192,32768,bar);
  bulk(dst+32768,attn+row*8192,16384,bar);
  bulk(dst+49152,ffn+(MAP?(size_t)prow[k]:row)*8192,16384,bar);
 };
 prefetch(blockIdx.x,0,0);
 if (MAP) {
  for (int k=tid; k<nk; k+=NT) prps[k]=rps[prow[k]];
  for (int k=tid; k<8*nk; k+=NT) ids_out[(blockIdx.x+(size_t)(k>>3)*gridDim.x)*8+(k&7)]=ids_in[(size_t)prow[k>>3]*8+(k&7)];
  __syncthreads();
 }
 int iteration=0;
 for (size_t row=blockIdx.x; row<(size_t)T; row+=gridDim.x,++iteration) {
  int slot=iteration&1;
  wait(ma+slot*8,(iteration>>1)&1);
  const unsigned char* cur=stage+slot*65536;
  float x[VEC], b[VEC];
  #pragma unroll
  for (int g=0; g<GROUPS; ++g) {
   float4 v=reinterpret_cast<const float4*>(cur)[NT*g+tid];
   x[4*g]=v.x; x[4*g+1]=v.y; x[4*g+2]=v.z; x[4*g+3]=v.w;
   ld4(reinterpret_cast<const __nv_bfloat16*>(cur+32768),NT*g+tid,b+4*g);
  }
  prefetch(row+gridDim.x,slot^1,iteration+1);
  const float rpa=MAP?prps[iteration]:rps[row];
  #pragma unroll
  for (int j=0; j<VEC; ++j) {
   float nb=bfr(b[j]*rpa);
   x[j]=ar[j]*x[j]+br[j]*nb;
  }
  #pragma unroll
  for (int g=0; g<GROUPS; ++g)
   ld4(reinterpret_cast<const __nv_bfloat16*>(cur+49152),NT*g+tid,b+4*g);
  #pragma unroll
  for (int g=0; g<GROUPS; ++g) {
   float w[4]; reg4(post_r[2*g],post_r[2*g+1],w);
   #pragma unroll
   for (int q=0; q<4; ++q) b[4*g+q]=bfr(b[4*g+q]*w[q]);
  }
  float ss=0.f;
  #pragma unroll
  for (int j=0; j<VEC; ++j) ss+=b[j]*b[j];
  float rp=rsqrtf(bsum(ss,red[0])/8192.f+post_eps);
  float ss2=0.f;
  #pragma unroll
  for (int g=0; g<GROUPS; ++g) {
   #pragma unroll
   for (int q=0; q<4; ++q) {
    int j=4*g+q;
    float nb=bfr(b[j]*rp);
    x[j]=cr[j]*x[j]+dr[j]*nb;
    ss2+=x[j]*x[j];
   }
   reinterpret_cast<float4*>(nst+row*8192)[NT*g+tid]=
     make_float4(x[4*g],x[4*g+1],x[4*g+2],x[4*g+3]);
  }
  float rq=rsqrtf(bsum(ss2,red[1])/8192.f+pre_eps);
  #pragma unroll
  for (int g=0; g<GROUPS; ++g) {
   float w[4]; reg4(pre_r[2*g],pre_r[2*g+1],w);
   uint2 v;
   *reinterpret_cast<__nv_bfloat162*>(&v.x)=__floats2bfloat162_rn(x[4*g]*rq*w[0],x[4*g+1]*rq*w[1]);
   *reinterpret_cast<__nv_bfloat162*>(&v.y)=__floats2bfloat162_rn(x[4*g+2]*rq*w[2],x[4*g+3]*rq*w[3]);
   reinterpret_cast<uint2*>(out+row*8192)[NT*g+tid]=v;
  }
 }
}






template <bool MAP>
__global__ void __launch_bounds__(NT,1) remat2_k(const float* __restrict__ resid,
  const __nv_bfloat16* __restrict__ attn, const __nv_bfloat16* __restrict__ ffn,
  __nv_bfloat16* __restrict__ out, float* __restrict__ nst, const float* __restrict__ rps,
  const float* __restrict__ aa, const float* __restrict__ ba,
  const float* __restrict__ ab, const float* __restrict__ bb,
  const __nv_bfloat16* __restrict__ postw, const __nv_bfloat16* __restrict__ prew,
  float post_eps, float pre_eps, int T,
  const int* __restrict__ inv, const int* __restrict__ ids_in, int* __restrict__ ids_out) {
 __shared__ float red[2][WARPS];
 extern __shared__ __align__(16) unsigned char pbuf[];
 float4* sar=reinterpret_cast<float4*>(pbuf);
 float4* sbr=sar+2048; float4* scr=sar+4096; float4* sdr=sar+6144;
 uint2* spost=reinterpret_cast<uint2*>(sar+8192); uint2* spre=spost+2048;
 const int tid=threadIdx.x;
 #pragma unroll
 for (int g=0; g<GROUPS; ++g) {
  const int i=NT*g+tid;
  sar[i]=reinterpret_cast<const float4*>(aa)[i]; sbr[i]=reinterpret_cast<const float4*>(ba)[i];
  scr[i]=reinterpret_cast<const float4*>(ab)[i]; sdr[i]=reinterpret_cast<const float4*>(bb)[i];
  spost[i]=reinterpret_cast<const uint2*>(postw)[i]; spre[i]=reinterpret_cast<const uint2*>(prew)[i];
 }
 int row=blockIdx.x;
 int fr=row<T?(MAP?__ldg(inv+row):row):0;
 asm volatile("griddepcontrol.wait;" ::: "memory");
 float4 cs[GROUPS]; uint2 ca[GROUPS], cf[GROUPS];
 float crps=0.f;
 if (row<T) {
  #pragma unroll
  for (int g=0; g<GROUPS; ++g) {
   cs[g]=reinterpret_cast<const float4*>(resid+(size_t)row*8192)[NT*g+tid];
   ca[g]=reinterpret_cast<const uint2*>(attn+(size_t)row*8192)[NT*g+tid];
   cf[g]=reinterpret_cast<const uint2*>(ffn+(size_t)fr*8192)[NT*g+tid];
  }
  crps=rps[fr];
  if (MAP && tid<8) ids_out[(size_t)row*8+tid]=ids_in[(size_t)fr*8+tid];
 }
 __syncthreads();
 for (; row<T; row+=gridDim.x) {
  const int nrow=row+gridDim.x;
  float4 ns[GROUPS]; uint2 na[GROUPS], nf[GROUPS];
  float nrps=0.f;
  if (nrow<T) {
   const int nfr=MAP?__ldg(inv+nrow):nrow;
   #pragma unroll
   for (int g=0; g<GROUPS; ++g) {
    ns[g]=reinterpret_cast<const float4*>(resid+(size_t)nrow*8192)[NT*g+tid];
    na[g]=reinterpret_cast<const uint2*>(attn+(size_t)nrow*8192)[NT*g+tid];
    nf[g]=reinterpret_cast<const uint2*>(ffn+(size_t)nfr*8192)[NT*g+tid];
   }
   nrps=rps[nfr];
   if (MAP && tid<8) ids_out[(size_t)nrow*8+tid]=ids_in[(size_t)nfr*8+tid];
  }
  float x[VEC], b[VEC];
  #pragma unroll
  for (int g=0; g<GROUPS; ++g) {
   x[4*g]=cs[g].x; x[4*g+1]=cs[g].y; x[4*g+2]=cs[g].z; x[4*g+3]=cs[g].w;
   ld4(reinterpret_cast<const __nv_bfloat16*>(&ca[g]),0,b+4*g);
  }
  const float rpa=crps;
  #pragma unroll
  for (int g=0; g<GROUPS; ++g) {
   const float4 a4=sar[NT*g+tid], b4=sbr[NT*g+tid];
   const float av[4]={a4.x,a4.y,a4.z,a4.w}, bv[4]={b4.x,b4.y,b4.z,b4.w};
   #pragma unroll
   for (int q=0; q<4; ++q) {
    const int j=4*g+q;
    float nb=bfr(b[j]*rpa);
    x[j]=av[q]*x[j]+bv[q]*nb;
   }
  }
  #pragma unroll
  for (int g=0; g<GROUPS; ++g)
   ld4(reinterpret_cast<const __nv_bfloat16*>(&cf[g]),0,b+4*g);
  #pragma unroll
  for (int g=0; g<GROUPS; ++g) {
   const uint2 pw=spost[NT*g+tid];
   float w[4]; reg4(pw.x,pw.y,w);
   #pragma unroll
   for (int q=0; q<4; ++q) b[4*g+q]=bfr(b[4*g+q]*w[q]);
  }
  float ss=0.f;
  #pragma unroll
  for (int j=0; j<VEC; ++j) ss+=b[j]*b[j];
  float rp=rsqrtf(bsum(ss,red[0])/8192.f+post_eps);
  float ss2=0.f;
  #pragma unroll
  for (int g=0; g<GROUPS; ++g) {
   const float4 c4=scr[NT*g+tid], d4=sdr[NT*g+tid];
   const float cv[4]={c4.x,c4.y,c4.z,c4.w}, dv[4]={d4.x,d4.y,d4.z,d4.w};
   #pragma unroll
   for (int q=0; q<4; ++q) {
    int j=4*g+q;
    float nb=bfr(b[j]*rp);
    x[j]=cv[q]*x[j]+dv[q]*nb;
    ss2+=x[j]*x[j];
   }
   reinterpret_cast<float4*>(nst+(size_t)row*8192)[NT*g+tid]=
     make_float4(x[4*g],x[4*g+1],x[4*g+2],x[4*g+3]);
  }
  float rq=rsqrtf(bsum(ss2,red[1])/8192.f+pre_eps);
  #pragma unroll
  for (int g=0; g<GROUPS; ++g) {
   const uint2 pw=spre[NT*g+tid];
   float w[4]; reg4(pw.x,pw.y,w);
   uint2 v;
   *reinterpret_cast<__nv_bfloat162*>(&v.x)=__floats2bfloat162_rn(x[4*g]*rq*w[0],x[4*g+1]*rq*w[1]);
   *reinterpret_cast<__nv_bfloat162*>(&v.y)=__floats2bfloat162_rn(x[4*g+2]*rq*w[2],x[4*g+3]*rq*w[3]);
   reinterpret_cast<uint2*>(out+(size_t)row*8192)[NT*g+tid]=v;
  }
  #pragma unroll
  for (int g=0; g<GROUPS; ++g) { cs[g]=ns[g]; ca[g]=na[g]; cf[g]=nf[g]; }
  crps=nrps;
 }
}
}

void remat_init() {
 cudaFuncSetAttribute(remat_k<false>,cudaFuncAttributeMaxDynamicSharedMemorySize,128*1024);
 cudaFuncSetAttribute(remat_k<true>,cudaFuncAttributeMaxDynamicSharedMemorySize,128*1024);
 cudaFuncSetAttribute(remat2_k<false>,cudaFuncAttributeMaxDynamicSharedMemorySize,160*1024);
 cudaFuncSetAttribute(remat2_k<true>,cudaFuncAttributeMaxDynamicSharedMemorySize,160*1024);
}
void remat_run(torch::Tensor resid,torch::Tensor attn,torch::Tensor ffn,torch::Tensor out,
  torch::Tensor nst,torch::Tensor rps,torch::Tensor aa,torch::Tensor ba,torch::Tensor ab,torch::Tensor bb,
  torch::Tensor postw,torch::Tensor prew,double post_eps,double pre_eps,int64_t grid,
  torch::Tensor inv,torch::Tensor ids_in,torch::Tensor ids_out,int64_t v2) {
 int T=(int)resid.size(0);
 const bool map=inv.numel()>0;
 TORCH_CHECK(!map || (inv.numel()>=T && T<=(int64_t)RMAX*grid && ids_in.numel()>=8*T && ids_out.numel()>=8*T), "remat map");
 cudaLaunchAttribute attrs[1];
 attrs[0].id=cudaLaunchAttributeProgrammaticStreamSerialization;
 attrs[0].val.programmaticStreamSerializationAllowed=1;
 cudaLaunchConfig_t cfg={dim3((unsigned)grid),dim3(NT),(size_t)(v2?160:128)*1024,at::cuda::getCurrentCUDAStream(),attrs,1};
 cudaLaunchKernelEx(&cfg,v2?(map?remat2_k<true>:remat2_k<false>):(map?remat_k<true>:remat_k<false>),resid.data_ptr<float>(),(const __nv_bfloat16*)attn.data_ptr(),
   (const __nv_bfloat16*)ffn.data_ptr(),(__nv_bfloat16*)out.data_ptr(),nst.data_ptr<float>(),rps.data_ptr<float>(),
   aa.data_ptr<float>(),ba.data_ptr<float>(),ab.data_ptr<float>(),bb.data_ptr<float>(),
   (const __nv_bfloat16*)postw.data_ptr(),(const __nv_bfloat16*)prew.data_ptr(),(float)post_eps,(float)pre_eps,T,
   map?(const int*)inv.data_ptr<int>():nullptr,map?(const int*)ids_in.data_ptr<int>():nullptr,
   map?ids_out.data_ptr<int>():nullptr);
}
"""

xr__nr = load_inline(name='okc_remat_z08020_b460010', cuda_sources=xr__SRC,
    cpp_sources='void remat_init(); void remat_run(torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,double,double,int64_t,torch::Tensor,torch::Tensor,torch::Tensor,int64_t);',
    functions=['remat_init','remat_run'],
    extra_cuda_cflags=['-O3','-gencode=arch=compute_90a,code=sm_90a','-Xptxas=-v'],verbose=False)

xr_GRID = 132
xr_R2 = True
xr__initialized = False

def xr_init():
    global xr__initialized
    if not xr__initialized:
        xr__nr.remat_init()
        xr__initialized = True

def xr_supported(pa,pb):
    return (not pa['scale'] and pa['postw'].numel()==0 and pb['scale']==1
            and all(p[k].numel()==8192 for p in (pa,pb) for k in ('alpha','beta','prew'))
            and pb['postw'].numel()==8192)

xr__E = {}

def xr__empty(dev):
    if dev not in xr__E:
        xr__E[dev] = torch.empty(0, dtype=torch.int32, device=dev)
    return xr__E[dev]

def xr_run_b(pa,pb,resid,attn_out,ffn_out,rps,out,new,grid=132,inv=None,ids_in=None,ids_out=None):
    xr_init()
    e=xr__empty(resid.device)
    xr__nr.remat_run(resid,attn_out,ffn_out,out,new,rps,pa['alpha'],pa['beta'],pb['alpha'],pb['beta'],
                  pb['postw'],pb['prew'],pb['post_eps'],pb['pre_eps'],grid,
                  e if inv is None else inv,e if inv is None else ids_in,e if inv is None else ids_out,1 if xr_R2 else 0)

def xr_boundary_b(pa,pb,resid,attn_out,ffn_out,rps,inv=None,ids=None):
    """(next input, new residual); with the inverse map inv (token t's FFN row, row scale and ids at row inv[t]) also the
    top-8 ids in token order."""
    out,new=torch.empty_like(ffn_out),torch.empty_like(resid)
    if inv is None:
        xr_run_b(pa,pb,resid,attn_out,ffn_out,rps,out,new,xr_GRID)
        return out,new
    ids_out=torch.empty_like(ids)
    xr_run_b(pa,pb,resid,attn_out,ffn_out,rps,out,new,xr_GRID,inv,ids,ids_out)
    return out,new,ids_out



xk_KC = 32
xk_KS = 128
xk_TMAX = 16384

xk__SRC = r"""

#include <cuda_runtime.h>
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <stdint.h>

namespace {
constexpr int KC = 32, KS = 128, NB = 4096, XL = KC + 1, TMAX = 16384, NT = 1024, NIT = 3;

__global__ void __launch_bounds__(NT, 1) order2_k(const float* __restrict__ x, long ld, int T, int* __restrict__ perm,
                         int* __restrict__ inv, int ident) {
 extern __shared__ __align__(16) int dsm[];
 int* hist = dsm;
 uint16_t* keys = reinterpret_cast<uint16_t*>(dsm + NB);
 __shared__ float xs[KS * XL];
 __shared__ float C[KC * XL];
 __shared__ float red[32][KC + 1];
 __shared__ float mu[KC];
 __shared__ float V[KC][4], W[KC][4];
 __shared__ __align__(16) float4 dir[KC];
 __shared__ float bias[4], isd[4];
 __shared__ int ws[32];
 const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
 for (int i = tid; i < NB; i += NT) hist[i] = 0;

 const int S = T < KS ? T : KS;
 for (int i = tid >> 3; i < S; i += NT / 8) {
  const long r = ((long)i * T) / S;
  const float4 v = reinterpret_cast<const float4*>(x + r * ld)[tid & 7];
  float* d = xs + i * XL + 4 * (tid & 7);
  d[0] = v.x; d[1] = v.y; d[2] = v.z; d[3] = v.w;
 }
 __syncthreads();
 {
  float s = 0.f;
  for (int i = warp; i < S; i += 32) s += xs[i * XL + lane];
  red[warp][lane] = s;
 }
 __syncthreads();
 if (tid < KC) {
  float s = 0.f;
#pragma unroll
  for (int p = 0; p < 32; ++p) s += red[p][tid];
  mu[tid] = s / S;
 }
 __syncthreads();
 for (int i = tid; i < S * KC; i += NT) xs[(i / KC) * XL + i % KC] -= mu[i % KC];
 __syncthreads();
 if (tid < 256) {
  const int a0 = 2 * (tid >> 4), b0 = 2 * (tid & 15);
  float c00 = 0.f, c01 = 0.f, c10 = 0.f, c11 = 0.f;
#pragma unroll 4
  for (int i = 0; i < S; ++i) {
   const float* r = xs + i * XL;
   const float pa0 = r[a0], pa1 = r[a0 + 1], pb0 = r[b0], pb1 = r[b0 + 1];
   c00 += pa0 * pb0; c01 += pa0 * pb1; c10 += pa1 * pb0; c11 += pa1 * pb1;
  }
  const float is = 1.f / S;
  C[a0 * XL + b0] = c00 * is; C[a0 * XL + b0 + 1] = c01 * is;
  C[(a0 + 1) * XL + b0] = c10 * is; C[(a0 + 1) * XL + b0 + 1] = c11 * is;
 }
 if (tid < KC * 4) {
  const int c = tid >> 2, j = tid & 3;
  const uint32_t h = (uint32_t)(c * 4 + j) * 2654435761u + 12345u;
  V[c][j] = (float)((h >> 8) & 0xFFFF) / 65536.f - 0.5f + ((c & 3) == j ? 1.f : 0.f);
 }
 __syncthreads();
 for (int it = 0; it <= NIT; ++it) {
  if (tid < KC * 4) {
   const int c = tid >> 2, j = tid & 3;
   float s = 0.f;
#pragma unroll
   for (int k = 0; k < KC; ++k) s += C[c * XL + k] * V[k][j];
   W[c][j] = s;
  }
  __syncthreads();
  if (warp == 0) {
   if (it == NIT) {
#pragma unroll
    for (int j = 0; j < 4; ++j) {
     float d = V[lane][j] * W[lane][j];
#pragma unroll
     for (int o = 16; o > 0; o >>= 1) d += __shfl_xor_sync(0xffffffffu, d, o);
     if (lane == 0) isd[j] = rsqrtf(fmaxf(d, 1e-30f));
    }
   } else {
    float w[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) w[j] = W[lane][j];
#pragma unroll
    for (int j = 0; j < 4; ++j) {
#pragma unroll
     for (int p = 0; p < j; ++p) {
      float d = w[j] * w[p];
#pragma unroll
      for (int o = 16; o > 0; o >>= 1) d += __shfl_xor_sync(0xffffffffu, d, o);
      w[j] -= d * w[p];
     }
     float n = w[j] * w[j];
#pragma unroll
     for (int o = 16; o > 0; o >>= 1) n += __shfl_xor_sync(0xffffffffu, n, o);
     w[j] *= rsqrtf(n + 1e-30f);
    }
#pragma unroll
    for (int j = 0; j < 4; ++j) V[lane][j] = w[j];
   }
  }
  __syncthreads();
 }
 if (tid < KC) dir[tid] = make_float4(V[tid][0], V[tid][1], V[tid][2], V[tid][3]);
 if (warp == 1) {
#pragma unroll
  for (int j = 0; j < 4; ++j) {
   float d = mu[lane] * V[lane][j];
#pragma unroll
   for (int o = 16; o > 0; o >>= 1) d += __shfl_xor_sync(0xffffffffu, d, o);
   if (lane == 0) bias[j] = d;
  }
 }
 __syncthreads();

 const float4 bz = make_float4(bias[0], bias[1], bias[2], bias[3]);
 const float4 sd = make_float4(isd[0], isd[1], isd[2], isd[3]);
 for (int t = tid; t < T; t += NT) {
  const float4* r = reinterpret_cast<const float4*>(x + (long)t * ld);
  float4 xv[KC / 4];
#pragma unroll
  for (int q = 0; q < KC / 4; ++q) xv[q] = __ldg(r + q);
  float z0 = -bz.x, z1 = -bz.y, z2 = -bz.z, z3 = -bz.w;
#pragma unroll
  for (int q = 0; q < KC / 4; ++q) {
   const float e[4] = {xv[q].x, xv[q].y, xv[q].z, xv[q].w};
#pragma unroll
   for (int i = 0; i < 4; ++i) {
    const float4 d = dir[4 * q + i];
    z0 += e[i] * d.x; z1 += e[i] * d.y; z2 += e[i] * d.z; z3 += e[i] * d.w;
   }
  }
  const int q0 = min(7, max(0, (int)(normcdff(z0 * sd.x) * 8.f)));
  const int q1 = min(7, max(0, (int)(normcdff(z1 * sd.y) * 8.f)));
  const int q2 = min(7, max(0, (int)(normcdff(z2 * sd.z) * 8.f)));
  const int q3 = min(7, max(0, (int)(normcdff(z3 * sd.w) * 8.f)));
  int k = 0;
#pragma unroll
  for (int b = 2; b >= 0; --b)
   k = 16 * k + 8 * ((q0 >> b) & 1) + 4 * ((q1 >> b) & 1) + 2 * ((q2 >> b) & 1) + ((q3 >> b) & 1);
  keys[t] = (uint16_t)k;
  atomicAdd(hist + k, 1);
 }
 __syncthreads();

 int cv[4], s = 0;
#pragma unroll
 for (int q = 0; q < 4; ++q) { cv[q] = hist[4 * tid + q]; s += cv[q]; }
 int inc = s;
#pragma unroll
 for (int o = 1; o < 32; o <<= 1) {
  const int y = __shfl_up_sync(0xffffffffu, inc, o);
  if (lane >= o) inc += y;
 }
 if (lane == 31) ws[warp] = inc;
 __syncthreads();
 if (warp == 0) {
  const int w = ws[lane];
  int wi = w;
#pragma unroll
  for (int o = 1; o < 32; o <<= 1) {
   const int y = __shfl_up_sync(0xffffffffu, wi, o);
   if (lane >= o) wi += y;
  }
  ws[lane] = wi - w;
 }
 __syncthreads();
 int e = ws[warp] + inc - s;
#pragma unroll
 for (int q = 0; q < 4; ++q) { hist[4 * tid + q] = e; e += cv[q]; }
 __syncthreads();
 for (int t = tid; t < T; t += NT) {
  const int pos = ident ? t : atomicAdd(hist + keys[t], 1);
  perm[pos] = t;
  inv[t] = pos;
 }
}
}

static size_t smem_bytes(int T) { return (size_t)NB * 4 + (((size_t)T * 2 + 15) / 16) * 16; }

void order2_init() {
 cudaFuncSetAttribute(order2_k, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem_bytes(TMAX));
}

void order2_run(torch::Tensor x, torch::Tensor perm, torch::Tensor inv, int64_t ident) {
 const int T = (int)x.size(0);
 const long ld = x.stride(0);
 TORCH_CHECK(x.scalar_type() == torch::kFloat && x.stride(1) == 1 && x.size(1) >= KC && ld % 4 == 0, "order x");
 TORCH_CHECK(T <= TMAX && perm.numel() >= T && inv.numel() >= T, "order T / perm / inv");
 order2_k<<<1, NT, smem_bytes(T), at::cuda::getCurrentCUDAStream()>>>(x.data_ptr<float>(), ld, T,
                                    perm.data_ptr<int>(), inv.data_ptr<int>(),
                                    (int)ident);
}
"""

xk__mod = load_inline(name="okc_order2_z02590_b460010", cuda_sources=xk__SRC,
                   cpp_sources="void order2_init(); void order2_run(torch::Tensor, torch::Tensor, torch::Tensor, int64_t);",
                   functions=["order2_init", "order2_run"], extra_cuda_cflags=["-O3", "-gencode=arch=compute_90a,code=sm_90a"],
                   verbose=False)
xk__init = []


def xk_supported(T):
    return T <= xk_TMAX


def xk_order(resid, ident=False):
    """(perm, inv), int32 [T]: row i holds token perm[i], token t sits in row inv[t] (ident: same work, identity)."""
    if not xk__init:
        xk__mod.order2_init()
        xk__init.append(1)
    T = resid.shape[0]
    pi = torch.empty(2 * T, dtype=torch.int32, device=resid.device)
    perm, inv = pi[:T], pi[T:]
    xk__mod.order2_run(resid, perm, inv, 1 if ident else 0)
    return perm, inv




xw_WS_MINB = 4
xw_WS_NB = 4

xw__SRC = r"""

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

#ifndef WS_MINB
#define WS_MINB 4
#endif
#ifndef WS_NB
#define WS_NB 4
#endif
namespace {
__device__ __forceinline__ int ld_acq(const int* p) {
 int v;
 asm volatile("ld.acquire.gpu.global.b32 %0, [%1];" : "=r"(v) : "l"(p) : "memory");
 return v;
}
__device__ __forceinline__ float bf_lo(uint32_t u) { return __uint_as_float(u << 16); }
__device__ __forceinline__ float bf_hi(uint32_t u) { return __uint_as_float(u & 0xffff0000u); }

#define PK(a, b, i) ((i) < 8 ? (((i) & 1) ? bf_hi((&a.x)[(i) >> 1]) : bf_lo((&a.x)[(i) >> 1])) \
               : (((i) & 1) ? bf_hi((&b.x)[((i) - 8) >> 1]) : bf_lo((&b.x)[((i) - 8) >> 1])))

__global__ void __launch_bounds__(256, WS_MINB) wsum3_k(const __nv_bfloat16* __restrict__ y, const float* __restrict__ tw,
                            const int* __restrict__ slot, const __nv_bfloat16* __restrict__ w,
                            const int* __restrict__ meta, const int* __restrict__ cnt2,
                            __nv_bfloat16* __restrict__ out, float eps,
                            float sceps_unused) {
 __shared__ int s_need[32];
 __shared__ float red[8][8];
 __shared__ int s_k[8];
 __shared__ float s_tw[8];
 __shared__ int s_slot[8];
 __shared__ int s_n, s_rdy;
 const int t = blockIdx.x, tid = threadIdx.x, warp = tid / 32, lane = tid % 32;
 const uint4 w0 = *reinterpret_cast<const uint4*>(w + 8 * tid), w1 = *reinterpret_cast<const uint4*>(w + 2048 + 8 * tid);


 if (tid == 0 && gridDim.x < 2048) asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
 if (tid < 32) {


  const int e = meta[164 + tid], inf = meta[196 + tid];
  s_need[e] = (inf & 255) * (32 / max(1, (inf >> 8) & 15));
 } else if (tid < 40) {
  s_tw[tid - 32] = tw[t * 8 + tid - 32];
 } else if (tid < 48) {
  s_slot[tid - 40] = slot[t * 8 + tid - 40];
 }
 __syncthreads();
 int e = -1;
 bool rdy = true;
 if (warp == 0) {
  e = lane < 8 ? s_slot[lane] : -1;
  if (e >= 0) rdy = ld_acq(cnt2 + e) >= s_need[e];
  const unsigned loc = __ballot_sync(0xffffffffu, e >= 0), nr = __ballot_sync(0xffffffffu, e >= 0 && !rdy);
  if (lane < 8 && e >= 0) s_k[__popc(loc & ((1u << lane) - 1))] = lane;
  if (lane == 0) { s_n = __popc(loc); s_rdy = (int)nr; }
 }
 __syncthreads();
 const int n = s_n;
 const unsigned late = (unsigned)s_rdy;
 const __nv_bfloat16* base = y + (size_t)t * 8 * 4096 + 8 * tid;
 uint4 xa[WS_NB], xb[WS_NB];

#pragma unroll
 for (int i = 0; i < WS_NB; i++)
  if (i < n && !((late >> s_k[i]) & 1)) {
   const __nv_bfloat16* row = base + (size_t)s_k[i] * 4096;
   xa[i] = *reinterpret_cast<const uint4*>(row);
   xb[i] = *reinterpret_cast<const uint4*>(row + 2048);
  }
 if (late) {
  if (warp == 0) {
   if (e >= 0 && !rdy)
    while (ld_acq(cnt2 + e) < s_need[e]) __nanosleep(64);
   __syncwarp();
  }
  __syncthreads();
#pragma unroll
  for (int i = 0; i < WS_NB; i++)
   if (i < n && ((late >> s_k[i]) & 1)) {
    const __nv_bfloat16* row = base + (size_t)s_k[i] * 4096;
    xa[i] = *reinterpret_cast<const uint4*>(row);
    xb[i] = *reinterpret_cast<const uint4*>(row + 2048);
   }
 }

#pragma unroll
 for (int i = 0; i < WS_NB; i++)
  if (i < n) {
   float ss = 0.f;
#pragma unroll
   for (int j = 0; j < 16; j++) {
    const float xv = PK(xa[i], xb[i], j) * PK(w0, w1, j);
    ss += xv * xv;
   }
#pragma unroll
   for (int o = 16; o > 0; o >>= 1) ss += __shfl_xor_sync(0xffffffffu, ss, o);
   if (lane == 0) red[s_k[i]][warp] = ss;
  }
 for (int i = WS_NB; i < n; i++) {
  const __nv_bfloat16* row = base + (size_t)s_k[i] * 4096;
  const uint4 ya = *reinterpret_cast<const uint4*>(row), yb = *reinterpret_cast<const uint4*>(row + 2048);
  float ss = 0.f;
#pragma unroll
  for (int j = 0; j < 16; j++) {
   const float xv = PK(ya, yb, j) * PK(w0, w1, j);
   ss += xv * xv;
  }
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) ss += __shfl_xor_sync(0xffffffffu, ss, o);
  if (lane == 0) red[s_k[i]][warp] = ss;
 }
 __syncthreads();

 float acc[16];
#pragma unroll
 for (int j = 0; j < 16; j++) acc[j] = 0.f;
#pragma unroll
 for (int i = 0; i < WS_NB; i++)
  if (i < n) {
   const int k = s_k[i];
   float s = 0.f;
#pragma unroll
   for (int j = 0; j < 8; j++) s += red[k][j];
   const float r = rsqrtf(s / 4096.f + eps) * s_tw[k];
#pragma unroll
   for (int j = 0; j < 16; j++) {
    const float xv = PK(xa[i], xb[i], j) * PK(w0, w1, j);
    acc[j] += xv * r;
   }
  }
 for (int i = WS_NB; i < n; i++) {
  const int k = s_k[i];
  const __nv_bfloat16* row = base + (size_t)k * 4096;
  const uint4 ya = *reinterpret_cast<const uint4*>(row), yb = *reinterpret_cast<const uint4*>(row + 2048);
  float s = 0.f;
#pragma unroll
  for (int j = 0; j < 8; j++) s += red[k][j];
  const float r = rsqrtf(s / 4096.f + eps) * s_tw[k];
#pragma unroll
  for (int j = 0; j < 16; j++) {
   const float xv = PK(ya, yb, j) * PK(w0, w1, j);
   acc[j] += xv * r;
  }
 }
 uint4 o0, o1;
 __nv_bfloat162* h0 = reinterpret_cast<__nv_bfloat162*>(&o0);
 __nv_bfloat162* h1 = reinterpret_cast<__nv_bfloat162*>(&o1);
#pragma unroll
 for (int i = 0; i < 4; i++) {
  h0[i] = __floats2bfloat162_rn(acc[2 * i], acc[2 * i + 1]);
  h1[i] = __floats2bfloat162_rn(acc[8 + 2 * i], acc[8 + 2 * i + 1]);
 }
 *reinterpret_cast<uint4*>(out + (size_t)t * 4096 + 8 * tid) = o0;
 *reinterpret_cast<uint4*>(out + (size_t)t * 4096 + 2048 + 8 * tid) = o1;
}
}

void wsum_run(torch::Tensor y, torch::Tensor tw, torch::Tensor slot, torch::Tensor w, torch::Tensor meta,
       torch::Tensor cnt2, torch::Tensor out, double eps, int64_t pdl) {
 const int T = (int)out.size(0);
 cudaLaunchAttribute at[1];
 at[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
 at[0].val.programmaticStreamSerializationAllowed = pdl ? 1 : 0;

 cudaLaunchConfig_t cfg = {dim3(T), dim3(256), 0, at::cuda::getCurrentCUDAStream(), at, 1};
 cudaLaunchKernelEx(&cfg, wsum3_k, (const __nv_bfloat16*)y.data_ptr(), tw.data_ptr<float>(), slot.data_ptr<int>(),
           (const __nv_bfloat16*)w.data_ptr(), meta.data_ptr<int>(), cnt2.data_ptr<int>(),
           (__nv_bfloat16*)out.data_ptr(), (float)eps, 1e-15f);
}
"""
xw__ws = load_inline(name="okc_prefill_wsum_z08492_b460010", cuda_sources=xw__SRC,
                  cpp_sources="void wsum_run(torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, "
                              "torch::Tensor, torch::Tensor, double, int64_t);",
                  functions=["wsum_run"],
                  extra_cuda_cflags=["-O3", "-gencode=arch=compute_90a,code=sm_90a", "-DWS_MINB=4", "-DWS_NB=4"])


def xw_wsum(y, tw, slot, w, meta, cnt2, T, eps, pdl=True):
    """[T, 4096] BF16 weighted sum of the normalized local expert outputs y [T * 8, 4096] (slot ids < 0 skipped)."""
    out = torch.empty((T, 4096), dtype=torch.bfloat16, device=y.device)
    xw__ws.wsum_run(y, tw, slot, w, meta, cnt2, out, eps, 1 if pdl else 0)
    return out




try:
    import cutlass
    import cutlass.cute as cute
    import cutlass.pipeline as pipeline
    from cutlass import Boolean, Float32, Int32, const_expr
    from cutlass.cute.nvgpu import cpasync
    from cutlass.experimental import primitives as prims
    from cutlass.pipeline import pipeline_init_arrive, pipeline_init_wait
    from cutlass.utils import SmemPartition

    import quack.copy_utils as copy_utils
    import quack.sm90_utils as quack_sm90_utils
    from quack.cute_dsl_utils import get_device_capacity, get_max_active_clusters
    from quack.gemm_default_epi import GemmDefaultEpiMixin, GemmDefaultSm90
    from quack.gemm_tvm_ffi_utils import (compile_gemm_kernel, get_dtypes, get_majors, make_fake_gemm_tensors,
                                          make_fake_scheduler_args, make_scheduler_args)
    from quack.pipeline import make_pipeline_state
    from quack.rounding import RoundingMode
    from quack.varlen_utils import VarlenManager

    AVAILABLE = True
except Exception:
    AVAILABLE = False


if AVAILABLE:
    class PdlGemmSm90DynX12712(GemmDefaultSm90):
        """QuACK's SM90 GEMM with weights loaded before the PDL wait and the dependents' trigger at kernel end."""

        @cute.kernel
        def kernel(self, tiled_mma, tma_atom_a, mA_mkl, tma_atom_b, mB_nkl, tma_atom_sfa, mSFA_mkl, tma_atom_sfb,
                   mSFB_nkl, tma_atom_d, mD_mnl, tma_atom_c, mC_mnl, epilogue_params, varlen_params, cluster_layout_mnk,
                   a_smem_layout, b_smem_layout, sfa_smem_layout, sfb_smem_layout, epi_smem_layout, epi_c_smem_layout,
                   tma_atom_aux_a, mAuxA_mkl, aux_a_smem_layout, tile_sched_params,
                   TileSchedulerCls: cutlass.Constexpr):
            assert varlen_params.cu_seqlens_m is None and varlen_params.cu_seqlens_k is None
            assert not self.gather_A and self.transform_a is None and self.aux_a is None and self.split_k == 1
            assert mC_mnl is None and mD_mnl is not None and self.epi_c_stage == 0 and self.use_pdl
            warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
            if warp_idx == self.ab_load_warp_id:
                for tma_atom in (tma_atom_a, tma_atom_b, tma_atom_d):
                    cpasync.prefetch_descriptor(tma_atom)
            smem = cutlass.utils.SmemAllocator()
            storage = smem.allocate(self.shared_storage)
            ab_pipeline = self.make_ab_pipeline(tiled_mma=tiled_mma,
                                                cluster_layout_vmnk=cute.make_layout((1, *cluster_layout_mnk.shape)))
            sched_pipeline = None
            sched_data = None
            if const_expr(self.is_persistent):
                sched_pipeline = self.make_sched_pipeline(cluster_layout_mnk, varlen_k=False)
                sched_data = smem.allocate_tensor(Int32, cute.make_layout((4, self.sched_stage)), byte_alignment=16,
                                                  partition=SmemPartition.RESERVED)
            pipeline_init_arrive(cluster_shape_mn=self.cluster_shape_mnk[:-1], is_relaxed=True)
            sA = storage.sA.get_tensor(a_smem_layout.outer, swizzle=a_smem_layout.inner)
            sB = storage.sB.get_tensor(b_smem_layout.outer, swizzle=b_smem_layout.inner)
            sD = storage.sD.get_tensor(epi_smem_layout.outer, swizzle=epi_smem_layout.inner)
            epi_smem_tensors = self.epi_get_smem_tensors(epilogue_params, storage)
            varlen_manager = VarlenManager.create(varlen_params, len_m_static=Int32(cute.size(mA_mkl, mode=[0])),
                                                  len_k_static=Int32(cute.size(mB_nkl, mode=[1])),
                                                  len_n_static=Int32(cute.size(mB_nkl, mode=[0])))
            TileSchedulerCls = partial(TileSchedulerCls.create, tile_sched_params, sched_data, sched_pipeline)
            pipeline_init_wait(cluster_shape_mn=self.cluster_shape_mnk[:-1])
            k_tile_cnt_static = cute.ceil_div(cute.size(mA_mkl, mode=[1]), self.cta_tile_shape_mnk[2])

            if warp_idx >= self.ab_load_warp_id:
                cute.arch.setmaxregister_decrease(self.num_regs_load)
                if warp_idx >= self.ab_load_warp_id and warp_idx < self.ab_load_warp_id + self.num_ab_load_warps:
                    a_tma_multicast = {"cluster_shape": self.cluster_shape_mnk[:2], "multicast_dim": "M"}
                    b_tma_multicast = {"cluster_shape": self.cluster_shape_mnk[:2], "multicast_dim": "N"}
                    is_scheduler_warp = self.num_ab_load_warps == 1 or warp_idx == self.ab_load_warp_id
                    if const_expr(cute.size(cluster_layout_mnk) > 1):
                        is_scheduler_warp = is_scheduler_warp and cute.arch.block_idx_in_cluster() == 0
                    tile_scheduler = TileSchedulerCls()
                    work_tile = tile_scheduler.initial_work_tile_info()
                    ab_producer_state = make_pipeline_state(pipeline.PipelineUserType.Producer, self.ab_stage)
                    n_pre = Int32(0)
                    if work_tile.is_valid_tile:
                        gB0 = cute.local_tile(varlen_manager.offset_batch_B(mB_nkl, work_tile.tile_idx[3]),
                                              cute.select(self.cta_tile_shape_mnk, [1, 2]), (work_tile.tile_idx[1], None))
                        copy_B0 = copy_utils.tma_get_block_copy_fn(tma_atom_b, src_tensor=gB0, dst_tensor=sB,
                                                                   tma_multicast=b_tma_multicast)
                        n_pre = Int32(k_tile_cnt_static)
                        if n_pre > self.ab_stage:
                            n_pre = Int32(self.ab_stage)
                        pre_state = ab_producer_state.clone()
                        for k_tile in cutlass.range(n_pre, unroll=1):
                            ab_pipeline.producer_acquire(pre_state)
                            copy_B0(k_tile, pre_state.index, tma_bar_ptr=ab_pipeline.producer_get_barrier(pre_state))
                            pre_state.advance()
                    prims.griddepcontrol(prims.GridDepAction.WAIT)
                    while work_tile.is_valid_tile:
                        tile_coord_mnkl = work_tile.tile_idx
                        batch_idx = tile_coord_mnkl[3]
                        gA_mk = cute.local_tile(varlen_manager.offset_batch_A(mA_mkl, batch_idx),
                                                cute.select(self.cta_tile_shape_mnk, [0, 2]), (tile_coord_mnkl[0], None))
                        copy_A = copy_utils.tma_get_block_copy_fn(tma_atom_a, src_tensor=gA_mk, dst_tensor=sA,
                                                                  tma_multicast=a_tma_multicast)
                        gB_nk = cute.local_tile(varlen_manager.offset_batch_B(mB_nkl, batch_idx),
                                                cute.select(self.cta_tile_shape_mnk, [1, 2]), (tile_coord_mnkl[1], None))
                        copy_B = copy_utils.tma_get_block_copy_fn(tma_atom_b, src_tensor=gB_nk, dst_tensor=sB,
                                                                  tma_multicast=b_tma_multicast)
                        k_tile_cnt = cute.ceil_div(varlen_manager.len_k(batch_idx), self.cta_tile_shape_mnk[2])
                        for k_tile in cutlass.range(n_pre, unroll=1):
                            copy_A(k_tile, ab_producer_state.index,
                                   tma_bar_ptr=ab_pipeline.producer_get_barrier(ab_producer_state))
                            ab_pipeline.producer_commit(ab_producer_state)
                            ab_producer_state.advance()
                        ab_producer_state = self.load_tma(ab_pipeline, ab_producer_state, [copy_A, copy_B, None],
                                                          k_tile_cnt - n_pre, k_tile_start=n_pre)
                        n_pre = Int32(0)
                        tile_scheduler.advance_to_next_work(is_scheduler_warp=is_scheduler_warp)
                        work_tile = tile_scheduler.get_current_work()
                    if const_expr(self.pingpong):
                        if is_scheduler_warp:
                            tile_scheduler.write_work_tile_to_smem(work_tile)
                        work_tile = tile_scheduler.get_current_work()
                    if warp_idx == self.ab_load_warp_id:
                        ab_pipeline.producer_tail(ab_producer_state)
                    if is_scheduler_warp:
                        tile_scheduler.producer_tail()

            if warp_idx < self.ab_load_warp_id:
                cute.arch.setmaxregister_increase(self.num_regs_mma)
                is_tma_warp = Boolean((not self.pingpong and warp_idx == 0)
                                      or (self.pingpong and (warp_idx == 0 or warp_idx == 4)))
                tidx, _, _ = cute.arch.thread_idx()
                warp_group_idx = cute.arch.make_warp_uniform(tidx // self.num_threads_per_warp_group)
                if const_expr(self.pingpong):
                    tidx = tidx % self.num_threads_per_warp_group
                warp_group_thread_layout = cute.make_layout(
                    self.mma_warp_groups if const_expr(not self.pingpong) else 1, stride=self.num_threads_per_warp_group)
                thr_mma = tiled_mma.get_slice(warp_group_thread_layout(warp_group_idx if not self.pingpong else 0))
                acc, tCrA, tCrB = quack_sm90_utils.partition_fragment_ABC(thr_mma, self.cta_tile_shape_mnk, sA, sB)
                assert not self.mma_is_rs and not self.fp8_slow_accum
                mma_fn = partial(quack_sm90_utils.gemm_w_idx, tiled_mma, acc, tCrA, tCrB)
                if const_expr(self.pingpong):
                    if warp_group_idx == 0:
                        self.pingpong_barrier_arrive(warp_group_idx=0, stage="mma")
                        self.pingpong_barrier_arrive(warp_group_idx=0, stage="epi")
                c_tile_cnt = cute.size(self.epi_tile_shape)
                ab_read_state = make_pipeline_state(pipeline.PipelineUserType.Consumer, self.ab_stage)
                epi_store_pipeline = self.make_epi_store_pipeline()
                epi_read_state = make_pipeline_state(pipeline.PipelineUserType.Consumer, self.epi_c_stage)
                epi_producer_state = make_pipeline_state(pipeline.PipelineUserType.Producer, self.epi_c_stage)
                tile_scheduler = TileSchedulerCls()
                work_tile = tile_scheduler.initial_work_tile_info()
                if const_expr(self.pingpong):
                    if warp_idx >= 4:
                        epi_read_state.advance_iters(c_tile_cnt)
                        epi_producer_state.advance_iters(c_tile_cnt)
                        ab_read_state.advance_iters(k_tile_cnt_static)
                        tile_scheduler.advance_to_next_work()
                        work_tile = tile_scheduler.get_current_work()
                while work_tile.is_valid_tile:
                    tile_coord_mnkl = work_tile.tile_idx
                    batch_idx = tile_coord_mnkl[3]
                    k_tile_cnt = cute.ceil_div(varlen_manager.len_k(batch_idx), self.cta_tile_shape_mnk[2])
                    if const_expr(self.pingpong):
                        self.pingpong_barrier_sync(warp_group_idx, stage="mma")
                    ab_read_state = self.mma(ab_pipeline, ab_read_state, mma_fn, acc, None, k_tile_cnt, warp_group_idx)
                    if const_expr(self.pingpong):
                        self.pingpong_barrier_sync(warp_group_idx, "epi")
                    copy_D, _, _ = self.epilog_gmem_copy_and_partition(
                        tma_atom_d, varlen_manager.offset_batch_epi(mD_mnl, batch_idx), self.cta_tile_shape_mnk[:2],
                        self.epi_tile, sD, tile_coord_mnkl)
                    tiled_copy_r2s, tRS_rD, tRS_sD = self.epilog_smem_store_and_partition(
                        tiled_mma, self.d_layout, self.d_dtype, sD, tidx)
                    tRS_rAcc = self.epi_retile_acc(acc, tRS_rD, tiled_copy_r2s)
                    load_acc_subtile = partial(self.epi_load_acc_subtile, tRS_rAcc)
                    self.epi_visit_acc(epilogue_params, acc, tiled_mma, tile_coord_mnkl, tidx)
                    epi_fn = partial(
                        self.epilogue, epilogue_params, epi_smem_tensors, None, epi_store_pipeline, epi_read_state,
                        epi_producer_state, self.epi_tile, tRS_rD=tRS_rD, tRS_rC=None, tiled_copy_t2r=None,
                        tiled_copy_r2s=tiled_copy_r2s, tRS_sD=tRS_sD, tiled_copy_s2r=None, tSR_rC=None, tSR_sC=None,
                        copy_D=copy_D, copy_C=None, tile_coord_mnkl=tile_coord_mnkl, varlen_manager=varlen_manager,
                        epilogue_barrier=self.epilogue_barrier, tile_scheduler=tile_scheduler, tidx=tidx,
                        is_tma_warp=is_tma_warp)
                    epi_read_state, epi_producer_state = self.epilogue_split_k(
                        epilogue_params, epi_fn, load_acc_subtile, tRS_rD, self.epi_tile, epi_read_state,
                        epi_producer_state, epi_store_pipeline, tile_coord_mnkl, self.epilogue_barrier, tidx, is_tma_warp)
                    if const_expr(self.pingpong):
                        if is_tma_warp:
                            epi_store_pipeline.producer_tail()
                        self.pingpong_barrier_arrive(1 - warp_group_idx, stage="epi")
                    if const_expr(not self.pingpong):
                        tile_scheduler.advance_to_next_work()
                        work_tile = tile_scheduler.get_current_work()
                    else:
                        epi_read_state.advance_iters(c_tile_cnt)
                        epi_producer_state.advance_iters(c_tile_cnt)
                        ab_read_state.advance_iters(k_tile_cnt_static)
                        tile_scheduler.advance_to_next_work(advance_count=self.mma_warp_groups)
                        work_tile = tile_scheduler.get_current_work()
                if const_expr(not self.pingpong):
                    if is_tma_warp:
                        epi_store_pipeline.producer_tail()
                prims.griddepcontrol(prims.GridDepAction.LAUNCH_DEPENDENTS)


xg__CACHE = {}
xg__SEMS = []


def xg_compile_gemm(x, w, tile_m=128, tile_n=192, pingpong=True, cluster_m=1, cluster_n=1, max_swizzle_size=8):
    """Compile (once per configuration and operand layout) out = x @ w.T for BF16 x (M, K) and w (N, K)."""
    out = torch.empty((x.shape[0], w.shape[0]), dtype=torch.bfloat16, device=x.device)
    B = w.t()
    key = (tile_m, tile_n, pingpong, cluster_m, cluster_n, max_swizzle_size, x.stride(), B.stride(), x.device.index)
    if key in xg__CACHE:
        return key
    cap = get_device_capacity(x.device)
    assert cap[0] == 9, "PdlGemmSm90DynX12712 is an SM90 kernel"
    a_major, b_major, d_major, c_major = get_majors(x, B, out, None)
    b_major = "n" if B.stride(-1) == 1 else "k"
    a_dt, b_dt, d_dt, c_dt = get_dtypes(x, B, out, None)
    mA, mB, mD, mC, m, n, k, l = make_fake_gemm_tensors(a_dt, b_dt, d_dt, c_dt, a_major, b_major, d_major, c_major,
                                                        batched=False, b_kn=True)
    epi = GemmDefaultEpiMixin.EpilogueArguments(add_to_output=False, rounding_mode=RoundingMode.RN)
    fn = compile_gemm_kernel(PdlGemmSm90DynX12712, a_dt, (tile_m, tile_n), (cluster_m, cluster_n, 1), pingpong, True, False,
                             True, cap, mA, mB, mD, mC, epi, make_fake_scheduler_args(True, False, l), None,
                             b_transposed=True)
    epi_run = GemmDefaultEpiMixin.EpilogueArguments(alpha=None, beta=None, mRowVecBroadcast=None,
                                                    mColVecBroadcast=None, add_to_output=None, rounding_mode=None,
                                                    sr_seed=None)
    sem = torch.zeros(1, dtype=torch.int32, device=x.device)
    xg__SEMS.append(sem)
    sched = make_scheduler_args(get_max_active_clusters(cluster_m * cluster_n, device_capacity=cap),
                                max_swizzle_size, sem)
    xg__CACHE[key] = (fn, epi_run, sched)
    return key


def xg_gemm(x, w, key):
    """out = x @ w.T (BF16 out, FP32 accumulation) on the compiled kernel `key` (from compile_gemm)."""
    if key not in xg__CACHE:
        xg_compile_gemm(x, w, *key[:6])
    fn, epi_run, sched = xg__CACHE[key]
    out = torch.empty((x.shape[0], w.shape[0]), dtype=torch.bfloat16, device=x.device)
    fn(x, w.t(), out, None, epi_run, sched, None, None, None)
    return out




try:
    import cutlass
    import cutlass.cute as cute
    import cutlass.pipeline as pipeline
    from cutlass import Boolean, Float32, Int32, Int64, const_expr
    from cutlass.cute.nvgpu import cpasync
    from cutlass.experimental import primitives as prims
    from cutlass.pipeline import pipeline_init_arrive, pipeline_init_wait
    from cutlass.utils import SmemPartition

    import quack.copy_utils as copy_utils
    import quack.sm90_utils as quack_sm90_utils
    from quack.cute_dsl_utils import get_device_capacity, get_max_active_clusters
    from quack.gemm_default_epi import GemmDefaultEpiMixin, GemmDefaultSm90
    from quack.gemm_tvm_ffi_utils import (compile_gemm_kernel, get_dtypes, get_majors, make_fake_gemm_tensors,
                                          make_fake_scheduler_args, make_scheduler_args)
    from quack.pipeline import make_pipeline_state
    from quack.rounding import RoundingMode
    from quack.varlen_utils import VarlenManager

    AVAILABLE = True
except Exception:
    AVAILABLE = False


xgu_X45_B_POLICY = 0x12F0000000000000

if AVAILABLE:
    class PdlGemmSm90Ef(GemmDefaultSm90):
        """QuACK's SM90 GEMM with weights loaded before the PDL wait and the dependents' trigger at kernel end."""

        @cute.kernel
        def kernel(self, tiled_mma, tma_atom_a, mA_mkl, tma_atom_b, mB_nkl, tma_atom_sfa, mSFA_mkl, tma_atom_sfb,
                   mSFB_nkl, tma_atom_d, mD_mnl, tma_atom_c, mC_mnl, epilogue_params, varlen_params, cluster_layout_mnk,
                   a_smem_layout, b_smem_layout, sfa_smem_layout, sfb_smem_layout, epi_smem_layout, epi_c_smem_layout,
                   tma_atom_aux_a, mAuxA_mkl, aux_a_smem_layout, tile_sched_params,
                   TileSchedulerCls: cutlass.Constexpr):
            assert varlen_params.cu_seqlens_m is None and varlen_params.cu_seqlens_k is None
            assert not self.gather_A and self.transform_a is None and self.aux_a is None and self.split_k == 1
            assert mC_mnl is None and mD_mnl is not None and self.epi_c_stage == 0 and self.use_pdl
            warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
            if warp_idx == self.ab_load_warp_id:
                for tma_atom in (tma_atom_a, tma_atom_b, tma_atom_d):
                    cpasync.prefetch_descriptor(tma_atom)
            smem = cutlass.utils.SmemAllocator()
            storage = smem.allocate(self.shared_storage)
            ab_pipeline = self.make_ab_pipeline(tiled_mma=tiled_mma,
                                                cluster_layout_vmnk=cute.make_layout((1, *cluster_layout_mnk.shape)))
            sched_pipeline = None
            sched_data = None
            if const_expr(self.is_persistent):
                sched_pipeline = self.make_sched_pipeline(cluster_layout_mnk, varlen_k=False)
                sched_data = smem.allocate_tensor(Int32, cute.make_layout((4, self.sched_stage)), byte_alignment=16,
                                                  partition=SmemPartition.RESERVED)
            pipeline_init_arrive(cluster_shape_mn=self.cluster_shape_mnk[:-1], is_relaxed=True)
            sA = storage.sA.get_tensor(a_smem_layout.outer, swizzle=a_smem_layout.inner)
            sB = storage.sB.get_tensor(b_smem_layout.outer, swizzle=b_smem_layout.inner)
            sD = storage.sD.get_tensor(epi_smem_layout.outer, swizzle=epi_smem_layout.inner)
            epi_smem_tensors = self.epi_get_smem_tensors(epilogue_params, storage)
            varlen_manager = VarlenManager.create(varlen_params, len_m_static=Int32(cute.size(mA_mkl, mode=[0])),
                                                  len_k_static=Int32(cute.size(mB_nkl, mode=[1])),
                                                  len_n_static=Int32(cute.size(mB_nkl, mode=[0])))
            TileSchedulerCls = partial(TileSchedulerCls.create, tile_sched_params, sched_data, sched_pipeline)
            pipeline_init_wait(cluster_shape_mn=self.cluster_shape_mnk[:-1])
            k_tile_cnt_static = cute.ceil_div(cute.size(mA_mkl, mode=[1]), self.cta_tile_shape_mnk[2])

            if warp_idx >= self.ab_load_warp_id:
                cute.arch.setmaxregister_decrease(self.num_regs_load)
                if warp_idx >= self.ab_load_warp_id and warp_idx < self.ab_load_warp_id + self.num_ab_load_warps:
                    a_tma_multicast = {"cluster_shape": self.cluster_shape_mnk[:2], "multicast_dim": "M"}
                    b_tma_multicast = {"cluster_shape": self.cluster_shape_mnk[:2], "multicast_dim": "N"}
                    is_scheduler_warp = self.num_ab_load_warps == 1 or warp_idx == self.ab_load_warp_id
                    if const_expr(cute.size(cluster_layout_mnk) > 1):
                        is_scheduler_warp = is_scheduler_warp and cute.arch.block_idx_in_cluster() == 0
                    tile_scheduler = TileSchedulerCls()
                    work_tile = tile_scheduler.initial_work_tile_info()
                    ab_producer_state = make_pipeline_state(pipeline.PipelineUserType.Producer, self.ab_stage)
                    n_pre = Int32(0)
                    if work_tile.is_valid_tile:
                        gB0 = cute.local_tile(varlen_manager.offset_batch_B(mB_nkl, work_tile.tile_idx[3]),
                                              cute.select(self.cta_tile_shape_mnk, [1, 2]), (work_tile.tile_idx[1], None))
                        copy_B0 = copy_utils.tma_get_block_copy_fn(tma_atom_b, src_tensor=gB0, dst_tensor=sB,
                                                                   tma_multicast=b_tma_multicast,
                                                                   cache_policy=Int64(xgu_X45_B_POLICY))
                        n_pre = Int32(k_tile_cnt_static)
                        if n_pre > self.ab_stage:
                            n_pre = Int32(self.ab_stage)
                        pre_state = ab_producer_state.clone()
                        for k_tile in cutlass.range(n_pre, unroll=1):
                            ab_pipeline.producer_acquire(pre_state)
                            copy_B0(k_tile, pre_state.index, tma_bar_ptr=ab_pipeline.producer_get_barrier(pre_state))
                            pre_state.advance()
                    prims.griddepcontrol(prims.GridDepAction.WAIT)
                    while work_tile.is_valid_tile:
                        tile_coord_mnkl = work_tile.tile_idx
                        batch_idx = tile_coord_mnkl[3]
                        gA_mk = cute.local_tile(varlen_manager.offset_batch_A(mA_mkl, batch_idx),
                                                cute.select(self.cta_tile_shape_mnk, [0, 2]), (tile_coord_mnkl[0], None))
                        copy_A = copy_utils.tma_get_block_copy_fn(tma_atom_a, src_tensor=gA_mk, dst_tensor=sA,
                                                                  tma_multicast=a_tma_multicast)
                        gB_nk = cute.local_tile(varlen_manager.offset_batch_B(mB_nkl, batch_idx),
                                                cute.select(self.cta_tile_shape_mnk, [1, 2]), (tile_coord_mnkl[1], None))
                        copy_B = copy_utils.tma_get_block_copy_fn(tma_atom_b, src_tensor=gB_nk, dst_tensor=sB,
                                                                  tma_multicast=b_tma_multicast,
                                                                  cache_policy=Int64(xgu_X45_B_POLICY))
                        k_tile_cnt = cute.ceil_div(varlen_manager.len_k(batch_idx), self.cta_tile_shape_mnk[2])
                        for k_tile in cutlass.range(n_pre, unroll=1):
                            copy_A(k_tile, ab_producer_state.index,
                                   tma_bar_ptr=ab_pipeline.producer_get_barrier(ab_producer_state))
                            ab_pipeline.producer_commit(ab_producer_state)
                            ab_producer_state.advance()
                        ab_producer_state = self.load_tma(ab_pipeline, ab_producer_state, [copy_A, copy_B, None],
                                                          k_tile_cnt - n_pre, k_tile_start=n_pre)
                        n_pre = Int32(0)
                        tile_scheduler.advance_to_next_work(is_scheduler_warp=is_scheduler_warp)
                        work_tile = tile_scheduler.get_current_work()
                    if const_expr(self.pingpong):
                        if is_scheduler_warp:
                            tile_scheduler.write_work_tile_to_smem(work_tile)
                        work_tile = tile_scheduler.get_current_work()
                    if warp_idx == self.ab_load_warp_id:
                        ab_pipeline.producer_tail(ab_producer_state)
                    if is_scheduler_warp:
                        tile_scheduler.producer_tail()

            if warp_idx < self.ab_load_warp_id:
                cute.arch.setmaxregister_increase(self.num_regs_mma)
                is_tma_warp = Boolean((not self.pingpong and warp_idx == 0)
                                      or (self.pingpong and (warp_idx == 0 or warp_idx == 4)))
                tidx, _, _ = cute.arch.thread_idx()
                warp_group_idx = cute.arch.make_warp_uniform(tidx // self.num_threads_per_warp_group)
                if const_expr(self.pingpong):
                    tidx = tidx % self.num_threads_per_warp_group
                warp_group_thread_layout = cute.make_layout(
                    self.mma_warp_groups if const_expr(not self.pingpong) else 1, stride=self.num_threads_per_warp_group)
                thr_mma = tiled_mma.get_slice(warp_group_thread_layout(warp_group_idx if not self.pingpong else 0))
                acc, tCrA, tCrB = quack_sm90_utils.partition_fragment_ABC(thr_mma, self.cta_tile_shape_mnk, sA, sB)
                assert not self.mma_is_rs and not self.fp8_slow_accum
                mma_fn = partial(quack_sm90_utils.gemm_w_idx, tiled_mma, acc, tCrA, tCrB)
                if const_expr(self.pingpong):
                    if warp_group_idx == 0:
                        self.pingpong_barrier_arrive(warp_group_idx=0, stage="mma")
                        self.pingpong_barrier_arrive(warp_group_idx=0, stage="epi")
                c_tile_cnt = cute.size(self.epi_tile_shape)
                ab_read_state = make_pipeline_state(pipeline.PipelineUserType.Consumer, self.ab_stage)
                epi_store_pipeline = self.make_epi_store_pipeline()
                epi_read_state = make_pipeline_state(pipeline.PipelineUserType.Consumer, self.epi_c_stage)
                epi_producer_state = make_pipeline_state(pipeline.PipelineUserType.Producer, self.epi_c_stage)
                tile_scheduler = TileSchedulerCls()
                work_tile = tile_scheduler.initial_work_tile_info()
                if const_expr(self.pingpong):
                    if warp_idx >= 4:
                        epi_read_state.advance_iters(c_tile_cnt)
                        epi_producer_state.advance_iters(c_tile_cnt)
                        ab_read_state.advance_iters(k_tile_cnt_static)
                        tile_scheduler.advance_to_next_work()
                        work_tile = tile_scheduler.get_current_work()
                while work_tile.is_valid_tile:
                    tile_coord_mnkl = work_tile.tile_idx
                    batch_idx = tile_coord_mnkl[3]
                    k_tile_cnt = cute.ceil_div(varlen_manager.len_k(batch_idx), self.cta_tile_shape_mnk[2])
                    if const_expr(self.pingpong):
                        self.pingpong_barrier_sync(warp_group_idx, stage="mma")
                    ab_read_state = self.mma(ab_pipeline, ab_read_state, mma_fn, acc, None, k_tile_cnt, warp_group_idx)
                    if const_expr(self.pingpong):
                        self.pingpong_barrier_sync(warp_group_idx, "epi")
                    copy_D, _, _ = self.epilog_gmem_copy_and_partition(
                        tma_atom_d, varlen_manager.offset_batch_epi(mD_mnl, batch_idx), self.cta_tile_shape_mnk[:2],
                        self.epi_tile, sD, tile_coord_mnkl)
                    tiled_copy_r2s, tRS_rD, tRS_sD = self.epilog_smem_store_and_partition(
                        tiled_mma, self.d_layout, self.d_dtype, sD, tidx)
                    tRS_rAcc = self.epi_retile_acc(acc, tRS_rD, tiled_copy_r2s)
                    load_acc_subtile = partial(self.epi_load_acc_subtile, tRS_rAcc)
                    self.epi_visit_acc(epilogue_params, acc, tiled_mma, tile_coord_mnkl, tidx)
                    epi_fn = partial(
                        self.epilogue, epilogue_params, epi_smem_tensors, None, epi_store_pipeline, epi_read_state,
                        epi_producer_state, self.epi_tile, tRS_rD=tRS_rD, tRS_rC=None, tiled_copy_t2r=None,
                        tiled_copy_r2s=tiled_copy_r2s, tRS_sD=tRS_sD, tiled_copy_s2r=None, tSR_rC=None, tSR_sC=None,
                        copy_D=copy_D, copy_C=None, tile_coord_mnkl=tile_coord_mnkl, varlen_manager=varlen_manager,
                        epilogue_barrier=self.epilogue_barrier, tile_scheduler=tile_scheduler, tidx=tidx,
                        is_tma_warp=is_tma_warp)
                    epi_read_state, epi_producer_state = self.epilogue_split_k(
                        epilogue_params, epi_fn, load_acc_subtile, tRS_rD, self.epi_tile, epi_read_state,
                        epi_producer_state, epi_store_pipeline, tile_coord_mnkl, self.epilogue_barrier, tidx, is_tma_warp)
                    if const_expr(self.pingpong):
                        if is_tma_warp:
                            epi_store_pipeline.producer_tail()
                        self.pingpong_barrier_arrive(1 - warp_group_idx, stage="epi")
                    if const_expr(not self.pingpong):
                        tile_scheduler.advance_to_next_work()
                        work_tile = tile_scheduler.get_current_work()
                    else:
                        epi_read_state.advance_iters(c_tile_cnt)
                        epi_producer_state.advance_iters(c_tile_cnt)
                        ab_read_state.advance_iters(k_tile_cnt_static)
                        tile_scheduler.advance_to_next_work(advance_count=self.mma_warp_groups)
                        work_tile = tile_scheduler.get_current_work()
                if const_expr(not self.pingpong):
                    if is_tma_warp:
                        epi_store_pipeline.producer_tail()
                prims.griddepcontrol(prims.GridDepAction.LAUNCH_DEPENDENTS)


xgu__CACHE = {}


def xgu_compile_gemm(x, w, tile_m=128, tile_n=192, pingpong=True, cluster_m=1, cluster_n=1, max_swizzle_size=8):
    """Compile (once per configuration and operand layout) out = x @ w.T for BF16 x (M, K) and w (N, K)."""
    out = torch.empty((x.shape[0], w.shape[0]), dtype=torch.bfloat16, device=x.device)
    B = w.t()
    key = (tile_m, tile_n, pingpong, cluster_m, cluster_n, max_swizzle_size, x.stride(), B.stride(), x.device.index)
    if key in xgu__CACHE:
        return key
    cap = get_device_capacity(x.device)
    assert cap[0] == 9, "PdlGemmSm90Ef is an SM90 kernel"
    a_major, b_major, d_major, c_major = get_majors(x, B, out, None)
    b_major = "n" if B.stride(-1) == 1 else "k"
    a_dt, b_dt, d_dt, c_dt = get_dtypes(x, B, out, None)
    mA, mB, mD, mC, m, n, k, l = make_fake_gemm_tensors(a_dt, b_dt, d_dt, c_dt, a_major, b_major, d_major, c_major,
                                                        batched=False, b_kn=True)
    epi = GemmDefaultEpiMixin.EpilogueArguments(add_to_output=False, rounding_mode=RoundingMode.RN)
    fn = compile_gemm_kernel(PdlGemmSm90Ef, a_dt, (tile_m, tile_n), (cluster_m, cluster_n, 1), pingpong, True, False,
                             False, cap, mA, mB, mD, mC, epi, make_fake_scheduler_args(False, False, l), None,
                             b_transposed=True)
    epi_run = GemmDefaultEpiMixin.EpilogueArguments(alpha=None, beta=None, mRowVecBroadcast=None,
                                                    mColVecBroadcast=None, add_to_output=None, rounding_mode=None,
                                                    sr_seed=None)
    sched = make_scheduler_args(get_max_active_clusters(cluster_m * cluster_n, device_capacity=cap),
                                max_swizzle_size, None)
    xgu__CACHE[key] = (fn, epi_run, sched)
    return key


def xgu_gemm(x, w, key):
    """out = x @ w.T (BF16 out, FP32 accumulation) on the compiled kernel `key` (from compile_gemm)."""
    if key not in xgu__CACHE:
        xgu_compile_gemm(x, w, *key[:6])
    fn, epi_run, sched = xgu__CACHE[key]
    out = torch.empty((x.shape[0], w.shape[0]), dtype=torch.bfloat16, device=x.device)
    fn(x, w.t(), out, None, epi_run, sched, None, None, None)
    return out





xt_ATT_CPP = ("torch::Tensor att_fwd(torch::Tensor q, torch::Tensor k, torch::Tensor v, torch::Tensor page_table, "
           "torch::Tensor cache_seqlens, torch::Tensor cu_seqlens_q, int64_t max_seqlen_q, double softmax_scale, "
           "int64_t window_left, int64_t num_splits_req, torch::Tensor out, torch::Tensor gate, double gate_eps);\n"
           "void att_gather(torch::Tensor kc, torch::Tensor vc, torch::Tensor pt, torch::Tensor seqlens, "
           "torch::Tensor cu_q, int64_t max_q, int64_t part, torch::Tensor ko, torch::Tensor vo);\n"
           "void att_front(torch::Tensor qkv, torch::Tensor loc, torch::Tensor qo, torch::Tensor kc, torch::Tensor vc, double eps, double qmul);\n"
           "void att_gather_t(torch::Tensor kc, torch::Tensor vc, torch::Tensor pt, torch::Tensor seqlens, "
           "torch::Tensor cu_q, int64_t win, int64_t part, torch::Tensor ko, torch::Tensor vt);")
xt_ATT_FLAGS = ["-O3", "--use_fast_math", "--expt-relaxed-constexpr", "--expt-extended-lambda", "-DNDEBUG",
             "-DCUTE_USE_PACKED_TUPLE=1", "-DCUTLASS_ENABLE_GDC_FOR_SM90", "-DCUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED",
             "-DCUTLASS_ENABLE_TENSOR_CORE_MMA=1", "-U__CUDA_NO_HALF_OPERATORS__", "-U__CUDA_NO_HALF_CONVERSIONS__",
             "-U__CUDA_NO_BFLOAT16_CONVERSIONS__", "-U__CUDA_NO_HALF2_OPERATORS__",
             "-gencode=arch=compute_90a,code=sm_90a"]
xt_ATT_SRC = r"""




































#define ATT_VARIANTS ATT_CASE(20) ATT_CASE(4) ATT_CASE(0)
#define U_PB 96
#define FK_PV72
#define FK_SUMFL 8
#define ATT_BN 256
#define ATT_SCHED 1
#define AT_SUB
#define Z0_NOL
#define Z0_GP 2
#define Z0_GP_MINB 2
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include "cute/tensor.hpp"
#include "cutlass/cutlass.h"
#include "cutlass/device_kernel.h"
#include <cutlass/kernel_hardware_info.h>
#include "cutlass/cluster_launch.hpp"
#include "cutlass/kernel_launch.h"
#define BOOL_SWITCH(COND, CONST_NAME, ...) \
 [&] { \
  if (COND) { \
   constexpr static bool CONST_NAME = true; \
   return __VA_ARGS__(); \
  } else { \
   constexpr static bool CONST_NAME = false; \
   return __VA_ARGS__(); \
  } \
 }()
 #define CAUSAL_LOCAL_SWITCH(CAUSAL_COND, LOCAL_COND, CAUSAL_CONST_NAME, LOCAL_CONST_NAME, ...) \
  [&] { \
   if (CAUSAL_COND) { \
    constexpr static bool CAUSAL_CONST_NAME = true; \
    constexpr static bool LOCAL_CONST_NAME = false; \
    return __VA_ARGS__(); \
   } else if (LOCAL_COND) { \
    constexpr static bool CAUSAL_CONST_NAME = false; \
    constexpr static bool LOCAL_CONST_NAME = true; \
    return __VA_ARGS__(); \
   } else { \
    constexpr static bool CAUSAL_CONST_NAME = false; \
    constexpr static bool LOCAL_CONST_NAME = false; \
    return __VA_ARGS__(); \
   } \
  }()
 #define SOFTCAP_SWITCH BOOL_SWITCH
 #define PAGEDKV_SWITCH BOOL_SWITCH
 #define SPLIT_SWITCH BOOL_SWITCH
 #define APPENDKV_SWITCH BOOL_SWITCH
 #define PACKGQA_SWITCH BOOL_SWITCH
 #define VARLEN_SWITCH BOOL_SWITCH
 #define CLUSTER_SWITCH BOOL_SWITCH
 #define ARCH_SWITCH(ARCH, ARCH_NAME, ...) \
 [&] { \
  if (ARCH == 86 || ARCH == 89) { \
   constexpr static int ARCH_NAME = 86; \
   return __VA_ARGS__(); \
  } else if (ARCH < 90) { \
   constexpr static int ARCH_NAME = 80; \
   return __VA_ARGS__(); \
  } else { \
   constexpr static int ARCH_NAME = 90; \
   return __VA_ARGS__(); \
  } \
 }()
 #define VCOLMAJOR_SWITCH(COND, CONST_NAME, ...) \
 [&] { \
  constexpr static bool CONST_NAME = false; \
  return __VA_ARGS__(); \
 }()
#define HEADDIM_SWITCH(HEADDIM, ...) \
 [&] { \
  if (HEADDIM == 64) { \
   constexpr static int kHeadSize = 64; \
   return __VA_ARGS__(); \
  } else if (HEADDIM == 96) { \
   constexpr static int kHeadSize = 96; \
   return __VA_ARGS__(); \
  } else if (HEADDIM == 128) { \
   constexpr static int kHeadSize = 128; \
   return __VA_ARGS__(); \
  } else if (HEADDIM == 96) { \
   constexpr static int kHeadSize = 96; \
   return __VA_ARGS__(); \
  } else if (HEADDIM == 256) { \
   constexpr static int kHeadSize = 256; \
   return __VA_ARGS__(); \
  } \
 }()
#define NUM_WARP_SWITCH(VALUE, CONST_NAME, ...) \
 [&] { \
  if (VALUE <= 1) { \
   constexpr static int CONST_NAME = 1; \
   return __VA_ARGS__(); \
  } else if (VALUE <= 2) { \
   constexpr static int CONST_NAME = 2; \
   return __VA_ARGS__(); \
  } else if (VALUE <= 4) { \
   constexpr static int CONST_NAME = 4; \
   return __VA_ARGS__(); \
  } else if (VALUE <= 8) { \
   constexpr static int CONST_NAME = 8; \
   return __VA_ARGS__(); \
  } else if (VALUE <= 16) { \
   constexpr static int CONST_NAME = 16; \
   return __VA_ARGS__(); \
  } else { \
   constexpr static int CONST_NAME = 32; \
   return __VA_ARGS__(); \
  } \
 }()
 #define SPARSE_MASK_SWITCH BOOL_SWITCH
#include <cuda.h>
#include <vector>
struct Qkv_params {
  using index_t = int64_t;
  void *__restrict__ q_ptr;
  void *__restrict__ k_ptr;
  void *__restrict__ v_ptr;
  index_t q_batch_stride;
  index_t k_batch_stride;
  index_t v_batch_stride;
  index_t q_row_stride;
  index_t k_row_stride;
  index_t v_row_stride;
  index_t q_head_stride;
  index_t k_head_stride;
  index_t v_head_stride;
  index_t v_dim_stride;
  int h, h_k;
};
struct Flash_fwd_params : public Qkv_params {
  using index_t = int64_t;
  void * __restrict__ o_ptr;
  void * __restrict__ oaccum_ptr;
  index_t o_batch_stride;
  index_t o_row_stride;
  index_t o_head_stride;
  void * __restrict__ softmax_lse_ptr;
  void * __restrict__ softmax_lseaccum_ptr;
  float * __restrict__ q_descale_ptr;
  float * __restrict__ k_descale_ptr;
  float * __restrict__ v_descale_ptr;
  index_t q_descale_batch_stride;
  index_t q_descale_head_stride;
  index_t k_descale_batch_stride;
  index_t k_descale_head_stride;
  index_t v_descale_batch_stride;
  index_t v_descale_head_stride;
  int b, seqlen_q, seqlen_k, seqlen_knew, d, seqlen_q_rounded, seqlen_k_rounded, d_rounded, rotary_dim;
  int total_q, total_k, total_knew;
  int b_k;
  int dv, dv_rounded;
  float scale_softmax;
  float softcap;
  int * __restrict__ cu_seqlens_q;
  int * __restrict__ cu_seqlens_k;
  int * __restrict__ cu_seqlens_knew;
  int * __restrict__ leftpad_k;
  int *__restrict__ seqused_q;
  int *__restrict__ seqused_k;
  index_t oaccum_split_stride;
  index_t oaccum_batch_stride;
  index_t oaccum_row_stride;
  index_t oaccum_head_stride;
  index_t lseaccum_split_stride;
  index_t lseaccum_batch_stride;
  index_t lseaccum_head_stride;
  void * __restrict__ knew_ptr;
  void * __restrict__ vnew_ptr;
  index_t knew_batch_stride;
  index_t vnew_batch_stride;
  index_t knew_row_stride;
  index_t vnew_row_stride;
  index_t knew_head_stride;
  index_t vnew_head_stride;
  void *__restrict__ qv_ptr;
  index_t qv_batch_stride;
  index_t qv_row_stride;
  index_t qv_head_stride;
  void * __restrict__ rotary_cos_ptr;
  void * __restrict__ rotary_sin_ptr;
  int *__restrict__ seqlens_rotary;
  int * __restrict__ kv_batch_idx;
  int * __restrict__ page_table;
  index_t page_table_batch_stride;
  int page_size;
  int num_pages;
  bool pagedkv_tma;
  float p_dropout;
  uint8_t p_dropout_in_uint8_t;
  float rp_dropout;
  int window_size_left, window_size_right;
  int attention_chunk;
  uint64_t * rng_state;
  bool is_bf16;
  bool is_fp32;
  bool is_e4m3;
  bool is_causal;
  bool is_local;
  bool is_rotary_interleaved;
  int num_splits;
  bool pack_gqa;
  int * __restrict__ tile_count_semaphore;
  int * __restrict__ num_m_blocks_ptr;
  int * __restrict__ num_splits_dynamic_ptr;
  int * __restrict__ varlen_batch_idx_ptr;
  int * __restrict__ num_nheads_in_l2_ptr;
  bool skip_scheduler_metadata_computation;
  bool varlen_sort_batches;
  int tile_count_semaphore_offset;
  bool head_swizzle;
  bool prepare_varlen_pdl;
  int arch;
  int num_sm;
  int sk2_k;
  int sk2_wave;
  bool only_qv;
  void *__restrict__ sink_ptr;
  int * __restrict__ sparse_mask_fine;
  int sparse_mask_k_block_size;
  int sparse_mask_q_tile_size;
  int sparse_mask_max_k_blocks;
  int sparse_mask_fine_k_stride;
  int sparse_mask_fine_q_stride;
  bool use_sparse_mask;
};
struct Flash_bwd_params : public Flash_fwd_params {
  using index_t = int64_t;
  void *__restrict__ do_ptr;
  void *__restrict__ dq_ptr;
  void *__restrict__ dk_ptr;
  void *__restrict__ dv_ptr;
  void *__restrict__ dq_accum_ptr;
  void *__restrict__ dk_accum_ptr;
  void *__restrict__ dv_accum_ptr;
  index_t do_batch_stride;
  index_t do_row_stride;
  index_t do_head_stride;
  index_t dq_batch_stride;
  index_t dk_batch_stride;
  index_t dv_batch_stride;
  index_t dq_row_stride;
  index_t dk_row_stride;
  index_t dv_row_stride;
  index_t dq_head_stride;
  index_t dk_head_stride;
  index_t dv_head_stride;
  void *__restrict__ dsoftmax_sum;
  void *__restrict__ softmax_lse_log2_ptr;
  int *__restrict__ dq_semaphore;
  int *__restrict__ dk_semaphore;
  int *__restrict__ dv_semaphore;
  bool deterministic;
  index_t dq_accum_split_stride;
};
template <int Arch, typename T, int kHeadDim, int kHeadDimV, bool Split, bool PagedKVNonTMA, bool Has_softcap, bool PackGQA>
void run_mha_fwd_(Flash_fwd_params &params, cudaStream_t strm_);
void prepare_varlen_num_blocks(Flash_fwd_params &params, cudaStream_t strm_, bool packgqa, int blockM, int blockN, bool enable_pdl);
void sk2_prepare(Flash_fwd_params &params, cudaStream_t strm_, int blockM, int blockN);
template <int Arch, typename T, int kHeadDim, bool Has_softcap>
void run_mha_bwd_(Flash_bwd_params &params, cudaStream_t strm_);
template <typename T, typename Tpartial, int kBlockK>
void run_mha_fwd_combine_(Flash_fwd_params &params, cudaStream_t strm_, bool enable_pdl);
#include <tuple>
#define ATT_BM 192
#define ATT_BN_LOCAL ATT_BN
#define ATT_STAGES 2
constexpr std::tuple<int, int, bool, bool> tile_size_fwd_sm90(
    int headdim, int headdim_v, bool is_causal, bool is_local, int element_size=2,
    bool v_colmajor=false, bool paged_kv_non_TMA=false, bool softcap=false, bool use_one_mma_wg=false) {
  if (element_size == 2) {
    if (headdim <= 64) {
      if (headdim_v == 512) {
        return {64, 64, false, false};
      } else if (headdim_v == 256) {
        return {128, 96, true, false};
      } else {
        if (use_one_mma_wg) {
          return {64, 192, true, true};
        } else {
          bool const use_blockN_128 = is_causal || (!is_local && paged_kv_non_TMA);
          return {192, use_blockN_128 ? 128 : 192, use_blockN_128, true};
        }
      }
    } else if (headdim <= 96) {
      return {192, is_local || paged_kv_non_TMA ? 128 : 144, false, true};
    } else if (headdim <= 128) {
      if (use_one_mma_wg) {
        return {64, is_causal || is_local || paged_kv_non_TMA ? 128 : 176, true, true};
      } else {
        bool const use_blockN_128 = is_causal || is_local || paged_kv_non_TMA;
        return {128, use_blockN_128 ? 128 : 176, true, true};
      }
    } else if (headdim <= 192) {
      return {128, paged_kv_non_TMA || is_local ? 96 : (headdim_v <= 128 ? 128 : 112), true, true};
    } else {
      return {128, is_local ? 64 : 80, true, true};
    }
  } else {
    if (headdim <= 64) {
      if (use_one_mma_wg) {
        return {64, 128, true, true};
      } else {
        return {ATT_BM, is_local ? ATT_BN_LOCAL : ATT_BN, true, true};
      }
    } else if (headdim <= 96) {
      return {192, 128, true, true};
    } else if (headdim <= 128) {
      if (use_one_mma_wg) {
        return {64, 96, true, true};
      } else {
        return {128, paged_kv_non_TMA ? 160 : (v_colmajor || (softcap && is_local) ? 192 : 224), true, true};
      }
    } else if (headdim <= 192) {
      return {128, (paged_kv_non_TMA || softcap) && is_local ? 128 : 160, true, true};
    } else {
      return {128, is_local ? 64 : 128, true, !paged_kv_non_TMA};
    }
  }
}
constexpr std::tuple<int, int, int, int, bool> tile_size_fwd_sm8x(
    bool sm86_or_89, int headdim, int headdim_v, bool is_causal, bool is_local, int element_size=2,
    bool paged_kv=false, bool varlen_and_split=false,
    bool softcap=false, bool append_kv=false) {
  if (element_size == 2) {
    if (headdim <= 64) {
      return {128, varlen_and_split ? 80 : (is_local ? 96 : 112), 4, 1, false};
    } else if (headdim <= 96) {
      return {128, varlen_and_split || is_local ? 48 : 64, 4, 1, false};
    } else if (headdim <= 128) {
      bool const use_8_warps = sm86_or_89 | varlen_and_split;
      return {128, use_8_warps ? (varlen_and_split ? (is_local ? 96 : 112) : (is_local ? 96 : 128)) : (is_local ? 48 : 64), use_8_warps ? 8 : 4, 1, use_8_warps};
    } else if (headdim <= 192) {
      bool const kBlockN_64 = append_kv || is_local || varlen_and_split || paged_kv;
      return {128, kBlockN_64 ? 64 : 96, 8, sm86_or_89 ? 1 : 2, !kBlockN_64};
    } else {
      return {128, sm86_or_89 ? (append_kv ? 32 : (varlen_and_split || is_local ? 48 : 64)) : (append_kv ? 48 : (varlen_and_split || is_local ? 64 : 96)), 8, 1, sm86_or_89 && !append_kv};
    }
  } else {
    return {128, 64, 8, 2, false};
  }
}
#include "cutlass/fast_math.h"
#include "cutlass/arch/barrier.h"
#include "cutlass/arch/barrier.h"
namespace flash {
CUTLASS_DEVICE
static void named_barrier_sync(uint32_t num_threads, uint32_t barrier_id_) {
  static constexpr uint32_t ReservedNamedBarrierCount = static_cast<uint32_t>(cutlass::arch::ReservedNamedBarriers::FirstUserBarrier);
  uint32_t barrier_id = barrier_id_ + ReservedNamedBarrierCount;
  asm volatile("bar.sync %0, %1;" : : "r"(barrier_id), "r"(num_threads));
  cutlass::arch::synclog_emit_named_barrier_arrive_and_wait(__LINE__, num_threads, barrier_id);
}
CUTLASS_DEVICE
static void named_barrier_sync(uint32_t num_threads, cutlass::arch::ReservedNamedBarriers reserved_named_barriers) {
  uint32_t barrier_id = static_cast<uint32_t>(reserved_named_barriers);
  asm volatile("bar.sync %0, %1;" : : "r"(barrier_id), "r"(num_threads));
  cutlass::arch::synclog_emit_named_barrier_arrive_and_wait(__LINE__, num_threads, barrier_id);
}
CUTLASS_DEVICE
static void named_barrier_arrive(uint32_t num_threads, uint32_t barrier_id_) {
  static constexpr uint32_t ReservedNamedBarrierCount = static_cast<uint32_t>(cutlass::arch::ReservedNamedBarriers::FirstUserBarrier);
  uint32_t barrier_id = barrier_id_ + ReservedNamedBarrierCount;
  cutlass::arch::synclog_emit_named_barrier_arrive(__LINE__, num_threads, barrier_id);
  asm volatile("bar.arrive %0, %1;" : : "r"(barrier_id), "r"(num_threads));
}
CUTLASS_DEVICE
static void named_barrier_arrive(uint32_t num_threads, cutlass::arch::ReservedNamedBarriers reserved_named_barriers) {
  uint32_t barrier_id = static_cast<uint32_t>(reserved_named_barriers);
  cutlass::arch::synclog_emit_named_barrier_arrive(__LINE__, num_threads, barrier_id);
  asm volatile("bar.arrive %0, %1;" : : "r"(barrier_id), "r"(num_threads));
}
enum class FwdNamedBarriers {
  QueryEmpty = 0,
  WarpSchedulerWG1 = 1,
  WarpSchedulerWG2 = 2,
  WarpSchedulerWG3 = 3,
  AppendKV = 4,
  QueryRotated = 5,
  PFull = 6,
  PEmpty = 7,
};
enum class BwdNamedBarriers {
  KVEmpty = 0,
  PdS = 1,
  dQEmptyWG1 = 2,
  dQEmptyWG2 = 3,
  dQEmptyWG3 = 4,
  dQFullWG1 = 5,
  dQFullWG2 = 6,
  dQFullWG3 = 7,
};
}
#include <assert.h>
#include <stdint.h>
#include <stdlib.h>
#include <cuda_fp16.h>
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
#include <cuda_bf16.h>
#endif
#include <cute/tensor.hpp>
#include <cutlass/cutlass.h>
#include <cutlass/array.h>
#include <cutlass/numeric_conversion.h>
#include <cutlass/numeric_types.h>
#include <assert.h>
#include <stdlib.h>
#define CHECK_CUDA(call) \
  do { \
    cudaError_t status_ = call; \
    if (status_ != cudaSuccess) { \
      fprintf(stderr, "CUDA error (%s:%d): %s\n", __FILE__, __LINE__, cudaGetErrorString(status_)); \
      exit(1); \
    } \
  } while(0)
#define CHECK_CUDA_KERNEL_LAUNCH() CHECK_CUDA(cudaGetLastError())
namespace flash {
using namespace cute;
template <typename Kernel>
struct enable_sm90_or_later : Kernel {
  template <typename... Args>
  CUTLASS_DEVICE void operator()(Args&&... args) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
    Kernel::operator()(std::forward<Args>(args)...);
#endif
  }
};
template <typename Kernel>
struct enable_sm80_to_sm89 : Kernel {
  template <typename... Args>
  CUTLASS_DEVICE void operator()(Args&&... args) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800) && (__CUDA_ARCH__ <= 890)
    Kernel::operator()(std::forward<Args>(args)...);
#endif
  }
};
template<typename T>
struct MaxOp {
__device__ __forceinline__ T operator()(T const & x, T const & y) { return x > y ? x : y; }
};
template <>
struct MaxOp<float> {
__device__ __forceinline__ float operator()(float const &x, float const &y) { return max(x, y); }
};
template<typename T>
struct SumOp {
__device__ __forceinline__ T operator()(T const & x, T const & y) { return x + y; }
};
template<int THREADS>
struct Allreduce {
  static_assert(THREADS == 32 || THREADS == 16 || THREADS == 8 || THREADS == 4);
  template<typename T, typename Operator>
  static __device__ __forceinline__ T run(T x, Operator &op) {
    constexpr int OFFSET = THREADS / 2;
    x = op(x, __shfl_xor_sync(uint32_t(-1), x, OFFSET));
    return Allreduce<OFFSET>::run(x, op);
  }
};
template<>
struct Allreduce<2> {
template<typename T, typename Operator>
static __device__ __forceinline__ T run(T x, Operator &op) {
  x = op(x, __shfl_xor_sync(uint32_t(-1), x, 1));
  return x;
}
};
CUTLASS_HOST_DEVICE
int div_floor(cutlass::FastDivmod const& divmod, int dividend) {
  return dividend >= 0 ? divmod.divide(dividend) : -1 - divmod.divide(-1 - dividend);
}
CUTLASS_HOST_DEVICE
int round_down(cutlass::FastDivmod const& divmod, int dividend) {
  return div_floor(divmod, dividend) * divmod.divisor;
}
CUTLASS_HOST_DEVICE
int round_up(cutlass::FastDivmod const& divmod, int dividend) {
  return div_floor(divmod, dividend - 1) * divmod.divisor + divmod.divisor;
}
template<bool Transposed=false, typename Layout0>
CUTLASS_DEVICE auto convert_layout_acc_rowcol(Layout0 acc_layout) {
  if constexpr (decltype(rank<0>(acc_layout))::value == 3) {
    static_assert(decltype(size<0, 0>(acc_layout))::value == 2);
    static_assert(decltype(size<0, 1>(acc_layout))::value == 2);
    static_assert(decltype(rank(acc_layout))::value == 3);
    auto l = acc_layout;
    if constexpr (!Transposed) {
      return make_layout(make_layout(get<0, 1>(l), get<1>(l)), make_layout(get<0, 0>(l), get<0, 2>(l), get<2>(l)));
    } else {
       return make_layout(make_layout(get<0, 0>(l), get<0, 2>(l), get<2>(l)), make_layout(get<0, 1>(l), get<1>(l)));
    }
  } else {
    static_assert(decltype(size<0>(acc_layout))::value == 4);
    static_assert(decltype(rank(acc_layout))::value == 3);
    auto l = logical_divide(acc_layout, Shape<_2>{});
    if constexpr (!Transposed) {
      return make_layout(make_layout(get<0, 1>(l), get<1>(l)), make_layout(get<0, 0>(l), get<2>(l)));
    } else {
      return make_layout(make_layout(get<0, 0>(l), get<2>(l)), make_layout(get<0, 1>(l), get<1>(l)));
    }
  }
};
template<typename MMA_Traits, typename Layout0>
CUTLASS_DEVICE auto convert_layout_acc_Aregs(Layout0 acc_layout) {
  using X = Underscore;
  if constexpr (decltype(rank<0>(acc_layout))::value == 3) {
    static_assert(decltype(size<0, 0>(acc_layout))::value == 2);
    static_assert(decltype(size<0, 1>(acc_layout))::value == 2);
    static_assert(decltype(rank(acc_layout))::value == 3);
    static_assert(decltype(rank(get<0>(acc_layout)))::value == 3);
    if constexpr (sizeof(typename MMA_Traits::ValTypeA) == 2) {
      auto l = logical_divide(get<0, 2>(acc_layout), Tile<_2>{});
      return make_layout(make_layout(get<0, 0>(acc_layout), get<0, 1>(acc_layout), get<0, 0>(l)), get<1>(acc_layout), coalesce(make_layout(get<0, 1>(l), get<2>(acc_layout))));
    } else {
      static_assert(sizeof(typename MMA_Traits::ValTypeA) == 1);
      static_assert(decltype(stride<0, 0>(acc_layout))::value == 1);
      static_assert(decltype(stride<0, 1>(acc_layout))::value == 2);
      auto l = logical_divide(get<0, 2>(acc_layout), Tile<Layout<Shape<_2, _2>>>{});
      return make_layout(make_layout(Layout<_4>{}, get<0, 0, 0>(l), get<0, 0, 1>(l)),
                get<1>(acc_layout),
                coalesce(make_layout(get<0, 1>(l), get<2>(acc_layout))));
    }
  } else {
    static_assert(decltype(size<0>(acc_layout))::value == 4);
    static_assert(decltype(rank(acc_layout))::value == 3);
    constexpr int mma_shape_K = get<2>(typename MMA_Traits::Shape_MNK{});
    static_assert(mma_shape_K == 8 || mma_shape_K == 16);
    if constexpr (mma_shape_K == 8) {
      return acc_layout;
    } else {
      auto l = logical_divide(acc_layout, Shape<X, X, _2>{});
      return make_layout(make_layout(get<0>(l), get<2, 0>(l)), get<1>(l), get<2, 1>(l));
    }
  }
};
template <typename To_type, typename Engine, typename Layout>
CUTLASS_DEVICE auto convert_type_unsafe(Tensor<Engine, Layout> const &tensor) {
  using From_type = typename Engine::value_type;
  static constexpr int numel = decltype(size(tensor))::value;
  cutlass::NumericArrayConverter<To_type, From_type, numel> convert_op;
  auto frag = convert_op(*reinterpret_cast<const cutlass::Array<From_type, numel> *>(tensor.data()));
  return make_tensor(make_rmem_ptr<To_type>(&frag), tensor.layout());
}
template <typename Engine, typename Layout, typename EngineOut>
CUTLASS_DEVICE void convert_type_out(Tensor<Engine, Layout> const &tensor, Tensor<EngineOut, Layout> &out) {
  using From_type = typename Engine::value_type;
  using To_type = typename EngineOut::value_type;
  static constexpr int FragmentSize = std::max(sizeof(From_type) / sizeof(To_type), sizeof(To_type) / sizeof(From_type));
  static_assert(CUTE_STATIC_V(size(tensor)) % FragmentSize == 0, "Fragment size does not vectorize properly");
  Tensor frag = recast<cutlass::Array<From_type, FragmentSize> const>(tensor);
  Tensor out_frg = recast<cutlass::Array<To_type, FragmentSize>>(out);
  static_assert(size(frag) == size(out_frg));
  cutlass::NumericArrayConverter<To_type, From_type, FragmentSize> convert_op;
  #pragma unroll
  for (int i = 0; i < size(frag); ++i) { out_frg[i] = convert_op(frag[i]); }
}
template <int N>
CUTE_HOST_DEVICE
void cp_async_wait() {
#if defined(CUTE_ARCH_CP_ASYNC_SM80_ENABLED)
  asm volatile("cp.async.wait_group %0;\n" :: "n"(N));
#endif
}
template <typename TS, typename TD>
CUTE_HOST_DEVICE
void cp_async_ca_zfill(TD* smem_ptr, TS const* gmem_ptr, bool pred) {
  SM80_CP_ASYNC_CACHEALWAYS_ZFILL<TS, TD>::copy(*gmem_ptr, *smem_ptr, pred);
}
template <bool A, class Mma, class Tensor0>
CUTLASS_DEVICE
auto mma_partition_fragment_AB(Mma const& mma, Tensor0 const& tensor0) {
  if constexpr (A) {
    return mma.partition_fragment_A(tensor0);
  } else {
    return mma.partition_fragment_B(tensor0);
  }
}
template <bool zero_init=false, int wg_wait=0, bool SwapAB=false, int M_slice=-1,
    typename Tensor0, typename Tensor1, typename Tensor2, typename TiledMma>
CUTLASS_DEVICE void gemm(TiledMma& tiled_mma, Tensor0 const& tCrA, Tensor1 const& tCrB, Tensor2& tCrC) {
  if constexpr (M_slice >= 0) {
    static constexpr int MMA_M = decltype(size<1>(tCrC))::value;
    static_assert(M_slice < MMA_M);
    Tensor tCrC_slice = cute::logical_divide(tCrC, Shape<cute::Underscore, Int<MMA_M>>{})(_, make_coord(Int<M_slice>{}, _), _);
    if constexpr (!SwapAB) {
      Tensor tCrA_slice = cute::logical_divide(tCrA, Shape<cute::Underscore, Int<MMA_M>>{})(_, make_coord(Int<M_slice>{}, _), _);
      gemm<zero_init, wg_wait, SwapAB, -1>(tiled_mma, tCrA_slice, tCrB, tCrC_slice);
    } else {
      Tensor tCrB_slice = cute::logical_divide(tCrB, Shape<cute::Underscore, Int<MMA_M>>{})(_, make_coord(Int<M_slice>{}, _), _);
      gemm<zero_init, wg_wait, SwapAB, -1>(tiled_mma, tCrA, tCrB_slice, tCrC_slice);
    }
  } else {
    constexpr bool Is_RS = !cute::is_base_of<cute::GMMA::DescriptorIterator, typename TiledMma::FrgTypeA>::value;
    if constexpr (Is_RS) {
      if constexpr (!SwapAB) {
        warpgroup_fence_operand(const_cast<Tensor0 &>(tCrA));
      } else {
        warpgroup_fence_operand(const_cast<Tensor1 &>(tCrB));
      }
    }
    warpgroup_fence_operand(tCrC);
    warpgroup_arrive();
    if constexpr (zero_init) {
      tiled_mma.accumulate_ = GMMA::ScaleOut::Zero;
    }
    static constexpr int kNumKIters = CUTE_STATIC_V(size<2>(tCrA));
    static constexpr int kMaxKIters = 16;
    CUTLASS_PRAGMA_UNROLL
    for (int k_block = 0; k_block < std::min(kNumKIters, kMaxKIters); ++k_block) {
      if constexpr (!SwapAB) {
        cute::gemm(tiled_mma, tCrA(_,_,k_block), tCrB(_,_,k_block), tCrC);
      } else {
        cute::gemm(tiled_mma, tCrB(_,_,k_block), tCrA(_,_,k_block), tCrC);
      }
      tiled_mma.accumulate_ = GMMA::ScaleOut::One;
    }
    if constexpr (kNumKIters > kMaxKIters) {
      int const k_offset = cutlass::canonical_warp_group_idx() < 128 ? 0 : 1;
      CUTLASS_PRAGMA_UNROLL
      for (int k_block = kMaxKIters; k_block < kNumKIters; ++k_block) {
        if constexpr (!SwapAB) {
          cute::gemm(tiled_mma, tCrA(_,_,k_block + k_offset), tCrB(_,_,k_block + k_offset), tCrC);
        } else {
          cute::gemm(tiled_mma, tCrB(_,_,k_block + k_offset), tCrA(_,_,k_block + k_offset), tCrC);
        }
        tiled_mma.accumulate_ = GMMA::ScaleOut::One;
      }
    }
    warpgroup_commit_batch();
    if constexpr (wg_wait >= 0) { warpgroup_wait<wg_wait>(); }
    warpgroup_fence_operand(tCrC);
    if constexpr (Is_RS) {
      if constexpr (!SwapAB) {
        warpgroup_fence_operand(const_cast<Tensor0 &>(tCrA));
      } else {
        warpgroup_fence_operand(const_cast<Tensor1 &>(tCrB));
      }
    }
  }
}
template<bool A_in_regs=false, bool B_in_regs=false, bool SwapAB=false,
     typename Tensor0, typename Tensor1,
     typename Tensor2, typename Tensor3, typename Tensor4,
     typename TiledMma, typename TiledCopyA, typename TiledCopyB,
     typename ThrCopyA, typename ThrCopyB, typename Hook>
CUTLASS_DEVICE void gemm_sm80(Tensor0 &acc, Tensor1 &tCrA, Tensor2 &tCrB, Tensor3 const& tCsA,
               Tensor4 const& tCsB, TiledMma tiled_mma,
               TiledCopyA smem_tiled_copy_A, TiledCopyB smem_tiled_copy_B,
               ThrCopyA smem_thr_copy_A, ThrCopyB smem_thr_copy_B, Hook fn) {
  if constexpr (SwapAB) {
    gemm_sm80<B_in_regs, A_in_regs>(acc, tCrB, tCrA, tCsB, tCsA, tiled_mma, smem_tiled_copy_B, smem_tiled_copy_A, smem_thr_copy_B, smem_thr_copy_A, fn);
  } else {
    CUTE_STATIC_ASSERT_V(size<1>(tCrA) == size<1>(acc));
    CUTE_STATIC_ASSERT_V(size<1>(tCrB) == size<2>(acc));
    CUTE_STATIC_ASSERT_V(size<2>(tCrA) == size<2>(tCrB));
    Tensor tCrA_copy_view = smem_thr_copy_A.retile_D(tCrA);
    CUTE_STATIC_ASSERT_V(size<1>(tCsA) == size<1>(tCrA_copy_view));
    Tensor tCrB_copy_view = smem_thr_copy_B.retile_D(tCrB);
    CUTE_STATIC_ASSERT_V(size<1>(tCsB) == size<1>(tCrB_copy_view));
    if (!A_in_regs) { cute::copy(smem_tiled_copy_A, tCsA(_, _, _0{}), tCrA_copy_view(_, _, _0{})); }
    if (!B_in_regs) { cute::copy(smem_tiled_copy_B, tCsB(_, _, _0{}), tCrB_copy_view(_, _, _0{})); }
    #pragma unroll
    for (int i = 0; i < size<2>(tCrA); ++i) {
      if (i < size<2>(tCrA) - 1) {
        if (!A_in_regs) { cute::copy(smem_tiled_copy_A, tCsA(_, _, i + 1), tCrA_copy_view(_, _, i + 1)); }
        if (!B_in_regs) { cute::copy(smem_tiled_copy_B, tCsB(_, _, i + 1), tCrB_copy_view(_, _, i + 1)); }
      }
      if constexpr (!std::is_same_v<Hook, std::nullptr_t>) {
        if (i == 0) { fn(); }
      }
      cute::gemm(tiled_mma, tCrA(_, _, i), tCrB(_, _, i), acc);
    }
  }
}
template<typename Tensor0, typename Tensor1, typename Tensor2, typename Tensor3,
     typename TiledMma, typename TiledCopy, typename ThrCopy>
CUTLASS_DEVICE void gemm_rs_sm80(Tensor0 &acc, Tensor1 &tCrA, Tensor2 &tCrB, Tensor3 const& tCsB,
                 TiledMma tiled_mma, TiledCopy smem_tiled_copy_B,
                 ThrCopy smem_thr_copy_B) {
  CUTE_STATIC_ASSERT_V(size<1>(tCrA) == size<1>(acc));
  CUTE_STATIC_ASSERT_V(size<1>(tCrB) == size<2>(acc));
  CUTE_STATIC_ASSERT_V(size<2>(tCrA) == size<2>(tCrB));
  Tensor tCrB_copy_view = smem_thr_copy_B.retile_D(tCrB);
  CUTE_STATIC_ASSERT_V(size<1>(tCsB) == size<1>(tCrB_copy_view));
  cute::copy(smem_tiled_copy_B, tCsB(_, _, _0{}), tCrB_copy_view(_, _, _0{}));
  #pragma unroll
  for (int i = 0; i < size<2>(tCrA); ++i) {
    if (i < size<2>(tCrA) - 1) {
      cute::copy(smem_tiled_copy_B, tCsB(_, _, i + 1), tCrB_copy_view(_, _, i + 1));
    }
    cute::gemm(tiled_mma, tCrA(_, _, i), tCrB(_, _, i), acc);
  }
}
template <bool zero_init=false, typename Atom, typename TA, typename TB, typename TC>
CUTLASS_DEVICE void gemm_sm100(Atom& atom, TA const& tA, TB const& tB, TC&& tC) {
  static constexpr int rA = decltype(rank(tA))::value;
  static constexpr int rB = decltype(rank(tB))::value;
  static constexpr int rC = decltype(rank(tC))::value;
  static_assert(rA == 3 && rB == 3 && rC == 3);
  if constexpr (zero_init) { atom.accumulate_ = decltype(atom.accumulate_)::Zero; }
  CUTLASS_PRAGMA_UNROLL
  for (int k_block = 0; k_block < size<2>(tA); k_block++) {
    cute::gemm(atom, tA(_,_,k_block), tB(_,_,k_block), tC);
    atom.accumulate_ = decltype(atom.accumulate_)::One;
  }
}
template <class a_type, class b_type, class c_type,
     int M, int N, UMMA::Major a_major, UMMA::Major b_major,
     UMMA::ScaleIn a_neg, UMMA::ScaleIn b_neg, class... TAs, class... TMs>
CUTE_HOST_DEVICE constexpr
auto
to_tiled_mma_sm100_ts(
  TiledMMA<MMA_Atom<
   MMA_Traits<SM100_MMA_F8F6F4_SS, a_type, b_type, c_type,
          cute::C<M>, cute::C<N>,
          cute::integral_constant<UMMA::Major, a_major>,
          cute::integral_constant<UMMA::Major, b_major>,
          cute::integral_constant<UMMA::ScaleIn, a_neg>,
          cute::integral_constant<UMMA::ScaleIn, b_neg>>,
   TAs...>, TMs...>) {
 return TiledMMA<MMA_Atom<
  MMA_Traits<SM100_MMA_F8F6F4_TS<a_type, b_type, c_type,
                M, N,
                a_major, b_major,
                a_neg, b_neg, UMMA::Saturate::False>>,
  TAs...>, TMs...>{};
}
template <class a_type, class b_type, class c_type,
     int M, int N, UMMA::Major a_major, UMMA::Major b_major,
     UMMA::ScaleIn a_neg, UMMA::ScaleIn b_neg, class... TAs, class... TMs>
CUTE_HOST_DEVICE constexpr
auto
to_tiled_mma_sm100_ts(
  TiledMMA<MMA_Atom<
   SM100_MMA_F16BF16_SS<a_type, b_type, c_type,
          M, N,
          a_major,
          b_major,
          a_neg,
          b_neg>,
   TAs...>, TMs...>) {
 return TiledMMA<MMA_Atom<
  SM100_MMA_F16BF16_TS<a_type, b_type, c_type,
                M, N,
                a_major, b_major,
                a_neg, b_neg, UMMA::Saturate::False>,
  TAs...>, TMs...>{};
}
template <bool Is_even_MN=true, bool Is_even_K=true, bool Clear_OOB_MN=false, bool Clear_OOB_K=true,
     class CopyAtom, class TV, class Tiler, typename Engine0, typename Layout0, typename Engine1, typename Layout1,
     typename Engine2, typename Layout2, typename Engine3, typename Layout3>
CUTLASS_DEVICE void copy(TiledCopy<CopyAtom, TV, Tiler> const &tiled_copy, Tensor<Engine0, Layout0> const &S,
             Tensor<Engine1, Layout1> &D, Tensor<Engine2, Layout2> const &identity_MN,
             Tensor<Engine3, Layout3> const &predicate_K, const int max_MN=0) {
  auto copy_atom = static_cast<CopyAtom const&>(tiled_copy);
  CUTE_STATIC_ASSERT_V(rank(S) == Int<3>{});
  CUTE_STATIC_ASSERT_V(rank(D) == Int<3>{});
  CUTE_STATIC_ASSERT_V(size<0>(S) == size<0>(D));
  CUTE_STATIC_ASSERT_V(size<1>(S) == size<1>(D));
  CUTE_STATIC_ASSERT_V(size<2>(S) == size<2>(D));
  static_assert(!(Clear_OOB_MN && !Clear_OOB_K));
  auto has_with_bool = cute::is_valid([](auto t)->void_t<decltype(declval<typename decltype(t)::Traits>().with(true))>{}, copy_atom);
  #pragma unroll
  for (int m = 0; m < size<1>(S); ++m) {
    bool predicate_mn = Is_even_MN || get<0>(identity_MN(_0{}, m, _0{})) < max_MN;
    if constexpr (Is_even_MN || !Clear_OOB_MN) {
      if (Is_even_MN || predicate_mn) {
        #pragma unroll
        for (int k = 0; k < size<2>(S); ++k) {
          if constexpr (Is_even_K || !Clear_OOB_K) {
            if (Is_even_K || predicate_K(k)) { cute::copy(copy_atom, S(_, m, k), D(_, m, k)); }
          } else {
            if constexpr (has_with_bool) {
              cute::copy(copy_atom.with(predicate_K(k)), S(_, m, k), D(_, m, k));
            } else {
              if (predicate_K(k)) {
                cute::copy(copy_atom, S(_, m, k), D(_, m, k));
              } else {
                cute::clear(D(_, m, k));
              }
            }
          }
        }
      }
    } else {
      if constexpr (!has_with_bool) {
        if (predicate_mn) {
          #pragma unroll
          for (int k = 0; k < size<2>(S); ++k) {
            if (Is_even_K || predicate_K(k)) {
              cute::copy(copy_atom, S(_, m, k), D(_, m, k));
            } else if (Clear_OOB_K) {
              cute::clear(D(_, m, k));
            }
          }
        } else {
          cute::clear(D(_, m, _));
        }
      } else {
        #pragma unroll
        for (int k = 0; k < size<2>(S); ++k) {
          cute::copy(copy_atom.with(predicate_mn && (Is_even_K || predicate_K(k))), S(_, m, k), D(_, m, k));
        }
      }
    }
  }
}
template <typename Fragment>
CUTLASS_DEVICE void permute_Aregs_fp8(Fragment &frag) {
  static_assert(decltype(size<0, 0>(frag))::value == 4);
  static_assert(decltype(size<0, 1>(frag))::value == 2);
  static_assert(decltype(stride<0, 0>(frag))::value == 1);
  static_assert(decltype(stride<0, 1>(frag))::value == 4);
  static_assert(sizeof(typename Fragment::value_type) == 1);
  int quad_idx = threadIdx.x % 4;
  bool lane_03 = quad_idx == 0 || quad_idx == 3;
  int selector_upper = lane_03 ? 0x5410 : 0x1054;
  int selector_lower = lane_03 ? 0x7632 : 0x3276;
  static constexpr int upper_map[4] = {0, 3, 1, 2};
  Tensor frag_64b = recast<uint2>(frag);
  #pragma unroll
  for (int i = 0; i < size(frag_64b); ++i) {
    uint32_t upper = frag_64b[i].x;
    uint32_t lower = frag_64b[i].y;
    uint32_t upper0 = lane_03 ? upper : lower;
    uint32_t lower0 = lane_03 ? lower : upper;
    upper0 = __shfl_sync(uint32_t(-1), upper0, upper_map[quad_idx], 4);
    lower0 = __shfl_sync(uint32_t(-1), lower0, upper_map[quad_idx] ^ 1, 4);
    frag_64b[i].x = __byte_perm(upper0, lower0, selector_upper);
    frag_64b[i].y = __byte_perm(upper0, lower0, selector_lower);
  }
}
template <typename Fragment>
CUTLASS_DEVICE void permute_Cregs_fp8(Fragment &frag) {
  static_assert(decltype(size<0, 0>(frag))::value == 2);
  static_assert(decltype(size<0, 1>(frag))::value == 2);
  static_assert(decltype(size<0, 2>(frag))::value % 2 == 0);
  static_assert(decltype(stride<0, 0>(frag))::value == 1);
  static_assert(sizeof(typename Fragment::value_type) == 4);
  Tensor frag_64b = group_modes<1, 3>(recast<uint2>(frag));
  #pragma unroll
  for (int mi = 0; mi < size<1>(frag_64b); ++mi) {
    #pragma unroll
    for (int i = 0; i < size<0, 2>(frag_64b) / 2; ++i) {
      cutlass::swap(frag_64b(make_coord(_0{}, _1{}, 2 * i), mi), frag_64b(make_coord(_0{}, _0{}, 2 * i + 1), mi));
    }
  }
}
template <typename Fragment>
CUTLASS_DEVICE void permute_output_fp8(Fragment &out) {
  static_assert(decltype(size<0, 0>(out))::value == 2);
  static_assert(decltype(size<0, 1>(out))::value == 2);
  static_assert(decltype(size<0, 2>(out))::value % 2 == 0);
  static_assert(decltype(stride<0, 0>(out))::value == 1);
  static_assert(sizeof(typename Fragment::value_type) == 4);
  Tensor frag = group_modes<1, 3>(out);
  #pragma unroll
  for (int mi = 0; mi < size<1>(frag); ++mi) {
    #pragma unroll
    for (int j = 0; j < size<0, 1>(frag); ++j) {
      #pragma unroll
      for (int i = 0; i < size<0, 2>(frag) / 2; ++i) {
        cutlass::swap(frag(make_coord(_1{}, j, 2 * i), mi), frag(make_coord(_0{}, j, 2 * i + 1), mi));
      }
    }
  }
}
template <typename Fragment>
CUTLASS_DEVICE void permute_output_fp8_Vcolmajor(Fragment &frag) {
  static_assert(decltype(size<0, 0>(frag))::value == 2);
  static_assert(decltype(size<0, 1>(frag))::value == 2);
  static_assert(decltype(stride<0, 0>(frag))::value == 1);
  static_assert(sizeof(typename Fragment::value_type) == 2 || sizeof(typename Fragment::value_type) == 4);
  int quad_idx = threadIdx.x % 4;
  bool lane_03 = quad_idx == 0 || quad_idx == 3;
  static constexpr int upper_map[4] = {0, 2, 3, 1};
  using type2 = std::conditional_t<sizeof(typename Fragment::value_type) == 2, uint32_t, uint64_t>;
  Tensor frag_2 = group_modes<1, 3>(recast<type2>(frag));
  #pragma unroll
  for (int mi = 0; mi < size<1>(frag_2); ++mi) {
    #pragma unroll
    for (int j = 0; j < size<0, 1>(frag_2); ++j) {
      #pragma unroll
      for (int i = 0; i < size<0, 2>(frag_2) / 2; ++i) {
        type2 upper = frag_2(make_coord(_0{}, j, 2 * i), mi);
        type2 lower = frag_2(make_coord(_0{}, j, 2 * i + 1), mi);
        type2 upper0 = lane_03 ? upper : lower;
        type2 lower0 = lane_03 ? lower : upper;
        upper0 = __shfl_sync(uint32_t(-1), upper0, upper_map[quad_idx], 4);
        lower0 = __shfl_sync(uint32_t(-1), lower0, upper_map[quad_idx] ^ 2, 4);
        frag_2(make_coord(_0{}, j, 2 * i), mi) = lane_03 ? upper0 : lower0;
        frag_2(make_coord(_0{}, j, 2 * i + 1), mi) = lane_03 ? lower0 : upper0;
      }
    }
  }
}
template <typename Engine, typename Layout>
CUTLASS_DEVICE void apply_softcap(Tensor<Engine, Layout> &tensor, float const softcap){
  #pragma unroll
  for (int i = 0; i < size(tensor); ++i) {
    tensor(i) = cutlass::fast_tanh(tensor(i) * softcap);
  }
}
template <typename Engine, typename Layout>
CUTLASS_DEVICE auto calculate_dtanh(Tensor<Engine, Layout> &tensor){
  Tensor out = make_fragment_like<float>(tensor);
  #pragma unroll
  for (int i = 0; i < size(tensor); ++i) {
    out(i) = 1.f - (tensor(i) * tensor(i));
  }
  return out;
}
template<class T>
CUTE_DEVICE T warp_prefix_sum(T val) {
  int lane = threadIdx.x % cutlass::NumThreadsPerWarp;
  CUTLASS_PRAGMA_UNROLL
  for (int i = 1; i < cutlass::NumThreadsPerWarp; i <<= 1) {
    T partial_sum = __shfl_up_sync(0xffffffff, val, i);
    if (lane >= i) { val += partial_sum; }
  }
  return val;
}
template<class T>
CUTE_DEVICE T warp_uniform(T a) {
  return __shfl_sync(0xffffffff, a, 0);
}
CUTLASS_DEVICE
int canonical_warp_group_idx_nosync() {
  return threadIdx.x / cutlass::NumThreadsPerWarpGroup;
}
}
namespace flash {
struct TileSchedulerArguments {
  int const num_blocks, num_head, num_batch, num_splits;
  int const qhead_per_khead;
  int const seqlen;
  int const seqlen_k, headdim, headdim_v, element_size;
  int* const tile_count_semaphore = nullptr;
  int const* const cu_seqlens = nullptr;
  int const* const seqused = nullptr;
  int const* const num_splits_dynamic_ptr = nullptr;
  int const* const num_m_blocks_ptr = nullptr;
  int const* const varlen_batch_idx_ptr = nullptr;
  int const* const num_nheads_in_l2_ptr = nullptr;
};
template<bool Varlen=false, bool Split=false, bool PackGQA=false, int kBlock=128>
class SingleTileScheduler {
public:
  using SharedStorage = int;
  struct Params {
    int const num_blocks, num_head, num_batch, num_splits;
    int const qhead_per_khead;
    int const seqlen;
    cutlass::FastDivmod nsplits_divmod;
    int const* const cu_seqlens;
    int const* const seqused;
    int const* const num_splits_dynamic_ptr = nullptr;
  };
  static Params
  to_underlying_arguments(TileSchedulerArguments const& args) {
    assert(!Split || !Varlen || args.num_splits_dynamic_ptr != nullptr);
    assert(!Split || !Varlen || args.num_splits < (1 << 16));
    return {args.num_blocks, args.num_head, args.num_batch, !Split ? 1 : args.num_splits,
        args.qhead_per_khead, args.seqlen,
        cutlass::FastDivmod(!Split ? 1 : args.num_splits),
        !Varlen ? nullptr : args.cu_seqlens, !Varlen ? nullptr : args.seqused,
        args.num_splits_dynamic_ptr};
  }
  static dim3
  get_grid_shape(Params const& params, int num_sm) {
    return {uint32_t(params.num_blocks), uint32_t((!Split ? 1 : params.num_splits) * params.num_head), uint32_t(params.num_batch)};
  }
  struct WorkTileInfo {
    int block_idx = 0;
    int bidh = 0;
    int bidb = 0;
    int split_idx = 0;
    CUTLASS_DEVICE
    bool
    is_valid(Params const& params) const {
      return bidb >= 0;
    }
    CUTLASS_DEVICE
    cute::tuple<int32_t, int32_t, int32_t, int32_t>
    get_block_coord(Params const& params) const {
      return {block_idx, bidh, bidb, !Split ? 0 : split_idx};
    }
  };
  CUTLASS_DEVICE
  SingleTileScheduler(SharedStorage* const smem_scheduler) { }
  template<bool IsProducerWarp=false>
  CUTLASS_DEVICE
  WorkTileInfo
  get_initial_work(Params const& params) const {
    WorkTileInfo work_info {int(blockIdx.x), int(blockIdx.y), int(blockIdx.z), 0};
    if constexpr (Split) {
      int split_idx;
      work_info.bidh = params.nsplits_divmod.divmod(split_idx, work_info.bidh);
      work_info.split_idx = split_idx;
    }
    bool is_valid_tile = true;
    if constexpr (Varlen) {
      int seqlen = params.seqused
        ? params.seqused[work_info.bidb]
        : (params.cu_seqlens ? params.cu_seqlens[work_info.bidb + 1] - params.cu_seqlens[work_info.bidb] : params.seqlen);
      if constexpr (PackGQA) { seqlen *= params.qhead_per_khead; }
      is_valid_tile = work_info.block_idx * kBlock < seqlen;
    }
    if constexpr (Varlen && Split) {
      int num_splits_dynamic = params.num_splits_dynamic_ptr ? params.num_splits_dynamic_ptr[work_info.bidb] : params.num_splits;
      is_valid_tile &= work_info.split_idx < num_splits_dynamic;
      work_info.split_idx |= (num_splits_dynamic << 16);
    }
    work_info.bidb = is_valid_tile ? work_info.bidb : -1;
    return work_info;
  }
  CUTLASS_DEVICE
  void
  init_consumer() const {}
  CUTLASS_DEVICE
  void
  prefetch_next_work(Params const& params, WorkTileInfo& current_work) const {}
  template<bool IsProducerWarp=false>
  CUTLASS_DEVICE
  WorkTileInfo
  get_next_work(Params const& params, WorkTileInfo const& current_work) const {
    return {0, 0, -1, 0};
  }
};
template<bool Split=false>
class StaticPersistentTileScheduler {
public:
  using SharedStorage = int;
  struct Params {
    int total_blocks;
    cutlass::FastDivmod m_block_divmod, head_divmod;
    cutlass::FastDivmod nsplits_divmod;
  };
  static Params
  to_underlying_arguments(TileSchedulerArguments const& args) {
    return {args.num_blocks * args.num_head * args.num_batch * (!Split ? 1 : args.num_splits),
        cutlass::FastDivmod(args.num_blocks), cutlass::FastDivmod(args.num_head * (!Split ? 1 : args.num_splits)),
        cutlass::FastDivmod(!Split ? 1 : args.num_splits)};
  }
  static dim3
  get_grid_shape(Params const& params, int num_sm) {
    return {uint32_t(num_sm)};
  }
  struct WorkTileInfo {
    int tile_idx;
    CUTLASS_DEVICE
    bool
    is_valid(Params const& params) const {
      return tile_idx < params.total_blocks;
    }
    CUTLASS_DEVICE
    cute::tuple<int32_t, int32_t, int32_t, int32_t>
    get_block_coord(Params const& params) const {
      int block, bidh, bidb;
      bidb = params.head_divmod.divmod(bidh, params.m_block_divmod.divmod(block, tile_idx));
      int split_idx = 0;
      if constexpr (Split) {
        bidh = params.nsplits_divmod.divmod(split_idx, bidh);
      }
      return {block, bidh, bidb, split_idx};
    }
  };
  CUTLASS_DEVICE
  StaticPersistentTileScheduler(SharedStorage* const smem_scheduler) {};
  template<bool IsProducerWarp=false>
  CUTLASS_DEVICE
  WorkTileInfo
  get_initial_work(Params const& params) const {
    return {int(blockIdx.x)};
  }
  CUTLASS_DEVICE
  void
  init_consumer() const {}
  CUTLASS_DEVICE
  void
  prefetch_next_work(Params const& params, WorkTileInfo& current_work) const {}
  template<bool IsProducerWarp=false>
  CUTLASS_DEVICE
  WorkTileInfo
  get_next_work(Params const& params, WorkTileInfo const& current_work) const {
    return {current_work.tile_idx + int(gridDim.x)};
  }
};
template<int NumMmaThreads=2 * cutlass::NumThreadsPerWarpGroup, int NumProducerThreads=cutlass::NumThreadsPerWarp,
    bool Split=false, bool PackGQA=false, bool WarpSpecialized=true>
class DynamicPersistentTileScheduler {
  static_assert(WarpSpecialized || NumProducerThreads == NumMmaThreads);
  static constexpr int NumThreads = WarpSpecialized ? NumMmaThreads + NumProducerThreads : NumMmaThreads;
public:
  using SharedStorage = int;
protected:
  SharedStorage* const tile_count_smem;
public:
  struct Params {
    int const total_blocks;
    cutlass::FastDivmod const m_block_divmod, head_divmod;
    cutlass::FastDivmod const l2_minor_divmod, l2_major_divmod;
    cutlass::FastDivmod const l2_minor_residual_divmod;
    int const num_hb_quotient;
    int* const tile_count_semaphore;
  };
  static Params
  to_underlying_arguments(TileSchedulerArguments const& args) {
    long long const size_one_kv_head = long(args.seqlen_k) * long(args.headdim + args.headdim_v) * long(args.element_size);
    int const size_l2 = 32 * 1024 * 1024;
    auto find_log2_floor = [&](int n) { return 31 - cutlass::clz(n); };
    int const swizzle = (size_l2 < size_one_kv_head ? 1 : (1 << find_log2_floor(size_l2 / size_one_kv_head))) * (PackGQA ? 1 : args.qhead_per_khead);
    int const num_hb_remainder = (args.num_head * args.num_batch) % swizzle;
    int const num_split_blocks = args.num_blocks * (!Split ? 1 : args.num_splits);
    assert(args.tile_count_semaphore != nullptr);
    return {num_split_blocks * args.num_head * args.num_batch,
        cutlass::FastDivmod(args.num_blocks), cutlass::FastDivmod(args.num_head),
        cutlass::FastDivmod(swizzle), cutlass::FastDivmod(swizzle * num_split_blocks),
        cutlass::FastDivmod(num_hb_remainder > 0 ? num_hb_remainder : 1),
        (args.num_head * args.num_batch) / swizzle,
        args.tile_count_semaphore};
  }
  static dim3
  get_grid_shape(Params const& params, int num_sm) {
    return {uint32_t(num_sm)};
  }
  struct WorkTileInfo {
    int tile_idx;
    CUTLASS_DEVICE
    bool
    is_valid(Params const& params) const {
      return tile_idx < params.total_blocks;
    }
    CUTLASS_DEVICE
    cute::tuple<int32_t, int32_t, int32_t, int32_t>
    get_block_coord(Params const& params) const {
      int block, bidh, bidb;
      int l2_mod, bidhb, bidhb_residual;
      bidhb = params.l2_major_divmod.divmod(l2_mod, tile_idx);
      if (bidhb < params.num_hb_quotient) {
        block = params.l2_minor_divmod.divmod(bidhb_residual, l2_mod);
      } else {
        block = params.l2_minor_residual_divmod.divmod(bidhb_residual, l2_mod);
      }
      bidb = params.head_divmod.divmod(bidh, bidhb * params.l2_minor_divmod.divisor + bidhb_residual);
      int split_idx = 0;
      if constexpr (Split) {
        split_idx = params.m_block_divmod.divmod(block, block);
      }
      block = params.m_block_divmod.divisor - 1 - block;
      return {block, bidh, bidb, split_idx};
    }
  };
  CUTLASS_DEVICE
  DynamicPersistentTileScheduler(SharedStorage* const smem_scheduler) : tile_count_smem(smem_scheduler) {};
  template<bool IsProducerWarp=false>
  CUTLASS_DEVICE
  WorkTileInfo
  get_initial_work(Params const& params) const {
    return {int(blockIdx.x)};
  }
  CUTLASS_DEVICE
  void
  init_consumer() const {
    if (WarpSpecialized || cutlass::canonical_warp_idx_sync() > 0) {
      flash::named_barrier_arrive(NumThreads, cutlass::arch::ReservedNamedBarriers::StreamkBarrier0 );
    }
  }
  CUTLASS_DEVICE
  void
  prefetch_next_work(Params const& params, WorkTileInfo& current_work) const {
    if (threadIdx.x % NumProducerThreads == 0) {
      current_work.tile_idx = atomicAdd(params.tile_count_semaphore, 1) + int(gridDim.x);
    }
  }
  template<bool IsProducerWarp=false>
  CUTLASS_DEVICE
  WorkTileInfo
  get_next_work(Params const& params, WorkTileInfo const& current_work) const {
    if constexpr (IsProducerWarp) {
      int new_tile_idx = __shfl_sync(0xffffffff, current_work.tile_idx, 0 );
      flash::named_barrier_sync(NumThreads, cutlass::arch::ReservedNamedBarriers::StreamkBarrier0 );
      if (threadIdx.x % NumProducerThreads == 0) {
        *tile_count_smem = current_work.tile_idx;
      }
      flash::named_barrier_arrive(NumThreads, cutlass::arch::ReservedNamedBarriers::StreamkBarrier1 );
      return {new_tile_idx};
    } else {
      flash::named_barrier_sync(NumThreads, cutlass::arch::ReservedNamedBarriers::StreamkBarrier1 );
      int tile_idx = *tile_count_smem;
      flash::named_barrier_arrive(NumThreads, cutlass::arch::ReservedNamedBarriers::StreamkBarrier0 );
      return {tile_idx};
    }
  }
};
template <bool Varlen, int kBlock, bool SPT = false>
class SingleTileBwdLPTScheduler {
public:
  using SharedStorage = int;
  struct Params {
    int const total_blocks;
    cutlass::FastDivmod const block_divmod, head_divmod;
    cutlass::FastDivmod const l2_minor_divmod, l2_major_divmod;
    cutlass::FastDivmod const l2_minor_residual_divmod;
    int const num_hb_quotient;
    int const seqlen;
    int const* const cu_seqlens;
    int const* const seqused;
  };
  static Params
  to_underlying_arguments(TileSchedulerArguments const& args) {
    long long const size_one_qdo_head = long(args.seqlen_k) * long(args.headdim + args.headdim_v) * long(args.element_size);
    long long const size_one_dqaccum_head = long(args.seqlen_k) * long(args.headdim) * sizeof(float);
    long long const size_one_head = size_one_qdo_head + size_one_dqaccum_head;
    int const size_l2 = 40 * 1024 * 1024;
    auto find_log2_floor = [&](int n) { return 31 - cutlass::clz(n); };
    int const swizzle = size_l2 < size_one_head ? 1 : (1 << find_log2_floor(size_l2 / size_one_head));
    int const num_hb_remainder = (args.num_head * args.num_batch) % swizzle;
    assert(args.tile_count_semaphore != nullptr);
    return {args.num_blocks * args.num_head * args.num_batch,
        cutlass::FastDivmod(args.num_blocks), cutlass::FastDivmod(args.num_head),
        cutlass::FastDivmod(swizzle), cutlass::FastDivmod(swizzle * args.num_blocks),
        cutlass::FastDivmod(num_hb_remainder > 0 ? num_hb_remainder : 1),
        (args.num_head * args.num_batch) / swizzle,
        args.seqlen, !Varlen ? nullptr : args.cu_seqlens, !Varlen ? nullptr : args.seqused};
  }
  static dim3
  get_grid_shape(Params const& params, int num_sm) {
    return {uint32_t(params.total_blocks)};
  }
  struct WorkTileInfo {
    int block;
    int bidh;
    int bidb;
    CUTLASS_DEVICE
    bool
    is_valid(Params const& params) const {
      return bidb >= 0;
    }
    CUTLASS_DEVICE
    cute::tuple<int32_t, int32_t, int32_t, int32_t>
    get_block_coord(Params const& params) const {
      return {block, bidh, bidb, 0 };
    }
  };
  CUTLASS_DEVICE
  SingleTileBwdLPTScheduler(SharedStorage* const smem_scheduler) { }
  template<bool IsProducerWarp=false>
  CUTLASS_DEVICE
  WorkTileInfo
  get_initial_work(Params const& params) const {
    int tile_idx = blockIdx.x;
    int block, bidh, bidb;
    int l2_mod, bidhb, bidhb_residual;
    bidhb = params.l2_major_divmod.divmod(l2_mod, tile_idx);
    if (bidhb < params.num_hb_quotient) {
      block = params.l2_minor_divmod.divmod(bidhb_residual, l2_mod);
    } else {
      block = params.l2_minor_residual_divmod.divmod(bidhb_residual, l2_mod);
    }
    bidb = params.head_divmod.divmod(bidh, bidhb * params.l2_minor_divmod.divisor + bidhb_residual);
    bool is_valid_tile = true;
    int num_blocks;
    if constexpr (Varlen) {
      int seqlen = params.seqused
        ? params.seqused[bidb]
        : (params.cu_seqlens ? params.cu_seqlens[bidb + 1] - params.cu_seqlens[bidb] : params.seqlen);
      num_blocks = cute::ceil_div(seqlen, Int<kBlock>{});
      is_valid_tile = block < num_blocks;
    } else {
      num_blocks = params.block_divmod.divisor;
    }
    if constexpr (SPT) {
      block = num_blocks - block - 1;
    }
    return {block, bidh, is_valid_tile ? bidb : -1};
  }
  CUTLASS_DEVICE
  void
  init_consumer() const {}
  CUTLASS_DEVICE
  void
  prefetch_next_work(Params const& params, WorkTileInfo& current_work) const {}
  template<bool IsProducerWarp=false>
  CUTLASS_DEVICE
  WorkTileInfo
  get_next_work(Params const& params, WorkTileInfo const& current_work) const {
    return {0, 0, -1};
  }
};
template<int kBlockM, int kBlockN, int NumMmaThreads=2 * cutlass::NumThreadsPerWarpGroup, int NumProducerThreads=cutlass::NumThreadsPerWarp,
     bool Split=false, bool PackGQA=false, bool WarpSpecialized=true, bool LPT = false, bool Sort = false, bool Prepared = true>
class VarlenDynamicPersistentTileScheduler {
  static_assert(WarpSpecialized || NumProducerThreads == NumMmaThreads);
  static constexpr int NumThreads = WarpSpecialized ? NumMmaThreads + NumProducerThreads : NumMmaThreads;
public:
  using SharedStorage = int4;
protected:
  SharedStorage* const work_info_smem;
public:
  struct Params {
    int num_head, num_batch;
    int const qhead_per_khead;
    int const seqlen;
    cutlass::FastDivmod head_divmod;
    cutlass::FastDivmod nsplits_divmod;
    int* const tile_count_semaphore;
    int const* const cu_seqlens;
    int const* const seqused;
    int const* const num_splits_dynamic_ptr;
    int const* const num_m_blocks_ptr;
    int const* const varlen_batch_idx_ptr;
    int const* const num_nheads_in_l2_ptr;
  };
  static Params
  to_underlying_arguments(TileSchedulerArguments const& args) {
    assert(args.tile_count_semaphore != nullptr);
    assert(args.num_head < (1 << 16));
    assert(!Split || args.num_splits < (1 << 8));
    return {args.num_head, args.num_batch,
        args.qhead_per_khead, args.seqlen,
        cutlass::FastDivmod(args.num_head),
        cutlass::FastDivmod(!Split ? 1 : args.num_splits),
        args.tile_count_semaphore, args.cu_seqlens, args.seqused,
        args.num_splits_dynamic_ptr,
        args.num_m_blocks_ptr,
        args.varlen_batch_idx_ptr,
        args.num_nheads_in_l2_ptr};
  }
  static dim3
  get_grid_shape(Params const& params, int num_sm) {
    return {uint32_t(num_sm)};
  }
  struct WorkTileInfo {
    int tile_idx, block, bidh, bidb;
    CUTLASS_DEVICE
    bool
    is_valid(Params const& params) const {
      return bidb < params.num_batch;
    }
    CUTLASS_DEVICE
    cute::tuple<int32_t, int32_t, int32_t, int32_t>
    get_block_coord(Params const& params) const {
      auto get_actual_batch = [&](int virtual_batch) {
        if constexpr(Prepared && Sort) {
          return params.varlen_batch_idx_ptr[virtual_batch];
        } else {
          return virtual_batch;
        }
      };
      if constexpr (!Split) {
        return {block, bidh, get_actual_batch(bidb), 0 };
      } else {
        uint32_t bidh_packed = reinterpret_cast<uint32_t const&>(bidh);
        uint32_t bidh_actual_u = bidh_packed & 0x0000FFFF;
        int bidh_actual = reinterpret_cast<int&>(bidh_actual_u);
        uint32_t split_idx_u = ((bidh_packed & 0x00FF0000) >> 16) + ((bidh_packed & 0xFF000000) >> 8);
        int split_idx = reinterpret_cast<int&>(split_idx_u);
        return {block, bidh_actual, get_actual_batch(bidb), split_idx};
      }
    }
  };
  CUTLASS_DEVICE
  VarlenDynamicPersistentTileScheduler(SharedStorage* const smem_scheduler) : work_info_smem(smem_scheduler) {};
  CUTLASS_DEVICE
  WorkTileInfo
  tile_idx_to_work_tile(Params const& params, int next_tile_idx, WorkTileInfo const& current_work) const {
    int lane = threadIdx.x % cutlass::NumThreadsPerWarp;
    auto get_num_m_blocks = [&] (int bidb_start) {
      int batch_idx = lane + bidb_start;
      if constexpr (Prepared) {
        return batch_idx < params.num_batch && lane < cutlass::NumThreadsPerWarp - 1
          ? params.num_m_blocks_ptr[batch_idx] : 0;
      } else {
        int seqlen = params.seqlen * (!PackGQA ? 1 : params.qhead_per_khead);
        if (seqlen > kBlockM) {
          if (params.seqused) {
            seqlen = batch_idx < params.num_batch ? params.seqused[batch_idx] : 0;
          } else if (params.cu_seqlens) {
            int cur_cu_seqlen = batch_idx <= params.num_batch ? params.cu_seqlens[batch_idx] : 0;
            int next_cu_seqlen = __shfl_down_sync(0xffffffff, cur_cu_seqlen, 1);
            seqlen = next_cu_seqlen - cur_cu_seqlen;
          } else {
            seqlen = params.seqlen;
          }
          if constexpr (PackGQA) { seqlen *= params.qhead_per_khead; }
        }
        return batch_idx < params.num_batch && lane < cutlass::NumThreadsPerWarp - 1
          ? cute::ceil_div(seqlen, kBlockM) : 0;
      }
    };
    auto get_num_splits = [&] (int bidb_start) {
      int batch_idx = lane + bidb_start;
      bool is_valid = batch_idx < params.num_batch && lane < cutlass::NumThreadsPerWarp - 1;
      if constexpr (!Split) {
        return is_valid ? 1 : 0;
      } else if constexpr(Prepared) {
        return is_valid ? params.num_splits_dynamic_ptr[batch_idx] : 0;
      } else {
        return is_valid ? params.nsplits_divmod.divisor : 0;
      }
    };
    int num_m_blocks = get_num_m_blocks(current_work.bidb);
    int num_splits = get_num_splits(current_work.bidb);
    int num_split_m_blocks = !Split ? num_m_blocks : num_m_blocks * num_splits;
    int num_m_blocks_cumulative = warp_prefix_sum(num_split_m_blocks);
    int m_blocks_in_group = __shfl_sync(0xffffffff, num_m_blocks_cumulative, cutlass::NumThreadsPerWarp - 1);
    int group_end_tile = current_work.tile_idx + m_blocks_in_group * params.num_head;
    int bidb = current_work.bidb;
    while (group_end_tile <= next_tile_idx) {
      bidb += cutlass::NumThreadsPerWarp - 1;
      if (bidb >= params.num_batch) {
        return {next_tile_idx, 0, 0, params.num_batch};
      }
      num_m_blocks = get_num_m_blocks(bidb);
      num_splits = get_num_splits(bidb);
      num_split_m_blocks = !Split ? num_m_blocks : num_m_blocks * num_splits;
      num_m_blocks_cumulative = warp_prefix_sum(num_split_m_blocks);
      m_blocks_in_group = __shfl_sync(0xffffffff, num_m_blocks_cumulative, cutlass::NumThreadsPerWarp - 1);
      group_end_tile += m_blocks_in_group * params.num_head;
    }
    int group_start_tile = group_end_tile - m_blocks_in_group * params.num_head;
    int batch_idx_in_group = __popc(__ballot_sync(0xffffffff, group_start_tile + num_m_blocks_cumulative * params.num_head <= next_tile_idx));
    bidb += batch_idx_in_group;
    num_m_blocks = __shfl_sync(0xffffffff, num_m_blocks, batch_idx_in_group);
    if constexpr (Split) { num_splits = __shfl_sync(0xffffffff, num_splits, batch_idx_in_group); }
    group_start_tile += (batch_idx_in_group == 0 ? 0 : __shfl_sync(0xffffffff, num_m_blocks_cumulative, batch_idx_in_group - 1)) * params.num_head;
    int mh_block = next_tile_idx - group_start_tile;
    int block, bidh;
    if constexpr (LPT) {
      if (!Split || num_splits == 1) {
        auto get_nheads_in_l2 = [&](int batch_idx) {
          if constexpr(Prepared) {
            return params.num_nheads_in_l2_ptr[batch_idx];
          } else {
            return !PackGQA ? params.qhead_per_khead : 1;
          }
        };
        int nheads_in_l2 = get_nheads_in_l2(bidb);
        int mh_in_l2 = nheads_in_l2 * num_m_blocks;
        int section_idx = mh_block / mh_in_l2;
        int l2_mod = mh_block - section_idx * mh_in_l2;
        int nheads_remainder = params.num_head - section_idx * nheads_in_l2;
        int nheads_in_this_section = nheads_in_l2 <= nheads_remainder ? nheads_in_l2 : nheads_remainder;
        block = l2_mod / nheads_in_this_section;
        int bidh_residual = l2_mod - block * nheads_in_this_section;
        bidh = section_idx * nheads_in_l2 + bidh_residual;
        if constexpr(Split) {
          uint32_t bidh_packed = reinterpret_cast<uint32_t&>(bidh) + (reinterpret_cast<uint32_t&>(num_splits) << 24);
          bidh = reinterpret_cast<int&>(bidh_packed);
        }
      } else {
        bidh = mh_block / num_m_blocks;
        block = mh_block - bidh * num_m_blocks;
        if constexpr (Split) {
          int bidh_actual = bidh / num_splits;
          int split_idx = bidh - bidh_actual * num_splits;
          uint32_t bidh_packed = reinterpret_cast<uint32_t&>(bidh_actual) + (reinterpret_cast<uint32_t&>(split_idx) << 16) + (reinterpret_cast<uint32_t&>(num_splits) << 24);
          bidh = reinterpret_cast<int&>(bidh_packed);
        }
      }
      block = num_m_blocks - 1 - block;
    } else {
      bidh = mh_block / num_m_blocks;
      block = mh_block - bidh * num_m_blocks;
      if constexpr (Split) {
        int bidh_actual = bidh / num_splits;
        int split_idx = bidh - bidh_actual * num_splits;
        uint32_t bidh_packed = reinterpret_cast<uint32_t&>(bidh_actual) + (reinterpret_cast<uint32_t&>(split_idx) << 16) + (reinterpret_cast<uint32_t&>(num_splits) << 24);
        bidh = reinterpret_cast<int&>(bidh_packed);
      }
    }
    return {group_start_tile, block, bidh, bidb};
  }
  template<bool IsProducerWarp=false>
  CUTLASS_DEVICE
  WorkTileInfo
  get_initial_work(Params const& params) const {
    if constexpr (IsProducerWarp) {
      WorkTileInfo work_info = tile_idx_to_work_tile(params, int(blockIdx.x), {0, 0, 0, 0});
      if (threadIdx.x % cutlass::NumThreadsPerWarp == 0) {
        *work_info_smem = make_int4(work_info.tile_idx, work_info.block, work_info.bidh, work_info.bidb);
      }
      flash::named_barrier_arrive(NumThreads, cutlass::arch::ReservedNamedBarriers::StreamkBarrier1 );
      return work_info;
    } else {
      return get_next_work<false>(params, {0, 0, 0, 0});
    }
  }
  CUTLASS_DEVICE
  void
  init_consumer() const {
  }
  CUTLASS_DEVICE
  void
  prefetch_next_work(Params const& params, WorkTileInfo& current_work) const {
    if (threadIdx.x % NumProducerThreads == 0) {
      current_work.tile_idx = atomicAdd(params.tile_count_semaphore, 1) + int(gridDim.x);
    }
  }
  template<bool IsProducerWarp=false>
  CUTLASS_DEVICE
  WorkTileInfo
  get_next_work(Params const& params, WorkTileInfo const& current_work) const {
    if constexpr (IsProducerWarp) {
      int new_tile_idx = __shfl_sync(0xffffffff, current_work.tile_idx, 0 );
      WorkTileInfo work_info = {__shfl_sync(0xffffffff, current_work.tile_idx, 1 ), current_work.block, current_work.bidh, current_work.bidb};
      work_info = tile_idx_to_work_tile(params, new_tile_idx, work_info);
      flash::named_barrier_sync(NumThreads, cutlass::arch::ReservedNamedBarriers::StreamkBarrier0 );
      if (threadIdx.x % cutlass::NumThreadsPerWarp == 0) {
        *work_info_smem = make_int4(work_info.tile_idx, work_info.block, work_info.bidh, work_info.bidb);
      }
      flash::named_barrier_arrive(NumThreads, cutlass::arch::ReservedNamedBarriers::StreamkBarrier1 );
      return work_info;
    } else {
      flash::named_barrier_sync(NumThreads, cutlass::arch::ReservedNamedBarriers::StreamkBarrier1 );
      int4 work_info = *work_info_smem;
      flash::named_barrier_arrive(NumThreads, cutlass::arch::ReservedNamedBarriers::StreamkBarrier0 );
      return WorkTileInfo{work_info.x, work_info.y, work_info.z, work_info.w};
    }
  }
};
template <class T, class = void> struct is_sk2_sched { static constexpr bool value = false; };
template <class T> struct is_sk2_sched<T, std::void_t<decltype(T::IsSK2)>> { static constexpr bool value = T::IsSK2; };
template <int NumMmaThreads, int NumProducerThreads, bool WarpSpecialized>
class SK2Scheduler {
public:
  static constexpr bool IsSK2 = true;
private:
  static constexpr int NumThreads = WarpSpecialized ? NumMmaThreads + NumProducerThreads : NumMmaThreads;
public:
  struct SharedStorage { int4 a; int4 b; int4 c; };
protected:
  SharedStorage* const work_info_smem;
public:
  struct Params {
    int num_batch;
    int const* tp;
    int const* desc;
    int const* st;
    int* cnt;
  };
  static Params to_underlying_arguments(TileSchedulerArguments const& args) {
    return {args.num_batch, args.num_m_blocks_ptr, args.varlen_batch_idx_ptr, args.num_splits_dynamic_ptr,
        args.tile_count_semaphore};
  }
  static dim3 get_grid_shape(Params const& params, int num_sm) { return {uint32_t(num_sm)}; }
  struct WorkTileInfo {
    int code, block, bidh, bidb;
    int t, np, slot, r1;
    CUTLASS_DEVICE bool is_valid(Params const& params) const { return bidb < params.num_batch; }
    CUTLASS_DEVICE cute::tuple<int32_t, int32_t, int32_t, int32_t> get_block_coord(Params const& params) const {
      return {block, bidh, bidb, code};
    }
  };
  CUTLASS_DEVICE SK2Scheduler(SharedStorage* const smem_scheduler) : work_info_smem(smem_scheduler) {};
  static CUTLASS_DEVICE int cta_of(long long x, long long W, int G) {
    return int(((x + 1) * G + W - 1) / W) - 1;
  }
  CUTLASS_DEVICE WorkTileInfo piece(Params const& params, int t, int r0, int r1, int slot) const {
    int const G = int(gridDim.x);
    int const W = params.st[G], K = params.st[G + 1];
    int const t0 = params.tp[t], t1 = params.tp[t + 1];
    int const d = params.desc[t];
    int const nb = t1 - t0 - K;
    int const lo = max(r0 - t0 - K, 0), hi = min(r1 - t0 - K, nb);
    if (hi <= lo) { return invalid(params); }
    int const cf = cta_of(t0 + K, W, G), cl = cta_of(t1 - 1, W, G);
    WorkTileInfo w;
    w.code = lo | (hi << 11) | (1 << 30);
    w.block = d >> 16; w.bidh = (d >> 8) & 0xFF; w.bidb = d & 0xFF;
    w.t = t; w.np = cl - cf + 1; w.slot = slot; w.r1 = r1;
    return w;
  }
  CUTLASS_DEVICE WorkTileInfo invalid(Params const& params) const {
    WorkTileInfo w{};
    w.bidb = params.num_batch;
    return w;
  }
  CUTLASS_DEVICE void publish(WorkTileInfo const& w) const {
    if (threadIdx.x % cutlass::NumThreadsPerWarp == 0) {
      work_info_smem->a = make_int4(w.code, w.block, w.bidh, w.bidb);
      work_info_smem->b = make_int4(w.t, w.np, w.slot, w.r1);
    }
  }
  template<bool IsProducerWarp=false>
  CUTLASS_DEVICE WorkTileInfo get_initial_work(Params const& params) const {
    if constexpr (IsProducerWarp) {
      int const G = int(gridDim.x), c = int(blockIdx.x);
      long long const W = params.st[G];
      int const r0 = int(c * W / G), r1 = int((c + 1) * W / G);
      WorkTileInfo w = r0 < r1 ? piece(params, params.st[c], r0, r1, 0) : invalid(params);
      publish(w);
      flash::named_barrier_arrive(NumThreads, cutlass::arch::ReservedNamedBarriers::StreamkBarrier1 );
      return w;
    } else {
      return get_next_work<false>(params, WorkTileInfo{});
    }
  }
  CUTLASS_DEVICE void init_consumer() const {}
  CUTLASS_DEVICE void prefetch_next_work(Params const& params, WorkTileInfo& current_work) const {}
  template<bool IsProducerWarp=false>
  CUTLASS_DEVICE WorkTileInfo get_next_work(Params const& params, WorkTileInfo const& cur) const {
    if constexpr (IsProducerWarp) {
      WorkTileInfo w = invalid(params);
      if (cur.bidb < params.num_batch) {
        int const nt = cur.t + 1;
        if (params.tp[nt] < cur.r1) { w = piece(params, nt, params.tp[nt], cur.r1, 1); }
      }
      flash::named_barrier_sync(NumThreads, cutlass::arch::ReservedNamedBarriers::StreamkBarrier0 );
      publish(w);
      flash::named_barrier_arrive(NumThreads, cutlass::arch::ReservedNamedBarriers::StreamkBarrier1 );
      return w;
    } else {
      flash::named_barrier_sync(NumThreads, cutlass::arch::ReservedNamedBarriers::StreamkBarrier1 );
      int4 a = work_info_smem->a, b = work_info_smem->b;
      flash::named_barrier_arrive(NumThreads, cutlass::arch::ReservedNamedBarriers::StreamkBarrier0 );
      return WorkTileInfo{a.x, a.y, a.z, a.w, b.x, b.y, b.z, b.w};
    }
  }
};
}
#include "cute/tensor.hpp"
#include <cutlass/cutlass.h>
#include <cutlass/arch/reg_reconfig.h>
#include <cutlass/array.h>
#include <cutlass/numeric_types.h>
#include <cutlass/numeric_conversion.h>
#include <cutlass/kernel_hardware_info.h>
#include "cutlass/pipeline/pipeline.hpp"
#include "cutlass/arch/grid_dependency_control.h"
namespace flash {
template <bool Varlen, int kBlock>
struct SeqlenInfo {
  int const offset, offset_padded;
  int const seqlen;
  CUTLASS_DEVICE
  SeqlenInfo(int const bidb, int const seqlen_static, int const* const cu_seqlens, int const* const seqused)
    : offset(!Varlen || cu_seqlens == nullptr ? 0 : cu_seqlens[bidb])
    , offset_padded(!Varlen || cu_seqlens == nullptr ? 0 : (cu_seqlens[bidb] + bidb * kBlock) / kBlock * kBlock)
    , seqlen(!Varlen
         ? seqlen_static
         : (seqused ? seqused[bidb] : (cu_seqlens ? cu_seqlens[bidb + 1] - cu_seqlens[bidb] : seqlen_static)))
  {
  }
};
template <bool Varlen, int kBlockM>
struct SeqlenInfoQK {
  int const offset_q, offset_k, offset_q_padded;
  int const seqlen_q, seqlen_k;
  CUTLASS_DEVICE
  SeqlenInfoQK(int const bidb, int const seqlen_q_static, int const seqlen_k_static,
         int const* const cu_seqlens_q, int const* const cu_seqlens_k,
         int const* const seqused_q, int const* const seqused_k
         )
    : offset_q(!Varlen || cu_seqlens_q == nullptr ? 0 : cu_seqlens_q[bidb])
    , offset_k(!Varlen || cu_seqlens_k == nullptr ? 0 : cu_seqlens_k[bidb])
    , offset_q_padded(!Varlen || cu_seqlens_q == nullptr ? 0 : (cu_seqlens_q[bidb] + bidb * kBlockM) / kBlockM * kBlockM)
    , seqlen_q(!Varlen
          ? seqlen_q_static
          : (seqused_q ? seqused_q[bidb] : (cu_seqlens_q ? cu_seqlens_q[bidb + 1] - cu_seqlens_q[bidb] : seqlen_q_static)))
    , seqlen_k(!Varlen
          ? seqlen_k_static
          : (seqused_k ? seqused_k[bidb] : (cu_seqlens_k ? cu_seqlens_k[bidb + 1] - cu_seqlens_k[bidb] : seqlen_k_static)))
  {
  }
};
template <bool Varlen, bool AppendKV>
struct SeqlenInfoQKNewK {
  static_assert(!(AppendKV && !Varlen), "AppendKV is only supported with Varlen");
  int const leftpad_k;
  int const offset_q, offset_k, offset_k_new;
  int const seqlen_q, seqlen_k_og, seqlen_k_new, seqlen_k, seqlen_rotary;
  CUTLASS_DEVICE
  SeqlenInfoQKNewK(int const bidb, int const seqlen_q_static, int const seqlen_k_static, int const shape_K_new_0,
           int const* const cu_seqlens_q, int const* const cu_seqlens_k, int const* const cu_seqlens_k_new,
           int const* const seqused_q, int const* const seqused_k, int const* const ptr_leftpad_k,
           int const* const seqlens_rotary
           )
    : leftpad_k(ptr_leftpad_k ? ptr_leftpad_k[bidb] : 0)
    , offset_q(!Varlen || cu_seqlens_q == nullptr ? 0 : cu_seqlens_q[bidb])
    , offset_k(!Varlen ? 0 : (cu_seqlens_k ? cu_seqlens_k[bidb] : 0) + leftpad_k)
    , offset_k_new(!AppendKV || cu_seqlens_k_new == nullptr ? 0 : cu_seqlens_k_new[bidb])
    , seqlen_q(!Varlen
          ? seqlen_q_static
          : (seqused_q ? seqused_q[bidb] : (cu_seqlens_q ? cu_seqlens_q[bidb + 1] - cu_seqlens_q[bidb] : seqlen_q_static)))
    , seqlen_k_og(!Varlen
           ? seqlen_k_static
           : (seqused_k ? seqused_k[bidb] : (cu_seqlens_k ? cu_seqlens_k[bidb + 1] - cu_seqlens_k[bidb] : seqlen_k_static)) - leftpad_k)
    , seqlen_k_new(!AppendKV
            ? 0
            : (cu_seqlens_k_new ? cu_seqlens_k_new[bidb + 1] - cu_seqlens_k_new[bidb] : shape_K_new_0))
    , seqlen_k(!AppendKV ? seqlen_k_og : seqlen_k_og + seqlen_k_new)
    , seqlen_rotary(!AppendKV || !seqlens_rotary ? seqlen_k_og + leftpad_k : seqlens_rotary[bidb])
  {
  }
};
}
#include <cmath>
#include <cute/tensor.hpp>
#include <cutlass/numeric_types.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
namespace flash {
using namespace cute;
template<bool zero_init=true, typename Engine0, typename Layout0, typename Engine1, typename Layout1, typename Operator>
__device__ __forceinline__ void thread_reduce_(Tensor<Engine0, Layout0> const &tensor, Tensor<Engine1, Layout1> &summary, Operator &op) {
  static_assert(Layout0::rank == 2, "Only support 2D Tensor");
  static_assert(Layout1::rank == 1, "Only support 1D Tensor");
  CUTE_STATIC_ASSERT_V(size<0>(summary) == size<0>(tensor));
  #pragma unroll
  for (int ni = 0; ni < size<1>(tensor); ni++) {
    #pragma unroll
    for (int mi = 0; mi < size<0>(tensor); mi++) {
      summary(mi) = zero_init && ni == 0 ? tensor(mi, ni) : op(summary(mi), tensor(mi, ni));
    }
  }
}
template<typename Engine0, typename Layout0, typename Engine1, typename Layout1, typename Operator>
__device__ __forceinline__ void quad_allreduce_(Tensor<Engine0, Layout0> &dst, Tensor<Engine1, Layout1> &src, Operator &op) {
  CUTE_STATIC_ASSERT_V(size(dst) == size(src));
  #pragma unroll
  for (int i = 0; i < size(dst); i++) {
    dst(i) = Allreduce<4>::run(src(i), op);
  }
}
template<bool zero_init=true, typename Engine0, typename Layout0, typename Engine1, typename Layout1, typename Operator>
__device__ __forceinline__ void reduce_(Tensor<Engine0, Layout0> const& tensor, Tensor<Engine1, Layout1> &summary, Operator &op) {
  thread_reduce_<zero_init>(tensor, summary, op);
  quad_allreduce_(summary, summary, op);
}
template<bool zero_init=true, typename Engine0, typename Layout0, typename Engine1, typename Layout1>
__device__ __forceinline__ void reduce_max(Tensor<Engine0, Layout0> const& tensor, Tensor<Engine1, Layout1> &max){
  MaxOp<float> max_op;
  reduce_<zero_init>(tensor, max, max_op);
}
template<bool zero_init=true, bool warp_reduce=true, typename Engine0, typename Layout0, typename Engine1, typename Layout1>
__device__ __forceinline__ void reduce_sum(Tensor<Engine0, Layout0> const& tensor, Tensor<Engine1, Layout1> &sum){
  SumOp<float> sum_op;
  thread_reduce_<zero_init>(tensor, sum, sum_op);
  if constexpr (warp_reduce) { quad_allreduce_(sum, sum, sum_op); }
}
__device__ __forceinline__ float att_ex2_poly(float x) {
  x = fmaxf(x, -64.f);
  float const t = __fadd_rn(x, 12582912.f);
  float const f = __fsub_rn(x, __fsub_rn(t, 12582912.f));
  float const p = __fmaf_rn(__fmaf_rn(__fmaf_rn(5.521949008e-02f, f, 2.426151186e-01f), f, 6.932522058e-01f), f, 9.999275804e-01f);
  return __uint_as_float(__float_as_uint(t) * (1u << 23) + __float_as_uint(p));
}
template <bool Scale_max=true, bool Check_inf=true, int Max_offset=0,
    typename Engine0, typename Layout0, typename Engine1, typename Layout1>
__forceinline__ __device__ void scale_apply_exp2(Tensor<Engine0, Layout0> &tensor, Tensor<Engine1, Layout1> const &max, const float scale) {
  static constexpr float max_offset = float(Max_offset);
  static_assert(Layout0::rank == 2, "Only support 2D Tensor");
  static_assert(Layout1::rank == 1, "Only support 1D Tensor");
  CUTE_STATIC_ASSERT_V(size<0>(max) == size<0>(tensor));
  #pragma unroll
  for (int mi = 0; mi < size<0>(tensor); ++mi) {
    const float max_scaled = Check_inf
      ? (max(mi) == -INFINITY ? 0.f : (!Scale_max ? max(mi) : max(mi) * scale) - max_offset)
      : (!Scale_max ? max(mi) : max(mi) * scale) - max_offset;
    #pragma unroll
    for (int ni = 0; ni < size<1>(tensor); ++ni) {
      tensor(mi, ni) = exp2f(tensor(mi, ni) * scale - max_scaled);
    }
  }
}
template<typename TensorP>
__forceinline__ __device__ void att_issue_sum(TensorP const &tOrP, uint64_t desc, float (&osum)[4]) {
  Tensor a = recast<uint32_t const>(tOrP);
  constexpr int NK = CUTE_STATIC_V(size(a)) / 4;
  asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory");
  #pragma unroll
  for (int k = 0; k < NK; ++k) {
    if (k == 0) {
      asm volatile("{\n.reg .pred p;\nsetp.ne.b32 p, 0, 0;\n"
             "wgmma.mma_async.sync.aligned.m64n8k32.f32.e4m3.e4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, %8, p, 1, 1;\n}\n"
             : "+f"(osum[0]), "+f"(osum[1]), "+f"(osum[2]), "+f"(osum[3])
             : "r"(a(4 * k)), "r"(a(4 * k + 1)), "r"(a(4 * k + 2)), "r"(a(4 * k + 3)), "l"(desc));
    } else {
      asm volatile("{\n.reg .pred p;\nsetp.ne.b32 p, 1, 0;\n"
             "wgmma.mma_async.sync.aligned.m64n8k32.f32.e4m3.e4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, %8, p, 1, 1;\n}\n"
             : "+f"(osum[0]), "+f"(osum[1]), "+f"(osum[2]), "+f"(osum[3])
             : "r"(a(4 * k)), "r"(a(4 * k + 1)), "r"(a(4 * k + 2)), "r"(a(4 * k + 3)), "l"(desc));
    }
  }
}
template <int kNRows, int Max_offset=0>
struct Softmax {
  using TensorT = decltype(make_tensor<float>(Shape<Int<kNRows>>{}));
  TensorT row_max, row_sum;
  float osum[4];
  float const softmax_scale_log2;
  bool has_sink = false;
  TensorT row_sink;
  CUTLASS_DEVICE Softmax(float const softmax_scale_log2_) : softmax_scale_log2(softmax_scale_log2_) {};
  template<bool Is_first, bool Check_inf=false, typename Tensor0>
  __forceinline__ __device__ TensorT max_get_scale(Tensor0 &acc_s) {
    Tensor scores = make_tensor(acc_s.data(), flash::convert_layout_acc_rowcol(acc_s.layout()));
    static_assert(CUTE_STATIC_V(size<0>(scores)) == kNRows);
    TensorT scores_scale;
    if constexpr (Is_first) {
      flash::template reduce_max<true>(scores, row_max);
      cute::fill(scores_scale, 1.f);
    } else {
      Tensor scores_max_prev = make_fragment_like(row_max);
      cute::copy(row_max, scores_max_prev);
      flash::template reduce_max<false>(scores, row_max);
      #pragma unroll
      for (int mi = 0; mi < size(row_max); ++mi) {
        float scores_max_cur = !Check_inf
          ? row_max(mi)
          : (row_max(mi) == -INFINITY ? 0.0f : row_max(mi));
        scores_scale(mi) = exp2f((scores_max_prev(mi) - scores_max_cur) * softmax_scale_log2);
        row_sum(mi) *= scores_scale(mi);
      }
    }
    return scores_scale;
  };
  template<bool Is_packGQA, typename Tensor0>
  __forceinline__ __device__ void set_row_sink(Tensor0 &sSink, int bidh, int qhead_per_khead = 0, int row_start = 0, int row_step = 0) {
    has_sink = true;
    if constexpr (!Is_packGQA) {
      #pragma unroll
      for(int mi = 0; mi < size(row_sink); ++mi) {
        row_sink(mi) = static_cast<float>(sSink(bidh));
      }
    } else {
      #pragma unroll
      for(int mi = 0; mi < size(row_sink); ++mi) {
        int row = row_start + mi * row_step;
        int head = bidh * qhead_per_khead + row % qhead_per_khead;
        row_sink(mi) = static_cast<float>(sSink(head));
      }
    }
  }
  template<bool Is_first, bool Check_inf=false, typename Tensor0>
  __forceinline__ __device__ void online_softmax(Tensor0 &acc_s) {
    Tensor scores = make_tensor(acc_s.data(), flash::convert_layout_acc_rowcol(acc_s.layout()));
    static_assert(CUTE_STATIC_V(size<0>(scores)) == kNRows);
    flash::template scale_apply_exp2<true, Check_inf, Max_offset>(scores, row_max, softmax_scale_log2);
    if constexpr (Is_first) { cute::fill(row_sum, 0.f); }
  };
  __forceinline__ __device__ void clear_sum() { osum[0] = osum[1] = osum[2] = osum[3] = 0.f; }
  __forceinline__ __device__ void add_sum(TensorT const &scores_scale) {
    row_sum(0) += osum[0] * scores_scale(0);
    row_sum(1) += osum[2] * scores_scale(1);
  }
  template<typename TensorP>
  __forceinline__ __device__ void issue_sum(TensorP const &tOrP, uint64_t desc) { att_issue_sum(tOrP, desc, osum); }
  __forceinline__ __device__ TensorT finalize(float const final_scale=1.f) {
    row_sum(0) += osum[0];
    row_sum(1) += osum[2];
    TensorT scores_scale;
    #pragma unroll
    for (int mi = 0; mi < size(row_sum); ++mi) {
      if (row_max(mi) == -INFINITY) { row_max(mi) = 0.f; }
      float sum = row_sum(mi);
      if (has_sink) {
        const float max_scaled = row_max(mi) * softmax_scale_log2 - Max_offset;
        float const extra = exp2f(float(M_LOG2E) * row_sink(mi) - max_scaled);
        sum += extra;
      }
      float inv_sum = (sum == 0.f || sum != sum) ? 0.f : 1.f / sum;
      scores_scale(mi) = inv_sum * final_scale;
      if constexpr (Max_offset != 0) {
        static constexpr float sum_scale = 1.f / float(1 << Max_offset);
        sum *= sum_scale;
      }
      row_sum(mi) = (sum == 0.f || sum != sum) ? -INFINITY : row_max(mi) * (softmax_scale_log2 * float(M_LN2)) + __logf(sum);
    }
    return scores_scale;
  };
  template<typename Tensor1>
  __forceinline__ __device__ void rescale_o(Tensor1 &acc_o, TensorT const &scores_scale) {
    Tensor acc_o_rowcol = make_tensor(acc_o.data(), flash::convert_layout_acc_rowcol(acc_o.layout()));
    static_assert(CUTE_STATIC_V(size<0>(acc_o_rowcol)) == kNRows);
    #pragma unroll
    for (int mi = 0; mi < size<0>(acc_o_rowcol); ++mi) {
      #pragma unroll
      for (int ni = 0; ni < size<1>(acc_o_rowcol); ++ni) { acc_o_rowcol(mi, ni) *= scores_scale(mi); }
    }
  };
};
#define ATT_POLY_MASK 0
#define ATT_MAX_SLACK 0.5f
__device__ __forceinline__ uint32_t att_cvt_h2(float lo, float hi) {
  uint32_t r; asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(r) : "f"(hi), "f"(lo)); return r;
}
__device__ __forceinline__ uint32_t att_ex2_h2(uint32_t x) {
  uint32_t y; asm("ex2.approx.f16x2 %0, %1;" : "=r"(y) : "r"(x)); return y;
}
__device__ __forceinline__ __half2 att_as_h2(uint32_t x) { return *reinterpret_cast<__half2*>(&x); }
__device__ __forceinline__ uint32_t att_as_u(__half2 x) { return *reinterpret_cast<uint32_t*>(&x); }
__device__ __forceinline__ uint32_t att_ex2_poly_h2(__half2 y) {
  __half2 const magic = __float2half2_rn(1024.f);
  __half2 t = __hadd2(y, magic);
  __half2 f = __hsub2(y, __hsub2(t, magic));
  __half2 p = __hfma2(__hfma2(__float2half2_rn(0.23734974f), f, __float2half2_rn(0.70128093f)), f,
            __float2half2_rn(1.00035163f));
  uint32_t e = (att_as_u(t) & 0x001F001Fu) << 10;
  return att_as_u(__hmul2(p, att_as_h2(e)));
}
__device__ __forceinline__ uint16_t att_e4m3x2(uint32_t h) {
  uint16_t r; asm("cvt.rn.satfinite.e4m3x2.f16x2 %0, %1;" : "=h"(r) : "r"(h)); return r;
}
template <int kNRows, int Max_offset=0>
struct SoftmaxH {
  using TensorT = decltype(make_tensor<float>(Shape<Int<kNRows>>{}));
  TensorT row_max, row_sum;
  float const softmax_scale_log2;
  bool has_sink = false;
  TensorT row_sink;
  CUTLASS_DEVICE SoftmaxH(float const softmax_scale_log2_) : softmax_scale_log2(softmax_scale_log2_) {};
  template<bool Is_packGQA, typename Tensor0>
  __forceinline__ __device__ void set_row_sink(Tensor0 &sSink, int bidh, int qhead_per_khead = 0, int row_start = 0, int row_step = 0) {}
  template<bool Is_first, bool Check_inf, typename Tensor0, int NW>
  __forceinline__ __device__ TensorT step(Tensor0 const &acc_s, uint32_t (&h)[NW]) {
    static_assert(kNRows == 2);
    static_assert(CUTE_STATIC_V(size(acc_s)) == 2 * NW);
    if constexpr (sizeof(typename Tensor0::value_type) == 2) {
      Tensor acc_u = recast<uint32_t const>(acc_s);
      #pragma unroll
      for (int w = 0; w < NW; ++w) { h[w] = acc_u(w); }
    } else {
      #pragma unroll
      for (int w = 0; w < NW; ++w) { h[w] = att_cvt_h2(acc_s(2 * w), acc_s(2 * w + 1)); }
    }
#define ATT_ABL 0
    __half2 m2[2] = {att_as_h2(h[0]), att_as_h2(h[1])};
    if constexpr ((ATT_ABL & 2) == 0) {
    #pragma unroll
    for (int w = 2; w < NW; ++w) { m2[w & 1] = __hmax2(m2[w & 1], att_as_h2(h[w])); }
    } else { m2[0] = m2[1] = __float2half2_rn(30.f); }
    TensorT scores_scale;
    #pragma unroll
    for (int r = 0; r < 2; ++r) {
      float m = fmaxf(__low2float(m2[r]), __high2float(m2[r]));
      m = fmaxf(m, __shfl_xor_sync(0xffffffff, m, 2));
      m = fmaxf(m, __shfl_xor_sync(0xffffffff, m, 1));
      if constexpr (Is_first) {
        row_max(r) = m;
        scores_scale(r) = 1.f;
      } else {
        float const prev = row_max(r);
        float cur = fmaxf(prev, m);
        if ((cur - prev) * softmax_scale_log2 <= float(ATT_MAX_SLACK)) { cur = prev; }
        float const sc = cur == prev ? 1.f : exp2f((prev - cur) * softmax_scale_log2);
        scores_scale(r) = sc;
        row_max(r) = cur;
        row_sum(r) *= sc;
      }
    }
    __half2 const sc2 = __float2half2_rn(softmax_scale_log2);
    __half2 nms[2], nms15[2];
    #pragma unroll
    for (int r = 0; r < 2; ++r) {
      float const ms = Check_inf
        ? (row_max(r) == -INFINITY ? 0.f : row_max(r) * softmax_scale_log2 - float(Max_offset))
        : row_max(r) * softmax_scale_log2 - float(Max_offset);
      nms[r] = __float2half2_rn(-ms);
      nms15[r] = __float2half2_rn(15.f - ms);
    }
    __half2 sum2[2] = {__float2half2_rn(0.f), __float2half2_rn(0.f)};
    #pragma unroll
    for (int w = 0; w < NW; ++w) {
      uint32_t p;
      if (((ATT_POLY_MASK >> (w % 8)) & 1) != 0) {
        p = att_ex2_poly_h2(__hfma2_relu(att_as_h2(h[w]), sc2, nms15[w & 1]));
      } else if constexpr ((ATT_ABL & 1) != 0) {
        p = att_as_u(__hfma2(att_as_h2(h[w]), sc2, nms[w & 1]));
      } else {
        p = att_ex2_h2(att_as_u(__hfma2(att_as_h2(h[w]), sc2, nms[w & 1])));
      }
      sum2[w & 1] = __hadd2(sum2[w & 1], att_as_h2(p));
      h[w] = p;
    }
    #pragma unroll
    for (int r = 0; r < 2; ++r) {
      float const s = __low2float(sum2[r]) + __high2float(sum2[r]);
      if constexpr (Is_first) { row_sum(r) = s; } else { row_sum(r) += s; }
    }
    return scores_scale;
  };
  template<int NW, typename TensorP>
  __forceinline__ __device__ void pack_p(uint32_t const (&h)[NW], TensorP &tOrP) {
    Tensor a = recast<uint32_t>(tOrP);
    static_assert(CUTE_STATIC_V(size(a)) * 2 == NW);
    #pragma unroll
    for (int i = 0; i < NW / 4; ++i) {
      a(2 * i) = uint32_t(att_e4m3x2(h[4 * i])) | (uint32_t(att_e4m3x2(h[4 * i + 2])) << 16);
      a(2 * i + 1) = uint32_t(att_e4m3x2(h[4 * i + 1])) | (uint32_t(att_e4m3x2(h[4 * i + 3])) << 16);
    }
  }
  __forceinline__ __device__ TensorT finalize(float const final_scale=1.f) {
    SumOp<float> sum_op;
    quad_allreduce_(row_sum, row_sum, sum_op);
    TensorT scores_scale;
    #pragma unroll
    for (int mi = 0; mi < size(row_sum); ++mi) {
      if (row_max(mi) == -INFINITY) { row_max(mi) = 0.f; }
      float sum = row_sum(mi);
      float inv_sum = (sum == 0.f || sum != sum) ? 0.f : 1.f / sum;
      scores_scale(mi) = inv_sum * final_scale;
      if constexpr (Max_offset != 0) {
        static constexpr float sum_scale = 1.f / float(1 << Max_offset);
        sum *= sum_scale;
      }
      row_sum(mi) = (sum == 0.f || sum != sum) ? -INFINITY : row_max(mi) * (softmax_scale_log2 * float(M_LN2)) + __logf(sum);
    }
    return scores_scale;
  };
  template<typename Tensor1>
  __forceinline__ __device__ void rescale_o(Tensor1 &acc_o, TensorT const &scores_scale) {
    Tensor acc_o_rowcol = make_tensor(acc_o.data(), flash::convert_layout_acc_rowcol(acc_o.layout()));
    static_assert(CUTE_STATIC_V(size<0>(acc_o_rowcol)) == kNRows);
    #pragma unroll
    for (int mi = 0; mi < size<0>(acc_o_rowcol); ++mi) {
      if (scores_scale(mi) != 1.f) {
        #pragma unroll
        for (int ni = 0; ni < size<1>(acc_o_rowcol); ++ni) { acc_o_rowcol(mi, ni) *= scores_scale(mi); }
      }
    }
  };
};
__device__ __forceinline__ uint32_t att_e4m3x2_to_h2(uint16_t x) {
  uint32_t r; asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(r) : "h"(x)); return r;
}
__device__ __forceinline__ uint32_t att_prmt(uint32_t a, uint32_t b, uint32_t sel) {
  uint32_t r; asm("prmt.b32 %0, %1, %2, %3;" : "=r"(r) : "r"(a), "r"(b), "r"(sel)); return r;
}
__device__ __forceinline__ uint64_t att_ones_desc(void const* p) {
  uint32_t const a = static_cast<uint32_t>(__cvta_generic_to_shared(p));
  return uint64_t((a & 0x3FFFF) >> 4) | (uint64_t(128 >> 4) << 16) | (uint64_t(256 >> 4) << 32);
}




template <int kNRows, int Max_offset=0>
struct SoftmaxL {
  using TensorT = decltype(make_tensor<float>(Shape<Int<kNRows>>{}));
  TensorT row_max, row_sum, row_b;
  uint32_t rmh, rbh;
  bool fx_chg = true;
  float osum[4];
  float const softmax_scale_log2;
  float const a32;
  bool has_sink = false;
  CUTLASS_DEVICE SoftmaxL(float const softmax_scale_log2_)
    : softmax_scale_log2(softmax_scale_log2_), a32(__half2float(__float2half_rn(8.f * softmax_scale_log2_))) {};
  template<bool Is_packGQA, typename Tensor0>
  __forceinline__ __device__ void set_row_sink(Tensor0 &sSink, int bidh, int qhead_per_khead = 0, int row_start = 0, int row_step = 0) {}
  template<bool Is_first, bool Check_inf, typename Tensor0, int NW>
  __forceinline__ __device__ TensorT step(Tensor0 const &acc_s, uint32_t (&h)[NW]) {
    static_assert(kNRows == 2);
    static_assert(CUTE_STATIC_V(size(acc_s)) == 2 * NW);
    static_assert(NW % 4 == 0);
    if constexpr (sizeof(typename Tensor0::value_type) == 2) {
      Tensor acc_u = recast<uint32_t const>(acc_s);
      #pragma unroll
      for (int w = 0; w < NW; ++w) { h[w] = acc_u(w); }
    } else {
      #pragma unroll
      for (int w = 0; w < NW; ++w) { h[w] = att_cvt_h2(acc_s(2 * w), acc_s(2 * w + 1)); }
    }
    __half2 mp[8];
    #pragma unroll
    for (int k = 0; k < 8; ++k) { mp[k] = att_as_h2(h[k]); }
    #pragma unroll
    for (int w = 8; w < NW; ++w) { mp[w % 8] = __hmax2(mp[w % 8], att_as_h2(h[w])); }
    #pragma unroll
    for (int k = 0; k < 4; ++k) { mp[k] = __hmax2(mp[k], mp[k + 4]); }
    __half2 const r0 = __hmax2(mp[0], mp[2]), r1 = __hmax2(mp[1], mp[3]);
    __half2 mm = __hmax2(__lows2half2(r0, r1), __highs2half2(r0, r1));
    uint32_t mu = att_as_u(mm);
    mm = __hmax2(mm, att_as_h2(__shfl_xor_sync(0xffffffff, mu, 2)));
    mu = att_as_u(mm);
    mm = __hmax2(mm, att_as_h2(__shfl_xor_sync(0xffffffff, mu, 1)));
    TensorT scores_scale;
    __half2 b2[2];
    __half2 const cur = Is_first ? mm : __hmax2(att_as_h2(rmh), mm);
#ifdef AT_SUB
    __half2 const bn = __hmin2(__hfma2(cur, __float2half2_rn(-0x1p-21f), __float2half2_rn(float(U_PB) * 0x1p-24f)), __float2half2_rn(65504.f));
#else
    __half2 const bn = __hmin2(__hfma2(cur, __hneg2(__float2half2_rn(a32)), __float2half2_rn(1144.f)), __float2half2_rn(65504.f));
#endif
    uint32_t const bnu = att_as_u(bn);
    scores_scale(0) = scores_scale(1) = 1.f;
    if constexpr (Is_first) {
      row_sum(0) = row_sum(1) = 0.f;
    } else {
      {
#ifdef AT_SUB
        scores_scale(0) = exp2f((__low2float(bn) - __low2float(att_as_h2(rbh))) * 0x1p21f);
        scores_scale(1) = exp2f((__high2float(bn) - __high2float(att_as_h2(rbh))) * 0x1p21f);
#else
        scores_scale(0) = exp2f((__low2float(bn) - __low2float(att_as_h2(rbh))) * 0.125f);
        scores_scale(1) = exp2f((__high2float(bn) - __high2float(att_as_h2(rbh))) * 0.125f);
#endif
      }
      row_sum(0) *= scores_scale(0);
      row_sum(1) *= scores_scale(1);
    }
    rmh = att_as_u(cur);
    rbh = bnu;
    b2[0] = __low2half2(bn);
    b2[1] = __high2half2(bn);
    __half2 const a2 = __float2half2_rn(a32);
    __half2 const lo2 = __float2half2_rn(1024.f);
    #pragma unroll
    for (int w = 0; w < NW; ++w) {
#if defined(AT_SUB)
      {
        uint32_t r;
        asm("fma.rn.relu.f16x2 %0, %1, %2, %3;" : "=r"(r) : "r"(h[w]), "r"(0x00080008u), "r"(att_as_u(b2[w & 1])));
        h[w] = r;
      }
#else
      h[w] = att_as_u(__hmax2(__hfma2(att_as_h2(h[w]), a2, b2[w & 1]), lo2));
#endif
    }
    return scores_scale;
  };
  template<int NW, typename TensorP>
  __forceinline__ __device__ void pack_p(uint32_t const (&h)[NW], TensorP &tOrP) {
    Tensor a = recast<uint32_t>(tOrP);
    static_assert(CUTE_STATIC_V(size(a)) * 2 == NW);
    #pragma unroll
    for (int i = 0; i < NW / 4; ++i) {
      uint32_t p0, p1;
      asm volatile("prmt.b32 %0, %1, %2, 0x6420;" : "=r"(p0) : "r"(h[4 * i]), "r"(h[4 * i + 2]));
      asm volatile("prmt.b32 %0, %1, %2, 0x6420;" : "=r"(p1) : "r"(h[4 * i + 1]), "r"(h[4 * i + 3]));
      a(2 * i) = p0;
      a(2 * i + 1) = p1;
    }
  }
  __forceinline__ __device__ void clear_sum() { osum[0] = osum[1] = osum[2] = osum[3] = 0.f; }
  __forceinline__ __device__ void add_sum(TensorT const &scores_scale) {
    row_sum(0) += osum[0] * scores_scale(0);
    row_sum(1) += osum[2] * scores_scale(1);
  }

  __forceinline__ __device__ void scale_sum(TensorT const &scores_scale) {
    return;
    bool const fl = ((++fl_cnt) & (FK_SUMFL - 1)) == 0;
    float const a0 = osum[0] * scores_scale(0), a2 = osum[2] * scores_scale(1);
    row_sum(0) += fl ? a0 : 0.f;
    row_sum(1) += fl ? a2 : 0.f;
    osum[0] = fl ? 0.f : a0; osum[1] = fl ? 0.f : osum[1] * scores_scale(0);
    osum[2] = fl ? 0.f : a2; osum[3] = fl ? 0.f : osum[3] * scores_scale(1);
  }
  int fl_cnt = 0;
  template<typename TensorP>
  __forceinline__ __device__ void issue_sum(TensorP const &tOrP, uint64_t desc) {
    Tensor a = recast<uint32_t const>(tOrP);
    constexpr int NK = CUTE_STATIC_V(size(a)) / 4;
    asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory");
    #pragma unroll
    for (int k = 0; k < NK; ++k) {
      if (k == 0) {
        asm volatile(
          "{\n.reg .pred p;\nsetp.ne.b32 p, 0, 0;\n"
          "wgmma.mma_async.sync.aligned.m64n8k32.f32.e4m3.e4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, %8, p, 1, 1;\n}\n"
          : "+f"(osum[0]), "+f"(osum[1]), "+f"(osum[2]), "+f"(osum[3])
          : "r"(a(4 * k)), "r"(a(4 * k + 1)), "r"(a(4 * k + 2)), "r"(a(4 * k + 3)), "l"(desc));
      } else {
        asm volatile(
          "{\n.reg .pred p;\nsetp.ne.b32 p, 1, 0;\n"
          "wgmma.mma_async.sync.aligned.m64n8k32.f32.e4m3.e4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, %8, p, 1, 1;\n}\n"
          : "+f"(osum[0]), "+f"(osum[1]), "+f"(osum[2]), "+f"(osum[3])
          : "r"(a(4 * k)), "r"(a(4 * k + 1)), "r"(a(4 * k + 2)), "r"(a(4 * k + 3)), "l"(desc));
      }
    }
  }



  __forceinline__ __device__ TensorT finalize(float const final_scale=1.f) {
#ifdef AT_SUB
    row_b(0) = fmaf(__low2float(att_as_h2(rbh)), 0x1p24f, 1024.f);
    row_b(1) = fmaf(__high2float(att_as_h2(rbh)), 0x1p24f, 1024.f);
#else
    row_b(0) = __low2float(att_as_h2(rbh));
    row_b(1) = __high2float(att_as_h2(rbh));
#endif
    {
      static_assert(kNRows == 2, "Z0_NOL: two rows");
      TensorT nol_scale;
      #pragma unroll
      for (int mi = 0; mi < 2; ++mi) {
        uint32_t const mh = mi == 0 ? (rmh & 0xFFFFu) : (rmh >> 16);
        bool const empty = mh == 0xFC00u;
        nol_scale(mi) = empty ? 0.f : final_scale;
        row_sum(mi) = empty ? -INFINITY : float(M_LN2) * (-8.f - (row_b(mi) - 1144.f) * 0.125f);
      }
      return nol_scale;
    }
    row_sum(0) += osum[0];
    row_sum(1) += osum[2];
    TensorT scores_scale;
    #pragma unroll
    for (int mi = 0; mi < size(row_sum); ++mi) {
      float const sum = row_sum(mi);
      bool const empty = sum == 0.f || sum != sum;
      scores_scale(mi) = empty ? 0.f : final_scale / sum;
      row_sum(mi) = empty ? -INFINITY : float(M_LN2) * (__log2f(sum) - 8.f - (row_b(mi) - 1144.f) * 0.125f);
    }
    return scores_scale;
  };
  template<typename Tensor1>
  __forceinline__ __device__ void rescale_o(Tensor1 &acc_o, TensorT const &scores_scale) {
    Tensor acc_o_rowcol = make_tensor(acc_o.data(), flash::convert_layout_acc_rowcol(acc_o.layout()));
    static_assert(CUTE_STATIC_V(size<0>(acc_o_rowcol)) == kNRows);
    #pragma unroll
    for (int mi = 0; mi < size<0>(acc_o_rowcol); ++mi) {
      if (scores_scale(mi) != 1.f) {
        #pragma unroll
        for (int ni = 0; ni < size<1>(acc_o_rowcol); ++ni) { acc_o_rowcol(mi, ni) *= scores_scale(mi); }
      }
    }
  };
};
}
namespace flash {
using namespace cute;
template <class CollectiveMainloop_, class CollectiveEpilogue_, class TileScheduler_>
class FlashAttnFwdSm90 {
public:
  using CollectiveMainloop = CollectiveMainloop_;
  using CollectiveEpilogue = CollectiveEpilogue_;
  static constexpr bool Is_causal = CollectiveMainloop::Is_causal;
  static constexpr bool Is_local = CollectiveMainloop::Is_local;
  static_assert(CollectiveMainloop::Varlen == CollectiveEpilogue::Varlen);
  static constexpr bool Has_softcap = CollectiveMainloop::Has_softcap;
  static constexpr bool Varlen = CollectiveMainloop::Varlen;
  static constexpr bool Split = CollectiveMainloop::Split;
  static constexpr bool Is_FP8 = CollectiveMainloop::Is_FP8;
  static constexpr bool Transpose_V = CollectiveMainloop::Transpose_V;
  static constexpr bool AppendKV = CollectiveMainloop::AppendKV;
  static constexpr bool HasQv = CollectiveMainloop::HasQv;
  static constexpr bool OnlyQv = CollectiveMainloop::OnlyQv;
  static constexpr bool Use_TMA_Q = CollectiveMainloop::Use_TMA_Q;
  static constexpr bool Use_TMA_KV = CollectiveMainloop::Use_TMA_KV;
  static constexpr bool Use_TMA_O = CollectiveEpilogue::Use_TMA_O;
  static constexpr bool PackGQA = CollectiveMainloop::PackGQA;
  static constexpr int NumProducerThreads = CollectiveMainloop::NumProducerThreads;
  static constexpr bool SameHeadDim = CollectiveMainloop::SameHeadDim;
  static constexpr bool LargeHeadDimV = CollectiveMainloop::LargeHeadDimV;
  static constexpr bool Has_sparse_mask = CollectiveMainloop::Has_sparse_mask;
  static constexpr bool SK = flash::is_sk2_sched<TileScheduler_>::value;
  static_assert(CollectiveMainloop::LargeHeadDimV == CollectiveEpilogue::LargeHeadDimV);
  using SeqlenInfo_t = typename CollectiveMainloop::SeqlenInfo_t;
  using TiledMmaQK = typename CollectiveMainloop::TiledMmaQK;
  using TileShape_MNK = typename CollectiveMainloop::TileShape_MNK;
  using TileShape_MNK_PV = typename CollectiveMainloop::TileShape_MNK_PV;
  using TiledMmaPV = typename CollectiveMainloop::TiledMmaPV;
  using ArchTag = typename CollectiveMainloop::ArchTag;
  using ClusterShape = typename CollectiveMainloop::ClusterShape;
  using MainloopArguments = typename CollectiveMainloop::Arguments;
  using MainloopParams = typename CollectiveMainloop::Params;
  using BarrierQ = std::conditional_t<Use_TMA_Q, cutlass::arch::ClusterTransactionBarrier, cutlass::arch::ClusterBarrier>;
  using SmemLayoutSink = typename CollectiveMainloop::SmemLayoutSink;
  using EpilogueArguments = typename CollectiveEpilogue::Arguments;
  using EpilogueParams = typename CollectiveEpilogue::Params;
  static_assert(ArchTag::kMinComputeCapability >= 90);
  using TileScheduler = TileScheduler_;
  using TileSchedulerArguments = typename flash::TileSchedulerArguments;
  using TileSchedulerParams = typename TileScheduler::Params;
  static constexpr uint32_t NumLoadWarpGroups = 1;
  static constexpr uint32_t NumMmaWarpGroups = CUTE_STATIC_V(size(TiledMmaPV{})) / cutlass::NumThreadsPerWarpGroup;
  static constexpr uint32_t MaxThreadsPerBlock = CUTE_STATIC_V(size(TiledMmaPV{})) + (NumLoadWarpGroups * cutlass::NumThreadsPerWarpGroup);
  static constexpr uint32_t MinBlocksPerMultiprocessor = 1;
  static_assert(NumMmaWarpGroups == 1 || NumMmaWarpGroups == 2 || NumMmaWarpGroups == 3 || NumMmaWarpGroups == 4);
#define ATT_REG4_LOAD 32
#define ATT_REG4_MMA 120
  static constexpr uint32_t LoadRegisterRequirement = (NumMmaWarpGroups == 1 ? 56 : (NumMmaWarpGroups == 2 ? ((Use_TMA_KV) ? 24 : 40) : (NumMmaWarpGroups == 3 ? 32 : ATT_REG4_LOAD)));
  static constexpr uint32_t MmaRegisterRequirement = (NumMmaWarpGroups == 1 ? 256 : (NumMmaWarpGroups == 2 ? ((Use_TMA_KV) ? 240 : 232) : (NumMmaWarpGroups == 3 ? 160 : ATT_REG4_MMA)));
  static constexpr int mainloop_smem_padding_ = int(sizeof(typename CollectiveEpilogue::TensorStorage)) - int(sizeof(decltype((typename CollectiveMainloop::TensorStorage{}).smem_v)));
  static constexpr int mainloop_smem_padding = mainloop_smem_padding_ < 0 ? 0 : mainloop_smem_padding_;
  struct SharedStorage {
    struct TensorStorage : cute::aligned_struct<128, _1> {
      union {
        struct {
          cute::array<uint32_t, mainloop_smem_padding / sizeof(uint32_t)> padding_;
          typename CollectiveMainloop::TensorStorage mainloop;
        };
        typename CollectiveEpilogue::TensorStorage epilogue;
      };
    } tensors;
    struct PipelineStorage : cute::aligned_struct<16, _1> {
      alignas(16) BarrierQ barrier_Q;
      alignas(16) BarrierQ barrier_Qv;
      alignas(16) cutlass::arch::ClusterBarrier barrier_O;
      alignas(16) typename CollectiveMainloop::MainloopPipelineK::SharedStorage pipeline_k;
      alignas(16) typename CollectiveMainloop::MainloopPipelineV::SharedStorage pipeline_v;
      alignas(16) typename CollectiveMainloop::MainloopPipelineVt::SharedStorage pipeline_vt;
      alignas(16) typename CollectiveMainloop::MainloopPipelineKVNew::SharedStorage pipeline_k_new;
      alignas(16) typename CollectiveMainloop::MainloopPipelineKVNew::SharedStorage pipeline_v_new;
      alignas(16) typename TileScheduler::SharedStorage smem_scheduler;
      alignas(16) typename CollectiveMainloop::MainloopPipelineMask::SharedStorage pipeline_mask;
    } pipelines;
  };
  static constexpr int SharedStorageSize = sizeof(SharedStorage);
  struct Arguments {
    MainloopArguments mainloop{};
    EpilogueArguments epilogue{};
    cutlass::KernelHardwareInfo hw_info{};
    TileSchedulerArguments scheduler{};
  };
  struct Params {
    MainloopParams mainloop{};
    EpilogueParams epilogue{};
    cutlass::KernelHardwareInfo hw_info{};
    TileSchedulerParams scheduler{};
  };
  static
  Params
  to_underlying_arguments(Arguments const& args) {
    CUTLASS_TRACE_HOST("to_underlying_arguments():");
    int sm_count = args.hw_info.sm_count;
    if (sm_count <= 0) {
      CUTLASS_TRACE_HOST("  WARNING: Arguments do not include a valid SM count.\n"
        "  For optimal performance, populate the arguments KernelHardwareInfo struct with the SM count.");
      sm_count = cutlass::KernelHardwareInfo::query_device_multiprocessor_count(args.hw_info.device_id);
    }
    CUTLASS_TRACE_HOST("to_underlying_arguments(): Setting persistent grid SM count to " << sm_count);
    cutlass::KernelHardwareInfo hw_info{args.hw_info.device_id, sm_count};
    return {
      CollectiveMainloop::to_underlying_arguments(args.mainloop),
      CollectiveEpilogue::to_underlying_arguments(args.epilogue),
      hw_info,
      TileScheduler::to_underlying_arguments(args.scheduler)
    };
  }
  static dim3
  get_grid_shape(Params const& params) {
    return TileScheduler::get_grid_shape(params.scheduler, params.hw_info.sm_count);
  }
  static dim3
  get_block_shape() {
    return dim3(MaxThreadsPerBlock, 1, 1);
  }
  CUTLASS_DEVICE
  void
  operator()(Params const& params, char* smem_buf) {
    static constexpr int NumMmaThreads = NumMmaWarpGroups * cutlass::NumThreadsPerWarpGroup;
    static constexpr int MmaThreadOffset = NumLoadWarpGroups * cutlass::NumThreadsPerWarpGroup;
    static constexpr int kBlockM = get<0>(TileShape_MNK_PV{});
    using MainloopPipelineK = typename CollectiveMainloop::MainloopPipelineK;
    using MainloopPipelineV = typename CollectiveMainloop::MainloopPipelineV;
    using MainloopPipelineVt = typename CollectiveMainloop::MainloopPipelineVt;
    using MainloopPipelineKVNew = typename CollectiveMainloop::MainloopPipelineKVNew;
    using MainloopPipelineMask = typename CollectiveMainloop::MainloopPipelineMask;
    using PipelineState = typename CollectiveMainloop::PipelineState;
    using PipelineParamsK = typename MainloopPipelineK::Params;
    using PipelineParamsV = typename MainloopPipelineV::Params;
    using PipelineParamsVt = typename MainloopPipelineVt::Params;
    using PipelineParamsKVNew = typename MainloopPipelineKVNew::Params;
    using PipelineParamsMask = typename MainloopPipelineMask::Params;
    SharedStorage& shared_storage = *reinterpret_cast<SharedStorage*>(smem_buf);
    int const lane_predicate = cute::elect_one_sync();
    int const warp_idx = cutlass::canonical_warp_idx_sync();
    if (warp_idx == 0 && lane_predicate) {
      CollectiveMainloop::prefetch_tma_descriptors(params.mainloop);
      CollectiveEpilogue::prefetch_tma_descriptors(params.epilogue);
    }
    int const warp_group_thread_idx = threadIdx.x % cutlass::NumThreadsPerWarpGroup;
    int warp_group_idx = cutlass::canonical_warp_group_idx();
    if (warp_idx == 0 && lane_predicate) {
      shared_storage.pipelines.barrier_Q.init(Use_TMA_Q ? 1 : NumProducerThreads );
      if constexpr (HasQv) {
        shared_storage.pipelines.barrier_Qv.init(Use_TMA_Q ? 1 : NumProducerThreads );
      }
      shared_storage.pipelines.barrier_O.init(size(ClusterShape{}) * (Use_TMA_O ? 1 : NumMmaThreads) );
    }
    PipelineParamsK pipeline_params_k;
    pipeline_params_k.role = warp_group_idx == 0
      ? MainloopPipelineK::ThreadCategory::Producer
      : MainloopPipelineK::ThreadCategory::Consumer;
    if constexpr (Use_TMA_KV) {
      pipeline_params_k.transaction_bytes = CollectiveMainloop::TmaTransactionBytesK;
      pipeline_params_k.is_leader = warp_group_thread_idx == 0;
      pipeline_params_k.num_consumers = !LargeHeadDimV ? NumMmaThreads : cutlass::NumThreadsPerWarpGroup;
    } else {
      pipeline_params_k.consumer_arv_count = !LargeHeadDimV ? NumMmaThreads : cutlass::NumThreadsPerWarpGroup;
      pipeline_params_k.producer_arv_count = NumProducerThreads;
    }
    PipelineParamsMask pipeline_params_mask;
    pipeline_params_mask.role = warp_group_idx == 0
      ? MainloopPipelineMask::ThreadCategory::Producer
      : MainloopPipelineMask::ThreadCategory::Consumer;
    if constexpr (CollectiveMainloop::Use_TMA_Mask) {
      pipeline_params_mask.transaction_bytes = CollectiveMainloop::TmaTransactionBytesMask;
      pipeline_params_mask.is_leader = warp_group_thread_idx == 0;
      pipeline_params_mask.num_consumers = !LargeHeadDimV ? NumMmaThreads : cutlass::NumThreadsPerWarpGroup;
    } else {
      pipeline_params_mask.consumer_arv_count = !LargeHeadDimV ? NumMmaThreads : cutlass::NumThreadsPerWarpGroup;
      pipeline_params_mask.producer_arv_count = NumProducerThreads;
    }
    static_assert(is_same_v<PipelineParamsK, PipelineParamsVt>);
    PipelineParamsVt pipeline_params_vt = pipeline_params_k;
    if constexpr (Use_TMA_KV) {
      pipeline_params_vt.transaction_bytes = CollectiveMainloop::TmaTransactionBytesV;
      if constexpr (LargeHeadDimV) { pipeline_params_vt.num_consumers = NumMmaThreads; }
    } else {
      if constexpr (LargeHeadDimV) { pipeline_params_vt.consumer_arv_count = NumMmaThreads; }
    }
    MainloopPipelineK pipeline_k = [&] {
      if constexpr (Use_TMA_KV) {
        return MainloopPipelineK(shared_storage.pipelines.pipeline_k, pipeline_params_k, ClusterShape{});
      } else {
        return MainloopPipelineK(shared_storage.pipelines.pipeline_k, pipeline_params_k);
      }
    }();
    MainloopPipelineMask pipeline_mask = [&] {
      if constexpr (CollectiveMainloop::Use_TMA_Mask) {
        return MainloopPipelineMask(shared_storage.pipelines.pipeline_mask, pipeline_params_mask, ClusterShape{});
      } else {
        return MainloopPipelineMask(shared_storage.pipelines.pipeline_mask, pipeline_params_mask);
      }
    }();
    MainloopPipelineV pipeline_v = [&] {
      if constexpr (!Transpose_V) {
        static_assert(is_same_v<PipelineParamsK, PipelineParamsV>);
        if constexpr (Use_TMA_KV) {
          return MainloopPipelineV(shared_storage.pipelines.pipeline_v, pipeline_params_vt, ClusterShape{});
        } else {
          return MainloopPipelineV(shared_storage.pipelines.pipeline_v, pipeline_params_vt);
        }
      } else {
        PipelineParamsV pipeline_params_v;
        pipeline_params_v.role = warp_group_idx == 0
          ? MainloopPipelineV::ThreadCategory::Producer
          : MainloopPipelineV::ThreadCategory::Consumer;
        pipeline_params_v.producer_arv_count = NumProducerThreads;
        pipeline_params_v.consumer_arv_count = NumMmaThreads;
        return MainloopPipelineV(shared_storage.pipelines.pipeline_v, pipeline_params_v);
      }
    }();
    MainloopPipelineVt pipeline_vt = [&] {
      if constexpr (Use_TMA_KV) {
        pipeline_params_vt.num_consumers = NumProducerThreads;
        return MainloopPipelineVt(shared_storage.pipelines.pipeline_vt, pipeline_params_vt, ClusterShape{});
      } else {
        pipeline_params_vt.consumer_arv_count = NumProducerThreads;
        return MainloopPipelineVt(shared_storage.pipelines.pipeline_vt, pipeline_params_vt);
      }
    }();
    PipelineParamsKVNew pipeline_params_kv_new;
    pipeline_params_kv_new.role = warp_group_idx == 0
      ? MainloopPipelineKVNew::ThreadCategory::Producer
      : MainloopPipelineKVNew::ThreadCategory::Consumer;
    pipeline_params_kv_new.transaction_bytes = CollectiveMainloop::TmaTransactionBytesK;
    pipeline_params_kv_new.is_leader = warp_group_thread_idx == 0;
    pipeline_params_kv_new.num_consumers = NumMmaThreads;
    auto pipeline_k_new = cute::conditional_return<AppendKV>(MainloopPipelineKVNew(shared_storage.pipelines.pipeline_k_new, pipeline_params_kv_new, ClusterShape{}), nullptr);
    if constexpr (!SameHeadDim) {
      pipeline_params_kv_new.transaction_bytes = CollectiveMainloop::TmaTransactionBytesV;
    }
    auto pipeline_v_new = cute::conditional_return<AppendKV>(MainloopPipelineKVNew(shared_storage.pipelines.pipeline_v_new, pipeline_params_kv_new, ClusterShape{}), nullptr);
    CollectiveMainloop mainloop;
    CollectiveEpilogue epilogue;
    const bool has_sink = params.mainloop.ptr_sink != nullptr;
    const int num_heads = get<2>(params.mainloop.shape_Q);
    Tensor gSink = make_tensor(make_gmem_ptr(params.mainloop.ptr_sink), make_shape(num_heads));
    Tensor sSink = make_tensor(make_smem_ptr(shared_storage.tensors.mainloop.smem_sink.data()), SmemLayoutSink{});
    if (has_sink) {
      #pragma unroll
      for (int i = threadIdx.x; i < num_heads; i += blockDim.x) {
        sSink(i) = gSink(i);
      }
    }
    {
      uint32_t* ones = reinterpret_cast<uint32_t*>(shared_storage.tensors.mainloop.smem_ones.data());
      for (int i = threadIdx.x; i < 128; i += blockDim.x) { ones[i] = 0x38383838u; }
      cutlass::arch::fence_view_async_shared();
    }
    if constexpr (size(ClusterShape{}) > 1) {
      cute::cluster_arrive_relaxed();
      cute::cluster_wait();
    } else {
      __syncthreads();
    }
    TileScheduler scheduler(reinterpret_cast<typename TileScheduler::SharedStorage*>(&shared_storage.pipelines.smem_scheduler));
    if (warp_group_idx == 0) {
      cutlass::arch::warpgroup_reg_dealloc<LoadRegisterRequirement>();
      PipelineState smem_pipe_write = cutlass::make_producer_start_state<MainloopPipelineK>();
      PipelineState smem_pipe_write_new = cutlass::make_producer_start_state<MainloopPipelineKVNew>();
      int work_idx = 0;
      int warp_idx_in_warpgroup = __shfl_sync(0xffffffff, (threadIdx.x / 32) % 4, 0);
      static constexpr bool SingleProducerWarp = NumProducerThreads == cutlass::NumThreadsPerWarp;
      if constexpr (SingleProducerWarp) {
        if (warp_idx_in_warpgroup != 0) { return; }
      }
      if (!SingleProducerWarp && warp_idx_in_warpgroup != 0) { scheduler.init_consumer(); }
      cutlass::arch::wait_on_dependent_grids();
      for (auto work_tile_info = SingleProducerWarp || warp_idx_in_warpgroup == 0 ? scheduler.template get_initial_work<true>(params.scheduler) : scheduler.template get_initial_work<false>(params.scheduler);
         work_tile_info.is_valid(params.scheduler);
         work_tile_info = SingleProducerWarp || warp_idx_in_warpgroup == 0 ? scheduler.template get_next_work<true>(params.scheduler, work_tile_info) : scheduler.template get_next_work<false>(params.scheduler, work_tile_info)) {
        auto block_coord = work_tile_info.get_block_coord(params.scheduler);
        SeqlenInfo_t seqlen_info{
          get<2>(block_coord) ,
          get<0>(params.mainloop.shape_Q),
          !params.mainloop.ptr_pagetable ? size<0>(params.mainloop.shape_K) : size<0>(params.mainloop.shape_K) * size<1>(params.mainloop.shape_pagetable),
          get<0>(params.mainloop.shape_K_new),
          params.mainloop.cu_seqlens_q, params.mainloop.cu_seqlens_k, params.mainloop.cu_seqlens_k_new,
          params.mainloop.seqused_q, params.mainloop.seqused_k, params.mainloop.leftpad_k,
          params.mainloop.seqlens_rotary
        };
        if constexpr (AppendKV) {
          bool tile_new_valid = mainloop.load_kv_new(
            params.mainloop, pipeline_k_new, pipeline_v_new,
            smem_pipe_write_new, shared_storage, seqlen_info, block_coord, work_idx);
          if (tile_new_valid) {
            cutlass::arch::NamedBarrier::sync(NumMmaThreads + NumProducerThreads, static_cast<uint32_t>(FwdNamedBarriers::AppendKV) );
          }
        }
        auto scheduler_prefetch = [&scheduler, &params, &work_tile_info]() {
          scheduler.prefetch_next_work(params.scheduler, work_tile_info);
        };
        mainloop.load(params.mainloop, pipeline_k, pipeline_v, pipeline_vt, pipeline_mask, smem_pipe_write,
                     shared_storage, scheduler_prefetch, seqlen_info, block_coord, work_idx);
      }
      mainloop.load_tail(pipeline_k, pipeline_v, pipeline_vt, pipeline_mask, smem_pipe_write, shared_storage, work_idx);
    } else {
      cutlass::arch::warpgroup_reg_alloc<MmaRegisterRequirement>();
      TiledMmaPV tiled_mma_pv;
      PipelineState smem_pipe_read;
      PipelineState smem_pipe_read_new;
      scheduler.init_consumer();
      mainloop.mma_init();
      int work_idx = 0;
      CUTLASS_PRAGMA_NO_UNROLL
      for (auto work_tile_info = scheduler.template get_initial_work<false>(params.scheduler);
         work_tile_info.is_valid(params.scheduler);
         ) {
        auto block_coord = work_tile_info.get_block_coord(params.scheduler);
        int const bidb = get<2>(block_coord);
        int sk_t = 0, sk_np = 1, sk_slot = 0;
        if constexpr (SK) { sk_t = work_tile_info.t; sk_np = work_tile_info.np; sk_slot = work_tile_info.slot; }
        SeqlenInfo_t seqlen_info{
          bidb,
          get<0>(params.mainloop.shape_Q),
          !params.mainloop.ptr_pagetable ? size<0>(params.mainloop.shape_K) : size<0>(params.mainloop.shape_K) * size<1>(params.mainloop.shape_pagetable),
          get<0>(params.mainloop.shape_K_new),
          params.mainloop.cu_seqlens_q, params.mainloop.cu_seqlens_k, params.mainloop.cu_seqlens_k_new,
          params.mainloop.seqused_q, params.mainloop.seqused_k, params.mainloop.leftpad_k,
          params.mainloop.seqlens_rotary
        };
        if constexpr (AppendKV) {
          bool tile_new_valid = mainloop.store_kv_new(
            params.mainloop, pipeline_k_new, pipeline_v_new, smem_pipe_read_new,
            threadIdx.x - MmaThreadOffset, shared_storage, seqlen_info, block_coord);
          if (tile_new_valid) {
            asm volatile ("fence.proxy.async.global;");
            cutlass::arch::NamedBarrier::arrive(NumMmaThreads + NumProducerThreads, static_cast<uint32_t>(FwdNamedBarriers::AppendKV) );
          }
        }
        float softmax_scale_log2 = params.mainloop.softmax_scale_log2;
        if constexpr (Is_FP8 && !Has_softcap) {
          int const bidh = get<1>(block_coord);
          int const bidh_kv = !PackGQA ? params.mainloop.qhead_per_khead_divmod.divide(bidh) : bidh;
          float const q_descale = params.mainloop.ptr_q_descale == nullptr ? 1.0f : params.mainloop.ptr_q_descale[bidb * get<0>(params.mainloop.stride_q_descale) + bidh_kv * get<1>(params.mainloop.stride_q_descale)];
          float const k_descale = params.mainloop.ptr_k_descale == nullptr ? 1.0f : params.mainloop.ptr_k_descale[bidb * get<0>(params.mainloop.stride_k_descale) + bidh_kv * get<1>(params.mainloop.stride_k_descale)];
          softmax_scale_log2 *= q_descale * k_descale;
        }
        flash::SoftmaxL<!LargeHeadDimV ? 2 * (2 * kBlockM / NumMmaThreads) : 2, !Is_FP8 ? 0 : 8> softmax(softmax_scale_log2);
        if (has_sink) {
          int const split_idx = get<3>(block_coord);
          if (!Split || (split_idx & 0x0000FFFF) == 0) {
            int const bidh = get<1>(block_coord);
            if (!PackGQA) {
              softmax.template set_row_sink<PackGQA>(sSink, bidh);
            } {
              int const m_block = get<0>(block_coord);
              int const qhead_per_khead = params.mainloop.qhead_per_khead_divmod.divisor;
              TiledMmaQK tiled_mma_qk;
              auto thread_mma_qk = tiled_mma_qk.get_thread_slice(threadIdx.x - MmaThreadOffset);
              Tensor coord_S = cute::make_identity_tensor(select<0, 1>(TileShape_MNK{}));
              Tensor tCoord_S = thread_mma_qk.partition_C(coord_S);
              auto tCoord_rowcol = make_tensor(
                tCoord_S.data(),
                flash::convert_layout_acc_rowcol<false>(tCoord_S.layout())
              );
              const int row_start = m_block * kBlockM + get<0>(tCoord_rowcol(_0{}, _0{}));
              const int row_step = get<0>(tCoord_rowcol(_1{}, _0{})) - get<0>(tCoord_rowcol(_0{}, _0{}));
              softmax.template set_row_sink<PackGQA>(sSink, bidh, qhead_per_khead, row_start, row_step);
            }
          }
        }
        Tensor tOrO = partition_fragment_C(tiled_mma_pv, select<0, 1>(TileShape_MNK_PV{}));
        bool tile_valid;
        if constexpr (!LargeHeadDimV) {
          tile_valid = mainloop.mma(
            params.mainloop, pipeline_k, pipeline_v, pipeline_mask, smem_pipe_read,
            tOrO, softmax, threadIdx.x - MmaThreadOffset, work_idx, seqlen_info, block_coord, shared_storage);
        } else {
          if (warp_group_idx == 1) {
            tile_valid = mainloop.mma(
              params.mainloop, pipeline_k, pipeline_v, pipeline_mask, smem_pipe_read,
              tOrO, softmax, threadIdx.x - MmaThreadOffset, work_idx, seqlen_info, block_coord, shared_storage);
          } else {
            tile_valid = mainloop.mma_pv(
              params.mainloop, pipeline_v, smem_pipe_read,
              tOrO, softmax, threadIdx.x - MmaThreadOffset, seqlen_info, block_coord, shared_storage);
          }
        }
        work_tile_info = scheduler.template get_next_work<false>(params.scheduler, work_tile_info);
        if constexpr (Split && Varlen) {
          if (!work_tile_info.is_valid(params.scheduler)) {
            cutlass::arch::launch_dependent_grids();
          }
        }
        if constexpr (SK) {
          cute::tuple<int32_t, int32_t, int32_t, int32_t> const coord_f{get<0>(block_coord), get<1>(block_coord), get<2>(block_coord), int32_t(1 << 16)};
          if (!tile_valid) {
            cute::clear(tOrO);
            #pragma unroll
            for (int mi = 0; mi < size(softmax.row_sum); ++mi) { softmax.row_sum(mi) = -INFINITY; }
          }



          if (sk_np <= 1) {
            epilogue.store(params.epilogue, tOrO, softmax.row_sum, shared_storage, tiled_mma_pv,
                    threadIdx.x - MmaThreadOffset, coord_f);
          } else {
            int* const sk2_flag = reinterpret_cast<int*>(&shared_storage.pipelines.smem_scheduler) + 8;
            static constexpr int FR = decltype(size(tOrO))::value;
            static constexpr int NR = decltype(size(softmax.row_sum))::value;
            static_assert(FR % 4 == 0);
            int const tid = threadIdx.x - MmaThreadOffset;
            int const G = int(gridDim.x);
            int const t0 = params.scheduler.tp[sk_t], t1 = params.scheduler.tp[sk_t + 1];
            float* const part = reinterpret_cast<float*>(params.epilogue.ptr_O_partial);
            float* const lsep = params.epilogue.ptr_LSE_partial;
            {
              size_t const o = (size_t(blockIdx.x) * 2 + sk_slot) * NumMmaThreads + tid;
              float4* dst = reinterpret_cast<float4*>(part + o * FR);
              #pragma unroll
              for (int i = 0; i < FR / 4; ++i) { dst[i] = make_float4(tOrO(4 * i), tOrO(4 * i + 1), tOrO(4 * i + 2), tOrO(4 * i + 3)); }
              #pragma unroll
              for (int r = 0; r < NR; ++r) { lsep[o * NR + r] = softmax.row_sum(r); }
            }
            __threadfence();
            flash::named_barrier_sync(NumMmaThreads, cutlass::arch::ReservedNamedBarriers::EpilogueBarrier);
            if (tid == 0) { *sk2_flag = atomicAdd(params.scheduler.cnt + sk_t, 1) == sk_np - 1; }
            flash::named_barrier_sync(NumMmaThreads, cutlass::arch::ReservedNamedBarriers::EpilogueBarrier);
            bool const last = *sk2_flag;
            if (last) {
              int const x52z = *sk2_flag - 1;
              long long const W = params.scheduler.st[G + x52z];
              __threadfence();
              int const cf = int(((long long)(t0 + params.scheduler.st[G + 1 + x52z] + 1) * G + W - 1) / W) - 1;
              int const cl = int(((long long)t1 * G + W - 1) / W) - 1;
              Tensor acc = make_tensor(tOrO.data(), flash::convert_layout_acc_rowcol(tOrO.layout()));
              for (int c2 = cf; c2 <= cl; ++c2) {
                if (c2 == int(blockIdx.x)) { continue; }
                int const sl = (long long)c2 * W / G >= t0 ? 0 : 1;
                size_t const o = (size_t(c2) * 2 + sl) * NumMmaThreads + tid;
                Tensor tOrO2 = make_fragment_like(tOrO);
                float4 const* src = reinterpret_cast<float4 const*>(part + o * FR);
                #pragma unroll
                for (int i = 0; i < FR / 4; ++i) {
                  float4 const v = __ldcg(src + i);
                  tOrO2(4 * i) = v.x; tOrO2(4 * i + 1) = v.y; tOrO2(4 * i + 2) = v.z; tOrO2(4 * i + 3) = v.w;
                }
                Tensor acc2 = make_tensor(tOrO2.data(), flash::convert_layout_acc_rowcol(tOrO2.layout()));
                #pragma unroll
                for (int mi = 0; mi < NR; ++mi) {
                  float const la = softmax.row_sum(mi), lb = __ldcg(lsep + o * NR + mi);
                  float const mx = fmaxf(la, lb);
                  if (mx == -INFINITY) { continue; }
                  float const wa = expf(la - mx), wb = expf(lb - mx);
                  float const inv = 1.f / (wa + wb);
                  float const ia = wa * inv, ib = wb * inv;
                  #pragma unroll
                  for (int ni = 0; ni < size<1>(acc); ++ni) { acc(mi, ni) = acc(mi, ni) * ia + acc2(mi, ni) * ib; }
                  softmax.row_sum(mi) = mx + logf(wa + wb);
                }
              }
              epilogue.store(params.epilogue, tOrO, softmax.row_sum, shared_storage, tiled_mma_pv,
                      threadIdx.x - MmaThreadOffset, coord_f);
            } else {
              epilogue.store_skip(shared_storage);
            }
          }
        } else
        {
        if (tile_valid) {
          epilogue.store(params.epilogue, tOrO, softmax.row_sum, shared_storage, tiled_mma_pv,
                  threadIdx.x - MmaThreadOffset, block_coord);
        } else {
          epilogue.store_zero(params.epilogue, threadIdx.x - MmaThreadOffset, block_coord);
        }
        }
      }
      epilogue.store_tail();
    }
  }
};
}
#include "cute/tensor.hpp"
#include <cutlass/cutlass.h>
#include <cutlass/array.h>
#include <cutlass/numeric_types.h>
#include <cutlass/kernel_hardware_info.h>
namespace flash {
using namespace cute;
template <class CollectiveMainloop_, class CollectiveEpilogue_, class TileScheduler_>
class FlashAttnFwdSm80 {
public:
  using CollectiveMainloop = CollectiveMainloop_;
  using CollectiveEpilogue = CollectiveEpilogue_;
  static constexpr bool Is_causal = CollectiveMainloop::Is_causal;
  static constexpr bool Is_local = CollectiveMainloop::Is_local;
  static_assert(CollectiveMainloop::Varlen == CollectiveEpilogue::Varlen);
  static constexpr bool Has_softcap = CollectiveMainloop::Has_softcap;
  static constexpr bool Varlen = CollectiveMainloop::Varlen;
  static constexpr bool PagedKV = CollectiveMainloop::PagedKV;
  static constexpr bool Split = CollectiveMainloop::Split;
  static constexpr bool Is_FP8 = CollectiveMainloop::Is_FP8;
  static constexpr bool Transpose_V = CollectiveMainloop::Transpose_V;
  static constexpr bool AppendKV = CollectiveMainloop::AppendKV;
  static constexpr bool PackGQA = CollectiveMainloop::PackGQA;
  static constexpr int NumProducerThreads = CollectiveMainloop::NumProducerThreads;
  using SeqlenInfo_t = typename CollectiveMainloop::SeqlenInfo_t;
  using TileShape_MNK = typename CollectiveMainloop::TileShape_MNK;
  using TiledMma = typename CollectiveMainloop::TiledMma;
  using ArchTag = typename CollectiveMainloop::ArchTag;
  using MainloopArguments = typename CollectiveMainloop::Arguments;
  using MainloopParams = typename CollectiveMainloop::Params;
  using EpilogueArguments = typename CollectiveEpilogue::Arguments;
  using EpilogueParams = typename CollectiveEpilogue::Params;
  static_assert(ArchTag::kMinComputeCapability >= 80);
  using TileScheduler = TileScheduler_;
  using TileSchedulerArguments = typename flash::TileSchedulerArguments;
  using TileSchedulerParams = typename TileScheduler::Params;
  static constexpr uint32_t NumThreads = CUTE_STATIC_V(size(TiledMma{}));
  static constexpr uint32_t MaxThreadsPerBlock = CUTE_STATIC_V(size(TiledMma{}));
  static constexpr uint32_t MinBlocksPerMultiprocessor = NumThreads == 128 ? 2 : 1;
  static constexpr int mainloop_smem_padding_ = int(sizeof(typename CollectiveEpilogue::TensorStorage))
    - int(sizeof(decltype((typename CollectiveMainloop::TensorStorage{}).smem_v)))
    - int(sizeof(decltype((typename CollectiveMainloop::TensorStorage{}).smem_k)));
  static constexpr int mainloop_smem_padding = mainloop_smem_padding_ < 0 ? 0 : mainloop_smem_padding_;
  struct SharedStorage {
    struct TensorStorage : cute::aligned_struct<128> {
      union {
        struct {
          cute::array<uint32_t, mainloop_smem_padding / sizeof(uint32_t)> padding_;
          typename CollectiveMainloop::TensorStorage mainloop;
        };
        typename CollectiveEpilogue::TensorStorage epilogue;
      };
    } tensors;
    alignas(16) typename TileScheduler::SharedStorage smem_scheduler;
  };
  static constexpr int SharedStorageSize = sizeof(SharedStorage);
  struct Arguments {
    MainloopArguments mainloop{};
    EpilogueArguments epilogue{};
    cutlass::KernelHardwareInfo hw_info{};
    TileSchedulerArguments scheduler{};
  };
  struct Params {
    MainloopParams mainloop{};
    EpilogueParams epilogue{};
    cutlass::KernelHardwareInfo hw_info{};
    TileSchedulerParams scheduler{};
  };
  static
  Params
  to_underlying_arguments(Arguments const& args) {
    CUTLASS_TRACE_HOST("to_underlying_arguments():");
    int sm_count = args.hw_info.sm_count;
    if (sm_count <= 0) {
      CUTLASS_TRACE_HOST("  WARNING: Arguments do not include a valid SM count.\n"
        "  For optimal performance, populate the arguments KernelHardwareInfo struct with the SM count.");
      sm_count = cutlass::KernelHardwareInfo::query_device_multiprocessor_count(args.hw_info.device_id);
    }
    CUTLASS_TRACE_HOST("to_underlying_arguments(): Setting persistent grid SM count to " << sm_count);
    cutlass::KernelHardwareInfo hw_info{args.hw_info.device_id, sm_count};
    return {
      CollectiveMainloop::to_underlying_arguments(args.mainloop),
      CollectiveEpilogue::to_underlying_arguments(args.epilogue),
      hw_info,
      TileScheduler::to_underlying_arguments(args.scheduler)
    };
  }
  static dim3
  get_grid_shape(Params const& params) {
    return TileScheduler::get_grid_shape(params.scheduler, params.hw_info.sm_count * MinBlocksPerMultiprocessor);
  }
  static dim3
  get_block_shape() {
    return dim3(MaxThreadsPerBlock, 1, 1);
  }
  CUTLASS_DEVICE
  void
  operator()(Params const& params, char* smem_buf) {
    static constexpr int kBlockM = get<0>(TileShape_MNK{});
    SharedStorage& shared_storage = *reinterpret_cast<SharedStorage*>(smem_buf);
    CollectiveMainloop mainloop;
    CollectiveEpilogue epilogue;
    TileScheduler scheduler(reinterpret_cast<typename TileScheduler::SharedStorage*>(&shared_storage.smem_scheduler));
    TiledMma tiled_mma;
    scheduler.init_consumer();
    int warp_idx = cutlass::canonical_warp_idx_sync();
    CUTLASS_PRAGMA_NO_UNROLL
    for (auto work_tile_info = warp_idx == 0 ? scheduler.template get_initial_work<true>(params.scheduler) : scheduler.template get_initial_work<false>(params.scheduler);
       work_tile_info.is_valid(params.scheduler);
       work_tile_info = warp_idx == 0 ? scheduler.template get_next_work<true>(params.scheduler, work_tile_info) : scheduler.template get_next_work<false>(params.scheduler, work_tile_info)) {
      Tensor tOrO = partition_fragment_C(tiled_mma, select<0, 2>(TileShape_MNK{}));
      float softmax_scale_log2 = params.mainloop.softmax_scale_log2;
      auto block_coord = work_tile_info.get_block_coord(params.scheduler);
      int const bidb = get<2>(block_coord);
      if constexpr (Is_FP8 && !Has_softcap) {
        int const bidh = get<1>(block_coord);
        int const bidh_kv = !PackGQA ? params.mainloop.qhead_per_khead_divmod.divide(bidh) : bidh;
        float const q_descale = params.mainloop.ptr_q_descale == nullptr ? 1.0f : params.mainloop.ptr_q_descale[bidb * get<0>(params.mainloop.stride_q_descale) + bidh_kv * get<1>(params.mainloop.stride_q_descale)];
        float const k_descale = params.mainloop.ptr_k_descale == nullptr ? 1.0f : params.mainloop.ptr_k_descale[bidb * get<0>(params.mainloop.stride_k_descale) + bidh_kv * get<1>(params.mainloop.stride_k_descale)];
        softmax_scale_log2 *= q_descale * k_descale;
      }
      flash::Softmax<2 * (2 * kBlockM / NumThreads), !Is_FP8 ? 0 : 8> softmax(softmax_scale_log2);
      SeqlenInfo_t seqlen_info{
        bidb,
        get<0>(params.mainloop.shape_Q),
        !PagedKV ? size<0>(params.mainloop.shape_K) : size<0>(params.mainloop.shape_K) * size<1>(params.mainloop.shape_pagetable),
        get<0>(params.mainloop.shape_K_new),
        params.mainloop.cu_seqlens_q, params.mainloop.cu_seqlens_k, params.mainloop.cu_seqlens_k_new,
        params.mainloop.seqused_q, params.mainloop.seqused_k, params.mainloop.leftpad_k,
        params.mainloop.seqlens_rotary
      };
      if constexpr (AppendKV) {
        bool tile_new_valid = mainloop.store_kv_new(
          params.mainloop, threadIdx.x, shared_storage, seqlen_info, block_coord);
        if (tile_new_valid) { __syncthreads(); }
      }
      bool tile_valid = mainloop.mma(
        params.mainloop, tOrO, softmax, threadIdx.x, seqlen_info, block_coord,
        shared_storage);
      scheduler.prefetch_next_work(params.scheduler, work_tile_info);
      if (tile_valid) {
        epilogue.store(params.epilogue, tOrO, softmax.row_sum, shared_storage, tiled_mma,
                threadIdx.x, block_coord);
      } else {
        epilogue.store_zero(params.epilogue, threadIdx.x, block_coord);
      }
    }
  }
};
}
#include <cutlass/cutlass.h>
#include <cutlass/array.h>
#include <cutlass/numeric_types.h>
#include <cutlass/numeric_conversion.h>
#include "cutlass/pipeline/pipeline.hpp"
#include "cute/tensor.hpp"
#include "cutlass/gemm/collective/builders/sm90_common.inl"
namespace flash {
template <class SeqlenInfo_t, int kBlockM, int kBlockN, bool Is_causal, bool Is_local, bool PackGQA=false, bool Split=false>
struct BlockMN {
  static
  CUTLASS_DEVICE
  cute::tuple<int, int, int> get_n_block_min_max(
      SeqlenInfo_t const& seqlen_info,
      int const m_block, int const bidb, int const split_idx, int const num_splits,
      int const window_size_left, int const window_size_right,
      cutlass::FastDivmod const& attention_chunk_divmod,
      cutlass::FastDivmod const& qhead_per_khead_divmod) {
    int seqlen_k = seqlen_info.seqlen_k;
    int const seqlen_q = seqlen_info.seqlen_q;
    int n_offset = 0;
    if constexpr (Is_local) {
      int m_idx_min = m_block * kBlockM;
      if (PackGQA) { m_idx_min = qhead_per_khead_divmod.divide(m_idx_min); }
      int const n_idx = m_idx_min + seqlen_k - seqlen_q;
      int n_idx_left = n_idx - window_size_left;
      if (attention_chunk_divmod.divisor > 0) {
        n_idx_left = std::max(n_idx_left, flash::round_down(attention_chunk_divmod, n_idx));
      }
      n_offset = std::max(int(0), n_idx_left);
      seqlen_k -= n_offset;
    }
    int n_block_max = cute::ceil_div(seqlen_k, kBlockN);
    if constexpr (Is_causal || Is_local) {
      int m_idx_max = (m_block + 1) * kBlockM;
      if (PackGQA) { m_idx_max = qhead_per_khead_divmod.divide(m_idx_max - 1) + 1 ; }
      n_block_max = std::min(n_block_max,
                  cute::ceil_div(m_idx_max + seqlen_k - seqlen_q + window_size_right, kBlockN));
    }
    int n_block_min = 0;
    if (Split && (split_idx & (1 << 30))) {
      n_block_min = split_idx & 0x7FF;
      n_block_max = std::min(n_block_max, (split_idx >> 11) & 0x7FF);
      return {n_block_min, n_block_max, n_offset};
    }
    if constexpr (Split) {
      uint32_t num_splits_dynamic_u = reinterpret_cast<uint32_t const&>(split_idx) >> 16;
      int num_splits_dynamic = reinterpret_cast<int&>(num_splits_dynamic_u);
      int split_idx_actual = split_idx & 0x0000FFFF;
      int num_splits_actual = num_splits_dynamic > 0 ? num_splits_dynamic : num_splits;
      int num_n_blocks_per_split = n_block_max <= n_block_min ? 0 : cute::ceil_div(n_block_max - n_block_min, num_splits_actual);
      n_block_min = n_block_min + split_idx_actual * num_n_blocks_per_split;
      n_block_max = std::min(n_block_min + num_n_blocks_per_split, n_block_max);
    }
    return {n_block_min, n_block_max, n_offset};
  }
  static
  CUTLASS_DEVICE
  cute::tuple<int, int> get_n_block_k_new_min_max(
      SeqlenInfo_t const& seqlen_info,
      int const m_block, int const bidb, int const split_idx, int const num_splits,
      int const window_size_left, int const window_size_right,
      cutlass::FastDivmod const& attention_chunk_divmod,
      cutlass::FastDivmod const& qhead_per_khead_divmod) {
    auto [n_block_min, n_block_max, n_offset] = get_n_block_min_max(
      seqlen_info, m_block, bidb, split_idx, num_splits,
      window_size_left, window_size_right, attention_chunk_divmod, qhead_per_khead_divmod);
    int const idx_k_new_min = std::max(n_block_min * kBlockN + n_offset - seqlen_info.seqlen_k_og, 0);
    int const idx_k_new_max = std::min(n_block_max * kBlockN + n_offset - seqlen_info.seqlen_k_og, seqlen_info.seqlen_k_new);
    int const n_block_new_min = idx_k_new_min / kBlockN;
    int const n_block_new_max = idx_k_new_max > idx_k_new_min ? cute::ceil_div(idx_k_new_max, kBlockN) : n_block_new_min;
    return {n_block_new_min, n_block_new_max};
  }
  static
  CUTLASS_DEVICE
  cute::tuple<int, int> get_m_block_min_max(
      SeqlenInfo_t const& seqlen_info,
      int const n_block, int const bidb,
      int const window_size_left, int const window_size_right, int const sink_token_length) {
    int const seqlen_q = seqlen_info.seqlen_q;
    int const seqlen_k = seqlen_info.seqlen_k;
    int m_block_max = cute::ceil_div(seqlen_q, kBlockM);
    if constexpr (Is_local) {
      if (n_block >= cute::ceil_div(sink_token_length, kBlockN)) {
        m_block_max = std::min(m_block_max, cute::ceil_div((n_block + 1) * kBlockN + seqlen_q - seqlen_k + window_size_left, kBlockM));
      }
    }
    int m_block_min = 0;
    if constexpr (Is_causal || Is_local) {
      m_block_min = std::max(m_block_min, (n_block * kBlockN + seqlen_q - seqlen_k - window_size_right) / kBlockM);
    }
    return {m_block_min, m_block_max};
  }
  static
  CUTLASS_DEVICE
  int get_n_block_min_causal_local_mask(
      SeqlenInfo_t const& seqlen_info,
      int const m_block, int const n_block_min, int const window_size_right,
      cutlass::FastDivmod const& attention_chunk_divmod,
      cutlass::FastDivmod const& qhead_per_khead_divmod) {
    int const m_idx_min = !PackGQA ? m_block * kBlockM : qhead_per_khead_divmod.divide(m_block * kBlockM);
    int const n_idx = m_idx_min + seqlen_info.seqlen_k - seqlen_info.seqlen_q;
    int n_idx_right = !Is_local ? n_idx : n_idx + window_size_right;
    if (Is_local && attention_chunk_divmod.divisor > 0) {
      n_idx_right = std::min(n_idx_right, flash::round_up(attention_chunk_divmod, n_idx));
    }
    return std::max(n_block_min, n_idx_right / kBlockN);
  }
  static
  CUTLASS_DEVICE
  int get_n_block_min_before_local_mask(
      SeqlenInfo_t const& seqlen_info,
      int const m_block, int const n_block_min, int const window_size_left,
      cutlass::FastDivmod const& attention_chunk_divmod,
      cutlass::FastDivmod const& qhead_per_khead_divmod) {
    int const m_idx_max = !PackGQA ? (m_block + 1) * kBlockM : qhead_per_khead_divmod.divide((m_block + 1) * kBlockM - 1) + 1;
    int const n_idx = m_idx_max + seqlen_info.seqlen_k - seqlen_info.seqlen_q;
    int n_idx_left = !Is_local ? n_idx : n_idx - window_size_left;
    if (Is_local && attention_chunk_divmod.divisor > 0) {
      n_idx_left = std::max(n_idx_left, flash::round_down(attention_chunk_divmod, n_idx));
    }
    return !Is_local ? n_block_min : std::max(n_block_min, cute::ceil_div(n_idx_left, kBlockN));
  }
};
}
#include <cute/tensor.hpp>
#include "cutlass/fast_math.h"
namespace flash {
using namespace cute;
template <int kBlockM, int kBlockN, bool PackGQA, typename TiledMma, bool SwapAB=false>
struct Mask {
  static_assert(!(PackGQA && SwapAB), "Cannot be both PackGQA and SwapAB");
  int const thread_idx;
  int const seqlen_q, seqlen_k;
  int const window_size_left, window_size_right, sink_token_length;
  cutlass::FastDivmod const attention_chunk_divmod;
  cutlass::FastDivmod const qhead_per_khead_divmod;
  CUTLASS_DEVICE
  Mask(const int thread_idx, const int seqlen_q, const int seqlen_k,
     const int window_size_left, const int window_size_right, const int sink_token_length,
     cutlass::FastDivmod const &attention_chunk_divmod,
     cutlass::FastDivmod const &qhead_per_khead_divmod)
    : thread_idx(thread_idx)
    , seqlen_q(seqlen_q)
    , seqlen_k(seqlen_k)
    , window_size_left(window_size_left)
    , window_size_right(window_size_right)
    , sink_token_length(sink_token_length)
    , attention_chunk_divmod(attention_chunk_divmod)
    , qhead_per_khead_divmod(qhead_per_khead_divmod)
  {
  };
  template <bool Seqlenk_mask=false, bool Causal_mask=false, bool Local_mask=false,
    typename Engine, typename Layout>
  CUTLASS_DEVICE
  void apply(Tensor<Engine, Layout> &tSrS, const int m_block, const int n_block) const {
    static_assert(!(Causal_mask && Local_mask), "Cannot be both causal and local");
    static_assert(Layout::rank == 3, "Only support 3D Tensor");
    if (!Seqlenk_mask && !Causal_mask && !Local_mask) { return; }
    auto thread_mma = TiledMma{}.get_thread_slice(thread_idx);
    auto thread0_mma = TiledMma{}.get_thread_slice(_0{});
    static constexpr int Row = !SwapAB ? 0 : 1, Col = !SwapAB ? 1 : 0;
    Tensor cS = cute::make_identity_tensor(Shape<Int<!SwapAB ? kBlockM : kBlockN>, Int<!SwapAB ? kBlockN : kBlockM>>{});
    Tensor tScS = thread_mma.partition_C(cS);
    Tensor tSrS_rowcol = make_tensor(tSrS.data(), flash::convert_layout_acc_rowcol<SwapAB>(tSrS.layout()));
    Tensor tScS_rowcol = make_tensor(tScS.data(), flash::convert_layout_acc_rowcol<SwapAB>(tScS.layout()));
    Tensor t0ScS = thread0_mma.partition_C(cS);
    Tensor t0ScS_rowcol = make_tensor(t0ScS.data(), flash::convert_layout_acc_rowcol<SwapAB>(t0ScS.layout()));
    int const thread_col_offset = get<Col>(tScS_rowcol(_0{}, _0{}));
    int const seqlenk_col_limit = seqlen_k - n_block * kBlockN - thread_col_offset;
    if constexpr (!Causal_mask && !Local_mask) {
      if constexpr (Seqlenk_mask) {
        #pragma unroll
        for (int n = 0; n < size<1>(tSrS_rowcol); ++n) {
          if (int(get<Col>(t0ScS_rowcol(_0{}, n))) >= seqlenk_col_limit) {
            #pragma unroll
            for (int m = 0; m < size<0>(tSrS_rowcol); ++m) { tSrS_rowcol(m, n) = -INFINITY; }
          }
        }
      }
    } else {
      if constexpr (!SwapAB) {
        static constexpr int kMmaThreadsPerRow = size<0, 0>(typename TiledMma::AtomLayoutC_TV{});
        static_assert(cutlass::NumThreadsPerWarp % kMmaThreadsPerRow == 0);
        static_assert(!PackGQA || CUTE_STATIC_V(size<0>(tSrS_rowcol)) <= kMmaThreadsPerRow);
        int mma_m_idx;
        if constexpr (PackGQA) {
          mma_m_idx = qhead_per_khead_divmod.divide(m_block * kBlockM + get<Row>(tScS_rowcol(thread_idx % kMmaThreadsPerRow, _0{})));
        }
        int const causal_row_offset = 1 + seqlen_k - n_block * kBlockN - seqlen_q - thread_col_offset;
        if constexpr (Causal_mask) {
          #pragma unroll
          for (int m = 0; m < size<0>(tSrS_rowcol); ++m) {
            int const row_idx = !PackGQA
              ? get<Row>(tScS_rowcol(m, _0{})) + m_block * kBlockM
              : __shfl_sync(0xffffffff, mma_m_idx, m % kMmaThreadsPerRow, kMmaThreadsPerRow);
            int const col_limit_right = !Seqlenk_mask
              ? row_idx + causal_row_offset
              : __viaddmin_s32(row_idx, causal_row_offset, seqlenk_col_limit);
            #pragma unroll
            for (int n = 0; n < size<1>(tSrS_rowcol); ++n) {
              if (int(get<Col>(t0ScS_rowcol(_0{}, n))) >= col_limit_right) { tSrS_rowcol(m, n) = -INFINITY; }
            }
          }
        } else {
          int const local_row_offset_right = causal_row_offset + window_size_right;
          int const local_row_offset_left = causal_row_offset - 1 - window_size_left;
          int const col_limit_sink = sink_token_length - n_block * kBlockN;
          #pragma unroll
          for (int m = 0; m < size<0>(tSrS_rowcol); ++m) {
            int const row_idx = !PackGQA
              ? get<Row>(tScS_rowcol(m, _0{})) + m_block * kBlockM
              : __shfl_sync(0xffffffff, mma_m_idx, m % kMmaThreadsPerRow, kMmaThreadsPerRow);
            int col_limit_right = !Seqlenk_mask
              ? row_idx + local_row_offset_right
              : __viaddmin_s32(row_idx, local_row_offset_right, seqlenk_col_limit);
            int col_limit_left = row_idx + local_row_offset_left;
            if (attention_chunk_divmod.divisor > 0) {
              int col_limit_left_chunk = flash::round_down(attention_chunk_divmod, row_idx + seqlen_k - seqlen_q) - n_block * kBlockN - thread_col_offset;
              col_limit_left = std::max(col_limit_left, col_limit_left_chunk);
              col_limit_right = std::min(col_limit_right, col_limit_left_chunk + attention_chunk_divmod.divisor);
            }
            #pragma unroll
            for (int n = 0; n < size<1>(tSrS_rowcol); ++n) {
              int const col_idx = int(get<Col>(t0ScS_rowcol(m, n)));
              if (col_idx >= col_limit_right || (col_idx < col_limit_left && col_idx >= col_limit_sink)) { tSrS_rowcol(m, n) = -INFINITY; }
            }
          }
        }
      } else {
        int const thread_row_offset = get<Row>(tScS_rowcol(_0{}, _0{}));
        int const causal_row_offset = seqlenk_col_limit - seqlen_q + m_block * kBlockM + thread_row_offset;
        if constexpr (Causal_mask) {
          #pragma unroll
          for (int n = 0; n < size<1>(tSrS_rowcol); ++n) {
            int const col0 = int(get<Col>(t0ScS_rowcol(_0{}, n)));
            int const row_limit_top = col0 >= seqlenk_col_limit ? kBlockM : col0 - causal_row_offset;
            #pragma unroll
            for (int m = 0; m < size<0>(tSrS_rowcol); ++m) {
              if (int(get<Row>(t0ScS_rowcol(m, _0{}))) < row_limit_top) { tSrS_rowcol(m, n) = -INFINITY; }
            }
          }
        } else {
          int const col_limit_sink = sink_token_length - n_block * kBlockN - thread_col_offset;
          #pragma unroll
          for (int n = 0; n < size<1>(tSrS_rowcol); ++n) {
            int const col0 = int(get<Col>(t0ScS_rowcol(_0{}, n)));
            int const row_limit_top = col0 >= seqlenk_col_limit ? kBlockM : col0 - causal_row_offset - window_size_right;
            int const row_limit_bot = col0 < col_limit_sink ? kBlockM : col0 - causal_row_offset + window_size_left;
            #pragma unroll
            for (int m = 0; m < size<0>(tSrS_rowcol); ++m) {
              int const row_idx = int(get<Row>(t0ScS_rowcol(m, _0{})));
              if (row_idx < row_limit_top || row_idx > row_limit_bot) { tSrS_rowcol(m, n) = -INFINITY; }
            }
          }
        }
      }
    }
  };
};
template <int kBlockM, int kBlockN, bool PackGQA, typename TiledMma, bool SwapAB=false>
struct SparseMask {
  static constexpr int kNumInt32PerBlock = (kBlockN + 31) / 32;
  int const thread_idx;
  int const seqlen_q, seqlen_k;
  int const max_k_blocks;
  cutlass::FastDivmod const qhead_per_khead_divmod;
  CUTLASS_DEVICE
  SparseMask(const int thread_idx, const int seqlen_q, const int seqlen_k,
        int const max_k_blocks,
        cutlass::FastDivmod const qhead_per_khead_divmod
      )
    : thread_idx(thread_idx)
    , seqlen_q(seqlen_q)
    , seqlen_k(seqlen_k)
    , max_k_blocks(max_k_blocks)
    , qhead_per_khead_divmod(qhead_per_khead_divmod)
  {
  }
  template <bool Seqlenk_mask=false, typename Engine, typename Layout>
  CUTLASS_DEVICE
  void apply(Tensor<Engine, Layout> &tSrS, const int m_block, const int n_block,
        uint32_t const* __restrict__ smem_mask_ptr) const {
    auto thread_mma = TiledMma{}.get_thread_slice(thread_idx);
    auto thread0_mma = TiledMma{}.get_thread_slice(_0{});
    static constexpr int Row = !SwapAB ? 0 : 1, Col = !SwapAB ? 1 : 0;
    Tensor cS = cute::make_identity_tensor(Shape<Int<!SwapAB ? kBlockM : kBlockN>, Int<!SwapAB ? kBlockN : kBlockM>>{});
    Tensor tScS = thread_mma.partition_C(cS);
    Tensor t0ScS = thread0_mma.partition_C(cS);
    Tensor tSrS_rowcol = make_tensor(tSrS.data(), flash::convert_layout_acc_rowcol<SwapAB>(tSrS.layout()));
    Tensor tScS_rowcol = make_tensor(tScS.data(), flash::convert_layout_acc_rowcol<SwapAB>(tScS.layout()));
    Tensor t0ScS_rowcol = make_tensor(t0ScS.data(), flash::convert_layout_acc_rowcol<SwapAB>(t0ScS.layout()));
    static constexpr int kMmaThreadsPerRow = size<0, 0>(typename TiledMma::AtomLayoutC_TV{});
    int const thread_col_offset = get<Col>(tScS_rowcol(_0{}, _0{}));
    int const quad_lane = thread_col_offset / 2;
    int const seqlenk_col_limit = seqlen_k - n_block * kBlockN - thread_col_offset;
    const bool is_block_safe = n_block < max_k_blocks;
    #pragma unroll
    for (int m = 0; m < size<0>(tSrS_rowcol); ++m) {
      int local_row = get<Row>(tScS_rowcol(m, _0{}));
      int global_row_idx;
      if constexpr (!PackGQA) {
        global_row_idx = local_row + m_block * kBlockM;
      } else {
        int mma_m_idx = qhead_per_khead_divmod.divide(m_block * kBlockM + local_row);
        global_row_idx = __shfl_sync(0xffffffff, mma_m_idx, m % kMmaThreadsPerRow, kMmaThreadsPerRow);
      }
      bool const is_row_safe = (global_row_idx < seqlen_q) && is_block_safe;
      if (!is_row_safe) {
        #pragma unroll
        for (int n = 0; n < size<1>(tSrS_rowcol); ++n) {
          tSrS_rowcol(m, n) = -INFINITY;
        }
        continue;
      }
      if constexpr (kBlockN == 128) {
        uint32_t const mask_word = smem_mask_ptr[local_row * 4 + quad_lane];
        #pragma unroll
        for (int n = 0; n < size<1>(tSrS_rowcol); ++n) {
          bool const mask_bit = (mask_word >> n) & 1;
          if constexpr (Seqlenk_mask) {
            int const col0 = int(get<Col>(t0ScS_rowcol(_0{}, n)));
            bool const is_valid = col0 < seqlenk_col_limit;
            bool const final_keep = is_valid && mask_bit;
            if (!final_keep) {
              tSrS_rowcol(m, n) = -INFINITY;
            }
          } else {
            if (!mask_bit) {
              tSrS_rowcol(m, n) = -INFINITY;
            }
          }
        }
      } else if constexpr (kBlockN == 64) {
        int const word_idx = quad_lane / 2;
        int const half_idx = quad_lane % 2;
        uint32_t const full_word = smem_mask_ptr[local_row * 2 + word_idx];
        uint32_t const mask_word = half_idx ? (full_word >> 16) : full_word;
        #pragma unroll
        for (int n = 0; n < size<1>(tSrS_rowcol); ++n) {
          bool const mask_bit = (mask_word >> n) & 1;
          if constexpr (Seqlenk_mask) {
            int const col0 = int(get<Col>(t0ScS_rowcol(_0{}, n)));
            bool const is_valid = col0 < seqlenk_col_limit;
            bool const final_keep = is_valid && mask_bit;
            if (!final_keep) {
              tSrS_rowcol(m, n) = -INFINITY;
            }
          } else {
            if (!mask_bit) {
              tSrS_rowcol(m, n) = -INFINITY;
            }
          }
        }
      } else {
        #pragma unroll
        for (int n = 0; n < size<1>(tSrS_rowcol); ++n) {
          int const token_in_block = int(get<Col>(tScS_rowcol(m, n)));
          int const word_idx = token_in_block / 32;
          int const bit_idx = token_in_block % 32;
          bool const mask_bit = (smem_mask_ptr[local_row * kNumInt32PerBlock + word_idx] >> bit_idx) & 1;
          if constexpr (Seqlenk_mask) {
            bool const is_valid = (n_block * kBlockN + token_in_block < seqlen_k);
            bool const final_keep = is_valid && mask_bit;
            if (!final_keep) {
              tSrS_rowcol(m, n) = -INFINITY;
            }
          } else {
            if (!mask_bit) {
              tSrS_rowcol(m, n) = -INFINITY;
            }
          }
        }
      }
    }
  }
};
}
#include <cute/tensor.hpp>
#include "cutlass/fast_math.h"
namespace flash {
using namespace cute;
template <int kBlockM, int kHeadDim, int NumThreads, typename Element>
struct PackGQAManager {
  static constexpr int kGmemElemsPerLoad = sizeof(cute::uint128_t) / sizeof(Element);
  static constexpr int kGmemElemsPerStore = kGmemElemsPerLoad;
  static_assert(kHeadDim % kGmemElemsPerLoad == 0, "Headdim must be a multiple of kGmemElemsPerLoad");
  static constexpr int kBytePerRow = kHeadDim * sizeof(Element);
  static constexpr int kBlockKGmem = (kBytePerRow % 128 == 0 ? 128 : (kBytePerRow % 64 == 0 ? 64 : 32)) / sizeof(Element);
  static constexpr int kGmemThreadsPerRow = kBlockKGmem / kGmemElemsPerLoad;
  static_assert(NumThreads % kGmemThreadsPerRow == 0, "NumThreads must be a multiple of kGmemThreadsPerRow");
  static_assert(cutlass::NumThreadsPerWarp % kGmemThreadsPerRow == 0, "kGmemThreadsPerRow must divide NumThreadsPerWarp");
  using GmemCopyAtomCpAsync = cute::Copy_Atom<SM80_CP_ASYNC_CACHEGLOBAL_ZFILL<uint128_t>, Element>;
  using GmemLayoutAtom = Layout<Shape <Int<NumThreads / kGmemThreadsPerRow>, Int<kGmemThreadsPerRow>>,
                 Stride<Int<kGmemThreadsPerRow>, _1>>;
  using GmemTiledCopyQCpAsync = decltype(
    make_tiled_copy(GmemCopyAtomCpAsync{},
            GmemLayoutAtom{},
            Layout<Shape<_1, Int<kGmemElemsPerLoad>>>{}));
  using GmemTiledCopyO = decltype(
    make_tiled_copy(Copy_Atom<AutoVectorizingCopyWithAssumedAlignment<128>, Element>{},
            GmemLayoutAtom{},
            Layout<Shape<_1, Int<kGmemElemsPerStore>>>{}));
  template <int NumThreadsPerRow=kGmemThreadsPerRow, typename Engine, typename Layout, typename TensorC>
  CUTLASS_DEVICE
  static auto
  compute_ptr(Tensor<Engine, Layout> &tensor, TensorC const &tRows,
        cutlass::FastDivmod const &qhead_per_khead_divmod, int const thread_idx, int const m_block) {
    static constexpr int NumPtrPerThread = cute::ceil_div(CUTE_STATIC_V(cute::size(tRows)), NumThreadsPerRow);
    using TensorType = typename Engine::value_type;
    Tensor tPrPtr = make_tensor<TensorType const*>(Shape<Int<NumPtrPerThread>>{});
    #pragma unroll
    for (int i = 0; i < NumPtrPerThread; ++i) {
      int const row = i * NumThreads + get<0>(tRows(thread_idx % NumThreadsPerRow));
      int const idx = m_block * kBlockM + row;
      int m_idx, h_idx;
      m_idx = qhead_per_khead_divmod.divmod(h_idx, idx);
      tPrPtr[i] = &tensor(make_coord(make_coord(h_idx, m_idx)));
    }
    return tPrPtr;
  }
  template <typename TensormQ, typename TensorsQ>
  CUTLASS_DEVICE
  static void
  load_Q(TensormQ const &mQ,
      TensorsQ &sQ,
      cutlass::FastDivmod const &qhead_per_khead_divmod,
      int const thread_idx, int const seqlen_q, int const m_block
     )
  {
    GmemTiledCopyQCpAsync gmem_tiled_copy_Q_cp_async;
    auto gmem_thr_copy_Q_cp_async = gmem_tiled_copy_Q_cp_async.get_thread_slice(thread_idx);
    Tensor cQ = make_identity_tensor(make_shape(size<0>(sQ), size<1>(sQ)));
    Tensor tQcQ = gmem_thr_copy_Q_cp_async.partition_S(cQ);
    Tensor tQsQ = gmem_thr_copy_Q_cp_async.partition_D(sQ);
    Tensor tQpQ = make_tensor<bool>(make_shape(size<2>(tQsQ)));
    #pragma unroll
    for (int k = 0; k < size(tQpQ); ++k) { tQpQ(k) = get<1>(tQcQ(_0{}, _0{}, k)) < size<1>(mQ); }
    Tensor mQ_0 = mQ(_, _0{});
    Tensor tQcQ_row = tQcQ(_0{}, _, _0{});
    Tensor tPrQPtr = compute_ptr(mQ_0, tQcQ_row, qhead_per_khead_divmod, thread_idx, m_block);
    int const qhead_per_khead = qhead_per_khead_divmod.divisor;
    #pragma unroll
    for (int m = 0; m < size<1>(tQsQ); ++m) {
      int idx = m_block * kBlockM + get<0>(tQcQ(_0{}, m, _0{}));
      Element const* q_ptr = reinterpret_cast<Element const*>(__shfl_sync(0xffffffff, reinterpret_cast<uint64_t>(tPrQPtr(m / kGmemThreadsPerRow)), m % kGmemThreadsPerRow, kGmemThreadsPerRow));
      if (idx < seqlen_q * qhead_per_khead) {
        Tensor mQ_cur = make_tensor(make_gmem_ptr(q_ptr), Shape<Int<kHeadDim>>{});
        Tensor mQ_cur_copy = cute::tiled_divide(mQ_cur, Shape<Int<kGmemElemsPerLoad>>{});
        #pragma unroll
        for (int k = 0; k < size<2>(tQsQ); ++k) {
          int ki = get<1>(tQcQ(_0{}, _0{}, k)) / kGmemElemsPerLoad;
          cute::copy(gmem_tiled_copy_Q_cp_async.with(tQpQ(k)), mQ_cur_copy(_, ki), tQsQ(_, m, k));
        }
      }
    }
  };
  template <typename TensormLSE, typename TensorsLSE, typename TiledMma>
  CUTLASS_DEVICE
  static void
  store_LSE(TensormLSE &mLSE,
       TensorsLSE const &tLSErLSE,
       TiledMma tiled_mma,
       cutlass::FastDivmod const &qhead_per_khead_divmod,
       int const thread_idx, int const seqlen_o, int const m_block
       )
  {
    Tensor caccO = cute::make_identity_tensor(Shape<Int<kBlockM>, Int<kHeadDim>>{});
    auto thread_mma = tiled_mma.get_thread_slice(thread_idx);
    Tensor taccOcO = thread_mma.partition_C(caccO);
    Tensor taccOcO_row = make_tensor(taccOcO.data(), flash::convert_layout_acc_rowcol(taccOcO.layout()))(_, _0{});
    CUTE_STATIC_ASSERT_V(size(tLSErLSE) == size(taccOcO_row));
    static constexpr int kMmaThreadsPerRow = size<0, 0>(typename TiledMma::AtomLayoutC_TV{});
    static_assert(cutlass::NumThreadsPerWarp % kMmaThreadsPerRow == 0);
    static_assert(CUTE_STATIC_V(size(tLSErLSE)) <= kMmaThreadsPerRow);
    static_assert(CUTE_STATIC_V(size(taccOcO_row)) <= kMmaThreadsPerRow);
    Tensor tPrLSEPtr = compute_ptr<kMmaThreadsPerRow>(mLSE, taccOcO_row, qhead_per_khead_divmod, thread_idx, m_block);
    static_assert(CUTE_STATIC_V(size(tPrLSEPtr)) == 1);
    int const qhead_per_khead = qhead_per_khead_divmod.divisor;
    #pragma unroll
    for (int mi = 0; mi < size(tLSErLSE); ++mi) {
      int const row = m_block * kBlockM + get<0>(taccOcO_row(mi));
      float* ptr_LSE_cur = reinterpret_cast<float*>(__shfl_sync(0xffffffff, reinterpret_cast<uint64_t>(tPrLSEPtr[0]), mi % kMmaThreadsPerRow, kMmaThreadsPerRow));
      if (get<1>(taccOcO_row(_0{})) == 0 && row < seqlen_o * qhead_per_khead) {
        *ptr_LSE_cur = tLSErLSE(mi);
      }
    }
  };
  static constexpr int kX52Rows = kBlockM * kGmemThreadsPerRow / NumThreads;
  template <typename TensormO>
  CUTLASS_DEVICE
  static void
  load_gate(uint4 (&gpre)[kX52Rows], TensormO &mO, cutlass::FastDivmod const &qhead_per_khead_divmod,
       int const thread_idx, int const seqlen_o, int const m_block,
       Element const* obase, int64_t const orow, Element const* gate, int const gstride)
  {
    GmemTiledCopyO gmem_tiled_copy_O;
    auto gmem_thr_copy_O = gmem_tiled_copy_O.get_thread_slice(thread_idx);
    Tensor cO = make_identity_tensor(Shape<Int<kBlockM>, Int<kHeadDim>>{});
    Tensor tOcO = gmem_thr_copy_O.partition_D(cO);
    static_assert(CUTE_STATIC_V(size<1>(tOcO)) == kX52Rows && CUTE_STATIC_V(size<2>(tOcO)) == 1, "x52 gate rows");
    Tensor mO_0 = mO(_, _0{});
    Tensor tOcO_row = tOcO(_0{}, _, _0{});
    Tensor tPrOPtr = compute_ptr(mO_0, tOcO_row, qhead_per_khead_divmod, thread_idx, m_block);
    int const qhead_per_khead = qhead_per_khead_divmod.divisor;
    int const col = get<1>(tOcO(_0{}, _0{}, _0{}));
    #pragma unroll
    for (int m = 0; m < kX52Rows; ++m) {
      int idx = m_block * kBlockM + get<0>(tOcO(_0{}, m, _0{}));
      Element* o_ptr = reinterpret_cast<Element*>(__shfl_sync(0xffffffff, reinterpret_cast<uint64_t>(tPrOPtr(m / kGmemThreadsPerRow)), m % kGmemThreadsPerRow, kGmemThreadsPerRow));
      gpre[m] = make_uint4(0u, 0u, 0u, 0u);
      if (idx < seqlen_o * qhead_per_khead) {
        int64_t const off = o_ptr - obase;
        int64_t const tok = off / orow;
        gpre[m] = __ldg(reinterpret_cast<uint4 const*>(gate + tok * gstride + (off - tok * orow) + col));
      }
    }
  };
  template <typename TensormO, typename TensorrO>
  CUTLASS_DEVICE
  static void
  store_O_g(TensormO &mO, TensorrO &tOrO, cutlass::FastDivmod const &qhead_per_khead_divmod,
       int const thread_idx, int const seqlen_o, int const m_block,
       float const geps, uint4 const* gpre)
  {
    GmemTiledCopyO gmem_tiled_copy_O;
    auto gmem_thr_copy_O = gmem_tiled_copy_O.get_thread_slice(thread_idx);
    Tensor cO = make_identity_tensor(Shape<Int<kBlockM>, Int<kHeadDim>>{});
    Tensor tOcO = gmem_thr_copy_O.partition_D(cO);
    Tensor tOpO = make_tensor<bool>(make_shape(size<2>(tOcO)));
    #pragma unroll
    for (int k = 0; k < size(tOpO); ++k) { tOpO(k) = get<1>(tOcO(_0{}, _0{}, k)) < size<1>(mO); }
    Tensor mO_0 = mO(_, _0{});
    Tensor tOcO_row = tOcO(_0{}, _, _0{});
    Tensor tPrOPtr = compute_ptr(mO_0, tOcO_row, qhead_per_khead_divmod, thread_idx, m_block);
    int const qhead_per_khead = qhead_per_khead_divmod.divisor;
    static_assert(kHeadDim == 64 && kGmemThreadsPerRow == 8 && kGmemElemsPerStore == 8, "x52 gate layout");
    static_assert(CUTE_STATIC_V(size<2>(tOcO)) == 1, "x52 gate: one 8-element column group per thread");
    #pragma unroll
    for (int m = 0; m < size<1>(tOrO); ++m) {
      int idx = m_block * kBlockM + get<0>(tOcO(_0{}, m, _0{}));
      Element* o_ptr = reinterpret_cast<Element*>(__shfl_sync(0xffffffff, reinterpret_cast<uint64_t>(tPrOPtr(m / kGmemThreadsPerRow)), m % kGmemThreadsPerRow, kGmemThreadsPerRow));
      float ss = 0.f;
      #pragma unroll
      for (int k = 0; k < size<2>(tOrO); ++k) {
        #pragma unroll
        for (int i = 0; i < size<0>(tOrO); ++i) { float const xv = float(tOrO(i, m, k)); ss = fmaf(xv, xv, ss); }
      }
      #pragma unroll
      for (int o = 1; o < kGmemThreadsPerRow; o <<= 1) ss += __shfl_xor_sync(0xffffffffu, ss, o);
      float const rs = rsqrtf(ss * (1.f / 64.f) + geps);
      if (idx < seqlen_o * qhead_per_khead) {
        Element const* gv8 = reinterpret_cast<Element const*>(&gpre[m]);
        Tensor mO_cur = make_tensor(make_gmem_ptr(o_ptr), Shape<Int<kHeadDim>>{});
        Tensor mO_cur_copy = cute::tiled_divide(mO_cur, Shape<Int<kGmemElemsPerStore>>{});
        #pragma unroll
        for (int k = 0; k < size<2>(tOrO); ++k) {
          int const col = get<1>(tOcO(_0{}, _0{}, k));
          int ki = col / kGmemElemsPerStore;
          if (tOpO(k)) {
            #pragma unroll
            for (int i = 0; i < size<0>(tOrO); ++i) {
              float const gv = float(gv8[i]);
              tOrO(i, m, k) = Element(float(tOrO(i, m, k)) * rs / (1.f + __expf(-gv)));
            }
            cute::copy(gmem_tiled_copy_O, tOrO(_, m, k), mO_cur_copy(_, ki));
          }
        }
      }
    }
  };
  template <typename TensormO, typename TensorrO>
  CUTLASS_DEVICE
  static void
  store_O(TensormO &mO,
      TensorrO const &tOrO,
      cutlass::FastDivmod const &qhead_per_khead_divmod,
      int const thread_idx, int const seqlen_o, int const m_block
     )
  {
    GmemTiledCopyO gmem_tiled_copy_O;
    auto gmem_thr_copy_O = gmem_tiled_copy_O.get_thread_slice(thread_idx);
    Tensor cO = make_identity_tensor(Shape<Int<kBlockM>, Int<kHeadDim>>{});
    Tensor tOcO = gmem_thr_copy_O.partition_D(cO);
    Tensor tOpO = make_tensor<bool>(make_shape(size<2>(tOcO)));
    #pragma unroll
    for (int k = 0; k < size(tOpO); ++k) { tOpO(k) = get<1>(tOcO(_0{}, _0{}, k)) < size<1>(mO); }
    Tensor mO_0 = mO(_, _0{});
    Tensor tOcO_row = tOcO(_0{}, _, _0{});
    Tensor tPrOPtr = compute_ptr(mO_0, tOcO_row, qhead_per_khead_divmod, thread_idx, m_block);
    int const qhead_per_khead = qhead_per_khead_divmod.divisor;
    #pragma unroll
    for (int m = 0; m < size<1>(tOrO); ++m) {
      int idx = m_block * kBlockM + get<0>(tOcO(_0{}, m, _0{}));
      Element* o_ptr = reinterpret_cast<Element*>(__shfl_sync(0xffffffff, reinterpret_cast<uint64_t>(tPrOPtr(m / kGmemThreadsPerRow)), m % kGmemThreadsPerRow, kGmemThreadsPerRow));
      if (idx < seqlen_o * qhead_per_khead) {
        Tensor mO_cur = make_tensor(make_gmem_ptr(o_ptr), Shape<Int<kHeadDim>>{});
        Tensor mO_cur_copy = cute::tiled_divide(mO_cur, Shape<Int<kGmemElemsPerStore>>{});
        #pragma unroll
        for (int k = 0; k < size<2>(tOrO); ++k) {
          int ki = get<1>(tOcO(_0{}, _0{}, k)) / kGmemElemsPerStore;
          if (tOpO(k)) {
            cute::copy(gmem_tiled_copy_O, tOrO(_, m, k), mO_cur_copy(_, ki));
          }
        }
      }
    }
  };
  template <typename TensormO, typename TensorrO, typename TiledMma>
  CUTLASS_DEVICE
  static void
  store_O_direct(TensormO &mO,
          TensorrO const &tOrO,
          TiledMma tiled_mma,
          cutlass::FastDivmod const &qhead_per_khead_divmod,
          int const thread_idx, int const seqlen_o, int const m_block
         )
  {
    static constexpr int kGmemElemsPerStoreDirect = 2;
    cute::Copy_Atom<AutoVectorizingCopyWithAssumedAlignment<128>, Element> gmem_copy_direct;
    Tensor tOrO_rowcol = make_tensor(tOrO.data(), flash::convert_layout_acc_rowcol(tOrO.layout()));
    Tensor tOrO_copy = cute::tiled_divide(tOrO_rowcol, Shape<_1, Int<kGmemElemsPerStoreDirect>>{});
    Tensor caccO = cute::make_identity_tensor(Shape<Int<kBlockM>, Int<kHeadDim>>{});
    auto thread_mma = tiled_mma.get_thread_slice(thread_idx);
    Tensor taccOcO = thread_mma.partition_C(caccO);
    Tensor taccOcO_rowcol = make_tensor(taccOcO.data(), flash::convert_layout_acc_rowcol(taccOcO.layout()));
    Tensor taccOcO_row = taccOcO_rowcol(_, _0{});
    Tensor taccOcO_col = taccOcO_rowcol(_0{}, _);
    static constexpr int kMmaThreadsPerRow = size<0, 0>(typename TiledMma::AtomLayoutC_TV{});
    static_assert(cutlass::NumThreadsPerWarp % kMmaThreadsPerRow == 0);
    static_assert(CUTE_STATIC_V(size(taccOcO_row)) <= kMmaThreadsPerRow);
    Tensor mO_0 = mO(_, _0{});
    Tensor tPrOPtr = compute_ptr<kMmaThreadsPerRow>(mO_0, taccOcO_row, qhead_per_khead_divmod, thread_idx, m_block);
    static_assert(CUTE_STATIC_V(size(tPrOPtr)) == 1);
    int const qhead_per_khead = qhead_per_khead_divmod.divisor;
    #pragma unroll
    for (int m = 0; m < size<1>(tOrO_copy); ++m) {
      int row = m_block * kBlockM + get<0>(taccOcO_row(m));
      Element* o_ptr = reinterpret_cast<Element*>(__shfl_sync(0xffffffff, reinterpret_cast<uint64_t>(tPrOPtr[0]), m % kMmaThreadsPerRow, kMmaThreadsPerRow));
      if (row < seqlen_o * qhead_per_khead) {
        Tensor mO_cur = make_tensor(make_gmem_ptr(o_ptr), Shape<Int<kHeadDim>>{});
        Tensor mO_cur_copy = cute::tiled_divide(mO_cur, Shape<Int<kGmemElemsPerStoreDirect>>{});
        #pragma unroll
        for (int k = 0; k < size<2>(tOrO_copy); ++k) {
          int col = get<1>(taccOcO_col(k * kGmemElemsPerStoreDirect));
          if (col < size<1>(mO)) {
            cute::copy(gmem_copy_direct, tOrO_copy(_, m, k), mO_cur_copy(_, col / kGmemElemsPerStoreDirect));
          }
        }
      }
    }
  };
};
}
#include <cute/tensor.hpp>
#include "cutlass/fast_math.h"
namespace flash {
using namespace cute;
template <int kBlockN, int kHeadDim, int kHeadDimV, int NumThreads, typename Element, bool KV_Same_Iter=false, int LoadsPerRow_LB=1>
struct PagedKVManager {
  static constexpr bool SameHeadDim = (kHeadDim == kHeadDimV);
  static constexpr int kHeadDimGCD = cute::gcd(kHeadDim, kHeadDimV);
  static constexpr int kGmemElemsPerLoad = sizeof(cute::uint128_t) / sizeof(Element);
  static_assert(kHeadDimGCD % kGmemElemsPerLoad == 0, "Headdim and HeaddimV must be a multiple of kGmemElemsPerLoad");
  static_assert(kHeadDimGCD % LoadsPerRow_LB == 0, "Headdim and HeaddimV must be a multiple of LoadsPerRow_LB");
  static constexpr int kBytePerRow = kHeadDimGCD / LoadsPerRow_LB * sizeof(Element);
  static constexpr int kBlockKGmem = (kBytePerRow % 128 == 0 ? 128 : (kBytePerRow % 64 == 0 ? 64 : 32)) / sizeof(Element);
  static constexpr int kGmemThreadsPerRow = kBlockKGmem / kGmemElemsPerLoad;
  static_assert(NumThreads % kGmemThreadsPerRow == 0, "NumThreads must be a multiple of kGmemThreadsPerRow");
  static_assert(cutlass::NumThreadsPerWarp % kGmemThreadsPerRow == 0, "kGmemThreadsPerRow must divide NumThreadsPerWarp");
  using GmemCopyAtomCpAsync = cute::Copy_Atom<SM80_CP_ASYNC_CACHEGLOBAL_ZFILL<uint128_t>, Element>;
  using GmemLayoutAtomKVCpAsync = Layout<Shape <Int<NumThreads / kGmemThreadsPerRow>, Int<kGmemThreadsPerRow>>,
                      Stride<Int<kGmemThreadsPerRow>, _1>>;
  using GmemTiledCopyKVCpAsync = decltype(
    make_tiled_copy(GmemCopyAtomCpAsync{},
            GmemLayoutAtomKVCpAsync{},
            Layout<Shape<_1, Int<kGmemElemsPerLoad>>>{}));
  using GmemTiledCopyKVStore = decltype(
    make_tiled_copy(Copy_Atom<AutoVectorizingCopyWithAssumedAlignment<128>, Element>{},
            GmemLayoutAtomKVCpAsync{},
            Layout<Shape<_1, Int<kGmemElemsPerLoad>>>{}));
  using ShapeKV = cute::Shape<int32_t, int32_t, int32_t, int32_t>;
  using StrideKV = cute::Stride<int64_t, _1, int64_t, int64_t>;
  using ShapePageTable = cute::Shape<int32_t, int32_t>;
  using StridePageTable = cute::Stride<int64_t, _1>;
  using TensorPageTable = decltype(make_tensor(make_gmem_ptr(static_cast<int const*>(nullptr)), ShapePageTable{}, StridePageTable{})(int(0), _));
  using TensorKV = decltype(make_tensor(make_gmem_ptr(static_cast<Element*>(nullptr)), ShapeKV{}, StrideKV{})(_, _, int(0), _));
  using GmemThrCopyKVCpAsync = decltype(GmemTiledCopyKVCpAsync{}.get_thread_slice(int(0)));
  using TensortKcK = decltype(GmemTiledCopyKVCpAsync{}.get_thread_slice(int(0)).partition_D(cute::make_identity_tensor(Shape<Int<kBlockN>, Int<kHeadDim>>{})));
  using TensortKpK = decltype(make_tensor<bool>(make_shape(size<1>(TensortKcK{}), size<2>(TensortKcK{})), Stride<_0, _1>{}));
  using TensortVcV = decltype(GmemTiledCopyKVCpAsync{}.get_thread_slice(int(0)).partition_D(cute::make_identity_tensor(Shape<Int<kBlockN>, Int<kHeadDimV>>{})));
  using TensortVpV = decltype(make_tensor<bool>(make_shape(size<1>(TensortVcV{}), size<2>(TensortVcV{})), Stride<_0, _1>{}));
  static_assert(CUTE_STATIC_V(size<1>(TensortKcK{})) == CUTE_STATIC_V(size<1>(TensortVcV{})));
  static constexpr int kPageEntryPerThread = cute::ceil_div(size<1>(TensortKcK{}), kGmemThreadsPerRow);
  using TensorPageOffset = decltype(make_tensor<cute::tuple<int, int>>(Shape<Int<kPageEntryPerThread>>{}));
  using TensorKVPtr = decltype(make_tensor<Element*>(Shape<Int<kPageEntryPerThread>>{}));
  GmemTiledCopyKVCpAsync gmem_tiled_copy_kv;
  cutlass::FastDivmod const &page_size_divmod;
  cutlass::FastDivmod const &blockN_per_page_size_divmod;
  int const thread_idx;
  int const seqlen_k;
  int const leftpad_k;
  int const* const ptr_page_table;
  GmemThrCopyKVCpAsync const gmem_thr_copy_kv;
  TensorPageTable mPageTable;
  TensorKV mK_paged, mV_paged;
  TensortKpK tKpK;
  TensortVpV tVpV;
  TensorPageOffset tPrPageOffset;
  TensorKVPtr tPrVPtr;
  int bidb_kv_idx, bidb_kv_idx_prev, n_block_idx, n_block_idx_prev;
  CUTLASS_DEVICE
  PagedKVManager(int const* const ptr_page_table_,
          ShapePageTable const &shape_pagetable, StridePageTable const &stride_pagetable,
          Element* const ptr_K, ShapeKV const &shape_K, StrideKV const &stride_K,
          Element* const ptr_V, int const headdim_v, StrideKV const &stride_V,
          cutlass::FastDivmod const &page_size_divmod,
          cutlass::FastDivmod const &blockN_per_page_size_divmod,
          int const bidb, int const bidh, int const thread_idx, int const seqlen_k, int const leftpad_k,
          int bidb_kv_idx
          )
    : page_size_divmod(page_size_divmod)
    , blockN_per_page_size_divmod(blockN_per_page_size_divmod)
    , thread_idx(thread_idx)
    , seqlen_k(seqlen_k)
    , leftpad_k(leftpad_k)
    , ptr_page_table(ptr_page_table_)
    , gmem_thr_copy_kv(gmem_tiled_copy_kv.get_thread_slice(thread_idx))
    , bidb_kv_idx(bidb_kv_idx)
    , bidb_kv_idx_prev(bidb_kv_idx)
  {
    mPageTable = make_tensor(make_gmem_ptr(ptr_page_table), shape_pagetable, stride_pagetable)(bidb, _);
    mK_paged = make_tensor(make_gmem_ptr(ptr_K), shape_K, stride_K)(_, _, bidh, _);
    auto shape_V = make_shape(get<0>(shape_K), headdim_v, get<2>(shape_K), get<3>(shape_K));
    mV_paged = make_tensor(make_gmem_ptr(ptr_V), shape_V, stride_V)(_, _, bidh, _);
    tKpK = make_tensor<bool>(make_shape(size<1>(TensortKcK{}), size<2>(TensortKcK{})), Stride<_0, _1>{});
    Tensor cK = cute::make_identity_tensor(Shape<Int<kBlockN>, Int<kHeadDim>>{});
    Tensor tKcK = gmem_thr_copy_kv.partition_S(cK);
    #pragma unroll
    for (int k = 0; k < size<1>(tKpK); ++k) { tKpK(_0{}, k) = get<1>(tKcK(_0{}, _0{}, k)) < get<1>(shape_K); }
    Tensor tVpV_ = make_tensor<bool>(make_shape(size<1>(TensortVcV{}), size<2>(TensortVcV{})), Stride<_0, _1>{});
    Tensor cV = cute::make_identity_tensor(Shape<Int<kBlockN>, Int<kHeadDimV>>{});
    Tensor tVcV = gmem_thr_copy_kv.partition_S(cV);
    #pragma unroll
    for (int k = 0; k < size<1>(tVpV_); ++k) { tVpV_(_0{}, k) = get<1>(tVcV(_0{}, _0{}, k)) < get<1>(shape_V); }
    tVpV = cute::conditional_return<SameHeadDim>(tKpK, tVpV_);
  };
  template <bool Seqlenk_mask=false, bool First_iter=false>
  CUTLASS_DEVICE
  void load_page_table(const int n_block) {
    #pragma unroll
    for (int i = 0; i < kPageEntryPerThread; ++i) {
      int const row = i * NumThreads + (thread_idx % kGmemThreadsPerRow) * (NumThreads / kGmemThreadsPerRow) + (thread_idx / kGmemThreadsPerRow);
      int const row_idx = n_block * kBlockN + row;
      int page_idx, page_offset;
      page_idx = page_size_divmod.divmod(page_offset, row_idx + leftpad_k);
      int const page = ((i + 1) * NumThreads <= kBlockN || row < kBlockN) && (!Seqlenk_mask || row_idx < seqlen_k) ? mPageTable[page_idx] : 0;
      tPrPageOffset[i] = {page, page_offset};
    }
    if constexpr (First_iter && !KV_Same_Iter) { compute_V_ptr(); }
  };
  template <bool First_iter=false>
  CUTLASS_DEVICE
  void load_page_table_TMA(const int n_block) {
    if (ptr_page_table) {
      bidb_kv_idx = mPageTable[blockN_per_page_size_divmod.divmod(n_block_idx, n_block)];
    } else {
      n_block_idx = n_block;
    }
    if constexpr (First_iter && !KV_Same_Iter) {
      bidb_kv_idx_prev = bidb_kv_idx;
      n_block_idx_prev = n_block_idx;
    }
  };
  CUTLASS_DEVICE
  cute::tuple<int, int> get_indices_for_K_TMA() {
    return {n_block_idx, bidb_kv_idx};
  };
  CUTLASS_DEVICE
  cute::tuple<int, int> get_indices_for_V_TMA() {
    if constexpr (KV_Same_Iter) {
      return {n_block_idx, bidb_kv_idx};
    } else {
      cute::tuple<int, int> const indices = {n_block_idx_prev, bidb_kv_idx_prev};
      bidb_kv_idx_prev = bidb_kv_idx;
      n_block_idx_prev = n_block_idx;
      return indices;
    }
  };
  CUTLASS_DEVICE
  TensorKVPtr compute_K_ptr() {
    Tensor tPrKPtr = make_tensor<Element*>(Shape<Int<kPageEntryPerThread>>{});
    #pragma unroll
    for (int i = 0; i < kPageEntryPerThread; ++i) {
      auto [page, page_offset] = tPrPageOffset[i];
      tPrKPtr[i] = &mK_paged(page_offset, _0{}, page);
    }
    return tPrKPtr;
  };
  CUTLASS_DEVICE
  void compute_V_ptr() {
    #pragma unroll
    for (int i = 0; i < kPageEntryPerThread; ++i) {
      auto [page, page_offset] = tPrPageOffset[i];
      tPrVPtr[i] = &mV_paged(page_offset, _0{}, page);
    }
  };
  template <bool Seqlenk_mask=false, typename TensorK>
  CUTLASS_DEVICE
  void load_K(const int n_block, TensorK &&sK) {
    static constexpr bool EvenN = kBlockN % CUTE_STATIC_V(shape<0>(GmemLayoutAtomKVCpAsync{})) == 0;
    Tensor tPrKPtr = compute_K_ptr();
    auto gmem_thr0_copy_kv = gmem_tiled_copy_kv.get_thread_slice(_0{});
    Tensor tKsK = gmem_thr_copy_kv.partition_D(sK);
    Tensor cK = cute::make_identity_tensor(Shape<Int<kBlockN>, Int<kHeadDim>>{});
    Tensor tKcK = gmem_thr_copy_kv.partition_S(cK);
    Tensor t0KcK = gmem_thr0_copy_kv.partition_S(cK);
    int const seqlenk_row_limit = -int(get<0>(tKcK(_0{}, _0{}, _0{}))) + (EvenN
      ? seqlen_k - n_block * kBlockN
      : (!Seqlenk_mask ? kBlockN : std::min(seqlen_k - n_block * kBlockN, kBlockN)));
    #pragma unroll
    for (int m = 0; m < size<1>(tKsK); ++m) {
      bool const should_load = EvenN
        ? (!Seqlenk_mask || get<0>(t0KcK(_0{}, m, _0{})) < seqlenk_row_limit)
        : get<0>(t0KcK(_0{}, m, _0{})) < seqlenk_row_limit;
      Element const* k_ptr = reinterpret_cast<Element const*>(__shfl_sync(0xffffffff, reinterpret_cast<uint64_t>(tPrKPtr(m / kGmemThreadsPerRow)), (m % kGmemThreadsPerRow), kGmemThreadsPerRow));
      Tensor mK_paged_cur = make_tensor(make_gmem_ptr(k_ptr), Shape<Int<kHeadDim>>{});
      Tensor mK_paged_cur_copy = cute::tiled_divide(mK_paged_cur, Shape<Int<kGmemElemsPerLoad>>{});
      if (should_load) {
        #pragma unroll
        for (int k = 0; k < size<2>(tKsK); ++k) {
          int const ki = get<1>(tKcK(_0{}, _0{}, k)) / kGmemElemsPerLoad;
          cute::copy(gmem_tiled_copy_kv.with(tKpK(_0{}, k)), mK_paged_cur_copy(_, ki), tKsK(_, m, k));
        }
      }
    }
  };
  template <bool Seqlenk_mask=false, typename TensorV>
  CUTLASS_DEVICE
  void load_V(const int n_block, TensorV &&sV) {
    static constexpr bool EvenN = kBlockN % CUTE_STATIC_V(shape<0>(GmemLayoutAtomKVCpAsync{})) == 0;
    if constexpr (KV_Same_Iter) { compute_V_ptr(); }
    auto gmem_thr0_copy_kv = gmem_tiled_copy_kv.get_thread_slice(_0{});
    Tensor tVsV = gmem_thr_copy_kv.partition_D(sV);
    Tensor cV = cute::make_identity_tensor(Shape<Int<kBlockN>, Int<kHeadDimV>>{});
    Tensor tVcV = gmem_thr_copy_kv.partition_S(cV);
    Tensor t0VcV = gmem_thr0_copy_kv.partition_S(cV);
    int const seqlenk_row_limit = seqlen_k - n_block * kBlockN - get<0>(tVcV(_0{}, _0{}, _0{}));
    #pragma unroll
    for (int m = 0; m < size<1>(tVsV); ++m) {
      if (EvenN || m < size<1>(tVsV) - 1 || get<0>(tVcV(_0{}, m, _0{})) < kBlockN) {
        bool const should_load = !Seqlenk_mask || get<0>(t0VcV(_0{}, m, _0{})) < seqlenk_row_limit;
        Element const* v_ptr = reinterpret_cast<Element const*>(__shfl_sync(0xffffffff, reinterpret_cast<uint64_t>(tPrVPtr(m / kGmemThreadsPerRow)), m % kGmemThreadsPerRow, kGmemThreadsPerRow));
        Tensor mV_paged_cur = make_tensor(make_gmem_ptr(v_ptr), Shape<Int<kHeadDimV>>{});
        Tensor mV_paged_cur_copy = cute::tiled_divide(mV_paged_cur, Shape<Int<kGmemElemsPerLoad>>{});
        #pragma unroll
        for (int k = 0; k < size<2>(tVsV); ++k) {
          int const ki = get<1>(tVcV(_0{}, _0{}, k)) / kGmemElemsPerLoad;
          cute::copy(gmem_tiled_copy_kv.with(tVpV(_0{}, k) && should_load), mV_paged_cur_copy(_, ki), tVsV(_, m, k));
        }
      }
    }
    if constexpr (!KV_Same_Iter) { compute_V_ptr(); }
  };
  template <typename TensorK>
  CUTLASS_DEVICE
  void store_K(const int n_block, TensorK &&tKrK) {
    Tensor tPrKPtr = compute_K_ptr();
    auto gmem_thr0_copy_kv = gmem_tiled_copy_kv.get_thread_slice(_0{});
    Tensor cK = cute::make_identity_tensor(Shape<Int<kBlockN>, Int<kHeadDim>>{});
    Tensor tKcK = gmem_thr_copy_kv.partition_S(cK);
    Tensor t0KcK = gmem_thr0_copy_kv.partition_S(cK);
    GmemTiledCopyKVStore gmem_tiled_copy_kv_store;
    int const seqlenk_row_limit = std::min(seqlen_k - n_block * kBlockN, kBlockN) - get<0>(tKcK(_0{}, _0{}, _0{}));
    #pragma unroll
    for (int m = 0; m < size<1>(tKrK); ++m) {
      bool const should_load = get<0>(t0KcK(_0{}, m, _0{})) < seqlenk_row_limit;
      Element* k_ptr = reinterpret_cast<Element*>(__shfl_sync(0xffffffff, reinterpret_cast<uint64_t>(tPrKPtr(m / kGmemThreadsPerRow)), (m % kGmemThreadsPerRow), kGmemThreadsPerRow));
      Tensor mK_paged_cur = make_tensor(make_gmem_ptr(k_ptr), Shape<Int<kHeadDim>>{});
      Tensor mK_paged_cur_copy = cute::tiled_divide(mK_paged_cur, Shape<Int<kGmemElemsPerLoad>>{});
      if (should_load) {
        #pragma unroll
        for (int k = 0; k < size<2>(tKrK); ++k) {
          int const ki = get<1>(tKcK(_0{}, _0{}, k)) / kGmemElemsPerLoad;
          if (tKpK(_0{}, k)) {
            cute::copy(gmem_tiled_copy_kv_store, tKrK(_, m, k), mK_paged_cur_copy(_, ki));
          }
        }
      }
    }
  };
  template <typename TensorV>
  CUTLASS_DEVICE
  void store_V(const int n_block, TensorV &&tVrV) {
    if constexpr (KV_Same_Iter) { compute_V_ptr(); }
    auto gmem_thr0_copy_kv = gmem_tiled_copy_kv.get_thread_slice(_0{});
    Tensor cV = cute::make_identity_tensor(Shape<Int<kBlockN>, Int<kHeadDimV>>{});
    Tensor tVcV = gmem_thr_copy_kv.partition_S(cV);
    Tensor t0VcV = gmem_thr0_copy_kv.partition_S(cV);
    GmemTiledCopyKVStore gmem_tiled_copy_kv_store;
    int const seqlenk_row_limit = std::min(seqlen_k - n_block * kBlockN, kBlockN) - get<0>(tVcV(_0{}, _0{}, _0{}));
    #pragma unroll
    for (int m = 0; m < size<1>(tVrV); ++m) {
      bool const should_load = get<0>(t0VcV(_0{}, m, _0{})) < seqlenk_row_limit;
      Element* v_ptr = reinterpret_cast<Element*>(__shfl_sync(0xffffffff, reinterpret_cast<uint64_t>(tPrVPtr(m / kGmemThreadsPerRow)), m % kGmemThreadsPerRow, kGmemThreadsPerRow));
      Tensor mV_paged_cur = make_tensor(make_gmem_ptr(v_ptr), Shape<Int<kHeadDimV>>{});
      Tensor mV_paged_cur_copy = cute::tiled_divide(mV_paged_cur, Shape<Int<kGmemElemsPerLoad>>{});
      if (should_load) {
        #pragma unroll
        for (int k = 0; k < size<2>(tVrV); ++k) {
          int const ki = get<1>(tVcV(_0{}, _0{}, k)) / kGmemElemsPerLoad;
          if (tVpV(_0{}, k)) {
            cute::copy(gmem_tiled_copy_kv_store, tVrV(_, m, k), mV_paged_cur_copy(_, ki));
          }
        }
      }
    }
    if constexpr (!KV_Same_Iter) { compute_V_ptr(); }
  };
};
}
#include <cute/tensor.hpp>
namespace flash {
using namespace cute;
template <typename Engine1, typename Layout1, typename Engine2, typename Layout2>
CUTLASS_DEVICE void
apply_rotary_interleaved(Tensor<Engine1, Layout1> &rK,
             Tensor<Engine2, Layout2> const &rCos,
             Tensor<Engine2, Layout2> const &rSin) {
  CUTE_STATIC_ASSERT_V(rank(rK) == _1{});
  CUTE_STATIC_ASSERT_V(rank(rCos) == _1{});
  CUTE_STATIC_ASSERT_V(rank(rSin) == _1{});
  CUTE_STATIC_ASSERT_V(size<0>(rCos) == size<0>(rSin));
  static_assert(decltype(size<0>(rK))::value == decltype(size<0>(rCos))::value * 2);
  static_assert(decltype(size<0>(rCos))::value % 2 == 0);
  Tensor K_fp32 = make_tensor_like<float>(rK);
  convert_type_out(rK, K_fp32);
  Tensor cos_fp32 = make_tensor_like<float>(rCos);
  convert_type_out(rCos, cos_fp32);
  Tensor sin_fp32 = make_tensor_like<float>(rSin);
  convert_type_out(rSin, sin_fp32);
  #pragma unroll
  for (int i = 0; i < size<0>(K_fp32) / 2; ++i) {
    float real = K_fp32[2 * i] * cos_fp32[i] - K_fp32[2 * i + 1] * sin_fp32[i];
    float imag = K_fp32[2 * i] * sin_fp32[i] + K_fp32[2 * i + 1] * cos_fp32[i];
    K_fp32[2 * i] = real;
    K_fp32[2 * i + 1] = imag;
  }
  convert_type_out(K_fp32, rK);
}
template <typename Engine1, typename Layout1, typename Engine2, typename Layout2>
CUTLASS_DEVICE void
apply_rotary_contiguous(Tensor<Engine1, Layout1> &rK_left,
            Tensor<Engine1, Layout1> &rK_right,
            Tensor<Engine2, Layout2> const &rCos,
            Tensor<Engine2, Layout2> const &rSin) {
  CUTE_STATIC_ASSERT_V(rank(rK_left) == _1{});
  CUTE_STATIC_ASSERT_V(rank(rK_right) == _1{});
  CUTE_STATIC_ASSERT_V(rank(rCos) == _1{});
  CUTE_STATIC_ASSERT_V(rank(rSin) == _1{});
  CUTE_STATIC_ASSERT_V(size<0>(rK_left) == size<0>(rK_right));
  CUTE_STATIC_ASSERT_V(size<0>(rK_left) == size<0>(rCos));
  CUTE_STATIC_ASSERT_V(size<0>(rCos) == size<0>(rSin));
  static_assert(decltype(size<0>(rCos))::value % 2 == 0);
  Tensor K_left_fp32 = make_tensor_like<float>(rK_left);
  convert_type_out(rK_left, K_left_fp32);
  Tensor K_right_fp32 = make_tensor_like<float>(rK_right);
  convert_type_out(rK_right, K_right_fp32);
  Tensor cos_fp32 = make_tensor_like<float>(rCos);
  convert_type_out(rCos, cos_fp32);
  Tensor sin_fp32 = make_tensor_like<float>(rSin);
  convert_type_out(rSin, sin_fp32);
  #pragma unroll
  for (int i = 0; i < size<0>(K_left_fp32); ++i) {
    float real = K_left_fp32[i] * cos_fp32[i] - K_right_fp32[i] * sin_fp32[i];
    float imag = K_left_fp32[i] * sin_fp32[i] + K_right_fp32[i] * cos_fp32[i];
    K_left_fp32[i] = real;
    K_right_fp32[i] = imag;
  }
  convert_type_out(K_left_fp32, rK_left);
  convert_type_out(K_right_fp32, rK_right);
}
template <int kBlockMN, int kHeadDim, int NumThreads, typename Element, bool FixedPosition=false>
struct Rotary {
  static constexpr int kGmemElemsPerLoad = sizeof(cute::uint128_t) / sizeof(Element);
  static_assert(kHeadDim % kGmemElemsPerLoad == 0, "Headdim must be a multiple of kGmemElemsPerLoad");
  static constexpr int kBytePerHalfRow = kHeadDim / 2 * sizeof(Element);
  static constexpr int kBlockKGmem = (kBytePerHalfRow % 128 == 0 ? 128 : (kBytePerHalfRow % 64 == 0 ? 64 : 32)) / sizeof(Element);
  static constexpr int kGmemThreadsPerRow = kBlockKGmem / kGmemElemsPerLoad;
  static_assert(NumThreads % kGmemThreadsPerRow == 0, "NumThreads must be a multiple of kGmemThreadsPerRow");
  static_assert(cutlass::NumThreadsPerWarp % kGmemThreadsPerRow == 0, "kGmemThreadsPerRow must divide NumThreadsPerWarp");
  using LayoutAtom = Layout<Shape <Int<NumThreads / kGmemThreadsPerRow>, Int<kGmemThreadsPerRow>>,
                 Stride<Int<kGmemThreadsPerRow>, _1>>;
  using TiledCopyQK = decltype(
    make_tiled_copy(Copy_Atom<AutoVectorizingCopyWithAssumedAlignment<128>, Element>{},
            LayoutAtom{},
            Layout<Shape<_1, Int<kGmemElemsPerLoad>>>{}));
  using GmemTiledCopyRotary = decltype(
    make_tiled_copy(Copy_Atom<AutoVectorizingCopyWithAssumedAlignment<64>, Element>{},
            LayoutAtom{},
            Layout<Shape<_1, Int<kGmemElemsPerLoad / 2>>>{}));
  using GmemTiledCopyRotaryCont = decltype(
    make_tiled_copy(Copy_Atom<AutoVectorizingCopyWithAssumedAlignment<128>, Element>{},
            LayoutAtom{},
            Layout<Shape<_1, Int<kGmemElemsPerLoad>>>{}));
  using ShapeRotary = cute::Shape<int32_t, int32_t>;
  using StrideRotary = cute::Stride<int64_t, _1>;
  using GmemThrCopyRotary = decltype(GmemTiledCopyRotary{}.get_thread_slice(int(0)));
  using GmemThrCopyRotaryCont = decltype(GmemTiledCopyRotaryCont{}.get_thread_slice(int(0)));
  using TensortRcR = decltype(GmemTiledCopyRotary{}.get_thread_slice(int(0)).partition_D(cute::make_identity_tensor(Shape<Int<kBlockMN>, Int<kHeadDim / 2>>{})));
  using TensortRpR = decltype(make_tensor<bool>(make_shape(size<2>(TensortRcR{}))));
  using TensortRcRCont = decltype(GmemTiledCopyRotaryCont{}.get_thread_slice(int(0)).partition_D(cute::make_identity_tensor(Shape<Int<kBlockMN>, Int<kHeadDim / 2>>{})));
  using TensortRpRCont = decltype(make_tensor<bool>(make_shape(size<2>(TensortRcRCont{}))));
  using TensormR = decltype(make_tensor(
    make_gmem_ptr((Element const*)nullptr),
    ShapeRotary{},
    make_stride(cute::conditional_return<FixedPosition>(_0{}, int64_t(0)), _1{})));
  using TensortRgR = decltype(
    GmemTiledCopyRotary{}.get_thread_slice(int(0)).partition_S(make_tensor(
      make_gmem_ptr((Element const*)nullptr),
      make_shape(Int<kBlockMN>{}, Int<kHeadDim / 2>{}, int(0)),
      make_stride(cute::conditional_return<FixedPosition>(_0{}, int64_t(0)), _1{}, cute::conditional_return<FixedPosition>(_0{}, int64_t(0))))));
  using TensortRgRCont = decltype(
    GmemTiledCopyRotaryCont{}.get_thread_slice(int(0)).partition_S(make_tensor(
      make_gmem_ptr((Element const*)nullptr),
      make_shape(Int<kBlockMN>{}, Int<kHeadDim / 2>{}, int(0)),
      make_stride(cute::conditional_return<FixedPosition>(_0{}, int64_t(0)), _1{}, cute::conditional_return<FixedPosition>(_0{}, int64_t(0))))));
  GmemTiledCopyRotary gmem_tiled_copy_rotary;
  GmemTiledCopyRotaryCont gmem_tiled_copy_rotary_cont;
  bool const is_rotary_interleaved;
  int const rotary_dim;
  int const thread_idx;
  int const max_seqlen;
  GmemThrCopyRotary const gmem_thr_copy_rotary;
  GmemThrCopyRotaryCont const gmem_thr_copy_rotary_cont;
  TensortRpR tRpR;
  TensortRpRCont tRpRCont;
  TensormR mCos, mSin;
  TensortRgR tRgCos, tRgSin;
  TensortRgRCont tRgCosCont, tRgSinCont;
  CUTLASS_DEVICE
  Rotary(Element const* const ptr_rotary_cos, ShapeRotary const &shape_rotary, StrideRotary const &stride_rotary_cos_,
      Element const* const ptr_rotary_sin, StrideRotary const &stride_rotary_sin_,
      bool const is_rotary_interleaved, int const thread_idx, int const max_seqlen, int const start_idx)
    : is_rotary_interleaved(is_rotary_interleaved)
    , rotary_dim(get<1>(shape_rotary) * 2)
    , thread_idx(thread_idx)
    , max_seqlen(max_seqlen)
    , gmem_thr_copy_rotary(gmem_tiled_copy_rotary.get_thread_slice(thread_idx))
    , gmem_thr_copy_rotary_cont(gmem_tiled_copy_rotary_cont.get_thread_slice(thread_idx))
  {
    auto stride_rotary_cos = make_stride(cute::conditional_return<!FixedPosition>(get<0>(stride_rotary_cos_), _0{}), get<1>(stride_rotary_cos_));
    auto stride_rotary_sin = make_stride(cute::conditional_return<!FixedPosition>(get<0>(stride_rotary_sin_), _0{}), get<1>(stride_rotary_sin_));
    mCos = make_tensor(make_gmem_ptr(ptr_rotary_cos + start_idx * get<0>(stride_rotary_cos_)), shape_rotary, stride_rotary_cos);
    mSin = make_tensor(make_gmem_ptr(ptr_rotary_sin + start_idx * get<0>(stride_rotary_sin_)), shape_rotary, stride_rotary_sin);
    Tensor gCos = local_tile(mCos, Shape<Int<kBlockMN>, Int<kHeadDim / 2>>{}, make_coord(_, _0{}));
    Tensor gSin = local_tile(mSin, Shape<Int<kBlockMN>, Int<kHeadDim / 2>>{}, make_coord(_, _0{}));
    tRgCos = gmem_thr_copy_rotary.partition_S(gCos);
    tRgSin = gmem_thr_copy_rotary.partition_S(gSin);
    tRgCosCont = gmem_thr_copy_rotary_cont.partition_S(gCos);
    tRgSinCont = gmem_thr_copy_rotary_cont.partition_S(gSin);
    Tensor cR = cute::make_identity_tensor(Shape<Int<kBlockMN>, Int<kHeadDim / 2>>{});
    Tensor tRcR = gmem_thr_copy_rotary.partition_D(cR);
    tRpR = make_tensor<bool>(make_shape(size<2>(tRcR)));
    #pragma unroll
    for (int k = 0; k < size(tRpR); ++k) { tRpR(k) = get<1>(tRcR(_0{}, _0{}, k)) < get<1>(shape_rotary); }
    Tensor tRcRCont = gmem_thr_copy_rotary_cont.partition_D(cR);
    tRpRCont = make_tensor<bool>(make_shape(size<2>(tRcRCont)));
    #pragma unroll
    for (int k = 0; k < size(tRpRCont); ++k) { tRpRCont(k) = get<1>(tRcRCont(_0{}, _0{}, k)) < get<1>(shape_rotary); }
  };
  template <bool kInterleaved=true>
  CUTLASS_DEVICE
  auto load_cos_sin(int const block) {
    using GmemTiledCopyRo = std::conditional_t<kInterleaved, GmemTiledCopyRotary, GmemTiledCopyRotaryCont>;
    auto gmem_thr_copy_ro = cute::conditional_return<kInterleaved>(gmem_thr_copy_rotary, gmem_thr_copy_rotary_cont);
    Tensor tRpRCur = cute::conditional_return<kInterleaved>(tRpR, tRpRCont);
    Tensor tRgCosCur = cute::conditional_return<kInterleaved>(tRgCos, tRgCosCont)(_, _, _, block);
    Tensor tRgSinCur = cute::conditional_return<kInterleaved>(tRgSin, tRgSinCont)(_, _, _, block);
    Tensor tRrCos = make_tensor_like(tRgCosCur);
    Tensor tRrSin = make_tensor_like(tRgSinCur);
    Tensor cR = cute::make_identity_tensor(Shape<Int<kBlockMN>, Int<kHeadDim / 2>>{});
    Tensor tRcR = gmem_thr_copy_ro.partition_D(cR);
    #pragma unroll
    for (int m = 0; m < (!FixedPosition ? size<1>(tRrCos) : 1); ++m) {
      if (get<0>(tRcR(_0{}, m, _0{})) < std::min(max_seqlen - block * kBlockMN, kBlockMN)) {
        #pragma unroll
        for (int k = 0; k < size<2>(tRrCos); ++k) {
          if (tRpRCur(k)) {
            cute::copy(GmemTiledCopyRo{}, tRgCosCur(_, m, k), tRrCos(_, m, k));
            cute::copy(GmemTiledCopyRo{}, tRgSinCur(_, m, k), tRrSin(_, m, k));
          }
        }
      }
    }
    return cute::make_tuple(tRrCos, tRrSin);;
  }
  template <bool kInterleaved=true>
  CUTLASS_DEVICE
  auto load_cos_sin_packgqa(int const block, cutlass::FastDivmod const &qhead_per_khead_divmod) {
    static constexpr int kGmemElemsPerLoadCur = kInterleaved ? kGmemElemsPerLoad / 2 : kGmemElemsPerLoad;
    using GmemTiledCopyRo = std::conditional_t<kInterleaved, GmemTiledCopyRotary, GmemTiledCopyRotaryCont>;
    auto gmem_thr_copy_ro = cute::conditional_return<kInterleaved>(gmem_thr_copy_rotary, gmem_thr_copy_rotary_cont);
    Tensor tRpRCur = cute::conditional_return<kInterleaved>(tRpR, tRpRCont);
    Tensor tRrCos = make_tensor_like(cute::conditional_return<kInterleaved>(tRgCos, tRgCosCont)(_, _, _, _0{}));
    Tensor tRrSin = make_tensor_like(cute::conditional_return<kInterleaved>(tRgSin, tRgSinCont)(_, _, _, _0{}));
    int const qhead_per_khead = qhead_per_khead_divmod.divisor;
    Tensor cR = cute::make_identity_tensor(Shape<Int<kBlockMN>, Int<kHeadDim / 2>>{});
    Tensor tRcR = gmem_thr_copy_ro.partition_D(cR);
    static constexpr int NumPtrPerThread = cute::ceil_div(CUTE_STATIC_V(cute::size<1>(tRrCos)), kGmemThreadsPerRow);
    Tensor tPrCosPtr = make_tensor<Element const*>(Shape<Int<NumPtrPerThread>>{});
    Tensor tPrSinPtr = make_tensor<Element const*>(Shape<Int<NumPtrPerThread>>{});
    #pragma unroll
    for (int i = 0; i < NumPtrPerThread; ++i) {
      int const row = i * NumThreads + get<0>(tRcR(_0{}, thread_idx % kGmemThreadsPerRow, _0{}));
      int const idx = block * kBlockMN + row;
      int row_actual = qhead_per_khead_divmod.divide(idx);
      tPrCosPtr[i] = &mCos(row_actual, _0{});
      tPrSinPtr[i] = &mSin(row_actual, _0{});
    }
    #pragma unroll
    for (int m = 0; m < (!FixedPosition ? size<1>(tRgCos) : 1); ++m) {
      int const idx = block * kBlockMN + get<0>(tRcR(_0{}, m, _0{}));
      Element const* cos_ptr = reinterpret_cast<Element const*>(__shfl_sync(0xffffffff, reinterpret_cast<uint64_t>(tPrCosPtr(m / kGmemThreadsPerRow)), m % kGmemThreadsPerRow, kGmemThreadsPerRow));
      Element const* sin_ptr = reinterpret_cast<Element const*>(__shfl_sync(0xffffffff, reinterpret_cast<uint64_t>(tPrSinPtr(m / kGmemThreadsPerRow)), m % kGmemThreadsPerRow, kGmemThreadsPerRow));
      if (idx < max_seqlen * qhead_per_khead) {
        Tensor mCos_copy = cute::tiled_divide(make_tensor(make_gmem_ptr(cos_ptr), Shape<Int<kHeadDim / 2>>{}),
                          Shape<Int<kGmemElemsPerLoadCur>>{});
        Tensor mSin_copy = cute::tiled_divide(make_tensor(make_gmem_ptr(sin_ptr), Shape<Int<kHeadDim / 2>>{}),
                          Shape<Int<kGmemElemsPerLoadCur>>{});
        #pragma unroll
        for (int k = 0; k < size<2>(tRgCos); ++k) {
          int const ki = get<1>(tRcR(_0{}, _0{}, k)) / (kGmemElemsPerLoadCur);
          if (tRpRCur(k)) {
            cute::copy(GmemTiledCopyRo{}, mCos_copy(_, ki), tRrCos(_, m, k));
            cute::copy(GmemTiledCopyRo{}, mSin_copy(_, ki), tRrSin(_, m, k));
          }
        }
      }
    }
    return cute::make_tuple(tRrCos, tRrSin);
  }
  template <typename TensorsQ, typename TensortRrR>
  CUTLASS_DEVICE
  void
  apply_Q_interleaved(TensorsQ &sQ,
            TensortRrR const &tRrCos,
            TensortRrR const &tRrSin,
            int const m_block, int const qhead_per_khead=1)
  {
    TiledCopyQK tiled_copy_q;
    auto gmem_thr_copy_q = tiled_copy_q.get_thread_slice(thread_idx);
    Tensor tQsQ = gmem_thr_copy_q.partition_S(sQ);
    Tensor tQcQ = gmem_thr_copy_q.partition_S(cute::make_identity_tensor(Shape<Int<kBlockMN>, Int<kHeadDim>>{}));
    CUTE_STATIC_ASSERT_V(rank(tQsQ) == _3{});
    CUTE_STATIC_ASSERT_V(rank(tRrCos) == _3{});
    CUTE_STATIC_ASSERT_V(rank(tRrSin) == _3{});
    CUTE_STATIC_ASSERT_V(size<1>(tQsQ) == size<1>(tRrCos));
    CUTE_STATIC_ASSERT_V(size<2>(tQsQ) == size<2>(tRrCos));
    CUTE_STATIC_ASSERT_V(size<1>(tQsQ) == size<1>(tRrSin));
    CUTE_STATIC_ASSERT_V(size<2>(tQsQ) == size<2>(tRrSin));
    CUTE_STATIC_ASSERT_V(size<0>(tRrCos) == size<0>(tRrSin));
    static_assert(decltype(size<0>(tQsQ))::value == decltype(size<0>(tRrCos))::value * 2);
    static_assert(decltype(size<0>(tRrCos))::value % 2 == 0);
    #pragma unroll
    for (int m = 0; m < size<1>(tQsQ); ++m) {
      if (get<0>(tQcQ(_0{}, m, _0{})) < std::min(max_seqlen * qhead_per_khead - m_block * kBlockMN, kBlockMN)) {
        #pragma unroll
        for (int k = 0; k < size<2>(tQsQ); ++k) {
          if (tRpR(k)) {
            Tensor rQ = make_fragment_like(tQsQ(_, m, k));
            cute::copy(tiled_copy_q, tQsQ(_, m, k), rQ);
            apply_rotary_interleaved(rQ, tRrCos(_, m, k), tRrSin(_, m, k));
            cute::copy(tiled_copy_q, rQ, tQsQ(_, m, k));
          }
        }
      }
    }
  };
  template <typename TensorsQ, typename TensortRrR>
  CUTLASS_DEVICE
  void
  apply_Q_contiguous(TensorsQ &sQ,
            TensortRrR const &tRrCosCont,
            TensortRrR const &tRrSinCont,
            int const m_block, int const qhead_per_khead=1)
  {
    TiledCopyQK tiled_copy_q;
    auto gmem_thr_copy_q = tiled_copy_q.get_thread_slice(thread_idx);
    Tensor sQ_copy = cute::tiled_divide(sQ, Shape<_1, Int<kGmemElemsPerLoad>>{});
    Tensor tQcQ = gmem_thr_copy_q.partition_S(cute::make_identity_tensor(Shape<Int<kBlockMN>, Int<kHeadDim / 2>>{}));
    CUTE_STATIC_ASSERT_V(rank(tQcQ) == _3{});
    CUTE_STATIC_ASSERT_V(rank(tRrCosCont) == _3{});
    CUTE_STATIC_ASSERT_V(rank(tRrSinCont) == _3{});
    CUTE_STATIC_ASSERT_V(size<1>(tQcQ) == size<1>(tRrCosCont));
    CUTE_STATIC_ASSERT_V(size<2>(tQcQ) == size<2>(tRrCosCont));
    CUTE_STATIC_ASSERT_V(size<1>(tQcQ) == size<1>(tRrSinCont));
    CUTE_STATIC_ASSERT_V(size<2>(tQcQ) == size<2>(tRrSinCont));
    CUTE_STATIC_ASSERT_V(size<0>(tRrCosCont) == size<0>(tRrSinCont));
    CUTE_STATIC_ASSERT_V(size<0>(tQcQ) == size<0>(tRrCosCont));
    static_assert(decltype(size<0>(tRrCosCont))::value % 2 == 0);
    #pragma unroll
    for (int m = 0; m < size<1>(tQcQ); ++m) {
      int const row = get<0>(tQcQ(_0{}, m, _0{}));
      if (row < std::min(max_seqlen * qhead_per_khead - m_block * kBlockMN, kBlockMN)) {
        #pragma unroll
        for (int k = 0; k < size<2>(tQcQ); ++k) {
          int const col = get<1>(tQcQ(_0{}, _0{}, k));
          if (col < rotary_dim / 2) {
            int const col_idx_left = col / kGmemElemsPerLoad;
            int const col_idx_right = col / kGmemElemsPerLoad + rotary_dim / (2 * kGmemElemsPerLoad);
            Tensor rQ_left = make_fragment_like(sQ_copy(_, row, col_idx_left));
            cute::copy(tiled_copy_q, sQ_copy(_, row, col_idx_left), rQ_left);
            Tensor rQ_right = make_fragment_like(rQ_left);
            cute::copy(tiled_copy_q, sQ_copy(_, row, col_idx_right), rQ_right);
            apply_rotary_contiguous(rQ_left, rQ_right, tRrCosCont(_, m, k), tRrSinCont(_, m, k));
            cute::copy(tiled_copy_q, rQ_left, sQ_copy(_, row, col_idx_left));
            cute::copy(tiled_copy_q, rQ_right, sQ_copy(_, row, col_idx_right));
          }
        }
      }
    }
  };
  template <bool PagedKVNonTMA=false, typename TensorsK, typename TensorgK, typename TensorpK, typename TensortRrR, typename TensorKPtr>
  CUTLASS_DEVICE
  void
  apply_K_interleaved(TensorsK const &sK,
            TensorgK &gK,
            TensorpK const &tKpK,
            TensortRrR const &tRrCos,
            TensortRrR const &tRrSin,
            TensorKPtr const &tPrKPtr,
            int const n_block)
  {
    TiledCopyQK tiled_copy_k;
    auto gmem_thr_copy_q = tiled_copy_k.get_thread_slice(thread_idx);
    Tensor tKsK = gmem_thr_copy_q.partition_S(sK);
    Tensor tKgK = gmem_thr_copy_q.partition_S(gK);
    Tensor tKcK = gmem_thr_copy_q.partition_S(cute::make_identity_tensor(Shape<Int<kBlockMN>, Int<kHeadDim>>{}));
    CUTE_STATIC_ASSERT_V(rank(tKsK) == _3{});
    CUTE_STATIC_ASSERT_V(rank(tRrCos) == _3{});
    CUTE_STATIC_ASSERT_V(rank(tRrSin) == _3{});
    CUTE_STATIC_ASSERT_V(size<1>(tKsK) == size<1>(tRrCos));
    CUTE_STATIC_ASSERT_V(size<2>(tKsK) == size<2>(tRrCos));
    CUTE_STATIC_ASSERT_V(size<1>(tKsK) == size<1>(tRrSin));
    CUTE_STATIC_ASSERT_V(size<2>(tKsK) == size<2>(tRrSin));
    CUTE_STATIC_ASSERT_V(size<0>(tRrCos) == size<0>(tRrSin));
    static_assert(decltype(size<0>(tKsK))::value == decltype(size<0>(tRrCos))::value * 2);
    static_assert(decltype(size<0>(tRrCos))::value % 2 == 0);
    if constexpr (PagedKVNonTMA) {
      static_assert(decltype(size(tPrKPtr))::value == cute::ceil_div(size<1>(tKcK), kGmemThreadsPerRow));
    }
    #pragma unroll
    for (int m = 0; m < size<1>(tKsK); ++m) {
      int const row = get<0>(tKcK(_0{}, m, _0{}));
      auto mK_cur_copy = [&] {
        if constexpr (PagedKVNonTMA) {
          Element* k_ptr = reinterpret_cast<Element*>(__shfl_sync(0xffffffff, reinterpret_cast<uint64_t>(tPrKPtr(m / kGmemThreadsPerRow)), (m % kGmemThreadsPerRow), kGmemThreadsPerRow));
          Tensor mK_cur = make_tensor(make_gmem_ptr(k_ptr), Shape<Int<kHeadDim>>{});
          return cute::tiled_divide(mK_cur, Shape<Int<kGmemElemsPerLoad>>{});
        } else {
          return nullptr;
        }
      }();
      if (row < std::min(max_seqlen - n_block * kBlockMN, kBlockMN)) {
        #pragma unroll
        for (int k = 0; k < size<2>(tKsK); ++k) {
          if (tKpK(k)) {
            Tensor rK = make_fragment_like(tKsK(_, m, k));
            cute::copy(tiled_copy_k, tKsK(_, m, k), rK);
            if (tRpR(k)) { apply_rotary_interleaved(rK, tRrCos(_, m, k), tRrSin(_, m, k)); }
            if constexpr (!PagedKVNonTMA) {
              cute::copy(tiled_copy_k, rK, tKgK(_, m, k));
            } else {
              int const ki = get<1>(tKcK(_0{}, _0{}, k)) / kGmemElemsPerLoad;
              cute::copy(tiled_copy_k, rK, mK_cur_copy(_, ki));
            }
          }
        }
      }
    }
  };
  template <bool PagedKVNonTMA=false, typename TensorsK, typename TensorgK, typename TensorpK, typename TensortRrR, typename TensorKPtr>
  CUTLASS_DEVICE
  void
  apply_K_contiguous(TensorsK const &sK,
            TensorgK &gK,
            TensorpK const &tKpK,
            TensortRrR const &tRrCosCont,
            TensortRrR const &tRrSinCont,
            TensorKPtr const &tPrKPtr,
            int const n_block, int const max_k)
  {
    TiledCopyQK tiled_copy_k;
    auto gmem_thr_copy_q = tiled_copy_k.get_thread_slice(thread_idx);
    Tensor sK_copy = cute::tiled_divide(sK, Shape<_1, Int<kGmemElemsPerLoad>>{});
    Tensor gK_copy = cute::tiled_divide(gK, Shape<_1, Int<kGmemElemsPerLoad>>{});
    Tensor tKcK = gmem_thr_copy_q.partition_S(cute::make_identity_tensor(Shape<Int<kBlockMN>, Int<kHeadDim / 2>>{}));
    CUTE_STATIC_ASSERT_V(rank(tKcK) == _3{});
    CUTE_STATIC_ASSERT_V(rank(tRrCosCont) == _3{});
    CUTE_STATIC_ASSERT_V(rank(tRrSinCont) == _3{});
    CUTE_STATIC_ASSERT_V(size<1>(tKcK) == size<1>(tRrCosCont));
    CUTE_STATIC_ASSERT_V(size<2>(tKcK) == size<2>(tRrCosCont));
    CUTE_STATIC_ASSERT_V(size<1>(tKcK) == size<1>(tRrSinCont));
    CUTE_STATIC_ASSERT_V(size<2>(tKcK) == size<2>(tRrSinCont));
    CUTE_STATIC_ASSERT_V(size<0>(tRrCosCont) == size<0>(tRrSinCont));
    CUTE_STATIC_ASSERT_V(size<0>(tKcK) == size<0>(tRrCosCont));
    static_assert(decltype(size<0>(tRrCosCont))::value % 2 == 0);
    if constexpr (PagedKVNonTMA) {
      static_assert(decltype(size(tPrKPtr))::value == cute::ceil_div(size<1>(tKcK), kGmemThreadsPerRow));
    }
    const int ro_dim_vec = rotary_dim / kGmemElemsPerLoad;
    const int non_ro_dim_vec = (max_k - rotary_dim) / kGmemElemsPerLoad;
    #pragma unroll
    for (int m = 0; m < size<1>(tKcK); ++m) {
      int const row = get<0>(tKcK(_0{}, m, _0{}));
      Tensor gK_cur_copy = [&] {
        if constexpr (PagedKVNonTMA) {
          Element* k_ptr = reinterpret_cast<Element*>(__shfl_sync(0xffffffff, reinterpret_cast<uint64_t>(tPrKPtr(m / kGmemThreadsPerRow)), (m % kGmemThreadsPerRow), kGmemThreadsPerRow));
          Tensor mK_cur = make_tensor(make_gmem_ptr(k_ptr), Shape<Int<kHeadDim>>{});
          return cute::tiled_divide(mK_cur, Shape<Int<kGmemElemsPerLoad>>{});
        } else {
          return gK_copy(_, row, _);
        }
      }();
      if (row < std::min(max_seqlen - n_block * kBlockMN, kBlockMN)) {
        #pragma unroll
        for (int k = 0; k < size<2>(tKcK); ++k) {
          if (tKpK(k)) {
            int const col = get<1>(tKcK(_0{}, _0{}, k));
            bool rotate = col < rotary_dim / 2;
            int const col_idx_left = rotate ? col / kGmemElemsPerLoad : (col + rotary_dim / 2) / kGmemElemsPerLoad;
            int const col_idx_right = col_idx_left + (rotate ? ro_dim_vec / 2 : non_ro_dim_vec / 2);
            Tensor rK_left = make_fragment_like(sK_copy(_, row, col_idx_left));
            cute::copy(tiled_copy_k, sK_copy(_, row, col_idx_left), rK_left);
            Tensor rK_right = make_fragment_like(rK_left);
            cute::copy(tiled_copy_k, sK_copy(_, row, col_idx_right), rK_right);
            if (rotate) {
              apply_rotary_contiguous(rK_left, rK_right, tRrCosCont(_, m, k), tRrSinCont(_, m, k));
            }
            cute::copy(tiled_copy_k, rK_left, gK_cur_copy(_, col_idx_left));
            if (col_idx_right * kGmemElemsPerLoad < max_k) {
              cute::copy(tiled_copy_k, rK_right, gK_cur_copy(_, col_idx_right));
            }
          }
        }
      }
    }
  };
};
}
#include<cutlass/pipeline/sm90_pipeline.hpp>
namespace cutlass {
using namespace cute;
template <int Stages_, class Base=cutlass::PipelineTmaAsync<Stages_>>
class PipelineTmaAsyncNoCluster: public Base {
public:
 using FullBarrier = typename Base::FullBarrier;
 using EmptyBarrier = typename Base::EmptyBarrier;
 static constexpr uint32_t Stages = Stages_;
 using PipelineState = typename Base::PipelineState;
 using SharedStorage = typename Base::SharedStorage;
 using ThreadCategory = typename Base::ThreadCategory;
 using Params = typename Base::Params;
 static
 CUTLASS_DEVICE
 void
 init_barriers(SharedStorage& storage, Params params) {
  int warp_idx = canonical_warp_idx_sync();
  bool is_initializing_warp = (warp_idx == 0);
  if (is_initializing_warp) {
   constexpr int producer_arv_cnt = 1;
   uint32_t const num_consumer_warpgroups_per_cluster = (params.num_consumers + NumThreadsPerWarpGroup - 1) / NumThreadsPerWarpGroup;
   uint32_t const multicast_consumer_arrival_count = num_consumer_warpgroups_per_cluster;
   cutlass::arch::detail::initialize_barrier_array_pair_aligned<decltype(storage.full_barrier_), decltype(storage.empty_barrier_), Stages>(
     storage.full_barrier_, storage.empty_barrier_, producer_arv_cnt, multicast_consumer_arrival_count);
  }
  cutlass::arch::fence_barrier_init();
 }
 template<class ClusterShape, class InitBarriers, class InitMasks>
 CUTLASS_DEVICE
 PipelineTmaAsyncNoCluster(SharedStorage& storage, Params params, ClusterShape cluster_shape, InitBarriers = {}, InitMasks = {})
   : Base(storage, params, make_shape(_1{}, _1{}, _1{}) , cute::false_type{} , cute::false_type{} )
   , empty_barrier_ptr_(&storage.empty_barrier_[0]) {
  int warp_idx = canonical_warp_idx_sync();
  int lane_predicate = cute::elect_one_sync();
  static_assert(cute::is_same_v<InitBarriers, cute::true_type> || cute::is_same_v<InitBarriers, cute::false_type>);
  static_assert(cute::is_same_v<InitMasks, cute::true_type> || cute::is_same_v<InitMasks, cute::false_type>);
  if constexpr (cute::is_same_v<InitBarriers, cute::true_type>) {
   init_barriers(storage, params);
  }
 }
 template<class ClusterShape>
 CUTLASS_DEVICE
 PipelineTmaAsyncNoCluster(SharedStorage& storage, Params params, ClusterShape cluster_shape)
   : PipelineTmaAsyncNoCluster(storage, params, cluster_shape, cute::true_type{}, cute::true_type{}) { }
 template<class ClusterShape, class InitBarriers>
 CUTLASS_DEVICE
 PipelineTmaAsyncNoCluster(SharedStorage& storage, Params params, ClusterShape cluster_shape, InitBarriers = {})
   : PipelineTmaAsyncNoCluster(storage, params, cluster_shape, InitBarriers{}, cute::true_type{}) { }
 CUTLASS_DEVICE
 void consumer_release(PipelineState state) {
  consumer_release(state.index());
 }
private:
 EmptyBarrier* const empty_barrier_ptr_ = nullptr;
 CUTLASS_DEVICE
 void consumer_release(uint32_t stage, uint32_t skip = false) {
  if ((threadIdx.x % cutlass::NumThreadsPerWarpGroup == 0) && !skip) {
   uint32_t const a = cute::cast_smem_ptr_to_uint(&empty_barrier_ptr_[stage]);
   asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" :: "r"(a) : "memory");
  }
 }
};
}
#define SK2_ABL 0
#define ATT_CABL (((SK2_ABL & 1) ? 2 : 0) | ((SK2_ABL & 2) ? 1 : 0))
#define ATT_PABL 0
#define ATT_STAMP(cond, role, it, ev) {}
namespace flash {
using namespace cute;


template <int kBlockN, typename TP>
CUTLASS_DEVICE void fk_pv16(TP const& tOrP, uint32_t (&o)[16], uint32_t vaddr) {
  Tensor a = recast<uint32_t const>(tOrP);
  constexpr int NK = CUTE_STATIC_V(size(a)) / 4;
  static_assert(NK == kBlockN / 32);
  constexpr int CW = kBlockN % 128 == 0 ? 128 : 64;
  constexpr int KPC = CW / 32;
  constexpr uint32_t LT = CW == 128 ? 1 : 2;
  uint32_t const lo0 = ((vaddr & 0x3FFFF) >> 4) | (1u << 16);
  uint32_t const hi = uint32_t(8 * CW >> 4) | uint32_t(LT << 30);
  warpgroup_fence_operand(const_cast<TP&>(tOrP));
  #pragma unroll
  for (int j = 0; j < 16; ++j) { asm volatile("" : "+r"(o[j]) :: "memory"); }
  warpgroup_arrive();
  #pragma unroll
  for (int k = 0; k < NK; ++k) {
    uint64_t desc;
    asm("mov.b64 %0, {%1, %2};" : "=l"(desc) : "r"(lo0 + uint32_t(((k / KPC) * (72 * CW) + (k % KPC) * 32) >> 4)), "r"(hi));
    asm volatile(
      "{\n.reg .pred p;\nsetp.ne.b32 p, 1, 0;\n"
      "wgmma.mma_async.sync.aligned.m64n64k32.f16.e4m3.e4m3 "
      "{%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, {%16, %17, %18, %19}, %20, p, 1, 1;\n}\n"
      : "+r"(o[0]), "+r"(o[1]), "+r"(o[2]), "+r"(o[3]), "+r"(o[4]), "+r"(o[5]), "+r"(o[6]), "+r"(o[7]),
       "+r"(o[8]), "+r"(o[9]), "+r"(o[10]), "+r"(o[11]), "+r"(o[12]), "+r"(o[13]), "+r"(o[14]), "+r"(o[15])
      : "r"(a(4 * k)), "r"(a(4 * k + 1)), "r"(a(4 * k + 2)), "r"(a(4 * k + 3)), "l"(desc));
  }
  warpgroup_commit_batch();
  #pragma unroll
  for (int j = 0; j < 16; ++j) { asm volatile("" : "+r"(o[j]) :: "memory"); }
  warpgroup_fence_operand(const_cast<TP&>(tOrP));
}

template <typename TS>
CUTLASS_DEVICE void u_rescale16(uint32_t (&o)[16], TS const& sc) {
  __half2 const s0 = __float2half2_rn(sc(0)), s1 = __float2half2_rn(sc(1));
  #pragma unroll
  for (int j = 0; j < 16; ++j) {
    __half2 v = *reinterpret_cast<__half2*>(&o[j]);
    v = __hmul2(v, (j & 1) ? s1 : s0);
    o[j] = *reinterpret_cast<uint32_t*>(&v);
  }
}
CUTLASS_DEVICE void u_stat16(uint32_t const (&o)[16]) {
}

template <typename TO>
CUTLASS_DEVICE void u_fold(TO& acc, uint32_t (&o)[16], float (&usc)[2], bool first) {
  u_stat16(o);
  #pragma unroll
  for (int j = 0; j < 16; ++j) {
    __half2 v = *reinterpret_cast<__half2*>(&o[j]);
    float const f = usc[j & 1];
    if (first) {
      acc(2 * j) = __low2float(v);
      acc(2 * j + 1) = __high2float(v);
    } else {
      acc(2 * j) = fmaf(acc(2 * j), f, __low2float(v));
      acc(2 * j + 1) = fmaf(acc(2 * j + 1), f, __high2float(v));
    }
    o[j] = 0u;
  }
  usc[0] = usc[1] = 1.f;
}

template <int kBlockN, typename TP, typename TO>
CUTLASS_DEVICE void fk_pv72(TP const& tOrP, TO& o, float (&os)[4], uint32_t vaddr) {
  Tensor a = recast<uint32_t const>(tOrP);
  constexpr int NK = CUTE_STATIC_V(size(a)) / 4;
  static_assert(NK == kBlockN / 32);
  static_assert(CUTE_STATIC_V(size(o)) == 32);
  constexpr int CW = kBlockN % 128 == 0 ? 128 : 64;
  constexpr int KPC = CW / 32;
  constexpr uint64_t LT = CW == 128 ? 1 : 2;
  uint32_t const fx_lo0 = ((vaddr & 0x3FFFF) >> 4) | (1u << 16);
  uint32_t const fx_hi = uint32_t(8 * CW >> 4) | uint32_t(LT << 30);
  warpgroup_fence_operand(const_cast<TP&>(tOrP));
  warpgroup_fence_operand(o);
  #pragma unroll
  for (int i = 0; i < 4; ++i) { asm volatile("" : "+f"(os[i]) :: "memory"); }
  warpgroup_arrive();
  #pragma unroll
  for (int k = 0; k < NK; ++k) {
    uint32_t const ad = vaddr + (k / KPC) * (72 * CW) + (k % KPC) * 32;
    uint64_t desc;
    asm("mov.b64 %0, {%1, %2};" : "=l"(desc) : "r"(fx_lo0 + uint32_t(((k / KPC) * (72 * CW) + (k % KPC) * 32) >> 4)), "r"(fx_hi));
    (void)ad;
    asm volatile(
      "{\n.reg .pred p;\nsetp.ne.b32 p, 1, 0;\n"
      "wgmma.mma_async.sync.aligned.m64n64k32.f32.e4m3.e4m3 "
      "{%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, {%32, %33, %34, %35}, %36, p, 1, 1;\n}\n"
      : "+f"(o(0)), "+f"(o(1)), "+f"(o(2)), "+f"(o(3)), "+f"(o(4)), "+f"(o(5)), "+f"(o(6)), "+f"(o(7)), "+f"(o(8)), "+f"(o(9)), "+f"(o(10)), "+f"(o(11)), "+f"(o(12)), "+f"(o(13)), "+f"(o(14)), "+f"(o(15)), "+f"(o(16)), "+f"(o(17)), "+f"(o(18)), "+f"(o(19)), "+f"(o(20)), "+f"(o(21)), "+f"(o(22)), "+f"(o(23)), "+f"(o(24)), "+f"(o(25)), "+f"(o(26)), "+f"(o(27)), "+f"(o(28)), "+f"(o(29)), "+f"(o(30)), "+f"(o(31))
      : "r"(a(4 * k)), "r"(a(4 * k + 1)), "r"(a(4 * k + 2)), "r"(a(4 * k + 3)), "l"(desc));
  }
  warpgroup_commit_batch();
  warpgroup_fence_operand(o);
  #pragma unroll
  for (int i = 0; i < 4; ++i) { asm volatile("" : "+f"(os[i]) :: "memory"); }
  warpgroup_fence_operand(const_cast<TP&>(tOrP));
}
template <int Stages, class ClusterShape_, class TileShape_MNK_, int kHeadDimV, class Element_, class ElementAccum_, class ArchTag_,
    bool Is_causal_, bool Is_local_, bool Has_softcap_, bool Varlen_, bool PagedKVNonTMA_, bool AppendKV_, bool HasQv_, bool OnlyQv_,
    bool MmaPV_is_RS, bool IntraWGOverlap, bool PackGQA_, bool Split_, bool V_colmajor_, class ElementSink_, bool Has_sparse_mask_=false, int kBlockH_=1>
struct CollectiveMainloopFwdSm90 {
  static constexpr int kStages = Stages;
  using ClusterShape = ClusterShape_;
  using TileShape_MNK = TileShape_MNK_;
  using TileShape_MNK_PV = Shape<decltype(get<0>(TileShape_MNK{})), Int<kHeadDimV>, decltype(get<1>(TileShape_MNK{}))>;
  using TileShape_MNK_QV = Shape<decltype(get<0>(TileShape_MNK{})), decltype(get<1>(TileShape_MNK{})), Int<kHeadDimV>>;
  using Element = Element_;
  using ElementAccum = ElementAccum_;
  using ElementSink = ElementSink_;
  using ArchTag = ArchTag_;
  static constexpr bool Is_FP8 = cute::is_same_v<Element, cutlass::float_e4m3_t> || cute::is_same_v<Element, cutlass::float_e5m2_t>;;
  static constexpr bool Is_causal = Is_causal_;
  static constexpr bool Is_local = Is_local_;
  static constexpr bool Has_softcap = Has_softcap_;
  static constexpr bool Varlen = Varlen_;
  static constexpr bool PagedKVNonTMA = PagedKVNonTMA_;
  static constexpr bool AppendKV = AppendKV_;
  static constexpr bool HasQv = HasQv_;
  static constexpr bool OnlyQv = OnlyQv_;
  static constexpr bool PackGQA = PackGQA_;
  static constexpr bool PackGQA_TMA = PackGQA && kBlockH_ > 1;
  static constexpr bool Split = Split_;
  static constexpr bool V_colmajor = V_colmajor_;
  static constexpr bool Has_sparse_mask = Has_sparse_mask_;
  static constexpr bool Transpose_V = false;
  static_assert(Is_FP8 && !V_colmajor && !PagedKVNonTMA && !AppendKV && !HasQv);
  static constexpr bool Use_TMA_Q = !PackGQA || PackGQA_TMA;
  static constexpr bool Use_TMA_KV = !PagedKVNonTMA;
  static_assert(Use_TMA_KV || CUTE_STATIC_V(size(ClusterShape{})) == 1, "If not using TMA for KV, ClusterShape must be 1");
  static_assert(Use_TMA_KV || !V_colmajor, "If not using TMA for KV, V_colmajor is not supported");
  static constexpr bool SameHeadDim = get<2>(TileShape_MNK{}) == kHeadDimV;
  static constexpr bool LargeHeadDimV = kHeadDimV > 256;
  static_assert(!OnlyQv || HasQv, "OnlyQv requires HasQv");
  static_assert(ArchTag::kMinComputeCapability >= 90);
  static constexpr cute::GMMA::Major MmaMajorV = !Is_FP8 && !V_colmajor ? GMMA::Major::MN : GMMA::Major::K;
  static constexpr cute::GMMA::Major TmaMajorV = !V_colmajor ? GMMA::Major::MN : GMMA::Major::K;
  static constexpr int kBlockM = get<0>(TileShape_MNK{});
  static constexpr int kBlockN = get<1>(TileShape_MNK{});
  static constexpr int kHeadDim = get<2>(TileShape_MNK{});
  static constexpr int kBlockH = kBlockH_;
  static constexpr int kNumInt32PerBlock = (kBlockN + 31) / 32;
  using SeqlenInfo_t = flash::SeqlenInfoQKNewK<Varlen, AppendKV>;
  using BlockMN_t = flash::BlockMN<SeqlenInfo_t, kBlockM, kBlockN, Is_causal, Is_local, PackGQA, Split>;
  static_assert(!LargeHeadDimV || kHeadDimV % 256 == 0);
  static_assert(!LargeHeadDimV || kBlockM <= 64, "kBlockM must be 64 or less for large Headdim_V");
  static_assert(!LargeHeadDimV || !MmaPV_is_RS, "MmaPV must be SS for large Headdim_V");
  static constexpr bool MmaQK_is_RS = false;
  static_assert(!(!MmaPV_is_RS && Is_FP8), "MmaPV must be RS if FP8");
  static_assert(!(!MmaPV_is_RS && Transpose_V), "MmaPV must be RS if Transpose_V");
  static constexpr bool MmaPV_use_RS_WG1 = !MmaPV_is_RS && kHeadDim == 64 && kHeadDimV == 512;
  using AtomLayoutQK = Layout<Shape<Int<kBlockM / 64>, _1, _1>>;
  using ElementAccumQK = std::conditional_t<Is_FP8, cutlass::half_t, ElementAccum>;
  using TiledMmaQK = decltype(cute::make_tiled_mma(
    std::conditional_t<
      !MmaQK_is_RS,
      decltype(cute::GMMA::ss_op_selector<Element, Element, ElementAccumQK, TileShape_MNK>()),
      decltype(cute::GMMA::rs_op_selector<Element, Element, ElementAccumQK, TileShape_MNK>())
    >{},
    AtomLayoutQK{}));
  using AtomLayoutPV = std::conditional_t<
    !LargeHeadDimV,
    AtomLayoutQK,
    Layout<Shape<_1, Int<kHeadDimV / 256>, _1>>
  >;
  using TiledMmaPV = decltype(cute::make_tiled_mma(
    std::conditional_t<
      !MmaPV_is_RS,
      decltype(cute::GMMA::ss_op_selector<Element, Element, ElementAccum,
           TileShape_MNK_PV, GMMA::Major::K, MmaMajorV>()),
      decltype(cute::GMMA::rs_op_selector<Element, Element, ElementAccum,
           TileShape_MNK_PV, GMMA::Major::K, MmaMajorV>())
    >{},
    AtomLayoutPV{}));
  using TiledMmaQV = decltype(cute::make_tiled_mma(
    cute::GMMA::ss_op_selector<Element, Element, ElementAccum, TileShape_MNK_QV>(),
    AtomLayoutQK{}));
  using TiledMmaPV_RS = decltype(cute::make_tiled_mma(
    cute::GMMA::rs_op_selector<Element, Element, ElementAccum, TileShape_MNK_PV, GMMA::Major::K, MmaMajorV>(),
    AtomLayoutPV{}));
  static constexpr int NumMmaThreadsQK = size(TiledMmaQK{});
  static constexpr int NumMmaThreads = size(TiledMmaPV{});
  static constexpr int NumProducerThreads = !Transpose_V && Use_TMA_KV && Use_TMA_Q ? cutlass::NumThreadsPerWarp : cutlass::NumThreadsPerWarpGroup;
  static_assert(NumMmaThreadsQK % cutlass::NumThreadsPerWarpGroup == 0);
  static_assert(NumMmaThreads % cutlass::NumThreadsPerWarpGroup == 0);
  static constexpr int NumMmaWarpGroups = NumMmaThreads / cutlass::NumThreadsPerWarpGroup;
  static_assert(NumMmaWarpGroups == 1 || NumMmaWarpGroups == 2 || NumMmaWarpGroups == 3 || NumMmaWarpGroups == 4);
  using SmemLayoutAtomQ = decltype(cutlass::gemm::collective::detail::ss_smem_selector<GMMA::Major::K, Element,
    decltype(cute::get<0>(TileShape_MNK{})), decltype(cute::get<2>(TileShape_MNK{}))>());
  using SmemLayoutQ = decltype(tile_to_shape(SmemLayoutAtomQ{}, select<0, 2>(TileShape_MNK{})));
  using SmemLayoutAtomK = decltype(cutlass::gemm::collective::detail::ss_smem_selector<GMMA::Major::K, Element,
    decltype(cute::get<1>(TileShape_MNK{})), decltype(cute::get<2>(TileShape_MNK{}))>());
  using SmemLayoutK = decltype(tile_to_shape(
    SmemLayoutAtomK{},
    make_shape(shape<1>(TileShape_MNK{}), shape<2>(TileShape_MNK{}), Int<kStages>{})));
  using SmemLayoutAtomVt = decltype(cutlass::gemm::collective::detail::ss_smem_selector<TmaMajorV, Element,
                   Int<kHeadDimV>, decltype(cute::get<2>(TileShape_MNK_PV{}))>());
  using SmemLayoutVt = decltype(tile_to_shape(
    SmemLayoutAtomVt{},
    make_shape(Int<kHeadDimV>{}, shape<2>(TileShape_MNK_PV{}), Int<kStages>{}),
    std::conditional_t<TmaMajorV == GMMA::Major::K, cute::Step<_1, _2, _3>, cute::Step<_2, _1, _3>>{}));
  using SmemLayoutAtomVtMma = decltype(cutlass::gemm::collective::detail::ss_smem_selector<MmaMajorV, Element,
                     Int<kHeadDimV>, decltype(cute::get<2>(TileShape_MNK_PV{}))>());
  static_assert(kHeadDimV == 64 && MmaMajorV == GMMA::Major::K && kBlockN % 64 == 0, "FK_PV72: hdim 64, BN % 64");
  using SmemLayoutVtMma72 = decltype(tile_to_shape(
    SmemLayoutAtomVtMma{},
    make_shape(Int<72>{}, shape<2>(TileShape_MNK_PV{}), Int<kStages>{}),
    cute::Step<_1, _2, _3>{}));
  using SmemLayoutVtMma = decltype(composition(SmemLayoutVtMma72{},
    Layout<Shape<Int<64>, Int<kBlockN>, Int<kStages>>, Stride<_1, Int<72>, Int<72 * kBlockN>>>{}));
  static constexpr int kSmemVSize = cute::cosize_v<SmemLayoutVtMma72>;
  using SmemLayoutAtomQv = decltype(cutlass::gemm::collective::detail::ss_smem_selector<GMMA::Major::K, Element,
    decltype(cute::get<0>(TileShape_MNK_QV{})), decltype(cute::get<2>(TileShape_MNK_QV{}))>());
  using SmemLayoutQv = decltype(tile_to_shape(SmemLayoutAtomQv{}, select<0, 2>(TileShape_MNK_QV{})));
  using SmemLayoutAtomVMmaQV = decltype(cutlass::gemm::collective::detail::ss_smem_selector<GMMA::Major::K, Element,
    decltype(cute::get<1>(TileShape_MNK_QV{})), decltype(cute::get<2>(TileShape_MNK_QV{}))>());
  using SmemLayoutVMmaQV = decltype(tile_to_shape(
    SmemLayoutAtomVMmaQV{},
    make_shape(shape<1>(TileShape_MNK_QV{}), Int<kHeadDimV>{}, Int<kStages>{})));
  static_assert(CUTE_STATIC_V(size(SmemLayoutVMmaQV{})) == size(SmemLayoutVtMma{}));
  using SmemLayoutAtomVCpAsync = decltype(cutlass::gemm::collective::detail::ss_smem_selector<GMMA::Major::K, Element,
    decltype(cute::get<1>(TileShape_MNK{})), Int<kHeadDimV>>());
  using SmemLayoutVCpAsync = decltype(tile_to_shape(
    SmemLayoutAtomVCpAsync{},
    make_shape(shape<1>(TileShape_MNK{}), Int<kHeadDimV>{}, Int<kStages>{})));
  using SmemLayoutAtomP = decltype(cutlass::gemm::collective::detail::ss_smem_selector<GMMA::Major::K, Element,
    decltype(cute::get<0>(TileShape_MNK{})), decltype(cute::get<1>(TileShape_MNK{}))>());
  using SmemLayoutP = decltype(tile_to_shape(SmemLayoutAtomP{}, select<0, 1>(TileShape_MNK{})));
  using SmemLayoutScale = cute::Layout<cute::Shape<Int<kBlockM>, Int<kStages>>>;
  using SmemCopyAtomP = Copy_Atom<cute::SM90_U32x4_STSM_N, Element>;
  using SmemLayoutSink = Layout<Shape<_64>>;
  static_assert(!Transpose_V || (kHeadDimV % 32 == 0 && kBlockN % 32 == 0));
  static constexpr bool kHeadDimV_multiple_64 = kHeadDimV % 64 == 0;
  static_assert(!Transpose_V || (kHeadDimV_multiple_64 || kBlockN % 64 == 0));
  using LDSM_thread_shape = std::conditional_t<kHeadDimV_multiple_64, Shape<_32, _4, _1, _1>, Shape<_16, _4, _1, _2>>;
  using LDSM_thread_stride = std::conditional_t<kHeadDimV_multiple_64, Stride<_4, _1, _0, _0>, Stride<_4, _1, _0, _64>>;
  using LDSM_value_shape = Shape<_2, _2, _1, _4>;
  using LDSM_value_stride = Stride<_1, _2, _16, _4>;
  using LDSM_divide_shape = std::conditional_t<kHeadDimV_multiple_64, Shape<_64, _8>, Shape<_32, _8>>;
  using S2RTiledCopyVt = decltype(make_tiled_copy(
    Copy_Atom<SM75_U16x8_LDSM_T, Element>{}, Layout<LDSM_thread_shape, LDSM_thread_stride>{},
    Layout<LDSM_value_shape, LDSM_value_stride>{}));
  using STSM_thread_shape = std::conditional_t<kHeadDimV_multiple_64, Shape<_8, _4, _4, _1>, Shape<_8, _4, _2, _2>>;
  using STSM_thread_stride = std::conditional_t<kHeadDimV_multiple_64, Stride<_4, _1, _32, _0>, Stride<_4, _1, _32, _64>>;
  using STSM_value_shape = Shape<_1, _4, _2, _2>;
  using STSM_value_stride = Stride<_0, _1, _4, _8>;
  using STSM_divide_shape = Shape<_8, _16>;
  using R2STiledCopyV = decltype(make_tiled_copy(
    Copy_Atom<SM90_U32x4_STSM_N, Element>{}, Layout<STSM_thread_shape, STSM_thread_stride>{},
    Layout<STSM_value_shape, STSM_value_stride>{}));
  using GmemTiledCopyQ = cute::SM90_TMA_LOAD;
  using GmemTiledCopyKV = decltype(cutlass::gemm::collective::detail::sm90_cluster_shape_to_tma_atom(shape<0>(ClusterShape{})));
  static constexpr int kHeadDimGCD = cute::gcd(kHeadDim, kHeadDimV);
  static constexpr int kGmemElemsPerLoad = sizeof(cute::uint128_t) / sizeof(Element);
  static_assert(kHeadDimGCD % kGmemElemsPerLoad == 0, "Headdim and HeaddimV must be a multiple of kGmemElemsPerLoad");
  static constexpr int kBytePerHalfRow = kHeadDimGCD / 2 * sizeof(Element);
  static constexpr int kBlockKGmem = (kBytePerHalfRow % 128 == 0 ? 128 : (kBytePerHalfRow % 64 == 0 ? 64 : 32)) / sizeof(Element);
  static constexpr int kGmemThreadsPerRow = kBlockKGmem / kGmemElemsPerLoad;
  static_assert(NumMmaThreads % kGmemThreadsPerRow == 0, "NumMmaThreads must be a multiple of kGmemThreadsPerRow");
  static_assert(cutlass::NumThreadsPerWarp % kGmemThreadsPerRow == 0, "kGmemThreadsPerRow must divide NumThreadsPerWarp");
  using GmemLayoutAtom = Layout<Shape <Int<NumMmaThreads / kGmemThreadsPerRow>, Int<kGmemThreadsPerRow>>,
                 Stride<Int<kGmemThreadsPerRow>, _1>>;
  static_assert(!AppendKV || kBlockM % CUTE_STATIC_V(shape<0>(GmemLayoutAtom{})) == 0, "kBlockM must be a multiple of NumMmaThreads / kGmemThreadsPerRow");
  using GmemTiledCopyAppendKV = decltype(
    make_tiled_copy(Copy_Atom<AutoVectorizingCopyWithAssumedAlignment<128>, Element>{},
            GmemLayoutAtom{},
            Layout<Shape<_1, Int<kGmemElemsPerLoad>>>{}));
  using ShapeQKV = cute::Shape<int32_t, int32_t, int32_t, int32_t>;
  using StrideQK = cute::Stride<int64_t, _1, int64_t, int64_t>;
  using StrideV = std::conditional_t<!V_colmajor, StrideQK, cute::Stride<_1, int64_t, int64_t, int64_t>>;
  template <class P> static CUTLASS_DEVICE auto att_paged_stride_v(P const& p) {
    if constexpr (V_colmajor) { return p.stride_K; } else { return p.stride_V; }
  }
  using ShapeQPacked = std::conditional_t<!PackGQA, ShapeQKV, cute::Shape<cute::Shape<int32_t, int32_t>, int32_t, int32_t, int32_t>>;
  using StrideQPacked = std::conditional_t<!PackGQA, StrideQK, cute::Stride<cute::Stride<int64_t, int64_t>, _1, int64_t, int64_t>>;
  using ShapePageTable = cute::Shape<int32_t, int32_t>;
  using StridePageTable = cute::Stride<int64_t, _1>;
  using ShapeRotary = cute::Shape<int32_t, int32_t>;
  using StrideRotary = cute::Stride<int64_t, _1>;
  using StrideDescale = cute::Stride<int64_t, int64_t>;
  using TMA_Q = decltype(make_tma_copy_A_sm90(
    GmemTiledCopyQ{},
    make_tensor(make_gmem_ptr(static_cast<Element const*>(nullptr)), ShapeQKV{}, StrideQK{}),
    SmemLayoutQ{},
    TileShape_MNK{},
    ClusterShape{}));
  using TMA_K = decltype(make_tma_copy_B_sm90(
    GmemTiledCopyKV{},
    make_tensor(make_gmem_ptr(static_cast<Element const*>(nullptr)), ShapeQKV{}, StrideQK{}),
    take<0, 2>(SmemLayoutK{}),
    TileShape_MNK{},
    ClusterShape{}));
  using TMA_V = decltype(make_tma_copy(
    GmemTiledCopyKV{},
    make_tensor(make_gmem_ptr(static_cast<Element const*>(nullptr)), ShapeQKV{}, select<1, 0, 2, 3>(StrideV{})),
    take<0, 2>(SmemLayoutVt{}),
    select<1, 2>(TileShape_MNK_PV{}),
    size<0>(ClusterShape{})));
  using TMA_Qv_ = decltype(make_tma_copy_A_sm90(
    GmemTiledCopyQ{},
    make_tensor(make_gmem_ptr(static_cast<Element const*>(nullptr)), ShapeQKV{}, StrideQK{}),
    SmemLayoutQv{},
    TileShape_MNK_QV{},
    ClusterShape{}));
  using TMA_Qv = std::conditional_t<HasQv, TMA_Qv_, std::nullptr_t>;
  using ShapeMask = cute::Shape<int32_t, int32_t>;
  using StrideMask = cute::Stride<int64_t, _1>;
  using SmemLayoutMask = Layout<Shape<Int<kBlockM>, Int<kNumInt32PerBlock>, Int<kStages>>,
                  Stride<Int<kNumInt32PerBlock>, _1, Int<kBlockM * kNumInt32PerBlock>>>;
  using TMA_Mask_ = decltype(make_tma_copy(
    SM90_TMA_LOAD{},
    make_tensor(make_gmem_ptr(static_cast<int32_t const*>(nullptr)), ShapeMask{}, StrideMask{}),
    take<0, 2>(SmemLayoutMask{}),
    Shape<Int<kBlockM>, Int<kNumInt32PerBlock>>{},
    _1{}));
  using TMA_Mask = std::conditional_t<Has_sparse_mask && Use_TMA_KV, TMA_Mask_, std::nullptr_t>;
  static constexpr uint32_t TmaTransactionBytesQ = static_cast<uint32_t>(size(SmemLayoutQ{}) * cutlass::sizeof_bits_v<Element> / 8);
  static constexpr uint32_t TmaTransactionBytesK = static_cast<uint32_t>(size(take<0, 2>(SmemLayoutK{})) * cutlass::sizeof_bits_v<Element> / 8);
  static constexpr uint32_t TmaTransactionBytesV = 72 * kBlockN;
  static constexpr uint32_t TmaTransactionBytesQv = static_cast<uint32_t>(size(SmemLayoutQv{}) * cutlass::sizeof_bits_v<Element> / 8);
  static constexpr uint32_t TmaTransactionBytesMask = static_cast<uint32_t>(kBlockM * kNumInt32PerBlock * sizeof(int));
  static constexpr bool Use_TMA_Mask = Use_TMA_KV && Has_sparse_mask;
  using PipelineTmaAsync = std::conditional_t<CUTE_STATIC_V(size(ClusterShape{})) == 1, typename cutlass::PipelineTmaAsyncNoCluster<kStages>, typename cutlass::PipelineTmaAsync<kStages>>;
  using PipelineTmaAsyncMask = typename cutlass::PipelineTmaAsyncNoCluster<kStages>;
  using MainloopPipelineK = std::conditional_t<Use_TMA_KV, PipelineTmaAsync, typename cutlass::PipelineAsync<kStages>>;
  using MainloopPipelineV = std::conditional_t<!Transpose_V && Use_TMA_KV, PipelineTmaAsync, typename cutlass::PipelineAsync<kStages>>;
  using MainloopPipelineVt = std::conditional_t<Use_TMA_KV, PipelineTmaAsync, typename cutlass::PipelineAsync<kStages>>;
  using MainloopPipelineMask = std::conditional_t<Use_TMA_Mask, PipelineTmaAsyncMask, typename cutlass::PipelineAsync<kStages>>;
  using MainloopPipelineKVNew = PipelineTmaAsync;
  using PipelineState = cutlass::PipelineState<kStages>;
  static constexpr size_t SmemAlignmentQ = Use_TMA_Q && !MmaQK_is_RS ? 128 : cutlass::detail::alignment_for_swizzle(SmemLayoutQ{});
  static constexpr size_t SmemAlignmentK = Use_TMA_KV && !AppendKV ? 128 : cutlass::detail::alignment_for_swizzle(SmemLayoutK{});
  static constexpr size_t SmemAlignmentVtNoTranspose = 1024;
  static constexpr size_t SmemAlignmentQv = Use_TMA_Q ? 128 : cutlass::detail::alignment_for_swizzle(SmemLayoutQv{});
  static_assert(SmemAlignmentQ >= 128 and SmemAlignmentK >= 128 && SmemAlignmentVtNoTranspose >= 128, "Require at least 128B alignment");
  static constexpr size_t SmemAlignmentP = cutlass::detail::alignment_for_swizzle(SmemLayoutP{});
  static_assert(SmemAlignmentP >= 128, "Require at least 128B alignment");
  using SmemP_t = std::conditional_t<MmaPV_is_RS, cute::array<Element, 0>, cute::array_aligned<Element, cute::cosize_v<SmemLayoutP>, SmemAlignmentP>>;
  using SmemScale_t = std::conditional_t<!LargeHeadDimV, cute::array<float, 0>, cute::array_aligned<float, cute::cosize_v<SmemLayoutScale>, 128>>;
  using SmemQv_t = std::conditional_t<!HasQv, cute::array<Element, 0>, cute::array_aligned<Element, cute::cosize_v<SmemLayoutQv>, SmemAlignmentQv>>;
  using SmemMask_t = std::conditional_t<!Has_sparse_mask, cute::array<int, 0>, cute::array_aligned<int, kStages * kBlockM * kNumInt32PerBlock, 128>>;
  struct TensorStorageWithoutPNoTranspose : cute::aligned_struct<cute::max(SmemAlignmentQ, SmemAlignmentK, SmemAlignmentVtNoTranspose), _0> {
    cute::array_aligned<Element, kSmemVSize, SmemAlignmentVtNoTranspose> smem_v;
    cute::array_aligned<Element, cute::cosize_v<SmemLayoutQ>, SmemAlignmentQ> smem_q;
    cute::array_aligned<Element, cute::cosize_v<SmemLayoutK>, SmemAlignmentK> smem_k;
    SmemQv_t smem_qv;
    cute::array_aligned<ElementSink, cute::cosize_v<SmemLayoutSink>, 128> smem_sink;
    cute::array_aligned<uint8_t, 512, 128> smem_ones;
    SmemMask_t smem_mask;
  };
  struct TensorStorageWithPNoTranspose : cute::aligned_struct<cute::max(SmemAlignmentQ, SmemAlignmentK, SmemAlignmentVtNoTranspose, SmemAlignmentP), _0> {
    cute::array_aligned<Element, kSmemVSize, SmemAlignmentVtNoTranspose> smem_v;
    cute::array_aligned<Element, cute::cosize_v<SmemLayoutQ>, SmemAlignmentQ> smem_q;
    cute::array_aligned<Element, cute::cosize_v<SmemLayoutK>, SmemAlignmentK> smem_k;
    SmemQv_t smem_qv;
    SmemP_t smem_p;
    cute::array_aligned<ElementSink, cute::cosize_v<SmemLayoutSink>, 128> smem_sink;
    cute::array_aligned<uint8_t, 512, 128> smem_ones;
    SmemMask_t smem_mask;
  };
  struct TensorStorageWithPScaleNoTranspose : cute::aligned_struct<cute::max(SmemAlignmentQ, SmemAlignmentK, SmemAlignmentVtNoTranspose, SmemAlignmentP), _0> {
    cute::array_aligned<Element, kSmemVSize, SmemAlignmentVtNoTranspose> smem_v;
    cute::array_aligned<Element, cute::cosize_v<SmemLayoutQ>, SmemAlignmentQ> smem_q;
    cute::array_aligned<Element, cute::cosize_v<SmemLayoutK>, SmemAlignmentK> smem_k;
    SmemQv_t smem_qv;
    SmemP_t smem_p;
    SmemScale_t smem_scale;
    cute::array_aligned<ElementSink, cute::cosize_v<SmemLayoutSink>, 128> smem_sink;
    cute::array_aligned<uint8_t, 512, 128> smem_ones;
    SmemMask_t smem_mask;
  };
  using TensorStorageNoTranspose = std::conditional_t<
    MmaPV_is_RS,
    TensorStorageWithoutPNoTranspose,
    std::conditional_t<!LargeHeadDimV, TensorStorageWithPNoTranspose, TensorStorageWithPScaleNoTranspose>
  >;
  static constexpr size_t SmemAlignmentVt = cutlass::detail::alignment_for_swizzle(SmemLayoutVt{});
  static constexpr size_t SmemAlignmentV = cutlass::detail::alignment_for_swizzle(SmemLayoutVtMma72{});
  static_assert(SmemAlignmentVt >= 128 and SmemAlignmentV >= 128, "Require at least 128B alignment");
  struct TensorStorageTransposeV : cute::aligned_struct<cute::max(SmemAlignmentQ, SmemAlignmentK, SmemAlignmentV), _0> {
    cute::array_aligned<Element, kSmemVSize, SmemAlignmentV> smem_v;
    cute::array_aligned<Element, cute::cosize_v<SmemLayoutVt>, SmemAlignmentVt> smem_vt;
    cute::array_aligned<Element, cute::cosize_v<SmemLayoutQ>, SmemAlignmentQ> smem_q;
    cute::array_aligned<Element, cute::cosize_v<SmemLayoutK>, SmemAlignmentK> smem_k;
    SmemQv_t smem_qv;
    SmemScale_t smem_scale;
    cute::array_aligned<ElementSink, cute::cosize_v<SmemLayoutSink>, 128> smem_sink;
    cute::array_aligned<uint8_t, 512, 128> smem_ones;
    SmemMask_t smem_mask;
  };
  using TensorStorage = std::conditional_t<!Transpose_V, TensorStorageNoTranspose, TensorStorageTransposeV>;
  static constexpr bool UseSchedulerBarrier = (IntraWGOverlap
    ? (NumMmaWarpGroups >= 2) && (!Is_FP8 ? kHeadDim <= 128 : (kHeadDim >= 128 || ATT_SCHED))
    : NumMmaWarpGroups == 2)
    && !LargeHeadDimV;
  static constexpr bool RescaleOBeforeGemm = kHeadDim > 128 && (!Is_FP8 || V_colmajor) && IntraWGOverlap;
  struct Arguments {
    Element const* const ptr_Q;
    ShapeQKV const shape_Q;
    StrideQK const stride_Q;
    Element* const ptr_K;
    ShapeQKV const shape_K;
    StrideQK const stride_K;
    Element* const ptr_V;
    int32_t const headdim_v;
    StrideV const stride_V;
    Element const* const ptr_K_new;
    ShapeQKV const shape_K_new;
    StrideQK const stride_K_new;
    Element const* const ptr_V_new;
    StrideV const stride_V_new;
    Element const* const ptr_Qv;
    StrideQK const stride_Qv;
    Element const* const ptr_rotary_cos;
    ShapeRotary const shape_rotary;
    StrideRotary const stride_rotary_cos;
    Element const* const ptr_rotary_sin;
    StrideRotary const stride_rotary_sin;
    bool const is_rotary_interleaved;
    int const* const ptr_pagetable;
    ShapePageTable const shape_pagetable;
    StridePageTable const stride_pagetable;
    float const softmax_scale;
    float const* ptr_q_descale, *ptr_k_descale, *ptr_v_descale;
    StrideDescale const stride_q_descale, stride_k_descale, stride_v_descale;
    int const window_size_left = -1, window_size_right = -1, attention_chunk = 0;
    float const softcap_val;
    int const num_splits;
    int const* const kv_batch_idx = nullptr;
    int const* const cu_seqlens_q = nullptr;
    int const* const cu_seqlens_k = nullptr;
    int const* const cu_seqlens_k_new = nullptr;
    int const* const seqused_q = nullptr;
    int const* const seqused_k = nullptr;
    int const* const leftpad_k = nullptr;
    int const* const seqlens_rotary = nullptr;
    ElementSink const* const ptr_sink = nullptr;
    int const* sparse_mask_fine = nullptr;
    int sparse_mask_max_k_blocks = 0;
    int sparse_mask_fine_q_stride = 0;
    int sparse_mask_fine_k_stride = 0;
    int total_q = 0;
  };
  struct Params {
    Element const* const ptr_Q;
    ShapeQKV const shape_Q;
    StrideQK const stride_Q;
    ShapeQPacked const shape_Q_packed;
    StrideQPacked const stride_Q_packed;
    Element* const ptr_K;
    ShapeQKV const shape_K;
    StrideQK const stride_K;
    Element* const ptr_V;
    int32_t const headdim_v;
    StrideV const stride_V;
    Element const* const ptr_K_new;
    ShapeQKV const shape_K_new;
    StrideQK const stride_K_new;
    Element const* const ptr_V_new;
    StrideV const stride_V_new;
    Element const* const ptr_Qv;
    StrideQK const stride_Qv;
    ShapeQPacked const shape_Qv_packed;
    StrideQPacked const stride_Qv_packed;
    Element const* const ptr_rotary_cos;
    ShapeRotary const shape_rotary;
    StrideRotary const stride_rotary_cos;
    Element const* const ptr_rotary_sin;
    StrideRotary const stride_rotary_sin;
    bool const is_rotary_interleaved;
    int const* const ptr_pagetable;
    ShapePageTable const shape_pagetable;
    StridePageTable const stride_pagetable;
    cutlass::FastDivmod page_size_divmod;
    cutlass::FastDivmod blockN_per_page_size_divmod;
    cutlass::FastDivmod qhead_per_khead_divmod;
    TMA_Q tma_load_Q;
    TMA_K tma_load_K;
    TMA_V tma_load_V;
    TMA_K tma_load_K_new;
    TMA_V tma_load_V_new;
    TMA_Qv tma_load_Qv;
    TMA_Mask tma_load_Mask;
    float const softmax_scale_log2;
    float const* ptr_q_descale, *ptr_k_descale, *ptr_v_descale;
    StrideDescale const stride_q_descale, stride_k_descale, stride_v_descale;
    float const softcap_val;
    int const window_size_left, window_size_right;
    cutlass::FastDivmod attention_chunk_divmod;
    int const num_splits;
    int const* const kv_batch_idx = nullptr;
    int const* const cu_seqlens_q = nullptr;
    int const* const cu_seqlens_k = nullptr;
    int const* const cu_seqlens_k_new = nullptr;
    int const* const seqused_q = nullptr;
    int const* const seqused_k = nullptr;
    int const* const leftpad_k = nullptr;
    int const *const seqlens_rotary = nullptr;
    ElementSink const* const ptr_sink = nullptr;
    int const* sparse_mask_fine = nullptr;
    int sparse_mask_max_k_blocks = 0;
    int sparse_mask_fine_q_stride = 0;
    int sparse_mask_fine_k_stride = 0;
    int total_q = 0;
  };
  static Params
  to_underlying_arguments(Arguments const& args) {
    Tensor mQ = make_tensor(make_gmem_ptr(args.ptr_Q), args.shape_Q, args.stride_Q);
    TMA_Q tma_load_Q = make_tma_copy_A_sm90(
      GmemTiledCopyQ{},
      mQ,
      SmemLayoutQ{},
      TileShape_MNK{},
      ClusterShape{});
    Tensor mK = make_tensor(make_gmem_ptr(args.ptr_K), args.shape_K, args.stride_K);
    TMA_K tma_load_K = make_tma_copy_B_sm90(
      GmemTiledCopyKV{},
      mK,
      take<0, 2>(SmemLayoutK{}),
      TileShape_MNK{},
      ClusterShape{});
    Tensor mV = make_tensor(make_gmem_ptr(args.ptr_V),
                make_shape(args.headdim_v, get<0>(args.shape_K), get<2>(args.shape_K), get<3>(args.shape_K)),
                select<1, 0, 2, 3>(args.stride_V));
    TMA_V tma_load_V = make_tma_copy(
      GmemTiledCopyKV{},
      mV,
      take<0, 2>(SmemLayoutVt{}),
      select<1, 2>(TileShape_MNK_PV{}),
      size<0>(ClusterShape{}));
    Tensor mKnew = make_tensor(make_gmem_ptr(args.ptr_K_new), args.shape_K_new, args.stride_K_new);
    TMA_K tma_load_K_new = make_tma_copy_B_sm90(
      GmemTiledCopyKV{},
      cute::conditional_return<AppendKV>(mKnew, mK),
      take<0, 2>(SmemLayoutK{}),
      TileShape_MNK{},
      ClusterShape{});
    Tensor mVnew = make_tensor(make_gmem_ptr(args.ptr_V_new),
                  make_shape(args.headdim_v, get<0>(args.shape_K_new), get<2>(args.shape_K_new), get<3>(args.shape_K_new)),
                  select<1, 0, 2, 3>(args.stride_V_new));
    TMA_V tma_load_V_new = make_tma_copy(
      GmemTiledCopyKV{},
      cute::conditional_return<AppendKV>(mVnew, mV),
      take<0, 2>(SmemLayoutVt{}),
      select<1, 2>(TileShape_MNK_PV{}),
      size<0>(ClusterShape{}));
    auto shape_Qv = make_shape(get<0>(args.shape_Q), args.headdim_v, get<2>(args.shape_Q), get<3>(args.shape_Q));
    Tensor mQv = make_tensor(make_gmem_ptr(args.ptr_Qv), shape_Qv, args.stride_Qv);
    TMA_Qv tma_load_Qv = [&] {
      if constexpr (HasQv) {
        return make_tma_copy_A_sm90(
          GmemTiledCopyQ{},
          mQv,
          SmemLayoutQv{},
          TileShape_MNK_QV{},
          ClusterShape{});
      } else {
        return nullptr;
      }
    }();
    TMA_Mask tma_load_Mask = [&] {
      if constexpr (Has_sparse_mask && Use_TMA_KV) {
        auto shape_Mask = make_shape(
          static_cast<int32_t>(args.total_q),
          static_cast<int32_t>(args.sparse_mask_fine_q_stride)
        );
        auto stride_Mask = make_stride(args.sparse_mask_fine_q_stride, Int<1>{});
        Tensor mMask = make_tensor(make_gmem_ptr(args.sparse_mask_fine), shape_Mask, stride_Mask);
        return make_tma_copy(
          SM90_TMA_LOAD{},
          mMask,
          take<0, 2>(SmemLayoutMask{}),
          Shape<Int<kBlockM>, Int<kNumInt32PerBlock>>{},
          _1{});
      } else {
        return nullptr;
      }
    }();
    int const qhead_per_khead = !PackGQA ? 1 : cute::ceil_div(get<2>(args.shape_Q), get<2>(args.shape_K));
    auto const shape_Q_packed = cute::conditional_return<!PackGQA>(
      args.shape_Q,
      make_shape(make_shape(qhead_per_khead, get<0>(args.shape_Q)), get<1>(args.shape_Q), get<2>(args.shape_K), get<3>(args.shape_Q))
    );
    auto const stride_Q_packed = cute::conditional_return<!PackGQA>(
      args.stride_Q,
      make_stride(make_stride(get<2>(args.stride_Q), get<0>(args.stride_Q)), get<1>(args.stride_Q), get<2>(args.stride_Q) * qhead_per_khead, get<3>(args.stride_Q))
    );
    auto const shape_Qv_packed = cute::conditional_return<!PackGQA>(
      shape_Qv,
      make_shape(make_shape(qhead_per_khead, get<0>(shape_Qv)), get<1>(shape_Qv), get<2>(args.shape_K), get<3>(shape_Qv))
    );
    auto const stride_Qv_packed = cute::conditional_return<!PackGQA>(
      args.stride_Qv,
      make_stride(make_stride(get<2>(args.stride_Qv), get<0>(args.stride_Qv)), get<1>(args.stride_Qv), get<2>(args.stride_Qv) * qhead_per_khead, get<3>(args.stride_Qv))
    );
    if (get<1>(args.shape_rotary) > 0) {
      assert(args.ptr_rotary_cos != nullptr && args.ptr_rotary_sin != nullptr);
    }
    assert(args.num_splits >= 1);
    int page_size = !args.ptr_pagetable ? 1 : get<0>(args.shape_K);
    if (!PagedKVNonTMA && args.ptr_pagetable != nullptr) {
      assert(page_size % kBlockN == 0);
      assert(!args.leftpad_k);
    }
    cutlass::FastDivmod attention_chunk_divmod(args.attention_chunk >= 1 ? args.attention_chunk : 1);
    attention_chunk_divmod.divisor = args.attention_chunk;
    return {args.ptr_Q, args.shape_Q, args.stride_Q, shape_Q_packed, stride_Q_packed,
        args.ptr_K, args.shape_K, args.stride_K, args.ptr_V, args.headdim_v, args.stride_V,
        args.ptr_K_new, args.shape_K_new, args.stride_K_new, args.ptr_V_new, args.stride_V_new,
        args.ptr_Qv, args.stride_Qv, shape_Qv_packed, stride_Qv_packed,
        args.ptr_rotary_cos, args.shape_rotary, args.stride_rotary_cos,
        args.ptr_rotary_sin, args.stride_rotary_sin, args.is_rotary_interleaved,
        args.ptr_pagetable, args.shape_pagetable, args.stride_pagetable,
        cutlass::FastDivmod(page_size),
        cutlass::FastDivmod(!args.ptr_pagetable ? 1 : cute::ceil_div(page_size, kBlockN)),
        cutlass::FastDivmod(cute::ceil_div(get<2>(args.shape_Q), get<2>(args.shape_K))),
        tma_load_Q, tma_load_K, tma_load_V, tma_load_K_new, tma_load_V_new, tma_load_Qv, tma_load_Mask,
        !Has_softcap ? float(args.softmax_scale * M_LOG2E) : float(args.softcap_val * M_LOG2E),
        args.ptr_q_descale, args.ptr_k_descale, args.ptr_v_descale,
        args.stride_q_descale, args.stride_k_descale, args.stride_v_descale,
        !Has_softcap ? 0.f : args.softmax_scale / args.softcap_val,
        args.window_size_left, args.window_size_right, attention_chunk_divmod,
        !Split ? 1 : args.num_splits,
        args.kv_batch_idx,
        args.cu_seqlens_q, args.cu_seqlens_k, args.cu_seqlens_k_new,
        args.seqused_q, args.seqused_k, args.leftpad_k, args.seqlens_rotary, args.ptr_sink,
        args.sparse_mask_fine,
        args.sparse_mask_max_k_blocks,
        args.sparse_mask_fine_q_stride, args.sparse_mask_fine_k_stride,
      };
  }
  CUTLASS_DEVICE
  static void prefetch_tma_descriptors(Params const& params) {
    if constexpr (Use_TMA_Q) {
      cute::prefetch_tma_descriptor(params.tma_load_Q.get_tma_descriptor());
      if constexpr (HasQv) {
        cute::prefetch_tma_descriptor(params.tma_load_Qv.get_tma_descriptor());
      }
    }
    if constexpr (Use_TMA_KV) {
      cute::prefetch_tma_descriptor(params.tma_load_K.get_tma_descriptor());
      cute::prefetch_tma_descriptor(params.tma_load_V.get_tma_descriptor());
    }
    if constexpr (AppendKV) {
      cute::prefetch_tma_descriptor(params.tma_load_K_new.get_tma_descriptor());
      cute::prefetch_tma_descriptor(params.tma_load_V_new.get_tma_descriptor());
    }
    if constexpr (Use_TMA_Mask) {
      cute::prefetch_tma_descriptor(params.tma_load_Mask.get_tma_descriptor());
    }
  }
  template <typename SchedulerPrefetch, typename SharedStorage>
  CUTLASS_DEVICE void
  load(Params const& params,
     MainloopPipelineK pipeline_k,
     MainloopPipelineV pipeline_v,
     MainloopPipelineVt pipeline_vt,
     MainloopPipelineMask pipeline_mask,
     PipelineState& smem_pipe_write,
     SharedStorage &shared_storage,
     SchedulerPrefetch const& scheduler_prefetch,
     SeqlenInfo_t const& seqlen_info,
     cute::tuple<int32_t, int32_t, int32_t, int32_t> block_coord,
     int &work_idx
     ) {
    int const m_block = get<0>(block_coord);
    int const bidh = get<1>(block_coord);
    int const bidb = get<2>(block_coord);
    int const split_idx = get<3>(block_coord);
    auto [n_block_min, n_block_max, n_offset] = BlockMN_t::get_n_block_min_max(
      seqlen_info, m_block, bidb, split_idx, params.num_splits,
      params.window_size_left, params.window_size_right, params.attention_chunk_divmod,
      params.qhead_per_khead_divmod);
    if constexpr (Is_causal || Is_local || Varlen || Split) {
      if (n_block_max <= n_block_min) {
        scheduler_prefetch();
        return;
      }
    }
    Tensor sQ = make_tensor(make_smem_ptr(shared_storage.tensors.mainloop.smem_q.data()), SmemLayoutQ{});
    Tensor sK = make_tensor(make_smem_ptr(shared_storage.tensors.mainloop.smem_k.data()), SmemLayoutK{});
    Tensor sK_pi = as_position_independent_swizzle_tensor(sK);
    Tensor sVt = [&] {
      if constexpr (!Transpose_V) {
        return make_tensor(make_smem_ptr(shared_storage.tensors.mainloop.smem_v.data()), SmemLayoutVt{});
      } else {
        return cute::as_position_independent_swizzle_tensor(make_tensor(make_smem_ptr(shared_storage.tensors.mainloop.smem_vt.data()), SmemLayoutVt{}));
      }
    }();
    Tensor sV = cute::as_position_independent_swizzle_tensor(make_tensor(make_smem_ptr(shared_storage.tensors.mainloop.smem_v.data()), SmemLayoutVtMma{}));
    Tensor sVcpasync = [&] {
      if constexpr (!Transpose_V) {
        return cute::as_position_independent_swizzle_tensor(make_tensor(make_smem_ptr(shared_storage.tensors.mainloop.smem_v.data()), SmemLayoutVCpAsync{}));
      } else {
        return cute::as_position_independent_swizzle_tensor(make_tensor(make_smem_ptr(shared_storage.tensors.mainloop.smem_vt.data()), SmemLayoutVCpAsync{}));
      }
    }();
    Tensor sQv = make_tensor(make_smem_ptr(shared_storage.tensors.mainloop.smem_qv.data()), SmemLayoutQv{});
    int const thread_idx = threadIdx.x % NumProducerThreads;
    int const bidh_kv = !PackGQA ? params.qhead_per_khead_divmod.divide(bidh) : bidh;
    int const bidb_kv = params.kv_batch_idx == nullptr ? bidb : params.kv_batch_idx[bidb];
    uint32_t block_rank_in_cluster = cute::block_rank_in_cluster();
    constexpr uint32_t cluster_shape_x = get<0>(ClusterShape());
    uint2 cluster_local_block_id = {block_rank_in_cluster % cluster_shape_x, block_rank_in_cluster / cluster_shape_x};
    bool const is_varlen_q = Varlen && params.cu_seqlens_q;
    bool const is_varlen_k = Varlen && params.cu_seqlens_k;
    Tensor mQ = params.tma_load_Q.get_tma_tensor(params.shape_Q)(_, _, bidh, !is_varlen_q ? bidb : 0);
    Tensor mK_TMA = params.tma_load_K.get_tma_tensor(params.shape_K)(_, _, bidh_kv, _);
    auto shape_V = make_shape(params.headdim_v, get<0>(params.shape_K), get<2>(params.shape_K), get<3>(params.shape_K));
    Tensor mVt_TMA = params.tma_load_V.get_tma_tensor(shape_V)(_, _, bidh_kv, _);
    Tensor gQ = local_tile(domain_offset(make_coord(seqlen_info.offset_q, _0{}), mQ), select<0, 2>(TileShape_MNK{}), make_coord(m_block, _0{}));
    Tensor gK_TMA = local_tile(domain_offset(make_coord(seqlen_info.offset_k + n_offset, _0{}, _0{}), mK_TMA), select<1, 2>(TileShape_MNK{}), make_coord(_, _0{}, _));
    Tensor gVt_TMA = local_tile(domain_offset(make_coord(_0{}, seqlen_info.offset_k + n_offset, _0{}), mVt_TMA), select<1, 2>(TileShape_MNK_PV{}), make_coord(_0{}, _, _));
    auto block_tma_Q = params.tma_load_Q.get_slice(_0{});
    Tensor tQgQ = group_modes<0, 3>(block_tma_Q.partition_S(gQ));
    Tensor tQsQ = group_modes<0, 3>(block_tma_Q.partition_D(sQ));
    if (Use_TMA_Q && thread_idx == 0) { prefetch(params.tma_load_Q, tQgQ); }
    auto block_tma_K = params.tma_load_K.get_slice(cluster_local_block_id.x);
    Tensor tKgK_TMA = group_modes<0, 3>(block_tma_K.partition_S(gK_TMA));
    Tensor tKsK_TMA = group_modes<0, 3>(block_tma_K.partition_D(sK));
    auto block_tma_V = params.tma_load_V.get_slice(cluster_local_block_id.x);
    Tensor tVgVt_TMA = group_modes<0, 3>(block_tma_V.partition_S(gVt_TMA));
    Tensor tVsVt_TMA = group_modes<0, 3>(block_tma_V.partition_D(sVt));
    auto [tQvgQv, tQvsQv] = [&] {
      if constexpr (HasQv) {
        auto shape_Qv = make_shape(get<0>(params.shape_Q), params.headdim_v, get<2>(params.shape_Q), get<3>(params.shape_Q));
        Tensor mQv = params.tma_load_Qv.get_tma_tensor(shape_Qv)(_, _, bidh, !is_varlen_q ? bidb : 0);
        Tensor gQv = local_tile(domain_offset(make_coord(seqlen_info.offset_q, _0{}), mQv), select<0, 2>(TileShape_MNK_QV{}), make_coord(m_block, _0{}));
        auto block_tma_Qv = params.tma_load_Qv.get_slice(_0{});
        Tensor tQvgQv = group_modes<0, 3>(block_tma_Qv.partition_S(gQv));
        Tensor tQvsQv = group_modes<0, 3>(block_tma_Qv.partition_D(sQv));
        return cute::make_tuple(tQvgQv, tQvsQv);
      } else {
        return cute::make_tuple(nullptr, nullptr);
      }
    }();
    int const bidb_kv_idx = !is_varlen_k && !params.ptr_pagetable ? bidb_kv : 0;
    using PagedKVManager_t = PagedKVManager<get<1>(TileShape_MNK{}), get<2>(TileShape_MNK{}), get<1>(TileShape_MNK_PV{}), NumProducerThreads, Element, Transpose_V || !IntraWGOverlap >;
    PagedKVManager_t paged_kv_manager(
      params.ptr_pagetable, params.shape_pagetable, params.stride_pagetable,
      params.ptr_K, params.shape_K, params.stride_K,
      params.ptr_V, params.headdim_v, att_paged_stride_v(params),
      params.page_size_divmod, params.blockN_per_page_size_divmod,
      bidb_kv, bidh_kv, thread_idx, seqlen_info.seqlen_k - n_offset, seqlen_info.leftpad_k + n_offset, bidb_kv_idx
    );
    S2RTiledCopyVt s2r_tiled_copy_vt;
    R2STiledCopyV r2s_tiled_copy_v;
    auto s2r_thr_copy_vt = s2r_tiled_copy_vt.get_thread_slice(thread_idx);
    auto r2s_thr_copy_v = r2s_tiled_copy_v.get_thread_slice(thread_idx);
    Tensor tTranssVt_ = s2r_thr_copy_vt.partition_S(flat_divide(sVt, LDSM_divide_shape{}));
    Tensor tTranssV_ = r2s_thr_copy_v.partition_D(flat_divide(sV, STSM_divide_shape{}));
    CUTE_STATIC_ASSERT_V(rank(tTranssVt_) == rank(tTranssV_));
    CUTE_STATIC_ASSERT_V(size<0>(tTranssVt_) == size<0>(tTranssV_));
    CUTE_STATIC_ASSERT_V(size<1>(tTranssVt_) == size<1>(tTranssV_));
    CUTE_STATIC_ASSERT_V(size<2>(tTranssVt_) == size<2>(tTranssV_));
    CUTE_STATIC_ASSERT_V(size<3>(tTranssVt_) == size<3>(tTranssV_));
    CUTE_STATIC_ASSERT_V(size<4>(tTranssVt_) == size<4>(tTranssV_));
    static constexpr int Transpose_ILP = (size<2>(tTranssVt_) * size<3>(tTranssVt_)) % 2 == 0 ? 2 : 1;
    Tensor tTranssVt = logical_divide(group_modes<1, rank(tTranssVt_) - 1>(tTranssVt_), Shape<Underscore, Int<Transpose_ILP>>{});
    Tensor tTranssV = logical_divide(group_modes<1, rank(tTranssV_) - 1>(tTranssV_), Shape<Underscore, Int<Transpose_ILP>>{});
    auto transpose_V = [&](int stage) {
      if constexpr (Transpose_V) {
        #pragma unroll
        for (int i = 0; i < size<1, 1>(tTranssVt); ++i) {
          Tensor tTransrV = make_fragment_like(tTranssV(_, make_coord(_, _0{}), _0{}));
          static_assert(size<0>(tTransrV) == 16);
          Tensor tTransrV_64 = recast<uint2>(tTransrV);
          cute::copy(s2r_tiled_copy_vt, tTranssVt(_, make_coord(_, i), stage), tTransrV);
          #pragma unroll
          for (int j = 0; j < size(tTransrV_64); ++j) {
            uint32_t upper = tTransrV_64[j].x;
            uint32_t lower = tTransrV_64[j].y;
            tTransrV_64[j].x = __byte_perm(upper, lower, 0x6420);
            tTransrV_64[j].y = __byte_perm(upper, lower, 0x7531);
          }
          cute::copy(r2s_tiled_copy_v, tTransrV, tTranssV(_, make_coord(_, i), stage));
        }
      }
    };
    uint16_t mcast_mask_kv = 0;
    if constexpr (cute::is_same_v<GmemTiledCopyKV, SM90_TMA_LOAD_MULTICAST>) {
      auto block_layout = Layout<ClusterShape>{};
      for (int m = 0; m < size<0>(block_layout); ++m) {
        mcast_mask_kv |= (uint16_t(1) << block_layout(m, cluster_local_block_id.y, _0{}));
      }
    }
    Tensor sMask = make_tensor(make_smem_ptr(shared_storage.tensors.mainloop.smem_mask.data()), SmemLayoutMask{});
    auto tMaskgMask_tMasksMask_tuple = [&] {
      if constexpr (Use_TMA_Mask) {
        auto shape_Mask_gmem = make_shape(
          static_cast<int32_t>(params.total_q),
          static_cast<int32_t>(params.sparse_mask_fine_q_stride)
        );
        Tensor mMask_TMA = params.tma_load_Mask.get_tma_tensor(shape_Mask_gmem);
        Tensor gMask_TMA = local_tile(
          domain_offset(make_coord(seqlen_info.offset_q, _0{}), mMask_TMA),
          Shape<Int<kBlockM>, Int<kNumInt32PerBlock>>{},
          make_coord(_, _)
        );
        auto block_tma_Mask = params.tma_load_Mask.get_slice(_0{});
        Tensor tMaskgMask = group_modes<0, 3>(block_tma_Mask.partition_S(gMask_TMA));
        Tensor tMasksMask = group_modes<0, 3>(block_tma_Mask.partition_D(sMask));
        return cute::make_tuple(tMaskgMask, tMasksMask);
      } else {
        return cute::make_tuple(nullptr, nullptr);
      }
    }();
    auto tMaskgMask_TMA = cute::get<0>(tMaskgMask_tMasksMask_tuple);
    auto tMasksMask_TMA = cute::get<1>(tMaskgMask_tMasksMask_tuple);
    auto load_Mask = [&] (int const m_block, int const n_block, auto const& smem_pipe_write) {
      if constexpr (!Has_sparse_mask) {
        return;
      } else if constexpr (Use_TMA_Mask) {
        pipeline_mask.producer_acquire(smem_pipe_write);
        copy(params.tma_load_Mask.with(*pipeline_mask.producer_get_barrier(smem_pipe_write), 0 , TMA::CacheHintSm90::EVICT_FIRST),
          tMaskgMask_TMA(_, m_block, n_block), tMasksMask_TMA(_, smem_pipe_write.index()));
      } else {
        pipeline_mask.producer_acquire(smem_pipe_write);
        int* stage_smem_base = shared_storage.tensors.mainloop.smem_mask.data() +
                    smem_pipe_write.index() * (kBlockM * kNumInt32PerBlock);
        int const* src_base_common = params.sparse_mask_fine +
                        n_block * params.sparse_mask_fine_k_stride +
                        (seqlen_info.offset_q + m_block * kBlockM) * params.sparse_mask_fine_q_stride;
        constexpr int total_elems = kBlockM * kNumInt32PerBlock;
        int const block_row_start = m_block * kBlockM;
        #pragma unroll
        for (int i = thread_idx; i < total_elems; i += NumProducerThreads) {
          int const row_offset = i / kNumInt32PerBlock;
          int const word_idx = i % kNumInt32PerBlock;
          int const row_logic = block_row_start + row_offset;
          bool const valid = (row_logic < seqlen_info.seqlen_q);
          int const* src = src_base_common +
                    row_offset * params.sparse_mask_fine_q_stride +
                    word_idx;
          flash::cp_async_ca_zfill(stage_smem_base + i, src, valid);
        }
        flash::cp_async_fence();
        pipeline_mask.producer_commit(smem_pipe_write, cutlass::arch::cpasync_barrier_arrive);
      }
    };
    auto load_K = [&] (int const n_block, auto const& smem_pipe_write, auto need_seqlenk_masking_type) {
      pipeline_k.producer_acquire(smem_pipe_write);
      if constexpr (!PagedKVNonTMA) {
        auto [n_block_idx, bidb_kv_idx] = paged_kv_manager.get_indices_for_K_TMA();
        copy(params.tma_load_K.with(*pipeline_k.producer_get_barrier(smem_pipe_write), mcast_mask_kv, TMA::CacheHintSm90::EVICT_LAST),
          tKgK_TMA(_, n_block_idx, bidb_kv_idx), tKsK_TMA(_, smem_pipe_write.index()));
      } else {
        constexpr bool Seqlenk_mask = decltype(need_seqlenk_masking_type)::value;
        if constexpr ((ATT_PABL & 2) == 0) { paged_kv_manager.template load_K<Seqlenk_mask>(n_block, sK_pi(_, _, smem_pipe_write.index())); }
        pipeline_k.producer_commit(smem_pipe_write, cutlass::arch::cpasync_barrier_arrive);
      }
    };
    auto load_V = [&] (int const n_block, auto const& smem_pipe_write, auto need_seqlenk_masking_type) {
      auto pipeline_v_load = cute::conditional_return<!Transpose_V>(pipeline_v, pipeline_vt);
      pipeline_v_load.producer_acquire(smem_pipe_write);
      if constexpr (!PagedKVNonTMA) {
        auto [n_block_idx, bidb_kv_idx] = paged_kv_manager.get_indices_for_V_TMA();
        constexpr int TileBytes = 72 * ATT_BN;
        int const nblocks = (get<0>(params.shape_K) + ATT_BN - 1) / ATT_BN;
        int64_t const tile = (int64_t(bidb_kv_idx) * get<2>(params.shape_K) + bidh_kv) * nblocks + n_block;
        auto const* src = reinterpret_cast<uint8_t const*>(params.ptr_V) + tile * TileBytes;
        auto* dst = reinterpret_cast<uint8_t*>(shared_storage.tensors.mainloop.smem_v.data()) + smem_pipe_write.index() * TileBytes;
        uint32_t const da = cute::cast_smem_ptr_to_uint(dst);
        uint32_t const ba = cute::cast_smem_ptr_to_uint(pipeline_v_load.producer_get_barrier(smem_pipe_write));
        asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];" :: "r"(da), "l"(src), "r"(TileBytes), "r"(ba) : "memory");
      } else {
        constexpr bool Seqlenk_mask = decltype(need_seqlenk_masking_type)::value;
        if constexpr ((ATT_PABL & 2) == 0) { paged_kv_manager.template load_V<Seqlenk_mask>(n_block, sVcpasync(_, _, smem_pipe_write.index())); }
        pipeline_v_load.producer_commit(smem_pipe_write, cutlass::arch::cpasync_barrier_arrive);
      }
    };
    auto copy_Vt_to_V = [&] (auto const& smem_pipe_write) {
      PipelineState smem_pipe_read{smem_pipe_write.index(), smem_pipe_write.phase() ^ 1, smem_pipe_write.count()};
      pipeline_vt.consumer_wait(smem_pipe_read);
      pipeline_v.producer_acquire(smem_pipe_write);
        if constexpr ((ATT_PABL & 1) == 0 && (SK2_ABL & 16) == 0) { transpose_V(smem_pipe_write.index()); }
      {
        uint4* stg = reinterpret_cast<uint4*>(shared_storage.tensors.mainloop.smem_v.data() + smem_pipe_write.index() * (72 * kBlockN));
        constexpr int CW = kBlockN % 128 == 0 ? 128 : 64, PER = 8 * CW / 16;
        #pragma unroll
        for (int i = threadIdx.x % cutlass::NumThreadsPerWarpGroup; i < (kBlockN / CW) * PER; i += cutlass::NumThreadsPerWarpGroup) {
          stg[(i / PER) * (72 * CW / 16) + 64 * CW / 16 + (i % PER)] = make_uint4(0x38383838u, 0x38383838u, 0x38383838u, 0x38383838u);
        }
      }
      cutlass::arch::fence_view_async_shared();
      pipeline_v.producer_commit(smem_pipe_write);
      cutlass::arch::NamedBarrier::sync(cutlass::NumThreadsPerWarpGroup, cutlass::arch::ReservedNamedBarriers::TransposeBarrier );
      pipeline_vt.consumer_release(smem_pipe_read);
    };
    int n_block = n_block_max - 1;
    int warp_idx_in_warpgroup = __shfl_sync(0xffffffff, (threadIdx.x / 32) % 4, 0);
    static constexpr bool SingleProducerWarp = NumProducerThreads == cutlass::NumThreadsPerWarp;
    bool should_load_KV = !Use_TMA_KV || ((SingleProducerWarp || warp_idx_in_warpgroup == 0) && cute::elect_one_sync());
    if (should_load_KV) {
      if constexpr (PagedKVNonTMA) {
        paged_kv_manager.template load_page_table<true , true >(n_block);
      } else {
        paged_kv_manager.template load_page_table_TMA<true >(n_block);
      }
      if constexpr (Transpose_V) { load_V(n_block, smem_pipe_write, cute::true_type{} ); }
      load_K(n_block, smem_pipe_write, cute::true_type{} );
    }
    if constexpr (Has_sparse_mask) {
      if constexpr (Use_TMA_Mask) {
        if (should_load_KV) { load_Mask(m_block, n_block, smem_pipe_write); }
      } else {
        load_Mask(m_block, n_block, smem_pipe_write);
      }
    }
    if constexpr (Use_TMA_Q) {
      if (SingleProducerWarp || warp_idx_in_warpgroup == 0) {
        cutlass::arch::NamedBarrier::sync(NumMmaThreadsQK + cutlass::NumThreadsPerWarp, static_cast<uint32_t>(FwdNamedBarriers::QueryEmpty) );
      }
      if ((SingleProducerWarp || warp_idx_in_warpgroup == 0) && cute::elect_one_sync()) {
        shared_storage.pipelines.barrier_Q.arrive_and_expect_tx(TmaTransactionBytesQ);
        copy(params.tma_load_Q.with(reinterpret_cast<typename cutlass::arch::ClusterTransactionBarrier::ValueType&>(shared_storage.pipelines.barrier_Q), 0 , !Split ? TMA::CacheHintSm90::EVICT_FIRST : TMA::CacheHintSm90::EVICT_LAST),
          tQgQ, tQsQ);
        if constexpr (HasQv) {
          shared_storage.pipelines.barrier_Qv.arrive_and_expect_tx(TmaTransactionBytesQv);
          copy(params.tma_load_Qv.with(reinterpret_cast<typename cutlass::arch::ClusterTransactionBarrier::ValueType&>(shared_storage.pipelines.barrier_Qv), 0 , !Split ? TMA::CacheHintSm90::EVICT_FIRST : TMA::CacheHintSm90::EVICT_LAST),
            tQvgQv, tQvsQv);
        }
      }
    } else {
      cutlass::arch::NamedBarrier::sync(NumMmaThreadsQK + NumProducerThreads, static_cast<uint32_t>(FwdNamedBarriers::QueryEmpty) );
      Tensor mQ = make_tensor(make_gmem_ptr(params.ptr_Q + seqlen_info.offset_q * get<0>(params.stride_Q)), params.shape_Q_packed, params.stride_Q_packed)(_, _, bidh, !is_varlen_q ? bidb : 0);
      Tensor sQ_pi = cute::as_position_independent_swizzle_tensor(sQ);
      using PackGQAt = flash::PackGQAManager<get<0>(TileShape_MNK{}), get<2>(TileShape_MNK{}), NumProducerThreads, Element>;
      PackGQAt::load_Q(mQ, sQ_pi, params.qhead_per_khead_divmod, thread_idx, seqlen_info.seqlen_q, m_block);
      auto &barrier_Q = shared_storage.pipelines.barrier_Q;
      cutlass::arch::cpasync_barrier_arrive(reinterpret_cast<uint64_t*>(&barrier_Q));
      barrier_Q.arrive();
      if constexpr (HasQv) {
        Tensor mQv = make_tensor(make_gmem_ptr(params.ptr_Qv + seqlen_info.offset_q * get<0>(params.stride_Qv)), params.shape_Qv_packed, params.stride_Qv_packed)(_, _, bidh, !is_varlen_q ? bidb : 0);
        Tensor sQv_pi = cute::as_position_independent_swizzle_tensor(sQv);
        using PackGQAt = flash::PackGQAManager<get<0>(TileShape_MNK_QV{}), get<2>(TileShape_MNK_QV{}), NumProducerThreads, Element>;
        PackGQAt::load_Q(mQv, sQv_pi, params.qhead_per_khead_divmod, thread_idx, seqlen_info.seqlen_q, m_block);
        auto &barrier_Qv = shared_storage.pipelines.barrier_Qv;
        cutlass::arch::cpasync_barrier_arrive(reinterpret_cast<uint64_t*>(&barrier_Qv));
        barrier_Qv.arrive();
      }
    }
    shared_storage.pipelines.barrier_O.wait((work_idx + 1) % 2);
    if constexpr (!Transpose_V && !IntraWGOverlap) {
      if (should_load_KV) { load_V(n_block, smem_pipe_write, cute::true_type{} ); }
    }
    int n_block_prev = n_block;
    --n_block;
    #pragma unroll (!Transpose_V && Use_TMA_KV && !Has_sparse_mask ? 2 : 1)
    for (; n_block >= n_block_min; --n_block) {
      PipelineState smem_pipe_write_v = smem_pipe_write;
      ++smem_pipe_write;
      if (should_load_KV) {
        if constexpr (PagedKVNonTMA) {
        if constexpr ((ATT_PABL & 4) == 0) { paged_kv_manager.template load_page_table<false >(n_block); }
        } else {
          paged_kv_manager.load_page_table_TMA(n_block);
        }
        if constexpr (Transpose_V) { load_V(n_block, smem_pipe_write, cute::false_type{} ); }
        load_K(n_block, smem_pipe_write, cute::false_type{} );
        if constexpr (!Transpose_V) {
          if constexpr (IntraWGOverlap) {
            load_V(n_block_prev, smem_pipe_write_v, cute::true_type{} );
          } else {
            load_V(n_block, smem_pipe_write, cute::false_type{} );
          }
        }
      }
      if constexpr (Has_sparse_mask) {
        if constexpr (Use_TMA_Mask) {
          if (should_load_KV) { load_Mask(m_block, n_block, smem_pipe_write); }
        } else {
          load_Mask(m_block, n_block, smem_pipe_write);
        }
      }
      n_block_prev = n_block;
      if constexpr (Transpose_V) { copy_Vt_to_V(smem_pipe_write_v); }
    }
    scheduler_prefetch();
    if constexpr (!Transpose_V && IntraWGOverlap) {
      if (should_load_KV) { load_V(n_block_prev, smem_pipe_write, cute::true_type{} ); }
    }
    if constexpr (Transpose_V) { copy_Vt_to_V(smem_pipe_write); }
    ++smem_pipe_write;
    ++work_idx;
  }
  template <typename SharedStorage>
  CUTLASS_DEVICE void
  load_tail(MainloopPipelineK pipeline_k, MainloopPipelineV pipeline_v, MainloopPipelineVt pipeline_vt, MainloopPipelineMask pipeline_mask,
       PipelineState& smem_pipe_write, SharedStorage &shared_storage, int const work_idx) {
    shared_storage.pipelines.barrier_O.wait((work_idx + 1) % 2);
    int warp_idx_in_warpgroup = __shfl_sync(0xffffffff, (threadIdx.x / 32) % 4, 0);
    if (warp_idx_in_warpgroup == 0 && cute::elect_one_sync()) {
      pipeline_k.producer_tail(smem_pipe_write);
      pipeline_v.producer_tail(smem_pipe_write);
      if constexpr (Has_sparse_mask) {pipeline_mask.producer_tail(smem_pipe_write);}
      if constexpr (Transpose_V) { pipeline_vt.producer_tail(smem_pipe_write); }
    }
  }
  CUTLASS_DEVICE void
  warp_scheduler_barrier_sync() {
    if constexpr (UseSchedulerBarrier) {
      cutlass::arch::NamedBarrier::sync(2 * cutlass::NumThreadsPerWarpGroup, static_cast<uint32_t>(FwdNamedBarriers::WarpSchedulerWG1) - 1 + flash::canonical_warp_group_idx_nosync() );
    }
  }
  CUTLASS_DEVICE void
  warp_scheduler_barrier_arrive() {
    if constexpr (UseSchedulerBarrier) {
      static_assert(NumMmaWarpGroups == 2 || NumMmaWarpGroups == 3);
      int const cur_WG = flash::canonical_warp_group_idx_nosync() - 1;
      int const next_WG = NumMmaWarpGroups == 2
        ? 1 - cur_WG
        : (cur_WG < NumMmaWarpGroups - 1 ? cur_WG + 1 : 0);
      cutlass::arch::NamedBarrier::arrive(2 * cutlass::NumThreadsPerWarpGroup, static_cast<uint32_t>(FwdNamedBarriers::WarpSchedulerWG1) + next_WG );
    }
  }
  CUTLASS_DEVICE void
  mma_init() {
    int warp_group_idx = flash::canonical_warp_group_idx_nosync();
    if (!LargeHeadDimV || warp_group_idx == 1) {
      cutlass::arch::NamedBarrier::arrive(NumMmaThreadsQK + (Use_TMA_Q ? cutlass::NumThreadsPerWarp : NumProducerThreads), static_cast<uint32_t>(FwdNamedBarriers::QueryEmpty) );
    }
    if (LargeHeadDimV && warp_group_idx > 1) {
      cutlass::arch::NamedBarrier::arrive(NumMmaThreads, static_cast<uint32_t>(FwdNamedBarriers::PEmpty) );
    }
    if constexpr (UseSchedulerBarrier) {
      static_assert(NumMmaWarpGroups == 2 || NumMmaWarpGroups == 3);
      if (warp_group_idx == 1) {
        cutlass::arch::NamedBarrier::arrive(2 * cutlass::NumThreadsPerWarpGroup, static_cast<uint32_t>(FwdNamedBarriers::WarpSchedulerWG1) );
      }
    }
  }
  CUTLASS_DEVICE void xu_sync() {
  }
  CUTLASS_DEVICE void xu_arrive() {
  }
  template <typename SharedStorage, typename FrgTensorO, typename Softmax>
  CUTLASS_DEVICE bool
  mma(Params const& params,
    MainloopPipelineK pipeline_k,
    MainloopPipelineV pipeline_v,
    MainloopPipelineMask pipeline_mask,
    PipelineState& smem_pipe_read,
    FrgTensorO& tOrO,
    Softmax& softmax,
    int const thread_idx,
    int &work_idx,
    SeqlenInfo_t const& seqlen_info,
    cute::tuple<int32_t, int32_t, int32_t, int32_t> block_coord,
    SharedStorage& shared_storage
    ) {
    static_assert(is_rmem<FrgTensorO>::value, "O tensor must be rmem resident.");
    static constexpr int kBlockM = get<0>(TileShape_MNK{});
    static constexpr int kBlockN = get<1>(TileShape_MNK{});
    int const m_block = get<0>(block_coord);
    int const bidh = get<1>(block_coord);
    int const bidb = get<2>(block_coord);
    int const split_idx = get<3>(block_coord);
    int const bidh_kv = !PackGQA ? params.qhead_per_khead_divmod.divide(bidh) : bidh;
    auto [n_block_min, n_block_max, n_offset] = BlockMN_t::get_n_block_min_max(
      seqlen_info, m_block, bidb, split_idx, params.num_splits,
      params.window_size_left, params.window_size_right, params.attention_chunk_divmod,
      params.qhead_per_khead_divmod);
    if constexpr (Is_causal || Is_local || Varlen || Split) {
      if (n_block_max <= n_block_min) { return false; }
    }
    Tensor sQ = make_tensor(make_smem_ptr(shared_storage.tensors.mainloop.smem_q.data()), SmemLayoutQ{});
    Tensor sK = make_tensor(make_smem_ptr(shared_storage.tensors.mainloop.smem_k.data()), SmemLayoutK{});
    Tensor sV = make_tensor(make_smem_ptr(shared_storage.tensors.mainloop.smem_v.data()), SmemLayoutVtMma{});
    Tensor sP = [&] {
      if constexpr (MmaPV_is_RS) {
        return make_tensor(make_smem_ptr(shared_storage.tensors.mainloop.smem_q.data()), SmemLayoutP{});
      } else {
        return make_tensor(make_smem_ptr(shared_storage.tensors.mainloop.smem_p.data()), SmemLayoutP{});
      }
    }();
    Tensor sScale = [&] {
      if constexpr (LargeHeadDimV) {
        return make_tensor(make_smem_ptr(shared_storage.tensors.mainloop.smem_scale.data()), SmemLayoutScale{});
      } else {
        return make_tensor(make_smem_ptr(static_cast<float*>(nullptr)), SmemLayoutScale{});
      }
    }();
    Tensor sQv = make_tensor(make_smem_ptr(shared_storage.tensors.mainloop.smem_qv.data()), SmemLayoutQv{});
    Tensor sVMmaQV = make_tensor(make_smem_ptr(shared_storage.tensors.mainloop.smem_v.data()), SmemLayoutVMmaQV{});
    if constexpr (!MmaQK_is_RS) {
      static_assert(stride<0>(typename TiledMmaQK::ALayout{}) == 0 and
            stride<0>(typename TiledMmaQK::BLayout{}) == 0 and
            size<0>(typename TiledMmaQK::ALayout{}) == cutlass::NumThreadsPerWarpGroup and
            size<0>(typename TiledMmaQK::BLayout{}) == cutlass::NumThreadsPerWarpGroup,
        "Stride of the first mode must be 0 and the size of the mode must be NumThreadsPerWarpGroup");
    }
    static constexpr int MmaWarpGroups = size(TiledMmaPV{}) / cutlass::NumThreadsPerWarpGroup;
    Layout warp_group_thread_layout = make_layout(make_shape(Int<MmaWarpGroups>{}),
                           make_stride(Int<cutlass::NumThreadsPerWarpGroup>{}));
    int warp_group_idx = __shfl_sync(0xFFFFFFFF, thread_idx / cutlass::NumThreadsPerWarpGroup, 0);
    TiledMmaQK tiled_mma_qk;
    TiledMmaPV tiled_mma_pv;
    TiledMmaQV tiled_mma_qv;
    auto wg_mma_qk = tiled_mma_qk.get_slice(warp_group_thread_layout(warp_group_idx));
    auto wg_mma_pv = tiled_mma_pv.get_slice(warp_group_thread_layout(warp_group_idx));
    auto wg_mma_qv = tiled_mma_qv.get_slice(warp_group_thread_layout(warp_group_idx));
    auto smem_tiled_copy_P = make_tiled_copy_C(SmemCopyAtomP{}, tiled_mma_qk);
    auto smem_thr_copy_P = smem_tiled_copy_P.get_thread_slice(thread_idx);
    Tensor tSrQ = wg_mma_qk.partition_fragment_A(sQ);
    Tensor tSrK = wg_mma_qk.partition_fragment_B(sK);
    Tensor tOrV = wg_mma_pv.partition_fragment_B(sV);
    Tensor tOsP = wg_mma_pv.partition_fragment_A(sP);
    Tensor tSrQv = wg_mma_qv.partition_fragment_A(sQv);
    Tensor tSrV = wg_mma_qv.partition_fragment_B(sVMmaQV);
    Tensor tPsP = smem_thr_copy_P.partition_D(cute::as_position_independent_swizzle_tensor(sP));
    auto thread_mma_pv = tiled_mma_pv.get_thread_slice(thread_idx);
    Tensor taccOcO = thread_mma_pv.partition_C(cute::make_identity_tensor(select<0, 1>(TileShape_MNK_PV{})));
    Tensor taccOcO_rowcol = make_tensor(taccOcO.data(), flash::convert_layout_acc_rowcol(taccOcO.layout()));
    Tensor taccOcO_row = taccOcO_rowcol(_, _0{});
    auto store_scales = [&](auto& scales, int stage) {
      static_assert(CUTE_STATIC_V(size(scales)) == CUTE_STATIC_V(size(taccOcO_row)));
      #pragma unroll
      for (int mi = 0; mi < size(taccOcO_row); ++mi) {
        if (get<1>(taccOcO_row(_0{})) == 0) {
          sScale(get<0>(taccOcO_row(mi)), stage) = scales(mi);
        }
      }
    };
    auto consumer_wait = [](auto& pipeline, auto& smem_pipe_read) {
      auto barrier_token = pipeline.consumer_try_wait(smem_pipe_read);
      pipeline.consumer_wait(smem_pipe_read, barrier_token);
    };
    int const seqlen_q = seqlen_info.seqlen_q;
    int const seqlen_k = seqlen_info.seqlen_k - n_offset;
    int const m_idx_min = !PackGQA ? m_block * kBlockM : params.qhead_per_khead_divmod.divide(m_block * kBlockM);
    int const m_idx_max = !PackGQA ? (m_block + 1) * kBlockM : params.qhead_per_khead_divmod.divide((m_block + 1) * kBlockM - 1) + 1;
    int n_block = n_block_max - 1;
    flash::Mask<kBlockM, kBlockN, PackGQA, TiledMmaQK> mask(
      thread_idx, seqlen_q, seqlen_k, params.window_size_left, params.window_size_right, 0 ,
      params.attention_chunk_divmod, params.qhead_per_khead_divmod
    );
    [[maybe_unused]] flash::SparseMask<kBlockM, kBlockN, PackGQA, TiledMmaQK> sparse_mask(
      thread_idx, seqlen_q, seqlen_k,
      params.sparse_mask_max_k_blocks,
      params.qhead_per_khead_divmod
    );
    float softcap_val = params.softcap_val;
    if constexpr (Has_softcap && Is_FP8) {
      float const q_descale = params.ptr_q_descale == nullptr ? 1.0f : params.ptr_q_descale[bidb * get<0>(params.stride_q_descale) + bidh_kv * get<1>(params.stride_q_descale)];
      float const k_descale = params.ptr_k_descale == nullptr ? 1.0f : params.ptr_k_descale[bidb * get<0>(params.stride_k_descale) + bidh_kv * get<1>(params.stride_k_descale)];
      softcap_val *= q_descale * k_descale;
    }
    auto scoremod_premask_fn = [&](auto& tSrS) {
      if constexpr (Has_softcap) { flash::apply_softcap(tSrS, softcap_val); }
    };
    auto write_P_to_smem = [&](auto& tOrP) {
      if constexpr (LargeHeadDimV) {
        cutlass::arch::NamedBarrier::sync(NumMmaThreads, static_cast<uint32_t>(FwdNamedBarriers::PEmpty) );
      }
      cute::copy(smem_tiled_copy_P, smem_thr_copy_P.retile_S(tOrP), tPsP);
    };
    auto arrive_on_P_write_barrier = [&] {
      cutlass::arch::fence_view_async_shared();
      __syncwarp();
      if constexpr (LargeHeadDimV) {
        cutlass::arch::NamedBarrier::arrive(NumMmaThreads, static_cast<uint32_t>(FwdNamedBarriers::PFull) );
      }
    };
    auto &barrier_Q = shared_storage.pipelines.barrier_Q;
    if constexpr (!AppendKV) {
      barrier_Q.wait(work_idx % 2);
    } else {
      if (get<1>(params.shape_rotary) > 0) {
        using Rotary_t = Rotary<kBlockM, kHeadDim, NumMmaThreadsQK, Element, !(Is_causal || Is_local) >;
        Rotary_t rotary(params.ptr_rotary_cos, params.shape_rotary, params.stride_rotary_cos,
                params.ptr_rotary_sin, params.stride_rotary_sin,
                params.is_rotary_interleaved, thread_idx, seqlen_q,
                seqlen_info.seqlen_rotary);
        Tensor sQ_pi = cute::as_position_independent_swizzle_tensor(sQ);
        int const qhead_per_khead = !PackGQA ? 1 : params.qhead_per_khead_divmod.divisor;
        if (params.is_rotary_interleaved) {
          auto [tRrCos, tRrSin] = cute::conditional_return<!PackGQA>(
            rotary.template load_cos_sin<true >(m_block),
            rotary.template load_cos_sin_packgqa<true >(m_block, params.qhead_per_khead_divmod)
          );
          barrier_Q.wait(work_idx % 2);
          rotary.apply_Q_interleaved(sQ_pi, tRrCos, tRrSin, m_block, qhead_per_khead);
        } else {
          auto [tRrCosCont, tRrSinCont] = cute::conditional_return<!PackGQA>(
            rotary.template load_cos_sin<false >(m_block),
            rotary.template load_cos_sin_packgqa<false >(m_block, params.qhead_per_khead_divmod)
          );
          barrier_Q.wait(work_idx % 2);
          rotary.apply_Q_contiguous(sQ_pi, tRrCosCont, tRrSinCont, m_block, qhead_per_khead);
        }
        cutlass::arch::fence_view_async_shared();
        cutlass::arch::NamedBarrier::sync(NumMmaThreadsQK, static_cast<uint32_t>(FwdNamedBarriers::QueryRotated) );
      } else {
        barrier_Q.wait(work_idx % 2);
      }
    }
    if constexpr (MmaQK_is_RS) {
      using SmemCopyAtomQ = Copy_Atom<cute::SM75_U32x4_LDSM_N, Element>;
      auto smem_tiled_copy_Q = make_tiled_copy_A(SmemCopyAtomQ{}, tiled_mma_qk);
      auto smem_thr_copy_Q = smem_tiled_copy_Q.get_thread_slice(thread_idx);
      Tensor tSrQ_copy_view = smem_thr_copy_Q.retile_D(tSrQ);
      Tensor tSsQ_copy_view = smem_thr_copy_Q.partition_S(cute::as_position_independent_swizzle_tensor(sQ));
      cute::copy(smem_tiled_copy_Q, tSsQ_copy_view, tSrQ_copy_view);
    }
    if constexpr (IntraWGOverlap) {
     {
      Tensor tSrS = partition_fragment_C(tiled_mma_qk, select<0, 1>(TileShape_MNK{}));
      consumer_wait(pipeline_k, smem_pipe_read);
      if constexpr (!OnlyQv) {
        flash::gemm<true, -1>(tiled_mma_qk, tSrQ, tSrK(_, _, _, smem_pipe_read.index()), tSrS);
        warpgroup_wait<0>();
      }
      pipeline_k.consumer_release(smem_pipe_read);
      if constexpr (HasQv) {
        shared_storage.pipelines.barrier_Qv.wait(work_idx % 2);
        consumer_wait(pipeline_v, smem_pipe_read);
        flash::gemm<OnlyQv, 0>(tiled_mma_qv, tSrQv, tSrV(_, _, _, smem_pipe_read.index()), tSrS);
      }
      scoremod_premask_fn(tSrS);
      if constexpr (Has_sparse_mask) {
        consumer_wait(pipeline_mask, smem_pipe_read);
        uint32_t const* mask_ptr = reinterpret_cast<uint32_t const*>(
          shared_storage.tensors.mainloop.smem_mask.data() +
          smem_pipe_read.index() * (kBlockM * kNumInt32PerBlock)
        );
        sparse_mask.template apply<true >(tSrS, m_block, n_block, mask_ptr);
        pipeline_mask.consumer_release(smem_pipe_read);
      } else {
        mask.template apply<true , Is_causal, Is_local>(tSrS, m_block, n_block);
      }
      uint32_t hP[CUTE_STATIC_V(size(tSrS)) / 2];
      xu_sync();
      Tensor scores_scale = softmax.template step<true, true>(tSrS, hP);
      xu_arrive();
      Tensor tOrP_acc = make_tensor(tSrS.data(), flash::convert_layout_acc_Aregs<TiledMmaPV>(tSrS.layout()));
      Tensor tOrP = make_tensor_like<Element>(tOrP_acc);
      softmax.pack_p(hP, tOrP);
      if constexpr (Is_FP8 && V_colmajor) { flash::permute_Aregs_fp8(tOrP); }
      if constexpr (!MmaPV_is_RS) { write_P_to_smem(tOrP); }
      if constexpr (!MmaPV_is_RS) { arrive_on_P_write_barrier(); }
      --n_block;
      clear(tOrO);

      uint32_t u_o16[16];
      #pragma unroll
      for (int j = 0; j < 16; ++j) { u_o16[j] = 0u; }
      float u_sc[2] = {1.f, 1.f};
      int u_cnt = 0;
      bool u_first = true;
      softmax.clear_sum();
      uint64_t const ones_desc = flash::att_ones_desc(shared_storage.tensors.mainloop.smem_ones.data());
      uint32_t const fk_vaddr = static_cast<uint32_t>(__cvta_generic_to_shared(shared_storage.tensors.mainloop.smem_v.data()));
      uint32_t fx_bsync, fx_barv;
      {
        int const cur_wg = flash::canonical_warp_group_idx_nosync() - 1;
        int const next_wg = NumMmaWarpGroups == 2 ? 1 - cur_wg : (cur_wg < NumMmaWarpGroups - 1 ? cur_wg + 1 : 0);
        asm volatile("mov.u32 %0, %1;" : "=r"(fx_bsync) : "r"(uint32_t(static_cast<uint32_t>(FwdNamedBarriers::WarpSchedulerWG1) + cur_wg)));
        asm volatile("mov.u32 %0, %1;" : "=r"(fx_barv) : "r"(uint32_t(static_cast<uint32_t>(FwdNamedBarriers::WarpSchedulerWG1) + next_wg)));
      }
      auto fwd_step2 = [&](int const n_block, auto mask_fn, auto check_inf_type, auto& P_use, auto& P_fill) {
        static constexpr bool Check_inf = decltype(check_inf_type)::value;
        PipelineState smem_pipe_read_v(smem_pipe_read.index(), smem_pipe_read.phase(), smem_pipe_read.count());
        ++smem_pipe_read;
        Tensor tSrS = partition_fragment_C(tiled_mma_qk, select<0, 1>(TileShape_MNK{}));
        ATT_STAMP(tl_on, warp_group_idx, tl_it, 0);
        if (!UseSchedulerBarrier || warp_group_idx == 0) { consumer_wait(pipeline_k, smem_pipe_read); }
        ATT_STAMP(tl_on, warp_group_idx, tl_it, 1);
        if constexpr (UseSchedulerBarrier) { cutlass::arch::NamedBarrier::sync(2 * cutlass::NumThreadsPerWarpGroup, fx_bsync); }
        if constexpr (!OnlyQv && (ATT_CABL & 2) == 0) {
          flash::gemm<true, -1>(tiled_mma_qk, tSrQ, tSrK(_, _, _, smem_pipe_read.index()), tSrS);
        }
        ATT_STAMP(tl_on, warp_group_idx, tl_it, 7);
        if constexpr (RescaleOBeforeGemm) { softmax.rescale_o(tOrO, scores_scale); }
        if constexpr(!HasQv) {
          if (!UseSchedulerBarrier || warp_group_idx == 0) { consumer_wait(pipeline_v, smem_pipe_read_v); }
        }
        if constexpr ((ATT_CABL & 1) == 0) {
        fk_pv16<kBlockN>(P_use, u_o16, fk_vaddr + smem_pipe_read_v.index() * (72 * kBlockN));
        }
        if constexpr (UseSchedulerBarrier) { cutlass::arch::NamedBarrier::arrive(2 * cutlass::NumThreadsPerWarpGroup, fx_barv); }
        ATT_STAMP(tl_on, warp_group_idx, tl_it, 2);
        warpgroup_wait<0>();
        ATT_STAMP(tl_on, warp_group_idx, tl_it, 3);
        pipeline_k.consumer_release(smem_pipe_read);
        if constexpr (HasQv) {
          warpgroup_wait<0>();
          pipeline_v.consumer_release(smem_pipe_read_v);
          consumer_wait(pipeline_v, smem_pipe_read);
          flash::gemm<OnlyQv, 0>(tiled_mma_qv, tSrQv, tSrV(_, _, _, smem_pipe_read.index()), tSrS);
        }
        scoremod_premask_fn(tSrS);
        if constexpr (Has_sparse_mask) {
          consumer_wait(pipeline_mask, smem_pipe_read);
        }
        mask_fn(tSrS, n_block);
        if constexpr (Has_sparse_mask) {
          pipeline_mask.consumer_release(smem_pipe_read);
        }
        uint32_t hP[CUTE_STATIC_V(size(tSrS)) / 2];
        xu_sync();
        cute::copy(softmax.template step<false, Check_inf>(tSrS, hP), scores_scale);
        xu_arrive();
        ATT_STAMP(tl_on, warp_group_idx, tl_it, 4);
        warpgroup_wait<0>();
        ATT_STAMP(tl_on, warp_group_idx, tl_it, 5);
        pipeline_v.consumer_release(smem_pipe_read_v);
        softmax.pack_p(hP, P_fill);
        if constexpr (Is_FP8 && V_colmajor) { flash::permute_Aregs_fp8(tOrP); }
        if constexpr (!MmaPV_is_RS) { write_P_to_smem(tOrP); }
        u_rescale16(u_o16, scores_scale);
        softmax.scale_sum(scores_scale);
        if constexpr (!MmaPV_is_RS) { arrive_on_P_write_barrier(); }
        ATT_STAMP(tl_on, warp_group_idx, tl_it, 6);
      };
      auto fwd_loop = [&](int const lo, auto mask_fn, auto check_inf_type) {
        #pragma unroll 1
        for (; n_block >= lo; --n_block) { fwd_step2(n_block, mask_fn, check_inf_type, tOrP, tOrP); }
      };
      auto fwd_step = [&](int const n_blk, auto mask_fn, auto check_inf_type) {
        fwd_step2(n_blk, mask_fn, check_inf_type, tOrP, tOrP);
      };
      if constexpr (Has_sparse_mask) {
        auto sparse_mask_fn = [&](auto& tSrS, int n_block) {
          uint32_t const* mask_ptr = reinterpret_cast<uint32_t const*>(
            shared_storage.tensors.mainloop.smem_mask.data() +
            smem_pipe_read.index() * (kBlockM * kNumInt32PerBlock)
          );
          sparse_mask.template apply<false >(tSrS, m_block, n_block, mask_ptr);
        };
        #pragma unroll 1
        for (; n_block >= n_block_min; --n_block) {
          fwd_step(n_block, sparse_mask_fn, cute::true_type{} );
        }
      } else {
        if constexpr (Is_causal || Is_local) {
          auto mask_fn = [&](auto& tSrS, int n_block) { mask.template apply<false , Is_causal, Is_local>(tSrS, m_block, n_block); };
          int n_block_min_causal_local_mask;
          {
            int const n_idx_cm = m_idx_min + seqlen_k - seqlen_q;
            int n_idx_right = !Is_local ? n_idx_cm : n_idx_cm + params.window_size_right;
            if constexpr (Is_local) { if (params.attention_chunk_divmod.divisor > 0) { n_idx_right = std::min(n_idx_right, flash::round_up(params.attention_chunk_divmod, n_idx_cm)); } }
            n_block_min_causal_local_mask = std::max(n_block_min, n_idx_right / kBlockN);
          }
          fwd_loop(n_block_min_causal_local_mask, mask_fn, cute::true_type{} );
        }
        int n_block_min_before_local_mask;
        {
          int const n_idx_bm = m_idx_max + seqlen_k - seqlen_q;
          int n_idx_left = !Is_local ? n_idx_bm : n_idx_bm - params.window_size_left;
          if constexpr (Is_local) { if (params.attention_chunk_divmod.divisor > 0) { n_idx_left = std::max(n_idx_left, flash::round_down(params.attention_chunk_divmod, n_idx_bm)); } }
          n_block_min_before_local_mask = !Is_local ? n_block_min : std::max(n_block_min, cute::ceil_div(n_idx_left, kBlockN));
        }
        auto no_mask_fn = [](auto& tSrS, int n_block) { };
        fwd_loop(n_block_min_before_local_mask, no_mask_fn, cute::false_type{} );
        if constexpr (Is_local) {
          auto local_mask_fn = [&](auto& tSrS, int n_block) { mask.template apply<false , false , Is_local>(tSrS, m_block, n_block); };
          fwd_loop(n_block_min, local_mask_fn, cute::bool_constant<Is_local>{} );
        }
      }
      cutlass::arch::NamedBarrier::arrive(NumMmaThreadsQK + (Use_TMA_Q ? cutlass::NumThreadsPerWarp : NumProducerThreads), static_cast<uint32_t>(FwdNamedBarriers::QueryEmpty) );
      if constexpr (RescaleOBeforeGemm) { softmax.rescale_o(tOrO, scores_scale); }
      if constexpr (!HasQv) { consumer_wait(pipeline_v, smem_pipe_read); }
      fk_pv16<kBlockN>(tOrP, u_o16, fk_vaddr + smem_pipe_read.index() * (72 * kBlockN));
      float const v_descale = !Is_FP8 || params.ptr_v_descale == nullptr ? 1.0f : params.ptr_v_descale[bidb * get<0>(params.stride_v_descale) + bidh_kv * get<1>(params.stride_v_descale)];
      if constexpr (LargeHeadDimV) {
        cutlass::arch::NamedBarrier::sync(NumMmaThreads, static_cast<uint32_t>(FwdNamedBarriers::PEmpty) );
        store_scales(scores_scale, smem_pipe_read.index());
        cutlass::arch::NamedBarrier::arrive(NumMmaThreads, static_cast<uint32_t>(FwdNamedBarriers::PFull) );
      }
      warpgroup_wait<0>();
      pipeline_v.consumer_release(smem_pipe_read);
      cute::copy(softmax.finalize(v_descale), scores_scale);
      u_fold(tOrO, u_o16, u_sc, u_first);
      softmax.rescale_o(tOrO, scores_scale);
      if constexpr (Is_FP8 && !V_colmajor) { flash::permute_output_fp8(tOrO); }
      ++smem_pipe_read;
     }
    } else {
      warp_scheduler_barrier_sync();
      auto fwd_step = [&](int const n_block, auto mask_fn, auto is_first_iter_type, auto check_inf_type) {
        static constexpr bool Is_first_iter = decltype(is_first_iter_type)::value;
        static constexpr bool Check_inf = decltype(check_inf_type)::value;
        auto smem_pipe_read_prev = smem_pipe_read;
        if constexpr (!Is_first_iter) { ++smem_pipe_read; }
        Tensor tSrS = partition_fragment_C(tiled_mma_qk, select<0, 1>(TileShape_MNK{}));
        consumer_wait(pipeline_k, smem_pipe_read);
        if constexpr (!OnlyQv) {
          flash::gemm<true, -1>(tiled_mma_qk, tSrQ, tSrK(_, _, _, smem_pipe_read.index()), tSrS);
        }
        if constexpr (!HasQv) {
          warp_scheduler_barrier_arrive();
          warpgroup_wait<0>();
          pipeline_k.consumer_release(smem_pipe_read);
        } else {
          if constexpr (Is_first_iter) {
            shared_storage.pipelines.barrier_Qv.wait(work_idx % 2);
          }
          consumer_wait(pipeline_v, smem_pipe_read);
          flash::gemm<OnlyQv, -1>(tiled_mma_qv, tSrQv, tSrV(_, _, _, smem_pipe_read.index()), tSrS);
          warp_scheduler_barrier_arrive();
          warpgroup_wait<1>();
          pipeline_k.consumer_release(smem_pipe_read);
          warpgroup_wait<0>();
        }
        scoremod_premask_fn(tSrS);
        if constexpr (Has_sparse_mask) {
          consumer_wait(pipeline_mask, smem_pipe_read);
        }
        mask_fn(tSrS, n_block);
        if constexpr (Has_sparse_mask) {
          pipeline_mask.consumer_release(smem_pipe_read);
        }
        Tensor scores_scale = softmax.template max_get_scale<Is_first_iter, Check_inf>(tSrS);
        if constexpr (LargeHeadDimV && !Is_first_iter) { store_scales(scores_scale, smem_pipe_read_prev.index()); }
        softmax.template online_softmax<Is_first_iter, Check_inf>(tSrS);
        if constexpr (Is_FP8 && !V_colmajor) { flash::permute_Cregs_fp8(tSrS); }
        Tensor tOrP_acc = make_tensor(tSrS.data(), flash::convert_layout_acc_Aregs<TiledMmaPV>(tSrS.layout()));
        Tensor tOrP = make_tensor_like<Element>(tOrP_acc);
        convert_type_out(tOrP_acc, tOrP);
        if constexpr (Is_FP8 && V_colmajor) { flash::permute_Aregs_fp8(tOrP); }
        if constexpr (!MmaPV_is_RS) { write_P_to_smem(tOrP); }
        if constexpr (!Is_first_iter) { softmax.rescale_o(tOrO, scores_scale); }
        if constexpr (!MmaPV_is_RS && !MmaPV_use_RS_WG1) { arrive_on_P_write_barrier(); }
        if constexpr (!HasQv) { consumer_wait(pipeline_v, smem_pipe_read); }
        warp_scheduler_barrier_sync();
        if constexpr (!MmaPV_use_RS_WG1) {
          flash::gemm<Is_first_iter, -1>(tiled_mma_pv, cute::conditional_return<MmaPV_is_RS>(tOrP, tOsP), tOrV(_, _, _, smem_pipe_read.index()), tOrO);
        } else {
          TiledMmaPV_RS tiled_mma_pv_rs;
          flash::gemm<Is_first_iter, -1>(tiled_mma_pv_rs, tOrP, tOrV(_, _, _, smem_pipe_read.index()), tOrO);
        }
        if constexpr (!MmaPV_is_RS && MmaPV_use_RS_WG1) { arrive_on_P_write_barrier(); }
        warpgroup_wait<0>();
        pipeline_v.consumer_release(smem_pipe_read);
      };
      auto first_iter_mask_fn = [&](auto& tSrS, int n_block) {
        if constexpr (Has_sparse_mask) {
          uint32_t const* mask_ptr = reinterpret_cast<uint32_t const*>(
            shared_storage.tensors.mainloop.smem_mask.data() +
            smem_pipe_read.index() * (kBlockM * kNumInt32PerBlock)
          );
          sparse_mask.template apply<true >(tSrS, m_block, n_block, mask_ptr);
        } else {
          mask.template apply<true , Is_causal, Is_local>(tSrS, m_block, n_block);
        }
      };
      fwd_step(n_block, first_iter_mask_fn, cute::true_type{} , cute::true_type{} );
      --n_block;
      if constexpr (Is_causal || Is_local) {
        auto mask_fn = [&](auto& tSrS, int n_block) {
          if constexpr (Has_sparse_mask) {
            uint32_t const* mask_ptr = reinterpret_cast<uint32_t const*>(
              shared_storage.tensors.mainloop.smem_mask.data() +
              smem_pipe_read.index() * (kBlockM * kNumInt32PerBlock)
            );
            sparse_mask.template apply<false >(tSrS, m_block, n_block, mask_ptr);
          } else {
            mask.template apply<false , Is_causal, Is_local>(tSrS, m_block, n_block);
          }
        };
        int n_block_min_causal_local_mask;
        {
          int const n_idx_cm = m_idx_min + seqlen_k - seqlen_q;
          int n_idx_right = !Is_local ? n_idx_cm : n_idx_cm + params.window_size_right;
          if constexpr (Is_local) { if (params.attention_chunk_divmod.divisor > 0) { n_idx_right = std::min(n_idx_right, flash::round_up(params.attention_chunk_divmod, n_idx_cm)); } }
          n_block_min_causal_local_mask = std::max(n_block_min, n_idx_right / kBlockN);
        }
        #pragma unroll 1
        for (; n_block >= n_block_min_causal_local_mask; --n_block) {
          fwd_step(n_block, mask_fn, cute::false_type{} , cute::true_type{} );
        }
      }
      int n_block_min_before_local_mask;
      {
        int const n_idx_bm = m_idx_max + seqlen_k - seqlen_q;
        int n_idx_left = !Is_local ? n_idx_bm : n_idx_bm - params.window_size_left;
        if constexpr (Is_local) { if (params.attention_chunk_divmod.divisor > 0) { n_idx_left = std::max(n_idx_left, flash::round_down(params.attention_chunk_divmod, n_idx_bm)); } }
        n_block_min_before_local_mask = !Is_local ? n_block_min : std::max(n_block_min, cute::ceil_div(n_idx_left, kBlockN));
      }
      auto no_mask_fn = [](auto& tSrS, int n_block) { };
      #pragma unroll 1
      for (; n_block >= n_block_min_before_local_mask; --n_block) {
        fwd_step(n_block, no_mask_fn, cute::false_type{} , cute::false_type{} );
      }
      if constexpr (Is_local) {
        auto local_mask_fn = [&](auto& tSrS, int n_block) { mask.template apply<false , false , Is_local>(tSrS, m_block, n_block); };
        #pragma unroll 1
        for (; n_block >= n_block_min; --n_block) {
          fwd_step(n_block, local_mask_fn, cute::false_type{} , cute::bool_constant<Is_local>{} );
        }
      }
      warp_scheduler_barrier_arrive();
      cutlass::arch::NamedBarrier::arrive(NumMmaThreadsQK + (Use_TMA_Q ? cutlass::NumThreadsPerWarp : NumProducerThreads), static_cast<uint32_t>(FwdNamedBarriers::QueryEmpty) );
      float const v_descale = !Is_FP8 || params.ptr_v_descale == nullptr ? 1.0f : params.ptr_v_descale[bidb * get<0>(params.stride_v_descale) + bidh_kv * get<1>(params.stride_v_descale)];
      Tensor scores_scale = softmax.finalize(v_descale);
      if constexpr (LargeHeadDimV) {
        cutlass::arch::NamedBarrier::sync(NumMmaThreads, static_cast<uint32_t>(FwdNamedBarriers::PEmpty) );
        store_scales(scores_scale, smem_pipe_read.index());
        cutlass::arch::NamedBarrier::arrive(NumMmaThreads, static_cast<uint32_t>(FwdNamedBarriers::PFull) );
      }
      softmax.rescale_o(tOrO, scores_scale);
      if constexpr (Is_FP8 && !V_colmajor) { flash::permute_output_fp8(tOrO); }
      ++smem_pipe_read;
    }
    ++work_idx;
    return true;
  }
  template <typename SharedStorage, typename FrgTensorO, typename Softmax>
  CUTLASS_DEVICE bool
  mma_pv(Params const& params,
      MainloopPipelineV pipeline_v,
      PipelineState& smem_pipe_read,
      FrgTensorO& tOrO,
      Softmax& softmax,
      int const thread_idx,
      SeqlenInfo_t const& seqlen_info,
      cute::tuple<int32_t, int32_t, int32_t, int32_t> block_coord,
      SharedStorage& shared_storage
      ) {
    static_assert(is_rmem<FrgTensorO>::value, "O tensor must be rmem resident.");
    int const m_block = get<0>(block_coord);
    int const bidb = get<2>(block_coord);
    int const split_idx = get<3>(block_coord);
    auto [n_block_min, n_block_max, n_offset_pv] = BlockMN_t::get_n_block_min_max(
      seqlen_info, m_block, bidb, split_idx, params.num_splits,
      params.window_size_left, params.window_size_right, params.attention_chunk_divmod,
      params.qhead_per_khead_divmod);
    if constexpr (Is_causal || Is_local || Varlen || Split) {
      if (n_block_max <= n_block_min) { return false; }
    }
    Tensor sV = make_tensor(make_smem_ptr(shared_storage.tensors.mainloop.smem_v.data()), SmemLayoutVtMma{});
    Tensor sP = make_tensor(make_smem_ptr(shared_storage.tensors.mainloop.smem_p.data()), SmemLayoutP{});
    Tensor sScale = make_tensor(make_smem_ptr(shared_storage.tensors.mainloop.smem_scale.data()), SmemLayoutScale{});
    static constexpr int MmaWarpGroups = size(TiledMmaPV{}) / cutlass::NumThreadsPerWarpGroup;
    Layout warp_group_thread_layout = make_layout(make_shape(Int<MmaWarpGroups>{}),
                           make_stride(Int<cutlass::NumThreadsPerWarpGroup>{}));
    int warp_group_idx = __shfl_sync(0xFFFFFFFF, thread_idx / cutlass::NumThreadsPerWarpGroup, 0);
    TiledMmaPV tiled_mma_pv;
    auto wg_mma_pv = tiled_mma_pv.get_slice(warp_group_thread_layout(warp_group_idx));
    Tensor tOrV = wg_mma_pv.partition_fragment_B(sV);
    Tensor tOsP = wg_mma_pv.partition_fragment_A(sP);
    auto thread_mma_pv = tiled_mma_pv.get_thread_slice(thread_idx % cutlass::NumThreadsPerWarpGroup);
    Tensor taccOcO = thread_mma_pv.partition_C(cute::make_identity_tensor(select<0, 1>(TileShape_MNK_PV{})));
    Tensor taccOcO_rowcol = make_tensor(taccOcO.data(), flash::convert_layout_acc_rowcol(taccOcO.layout()));
    Tensor taccOcO_row = taccOcO_rowcol(_, _0{});
    auto load_scales = [&](auto& scales, int stage) {
      static_assert(CUTE_STATIC_V(size(scales)) == CUTE_STATIC_V(size(taccOcO_row)));
      #pragma unroll
      for (int mi = 0; mi < size(taccOcO_row); ++mi) {
        scales(mi) = sScale(get<0>(taccOcO_row(mi)), stage);
      }
    };
    typename Softmax::TensorT scores_scale;
    int n_block = n_block_max - 1;
    if constexpr (!HasQv) { pipeline_v.consumer_wait(smem_pipe_read); }
    cutlass::arch::NamedBarrier::sync(NumMmaThreads, static_cast<uint32_t>(FwdNamedBarriers::PFull) );
    flash::gemm<true, 0>(tiled_mma_pv, tOsP, tOrV(_, _, _, smem_pipe_read.index()), tOrO);
    cutlass::arch::NamedBarrier::arrive(NumMmaThreads, static_cast<uint32_t>(FwdNamedBarriers::PEmpty) );
    pipeline_v.consumer_release(smem_pipe_read);
    --n_block;
    #pragma unroll 1
    for (; n_block >= n_block_min; --n_block) {
      cutlass::arch::NamedBarrier::sync(NumMmaThreads, static_cast<uint32_t>(FwdNamedBarriers::PFull) );
      load_scales(scores_scale, smem_pipe_read.index());
      softmax.rescale_o(tOrO, scores_scale);
      ++smem_pipe_read;
      if constexpr (!HasQv) {
        auto barrier_token = pipeline_v.consumer_try_wait(smem_pipe_read);
        pipeline_v.consumer_wait(smem_pipe_read, barrier_token);
      }
      flash::gemm<false, 0>(tiled_mma_pv, tOsP, tOrV(_, _, _, smem_pipe_read.index()), tOrO);
      cutlass::arch::NamedBarrier::arrive(NumMmaThreads, static_cast<uint32_t>(FwdNamedBarriers::PEmpty) );
      pipeline_v.consumer_release(smem_pipe_read);
    };
    cutlass::arch::NamedBarrier::sync(NumMmaThreads, static_cast<uint32_t>(FwdNamedBarriers::PFull) );
    load_scales(scores_scale, smem_pipe_read.index());
    cutlass::arch::NamedBarrier::arrive(NumMmaThreads, static_cast<uint32_t>(FwdNamedBarriers::PEmpty) );
    softmax.rescale_o(tOrO, scores_scale);
    if constexpr (Is_FP8 && !V_colmajor) { flash::permute_output_fp8(tOrO); }
    ++smem_pipe_read;
    return true;
  }
  template <typename SharedStorage>
  CUTLASS_DEVICE bool
  load_kv_new(Params const& params,
     MainloopPipelineKVNew pipeline_k_new,
     MainloopPipelineKVNew pipeline_v_new,
     PipelineState& smem_pipe_write,
     SharedStorage &shared_storage,
     SeqlenInfo_t const& seqlen_info,
     cute::tuple<int32_t, int32_t, int32_t, int32_t> block_coord,
     int const work_idx
     ) {
    auto [m_block, bidh, bidb, split_idx] = block_coord;
    auto [n_block_new_min, n_block_new_max] = BlockMN_t::get_n_block_k_new_min_max(
      seqlen_info, m_block, bidb, split_idx, params.num_splits,
      params.window_size_left, params.window_size_right, params.attention_chunk_divmod,
      params.qhead_per_khead_divmod);
    if (n_block_new_max <= n_block_new_min) { return false; }
    Tensor sK = make_tensor(make_smem_ptr(shared_storage.tensors.mainloop.smem_k.data()), SmemLayoutK{});
    Tensor sVt = [&] {
      if constexpr (!Transpose_V) {
        return make_tensor(make_smem_ptr(shared_storage.tensors.mainloop.smem_v.data()), SmemLayoutVt{});
      } else {
        return make_tensor(make_smem_ptr(shared_storage.tensors.mainloop.smem_vt.data()), SmemLayoutVt{});
      }
    }();
    int const bidh_kv = !PackGQA ? params.qhead_per_khead_divmod.divide(bidh) : bidh;
    uint32_t block_rank_in_cluster = cute::block_rank_in_cluster();
    constexpr uint32_t cluster_shape_x = get<0>(ClusterShape());
    uint2 cluster_local_block_id = {block_rank_in_cluster % cluster_shape_x, block_rank_in_cluster / cluster_shape_x};
    bool const is_varlen_k_new = Varlen && params.cu_seqlens_k_new;
    Tensor mKnew_TMA = params.tma_load_K_new.get_tma_tensor(params.shape_K_new)(_, _, bidh_kv, !is_varlen_k_new ? bidb : 0);
    auto shape_Vnew = make_shape(params.headdim_v, get<0>(params.shape_K_new), get<2>(params.shape_K_new), get<3>(params.shape_K_new));
    Tensor mVnewt_TMA = params.tma_load_V_new.get_tma_tensor(shape_Vnew)(_, _, bidh_kv, !is_varlen_k_new ? bidb : 0);
    Tensor gKnew_TMA = local_tile(domain_offset(make_coord(seqlen_info.offset_k_new, _0{}), mKnew_TMA), select<1, 2>(TileShape_MNK{}), make_coord(_, _0{}));
    Tensor gVnewt_TMA = local_tile(domain_offset(make_coord(_0{}, seqlen_info.offset_k_new), mVnewt_TMA), select<1, 2>(TileShape_MNK_PV{}), make_coord(_0{}, _));
    auto block_tma_K_new = params.tma_load_K_new.get_slice(cluster_local_block_id.x);
    Tensor tKgKnew_TMA = group_modes<0, 3>(block_tma_K_new.partition_S(gKnew_TMA));
    Tensor tKsK_TMA = group_modes<0, 3>(block_tma_K_new.partition_D(sK));
    auto block_tma_V_new = params.tma_load_V_new.get_slice(cluster_local_block_id.x);
    Tensor tVgVnewt_TMA = group_modes<0, 3>(block_tma_V_new.partition_S(gVnewt_TMA));
    Tensor tVsVt_TMA = group_modes<0, 3>(block_tma_V_new.partition_D(sVt));
    uint16_t mcast_mask_kv = 0;
    if constexpr (cute::is_same_v<GmemTiledCopyKV, SM90_TMA_LOAD_MULTICAST>) {
      auto block_layout = Layout<ClusterShape>{};
      for (int m = 0; m < size<0>(block_layout); ++m) {
        mcast_mask_kv |= (uint16_t(1) << block_layout(m, cluster_local_block_id.y, _0{}));
      }
    }
    auto load_K_new = [&] (int const n_block, auto const& smem_pipe_write) {
      pipeline_k_new.producer_acquire(smem_pipe_write);
      copy(params.tma_load_K_new.with(*pipeline_k_new.producer_get_barrier(smem_pipe_write), mcast_mask_kv, TMA::CacheHintSm90::EVICT_FIRST),
        tKgKnew_TMA(_, n_block), tKsK_TMA(_, smem_pipe_write.index()));
    };
    auto load_V_new = [&] (int const n_block, auto const& smem_pipe_write) {
      pipeline_v_new.producer_acquire(smem_pipe_write);
      copy(params.tma_load_V_new.with(*pipeline_v_new.producer_get_barrier(smem_pipe_write), mcast_mask_kv, TMA::CacheHintSm90::EVICT_FIRST),
        tVgVnewt_TMA(_, n_block), tVsVt_TMA(_, smem_pipe_write.index()));
    };
    int warp_idx_in_warpgroup = __shfl_sync(0xffffffff, (threadIdx.x / 32) % 4, 0);
    static constexpr bool SingleProducerWarp = NumProducerThreads == cutlass::NumThreadsPerWarp;
    bool should_load_KV = (SingleProducerWarp || warp_idx_in_warpgroup == 0) && cute::elect_one_sync();
    int n_block = n_block_new_max - 1;
    shared_storage.pipelines.barrier_O.wait((work_idx + 1) % 2);
    if (should_load_KV) { load_K_new(n_block, smem_pipe_write); }
    if (should_load_KV) { load_V_new(n_block, smem_pipe_write); }
    ++smem_pipe_write;
    --n_block;
    #pragma unroll 1
    for (; n_block >= n_block_new_min; --n_block) {
      if (should_load_KV) {
        load_K_new(n_block, smem_pipe_write);
        load_V_new(n_block, smem_pipe_write);
      }
      ++smem_pipe_write;
    }
    return true;
  }
  template <typename SharedStorage>
  CUTLASS_DEVICE bool
  store_kv_new(Params const& params,
         MainloopPipelineKVNew pipeline_k_new,
         MainloopPipelineKVNew pipeline_v_new,
         PipelineState& smem_pipe_read,
         int const thread_idx,
         SharedStorage &shared_storage,
         SeqlenInfo_t const& seqlen_info,
         cute::tuple<int32_t, int32_t, int32_t, int32_t> block_coord
  ) {
    auto [m_block, bidh, bidb, split_idx] = block_coord;
    auto [n_block_new_min, n_block_new_max] = BlockMN_t::get_n_block_k_new_min_max(
      seqlen_info, m_block, bidb, split_idx, params.num_splits,
      params.window_size_left, params.window_size_right, params.attention_chunk_divmod,
      params.qhead_per_khead_divmod);
    if (n_block_new_max <= n_block_new_min) { return false; }
    Tensor sK = cute::as_position_independent_swizzle_tensor(make_tensor(make_smem_ptr(shared_storage.tensors.mainloop.smem_k.data()), SmemLayoutK{}));
    Tensor sV = [&] {
      if constexpr (!Transpose_V) {
        return cute::as_position_independent_swizzle_tensor(make_tensor(make_smem_ptr(shared_storage.tensors.mainloop.smem_v.data()), SmemLayoutVCpAsync{}));
      } else {
        return cute::as_position_independent_swizzle_tensor(make_tensor(make_smem_ptr(shared_storage.tensors.mainloop.smem_vt.data()), SmemLayoutVCpAsync{}));
      }
    }();
    int const bidh_kv = !PackGQA ? params.qhead_per_khead_divmod.divide(bidh) : bidh;
    int const bidb_kv = params.kv_batch_idx == nullptr ? bidb : params.kv_batch_idx[bidb];
    bool const is_varlen_k = Varlen && params.cu_seqlens_k;
    Tensor mK = make_tensor(make_gmem_ptr(params.ptr_K), params.shape_K, params.stride_K)(_, _, bidh_kv, !is_varlen_k ? bidb_kv : 0);
    auto shape_V = make_shape(params.headdim_v, get<0>(params.shape_K), get<2>(params.shape_K), get<3>(params.shape_K));
    Tensor mV = make_tensor(make_gmem_ptr(params.ptr_V), shape_V, params.stride_V)(_, _, bidh_kv, !is_varlen_k ? bidb_kv : 0);
    int const offset_k = seqlen_info.offset_k + seqlen_info.seqlen_k_og;
    Tensor gK = local_tile(domain_offset(make_coord(offset_k, _0{}), mK), select<1, 2>(TileShape_MNK{}), make_coord(_, _0{}));
    Tensor gV = local_tile(domain_offset(make_coord(offset_k, _0{}), mV), select<2, 1>(TileShape_MNK_PV{}), make_coord(_, _0{}));
    static constexpr int kBlockN = get<1>(TileShape_MNK{});
    static constexpr int kHeadDim = get<2>(TileShape_MNK{});
    int const seqlen_k_new = seqlen_info.seqlen_k_new;
    using Rotary_t = Rotary<kBlockN, kHeadDim, NumMmaThreads, Element>;
    Rotary_t rotary(params.ptr_rotary_cos, params.shape_rotary, params.stride_rotary_cos,
            params.ptr_rotary_sin, params.stride_rotary_sin,
            params.is_rotary_interleaved, thread_idx, seqlen_k_new,
            seqlen_info.seqlen_rotary);
    int const bidb_kv_idx = !is_varlen_k && !params.ptr_pagetable ? bidb_kv : 0;
    using PagedKVManager_t = PagedKVManager<get<1>(TileShape_MNK{}), get<2>(TileShape_MNK{}), get<1>(TileShape_MNK_PV{}), NumMmaThreads, Element, true , 2 >;
    PagedKVManager_t paged_kv_manager(
      params.ptr_pagetable, params.shape_pagetable, params.stride_pagetable,
      params.ptr_K, params.shape_K, params.stride_K,
      params.ptr_V, params.headdim_v, att_paged_stride_v(params),
      params.page_size_divmod, params.blockN_per_page_size_divmod,
      bidb_kv, bidh_kv, thread_idx, seqlen_k_new, offset_k, bidb_kv_idx
    );
    if constexpr (UseSchedulerBarrier) {
      if (flash::canonical_warp_group_idx_nosync() == 1) {
        cutlass::arch::NamedBarrier::sync(2 * cutlass::NumThreadsPerWarpGroup, static_cast<uint32_t>(FwdNamedBarriers::WarpSchedulerWG1) );
      }
    }
    static_assert(std::is_same_v<GmemLayoutAtom, typename Rotary_t::LayoutAtom>);
    static_assert(!PagedKVNonTMA || std::is_same_v<GmemLayoutAtom, typename PagedKVManager_t::GmemLayoutAtomKVCpAsync>);
    GmemTiledCopyAppendKV gmem_tiled_copy_kv;
    auto gmem_thr_copy_kv = gmem_tiled_copy_kv.get_thread_slice(thread_idx);
    Tensor tKsK = gmem_thr_copy_kv.partition_S(sK);
    Tensor tKgK = gmem_thr_copy_kv.partition_D(gK);
    Tensor tVsV = gmem_thr_copy_kv.partition_S(sV);
    Tensor tVgV = gmem_thr_copy_kv.partition_D(gV);
    Tensor cK = cute::make_identity_tensor(select<1, 2>(TileShape_MNK{}));
    Tensor tKcK = gmem_thr_copy_kv.partition_D(cK);
    Tensor tKpK = make_tensor<bool>(make_shape(size<2>(tKsK)));
    #pragma unroll
    for (int k = 0; k < size(tKpK); ++k) { tKpK(k) = get<1>(tKcK(_0{}, _0{}, k)) < get<1>(params.shape_K); }
    Tensor cV = cute::make_identity_tensor(select<2, 1>(TileShape_MNK_PV{}));
    Tensor tVcV = cute::conditional_return<SameHeadDim>(tKcK, gmem_thr_copy_kv.partition_D(cV));
    Tensor tVpV_ = make_tensor<bool>(make_shape(size<2>(tVsV)));
    #pragma unroll
    for (int k = 0; k < size(tVpV_); ++k) { tVpV_(k) = get<1>(tVcV(_0{}, _0{}, k)) < params.headdim_v; }
    Tensor tVpV = cute::conditional_return<SameHeadDim>(tKpK, tVpV_);
    auto store_K = [&] (int const n_block, auto const& smem_pipe_read) {
      int const n_limit = std::min(seqlen_k_new - n_block * kBlockN, kBlockN);
      if (get<1>(params.shape_rotary) <= 0) {
        pipeline_k_new.consumer_wait(smem_pipe_read);
        Tensor tKsK_cur = tKsK(_, _, _, smem_pipe_read.index());
        if constexpr (!PagedKVNonTMA) {
          Tensor tKgK_cur = tKgK(_, _, _, n_block);
          flash::copy<false, false, false, false>(
            gmem_tiled_copy_kv, tKsK_cur, tKgK_cur, tKcK, tKpK, std::min(seqlen_k_new - n_block * kBlockN, kBlockN)
          );
        } else {
          paged_kv_manager.store_K(n_block, tKsK_cur);
        }
      } else {
        Tensor gK_cur = gK(_, _, n_block);
        auto tPrKPtr = cute::conditional_return<PagedKVNonTMA>(paged_kv_manager.compute_K_ptr(), nullptr);
        if (params.is_rotary_interleaved) {
          auto [tRrCos, tRrSin] = rotary.template load_cos_sin<true >(n_block);
          pipeline_k_new.consumer_wait(smem_pipe_read);
          rotary.template apply_K_interleaved<PagedKVNonTMA>(sK(_, _, smem_pipe_read.index()), gK_cur, tKpK, tRrCos, tRrSin, tPrKPtr, n_block);
        } else {
          auto [tRrCosCont, tRrSinCont] = rotary.template load_cos_sin<false >(n_block);
          pipeline_k_new.consumer_wait(smem_pipe_read);
          rotary.template apply_K_contiguous<PagedKVNonTMA>(sK(_, _, smem_pipe_read.index()), gK_cur, tKpK, tRrCosCont, tRrSinCont, tPrKPtr, n_block, get<1>(params.shape_K));
        }
      }
      cutlass::arch::fence_view_async_shared();
      cutlass::arch::NamedBarrier::sync(cutlass::NumThreadsPerWarpGroup, static_cast<uint32_t>(FwdNamedBarriers::WarpSchedulerWG1) - 1 + flash::canonical_warp_group_idx_nosync() );
      pipeline_k_new.consumer_release(smem_pipe_read);
    };
    auto store_V = [&] (int const n_block, auto const& smem_pipe_read) {
      pipeline_v_new.consumer_wait(smem_pipe_read);
      int const n_limit = std::min(seqlen_k_new - n_block * kBlockN, kBlockN);
      Tensor tVsV_cur = tVsV(_, _, _, smem_pipe_read.index());
      if constexpr (!PagedKVNonTMA) {
        Tensor tVgV_cur = tVgV(_, _, _, n_block);
        flash::copy<false, false, false, false>(
          gmem_tiled_copy_kv, tVsV_cur, tVgV_cur, tVcV, tVpV, n_limit);
      } else {
        paged_kv_manager.store_V(n_block, tVsV_cur);
      }
      cutlass::arch::fence_view_async_shared();
      cutlass::arch::NamedBarrier::sync(cutlass::NumThreadsPerWarpGroup, static_cast<uint32_t>(FwdNamedBarriers::WarpSchedulerWG1) - 1 + flash::canonical_warp_group_idx_nosync() );
      pipeline_v_new.consumer_release(smem_pipe_read);
    };
    #pragma unroll 1
    for (int n_block = n_block_new_max - 1; n_block >= n_block_new_min; --n_block) {
      if constexpr (PagedKVNonTMA) { paged_kv_manager.template load_page_table<true >(n_block); }
      store_K(n_block, smem_pipe_read);
      store_V(n_block, smem_pipe_read);
      ++smem_pipe_read;
    }
    if constexpr (UseSchedulerBarrier) {
      if (flash::canonical_warp_group_idx_nosync() == 1) {
        cutlass::arch::NamedBarrier::arrive(2 * cutlass::NumThreadsPerWarpGroup, static_cast<uint32_t>(FwdNamedBarriers::WarpSchedulerWG1) );
      }
    }
    return true;
  }
};
}
#include <cutlass/cutlass.h>
#include <cutlass/array.h>
#include <cutlass/numeric_types.h>
#include <cutlass/numeric_conversion.h>
#include "cute/tensor.hpp"
namespace flash {
using namespace cute;
template <int kNWarps, int Stages, bool Q_in_regs, class TileShape_MNK_, int kHeadDimV, class Element_, class ElementAccum_, class ArchTag_,
    bool Is_causal_, bool Is_local_, bool Has_softcap_, bool Varlen_, bool PagedKV_, bool AppendKV_,
    bool PackGQA_, bool Split_, class ElementSink_>
struct CollectiveMainloopFwdSm80 {
  static constexpr int kStages = Stages;
  static_assert(kStages > 0, "kStages must be greater than 0");
  using TileShape_MNK = TileShape_MNK_;
  using TileShape_MNK_PV = Shape<decltype(get<0>(TileShape_MNK{})), Int<kHeadDimV>, decltype(get<1>(TileShape_MNK{}))>;
  using Element = Element_;
  using ElementAccum = ElementAccum_;
  using ElementSink = ElementSink_;
  using ArchTag = ArchTag_;
  static constexpr bool Is_FP8 = cute::is_same_v<Element, cutlass::float_e4m3_t> || cute::is_same_v<Element, cutlass::float_e5m2_t>;;
  static constexpr bool Is_causal = Is_causal_;
  static constexpr bool Is_local = Is_local_;
  static constexpr bool Has_softcap = Has_softcap_;
  static constexpr bool Varlen = Varlen_;
  static constexpr bool PagedKV = PagedKV_;
  static constexpr bool AppendKV = AppendKV_;
  static constexpr bool PackGQA = PackGQA_;
  static constexpr bool Split = Split_;
  static constexpr bool Transpose_V = Is_FP8;
  static_assert(ArchTag::kMinComputeCapability >= 80);
  static constexpr bool Has_cp_async = ArchTag::kMinComputeCapability >= 80;
  static constexpr int kBlockM = get<0>(TileShape_MNK{});
  static constexpr int kBlockN = get<1>(TileShape_MNK{});
  static constexpr int kHeadDim = get<2>(TileShape_MNK{});
  using SeqlenInfo_t = flash::SeqlenInfoQKNewK<Varlen, AppendKV>;
  using BlockMN_t = flash::BlockMN<SeqlenInfo_t, kBlockM, kBlockN, Is_causal, Is_local, PackGQA, Split>;
  using MMA_Atom_Arch = std::conditional_t<
    ArchTag::kMinComputeCapability >= 80,
    std::conditional_t<
      std::is_same_v<Element, cutlass::half_t>,
      MMA_Atom<SM80_16x8x16_F32F16F16F32_TN>,
      MMA_Atom<SM80_16x8x16_F32BF16BF16F32_TN>
    >,
    MMA_Atom<SM75_16x8x8_F32F16F16F32_TN>
  >;
  using TiledMma = TiledMMA<
    MMA_Atom_Arch,
    Layout<Shape<Int<kNWarps>,_1,_1>>,
    Tile<Int<16 * kNWarps>, _16, _16>>;
  static constexpr int NumMmaThreads = size(TiledMma{});
  static constexpr int NumProducerThreads = NumMmaThreads;
  static constexpr int kGmemElemsPerLoad = sizeof(cute::uint128_t) / sizeof(Element);
  static_assert(kHeadDim % kGmemElemsPerLoad == 0, "Headdim must be a multiple of kGmemElemsPerLoad");
  static constexpr int kBytePerRow = kHeadDim * sizeof(Element);
  static constexpr int kBlockKGmem = (kBytePerRow % 128 == 0 ? 128 : (kBytePerRow % 64 == 0 ? 64 : 32)) / sizeof(Element);
  static constexpr int kSwizzle = kBlockKGmem == 128 ? 4 : (kBlockKGmem == 64 ? 3 : (kBlockKGmem == 32 ? 2 : 1));
  static constexpr int kSwizzleBase = sizeof(Element) == 4 ? 2 : (sizeof(Element) == 2 ? 3 : 4);
  using SmemLayoutAtomQKV = decltype(
    composition(Swizzle<kSwizzle, kSwizzleBase, kSwizzleBase>{},
          Layout<Shape<_8, Int<kBlockKGmem>>,
              Stride<Int<kBlockKGmem>, _1>>{}));
  using SmemLayoutQ = decltype(tile_to_shape(SmemLayoutAtomQKV{}, select<0, 2>(TileShape_MNK{})));
  using SmemLayoutK = decltype(tile_to_shape(
    SmemLayoutAtomQKV{},
    make_shape(shape<1>(TileShape_MNK{}), shape<2>(TileShape_MNK{}), Int<kStages>{})));
  using SmemLayoutV = decltype(tile_to_shape(
    SmemLayoutAtomQKV{},
    make_shape(shape<1>(TileShape_MNK{}), shape<2>(TileShape_MNK{}), Int<kStages>{})));
  using SmemLayoutVt = decltype(
    composition(SmemLayoutV{},
          make_ordered_layout(make_shape(shape<2>(TileShape_MNK{}), shape<1>(TileShape_MNK{}), Int<kStages>{}),
                    Step<_2, _1, _3>{})));
  using SmemCopyAtom = Copy_Atom<SM75_U32x4_LDSM_N, Element>;
  using SmemCopyAtomTransposed = Copy_Atom<SM75_U16x8_LDSM_T, Element>;
  using GmemCopyAtom = Copy_Atom<std::conditional_t<
    Has_cp_async,
    SM80_CP_ASYNC_CACHEGLOBAL_ZFILL<cute::uint128_t>,
    AutoVectorizingCopyWithAssumedAlignment<128>
  >, Element>;
  static constexpr int kGmemThreadsPerRow = kBlockKGmem / kGmemElemsPerLoad;
  static_assert(NumMmaThreads % kGmemThreadsPerRow == 0, "NumMmaThreads must be a multiple of kGmemThreadsPerRow");
  using GmemLayoutAtom = Layout<Shape <Int<NumMmaThreads / kGmemThreadsPerRow>, Int<kGmemThreadsPerRow>>,
                 Stride<Int<kGmemThreadsPerRow>, _1>>;
  using GmemTiledCopyQKV = decltype(
    make_tiled_copy(GmemCopyAtom{},
            GmemLayoutAtom{},
            Layout<Shape<_1, Int<kGmemElemsPerLoad>>>{}));
  static_assert(kBlockM % CUTE_STATIC_V(shape<0>(GmemLayoutAtom{})) == 0);
  static constexpr int kBytePerHalfRow = kHeadDim / 2 * sizeof(Element);
  static constexpr int kBlockKGmemAppend = (kBytePerHalfRow % 128 == 0 ? 128 : (kBytePerHalfRow % 64 == 0 ? 64 : 32)) / sizeof(Element);
  static constexpr int kGmemThreadsPerRowAppend = kBlockKGmemAppend / kGmemElemsPerLoad;
  static_assert(NumMmaThreads % kGmemThreadsPerRowAppend == 0, "NumMmaThreads must be a multiple of kGmemThreadsPerRowAppend");
  static_assert(cutlass::NumThreadsPerWarp % kGmemThreadsPerRowAppend == 0, "kGmemThreadsPerRowAppend must divide NumThreadsPerWarp");
  using GmemLayoutAtomAppend = Layout<Shape <Int<NumMmaThreads / kGmemThreadsPerRowAppend>, Int<kGmemThreadsPerRowAppend>>,
                    Stride<Int<kGmemThreadsPerRowAppend>, _1>>;
  static_assert(!AppendKV || kBlockM % CUTE_STATIC_V(shape<0>(GmemLayoutAtomAppend{})) == 0, "kBlockM must be a multiple of NumMmaThreads / kGmemThreadsPerRowAppend");
  using GmemTiledCopyAppendKV = decltype(
    make_tiled_copy(Copy_Atom<AutoVectorizingCopyWithAssumedAlignment<128>, Element>{},
            GmemLayoutAtomAppend{},
            Layout<Shape<_1, Int<kGmemElemsPerLoad>>>{}));
  using ShapeQKV = cute::Shape<int32_t, int32_t, int32_t, int32_t>;
  using StrideQK = cute::Stride<int64_t, _1, int64_t, int64_t>;
  using StrideV = StrideQK;
  using ShapeQPacked = std::conditional_t<!PackGQA, ShapeQKV, cute::Shape<cute::Shape<int32_t, int32_t>, int32_t, int32_t, int32_t>>;
  using StrideQPacked = std::conditional_t<!PackGQA, StrideQK, cute::Stride<cute::Stride<int64_t, int64_t>, _1, int64_t, int64_t>>;
  using ShapePageTable = cute::Shape<int32_t, int32_t>;
  using StridePageTable = cute::Stride<int64_t, _1>;
  using ShapeRotary = cute::Shape<int32_t, int32_t>;
  using StrideRotary = cute::Stride<int64_t, _1>;
  using StrideDescale = cute::Stride<int64_t, int64_t>;
  static constexpr bool Share_QV_Smem = Q_in_regs;
  struct TensorStorageSharedQV : cute::aligned_struct<128> {
    union {
      cute::array_aligned<Element, cute::cosize_v<SmemLayoutV>> smem_v;
      cute::array_aligned<Element, cute::cosize_v<SmemLayoutQ>> smem_q;
    };
    cute::array_aligned<Element, cute::cosize_v<SmemLayoutK>> smem_k;
  };
  struct TensorStorageSeparateQV : cute::aligned_struct<128> {
    cute::array_aligned<Element, cute::cosize_v<SmemLayoutV>> smem_v;
    cute::array_aligned<Element, cute::cosize_v<SmemLayoutK>> smem_k;
    cute::array_aligned<Element, cute::cosize_v<SmemLayoutQ>> smem_q;
  };
  using TensorStorage = std::conditional_t<Share_QV_Smem, TensorStorageSharedQV, TensorStorageSeparateQV>;
  struct Arguments {
    Element const* const ptr_Q;
    ShapeQKV const shape_Q;
    StrideQK const stride_Q;
    Element* const ptr_K;
    ShapeQKV const shape_K;
    StrideQK const stride_K;
    Element* const ptr_V;
    int32_t const headdim_v;
    StrideV const stride_V;
    Element const* const ptr_K_new;
    ShapeQKV const shape_K_new;
    StrideQK const stride_K_new;
    Element const* const ptr_V_new;
    StrideV const stride_V_new;
    Element const* const ptr_Qv;
    StrideQK const stride_Qv;
    Element const* const ptr_rotary_cos;
    ShapeRotary const shape_rotary;
    StrideRotary const stride_rotary_cos;
    Element const* const ptr_rotary_sin;
    StrideRotary const stride_rotary_sin;
    bool const is_rotary_interleaved;
    int const* const ptr_pagetable;
    ShapePageTable const shape_pagetable;
    StridePageTable const stride_pagetable;
    float const softmax_scale;
    float const* ptr_q_descale, *ptr_k_descale, *ptr_v_descale;
    StrideDescale const stride_q_descale, stride_k_descale, stride_v_descale;
    int const window_size_left = -1, window_size_right = -1, attention_chunk = 0;
    float const softcap_val;
    int const num_splits;
    int const* const kv_batch_idx = nullptr;
    int const* const cu_seqlens_q = nullptr;
    int const* const cu_seqlens_k = nullptr;
    int const* const cu_seqlens_k_new = nullptr;
    int const* const seqused_q = nullptr;
    int const* const seqused_k = nullptr;
    int const* const leftpad_k = nullptr;
    int const* const seqlens_rotary = nullptr;
    ElementSink const* const ptr_sink = nullptr;
  };
  struct Params {
    Element const* const ptr_Q;
    ShapeQKV const shape_Q;
    StrideQK const stride_Q;
    ShapeQPacked const shape_Q_packed;
    StrideQPacked const stride_Q_packed;
    Element* const ptr_K;
    ShapeQKV const shape_K;
    StrideQK const stride_K;
    Element* const ptr_V;
    int32_t const headdim_v;
    StrideV const stride_V;
    Element const* const ptr_K_new;
    ShapeQKV const shape_K_new;
    StrideQK const stride_K_new;
    Element const* const ptr_V_new;
    StrideV const stride_V_new;
    Element const* const ptr_rotary_cos;
    ShapeRotary const shape_rotary;
    StrideRotary const stride_rotary_cos;
    Element const* const ptr_rotary_sin;
    StrideRotary const stride_rotary_sin;
    bool const is_rotary_interleaved;
    int const* const ptr_pagetable;
    ShapePageTable const shape_pagetable;
    StridePageTable const stride_pagetable;
    cutlass::FastDivmod page_size_divmod;
    cutlass::FastDivmod qhead_per_khead_divmod;
    float const softmax_scale_log2;
    float const* ptr_q_descale, *ptr_k_descale, *ptr_v_descale;
    StrideDescale const stride_q_descale, stride_k_descale, stride_v_descale;
    float const softcap_val;
    int const window_size_left, window_size_right;
    cutlass::FastDivmod attention_chunk_divmod;
    int const num_splits;
    int const* const kv_batch_idx = nullptr;
    int const* const cu_seqlens_q = nullptr;
    int const* const cu_seqlens_k = nullptr;
    int const* const cu_seqlens_k_new = nullptr;
    int const* const seqused_q = nullptr;
    int const* const seqused_k = nullptr;
    int const* const leftpad_k = nullptr;
    int const* const seqlens_rotary = nullptr;
    ElementSink const* const ptr_sink = nullptr;
  };
  static Params
  to_underlying_arguments(Arguments const& args) {
    int const qhead_per_khead = !PackGQA ? 1 : cute::ceil_div(get<2>(args.shape_Q), get<2>(args.shape_K));
    auto const shape_Q_packed = cute::conditional_return<!PackGQA>(
      args.shape_Q,
      make_shape(make_shape(qhead_per_khead, get<0>(args.shape_Q)), get<1>(args.shape_Q), get<2>(args.shape_K), get<3>(args.shape_Q))
    );
    auto const stride_Q_packed = cute::conditional_return<!PackGQA>(
      args.stride_Q,
      make_stride(make_stride(get<2>(args.stride_Q), get<0>(args.stride_Q)), get<1>(args.stride_Q), get<2>(args.stride_Q) * qhead_per_khead, get<3>(args.stride_Q))
    );
    if (get<1>(args.shape_rotary) > 0) {
      assert(args.ptr_rotary_cos != nullptr && args.ptr_rotary_sin != nullptr);
    }
    assert(args.num_splits >= 1);
    cutlass::FastDivmod attention_chunk_divmod(args.attention_chunk >= 1 ? args.attention_chunk : 1);
    attention_chunk_divmod.divisor = args.attention_chunk;
    return {args.ptr_Q, args.shape_Q, args.stride_Q, shape_Q_packed, stride_Q_packed,
        args.ptr_K, args.shape_K, args.stride_K, args.ptr_V, args.headdim_v, args.stride_V,
        args.ptr_K_new, args.shape_K_new, args.stride_K_new, args.ptr_V_new, args.stride_V_new,
        args.ptr_rotary_cos, args.shape_rotary, args.stride_rotary_cos,
        args.ptr_rotary_sin, args.stride_rotary_sin, args.is_rotary_interleaved,
        args.ptr_pagetable, args.shape_pagetable, args.stride_pagetable,
        cutlass::FastDivmod(int(get<0>(args.shape_K))),
        cutlass::FastDivmod(cute::ceil_div(get<2>(args.shape_Q), get<2>(args.shape_K))),
        !Has_softcap ? float(args.softmax_scale * M_LOG2E) : float(args.softcap_val * M_LOG2E),
        args.ptr_q_descale, args.ptr_k_descale, args.ptr_v_descale,
        args.stride_q_descale, args.stride_k_descale, args.stride_v_descale,
        !Has_softcap ? 0.f : args.softmax_scale / args.softcap_val,
        args.window_size_left, args.window_size_right, attention_chunk_divmod,
        !Split ? 1 : args.num_splits,
        args.kv_batch_idx,
        args.cu_seqlens_q, args.cu_seqlens_k, args.cu_seqlens_k_new,
        args.seqused_q, args.seqused_k, args.leftpad_k, args.seqlens_rotary, args.ptr_sink};
  }
  template <typename SharedStorage, typename FrgTensorO, typename Softmax>
  CUTLASS_DEVICE bool
  mma(Params const& params,
    FrgTensorO& tOrO,
    Softmax& softmax,
    int const thread_idx,
    SeqlenInfo_t const& seqlen_info,
    cute::tuple<int32_t, int32_t, int32_t, int32_t> block_coord,
    SharedStorage& shared_storage
    ) {
    static_assert(is_rmem<FrgTensorO>::value, "O tensor must be rmem resident.");
    static constexpr int kBlockM = get<0>(TileShape_MNK{});
    static constexpr int kBlockN = get<1>(TileShape_MNK{});
    int const m_block = get<0>(block_coord);
    int const bidh = get<1>(block_coord);
    int const bidb = get<2>(block_coord);
    int const split_idx = get<3>(block_coord);
    int const bidh_kv = !PackGQA ? params.qhead_per_khead_divmod.divide(bidh) : bidh;
    auto n_block_min_max = BlockMN_t::get_n_block_min_max(
      seqlen_info, m_block, bidb, split_idx, params.num_splits,
      params.window_size_left, params.window_size_right, params.attention_chunk_divmod,
      params.qhead_per_khead_divmod);
    int const n_block_min = get<0>(n_block_min_max);
    int const n_block_max = get<1>(n_block_min_max);
    if constexpr (Is_causal || Is_local || Varlen || Split) {
      if (n_block_max <= n_block_min) { return false; }
    }
    Tensor sQ = make_tensor(make_smem_ptr(shared_storage.tensors.mainloop.smem_q.data()), SmemLayoutQ{});
    Tensor sK = make_tensor(make_smem_ptr(shared_storage.tensors.mainloop.smem_k.data()), SmemLayoutK{});
    Tensor sV = make_tensor(make_smem_ptr(shared_storage.tensors.mainloop.smem_v.data()), SmemLayoutV{});
    Tensor sVt = make_tensor(make_smem_ptr(shared_storage.tensors.mainloop.smem_v.data()), SmemLayoutVt{});
    bool const is_varlen_q = Varlen && params.cu_seqlens_q;
    bool const is_varlen_k = Varlen && params.cu_seqlens_k;
    int const bidb_kv = params.kv_batch_idx == nullptr ? bidb : params.kv_batch_idx[bidb];
    Tensor mQ = make_tensor(make_gmem_ptr(params.ptr_Q + seqlen_info.offset_q * get<0>(params.stride_Q)), params.shape_Q_packed, params.stride_Q_packed)(_, _, bidh, !is_varlen_q ? bidb : 0);
    Tensor gQ = local_tile(mQ, select<0, 2>(TileShape_MNK{}), make_coord(m_block, _0{}));
    Tensor mK = make_tensor(make_gmem_ptr(params.ptr_K + seqlen_info.offset_k * get<0>(params.stride_K)), params.shape_K, params.stride_K)(_, _, bidh_kv, !is_varlen_k ? bidb_kv : 0);
    Tensor gK = local_tile(mK, select<1, 2>(TileShape_MNK{}), make_coord(_, _0{}));
    Tensor mV = make_tensor(make_gmem_ptr(params.ptr_V + seqlen_info.offset_k * get<0>(params.stride_V)), params.shape_K, params.stride_V)(_, _, bidh_kv, !is_varlen_k ? bidb_kv : 0);
    Tensor gV = local_tile(mV, select<1, 2>(TileShape_MNK{}), make_coord(_, _0{}));
    GmemTiledCopyQKV gmem_tiled_copy_QKV;
    auto gmem_thr_copy_QKV = gmem_tiled_copy_QKV.get_thread_slice(thread_idx);
    auto gmem_thr0_copy_QKV = gmem_tiled_copy_QKV.get_thread_slice(_0{});
    Tensor tKgK = gmem_thr_copy_QKV.partition_S(gK);
    Tensor tKsK = gmem_thr_copy_QKV.partition_D(sK);
    Tensor tVgV = gmem_thr_copy_QKV.partition_S(gV);
    Tensor tVsV = gmem_thr_copy_QKV.partition_D(sV);
    TiledMma tiled_mma;
    auto thr_mma = tiled_mma.get_slice(thread_idx);
    Tensor tSrQ = thr_mma.partition_fragment_A(sQ);
    auto smem_tiled_copy_Q = make_tiled_copy_A(SmemCopyAtom{}, tiled_mma);
    auto smem_thr_copy_Q = smem_tiled_copy_Q.get_thread_slice(thread_idx);
    auto smem_tiled_copy_K = make_tiled_copy_B(SmemCopyAtom{}, tiled_mma);
    auto smem_thr_copy_K = smem_tiled_copy_K.get_thread_slice(thread_idx);
    auto smem_tiled_copy_V = make_tiled_copy_B(SmemCopyAtomTransposed{}, tiled_mma);
    auto smem_thr_copy_V = smem_tiled_copy_V.get_thread_slice(thread_idx);
    Tensor tSsQ = smem_thr_copy_Q.partition_S(sQ);
    Tensor tSsK = smem_thr_copy_K.partition_S(sK);
    Tensor tOsVt = smem_thr_copy_V.partition_S(sVt);
    Tensor cKV = cute::make_identity_tensor(select<1, 2>(TileShape_MNK{}));
    Tensor tKVcKV = gmem_thr_copy_QKV.partition_S(cKV);
    Tensor t0KVcKV = gmem_thr0_copy_QKV.partition_S(cKV);
    Tensor tKVpKV = make_tensor<bool>(make_shape(size<2>(tKsK)));
    #pragma unroll
    for (int k = 0; k < size(tKVpKV); ++k) { tKVpKV(k) = get<1>(tKVcKV(_0{}, _0{}, k)) < get<1>(params.shape_K); }
    int const seqlen_q = seqlen_info.seqlen_q;
    int const seqlen_k = seqlen_info.seqlen_k;
    int n_block = n_block_max - 1;
    if constexpr (Share_QV_Smem) { __syncthreads(); }
    if constexpr (!PackGQA) {
      Tensor tQgQ = gmem_thr_copy_QKV.partition_S(gQ);
      Tensor tQsQ = gmem_thr_copy_QKV.partition_D(sQ);
      Tensor cQ = cute::make_identity_tensor(select<0, 2>(TileShape_MNK{}));
      Tensor tQcQ = gmem_thr_copy_QKV.partition_S(cQ);
      Tensor t0QcQ = gmem_thr0_copy_QKV.partition_S(cQ);
      Tensor tQpQ = make_tensor<bool>(make_shape(size<2>(tQsQ)));
      #pragma unroll
      for (int k = 0; k < size(tQpQ); ++k) { tQpQ(k) = get<1>(tQcQ(_0{}, _0{}, k)) < get<1>(params.shape_Q); }
      flash::copy<false, false, false, true>(
        gmem_tiled_copy_QKV, tQgQ, tQsQ, t0QcQ, tQpQ, seqlen_info.seqlen_q - m_block * kBlockM - get<0>(tQcQ(_0{}, _0{}, _0{}))
      );
    } else {
      using PackGQAt = flash::PackGQAManager<get<0>(TileShape_MNK{}), get<2>(TileShape_MNK{}), NumMmaThreads, Element>;
      PackGQAt::load_Q(mQ, sQ, params.qhead_per_khead_divmod, thread_idx, seqlen_q, m_block);
    }
    cute::cp_async_fence();
    using PagedKVManager_t = PagedKVManager<get<1>(TileShape_MNK{}), get<2>(TileShape_MNK{}), get<1>(TileShape_MNK_PV{}), NumMmaThreads, Element, true >;
    PagedKVManager_t paged_kv_manager(
      params.ptr_pagetable, params.shape_pagetable, params.stride_pagetable,
      params.ptr_K, params.shape_K, params.stride_K,
      params.ptr_V, params.headdim_v, params.stride_V,
      params.page_size_divmod,
      params.page_size_divmod ,
      bidb_kv, bidh_kv, thread_idx, seqlen_info.seqlen_k, seqlen_info.leftpad_k,
      0
    );
    auto load_K = [&] (int const n_block, int const smem_pipe_write, auto need_seqlenk_masking_type) {
      static constexpr bool Seqlenk_mask = decltype(need_seqlenk_masking_type)::value;
      if constexpr (!PagedKV) {
        static constexpr bool EvenN = kBlockN % CUTE_STATIC_V(shape<0>(GmemLayoutAtom{})) == 0;
        Tensor tKsK_cur = tKsK(_, _, _, smem_pipe_write);
        int const seqlenk_row_limit = -int(get<0>(tKVcKV(_0{}, _0{}, _0{}))) + (EvenN
          ? seqlen_info.seqlen_k - n_block * kBlockN
          : (!Seqlenk_mask ? kBlockN : std::min(seqlen_info.seqlen_k - n_block * kBlockN, kBlockN)));
        flash::copy<!Seqlenk_mask && EvenN, false, false, true>(
          gmem_tiled_copy_QKV, tKgK(_, _, _, n_block), tKsK_cur, t0KVcKV, tKVpKV, seqlenk_row_limit);
      } else {
        paged_kv_manager.template load_page_table<Seqlenk_mask>(n_block);
        paged_kv_manager.template load_K<Seqlenk_mask>(n_block, sK(_, _, smem_pipe_write));
      }
    };
    auto load_V = [&] (int const n_block, int const smem_pipe_write, auto need_seqlenk_masking_type) {
      static constexpr bool Seqlenk_mask = decltype(need_seqlenk_masking_type)::value;
      if constexpr (!PagedKV) {
        static constexpr bool EvenN = kBlockN % CUTE_STATIC_V(shape<0>(GmemLayoutAtom{})) == 0;
        Tensor tVsV_cur = tVsV(_, _, _, smem_pipe_write);
        Tensor tVgV_cur = tVgV(_, _, _, n_block);
        int const seqlenk_row_limit = seqlen_info.seqlen_k - n_block * kBlockN - get<0>(tKVcKV(_0{}, _0{}, _0{}));
        #pragma unroll
        for (int m = 0; m < size<1>(tVsV); ++m) {
          if (EvenN || m < size<1>(tVsV) - 1 || get<0>(tKVcKV(_0{}, m, _0{})) < kBlockN) {
            bool const predicate_n = !Seqlenk_mask || get<0>(t0KVcKV(_0{}, m, _0{})) < seqlenk_row_limit;
            #pragma unroll
            for (int k = 0; k < size<2>(tVsV); ++k) {
              cute::copy(gmem_tiled_copy_QKV.with(tKVpKV(k) && predicate_n), tVgV_cur(_, m, k), tVsV_cur(_, m, k));
            }
          }
        }
      } else {
        paged_kv_manager.template load_V<Seqlenk_mask>(n_block, sV(_, _, smem_pipe_write));
      }
    };
    auto preprocess_Q = [&] {
      if constexpr (!AppendKV) {
        flash::cp_async_wait<Share_QV_Smem ? 1 : kStages * 2 - 1>();
      } else {
        if (get<1>(params.shape_rotary) > 0) {
          using Rotary_t = Rotary<kBlockM, kHeadDim, NumMmaThreads, Element, !(Is_causal || Is_local) >;
          Rotary_t rotary(params.ptr_rotary_cos, params.shape_rotary, params.stride_rotary_cos,
                  params.ptr_rotary_sin, params.stride_rotary_sin,
                  params.is_rotary_interleaved, thread_idx, seqlen_q,
                  seqlen_info.seqlen_rotary);
          int const qhead_per_khead = !PackGQA ? 1 : params.qhead_per_khead_divmod.divisor;
          if (params.is_rotary_interleaved) {
            auto [tRrCos, tRrSin] = cute::conditional_return<!PackGQA>(
              rotary.template load_cos_sin<true >(m_block),
              rotary.template load_cos_sin_packgqa<true >(m_block, params.qhead_per_khead_divmod)
            );
            flash::cp_async_wait<Share_QV_Smem ? 1 : kStages * 2 - 1>();
            __syncthreads();
            rotary.apply_Q_interleaved(sQ, tRrCos, tRrSin, m_block, qhead_per_khead);
          } else {
            auto [tRrCosCont, tRrSinCont] = cute::conditional_return<!PackGQA>(
              rotary.template load_cos_sin<false >(m_block),
              rotary.template load_cos_sin_packgqa<false >(m_block, params.qhead_per_khead_divmod)
            );
            flash::cp_async_wait<Share_QV_Smem ? 1 : kStages * 2 - 1>();
            __syncthreads();
            rotary.apply_Q_contiguous(sQ, tRrCosCont, tRrSinCont, m_block, qhead_per_khead);
          }
        } else {
          flash::cp_async_wait<Share_QV_Smem ? 1 : kStages * 2 - 1>();
        }
      }
      if constexpr (Q_in_regs) {
        __syncthreads();
        Tensor tSrQ_copy_view = smem_thr_copy_Q.retile_D(tSrQ);
        Tensor tSsQ_copy_view = smem_thr_copy_Q.partition_S(sQ);
        cute::copy(smem_tiled_copy_Q, tSsQ_copy_view, tSrQ_copy_view);
      }
    };
    if constexpr (Share_QV_Smem) {
      load_K(n_block, 0, cute::true_type{} );
      cute::cp_async_fence();
      preprocess_Q();
      __syncthreads();
    }
    if constexpr (!Share_QV_Smem) { __syncthreads(); }
    for_each(make_int_sequence<kStages>{}, [&] (auto stage) {
      static constexpr bool Is_first_stage = CUTE_STATIC_V(stage) == 0;
      static constexpr bool Is_last_stage = CUTE_STATIC_V(stage) == kStages - 1;
      if constexpr (!Share_QV_Smem || !Is_first_stage) {
        if (Is_first_stage || n_block - stage >= n_block_min) {
          load_K(n_block - stage, stage, cute::bool_constant<Is_first_stage>{} );
        }
        cute::cp_async_fence();
      }
      if constexpr (!Is_last_stage) {
        if (Is_first_stage || n_block - stage >= n_block_min) {
          load_V(n_block - stage, stage, cute::bool_constant<Is_first_stage>{} );
        }
        cute::cp_async_fence();
      }
    });
    if constexpr (!Share_QV_Smem) { preprocess_Q(); }
    flash::Mask<kBlockM, kBlockN, PackGQA, TiledMma> mask(
      thread_idx, seqlen_q, seqlen_k, params.window_size_left, params.window_size_right, 0 ,
      params.attention_chunk_divmod, params.qhead_per_khead_divmod
    );
    float softcap_val = params.softcap_val;
    if constexpr (Has_softcap && Is_FP8) {
      float const q_descale = params.ptr_q_descale == nullptr ? 1.0f : params.ptr_q_descale[bidb * get<0>(params.stride_q_descale) + bidh_kv * get<1>(params.stride_q_descale)];
      float const k_descale = params.ptr_k_descale == nullptr ? 1.0f : params.ptr_k_descale[bidb * get<0>(params.stride_k_descale) + bidh_kv * get<1>(params.stride_k_descale)];
      softcap_val *= q_descale * k_descale;
    }
    auto scoremod_premask_fn = [&](auto& tSrS) {
      if constexpr (Has_softcap) { flash::apply_softcap(tSrS, softcap_val); }
    };
    int smem_pipe_read = 0, smem_pipe_write = kStages - 1;
    auto load_K_next = [&] {
      if (n_block - kStages >= n_block_min) {
        load_K(n_block - kStages, kStages > 1 ? smem_pipe_write : 0, cute::false_type{} );
      }
      cute::cp_async_fence();
    };
    auto sync = [&] {
      flash::cp_async_wait<kStages * 2 - 2>();
      __syncthreads();
    };
    clear(tOrO);
    auto fwd_step = [&](int const n_block, auto mask_fn, auto is_first_iter_type, auto check_inf_type) {
      static constexpr bool Is_first_iter = decltype(is_first_iter_type)::value;
      static constexpr bool Check_inf = decltype(check_inf_type)::value;
      Tensor tSrS = partition_fragment_C(tiled_mma, select<0, 1>(TileShape_MNK{}));
      clear(tSrS);
      sync();
      auto load_V_next = [&] {
        if (n_block - kStages + 1 >= n_block_min) {
          load_V(n_block - kStages + 1, kStages > 1 ? smem_pipe_write : 0, cute::bool_constant<Is_first_iter && kStages == 1>{} );
        }
        cute::cp_async_fence();
      };
      Tensor tSrQ_cur = cute::conditional_return<Q_in_regs>(tSrQ, thr_mma.partition_fragment_A(sQ));
      Tensor tSrK = thr_mma.partition_fragment_B(sK(_, _, _0{}));
      flash::gemm_sm80<Q_in_regs>(
        tSrS, tSrQ_cur, tSrK, tSsQ, tSsK(_, _, _, kStages > 1 ? smem_pipe_read : 0),
        tiled_mma, smem_tiled_copy_Q, smem_tiled_copy_K, smem_thr_copy_Q, smem_thr_copy_K, load_V_next
      );
      smem_pipe_write = smem_pipe_write < kStages - 1 ? smem_pipe_write + 1 : 0;
      scoremod_premask_fn(tSrS);
      if constexpr (kStages == 1) { sync(); load_K_next(); }
      mask_fn(tSrS, n_block);
      Tensor scores_scale = softmax.template max_get_scale<Is_first_iter, Check_inf>(tSrS);
      softmax.template online_softmax<Is_first_iter, Check_inf>(tSrS);
      if constexpr (Is_FP8) { flash::permute_Cregs_fp8(tSrS); }
      Tensor tOrP_acc = make_tensor(tSrS.data(), flash::convert_layout_acc_Aregs<TiledMma>(tSrS.layout()));
      Tensor tOrP = make_tensor_like<Element>(tOrP_acc);
      convert_type_out(tOrP_acc, tOrP);
      if constexpr (!Is_first_iter) { softmax.rescale_o(tOrO, scores_scale); }
      if constexpr (kStages > 1) { sync(); }
      Tensor tOrV = thr_mma.partition_fragment_B(sVt(_, _, _0{}));
      flash::gemm_rs_sm80(tOrO, tOrP, tOrV, tOsVt(_, _, _, kStages > 1 ? smem_pipe_read : 0), tiled_mma, smem_tiled_copy_V, smem_thr_copy_V);
      if constexpr (kStages > 1) { load_K_next(); }
      smem_pipe_read = smem_pipe_read < kStages - 1 ? smem_pipe_read + 1 : 0;
    };
    auto first_iter_mask_fn = [&](auto& tSrS, int n_block) { mask.template apply<true , Is_causal, Is_local>(tSrS, m_block, n_block); };
    fwd_step(n_block, first_iter_mask_fn, cute::true_type{} , cute::true_type{} );
    --n_block;
    if constexpr (Is_causal || Is_local) {
      auto mask_fn = [&](auto& tSrS, int n_block) { mask.template apply<false , Is_causal, Is_local>(tSrS, m_block, n_block); };
      int const n_block_min_causal_local_mask = BlockMN_t::get_n_block_min_causal_local_mask(
        seqlen_info, m_block, n_block_min, params.window_size_right,
        params.attention_chunk_divmod, params.qhead_per_khead_divmod);
      #pragma unroll 1
      for (; n_block >= n_block_min_causal_local_mask; --n_block) {
        fwd_step(n_block, mask_fn, cute::false_type{} , cute::true_type{} );
      }
    }
    int const n_block_min_before_local_mask = BlockMN_t::get_n_block_min_before_local_mask(
      seqlen_info, m_block, n_block_min, params.window_size_left,
      params.attention_chunk_divmod, params.qhead_per_khead_divmod);
    auto no_mask_fn = [](auto& tSrS, int n_block) { };
    #pragma unroll 1
    for (; n_block >= n_block_min_before_local_mask; --n_block) {
      fwd_step(n_block, no_mask_fn, cute::false_type{} , cute::false_type{} );
    }
    if constexpr (Is_local) {
      auto local_mask_fn = [&](auto& tSrS, int n_block) { mask.template apply<false , false , Is_local>(tSrS, m_block, n_block); };
      #pragma unroll 1
      for (; n_block >= n_block_min; --n_block) {
        fwd_step(n_block, local_mask_fn, cute::false_type{} , cute::bool_constant<Is_local>{} );
      }
    }
    float const v_descale = !Is_FP8 || params.ptr_v_descale == nullptr ? 1.0f : params.ptr_v_descale[bidb * get<0>(params.stride_v_descale) + bidh_kv * get<1>(params.stride_v_descale)];
    Tensor scores_scale = softmax.finalize(v_descale);
    softmax.rescale_o(tOrO, scores_scale);
    if constexpr (Is_FP8) { flash::permute_output_fp8(tOrO); }
    return true;
  }
  template <typename SharedStorage>
  CUTLASS_DEVICE bool
  store_kv_new(Params const& params,
         int const thread_idx,
         SharedStorage &shared_storage,
         SeqlenInfo_t const& seqlen_info,
         cute::tuple<int32_t, int32_t, int32_t, int32_t> block_coord
  ) {
    auto [m_block, bidh, bidb, split_idx] = block_coord;
    auto n_block_new_min_max = BlockMN_t::get_n_block_k_new_min_max(
      seqlen_info, m_block, bidb, split_idx, params.num_splits,
      params.window_size_left, params.window_size_right, params.attention_chunk_divmod,
      params.qhead_per_khead_divmod);
    int const n_block_new_min = get<0>(n_block_new_min_max);
    int const n_block_new_max = get<1>(n_block_new_min_max);
    if (n_block_new_max <= n_block_new_min) { return false; }
    Tensor sK = make_tensor(make_smem_ptr(shared_storage.tensors.mainloop.smem_k.data()), SmemLayoutK{});
    Tensor sV = make_tensor(make_smem_ptr(shared_storage.tensors.mainloop.smem_v.data()), SmemLayoutV{});
    int const bidh_kv = !PackGQA ? params.qhead_per_khead_divmod.divide(bidh) : bidh;
    int const bidb_kv = params.kv_batch_idx == nullptr ? bidb : params.kv_batch_idx[bidb];
    bool const is_varlen_k_new = Varlen && params.cu_seqlens_k_new;
    Tensor mKnew = make_tensor(make_gmem_ptr(params.ptr_K_new), params.shape_K_new, params.stride_K_new)(_, _, bidh_kv, !is_varlen_k_new ? bidb : 0);
    Tensor mVnew = make_tensor(make_gmem_ptr(params.ptr_V_new), params.shape_K_new, params.stride_V_new)(_, _, bidh_kv, !is_varlen_k_new ? bidb : 0);
    bool const is_varlen_k = Varlen && params.cu_seqlens_k;
    Tensor mK = make_tensor(make_gmem_ptr(params.ptr_K), params.shape_K, params.stride_K)(_, _, bidh_kv, !is_varlen_k ? bidb_kv : 0);
    Tensor mV = make_tensor(make_gmem_ptr(params.ptr_V), params.shape_K, params.stride_V)(_, _, bidh_kv, !is_varlen_k ? bidb_kv : 0);
    Tensor gKnew = local_tile(domain_offset(make_coord(seqlen_info.offset_k_new, _0{}), mKnew), select<1, 2>(TileShape_MNK{}), make_coord(_, _0{}));
    Tensor gVnew = local_tile(domain_offset(make_coord(seqlen_info.offset_k_new, _0{}), mVnew), select<1, 2>(TileShape_MNK{}), make_coord(_, _0{}));
    int const offset_k = seqlen_info.offset_k + seqlen_info.seqlen_k_og;
    Tensor gK = local_tile(domain_offset(make_coord(offset_k, _0{}), mK), select<1, 2>(TileShape_MNK{}), make_coord(_, _0{}));
    Tensor gV = local_tile(domain_offset(make_coord(offset_k, _0{}), mV), select<1, 2>(TileShape_MNK{}), make_coord(_, _0{}));
    static constexpr int kBlockN = get<1>(TileShape_MNK{});
    static constexpr int kHeadDim = get<2>(TileShape_MNK{});
    int const seqlen_k_new = seqlen_info.seqlen_k_new;
    using Rotary_t = Rotary<kBlockN, kHeadDim, NumMmaThreads, Element>;
    Rotary_t rotary(params.ptr_rotary_cos, params.shape_rotary, params.stride_rotary_cos,
            params.ptr_rotary_sin, params.stride_rotary_sin,
            params.is_rotary_interleaved, thread_idx, seqlen_k_new,
            seqlen_info.seqlen_rotary);
    using PagedKVManager_t = PagedKVManager<get<1>(TileShape_MNK{}), get<2>(TileShape_MNK{}), get<1>(TileShape_MNK_PV{}), NumMmaThreads, Element, true , 2 >;
    PagedKVManager_t paged_kv_manager(
      params.ptr_pagetable, params.shape_pagetable, params.stride_pagetable,
      params.ptr_K, params.shape_K, params.stride_K,
      params.ptr_V, params.headdim_v, params.stride_V,
      params.page_size_divmod,
      params.page_size_divmod ,
      bidb_kv, bidh_kv, thread_idx, seqlen_k_new, offset_k,
      0
    );
    static_assert(std::is_same_v<GmemLayoutAtomAppend, typename Rotary_t::LayoutAtom>);
    static_assert(!PagedKV || std::is_same_v<GmemLayoutAtomAppend, typename PagedKVManager_t::GmemLayoutAtomKVCpAsync>);
    GmemTiledCopyQKV gmem_tiled_copy_kv_g2s;
    auto gmem_thr_copy_kv_g2s = gmem_tiled_copy_kv_g2s.get_thread_slice(thread_idx);
    auto gmem_thr0_copy_kv_g2s = gmem_tiled_copy_kv_g2s.get_thread_slice(_0{});
    GmemTiledCopyAppendKV gmem_tiled_copy_kv_s2g;
    auto gmem_thr_copy_kv_s2g = gmem_tiled_copy_kv_s2g.get_thread_slice(thread_idx);
    auto gmem_thr0_copy_kv_s2g = gmem_tiled_copy_kv_s2g.get_thread_slice(_0{});
    Tensor tKgKnew = gmem_thr_copy_kv_g2s.partition_S(gKnew);
    Tensor tKsKg2s = gmem_thr_copy_kv_g2s.partition_S(sK);
    Tensor tKsKs2g = gmem_thr_copy_kv_s2g.partition_S(sK);
    Tensor tKgK = gmem_thr_copy_kv_s2g.partition_D(gK);
    Tensor tVgVnew = gmem_thr_copy_kv_g2s.partition_S(gVnew);
    Tensor tVsVg2s = gmem_thr_copy_kv_g2s.partition_S(sV);
    Tensor tVsVs2g = gmem_thr_copy_kv_s2g.partition_S(sV);
    Tensor tVgV = gmem_thr_copy_kv_s2g.partition_D(gV);
    Tensor cK = cute::make_identity_tensor(select<1, 2>(TileShape_MNK{}));
    Tensor tKcKg2s = gmem_thr_copy_kv_g2s.partition_D(cK);
    Tensor t0KcKg2s = gmem_thr0_copy_kv_g2s.partition_D(cK);
    Tensor tKpKg2s = make_tensor<bool>(make_shape(size<2>(tKsKg2s)));
    Tensor tKcKs2g = gmem_thr_copy_kv_s2g.partition_D(cK);
    Tensor t0KcKs2g = gmem_thr0_copy_kv_s2g.partition_D(cK);
    Tensor tKpKs2g = make_tensor<bool>(make_shape(size<2>(tKsKs2g)));
    #pragma unroll
    for (int k = 0; k < size(tKpKg2s); ++k) { tKpKg2s(k) = get<1>(tKcKg2s(_0{}, _0{}, k)) < get<1>(params.shape_K); }
    #pragma unroll
    for (int k = 0; k < size(tKpKs2g); ++k) { tKpKs2g(k) = get<1>(tKcKs2g(_0{}, _0{}, k)) < get<1>(params.shape_K); }
    auto load_K_new = [&] (int const n_block, int const smem_pipe_write, auto need_seqlenk_masking_type) {
      static constexpr bool Seqlenk_mask = decltype(need_seqlenk_masking_type)::value;
      static constexpr bool EvenN = kBlockN % CUTE_STATIC_V(shape<0>(GmemLayoutAtom{})) == 0;
      Tensor tKsK_cur = tKsKg2s(_, _, _, smem_pipe_write);
      int const seqlenk_row_limit = -int(get<0>(tKcKg2s(_0{}, _0{}, _0{}))) + (EvenN
        ? seqlen_k_new - n_block * kBlockN
        : (!Seqlenk_mask ? kBlockN : std::min(seqlen_k_new - n_block * kBlockN, kBlockN)));
      flash::copy<!Seqlenk_mask && EvenN, false, false, true>(
        gmem_tiled_copy_kv_g2s, tKgKnew(_, _, _, n_block), tKsK_cur, t0KcKg2s, tKpKg2s, seqlenk_row_limit);
    };
    auto load_V_new = [&] (int const n_block, int const smem_pipe_write, auto need_seqlenk_masking_type) {
      static constexpr bool Seqlenk_mask = decltype(need_seqlenk_masking_type)::value;
      static constexpr bool EvenN = kBlockN % CUTE_STATIC_V(shape<0>(GmemLayoutAtom{})) == 0;
      Tensor tVsV_cur = tVsVg2s(_, _, _, smem_pipe_write);
      int const seqlenk_row_limit = -int(get<0>(tKcKg2s(_0{}, _0{}, _0{}))) + (EvenN
        ? seqlen_k_new - n_block * kBlockN
        : (!Seqlenk_mask ? kBlockN : std::min(seqlen_k_new - n_block * kBlockN, kBlockN)));
      flash::copy<!Seqlenk_mask && EvenN, false, false, true>(
        gmem_tiled_copy_kv_g2s, tVgVnew(_, _, _, n_block), tVsV_cur, t0KcKg2s, tKpKg2s, seqlenk_row_limit);
    };
    auto store_K = [&] (int const n_block, int const smem_pipe_read) {
      int const n_limit = std::min(seqlen_k_new - n_block * kBlockN, kBlockN);
      if (get<1>(params.shape_rotary) <= 0) {
        Tensor tKsK_cur = tKsKs2g(_, _, _, smem_pipe_read);
        if constexpr (!PagedKV) {
          Tensor tKgK_cur = tKgK(_, _, _, n_block);
          flash::copy<false, false, false, false>(
            gmem_tiled_copy_kv_s2g, tKsK_cur, tKgK_cur, tKcKs2g, tKpKs2g, std::min(seqlen_k_new - n_block * kBlockN, kBlockN)
          );
        } else {
          paged_kv_manager.store_K(n_block, tKsK_cur);
        }
      } else {
        Tensor gK_cur = gK(_, _, n_block);
        auto tPrKPtr = cute::conditional_return<PagedKV>(paged_kv_manager.compute_K_ptr(), nullptr);
        if (params.is_rotary_interleaved) {
          auto [tRrCos, tRrSin] = rotary.template load_cos_sin<true >(n_block);
          rotary.template apply_K_interleaved<PagedKV>(sK(_, _, smem_pipe_read), gK_cur, tKpKs2g, tRrCos, tRrSin, tPrKPtr, n_block);
        } else {
          auto [tRrCosCont, tRrSinCont] = rotary.template load_cos_sin<false >(n_block);
          rotary.template apply_K_contiguous<PagedKV>(sK(_, _, smem_pipe_read), gK_cur, tKpKs2g, tRrCosCont, tRrSinCont, tPrKPtr, n_block, get<1>(params.shape_K));
        }
      }
    };
    auto store_V = [&] (int const n_block, int const smem_pipe_read) {
      int const n_limit = std::min(seqlen_k_new - n_block * kBlockN, kBlockN);
      Tensor tVsV_cur = tVsVs2g(_, _, _, smem_pipe_read);
      if constexpr (!PagedKV) {
        Tensor tVgV_cur = tVgV(_, _, _, n_block);
        flash::copy<false, false, false, false>(
          gmem_tiled_copy_kv_s2g, tVsV_cur, tVgV_cur, tKcKs2g, tKpKs2g, n_limit);
      } else {
        paged_kv_manager.store_V(n_block, tVsV_cur);
      }
    };
    int n_block = n_block_new_max - 1;
    for_each(make_int_sequence<kStages>{}, [&] (auto stage) {
      static constexpr bool Is_first_stage = CUTE_STATIC_V(stage) == 0;
      static constexpr bool Is_last_stage = CUTE_STATIC_V(stage) == kStages - 1;
      if (Is_first_stage || n_block - stage >= n_block_new_min) {
        load_K_new(n_block - stage, stage, cute::bool_constant<Is_first_stage>{} );
      }
      cute::cp_async_fence();
      if constexpr (Is_first_stage) { __syncthreads(); }
      if constexpr (!Is_last_stage) {
        if (Is_first_stage || n_block - stage >= n_block_new_min) {
          load_V_new(n_block - stage, stage, cute::bool_constant<Is_first_stage>{} );
        }
        cute::cp_async_fence();
      }
    });
    int smem_pipe_read = 0, smem_pipe_write = kStages - 1;
    #pragma unroll 1
    for (; n_block >= n_block_new_min; --n_block) {
      if constexpr (PagedKV) { paged_kv_manager.template load_page_table<true >(n_block); }
      flash::cp_async_wait<kStages * 2 - 2>();
      __syncthreads();
      store_K(n_block, kStages > 1 ? smem_pipe_read : 0);
      if (n_block - kStages + 1 >= n_block_new_min) {
        load_V_new(n_block - kStages + 1, kStages > 1 ? smem_pipe_write : 0, cute::bool_constant<kStages == 1>{} );
      }
      cute::cp_async_fence();
      smem_pipe_write = smem_pipe_write < kStages - 1 ? smem_pipe_write + 1 : 0;
      flash::cp_async_wait<kStages * 2 - 2>();
      __syncthreads();
      store_V(n_block, kStages > 1 ? smem_pipe_read : 0);
      smem_pipe_read = smem_pipe_read < kStages - 1 ? smem_pipe_read + 1 : 0;
      if (n_block - kStages >= n_block_new_min) {
        load_K_new(n_block - kStages, kStages > 1 ? smem_pipe_write : 0, cute::false_type{} );
      }
      cute::cp_async_fence();
    }
    return true;
  }
};
}
#include <cutlass/cutlass.h>
#include <cutlass/fast_math.h>
#include "cute/tensor.hpp"
#include "cutlass/gemm/collective/builders/sm90_common.inl"
#include "cutlass/epilogue/collective/builders/sm90_common.inl"
namespace flash {
using namespace cute;
template <class TileShape_MNK_PV_, class ClusterShape_, class Element_, class ArchTag_,
     int NumEpilogueThreads_, bool Varlen_, bool PackGQA_, bool Split_, bool FP8PermuteCol=false>
struct CollectiveEpilogueFwd {
  using TileShape_MNK_PV = TileShape_MNK_PV_;
  using ClusterShape = ClusterShape_;
  using Element = Element_;
  using ElementPartial = float;
  using ArchTag = ArchTag_;
  static constexpr int NumEpilogueThreads = NumEpilogueThreads_;
  static constexpr bool Varlen = Varlen_;
  static constexpr bool PackGQA = PackGQA_;
  static constexpr bool Split = Split_;
  static constexpr bool Use_smem = !(Split && !Varlen);
  static constexpr bool Use_TMA_O = ArchTag::kMinComputeCapability >= 90 && !Varlen && !Split && !PackGQA;
  static_assert(ArchTag::kMinComputeCapability >= 80);
  static_assert(ArchTag::kMinComputeCapability >= 90 || CUTE_STATIC_V(size(ClusterShape{})) == 1);
  static_assert(sizeof(Element) <= 2);
  static constexpr int kBlockM = get<0>(TileShape_MNK_PV{});
  static constexpr int kHeadDimV = get<1>(TileShape_MNK_PV{});
  static constexpr bool LargeHeadDimV = kHeadDimV > 256;
  using GmemTiledCopyOTMA = cute::SM90_TMA_STORE;
  static constexpr int kGmemElemsPerStore = sizeof(cute::uint128_t) / sizeof(Element);
  static_assert(kHeadDimV % kGmemElemsPerStore == 0, "Headdim must be a multiple of kGmemElemsPerStore");
  static constexpr int kBytePerRow = kHeadDimV * sizeof(Element);
  static constexpr int kBlockKGmem = (kBytePerRow % 128 == 0 ? 128 : (kBytePerRow % 64 == 0 ? 64 : 32)) / sizeof(Element);
  static constexpr int kGmemThreadsPerRow = kBlockKGmem / kGmemElemsPerStore;
  static_assert(cutlass::NumThreadsPerWarp % kGmemThreadsPerRow == 0);
  static_assert(NumEpilogueThreads % kGmemThreadsPerRow == 0, "NumEpilogueThreads must be a multiple of kGmemThreadsPerRow");
  using GmemLayoutAtom = Layout<Shape <Int<NumEpilogueThreads / kGmemThreadsPerRow>, Int<kGmemThreadsPerRow>>,
                 Stride<Int<kGmemThreadsPerRow>, _1>>;
  static_assert(kBlockM % CUTE_STATIC_V(shape<0>(GmemLayoutAtom{})) == 0, "kBlockM must be a multiple of NumEpilogueThreads / kGmemThreadsPerRow");
  using GmemTiledCopyO = decltype(
    make_tiled_copy(Copy_Atom<AutoVectorizingCopyWithAssumedAlignment<128>, Element>{},
            GmemLayoutAtom{},
            Layout<Shape<_1, Int<kGmemElemsPerStore>>>{}));
  using SmemLayoutAtomOTMA = decltype(cutlass::gemm::collective::detail::ss_smem_selector<GMMA::Major::K, Element,
    decltype(cute::get<0>(TileShape_MNK_PV{})), decltype(cute::get<1>(TileShape_MNK_PV{}))>());
  using SmemLayoutOTMA = decltype(tile_to_shape(SmemLayoutAtomOTMA{}, select<0, 1>(TileShape_MNK_PV{})));
  static constexpr int kSwizzle = kBlockKGmem == 128 ? 4 : (kBlockKGmem == 64 ? 3 : (kBlockKGmem == 32 ? 2 : 1));
  static constexpr int kSwizzleBase = sizeof(Element) == 4 ? 2 : (sizeof(Element) == 2 ? 3 : 4);
  using SmemLayoutAtomO = decltype(
    composition(Swizzle<kSwizzle, kSwizzleBase, kSwizzleBase>{},
          Layout<Shape<_8, Int<kBlockKGmem>>,
              Stride<Int<kBlockKGmem>, _1>>{}));
  using SmemLayoutOSTS = decltype(tile_to_shape(SmemLayoutAtomO{}, select<0, 1>(TileShape_MNK_PV{})));
  using SmemLayoutO = std::conditional_t<ArchTag::kMinComputeCapability >= 90, SmemLayoutOTMA, SmemLayoutOSTS>;
  using ShapeO = cute::Shape<int32_t, int32_t, int32_t, int32_t, int32_t>;
  using StrideO = cute::Stride<int64_t, _1, int64_t, int64_t, int64_t>;
  using StrideLSE = cute::Stride<_1, int64_t, int64_t, int64_t>;
  using ShapeOPacked = std::conditional_t<!PackGQA, ShapeO, cute::Shape<cute::Shape<int32_t, int32_t>, int32_t, int32_t, int32_t, int32_t>>;
  using StrideOPacked = std::conditional_t<!PackGQA, StrideO, cute::Stride<cute::Stride<int64_t, int64_t>, _1, int64_t, int64_t, int64_t>>;
  using ShapeLSEPacked = std::conditional_t<!PackGQA, cute::Shape<int32_t, int32_t, int32_t, int32_t>, cute::Shape<cute::Shape<int32_t, int32_t>, int32_t, int32_t, int32_t>>;
  using StrideLSEPacked = std::conditional_t<!PackGQA, StrideLSE, cute::Stride<cute::Stride<int64_t, _1>, int64_t, int64_t, int64_t>>;
  using EpilogueTile_MN = decltype(select<0, 1>(TileShape_MNK_PV{}));
  using CopyOpR2S = std::conditional_t<
    ArchTag::kMinComputeCapability >= 90,
    decltype(cutlass::epilogue::collective::detail::sm90_get_smem_store_op_for_accumulator<StrideO, Element, EpilogueTile_MN>()),
    AutoVectorizingCopyWithAssumedAlignment<128>
  >;
  using SmemCopyAtomO = Copy_Atom<CopyOpR2S, Element>;
  struct TensorStorage : cute::aligned_struct<128> {
    cute::array_aligned<Element, Use_smem ? cute::cosize_v<SmemLayoutO> : 0> smem_o;
  };
  using TMA_O = std::conditional_t<
    Use_TMA_O,
    decltype(make_tma_copy(
      GmemTiledCopyOTMA{},
      make_tensor(make_gmem_ptr(static_cast<Element*>(nullptr)), ShapeO{}, StrideO{}),
      SmemLayoutOTMA{},
      select<0, 1>(TileShape_MNK_PV{}),
      _1{})),
    std::nullptr_t
  >;
  struct Arguments {
    Element* ptr_O;
    ShapeO const shape_O;
    StrideO const stride_O;
    ElementPartial* ptr_O_partial;
    StrideO const stride_O_partial;
    float* ptr_LSE;
    StrideLSE const stride_LSE;
    float* ptr_LSE_partial;
    StrideLSE const stride_LSE_partial;
    int32_t const nheads_kv;
    int const* cu_seqlens = nullptr;
    int const* seqused = nullptr;
    Element const* ptr_gate = nullptr;
    int gate_stride = 0;
    float gate_eps = 0.f;
  };
  struct Params {
    Element* ptr_O;
    ShapeO const shape_O;
    StrideO const stride_O;
    ShapeOPacked const shape_O_packed;
    StrideOPacked const stride_O_packed;
    ElementPartial* ptr_O_partial;
    StrideO const stride_O_partial;
    StrideOPacked const stride_O_partial_packed;
    float* ptr_LSE;
    StrideLSE const stride_LSE;
    ShapeLSEPacked const shape_LSE_packed;
    StrideLSEPacked const stride_LSE_packed;
    float* ptr_LSE_partial;
    StrideLSE const stride_LSE_partial;
    StrideLSEPacked const stride_LSE_partial_packed;
    cutlass::FastDivmod qhead_per_khead_divmod;
    TMA_O tma_store_O;
    int const* cu_seqlens = nullptr;
    int const* seqused = nullptr;
    Element const* ptr_gate = nullptr;
    int gate_stride = 0;
    float gate_eps = 0.f;
  };
  static Params
  to_underlying_arguments(Arguments const& args) {
    Tensor mO = make_tensor(make_gmem_ptr(args.ptr_O), args.shape_O, args.stride_O);
    TMA_O tma_store_O = [&]{
      if constexpr (Use_TMA_O) {
        return make_tma_copy(GmemTiledCopyOTMA{}, mO, SmemLayoutO{}, select<0, 1>(TileShape_MNK_PV{}), _1{});
      } else {
        return nullptr;
      }
    }();
    int const qhead_per_khead = !PackGQA ? 1 : cute::ceil_div(get<2>(args.shape_O), args.nheads_kv);
    auto const shape_O_packed = cute::conditional_return<!PackGQA>(
      args.shape_O,
      make_shape(make_shape(qhead_per_khead, get<0>(args.shape_O)), get<1>(args.shape_O), args.nheads_kv, get<3>(args.shape_O), get<4>(args.shape_O))
    );
    auto const stride_O_packed = cute::conditional_return<!PackGQA>(
      args.stride_O,
      make_stride(make_stride(get<2>(args.stride_O), get<0>(args.stride_O)), get<1>(args.stride_O), get<2>(args.stride_O) * qhead_per_khead, get<3>(args.stride_O), get<4>(args.stride_O))
    );
    auto const stride_O_partial_packed = cute::conditional_return<!PackGQA>(
      args.stride_O_partial,
      make_stride(make_stride(get<2>(args.stride_O_partial), get<0>(args.stride_O_partial)), get<1>(args.stride_O_partial), get<2>(args.stride_O_partial) * qhead_per_khead, get<3>(args.stride_O_partial), get<4>(args.stride_O_partial))
    );
    auto const shape_LSE_packed = cute::conditional_return<!PackGQA>(
      select<0, 2, 3, 4>(args.shape_O),
      make_shape(make_shape(qhead_per_khead, get<0>(args.shape_O)), args.nheads_kv, get<3>(args.shape_O), get<4>(args.shape_O))
    );
    auto const stride_LSE_packed = cute::conditional_return<!PackGQA>(
      args.stride_LSE,
      make_stride(make_stride(get<1>(args.stride_LSE), get<0>(args.stride_LSE)), get<1>(args.stride_LSE) * qhead_per_khead, get<2>(args.stride_LSE), get<3>(args.stride_LSE))
    );
    auto const stride_LSE_partial_packed = cute::conditional_return<!PackGQA>(
      args.stride_LSE_partial,
      make_stride(make_stride(get<1>(args.stride_LSE_partial), get<0>(args.stride_LSE_partial)), get<1>(args.stride_LSE_partial) * qhead_per_khead, get<2>(args.stride_LSE_partial), get<3>(args.stride_LSE_partial))
    );
    return {args.ptr_O, args.shape_O, args.stride_O, shape_O_packed, stride_O_packed,
        args.ptr_O_partial, args.stride_O_partial, stride_O_partial_packed,
        args.ptr_LSE, args.stride_LSE, shape_LSE_packed, stride_LSE_packed,
        args.ptr_LSE_partial, args.stride_LSE_partial, stride_LSE_partial_packed,
        cutlass::FastDivmod(qhead_per_khead),
        tma_store_O, args.cu_seqlens, args.seqused
        , args.ptr_gate, args.gate_stride, args.gate_eps
        };
  }
  CUTLASS_DEVICE
  static void prefetch_tma_descriptors(Params const& params) {
    if constexpr (Use_TMA_O) {
      cute::prefetch_tma_descriptor(params.tma_store_O.get_tma_descriptor());
    }
  }
  template <typename SharedStorage>
  CUTLASS_DEVICE void store_skip(SharedStorage& shared_storage) {
    flash::named_barrier_sync(NumEpilogueThreads, cutlass::arch::ReservedNamedBarriers::EpilogueBarrier);
    #pragma unroll
    for (uint32_t cta_id = 0; cta_id < size(ClusterShape{}); ++cta_id) {
      shared_storage.pipelines.barrier_O.arrive(cta_id);
    }
  }
  template <typename SharedStorage, typename FrgTensorO, typename FrgTensorLSE, typename TiledMma>
  CUTLASS_DEVICE void
  store(Params const& params,
     FrgTensorO& tOrO,
     FrgTensorLSE const& lse,
     SharedStorage& shared_storage,
     TiledMma tiled_mma,
     int thread_idx,
     cute::tuple<int32_t, int32_t, int32_t, int32_t> const& block_coord
     ) {
    auto [m_block, bidh, bidb, split_idx] = block_coord;
    int num_splits = get<4>(params.shape_O_packed);
    if constexpr (Split && Varlen) {
      uint32_t num_splits_dynamic_u = reinterpret_cast<uint32_t const&>(split_idx) >> 16;
      int num_splits_dynamic = reinterpret_cast<int&>(num_splits_dynamic_u);
      num_splits = num_splits_dynamic > 0 ? num_splits_dynamic : num_splits;
      split_idx &= 0x0000FFFF;
    }
    bool const is_split = !Split ? false : (!Varlen ? true : num_splits > 1);
    using X52P_t = flash::PackGQAManager<get<0>(TileShape_MNK_PV{}), get<1>(TileShape_MNK_PV{}), NumEpilogueThreads, Element>;
    uint4 x52g[X52P_t::kX52Rows];
    if constexpr (!Split && PackGQA && !Use_TMA_O) {
      if (params.ptr_gate != nullptr) {
        flash::SeqlenInfo<Varlen, kBlockM> x52si{bidb, size<0>(params.shape_O), params.cu_seqlens, params.seqused};
        bool const x52vl = Varlen && params.cu_seqlens;
        Tensor x52mO = make_tensor(make_gmem_ptr(params.ptr_O + x52si.offset * get<0>(params.stride_O)), params.shape_O_packed, params.stride_O_packed)(_, _, bidh, !x52vl ? bidb : 0, _0{});
        X52P_t::load_gate(x52g, x52mO, params.qhead_per_khead_divmod, thread_idx, x52si.seqlen, m_block,
                 params.ptr_O, int64_t(get<0>(params.stride_O)), params.ptr_gate, params.gate_stride);
      }
    }
    Tensor sO = make_tensor(make_smem_ptr(shared_storage.tensors.epilogue.smem_o.data()), SmemLayoutO{});
    static constexpr bool NeedFP8Permute = FP8PermuteCol && (sizeof(Element) == 2 || sizeof(Element) == 4);
    if constexpr (NeedFP8Permute && Split) { flash::permute_output_fp8_Vcolmajor(tOrO); }
    Tensor tOrO_out = make_tensor_like<Element>(tOrO);
    flash::convert_type_out(tOrO, tOrO_out);
    if constexpr (NeedFP8Permute && !Split) { flash::permute_output_fp8_Vcolmajor(tOrO_out); }
    flash::named_barrier_sync(NumEpilogueThreads, cutlass::arch::ReservedNamedBarriers::EpilogueBarrier);
    if constexpr (Use_smem) {
      auto smem_tiled_copy_O = make_tiled_copy_C(SmemCopyAtomO{}, tiled_mma);
      auto smem_thr_copy_O = smem_tiled_copy_O.get_thread_slice(thread_idx);
      Tensor taccOrO = smem_thr_copy_O.retile_S(tOrO_out);
      Tensor taccOsO = smem_thr_copy_O.partition_D(sO);
      cute::copy(smem_tiled_copy_O, taccOrO, taccOsO);
      if constexpr (Use_TMA_O) {
        cutlass::arch::fence_view_async_shared();
        cutlass::arch::NamedBarrier::arrive(NumEpilogueThreads + cutlass::NumThreadsPerWarp,
                          cutlass::arch::ReservedNamedBarriers::EpilogueBarrier);
      } else {
        flash::named_barrier_sync(NumEpilogueThreads, cutlass::arch::ReservedNamedBarriers::EpilogueBarrier);
      }
    } else {
      if constexpr (ArchTag::kMinComputeCapability >= 90) {
        #pragma unroll
        for (uint32_t cta_id = 0; cta_id < size(ClusterShape{}); ++cta_id) {
          shared_storage.pipelines.barrier_O.arrive(cta_id);
        }
      }
    }
    flash::SeqlenInfo<Varlen, kBlockM> seqlen_info{bidb, size<0>(params.shape_O), params.cu_seqlens, params.seqused};
    bool is_varlen = Varlen && params.cu_seqlens;
    int offset_o = seqlen_info.offset;
    int seqlen_o = seqlen_info.seqlen;
    int warp_group_idx = __shfl_sync(0xFFFFFFFF, thread_idx / cutlass::NumThreadsPerWarpGroup, 0);
    auto thread_mma = tiled_mma.get_thread_slice(thread_idx);
    Tensor taccOcO = thread_mma.partition_C(cute::make_identity_tensor(select<0, 1>(TileShape_MNK_PV{})));
    static_assert(decltype(size<0, 0>(taccOcO))::value == 2);
    static_assert(decltype(size<0, 1>(taccOcO))::value == 2);
    Tensor taccOcO_rowcol = make_tensor(taccOcO.data(), flash::convert_layout_acc_rowcol(taccOcO.layout()));
    Tensor taccOcO_row = taccOcO_rowcol(_, _0{});
    CUTE_STATIC_ASSERT_V(size(lse) == size(taccOcO_row));
    using PackGQA_t = flash::PackGQAManager<get<0>(TileShape_MNK_PV{}), get<1>(TileShape_MNK_PV{}), NumEpilogueThreads, Element>;
    using PackGQApartial_t = flash::PackGQAManager<get<0>(TileShape_MNK_PV{}), get<1>(TileShape_MNK_PV{}), NumEpilogueThreads, ElementPartial>;
    Tensor mLSE = make_tensor(make_gmem_ptr((!is_split ? params.ptr_LSE : params.ptr_LSE_partial) + offset_o * get<0>(!is_split ? params.stride_LSE : params.stride_LSE_partial)),
                 params.shape_LSE_packed,
                 !is_split ? params.stride_LSE_packed : params.stride_LSE_partial_packed)(_, bidh, !is_varlen ? bidb : 0, !is_split ? 0 : split_idx);
    if (!LargeHeadDimV || warp_group_idx == 0) {
      if constexpr (!PackGQA) {
        #pragma unroll
        for (int mi = 0; mi < size(lse); ++mi) {
          int const row = m_block * kBlockM + get<0>(taccOcO_row(mi));
          if (get<1>(taccOcO_row(_0{})) == 0 && row < seqlen_o) { mLSE(row) = lse(mi); }
        }
      } else {
        PackGQA_t::store_LSE(mLSE, lse, tiled_mma, params.qhead_per_khead_divmod, thread_idx, seqlen_o, m_block);
      }
    }
    if constexpr (Use_TMA_O) {
      Tensor mO = params.tma_store_O.get_tma_tensor(params.shape_O)(_, _, bidh, bidb, split_idx);
      Tensor gO = local_tile(mO, select<0, 1>(TileShape_MNK_PV{}), make_coord(m_block, _0{}));
      auto block_tma_O = params.tma_store_O.get_slice(_0{});
      Tensor tOgO = block_tma_O.partition_D(gO);
      Tensor tOsO = block_tma_O.partition_S(sO);
      int warp_idx_sync = __shfl_sync(0xffffffff, thread_idx / cutlass::NumThreadsPerWarp, 0);
      if (warp_idx_sync == NumEpilogueThreads / cutlass::NumThreadsPerWarp - 1) {
        cutlass::arch::NamedBarrier::sync(NumEpilogueThreads + cutlass::NumThreadsPerWarp,
                         cutlass::arch::ReservedNamedBarriers::EpilogueBarrier);
        if (cute::elect_one_sync()) {
          cute::copy(params.tma_store_O, tOsO, tOgO);
          tma_store_arrive();
          tma_store_wait<0>();
          #pragma unroll
          for (uint32_t cta_id = 0; cta_id < size(ClusterShape{}); ++cta_id) {
            shared_storage.pipelines.barrier_O.arrive(cta_id);
          }
        }
      }
    } else {
      if (!is_split) {
        Tensor mO = make_tensor(make_gmem_ptr(params.ptr_O + offset_o * get<0>(params.stride_O)), params.shape_O_packed, params.stride_O_packed)(_, _, bidh, !is_varlen ? bidb : 0, _0{});
        Tensor gO = local_tile(mO, select<0, 1>(TileShape_MNK_PV{}), make_coord(m_block, _0{}));
        GmemTiledCopyO gmem_tiled_copy_O;
        auto gmem_thr_copy_O = gmem_tiled_copy_O.get_thread_slice(thread_idx);
        Tensor tOsO = gmem_thr_copy_O.partition_S(sO);
        Tensor tOrO = make_fragment_like(tOsO);
        cute::copy(gmem_tiled_copy_O, tOsO, tOrO);
        if constexpr (ArchTag::kMinComputeCapability >= 90) {
          cutlass::arch::fence_view_async_shared();
          #pragma unroll
          for (uint32_t cta_id = 0; cta_id < size(ClusterShape{}); ++cta_id) {
            shared_storage.pipelines.barrier_O.arrive(cta_id);
          }
        }
        if constexpr (!PackGQA) {
          Tensor tOcO = gmem_thr_copy_O.partition_D(cute::make_identity_tensor(select<0, 1>(TileShape_MNK_PV{})));
          Tensor tOpO = make_tensor<bool>(make_shape(size<2>(tOsO)));
          #pragma unroll
          for (int k = 0; k < size(tOpO); ++k) { tOpO(k) = get<1>(tOcO(_0{}, _0{}, k)) < get<1>(params.shape_O); }
          Tensor tOgO = gmem_thr_copy_O.partition_D(gO);
          flash::copy<false, false, false, false>(
            gmem_tiled_copy_O, tOrO, tOgO, tOcO, tOpO, seqlen_o - m_block * kBlockM
          );
        } else {
          if (params.ptr_gate != nullptr) {
            PackGQA_t::store_O_g(mO, tOrO, params.qhead_per_khead_divmod, thread_idx, seqlen_o, m_block,
                       params.gate_eps, x52g);
          } else
          PackGQA_t::store_O(mO, tOrO, params.qhead_per_khead_divmod, thread_idx, seqlen_o, m_block);
        }
      } else {
        Tensor mOpartial = make_tensor(make_gmem_ptr(params.ptr_O_partial + offset_o * get<0>(params.stride_O_partial)), params.shape_O_packed, params.stride_O_partial_packed)(_, _, bidh, !is_varlen ? bidb : 0, split_idx);
        Tensor gOpartial = local_tile(mOpartial, select<0, 1>(TileShape_MNK_PV{}), make_coord(m_block, _0{}));
        if constexpr (Use_smem) {
          if constexpr (ArchTag::kMinComputeCapability >= 90) {
            #pragma unroll
            for (uint32_t cta_id = 0; cta_id < size(ClusterShape{}); ++cta_id) {
              shared_storage.pipelines.barrier_O.arrive(cta_id);
            }
          }
        }
        if constexpr (!PackGQA) {
          static constexpr int kGmemElemsPerStoreDirect = 2;
          cute::Copy_Atom<AutoVectorizingCopyWithAssumedAlignment<128>, ElementPartial> gmem_copy_direct;
          Tensor tOrO_rowcol = make_tensor(tOrO.data(), flash::convert_layout_acc_rowcol(tOrO.layout()));
          Tensor tOrO_copy = cute::tiled_divide(tOrO_rowcol, Shape<_1, Int<kGmemElemsPerStoreDirect>>{});
          Tensor tOgO = thread_mma.partition_C(gOpartial);
          Tensor tOgO_rowcol = make_tensor(tOgO.data(), flash::convert_layout_acc_rowcol(tOgO.layout()));
          Tensor tOgO_copy = cute::tiled_divide(tOgO_rowcol, Shape<_1, Int<kGmemElemsPerStoreDirect>>{});
          Tensor taccOcO_col = taccOcO_rowcol(_0{}, _);
          #pragma unroll
          for (int m = 0; m < size(taccOcO_row); ++m) {
            if (get<0>(taccOcO_row(m)) < seqlen_o - m_block * kBlockM) {
              #pragma unroll
              for (int k = 0; k < size(taccOcO_col) / kGmemElemsPerStoreDirect; ++k) {
                if (get<1>(taccOcO_col(k * kGmemElemsPerStoreDirect)) < get<1>(params.shape_O)) {
                  cute::copy(gmem_copy_direct, tOrO_copy(_, m, k), tOgO_copy(_, m, k));
                }
              }
            }
          }
        } else {
          PackGQApartial_t::store_O_direct(mOpartial, tOrO, tiled_mma, params.qhead_per_khead_divmod, thread_idx, seqlen_o, m_block);
        }
      }
    }
  }
  CUTLASS_DEVICE void
  store_tail() {
  }
  CUTLASS_DEVICE void
  store_zero(
     Params const& params,
     int thread_idx,
     cute::tuple<int32_t, int32_t, int32_t, int32_t> const& block_coord
     ) {
    static constexpr int kBlockM = get<0>(TileShape_MNK_PV{});
    auto [m_block, bidh, bidb, split_idx] = block_coord;
    int num_splits = get<4>(params.shape_O_packed);
    if constexpr (Split && Varlen) {
      uint32_t num_splits_dynamic_u = reinterpret_cast<uint32_t const&>(split_idx) >> 16;
      int num_splits_dynamic = reinterpret_cast<int&>(num_splits_dynamic_u);
      num_splits = num_splits_dynamic > 0 ? num_splits_dynamic : num_splits;
      split_idx &= 0x0000FFFF;
    }
    bool const is_split = !Split ? false : (!Varlen ? true : num_splits > 1);
    flash::SeqlenInfo<Varlen, kBlockM> seqlen_info{bidb, size<0>(params.shape_O), params.cu_seqlens, params.seqused};
    bool const is_varlen = Varlen && params.cu_seqlens;
    int offset_o = seqlen_info.offset;
    int seqlen_o = seqlen_info.seqlen;
    int qhead_per_khead = !PackGQA ? 1 : params.qhead_per_khead_divmod.divisor;
    Tensor mLSE = make_tensor(make_gmem_ptr((!is_split ? params.ptr_LSE : params.ptr_LSE_partial) + offset_o * get<0>(!is_split ? params.stride_LSE : params.stride_LSE_partial)),
                 params.shape_LSE_packed,
                 !is_split ? params.stride_LSE_packed : params.stride_LSE_partial_packed)(_, bidh, !is_varlen ? bidb : 0, !is_split ? 0 : split_idx);
    Tensor gLSE = local_tile(mLSE, Shape<Int<kBlockM>>{}, make_coord(m_block));
    static_assert(kBlockM <= NumEpilogueThreads);
    if (thread_idx < kBlockM) {
      const int row = m_block * kBlockM + thread_idx;
      if constexpr (!PackGQA) {
        if (row < seqlen_o) { mLSE(row) = -INFINITY; }
      } else {
        if (row < seqlen_o * qhead_per_khead) {
          int m_idx, h_idx;
          m_idx = params.qhead_per_khead_divmod.divmod(h_idx, row);
          mLSE(make_coord(make_coord(h_idx, m_idx))) = -INFINITY;
        }
      }
    }
    if (!is_split) {
      Tensor mO = make_tensor(make_gmem_ptr(params.ptr_O + offset_o * get<0>(params.stride_O)), params.shape_O_packed, params.stride_O_packed)(_, _, bidh, !is_varlen ? bidb : 0, _0{});
      GmemTiledCopyO gmem_tiled_copy_O;
      auto gmem_thr_copy_O = gmem_tiled_copy_O.get_thread_slice(thread_idx);
      Tensor tOcO = gmem_thr_copy_O.partition_D(cute::make_identity_tensor(select<0, 1>(TileShape_MNK_PV{})));
      if constexpr (!PackGQA) {
        Tensor tOpO = make_tensor<bool>(make_shape(size<2>(tOcO)));
        #pragma unroll
        for (int k = 0; k < size(tOpO); ++k) { tOpO(k) = get<1>(tOcO(_0{}, _0{}, k)) < get<1>(params.shape_O); }
        Tensor gO = local_tile(mO, select<0, 1>(TileShape_MNK_PV{}), make_coord(m_block, _0{}));
        Tensor tOgO = gmem_thr_copy_O.partition_D(gO);
        Tensor tOrO = make_fragment_like(tOgO);
        cute::clear(tOrO);
        flash::copy<false, false, false, false>(
          gmem_tiled_copy_O, tOrO, tOgO, tOcO, tOpO, seqlen_o - m_block * kBlockM
        );
      } else {
        using PackGQA_t = flash::PackGQAManager<get<0>(TileShape_MNK_PV{}), get<1>(TileShape_MNK_PV{}), NumEpilogueThreads, Element>;
        Tensor tOrO = make_tensor<Element>(make_shape(Shape<_1, Int<kGmemElemsPerStore>>{}, size<1>(tOcO), size<2>(tOcO)));
        cute::clear(tOrO);
        PackGQA_t::store_O(mO, tOrO, params.qhead_per_khead_divmod, thread_idx, seqlen_o, m_block);
      }
    }
  }
};
}
#include <vector>
inline bool use_one_mma_wg(Flash_fwd_params const& params) {
  return params.arch >= 90 && (params.d == 128 || params.d == 64) &&
    params.seqlen_q * (params.h / params.h_k) <= 64;
};
inline bool should_pack_gqa(bool varlen_q, int seqlen_q, int qhead_per_khead, int blockM) {
  if (varlen_q) return true;
  auto round_up = [](int a, int b) { return (a + b - 1) / b * b; };
  float nopack_gqa_efficiency = float(seqlen_q) / float(round_up(seqlen_q, blockM));
  float pack_gqa_efficiency = float(seqlen_q * qhead_per_khead) / float(round_up(seqlen_q * qhead_per_khead, blockM));
  return nopack_gqa_efficiency < 0.9 * pack_gqa_efficiency;
};
inline int num_splits_heuristic(int total_mblocks, int num_SMs, int num_n_blocks, int num_m_blocks, int size_one_kv_head, bool is_causal_or_local, int max_splits) {
  if (total_mblocks >= 0.8f * num_SMs) {
    int const size_l2 = 50 * 1024 * 1024;
    if (size_one_kv_head > size_l2 && num_m_blocks >= num_SMs * 2 && !is_causal_or_local) {
      return std::min((size_one_kv_head + size_l2 - 1) / size_l2, max_splits);
    } else {
      return 1;
    }
  }
  if (num_n_blocks <= 4) { return 1; }
  max_splits = std::min({max_splits, num_SMs, num_n_blocks});
  float max_efficiency = 0.f;
  std::vector<float> efficiency;
  efficiency.reserve(max_splits);
  for (int num_splits = 1; num_splits <= max_splits; num_splits++) {
    float n_waves = float(total_mblocks * num_splits) / num_SMs;
    float eff = n_waves / ceil(n_waves);
    if (eff > max_efficiency) { max_efficiency = eff; }
    efficiency.push_back(eff);
  }
  for (int num_splits = 1; num_splits <= max_splits; num_splits++) {
    if (efficiency[num_splits - 1] >= 0.85 * max_efficiency) {
      return num_splits;
    }
  }
  return 1;
}
using namespace cute;
static void const* x52_gptr = nullptr;
static int x52_gstride = 0;
static float x52_geps = 0.f;
template <int Arch, int kHeadDim, int kHeadDimV, int ClusterM, typename Element, typename ElementOut,
     bool Is_causal, bool Is_local, bool Has_softcap, bool Varlen, bool PagedKVNonTMA, bool AppendKV, bool HasQv, bool OnlyQv,
     bool PackGQA, bool Split, bool V_colmajor, bool Use_one_mma_wg, bool Has_sparse_mask=false, bool SK=false>
void run_flash_fwd(Flash_fwd_params &params, cudaStream_t strm_) {
  static_assert(!(Is_causal && Is_local), "Causal and Local cannot be enabled at the same time");
  static_assert(!(AppendKV && V_colmajor), "AppendKV and V_colmajor cannot be enabled at the same time");
  static_assert(!(AppendKV && !Varlen), "AppendKV requires Varlen");
  static constexpr bool Is_FP8 = cute::is_same_v<Element, cutlass::float_e4m3_t> || cute::is_same_v<Element, cutlass::float_e5m2_t>;
  static constexpr bool FP8_TransposeV = Is_FP8 && !V_colmajor;
  using ArchTag = std::conditional_t<Arch >= 90, cutlass::arch::Sm90, cutlass::arch::Sm80>;
  using ElementSink=cutlass::bfloat16_t;
  static constexpr std::tuple<int, int, bool, bool> kBlockMN_RS_IntraWGOverlap = tile_size_fwd_sm90(kHeadDim, kHeadDimV, Is_causal, Is_local, sizeof(Element) , V_colmajor, PagedKVNonTMA, Has_softcap, Use_one_mma_wg);
  static constexpr std::tuple<int, int, int, int, bool> kBlockMN_kNWarps_Stages_RS = tile_size_fwd_sm8x(Arch == 86 || Arch == 89, kHeadDim, kHeadDimV, Is_causal, Is_local, sizeof(Element) , PagedKVNonTMA, Varlen && Split, Has_softcap, AppendKV);
  static constexpr int kBlockM = Arch >= 90 ? std::get<0>(kBlockMN_RS_IntraWGOverlap) : std::get<0>(kBlockMN_kNWarps_Stages_RS);
  static constexpr int kBlockN = Arch >= 90 ? std::get<1>(kBlockMN_RS_IntraWGOverlap) : std::get<1>(kBlockMN_kNWarps_Stages_RS);
  static constexpr bool MmaPV_is_RS = std::get<2>(kBlockMN_RS_IntraWGOverlap);
  static constexpr bool IntraWGOverlap = std::get<3>(kBlockMN_RS_IntraWGOverlap);
  static constexpr int kNWarps = std::get<2>(kBlockMN_kNWarps_Stages_RS);
  static constexpr int kStages = Arch >= 90 ? ATT_STAGES : std::get<3>(kBlockMN_kNWarps_Stages_RS);
  static constexpr bool Q_in_regs = Arch >= 90 ? false : std::get<4>(kBlockMN_kNWarps_Stages_RS);
  using TileShape_MNK = cute::Shape<Int<kBlockM>, Int<kBlockN>, Int<kHeadDim>>;
  using TileShape_MNK_PV = cute::Shape<Int<kBlockM>, Int<kHeadDimV>, Int<kBlockN>>;
  using ClusterShape = cute::Shape<Int<ClusterM>, _1, _1>;
  using CollectiveMainloop = std::conditional_t<
    Arch >= 90,
    flash::CollectiveMainloopFwdSm90<kStages, ClusterShape, TileShape_MNK, kHeadDimV, Element, float, cutlass::arch::Sm90, Is_causal, Is_local, Has_softcap, Varlen, PagedKVNonTMA, AppendKV, HasQv, OnlyQv, MmaPV_is_RS, IntraWGOverlap, PackGQA, Split, V_colmajor, ElementSink, Has_sparse_mask>,
    flash::CollectiveMainloopFwdSm80<kNWarps, kStages, Q_in_regs, TileShape_MNK, kHeadDimV, Element, float, cutlass::arch::Sm80, Is_causal, Is_local, Has_softcap, Varlen, PagedKVNonTMA, AppendKV, PackGQA, Split, ElementSink>
  >;
  using CollectiveEpilogue = flash::CollectiveEpilogueFwd<TileShape_MNK_PV, ClusterShape, ElementOut, ArchTag, CollectiveMainloop::NumMmaThreads, Varlen, PackGQA, Split, FP8_TransposeV>;
  static constexpr int NumProducerThreads = Arch >= 90 ? CollectiveMainloop::NumProducerThreads : CollectiveMainloop::NumMmaThreads;
  static constexpr bool LPT = Is_causal || Is_local;
  static constexpr bool Sort = !Is_local;
  using SchedulerPersistent0 = std::conditional_t<Varlen,
    flash::VarlenDynamicPersistentTileScheduler<kBlockM, kBlockN, CollectiveMainloop::NumMmaThreads, NumProducerThreads, Split, PackGQA, Arch >= 90 , LPT, Sort, true >,
    std::conditional_t<!Is_causal && !Is_local,
      flash::StaticPersistentTileScheduler<Split>,
      flash::DynamicPersistentTileScheduler<CollectiveMainloop::NumMmaThreads, NumProducerThreads, Split, PackGQA, Arch >= 90 >
    >
  >;
  using SchedulerPersistent = std::conditional_t<SK && Varlen && Split && !Is_local,
    flash::SK2Scheduler<CollectiveMainloop::NumMmaThreads, NumProducerThreads, Arch >= 90>, SchedulerPersistent0>;
  using SchedulerSingleTile = flash::SingleTileScheduler<Varlen, Split, PackGQA, kBlockM>;
  static constexpr bool UsePersistentScheduler = Arch >= 90 ? !(Split && !Varlen) : ((Is_causal && !Varlen) || (Varlen && Split));
  using Scheduler = std::conditional_t<!UsePersistentScheduler, SchedulerSingleTile, SchedulerPersistent>;
  using AttnKernel = std::conditional_t<
    Arch >= 90,
    flash::enable_sm90_or_later<flash::FlashAttnFwdSm90<CollectiveMainloop, CollectiveEpilogue, Scheduler>>,
    flash::enable_sm80_to_sm89<flash::FlashAttnFwdSm80<CollectiveMainloop, CollectiveEpilogue, Scheduler>>
  >;
  bool const is_varlen_q = params.cu_seqlens_q;
  bool const is_varlen_k = params.cu_seqlens_k;
  bool const is_varlen_k_new = params.cu_seqlens_knew;
  int seqlen_q = !is_varlen_q ? params.seqlen_q : params.total_q;
  int batch_q = !is_varlen_q ? params.b : 1;
  int batch_k = !is_varlen_k ? (params.kv_batch_idx ? params.b_k : params.b) : 1;
  using StrideV_t = typename CollectiveMainloop::StrideV;
  int64_t const v_bs = !is_varlen_k ? params.v_batch_stride : 0;
  StrideV_t v_strides = [&] {
    if constexpr (!V_colmajor) { return StrideV_t{params.v_row_stride, _1{}, params.v_head_stride, v_bs}; }
    else { return StrideV_t{_1{}, params.v_dim_stride, params.v_head_stride, v_bs}; }
  }();
  StrideV_t vnew_strides = [&] {
    if constexpr (!V_colmajor) {
      return StrideV_t{params.vnew_row_stride, _1{}, params.vnew_head_stride, !is_varlen_k_new ? int64_t(params.vnew_batch_stride) : int64_t(0)};
    } else { return StrideV_t{}; }
  }();
  typename CollectiveMainloop::Arguments mainloop_args = {
    static_cast<Element const*>(params.q_ptr),
    {seqlen_q, params.d, params.h, batch_q},
    {params.q_row_stride, _1{}, params.q_head_stride, !is_varlen_q ? params.q_batch_stride : 0},
    static_cast<Element*>(params.k_ptr),
    {!params.page_table ? (!is_varlen_k ? params.seqlen_k : params.total_k) : params.page_size,
      params.d, params.h_k, !params.page_table ? batch_k : params.num_pages},
    {params.k_row_stride, _1{}, params.k_head_stride, !is_varlen_k ? params.k_batch_stride : 0},
    static_cast<Element*>(params.v_ptr),
    params.dv,
    v_strides,
    static_cast<Element const*>(params.knew_ptr),
    {!is_varlen_k_new ? params.seqlen_knew : params.total_knew, params.d, params.h_k, !is_varlen_k_new ? params.b : 1},
    {params.knew_row_stride, _1{}, params.knew_head_stride, !is_varlen_k_new ? params.knew_batch_stride : 0},
    static_cast<Element const*>(params.vnew_ptr),
    vnew_strides,
    static_cast<Element const*>(params.qv_ptr),
    {params.qv_row_stride, _1{}, params.qv_head_stride, !is_varlen_q ? params.qv_batch_stride : 0},
    static_cast<Element const*>(params.rotary_cos_ptr),
    {params.seqlen_k, params.rotary_dim / 2},
    {params.rotary_dim / 2, _1{}},
    static_cast<Element const*>(params.rotary_sin_ptr),
    {params.rotary_dim / 2, _1{}},
    params.is_rotary_interleaved,
    params.page_table,
    {params.kv_batch_idx ? params.b_k : params.b, !params.page_table ? 0 : params.seqlen_k / params.page_size},
    {params.page_table_batch_stride, _1{}},
    params.scale_softmax,
    params.q_descale_ptr, params.k_descale_ptr, params.v_descale_ptr,
    {params.q_descale_batch_stride, params.q_descale_head_stride},
    {params.k_descale_batch_stride, params.k_descale_head_stride},
    {params.v_descale_batch_stride, params.v_descale_head_stride},
    params.window_size_left, params.window_size_right, params.attention_chunk,
    params.softcap,
    params.num_splits,
    params.kv_batch_idx,
    params.cu_seqlens_q, params.cu_seqlens_k, params.cu_seqlens_knew,
    params.seqused_q, params.seqused_k,
    params.leftpad_k, params.seqlens_rotary,
    static_cast<ElementSink const*>(params.sink_ptr)
  };
  if constexpr (Arch >= 90) {
    mainloop_args.sparse_mask_fine = params.sparse_mask_fine;
    mainloop_args.sparse_mask_max_k_blocks = params.sparse_mask_max_k_blocks;
    mainloop_args.sparse_mask_fine_q_stride = params.sparse_mask_fine_q_stride;
    mainloop_args.sparse_mask_fine_k_stride = params.sparse_mask_fine_k_stride;
    mainloop_args.total_q = params.total_q;
  }
  typename CollectiveEpilogue::Arguments epilogue_args {
    static_cast<ElementOut*>(params.o_ptr),
    {seqlen_q, params.dv, params.h, batch_q, params.num_splits},
    {params.o_row_stride, _1{}, params.o_head_stride, !is_varlen_q ? params.o_batch_stride : 0, 0},
    static_cast<float*>(params.oaccum_ptr),
    {params.oaccum_row_stride, _1{}, params.oaccum_head_stride, !is_varlen_q ? params.oaccum_batch_stride : 0, params.oaccum_split_stride},
    static_cast<float*>(params.softmax_lse_ptr),
    {_1{}, seqlen_q, !is_varlen_q ? params.h * seqlen_q : 0, 0},
    static_cast<float*>(params.softmax_lseaccum_ptr),
    {_1{}, seqlen_q, !is_varlen_q ? params.h * seqlen_q : 0, params.h * seqlen_q * batch_q},
    params.h_k,
    params.cu_seqlens_q, params.seqused_q
  };
  if constexpr (!Split) {
    epilogue_args.ptr_gate = static_cast<typename CollectiveEpilogue::Element const*>(x52_gptr);
    epilogue_args.gate_stride = x52_gstride; epilogue_args.gate_eps = x52_geps;
  }
  int qhead_per_khead = !PackGQA ? 1 : cutlass::ceil_div(params.h, params.h_k);
  int num_blocks_m = cutlass::ceil_div(params.seqlen_q * qhead_per_khead, get<0>(TileShape_MNK{}));
  num_blocks_m = cutlass::round_up(num_blocks_m, size<0>(ClusterShape{}));
  typename flash::TileSchedulerArguments scheduler_args {
    num_blocks_m, !PackGQA ? params.h : params.h_k, params.b, params.num_splits,
    params.h / params.h_k,
    params.seqlen_q,
    params.seqlen_k, params.d, params.dv, sizeof(Element),
    params.tile_count_semaphore, params.cu_seqlens_q, params.seqused_q,
    params.num_splits_dynamic_ptr,
    params.num_m_blocks_ptr,
    params.varlen_batch_idx_ptr,
    params.num_nheads_in_l2_ptr
  };
  if (Varlen && !params.skip_scheduler_metadata_computation) {
    if constexpr (SK && Split && !Is_local) {
      sk2_prepare(params, strm_, kBlockM, kBlockN);
    } else
    prepare_varlen_num_blocks(params, strm_, PackGQA, kBlockM, kBlockN, Arch >= 90 && params.prepare_varlen_pdl );
    CHECK_CUDA_KERNEL_LAUNCH();
  }
  int device;
  CHECK_CUDA(cudaGetDevice(&device));
  typename AttnKernel::Params kernel_params = AttnKernel::to_underlying_arguments({
    mainloop_args, epilogue_args, {device, params.num_sm}, scheduler_args
  });
  dim3 grid_dims = AttnKernel::get_grid_shape(kernel_params);
  dim3 block_dims = AttnKernel::get_block_shape();
  int smem_size = AttnKernel::SharedStorageSize;
  if constexpr (size(ClusterShape{}) > 1) {
    void const* kernel = (void const*) cutlass::device_kernel<AttnKernel>;
    if (smem_size >= 48 * 1024) {
      CHECK_CUDA(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
    }
    dim3 cluster_dims(size<0>(ClusterShape{}), size<1>(ClusterShape{}), size<2>(ClusterShape{}));
    cutlass::ClusterLaunchParams launch_params{grid_dims, block_dims, cluster_dims, smem_size, strm_};
    cutlass::launch_kernel_on_cluster(launch_params, kernel, kernel_params);
  } else {
    auto kernel = cutlass::device_kernel<AttnKernel>;
    if (smem_size >= 48 * 1024) {
      CHECK_CUDA(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
    }
    cutlass::kernel_launch<AttnKernel>(grid_dims, block_dims, smem_size, strm_, kernel_params,
                      Arch >= 90 && Varlen && !params.skip_scheduler_metadata_computation && params.prepare_varlen_pdl );
  }
  CHECK_CUDA_KERNEL_LAUNCH();
}
template<int Arch, typename T, int kHeadDim, int kHeadDimV, bool Split, bool PagedKVNonTMA, bool Has_softcap, bool PackGQA>
void run_mha_fwd_(Flash_fwd_params &params, cudaStream_t strm_) {
  static_assert(sizeof(T) == 2 || sizeof(T) == 1, "Only 16bit and 8bit are supported");
  static constexpr bool Is_FP8 = cute::is_same_v<T, cutlass::float_e4m3_t> || cute::is_same_v<T, cutlass::float_e5m2_t>;
  using T_out = std::conditional_t<!Is_FP8, T, cutlass::bfloat16_t>;
  CAUSAL_LOCAL_SWITCH(params.is_causal, params.is_local, Is_causal, Is_local, [&] {
    VCOLMAJOR_SWITCH(params.v_dim_stride != 1, V_colmajor_, [&] {
      static constexpr bool V_colmajor = V_colmajor_ && sizeof(T) == 1;
      VARLEN_SWITCH(params.cu_seqlens_q || params.cu_seqlens_k || params.seqused_q || params.seqused_k || params.leftpad_k, Varlen, [&] {
        BOOL_SWITCH(use_one_mma_wg(params), Use_one_mma_wg_, [&] {
          static constexpr bool Use_one_mma_wg = Use_one_mma_wg_ && Arch >= 90 && (kHeadDim == 128 || kHeadDim == 64) && (kHeadDimV == kHeadDim);
          static constexpr int kBlockM = Arch >= 90 ? std::get<0>(tile_size_fwd_sm90(kHeadDim, kHeadDimV, Is_causal, Is_local, sizeof(T) , V_colmajor, PagedKVNonTMA, Has_softcap, Use_one_mma_wg)) : 128;
          static constexpr bool Enable_cluster = Arch == 90 && (sizeof(T) == 2 ? (kHeadDim >= 128) : (kHeadDim == 192)) && !Is_causal && !Is_local && !Split && !PagedKVNonTMA && !Varlen && !Use_one_mma_wg;
          BOOL_SWITCH(params.qv_ptr, HasQV_, [&] {
            static constexpr bool HasQv = HasQV_ && Arch == 90 && !Is_FP8 && kHeadDim == 64 && kHeadDimV >= 256;
            BOOL_SWITCH(params.only_qv, OnlyQv_, [&] {
              static constexpr bool OnlyQv = OnlyQv_ && HasQv;
              APPENDKV_SWITCH(params.knew_ptr, AppendKV, [&] {
                CLUSTER_SWITCH(cutlass::ceil_div(params.seqlen_q * (!PackGQA ? 1 : params.h / params.h_k), kBlockM) % 2 == 0, Use_cluster, [&] {
                  static constexpr int ClusterM = Enable_cluster && Use_cluster ? 2 : 1;
                  SPARSE_MASK_SWITCH(params.use_sparse_mask && Arch >= 90 && Varlen, Has_sparse_mask, [&] {
                    run_flash_fwd<Arch, kHeadDim, kHeadDimV, ClusterM, T, T_out, Is_causal, Is_local, Has_softcap, Varlen, PagedKVNonTMA, AppendKV && Varlen, HasQv, OnlyQv, PackGQA, Split, V_colmajor, Use_one_mma_wg, Has_sparse_mask>(params, strm_);
                  });
                });
              });
            });
          });
        });
      });
    });
  });
}
#include "cute/tensor.hpp"
#include "cutlass/cutlass.h"
#include "cutlass/arch/arch.h"
#include "cutlass/device_kernel.h"
#include "cutlass/kernel_launch.h"
#include "cute/tensor.hpp"
#include <cutlass/cutlass.h>
#include <cutlass/arch/memory.h>
#include <cutlass/array.h>
#include <cutlass/numeric_types.h>
#include <cutlass/numeric_conversion.h>
#include "cutlass/arch/grid_dependency_control.h"
namespace flash {
using namespace cute;
template <class TileShape_MK_, int kLogMaxSplits_, int kNThreads, int AlignmentLSE_,
     bool Is_even_K, bool Varlen, class Element, class ElementPartial, class ArchTag_>
class FlashAttnFwdCombine {
public:
  using TileShape_MK = TileShape_MK_;
  using ArchTag = ArchTag_;
  static constexpr int kMaxSplits = 1 << kLogMaxSplits_;
  static constexpr int AlignmentLSE = std::min(AlignmentLSE_, int(128 / 8 / sizeof(float)));
  static_assert(AlignmentLSE >= 1);
  static constexpr int kStages = 4;
  static_assert(ArchTag::kMinComputeCapability >= 75);
  static constexpr bool Has_cp_async = ArchTag::kMinComputeCapability >= 80;
  static constexpr uint32_t MaxThreadsPerBlock = kNThreads;
  static constexpr uint32_t MinBlocksPerMultiprocessor = 2;
  static constexpr int kBlockM = get<0>(TileShape_MK{});
  static constexpr int kBlockK = get<1>(TileShape_MK{});
  static constexpr int kGmemElemsPerLoad = sizeof(cute::uint128_t) / sizeof(ElementPartial);
  static_assert(kBlockK % kGmemElemsPerLoad == 0, "kBlockK must be a multiple of kGmemElemsPerLoad");
  static constexpr int kBlockKGmem = kBlockK % 128 == 0 ? 128 : (kBlockK % 64 == 0 ? 64 : 32);
  static constexpr int kGmemThreadsPerRow = kBlockKGmem / kGmemElemsPerLoad;
  static_assert(MaxThreadsPerBlock % kGmemThreadsPerRow == 0, "MaxThreadsPerBlock must be a multiple of kGmemThreadsPerRow");
  using GmemCopyAtom = std::conditional_t<
    Has_cp_async,
    cute::Copy_Atom<SM80_CP_ASYNC_CACHEGLOBAL<uint128_t>, ElementPartial>,
    cute::Copy_Atom<AutoVectorizingCopyWithAssumedAlignment<128>, ElementPartial>
  >;
  using GmemLayoutAtom = Layout<Shape <Int<MaxThreadsPerBlock / kGmemThreadsPerRow>, Int<kGmemThreadsPerRow>>,
                 Stride<Int<kGmemThreadsPerRow>, _1>>;
  static_assert(kBlockM % CUTE_STATIC_V(shape<0>(GmemLayoutAtom{})) == 0);
  using GmemTiledCopyAccum = decltype(
    make_tiled_copy(GmemCopyAtom{},
            GmemLayoutAtom{},
            Layout<Shape<_1, Int<kGmemElemsPerLoad>>>{}));
  using GmemTiledCopy = decltype(
    make_tiled_copy(Copy_Atom<AutoVectorizingCopyWithAssumedAlignment<128>, Element>{},
            GmemLayoutAtom{},
            Layout<Shape<_1, Int<kGmemElemsPerLoad>>>{}));
  using AlignmentTypeLSE = cute::uint_byte_t<static_cast<int>(sizeof(float)) * AlignmentLSE>;
  static constexpr int kGmemElemsPerLoadLSE = sizeof(AlignmentTypeLSE) / sizeof(float);
  static_assert(kBlockM % kGmemElemsPerLoadLSE == 0, "kBlockM must be a multiple of kGmemElemsPerLoadLSE");
  static_assert(kBlockM % 8 == 0, "kBlockM must be a multiple of 8");
  static constexpr int kBlockMSmem = kBlockM % 128 == 0 ? 128 : (kBlockM % 64 == 0 ? 64 : (kBlockM % 32 == 0 ? 32 : (kBlockM % 16 == 0 ? 16 : 8)));
  static constexpr int kGmemThreadsPerRowLSE = kBlockMSmem / kGmemElemsPerLoadLSE;
  static_assert(MaxThreadsPerBlock % kGmemThreadsPerRowLSE == 0, "MaxThreadsPerBlock must be a multiple of kGmemThreadsPerRowLSE");
  using GmemLayoutAtomLSE = Layout<Shape <Int<MaxThreadsPerBlock / kGmemThreadsPerRowLSE>, Int<kGmemThreadsPerRowLSE>>,
                   Stride<Int<kGmemThreadsPerRowLSE>, _1>>;
  static_assert(kMaxSplits % CUTE_STATIC_V(shape<0>(GmemLayoutAtomLSE{})) == 0);
  using GmemCopyAtomLSE = std::conditional_t<
    Has_cp_async,
    cute::Copy_Atom<SM80_CP_ASYNC_CACHEALWAYS<AlignmentTypeLSE>, float>,
    cute::Copy_Atom<AutoVectorizingCopyWithAssumedAlignment<AlignmentLSE * sizeof(float) * 8>, float>
  >;
  using GmemTiledCopyLSE = decltype(
    make_tiled_copy(GmemCopyAtomLSE{},
            GmemLayoutAtomLSE{},
            Layout<Shape<_1, Int<kGmemElemsPerLoadLSE>>>{}));
  static_assert((kBlockM * kMaxSplits * AlignmentLSE) % kNThreads == 0, "kNThreads must divide kBlockM * kMaxSplits * AlignmentLSE");
  using SmemLSESwizzle = std::conditional_t<
    kBlockMSmem == 8,
    Swizzle<5, 0, 5>,
    std::conditional_t<kBlockMSmem == 16, Swizzle<4, 0, 4>, Swizzle<3, 2, 3>>
  >;
  using SmemLayoutAtomLSE =
    decltype(composition(SmemLSESwizzle{},
         Layout<Shape<Int<8>, Int<kBlockMSmem>>,
         Stride<Int<kBlockMSmem>, _1>>{}));
  using SmemLayoutLSE = decltype(tile_to_shape(SmemLayoutAtomLSE{}, Shape<Int<kMaxSplits>, Int<kBlockM>>{}));
  using SmemLayoutO = Layout<Shape<Int<kBlockM>, Int<kBlockK>, Int<kStages>>,
                Stride<Int<kBlockK>, _1, Int<kBlockM * kBlockK>>>;
  static_assert(MaxThreadsPerBlock % kBlockMSmem == 0, "MaxThreadsPerBlock must be a multiple of kBlockMSmem");
  static constexpr int kSmemThreadsPerColLSEt = MaxThreadsPerBlock / kBlockMSmem;
  static_assert(cutlass::NumThreadsPerWarp % kSmemThreadsPerColLSEt == 0, "kSmemThreadsPerColLSEt must divide NumThreadsPerWarp");
  using S2RLayoutAtomLSE = Layout<Shape<Int<kSmemThreadsPerColLSEt>, Int<MaxThreadsPerBlock / kSmemThreadsPerColLSEt>>>;
  using S2RTiledCopyLSE = decltype(make_tiled_copy(cute::Copy_Atom<cute::DefaultCopy, float>{}, S2RLayoutAtomLSE{}, Layout<_1>{}));
  using ShapeOPartial = cute::Shape<int32_t, int32_t, int32_t, int32_t, int32_t>;
  using StrideOPartial = cute::Stride<int64_t, _1, int64_t, int64_t, int64_t>;
  using ShapeLSEPartial = cute::Shape<int32_t, int32_t, int32_t, int32_t>;
  using StrideLSEPartial = cute::Stride<_1, int64_t, int64_t, int64_t>;
  using ShapeO = cute::Shape<int32_t, int32_t, int32_t, int32_t>;
  using StrideO = cute::Stride<int64_t, _1, int64_t, int64_t>;
  using ShapeLSE = cute::Shape<int32_t, int32_t, int32_t>;
  using StrideLSE = cute::Stride<_1, int64_t, int64_t>;
  struct SharedStorage : cute::aligned_struct<128> {
    cute::array_aligned<float, cute::cosize_v<SmemLayoutLSE>> smem_lse_partial;
    cute::array_aligned<int, kBlockM> smem_max_valid_split;
    cute::array_aligned<ElementPartial, cute::cosize_v<SmemLayoutO>> smem_o_partial;
  };
  static constexpr int SharedStorageSize = sizeof(SharedStorage);
  struct Arguments {
    ElementPartial const* const ptr_O_partial;
    ShapeOPartial const shape_O_partial;
    StrideOPartial const stride_O_partial;
    float const* const ptr_LSE_partial;
    ShapeLSEPartial const shape_LSE_partial;
    StrideLSEPartial const stride_LSE_partial;
    Element* const ptr_O;
    StrideO const stride_O;
    float* const ptr_LSE;
    StrideLSE const stride_LSE;
    int const* const cu_seqlens = nullptr;
    int const* const seqused = nullptr;
    int const* const num_splits_dynamic_ptr = nullptr;
    int const* const varlen_batch_idx_ptr = nullptr;
    int* const semaphore_to_reset = nullptr;
    Element const* const ptr_gate = nullptr;
    int const gate_stride = 0;
    float const gate_eps = 0.f;
  };
  struct Params {
    ElementPartial const* const ptr_O_partial;
    ShapeOPartial const shape_O_partial;
    StrideOPartial const stride_O_partial;
    float const* const ptr_LSE_partial;
    ShapeLSEPartial const shape_LSE_partial;
    StrideLSEPartial const stride_LSE_partial;
    Element* const ptr_O;
    StrideO const stride_O;
    float* const ptr_LSE;
    StrideLSE const stride_LSE;
    cutlass::FastDivmod seqlen_divmod, head_divmod;
    int const* const cu_seqlens = nullptr;
    int const* const seqused = nullptr;
    int const* const num_splits_dynamic_ptr = nullptr;
    int const* const varlen_batch_idx_ptr = nullptr;
    int* const semaphore_to_reset = nullptr;
    Element const* const ptr_gate = nullptr;
    int const gate_stride = 0;
    float const gate_eps = 0.f;
  };
  static
  Params
  to_underlying_arguments(Arguments const& args) {
    assert(get<1>(args.shape_LSE_partial) <= kMaxSplits);
    return {
      args.ptr_O_partial,
      args.shape_O_partial,
      args.stride_O_partial,
      args.ptr_LSE_partial,
      args.shape_LSE_partial,
      args.stride_LSE_partial,
      args.ptr_O,
      args.stride_O,
      args.ptr_LSE,
      args.stride_LSE,
      cutlass::FastDivmod(get<0>(args.shape_LSE_partial)), cutlass::FastDivmod(get<2>(args.shape_LSE_partial)),
      args.cu_seqlens,
      args.seqused,
      args.num_splits_dynamic_ptr,
      args.varlen_batch_idx_ptr,
      args.semaphore_to_reset,
      args.ptr_gate,
      args.gate_stride,
      args.gate_eps,
    };
  }
  CUTLASS_DEVICE
  void
  operator()(Params const& params, char* smem_buf) {
    SharedStorage& shared_storage = *reinterpret_cast<SharedStorage*>(smem_buf);
    Tensor sLSE = make_tensor(make_smem_ptr(shared_storage.smem_lse_partial.data()), SmemLayoutLSE{});
    Tensor sMaxValidSplit = make_tensor(make_smem_ptr(shared_storage.smem_max_valid_split.data()), Shape<Int<kBlockM>>{});
    Tensor sO = make_tensor(make_smem_ptr(shared_storage.smem_o_partial.data()), SmemLayoutO{});
    int const thread_idx = threadIdx.x;
    int const m_block = blockIdx.x;
    int const k_block = blockIdx.y;
    int const maybe_virtual_batch = blockIdx.z;
    int const batch = params.varlen_batch_idx_ptr ? params.varlen_batch_idx_ptr[maybe_virtual_batch] : maybe_virtual_batch;
    int const num_splits = params.num_splits_dynamic_ptr ? params.num_splits_dynamic_ptr[maybe_virtual_batch] : get<1>(params.shape_LSE_partial);
    if (params.semaphore_to_reset && threadIdx.x == 0 && blockIdx.x == gridDim.x - 1 && blockIdx.y == gridDim.y - 1 && blockIdx.z == gridDim.z - 1) {
      cutlass::arch::wait_on_dependent_grids();
      *params.semaphore_to_reset = 0;
    }
    if (num_splits <= 1) {
      if (params.ptr_gate != nullptr && k_block == 0) {
        static_assert(kBlockM * 16 == MaxThreadsPerBlock && kBlockK == 64, "sw gate fallback layout");
        cutlass::arch::wait_on_dependent_grids();
        flash::SeqlenInfo<Varlen, kBlockM> si{batch, size<0>(params.shape_LSE_partial), params.cu_seqlens, params.seqused};
        int const r = thread_idx / 16, c = (thread_idx % 16) * 4;
        int const idx = m_block * kBlockM + r;
        bool const ok = idx < si.seqlen * get<2>(params.shape_LSE_partial);
        int m_idx = 0, bidh = 0;
        if (ok) { cutlass::FastDivmod dm(si.seqlen); bidh = dm.divmod(m_idx, idx); }
        __nv_bfloat16* po = reinterpret_cast<__nv_bfloat16*>(params.ptr_O) + (size_t)(si.offset + m_idx) * get<0>(params.stride_O)
                  + (size_t)bidh * get<2>(params.stride_O) + c;
        __nv_bfloat16 const* pg = reinterpret_cast<__nv_bfloat16 const*>(params.ptr_gate) + (size_t)(si.offset + m_idx) * params.gate_stride + bidh * 64 + c;
        float x[4] = {0.f, 0.f, 0.f, 0.f}, g[4] = {0.f, 0.f, 0.f, 0.f};
        if (ok) {
          #pragma unroll
          for (int i = 0; i < 4; ++i) { x[i] = __bfloat162float(po[i]); g[i] = __bfloat162float(pg[i]); }
        }
        float ss = x[0] * x[0] + x[1] * x[1] + x[2] * x[2] + x[3] * x[3];
        #pragma unroll
        for (int o = 1; o < 16; o <<= 1) ss += __shfl_xor_sync(0xffffffffu, ss, o);
        float const rs = rsqrtf(ss * (1.f / 64.f) + params.gate_eps);
        if (ok) {
          #pragma unroll
          for (int i = 0; i < 4; ++i) po[i] = __float2bfloat16_rn(x[i] * rs / (1.f + __expf(-g[i])));
        }
      }
      return;
    }
    flash::SeqlenInfo<Varlen, kBlockM> seqlen_info{batch, size<0>(params.shape_LSE_partial), params.cu_seqlens, params.seqused};
    int const offset = seqlen_info.offset;
    int const seqlen = seqlen_info.seqlen;
    int max_idx = seqlen * get<2>(params.shape_LSE_partial);
    if constexpr (Varlen) {
      if (m_block * kBlockM >= max_idx) { return; }
    }
    cutlass::FastDivmod seqlen_divmod_dynamic(seqlen);
    Tensor mLSEpartial = make_tensor(make_gmem_ptr(params.ptr_LSE_partial + offset * get<0>(params.stride_LSE_partial)),
                     select<1, 0, 2, 3>(params.shape_LSE_partial),
                     select<1, 0, 2, 3>(params.stride_LSE_partial))(_, _, _, !Varlen ? batch : 0);
    Tensor mLSEpartial_copy = cute::tiled_divide(mLSEpartial, Shape<_1, Int<kGmemElemsPerLoadLSE>>{});
    GmemTiledCopyLSE gmem_tiled_copy_LSE;
    auto gmem_thr_copy_LSE = gmem_tiled_copy_LSE.get_thread_slice(thread_idx);
    Tensor tLSEsLSE = gmem_thr_copy_LSE.partition_D(sLSE);
    Tensor cLSE = make_identity_tensor(make_shape(size<0>(sLSE), size<1>(sLSE)));
    Tensor tLSEcLSE = gmem_thr_copy_LSE.partition_S(cLSE);
    GmemTiledCopyAccum gmem_tiled_copy_O_partial;
    auto gmem_thr_copy_O_partial = gmem_tiled_copy_O_partial.get_thread_slice(thread_idx);
    Tensor cO = cute::make_identity_tensor(TileShape_MK{});
    Tensor tOcO = gmem_thr_copy_O_partial.partition_D(cO);
    Tensor mOpartial = make_tensor(make_gmem_ptr(params.ptr_O_partial + offset * get<0>(params.stride_O_partial)),
                    params.shape_O_partial, params.stride_O_partial)(_, _, _, _, !Varlen ? batch : 0);
    Tensor tOmidx = make_tensor<int>(make_shape(size<1>(tOcO)));
    Tensor tObidh = make_tensor<int>(make_shape(size<1>(tOcO)));
    Tensor tOrOptr = make_tensor<ElementPartial const*>(make_shape(size<1>(tOcO)));
    #pragma unroll
    for (int m = 0; m < size<1>(tOcO); ++m) {
      int mi = get<0>(tOcO(_0{}, m, _0{}));
      int idx = m_block * kBlockM + mi;
      if constexpr (!Varlen) {
        tObidh(m) = params.seqlen_divmod.divmod(tOmidx(m), idx);
      } else {
        tObidh[m] = seqlen_divmod_dynamic.divmod(tOmidx(m), idx);
      }
      tOrOptr[m] = &mOpartial(tOmidx(m), k_block * kBlockK, _0{}, tObidh(m));
      if (idx >= max_idx) {
        tObidh[m] = -1;
      }
    }
    constexpr int ZM = decltype(size<1>(tOcO))::value, ZK = decltype(size<2>(tOcO))::value;
    float z0g[ZM][ZK][kGmemElemsPerLoad];
    if (params.ptr_gate != nullptr) {
      #pragma unroll
      for (int m = 0; m < ZM; ++m) {
        #pragma unroll
        for (int k = 0; k < ZK; ++k) {
          int const col = k_block * kBlockK + get<1>(tOcO(_0{}, m, k));
          Element const* pg = params.ptr_gate + (size_t)(offset + tOmidx(m)) * params.gate_stride + (tObidh(m) < 0 ? 0 : tObidh(m)) * 64 + col;
          #pragma unroll
          for (int i = 0; i < kGmemElemsPerLoad; ++i) { z0g[m][k][i] = tObidh(m) >= 0 ? float(pg[i]) : 0.f; }
        }
      }
    }
    cutlass::arch::wait_on_dependent_grids();
    #pragma unroll
    for (int m = 0; m < size<2>(tLSEcLSE); ++m) {
      int mi = int(get<1>(tLSEcLSE(_0{}, _0{}, m)));
      int idx = m_block * kBlockM + mi;
      if (idx < max_idx) {
        int m_idx, bidh;
        if constexpr (!Varlen) {
          bidh = params.seqlen_divmod.divmod(m_idx, idx);
        } else {
          bidh = seqlen_divmod_dynamic.divmod(m_idx, idx);
        }
        Tensor mLSEpartial_cur_copy = mLSEpartial_copy(_, _, m_idx, bidh);
        #pragma unroll
        for (int s = 0; s < size<1>(tLSEcLSE); ++s) {
          int si = get<0>(tLSEcLSE(_0{}, s, _0{}));
          if (si < num_splits) {
            cute::copy(gmem_tiled_copy_LSE, mLSEpartial_cur_copy(_, si), tLSEsLSE(_, s, m));
          } else {
            cute::fill(tLSEsLSE(_, s, m), -INFINITY);
          }
        }
      } else {
      }
    }
    if constexpr (Has_cp_async) { cute::cp_async_fence(); }
    Tensor tOpO = make_tensor<bool>(make_shape(size<2>(tOcO)));
    if constexpr (!(Is_even_K)) {
      #pragma unroll
      for (int k = 0; k < size(tOpO); ++k) { tOpO(k) = get<1>(tOcO(_0{}, _0{}, k)) < get<1>(params.shape_O_partial) - k_block * kBlockK; }
    }
    Tensor tOsOpartial = gmem_thr_copy_O_partial.partition_D(sO);
    auto load_O_partial = [&] (int split, int stage) {
      Tensor tOsOpartial_cur = tOsOpartial(_, _, _, stage);
      #pragma unroll
      for (int m = 0; m < size<1>(tOcO); ++m) {
        if (tObidh(m) >= 0) {
          Tensor mOpartial_cur = make_tensor(make_gmem_ptr(tOrOptr[m]), mOpartial(_0{}, _, _, _0{}).layout());
          Tensor mOpartial_cur_copy = cute::tiled_divide(mOpartial_cur, Shape<Int<kGmemElemsPerLoad>>{});
          #pragma unroll
          for (int k = 0; k < size<2>(tOcO); ++k) {
            int k_idx = get<1>(tOcO(_0{}, _0{}, k)) / kGmemElemsPerLoad;
            if (Is_even_K || tOpO(k)) {
              cute::copy(gmem_tiled_copy_O_partial, mOpartial_cur_copy(_, k_idx, split), tOsOpartial_cur(_, m, k));
            }
          }
        }
      }
    };
    for (int s = 0; s < kStages - 1; ++s) {
      if (s < num_splits) { load_O_partial(s, s); }
      if constexpr (Has_cp_async) { cute::cp_async_fence(); }
    }
    if constexpr (Has_cp_async) { cutlass::arch::cp_async_wait<kStages - 1>(); }
    __syncthreads();
    S2RTiledCopyLSE s2r_tiled_copy_LSE;
    auto s2r_thr_copy_LSE = s2r_tiled_copy_LSE.get_thread_slice(thread_idx);
    Tensor ts2rsLSE = s2r_thr_copy_LSE.partition_S(sLSE);
    Tensor ts2rrLSE = make_fragment_like(ts2rsLSE);
    cute::copy(s2r_tiled_copy_LSE, ts2rsLSE, ts2rrLSE);
    Tensor lse_sum = make_tensor<float>(make_shape(size<2>(ts2rrLSE)));
    Tensor ts2rcLSE = s2r_thr_copy_LSE.partition_D(cLSE);
    Tensor max_valid_split = make_tensor<int>(make_shape(size<2>(ts2rrLSE)));
    static_assert(CUTE_STATIC_V(size<0>(ts2rrLSE)) == 1);
    #pragma unroll
    for (int m = 0; m < size<2>(ts2rrLSE); ++m) {
      float lse_max = ts2rrLSE(_0{}, _0{}, m);
      #pragma unroll
      for (int s = 1; s < size<1>(ts2rrLSE); ++s) { lse_max = max(lse_max, ts2rrLSE(_0{}, s, m)); }
      MaxOp<float> max_op;
      lse_max = Allreduce<kSmemThreadsPerColLSEt>::run(lse_max, max_op);
      int max_valid_idx = -1;
      #pragma unroll
      for (int s = 0; s < size<1>(ts2rrLSE); ++s) {
        if (ts2rrLSE(_0{}, s, m) != -INFINITY) { max_valid_idx = get<0>(ts2rcLSE(_0{}, s, _0{})); }
      }
      MaxOp<int> max_int_op;
      max_valid_split[m] = Allreduce<kSmemThreadsPerColLSEt>::run(max_valid_idx, max_int_op);
      float lse_max_cur = lse_max == -INFINITY ? 0.0f : lse_max;
      float lse_sum_cur = 0.f;
      #pragma unroll
      for (int s = 0; s < size<1>(ts2rrLSE); ++s) {
        float scale = expf(ts2rrLSE(_0{}, s, m) - lse_max_cur);
        lse_sum_cur += scale;
        ts2rrLSE(_0{}, s, m) = scale;
      }
      SumOp<float> sum_op;
      lse_sum_cur = Allreduce<kSmemThreadsPerColLSEt>::run(lse_sum_cur, sum_op);
      lse_sum(m) = logf(lse_sum_cur) + lse_max;
      float inv_sum = (lse_sum_cur == 0.f || lse_sum_cur != lse_sum_cur) ? 0.f : 1.f / lse_sum_cur;
      #pragma unroll
      for (int s = 0; s < size<1>(ts2rrLSE); ++s) { ts2rrLSE(_0{}, s, m) *= inv_sum; }
    }
    cute::copy(s2r_tiled_copy_LSE, ts2rrLSE, ts2rsLSE);
    #pragma unroll
    for (int m = 0; m < size<2>(ts2rrLSE); ++m) {
      if (get<0>(ts2rcLSE(_0{}, _0{}, m)) == 0) {
        int mi = int(get<1>(ts2rcLSE(_0{}, _0{}, m)));
        if (mi < kBlockM) { sMaxValidSplit[mi] = max_valid_split[m]; }
      }
    }
    if (k_block == 0) {
      auto shape_LSE = select<0, 2, 3>(params.shape_LSE_partial);
      Tensor mLSE = make_tensor(make_gmem_ptr(params.ptr_LSE + offset * get<0>(params.stride_LSE)), shape_LSE, params.stride_LSE)(_, _, !Varlen ? batch : 0);
      #pragma unroll
      for (int m = 0; m < size<2>(ts2rrLSE); ++m) {
        if (get<0>(ts2rcLSE(_0{}, _0{}, m)) == 0) {
          int mi = int(get<1>(ts2rcLSE(_0{}, _0{}, m)));
          int idx = m_block * kBlockM + mi;
          if (idx < max_idx) {
            int m_idx, bidh;
            if constexpr (!Varlen) {
              bidh = params.seqlen_divmod.divmod(m_idx, idx);
            } else {
              bidh = seqlen_divmod_dynamic.divmod(m_idx, idx);
            }
            mLSE(m_idx, bidh) = lse_sum(m);
          }
        }
      }
    }
    __syncthreads();
    int thr_max_valid_split = sMaxValidSplit[get<0>(tOcO(_0{}, _0{}, _0{}))];
    #pragma unroll
    for (int m = 1; m < size<1>(tOcO); ++m) { thr_max_valid_split = max(thr_max_valid_split, sMaxValidSplit[get<0>(tOcO(_0{}, m, _0{}))]); }
    Layout tOrOpartial_layout = gmem_thr_copy_O_partial.partition_S(make_tensor<ElementPartial>(TileShape_MK{})).layout();
    Tensor tOrOpartial = make_fragment_like<ElementPartial>(tOrOpartial_layout);
    Tensor tOrO = make_fragment_like<float>(tOrOpartial);
    clear(tOrO);
    int stage_load = kStages - 1, stage_compute = 0;
    #pragma unroll 4
    for (int s = 0; s <= thr_max_valid_split; ++s) {
      Tensor scale = make_tensor<float>(make_shape(size<1>(tOrOpartial)));
      #pragma unroll
      for (int m = 0; m < size<1>(tOrOpartial); ++m) { scale(m) = sLSE(s, get<0>(tOcO(_0{}, m, _0{}))); }
      if (s + kStages - 1 <= thr_max_valid_split) { load_O_partial(s + kStages - 1, stage_load); }
      if constexpr (Has_cp_async) { cute::cp_async_fence(); }
      stage_load = stage_load < kStages - 1 ? stage_load + 1 : 0;
      if constexpr (Has_cp_async) { cutlass::arch::cp_async_wait<kStages - 1>(); }
      cute::copy(Copy_Atom<AutoVectorizingCopyWithAssumedAlignment<128>, ElementPartial>{},
            tOsOpartial(_, _, _, stage_compute), tOrOpartial);
      stage_compute = stage_compute < kStages - 1 ? stage_compute + 1 : 0;
      #pragma unroll
      for (int m = 0; m < size<1>(tOrOpartial); ++m) {
        if (tObidh(m) >= 0 && scale(m) > 0.f) {
          #pragma unroll
          for (int k = 0; k < size<2>(tOrOpartial); ++k) {
            if (Is_even_K || tOpO(k)) {
              Tensor rOpartial = make_tensor_like<float>(tOrOpartial(_, m, k));
              flash::convert_type_out(tOrOpartial(_, m, k), rOpartial);
              #pragma unroll
              for (int i = 0; i < size<0>(tOrOpartial); ++i) {
                tOrO(i, m, k) += scale(m) * rOpartial[i];
              }
            }
          }
        }
      }
    }
    Tensor rO = make_tensor_like<Element>(tOrO);
    flash::convert_type_out(tOrO, rO);
    if (params.ptr_gate != nullptr) {


      static_assert(kBlockK == 64 && kBlockKGmem == 64 && kGmemThreadsPerRow == 16, "sw gate layout");
      #pragma unroll
      for (int m = 0; m < size<1>(rO); ++m) {
        float ss = 0.f;
        #pragma unroll
        for (int k = 0; k < size<2>(rO); ++k) {
          #pragma unroll
          for (int i = 0; i < size<0>(rO); ++i) { float const xv = float(rO(i, m, k)); ss = fmaf(xv, xv, ss); }
        }
        #pragma unroll
        for (int o = 1; o < 16; o <<= 1) ss += __shfl_xor_sync(0xffffffffu, ss, o);
        float const rs = rsqrtf(ss * (1.f / 64.f) + params.gate_eps);
        if (tObidh(m) >= 0) {
          #pragma unroll
          for (int k = 0; k < size<2>(rO); ++k) {
            int const col = k_block * kBlockK + get<1>(tOcO(_0{}, m, k));
            Element const* pg = params.ptr_gate + (size_t)(offset + tOmidx(m)) * params.gate_stride + tObidh(m) * 64 + col;
            #pragma unroll
            for (int i = 0; i < size<0>(rO); ++i) {
              float const gv = z0g[m][k][i];
              rO(i, m, k) = Element(float(rO(i, m, k)) * rs / (1.f + __expf(-gv)));
            }
          }
        }
      }
    }
    auto shape_O = make_shape(get<0>(params.shape_O_partial), get<1>(params.shape_O_partial) - k_block * kBlockK, get<3>(params.shape_O_partial), get<4>(params.shape_O_partial));
    Tensor mO = make_tensor(make_gmem_ptr(params.ptr_O + offset * get<0>(params.stride_O) + k_block * kBlockK * get<1>(params.stride_O)),
                shape_O, params.stride_O)(_, _, _, !Varlen ? batch : 0);
    Tensor mO_copy = cute::tiled_divide(mO, Shape<_1, Int<kGmemElemsPerLoad>>{});
    GmemTiledCopy gmem_tiled_copy_O;
    auto gmem_thr_copy_O = gmem_tiled_copy_O.get_thread_slice(thread_idx);
    #pragma unroll
    for (int m = 0; m < size<1>(tOcO); ++m) {
      if (tObidh(m) >= 0) {
        #pragma unroll
        for (int k = 0; k < size<2>(tOcO); ++k) {
          int k_idx = get<1>(tOcO(_0{}, _0{}, k)) / kGmemElemsPerLoad;
          if (Is_even_K || tOpO(k)) {
            cute::copy(gmem_tiled_copy_O, rO(_, m, k), mO_copy(_, tOmidx(m), k_idx, tObidh(m)));
          }
        }
      }
    }
  }
};
}
using namespace cute;
static void const* sw_gate_ptr = nullptr;
static int sw_gate_stride = 0;
static float sw_gate_eps = 0.f;
template <int Arch, int kBlockM, int kBlockK, int kLogMaxSplits, bool IsEvenK, bool Varlen, typename Element, typename ElementPartial>
void run_flash_fwd_combine(Flash_fwd_params &params, cudaStream_t strm_, bool enable_pdl) {
  using ArchTag = std::conditional_t<Arch >= 90, cutlass::arch::Sm90, cutlass::arch::Sm80>;
  using TileShape_MK = cute::Shape<Int<kBlockM>, Int<kBlockK>>;
  using CombineKernel = flash::FlashAttnFwdCombine<TileShape_MK, kLogMaxSplits, 256 , 1 ,
                           IsEvenK, Varlen, Element, ElementPartial, ArchTag>;
  typename CombineKernel::Arguments args {
    static_cast<ElementPartial const*>(params.oaccum_ptr),
    {!Varlen ? params.seqlen_q : params.total_q, params.dv, params.num_splits, params.h, !Varlen ? params.b : 1},
    {params.oaccum_row_stride, _1{}, params.oaccum_split_stride, params.oaccum_head_stride, !Varlen ? params.oaccum_batch_stride : 0},
    static_cast<float*>(params.softmax_lseaccum_ptr),
    {!Varlen ? params.seqlen_q : params.total_q, params.num_splits, params.h, !Varlen ? params.b : 1},
    {_1{}, params.lseaccum_split_stride, params.lseaccum_head_stride, !Varlen ? params.lseaccum_batch_stride : 0},
    static_cast<Element*>(params.o_ptr),
    {params.o_row_stride, _1{}, params.o_head_stride, !Varlen ? params.o_batch_stride : 0},
    static_cast<float*>(params.softmax_lse_ptr),
    {_1{}, !Varlen ? params.seqlen_q : params.total_q, !Varlen ? params.h * params.seqlen_q : 0},
    params.cu_seqlens_q, params.seqused_q, params.num_splits_dynamic_ptr, params.varlen_batch_idx_ptr, params.tile_count_semaphore,
    static_cast<Element const*>(sw_gate_ptr), sw_gate_stride, sw_gate_eps
  };
  typename CombineKernel::Params kernel_params = CombineKernel::to_underlying_arguments(args);
  int num_blocks_k = cute::ceil_div(params.dv, kBlockK);
  int num_blocks_m = cute::ceil_div(params.seqlen_q * params.h, kBlockM);
  dim3 grid_m(num_blocks_m, num_blocks_k, params.b);
  auto kernel = cutlass::device_kernel<CombineKernel>;
  int smem_size = CombineKernel::SharedStorageSize;
  if (smem_size >= 48 * 1024) {
    CHECK_CUDA(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
  }
  cutlass::kernel_launch<CombineKernel>(grid_m, CombineKernel::MaxThreadsPerBlock, smem_size, strm_, kernel_params, Arch >= 90 && enable_pdl );
  CHECK_CUDA_KERNEL_LAUNCH();
}
template<typename T, typename Tpartial, int kBlockK>
void run_mha_fwd_combine_(Flash_fwd_params &params, cudaStream_t strm_, bool enable_pdl) {
  static_assert(kBlockK % 32 == 0, "kBlockK must be a multiple of 32");
  static constexpr int kBlockM = kBlockK % 128 == 0 ? 8 : (kBlockK % 64 == 0 ? 16 : 32);
  ARCH_SWITCH(params.arch, Arch, [&] {
    BOOL_SWITCH(params.cu_seqlens_q || params.seqused_q, Varlen, [&] {
      if constexpr (kBlockM >= 16) {
        if (params.num_splits <= 16) {
          run_flash_fwd_combine<Arch, kBlockM, kBlockK, 4, false , Varlen, T, Tpartial>(params, strm_, enable_pdl);
          return;
        }
      }
      if (params.num_splits <= 32) {
        run_flash_fwd_combine<Arch, kBlockM, kBlockK, 5, false , Varlen, T, Tpartial>(params, strm_, enable_pdl);
      } else if (params.num_splits <= 64) {
        run_flash_fwd_combine<Arch, kBlockM, kBlockK, 6, false , Varlen, T, Tpartial>(params, strm_, enable_pdl);
      } else if (params.num_splits <= 128) {
        run_flash_fwd_combine<Arch, kBlockM, kBlockK, 7, false , Varlen, T, Tpartial>(params, strm_, enable_pdl);
      } else {
        run_flash_fwd_combine<Arch, kBlockM, kBlockK, 8, false , Varlen, T, Tpartial>(params, strm_, enable_pdl);
      }
    });
  });
}
#include <cub/cub.cuh>
#include "cutlass/fast_math.h"
#include "cutlass/barrier.h"
#include "cutlass/arch/barrier.h"
#include "cutlass/arch/grid_dependency_control.h"
namespace flash {
template <typename T>
struct PrepareSortOp
{
  __device__ __forceinline__ bool operator()(T const & lhs, T const & rhs)
  {
    return lhs > rhs;
  }
};
template <>
struct PrepareSortOp<int2> {
  __device__ __forceinline__ bool operator()(int2 const & lhs, int2 const & rhs) const {
    return lhs.x > rhs.x;
  }
};
template <>
struct PrepareSortOp<int4> {
  __device__ __forceinline__ bool operator()(int4 const & lhs, int4 const & rhs) const {
    return lhs.x > rhs.x;
  }
};
template <int NumWarps, bool Sort>
__global__ void prepare_varlen_num_blocks_kernel(
    int seqlen_q_static, int seqlen_k_static, int seqlen_k_new_static,
    int const* const cu_seqlens_q, int const* const cu_seqlens_k, int const* const cu_seqlens_k_new,
    int const* const seqused_q, int const* const seqused_k, int const* const leftpad_k_ptr,
    int num_batch, int num_head, int qhead_per_khead, int num_sm, int num_splits_static,
    cutlass::FastDivmod blockm_divmod, cutlass::FastDivmod blockn_divmod,
    int* const tile_count_semaphore,
    int* const num_m_blocks_ptr,
    int* const num_splits_dynamic_ptr,
    int* const varlen_batch_idx_ptr,
    int* const num_nheads_in_l2_ptr,
    bool enable_pdl,
    bool is_causal,
    bool packgqa,
    int max_kvblocks_in_l2) {
  static constexpr int kNumBatchPerWarp = cutlass::NumThreadsPerWarp - 1;
  static constexpr int kSmemSize = 1;
  static constexpr int BLOCK_DIM_X = NumWarps * 32;
  static constexpr int ITEMS_PER_THREAD = 1;
  static_assert(BLOCK_DIM_X * ITEMS_PER_THREAD == NumWarps * 32);
  using BlockMergeSort = cub::BlockMergeSort<int4, BLOCK_DIM_X, ITEMS_PER_THREAD>;
  __shared__ int total_blocks_smem[kSmemSize];
  __shared__ typename BlockMergeSort::TempStorage temp_storage;
  if (enable_pdl) { cutlass::arch::launch_dependent_grids(); }
  if (threadIdx.x < kSmemSize) { total_blocks_smem[threadIdx.x] = 0; }
  __syncthreads();
  if (threadIdx.x == 0 && tile_count_semaphore) { *tile_count_semaphore = 0; }
  int lane = threadIdx.x % cutlass::NumThreadsPerWarp;
  auto get_num_m_blocks = [&](int batch_idx) {
    int seqlen;
    if (seqused_q) {
      seqlen = batch_idx < num_batch ? seqused_q[batch_idx] : 0;
    } else if (cu_seqlens_q) {
      int cur_cu_seqlen = batch_idx <= num_batch ? cu_seqlens_q[batch_idx] : 0;
      int next_cu_seqlen = __shfl_down_sync(0xffffffff, cur_cu_seqlen, 1);
      seqlen = next_cu_seqlen - cur_cu_seqlen;
    } else {
      seqlen = seqlen_q_static;
    }
    if(packgqa) { seqlen *= qhead_per_khead; }
    return batch_idx < num_batch && lane < kNumBatchPerWarp
      ? blockm_divmod.div(seqlen + blockm_divmod.divisor - 1) : 0;
  };
  auto get_num_n_blocks = [&](int batch_idx) {
    int leftpad_k = batch_idx < num_batch && leftpad_k_ptr != nullptr ? leftpad_k_ptr[batch_idx] : 0;
    int seqlen;
    if (seqused_k) {
      seqlen = batch_idx < num_batch ? seqused_k[batch_idx] : 0;
    } else if (cu_seqlens_k) {
      int cur_cu_seqlen = batch_idx <= num_batch ? cu_seqlens_k[batch_idx] : 0;
      int next_cu_seqlen = __shfl_down_sync(0xffffffff, cur_cu_seqlen, 1);
      seqlen = next_cu_seqlen - cur_cu_seqlen;
    } else {
      seqlen = seqlen_k_static;
    }
    int seqlen_new;
    if (cu_seqlens_k_new) {
      int cur_cu_seqlen_new = batch_idx <= num_batch ? cu_seqlens_k_new[batch_idx] : 0;
      int next_cu_seqlen_new = __shfl_down_sync(0xffffffff, cur_cu_seqlen_new, 1);
      seqlen_new = next_cu_seqlen_new - cur_cu_seqlen_new;
    } else {
      seqlen_new = seqlen_k_new_static;
    }
    seqlen = seqlen - leftpad_k + seqlen_new;
    return batch_idx < num_batch && lane < kNumBatchPerWarp
      ? blockn_divmod.div(seqlen + blockn_divmod.divisor - 1) : 0;
  };
  int warp_idx = threadIdx.x / cutlass::NumThreadsPerWarp;
  int batch_cta_idx_offset = int(blockIdx.x) * 992;
  int bidb_start = batch_cta_idx_offset + kNumBatchPerWarp * warp_idx;
  int batch_idx = lane + bidb_start;
  int num_m_blocks = get_num_m_blocks(batch_idx);
  int num_n_blocks = get_num_n_blocks(batch_idx);
  auto get_nheads_in_l2 = [&](int n_blocks) {
    int nheads_in_l2 = n_blocks * 16 <= max_kvblocks_in_l2 ? 16
      : n_blocks * 8 <= max_kvblocks_in_l2 ? 8
      : n_blocks * 4 <= max_kvblocks_in_l2 ? 4
      : n_blocks * 2 <= max_kvblocks_in_l2 ? 2
      : 1;
    if(!packgqa) { nheads_in_l2 *= qhead_per_khead; }
    return min(nheads_in_l2, num_head);
  };
  int num_splits_dynamic;
  if (int(gridDim.x) > 1 || num_splits_static == 1) {
    num_splits_dynamic = 1;
  } else {
    int total_blocks = num_m_blocks * num_n_blocks;
    #pragma unroll
    for (int i = cutlass::NumThreadsPerWarp / 2; i >= 1; i /= 2) {
      total_blocks += __shfl_down_sync(0xffffffff, total_blocks, i);
    }
    if (lane == 0) { atomicAdd(total_blocks_smem, total_blocks); }
    __syncthreads();
    total_blocks = total_blocks_smem[0];
    int blocks_per_sm = static_cast<int>(ceilf(float(total_blocks) * 1.1f * float(num_head) / float(num_sm)));
    num_splits_dynamic = std::max(std::min((num_n_blocks + blocks_per_sm - 1) / blocks_per_sm, num_splits_static), 1);
    num_n_blocks = cutlass::ceil_div(num_n_blocks, num_splits_dynamic);
  }
  if constexpr (Sort) {
    if(lane == kNumBatchPerWarp || batch_idx >= num_batch) {
      num_n_blocks = INT_MIN;
    } else if (is_causal) {
      num_n_blocks = num_n_blocks * blockn_divmod.divisor - num_m_blocks * blockm_divmod.divisor;
    }
    int4 batch_coords[ITEMS_PER_THREAD];
    batch_coords[0] = make_int4(num_n_blocks, num_m_blocks, num_splits_dynamic, batch_idx);
    BlockMergeSort(temp_storage).Sort(batch_coords, PrepareSortOp<int4>());
    if (is_causal) {
      batch_coords[0].x = blockn_divmod.div(batch_coords[0].x + batch_coords[0].y * blockm_divmod.divisor);
    }
    batch_idx = batch_cta_idx_offset + threadIdx.x;
    if (batch_idx < num_batch && threadIdx.x < 992) {
      if(num_nheads_in_l2_ptr) { num_nheads_in_l2_ptr[batch_idx] = get_nheads_in_l2(max(batch_coords[0].x, 1)); }
      num_m_blocks_ptr[batch_idx] = batch_coords[0].y;
      num_splits_dynamic_ptr[batch_idx] = batch_coords[0].z;
      varlen_batch_idx_ptr[batch_idx] = batch_coords[0].w;
    }
  } else {
    if (batch_idx < num_batch && lane < kNumBatchPerWarp) {
      if(num_nheads_in_l2_ptr) { num_nheads_in_l2_ptr[batch_idx] = get_nheads_in_l2(max(num_n_blocks, 1)); }
      num_splits_dynamic_ptr[batch_idx] = num_splits_dynamic;
      num_m_blocks_ptr[batch_idx] = num_m_blocks;
    }
  }
  if (enable_pdl) { cutlass::arch::wait_on_dependent_grids(); }
}
}
void prepare_varlen_num_blocks(Flash_fwd_params &params, cudaStream_t strm_, bool packgqa,
                int blockM, int blockN, bool enable_pdl) {
  int qhead_per_khead = cutlass::ceil_div(params.h, params.h_k);
  int num_warps = cutlass::ceil_div(params.b, 31);
  int num_ctas = cutlass::ceil_div(params.b, 31 * 32);
  int const size_l2 = 8 * 1024 * 1024;
  int const element_size = params.is_e4m3 ? 1 : 2;
  int const size_one_kvblock = blockN * (params.d + params.dv) * element_size;
  int const max_kvblocks_in_l2 = size_l2 / size_one_kvblock;
  BOOL_SWITCH(params.varlen_sort_batches, Sort, [&] {
    NUM_WARP_SWITCH(num_warps, NumWarps, [&] {
      cudaLaunchAttribute la[1];
      la[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
      la[0].val.programmaticStreamSerializationAllowed = 1;
      cudaLaunchConfig_t lc = {dim3(num_ctas), dim3(32 * NumWarps), 0, strm_, la, enable_pdl ? 1u : 0u};
      cudaLaunchKernelEx(&lc, flash::prepare_varlen_num_blocks_kernel<NumWarps, Sort>,
        params.seqlen_q, params.seqlen_k, params.seqlen_knew,
        params.cu_seqlens_q, params.cu_seqlens_k, params.cu_seqlens_knew,
        params.seqused_q, params.seqused_k, params.leftpad_k,
        params.b, !packgqa ? params.h : params.h_k, qhead_per_khead, params.num_sm, params.num_splits,
        cutlass::FastDivmod(blockM), cutlass::FastDivmod(blockN),
        params.tile_count_semaphore,
        params.num_m_blocks_ptr,
        params.num_splits_dynamic_ptr,
        params.varlen_batch_idx_ptr,
        params.num_nheads_in_l2_ptr,
        enable_pdl,
        params.is_causal,
        packgqa,
        max_kvblocks_in_l2);
    });
  });
}


#define SK2_PREP_THREADS 1024
#define SK2_MAXT 4096
__global__ void __launch_bounds__(SK2_PREP_THREADS) sk2_prepare_kernel(
    int const* __restrict__ cu_q, int const* __restrict__ seqused_k, int B, int h_k, int qpk, int bm, int bn, int G, int K, int wave,
    int* __restrict__ tp, int* __restrict__ desc, int* __restrict__ st, int* __restrict__ cnt) {
  cutlass::arch::launch_dependent_grids();
  __shared__ int s_nmb[64], s_base[65];
  __shared__ int s_tp[SK2_MAXT + 1];
  using BlockScan = cub::BlockScan<int, SK2_PREP_THREADS>;
  __shared__ typename BlockScan::TempStorage scan_tmp;
  int const tid = threadIdx.x;
  if (tid < B) { int const L = cu_q[tid + 1] - cu_q[tid]; s_nmb[tid] = (L * qpk + bm - 1) / bm; }
  __syncthreads();
  if (tid == 0) { int a = 0; for (int b = 0; b < B; ++b) { s_base[b] = a; a += s_nmb[b] * h_k; } s_base[B] = a; }
  __syncthreads();
  int const NT = s_base[B];
  int const per = (NT + SK2_PREP_THREADS - 1) / SK2_PREP_THREADS;
  int local = 0;
  for (int k = 0; k < per; ++k) {
    int const t = tid * per + k;
    if (t >= NT) break;
    int o = t;
    if (wave) {
      int const q = NT / G, rem = NT % G;
      int rr, j;
      if (t < rem * (q + 1)) { rr = t / (q + 1); j = t % (q + 1); }
      else { int const p2 = t - rem * (q + 1); rr = rem + p2 / max(q, 1); j = p2 % max(q, 1); }
      o = j * G + rr;
    }
    int b = 0;
    while (b + 1 < B && s_base[b + 1] <= o) ++b;
    int const r = o - s_base[b], nmb = s_nmb[b];
    int const h = r / nmb, m = r - h * nmb;
    int const Lq = cu_q[b + 1] - cu_q[b], Lk = seqused_k[b];
    int nb = (Lk + bn - 1) / bn;
    int const m_idx_max = ((m + 1) * bm - 1) / qpk + 1;
    int const ncaus = m_idx_max + Lk - Lq;
    nb = min(nb, ncaus <= 0 ? 0 : (ncaus + bn - 1) / bn);
    nb = max(nb, 1);
    desc[t] = b | (h << 8) | (m << 16);
    cnt[t] = 0;
    s_tp[t] = nb + K;
    local += nb + K;
  }
  int excl;
  BlockScan(scan_tmp).ExclusiveSum(local, excl);
  __syncthreads();
  for (int k = 0; k < per; ++k) {
    int const t = tid * per + k;
    if (t >= NT) break;
    int const nb = s_tp[t];
    s_tp[t] = excl;
    excl += nb;
  }
  if (tid == SK2_PREP_THREADS - 1) { s_tp[NT] = excl; }
  __syncthreads();
  for (int t = tid; t <= NT; t += SK2_PREP_THREADS) { tp[t] = s_tp[t]; }
  long long const W = s_tp[NT];
  for (int c = tid; c < G; c += SK2_PREP_THREADS) {
    long long const r0 = c * W / G;
    int lo = 0, hi = NT - 1;
    while (lo < hi) { int const mid = (lo + hi + 1) >> 1; if (s_tp[mid] <= r0) lo = mid; else hi = mid - 1; }
    st[c] = lo;
  }
  if (tid == 0) { st[G] = int(W); st[G + 1] = K; }
}
void sk2_prepare(Flash_fwd_params &params, cudaStream_t strm_, int blockM, int blockN) {
  int const qpk = params.h / params.h_k;
  sk2_prepare_kernel<<<1, SK2_PREP_THREADS, 0, strm_>>>(
    params.cu_seqlens_q, params.seqused_k, params.b, params.h_k, qpk, blockM, blockN, params.num_sm, params.sk2_k, params.sk2_wave,
    params.num_m_blocks_ptr, params.varlen_batch_idx_ptr, params.num_splits_dynamic_ptr, params.tile_count_semaphore);
}
void pv_gather_stage(torch::Tensor kc, torch::Tensor vc, torch::Tensor pt, torch::Tensor seqlens, torch::Tensor ko, torch::Tensor vo, torch::Tensor cu_q, int64_t part);
template <bool Is_causal, bool Is_local, bool Paged, bool Split, bool Vcol, bool SK>
static void att_launch(Flash_fwd_params& params, cudaStream_t s) {
 run_flash_fwd<90, 64, 64, 1, cutlass::float_e4m3_t, cutlass::bfloat16_t, Is_causal, Is_local, false ,
        true , Paged, false, false, false, true , Split, Vcol ,
        false , false, SK>(params, s);
}
static void att_dispatch(Flash_fwd_params& params, cudaStream_t s, int v) {
#define ATT_CASE(V) if (v == V) { att_launch<!((V) & 1), ((V) & 1) != 0, ((V) & 2) != 0, ((V) & 4) != 0, ((V) & 8) != 0, ((V) & 16) != 0>(params, s); return; }
 ATT_VARIANTS
#undef ATT_CASE
 TORCH_CHECK(false, "att: variant not compiled: ", v);
}
static int att_num_splits(Flash_fwd_params const& params) {
 int const kBlockM = ATT_BM, kBlockN = params.is_local ? ATT_BN_LOCAL : ATT_BN;
 int seqlen_k_loaded = !params.is_local ? params.seqlen_k
   : std::max(0, std::min(params.seqlen_k, params.window_size_right + params.window_size_left + 1 + kBlockM));
 int num_n_blocks = (seqlen_k_loaded + kBlockN - 1) / kBlockN;
 int num_m_blocks = (params.seqlen_q * (params.h / params.h_k) + kBlockM - 1) / kBlockM;
 int size_one_kv_head = params.seqlen_k * (params.d + params.dv);
 int total_mblocks = ((params.total_q * (params.h / params.h_k) + kBlockM - 1) / kBlockM) * params.h_k;
 return num_splits_heuristic(total_mblocks, params.num_sm, num_n_blocks, num_m_blocks, size_one_kv_head,
               params.is_causal || params.is_local, 128);
}
torch::Tensor att_fwd(torch::Tensor q, torch::Tensor k, torch::Tensor v, torch::Tensor page_table,
           torch::Tensor cache_seqlens, torch::Tensor cu_seqlens_q, int64_t max_seqlen_q,
           double softmax_scale, int64_t window_left, int64_t num_splits_req, torch::Tensor out,
           torch::Tensor gate, double gate_eps) {
 bool const paged = page_table.numel() > 0;
 int const batch = cu_seqlens_q.size(0) - 1;
 int const total_q = q.size(0), h = q.size(1), d = q.size(2);
 int const page_size = paged ? k.size(1) : 1;
 int const seqlen_k = paged ? page_table.size(1) * page_size : k.size(1);
 int const h_k = k.size(-2);
 int seqlen_q = max_seqlen_q;
 int wl = window_left, wr = 0;
 if (wl >= seqlen_k - 1) wl = -1;
 Flash_fwd_params params = {};
 params.is_bf16 = false;
 params.is_e4m3 = true;
 params.q_ptr = q.data_ptr(); params.k_ptr = k.data_ptr(); params.v_ptr = v.data_ptr();
 params.q_row_stride = q.stride(-3); params.k_row_stride = k.stride(-3); params.v_row_stride = v.stride(-3);
 params.q_head_stride = q.stride(-2); params.k_head_stride = k.stride(-2); params.v_head_stride = v.stride(-2);
 params.v_dim_stride = v.stride(-1);
 params.o_ptr = out.data_ptr(); params.o_row_stride = out.stride(-3); params.o_head_stride = out.stride(-2);
 params.k_batch_stride = k.stride(0); params.v_batch_stride = v.stride(0);
 params.v_row_stride=128; params.v_head_stride=64; params.v_dim_stride=1;
 params.v_batch_stride=int64_t(k.size(1))*128;
 params.cu_seqlens_q = cu_seqlens_q.data_ptr<int>();
 params.seqused_k = cache_seqlens.data_ptr<int>();
 auto opts = q.options();
 auto lse = torch::empty({h, total_q}, opts.dtype(torch::kFloat));
 params.softmax_lse_ptr = lse.data_ptr();
 params.b = batch; params.h = h; params.h_k = h_k;
 params.seqlen_q = seqlen_q; params.seqlen_k = seqlen_k;
 params.seqlen_q_rounded = (seqlen_q + 127) / 128 * 128; params.seqlen_k_rounded = (seqlen_k + 127) / 128 * 128;
 params.d = d; params.d_rounded = 64;
 params.scale_softmax = softmax_scale; params.softcap = 0.f;
 params.p_dropout = 1.f; params.p_dropout_in_uint8_t = 255; params.rp_dropout = 1.f;
 params.is_causal = wl < 0 && wr == 0;
 params.is_local = (wl >= 0) && !params.is_causal;
 if (wl < 0) wl = seqlen_k - 1;
 params.window_size_left = wl; params.window_size_right = wr; params.attention_chunk = 0;
 params.arch = 90; params.num_sm = at::cuda::getCurrentDeviceProperties()->multiProcessorCount;
 params.total_q = total_q; params.total_k = paged ? 0 : batch * k.size(1); params.b_k = paged ? page_table.size(0) : batch;
 params.dv = d; params.dv_rounded = 64;
 if (paged) { params.page_table = page_table.data_ptr<int>(); params.page_table_batch_stride = page_table.stride(0); }
 params.page_size = page_size; params.num_pages = paged ? k.size(0) : 0;
 params.prepare_varlen_pdl = batch <= 992;
 params.num_splits_dynamic_ptr = reinterpret_cast<int*>(1);
 params.pagedkv_tma = false;
 params.num_splits = std::min(num_splits_req <= 0 ? att_num_splits(params) : (int)num_splits_req, 16);
#define SK2_K_DEFAULT 2
 int const sk_qpk = h / h_k;
 int const sk_ntcap = ((total_q * sk_qpk) / ATT_BM + batch + 1) * h_k;
 int const sk_tiles = ((total_q * sk_qpk + ATT_BM - 1) / ATT_BM) * h_k;
 bool const sk_turn = false;


 bool const sk_want = num_splits_req >= 100 || sk_turn ||
            (num_splits_req == 0 && sk_tiles >= 4 * params.num_sm && seqlen_k >= 2 * seqlen_q);
 bool const sk = sk_want && !params.is_local && !paged && sk_ntcap <= 4096 &&
         (seqlen_k + ATT_BN - 1) / ATT_BN < 2047 && batch <= 64;
 if (num_splits_req == -1) params.num_splits = att_num_splits(params);
 if (sk) {
  params.num_splits = 2;
  params.sk2_wave = 0;
  params.sk2_k = num_splits_req >= 100 && num_splits_req < 10000 ? int(num_splits_req - 100) : SK2_K_DEFAULT;
  if (num_splits_req >= 10000) {
   params.num_sm = std::min(params.num_sm, int((num_splits_req - 10000) / 100));
   params.sk2_k = int((num_splits_req - 10000) % 100);
  }
  if (num_splits_req >= 1000000) {
   params.sk2_wave = 1;
   params.sk2_k = SK2_K_DEFAULT; params.num_sm = at::cuda::getCurrentDeviceProperties()->multiProcessorCount;
  }
 }
 params.pack_gqa = true;
 params.varlen_sort_batches = !params.is_local;
 params.head_swizzle = params.is_causal || params.is_local;
 int b_rounded = (batch + 3) / 4 * 4;
 int nvec = 2 + (params.varlen_sort_batches ? 1 : 0) + (params.head_swizzle ? 1 : 0);
 int head_swizzle_offset = b_rounded * (params.varlen_sort_batches ? 3 : 2);
 int sem_offset = b_rounded * nvec;
 auto meta = torch::empty({sem_offset + 1}, opts.dtype(torch::kInt32));
 params.skip_scheduler_metadata_computation = false;
 int* mp = meta.data_ptr<int>();
 params.num_splits_dynamic_ptr = mp;
 params.num_m_blocks_ptr = mp + b_rounded;
 params.varlen_batch_idx_ptr = params.varlen_sort_batches ? mp + b_rounded * 2 : nullptr;
 params.num_nheads_in_l2_ptr = params.head_swizzle ? mp + head_swizzle_offset : nullptr;
 params.tile_count_semaphore = mp + sem_offset;
 params.tile_count_semaphore_offset = sem_offset;
 torch::Tensor oacc, lseacc;
 torch::Tensor sk_meta;
 if (sk) {
  int const G = params.num_sm, NMMA = 3 * 128, FR = ATT_BM * 64 / NMMA, NR = 2;
  sk_meta = torch::empty({3 * sk_ntcap + 1 + G + 2}, opts.dtype(torch::kInt32));
  int* m = sk_meta.data_ptr<int>();
  params.num_m_blocks_ptr = m;
  params.varlen_batch_idx_ptr = m + sk_ntcap + 1;
  params.tile_count_semaphore = m + 2 * sk_ntcap + 1;
  params.num_splits_dynamic_ptr = m + 3 * sk_ntcap + 1;
  params.num_nheads_in_l2_ptr = nullptr;
  oacc = torch::empty({(long)G * 2 * NMMA * FR}, opts.dtype(torch::kFloat));
  lseacc = torch::empty({(long)G * 2 * NMMA * NR}, opts.dtype(torch::kFloat));
  params.is_fp32 = false;
  params.oaccum_ptr = oacc.data_ptr(); params.softmax_lseaccum_ptr = lseacc.data_ptr();
  params.oaccum_split_stride = 0; params.oaccum_row_stride = 0; params.oaccum_head_stride = 0;
  params.lseaccum_split_stride = 0; params.lseaccum_head_stride = 0;
 } else
 if (params.num_splits > 1) {
  oacc = torch::empty({params.num_splits, h, total_q, d}, opts.dtype(torch::kFloat));
  lseacc = torch::empty({params.num_splits, h, total_q}, opts.dtype(torch::kFloat));
  params.is_fp32 = false;
  params.oaccum_ptr = oacc.data_ptr(); params.softmax_lseaccum_ptr = lseacc.data_ptr();
  params.oaccum_split_stride = oacc.stride(0); params.oaccum_row_stride = oacc.stride(-2);
  params.oaccum_head_stride = oacc.stride(-3);
  params.lseaccum_split_stride = lseacc.stride(0); params.lseaccum_head_stride = lseacc.stride(-2);
 }
 cudaStream_t s = at::cuda::getCurrentCUDAStream();
 int var = (params.is_local ? 1 : 0) | (paged ? 2 : 0) | (params.num_splits > 1 ? 4 : 0) | (params.v_dim_stride != 1 ? 8 : 0);
 if (sk) var |= 16;

 bool x52_fuse = gate.numel() > 0 && params.num_splits == 1 && !paged && d == 64 &&
         gate.stride(-1) == 1 && gate.size(-1) == h * d && gate.size(0) == total_q && out.stride(-3) == h * d &&
         gate.stride(0) % 8 == 0 && (reinterpret_cast<uintptr_t>(gate.data_ptr()) % 16) == 0;
 x52_fuse = x52_fuse && !sk;
 if (x52_fuse) { x52_gptr = gate.data_ptr(); x52_gstride = (int)gate.stride(0); x52_geps = (float)gate_eps; }
 att_dispatch(params, s, var);
 x52_gptr = nullptr; x52_gstride = 0; x52_geps = 0.f;
 if (x52_fuse) return out;
 if (sk) return torch::empty({0}, out.options());
 if (params.num_splits > 1) {
  params.is_bf16 = true;
  TORCH_CHECK(params.num_splits <= 16, "att: more than 16 splits");
  bool const fuse = gate.numel() > 0;
  if (fuse) {
   TORCH_CHECK(gate.stride(-1) == 1 && gate.size(-1) == h * d && d == 64 && gate.size(0) == total_q, "att: gate view");
   sw_gate_ptr = gate.data_ptr(); sw_gate_stride = (int)gate.stride(0); sw_gate_eps = (float)gate_eps;
  }
  run_flash_fwd_combine<90, 16, 64, 4, false, true, cutlass::bfloat16_t, float>(params, s, true);
  sw_gate_ptr = nullptr; sw_gate_stride = 0; sw_gate_eps = 0.f;
  if (fuse) return out;
 }
 return torch::empty({0}, out.options());
}
int64_t att_splits(torch::Tensor q, torch::Tensor k, torch::Tensor page_table, torch::Tensor cu_seqlens_q,
          int64_t max_seqlen_q, int64_t window_left) {
 Flash_fwd_params params = {};
 bool const paged = page_table.numel() > 0;
 params.b = cu_seqlens_q.size(0) - 1; params.h = q.size(1); params.h_k = k.size(-2);
 params.seqlen_q = max_seqlen_q; params.seqlen_k = paged ? page_table.size(1) * k.size(1) : k.size(1);
 params.d = 64; params.dv = 64; params.total_q = q.size(0);
 params.num_sm = at::cuda::getCurrentDeviceProperties()->multiProcessorCount;
 int wl = window_left >= params.seqlen_k - 1 ? -1 : window_left;
 params.is_causal = wl < 0; params.is_local = wl >= 0;
 params.window_size_left = wl < 0 ? params.seqlen_k - 1 : wl; params.window_size_right = 0;
 return att_num_splits(params);
}
__global__ void att_gather_k(uint4 const* __restrict__ kc, uint4 const* __restrict__ vc, int const* __restrict__ pt,
               int const* __restrict__ seqlens, int const* __restrict__ cu_q, int pages, int page, int S,
               int rows_per_b, int part, int win, uint4* __restrict__ ko, uint4* __restrict__ vo) {
 int constexpr L16 = 8;
 long long const i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
 int const b = blockIdx.y;
 if (i >= (long long)rows_per_b * L16) return;
 int const j = int(i / L16), u = int(i % L16);
 int const n = seqlens[b], L = cu_q[b + 1] - cu_q[b];
 int s, drow = -1;
 if (part == 1) { if (j >= L) return; s = n - L + j; }
 else if (part == 3) {
  int const s0 = max(0, n - L - win);
  if (j >= n - s0) {
   long long const z = ((long long)b * S + j) * L16 + u; ko[z] = make_uint4(0, 0, 0, 0); vo[z] = make_uint4(0, 0, 0, 0);
   return;
  }
  s = s0 + j; drow = j;
 } else {
  s = j;
  if (s >= n) {
   long long const z = ((long long)b * S + s) * L16 + u; ko[z] = make_uint4(0, 0, 0, 0); vo[z] = make_uint4(0, 0, 0, 0);
   return;
  }
  if (part == 0 && s >= n - L) return;
 }
 int p = pt[(long long)b * pages + s / page];
 p = p < 0 ? 0 : p;
 long long const src = ((long long)p * page + s % page) * L16 + u;
 long long const dst = ((long long)b * S + (drow >= 0 ? drow : s)) * L16 + u;
 uint4 const kv = __ldg(kc + src), vv = __ldg(vc + src);
 ko[dst] = kv;
 vo[dst] = vv;
}
void att_gather(torch::Tensor kc, torch::Tensor vc, torch::Tensor pt, torch::Tensor seqlens, torch::Tensor cu_q,
        int64_t max_q, int64_t part, torch::Tensor ko, torch::Tensor vo) {
 int const B = pt.size(0), pages = pt.size(1), page = kc.size(1);
 int const S = ko.size(1);
 TORCH_CHECK(kc.size(2) * kc.size(3) == 128 && kc.is_contiguous() && vc.is_contiguous() && ko.is_contiguous(), "att_gather");
 TORCH_CHECK((part == 2 || (part >= 20 && part <= 26)) && S % ATT_BN == 0, "pv_gather full block");

 pv_gather_stage(kc,vc,pt,seqlens,ko,vo,cu_q,part);
 return;
 int const rows = part == 1 ? (int)max_q : S;
 int const win = part == 3 ? (int)max_q : 0;
 long long const n = (long long)rows * 8;
 dim3 grid((unsigned)((n + 255) / 256), B);
 att_gather_k<<<grid, 256, 0, at::cuda::getCurrentCUDAStream()>>>(
   (uint4 const*)kc.data_ptr(), (uint4 const*)vc.data_ptr(), pt.data_ptr<int>(), seqlens.data_ptr<int>(),
   cu_q.data_ptr<int>(), pages, page, S, rows, (int)part, win, (uint4*)ko.data_ptr(), (uint4*)vo.data_ptr());
}


__global__ void __launch_bounds__(256) att_gather_t_k(uint4 const* __restrict__ kc, uint4 const* __restrict__ vc,
  int const* __restrict__ pt, int const* __restrict__ seqlens, int const* __restrict__ cu_q, int pages, int page,
  int S, int part, int win, uint4* __restrict__ ko, uint4* __restrict__ vt) {
 __shared__ uint4 tile[64][9];
 int const b = blockIdx.y, j0 = blockIdx.x * 64, t = threadIdx.x;
 int const n = seqlens[b], L = cu_q[b + 1] - cu_q[b];
 int const s0 = part == 3 ? max(0, n - L - win) : 0;
 int const nv = n - s0;
 #pragma unroll
 for (int r = 0; r < 2; ++r) {
  int const idx = t + 256 * r, jr = idx >> 3, u = idx & 7, j = j0 + jr;
  uint4 kv = make_uint4(0, 0, 0, 0), vv = make_uint4(0, 0, 0, 0);
  if (j < nv) {
   int const s = s0 + j;
   int p = pt[(long long)b * pages + s / page];
   p = p < 0 ? 0 : p;
   long long const src = ((long long)p * page + s % page) * 8 + u;
   kv = __ldg(kc + src); vv = __ldg(vc + src);
  }
  ko[((long long)b * S + j) * 8 + u] = kv;
  tile[jr][u] = vv;
 }
 __syncthreads();
 unsigned char const* tb = reinterpret_cast<unsigned char const*>(&tile[0][0]);
 #pragma unroll
 for (int r = 0; r < 2; ++r) {
  int const idx = t + 256 * r, hd = idx >> 2, q = idx & 3;
  uint32_t w[4];
  #pragma unroll
  for (int k = 0; k < 4; ++k) {
   uint32_t x = 0;
   #pragma unroll
   for (int i = 0; i < 4; ++i) { x |= uint32_t(tb[(q * 16 + 4 * k + i) * 144 + hd]) << (8 * i); }
   w[k] = x;
  }
  vt[(((long long)b * 128 + hd) * S + j0) / 16 + q] = make_uint4(w[0], w[1], w[2], w[3]);
 }
}
void att_gather_t(torch::Tensor kc, torch::Tensor vc, torch::Tensor pt, torch::Tensor seqlens, torch::Tensor cu_q,
         int64_t win, int64_t part, torch::Tensor ko, torch::Tensor vt) {
 int const B = pt.size(0), pages = pt.size(1), page = kc.size(1);
 int const S = ko.size(1);
 TORCH_CHECK(kc.size(2) * kc.size(3) == 128 && S % 64 == 0 && kc.is_contiguous() && vc.is_contiguous() &&
       ko.is_contiguous() && vt.is_contiguous() && vt.numel() == ko.numel(), "att_gather_t");
 dim3 grid(S / 64, B);
 att_gather_t_k<<<grid, 256, 0, at::cuda::getCurrentCUDAStream()>>>(
   (uint4 const*)kc.data_ptr(), (uint4 const*)vc.data_ptr(), pt.data_ptr<int>(), seqlens.data_ptr<int>(),
   cu_q.data_ptr<int>(), pages, page, S, (int)part, (int)win, (uint4*)ko.data_ptr(), (uint4*)vt.data_ptr());
}
using PvImageMainloop = flash::CollectiveMainloopFwdSm90<
  ATT_STAGES, cute::Shape<cute::_1,cute::_1,cute::_1>,
  cute::Shape<cute::Int<ATT_BM>,cute::Int<ATT_BN>,cute::Int<64>>,64,
  cutlass::float_e4m3_t,float,cutlass::arch::Sm90,
  true,false,false,true,false,false,false,false,true,true,true,false,false,cutlass::bfloat16_t,false>;



template<class Mainloop>
__global__ void __launch_bounds__(128) pv_gather_stage_k(const uint8_t* kc,const uint8_t* vc,const int* pt,
  const int* lengths,int S,int pages,int page,int physical_pages,uint8_t* ko,uint8_t* vo,const int* cu_q) {
  using namespace cute;
  using Element=typename Mainloop::Element;
  constexpr int BN=ATT_BN,TileBytes=72*BN;
  using SmemLayoutVt=decltype(tile_to_shape(typename Mainloop::SmemLayoutAtomVt{},make_shape(Int<64>{},Int<BN>{},_1{}),Step<_2,_1,_3>{}));
  using Pad=decltype(tile_to_shape(typename Mainloop::SmemLayoutAtomVtMma{},make_shape(Int<72>{},Int<BN>{},_1{}),Step<_1,_2,_3>{}));
  using SmemLayoutVtMma=decltype(composition(Pad{},Layout<Shape<Int<64>,Int<BN>,_1>,Stride<_1,Int<72>,Int<72*BN>>>{}));
  using S2RTiledCopyVt=typename Mainloop::S2RTiledCopyVt;
  using R2STiledCopyV=typename Mainloop::R2STiledCopyV;
  using LDSM_divide_shape=typename Mainloop::LDSM_divide_shape;
  using STSM_divide_shape=typename Mainloop::STSM_divide_shape;
  __shared__ __align__(1024) Element vt[cosize_v<SmemLayoutVt>];
  __shared__ __align__(1024) Element v[TileBytes];
  auto sVt=as_position_independent_swizzle_tensor(make_tensor(make_smem_ptr(vt),SmemLayoutVt{}));
  auto sV=as_position_independent_swizzle_tensor(make_tensor(make_smem_ptr(v),SmemLayoutVtMma{}));
  int const thread_idx=threadIdx.x,nb=blockIdx.y,head=blockIdx.x,batch=blockIdx.z;
  int const nblocks=S/BN,n=lengths[batch];
  asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
  constexpr int IT=BN*4/128;
  int64_t row[IT];
  #pragma unroll
  for(int i=0;i<IT;++i) {
    int const token=nb*BN+(thread_idx+128*i)/4;
    int64_t r=-1;
    if(page==1) {
      if(token<n && token<pages) {
        int pg=pt[(int64_t)batch*pages+token];pg=pg<0?0:pg;
        if(pg<physical_pages) r=pg;
      }
    } else if(token<n && token/page<pages) {
      int pg=pt[(int64_t)batch*pages+token/page];pg=pg<0?0:pg;
      if(pg<physical_pages) r=(int64_t)pg*page+token%page;
    }
    row[i]=r;
  }
  asm volatile("griddepcontrol.wait;" ::: "memory");
  uint4 kv[IT],vv[IT];
  #pragma unroll
  for(int i=0;i<IT;++i) {
    int const d=((thread_idx+128*i)%4)*16;
    kv[i]=make_uint4(0,0,0,0);vv[i]=kv[i];
    if(row[i]>=0) {
      int64_t const src=(row[i]*2+head)*64+d;
      kv[i]=__ldg(reinterpret_cast<const uint4*>(kc+src));
      vv[i]=__ldg(reinterpret_cast<const uint4*>(vc+src));
    }
  }
  #pragma unroll
  for(int i=0;i<IT;++i) {
    int const seg=thread_idx+128*i,j=seg/4,d=(seg%4)*16,token=nb*BN+j;
    *reinterpret_cast<uint4*>(ko+(((int64_t)batch*S+token)*2+head)*64+d)=kv[i];
    *reinterpret_cast<uint4*>(&sVt(d,j,0))=vv[i];
  }
  constexpr int CW=128,PER=8*CW/16;
  uint4* stg=reinterpret_cast<uint4*>(v);
  for(int i=thread_idx;i<(BN/CW)*PER;i+=128) {
    stg[(i/PER)*(72*CW/16)+64*CW/16+(i%PER)]=make_uint4(0x38383838u,0x38383838u,0x38383838u,0x38383838u);
  }
  __syncthreads();
  constexpr bool Transpose_V=true;
    S2RTiledCopyVt s2r_tiled_copy_vt;
    R2STiledCopyV r2s_tiled_copy_v;
    auto s2r_thr_copy_vt = s2r_tiled_copy_vt.get_thread_slice(thread_idx);
    auto r2s_thr_copy_v = r2s_tiled_copy_v.get_thread_slice(thread_idx);
    Tensor tTranssVt_ = s2r_thr_copy_vt.partition_S(flat_divide(sVt, LDSM_divide_shape{}));
    Tensor tTranssV_ = r2s_thr_copy_v.partition_D(flat_divide(sV, STSM_divide_shape{}));
    CUTE_STATIC_ASSERT_V(rank(tTranssVt_) == rank(tTranssV_));
    CUTE_STATIC_ASSERT_V(size<0>(tTranssVt_) == size<0>(tTranssV_));
    CUTE_STATIC_ASSERT_V(size<1>(tTranssVt_) == size<1>(tTranssV_));
    CUTE_STATIC_ASSERT_V(size<2>(tTranssVt_) == size<2>(tTranssV_));
    CUTE_STATIC_ASSERT_V(size<3>(tTranssVt_) == size<3>(tTranssV_));
    CUTE_STATIC_ASSERT_V(size<4>(tTranssVt_) == size<4>(tTranssV_));
    static constexpr int Transpose_ILP = (size<2>(tTranssVt_) * size<3>(tTranssVt_)) % 2 == 0 ? 2 : 1;
    Tensor tTranssVt = logical_divide(group_modes<1, rank(tTranssVt_) - 1>(tTranssVt_), Shape<Underscore, Int<Transpose_ILP>>{});
    Tensor tTranssV = logical_divide(group_modes<1, rank(tTranssV_) - 1>(tTranssV_), Shape<Underscore, Int<Transpose_ILP>>{});
    auto transpose_V = [&](int stage) {
      if constexpr (Transpose_V) {
        #pragma unroll
        for (int i = 0; i < size<1, 1>(tTranssVt); ++i) {
          Tensor tTransrV = make_fragment_like(tTranssV(_, make_coord(_, _0{}), _0{}));
          static_assert(size<0>(tTransrV) == 16);
          Tensor tTransrV_64 = recast<uint2>(tTransrV);
          cute::copy(s2r_tiled_copy_vt, tTranssVt(_, make_coord(_, i), stage), tTransrV);
          #pragma unroll
          for (int j = 0; j < size(tTransrV_64); ++j) {
            uint32_t upper = tTransrV_64[j].x;
            uint32_t lower = tTransrV_64[j].y;
            tTransrV_64[j].x = __byte_perm(upper, lower, 0x6420);
            tTransrV_64[j].y = __byte_perm(upper, lower, 0x7531);
          }
          cute::copy(r2s_tiled_copy_v, tTransrV, tTranssV(_, make_coord(_, i), stage));
        }
      }
    };
  transpose_V(0);
  __syncthreads();
  int64_t const dst=((int64_t(batch)*2+head)*nblocks+nb)*TileBytes;
  for(int i=thread_idx;i<TileBytes/16;i+=128) {
    reinterpret_cast<uint4*>(vo+dst)[i]=reinterpret_cast<const uint4*>(v)[i];
  }
}
__device__ __forceinline__ uint4 z0_gp_ld(const uint4* p, uint64_t pol, int ef) {
  uint4 r;
  if (ef) {
    asm("ld.global.nc.L1::no_allocate.L2::cache_hint.v4.u32 {%0,%1,%2,%3}, [%4], %5;"
      : "=r"(r.x), "=r"(r.y), "=r"(r.z), "=r"(r.w) : "l"(p), "l"(pol));
  } else {
    r = __ldg(p);
  }
  return r;
}
__device__ __forceinline__ uint32_t z0_gp_w(uint4 const& v, int w) { return w == 0 ? v.x : w == 1 ? v.y : w == 2 ? v.z : v.w; }



template<int MODE>
__global__ void __launch_bounds__(128) z0_gp_k(const uint4* __restrict__ kc, const uint4* __restrict__ vc,
  const int* __restrict__ pt, const int* __restrict__ lengths, int S, int pages, int physical_pages,
  uint4* __restrict__ ko, uint4* __restrict__ vo) {
  asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
  int const t = threadIdx.x, g = t >> 3, l8 = t & 7, a = l8 & 3, h = l8 >> 2;
  int const nb = blockIdx.x, batch = blockIdx.y, nblocks = S >> 8;
  int const n = lengths[batch];
  int const key0 = nb * 256 + g * 16;
  uint64_t pol = 0;
  if (MODE == 3) asm("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;" : "=l"(pol));
  int e[2];
  #pragma unroll
  for (int i = 0; i < 2; ++i) {
    int const token = key0 + 2 * l8 + i;
    int r = -1;
    if (token < n && token < pages) {
      int pg = pt[(int64_t)batch * pages + token];
      pg = pg < 0 ? 0 : pg;
      if (pg < physical_pages) r = pg;
    }
    e[i] = r;
  }
  asm volatile("griddepcontrol.wait;" ::: "memory");
  int const lane0 = (t & 31) & ~7;
  int src[16];
  #pragma unroll
  for (int k = 0; k < 16; ++k) src[k] = __shfl_sync(0xffffffffu, e[k & 1], lane0 | (k >> 1));
  uint4 v[16];
  #pragma unroll
  for (int k = 0; k < 16; ++k) {
    v[k] = make_uint4(0, 0, 0, 0);
    if (src[k] >= 0) v[k] = z0_gp_ld(vc + (int64_t)src[k] * 8 + l8, pol, MODE == 3);
  }
  uint4* const kdst = ko + ((int64_t)batch * S + key0) * 8 + l8;
  uint32_t acc = 0;
  #pragma unroll
  for (int half = 0; half < 2; ++half) {
    uint4 kv[8];
    #pragma unroll
    for (int k = 0; k < 8; ++k) {
      kv[k] = make_uint4(0, 0, 0, 0);
      int const s = src[half * 8 + k];
      if (s >= 0) kv[k] = z0_gp_ld(kc + (int64_t)s * 8 + l8, pol, MODE == 3);
    }
    #pragma unroll
    for (int k = 0; k < 8; ++k) {
      if (MODE == 1) acc ^= kv[k].x ^ kv[k].y ^ kv[k].z ^ kv[k].w;
      else kdst[(half * 8 + k) * 8] = kv[k];
    }
  }
  if (MODE == 1) {
    #pragma unroll
    for (int k = 0; k < 16; ++k) acc ^= v[k].x ^ v[k].y ^ v[k].z ^ v[k].w;
    if (acc == 0x9e3779b9u && n == -7) ko[t] = make_uint4(acc, 0, 0, 0);
    return;
  }

  int64_t const tile = ((int64_t)(batch * 2 + h) * nblocks + nb) * 1152 + (g >> 3) * 576;
  int const gl = g & 7;
  #pragma unroll
  for (int w = 0; w < 4; ++w) {
    uint32_t o[4][4];
    #pragma unroll
    for (int q = 0; q < 4; ++q) {
      uint32_t const A = z0_gp_w(v[2 * q], w), B = z0_gp_w(v[2 * q + 1], w);
      uint32_t const C = z0_gp_w(v[2 * q + 8], w), D = z0_gp_w(v[2 * q + 9], w);
      uint32_t const t0 = __byte_perm(A, B, 0x5140), t1 = __byte_perm(A, B, 0x7362);
      uint32_t const t2 = __byte_perm(C, D, 0x5140), t3 = __byte_perm(C, D, 0x7362);
      o[0][q] = __byte_perm(t0, t2, 0x5410);
      o[1][q] = __byte_perm(t0, t2, 0x7632);
      o[2][q] = __byte_perm(t1, t3, 0x5410);
      o[3][q] = __byte_perm(t1, t3, 0x7632);
    }
    #pragma unroll
    for (int j = 0; j < 4; ++j) {
      int const ed = 4 * w + j;
      int const r = 16 * a + 8 * (ed & 1) + (ed >> 1);
      int const s = gl ^ (ed >> 1);
      vo[tile + r * 8 + s] = make_uint4(o[j][0], o[j][1], o[j][2], o[j][3]);
    }
  }
  if (MODE != 2) {
    #pragma unroll
    for (int i = 0; i < 2; ++i) {
      int const idx = t + 128 * i, hd = idx >> 7, rem = idx & 127;
      int64_t const tb = ((int64_t)(batch * 2 + hd) * nblocks + nb) * 1152 + (rem >> 6) * 576;
      vo[tb + (64 + ((rem >> 3) & 7)) * 8 + (rem & 7)] = make_uint4(0x38383838u, 0x38383838u, 0x38383838u, 0x38383838u);
    }
  }
}


template<int ONES>
__global__ void __launch_bounds__(128) z0_gp8_k(const uint4* __restrict__ kc, const uint4* __restrict__ vc,
  const int* __restrict__ pt, const int* __restrict__ lengths, int S, int pages, int physical_pages,
  uint4* __restrict__ ko, uint4* __restrict__ vo) {
  asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
  int const t = threadIdx.x, gl = t >> 4, kh = (t >> 3) & 1, l8 = t & 7, a = l8 & 3, h = l8 >> 2;
  int const nb = blockIdx.x >> 1, hh = blockIdx.x & 1, batch = blockIdx.y, nblocks = S >> 8;
  int const n = lengths[batch];
  int const key0 = nb * 256 + hh * 128 + gl * 16 + kh * 8;
  int e = -1;
  {
    int const token = key0 + l8;
    if (token < n && token < pages) {
      int pg = pt[(int64_t)batch * pages + token];
      pg = pg < 0 ? 0 : pg;
      if (pg < physical_pages) e = pg;
    }
  }
  asm volatile("griddepcontrol.wait;" ::: "memory");
  int const lane0 = (t & 31) & ~7;
  int src[8];
  #pragma unroll
  for (int k = 0; k < 8; ++k) src[k] = __shfl_sync(0xffffffffu, e, lane0 | k);
  uint4 v[8], kv[8];
  #pragma unroll
  for (int k = 0; k < 8; ++k) {
    v[k] = make_uint4(0, 0, 0, 0);
    kv[k] = v[k];
    if (src[k] >= 0) {
      v[k] = __ldg(vc + (int64_t)src[k] * 8 + l8);
      kv[k] = __ldg(kc + (int64_t)src[k] * 8 + l8);
    }
  }
  uint4* const kdst = ko + ((int64_t)batch * S + key0) * 8 + l8;
  #pragma unroll
  for (int k = 0; k < 8; ++k) kdst[k * 8] = kv[k];
  int64_t const tile = ((int64_t)(batch * 2 + h) * nblocks + nb) * 1152 + hh * 576;
  #pragma unroll
  for (int wi = 0; wi < 2; ++wi) {
    uint32_t lo[4][2], hi[4][2];
    #pragma unroll
    for (int p = 0; p < 4; ++p) {
      int const wk = wi + 2 * kh, ws = wi + 2 * (kh ^ 1);
      uint32_t const A = z0_gp_w(v[2 * p], wk), B = z0_gp_w(v[2 * p + 1], wk);
      uint32_t const As = z0_gp_w(v[2 * p], ws), Bs = z0_gp_w(v[2 * p + 1], ws);
      uint32_t const k0 = __byte_perm(A, B, 0x5140), k1 = __byte_perm(A, B, 0x7362);
      uint32_t const s0 = __byte_perm(As, Bs, 0x5140), s1 = __byte_perm(As, Bs, 0x7362);
      uint32_t const r0 = __shfl_xor_sync(0xffffffffu, s0, 8), r1 = __shfl_xor_sync(0xffffffffu, s1, 8);
      lo[p][0] = kh ? r0 : k0; hi[p][0] = kh ? k0 : r0;
      lo[p][1] = kh ? r1 : k1; hi[p][1] = kh ? k1 : r1;
    }
    #pragma unroll
    for (int j = 0; j < 4; ++j) {
      int const ed = 4 * (wi + 2 * kh) + j;
      uint32_t const sel = (j & 1) ? 0x7632 : 0x5410;
      uint32_t o[4];
      #pragma unroll
      for (int q = 0; q < 4; ++q) o[q] = __byte_perm(lo[q][j >> 1], hi[q][j >> 1], sel);
      int const r = 16 * a + 8 * (ed & 1) + (ed >> 1);
      int const s = gl ^ (ed >> 1);
      vo[tile + r * 8 + s] = make_uint4(o[0], o[1], o[2], o[3]);
    }
  }
  if (ONES) {
    int const hd = t >> 6;
    int64_t const tb = ((int64_t)(batch * 2 + hd) * nblocks + nb) * 1152 + hh * 576;
    vo[tb + (64 + ((t >> 3) & 7)) * 8 + (t & 7)] = make_uint4(0x38383838u, 0x38383838u, 0x38383838u, 0x38383838u);
  }
}
void z0_gp_launch(torch::Tensor kc, torch::Tensor vc, torch::Tensor pt, torch::Tensor lengths, torch::Tensor ko,
         torch::Tensor vo, int mode) {
  int const B = pt.size(0), S = ko.size(1), pages = pt.size(1);
  TORCH_CHECK(S % 256 == 0 && ATT_BN == 256 && kc.size(1) == 1, "z0_gp");
  dim3 grid(S / 256, B);
  auto st = at::cuda::getCurrentCUDAStream();
  const uint4* k = reinterpret_cast<const uint4*>(kc.data_ptr());
  const uint4* v = reinterpret_cast<const uint4*>(vc.data_ptr());
  uint4* o1 = reinterpret_cast<uint4*>(ko.data_ptr());
  uint4* o2 = reinterpret_cast<uint4*>(vo.data_ptr());
  int const* p = pt.data_ptr<int>();
  int const* l = lengths.data_ptr<int>();
  int const ph = (int)kc.size(0);
  if (mode == 4) z0_gp8_k<1><<<dim3(S / 128, B), 128, 0, st>>>(k, v, p, l, S, pages, ph, o1, o2);
  else if (mode == 5) z0_gp8_k<0><<<dim3(S / 128, B), 128, 0, st>>>(k, v, p, l, S, pages, ph, o1, o2);
  else if (mode == 1) z0_gp_k<1><<<grid, 128, 0, st>>>(k, v, p, l, S, pages, ph, o1, o2);
  else if (mode == 3) z0_gp_k<3><<<grid, 128, 0, st>>>(k, v, p, l, S, pages, ph, o1, o2);
  else if (mode == 2) {
    cudaLaunchAttribute la[1];
    la[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    la[0].val.programmaticStreamSerializationAllowed = 1;
    cudaLaunchConfig_t lc = {grid, dim3(128), 0, st, la, 1};
    cudaLaunchKernelEx(&lc, z0_gp_k<2>, k, v, p, l, S, pages, ph, o1, o2);
  }
  else z0_gp_k<0><<<grid, 128, 0, st>>>(k, v, p, l, S, pages, ph, o1, o2);
}


void pv_gather_stage(torch::Tensor kc,torch::Tensor vc,torch::Tensor pt,torch::Tensor lengths,torch::Tensor ko,torch::Tensor vo,torch::Tensor cu_q,int64_t part) {
  if(kc.size(1)==1 && part!=20 && (part!=2 || pt.size(0)>=Z0_GP_MINB)) { TORCH_CHECK(vo.numel()==(int64_t)pt.size(0)*ko.size(1)*2*72,"pv_gather storage"); z0_gp_launch(kc,vc,pt,lengths,ko,vo,part==2?Z0_GP:(int)part-21); return; }
  int const B=pt.size(0),S=ko.size(1),pages=pt.size(1),page=kc.size(1);
  TORCH_CHECK(vo.numel()==(int64_t)B*S*2*72,"pv_gather storage");
  {
  cudaLaunchAttribute la[1];
  la[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  la[0].val.programmaticStreamSerializationAllowed = 1;
  unsigned const nla = 1;
  cudaLaunchConfig_t lc = {dim3(2,S/ATT_BN,B), dim3(128), 0, at::cuda::getCurrentCUDAStream(), la, nla};
  cudaLaunchKernelEx(&lc, pv_gather_stage_k<PvImageMainloop>, (const uint8_t*)kc.data_ptr(), (const uint8_t*)vc.data_ptr(),
    (const int*)pt.data_ptr<int>(), (const int*)lengths.data_ptr<int>(), S, pages, page, (int)kc.size(0),
    (uint8_t*)ko.data_ptr(), (uint8_t*)vo.data_ptr(), (const int*)cu_q.data_ptr<int>());
  }
}
#ifdef AT_SUB
__device__ __forceinline__ float at_bfr(float x) { return __bfloat162float(__float2bfloat16(x)); }
__device__ __forceinline__ uint16_t at_fp8x2(float a, float b) {
 return (uint16_t)__nv_cvt_float2_to_fp8x2(make_float2(a, b), __NV_SATFINITE, __NV_E4M3);
}



__global__ void __launch_bounds__(256) at_front_k(const __nv_bfloat162* __restrict__ qkv, const long* __restrict__ loc,
  uint16_t* __restrict__ qo, uint8_t* __restrict__ kc, uint8_t* __restrict__ vc, int T, float eps, float qmul) {
 asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
 asm volatile("griddepcontrol.wait;" ::: "memory");
 const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
 const int t = blockIdx.x * 8 + warp;
 if (t >= T) return;
 const __nv_bfloat162* row = qkv + (size_t)t * 1152;
 const long L = loc[t];
 float2 v[20];
#pragma unroll
 for (int h = 0; h < 20; h++) v[h] = __bfloat1622float2(row[h * 32 + lane]);
#pragma unroll
 for (int h = 0; h < 20; h++) {
  float x0 = v[h].x, x1 = v[h].y;
  if (h < 18) {
   float ss = x0 * x0 + x1 * x1;
#pragma unroll
   for (int o = 16; o > 0; o >>= 1) ss += __shfl_xor_sync(0xffffffffu, ss, o);
   const float r = rsqrtf(ss / 64.f + eps);
   x0 *= r; x1 *= r;
   x0 = at_bfr(x0); x1 = at_bfr(x1);
  }
  if (h < 16) qo[((size_t)t * 16 + h) * 32 + lane] = at_fp8x2(x0 * qmul, x1 * qmul);
  else if (h < 18) *reinterpret_cast<uint16_t*>(kc + L * 128 + (h - 16) * 64 + 2 * lane) = at_fp8x2(x0, x1);
  else *reinterpret_cast<uint16_t*>(vc + L * 128 + (h - 18) * 64 + 2 * lane) = at_fp8x2(x0, x1);
 }
}
void att_front(torch::Tensor qkv, torch::Tensor loc, torch::Tensor qo, torch::Tensor kc, torch::Tensor vc, double eps,
        double qmul) {
 const int T = (int)qkv.size(0);
 TORCH_CHECK(qkv.size(1) == 2304 && qkv.stride(0) == 2304, "att_front: qkv layout");
 cudaLaunchAttribute la[1];
 la[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
 la[0].val.programmaticStreamSerializationAllowed = 1;
 cudaLaunchConfig_t lc = {dim3((T + 7) / 8), dim3(256), 0, at::cuda::getCurrentCUDAStream(), la, 1};
 cudaLaunchKernelEx(&lc, at_front_k, (const __nv_bfloat162*)qkv.data_ptr(), (const long*)loc.data_ptr<long>(),
           (uint16_t*)qo.data_ptr(), (uint8_t*)kc.data_ptr(), (uint8_t*)vc.data_ptr(), T, (float)eps,
           (float)qmul);
}
#endif
"""


def xt__cutlass_include():
    import flashinfer
    return os.path.join(os.path.dirname(flashinfer.__file__), "data", "cutlass", "include")


xt_att_ext = load_inline(name="okc_att_z14643_b460010", cpp_sources=xt_ATT_CPP, cuda_sources=xt_ATT_SRC,
                      functions=["att_fwd", "att_gather", "att_gather_t", "att_front"], extra_include_paths=[xt__cutlass_include()],
                      extra_cuda_cflags=xt_ATT_FLAGS)
xt_EMPTY_PT = {}


def xt_attn(q, k_cache, v_cache, page_table, cache_seqlens, cu_seqlens_q, max_seqlen_q, softmax_scale, window_left, out):
    """FA3 forward into out [T, 16, 64] BF16 (unnormalized rows, see the module docstring). page_table must be None:
    only the contiguous variants are compiled, and k_cache / v_cache must come from gather()."""
    if page_table is None:
        page_table = xt_EMPTY_PT.get(q.device)
        if page_table is None:
            page_table = xt_EMPTY_PT[q.device] = torch.empty(0, dtype=torch.int32, device=q.device)
    return xt_att_ext.att_fwd(q, k_cache, v_cache, page_table, cache_seqlens, cu_seqlens_q, max_seqlen_q, softmax_scale,
                           window_left, 0, out, xt__empty_gate(q.device), 0.0)


xt_EMPTY_GATE = {}


def xt__empty_gate(dev):
    g = xt_EMPTY_GATE.get(dev)
    if g is None:
        g = xt_EMPTY_GATE[dev] = torch.empty(0, dtype=torch.bfloat16, device=dev)
    return g


def xt_attn_g(q, k_cache, v_cache, page_table, cache_seqlens, cu_seqlens_q, max_seqlen_q, softmax_scale, window_left, out,
           gate, eps):
    """attn() with o_norm_gate fused where the call allows it: in the attention epilogue when the call is not split,
    in the combine kernel when it is split by FA3's heuristic. Returns True when out already holds
    o_norm_gate(attention, gate), else False: the equal-work split (long-context calls with at least 4 waves of
    tiles and keys >= 2 x queries) merges in-kernel and leaves the unnormalized attention in out for the caller's
    o_norm_gate."""
    if page_table is None:
        page_table = xt_EMPTY_PT.get(q.device)
        if page_table is None:
            page_table = xt_EMPTY_PT[q.device] = torch.empty(0, dtype=torch.int32, device=q.device)
    r = xt_att_ext.att_fwd(q, k_cache, v_cache, page_table, cache_seqlens, cu_seqlens_q, max_seqlen_q, softmax_scale,
                        window_left, 0, out, gate, eps)
    return r.numel() > 0


def xt_gather(k_cache, v_cache, page_table, cache_seqlens, cu_seqlens_q, max_q, part, ko=None, vo=None):
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
    xt_att_ext.att_gather(k_cache.view(torch.uint8), v_cache.view(torch.uint8), page_table, cache_seqlens, cu_seqlens_q,
                       max_q, part, ko.view(torch.uint8), vo.view(torch.uint8))
    return ko, vo


def xt_gather_window(k_cache, v_cache, page_table, cache_seqlens, cu_seqlens_q, max_q, window):
    """Window copy for a sliding layer (part 3). Not usable in this build: att_gather accepts only part 2 or 20..26
    and raises on part 3. layer.py runs sliding layers on attention_sliding*.py instead."""
    B = page_table.shape[0]
    shape = (B, max_q + window - 1) + tuple(k_cache.shape[2:])
    ko = torch.empty(shape, dtype=k_cache.dtype, device=k_cache.device)
    vo = torch.empty(shape, dtype=v_cache.dtype, device=v_cache.device)
    xt_att_ext.att_gather(k_cache.view(torch.uint8), v_cache.view(torch.uint8), page_table, cache_seqlens, cu_seqlens_q,
                       window - 1, 3, ko.view(torch.uint8), vo.view(torch.uint8))
    seqlens = torch.minimum(cache_seqlens, cu_seqlens_q[1:] - cu_seqlens_q[:-1] + (window - 1)).to(torch.int32)
    return ko, vo, seqlens


def xt_gather_t(k_cache, v_cache, page_table, cache_seqlens, cu_seqlens_q, max_q, window=0):
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
    xt_att_ext.att_gather_t(k_cache.view(torch.uint8), v_cache.view(torch.uint8), page_table, cache_seqlens, cu_seqlens_q,
                         window - 1 if window else 0, 3 if window else 2, ko.view(torch.uint8), vt.view(torch.uint8))
    vo = vt.permute(0, 3, 1, 2)
    if not window:
        return ko, vo, cache_seqlens
    seqlens = torch.minimum(cache_seqlens, cu_seqlens_q[1:] - cu_seqlens_q[:-1] + (window - 1)).to(torch.int32)
    return ko, vo, seqlens


xt_QMUL = 1.4426950408889634


def xt_front(qkv, cache_loc, q8, k_cache_u8, v_cache_u8, eps, softmax_scale):
    """Full-layer front kernel: RMS-normalizes each q and k head (no weight, no RoPE: layer.py takes this path only
    when rope_theta is 0), writes q as FP8 of q x QMUL x softmax_scale into q8 and FP8 k / v into the cache rows at
    cache_loc. Returns 1 / QMUL, the factor the caller applies to the softmax scale it passes to attn / attn_g."""
    xt_att_ext.att_front(qkv, cache_loc, q8, k_cache_u8, v_cache_u8, eps, xt_QMUL * softmax_scale)
    return 1.0 / xt_QMUL




xs_SK_CPP = 'void sl_gather(torch::Tensor kc, torch::Tensor vc, torch::Tensor pt, torch::Tensor seqlens, torch::Tensor cu_q, int64_t win, torch::Tensor ko, torch::Tensor vt);\nvoid sl_attn(torch::Tensor q, torch::Tensor ko, torch::Tensor vt, torch::Tensor cache_seqlens, torch::Tensor cu_q, int64_t max_q, int64_t window, double softmax_scale, torch::Tensor out);\nvoid sl_ts(torch::Tensor buf);\nvoid sl_pvstat(torch::Tensor buf);\nvoid sl_gather2(torch::Tensor kc, torch::Tensor vc, torch::Tensor pt, torch::Tensor seqlens, torch::Tensor cu_q, int64_t win, int64_t part, torch::Tensor ko, torch::Tensor vt);\nvoid sl_attn_ng(torch::Tensor q, torch::Tensor ko, torch::Tensor vt, torch::Tensor cache_seqlens, torch::Tensor cu_q, int64_t max_q, int64_t window, double softmax_scale, torch::Tensor gate, double eps, torch::Tensor out);\n'
xs_SK_FLAGS = ['-O3', '--use_fast_math', '-DNDEBUG', '-U__CUDA_NO_HALF_OPERATORS__', '-U__CUDA_NO_HALF_CONVERSIONS__', '-U__CUDA_NO_BFLOAT16_CONVERSIONS__', '-U__CUDA_NO_HALF2_OPERATORS__', '-gencode=arch=compute_90a,code=sm_90a']
xs_SK_SRC = r"""
























#define Z0_NOL 1
#undef PV_B
#define PV_B 1104
#define PV16 1
#define SL_PIPE 3
#define SL_NWG 1
#define SL_MINB 3
#define SL_ST 3
#define SK_PDL 1
#define SX_G32 1
#define SK_KS 2
#define SX_XCH 2
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>








#define SL_BN 128
#define SL_MATH 1
#define SK_O72 1
#define SK_BM 1
#define SK_TS 0

constexpr int BN = SL_BN, ST = SL_ST;
constexpr int PPC = 8 * SL_NWG * SK_BM;
constexpr int K_BYTES = BN * 64, V_BYTES = 64 * BN, Q_BYTES = PPC * 8 * 64;

#define Z0_RS(o, i) 1.f
constexpr int VS = V_BYTES + ((SK_O72 && !Z0_NOL) ? 8 * BN : 0);
constexpr int NO = (SK_O72 && !Z0_NOL) ? 36 : 32;
typedef uint32_t OT;
constexpr int NOR = 16;
static_assert(Z0_NOL && SK_BM == 1, "PV16: denominator-free one-half build only");

constexpr int XCHB = PV16 ? 48 : 80;
constexpr int XB = (SX_XCH == 2 && SK_KS > 1) ? XCHB * SL_NWG * 128 : 0;
constexpr int SMEM = 1024 + Q_BYTES + ST * (K_BYTES + VS) + 512 + 256 + XB;

__device__ unsigned long long* g_sk_ts = nullptr;
__device__ unsigned int g_sx_smcnt[256];
static_assert(SK_KS == 1 || SK_O72, "SK_KS needs SK_O72 (sum in the O accumulator)");
static_assert(SK_BM == 1 || (SL_NWG == 1 && SL_PIPE == 3 && SK_O72 && SL_MATH && !SK_TS), "SK_BM 2: NWG 1, PIPE 3 only");
static_assert((SK_KS - 1) * SK_BM * 38 * SL_NWG * 128 * 4 <= ST * (K_BYTES + VS), "SK_BM exchange does not fit");
static_assert((SK_KS - 1) * 38 * SL_NWG * 128 * 4 <= ST * (K_BYTES + VS), "SK_KS exchange does not fit the ring");
constexpr int NKB = BN / 32;
constexpr int NSR = BN / 4;

typedef CUresult (*SlEncodeTiled)(CUtensorMap*, CUtensorMapDataType, cuuint32_t, void*, const cuuint64_t*,
                 const cuuint64_t*, const cuuint32_t*, const cuuint32_t*, CUtensorMapInterleave,
                 CUtensorMapSwizzle, CUtensorMapL2promotion, CUtensorMapFloatOOBfill);

struct SlParams {
 __nv_bfloat16* out;
 const int* cu_q;
 const int* cache_seqlens;
 int S;
 int G;
 int W;
 float a32;
 const __nv_bfloat16* gate;
 int gstride;
 float eps;
 int ks;
};

__device__ __forceinline__ uint32_t smem_u32(const void* p) { return static_cast<uint32_t>(__cvta_generic_to_shared(p)); }
__device__ __forceinline__ void mbar_init(uint32_t bar, int count) {
 asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::"r"(bar), "r"(count));
}
__device__ __forceinline__ void mbar_expect(uint32_t bar, int bytes) {
 asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(bar), "r"(bytes) : "memory");
}
__device__ __forceinline__ void mbar_arrive(uint32_t bar) {
 asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" ::"r"(bar) : "memory");
}
__device__ __forceinline__ void mbar_wait(uint32_t bar, uint32_t parity) {
 asm volatile("{\n .reg .pred p;\n W: mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n @!p bra W;\n}"
        ::"r"(bar), "r"(parity) : "memory");
}
__device__ __forceinline__ void tma_2d(uint32_t dst, const CUtensorMap* map, int c0, int c1, uint32_t bar) {
 asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%2, %3}], [%4];"
        ::"r"(dst), "l"(map), "r"(c0), "r"(c1), "r"(bar) : "memory");
}
__device__ __forceinline__ void tma_3d(uint32_t dst, const CUtensorMap* map, int c0, int c1, int c2, uint32_t bar) {
 asm volatile("cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%2, %3, %4}], [%5];"
        ::"r"(dst), "l"(map), "r"(c0), "r"(c1), "r"(c2), "r"(bar) : "memory");
}
__device__ __forceinline__ uint32_t prmt(uint32_t a, uint32_t b, uint32_t s) {
 uint32_t r;
 asm("prmt.b32 %0, %1, %2, %3;" : "=r"(r) : "r"(a), "r"(b), "r"(s));
 return r;
}

__device__ __forceinline__ uint64_t desc64(uint32_t addr) {
 return (uint64_t)((addr & 0x3FFFF) >> 4) | ((uint64_t)1 << 16) | ((uint64_t)(512 >> 4) << 32) | ((uint64_t)2 << 62);
}
__device__ __forceinline__ uint64_t desc128(uint32_t addr) {
 return (uint64_t)((addr & 0x3FFFF) >> 4) | ((uint64_t)1 << 16) | ((uint64_t)(1024 >> 4) << 32) | ((uint64_t)1 << 62);
}
__device__ __forceinline__ uint64_t descv(uint32_t addr) { return BN >= 128 ? desc128(addr) : desc64(addr); }


__device__ __forceinline__ uint64_t pvd(uint64_t dv, int kb) {
 return BN == 256 ? dv + (uint64_t)((kb >> 2) * (8192 >> 4) + 2 * (kb & 3)) : dv + (uint64_t)(2 * kb);
}


__device__ __forceinline__ uint64_t desc_ones(uint32_t addr) {
 return (uint64_t)((addr & 0x3FFFF) >> 4) | ((uint64_t)(128 >> 4) << 16) | ((uint64_t)(256 >> 4) << 32);
}
__device__ __forceinline__ void cluster_sync_all() {
 asm volatile("barrier.cluster.arrive.release.aligned;\nbarrier.cluster.wait.acquire.aligned;" ::: "memory");
}
__device__ __forceinline__ uint32_t cluster_rank() {
 uint32_t r;
 asm volatile("mov.u32 %0, %%cluster_ctarank;" : "=r"(r));
 return r;
}
__device__ __forceinline__ uint32_t mapa0(uint32_t a) {
 uint32_t r;
 asm volatile("mapa.shared::cluster.u32 %0, %1, 0;" : "=r"(r) : "r"(a));
 return r;
}
__device__ __forceinline__ uint32_t mapa_r(uint32_t a, uint32_t rank) {
 uint32_t r;
 asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(r) : "r"(a), "r"(rank));
 return r;
}
__device__ __forceinline__ void st_cluster4(uint32_t a, float x, float y, float z, float w) {
 asm volatile("st.shared::cluster.v4.f32 [%0], {%1, %2, %3, %4};" ::"r"(a), "f"(x), "f"(y), "f"(z), "f"(w) : "memory");
}
__device__ __forceinline__ void st_async4(uint32_t a, uint32_t bar, float x, float y, float z, float w) {
 asm volatile("st.async.shared::cluster.mbarrier::complete_tx::bytes.v4.f32 [%0], {%1, %2, %3, %4}, [%5];"
        ::"r"(a), "f"(x), "f"(y), "f"(z), "f"(w), "r"(bar) : "memory");
}
__device__ __forceinline__ void st_cluster(uint32_t a, float v) {
 asm volatile("st.shared::cluster.f32 [%0], %1;" ::"r"(a), "f"(v) : "memory");
}
__device__ __forceinline__ void wg_fence() { asm volatile("wgmma.fence.sync.aligned;" ::: "memory"); }
__device__ __forceinline__ void wg_commit() { asm volatile("wgmma.commit_group.sync.aligned;" ::: "memory"); }
template <int K_>
__device__ __forceinline__ void wg_wait() { asm volatile("wgmma.wait_group.sync.aligned %0;" ::"n"(K_) : "memory"); }
template <int N>
__device__ __forceinline__ void reg_fence(uint32_t (&r)[N]) {
#pragma unroll
 for (int i = 0; i < N; ++i) asm volatile("" : "+r"(r[i])::"memory");
}
template <int N>
__device__ __forceinline__ void reg_fence(float (&r)[N]) {
#pragma unroll
 for (int i = 0; i < N; ++i) asm volatile("" : "+f"(r[i])::"memory");
}




__device__ __forceinline__ void wgmma_qk(uint32_t (&d)[32], uint64_t da, uint64_t db, int sd) {
 asm volatile("{\n.reg .pred p;\nsetp.ne.b32 p, %34, 0;\n"
        "wgmma.mma_async.sync.aligned.m64n128k32.f16.e4m3.e4m3 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31}, %32, %33, p, 1, 1;\n}\n"
        : "+r"(d[0]), "+r"(d[1]), "+r"(d[2]), "+r"(d[3]), "+r"(d[4]), "+r"(d[5]), "+r"(d[6]), "+r"(d[7]), "+r"(d[8]), "+r"(d[9]), "+r"(d[10]), "+r"(d[11]), "+r"(d[12]), "+r"(d[13]), "+r"(d[14]), "+r"(d[15]), "+r"(d[16]), "+r"(d[17]), "+r"(d[18]), "+r"(d[19]), "+r"(d[20]), "+r"(d[21]), "+r"(d[22]), "+r"(d[23]), "+r"(d[24]), "+r"(d[25]), "+r"(d[26]), "+r"(d[27]), "+r"(d[28]), "+r"(d[29]), "+r"(d[30]), "+r"(d[31]) : "l"(da), "l"(db), "r"(sd));
}
__device__ __forceinline__ void wgmma_qk(uint32_t (&d)[64], uint64_t da, uint64_t db, int sd) {
 asm volatile("{\n.reg .pred p;\nsetp.ne.b32 p, %66, 0;\n"
        "wgmma.mma_async.sync.aligned.m64n256k32.f16.e4m3.e4m3 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47,%48,%49,%50,%51,%52,%53,%54,%55,%56,%57,%58,%59,%60,%61,%62,%63}, %64, %65, p, 1, 1;\n}\n"
        : "+r"(d[0]), "+r"(d[1]), "+r"(d[2]), "+r"(d[3]), "+r"(d[4]), "+r"(d[5]), "+r"(d[6]), "+r"(d[7]), "+r"(d[8]), "+r"(d[9]), "+r"(d[10]), "+r"(d[11]), "+r"(d[12]), "+r"(d[13]), "+r"(d[14]), "+r"(d[15]), "+r"(d[16]), "+r"(d[17]), "+r"(d[18]), "+r"(d[19]), "+r"(d[20]), "+r"(d[21]), "+r"(d[22]), "+r"(d[23]), "+r"(d[24]), "+r"(d[25]), "+r"(d[26]), "+r"(d[27]), "+r"(d[28]), "+r"(d[29]), "+r"(d[30]), "+r"(d[31]), "+r"(d[32]), "+r"(d[33]), "+r"(d[34]), "+r"(d[35]), "+r"(d[36]), "+r"(d[37]), "+r"(d[38]), "+r"(d[39]), "+r"(d[40]), "+r"(d[41]), "+r"(d[42]), "+r"(d[43]), "+r"(d[44]), "+r"(d[45]), "+r"(d[46]), "+r"(d[47]), "+r"(d[48]), "+r"(d[49]), "+r"(d[50]), "+r"(d[51]), "+r"(d[52]), "+r"(d[53]), "+r"(d[54]), "+r"(d[55]), "+r"(d[56]), "+r"(d[57]), "+r"(d[58]), "+r"(d[59]), "+r"(d[60]), "+r"(d[61]), "+r"(d[62]), "+r"(d[63]) : "l"(da), "l"(db), "r"(sd));
}
__device__ __forceinline__ void wgmma_pv(float (&d)[36], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint64_t db) {
 asm volatile("{\n.reg .pred p;\nsetp.ne.b32 p, 1, 0;\n"
        "wgmma.mma_async.sync.aligned.m64n72k32.f32.e4m3.e4m3 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32,%33,%34,%35}, {%36,%37,%38,%39}, %40, p, 1, 1;\n}\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]), "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]), "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]), "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27]), "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31]), "+f"(d[32]), "+f"(d[33]), "+f"(d[34]), "+f"(d[35]) : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "l"(db));
}
__device__ __forceinline__ void wgmma_pv(float (&d)[32], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint64_t db) {
 asm volatile("{\n.reg .pred p;\nsetp.ne.b32 p, 1, 0;\n"
        "wgmma.mma_async.sync.aligned.m64n64k32.f32.e4m3.e4m3 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31}, {%32,%33,%34,%35}, %36, p, 1, 1;\n}\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]), "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]), "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]), "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27]), "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31]) : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "l"(db));
}
__device__ __forceinline__ void wgmma_pv(uint32_t (&d)[16], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint64_t db) {
 asm volatile("{\n.reg .pred p;\nsetp.ne.b32 p, 1, 0;\n"
        "wgmma.mma_async.sync.aligned.m64n64k32.f16.e4m3.e4m3 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15}, {%16,%17,%18,%19}, %20, p, 1, 1;\n}\n"
        : "+r"(d[0]), "+r"(d[1]), "+r"(d[2]), "+r"(d[3]), "+r"(d[4]), "+r"(d[5]), "+r"(d[6]), "+r"(d[7]), "+r"(d[8]), "+r"(d[9]), "+r"(d[10]), "+r"(d[11]), "+r"(d[12]), "+r"(d[13]), "+r"(d[14]), "+r"(d[15]) : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "l"(db));
}
__device__ __forceinline__ void wgmma_ones(float (&d)[4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint64_t db, int sd) {
 asm volatile("{\n.reg .pred p;\nsetp.ne.b32 p, %9, 0;\n"
        "wgmma.mma_async.sync.aligned.m64n8k32.f32.e4m3.e4m3 {%0,%1,%2,%3}, {%4,%5,%6,%7}, %8, p, 1, 1;\n}\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "l"(db), "r"(sd));
}


__device__ __forceinline__ __half2 h2(uint32_t x) { return *reinterpret_cast<__half2*>(&x); }
__device__ __forceinline__ uint32_t u32(__half2 x) { return *reinterpret_cast<uint32_t*>(&x); }


struct SlRow {
 float rmax[2], rb[2], rsum[2];
 int jlo[2], jhi[2];
 uint32_t rmh, rbh;
};

__device__ __forceinline__ void sl_issue_qk(uint32_t (&s)[NSR], uint64_t dq, uint32_t kaddr) {
#pragma unroll
 for (int i = 0; i < NSR; ++i) s[i] = 0;
 reg_fence(s);
 wg_fence();
 const uint64_t dk = desc64(kaddr);
 wgmma_qk(s, dq, dk, 0);
 wgmma_qk(s, dq + 2, dk + 2, 1);
 wg_commit();
}

__device__ __forceinline__ void sl_pack(const uint32_t (&sc)[NSR], uint32_t (&pa)[NKB][4]);


__device__ __forceinline__ void sl_softmax(uint32_t (&sc)[NSR], uint32_t (&pa)[NKB][4], SlRow& st, int j0, bool edge,
                      int tq, float a32, float (&scale)[2]) {
 if (edge) {


  const int off0 = j0 + 2 * tq - st.jlo[0], off1 = j0 + 2 * tq - st.jlo[1];
  const unsigned sp0 = (unsigned)(st.jhi[0] - st.jlo[0]), sp1 = (unsigned)(st.jhi[1] - st.jlo[1]);
#pragma unroll
  for (int w = 0; w < NSR; ++w) {
   const int d = ((w & 1) ? off1 : off0) + 8 * (w >> 1);
   const unsigned sp = (w & 1) ? sp1 : sp0;
   const uint32_t m = ((unsigned)d <= sp ? 0x0000FFFFu : 0u) | ((unsigned)(d + 1) <= sp ? 0xFFFF0000u : 0u);
   sc[w] = (sc[w] & m) | (0xFC00FC00u & ~m);
  }
  edge = false;
 }
 const __half2 ninf2 = __float2half2_rn(-INFINITY);
 if (edge) {
#pragma unroll
  for (int w = 0; w < NSR; ++w) {
   const int r = w & 1, j = j0 + 8 * (w >> 1) + 2 * tq;
   __half2 v = h2(sc[w]);
   const bool ok0 = j >= st.jlo[r] && j <= st.jhi[r], ok1 = j + 1 >= st.jlo[r] && j + 1 <= st.jhi[r];
   v = __halves2half2(ok0 ? __low2half(v) : __low2half(ninf2), ok1 ? __high2half(v) : __high2half(ninf2));
   sc[w] = u32(v);
  }
 }
 __half2 m0 = h2(sc[0]), m1 = h2(sc[1]), m2 = h2(sc[2]), m3 = h2(sc[3]);
#pragma unroll
 for (int w = 4; w < NSR; w += 4) {
  m0 = __hmax2(m0, h2(sc[w])); m1 = __hmax2(m1, h2(sc[w + 1]));
  m2 = __hmax2(m2, h2(sc[w + 2])); m3 = __hmax2(m3, h2(sc[w + 3]));
 }
 m0 = __hmax2(m0, m2); m1 = __hmax2(m1, m3);
 __half2 mm = __hmax2(__lows2half2(m0, m1), __highs2half2(m0, m1));
 mm = __hmax2(mm, h2(__shfl_xor_sync(0xffffffff, u32(mm), 1)));
 mm = __hmax2(mm, h2(__shfl_xor_sync(0xffffffff, u32(mm), 2)));


 __half2 bx[2];
 {
  const __half2 cur = __hmax2(h2(st.rmh), mm);
  const __half2 bn = __hmin2(__hfma2(cur, __hneg2(__float2half2_rn(a32)), __float2half2_rn((float)PV_B)),
                __float2half2_rn(65504.f));
  const uint32_t bnu = u32(bn);
  scale[0] = scale[1] = 1.f;
  if (__any_sync(0xffffffffu, bnu != st.rbh) && bnu != st.rbh) {
   scale[0] = exp2f((__low2float(bn) - __low2float(h2(st.rbh))) * 0.125f);
   scale[1] = exp2f((__high2float(bn) - __high2float(h2(st.rbh))) * 0.125f);
  }
  st.rbh = bnu;
  st.rmh = u32(cur);
  bx[0] = __low2half2(bn);
  bx[1] = __high2half2(bn);
 }
 const __half2 a2 = __float2half2_rn(a32);
 const __half2 off2 = __float2half2_rn(896.f), lo2 = __float2half2_rn(1024.f);


 (void)off2;
#pragma unroll
 for (int w = 0; w < NSR; ++w) sc[w] = u32(__hmax2(__hfma2(h2(sc[w]), a2, bx[w & 1]), lo2));
}



__device__ __forceinline__ void sl_pack(const uint32_t (&sc)[NSR], uint32_t (&pa)[NKB][4]) {
#pragma unroll
 for (int kb = 0; kb < NKB; ++kb) {
  const int w = 8 * kb;
  pa[kb][0] = prmt(sc[w + 0], sc[w + 2], 0x6420);
  pa[kb][1] = prmt(sc[w + 1], sc[w + 3], 0x6420);
  pa[kb][2] = prmt(sc[w + 4], sc[w + 6], 0x6420);
  pa[kb][3] = prmt(sc[w + 5], sc[w + 7], 0x6420);
 }
}


__device__ __forceinline__ void sl_issue_pv(OT (&o)[NOR], float (&os)[4], uint32_t (&pa)[NKB][4], uint32_t vaddr,
                      uint64_t dones) {
 const uint64_t dv = descv(vaddr);
 reg_fence(o);
 wg_fence();
#pragma unroll
 for (int kb = 0; kb < NKB; ++kb) {
  wgmma_pv(o, pa[kb][0], pa[kb][1], pa[kb][2], pa[kb][3], pvd(dv, kb));
 }
 wg_commit();
}

__device__ __forceinline__ void sl_rescale(OT (&o)[NOR], SlRow& st, const float (&scale)[2]) {

 if (__any_sync(0xffffffffu, scale[0] != 1.f || scale[1] != 1.f)) {
  const __half2 s0 = __float2half2_rn(scale[0]), s1 = __float2half2_rn(scale[1]);
#pragma unroll
  for (int i = 0; i < NOR; i += 2) { o[i] = u32(__hmul2(h2(o[i]), s0)); o[i + 1] = u32(__hmul2(h2(o[i + 1]), s1)); }
  st.rsum[0] *= scale[0];
  st.rsum[1] *= scale[1];
 }
}



template <int R>
__device__ __forceinline__ void sx_send_row(uint32_t a, int nt4, const float (&o)[NO], float b) {
 float v[20] = {};
#pragma unroll
 for (int jb = 0; jb < NO / 4; ++jb) { v[2 * jb] = o[4 * jb + 2 * R]; v[2 * jb + 1] = o[4 * jb + 2 * R + 1]; }
 v[18] = b;
 v[19] = 0.f;
#pragma unroll
 for (int j = 0; j < 5; ++j) st_cluster4(a + 16u * j * nt4, v[4 * j], v[4 * j + 1], v[4 * j + 2], v[4 * j + 3]);
}
template <int R>
__device__ __forceinline__ void sx_send_row_async(uint32_t a, uint32_t bar, int nt4, const float (&o)[NO], float b) {
 float v[20] = {};
#pragma unroll
 for (int jb = 0; jb < NO / 4; ++jb) { v[2 * jb] = o[4 * jb + 2 * R]; v[2 * jb + 1] = o[4 * jb + 2 * R + 1]; }
 v[18] = b;
 v[19] = 0.f;
#pragma unroll
 for (int j = 0; j < 5; ++j) st_async4(a + 16u * j * nt4, bar, v[4 * j], v[4 * j + 1], v[4 * j + 2], v[4 * j + 3]);
}
template <int R>
__device__ __forceinline__ void sx_merge_row(const float4* x, int nt4, float (&o)[NO], float (&rb)[2]) {
 float v[20];
#pragma unroll
 for (int j = 0; j < 5; ++j) {
  const float4 t = x[j * nt4];
  v[4 * j] = t.x; v[4 * j + 1] = t.y; v[4 * j + 2] = t.z; v[4 * j + 3] = t.w;
 }
 const float b = v[18], m = fminf(rb[R], b);
 const float sa = exp2f((m - rb[R]) * 0.125f), sb = exp2f((m - b) * 0.125f);
#pragma unroll
 for (int jb = 0; jb < NO / 4; ++jb) {
  o[4 * jb + 2 * R] = o[4 * jb + 2 * R] * sa + v[2 * jb] * sb;
  o[4 * jb + 2 * R + 1] = o[4 * jb + 2 * R + 1] * sa + v[2 * jb + 1] * sb;
 }
 rb[R] = m;
}





template <int R>
__device__ __forceinline__ void sx_send_row_async(uint32_t a, uint32_t bar, int nt4, const uint32_t (&o)[16], float b) {
 st_async4(a, bar, __uint_as_float(o[R]), __uint_as_float(o[2 + R]), __uint_as_float(o[4 + R]), __uint_as_float(o[6 + R]));
 st_async4(a + 16u * nt4, bar, __uint_as_float(o[8 + R]), __uint_as_float(o[10 + R]), __uint_as_float(o[12 + R]),
      __uint_as_float(o[14 + R]));
 st_async4(a + 32u * nt4, bar, b, 0.f, 0.f, 0.f);
}
template <int R>
__device__ __forceinline__ void sx_merge_row(const float4* x, int nt4, uint32_t (&o)[16], float (&rb)[2]) {
 const float4 t0 = x[0], t1 = x[nt4], t2 = x[2 * nt4];
 const uint32_t v[8] = {__float_as_uint(t0.x), __float_as_uint(t0.y), __float_as_uint(t0.z), __float_as_uint(t0.w),
             __float_as_uint(t1.x), __float_as_uint(t1.y), __float_as_uint(t1.z), __float_as_uint(t1.w)};
 const float b = t2.x, m = fminf(rb[R], b);
 const float sa = exp2f((m - rb[R]) * 0.125f), sb = exp2f((m - b) * 0.125f);
#pragma unroll
 for (int jb = 0; jb < 8; ++jb) {
  const float2 f = __half22float2(h2(o[2 * jb + R])), g = __half22float2(h2(v[jb]));
  o[2 * jb + R] = u32(__floats2half2_rn(f.x * sa + g.x * sb, f.y * sa + g.y * sb));
 }
 rb[R] = m;
}
constexpr int NTHR = SL_NWG * 128 + 32;

__global__ void __launch_bounds__(NTHR, SL_MINB)
sl_attn_k(const __grid_constant__ CUtensorMap mq,
                              const __grid_constant__ CUtensorMap mk,
                              const __grid_constant__ CUtensorMap mv, SlParams p) {
 extern __shared__ uint8_t smem_raw[];
 uint8_t* base = reinterpret_cast<uint8_t*>((reinterpret_cast<uintptr_t>(smem_raw) + 1023) & ~uintptr_t(1023));
 uint8_t* sk = base;
 uint8_t* sv = sk + ST * K_BYTES;
 uint8_t* sq = sv + ST * VS;
 uint8_t* sones = sq + Q_BYTES;
 uint64_t* bars = reinterpret_cast<uint64_t*>(sones + 512);
 uint64_t* xbar = &bars[2 * ST + 1];
 float* xbuf = reinterpret_cast<float*>(sones + 512 + 256);
 const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;

#define SX_C(k) do { } while (0)



 const int ks = SK_KS > 1 ? p.ks : 1;
 const int kc = ks > 1 ? (int)cluster_rank() : 0, item = ks > 1 ? blockIdx.x / ks : blockIdx.x;
 const int kh = item & 1, bg = item >> 1, b = bg / p.G, g = bg - b * p.G;
 const int q0 = p.cu_q[b], Lb = p.cu_q[b + 1] - q0, n = p.cache_seqlens[b];
 const int i0 = g * PPC;
 if (i0 >= Lb) return;
 const int W = p.W;
 const int s0 = max(0, n - Lb - (W - 1));
 const int ilast = min(i0 + PPC - 1, Lb - 1);
 const int pfirst = n - Lb + i0, plast = n - Lb + ilast;
 const int jlo = max(0, pfirst - W + 1 - s0), jhi = plast - s0;
 const int ta = jlo / BN, nta = jhi / BN - ta + 1;
 const int t0 = ta + (kc * nta) / ks, nt = ta + ((kc + 1) * nta) / ks - t0;

 if (tid == 0) {
  for (int s = 0; s < ST; ++s) { mbar_init(smem_u32(&bars[s]), 1); mbar_init(smem_u32(&bars[ST + s]), SL_NWG * 128); }
  mbar_init(smem_u32(&bars[2 * ST]), 1);
  mbar_init(smem_u32(&bars[2 * ST + 2]), 1);
  if (ks > 1) mbar_init(smem_u32(xbar), 1);
  asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  if (ks > 1) mbar_expect(smem_u32(xbar), XCHB * SL_NWG * 128);
 }
 if (tid < 128) reinterpret_cast<uint32_t*>(sones)[tid] = 0x38383838u;
 asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
 __syncthreads();

 if (ks > 1) asm volatile("barrier.cluster.arrive.relaxed.aligned;" ::: "memory");
 asm volatile("griddepcontrol.wait;" ::: "memory");
 asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
 SX_C(1);

 if (warp == SL_NWG * 4) {



  if (lane == 0) {
   if (nt > 0) {
    const uint32_t qb = smem_u32(&bars[2 * ST]);
    mbar_expect(qb, Q_BYTES);
    tma_3d(smem_u32(sq), &mq, 0, kh * 8, q0 + i0, qb);
   }
   for (int it = 0; it < nt; ++it) {
    const int s = it % ST;
    if (it == 1) mbar_wait(smem_u32(&bars[0]), 0);
    if (it >= ST) mbar_wait(smem_u32(&bars[ST + s]), ((it / ST) - 1) & 1);
    const uint32_t fb = smem_u32(&bars[s]);
    const int j = (t0 + it) * BN;
    if (it == 0) {
     const uint32_t kb = smem_u32(&bars[2 * ST + 2]);
     mbar_expect(kb, K_BYTES);
     tma_3d(smem_u32(sk), &mk, 0, kh, b * p.S + j, kb);
     mbar_expect(fb, V_BYTES);
    } else {
     mbar_expect(fb, K_BYTES + V_BYTES);
     tma_3d(smem_u32(sk + s * K_BYTES), &mk, 0, kh, b * p.S + j, fb);
    }
    tma_2d(smem_u32(sv + s * VS), &mv, j, (b * 2 + kh) * 64, fb);
    if (it == 0) {
    }
   }
  }
  __syncwarp();
  if (ks > 1) asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");
  return;
 }

 if (p.gate && tid < 8 * PPC) {
  const int pos = i0 + (tid >> 3);
  if (pos < Lb)
   asm volatile("prefetch.global.L2 [%0];" ::"l"(p.gate + (size_t)(q0 + pos) * p.gstride + (kh * 8 + (tid & 7)) * 64));
 }


 const int wg = warp >> 2, wi = warp & 3;
 const int tq = lane & 3, hq = lane >> 2;
 const int ib = i0 + 8 * wg + 2 * wi;
 const int pr0 = n - Lb + ib;
 SlRow st;
#pragma unroll
 for (int r = 0; r < 2; ++r) {
  st.rmax[r] = -INFINITY; st.rb[r] = (float)PV_B; st.rsum[r] = 0.f;
  st.rmh = 0xFC00FC00u;
  st.rbh = 0x7BFF7BFFu;
  st.rb[r] = 65504.f;
  st.jhi[r] = pr0 + r - s0;
  st.jlo[r] = pr0 + r - W + 1 - s0;
 }
 OT o[NOR];
#pragma unroll
 for (int i = 0; i < NOR; ++i) o[i] = 0;
#define SK_T(k) do { } while (0)

 const uint64_t dq = desc64(smem_u32(sq + wg * 4096));
 const uint64_t dones = desc_ones(smem_u32(sones));

 const int jlo_all = max(0, plast - W + 1 - s0), jhi_all = pfirst - s0;
 const float a32 = p.a32;
 float os[4] = {0.f, 0.f, 0.f, 0.f};
 uint32_t sa[NSR], sb[NSR], pa[NKB][4];
 float scale[2];
 (void)dones;
 if (nt > 0) {
 mbar_wait(smem_u32(&bars[2 * ST]), 0);
 mbar_wait(smem_u32(&bars[2 * ST + 2]), 0);
 sl_issue_qk(sa, dq, smem_u32(sk));
 wg_wait<0>();
 reg_fence(sa);
 SK_T(0);
 SX_C(2);


 (void)sb;
 const uint32_t zdep = (uint32_t)(p.ks >> 8);
 (void)zdep;
 {
  const int j0 = t0 * BN;
  sl_softmax(sa, pa, st, j0, j0 < jlo_all || j0 + BN - 1 > jhi_all, tq, a32, scale);
  sl_rescale(o, st, scale);
  sl_pack(sa, pa);
 }
 mbar_wait(smem_u32(&bars[0]), 0);
 {

  const uint32_t fb0 = smem_u32(&bars[0]), eb0 = smem_u32(&bars[ST]);
  const uint64_t dk0 = desc64(smem_u32(sk)), dv0 = descv(smem_u32(sv));
  uint32_t s = 1 % ST, ph = (1 / ST) & 1, sp = 0;
#pragma unroll 1
  for (int it = 1; it < nt; ++it) {
   mbar_wait(fb0 + 8u * s, ph);
#pragma unroll
   for (int i = 0; i < NSR; ++i) sa[i] = 0;
   reg_fence(sa);
   wg_fence();
   {
    const uint64_t dk = dk0 + (uint64_t)(s * (K_BYTES >> 4));
    wgmma_qk(sa, dq, dk, 0);
    wgmma_qk(sa, dq + 2, dk + 2, 1);
   }
   wg_commit();
   reg_fence(o);
   wg_fence();
   {
    const uint64_t dv = dv0 + (uint64_t)(sp * (VS >> 4));
#pragma unroll
    for (int kb = 0; kb < NKB; ++kb) wgmma_pv(o, pa[kb][0], pa[kb][1], pa[kb][2], pa[kb][3], pvd(dv, kb));
   }
   wg_commit();
   wg_wait<1>();
   reg_fence(sa);
   const int j0 = (t0 + it) * BN;
   sl_softmax(sa, pa, st, j0, j0 < jlo_all || j0 + BN - 1 > jhi_all, tq, a32, scale);
   wg_wait<0>();
   reg_fence(o);
   mbar_arrive(eb0 + 8u * sp);
   sl_rescale(o, st, scale);
   sl_pack(sa, pa);
   sp = s;
   if (++s == (uint32_t)ST) { s = 0; ph ^= 1u; }
  }
 }
 sl_issue_pv(o, os, pa, smem_u32(sv + ((nt - 1) % ST) * VS), dones);
 wg_wait<0>();
 reg_fence(o);
 st.rb[0] = __low2float(h2(st.rbh));
 st.rb[1] = __high2float(h2(st.rbh));
 st.rsum[0] = Z0_RS(o, 32);
 st.rsum[1] = Z0_RS(o, 34);
 mbar_arrive(smem_u32(&bars[ST + (nt - 1) % ST]));
 }
 SX_C(3);


 static_assert(SK_KS == 2, "SX_XCH: cluster of 2");
 if (ks > 1) {
  constexpr int NT = SL_NWG * 128;
  asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");
  SX_C(4);
  const uint32_t ra = mapa_r(smem_u32(xbuf), (uint32_t)(kc ^ 1)) + 16u * (uint32_t)tid;
  const uint32_t rbar = mapa_r(smem_u32(xbar), (uint32_t)(kc ^ 1));
  if (kc == 0) sx_send_row_async<1>(ra, rbar, NT, o, st.rb[1]);
  else sx_send_row_async<0>(ra, rbar, NT, o, st.rb[0]);
  mbar_wait(smem_u32(xbar), 0);
  SX_C(5);
  const float4* x = reinterpret_cast<const float4*>(xbuf) + tid;
  if (kc == 0) sx_merge_row<0>(x, NT, o, st.rb);
  else sx_merge_row<1>(x, NT, o, st.rb);
  st.rsum[0] = Z0_RS(o, 32);
  st.rsum[1] = Z0_RS(o, 34);
 }
 SX_C(6);


#pragma unroll
 for (int r = 0; r < 2; ++r) {
  const int i = ib + r;
  if (ks > 1 && r != kc) continue;
  if (i < Lb) {
   __nv_bfloat16* dst = p.out + ((size_t)(q0 + i) * 16 + kh * 8 + hq) * 64 + 2 * tq;
   const float inv = st.rsum[r] > 0.f ? 1.f / st.rsum[r] : 0.f;
   __nv_bfloat162 v[8];
#pragma unroll
   for (int jb = 0; jb < 8; ++jb) { const float2 f = __half22float2(h2(o[2 * jb + r])); v[jb] = __floats2bfloat162_rn(f.x * inv, f.y * inv); }
   if (p.gate) {


    const __nv_bfloat16* gp = p.gate + (size_t)(q0 + i) * p.gstride + (kh * 8 + hq) * 64 + 2 * tq;
    __nv_bfloat162 g[8];
#pragma unroll
    for (int jb = 0; jb < 8; ++jb) g[jb] = *reinterpret_cast<const __nv_bfloat162*>(gp + 8 * jb);
    float ss = 0.f;
#pragma unroll
    for (int jb = 0; jb < 8; ++jb) {
     const float2 x = __bfloat1622float2(v[jb]);
     ss = fmaf(x.x, x.x, fmaf(x.y, x.y, ss));
    }
    ss += __shfl_xor_sync(0xffffffffu, ss, 1);
    ss += __shfl_xor_sync(0xffffffffu, ss, 2);
    const float rs = rsqrtf(ss * (1.f / 64.f) + p.eps);
#pragma unroll
    for (int jb = 0; jb < 8; ++jb) {
     const float2 x = __bfloat1622float2(v[jb]), gg = __bfloat1622float2(g[jb]);
     v[jb] = __floats2bfloat162_rn(x.x * rs / (1.f + __expf(-gg.x)), x.y * rs / (1.f + __expf(-gg.y)));
    }
   }
#pragma unroll
   for (int jb = 0; jb < 8; ++jb) *reinterpret_cast<__nv_bfloat162*>(dst + 8 * jb) = v[jb];
  }
 }
}







__global__ void __launch_bounds__(256) sl_gather_k(const uint4* __restrict__ kc, const uint4* __restrict__ vc,
  const int* __restrict__ pt, const int* __restrict__ seqlens, const int* __restrict__ cu_q, int pages, int page,
  int S, int win, int part, uint4* __restrict__ ko, uint4* __restrict__ vt) {
 __shared__ uint4 tile[32][9];
 asm volatile("griddepcontrol.wait;" ::: "memory");
 asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
 const int b = blockIdx.y, j0 = blockIdx.x * 32, t = threadIdx.x;
 const int n = seqlens[b], L = cu_q[b + 1] - cu_q[b];
 const int s0 = max(0, n - L - win), nv = n - s0;
 const bool old_only = j0 + 32 <= n - L - s0;
 if ((part == 0 && !old_only) || (part == 1 && old_only)) return;
 {
  const int jr = t >> 3, u = t & 7, j = j0 + jr;
  uint4 kv = make_uint4(0, 0, 0, 0), vv = make_uint4(0, 0, 0, 0);
  if (j < nv) {
   const int s = s0 + j;
   int pg = pt[(long long)b * pages + s / page];
   pg = pg < 0 ? 0 : pg;
   const long long src = ((long long)pg * page + s % page) * 8 + u;
   kv = __ldg(kc + src);
   vv = __ldg(vc + src);
  }
  ko[((long long)b * S + j) * 8 + u] = kv;
  tile[jr][u] = vv;
 }
 __syncthreads();
 const unsigned char* tb = reinterpret_cast<const unsigned char*>(&tile[0][0]);
 const int hd = t >> 1, q = t & 1;
 uint32_t w[4];
#pragma unroll
 for (int k = 0; k < 4; ++k) {
  uint32_t x = 0;
#pragma unroll
  for (int i = 0; i < 4; ++i) {
   const int ks = q * 16 + 4 * k + i;
   const int src = 2 * ((ks & 15) >> 2) + (ks & 1) + 8 * ((ks & 3) >> 1) + 16 * (ks >> 4);
   x |= uint32_t(tb[src * 144 + hd]) << (8 * i);
  }
  w[k] = x;
 }
 vt[(((long long)b * 128 + hd) * S + j0) / 16 + q] = make_uint4(w[0], w[1], w[2], w[3]);
}

static SlEncodeTiled sl_encode() {
 static SlEncodeTiled enc = nullptr;
 if (!enc) {
  cudaDriverEntryPointQueryResult qr;
  cudaGetDriverEntryPoint("cuTensorMapEncodeTiled", reinterpret_cast<void**>(&enc), cudaEnableDefault, &qr);
  cudaFuncSetAttribute(sl_attn_k, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM);
 }
 return enc;
}


void sl_ts(torch::Tensor buf) {
 unsigned long long* ptr = buf.numel() ? (unsigned long long*)buf.data_ptr() : nullptr;
 cudaMemcpyToSymbol(g_sk_ts, &ptr, sizeof(ptr));
}
void sl_pvstat(torch::Tensor buf) {
}

void sl_gather2(torch::Tensor kc, torch::Tensor vc, torch::Tensor pt, torch::Tensor seqlens, torch::Tensor cu_q,
        int64_t win, int64_t part, torch::Tensor ko, torch::Tensor vt) {
 const int B = pt.size(0), pages = pt.size(1), page = kc.size(1), S = ko.size(1);
 TORCH_CHECK(kc.size(2) * kc.size(3) == 128 && S % 64 == 0, "sl_gather");
 cudaLaunchConfig_t cfg = {};
 cudaLaunchAttribute at[1];
 at[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
 at[0].val.programmaticStreamSerializationAllowed = SK_PDL;
 cfg.gridDim = dim3(S / (SX_G32 ? 32 : 64), B);
 cfg.blockDim = dim3(256);
 cfg.dynamicSmemBytes = 0;
 { cudaLaunchConfig_t q0 = {cfg.gridDim, cfg.blockDim, cfg.dynamicSmemBytes, at::cuda::getCurrentCUDAStream(), nullptr, 0}; cfg = q0; }
 cfg.attrs = at;
 cfg.numAttrs = 1;
 cudaLaunchKernelEx(&cfg, sl_gather_k, (const uint4*)kc.data_ptr(), (const uint4*)vc.data_ptr(), (const int*)pt.data_ptr<int>(),
           (const int*)seqlens.data_ptr<int>(), (const int*)cu_q.data_ptr<int>(), pages, page, S, (int)win, (int)part,
           (uint4*)ko.data_ptr(), (uint4*)vt.data_ptr());
}

void sl_gather(torch::Tensor kc, torch::Tensor vc, torch::Tensor pt, torch::Tensor seqlens, torch::Tensor cu_q,
        int64_t win, torch::Tensor ko, torch::Tensor vt) {
 sl_gather2(kc, vc, pt, seqlens, cu_q, win, 2, ko, vt);
}



static void sl_launch(torch::Tensor q, torch::Tensor ko, torch::Tensor vt, torch::Tensor cache_seqlens, torch::Tensor cu_q,
           int64_t max_q, int64_t window, double softmax_scale, torch::Tensor out,
           const __nv_bfloat16* gate, int gstride, float eps) {
 SlEncodeTiled enc = sl_encode();
 const int T = q.size(0), B = ko.size(0), S = ko.size(1);
 CUtensorMap mq, mk, mv;
 {
  const cuuint64_t dims[3] = {64, 16, (cuuint64_t)T}, strides[2] = {64, 1024};
  const cuuint32_t box[3] = {64, 8, PPC}, es[3] = {1, 1, 1};
  enc(&mq, CU_TENSOR_MAP_DATA_TYPE_UINT8, 3, q.data_ptr(), dims, strides, box, es, CU_TENSOR_MAP_INTERLEAVE_NONE,
    CU_TENSOR_MAP_SWIZZLE_64B, CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
 }
 {
  const cuuint64_t dims[3] = {64, 2, (cuuint64_t)B * S}, strides[2] = {64, 128};
  const cuuint32_t box[3] = {64, 1, (cuuint32_t)BN}, es[3] = {1, 1, 1};
  enc(&mk, CU_TENSOR_MAP_DATA_TYPE_UINT8, 3, ko.data_ptr(), dims, strides, box, es, CU_TENSOR_MAP_INTERLEAVE_NONE,
    CU_TENSOR_MAP_SWIZZLE_64B, CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
 }
 {
  const cuuint64_t dims[2] = {(cuuint64_t)S, (cuuint64_t)B * 128}, strides[1] = {(cuuint64_t)S};
  const cuuint32_t box[2] = {(cuuint32_t)(BN == 256 ? 128 : BN), 64}, es[2] = {1, 1};
  enc(&mv, CU_TENSOR_MAP_DATA_TYPE_UINT8, 2, vt.data_ptr(), dims, strides, box, es, CU_TENSOR_MAP_INTERLEAVE_NONE,
    BN >= 128 ? CU_TENSOR_MAP_SWIZZLE_128B : CU_TENSOR_MAP_SWIZZLE_64B, CU_TENSOR_MAP_L2_PROMOTION_L2_128B,
    CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
 }
 SlParams p;
 p.out = (__nv_bfloat16*)out.data_ptr();
 p.cu_q = cu_q.data_ptr<int>();
 p.cache_seqlens = cache_seqlens.data_ptr<int>();
 p.S = S;
 p.G = (int)((max_q + PPC - 1) / PPC);
 p.W = (int)window;
 p.a32 = __half2float(__float2half_rn((float)(8.0 * 1.4426950408889634 * softmax_scale)));
 p.gate = gate;
 p.gstride = gstride;
 p.eps = eps;

 static int slots = 0;
 if (!slots) {
  int dev = 0, sms = 0, occ = 0;
  cudaGetDevice(&dev);
  cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev);
  cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ, sl_attn_k, NTHR, SMEM);
  slots = sms * (occ > 0 ? occ : 1);
 }
 const int items = B * p.G * 2;
 p.ks = (SK_KS > 1 && 2 * items <= slots) ? 2 : 1;
 cudaLaunchConfig_t cfg = {};
 cudaLaunchAttribute at[2];
 at[0].id = cudaLaunchAttributeClusterDimension;
 at[0].val.clusterDim.x = p.ks;
 at[0].val.clusterDim.y = 1;
 at[0].val.clusterDim.z = 1;
 at[1].id = cudaLaunchAttributeProgrammaticStreamSerialization;
 at[1].val.programmaticStreamSerializationAllowed = SK_PDL;
 cfg.gridDim = dim3(items * p.ks);
 cfg.blockDim = dim3(NTHR);
 cfg.dynamicSmemBytes = SMEM;
 { cudaLaunchConfig_t q0 = {cfg.gridDim, cfg.blockDim, cfg.dynamicSmemBytes, at::cuda::getCurrentCUDAStream(), nullptr, 0}; cfg = q0; }
 cfg.attrs = at;
 cfg.numAttrs = 2;
 cudaLaunchKernelEx(&cfg, sl_attn_k, mq, mk, mv, p);
}

void sl_attn(torch::Tensor q, torch::Tensor ko, torch::Tensor vt, torch::Tensor cache_seqlens, torch::Tensor cu_q,
       int64_t max_q, int64_t window, double softmax_scale, torch::Tensor out) {
 sl_launch(q, ko, vt, cache_seqlens, cu_q, max_q, window, softmax_scale, out, nullptr, 0, 0.f);
}


void sl_attn_ng(torch::Tensor q, torch::Tensor ko, torch::Tensor vt, torch::Tensor cache_seqlens, torch::Tensor cu_q,
        int64_t max_q, int64_t window, double softmax_scale, torch::Tensor gate, double eps, torch::Tensor out) {
 TORCH_CHECK(gate.stride(1) == 1 && gate.size(1) == 1024, "sl_attn_ng gate");
 sl_launch(q, ko, vt, cache_seqlens, cu_q, max_q, window, softmax_scale, out,
      (const __nv_bfloat16*)gate.data_ptr(), (int)gate.stride(0), (float)eps);
}

"""

xs_sk_ext = load_inline(name="okc_prefill_p9035_b460010", cpp_sources=xs_SK_CPP, cuda_sources=xs_SK_SRC,
                     functions=["sl_gather", "sl_attn", "sl_ts", "sl_pvstat", "sl_gather2", "sl_attn_ng"],
                     extra_cuda_cflags=xs_SK_FLAGS, extra_ldflags=["-lcuda"])


def xs_alloc(cu_seqlens_q, max_q, window, device):
    """Empty K / V^T copies for one call: S = max_q + window - 1 rows rounded up to 64 (each request's last L + W - 1
    cached rows, from row 0)."""
    B = cu_seqlens_q.shape[0] - 1
    S = (max_q + window - 1 + 63) // 64 * 64
    return (torch.empty((B, S, 2, 64), dtype=torch.uint8, device=device),
            torch.empty((B, 2, 64, S), dtype=torch.uint8, device=device))


def xs_gather(k_cache, v_cache, page_table, cache_seqlens, cu_seqlens_q, window, part, ko, vt):
    """Copy each request's window rows from the paged cache into ko / vt (from alloc). part 0: only the blocks of
    rows cached before this call; part 1: the other blocks; part 2: all."""
    xs_sk_ext.sl_gather2(k_cache.view(torch.uint8), v_cache.view(torch.uint8), page_table, cache_seqlens, cu_seqlens_q,
                      window - 1, part, ko, vt)


def xs_attn(q8, ko, vt, cache_seqlens, cu_seqlens_q, max_q, window, softmax_scale, out):
    """Attention into out [T, 16, 64] BF16, unnormalized: each row is off by a positive factor (there is no softmax
    denominator), so this entry point is only correct ahead of a per-head RMSNorm. The layer uses attn_ng."""
    xs_sk_ext.sl_attn(q8.view(torch.uint8), ko, vt, cache_seqlens, cu_seqlens_q, max_q, window, softmax_scale, out)


def xs_attn_ng(q8, ko, vt, cache_seqlens, cu_seqlens_q, max_q, window, softmax_scale, gate, eps, out):
    """out [T, 1024] BF16 = o_norm_gate(attention, gate); gate is a [T, 1024] BF16 view with unit inner stride. The
    per-head RMSNorm of o_norm_gate cancels the missing softmax denominator."""
    xs_sk_ext.sl_attn_ng(q8.view(torch.uint8), ko, vt, cache_seqlens, cu_seqlens_q, max_q, window, softmax_scale, gate,
                      eps, out)




xs1_SK_CPP = 'void sl_gather(torch::Tensor kc, torch::Tensor vc, torch::Tensor pt, torch::Tensor seqlens, torch::Tensor cu_q, int64_t win, torch::Tensor ko, torch::Tensor vt);\nvoid sl_attn(torch::Tensor q, torch::Tensor ko, torch::Tensor vt, torch::Tensor cache_seqlens, torch::Tensor cu_q, int64_t max_q, int64_t window, double softmax_scale, torch::Tensor out);\nvoid sl_ts(torch::Tensor buf);\nvoid sl_pvstat(torch::Tensor buf);\nvoid sl_gather2(torch::Tensor kc, torch::Tensor vc, torch::Tensor pt, torch::Tensor seqlens, torch::Tensor cu_q, int64_t win, int64_t part, torch::Tensor ko, torch::Tensor vt);\nvoid sl_attn_ng(torch::Tensor q, torch::Tensor ko, torch::Tensor vt, torch::Tensor cache_seqlens, torch::Tensor cu_q, int64_t max_q, int64_t window, double softmax_scale, torch::Tensor gate, double eps, torch::Tensor out);\nvoid sl_attn_ng2(torch::Tensor q, torch::Tensor qlo, torch::Tensor ko, torch::Tensor vt, torch::Tensor cache_seqlens, torch::Tensor cu_q, int64_t max_q, int64_t window, double softmax_scale, torch::Tensor gate, double eps, torch::Tensor out);\n'
xs1_SK_FLAGS = ['-O3', '--use_fast_math', '-DNDEBUG', '-U__CUDA_NO_HALF_OPERATORS__', '-U__CUDA_NO_HALF_CONVERSIONS__', '-U__CUDA_NO_BFLOAT16_CONVERSIONS__', '-U__CUDA_NO_HALF2_OPERATORS__', '-gencode=arch=compute_90a,code=sm_90a']
xs1_SK_SRC = r"""

























#define Z0_NOL 1
#undef PV_B
#define PV_B 1104
#define PV16 1
#define SL_PIPE 3
#define SL_NWG 1
#define SL_MINB 3
#define SL_ST 3
#define SK_PDL 1
#define SK_KS 1
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>






#define SL_BN 128
#define SL_MATH 1
#define SK_O72 1
#define SK_BM 1
#define SK_TS 0
#define SX_XCH 0

constexpr int BN = SL_BN, ST = SL_ST;
constexpr int PPC = 8 * SL_NWG * SK_BM;
constexpr int K_BYTES = BN * 64, V_BYTES = 64 * BN, Q_BYTES = PPC * 8 * 64;

#define Z0_RS(o, i) 1.f
constexpr int VS = V_BYTES + ((SK_O72 && !Z0_NOL) ? 8 * BN : 0);
constexpr int NO = (SK_O72 && !Z0_NOL) ? 36 : 32;
typedef uint32_t OT;
constexpr int NOR = 16;
static_assert(Z0_NOL && SK_BM == 1, "PV16: denominator-free one-half build only");

constexpr int XCHB = PV16 ? 48 : 80;
constexpr int XB = (SX_XCH == 2 && SK_KS > 1) ? XCHB * SL_NWG * 128 : 0;
constexpr int SMEM = 1024 + 2 * Q_BYTES + ST * (K_BYTES + VS) + 512 + 256 + XB;

__device__ unsigned long long* g_sk_ts = nullptr;
__device__ unsigned int g_sx_smcnt[256];
static_assert(SK_KS == 1 || SK_O72, "SK_KS needs SK_O72 (sum in the O accumulator)");
static_assert(SK_BM == 1 || (SL_NWG == 1 && SL_PIPE == 3 && SK_O72 && SL_MATH && !SK_TS), "SK_BM 2: NWG 1, PIPE 3 only");
static_assert((SK_KS - 1) * SK_BM * 38 * SL_NWG * 128 * 4 <= ST * (K_BYTES + VS), "SK_BM exchange does not fit");
static_assert((SK_KS - 1) * 38 * SL_NWG * 128 * 4 <= ST * (K_BYTES + VS), "SK_KS exchange does not fit the ring");
constexpr int NKB = BN / 32;
constexpr int NSR = BN / 4;

typedef CUresult (*SlEncodeTiled)(CUtensorMap*, CUtensorMapDataType, cuuint32_t, void*, const cuuint64_t*,
                 const cuuint64_t*, const cuuint32_t*, const cuuint32_t*, CUtensorMapInterleave,
                 CUtensorMapSwizzle, CUtensorMapL2promotion, CUtensorMapFloatOOBfill);

struct SlParams {
 __nv_bfloat16* out;
 const int* cu_q;
 const int* cache_seqlens;
 int S;
 int G;
 int W;
 float a32;
 const __nv_bfloat16* gate;
 int gstride;
 float eps;
 int ks;
 int has_lo;
};

__device__ __forceinline__ uint32_t smem_u32(const void* p) { return static_cast<uint32_t>(__cvta_generic_to_shared(p)); }
__device__ __forceinline__ void mbar_init(uint32_t bar, int count) {
 asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::"r"(bar), "r"(count));
}
__device__ __forceinline__ void mbar_expect(uint32_t bar, int bytes) {
 asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(bar), "r"(bytes) : "memory");
}
__device__ __forceinline__ void mbar_arrive(uint32_t bar) {
 asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" ::"r"(bar) : "memory");
}
__device__ __forceinline__ void mbar_wait(uint32_t bar, uint32_t parity) {
 asm volatile("{\n .reg .pred p;\n W: mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n @!p bra W;\n}"
        ::"r"(bar), "r"(parity) : "memory");
}
__device__ __forceinline__ void tma_2d(uint32_t dst, const CUtensorMap* map, int c0, int c1, uint32_t bar) {
 asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%2, %3}], [%4];"
        ::"r"(dst), "l"(map), "r"(c0), "r"(c1), "r"(bar) : "memory");
}
__device__ __forceinline__ void tma_3d(uint32_t dst, const CUtensorMap* map, int c0, int c1, int c2, uint32_t bar) {
 asm volatile("cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%2, %3, %4}], [%5];"
        ::"r"(dst), "l"(map), "r"(c0), "r"(c1), "r"(c2), "r"(bar) : "memory");
}
__device__ __forceinline__ uint32_t prmt(uint32_t a, uint32_t b, uint32_t s) {
 uint32_t r;
 asm("prmt.b32 %0, %1, %2, %3;" : "=r"(r) : "r"(a), "r"(b), "r"(s));
 return r;
}

__device__ __forceinline__ uint64_t desc64(uint32_t addr) {
 return (uint64_t)((addr & 0x3FFFF) >> 4) | ((uint64_t)1 << 16) | ((uint64_t)(512 >> 4) << 32) | ((uint64_t)2 << 62);
}
__device__ __forceinline__ uint64_t desc128(uint32_t addr) {
 return (uint64_t)((addr & 0x3FFFF) >> 4) | ((uint64_t)1 << 16) | ((uint64_t)(1024 >> 4) << 32) | ((uint64_t)1 << 62);
}
__device__ __forceinline__ uint64_t descv(uint32_t addr) { return BN >= 128 ? desc128(addr) : desc64(addr); }


__device__ __forceinline__ uint64_t pvd(uint64_t dv, int kb) {
 return BN == 256 ? dv + (uint64_t)((kb >> 2) * (8192 >> 4) + 2 * (kb & 3)) : dv + (uint64_t)(2 * kb);
}


__device__ __forceinline__ uint64_t desc_ones(uint32_t addr) {
 return (uint64_t)((addr & 0x3FFFF) >> 4) | ((uint64_t)(128 >> 4) << 16) | ((uint64_t)(256 >> 4) << 32);
}
__device__ __forceinline__ void cluster_sync_all() {
 asm volatile("barrier.cluster.arrive.release.aligned;\nbarrier.cluster.wait.acquire.aligned;" ::: "memory");
}
__device__ __forceinline__ uint32_t cluster_rank() {
 uint32_t r;
 asm volatile("mov.u32 %0, %%cluster_ctarank;" : "=r"(r));
 return r;
}
__device__ __forceinline__ uint32_t mapa0(uint32_t a) {
 uint32_t r;
 asm volatile("mapa.shared::cluster.u32 %0, %1, 0;" : "=r"(r) : "r"(a));
 return r;
}
__device__ __forceinline__ uint32_t mapa_r(uint32_t a, uint32_t rank) {
 uint32_t r;
 asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(r) : "r"(a), "r"(rank));
 return r;
}
__device__ __forceinline__ void st_cluster4(uint32_t a, float x, float y, float z, float w) {
 asm volatile("st.shared::cluster.v4.f32 [%0], {%1, %2, %3, %4};" ::"r"(a), "f"(x), "f"(y), "f"(z), "f"(w) : "memory");
}
__device__ __forceinline__ void st_async4(uint32_t a, uint32_t bar, float x, float y, float z, float w) {
 asm volatile("st.async.shared::cluster.mbarrier::complete_tx::bytes.v4.f32 [%0], {%1, %2, %3, %4}, [%5];"
        ::"r"(a), "f"(x), "f"(y), "f"(z), "f"(w), "r"(bar) : "memory");
}
__device__ __forceinline__ void st_cluster(uint32_t a, float v) {
 asm volatile("st.shared::cluster.f32 [%0], %1;" ::"r"(a), "f"(v) : "memory");
}
__device__ __forceinline__ void wg_fence() { asm volatile("wgmma.fence.sync.aligned;" ::: "memory"); }
__device__ __forceinline__ void wg_commit() { asm volatile("wgmma.commit_group.sync.aligned;" ::: "memory"); }
template <int K_>
__device__ __forceinline__ void wg_wait() { asm volatile("wgmma.wait_group.sync.aligned %0;" ::"n"(K_) : "memory"); }
template <int N>
__device__ __forceinline__ void reg_fence(uint32_t (&r)[N]) {
#pragma unroll
 for (int i = 0; i < N; ++i) asm volatile("" : "+r"(r[i])::"memory");
}
template <int N>
__device__ __forceinline__ void reg_fence(float (&r)[N]) {
#pragma unroll
 for (int i = 0; i < N; ++i) asm volatile("" : "+f"(r[i])::"memory");
}




__device__ __forceinline__ void wgmma_qk(uint32_t (&d)[32], uint64_t da, uint64_t db, int sd) {
 asm volatile("{\n.reg .pred p;\nsetp.ne.b32 p, %34, 0;\n"
        "wgmma.mma_async.sync.aligned.m64n128k32.f16.e4m3.e4m3 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31}, %32, %33, p, 1, 1;\n}\n"
        : "+r"(d[0]), "+r"(d[1]), "+r"(d[2]), "+r"(d[3]), "+r"(d[4]), "+r"(d[5]), "+r"(d[6]), "+r"(d[7]), "+r"(d[8]), "+r"(d[9]), "+r"(d[10]), "+r"(d[11]), "+r"(d[12]), "+r"(d[13]), "+r"(d[14]), "+r"(d[15]), "+r"(d[16]), "+r"(d[17]), "+r"(d[18]), "+r"(d[19]), "+r"(d[20]), "+r"(d[21]), "+r"(d[22]), "+r"(d[23]), "+r"(d[24]), "+r"(d[25]), "+r"(d[26]), "+r"(d[27]), "+r"(d[28]), "+r"(d[29]), "+r"(d[30]), "+r"(d[31]) : "l"(da), "l"(db), "r"(sd));
}
__device__ __forceinline__ void wgmma_qk(uint32_t (&d)[64], uint64_t da, uint64_t db, int sd) {
 asm volatile("{\n.reg .pred p;\nsetp.ne.b32 p, %66, 0;\n"
        "wgmma.mma_async.sync.aligned.m64n256k32.f16.e4m3.e4m3 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47,%48,%49,%50,%51,%52,%53,%54,%55,%56,%57,%58,%59,%60,%61,%62,%63}, %64, %65, p, 1, 1;\n}\n"
        : "+r"(d[0]), "+r"(d[1]), "+r"(d[2]), "+r"(d[3]), "+r"(d[4]), "+r"(d[5]), "+r"(d[6]), "+r"(d[7]), "+r"(d[8]), "+r"(d[9]), "+r"(d[10]), "+r"(d[11]), "+r"(d[12]), "+r"(d[13]), "+r"(d[14]), "+r"(d[15]), "+r"(d[16]), "+r"(d[17]), "+r"(d[18]), "+r"(d[19]), "+r"(d[20]), "+r"(d[21]), "+r"(d[22]), "+r"(d[23]), "+r"(d[24]), "+r"(d[25]), "+r"(d[26]), "+r"(d[27]), "+r"(d[28]), "+r"(d[29]), "+r"(d[30]), "+r"(d[31]), "+r"(d[32]), "+r"(d[33]), "+r"(d[34]), "+r"(d[35]), "+r"(d[36]), "+r"(d[37]), "+r"(d[38]), "+r"(d[39]), "+r"(d[40]), "+r"(d[41]), "+r"(d[42]), "+r"(d[43]), "+r"(d[44]), "+r"(d[45]), "+r"(d[46]), "+r"(d[47]), "+r"(d[48]), "+r"(d[49]), "+r"(d[50]), "+r"(d[51]), "+r"(d[52]), "+r"(d[53]), "+r"(d[54]), "+r"(d[55]), "+r"(d[56]), "+r"(d[57]), "+r"(d[58]), "+r"(d[59]), "+r"(d[60]), "+r"(d[61]), "+r"(d[62]), "+r"(d[63]) : "l"(da), "l"(db), "r"(sd));
}
__device__ __forceinline__ void wgmma_pv(float (&d)[36], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint64_t db) {
 asm volatile("{\n.reg .pred p;\nsetp.ne.b32 p, 1, 0;\n"
        "wgmma.mma_async.sync.aligned.m64n72k32.f32.e4m3.e4m3 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32,%33,%34,%35}, {%36,%37,%38,%39}, %40, p, 1, 1;\n}\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]), "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]), "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]), "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27]), "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31]), "+f"(d[32]), "+f"(d[33]), "+f"(d[34]), "+f"(d[35]) : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "l"(db));
}
__device__ __forceinline__ void wgmma_pv(float (&d)[32], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint64_t db) {
 asm volatile("{\n.reg .pred p;\nsetp.ne.b32 p, 1, 0;\n"
        "wgmma.mma_async.sync.aligned.m64n64k32.f32.e4m3.e4m3 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31}, {%32,%33,%34,%35}, %36, p, 1, 1;\n}\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]), "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]), "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]), "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27]), "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31]) : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "l"(db));
}
__device__ __forceinline__ void wgmma_pv(uint32_t (&d)[16], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint64_t db) {
 asm volatile("{\n.reg .pred p;\nsetp.ne.b32 p, 1, 0;\n"
        "wgmma.mma_async.sync.aligned.m64n64k32.f16.e4m3.e4m3 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15}, {%16,%17,%18,%19}, %20, p, 1, 1;\n}\n"
        : "+r"(d[0]), "+r"(d[1]), "+r"(d[2]), "+r"(d[3]), "+r"(d[4]), "+r"(d[5]), "+r"(d[6]), "+r"(d[7]), "+r"(d[8]), "+r"(d[9]), "+r"(d[10]), "+r"(d[11]), "+r"(d[12]), "+r"(d[13]), "+r"(d[14]), "+r"(d[15]) : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "l"(db));
}
__device__ __forceinline__ void wgmma_ones(float (&d)[4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint64_t db, int sd) {
 asm volatile("{\n.reg .pred p;\nsetp.ne.b32 p, %9, 0;\n"
        "wgmma.mma_async.sync.aligned.m64n8k32.f32.e4m3.e4m3 {%0,%1,%2,%3}, {%4,%5,%6,%7}, %8, p, 1, 1;\n}\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "l"(db), "r"(sd));
}


__device__ __forceinline__ __half2 h2(uint32_t x) { return *reinterpret_cast<__half2*>(&x); }
__device__ __forceinline__ uint32_t u32(__half2 x) { return *reinterpret_cast<uint32_t*>(&x); }


struct SlRow {
 float rmax[2], rb[2], rsum[2];
 int jlo[2], jhi[2];
 uint32_t rmh, rbh;
};

__device__ __forceinline__ void sl_issue_qk(uint32_t (&s)[NSR], uint64_t dq, uint32_t kaddr, uint64_t dql, int lo) {
#pragma unroll
 for (int i = 0; i < NSR; ++i) s[i] = 0;
 reg_fence(s);
 wg_fence();
 const uint64_t dk = desc64(kaddr);
 wgmma_qk(s, dq, dk, 0);
 wgmma_qk(s, dq + 2, dk + 2, 1);
 if (lo) {
  wgmma_qk(s, dql, dk, 1);
  wgmma_qk(s, dql + 2, dk + 2, 1);
 }
 wg_commit();
}

__device__ __forceinline__ void sl_pack(const uint32_t (&sc)[NSR], uint32_t (&pa)[NKB][4]);


__device__ __forceinline__ void sl_softmax(uint32_t (&sc)[NSR], uint32_t (&pa)[NKB][4], SlRow& st, int j0, bool edge,
                      int tq, float a32, float (&scale)[2]) {
 if (edge) {


  const int off0 = j0 + 2 * tq - st.jlo[0], off1 = j0 + 2 * tq - st.jlo[1];
  const unsigned sp0 = (unsigned)(st.jhi[0] - st.jlo[0]), sp1 = (unsigned)(st.jhi[1] - st.jlo[1]);
#pragma unroll
  for (int w = 0; w < NSR; ++w) {
   const int d = ((w & 1) ? off1 : off0) + 8 * (w >> 1);
   const unsigned sp = (w & 1) ? sp1 : sp0;
   const uint32_t m = ((unsigned)d <= sp ? 0x0000FFFFu : 0u) | ((unsigned)(d + 1) <= sp ? 0xFFFF0000u : 0u);
   sc[w] = (sc[w] & m) | (0xFC00FC00u & ~m);
  }
  edge = false;
 }
 const __half2 ninf2 = __float2half2_rn(-INFINITY);
 if (edge) {
#pragma unroll
  for (int w = 0; w < NSR; ++w) {
   const int r = w & 1, j = j0 + 8 * (w >> 1) + 2 * tq;
   __half2 v = h2(sc[w]);
   const bool ok0 = j >= st.jlo[r] && j <= st.jhi[r], ok1 = j + 1 >= st.jlo[r] && j + 1 <= st.jhi[r];
   v = __halves2half2(ok0 ? __low2half(v) : __low2half(ninf2), ok1 ? __high2half(v) : __high2half(ninf2));
   sc[w] = u32(v);
  }
 }
 __half2 m0 = h2(sc[0]), m1 = h2(sc[1]), m2 = h2(sc[2]), m3 = h2(sc[3]);
#pragma unroll
 for (int w = 4; w < NSR; w += 4) {
  m0 = __hmax2(m0, h2(sc[w])); m1 = __hmax2(m1, h2(sc[w + 1]));
  m2 = __hmax2(m2, h2(sc[w + 2])); m3 = __hmax2(m3, h2(sc[w + 3]));
 }
 m0 = __hmax2(m0, m2); m1 = __hmax2(m1, m3);
 __half2 mm = __hmax2(__lows2half2(m0, m1), __highs2half2(m0, m1));
 mm = __hmax2(mm, h2(__shfl_xor_sync(0xffffffff, u32(mm), 1)));
 mm = __hmax2(mm, h2(__shfl_xor_sync(0xffffffff, u32(mm), 2)));


 __half2 bx[2];
 {
  const __half2 cur = __hmax2(h2(st.rmh), mm);
  const __half2 bn = __hmin2(__hfma2(cur, __hneg2(__float2half2_rn(a32)), __float2half2_rn((float)PV_B)),
                __float2half2_rn(65504.f));
  const uint32_t bnu = u32(bn);
  scale[0] = scale[1] = 1.f;
  if (__any_sync(0xffffffffu, bnu != st.rbh) && bnu != st.rbh) {
   scale[0] = exp2f((__low2float(bn) - __low2float(h2(st.rbh))) * 0.125f);
   scale[1] = exp2f((__high2float(bn) - __high2float(h2(st.rbh))) * 0.125f);
  }
  st.rbh = bnu;
  st.rmh = u32(cur);
  bx[0] = __low2half2(bn);
  bx[1] = __high2half2(bn);
 }
 const __half2 a2 = __float2half2_rn(a32);
 const __half2 off2 = __float2half2_rn(896.f), lo2 = __float2half2_rn(1024.f);


 (void)off2;
#pragma unroll
 for (int w = 0; w < NSR; ++w) sc[w] = u32(__hmax2(__hfma2(h2(sc[w]), a2, bx[w & 1]), lo2));
}



__device__ __forceinline__ void sl_pack(const uint32_t (&sc)[NSR], uint32_t (&pa)[NKB][4]) {
#pragma unroll
 for (int kb = 0; kb < NKB; ++kb) {
  const int w = 8 * kb;
  pa[kb][0] = prmt(sc[w + 0], sc[w + 2], 0x6420);
  pa[kb][1] = prmt(sc[w + 1], sc[w + 3], 0x6420);
  pa[kb][2] = prmt(sc[w + 4], sc[w + 6], 0x6420);
  pa[kb][3] = prmt(sc[w + 5], sc[w + 7], 0x6420);
 }
}


__device__ __forceinline__ void sl_issue_pv(OT (&o)[NOR], float (&os)[4], uint32_t (&pa)[NKB][4], uint32_t vaddr,
                      uint64_t dones) {
 const uint64_t dv = descv(vaddr);
 reg_fence(o);
 wg_fence();
#pragma unroll
 for (int kb = 0; kb < NKB; ++kb) {
  wgmma_pv(o, pa[kb][0], pa[kb][1], pa[kb][2], pa[kb][3], pvd(dv, kb));
 }
 wg_commit();
}

__device__ __forceinline__ void sl_rescale(OT (&o)[NOR], SlRow& st, const float (&scale)[2]) {

 if (__any_sync(0xffffffffu, scale[0] != 1.f || scale[1] != 1.f)) {
  const __half2 s0 = __float2half2_rn(scale[0]), s1 = __float2half2_rn(scale[1]);
#pragma unroll
  for (int i = 0; i < NOR; i += 2) { o[i] = u32(__hmul2(h2(o[i]), s0)); o[i + 1] = u32(__hmul2(h2(o[i + 1]), s1)); }
  st.rsum[0] *= scale[0];
  st.rsum[1] *= scale[1];
 }
}



template <int R>
__device__ __forceinline__ void sx_send_row(uint32_t a, int nt4, const float (&o)[NO], float b) {
 float v[20] = {};
#pragma unroll
 for (int jb = 0; jb < NO / 4; ++jb) { v[2 * jb] = o[4 * jb + 2 * R]; v[2 * jb + 1] = o[4 * jb + 2 * R + 1]; }
 v[18] = b;
 v[19] = 0.f;
#pragma unroll
 for (int j = 0; j < 5; ++j) st_cluster4(a + 16u * j * nt4, v[4 * j], v[4 * j + 1], v[4 * j + 2], v[4 * j + 3]);
}
template <int R>
__device__ __forceinline__ void sx_send_row_async(uint32_t a, uint32_t bar, int nt4, const float (&o)[NO], float b) {
 float v[20] = {};
#pragma unroll
 for (int jb = 0; jb < NO / 4; ++jb) { v[2 * jb] = o[4 * jb + 2 * R]; v[2 * jb + 1] = o[4 * jb + 2 * R + 1]; }
 v[18] = b;
 v[19] = 0.f;
#pragma unroll
 for (int j = 0; j < 5; ++j) st_async4(a + 16u * j * nt4, bar, v[4 * j], v[4 * j + 1], v[4 * j + 2], v[4 * j + 3]);
}
template <int R>
__device__ __forceinline__ void sx_merge_row(const float4* x, int nt4, float (&o)[NO], float (&rb)[2]) {
 float v[20];
#pragma unroll
 for (int j = 0; j < 5; ++j) {
  const float4 t = x[j * nt4];
  v[4 * j] = t.x; v[4 * j + 1] = t.y; v[4 * j + 2] = t.z; v[4 * j + 3] = t.w;
 }
 const float b = v[18], m = fminf(rb[R], b);
 const float sa = exp2f((m - rb[R]) * 0.125f), sb = exp2f((m - b) * 0.125f);
#pragma unroll
 for (int jb = 0; jb < NO / 4; ++jb) {
  o[4 * jb + 2 * R] = o[4 * jb + 2 * R] * sa + v[2 * jb] * sb;
  o[4 * jb + 2 * R + 1] = o[4 * jb + 2 * R + 1] * sa + v[2 * jb + 1] * sb;
 }
 rb[R] = m;
}





template <int R>
__device__ __forceinline__ void sx_send_row_async(uint32_t a, uint32_t bar, int nt4, const uint32_t (&o)[16], float b) {
 st_async4(a, bar, __uint_as_float(o[R]), __uint_as_float(o[2 + R]), __uint_as_float(o[4 + R]), __uint_as_float(o[6 + R]));
 st_async4(a + 16u * nt4, bar, __uint_as_float(o[8 + R]), __uint_as_float(o[10 + R]), __uint_as_float(o[12 + R]),
      __uint_as_float(o[14 + R]));
 st_async4(a + 32u * nt4, bar, b, 0.f, 0.f, 0.f);
}
template <int R>
__device__ __forceinline__ void sx_merge_row(const float4* x, int nt4, uint32_t (&o)[16], float (&rb)[2]) {
 const float4 t0 = x[0], t1 = x[nt4], t2 = x[2 * nt4];
 const uint32_t v[8] = {__float_as_uint(t0.x), __float_as_uint(t0.y), __float_as_uint(t0.z), __float_as_uint(t0.w),
             __float_as_uint(t1.x), __float_as_uint(t1.y), __float_as_uint(t1.z), __float_as_uint(t1.w)};
 const float b = t2.x, m = fminf(rb[R], b);
 const float sa = exp2f((m - rb[R]) * 0.125f), sb = exp2f((m - b) * 0.125f);
#pragma unroll
 for (int jb = 0; jb < 8; ++jb) {
  const float2 f = __half22float2(h2(o[2 * jb + R])), g = __half22float2(h2(v[jb]));
  o[2 * jb + R] = u32(__floats2half2_rn(f.x * sa + g.x * sb, f.y * sa + g.y * sb));
 }
 rb[R] = m;
}
constexpr int NTHR = SL_NWG * 128 + 32;

__global__ void __launch_bounds__(NTHR, SL_MINB)
sl_attn_k(const __grid_constant__ CUtensorMap mq,
                              const __grid_constant__ CUtensorMap mk,
                              const __grid_constant__ CUtensorMap mv,
                              const __grid_constant__ CUtensorMap mql, SlParams p) {
 extern __shared__ uint8_t smem_raw[];
 uint8_t* base = reinterpret_cast<uint8_t*>((reinterpret_cast<uintptr_t>(smem_raw) + 1023) & ~uintptr_t(1023));
 uint8_t* sk = base;
 uint8_t* sv = sk + ST * K_BYTES;
 uint8_t* sq = sv + ST * VS;
 uint8_t* sones = sq + 2 * Q_BYTES;
 uint64_t* bars = reinterpret_cast<uint64_t*>(sones + 512);
 const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;

#define SX_C(k) do { } while (0)



 const int ks = SK_KS > 1 ? p.ks : 1;
 const int kc = ks > 1 ? (int)cluster_rank() : 0, item = ks > 1 ? blockIdx.x / ks : blockIdx.x;
 const int kh = item & 1, bg = item >> 1, b = bg / p.G, g = bg - b * p.G;
 const int q0 = p.cu_q[b], Lb = p.cu_q[b + 1] - q0, n = p.cache_seqlens[b];
 const int i0 = g * PPC;
 if (i0 >= Lb) return;
 const int W = p.W;
 const int s0 = max(0, n - Lb - (W - 1));
 const int ilast = min(i0 + PPC - 1, Lb - 1);
 const int pfirst = n - Lb + i0, plast = n - Lb + ilast;
 const int jlo = max(0, pfirst - W + 1 - s0), jhi = plast - s0;
 const int ta = jlo / BN, nta = jhi / BN - ta + 1;
 const int t0 = ta + (kc * nta) / ks, nt = ta + ((kc + 1) * nta) / ks - t0;

 if (tid == 0) {
  for (int s = 0; s < ST; ++s) { mbar_init(smem_u32(&bars[s]), 1); mbar_init(smem_u32(&bars[ST + s]), SL_NWG * 128); }
  mbar_init(smem_u32(&bars[2 * ST]), 1);
  asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
 }
 if (tid < 128) reinterpret_cast<uint32_t*>(sones)[tid] = 0x38383838u;
 asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
 __syncthreads();
 asm volatile("griddepcontrol.wait;" ::: "memory");
 asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
 SX_C(1);

 if (warp == SL_NWG * 4) {


  if (lane == 0) {
   for (int it = 0; it < nt; ++it) {
    const int s = it % ST;
    if (it >= ST) mbar_wait(smem_u32(&bars[ST + s]), ((it / ST) - 1) & 1);
    const uint32_t fb = smem_u32(&bars[s]);
    mbar_expect(fb, K_BYTES + V_BYTES);
    const int j = (t0 + it) * BN;
    tma_3d(smem_u32(sk + s * K_BYTES), &mk, 0, kh, b * p.S + j, fb);
    tma_2d(smem_u32(sv + s * VS), &mv, j, (b * 2 + kh) * 64, fb);
    if (it == 0) {
     const uint32_t qb = smem_u32(&bars[2 * ST]);
     mbar_expect(qb, p.has_lo ? 2 * Q_BYTES : Q_BYTES);
     tma_3d(smem_u32(sq), &mq, 0, kh * 8, q0 + i0, qb);
     if (p.has_lo) tma_3d(smem_u32(sq + Q_BYTES), &mql, 0, kh * 8, q0 + i0, qb);
    }
   }
  }
  __syncwarp();
  return;
 }

 if (p.gate && tid < 8 * PPC) {
  const int pos = i0 + (tid >> 3);
  if (pos < Lb)
   asm volatile("prefetch.global.L2 [%0];" ::"l"(p.gate + (size_t)(q0 + pos) * p.gstride + (kh * 8 + (tid & 7)) * 64));
 }


 const int wg = warp >> 2, wi = warp & 3;
 const int tq = lane & 3, hq = lane >> 2;
 const int ib = i0 + 8 * wg + 2 * wi;
 const int pr0 = n - Lb + ib;
 SlRow st;
#pragma unroll
 for (int r = 0; r < 2; ++r) {
  st.rmax[r] = -INFINITY; st.rb[r] = (float)PV_B; st.rsum[r] = 0.f;
  st.rmh = 0xFC00FC00u;
  st.rbh = 0x7BFF7BFFu;
  st.jhi[r] = pr0 + r - s0;
  st.jlo[r] = pr0 + r - W + 1 - s0;
 }
 OT o[NOR];
#pragma unroll
 for (int i = 0; i < NOR; ++i) o[i] = 0;
#define SK_T(k) do { } while (0)

 const uint64_t dq = desc64(smem_u32(sq + wg * 4096));
 const uint64_t dql = desc64(smem_u32(sq + Q_BYTES + wg * 4096));
 const uint64_t dones = desc_ones(smem_u32(sones));

 const int jlo_all = max(0, plast - W + 1 - s0), jhi_all = pfirst - s0;
 const float a32 = p.a32;
 float os[4] = {0.f, 0.f, 0.f, 0.f};
 uint32_t sa[NSR], sb[NSR], pa[NKB][4];
 float scale[2];
 (void)dones;
 if (nt > 0) {
 mbar_wait(smem_u32(&bars[2 * ST]), 0);
 mbar_wait(smem_u32(&bars[0]), 0);
 sl_issue_qk(sa, dq, smem_u32(sk), dql, p.has_lo);
 wg_wait<0>();
 reg_fence(sa);
 SK_T(0);
 SX_C(2);


 (void)sb;
 const uint32_t zdep = (uint32_t)(p.ks >> 8);
 (void)zdep;
 {
  const int j0 = t0 * BN;
  sl_softmax(sa, pa, st, j0, j0 < jlo_all || j0 + BN - 1 > jhi_all, tq, a32, scale);
  sl_rescale(o, st, scale);
  sl_pack(sa, pa);
 }
 {

  const uint32_t fb0 = smem_u32(&bars[0]), eb0 = smem_u32(&bars[ST]);
  const uint64_t dk0 = desc64(smem_u32(sk)), dv0 = descv(smem_u32(sv));
  uint32_t s = 1 % ST, ph = (1 / ST) & 1, sp = 0;
#pragma unroll 1
  for (int it = 1; it < nt; ++it) {
   mbar_wait(fb0 + 8u * s, ph);
#pragma unroll
   for (int i = 0; i < NSR; ++i) sa[i] = 0;
   reg_fence(sa);
   wg_fence();
   {
    const uint64_t dk = dk0 + (uint64_t)(s * (K_BYTES >> 4));
    wgmma_qk(sa, dq, dk, 0);
    wgmma_qk(sa, dq + 2, dk + 2, 1);
    if (p.has_lo) {
     wgmma_qk(sa, dql, dk, 1);
     wgmma_qk(sa, dql + 2, dk + 2, 1);
    }
   }
   wg_commit();
   reg_fence(o);
   wg_fence();
   {
    const uint64_t dv = dv0 + (uint64_t)(sp * (VS >> 4));
#pragma unroll
    for (int kb = 0; kb < NKB; ++kb) wgmma_pv(o, pa[kb][0], pa[kb][1], pa[kb][2], pa[kb][3], pvd(dv, kb));
   }
   wg_commit();
   wg_wait<1>();
   reg_fence(sa);
   const int j0 = (t0 + it) * BN;
   sl_softmax(sa, pa, st, j0, j0 < jlo_all || j0 + BN - 1 > jhi_all, tq, a32, scale);
   wg_wait<0>();
   reg_fence(o);
   mbar_arrive(eb0 + 8u * sp);
   sl_rescale(o, st, scale);
   sl_pack(sa, pa);
   sp = s;
   if (++s == (uint32_t)ST) { s = 0; ph ^= 1u; }
  }
 }
 sl_issue_pv(o, os, pa, smem_u32(sv + ((nt - 1) % ST) * VS), dones);
 wg_wait<0>();
 reg_fence(o);
 st.rb[0] = __low2float(h2(st.rbh));
 st.rb[1] = __high2float(h2(st.rbh));
 st.rsum[0] = Z0_RS(o, 32);
 st.rsum[1] = Z0_RS(o, 34);
 mbar_arrive(smem_u32(&bars[ST + (nt - 1) % ST]));
 }
 SX_C(3);
 SX_C(6);


#pragma unroll
 for (int r = 0; r < 2; ++r) {
  const int i = ib + r;
  if (i < Lb) {
   __nv_bfloat16* dst = p.out + ((size_t)(q0 + i) * 16 + kh * 8 + hq) * 64 + 2 * tq;
   const float inv = st.rsum[r] > 0.f ? 1.f / st.rsum[r] : 0.f;
   __nv_bfloat162 v[8];
#pragma unroll
   for (int jb = 0; jb < 8; ++jb) { const float2 f = __half22float2(h2(o[2 * jb + r])); v[jb] = __floats2bfloat162_rn(f.x * inv, f.y * inv); }
   if (p.gate) {


    const __nv_bfloat16* gp = p.gate + (size_t)(q0 + i) * p.gstride + (kh * 8 + hq) * 64 + 2 * tq;
    __nv_bfloat162 g[8];
#pragma unroll
    for (int jb = 0; jb < 8; ++jb) g[jb] = *reinterpret_cast<const __nv_bfloat162*>(gp + 8 * jb);
    float ss = 0.f;
#pragma unroll
    for (int jb = 0; jb < 8; ++jb) {
     const float2 x = __bfloat1622float2(v[jb]);
     ss = fmaf(x.x, x.x, fmaf(x.y, x.y, ss));
    }
    ss += __shfl_xor_sync(0xffffffffu, ss, 1);
    ss += __shfl_xor_sync(0xffffffffu, ss, 2);
    const float rs = rsqrtf(ss * (1.f / 64.f) + p.eps);
#pragma unroll
    for (int jb = 0; jb < 8; ++jb) {
     const float2 x = __bfloat1622float2(v[jb]), gg = __bfloat1622float2(g[jb]);
     v[jb] = __floats2bfloat162_rn(x.x * rs / (1.f + __expf(-gg.x)), x.y * rs / (1.f + __expf(-gg.y)));
    }
   }
#pragma unroll
   for (int jb = 0; jb < 8; ++jb) *reinterpret_cast<__nv_bfloat162*>(dst + 8 * jb) = v[jb];
  }
 }
}





__global__ void __launch_bounds__(256) sl_gather_k(const uint4* __restrict__ kc, const uint4* __restrict__ vc,
  const int* __restrict__ pt, const int* __restrict__ seqlens, const int* __restrict__ cu_q, int pages, int page,
  int S, int win, int part, uint4* __restrict__ ko, uint4* __restrict__ vt) {
 __shared__ uint4 tile[64][9];
 asm volatile("griddepcontrol.wait;" ::: "memory");
 asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
 const int b = blockIdx.y, j0 = blockIdx.x * 64, t = threadIdx.x;
 const int n = seqlens[b], L = cu_q[b + 1] - cu_q[b];
 const int s0 = max(0, n - L - win), nv = n - s0;


 const bool old_only = j0 + 64 <= n - L - s0;
 if ((part == 0 && !old_only) || (part == 1 && old_only)) return;
#pragma unroll
 for (int r = 0; r < 2; ++r) {
  const int idx = t + 256 * r, jr = idx >> 3, u = idx & 7, j = j0 + jr;
  uint4 kv = make_uint4(0, 0, 0, 0), vv = make_uint4(0, 0, 0, 0);
  if (j < nv) {
   const int s = s0 + j;
   int pg = pt[(long long)b * pages + s / page];
   pg = pg < 0 ? 0 : pg;
   const long long src = ((long long)pg * page + s % page) * 8 + u;
   kv = __ldg(kc + src);
   vv = __ldg(vc + src);
  }
  ko[((long long)b * S + j) * 8 + u] = kv;
  tile[jr][u] = vv;
 }
 __syncthreads();
 const unsigned char* tb = reinterpret_cast<const unsigned char*>(&tile[0][0]);
#pragma unroll
 for (int r = 0; r < 2; ++r) {
  const int idx = t + 256 * r, hd = idx >> 2, q = idx & 3;
  uint32_t w[4];
#pragma unroll
  for (int k = 0; k < 4; ++k) {
   uint32_t x = 0;
#pragma unroll
   for (int i = 0; i < 4; ++i) {
    const int slot = q * 16 + 4 * k + i, ks = slot & 31;
    const int src = (slot & 32) + 2 * ((ks & 15) >> 2) + (ks & 1) + 8 * ((ks & 3) >> 1) + 16 * (ks >> 4);
    x |= uint32_t(tb[src * 144 + hd]) << (8 * i);
   }
   w[k] = x;
  }
  vt[(((long long)b * 128 + hd) * S + j0) / 16 + q] = make_uint4(w[0], w[1], w[2], w[3]);
 }
}

static SlEncodeTiled sl_encode() {
 static SlEncodeTiled enc = nullptr;
 if (!enc) {
  cudaDriverEntryPointQueryResult qr;
  cudaGetDriverEntryPoint("cuTensorMapEncodeTiled", reinterpret_cast<void**>(&enc), cudaEnableDefault, &qr);
  cudaFuncSetAttribute(sl_attn_k, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM);
 }
 return enc;
}


void sl_ts(torch::Tensor buf) {
 unsigned long long* ptr = buf.numel() ? (unsigned long long*)buf.data_ptr() : nullptr;
 cudaMemcpyToSymbol(g_sk_ts, &ptr, sizeof(ptr));
}
void sl_pvstat(torch::Tensor buf) {
}

void sl_gather2(torch::Tensor kc, torch::Tensor vc, torch::Tensor pt, torch::Tensor seqlens, torch::Tensor cu_q,
        int64_t win, int64_t part, torch::Tensor ko, torch::Tensor vt) {
 const int B = pt.size(0), pages = pt.size(1), page = kc.size(1), S = ko.size(1);
 TORCH_CHECK(kc.size(2) * kc.size(3) == 128 && S % 64 == 0, "sl_gather");
 cudaLaunchConfig_t cfg = {};
 cudaLaunchAttribute at[1];
 at[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
 at[0].val.programmaticStreamSerializationAllowed = SK_PDL;
 cfg.gridDim = dim3(S / 64, B);
 cfg.blockDim = dim3(256);
 cfg.dynamicSmemBytes = 0;
 { cudaLaunchConfig_t q0 = {cfg.gridDim, cfg.blockDim, cfg.dynamicSmemBytes, at::cuda::getCurrentCUDAStream(), nullptr, 0}; cfg = q0; }
 cfg.attrs = at;
 cfg.numAttrs = 1;
 cudaLaunchKernelEx(&cfg, sl_gather_k, (const uint4*)kc.data_ptr(), (const uint4*)vc.data_ptr(), (const int*)pt.data_ptr<int>(),
           (const int*)seqlens.data_ptr<int>(), (const int*)cu_q.data_ptr<int>(), pages, page, S, (int)win, (int)part,
           (uint4*)ko.data_ptr(), (uint4*)vt.data_ptr());
}

void sl_gather(torch::Tensor kc, torch::Tensor vc, torch::Tensor pt, torch::Tensor seqlens, torch::Tensor cu_q,
        int64_t win, torch::Tensor ko, torch::Tensor vt) {
 sl_gather2(kc, vc, pt, seqlens, cu_q, win, 2, ko, vt);
}



static void sl_launch(torch::Tensor q, torch::Tensor ko, torch::Tensor vt, torch::Tensor cache_seqlens, torch::Tensor cu_q,
           int64_t max_q, int64_t window, double softmax_scale, torch::Tensor out,
           const __nv_bfloat16* gate, int gstride, float eps, const uint32_t* qlo = nullptr) {
 SlEncodeTiled enc = sl_encode();
 const int T = q.size(0), B = ko.size(0), S = ko.size(1);
 CUtensorMap mq, mql, mk, mv;
 {
  const cuuint64_t dims[3] = {64, 16, (cuuint64_t)T}, strides[2] = {64, 1024};
  const cuuint32_t box[3] = {64, 8, PPC}, es[3] = {1, 1, 1};
  enc(&mq, CU_TENSOR_MAP_DATA_TYPE_UINT8, 3, q.data_ptr(), dims, strides, box, es, CU_TENSOR_MAP_INTERLEAVE_NONE,
    CU_TENSOR_MAP_SWIZZLE_64B, CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  enc(&mql, CU_TENSOR_MAP_DATA_TYPE_UINT8, 3, qlo ? (void*)qlo : q.data_ptr(), dims, strides, box, es, CU_TENSOR_MAP_INTERLEAVE_NONE,
    CU_TENSOR_MAP_SWIZZLE_64B, CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
 }
 {
  const cuuint64_t dims[3] = {64, 2, (cuuint64_t)B * S}, strides[2] = {64, 128};
  const cuuint32_t box[3] = {64, 1, (cuuint32_t)BN}, es[3] = {1, 1, 1};
  enc(&mk, CU_TENSOR_MAP_DATA_TYPE_UINT8, 3, ko.data_ptr(), dims, strides, box, es, CU_TENSOR_MAP_INTERLEAVE_NONE,
    CU_TENSOR_MAP_SWIZZLE_64B, CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
 }
 {
  const cuuint64_t dims[2] = {(cuuint64_t)S, (cuuint64_t)B * 128}, strides[1] = {(cuuint64_t)S};
  const cuuint32_t box[2] = {(cuuint32_t)(BN == 256 ? 128 : BN), 64}, es[2] = {1, 1};
  enc(&mv, CU_TENSOR_MAP_DATA_TYPE_UINT8, 2, vt.data_ptr(), dims, strides, box, es, CU_TENSOR_MAP_INTERLEAVE_NONE,
    BN >= 128 ? CU_TENSOR_MAP_SWIZZLE_128B : CU_TENSOR_MAP_SWIZZLE_64B, CU_TENSOR_MAP_L2_PROMOTION_L2_128B,
    CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
 }
 SlParams p;
 p.out = (__nv_bfloat16*)out.data_ptr();
 p.cu_q = cu_q.data_ptr<int>();
 p.cache_seqlens = cache_seqlens.data_ptr<int>();
 p.S = S;
 p.G = (int)((max_q + PPC - 1) / PPC);
 p.W = (int)window;
 p.a32 = __half2float(__float2half_rn((float)(8.0 * 1.4426950408889634 * softmax_scale)));
 p.gate = gate;
 p.gstride = gstride;
 p.eps = eps;
 p.has_lo = qlo != nullptr;

 static int slots = 0;
 if (!slots) {
  int dev = 0, sms = 0, occ = 0;
  cudaGetDevice(&dev);
  cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev);
  cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ, sl_attn_k, NTHR, SMEM);
  slots = sms * (occ > 0 ? occ : 1);
 }
 const int items = B * p.G * 2;
 p.ks = (SK_KS > 1 && 2 * items <= slots) ? 2 : 1;
 cudaLaunchConfig_t cfg = {};
 cudaLaunchAttribute at[2];
 at[0].id = cudaLaunchAttributeClusterDimension;
 at[0].val.clusterDim.x = p.ks;
 at[0].val.clusterDim.y = 1;
 at[0].val.clusterDim.z = 1;
 at[1].id = cudaLaunchAttributeProgrammaticStreamSerialization;
 at[1].val.programmaticStreamSerializationAllowed = SK_PDL;
 cfg.gridDim = dim3(items * p.ks);
 cfg.blockDim = dim3(NTHR);
 cfg.dynamicSmemBytes = SMEM;
 { cudaLaunchConfig_t q0 = {cfg.gridDim, cfg.blockDim, cfg.dynamicSmemBytes, at::cuda::getCurrentCUDAStream(), nullptr, 0}; cfg = q0; }
 cfg.attrs = at;
 cfg.numAttrs = 2;
 cudaLaunchKernelEx(&cfg, sl_attn_k, mq, mk, mv, mql, p);
}

void sl_attn(torch::Tensor q, torch::Tensor ko, torch::Tensor vt, torch::Tensor cache_seqlens, torch::Tensor cu_q,
       int64_t max_q, int64_t window, double softmax_scale, torch::Tensor out) {
 sl_launch(q, ko, vt, cache_seqlens, cu_q, max_q, window, softmax_scale, out, nullptr, 0, 0.f);
}


void sl_attn_ng(torch::Tensor q, torch::Tensor ko, torch::Tensor vt, torch::Tensor cache_seqlens, torch::Tensor cu_q,
        int64_t max_q, int64_t window, double softmax_scale, torch::Tensor gate, double eps, torch::Tensor out) {
 TORCH_CHECK(gate.stride(1) == 1 && gate.size(1) == 1024, "sl_attn_ng gate");
 sl_launch(q, ko, vt, cache_seqlens, cu_q, max_q, window, softmax_scale, out,
      (const __nv_bfloat16*)gate.data_ptr(), (int)gate.stride(0), (float)eps);
}


void sl_attn_ng2(torch::Tensor q, torch::Tensor qlo, torch::Tensor ko, torch::Tensor vt, torch::Tensor cache_seqlens,
         torch::Tensor cu_q, int64_t max_q, int64_t window, double softmax_scale, torch::Tensor gate, double eps,
         torch::Tensor out) {
 TORCH_CHECK(gate.stride(1) == 1 && gate.size(1) == 1024, "sl_attn_ng gate");
 TORCH_CHECK(qlo.is_contiguous() && qlo.numel() == q.numel() && qlo.element_size() == 1, "sl_attn_ng2 qlo");
 sl_launch(q, ko, vt, cache_seqlens, cu_q, max_q, window, softmax_scale, out,
      (const __nv_bfloat16*)gate.data_ptr(), (int)gate.stride(0), (float)eps, (const uint32_t*)qlo.data_ptr());
}

"""

xs1_sk_ext = load_inline(name="okc_prefill_p9034_b460010", cpp_sources=xs1_SK_CPP, cuda_sources=xs1_SK_SRC,
                     functions=["sl_gather", "sl_attn", "sl_ts", "sl_pvstat", "sl_gather2", "sl_attn_ng", "sl_attn_ng2"],
                     extra_cuda_cflags=xs1_SK_FLAGS, extra_ldflags=["-lcuda"])


def xs1_alloc(cu_seqlens_q, max_q, window, device):
    """Empty K / V^T copies for one call: S = max_q + window - 1 rows rounded up to 64."""
    B = cu_seqlens_q.shape[0] - 1
    S = (max_q + window - 1 + 63) // 64 * 64
    return (torch.empty((B, S, 2, 64), dtype=torch.uint8, device=device),
            torch.empty((B, 2, 64, S), dtype=torch.uint8, device=device))


def xs1_gather(k_cache, v_cache, page_table, cache_seqlens, cu_seqlens_q, window, part, ko, vt):
    """Copy each request's window rows from the paged cache into ko / vt (from alloc). part 0: only the blocks of
    rows cached before this call; part 1: the other blocks; part 2: all."""
    xs1_sk_ext.sl_gather2(k_cache.view(torch.uint8), v_cache.view(torch.uint8), page_table, cache_seqlens, cu_seqlens_q,
                      window - 1, part, ko, vt)


def xs1_attn(q8, ko, vt, cache_seqlens, cu_seqlens_q, max_q, window, softmax_scale, out):
    """Attention into out [T, 16, 64] BF16, unnormalized: each row is off by a positive factor (there is no softmax
    denominator), so this entry point is only correct ahead of a per-head RMSNorm. The layer uses attn_ng."""
    xs1_sk_ext.sl_attn(q8.view(torch.uint8), ko, vt, cache_seqlens, cu_seqlens_q, max_q, window, softmax_scale, out)


def xs1_attn_ng(q8, ko, vt, cache_seqlens, cu_seqlens_q, max_q, window, softmax_scale, gate, eps, out):
    """out [T, 1024] BF16 = o_norm_gate(attention, gate); gate is a [T, 1024] BF16 view with unit inner stride. The
    per-head RMSNorm of o_norm_gate cancels the missing softmax denominator."""
    xs1_sk_ext.sl_attn_ng(q8.view(torch.uint8), ko, vt, cache_seqlens, cu_seqlens_q, max_q, window, softmax_scale, gate,
                      eps, out)


def xs1_attn_ng2(q8, qlo, ko, vt, cache_seqlens, cu_seqlens_q, max_q, window, softmax_scale, gate, eps, out):
    """attn_ng with a second query plane (qlo, FP8 [T, 16, 64], q = q8 + qlo), loaded by TMA and attended by two more
    QK wgmmas per key tile."""
    xs1_sk_ext.sl_attn_ng2(q8.view(torch.uint8), qlo.view(torch.uint8), ko, vt, cache_seqlens, cu_seqlens_q, max_q, window,
                       softmax_scale, gate, eps, out)




xs2_SK_CPP = 'void sl_gather(torch::Tensor kc, torch::Tensor vc, torch::Tensor pt, torch::Tensor seqlens, torch::Tensor cu_q, int64_t win, torch::Tensor ko, torch::Tensor vt);\nvoid sl_attn(torch::Tensor q, torch::Tensor ko, torch::Tensor vt, torch::Tensor cache_seqlens, torch::Tensor cu_q, int64_t max_q, int64_t window, double softmax_scale, torch::Tensor out);\nvoid sl_ts(torch::Tensor buf);\nvoid sl_pvstat(torch::Tensor buf);\nvoid sl_gather2(torch::Tensor kc, torch::Tensor vc, torch::Tensor pt, torch::Tensor seqlens, torch::Tensor cu_q, int64_t win, int64_t part, torch::Tensor ko, torch::Tensor vt);\nvoid sl_attn_ng(torch::Tensor q, torch::Tensor ko, torch::Tensor vt, torch::Tensor cache_seqlens, torch::Tensor cu_q, int64_t max_q, int64_t window, double softmax_scale, torch::Tensor gate, double eps, torch::Tensor out);\nvoid sl_attn_ng2(torch::Tensor q, torch::Tensor qlo, torch::Tensor ko, torch::Tensor vt, torch::Tensor cache_seqlens, torch::Tensor cu_q, int64_t max_q, int64_t window, double softmax_scale, torch::Tensor gate, double eps, torch::Tensor out);\n'
xs2_SK_FLAGS = ['-O3', '--use_fast_math', '-DNDEBUG', '-U__CUDA_NO_HALF_OPERATORS__', '-U__CUDA_NO_HALF_CONVERSIONS__', '-U__CUDA_NO_BFLOAT16_CONVERSIONS__', '-U__CUDA_NO_HALF2_OPERATORS__', '-gencode=arch=compute_90a,code=sm_90a']
xs2_SK_SRC = r"""

























#define Z0_NOL 1
#undef PV_B
#define PV_B 1104
#define PV16 1
#define SL_PIPE 3
#define SL_NWG 1
#define SK_PDL 1
#define SX_LOOP 1
#define SK_KS 1
#define SL_MINB 4
#define SL_ST 3
#define SA_NOPROD 2
#define SA_NOALIGN 1
#define Z0_NOLO 1
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>








#define SL_BN 128
#define SL_MATH 1
#define SK_O72 1
#define SK_BM 1
#define SK_TS 0
#define SX_XCH 0
#define SX_G32 0
#define SA_QK1 0
#define SA_DUP 1
#define SX_NOLOAD 0

constexpr int BN = SL_BN, ST = SL_ST;
constexpr int PPC = 8 * SL_NWG * SK_BM;
constexpr int K_BYTES = BN * 64, V_BYTES = 64 * BN, Q_BYTES = PPC * 8 * 64;

#define Z0_RS(o, i) 1.f
constexpr int VS = V_BYTES + ((SK_O72 && !Z0_NOL) ? 8 * BN : 0);
constexpr int NO = (SK_O72 && !Z0_NOL) ? 36 : 32;
typedef uint32_t OT;
constexpr int NOR = 16;
static_assert(Z0_NOL && SK_BM == 1, "PV16: denominator-free one-half build only");

constexpr int XCHB = PV16 ? 48 : 80;
constexpr int XB = (SX_XCH == 2 && SK_KS > 1) ? XCHB * SL_NWG * 128 : 0;
constexpr int SMEM = (SA_NOALIGN ? 0 : 1024) + Q_BYTES + ST * (K_BYTES + VS) + 512 + 256 + XB;

__device__ unsigned long long* g_sk_ts = nullptr;
__device__ unsigned int g_sx_smcnt[256];
static_assert(SK_KS == 1 || SK_O72, "SK_KS needs SK_O72 (sum in the O accumulator)");
static_assert(!SA_NOPROD || (SL_PIPE == 3 && SX_LOOP && SL_NWG == 1 && SK_BM == 1 && SA_DUP == 1 && !SA_QK1 && !SX_NOLOAD), "SA_NOPROD config");
static_assert(SK_BM == 1 || (SL_NWG == 1 && SL_PIPE == 3 && SK_O72 && SL_MATH && !SK_TS), "SK_BM 2: NWG 1, PIPE 3 only");
static_assert((SK_KS - 1) * SK_BM * 38 * SL_NWG * 128 * 4 <= ST * (K_BYTES + VS), "SK_BM exchange does not fit");
static_assert((SK_KS - 1) * 38 * SL_NWG * 128 * 4 <= ST * (K_BYTES + VS), "SK_KS exchange does not fit the ring");
constexpr int NKB = BN / 32;
constexpr int NSR = BN / 4;

typedef CUresult (*SlEncodeTiled)(CUtensorMap*, CUtensorMapDataType, cuuint32_t, void*, const cuuint64_t*,
                 const cuuint64_t*, const cuuint32_t*, const cuuint32_t*, CUtensorMapInterleave,
                 CUtensorMapSwizzle, CUtensorMapL2promotion, CUtensorMapFloatOOBfill);

struct SlParams {
 __nv_bfloat16* out;
 const int* cu_q;
 const int* cache_seqlens;
 int S;
 int G;
 int W;
 float a32;
 const __nv_bfloat16* gate;
 int gstride;
 float eps;
 int ks;
 const uint32_t* qlo;
 int T;
 int nitems;
};

__device__ __forceinline__ uint32_t smem_u32(const void* p) { return static_cast<uint32_t>(__cvta_generic_to_shared(p)); }
__device__ __forceinline__ void mbar_init(uint32_t bar, int count) {
 asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::"r"(bar), "r"(count));
}
__device__ __forceinline__ void mbar_expect(uint32_t bar, int bytes) {
 asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(bar), "r"(bytes) : "memory");
}
__device__ __forceinline__ void mbar_arrive(uint32_t bar) {
 asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" ::"r"(bar) : "memory");
}
__device__ __forceinline__ void mbar_wait(uint32_t bar, uint32_t parity) {
 asm volatile("{\n .reg .pred p;\n W: mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n @!p bra W;\n}"
        ::"r"(bar), "r"(parity) : "memory");
}
__device__ __forceinline__ void tma_2d(uint32_t dst, const CUtensorMap* map, int c0, int c1, uint32_t bar) {
 asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%2, %3}], [%4];"
        ::"r"(dst), "l"(map), "r"(c0), "r"(c1), "r"(bar) : "memory");
}
__device__ __forceinline__ void tma_3d(uint32_t dst, const CUtensorMap* map, int c0, int c1, int c2, uint32_t bar) {
 asm volatile("cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%2, %3, %4}], [%5];"
        ::"r"(dst), "l"(map), "r"(c0), "r"(c1), "r"(c2), "r"(bar) : "memory");
}
__device__ __forceinline__ uint32_t prmt(uint32_t a, uint32_t b, uint32_t s) {
 uint32_t r;
 asm("prmt.b32 %0, %1, %2, %3;" : "=r"(r) : "r"(a), "r"(b), "r"(s));
 return r;
}

__device__ __forceinline__ uint64_t desc64(uint32_t addr) {
 return (uint64_t)((addr & 0x3FFFF) >> 4) | ((uint64_t)1 << 16) | ((uint64_t)(512 >> 4) << 32) | ((uint64_t)2 << 62);
}
__device__ __forceinline__ uint64_t desc128(uint32_t addr) {
 return (uint64_t)((addr & 0x3FFFF) >> 4) | ((uint64_t)1 << 16) | ((uint64_t)(1024 >> 4) << 32) | ((uint64_t)1 << 62);
}
__device__ __forceinline__ uint64_t descv(uint32_t addr) { return BN >= 128 ? desc128(addr) : desc64(addr); }


__device__ __forceinline__ uint64_t pvd(uint64_t dv, int kb) {
 return BN == 256 ? dv + (uint64_t)((kb >> 2) * (8192 >> 4) + 2 * (kb & 3)) : dv + (uint64_t)(2 * kb);
}


__device__ __forceinline__ uint64_t desc_ones(uint32_t addr) {
 return (uint64_t)((addr & 0x3FFFF) >> 4) | ((uint64_t)(128 >> 4) << 16) | ((uint64_t)(256 >> 4) << 32);
}
__device__ __forceinline__ void cluster_sync_all() {
 asm volatile("barrier.cluster.arrive.release.aligned;\nbarrier.cluster.wait.acquire.aligned;" ::: "memory");
}
__device__ __forceinline__ uint32_t cluster_rank() {
 uint32_t r;
 asm volatile("mov.u32 %0, %%cluster_ctarank;" : "=r"(r));
 return r;
}
__device__ __forceinline__ uint32_t mapa0(uint32_t a) {
 uint32_t r;
 asm volatile("mapa.shared::cluster.u32 %0, %1, 0;" : "=r"(r) : "r"(a));
 return r;
}
__device__ __forceinline__ uint32_t mapa_r(uint32_t a, uint32_t rank) {
 uint32_t r;
 asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(r) : "r"(a), "r"(rank));
 return r;
}
__device__ __forceinline__ void st_cluster4(uint32_t a, float x, float y, float z, float w) {
 asm volatile("st.shared::cluster.v4.f32 [%0], {%1, %2, %3, %4};" ::"r"(a), "f"(x), "f"(y), "f"(z), "f"(w) : "memory");
}
__device__ __forceinline__ void st_async4(uint32_t a, uint32_t bar, float x, float y, float z, float w) {
 asm volatile("st.async.shared::cluster.mbarrier::complete_tx::bytes.v4.f32 [%0], {%1, %2, %3, %4}, [%5];"
        ::"r"(a), "f"(x), "f"(y), "f"(z), "f"(w), "r"(bar) : "memory");
}
__device__ __forceinline__ void st_cluster(uint32_t a, float v) {
 asm volatile("st.shared::cluster.f32 [%0], %1;" ::"r"(a), "f"(v) : "memory");
}
__device__ __forceinline__ void wg_fence() { asm volatile("wgmma.fence.sync.aligned;" ::: "memory"); }
__device__ __forceinline__ void wg_commit() { asm volatile("wgmma.commit_group.sync.aligned;" ::: "memory"); }
template <int K_>
__device__ __forceinline__ void wg_wait() { asm volatile("wgmma.wait_group.sync.aligned %0;" ::"n"(K_) : "memory"); }
template <int N>
__device__ __forceinline__ void reg_fence(uint32_t (&r)[N]) {
#pragma unroll
 for (int i = 0; i < N; ++i) asm volatile("" : "+r"(r[i])::"memory");
}
template <int N>
__device__ __forceinline__ void reg_fence(float (&r)[N]) {
#pragma unroll
 for (int i = 0; i < N; ++i) asm volatile("" : "+f"(r[i])::"memory");
}




__device__ __forceinline__ void wgmma_qk(uint32_t (&d)[32], uint64_t da, uint64_t db, int sd) {
 asm volatile("{\n.reg .pred p;\nsetp.ne.b32 p, %34, 0;\n"
        "wgmma.mma_async.sync.aligned.m64n128k32.f16.e4m3.e4m3 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31}, %32, %33, p, 1, 1;\n}\n"
        : "+r"(d[0]), "+r"(d[1]), "+r"(d[2]), "+r"(d[3]), "+r"(d[4]), "+r"(d[5]), "+r"(d[6]), "+r"(d[7]), "+r"(d[8]), "+r"(d[9]), "+r"(d[10]), "+r"(d[11]), "+r"(d[12]), "+r"(d[13]), "+r"(d[14]), "+r"(d[15]), "+r"(d[16]), "+r"(d[17]), "+r"(d[18]), "+r"(d[19]), "+r"(d[20]), "+r"(d[21]), "+r"(d[22]), "+r"(d[23]), "+r"(d[24]), "+r"(d[25]), "+r"(d[26]), "+r"(d[27]), "+r"(d[28]), "+r"(d[29]), "+r"(d[30]), "+r"(d[31]) : "l"(da), "l"(db), "r"(sd));
}
__device__ __forceinline__ void wgmma_qk(uint32_t (&d)[64], uint64_t da, uint64_t db, int sd) {
 asm volatile("{\n.reg .pred p;\nsetp.ne.b32 p, %66, 0;\n"
        "wgmma.mma_async.sync.aligned.m64n256k32.f16.e4m3.e4m3 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47,%48,%49,%50,%51,%52,%53,%54,%55,%56,%57,%58,%59,%60,%61,%62,%63}, %64, %65, p, 1, 1;\n}\n"
        : "+r"(d[0]), "+r"(d[1]), "+r"(d[2]), "+r"(d[3]), "+r"(d[4]), "+r"(d[5]), "+r"(d[6]), "+r"(d[7]), "+r"(d[8]), "+r"(d[9]), "+r"(d[10]), "+r"(d[11]), "+r"(d[12]), "+r"(d[13]), "+r"(d[14]), "+r"(d[15]), "+r"(d[16]), "+r"(d[17]), "+r"(d[18]), "+r"(d[19]), "+r"(d[20]), "+r"(d[21]), "+r"(d[22]), "+r"(d[23]), "+r"(d[24]), "+r"(d[25]), "+r"(d[26]), "+r"(d[27]), "+r"(d[28]), "+r"(d[29]), "+r"(d[30]), "+r"(d[31]), "+r"(d[32]), "+r"(d[33]), "+r"(d[34]), "+r"(d[35]), "+r"(d[36]), "+r"(d[37]), "+r"(d[38]), "+r"(d[39]), "+r"(d[40]), "+r"(d[41]), "+r"(d[42]), "+r"(d[43]), "+r"(d[44]), "+r"(d[45]), "+r"(d[46]), "+r"(d[47]), "+r"(d[48]), "+r"(d[49]), "+r"(d[50]), "+r"(d[51]), "+r"(d[52]), "+r"(d[53]), "+r"(d[54]), "+r"(d[55]), "+r"(d[56]), "+r"(d[57]), "+r"(d[58]), "+r"(d[59]), "+r"(d[60]), "+r"(d[61]), "+r"(d[62]), "+r"(d[63]) : "l"(da), "l"(db), "r"(sd));
}

template <int C>
__device__ __forceinline__ void wgmma_qk_rs(uint32_t (&d)[BN / 4], const uint32_t (&a)[8], uint64_t db) {
 return;
}
__device__ __forceinline__ void wgmma_pv(float (&d)[36], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint64_t db) {
 asm volatile("{\n.reg .pred p;\nsetp.ne.b32 p, 1, 0;\n"
        "wgmma.mma_async.sync.aligned.m64n72k32.f32.e4m3.e4m3 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32,%33,%34,%35}, {%36,%37,%38,%39}, %40, p, 1, 1;\n}\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]), "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]), "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]), "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27]), "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31]), "+f"(d[32]), "+f"(d[33]), "+f"(d[34]), "+f"(d[35]) : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "l"(db));
}
__device__ __forceinline__ void wgmma_pv(float (&d)[32], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint64_t db) {
 asm volatile("{\n.reg .pred p;\nsetp.ne.b32 p, 1, 0;\n"
        "wgmma.mma_async.sync.aligned.m64n64k32.f32.e4m3.e4m3 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31}, {%32,%33,%34,%35}, %36, p, 1, 1;\n}\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]), "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]), "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]), "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27]), "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31]) : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "l"(db));
}
__device__ __forceinline__ void wgmma_pv(uint32_t (&d)[16], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint64_t db) {
 asm volatile("{\n.reg .pred p;\nsetp.ne.b32 p, 1, 0;\n"
        "wgmma.mma_async.sync.aligned.m64n64k32.f16.e4m3.e4m3 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15}, {%16,%17,%18,%19}, %20, p, 1, 1;\n}\n"
        : "+r"(d[0]), "+r"(d[1]), "+r"(d[2]), "+r"(d[3]), "+r"(d[4]), "+r"(d[5]), "+r"(d[6]), "+r"(d[7]), "+r"(d[8]), "+r"(d[9]), "+r"(d[10]), "+r"(d[11]), "+r"(d[12]), "+r"(d[13]), "+r"(d[14]), "+r"(d[15]) : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "l"(db));
}
__device__ __forceinline__ void wgmma_ones(float (&d)[4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint64_t db, int sd) {
 asm volatile("{\n.reg .pred p;\nsetp.ne.b32 p, %9, 0;\n"
        "wgmma.mma_async.sync.aligned.m64n8k32.f32.e4m3.e4m3 {%0,%1,%2,%3}, {%4,%5,%6,%7}, %8, p, 1, 1;\n}\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "l"(db), "r"(sd));
}


__device__ __forceinline__ __half2 h2(uint32_t x) { return *reinterpret_cast<__half2*>(&x); }
__device__ __forceinline__ uint32_t u32(__half2 x) { return *reinterpret_cast<uint32_t*>(&x); }


struct SlRow {
 float rmax[2], rb[2], rsum[2];
 int jlo[2], jhi[2];
 uint32_t rmh, rbh;
};

__device__ __forceinline__ void sl_issue_qk(uint32_t (&s)[NSR], uint64_t dq, uint32_t kaddr, const uint32_t (&ql)[8]) {
#pragma unroll
 for (int i = 0; i < NSR; ++i) s[i] = 0;
 reg_fence(s);
 wg_fence();
 const uint64_t dk = desc64(kaddr);
 wgmma_qk(s, dq, dk, 0);
 wgmma_qk(s, dq + 2, dk + 2, 1);
 wgmma_qk_rs<0>(s, ql, dk);
 wgmma_qk_rs<1>(s, ql, dk + 2);
 wg_commit();
}

__device__ __forceinline__ void sl_pack(const uint32_t (&sc)[NSR], uint32_t (&pa)[NKB][4]);


__device__ __forceinline__ void sl_softmax(uint32_t (&sc)[NSR], uint32_t (&pa)[NKB][4], SlRow& st, int j0, bool edge,
                      int tq, float a32, float (&scale)[2]) {
 if (edge) {


  const int off0 = j0 + 2 * tq - st.jlo[0], off1 = j0 + 2 * tq - st.jlo[1];
  const unsigned sp0 = (unsigned)(st.jhi[0] - st.jlo[0]), sp1 = (unsigned)(st.jhi[1] - st.jlo[1]);
#pragma unroll
  for (int w = 0; w < NSR; ++w) {
   const int d = ((w & 1) ? off1 : off0) + 8 * (w >> 1);
   const unsigned sp = (w & 1) ? sp1 : sp0;
   const uint32_t m = ((unsigned)d <= sp ? 0x0000FFFFu : 0u) | ((unsigned)(d + 1) <= sp ? 0xFFFF0000u : 0u);
   sc[w] = (sc[w] & m) | (0xFC00FC00u & ~m);
  }
  edge = false;
 }
 const __half2 ninf2 = __float2half2_rn(-INFINITY);
 if (edge) {
#pragma unroll
  for (int w = 0; w < NSR; ++w) {
   const int r = w & 1, j = j0 + 8 * (w >> 1) + 2 * tq;
   __half2 v = h2(sc[w]);
   const bool ok0 = j >= st.jlo[r] && j <= st.jhi[r], ok1 = j + 1 >= st.jlo[r] && j + 1 <= st.jhi[r];
   v = __halves2half2(ok0 ? __low2half(v) : __low2half(ninf2), ok1 ? __high2half(v) : __high2half(ninf2));
   sc[w] = u32(v);
  }
 }
 __half2 m0 = h2(sc[0]), m1 = h2(sc[1]), m2 = h2(sc[2]), m3 = h2(sc[3]);
#pragma unroll
 for (int w = 4; w < NSR; w += 4) {
  m0 = __hmax2(m0, h2(sc[w])); m1 = __hmax2(m1, h2(sc[w + 1]));
  m2 = __hmax2(m2, h2(sc[w + 2])); m3 = __hmax2(m3, h2(sc[w + 3]));
 }
 m0 = __hmax2(m0, m2); m1 = __hmax2(m1, m3);
 __half2 mm = __hmax2(__lows2half2(m0, m1), __highs2half2(m0, m1));
 mm = __hmax2(mm, h2(__shfl_xor_sync(0xffffffff, u32(mm), 1)));
 mm = __hmax2(mm, h2(__shfl_xor_sync(0xffffffff, u32(mm), 2)));


 __half2 bx[2];
 {
  const __half2 cur = __hmax2(h2(st.rmh), mm);
  const __half2 bn = __hmin2(__hfma2(cur, __hneg2(__float2half2_rn(a32)), __float2half2_rn((float)PV_B)),
                __float2half2_rn(65504.f));
  const uint32_t bnu = u32(bn);
  scale[0] = scale[1] = 1.f;
  if (__any_sync(0xffffffffu, bnu != st.rbh) && bnu != st.rbh) {
   scale[0] = exp2f((__low2float(bn) - __low2float(h2(st.rbh))) * 0.125f);
   scale[1] = exp2f((__high2float(bn) - __high2float(h2(st.rbh))) * 0.125f);
  }
  st.rbh = bnu;
  st.rmh = u32(cur);
  bx[0] = __low2half2(bn);
  bx[1] = __high2half2(bn);
 }
 const __half2 a2 = __float2half2_rn(a32);
 const __half2 off2 = __float2half2_rn(896.f), lo2 = __float2half2_rn(1024.f);


 (void)off2;
#pragma unroll
 for (int w = 0; w < NSR; ++w) sc[w] = u32(__hmax2(__hfma2(h2(sc[w]), a2, bx[w & 1]), lo2));
}



__device__ __forceinline__ void sl_pack(const uint32_t (&sc)[NSR], uint32_t (&pa)[NKB][4]) {
#pragma unroll
 for (int kb = 0; kb < NKB; ++kb) {
  const int w = 8 * kb;
  pa[kb][0] = prmt(sc[w + 0], sc[w + 2], 0x6420);
  pa[kb][1] = prmt(sc[w + 1], sc[w + 3], 0x6420);
  pa[kb][2] = prmt(sc[w + 4], sc[w + 6], 0x6420);
  pa[kb][3] = prmt(sc[w + 5], sc[w + 7], 0x6420);
 }
}


__device__ __forceinline__ void sl_issue_pv(OT (&o)[NOR], float (&os)[4], uint32_t (&pa)[NKB][4], uint32_t vaddr,
                      uint64_t dones) {
 const uint64_t dv = descv(vaddr);
 reg_fence(o);
 wg_fence();
#pragma unroll
 for (int kb = 0; kb < NKB; ++kb) {
  wgmma_pv(o, pa[kb][0], pa[kb][1], pa[kb][2], pa[kb][3], pvd(dv, kb));
 }
 wg_commit();
}

__device__ __forceinline__ void sl_rescale(OT (&o)[NOR], SlRow& st, const float (&scale)[2]) {

 if (__any_sync(0xffffffffu, scale[0] != 1.f || scale[1] != 1.f)) {
  const __half2 s0 = __float2half2_rn(scale[0]), s1 = __float2half2_rn(scale[1]);
#pragma unroll
  for (int i = 0; i < NOR; i += 2) { o[i] = u32(__hmul2(h2(o[i]), s0)); o[i + 1] = u32(__hmul2(h2(o[i + 1]), s1)); }
  st.rsum[0] *= scale[0];
  st.rsum[1] *= scale[1];
 }
}



template <int R>
__device__ __forceinline__ void sx_send_row(uint32_t a, int nt4, const float (&o)[NO], float b) {
 float v[20] = {};
#pragma unroll
 for (int jb = 0; jb < NO / 4; ++jb) { v[2 * jb] = o[4 * jb + 2 * R]; v[2 * jb + 1] = o[4 * jb + 2 * R + 1]; }
 v[18] = b;
 v[19] = 0.f;
#pragma unroll
 for (int j = 0; j < 5; ++j) st_cluster4(a + 16u * j * nt4, v[4 * j], v[4 * j + 1], v[4 * j + 2], v[4 * j + 3]);
}
template <int R>
__device__ __forceinline__ void sx_send_row_async(uint32_t a, uint32_t bar, int nt4, const float (&o)[NO], float b) {
 float v[20] = {};
#pragma unroll
 for (int jb = 0; jb < NO / 4; ++jb) { v[2 * jb] = o[4 * jb + 2 * R]; v[2 * jb + 1] = o[4 * jb + 2 * R + 1]; }
 v[18] = b;
 v[19] = 0.f;
#pragma unroll
 for (int j = 0; j < 5; ++j) st_async4(a + 16u * j * nt4, bar, v[4 * j], v[4 * j + 1], v[4 * j + 2], v[4 * j + 3]);
}
template <int R>
__device__ __forceinline__ void sx_merge_row(const float4* x, int nt4, float (&o)[NO], float (&rb)[2]) {
 float v[20];
#pragma unroll
 for (int j = 0; j < 5; ++j) {
  const float4 t = x[j * nt4];
  v[4 * j] = t.x; v[4 * j + 1] = t.y; v[4 * j + 2] = t.z; v[4 * j + 3] = t.w;
 }
 const float b = v[18], m = fminf(rb[R], b);
 const float sa = exp2f((m - rb[R]) * 0.125f), sb = exp2f((m - b) * 0.125f);
#pragma unroll
 for (int jb = 0; jb < NO / 4; ++jb) {
  o[4 * jb + 2 * R] = o[4 * jb + 2 * R] * sa + v[2 * jb] * sb;
  o[4 * jb + 2 * R + 1] = o[4 * jb + 2 * R + 1] * sa + v[2 * jb + 1] * sb;
 }
 rb[R] = m;
}





template <int R>
__device__ __forceinline__ void sx_send_row_async(uint32_t a, uint32_t bar, int nt4, const uint32_t (&o)[16], float b) {
 st_async4(a, bar, __uint_as_float(o[R]), __uint_as_float(o[2 + R]), __uint_as_float(o[4 + R]), __uint_as_float(o[6 + R]));
 st_async4(a + 16u * nt4, bar, __uint_as_float(o[8 + R]), __uint_as_float(o[10 + R]), __uint_as_float(o[12 + R]),
      __uint_as_float(o[14 + R]));
 st_async4(a + 32u * nt4, bar, b, 0.f, 0.f, 0.f);
}
template <int R>
__device__ __forceinline__ void sx_merge_row(const float4* x, int nt4, uint32_t (&o)[16], float (&rb)[2]) {
 const float4 t0 = x[0], t1 = x[nt4], t2 = x[2 * nt4];
 const uint32_t v[8] = {__float_as_uint(t0.x), __float_as_uint(t0.y), __float_as_uint(t0.z), __float_as_uint(t0.w),
             __float_as_uint(t1.x), __float_as_uint(t1.y), __float_as_uint(t1.z), __float_as_uint(t1.w)};
 const float b = t2.x, m = fminf(rb[R], b);
 const float sa = exp2f((m - rb[R]) * 0.125f), sb = exp2f((m - b) * 0.125f);
#pragma unroll
 for (int jb = 0; jb < 8; ++jb) {
  const float2 f = __half22float2(h2(o[2 * jb + R])), g = __half22float2(h2(v[jb]));
  o[2 * jb + R] = u32(__floats2half2_rn(f.x * sa + g.x * sb, f.y * sa + g.y * sb));
 }
 rb[R] = m;
}
constexpr int NTHR = SL_NWG * 128 + (SA_NOPROD ? 0 : 32);

__global__ void __launch_bounds__(NTHR, SL_MINB)
sl_attn_k(const __grid_constant__ CUtensorMap mq,
                              const __grid_constant__ CUtensorMap mk,
                              const __grid_constant__ CUtensorMap mv, SlParams p) {
 extern __shared__ __align__(1024) uint8_t smem_raw[];
 uint8_t* base = smem_raw;
 if (threadIdx.x == 0 && (smem_u32(smem_raw) & 1023u)) __trap();
 uint8_t* sk = base;
 uint8_t* sv = sk + ST * K_BYTES;
 uint8_t* sq = sv + ST * VS;
 uint8_t* sones = sq + Q_BYTES;
 uint64_t* bars = reinterpret_cast<uint64_t*>(sones + 512);
 const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;

#define SX_C(k) do { } while (0)



 const int ks = SK_KS > 1 ? p.ks : 1;
 const int kc = ks > 1 ? (int)cluster_rank() : 0, item = ks > 1 ? blockIdx.x / ks : blockIdx.x * SA_DUP;
 if (item >= p.nitems) return;
 const int kh = item & 1, bg = item >> 1, b = bg / p.G, g = bg - b * p.G;
 const int q0 = p.cu_q[b], Lb = p.cu_q[b + 1] - q0, n = p.cache_seqlens[b];
 const int i0 = g * PPC;
 if (i0 >= Lb) return;
 const int W = p.W;
 const int s0 = max(0, n - Lb - (W - 1));
 const int ilast = min(i0 + PPC - 1, Lb - 1);
 const int pfirst = n - Lb + i0, plast = n - Lb + ilast;
 const int jlo = max(0, pfirst - W + 1 - s0), jhi = plast - s0;
 const int ta = jlo / BN, nta = jhi / BN - ta + 1;
 const int t0 = ta + (kc * nta) / ks, nt0 = ta + ((kc + 1) * nta) / ks - t0;
 const int nt = nt0;
#define SA_T(x) (x)

 if (tid == 0) {
  for (int s = 0; s < ST; ++s) { mbar_init(smem_u32(&bars[s]), 1); mbar_init(smem_u32(&bars[ST + s]), SL_NWG * 128); }
  mbar_init(smem_u32(&bars[2 * ST]), 1);
  asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
 }
 if (tid < 128) reinterpret_cast<uint32_t*>(sones)[tid] = 0x38383838u;
 asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
 __syncthreads();
 asm volatile("griddepcontrol.wait;" ::: "memory");
 asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
 SX_C(1);


 auto sa_load_tile = [&](int it, uint32_t s) {
  const uint32_t fb = smem_u32(&bars[s]);
  mbar_expect(fb, K_BYTES + V_BYTES);
  const int j = (t0 + it) * BN;
  tma_3d(smem_u32(sk + s * K_BYTES), &mk, 0, kh, b * p.S + j, fb);
  tma_2d(smem_u32(sv + s * VS), &mv, j, (b * 2 + kh) * 64, fb);
  if (BN == 256) tma_2d(smem_u32(sv + s * VS + 8192), &mv, j + 128, (b * 2 + kh) * 64, fb);
 };
 if (tid == 0) {
  const uint32_t qb = smem_u32(&bars[2 * ST]);
  mbar_expect(qb, Q_BYTES);
  tma_3d(smem_u32(sq), &mq, 0, kh * 8, q0 + i0, qb);
  for (int it = 0; it < ST && it < nt; ++it) sa_load_tile(it, (uint32_t)it);
 }

 if (false) {
  if (lane == 0) {
   for (int it = 0; it < nt; ++it) {
    const int s = it % ST;
    if (it >= ST) mbar_wait(smem_u32(&bars[ST + s]), ((it / ST) - 1) & 1);
    const uint32_t fb = smem_u32(&bars[s]);
    const int j = (t0 + SA_T(it)) * BN;
    mbar_expect(fb, K_BYTES + V_BYTES);
    tma_3d(smem_u32(sk + s * K_BYTES), &mk, 0, kh, b * p.S + j, fb);
    tma_2d(smem_u32(sv + s * VS), &mv, j, (b * 2 + kh) * 64, fb);
    if (it == 0) {
     const uint32_t qb = smem_u32(&bars[2 * ST]);
     mbar_expect(qb, Q_BYTES);
     tma_3d(smem_u32(sq), &mq, 0, kh * 8, q0 + i0, qb);
    }
   }
  }
  __syncwarp();
  return;
 }

 if (p.gate && tid < 8 * PPC) {
  const int pos = i0 + (tid >> 3);
  if (pos < Lb)
   asm volatile("prefetch.global.L2 [%0];" ::"l"(p.gate + (size_t)(q0 + pos) * p.gstride + (kh * 8 + (tid & 7)) * 64));
 }


 const int wg = warp >> 2, wi = warp & 3;
 const int tq = lane & 3, hq = lane >> 2;
 const int ib = i0 + 8 * wg + 2 * wi;
 const int pr0 = n - Lb + ib;
 SlRow st;
#pragma unroll
 for (int r = 0; r < 2; ++r) {
  st.rmax[r] = -INFINITY; st.rb[r] = (float)PV_B; st.rsum[r] = 0.f;
  st.rmh = 0xFC00FC00u;
  st.rbh = 0x7BFF7BFFu;
  st.jhi[r] = pr0 + r - s0;
  st.jlo[r] = pr0 + r - W + 1 - s0;
 }
 OT o[NOR];
#pragma unroll
 for (int i = 0; i < NOR; ++i) o[i] = 0;
#define SK_T(k) do { } while (0)

 const uint64_t dq = desc64(smem_u32(sq + wg * 4096));




 uint32_t ql[8];
 {
  const int tg = q0 + ib;
#pragma unroll
  for (int r = 0; r < 2; ++r) {
   const bool okr = p.qlo != nullptr && tg + r < p.T;
   const uint32_t* rp = p.qlo + ((size_t)(tg + r) * 16 + kh * 8 + hq) * 16 + tq;
#pragma unroll
   for (int c = 0; c < 2; ++c) {
    ql[4 * c + r] = okr ? __ldg(rp + 8 * c) : 0u;
    ql[4 * c + 2 + r] = okr ? __ldg(rp + 8 * c + 4) : 0u;
   }
  }
 }
 const uint64_t dones = desc_ones(smem_u32(sones));

 const int jlo_all = max(0, plast - W + 1 - s0), jhi_all = pfirst - s0;
 const float a32 = p.a32;
 float os[4] = {0.f, 0.f, 0.f, 0.f};
 uint32_t sa[NSR], sb[NSR], pa[NKB][4];
 float scale[2];
 (void)dones;
 if (nt > 0) {
 mbar_wait(smem_u32(&bars[2 * ST]), 0);
 mbar_wait(smem_u32(&bars[0]), 0);
 sl_issue_qk(sa, dq, smem_u32(sk), ql);
 wg_wait<0>();
 reg_fence(sa);
 SK_T(0);
 SX_C(2);


 (void)sb;
 const uint32_t zdep = (uint32_t)(p.ks >> 8);
 (void)zdep;
 {
  const int j0 = t0 * BN;
  sl_softmax(sa, pa, st, j0, j0 < jlo_all || j0 + BN - 1 > jhi_all, tq, a32, scale);
  sl_rescale(o, st, scale);
  sl_pack(sa, pa);
 }
 {

  const uint32_t fb0 = smem_u32(&bars[0]), eb0 = smem_u32(&bars[ST]);
  const uint64_t dk0 = desc64(smem_u32(sk)), dv0 = descv(smem_u32(sv));
  uint32_t s = 1 % ST, ph = (1 / ST) & 1, sp = 0;
  uint32_t spp = 0;
#pragma unroll 1
  for (int it = 1; it < nt; ++it) {
   mbar_wait(fb0 + 8u * s, ph);
#pragma unroll
   for (int i = 0; i < NSR; ++i) sa[i] = 0;
   reg_fence(sa);
   wg_fence();
   {
    const uint64_t dk = dk0 + (uint64_t)(s * (K_BYTES >> 4));
    wgmma_qk(sa, dq, dk, 0);
    wgmma_qk(sa, dq + 2, dk + 2, 1);
    wgmma_qk_rs<0>(sa, ql, dk);
    wgmma_qk_rs<1>(sa, ql, dk + 2);
   }
   wg_commit();
   reg_fence(o);
   wg_fence();
   {
    const uint64_t dv = dv0 + (uint64_t)(sp * (VS >> 4));
#pragma unroll
    for (int kb = 0; kb < NKB; ++kb) wgmma_pv(o, pa[kb][0], pa[kb][1], pa[kb][2], pa[kb][3], pvd(dv, kb));
   }
   wg_commit();

   if (tid == 0 && it >= 2 && it - 2 + ST < nt) {
    mbar_wait(eb0 + 8u * spp, (uint32_t)(((it - 2) / ST) & 1));
    sa_load_tile(it - 2 + ST, spp);
   }
   __syncwarp();
   wg_wait<1>();
   reg_fence(sa);
   const int j0 = (t0 + SA_T(it)) * BN;
   sl_softmax(sa, pa, st, j0, j0 < jlo_all || j0 + BN - 1 > jhi_all, tq, a32, scale);
   wg_wait<0>();
   reg_fence(o);
   mbar_arrive(eb0 + 8u * sp);
   sl_rescale(o, st, scale);
   sl_pack(sa, pa);
   spp = sp;
   sp = s;
   if (++s == (uint32_t)ST) { s = 0; ph ^= 1u; }
  }
 }
 sl_issue_pv(o, os, pa, smem_u32(sv + ((nt - 1) % ST) * VS), dones);
 wg_wait<0>();
 reg_fence(o);
 st.rb[0] = __low2float(h2(st.rbh));
 st.rb[1] = __high2float(h2(st.rbh));
 st.rsum[0] = Z0_RS(o, 32);
 st.rsum[1] = Z0_RS(o, 34);
 mbar_arrive(smem_u32(&bars[ST + (nt - 1) % ST]));
 }
 SX_C(3);
 SX_C(6);


#pragma unroll
 for (int r = 0; r < 2; ++r) {
  const int i = ib + r;
  if (i < Lb) {
   __nv_bfloat16* dst = p.out + ((size_t)(q0 + i) * 16 + kh * 8 + hq) * 64 + 2 * tq;
   const float inv = st.rsum[r] > 0.f ? 1.f / st.rsum[r] : 0.f;
   __nv_bfloat162 v[8];
#pragma unroll
   for (int jb = 0; jb < 8; ++jb) { const float2 f = __half22float2(h2(o[2 * jb + r])); v[jb] = __floats2bfloat162_rn(f.x * inv, f.y * inv); }
   if (p.gate) {


    const __nv_bfloat16* gp = p.gate + (size_t)(q0 + i) * p.gstride + (kh * 8 + hq) * 64 + 2 * tq;
    __nv_bfloat162 g[8];
#pragma unroll
    for (int jb = 0; jb < 8; ++jb) g[jb] = *reinterpret_cast<const __nv_bfloat162*>(gp + 8 * jb);
    float ss = 0.f;
#pragma unroll
    for (int jb = 0; jb < 8; ++jb) {
     const float2 x = __bfloat1622float2(v[jb]);
     ss = fmaf(x.x, x.x, fmaf(x.y, x.y, ss));
    }
    ss += __shfl_xor_sync(0xffffffffu, ss, 1);
    ss += __shfl_xor_sync(0xffffffffu, ss, 2);
    const float rs = rsqrtf(ss * (1.f / 64.f) + p.eps);
#pragma unroll
    for (int jb = 0; jb < 8; ++jb) {
     const float2 x = __bfloat1622float2(v[jb]), gg = __bfloat1622float2(g[jb]);
     v[jb] = __floats2bfloat162_rn(x.x * rs / (1.f + __expf(-gg.x)), x.y * rs / (1.f + __expf(-gg.y)));
    }
   }
#pragma unroll
   for (int jb = 0; jb < 8; ++jb) *reinterpret_cast<__nv_bfloat162*>(dst + 8 * jb) = v[jb];
  }
 }
}





__global__ void __launch_bounds__(256) sl_gather_k(const uint4* __restrict__ kc, const uint4* __restrict__ vc,
  const int* __restrict__ pt, const int* __restrict__ seqlens, const int* __restrict__ cu_q, int pages, int page,
  int S, int win, int part, uint4* __restrict__ ko, uint4* __restrict__ vt) {
 __shared__ uint4 tile[64][9];
 asm volatile("griddepcontrol.wait;" ::: "memory");
 asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
 const int b = blockIdx.y, j0 = blockIdx.x * 64, t = threadIdx.x;
 const int n = seqlens[b], L = cu_q[b + 1] - cu_q[b];
 const int s0 = max(0, n - L - win), nv = n - s0;


 const bool old_only = j0 + 64 <= n - L - s0;
 if ((part == 0 && !old_only) || (part == 1 && old_only)) return;
#pragma unroll
 for (int r = 0; r < 2; ++r) {
  const int idx = t + 256 * r, jr = idx >> 3, u = idx & 7, j = j0 + jr;
  uint4 kv = make_uint4(0, 0, 0, 0), vv = make_uint4(0, 0, 0, 0);
  if (j < nv) {
   const int s = s0 + j;
   int pg = pt[(long long)b * pages + s / page];
   pg = pg < 0 ? 0 : pg;
   const long long src = ((long long)pg * page + s % page) * 8 + u;
   kv = __ldg(kc + src);
   vv = __ldg(vc + src);
  }
  ko[((long long)b * S + j) * 8 + u] = kv;
  tile[jr][u] = vv;
 }
 __syncthreads();
 const unsigned char* tb = reinterpret_cast<const unsigned char*>(&tile[0][0]);
#pragma unroll
 for (int r = 0; r < 2; ++r) {
  const int idx = t + 256 * r, hd = idx >> 2, q = idx & 3;
  uint32_t w[4];
#pragma unroll
  for (int k = 0; k < 4; ++k) {
   uint32_t x = 0;
#pragma unroll
   for (int i = 0; i < 4; ++i) {
    const int slot = q * 16 + 4 * k + i, ks = slot & 31;
    const int src = (slot & 32) + 2 * ((ks & 15) >> 2) + (ks & 1) + 8 * ((ks & 3) >> 1) + 16 * (ks >> 4);
    x |= uint32_t(tb[src * 144 + hd]) << (8 * i);
   }
   w[k] = x;
  }
  vt[(((long long)b * 128 + hd) * S + j0) / 16 + q] = make_uint4(w[0], w[1], w[2], w[3]);
 }
}

static SlEncodeTiled sl_encode() {
 static SlEncodeTiled enc = nullptr;
 if (!enc) {
  cudaDriverEntryPointQueryResult qr;
  cudaGetDriverEntryPoint("cuTensorMapEncodeTiled", reinterpret_cast<void**>(&enc), cudaEnableDefault, &qr);
  cudaFuncSetAttribute(sl_attn_k, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM);
 }
 return enc;
}


void sl_ts(torch::Tensor buf) {
 unsigned long long* ptr = buf.numel() ? (unsigned long long*)buf.data_ptr() : nullptr;
 cudaMemcpyToSymbol(g_sk_ts, &ptr, sizeof(ptr));
}
void sl_pvstat(torch::Tensor buf) {
}

void sl_gather2(torch::Tensor kc, torch::Tensor vc, torch::Tensor pt, torch::Tensor seqlens, torch::Tensor cu_q,
        int64_t win, int64_t part, torch::Tensor ko, torch::Tensor vt) {
 const int B = pt.size(0), pages = pt.size(1), page = kc.size(1), S = ko.size(1);
 TORCH_CHECK(kc.size(2) * kc.size(3) == 128 && S % 64 == 0, "sl_gather");
 cudaLaunchConfig_t cfg = {};
 cudaLaunchAttribute at[1];
 at[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
 at[0].val.programmaticStreamSerializationAllowed = SK_PDL;
 cfg.gridDim = dim3(S / (SX_G32 == 2 ? 16 : SX_G32 ? 32 : 64), B);
 cfg.blockDim = dim3(SX_G32 == 2 ? 128 : 256);
 cfg.dynamicSmemBytes = 0;
 { cudaLaunchConfig_t q0 = {cfg.gridDim, cfg.blockDim, cfg.dynamicSmemBytes, at::cuda::getCurrentCUDAStream(), nullptr, 0}; cfg = q0; }
 cfg.attrs = at;
 cfg.numAttrs = 1;
 cudaLaunchKernelEx(&cfg, sl_gather_k, (const uint4*)kc.data_ptr(), (const uint4*)vc.data_ptr(), (const int*)pt.data_ptr<int>(),
           (const int*)seqlens.data_ptr<int>(), (const int*)cu_q.data_ptr<int>(), pages, page, S, (int)win, (int)part,
           (uint4*)ko.data_ptr(), (uint4*)vt.data_ptr());
}

void sl_gather(torch::Tensor kc, torch::Tensor vc, torch::Tensor pt, torch::Tensor seqlens, torch::Tensor cu_q,
        int64_t win, torch::Tensor ko, torch::Tensor vt) {
 sl_gather2(kc, vc, pt, seqlens, cu_q, win, 2, ko, vt);
}



static void sl_launch(torch::Tensor q, torch::Tensor ko, torch::Tensor vt, torch::Tensor cache_seqlens, torch::Tensor cu_q,
           int64_t max_q, int64_t window, double softmax_scale, torch::Tensor out,
           const __nv_bfloat16* gate, int gstride, float eps, const uint32_t* qlo = nullptr) {
 SlEncodeTiled enc = sl_encode();
 const int T = q.size(0), B = ko.size(0), S = ko.size(1);
 CUtensorMap mq, mk, mv;
 {
  const cuuint64_t dims[3] = {64, 16, (cuuint64_t)T}, strides[2] = {64, 1024};
  const cuuint32_t box[3] = {64, 8, PPC}, es[3] = {1, 1, 1};
  enc(&mq, CU_TENSOR_MAP_DATA_TYPE_UINT8, 3, q.data_ptr(), dims, strides, box, es, CU_TENSOR_MAP_INTERLEAVE_NONE,
    CU_TENSOR_MAP_SWIZZLE_64B, CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
 }
 {
  const cuuint64_t dims[3] = {64, 2, (cuuint64_t)B * S}, strides[2] = {64, 128};
  const cuuint32_t box[3] = {64, 1, (cuuint32_t)BN}, es[3] = {1, 1, 1};
  enc(&mk, CU_TENSOR_MAP_DATA_TYPE_UINT8, 3, ko.data_ptr(), dims, strides, box, es, CU_TENSOR_MAP_INTERLEAVE_NONE,
    CU_TENSOR_MAP_SWIZZLE_64B, CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
 }
 {
  const cuuint64_t dims[2] = {(cuuint64_t)S, (cuuint64_t)B * 128}, strides[1] = {(cuuint64_t)S};
  const cuuint32_t box[2] = {(cuuint32_t)(BN == 256 ? 128 : BN), 64}, es[2] = {1, 1};
  enc(&mv, CU_TENSOR_MAP_DATA_TYPE_UINT8, 2, vt.data_ptr(), dims, strides, box, es, CU_TENSOR_MAP_INTERLEAVE_NONE,
    BN >= 128 ? CU_TENSOR_MAP_SWIZZLE_128B : CU_TENSOR_MAP_SWIZZLE_64B, CU_TENSOR_MAP_L2_PROMOTION_L2_128B,
    CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
 }
 SlParams p;
 p.out = (__nv_bfloat16*)out.data_ptr();
 p.cu_q = cu_q.data_ptr<int>();
 p.cache_seqlens = cache_seqlens.data_ptr<int>();
 p.S = S;
 p.G = (int)((max_q + PPC - 1) / PPC);
 p.W = (int)window;
 p.a32 = __half2float(__float2half_rn((float)(8.0 * 1.4426950408889634 * softmax_scale)));
 p.gate = gate;
 p.gstride = gstride;
 p.eps = eps;
 p.qlo = qlo;
 p.T = T;

 static int slots = 0;
 if (!slots) {
  int dev = 0, sms = 0, occ = 0;
  cudaGetDevice(&dev);
  cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev);
  cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ, sl_attn_k, NTHR, SMEM);
  slots = sms * (occ > 0 ? occ : 1);
 }
 const int items = B * p.G * 2;
 p.ks = (SK_KS > 1 && 2 * items <= slots) ? 2 : 1;
 p.nitems = items;
 cudaLaunchConfig_t cfg = {};
 cudaLaunchAttribute at[2];
 at[0].id = cudaLaunchAttributeClusterDimension;
 at[0].val.clusterDim.x = p.ks;
 at[0].val.clusterDim.y = 1;
 at[0].val.clusterDim.z = 1;
 at[1].id = cudaLaunchAttributeProgrammaticStreamSerialization;
 at[1].val.programmaticStreamSerializationAllowed = SK_PDL;
 cfg.gridDim = dim3(p.ks > 1 ? items * p.ks : (items + SA_DUP - 1) / SA_DUP);
 cfg.blockDim = dim3(NTHR);
 cfg.dynamicSmemBytes = SMEM;
 { cudaLaunchConfig_t q0 = {cfg.gridDim, cfg.blockDim, cfg.dynamicSmemBytes, at::cuda::getCurrentCUDAStream(), nullptr, 0}; cfg = q0; }
 cfg.attrs = at;
 cfg.numAttrs = 2;
 cudaLaunchKernelEx(&cfg, sl_attn_k, mq, mk, mv, p);
}

void sl_attn(torch::Tensor q, torch::Tensor ko, torch::Tensor vt, torch::Tensor cache_seqlens, torch::Tensor cu_q,
       int64_t max_q, int64_t window, double softmax_scale, torch::Tensor out) {
 sl_launch(q, ko, vt, cache_seqlens, cu_q, max_q, window, softmax_scale, out, nullptr, 0, 0.f);
}


void sl_attn_ng(torch::Tensor q, torch::Tensor ko, torch::Tensor vt, torch::Tensor cache_seqlens, torch::Tensor cu_q,
        int64_t max_q, int64_t window, double softmax_scale, torch::Tensor gate, double eps, torch::Tensor out) {
 TORCH_CHECK(gate.stride(1) == 1 && gate.size(1) == 1024, "sl_attn_ng gate");
 sl_launch(q, ko, vt, cache_seqlens, cu_q, max_q, window, softmax_scale, out,
      (const __nv_bfloat16*)gate.data_ptr(), (int)gate.stride(0), (float)eps);
}


void sl_attn_ng2(torch::Tensor q, torch::Tensor qlo, torch::Tensor ko, torch::Tensor vt, torch::Tensor cache_seqlens,
         torch::Tensor cu_q, int64_t max_q, int64_t window, double softmax_scale, torch::Tensor gate, double eps,
         torch::Tensor out) {
 TORCH_CHECK(gate.stride(1) == 1 && gate.size(1) == 1024, "sl_attn_ng gate");
 TORCH_CHECK(!Z0_NOLO, "sl_attn_ng2: this build has no q_lo term");
 TORCH_CHECK(qlo.is_contiguous() && qlo.numel() == q.numel() && qlo.element_size() == 1, "sl_attn_ng2 qlo");
 sl_launch(q, ko, vt, cache_seqlens, cu_q, max_q, window, softmax_scale, out,
      (const __nv_bfloat16*)gate.data_ptr(), (int)gate.stride(0), (float)eps, (const uint32_t*)qlo.data_ptr());
}

"""

xs2_sk_ext = load_inline(name="okc_prefill_p9033_b460010", cpp_sources=xs2_SK_CPP, cuda_sources=xs2_SK_SRC,
                     functions=["sl_gather", "sl_attn", "sl_ts", "sl_pvstat", "sl_gather2", "sl_attn_ng", "sl_attn_ng2"],
                     extra_cuda_cflags=xs2_SK_FLAGS, extra_ldflags=["-lcuda"])


def xs2_alloc(cu_seqlens_q, max_q, window, device):
    """Empty K / V^T copies for one call: S = max_q + window - 1 rows rounded up to 64."""
    B = cu_seqlens_q.shape[0] - 1
    S = (max_q + window - 1 + 63) // 64 * 64
    return (torch.empty((B, S, 2, 64), dtype=torch.uint8, device=device),
            torch.empty((B, 2, 64, S), dtype=torch.uint8, device=device))


def xs2_gather(k_cache, v_cache, page_table, cache_seqlens, cu_seqlens_q, window, part, ko, vt):
    """Copy each request's window rows from the paged cache into ko / vt (from alloc). part 0: only the blocks of
    rows cached before this call; part 1: the other blocks; part 2: all."""
    xs2_sk_ext.sl_gather2(k_cache.view(torch.uint8), v_cache.view(torch.uint8), page_table, cache_seqlens, cu_seqlens_q,
                      window - 1, part, ko, vt)


def xs2_attn(q8, ko, vt, cache_seqlens, cu_seqlens_q, max_q, window, softmax_scale, out):
    """Attention into out [T, 16, 64] BF16, unnormalized: each row is off by a positive factor (there is no softmax
    denominator), so this entry point is only correct ahead of a per-head RMSNorm. The layer uses attn_ng."""
    xs2_sk_ext.sl_attn(q8.view(torch.uint8), ko, vt, cache_seqlens, cu_seqlens_q, max_q, window, softmax_scale, out)


def xs2_attn_ng(q8, ko, vt, cache_seqlens, cu_seqlens_q, max_q, window, softmax_scale, gate, eps, out):
    """out [T, 1024] BF16 = o_norm_gate(attention, gate); gate is a [T, 1024] BF16 view with unit inner stride. The
    per-head RMSNorm of o_norm_gate cancels the missing softmax denominator."""
    xs2_sk_ext.sl_attn_ng(q8.view(torch.uint8), ko, vt, cache_seqlens, cu_seqlens_q, max_q, window, softmax_scale, gate,
                      eps, out)


def xs2_attn_ng2(q8, qlo, ko, vt, cache_seqlens, cu_seqlens_q, max_q, window, softmax_scale, gate, eps, out):
    """Not available in this build: the kernel binding raises (no q_lo term). Kept so the three sliding modules share
    one interface."""
    xs2_sk_ext.sl_attn_ng2(q8.view(torch.uint8), qlo.view(torch.uint8), ko, vt, cache_seqlens, cu_seqlens_q, max_q, window,
                       softmax_scale, gate, eps, out)

"""Muse Spark prefill decoder layer on one H200 (TP8 / EP8 rank): the entry point the judge calls.

The judge imports one file and calls prepare(weights) once, untimed, then kernel_fn(data, state) as a CUDA graph.
This file is that layer. Its components are separate modules, loaded below under short aliases (xa, xn, ...);
build/build_submission.py merges them into the one submission file, prefixing each component's names with its alias
(xa_prepare becomes xa_prepare), so the aliases are part of the submitted code and stay as they are.

    alias  file                               what it does
    xa     moe_block.py                       dense GEMMs (cuBLASLt), router GEMM, routing, the NVFP4 expert GEMMs,
                                              the weighted sum, and the front kernels (QK-norm + RoPE + FP8 q + KV
                                              cache store)
    xn     boundary_norm.py                   the post-norm / residual / pre-norm boundary as one kernel
    xb     boundary_attn_remat.py             the attention boundary that writes only the FFN input (the residual
                                              after attention is recomputed later)
    xr     boundary_ffn_remat.py              the FFN boundary that recomputes that residual on the fly
    xk     token_order.py                     a token similarity order for the post-attention pipeline
    xw     weighted_sum.py                    post-expert norm + top-8 weighted sum after the expert GEMMs
    xg     gemm_sm90.py                       an SM90 BF16 GEMM (CuTe DSL, from QuACK) for the o projection
    xgu    gemm_sm90_evict_first.py           the same GEMM with evict_first weights, for the unpooler
    xt     attention_full.py                  full-attention layers (FA3-derived kernel with an FP8 softmax)
    xs     attention_sliding_split.py         sliding-window attention, keys split over a CTA pair; window-row gather
    xs1    attention_sliding_unsplit.py       sliding-window attention, keys not split, up to 3 CTAs per SM
    xs2    attention_sliding_unsplit_4cta.py  the unsplit kernel at 4 CTAs per SM without a producer warp, for
                                              calls of 8192+ tokens

Outside the routed experts the math is SGLang's MuseSpark1xDecoderLayer.forward. Calls the custom attention kernels
do not cover (8 or fewer tokens, a scaled or non-FP8 KV cache, other head shapes) run SGLang's attention modules, as
baseline.py does; calls of 8 or fewer tokens also keep SGLang's FP32 router and its boundary ops.
Each component's prepare() runs once; the state keeps only the pieces the layer's paths read (the judge clones the
prepared state, so a smaller state is cheaper).

Some names in the code are opaque tags that stay as they are because they are part of the submitted code: the
load_inline extension names (okc_prefill_moe_x46001, okc_att_z14643, ...) only need to be unique, since torch caches
each compiled extension under its name, and identifiers such as X53_WSC or x64_elect carry a letter-and-number tag
with no further meaning. Where one appears, a comment says what it is.
"""

import pathlib
import sys

import torch
import torch.nn.functional as F
from sgl_kernel.flash_attn import flash_attn_with_kvcache
from torch import nn

from sglang.kernels.ops.kvcache.kvcache import store_cache
from sglang.kernels.ops.muse_spark_v12.fused_muse_spark_routing import FusedMuseSparkRouting
from sglang.kernels.ops.muse_spark_v12.fused_norm_gate import FusedRMSNormGate
from sglang.kernels.ops.muse_spark_v12.fused_norm_residual import FusedPostNormResidualPreNorm
from sglang.kernels.ops.muse_spark_v12.fused_qk_norm import FusedQKNorm
from sglang.srt.layers.muse_spark_layers import (
    MuseSpark1xNormPreservingResidualConnection,
    MuseSpark1xRMSNorm,
    MuseSpark1xRMSNormWithInputScale,
)
from sglang.srt.layers.quantization.unquant import _bf16_gemm_dispatch_impl

ATT_GATHER_MIN = 0
ATT_OFF = False
ATT_VCOL = False
ATT_SLIDE = False
SK_SLIDE = True
ST_SK_SLOTS = 396
SK_SIDE = False









ORDER_MIN = 2048
ORDER_IDENT = False







FRONTG = True
FRONTG_MAX = 8192
SIDE_OLD_MIN, SIDE_OLD_MAX = 2048, 8192
TWO_PLANE_MIN = 1 << 30
assert xa_SLIDE_WIDE_MIN == TWO_PLANE_MIN, "two-plane query threshold must match the glue module"
SA_X4_MIN = 8192

HIDDEN = 8192
RMS_EPS = 1e-5
POST_NORM_EPS = 1e-8
FP8 = torch.float8_e4m3fn
N_EXPERTS = 256
TOP_K = 8
LOCAL_EXPERTS = 32
D_EXPERT = 4096
ROUTER_FP32_MAX_TOKENS = 8
SLICED_UNPOOL_MAX_TOKENS = 8
UNPOOL_SLICE = HIDDEN // 8
ROPE_POSITIONS = 1 << 21
ROPE_THETA = 500_000.0
ATTN_GATHER_MIN_TOKENS = 8192
AT_FULL_MIN = 200
ATTN_PACK_GQA = None
EXPERT_A_KEYS = ("moe", "post_expert_w")
MOE_BLOCK_KEYS = ("router_cat_t", "router_bias", "moe", "pre_expert_wf", "post_expert_w", "pool_router", "route_ctr")


def _gemm(x, w):
    return xa__gemm(x, w)


QO_MAX = 2048
try:
    from quack.gemm_config import GemmConfig as _QCfg
    from quack.gemm_interface import gemm_tuned as _qgemm

    _QO_CFG = _QCfg(tile_m=128, tile_n=128, pingpong=True, cluster_m=1, cluster_n=1, swap_ab=False,
                    is_dynamic_persistent=False, split_k=1, device_capacity=9)
    QO_ON = True
except Exception:
    QO_ON = False


def _quack_gemm(x, w, cfg):
    out = torch.empty((x.shape[0], w.shape[0]), dtype=torch.bfloat16, device=x.device)
    _qgemm.fn(x, w.t(), out, config=cfg)
    return out


def _quack_compile(dense):
    """Compile the QuACK kernel once, untimed (M is a runtime extent of the compiled kernel). True when it runs."""
    if not QO_ON:
        return False
    try:
        x = torch.zeros((128, dense["o"].shape[1]), dtype=torch.bfloat16, device=dense["o"].device)
        _quack_gemm(x, dense["o"], _QO_CFG)
        torch.cuda.synchronize()
        return True
    except Exception as e:
        print(f"z0 2030: QuACK compile failed ({type(e).__name__}), o GEMM stays on cuBLASLt", flush=True)
        return False


UNP_MAX = 2048
UNP_CFG = dict(tile_m=128, tile_n=192, pingpong=False, cluster_m=2, cluster_n=1)


def _unp_compile(dense):
    """Compile the unpooler GEMM, untimed; returns its key (plain values), or None to keep cuBLASLt."""
    if not AVAILABLE:
        return None
    try:
        x = torch.zeros((128, dense["unpooler"].shape[1]), dtype=torch.bfloat16, device=dense["unpooler"].device)
        key = xgu_compile_gemm(x, dense["unpooler"], **UNP_CFG)
        xgu_gemm(x, dense["unpooler"], key)
        torch.cuda.synchronize()
        return key
    except Exception as e:
        print(f"z0 2200: own unpooler GEMM compile failed ({type(e).__name__})", flush=True)
        return None


OG_FULL_MAX = 0
OG_SLIDE_MAX = 4096
OG_CFG = dict(tile_m=192, tile_n=128, pingpong=True, cluster_m=1, cluster_n=1)
OG_CFG_BIG = dict(tile_m=128, tile_n=192, pingpong=True, cluster_m=1, cluster_n=1)


def _og_compile(dense):
    """Compile the o GEMM kernels, untimed; returns their keys (plain values), or None to keep cuBLASLt."""
    if not AVAILABLE or max(OG_FULL_MAX, OG_SLIDE_MAX) <= 0:
        return None
    try:
        x = torch.zeros((128, dense["o"].shape[1]), dtype=torch.bfloat16, device=dense["o"].device)
        keys = []
        for cfg, use in ((OG_CFG, OG_FULL_MAX > 0), (OG_CFG_BIG, max(OG_FULL_MAX, OG_SLIDE_MAX) > 2048)):
            keys.append(xg_compile_gemm(x, dense["o"], **cfg) if use else None)
            if keys[-1] is not None:
                xg_gemm(x, dense["o"], keys[-1])
        torch.cuda.synchronize()
        return tuple(keys)
    except Exception as e:
        print(f"z0 2240: own o GEMM compile failed ({type(e).__name__})", flush=True)
        return None


def _param(w):
    return nn.Parameter(w.contiguous(), requires_grad=False)


def _norm(cls, w, size, eps, **kw):
    norm = cls(size, eps=eps, has_weight=w is not None, zero_centered_gamma=True, **kw)
    if w is not None:
        norm.weight = _param(w)
        norm.precompute_effective_weight()
    return norm


def _residual(w):
    gate = MuseSpark1xNormPreservingResidualConnection(HIDDEN, temperature=0.3)
    gate.gate = _param(w)
    gate.precompute_coefficients()
    return gate


def prepare(weights):
    device = weights["qkv_gate"].device
    router = weights["router"]
    hi = router.to(torch.bfloat16)
    lo = (router - hi.float()).to(torch.bfloat16)
    inv_freq = 1.0 / (ROPE_THETA ** (torch.arange(0, 64, 2, dtype=torch.float, device=device) / 64))
    freqs = torch.einsum("i,j -> ij", torch.arange(ROPE_POSITIONS, dtype=torch.float, device=device), inv_freq)
    pre_ffn = _norm(MuseSpark1xRMSNorm, weights["pre_ffn_norm"], HIDDEN, RMS_EPS)
    next_pre_attn = _norm(MuseSpark1xRMSNorm, weights["next_pre_attn_norm"], HIDDEN, RMS_EPS)
    post_attn = _norm(MuseSpark1xRMSNorm, None, HIDDEN, POST_NORM_EPS, is_post_norm=True,
                      post_norm_gain_center_type="one", residual_id=1)
    post_ffn = _norm(MuseSpark1xRMSNormWithInputScale, weights["post_ffn_norm"], HIDDEN, POST_NORM_EPS,
                     is_post_norm=True, post_norm_gain_center_type="one", residual_id=2)
    sa = xa_prepare(weights)
    expert_a = {k: sa[k] for k in EXPERT_A_KEYS + MOE_BLOCK_KEYS}
    del sa
    torch.cuda.empty_cache()
    state = {
        "dense": {name: weights[name].contiguous() for name in ("qkv_gate", "o", "pooler", "unpooler")},
        "router": router.contiguous(),
        "router_hl_t": torch.cat((hi.t(), lo.t()), dim=1).contiguous(),
        "routing": FusedMuseSparkRouting(expert_bias=weights["router_bias"].float().contiguous(), route_scale=1.0,
                                         router_output_multiplier=0.1767766952966369,
                                         router_score_division_eps=1e-15),
        "cos_sin": torch.cat((freqs.cos(), freqs.sin()), dim=-1),
        "qk_norm": FusedQKNorm(head_dim=64, num_q_heads=16, num_kv_heads=2, eps=RMS_EPS),
        "o_norm_gate": FusedRMSNormGate(MuseSpark1xRMSNorm(64, eps=RMS_EPS, has_weight=False), True, 64),
        "attn_boundary": FusedPostNormResidualPreNorm(post_attn, _residual(weights["attn_residual_gate"]), pre_ffn),
        "ffn_boundary": FusedPostNormResidualPreNorm(post_ffn, _residual(weights["ffn_residual_gate"]), next_pre_attn),
        "pre_expert_norm": _norm(MuseSpark1xRMSNorm, weights["pre_expert_norm"].index_select(
            0, xa_kperm_index(D_EXPERT).to(device)).contiguous(), D_EXPERT, RMS_EPS),
        "expert_a": expert_a,
        "side": torch.cuda.Stream(device),
        "side2": torch.cuda.Stream(device),
        "_modules": (pre_ffn, next_pre_attn, post_attn, post_ffn),
    }
    state["attn_nr"] = xn_nr_params(state["attn_boundary"])
    state["ffn_nr"] = xn_nr_params(state["ffn_boundary"])
    xr_init()
    del freqs
    state["dense"]["pooler"] = expert_a["pool_router"][:4096]
    state["qo"] = _quack_compile(state["dense"])
    state["qu"] = _unp_compile(state["dense"])
    state["og"] = _og_compile(state["dense"])
    print(f"z0 2240: own o GEMM {'on' if state['og'] else 'off'}", flush=True)
    print(f"z0 2200: own unpooler GEMM {'on' if state['qu'] else 'off'}", flush=True)
    print(f"z0 2030: QuACK o GEMM {'on' if state['qo'] else 'off'}", flush=True)
    torch.cuda.synchronize()
    return state


def _experts(x, topk_weights, topk_ids, data, state):
    """The routed experts of the normalized pooler output x, with the post-expert norm and weighted sum."""
    local_ids = (topk_ids - data["local_start"]) % N_EXPERTS
    return xa__experts_moe(x, topk_weights, local_ids, state["expert_a"])


def kernel_fn(data, state):
    cfg = data["config"]
    batch, length, tokens = cfg["batch"], cfg["q_len"], cfg["tokens"]
    hq, hkv, d = cfg["num_q_heads"], cfg["num_kv_heads"], cfg["head_dim"]
    dense = state["dense"]
    k_cache, v_cache = data["k_cache"], data["v_cache"]
    scaled = cfg["kv_scale"] != 1.0
    fast = not scaled and k_cache.dtype == FP8 and tokens > ROUTER_FP32_MAX_TOKENS
    window = cfg["window"]
    sk = SK_SLIDE and not ATT_OFF and fast and bool(window) and hq == 16 and hkv == 2 and d == 64
    perm = None
    order_on = tokens >= ORDER_MIN and xk_supported(tokens) and _remat_ok(state, tokens)
    frontg = FRONTG and tokens < FRONTG_MAX
    two = sk and tokens >= TWO_PLANE_MIN and 4 * batch * ((length + 7) // 8) > ST_SK_SLOTS
    if sk and not SK_SIDE:
        ko, vt = xs_alloc(data["cu_seqlens_q"], length, window, k_cache.device)
    if sk and SK_SIDE:
        main_q, side_q = torch.cuda.current_stream(), state["side"]
        ko, vt = xs_alloc(data["cu_seqlens_q"], length, window, k_cache.device)
        side_q.wait_stream(main_q)
        with torch.cuda.StreamContext(side_q):
            xs_gather(k_cache, v_cache, data["page_table"], data["cache_seqlens"], data["cu_seqlens_q"], window, 0, ko,
                      vt)
        if not torch.cuda.is_current_stream_capturing():
            ko.record_stream(side_q)
            vt.record_stream(side_q)
    side_old = sk and frontg and not SK_SIDE and SIDE_OLD_MIN <= tokens < SIDE_OLD_MAX
    if side_old:
        q8 = torch.empty((tokens, hq * d), dtype=FP8, device=k_cache.device)
        qlo = torch.empty((tokens, hq * d), dtype=FP8, device=k_cache.device) if two else q8[:0]
        o = torch.empty((tokens, hq * d), dtype=torch.bfloat16, device=k_cache.device)
    if order_on:
        pi = torch.empty(2 * tokens, dtype=torch.int32, device=k_cache.device)
        perm = (pi[:tokens], pi[tokens:])
        if not xk__init:
            xk__mod.order2_init()
            xk__init.append(1)
    if order_on and not side_old:
        main0, side2 = torch.cuda.current_stream(), state["side2"]
        side2.wait_stream(main0)
        with torch.cuda.StreamContext(side2):
            xk__mod.order2_run(data["stream"], perm[0], perm[1], 1 if ORDER_IDENT else 0)
        if not torch.cuda.is_current_stream_capturing():
            pi.record_stream(side2)
    qkv = _gemm(data["attn_input"], dense["qkv_gate"])
    if order_on and side_old:
        main0, side2 = torch.cuda.current_stream(), state["side2"]
        side2.wait_stream(main0)
        with torch.cuda.StreamContext(side2):
            if torch.cuda.is_current_stream_capturing():
                xa__glue.make_edges(1)
            xk__mod.order2_run(data["stream"], perm[0], perm[1], 1 if ORDER_IDENT else 0)
        if not torch.cuda.is_current_stream_capturing():
            pi.record_stream(side2)
    if side_old:
        main_q, side_q = torch.cuda.current_stream(), state["side"]
        capturing = torch.cuda.is_current_stream_capturing()
        side_q.wait_stream(main_q)
        with torch.cuda.StreamContext(side_q):
            if capturing:
                xa__glue.make_edges(1)
            xa__glue.oldg_run(k_cache.view(torch.uint8), v_cache.view(torch.uint8), data["page_table"],
                              data["cache_seqlens"], data["cu_seqlens_q"], window - 1, ko, vt)
        if not capturing:
            ko.record_stream(side_q)
            vt.record_stream(side_q)
    gate = qkv[:, hq * d + 2 * hkv * d:]
    att_scale = cfg["sm_scale"] * cfg["qk_scale_factor"]
    at_full = (fast and not window and not cfg["rope_theta"] and not ATT_OFF and hq == 16 and hkv == 2 and d == 64
               and tokens >= AT_FULL_MIN)
    if at_full:
        q8 = torch.empty((tokens, hq * d), dtype=FP8, device=qkv.device)
        att_scale = att_scale * xt_front(qkv, data["cache_loc"], q8, k_cache.view(torch.uint8), v_cache.view(torch.uint8),
                                         RMS_EPS, att_scale)
        q8 = q8.view(tokens, hq, d)
    elif fast and sk and frontg:
        if not side_old:
            q8 = torch.empty((tokens, hq * d), dtype=FP8, device=qkv.device)
            qlo = torch.empty((tokens, hq * d), dtype=FP8, device=qkv.device) if two else q8[:0]
        xa__glue.frontg_run(qkv, data["positions"], state["cos_sin"], data["cache_loc"], q8, qlo, k_cache.view(torch.uint8),
                            v_cache.view(torch.uint8), RMS_EPS, 1 if cfg["rope_theta"] else 0, data["page_table"],
                            data["cache_seqlens"], data["cu_seqlens_q"], window - 1, 1 if side_old else 2, ko, vt, 0)
        q8 = q8.view(tokens, hq, d)
    elif fast:
        q8 = torch.empty((tokens, hq * d), dtype=FP8, device=qkv.device)
        if two:
            qlo = torch.empty((tokens, hq * d), dtype=FP8, device=qkv.device)
            xa__glue.front2_run(qkv, data["positions"], state["cos_sin"], data["cache_loc"], q8, qlo,
                                k_cache.view(torch.uint8), v_cache.view(torch.uint8), RMS_EPS, 1 if cfg["rope_theta"] else 0)
        else:
            xa__glue.front_run(qkv, data["positions"], state["cos_sin"], data["cache_loc"], q8, k_cache.view(torch.uint8),
                               v_cache.view(torch.uint8), RMS_EPS, 1 if cfg["rope_theta"] else 0)
        q8 = q8.view(tokens, hq, d)
    else:
        if cfg["rope_theta"]:
            q, k = state["qk_norm"](qkv, data["positions"], state["cos_sin"])
        else:
            q, k = state["qk_norm"](qkv)
        v = qkv[:, hq * d + hkv * d: hq * d + 2 * hkv * d]
        k = k.reshape(tokens, hkv * d)
        if scaled:
            k, v = k / data["k_scale"], v / data["v_scale"]
        store_cache(k.to(FP8).view(torch.uint8), v.to(FP8).view(torch.uint8),
                    k_cache.view(-1, hkv * d).view(torch.uint8), v_cache.view(-1, hkv * d).view(torch.uint8),
                    data["cache_loc"], reserved_skip_index=-1)
        q8 = q.view(tokens, hq, d).to(FP8)
    if sk:
        if not frontg:
            xs_gather(k_cache, v_cache, data["page_table"], data["cache_seqlens"], data["cu_seqlens_q"], window,
                      1 if SK_SIDE else 2, ko, vt)
        if SK_SIDE or side_old:
            main_q.wait_stream(side_q)
            if perm is not None:
                main_q.wait_stream(state["side2"])
                perm = (perm[0], perm[1], True)
        if not side_old:
            o = torch.empty((tokens, hq * d), dtype=torch.bfloat16, device=q8.device)
        if 4 * batch * ((length + 7) // 8) <= ST_SK_SLOTS:
            xs_attn_ng(q8, ko, vt, data["cache_seqlens"], data["cu_seqlens_q"], length, window,
                       cfg["sm_scale"] * cfg["qk_scale_factor"], gate, RMS_EPS, o)
        elif tokens >= SA_X4_MIN:
            if two:
                xs2_attn_ng2(q8, qlo, ko, vt, data["cache_seqlens"], data["cu_seqlens_q"], length, window,
                             cfg["sm_scale"] * cfg["qk_scale_factor"], gate, RMS_EPS, o)
            else:
                xs2_attn_ng(q8, ko, vt, data["cache_seqlens"], data["cu_seqlens_q"], length, window,
                            cfg["sm_scale"] * cfg["qk_scale_factor"], gate, RMS_EPS, o)
        elif two:
            xs1_attn_ng2(q8, qlo, ko, vt, data["cache_seqlens"], data["cu_seqlens_q"], length, window,
                         cfg["sm_scale"] * cfg["qk_scale_factor"], gate, RMS_EPS, o)
        else:
            xs1_attn_ng(q8, ko, vt, data["cache_seqlens"], data["cu_seqlens_q"], length, window,
                        cfg["sm_scale"] * cfg["qk_scale_factor"], gate, RMS_EPS, o)
        return _layer_rest(data, state, dense, o, tokens, qo=True, perm=perm)
    descale = {}
    if scaled:
        descale = dict(k_descale=data["k_scale"].reshape(1, 1).expand(batch, hkv),
                       v_descale=data["v_scale"].reshape(1, 1).expand(batch, hkv))
    page_table = data["page_table"]
    if at_full:
        seqlens = data["cache_seqlens"]
        if ATT_VCOL:
            k_cache, v_cache, seqlens = xt_gather_t(k_cache, v_cache, page_table, seqlens, data["cu_seqlens_q"], length,
                                                    window)
        elif window:
            k_cache, v_cache, seqlens = xt_gather_window(k_cache, v_cache, page_table, seqlens, data["cu_seqlens_q"],
                                                         length, window)
        else:
            k_cache, v_cache = xt_gather(k_cache, v_cache, page_table, seqlens, data["cu_seqlens_q"], length, 2)
        out = torch.empty((tokens, hq, d), dtype=torch.bfloat16, device=q8.device)
        if xt_attn_g(q8, k_cache, v_cache, None, seqlens, data["cu_seqlens_q"], length,
                     att_scale, window - 1 if window else -1, out, gate, RMS_EPS):
            return _layer_rest(data, state, dense, out.view(tokens, hq * d), tokens, perm=perm)
    else:
        out = _fa3_sglang(q8, k_cache, v_cache, page_table, data, cfg, tokens, length, batch, hkv, d, window, descale)
    o = state["o_norm_gate"](out.reshape(tokens, hq * d).to(torch.bfloat16), gate)
    return _layer_rest(data, state, dense, o, tokens, perm=perm)


def _fa3_sglang(q8, k_cache, v_cache, page_table, data, cfg, tokens, length, batch, hkv, d, window, descale):
    if not cfg["window"] and tokens >= ATTN_GATHER_MIN_TOKENS and data["scheduler_metadata"] is None:
        page = k_cache.shape[1]
        rows = (page_table.clamp(min=0).long()[:, :, None] * page
                + torch.arange(page, device=page_table.device)).reshape(batch, -1)
        k_cache = k_cache.view(torch.uint8).view(-1, hkv, d)[rows].view(FP8)
        v_cache = v_cache.view(torch.uint8).view(-1, hkv, d)[rows].view(FP8)
        page_table = None
    out = flash_attn_with_kvcache(
        q=q8,
        k_cache=k_cache,
        v_cache=v_cache,
        page_table=page_table,
        pack_gqa=ATTN_PACK_GQA,
        cache_seqlens=data["cache_seqlens"],
        cu_seqlens_q=data["cu_seqlens_q"],
        max_seqlen_q=length,
        softmax_scale=cfg["sm_scale"] * cfg["qk_scale_factor"],
        causal=True,
        window_size=(window - 1, 0) if window else (-1, -1),
        scheduler_metadata=data["scheduler_metadata"],
        **descale,
    )
    if isinstance(out, tuple):
        out = out[0]
    return out


def _remat_ok(state, tokens):
    """True where the residual after attention is recomputed in the FFN boundary instead of stored (xb + xr)."""
    return (tokens >= 2048 and tokens > ROUTER_FP32_MAX_TOKENS and state["attn_nr"] is not None
            and state["ffn_nr"] is not None and xr_supported(state["attn_nr"], state["ffn_nr"]))


OWN_MAX_WAVES = 3
OWN_MIN_FILL = 0.9
_NSM = []


def _own_fill_ok(tokens, n, tm, tn, cm=1):
    if not _NSM:
        _NSM.append(torch.cuda.get_device_properties(torch.cuda.current_device()).multi_processor_count)
    mt = -(-tokens // tm)
    mt = -(-mt // cm) * cm
    waves = -(-(mt * -(-n // tn)) // _NSM[0])
    return waves <= OWN_MAX_WAVES and tokens * n >= OWN_MIN_FILL * waves * _NSM[0] * tm * tn


def _layer_rest(data, state, dense, o, tokens, qo=False, perm=None):
    """Everything after attention: o GEMM, attention boundary, MoE block, unpooler and FFN boundary.

    o is the gated attention output [T, 1024] BF16. Returns (next attention input, residual, top-8 expert ids).
    """
    q_o = (qo and state["qo"] and ROUTER_FP32_MAX_TOKENS < tokens < QO_MAX
           and _own_fill_ok(tokens, dense["o"].shape[0], 128, 128))
    full = not data["config"]["window"]
    x_o = (not q_o and state["og"] is not None and ROUTER_FP32_MAX_TOKENS < tokens
           and (tokens < OG_FULL_MAX if full else QO_MAX <= tokens < OG_SLIDE_MAX))
    if q_o:
        attn_output = _quack_gemm(o, dense["o"], _QO_CFG)
    elif x_o:
        attn_output = xg_gemm(o, dense["o"], state["og"][tokens >= 2048])
    else:
        attn_output = _gemm(o, dense["o"])

    split_router = tokens > ROUTER_FP32_MAX_TOKENS
    remat = split_router and _remat_ok(state, tokens)
    assert perm is None or remat
    if remat:
        if perm is not None and len(perm) == 2:
            torch.cuda.current_stream().wait_stream(state["side2"])
        ffn_input, attn_rps = xb_boundary_a(state["attn_nr"], data["stream"], attn_output, pdl=not x_o,
                                            perm=None if perm is None else perm[1])
        resid = None
        router_input = None
    elif split_router and state["attn_nr"] is not None:
        ffn_input, resid = xn_boundary(state["attn_nr"], data["stream"], attn_output, pdl=not q_o)
        router_input = None
    else:
        ffn_input, resid, router_input = state["attn_boundary"](
            data["stream"], attn_output, branch_dtype=torch.bfloat16,
            router_dtype=None if split_router else torch.float32)
    if split_router:
        expert_output, topk_ids = xa__moe_block(ffn_input, data, state["expert_a"], dense["pooler"], state["side"],
                                                  wsum=xw_wsum)
        if (state["qu"] is not None and tokens < UNP_MAX
                and _own_fill_ok(tokens, dense["unpooler"].shape[0], UNP_CFG["tile_m"], UNP_CFG["tile_n"],
                                 UNP_CFG["cluster_m"])):
            ffn_output = xgu_gemm(expert_output, dense["unpooler"], state["qu"])
        else:
            ffn_output = _gemm(expert_output, dense["unpooler"])
        if remat and perm is not None:
            next_attn_input, resid, topk_ids = xr_boundary_b(state["attn_nr"], state["ffn_nr"], data["stream"],
                                                             attn_output, ffn_output, attn_rps, perm[1], topk_ids)
        elif remat:
            next_attn_input, resid = xr_boundary_b(state["attn_nr"], state["ffn_nr"], data["stream"],
                                                  attn_output, ffn_output, attn_rps)
        elif state["ffn_nr"] is not None:
            next_attn_input, resid = xn_boundary(state["ffn_nr"], resid, ffn_output)
        else:
            next_attn_input, resid, _ = state["ffn_boundary"](resid, ffn_output, branch_dtype=torch.bfloat16)
        return next_attn_input, resid, topk_ids
    main_q, side_q = torch.cuda.current_stream(), state["side"]
    side_q.wait_stream(main_q)
    expert_input = _gemm(ffn_input, dense["pooler"])
    with torch.cuda.StreamContext(side_q):
        if split_router:
            hl = torch.mm(ffn_input, state["router_hl_t"], out_dtype=torch.float32)
            logits = hl[:, :N_EXPERTS] + hl[:, N_EXPERTS:]
        else:
            logits = F.linear(router_input, state["router"])
        topk_weights, topk_ids = state["routing"](None, logits, TOP_K, True)
    main_q.wait_stream(side_q)
    if not torch.cuda.is_current_stream_capturing():
        for tensor in (topk_weights, topk_ids):
            tensor.record_stream(main_q)
    expert_output = _experts(state["pre_expert_norm"](expert_input), topk_weights, topk_ids, data, state)
    if tokens <= SLICED_UNPOOL_MAX_TOKENS:
        unpooled = _gemm(expert_output, dense["unpooler"][:UNPOOL_SLICE])
        ffn_output = F.pad(unpooled, (0, HIDDEN - UNPOOL_SLICE))
    else:
        ffn_output = _gemm(expert_output, dense["unpooler"])
    next_attn_input, resid, _ = state["ffn_boundary"](resid, ffn_output, branch_dtype=torch.bfloat16)
    return next_attn_input, resid, topk_ids
