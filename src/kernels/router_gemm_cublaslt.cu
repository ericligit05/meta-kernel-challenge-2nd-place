// FP32-output cuBLASLt GEMMs for the router (moe_block.py): lt_router runs the router as S K-slices in one strided
// batched call (partial sums [S, T, 512], summed by the route kernel), lt_f32 runs one FP32-output GEMM (the pooler and
// router weights stacked, or the router rows alone). No workspace in either.
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
// Column-major: part_s^T [N, M] = (wT[K-slice s] viewed [N, Kc], ld N) * (x[:, K-slice s] viewed [Kc, M], ld K).
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
}  // namespace

// tile: a CUBLASLT_MATMUL_TILE id; the first heuristic algorithm with that tile is used, else the heuristic's first.
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

// Diagnostics (not called by the layer): per heuristic algorithm (tile id, split-K count, reduction scheme, CTA swizzle).
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

// Pooler and router as one GEMM: out [M, N] FP32 = x [M, K] @ w [N, K]^T (w = [pooler | router hi | router lo] rows),
// heuristic algorithm ai of the no-workspace list, or the forced configuration below.
namespace {
struct CPlan {
  cublasLtMatmulDesc_t op;
  cublasLtMatrixLayout_t a, b, c;
  cublasLtMatmulAlgo_t algo[16];
  int found;
  cublasLtMatmulAlgo_t forced;  // heuristic result 0 with tile / cluster / custom option set, if AlgoCheck accepts it
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
}  // namespace

void lt_f32(torch::Tensor x, torch::Tensor w, torch::Tensor out, int64_t ai, int64_t tile, int64_t cl, int64_t cu) {
  const int64_t M = x.size(0), K = x.size(1), N = w.size(0);
  CPlan* p = cplan(M, N, K, tile, cl, cu);
  const int i = ai < p->found ? (int)ai : 0;
  const float alpha = 1.f, beta = 0.f;
  auto r = cublasLtMatmul(rt_handle(), p->op, &alpha, w.data_ptr(), p->a, x.data_ptr(), p->b, &beta, out.data_ptr(),
                          p->c, out.data_ptr(), p->c, (p->has_forced ? &p->forced : &p->algo[i]), nullptr, 0, at::cuda::getCurrentCUDAStream());
  TORCH_CHECK(r == CUBLAS_STATUS_SUCCESS, "cublasLtMatmul (pooler + router) failed");
}
