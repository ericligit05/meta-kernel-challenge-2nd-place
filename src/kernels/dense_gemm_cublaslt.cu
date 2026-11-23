// Dense BF16 GEMMs (x @ w.T) through cuBLASLt with no workspace: lt_gemm picks a heuristic algorithm, optionally the
// one with a preferred tile / cluster shape / custom option; lt_gemm_like runs a GEMM on fewer rows with exactly the
// algorithm of the full-size call (moe_block.py's split pooler GEMM).
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
// Column-major view: out^T [N, M] = (w^T stored [K, N])^T * x^T [K, M]. Plans are cached per argument tuple: ai picks
// a heuristic index (else 0), smc > 0 sets the SM count target.
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
    // The heuristic result with this tile id and cluster shape id (measured faster at the turn shapes), else
    // algorithm ai. Same cuBLAS kernel family, whole K in one CTA: bit-identical outputs. With cu >= 0 the custom
    // option must match too; if only tile and cluster match, that result with the custom option set is used when
    // cublasLtMatmulAlgoCheck accepts it with no workspace.
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
}  // namespace

// The raw words of a plan's algorithm. They identify the nvjet kernel and hold no M field; for nvjet entries every
// config attribute reads 0, so these words are the only reliable way to tell two entries apart.
std::vector<uint64_t> algo_cfg(const cublasLtMatmulAlgo_t* a) {
  return std::vector<uint64_t>(a->data, a->data + 8);
}
// A plan for M rows whose algorithm has exactly the configuration of the plan for Mfull rows (the same nvjet
// kernel, whole K in one CTA, so every output row is the full call's row). Returns nullptr if no entry matches.
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
