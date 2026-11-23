// The post-expert norm weighted sum at 32+ tokens (moe_block.py's _moe_tail): out[t] = sum over the token's live
// slots j of score_j * rmsnorm(y_j * w), the per-route math of SGLang's fused_post_expert_norm_weighted_sum. A
// producer warp streams each token's live expert rows into shared memory by bulk copy through a ring of WSC_NS stages,
// at most WSC_RPS rows per token. The x73_ prefix of the two helpers below is a naming tag of the submitted code.
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

__device__ __forceinline__ bool x73_elect() {  // elect.sync over the full (converged) warp: true in one lane
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

// One CTA of NT consumer threads plus one producer warp, persistent over tokens tok = blockIdx.x + i * gridDim.x.
// Consumer tid owns columns [8 tid, 8 tid + 8) and [2048 + 8 tid, ...). The producer issues the next tokens' live rows
// (the first RPS live slots; a token's later live slots are read from global by the consumers, which happens for
// about 1.5% of tokens at 16384 tokens).
__global__ void __launch_bounds__(NT + 32, 2) wsc_k(const __nv_bfloat16* __restrict__ x, const __nv_bfloat16* __restrict__ w,
    const float* __restrict__ sc, __nv_bfloat16* __restrict__ out, const int* __restrict__ ids, int T, float eps) {
  extern __shared__ __align__(128) uint8_t wraw[];
  // One array: full barriers [0, NS) then empty barriers [NS, 2 NS), both addressed from fb. A separate empty array
  // that is only reached through fb + 8 * NS is never named, so the compiler may drop it and alias s_live there.
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
  // issue(i): stage i % NS gets token i's live list, scores and the bytes of its first RPS live rows.
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
  if (tid >= NT) {  // producer warp: lanes load ids / scores of 32 tokens at once, one elected lane issues their rows in order
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
        if (x73_elect()) issue(i, a2, b2, c2, d2);  // one elected lane of the converged warp
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
  // Shared addresses once per thread, opaque so ptxas keeps them in registers instead of rebuilding them
  // from SR_CgaCtaId for every row.
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
    asm volatile("bar.sync 1, %0;" ::"n"(NT) : "memory");  // all consumers done with stage s and s_red[i & 1]
    if (tid == 0) asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" ::"r"(fb + 8 * NS + 8 * s) : "memory");
  }
}
}  // namespace

void wsc3_run(torch::Tensor x, torch::Tensor w, torch::Tensor s, torch::Tensor out, torch::Tensor ids, int64_t T,
             double eps, int64_t ctas) {
  const int smem = NS * STB + 128;
  static bool once = false;
  if (!once) { cudaFuncSetAttribute(wsc_k, cudaFuncAttributeMaxDynamicSharedMemorySize, smem); once = true; }
  wsc_k<<<(int)ctas, NT + 32, smem, at::cuda::getCurrentCUDAStream()>>>(
      (const __nv_bfloat16*)x.data_ptr(), (const __nv_bfloat16*)w.data_ptr(), s.data_ptr<float>(),
      (__nv_bfloat16*)out.data_ptr(), ids.data_ptr<int>(), (int)T, (float)eps);
}
