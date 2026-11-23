// nr2_k: one layer boundary as one kernel: RMS post-norm of the branch, the gated residual update
// s' = alpha * s + beta * postnorm(branch), and the RMS pre-norm of s' (the next block's BF16 input).
// Every global access is coalesced: thread j owns columns 4j + 2048g, g = 0..3 (FP32 rows as float4, BF16 rows as
// 8-byte vectors), so each 32 B sector is touched by one request.
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
// One CTA of 512 threads per row of 8192. SCALE: post-norm with input scale (x * w rounded to BF16, then normalized);
// else plain post-norm, times the post weight when POSTW. COPY (mode 9): the same traffic with no math,
// a bandwidth-only timing variant.
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
    const float4 v = __ldlu(s4 + 512 * g + tid);  // last use: the residual is not read again
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
    if (!SCALE) {  // the attention boundary: its new residual is stored evict_last so it stays in L2 for the FFN boundary
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
}  // namespace

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
