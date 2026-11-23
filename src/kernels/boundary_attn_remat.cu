// nr3_k: the two layer boundaries without writing the mid-layer FP32 residual. The attention boundary (B false)
// writes only the FFN input and each row's post-norm scale; the FFN boundary (B true) recomputes the attention
// residual from the layer's input stream and the attention branch (the same float operations in the same order, so
// the values are bit-identical), then applies the FFN boundary. This saves 32 KB per token of FP32 residual written
// and read back, for 16 KB of BF16 attention branch read again.
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
  const __nv_bfloat16* postw;  // nullptr: no post weight
  const __nv_bfloat16* prew;
  const float* alpha;
  const float* beta;
  float post_eps, pre_eps;
  int scale;                   // post-norm with input scale (boundary_norm.cu mode 1)
};
// x (the residual, 16 values per thread) <- alpha x + beta postnorm(branch row); the same operations as the first
// half of boundary_norm.cu's nr2_k. rp_in >= 0: use it as the post-norm row scale instead of a row sum.
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
// out <- prenorm(x) as BF16 (the second half of nr2_k)
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

// B == false: attention boundary, FFN input only (branch row prefetched into registers).
// B == true: FFN boundary, recomputing the attention residual first; both branch rows are staged in shared memory
// by cp.async at the start (32 KB per CTA), so the kernel keeps nr2_k's register budget (2 CTAs x 512 threads).
// With a map (B false only), CTA t reads token t's rows in place and writes its outputs to row perm[t].
template <bool B>
__global__ void __launch_bounds__(512, 2) nr3_k(const float* __restrict__ st, const __nv_bfloat16* __restrict__ bra,
    const __nv_bfloat16* __restrict__ brb, __nv_bfloat16* __restrict__ out, float* __restrict__ nst, float* __restrict__ rps, NrP pa, NrP pb,
    const int* __restrict__ perm) {  // perm: the inverse map inv (token -> row) here
  __shared__ float red[4][16];
  __shared__ __align__(16) uint2 sb[B ? 2 : 1][B ? 2048 : 1];
  // The row map comes from token_order.cu on the side queue, which the caller joins with a full stream wait (not a
  // PDL edge), so it is safe to read before griddepcontrol.wait; only the store address depends on it.
  const size_t src = blockIdx.x;
  const int dst_i = perm ? __ldg(perm + src) : (int)src;
  const size_t row = (size_t)dst_i;  // the row this token's outputs go to (B: never with a map)
  asm volatile("griddepcontrol.wait;" ::: "memory");
  const int tid = threadIdx.x;
  const float4* s4 = reinterpret_cast<const float4*>(st + src * 8192);
  R4 ra;
  if (B) {
    const char* ga = reinterpret_cast<const char*>(bra + src * 8192);
    const char* gb = reinterpret_cast<const char*>(brb + row * 8192);
#pragma unroll
    for (int r = 0; r < 2; ++r) {
      const int c = tid + 512 * r;  // 16-byte chunk of the row
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
  if (!B && tid == 0) rps[row] = rpa;  // the attention branch's post-norm scale, for the FFN kernel
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
}  // namespace

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
