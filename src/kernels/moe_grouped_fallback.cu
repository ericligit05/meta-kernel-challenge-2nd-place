// Helpers for moe_block.py's _experts_grouped (dequantize the experts to BF16, then torch._grouped_mm). That path
// belongs to moe_block.py's own kernel_fn, which layer.py does not call.
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
__constant__ float kE2M1[8] = {0.f, 0.5f, 1.f, 1.5f, 2.f, 3.f, 4.f, 6.f};
// One thread per 16-value block: 8 packed bytes and one E4M3 scale -> 16 BF16 values.
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
// Rows of x for the sorted routes, only the first n_local = offs[31] (read on the device); 8 BF16 per copy.
__global__ void gather_k(const uint4* __restrict__ x, const long* __restrict__ order, const int* __restrict__ offs,
                         uint4* __restrict__ a, int K8) {
  const int r = blockIdx.x;
  if (r >= offs[31]) return;
  const long src = order[r] / 8;
  for (int c = threadIdx.x; c < K8; c += blockDim.x) a[(long)r * K8 + c] = x[src * K8 + c];
}
// inter = silu(g * gs) * (u * gs) for the first n_local rows of h [rows, 2I].
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
// y[slot] = y_sorted[inv[slot]] * gs2 for local slots, zeros elsewhere.
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
