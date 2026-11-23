// order2_k: token similarity order (perm, inv) from the first KC columns of the FP32 stream, in one CTA of
// 1024 threads (see token_order.py for the method). NIT subspace iterations, NB = 4096 key bins (12-bit keys).
#include <cuda_runtime.h>
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <stdint.h>

namespace {
constexpr int KC = 32, KS = 128, NB = 4096, XL = KC + 1, TMAX = 16384, NT = 1024, NIT = 3;

__global__ void __launch_bounds__(NT, 1) order2_k(const float* __restrict__ x, long ld, int T, int* __restrict__ perm,
                                                 int* __restrict__ inv, int ident) {
  extern __shared__ __align__(16) int dsm[];
  int* hist = dsm;                                            // NB counts, then offsets / cursors
  uint16_t* keys = reinterpret_cast<uint16_t*>(dsm + NB);     // T keys
  __shared__ float xs[KS * XL];                               // centred samples
  __shared__ float C[KC * XL];
  __shared__ float red[32][KC + 1];
  __shared__ float mu[KC];
  __shared__ float V[KC][4], W[KC][4];
  __shared__ __align__(16) float4 dir[KC];                    // v_j of column c, j = x, y, z, w
  __shared__ float bias[4], isd[4];
  __shared__ int ws[32];
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  for (int i = tid; i < NB; i += NT) hist[i] = 0;
  // samples: row (i * T) / S, 8 threads per row (one float4 each)
  const int S = T < KS ? T : KS;
  for (int i = tid >> 3; i < S; i += NT / 8) {
    const long r = ((long)i * T) / S;
    const float4 v = reinterpret_cast<const float4*>(x + r * ld)[tid & 7];
    float* d = xs + i * XL + 4 * (tid & 7);
    d[0] = v.x; d[1] = v.y; d[2] = v.z; d[3] = v.w;
  }
  __syncthreads();
  {  // column means: 32 parts x 32 columns, then a fixed-order sum
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
  if (tid < 256) {  // covariance, 2 x 2 tiles
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
  if (tid < KC * 4) {  // fixed start
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
      if (it == NIT) {  // the variance of each coordinate: v^T C v
#pragma unroll
        for (int j = 0; j < 4; ++j) {
          float d = V[lane][j] * W[lane][j];
#pragma unroll
          for (int o = 16; o > 0; o >>= 1) d += __shfl_xor_sync(0xffffffffu, d, o);
          if (lane == 0) isd[j] = rsqrtf(fmaxf(d, 1e-30f));
        }
      } else {  // Gram-Schmidt on W's columns, one row per lane
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
  if (warp == 1) {  // bias_j = mu . v_j
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      float d = mu[lane] * V[lane][j];
#pragma unroll
      for (int o = 16; o > 0; o >>= 1) d += __shfl_xor_sync(0xffffffffu, d, o);
      if (lane == 0) bias[j] = d;
    }
  }
  __syncthreads();
  // keys: one token per thread, the directions as broadcasts
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
  // exclusive scan of the counts (4 per thread), in place
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
}  // namespace

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
