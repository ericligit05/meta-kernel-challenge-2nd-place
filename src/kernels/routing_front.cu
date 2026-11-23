// The "_glue" extension of moe_block.py and layer.py: routing, the pre-expert norm and gather, the expert plan, and
// the attention front (QK-norm, RoPE, FP8 q and KV cache store, sliding-window gather).
//
// Routing writes per route (t * 8 + k): tw (weight), tid (expert id), slot (local expert 0..31 or -1) and the route
// list (order[row] = t * 8 + k, a[row] = the token's normed pooler row) grouped by local expert. The expert plan goes
// to meta (int32), which the expert kernel (moe_expert_gemm.cu) and weighted_sum.cu read:
//   meta[0..31]     route count per local expert
//   meta[32 + e], meta[132 + e]   first route-list row of expert e
//   meta[64 + p], meta[96]        GEMM1 tile offset of plan position p, total GEMM1 tiles
//   meta[97 + p], meta[129]       GEMM2 tile offset (in GEMM2 claim order), total GEMM2 tiles
//   meta[130], meta[131]          tile claim counters (zeroed)
//   meta[164 + p]   expert at plan position p (most routes first, ties by lower id)
// Mixed plans (plan2_* below; the expert kernel runs mixed tiles below 8192 tokens) also write:
//   meta[196 + p]   plan entry: ntt | mb << 8 | tile size << 12 | r << 21 (r: tokens of a trailing MB 2 tile)
//   meta[228 + p]   first token-tile index of position p (bits 0..15) | position of GEMM2 claim rank p (bits 16..)
//   meta[260 ...]   the expert kernel's per-token-tile cnt1 words (zeroed here)
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
constexpr int RW = 16;  // warps per route CTA (one token per warp at a time)

// Plan with uniform token tiles of tn (route_k at tn 256), one warp: counts in meta[0..31] -> list starts, expert
// order by count, tile offsets in that order, zeroed claim counters.
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

// Cost model of the expert kernel's mixed layouts (below 8192 tokens): MMA width of a tile (gl_mx_w, gl_tf_w1) and
// its modelled cycles per chunk (gl_mx_cost). gl_tf_w1 also sets the MB 1 width for px_c1 below; gl_mx_w and
// gl_mx_cost serve only plan2_warp, which no kernel here calls.
#define TF_WSET 15  // MB 1 widths enabled: bit 1 W 144, 2 W 176, 4 W 208, 8 W 240; bit 16 (MB 2 W 96) off. As moe_expert_gemm.cu.
__host__ __device__ __forceinline__ int gl_tf_w1(int nv) {  // MB 1 MMA width: steps of 16 where TF_WSET has the width
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
// Token tile costs from measured per-tile times (units of 0.01 us); an MB 1 tile counts as two 128-row blocks.
__device__ __forceinline__ int px_c1(int nv) {
  const int W = gl_tf_w1(nv);
  return 2 * (W <= 128 ? 2985 : W <= 144 ? TF_C144 : W <= 160 ? 3236 : W <= 176 ? TF_C176 : W <= 192 ? 3492
              : W <= 208 ? TF_C208 : W <= 224 ? 3700 : W <= 240 ? TF_C240 : 4188);
}
__device__ __forceinline__ int px_c2(int nv) { return nv <= 8 ? 1793 : nv <= 16 ? 2147 : nv <= 32 ? 2482 : nv <= 64 ? 3236 : (nv <= 96 && (TF_WSET & 16)) ? TF_C96 : 4714; }
// The plan entry for an expert with n routes, the cheapest by px_c1 / px_c2 of: pure MB 2, pure MB 1 (even splits),
// or MB 1 tiles plus one MB 2 tile of the last r tokens.
// x / k for 0 <= x < 2^13, 1 <= k <= 32: the fraction of x / k is 0 or >= 1 / k, so + 0.5 / k never carries.
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
__device__ int g_ptab[4096];  // px_plan(n) for every n < 4096 (filled once by px_init)
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

// The mixed plan for one warp (lane e = expert e, count n in a register), plan entries from px_get, every read-back
// from shared memory (no global round trips). Writes the meta words and returns the expert's route-list start.
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
  // GEMM2 claim order by the first token tile's time, longest first (LPT); narrow experts (MB 2, W <= 64) go last,
  // as in plan order (their GEMM1 tiles are claimed last); cnt[] is free here and holds rank -> position.
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

// Per token (one warp, 8 experts per lane): logits = sum over the S K-slice partials of hi + lo, s = sigmoid(mult *
// logit), top 8 of s + bias (ties to the lower id, as the reference's argmax), weights s / (sum + eps). Local routes
// get a rank inside this CTA's expert count; the last CTA to finish turns the per-CTA counts into the route list's
// starts and plan. Used with the separate router GEMM (moe_block.py route_run).
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
      // s + bias as an unsigned key in float order (no NaNs here), so a warp max is one redux.sync.
      const unsigned u = __float_as_uint(s[j] + b[j]);
      key[j] = (u & 0x80000000u) ? ~u : (u | 0x80000000u);
    }
    unsigned used = 0;
    float wsum = 0.f, my_s = 0.f, my_b = 0.f, v8 = 0.f;
    int my_id = 0;
#pragma unroll
    for (int k = 0; k < 8; k++) {
      // This lane's best unused (ties to the lower id), then the warp's max key and the lowest id holding it.
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
  // The last CTA, all warps: warp w sums the counts of CTAs [g0, g1) (independent loads, no serial chain), warp 0
  // totals and plans, then each warp writes its CTAs' starts.
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
      run = plan2_s(meta, tot, pc, pinf, pord);  // the expert kernel's mixed layouts
    }
    for (int w = 0; w < RW; w++) { const int x = part[w][lane]; part[w][lane] = run; run += x; }
  }
  __syncthreads();
  int run = part[warp][lane];
#pragma unroll
  for (int j = 0; j < 9; j++)
    if (g0 + j < g1) { base_tab[(g0 + j) * 32 + lane] = run; run += v[j]; }
}



// One CTA (256 threads) per token: the pre-expert RMSNorm of its pooler row (FP32 statistics, (1 + w) in FP32,
// rounded once to BF16), written to the sorted slot of each of its local routes, plus the route order. Slots come
// from route_k's per-CTA starts (base_tab). Rows [t0, T) only, so moe_block.py can run it in two parts.
__global__ void __launch_bounds__(256) normgather_k(const uint4* __restrict__ x, const float* __restrict__ w,
    const int* __restrict__ rinfo, const int* __restrict__ base_tab, int G, float eps, uint4* __restrict__ a,
    long* __restrict__ order, int T, int t0) {
  __shared__ float red[8];
  __shared__ int dst[8];
  GDC_WAIT();
  GDC_GO();
  const int tid = threadIdx.x;
  // Thread tid owns 16 B vectors tid and 256 + tid of the row (columns 8 tid.., 2048 + 8 tid..): coalesced.
  for (int t = t0 + (int)blockIdx.x; t < T; t += gridDim.x) {  // rows [t0, T)
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
// route_k without the grid tail (pooler + router as one GEMM): per token (one warp, 8 experts per lane) the logits
// hi + lo from row t of lg (row stride ldl floats), sigmoid, the biased top 8 (redux, ties to the lower id), weights;
// local routes get a rank inside this CTA's expert count; each CTA writes its counts and exits. normgather2_k turns
// the counts into slots (the kernel boundary is the grid-wide sync) and CTA 0 of it writes the expert plan.
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

// normgather_k on the FP32 pooler columns of the combined GEMM (row stride ldx floats; each value rounded to BF16
// first, as the pooler's BF16 output), with the slots from route2_k's per-CTA counts: per expert, the start (counts
// of all lower experts) plus the counts of lower route CTAs. CTA 0 also writes the counts and the expert plan
// (plan2_s) and zeroes the expert kernel's cnt1 words.
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
  // Thread tid owns columns 8 tid .. 8 tid + 7 and 2048 + 8 tid .. (its two 16 B BF16 vectors of the output row).
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
// plan2_s with fixed-capacity route lists: expert e's routes start at row start (e * cap; no prefix over the
// counts). Used by rng2_k.
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
  // mb is 1 or 2 in every plan entry (px_plan, px_get(0)), so 64 / pm and 32 / pm are selects, not divisions
  const int o1 = pn * (pm == 1 ? 64 : 32) - 32 * pr, o2 = pn * (pm == 1 ? 32 : 16) - 16 * pr;
  // GEMM2 claim order by the first token tile's time, longest first (LPT); narrow experts (MB 2, W <= 64) go last,
  // as in plan order (their GEMM1 tiles are claimed last); cnt[] is free here and holds rank -> position.
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

// plan2_cap with the plan entry from g_ptab (PR false, as plan2_cap) or from px_plan in registers (PR true).
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
  // mb is 1 or 2 in every plan entry (px_plan, px_get(0)), so 64 / pm and 32 / pm are selects, not divisions
  const int o1 = pn * (pm == 1 ? 64 : 32) - 32 * pr, o2 = pn * (pm == 1 ? 32 : 16) - 16 * pr;
  // GEMM2 claim order by the first token tile's time, longest first (LPT); narrow experts (MB 2, W <= 64) go last,
  // as in plan order (their GEMM1 tiles are claimed last); cnt[] is free here and holds rank -> position.
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

// Routing and the pre-expert norm + gather in one kernel, one CTA per token (after the pooler + router GEMM, FP32
// out, row stride ldx floats; logits hi | lo at column 4096): warp 0 takes the biased top 8 (as route_k) and claims a
// row for each local route in its expert's fixed-capacity list (row e * cap + a global per-expert counter), the CTA
// writes the token's normed row (BF16-rounded pooler values, FP32 statistics, (1 + w)) there. The last CTA swaps the
// counters back to zero, writes the plan (plan2_cap) and zeroes the expert kernel's cnt1 words. ctr: 32 expert
// counters + a done counter, persistent and zero between calls. rng2_run launches the 128-thread form rng3y_k
// below; rng2_k itself is not launched.
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
  // Done counter: this CTA's claims (warp 0's atomics) happened before its __syncthreads above.
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
#define RNG3_MINB 6  // rng3y_k: CTAs per SM the register budget aims at
// sigmoid as SGLang's routing.cuh / Triton tl.sigmoid computes it, operation by operation (so the routing matches).
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
// Relaxed GPU-scope load, for the planner CTA's poll.
__device__ __forceinline__ int ld_rlx_gpu(const int* p) {
  int v;
  asm volatile("ld.relaxed.gpu.global.b32 %0, [%1];" : "=r"(v) : "l"(p) : "memory");
  return v;
}
// Timing-stamp hooks in rng3y_k; they compile to nothing here.
#define Z29(k) do {} while (0)
#define Z29D(k, dep) do {} while (0)
#define Z29C(k) do {} while (0)
#define Z29V(k, v) do {} while (0)
__device__ __forceinline__ void red_add_gpu(int* p, int v) {
  asm volatile("red.relaxed.gpu.global.add.s32 [%0], %1;" ::"l"(p), "r"(v) : "memory");
}
// The planner CTA's plan: plan entries and key costs for counts < V2_PTN from tables (tp, tk) that warps 1-3 staged
// in shared memory, and the rank loops on 4 independent accumulators (plan2_capx's loops are one 32-step dependent
// chain). Writes the same meta words as plan2_capx<false>.
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
// rng2_k in CTAs of 128 threads (each thread does two of rng2_k's 256 column groups); lists start hd rows into each
// expert's block (row e * cap + hd + claim, cap = T + hd). D = 1 (moe_block.py's RNG3_XM bit 16): T + 1 CTAs, and
// CTA T is a planner that zeroes the cnt1 words and stages the plan tables first, then polls ctr[32] until all T
// tokens have arrived and writes the plan (plan2_v2). D = 0: the last CTA to finish plans, as rng2_k.
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
    // Planner CTA: everything that does not need the counts first, then the arrival poll, then the plan.
    // Warps 1-3 stage the plan tables while warp 0 waits for the arrivals (named barrier 1).
    __shared__ int v2_tp[V2_PTN];
    __shared__ unsigned short v2_tk[V2_PTN];
    const int v2_nt = min(T + 1, V2_PTN);  // a count is at most T (one claim per token and expert)
    if (warp != 0) {
      // 8 branch-free g_ptab loads per thread, all in flight at once (g_ptab[0] = px_plan(0) is px_get's n <= 0
      // entry), then the key costs; serial px_get calls (with their branches) were slower.
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
    emask[lane] = 0x7fffffff;  // no route moves to another expert: every emask word is set to 0x7fffffff
    for (int i = lane; i < nzero; i += 32) meta[260 + i] = 0;
    if (lane == 0)
      while (ld_rlx_gpu(ctr + 32) < T) __nanosleep(20);
    __syncwarp();
    const int n = ld_rlx_gpu(ctr + lane);
    if (lane == 0) Z29D(7, n);
    ctr[lane] = 0;
    if (lane == 0) ctr[32] = 0;
    asm volatile("bar.sync 1, 128;" ::: "memory");  // tables staged
    if (lane == 0) Z29(13);
    plan2_v2(meta, n, lane * cap + hd, pc, pord, v2_tp, v2_tk, v2_nt);
    if (lane == 0) { Z29(8); Z29C(11); }
    return;
  }
  const float* row = x + (size_t)t * ldx;
  const float4* xr = reinterpret_cast<const float4*>(row);
  float4 h0, h1, l0, l1, b0, b1;
  if (warp == 0) {  // the router logits and bias first (warp 0's routing waits on them, not on the pooler row)
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
      // Arrival: after every claim of this token has returned (the ballot reads the returned slots). It adds 1:
      // lanes 8-31 never hold a claim, so bm is never all ones.
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
    const int ot = tid + 128 * hh;  // the column group of rng2_k's thread ot
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
  // Done counter: this CTA's claims (warp 0's atomics) happened before its __syncthreads above.
  if (tid == 0) { amlast = atomicAdd(&ctr[32], 1) == T - 1; Z29D(6, amlast); }
  __syncthreads();
  if (!amlast) return;
  for (int i = tid; i < nzero; i += 128) meta[260 + i] = 0;
  if (warp == 0) {
    const int n = atomicExch(&ctr[lane], 0);
    if (lane == 0) { Z29D(7, n); ctr[32] = 0; }
    emask[lane] = 0x7fffffff;  // as the planner CTA
    plan2_capx<false>(meta, n, lane * cap + hd, pc, pinf, pord);
    if (lane == 0) { Z29(8); Z29C(11); }
  }
}
// One warp per token, FusedQKNorm's math with its FP8 KV store and none of its slot-0 skip: per-head RMSNorm of the
// 16 q and 2 k heads, the interleaved RoPE on the BF16-rounded row (when rope != 0), BF16 rounding, then FP8 q out
// and FP8 k (normed) and v (raw) into the cache rows at cache_loc. Lane l holds columns 2l and 2l + 1 of each head,
// so a RoPE pair sits in one lane.
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
      if (LO) {  // second q plane: q_lo = e4m3(x - e4m3(x)) of the same BF16-rounded x (residual exact in FP32)
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

// front_k fused with the sliding-window gather of attention_sliding_split.py (its gather's ko [B, S, 2, 64] and
// sigma-ordered vt [B, 2, 64, S]).
// One warp per copy row j of request b: j < jn an old window row (copied from the cache), jn <= j < nv a token of
// this call (front_k's math, then the K row also into ko and the V row into the CTA's tile), j >= nv zero.
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
  // part 1 starts at the first 32-row block holding a row of this call (oldg_k copies the blocks before it);
  // one 8-row group per CTA, S / 8 CTAs per request, the ones past the end exit at once.
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
  // Rows j0 .. j0 + 7 are keys 8a .. 8a + 7 of their 32-key block (a = (j0 / 8) % 4): key 8a + 2m + i (i = 0, 1) sits
  // in slot 16 (a / 2) + 2 (a % 2) + 4m + i (the inverse of the gather's sigma), so a head dim row gets 4 slot pairs.
  const int a = (j0 >> 3) & 3, sb = (j0 & ~31) + 16 * (a >> 1) + 2 * (a & 1);
#pragma unroll
  for (int r = 0; r < 2; ++r) {
    const int idx = threadIdx.x + 256 * r, hd = idx >> 2, m = idx & 3;
    const uint16_t w2 = (uint16_t)vtile[2 * m][hd] | (uint16_t)((uint16_t)vtile[2 * m + 1][hd] << 8);
    *reinterpret_cast<uint16_t*>(vt + ((size_t)b * 128 + hd) * S + sb + 4 * m) = w2;
  }
}
// The gather's part 0 (32-row blocks holding only rows cached before this call), for a side queue behind a
// launch-completion edge: no griddepcontrol (the edge launches it while the qkv GEMM still runs).
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
}  // namespace

void front_run(torch::Tensor qkv, torch::Tensor pos, torch::Tensor cs, torch::Tensor loc, torch::Tensor qo,
               torch::Tensor kc, torch::Tensor vc, double eps, int64_t rope) {
  const int T = (int)qkv.size(0);
  pdl_launch(front_k<false>, dim3((T + 7) / 8), dim3(256), 0,
      (const __nv_bfloat162*)qkv.data_ptr(), (const long*)pos.data_ptr<long>(), (const float*)(rope ? cs.data_ptr<float>() : nullptr),
      (const long*)loc.data_ptr<long>(), (uint16_t*)qo.data_ptr(), (uint8_t*)kc.data_ptr(), (uint8_t*)vc.data_ptr(), T, (float)eps,
      (int)rope, (uint16_t*)nullptr);
}

// front fused with the sliding gather. qlo empty: no q_lo plane. win = window - 1; part 1 skips the 32-row blocks
// of only old rows (oldg_k's), any other part does all rows. gx > 0 overrides the grid's x size.
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
// During graph capture, turn the current queue's capture dependencies into programmatic edges: port 0 the
// programmatic (trigger) port, 1 the launch-completion port, so the next kernel can start before the previous one
// ends. Returns the number of dependencies changed (< 0: not capturing or an error).
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
// front_run that also writes the q_lo plane (qlo [T, 16 * 64] e4m3, contiguous).
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
  // lg: [T, >= 512] FP32 view (hi | lo logits at its first 512 columns), any row stride
  const int T = (int)lg.size(0), G = (int)cnt_tab.size(0);
  TORCH_CHECK(G * RW >= T, "route2: too few CTAs");
  route2_k<<<G, RW * 32, 0, at::cuda::getCurrentCUDAStream()>>>(
      lg.data_ptr<float>(), (int)lg.stride(0), bias.data_ptr<float>(), lstart.data_ptr<int>(), T, (float)mult,
      (float)eps, tw.data_ptr<float>(), tid.data_ptr<int>(), slot.data_ptr<int>(), rinfo.data_ptr<int>(),
      cnt_tab.data_ptr<int>());
}
void normgather2_run(torch::Tensor x, torch::Tensor w, torch::Tensor rinfo, torch::Tensor cnt_tab, double eps,
                     torch::Tensor a, torch::Tensor order, torch::Tensor meta, int64_t nzero) {
  // x: [T, 4096] FP32 view, any row stride
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
  // x: the [T, 4608] FP32 output of the pooler + router GEMM (any row stride); a / order: 32 * (T + hd) rows.
  // Launches rng3y_k: with the planner CTA (T + 1 CTAs) when xm & 16, else T CTAs.
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
// normgather of rows [t0, t1) only (x holds all T rows; the slots and route order are those of the whole call).
void normgather_range_run(torch::Tensor x, torch::Tensor w, torch::Tensor rinfo, torch::Tensor base_tab, double eps,
                          torch::Tensor a, torch::Tensor order, int64_t t0, int64_t t1) {
  TORCH_CHECK(0 <= t0 && t0 < t1 && t1 <= x.size(0), "normgather_range_run");
  pdl_launch(normgather_k, dim3((unsigned)(t1 - t0)), dim3(256), 0,
      (const uint4*)x.data_ptr(), (const float*)w.data_ptr<float>(), (const int*)rinfo.data_ptr<int>(), (const int*)base_tab.data_ptr<int>(),
      (int)base_tab.size(0), (float)eps, (uint4*)a.data_ptr(), order.data_ptr<long>(), (int)t1, (int)t0);
}
