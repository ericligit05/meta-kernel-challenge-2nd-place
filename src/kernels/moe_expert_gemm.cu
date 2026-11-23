// Grouped NVFP4 x BF16 expert GEMMs for the routed experts (both GEMMs of the 32 local experts, gate/up then down):
// moe_fused (the wide layout, 8192+ tokens), moe_mixed (a layout per expert, below 2048 tokens) and moe_mixed_mc (the
// same plan on CTA pairs, 2048 to 8191 tokens), plus the route-list kernels moe_build / moe_gather run for
// _experts_moe in moe_block.py.
// Z_MIDSET / Z_M2SET: bit masks of the narrower MMA widths moe_mixed / moe_mixed_mc may issue inside MB 2 tiles
// (Z_MIDSET: 40 / 48 / 56 in W 64 tiles and 24 in W 32 tiles; Z_M2SET: 80 / 96 / 112 in W 128 tiles). All are on.
// Names with a letter-and-number tag (x64_elect, x91s / x91ph, X63_C, X65_W1 / X65_X1, x65w40, x40_pol_ef, z0_slot,
// z0_ex) are part of the submitted code; the tag only keeps the name unique, and each is described where it is defined.
#define Z_MIDSET 15
#define Z_M2SET 7
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <stdint.h>


// Launch with programmatic dependent launch (PDL) allowed: the kernel may start while the previous kernel on the
// stream finishes, and waits for it at griddepcontrol.wait (GDC_WAIT).
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
// ---- Grouped NVFP4 x BF16 expert GEMMs (wgmma, weights decoded straight into the A registers). ----
// The weights are the A operand (M = weight rows), the routed token rows the B operand (N = tokens). A CTA tile is
// 128 * MB weight rows of one expert (16-row tiles, MB per consumer warp) x up to TN of its routed tokens, over the
// full K = 4096 in 64 chunks of 64. prepare() (_pack_nvfp4 in moe_block.py) repacks each expert losslessly into units
// of 16 rows x 64 k: 512 B of E2M1 codes (per lane and k16 step one 32-bit word, bits placed for decode4) then the 64
// block scales as 8 E + m codes (see sc_bf16). A unit decodes to the m64nNk16 A fragments with shifts and masks
// (decode4) and one exact BF16 multiply by the block scale; the FP32 global scale goes on the output. The token rows
// come from the routes sorted by expert, by 2-D TMA (128 B swizzle). A producer warp streams [token chunk][weight
// units] through a ring of NS stages (full / empty mbarriers); two consumer warpgroups decode and multiply; three
// copier warps move each finished tile from the staging buffer to global memory. CTAs are persistent and claim tiles
// from a global counter. GEMM1 (MODE 1) writes silu(gate * gs) * (up * gs) in BF16 to the intermediate rows; GEMM2
// (MODE 2) writes acc * gs2 in BF16 straight to each route's slot. NVFP4 weights are so small that the token rows
// dominate the traffic, so a tile takes as many weight rows as the registers allow and the token tile is sized to
// the prefill.
constexpr int FZ_PAIR = 13312;  // the second chunk of a paired narrow stage (1024-aligned, after 8 KB x + 4.5 KB w)
constexpr int NW = 8, UNIT = 576, UNIT2 = 576;  // consumer warps; bytes per unit (512 B codes + 64 scale codes), both layouts
// MOE_SWZ: token tiles per group that run through every weight row block before the next group (1: one token tile).
// MOE_IL: chunk_mma's decode / issue interleave for moe_fused (1: one fence per k16 step).
#define MOE_SWZ 1
#define MOE_IL 1
constexpr int EBYTES = 256 * 136 * 2;  // epilogue BF16 staging: >= [128 tok][264] and [256 tok][136]
// Layouts: MB = 2 blocks of 64 rows per consumer warpgroup with TN = 128 tokens (256 weight rows per CTA tile), or
// MB = 1 with TN = 256 (128 rows; each decoded weight then serves twice the tokens).
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
__device__ __forceinline__ bool x64_elect() {  // elect.sync over the full (converged) warp: true in one lane
  uint32_t p;
  asm volatile("{\n .reg .pred P;\n elect.sync _|P, 0xffffffff;\n selp.u32 %0, 1, 0, P;\n}" : "=r"(p));
  return p != 0;
}
__device__ __forceinline__ void bulk_g2s(uint32_t dst, const void* src, int bytes, uint32_t bar) {
  asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];"
               ::"r"(dst), "l"(src), "r"(bytes), "r"(bar) : "memory");
}
// Four 8x8 BF16 blocks of accumulator fragments, transposed into the token-major staging buffer (one instruction for
// what took 16 two-byte stores per thread).
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
// FT_WSEM: an extra qualifier for ft_wait_cluster's try_wait (empty: the default semantics).
#define FT_WSEM ""
// Multicast TMA: the box lands at the same shared offset in every CTA of the cluster named in mask.
__device__ __forceinline__ void tma_2d_mc(uint32_t dst, const CUtensorMap* map, int c0, int c1, uint32_t bar, uint16_t mask) {
  asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster [%0], [%1, {%2, %3}], [%4], %5;"
               ::"r"(dst), "l"(map), "r"(c0), "r"(c1), "r"(bar), "h"(mask) : "memory");
}

// TMA loads with an L2 cache policy (the weight loads pass evict_first).
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
// CTA-pair helpers for moe_mixed_mc: a shared address in CTA r of the cluster, remote mbarrier arrives and waits,
// release / acquire arrives for the tile claim hand-off, a remote shared store, and a cluster-wide barrier.
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
__device__ __forceinline__ uint32_t ft_peer(uint32_t a) {  // a's address in the other CTA of the pair
  uint32_t r, o;
  asm volatile("mov.u32 %0, %%cluster_ctarank;" : "=r"(r));
  asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(o) : "r"(a), "r"(r ^ 1u));
  return o;
}
__device__ __forceinline__ void ft_cluster_sync() {  // every thread of both CTAs (divergent call sites: no .aligned)
  asm volatile("barrier.cluster.arrive.release;\nbarrier.cluster.wait.acquire;" ::: "memory");
}
__device__ __forceinline__ uint32_t prmt(uint32_t a, uint32_t b, uint32_t s) {
  uint32_t r;
  asm("prmt.b32 %0, %1, %2, %3;" : "=r"(r) : "r"(a), "r"(b), "r"(s));
  return r;
}
__device__ __forceinline__ uint64_t wg_desc(uint32_t addr) {  // K-major, 128 B swizzle, 8-row groups 1 KB apart
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

// moe_block.py's _wgmma_src() replaces the next line with the wgmma_n / wgmma_part specializations.
@@WGMMA@@
// Narrow tiles with at most MX_MS tokens: mma.sync m16n8k16 on the same decoded A fragments (a warp's 16 rows of a
// wgmma A operand are exactly the m16n8k16 A fragment) and B fragments by ldmatrix from the TMA-swizzled token rows;
// the m16n8 C fragment is the first 4 floats of the wgmma accumulator, so the epilogue is unchanged.
__device__ __forceinline__ void mma16816(float* d, uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t b0, uint32_t b1) {
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
               : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]) : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}
__device__ __forceinline__ void ldsm4(uint32_t addr, uint32_t& r0, uint32_t& r1, uint32_t& r2, uint32_t& r3) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];" : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(addr));
}

// Exact decode, with no table and no conversion of the codes: an E2M1 code c (sign bit 3) placed as BF16 bits
// (c & 7) << 6 | (c & 8) << 12 is exactly e2m1(c) * 2^-126 (code 1, the value 0.5, lands on the BF16 subnormal
// 2^-127). Multiplying by the block scale times 2^63 (exact: a 2-bit by 4-bit mantissa product) gives
// e2m1 * scale * 2^-63, a normal BF16; the 2^63 goes on the output with the global scale.
// e2m1_pair is this placement for a word with pair m's codes at bits 4m (first value) and 16 + 4m (second), with
// m = a0..a3 = (row g, k 2q), (row g + 8, k 2q), (row g, k 2q + 8), (row g + 8, k 2q + 8). It is not called: the
// packed weights use decode4's bit layout instead.
__device__ __forceinline__ uint32_t e2m1_pair(uint32_t w, int m) {
  const uint32_t lo = (m == 0 ? w << 6 : m == 1 ? w << 2 : m == 2 ? w >> 2 : w >> 6) & 0x01C001C0u;
  const uint32_t sg = (m == 0 ? w << 12 : m == 1 ? w << 8 : m == 2 ? w << 4 : w) & 0x80008000u;
  return lo | sg;
}
// The same BF16 bits for all four pairs of a word, in 6 ALU ops and 3 IMADs (LOP3 / SHF are half rate on the ALU
// pipe; the IMADs run beside them on the FMA pipe). prepare() places each code's bits (_POS in moe_block.py) so that
// every pair lands at bits 6..8 / 15 (first value) and 22..24 / 31 (second) under the mask M: pair 0 as stored,
// pairs 1 and 2 after q << 3 and q << 6 (ptxas issues both as IMAD.SHL).
__device__ __forceinline__ void decode4(uint32_t q, uint32_t* o) {
  const uint32_t M = 0x81C081C0u;
  o[0] = q & M;
  // Pair 3: rotate by 6, premask, and one carry-free multiply by 0x1084 (<< 2 + << 7 + << 12) that moves b0, b1 / b2
  // and the sign at once: the one-IMAD trick of onnxruntime's FP4 MoE GEMV decode (moe_gemv_fp4.cu, MIT License).
  o[1] = (q << 3) & M;
  o[2] = (q << 6) & M;
  o[3] = ((__funnelshift_l(q, q, 6) & 0x001B001Bu) * 0x1084u) & M;
}
// Two BF16 products (not called; the decode uses bmulh).
__device__ __forceinline__ uint32_t bmul(uint32_t a, uint32_t s) {
  __nv_bfloat162 r = __hmul2(*reinterpret_cast<__nv_bfloat162*>(&a), *reinterpret_cast<__nv_bfloat162*>(&s));
  return *reinterpret_cast<uint32_t*>(&r);
}
// Two block scales (bytes 0, 1 of v) -> BF16 bits of scale * 2^63, each replicated in both halves. prepare() stores each
// E4M3 scale losslessly as the code 8 E + m with scale = 1.m * 2^(E - 9) (E4M3 subnormals normalized; every positive E4M3
// value has a 3-bit mantissa and E in 0..17), so its BF16 bits are code * 16 + (181 << 7): one byte move and one IMAD.
__device__ __forceinline__ uint32_t sc_bf16(uint32_t v, uint32_t sel) { return prmt(v, 0, sel) * 16u + 0x5A805A80u; }
// a * (half h of s, broadcast): HMUL2 takes the half as an operand selector, so the scale pair needs no byte moves.
__device__ __forceinline__ uint32_t bmulh(uint32_t a, uint32_t s, int h) {
  const __nv_bfloat162 sv = *reinterpret_cast<__nv_bfloat162*>(&s);
  __nv_bfloat162 r = __hmul2(*reinterpret_cast<__nv_bfloat162*>(&a), h ? __high2bfloat162(sv) : __low2bfloat162(sv));
  return *reinterpret_cast<uint32_t*>(&r);
}
// Both scale bytes of v, each broadcast to a BF16 pair (not called).
__device__ __forceinline__ void e4m3x2(uint32_t v, uint32_t& s0, uint32_t& s1) {
  s0 = sc_bf16(v, 0x4040);
  s1 = sc_bf16(v, 0x4141);
}
// One lane's unit -> its 16 A registers (4 k16 steps x a0..a3), values e2m1 * scale * 2^-63. Used by chunk_mma's
// IL = 0 form, which no launch instantiates.
__device__ __forceinline__ void decode_unit(const uint8_t* up, int lane, uint32_t (&r)[16]) {
  const uint4 nb = *reinterpret_cast<const uint4*>(up + lane * 16);
  // K permutation (kperm_index): all of this lane's values in the chunk share one scale block, so one scale pair
  // (rows g, g + 8) serves all four k16 steps.
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

// One chunk: this warp's MB units decoded and issued as W-wide wgmmas on accumulators of width TN. IL = 1 decodes and
// issues one k16 step at a time with a fence per step, so the tensor pipe starts on step 0 while steps 1..3 decode;
// IL = 4 decodes the whole chunk first. IL = 0 (decode_unit, then all wgmmas) is not instantiated.
template <int W, int TN, int MB, int IL>
__device__ __forceinline__ void chunk_mma(float (&acc)[MB][TN / 2], uint32_t (&r)[MB][16], const uint8_t* wb, int ubase,
                                          int lane, uint32_t xa) {
  const uint64_t d0 = wg_desc(xa);  // k16 step kk is 32 B further: + 2 in the 16 B address field (no carry: stages start low)
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
    uint32_t S[MB];  // one scale pair (rows g | g + 8, this lane's block) per unit per chunk
#pragma unroll
    for (int b = 0; b < MB; b++) {
      const uint8_t* up = wb + (ubase + 4 * b) * (MB == 2 ? UNIT2 : UNIT);
      nb[b] = *reinterpret_cast<const uint4*>(up + lane * 16);
      if constexpr (false) S[b] = 0;  // never taken: both layouts decode their scales the same way
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
    if constexpr (IL == 4) {  // decode the whole chunk, one warpgroup fence, then all its wgmmas
      wg_fence();
#pragma unroll
      for (int kk = 0; kk < 4; kk++)
#pragma unroll
        for (int b = 0; b < MB; b++) mma(b, kk);
    }
  }
}
template <int V> using IC = std::integral_constant<int, V>;
// The wide layout (launched as <TN 256, MB 1> at 8192+ tokens). Tiles 0 .. total1 - 1 are GEMM1's (gate/up, SiLU
// applied by the copiers, into inter), the rest GEMM2's (down, into the route slots). phase 0 runs both in one
// persistent launch, with each GEMM2 tile's producer waiting until its token tile's RB1 GEMM1 blocks are written
// (cnt1, bumped by the copiers); moe_fused_run uses phase 1 then phase 2, two launches, which measured 1% faster.
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
  __shared__ int s_tq[8];                        // tile ids, producer -> consumers and copiers
  __shared__ int4 s_tl[8];                       // their (expert | GEMM << 8, row block, first row, tokens)
  __shared__ int s_tj[8];                        // and token tile (cnt1 index)
  __shared__ __align__(8) uint64_t tbars[8];
  __shared__ __align__(8) uint64_t zsfree;  // staging free again (the 96 copier threads arrive once per tile)
  __shared__ float s_gs[2][32];
  const int tid = threadIdx.x, warp = tid / 32, lane = tid % 32, g = lane / 4, q = lane % 4;
  const uint32_t zsf = (uint32_t)__cvta_generic_to_shared(&zsfree);
  const uint32_t raw = (uint32_t)__cvta_generic_to_shared(mraw);
  const uint32_t sb = (raw + 1023) & ~1023u;
  uint8_t* const sp0 = mraw + (sb - raw);
  // Epilogue staging after the ring, BF16 per token row: stride 136 (MB 1) or RW + 8 (GEMM2 at MB 2). ebuf, an FP32
  // view of it, is not used.
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
  // phase 0: both GEMMs; 1: GEMM1 tiles only; 2: GEMM2 tiles only (a second launch, with its own claim counter)
  const int tbeg = phase == 2 ? total1 : 0, tend = phase == 1 ? total1 : total;
  const uint32_t btq = (uint32_t)__cvta_generic_to_shared(tbars);
  // Tiles are numbered over the experts in perm order (most routes first, so the longest tiles are claimed first
  // and the last claims are short); toff is by position in that order. Binary search for the position.
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
    // Token tiles in groups of MOE_SWZ: a group's tiles run through every weight row block before the next group, so
    // its token rows stay in L2 instead of being re-read from HBM for each row block (measured 7% faster at 16K
    // tokens, same output).
    const int RBm = mode == 1 ? RB1 : RB2;
    const int ch = lt / (MOE_SWZ * RBm), rem = lt - ch * MOE_SWZ * RBm;
    const int gt = min(MOE_SWZ, ntt - ch * MOE_SWZ);
    rb = rem / gt;
    const int tt = ch * MOE_SWZ + rem % gt;
    jt = to[lo] / (mode == 1 ? RB1 : RB2) + tt;  // token tiles before this expert's, plus tt
    row0 = s_start[e] + tt * TN;
    nv = min(TN, s_cnt[e] - tt * TN);
  };
  if (warp >= NW) {  // last warpgroup: warp NW is the producer, warps NW + 1 .. NW + 3 the copiers
    asm volatile("setmaxnreg.dec.sync.aligned.u32 40;");
    if (warp == NW) {  // producer warp: all 32 lanes stay converged; one elected lane issues each copy
      // Tiles are claimed from a global counter (zeroed by the list build), so CTAs that finish early take more;
      // each tile id goes to the consumers and copiers through an 8-slot queue (s_tq / s_tl / s_tj, tbars).
      int gch = 0, x91s = 0, x91ph = 0;  // chunks issued, ring stage, stage parity
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
        const CUtensorMap* bmw = mode == 1 ? &bmap1w : &bmap2w;  // boxes of 64 k x 256 rows
        if (mode == 2 && phase != 2) {  // the token tile's intermediate rows: wait for its RB1 GEMM1 blocks (not in a
          // phase 2 launch: it runs after the phase 1 launch has completed, so every block is there)
          while (ld_acquire(&cnt1[jt]) < RB1) __nanosleep(128);
          asm volatile("fence.proxy.async.global;" ::: "memory");
        }
        const int wvp = nv <= 32 ? 32 : nv <= 64 ? 64 : nv <= 128 ? 128 : TN;
        const int nbox = (wvp + 63) / 64;
        if (TN == 256 && wvp <= 64) {
          // A tile of at most 64 tokens carries two chunks per ring stage (one 64-row x box and one weight
          // block each, the second at FZ_PAIR), so the ring holds 8 chunks ahead instead of 4.
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
          // Only the 64-row boxes of x that hold this tile's tokens (an MMA of width wv reads no further).
          if (x64_elect()) {
            mbar_expect(bfull + 8 * s, nbox * 8192 + WB);
            if (TN == 256 && nbox == 4) tma_2d(sb + s * SB, bmw, c * 64, row0, bfull + 8 * s);  // one 32 KB box
            else for (int bx = 0; bx < nbox; bx++) tma_2d(sb + s * SB + bx * 8192, bm, c * 64, row0 + bx * 64, bfull + 8 * s);
            bulk_g2s(sb + s * SB + XB, wp + (size_t)c * WB, WB, bfull + 8 * s);
          }
        }
      }
    } else if (warp > NW) {
      // Copy-out warps: each tile's output leaves the staging buffer here, overlapping the consumers' next tile.
      // Named barrier 2: staging written (consumers arrive, copiers sync); mbarrier zsfree: staging free again.
      const int ct = tid - (NC + 32);
      for (int k = 0;; k++) {
        mbar_wait(btq + 8 * (k & 7), (k >> 3) & 1);
        const int t = s_tq[k & 7];
        if (t < 0) break;
        const int4 tl = s_tl[k & 7];
        const int mode = tl.x >> 8, rb = tl.y, row0 = tl.z, nv = tl.w;
        int zs0 = 0, zs1 = 0, zs2 = 0;
        if (TN == 256 && mode == 2) {  // this copier's destination rows (at most 3), loaded before the barrier
          if (ct < nv) zs0 = (int)slot_of[row0 + ct];
          if (ct + 96 < nv) zs1 = (int)slot_of[row0 + ct + 96];
          if (ct + 192 < nv) zs2 = (int)slot_of[row0 + ct + 192];
        }
        asm volatile("bar.sync 2, %0;" ::"n"(NC + 96) : "memory");
        const uint32_t ob = (uint32_t)__cvta_generic_to_shared(obuf);
        if (mode == 1 && MB == 1) {
          // The consumers staged gate and up rounded to BF16 (SGLang's GEMM1 output is BF16); SiLU and the product
          // happen here, off the tensor pipe's critical path.
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
          if constexpr (TN == 256) {  // one bulk copy per token row (TMA engine): 1% faster at 16K
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
        asm volatile("cp.async.bulk.wait_group.read 0;" ::: "memory");  // the staging buffer has been read
        if (mode == 1 && phase != 1) {  // release this block of the token tile's intermediate rows to GEMM2's TMA reads
          // (not needed in a phase 1 launch: phase 2 is a later launch)
          asm volatile("cp.async.bulk.wait_group 0;" ::: "memory");
          asm volatile("fence.proxy.async.global;" ::: "memory");
          __threadfence();
          asm volatile("bar.sync 4, 96;" ::: "memory");
          if (ct == 0) atomicAdd(&cnt1[s_tj[k & 7]], 1);
        }
        mbar_arrive(zsf);  // staging free: the consumers may stage their next tile
      }
    }
    return;
  }
  asm volatile("setmaxnreg.inc.sync.aligned.u32 232;");
  int gch = 0, x91s = 0, x91ph = 0;  // chunks consumed, ring stage, stage parity
  uint32_t ra[MB][16], rb_[MB][16];  // decoded A registers, alternating between consecutive chunks
  // Unit of (this warp, block b) within a stage: ubase + 4 b. MB = 2: warpgroup 0 takes units 0..7, warpgroup 1
  // units 8..15; MB = 1: warp w takes unit w.
  const int ubase = MB == 2 ? (warp < 4 ? 0 : 8) + (warp % 4) : warp;
  for (int k = 0;; k++) {
    mbar_wait(btq + 8 * (k & 7), (k >> 3) & 1);
    const int t = s_tq[k & 7];
    if (t < 0) break;
    const int4 tl = s_tl[k & 7];
    const int mode = tl.x >> 8, e = tl.x & 255, nv = tl.w;
    // MMA width by the tile's real tokens: 32, 64, 128 or TN.
    const int wv = nv <= 32 ? 32 : nv <= 64 ? 64 : nv <= 128 ? 128 : TN;
    // One tile at MMA width W, on its own accumulator array: wgmmas of different shapes writing one array (with
    // the interleaved decode's per-step fences) are serialized by ptxas (warning C7512).
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
        wg_wait<1>();  // chunk c - 1's wgmma is done: release its stage
        if (c > 0) {
          __syncwarp();
          if (lane == 0) mbar_arrive(bempty + 8 * (x91s == 0 ? NS - 1 : x91s - 1));
        }
        gch++;
x91s = x91s + 1 == NS ? 0 : x91s + 1; x91ph ^= (x91s == 0);  // next stage; the parity flips on wrap
      };
      if constexpr (TN == 256 && W <= 64) {
        // Two chunks per stage (the producer's paired layout); a stage is released once both are read.
        for (int c = 0; c < C; c += 2) {
          const int s = x91s;
          mbar_wait(bfull + 8 * s, x91ph);
          const uint8_t* sp = sp0 + s * SB;
          const uint32_t xa = sb + s * SB;
          chunk_mma<W, W, MB, MOE_IL>(acc, ra, sp + 8192, ubase, lane, xa);
          wg_commit();
          wg_wait<1>();  // chunk c - 1 (the previous stage's second chunk) is done: release that stage
          if (c > 0) {
            __syncwarp();
            if (lane == 0) mbar_arrive(bempty + 8 * (x91s == 0 ? NS - 1 : x91s - 1));
          }
          chunk_mma<W, W, MB, MOE_IL>(acc, rb_, sp + FZ_PAIR + 8192, ubase, lane, xa + FZ_PAIR);
          wg_commit();
          wg_wait<1>();  // chunk c is done (ra free again); chunk c + 1 still reads this stage
          gch++;
x91s = x91s + 1 == NS ? 0 : x91s + 1; x91ph ^= (x91s == 0);  // next stage; the parity flips on wrap
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
      // Epilogue. acc[b][4j + 2h + c] = (row 64 b + 16 (warp % 4) + g + 8h of this warpgroup's rows, token 8j + 2q + c).
      const float sc = s_gs[mode - 1][e] * 9.223372036854775808e18f;  // global scale * 2^63 (the decode's offset)
      const int lr = (warp % 4) * 16 + g;
      if (k > 0) mbar_wait(zsf, (k - 1) & 1);  // the copiers are done with the previous tile's staging
      if constexpr (MB == 1) {
        // Staging [tok][136] BF16. GEMM1: each warp's 16 rows are 8 gate rows (g) and the same 8 up rows (g + 8), staged
        // at columns warp * 8 (gate) and 64 + warp * 8 (up); GEMM2: rows 64 (warp / 4) + 16 (warp % 4) + g (+ 8). Per
        // stmatrix: blocks (tokens 8j.., rows g), (8j.., g + 8), (8j + 8.., g), (8j + 8.., g + 8); lane 8i + t gives row
        // t of block i.
        const int i4 = lane >> 3, t8 = lane & 7;
        const int cb = mode == 1 ? 64 * (i4 & 1) + warp * 8 : (warp / 4) * 64 + (warp % 4) * 16 + 8 * (i4 & 1);
        const uint32_t base = (uint32_t)__cvta_generic_to_shared(obuf) + ((8 * (i4 >> 1) + t8) * 136 + cb) * 2;
  #pragma unroll
        for (int j = 0; j < W / 8; j += 2)
          stsm4t(base + j * 8 * 136 * 2, pkbf(acc[0][4 * j] * sc, acc[0][4 * j + 1] * sc),
                 pkbf(acc[0][4 * j + 2] * sc, acc[0][4 * j + 3] * sc), pkbf(acc[0][4 * j + 4] * sc, acc[0][4 * j + 5] * sc),
                 pkbf(acc[0][4 * j + 6] * sc, acc[0][4 * j + 7] * sc));
        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
        asm volatile("bar.arrive 2, %0;" ::"n"(NC + 96) : "memory");  // staging written: the copiers take it
      } else if (mode == 1) {
        // MB = 2 only (not instantiated): block 0 holds gate rows and block 1 the same up rows (prepare's grouping),
        // so silu(gate) * up is per thread.
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
        // MB = 2 only (not instantiated): acc * gs2 * 2^63 staged per token; the copiers write each row to its route's
        // slot of the [T * top_k, 4096] output (no unsort pass).
        constexpr int RW = 128 * MB, OS = RW + 8;  // rows per CTA tile, staging stride
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
    if constexpr (TN == 256) {  // MMA width = tokens rounded up to 32 (only an expert's last tile is short)
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

// ---- Below 8192 tokens (moe_mixed, moe_mixed_mc): the CTA tile layout is chosen per expert by a cost model (the
// planner in routing_front.cu, or plan2_k below). An expert with few tokens takes 256 weight rows x up to 128 tokens
// (MB = 2: the least decode per token), a large one 128 rows x up to 256 tokens (MB = 1: each decoded weight serves
// twice the tokens), its tokens split into ntt tiles of tsz (the last one shorter). One ring of stages sized for both:
// the token rows at 0, the weight units at mx_xoff(mb). info[pos] = ntt | mb << 8 | tsz << 12 | r << 21.
// chunk_mma interleave (IL) of the narrow MB 2 pair loops (W 24..64) and of the wider MB 2 tiles: 4 = decode the
// whole chunk, then one fence and its wgmmas. MOE_PIPE2 0: the narrow pair loops free each stage themselves, so the
// end-of-tile release skips them.
#define MOE_ILN 4
#define MOE_ILM 4
#define MOE_PIPE2 0
constexpr int MX_SB = 37888, MX_NS = 4;  // stage bytes (32 KB of token rows + MB 1 weights, 1 KB aligned), stages
constexpr int X63_C = 64;  // K chunks as a compile-time constant (the launch also passes 64)
// Narrow MB 2 tiles (W <= 64) put two chunks in a stage: x0 at 0, w0 at 8192, x1 at MX_PAIR, w1 at MX_PAIR + 8192.
constexpr int MX_PAIR = 18432;
// moe_mixed_mc's narrow stage: x0 at 0, w0 at 8192, w1 right after w0 at X65_W1, x1 at X65_X1 (1 KB aligned), so the
// two weight chunks are one contiguous TMA box.
constexpr uint32_t X65_W1 = 8192 + 2 * NW * UNIT2, X65_X1 = 28672;
#define MX_MS 16  // MB 2 tiles with at most this many tokens (W = 8, 16) use mma.sync m16n8k16
#define MX_OB 128  // tiles at least this wide load their token rows as one TMA box
// Token row maps per GEMM (a, inter): m[.][0] boxes of 64 rows, m[.][1..5] one box of 128 / 160 / 192 / 224 / 256
// rows, m[.][6..8] narrow boxes of 8 / 16 / 32 rows. w[GEMM][mb - 1]: the weight units as rows of 512 B.
struct MxMaps { CUtensorMap m[2][9]; CUtensorMap w[2][2]; };
__host__ __device__ constexpr int mx_xoff(int mb) { return mb == 2 ? 16384 : 32768; }
// Only bit 16 of TF_WSET is read: it would give MB 2 tiles of 65-96 tokens a width-96 accumulator (off; those tiles
// use W 128 accumulators with a narrower MMA instead).
#define TF_WSET 15
__host__ __device__ __forceinline__ int tf_w1(int nv) {  // MB 1 MMA width: 128, then steps of 8
  return nv <= 128 ? 128 : (nv + 7) / 8 * 8;
}
__host__ __device__ __forceinline__ int mx_w(int mb, int nv) {  // MMA width of a tile with nv tokens
  if (mb == 2) return nv <= MX_MS ? (nv + 7) / 8 * 8 : nv <= 32 ? 32 : nv <= 64 ? 64 : (nv <= 96 && (TF_WSET & 16)) ? 96 : 128;
  return tf_w1(nv);  // the token box rounds the width up to a multiple of 32 rows
}
// floor(a / b) for non-negative ints: a float quotient, corrected by one either way.
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
  constexpr int C = X63_C;  // the launch passes 64 too (C_ is unused)
  (void)C_;
  int* const counter = meta + 130;
  int* const cnt1 = meta + 260;
  extern __shared__ uint8_t mraw[];
  __shared__ __align__(8) uint64_t bars[2 * NS];
  __shared__ int s_toff[2][33], s_start[32], s_cnt[32], s_perm[32], s_info[32], s_jt0[32], s_map2[32];
  __shared__ int s_tq[8];
  __shared__ int4 s_tl[8];                       // (expert | GEMM << 8 | mb << 12 | width << 16, row block, first row, tokens)
  __shared__ int s_tj[8];
  __shared__ __align__(8) uint64_t tbars[8];
  __shared__ __align__(8) uint64_t sbars[8];  // queue slot k filled (scheduler -> producer)
  __shared__ __align__(8) uint64_t gbars[8];  // claim the next tile now (producer -> scheduler)
  __shared__ float s_gs[2][32];
  __shared__ int z0_slot[2][256];  // a GEMM2 tile's destination rows (route slots), double-buffered
  const int tid = threadIdx.x, warp = tid / 32, lane = tid % 32, g = lane / 4, q = lane % 4;
  const uint32_t raw = (uint32_t)__cvta_generic_to_shared(mraw);
  const uint32_t sb = (raw + 1023) & ~1023u;
  uint8_t* const sp0 = mraw + (sb - raw);
  __nv_bfloat16* const obuf = reinterpret_cast<__nv_bfloat16*>(sp0 + NS * SB);
  const uint32_t bfull = (uint32_t)__cvta_generic_to_shared(bars), bempty = bfull + 8 * NS;
  // Everything that does not read the routing kernel's output happens before griddepcontrol.wait (barrier init, the
  // global scales, the tensor map prefetch), so under PDL it overlaps the routing kernel's last CTAs.
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
  // A valid placeholder plan (one GEMM1 MB 1 tile of 128 tokens). The producer warp overwrites it with the real plan
  // after griddepcontrol.wait, before the scheduler thread locates any tile from it.
  if (tid < 33) { s_toff[0][tid] = tid ? 1 : 0; s_toff[1][tid] = 0; }
  if (tid < 32) {
    s_start[tid] = 0; s_cnt[tid] = 128; s_perm[tid] = 0; s_info[tid] = 1 | 1 << 8 | 128 << 12; s_jt0[tid] = 0; s_map2[tid] = 0;
  }
  __syncthreads();
  const uint32_t btq = (uint32_t)__cvta_generic_to_shared(tbars);
  const uint32_t bsl = (uint32_t)__cvta_generic_to_shared(sbars);
  const uint32_t bgo = (uint32_t)__cvta_generic_to_shared(gbars);
  // info = ntt | mb << 8 | tsz << 12 | r << 21. With r > 0 the expert's last r tokens are one MB 2 tile
  // (token tile k1 = ntt - 1) after k1 MB 1 tiles of tsz; its GEMM1 / GEMM2 tiles follow the MB 1 ones.
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
      uint32_t gch = 0;  // unsigned: % NS and / NS are a mask and a shift
      for (int k = 0;; k++) {
        int x_t = -1, x_mode = 0, x_e = 0, x_mb = 2, x_rb = 0, x_row0 = 0, x_nv = 0, x_jt = 0, x_rem = 0;  // tile 0, located in iteration 0 only
        if (k == 0) {
          GDC_WAIT();
          int ln;
          asm volatile("mov.u32 %0, %%laneid;" : "=r"(ln));
          const int* const mp = counter - 130;  // meta, rebuilt from a pointer the producer keeps anyway
          const int a0 = mp[64 + ln], a1 = mp[97 + ln], tot0 = mp[96], tot1 = mp[129], st = mp[32 + ln], cn = mp[ln];
          const int pe = mp[164 + ln], inf_ = mp[196 + ln], jm = mp[228 + ln];
          s_toff[0][ln] = a0; s_toff[1][ln] = a1;
          if (ln == 0) { s_toff[0][32] = tot0; s_toff[1][32] = tot1; }
          s_start[ln] = st; s_cnt[ln] = cn;
          s_perm[ln] = pe; s_info[ln] = inf_; s_jt0[ln] = jm & 0xFFFF;
          s_map2[ln] = jm >> 16;  // GEMM2 claim rank -> plan position
          // Tile 0 (= blockIdx) located by the whole warp from the plan registers: the binary search over the
          // monotone offsets is one ballot, every table read a shuffle (locate()'s math, same results).
          x_t = (int)blockIdx.x;
          if (x_t >= tot0 + tot1) x_t = -1;
          if (x_t >= 0) {  // warp-uniform
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
          __syncwarp();  // all 32 lanes stay in the loop (converged); issue ops are elected below
        }
        // CTA b's first tile is tile b; the counter then hands out tiles from gridDim on.
        int t, mode = 0, e = 0, mb = 2, rb = 0, row0 = 0, nv = 0, jt = 0, W;
        if (k == 0) {  // tile 0 is blockIdx, located above; later tiles come from the scheduler thread
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
        const CUtensorMap* bm = mb == 2 && W <= 32 ? &bmx.m[mode - 1][W == 8 ? 6 : W == 16 ? 7 : 8]  // narrow boxes of exactly W rows
                                                   : mode == 1 ? &bmap1 : &bmap2;
        const CUtensorMap* wm = &bmx.w[mode - 1][mb - 1];
        const int wr0 = (int)(((mode == 1 ? (size_t)e * ((64 >> (mb - 1))) : (size_t)e * ((32 >> (mb - 1)))) + rb) * C * (WB / 512));
        (void)wp;
        if (mode == 2 && k >= 0) {
          while (ld_acquire(&cnt1[jt]) < (64 >> (mb - 1))) __nanosleep(128);
          asm volatile("fence.proxy.async.global;" ::: "memory");
        }
        const int nbox = (W + 63) / 64;
        // One box of Wb rows when W >= 128 (Wb = W rounded up to 32: 128..256), else 64-row boxes.
        const bool one = W >= MX_OB && W >= 128;
        const int Wb = (W + 31) & ~31;  // box rows (a W in steps of 8 loads the next multiple of 32)
        const CUtensorMap* bmw = &bmx.m[mode - 1][one ? (Wb - 96) / 32 : 0];
        const int xbytes = one ? Wb * 128 : nbox * 8192;
        if (mb == 2 && W <= 64) {
          constexpr int WB2 = wbytes(2);
          const int z0_ex = 2 * (W * 128 + WB2);  // bytes per stage: two chunks of W token rows and MB 2 weights
          // Narrow tile: two chunks per stage (x, weights, x, weights at 0, 8192, MX_PAIR, MX_PAIR + 8192), so the ring
          // holds twice as many chunks ahead of these short, load-heavy chunks.
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
                if (c == C - 8) mbar_arrive(bgo + 8 * (k & 7));  // 4 stages from the end: the scheduler may claim tile k + 1
              }
            }
            if (k < 0) break;
          }
        } else {
          // Every tile issues a multiple of NS stages, so gch % NS == 0 here: one-box tiles unroll by NS with static
          // stage slots; the others use the generic loop.
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
                  if (s == 0 && c == X63_C - 4) mbar_arrive(bgo + 8 * (k & 7));  // 4 stages from the end: claim tile k + 1
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
                if (c == C - 4) mbar_arrive(bgo + 8 * (k & 7));  // 4 stages from the end: claim tile k + 1
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
        if (ct == 0) {  // copier thread 0 is the scheduler: it claims, locates and publishes tile k + 1
          const int kk = k + 1;
          mbar_wait(bgo + 8 * (k & 7), (k >> 3) & 1);  // claim late: once the producer is 4 stages from tile k's end
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
          // The tile's destination rows, one load per token, issued before the consumers' staging barrier
          // (double buffer: a copier writing tile k + 1's rows has passed tile k + 1's barrier, so every copier is done
          // with tile k - 1's buffer).
          for (int i = ct; i < nv; i += 96) zs[i] = (int)slot_of[row0 + i];
          asm volatile("bar.sync 4, 96;" ::: "memory");
        }
        asm volatile("bar.sync 2, %0;" ::"n"(NC + 96) : "memory");
        if (mode == 1) {
          // The consumers staged gate and up rounded to BF16 (SGLang's GEMM1 output is BF16); SiLU and the product
          // happen here.
          const int CW = 64 * mb, ST = mb == 1 ? 136 : 264, G8 = CW / 8;  // output columns, staging stride
          for (int i = ct; i < nv * G8; i += 96) {
            const int tok = i >> (2 + mb), c8 = (i & (G8 - 1)) * 8;  // G8 = 8 mb, a power of two
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
            const int tok = i >> (3 + mb), c8 = (i & (R8 - 1)) * 8;  // R8 = 16 mb, a power of two
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
          // GEMM2 tiles stored per expert, for the weighted sum that reads y while this kernel runs.
          __threadfence();
          asm volatile("bar.sync 4, 96;" ::: "memory");
          if (ct == 0) atomicAdd(&cnt2[tl.x & 255], 1 + (tl.y >> 16));  // a remainder tile counts as two
        }
        asm volatile("bar.arrive 3, %0;" ::"n"(NC + 96) : "memory");
      }
    }
    return;
  }
  asm volatile("setmaxnreg.inc.sync.aligned.u32 232;");
  GDC_WAIT();
  // A programmatic dependent (the weighted sum, which waits on cnt2 itself) may launch at once; its CTAs only fit on
  // SMs this kernel's CTAs have left.
  if (cnt2 && tid == 0) asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
  uint32_t gch = 0;  // unsigned: % NS and / NS are a mask and a shift
  for (int k = 0;; k++) {
    mbar_wait(btq + 8 * (k & 7), (k >> 3) & 1);
    const int t = s_tq[k & 7];
    if (t < 0) break;
    const int4 tl = s_tl[k & 7];
    const int mode = (tl.x >> 8) & 15, e = tl.x & 255, mbv = (tl.x >> 12) & 15, wv = tl.x >> 16;
    // One tile of layout MB at MMA width W, each (MB, W) on its own accumulator array (ptxas serializes the wgmmas
    // otherwise, warning C7512).
    auto tile = [&](auto MBc, auto Wc) {
      constexpr int MB = decltype(MBc)::value, W = decltype(Wc)::value;
      const int ubase = MB == 2 ? (warp < 4 ? 0 : 8) + (warp % 4) : warp;
      constexpr int AW = (W + 15) / 16 * 16;  // accumulator width; W = 8 stages a pair of n8 blocks, the extra one stays 0
      // MB 2 W 128 tiles of 65-120 tokens issue a narrower MMA (nw: tokens rounded up to 8, at least 72) on the same
      // W 128 accumulators
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
        if constexpr (MB == 1) {  // wait for this chunk's wgmmas and free its stage now (one chunk more lookahead)
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
        if constexpr (true) {  // a narrow-MMA step waits for this chunk's wgmmas and frees its stage now (else: unused)
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
        if constexpr (true) {  // a narrow-MMA step waits for this chunk's wgmmas and frees its stage now (else: unused)
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
        if constexpr (true) {  // a narrow-MMA step waits for this chunk's wgmmas and frees its stage now (else: unused)
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
        if constexpr (true) {  // a narrow-MMA step waits for this chunk's wgmmas and frees its stage now (else: unused)
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
        if constexpr (true) {  // a narrow-MMA step waits for this chunk's wgmmas and frees its stage now (else: unused)
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
        if constexpr (true) {  // a narrow-MMA step waits for this chunk's wgmmas and frees its stage now (else: unused)
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
        if constexpr (true) {  // a narrow-MMA step waits for this chunk's wgmmas and frees its stage now (else: unused)
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
        constexpr int NT = W / 8;  // n8 blocks
        {
          // At most MX_MS tokens: mma.sync m16n8k16 per n8 block (no wgmma), same stages and release order as the pair loop.
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
                const uint32_t xn = xr + n * 1024;  // token rows 8 n + lrow: the 128 B swizzle repeats every 8 rows
                ldsm4(xn + ((lblk ^ lrow) << 4), bf[n][0], bf[n][1], bf[n][2], bf[n][3]);
                ldsm4(xn + (((lblk + 4) ^ lrow) << 4), bf[n][4], bf[n][5], bf[n][6], bf[n][7]);
              }
              const uint8_t* wb = sp0 + s * SB + j * MX_PAIR + 8192;
  #pragma unroll
              for (int b = 0; b < 2; b++) {
                const uint8_t* up = wb + (ubase + 4 * b) * UNIT2;
                const uint4 nb = *reinterpret_cast<const uint4*>(up + lane * 16);
                const uint32_t S = sc_bf16(*reinterpret_cast<const uint16_t*>(up + 512 + lane * 2), 0x4140);  // this lane's scale pair
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
            __syncwarp();  // mma.sync read this stage synchronously, so it is free now
            // The last ldmatrix of the stage can still be outstanding here (ptxas issues the arrive before the HMMA
            // that consumes it); the proxy fence makes the release wait for the reads.
            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
            if (lane == 0) mbar_arrive(bempty + 8 * s);
          }
        }
      } else if constexpr (MB == 2 && W <= 64) {
        // Two chunks per stage (see the producer). Per stage: both chunks' wgmmas are issued (the second chunk decodes
        // while the first one's run), then wait 0 and the stage is released. nwm is a narrower MMA width inside the
        // W 64 (40 / 48 / 56) or W 32 (24) accumulators, each with its own copy of the loop.
        const int nwm = W == 64 ? (tl.w <= 40 ? 40 : tl.w <= 48 ? 48 : tl.w <= 56 ? 56 : 64)
                                : W == 32 ? (tl.w <= 24 ? 24 : 32) : W;
        (void)nwm;
        if constexpr (W == 64) {
          if ((Z_MIDSET & 1) && nwm == 40) {  // MMA width nwm, own pair loop
            for (int c = 0; c < C; c += 2, gch++) {
              const int s = gch % NS;
              mbar_wait(bfull + 8 * s, (gch / NS) & 1);
              chunk_mma<40, W, MB, MOE_ILN>(acc, ra, sp0 + s * SB + 8192, ubase, lane, sb + s * SB);
              wg_commit();
              chunk_mma<40, W, MB, MOE_ILN>(acc, rb_, sp0 + s * SB + MX_PAIR + 8192, ubase, lane, sb + s * SB + MX_PAIR);
              wg_commit();
              wg_wait<0>();  // both chunks done: release the stage
              __syncwarp();
              if (lane == 0) mbar_arrive(bempty + 8 * s);
            }
          } else if ((Z_MIDSET & 2) && nwm == 48) {  // MMA width nwm, own pair loop
            for (int c = 0; c < C; c += 2, gch++) {
              const int s = gch % NS;
              mbar_wait(bfull + 8 * s, (gch / NS) & 1);
              chunk_mma<48, W, MB, MOE_ILN>(acc, ra, sp0 + s * SB + 8192, ubase, lane, sb + s * SB);
              wg_commit();
              chunk_mma<48, W, MB, MOE_ILN>(acc, rb_, sp0 + s * SB + MX_PAIR + 8192, ubase, lane, sb + s * SB + MX_PAIR);
              wg_commit();
              wg_wait<0>();  // both chunks done: release the stage
              __syncwarp();
              if (lane == 0) mbar_arrive(bempty + 8 * s);
            }
          } else if ((Z_MIDSET & 4) && nwm == 56) {  // MMA width nwm, own pair loop
            for (int c = 0; c < C; c += 2, gch++) {
              const int s = gch % NS;
              mbar_wait(bfull + 8 * s, (gch / NS) & 1);
              chunk_mma<56, W, MB, MOE_ILN>(acc, ra, sp0 + s * SB + 8192, ubase, lane, sb + s * SB);
              wg_commit();
              chunk_mma<56, W, MB, MOE_ILN>(acc, rb_, sp0 + s * SB + MX_PAIR + 8192, ubase, lane, sb + s * SB + MX_PAIR);
              wg_commit();
              wg_wait<0>();  // both chunks done: release the stage
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
              wg_wait<0>();  // both chunks done: release the stage
              __syncwarp();
              if (lane == 0) mbar_arrive(bempty + 8 * s);
            }
          }
        } else if constexpr (W == 32) {
          if ((Z_MIDSET & 8) && nwm == 24) {  // MMA width nwm, own pair loop
            for (int c = 0; c < C; c += 2, gch++) {
              const int s = gch % NS;
              mbar_wait(bfull + 8 * s, (gch / NS) & 1);
              chunk_mma<24, W, MB, MOE_ILN>(acc, ra, sp0 + s * SB + 8192, ubase, lane, sb + s * SB);
              wg_commit();
              chunk_mma<24, W, MB, MOE_ILN>(acc, rb_, sp0 + s * SB + MX_PAIR + 8192, ubase, lane, sb + s * SB + MX_PAIR);
              wg_commit();
              wg_wait<0>();  // both chunks done: release the stage
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
              wg_wait<0>();  // both chunks done: release the stage
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
              wg_wait<0>();  // both chunks done: release the stage
              __syncwarp();
              if (lane == 0) mbar_arrive(bempty + 8 * s);
            }
        }
      } else {
        if (false) {
        } else if ((Z_M2SET & 1) && MB == 2 && W == 128 && nw == 80) {  // MMA width nw on W 128 accumulators, own loop
          for (int c = 0; c < C; c += 2) {
            step80(c, ra);
            step80(c + 1, rb_);
          }
        } else if ((Z_M2SET & 2) && MB == 2 && W == 128 && nw == 96) {  // MMA width nw on W 128 accumulators, own loop
          for (int c = 0; c < C; c += 2) {
            step96(c, ra);
            step96(c + 1, rb_);
          }
        } else if ((Z_M2SET & 4) && MB == 2 && W == 128 && nw == 112) {  // MMA width nw on W 128 accumulators, own loop
          for (int c = 0; c < C; c += 2) {
            step112(c, ra);
            step112(c + 1, rb_);
          }
        } else if (MB == 2 && W == 128 && nw == 72) {  // MMA width nw on W 128 accumulators, own loop
          for (int c = 0; c < C; c += 2) {
            step72(c, ra);
            step72(c + 1, rb_);
          }
        } else if (MB == 2 && W == 128 && nw == 88) {  // MMA width nw on W 128 accumulators, own loop
          for (int c = 0; c < C; c += 2) {
            step88(c, ra);
            step88(c + 1, rb_);
          }
        } else if (MB == 2 && W == 128 && nw == 104) {  // MMA width nw on W 128 accumulators, own loop
          for (int c = 0; c < C; c += 2) {
            step104(c, ra);
            step104(c + 1, rb_);
          }
        } else if (MB == 2 && W == 128 && nw == 120) {  // MMA width nw on W 128 accumulators, own loop
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
      // Only step()'s lagged release (MB 2 at full width) leaves the tile's last stage to free here.
      if (MB != 1 && !(MB == 2 && W <= 64 && W >= 32 && !MOE_PIPE2) && !(MB == 2 && W < 32) && nw == W && lane == 0) mbar_arrive(bempty + 8 * ((gch - 1) % NS));
      const float sc = s_gs[mode - 1][e] * 9.223372036854775808e18f;  // global scale * 2^63 (the decode's offset)
      const int lr = (warp % 4) * 16 + g;
      if (k > 0) asm volatile("bar.sync 3, %0;" ::"n"(NC + 96) : "memory");  // the copiers are done with the staging
      // Staging [tok][ST] BF16 by stmatrix.x4.trans (as in moe_fused's epilogue), GEMM1 as gate and up (SiLU and
      // the product move to the copiers, off the consumers' path). Columns: MB = 1 GEMM1 gate warp * 8, up 64 + warp * 8;
      // MB = 2 GEMM1 gate (warp / 4) * 64 + (warp % 4) * 16 (+ 8), up the same + 128; GEMM2 the tile's weight row.
      // Per stmatrix: blocks (tokens 8j.., rows g), (8j.., g + 8), (8j + 8.., g), (8j + 8.., g + 8); lane 8i + t gives
      // row t of block i.
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

// moe_mixed on clusters of two CTAs (launched for 16384+ routes, 2048 to 8191 tokens). The pair works on the same
// token tile: CTA rank r takes weight row blocks 2 q + r, so every claim covers two tiles. Rank 0 claims pairs from the
// counter and forwards each claim to rank 1 (remote store, release / acquire arrive). The token rows are multicast:
// each CTA issues half of a stage's token boxes (or rank 0 the single box) to both CTAs, and loads its own weights. A
// stage is free again only when the consumers of both CTAs are done with it (empty barriers count 2 NW arrives: each
// consumer warp arrives locally and on the peer). No PDL wait overlap here: the plan is read after griddepcontrol.wait.
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
  __shared__ int4 s_tl[8];                       // (expert | GEMM << 8 | mb << 12 | width << 16, row block, first row, tokens)
  __shared__ int s_tj[8];
  __shared__ __align__(8) uint64_t tbars[8];
  __shared__ float s_gs[2][32];
  __shared__ int z0_slot[2][256];  // a GEMM2 tile's destination rows (route slots), double-buffered
  constexpr bool MC = true;
  __shared__ int s_claim[8];  // claims forwarded by rank 0 (rank 1 reads them)
  __shared__ __align__(8) uint64_t cbars[8], peer_ready[NS];  // claim slot filled; peer_ready is initialized, not used
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
    s_map2[tid] = meta[228 + tid] >> 16;  // GEMM2 claim rank -> plan position
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
  // A programmatic dependent (the weighted sum, which waits on cnt2 itself) may launch at once; its CTAs only fit on
  // SMs this kernel's CTAs have left.
  if (cnt2 && tid == 0) asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
  const int total1 = s_toff[0][32], total = total1 + s_toff[1][32];
  const uint32_t btq = (uint32_t)__cvta_generic_to_shared(tbars);
  // info = ntt | mb << 8 | tsz << 12 | r << 21. With r > 0 the expert's last r tokens are one MB 2 tile
  // (token tile k1 = ntt - 1) after k1 MB 1 tiles of tsz; its GEMM1 / GEMM2 tiles follow the MB 1 ones.
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
    if (lt < k1 * per) {  // t is even: (lt >> 1) numbers the pair's tiles, rank r takes row block 2 q + r
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
    if (warp == NW) {  // producer warp, converged; it also claims (rank 0) or receives (rank 1) the tile pairs
      uint32_t gch = 0;  // unsigned: % NS and / NS are a mask and a shift
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
          if (k == 0) next_pair = claim(0);  // one claim ahead, so rank 1 never waits on rank 0's issue loop
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
        const CUtensorMap* bm = mb == 2 && W <= 32 ? &bmx.m[mode - 1][W == 8 ? 6 : W == 16 ? 7 : 8]  // narrow boxes of exactly W rows
                                                   : mode == 1 ? &bmap1 : &bmap2;
        const CUtensorMap* wm = &bmx.w[mode - 1][mb - 1];
        const CUtensorMap* wm40 = mode == 1 ? &x65w40a : &x65w40b;  // MB 2 weights, boxes of two chunks
        const int wr0 = (int)(((mode == 1 ? (size_t)e * ((64 >> (mb - 1))) : (size_t)e * ((32 >> (mb - 1)))) + rb) * C * (WB / 512));
        (void)wp;
        if (mode == 2) {
          while (ld_acquire(&cnt1[jt]) < (64 >> (mb - 1))) __nanosleep(128);
          asm volatile("fence.proxy.async.global;" ::: "memory");
        }
        const int nbox = (W + 63) / 64;
        // One box of Wb rows when W >= 128 (Wb = W rounded up to 32: 128..256), else 64-row boxes.
        const bool one = W >= MX_OB && W >= 128;
        const int Wb = (W + 31) & ~31;  // box rows (a W in steps of 8 loads the next multiple of 32)
        const CUtensorMap* bmw = &bmx.m[mode - 1][one ? (Wb - 96) / 32 : 0];
        const int xbytes = one ? Wb * 128 : nbox * 8192;
        if (mb == 2 && W <= 64) {
          constexpr int WB2 = wbytes(2);
          const int z0_ex = 2 * (W * 128 + WB2);  // bytes per stage: two chunks of W token rows and MB 2 weights
          // Narrow tile: two chunks per stage (x0 at 0, w0 w1 at 8192 as one box, x1 at X65_X1), so the ring holds twice
          // as many chunks ahead of these short, load-heavy chunks. Rank r multicasts chunk c + r's token rows.
          for (int c = 0; c < C; c += 2, gch++) {
            const int s = gch % NS;
            if (gch >= NS) ft_wait_cluster(bempty + 8 * s, ((gch / NS) - 1) & 1);
            if (x64_elect()) {
              mbar_expect(bfull + 8 * s, z0_ex);
              tma_2d_mc(sb + s * SB + (crank ? X65_X1 : 0u), bm, (c + (int)crank) * 64, row0, bfull + 8 * s, (uint16_t)3);
              tma_2d(sb + s * SB + 8192, wm40, 0, wr0 + c * (WB2 / 512), bfull + 8 * s);  // both chunks' weights
            }
          }
        } else {
          for (int c = 0; c < C; c++, gch++) {
            const int s = gch % NS;
            if (gch >= NS) ft_wait_cluster(bempty + 8 * s, ((gch / NS) - 1) & 1);
            if (x64_elect()) {
              mbar_expect(bfull + 8 * s, xbytes + WB);
              tma_2d(sb + s * SB + xo, wm, 0, wr0 + c * (WB / 512), bfull + 8 * s);  // own weights first
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
          // The tile's destination rows, one load per token, before the staging barrier.
          for (int i = ct; i < nv; i += 96) zs[i] = (int)slot_of[row0 + i];
          asm volatile("bar.sync 4, 96;" ::: "memory");
        }
        asm volatile("bar.sync 2, %0;" ::"n"(NC + 96) : "memory");
        if (mode == 1) {
          // The consumers staged gate and up rounded to BF16 (SGLang's GEMM1 output is BF16); SiLU and the product
          // happen here.
          const int CW = 64 * mb, ST = mb == 1 ? 136 : 264, G8 = CW / 8;  // output columns, staging stride
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
          // GEMM2 tiles stored per expert, for the weighted sum that reads y while this kernel runs.
          __threadfence();
          asm volatile("bar.sync 4, 96;" ::: "memory");
          if (ct == 0) atomicAdd(&cnt2[tl.x & 255], 1 + (tl.y >> 16));  // a remainder tile counts as two
        }
        asm volatile("bar.arrive 3, %0;" ::"n"(NC + 96) : "memory");
      }
    }
    ft_cluster_sync();
    return;
  }
  asm volatile("setmaxnreg.inc.sync.aligned.u32 224;");
  uint32_t gch = 0;  // unsigned: % NS and / NS are a mask and a shift
  for (int k = 0;; k++) {
    mbar_wait(btq + 8 * (k & 7), (k >> 3) & 1);
    const int t = s_tq[k & 7];
    if (t < 0) break;
    const int4 tl = s_tl[k & 7];
    const int mode = (tl.x >> 8) & 15, e = tl.x & 255, mbv = (tl.x >> 12) & 15, wv = tl.x >> 16;
    // One tile of layout MB at MMA width W, each (MB, W) on its own accumulator array (ptxas serializes the wgmmas
    // otherwise, warning C7512).
    auto tile = [&](auto MBc, auto Wc) {
      constexpr int MB = decltype(MBc)::value, W = decltype(Wc)::value;
      const int ubase = MB == 2 ? (warp < 4 ? 0 : 8) + (warp % 4) : warp;
      constexpr int AW = (W + 15) / 16 * 16;  // accumulator width; W = 8 stages a pair of n8 blocks, the extra one stays 0
      // MB 2 W 128 tiles of up to 80 / 96 / 112 tokens issue an MMA of that width on the same W 128 accumulators
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
        if constexpr (MB == 1) {  // wait for this chunk's wgmmas and free its stage now (one chunk more lookahead)
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
        if constexpr (true) {  // a narrow-MMA step waits for this chunk's wgmmas and frees its stage now (else: unused)
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
        if constexpr (true) {  // a narrow-MMA step waits for this chunk's wgmmas and frees its stage now (else: unused)
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
        if constexpr (true) {  // a narrow-MMA step waits for this chunk's wgmmas and frees its stage now (else: unused)
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
        constexpr int NT = W / 8;  // n8 blocks
        {
          // At most MX_MS tokens: mma.sync m16n8k16 per n8 block (no wgmma), same stages and release order as the pair loop.
          const int lrow = lane & 7, lblk = lane >> 3;
          for (int c = 0; c < C; c += 2, gch++) {
            const int s = gch % NS;
            mbar_wait(bfull + 8 * s, (gch / NS) & 1);
  #pragma unroll
            for (int j = 0; j < 2; j++) {
              const uint32_t xr = sb + s * SB + j * X65_X1 + lrow * 128;  // chunk j's token rows
              uint32_t bf[NT][8];
  #pragma unroll
              for (int n = 0; n < NT; n++) {
                const uint32_t xn = xr + n * 1024;  // token rows 8 n + lrow: the 128 B swizzle repeats every 8 rows
                ldsm4(xn + ((lblk ^ lrow) << 4), bf[n][0], bf[n][1], bf[n][2], bf[n][3]);
                ldsm4(xn + (((lblk + 4) ^ lrow) << 4), bf[n][4], bf[n][5], bf[n][6], bf[n][7]);
              }
              const uint8_t* wb = sp0 + s * SB + 8192 + j * (X65_W1 - 8192);  // chunk j's weights
  #pragma unroll
              for (int b = 0; b < 2; b++) {
                const uint8_t* up = wb + (ubase + 4 * b) * UNIT2;
                const uint4 nb = *reinterpret_cast<const uint4*>(up + lane * 16);
                const uint32_t S = sc_bf16(*reinterpret_cast<const uint16_t*>(up + 512 + lane * 2), 0x4140);  // this lane's scale pair
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
            __syncwarp();  // mma.sync read this stage synchronously, so it is free now
            // The last ldmatrix of the stage can still be outstanding here (ptxas issues the arrive before the HMMA
            // that consumes it); the proxy fence makes the release wait for the reads.
            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
            if (lane == 0) { mbar_arrive(bempty + 8 * s); ft_arrive_remote(peer_empty + 8 * s); }
          }
        }
      } else if constexpr (MB == 2 && W <= 64) {
        // Two chunks per stage (see the producer). Per stage: both chunks' wgmmas are issued (the second chunk decodes
        // while the first one's run), then wait 0 and the stage is released. nwm is a narrower MMA width inside the
        // W 64 (40 / 48 / 56) or W 32 (24) accumulators, each with its own copy of the loop.
        const int nwm = W == 64 ? (tl.w <= 40 ? 40 : tl.w <= 48 ? 48 : tl.w <= 56 ? 56 : 64)
                                : W == 32 ? (tl.w <= 24 ? 24 : 32) : W;
        (void)nwm;
        if constexpr (W == 64) {
          if ((Z_MIDSET & 1) && nwm == 40) {  // MMA width nwm, own pair loop
            for (int c = 0; c < C; c += 2, gch++) {
              const int s = gch % NS;
              mbar_wait(bfull + 8 * s, (gch / NS) & 1);
              chunk_mma<40, W, MB, MOE_ILN>(acc, ra, sp0 + s * SB + 8192, ubase, lane, sb + s * SB);
              wg_commit();
              chunk_mma<40, W, MB, MOE_ILN>(acc, rb_, sp0 + s * SB + X65_W1, ubase, lane, sb + s * SB + X65_X1);
              wg_commit();
              wg_wait<0>();  // both chunks done: release the stage
              __syncwarp();
              if (lane == 0) { mbar_arrive(bempty + 8 * s); ft_arrive_remote(peer_empty + 8 * s); }
            }
          } else if ((Z_MIDSET & 2) && nwm == 48) {  // MMA width nwm, own pair loop
            for (int c = 0; c < C; c += 2, gch++) {
              const int s = gch % NS;
              mbar_wait(bfull + 8 * s, (gch / NS) & 1);
              chunk_mma<48, W, MB, MOE_ILN>(acc, ra, sp0 + s * SB + 8192, ubase, lane, sb + s * SB);
              wg_commit();
              chunk_mma<48, W, MB, MOE_ILN>(acc, rb_, sp0 + s * SB + X65_W1, ubase, lane, sb + s * SB + X65_X1);
              wg_commit();
              wg_wait<0>();  // both chunks done: release the stage
              __syncwarp();
              if (lane == 0) { mbar_arrive(bempty + 8 * s); ft_arrive_remote(peer_empty + 8 * s); }
            }
          } else if ((Z_MIDSET & 4) && nwm == 56) {  // MMA width nwm, own pair loop
            for (int c = 0; c < C; c += 2, gch++) {
              const int s = gch % NS;
              mbar_wait(bfull + 8 * s, (gch / NS) & 1);
              chunk_mma<56, W, MB, MOE_ILN>(acc, ra, sp0 + s * SB + 8192, ubase, lane, sb + s * SB);
              wg_commit();
              chunk_mma<56, W, MB, MOE_ILN>(acc, rb_, sp0 + s * SB + X65_W1, ubase, lane, sb + s * SB + X65_X1);
              wg_commit();
              wg_wait<0>();  // both chunks done: release the stage
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
              wg_wait<0>();  // both chunks done: release the stage
              __syncwarp();
              if (lane == 0) { mbar_arrive(bempty + 8 * s); ft_arrive_remote(peer_empty + 8 * s); }
            }
          }
        } else if constexpr (W == 32) {
          if ((Z_MIDSET & 8) && nwm == 24) {  // MMA width nwm, own pair loop
            for (int c = 0; c < C; c += 2, gch++) {
              const int s = gch % NS;
              mbar_wait(bfull + 8 * s, (gch / NS) & 1);
              chunk_mma<24, W, MB, MOE_ILN>(acc, ra, sp0 + s * SB + 8192, ubase, lane, sb + s * SB);
              wg_commit();
              chunk_mma<24, W, MB, MOE_ILN>(acc, rb_, sp0 + s * SB + X65_W1, ubase, lane, sb + s * SB + X65_X1);
              wg_commit();
              wg_wait<0>();  // both chunks done: release the stage
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
              wg_wait<0>();  // both chunks done: release the stage
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
              wg_wait<0>();  // both chunks done: release the stage
              __syncwarp();
              if (lane == 0) { mbar_arrive(bempty + 8 * s); ft_arrive_remote(peer_empty + 8 * s); }
            }
        }
      } else {
        if (false) {
        } else if ((Z_M2SET & 1) && MB == 2 && W == 128 && nw == 80) {  // MMA width nw on W 128 accumulators, own loop
          for (int c = 0; c < C; c += 2) {
            step80(c, ra);
            step80(c + 1, rb_);
          }
        } else if ((Z_M2SET & 2) && MB == 2 && W == 128 && nw == 96) {  // MMA width nw on W 128 accumulators, own loop
          for (int c = 0; c < C; c += 2) {
            step96(c, ra);
            step96(c + 1, rb_);
          }
        } else if ((Z_M2SET & 4) && MB == 2 && W == 128 && nw == 112) {  // MMA width nw on W 128 accumulators, own loop
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
      // Only step()'s lagged release (MB 2 at full width) leaves the tile's last stage to free here, in both CTAs.
      if (MB != 1 && !(MB == 2 && W <= 64 && W >= 32 && !MOE_PIPE2) && !(MB == 2 && W < 32) && nw == W && lane == 0) { mbar_arrive(bempty + 8 * ((gch - 1) % NS)); ft_arrive_remote(peer_empty + 8 * ((gch - 1) % NS)); }
      const float sc = s_gs[mode - 1][e] * 9.223372036854775808e18f;  // global scale * 2^63 (the decode's offset)
      const int lr = (warp % 4) * 16 + g;
      if (k > 0) asm volatile("bar.sync 3, %0;" ::"n"(NC + 96) : "memory");  // the copiers are done with the staging
      // Staging [tok][ST] BF16 by stmatrix.x4.trans (as in moe_fused's epilogue), GEMM1 as gate and up (SiLU and
      // the product move to the copiers, off the consumers' path). Columns: MB = 1 GEMM1 gate warp * 8, up 64 + warp * 8;
      // MB = 2 GEMM1 gate (warp / 4) * 64 + (warp % 4) * 16 (+ 8), up the same + 128; GEMM2 the tile's weight row.
      // Per stmatrix: blocks (tokens 8j.., rows g), (8j.., g + 8), (8j + 8.., g), (8j + 8.., g + 8); lane 8i + t gives
      // row t of block i.
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
  const int dyn = ctas > 0 ? 1 : 0;  // negative ctas: static round-robin tiles
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
  CUtensorMap map[4];  // the sorted token rows [R][4096] BF16 (a, inter, a, inter): boxes of 64 k x 64 rows, then x 256 rows
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

int* g_cnt2 = nullptr;  // per-expert GEMM2 tile counters for the next launch (nullptr = off), set by moe_set_cnt2
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
  MxMaps mx;  // map[0..1]: 64-row boxes of a / inter; mx.m[a | inter][box rows 64, 128, 160, 192, 224, 256]
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
  for (int i = 0; i < 6; i++) {  // narrow token boxes of exactly 8 / 16 / 32 rows, [a | inter]
    torch::Tensor& b = i < 3 ? a : inter;
    const cuuint64_t dims[2] = {4096, (cuuint64_t)b.size(0)};
    const cuuint64_t strides[1] = {4096 * 2};
    const cuuint32_t box[2] = {64, 8u << (i % 3)}, estr[2] = {1, 1};
    encode(&mx.m[i / 3][6 + i % 3], CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, b.data_ptr(), dims, strides, box, estr,
           CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_128B,
           CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  }
  for (int i = 0; i < 4; i++) {  // [GEMM][mb - 1]: w1 (MB 2), w1w (MB 1), w2 (MB 2), w2w (MB 1)
    torch::Tensor& wt = i == 0 ? w1 : i == 1 ? w1w : i == 2 ? w2 : w2w;
    const int mbi = i % 2 == 0 ? 2 : 1;
    const cuuint64_t dims[2] = {256, (cuuint64_t)(wt.numel() * wt.element_size() / 512)};
    const cuuint64_t strides[1] = {512};
    const cuuint32_t box[2] = {256, (cuuint32_t)(1 * (mbi == 2 ? 18 : 9))}, estr[2] = {1, 1};  // one chunk: 18 rows (9216 B) at MB 2, 9 at MB 1
    CUresult r = encode(&mx.w[i / 2][mbi - 1], CU_TENSOR_MAP_DATA_TYPE_UINT16, 2, wt.data_ptr(), dims, strides, box, estr,
           CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_NONE, CU_TENSOR_MAP_L2_PROMOTION_NONE,
           CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    if (r != CUDA_SUCCESS) printf("pr_mk3: weight map %d encode failed %d\n", i, (int)r);
  }
  CUtensorMap x65w40[2];
  for (int i = 0; i < 2; i++) {  // moe_mixed_mc's MB 2 weights (w1, w2), boxes of two chunks
    torch::Tensor& wt = i == 0 ? w1 : w2;
    const cuuint64_t dims[2] = {256, (cuuint64_t)(wt.numel() * wt.element_size() / 512)};
    const cuuint64_t strides[1] = {512};
    const cuuint32_t box[2] = {256, (cuuint32_t)(2 * 2 * NW * UNIT2 / 512)}, estr[2] = {1, 1};  // 36 rows of 512 B
    CUresult r = encode(&x65w40[i], CU_TENSOR_MAP_DATA_TYPE_UINT16, 2, wt.data_ptr(), dims, strides, box, estr,
           CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_NONE, CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
           CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    // "x65: w40 map" names the x65w40 tensor maps above.
    if (r != CUDA_SUCCESS) printf("x65: w40 map %d encode failed %d\n", i, (int)r);
  }
  if (y.size(0) >= 2048 * 8) {  // 2048+ tokens: CTA pairs
    cudaLaunchAttribute attrs[1];  // plain cluster launch, no programmatic dependency
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
}  // namespace

void moe_set_cnt2(int64_t p) { g_cnt2 = reinterpret_cast<int*>(p); }

// cnt1, meta[196..] (tn 256) or meta[260..] (mixed): per token tile, the GEMM1 blocks written (zeroed with meta).
void moe_fused_run(torch::Tensor w1, torch::Tensor w2, torch::Tensor w1w, torch::Tensor w2w, torch::Tensor a,
                   torch::Tensor inter, torch::Tensor meta, torch::Tensor gs1, torch::Tensor gs2, torch::Tensor y,
                   int64_t ctas, int64_t tn, torch::Tensor slot_of) {
  const long* so = slot_of.data_ptr<long>();
  // The wide layout (8192+ tokens) runs the GEMMs as two launches (one launch measured 1% slower there).
  if (tn == 256) {
    launch_fused<256, 1>(w1w, w2w, a, inter, meta, gs1, gs2, y, (int)ctas, so, 1);
    launch_fused<256, 1>(w1w, w2w, a, inter, meta, gs1, gs2, y, (int)ctas, so, 2);
  } else {
    launch_mixed(w1, w2, w1w, w2w, a, inter, meta, gs1, gs2, y, (int)ctas, so);
  }
}

// The route list, grid-wide: (1) per-CTA histograms of the routes' local experts added into meta's counts (meta
// comes zeroed), (2) one CTA turns the counts into starts, both GEMMs' tile offsets for token tile tn, scatter
// cursors and zeroed tile counters, (3) a grid-wide scatter of route ids to their sorted positions. Warp-aggregated
// atomics; the order within an expert is arbitrary (each route is computed alone). moe_build runs these for
// _experts_moe; the layer's own routing path builds the same meta in routing_front.cu.
// meta (int32): cnt[32] | starts[32] | toff1[33] | toff2[33] | counters[2] | cursors[32] | perm[32] from 164, then
// cnt1 from 196 (plan_k) or info[32] from 196, jt0[32] from 228 and cnt1 from 260 (plan2_k).
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
// meta[164..195]: the experts by route count, most first (ties by id); the GEMMs number tiles in that order, with
// toff1 / toff2 by position. Rows stay sorted by expert id (starts).
__global__ void plan_k(int* __restrict__ meta, int tn, int rb1, int rb2) {
  __shared__ int cnt[32];
  const int e = threadIdx.x;
  const int n = meta[e];
  cnt[e] = n;
  __syncwarp();
  int pos = 0;
  for (int j = 0; j < 32; j++) pos += cnt[j] > n || (cnt[j] == n && j < e);
  meta[164 + pos] = e;
  // starts and scatter cursors by id
  int s = n;
  for (int o = 1; o < 32; o <<= 1) { const int v = __shfl_up_sync(0xffffffffu, s, o); if (e >= o) s += v; }
  meta[32 + e] = s - n; meta[132 + e] = s - n;
  // tile offsets by position: the expert at position e is the one with rank e
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
// Cycles per K chunk of one CTA tile by layout and MMA width (measured per tile at T = 753 and 3012; widths 96,
// 160, 192, 224 interpolated). MB = 2 covers 256 weight rows, MB = 1 128.
__device__ __forceinline__ int mx_cost(int mb, int W) {
  if (mb == 2) return W <= 32 ? 901 : W <= 64 ? 1027 : W <= 96 ? 1180 : 1334;
  return W <= 64 ? 715 : W <= 128 ? 883 : W <= 160 ? 940 : W <= 192 ? 1000 : W <= 224 ? 1060 : 1118;
}
// The mixed plan (tokens < 8192): per expert the layout (mb) and token tiling (ntt tiles of tsz, a multiple of 32)
// with the lower modelled cost over 256 weight rows; then by position (most routes first) the info word, both GEMMs'
// tile offsets and the token tile base. meta: info[196 + pos], jt0[228 + pos], cnt1 from 260.
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
// Rows of x for the sorted local routes (token = route / 8): one CTA per row, grid-stride.
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
  else plan2_k<<<1, 32, 0, st>>>(meta.data_ptr<int>(), tn < 0 ? (int)-tn : 0);  // tn -1 / -2: force that layout
  scatter_k<<<g, 1024, 0, st>>>(ids.data_ptr<int>(), R, order.data_ptr<long>(), meta.data_ptr<int>());
}
void moe_gather(torch::Tensor x, torch::Tensor order, torch::Tensor meta, torch::Tensor a) {
  gather2_k<<<2048, 256, 0, at::cuda::getCurrentCUDAStream()>>>((const uint4*)x.data_ptr(), order.data_ptr<long>(),
                                                               meta.data_ptr<int>(), (uint4*)a.data_ptr(), (int)(x.size(1) / 8));
}
