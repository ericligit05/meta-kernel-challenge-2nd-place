// Sliding-window prefill attention for the sliding layers (sl_attn_k) and the gather that builds its K / V copies
// (sl_gather_k). One attention kernel per call: no prepare kernel and no combine kernel.
//
// This build (SK_KS 1) never splits the key range: one CTA per work item, 160 threads (a consumer warpgroup and a
// producer warp), up to 3 CTAs per SM. It is the same kernel as attention_sliding_split.cu without the cluster split.
// sl_attn_ng2 also takes a second FP8 query plane (q_lo, q = q + q_lo), staged in shared memory by TMA and added to S
// by two more QK wgmmas per tile.
//
// Work layout: FP8 E4M3 q / K / V, head dim 64, 16 q heads, 2 KV heads, window W. A work item is (request, KV head,
// group of 8 query positions). Its 64 rows (8 positions x the 8 q heads that share the KV head, GQA packed) belong to
// one consumer warpgroup; one producer warp streams 128-key K and V^T tiles through a 3-stage TMA ring. K comes from a
// contiguous [B, S, 2, 64] copy, V from a transposed [B, 2, 64, S] copy whose keys are permuted inside each 32-key
// block (sigma, see sl_gather_k), so the QK accumulator registers pack straight into the A operand of the PV wgmma.
//
// Numerics:
// - S = q K^T accumulates in FP16 (wgmma .f16.e4m3.e4m3).
// - Softmax without exp: each P byte is the low byte of z = A s + B, one f16x2 FMA per two scores, with
//   A = 8 log2(e) softmax_scale (rounded to f16) and B = PV_B - A max. f16 values in [1024, 2048) are spaced 1 apart,
//   so the low byte of z is round(A (s - max)) + 80, clamped below at 0. Read as E4M3, a normal code c is
//   (1 + (c & 7) / 8) 2^((c >> 3) - 7), a piecewise-linear 2^(c / 8 - 7), so P is about 8 exp(softmax_scale (s - max)).
//   The row max gets code 80 (8.0). Codes 1 to 7 are E4M3 subnormals (c 2^-9), and keys more than about 10 octaves
//   below the row max get code 0.
// - When the running max of a row moves, its B moves with it and O is rescaled by exp2(dB / 8).
// - P V accumulates in FP16 (16 half2 registers per thread).
// - No row sum: O stays unnormalized, off by a positive factor per row. The per-head RMSNorm of the fused o_norm_gate
//   epilogue divides that factor out (up to its eps term), so the layer never needs the softmax denominator.
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

// Settings. PV_B: the f16 frame value of the row max (1104 = 1024 + E4M3 code 80). PV16: marks the FP16 P V build
// (the accumulator type is fixed below). SL_NWG: consumer warpgroups per CTA. SL_MINB: CTAs per SM asked of
// __launch_bounds__. SL_ST: TMA ring depth. SK_PDL: programmatic dependent launch. SK_KS 1: no key split. SL_BN: keys
// per tile. SK_BM 1: one 64-row half per warpgroup. Z0_NOL 1 (no row sum) turns off the all-ones V^T rows that SK_O72
// would add. SL_PIPE, SL_MATH and SK_TS are only checked by static_asserts; SX_XCH 0: no split exchange.
#define SL_BN 128
#define SL_MATH 1
#define SK_O72 1
#define SK_BM 1
#define SK_TS 0
#define SX_XCH 0

constexpr int BN = SL_BN, ST = SL_ST;
constexpr int PPC = 8 * SL_NWG * SK_BM;  // query positions per CTA
constexpr int K_BYTES = BN * 64, V_BYTES = 64 * BN, Q_BYTES = PPC * 8 * 64;
// Row sum of a row: always 1, the build is denominator-free (the o_norm_gate RMSNorm cancels the true sum).
#define Z0_RS(o, i) 1.f
constexpr int VS = V_BYTES + ((SK_O72 && !Z0_NOL) ? 8 * BN : 0);  // V^T stage stride (+ 8 ones rows only in row-sum builds)
constexpr int NO = (SK_O72 && !Z0_NOL) ? 36 : 32;                 // FP32 O registers (unused FP32 merge helpers only)
typedef uint32_t OT;
constexpr int NOR = 16;  // FP16 O registers: o[2 jb + r] = row r0 (r 0) or r0 + 8 (r 1), columns 8 jb + 2 tq, + 1
static_assert(Z0_NOL && SK_BM == 1, "PV16: denominator-free one-half build only");
// Split merge (SX_XCH 2, SK_KS > 1 only): an exchange buffer after the barriers, XCHB bytes per consumer thread.
constexpr int XCHB = PV16 ? 48 : 80;  // exchange bytes per consumer thread
constexpr int XB = (SX_XCH == 2 && SK_KS > 1) ? XCHB * SL_NWG * 128 : 0;
constexpr int SMEM = 1024 + 2 * Q_BYTES + ST * (K_BYTES + VS) + 512 + 256 + XB;  // second Q_BYTES: the q_lo tile
// g_sk_ts (set by sl_ts) and g_sx_smcnt are not read by any kernel in this file.
__device__ unsigned long long* g_sk_ts = nullptr;
__device__ unsigned int g_sx_smcnt[256];
static_assert(SK_KS == 1 || SK_O72, "SK_KS needs SK_O72 (sum in the O accumulator)");
static_assert(SK_BM == 1 || (SL_NWG == 1 && SL_PIPE == 3 && SK_O72 && SL_MATH && !SK_TS), "SK_BM 2: NWG 1, PIPE 3 only");
static_assert((SK_KS - 1) * SK_BM * 38 * SL_NWG * 128 * 4 <= ST * (K_BYTES + VS), "SK_BM exchange does not fit");
static_assert((SK_KS - 1) * 38 * SL_NWG * 128 * 4 <= ST * (K_BYTES + VS), "SK_KS exchange does not fit the ring");
constexpr int NKB = BN / 32;   // 32-key blocks per tile
constexpr int NSR = BN / 4;    // f16x2 S registers per thread (m64nBN f16 accumulator)

typedef CUresult (*SlEncodeTiled)(CUtensorMap*, CUtensorMapDataType, cuuint32_t, void*, const cuuint64_t*,
                                  const cuuint64_t*, const cuuint32_t*, const cuuint32_t*, CUtensorMapInterleave,
                                  CUtensorMapSwizzle, CUtensorMapL2promotion, CUtensorMapFloatOOBfill);

struct SlParams {
  __nv_bfloat16* out;     // [T, 16, 64]
  const int* cu_q;        // [B + 1]
  const int* cache_seqlens;  // [B] keys per request including this call's rows
  int S;                  // rows per request in the K / V copies
  int G;                  // position groups (of 8) per request: ceil(max L / 8)
  int W;                  // window
  float a32;              // A = 8 log2(e) softmax_scale, rounded to f16
  const __nv_bfloat16* gate;  // nullptr: unnormalized O; else fused o_norm_gate (per-head RMSNorm x sigmoid(gate))
  int gstride;            // gate row stride (elements)
  float eps;
  int ks;                 // key chunks per work item (cluster size), 1 .. SK_KS
  int has_lo;             // q_lo tile present (mql): two more QK wgmmas per tile
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
// K-major smem descriptors: 64 B rows with 64 B swizzle (8-row groups 512 B apart), 128 B rows with 128 B swizzle.
__device__ __forceinline__ uint64_t desc64(uint32_t addr) {
  return (uint64_t)((addr & 0x3FFFF) >> 4) | ((uint64_t)1 << 16) | ((uint64_t)(512 >> 4) << 32) | ((uint64_t)2 << 62);
}
__device__ __forceinline__ uint64_t desc128(uint32_t addr) {
  return (uint64_t)((addr & 0x3FFFF) >> 4) | ((uint64_t)1 << 16) | ((uint64_t)(1024 >> 4) << 32) | ((uint64_t)1 << 62);
}
__device__ __forceinline__ uint64_t descv(uint32_t addr) { return BN >= 128 ? desc128(addr) : desc64(addr); }
// PV B operand of 32-key block kb: 32 B further along the 128 B (128-key) V^T rows. The BN 256 branch (two 128-key
// boxes 8 KB apart) is not taken, SL_BN is 128.
__device__ __forceinline__ uint64_t pvd(uint64_t dv, int kb) {
  return BN == 256 ? dv + (uint64_t)((kb >> 2) * (8192 >> 4) + 2 * (kb & 3)) : dv + (uint64_t)(2 * kb);
}
// Descriptor of the 512 B all-ones tile (0x38 = 1.0 in E4M3) for a row-sum wgmma. This denominator-free build fills
// the tile but never issues that wgmma.
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

// S += Q K^T over 32 of the 64 head dims (k32), FP16 accumulate; sd 0 starts from zero. The n256 form is not used
// (SL_BN 128). Of the PV forms only the FP16 m64n64 one is issued here; the FP32 and m64n8 row-sum forms serve the
// builds with an FP32 O or a row sum.
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

// Per-thread softmax state for its two rows (r0 and r0 + 8). rmax is unused and rsum ends at 1 (no row sum).
struct SlRow {
  float rmax[2], rb[2], rsum[2];
  int jlo[2], jhi[2];
  uint32_t rmh, rbh;  // f16x2 {row 0, row 1}: running max, B (65504 = no key yet)
};

__device__ __forceinline__ void sl_issue_qk(uint32_t (&s)[NSR], uint64_t dq, uint32_t kaddr, uint64_t dql, int lo) {
#pragma unroll
  for (int i = 0; i < NSR; ++i) s[i] = 0;
  reg_fence(s);
  wg_fence();
  const uint64_t dk = desc64(kaddr);
  wgmma_qk(s, dq, dk, 0);
  wgmma_qk(s, dq + 2, dk + 2, 1);
  if (lo) {  // + q_lo . k (q_lo tile in smem)
    wgmma_qk(s, dql, dk, 1);
    wgmma_qk(s, dql + 2, dk + 2, 1);
  }
  wg_commit();
}

__device__ __forceinline__ void sl_pack(const uint32_t (&sc)[NSR], uint32_t (&pa)[NKB][4]);
// Softmax of one tile: mask (boundary tiles only), running max, new B, and the FP8 codes written over the scores.
// scale[] returns the factor that rescales each row's O.
__device__ __forceinline__ void sl_softmax(uint32_t (&sc)[NSR], uint32_t (&pa)[NKB][4], SlRow& st, int j0, bool edge,
                                           int tq, float a32, float (&scale)[2]) {
  if (edge) {
    // Mask: row r keeps column j when (unsigned)(j - jlo) <= jhi - jlo (two compares and one select per register);
    // masked scores become -inf (0xFC00). edge is then cleared, so the per-half mask after this block never runs.
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
  // B = PV_B - A max for both rows in one f16x2 FMA. O is rescaled by exp2(dB / 8) only when some lane's B moved
  // (warp vote); after the first tiles it rarely does.
  __half2 bx[2];
  {
    const __half2 cur = __hmax2(h2(st.rmh), mm);
    const __half2 bn = __hmin2(__hfma2(cur, __hneg2(__float2half2_rn(a32)), __float2half2_rn((float)PV_B)),
                               __float2half2_rn(65504.f));  // finite sentinel for rows with no key yet
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
  // Codes: z = max(A s + B, 1024). f16 steps by 1 in [1024, 2048), so the low byte of z is the E4M3 code z - 1024
  // (at most PV_B - 1024 = 80); sl_pack keeps only the low bytes. off2 is unused.
  (void)off2;
#pragma unroll
  for (int w = 0; w < NSR; ++w) sc[w] = u32(__hmax2(__hfma2(h2(sc[w]), a2, bx[w & 1]), lo2));
}

// Pack the codes (low byte of each f16) into PV A registers. Per 32-key block kb: a0 = row r0, n8 blocks 4kb and
// 4kb+1; a1 = row r0 + 8; a2 / a3 = the same rows, blocks 4kb+2 and 4kb+3.
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

// PV of the last tile: one FP16 m64n64k32 wgmma per 32-key block (os and dones are unused in this build).
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
  // Warp-uniform branch: after the first tiles the running max rarely moves, so the 16 half2 multiplies are skipped.
  if (__any_sync(0xffffffffu, scale[0] != 1.f || scale[1] != 1.f)) {
    const __half2 s0 = __float2half2_rn(scale[0]), s1 = __float2half2_rn(scale[1]);
#pragma unroll
    for (int i = 0; i < NOR; i += 2) { o[i] = u32(__hmul2(h2(o[i]), s0)); o[i + 1] = u32(__hmul2(h2(o[i + 1]), s1)); }
    st.rsum[0] *= scale[0];
    st.rsum[1] *= scale[1];
  }
}


// Split-merge helpers for an FP32 O (row R of a thread's O = o[4 jb + 2 R + {0, 1}]). Not called in this FP16 build.
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

// Split-merge helpers for the FP16 O. A thread sends its partner the row the partner keeps: 8 half2 values (two
// float4) and that row's B (one float4), 48 bytes by st.async onto the partner's exchange barrier. The merge scales
// both partials to the smaller B (the larger running max) by exp2((m - B) / 8), adds them in FP32 and rounds back to
// half2. A rank that saw no key sends B = 65504, which gives its zero partial weight 0.
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
  uint8_t* sk = base;                         // ST x K tile (1 KB aligned)
  uint8_t* sv = sk + ST * K_BYTES;            // ST x V^T tile (stride VS = V_BYTES, no ones rows)
  uint8_t* sq = sv + ST * VS;                 // Q tile, NWG x 64 rows
  uint8_t* sones = sq + 2 * Q_BYTES;          // 512 B of 0x38 after the q and q_lo tiles (filled but not read)
  uint64_t* bars = reinterpret_cast<uint64_t*>(sones + 512);  // full[ST], empty[ST], Q
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
// SX_C and SK_T mark phase points for optional instrumentation; both expand to nothing.
#define SX_C(k) do { } while (0)

  // Work item: blockIdx.x = ((b * G + g) * 2 + kh) * ks + c; group g = 8 query positions, kh the KV head, c the key
  // chunk (cluster rank) when the key range is split.
  const int ks = SK_KS > 1 ? p.ks : 1;
  const int kc = ks > 1 ? (int)cluster_rank() : 0, item = ks > 1 ? blockIdx.x / ks : blockIdx.x;
  const int kh = item & 1, bg = item >> 1, b = bg / p.G, g = bg - b * p.G;
  const int q0 = p.cu_q[b], Lb = p.cu_q[b + 1] - q0, n = p.cache_seqlens[b];
  const int i0 = g * PPC;
  if (i0 >= Lb) return;  // whole group past this request's queries (uneven q lengths)
  const int W = p.W;
  const int s0 = max(0, n - Lb - (W - 1));    // cache position of copy row 0 (the gather's layout)
  const int ilast = min(i0 + PPC - 1, Lb - 1);
  const int pfirst = n - Lb + i0, plast = n - Lb + ilast;
  const int jlo = max(0, pfirst - W + 1 - s0), jhi = plast - s0;  // copy rows any row of this CTA reads
  const int ta = jlo / BN, nta = jhi / BN - ta + 1;  // all tiles of the item; this CTA takes chunk kc of them
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
    // Producer (one lane): tile 0's K and V, then Q (both waited on before QK(0)), then the rest of the ring; a
    // stage is refilled after every consumer thread has released it (empty barrier).
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
          if (p.has_lo) tma_3d(smem_u32(sq + Q_BYTES), &mql, 0, kh * 8, q0 + i0, qb);  // q_lo on the same Q barrier
        }
      }
    }
    __syncwarp();
    return;
  }

  if (p.gate && tid < 8 * PPC) {  // warm L2 with this CTA's gate rows (read in the epilogue): one 128 B line each
    const int pos = i0 + (tid >> 3);
    if (pos < Lb)
      asm volatile("prefetch.global.L2 [%0];" ::"l"(p.gate + (size_t)(q0 + pos) * p.gstride + (kh * 8 + (tid & 7)) * 64));
  }
  // Consumer warpgroup wg: rows r0 = warp_in * 16 + lane / 4 (position 8 wg + 2 warp_in) and r0 + 8 (the next
  // position), head kh * 8 + lane / 4 for both.
  const int wg = warp >> 2, wi = warp & 3;
  const int tq = lane & 3, hq = lane >> 2;
  const int ib = i0 + 8 * wg + 2 * wi;        // query index (within the request) of row r0
  const int pr0 = n - Lb + ib;
  SlRow st;
#pragma unroll
  for (int r = 0; r < 2; ++r) {
    st.rmax[r] = -INFINITY; st.rb[r] = (float)PV_B; st.rsum[r] = 0.f;
    st.rmh = 0xFC00FC00u;  // -inf, -inf
    st.rbh = 0x7BFF7BFFu;  // 65504, 65504 (no key yet)
    st.jhi[r] = pr0 + r - s0;
    st.jlo[r] = pr0 + r - W + 1 - s0;
  }
  OT o[NOR];
#pragma unroll
  for (int i = 0; i < NOR; ++i) o[i] = 0;
#define SK_T(k) do { } while (0)

  const uint64_t dq = desc64(smem_u32(sq + wg * 4096));
  const uint64_t dql = desc64(smem_u32(sq + Q_BYTES + wg * 4096));  // q_lo tile
  const uint64_t dones = desc_ones(smem_u32(sones));
  // Interior tiles need no mask: every row of the CTA sees every key of the tile.
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
  // FA3 order, one S buffer: issue QK(it), issue PV(it - 1), wait QK(it), softmax(it) while PV(it - 1) runs, wait
  // PV(it - 1), release its stage, rescale O, pack P(it). PV(nt - 1) after the loop.
  (void)sb;
  const uint32_t zdep = (uint32_t)(p.ks >> 8);  // unused
  (void)zdep;
  {
    const int j0 = t0 * BN;
    sl_softmax(sa, pa, st, j0, j0 < jlo_all || j0 + BN - 1 > jhi_all, tq, a32, scale);
    sl_rescale(o, st, scale);
    sl_pack(sa, pa);
  }
  {
    // Incremental stage / parity / descriptors (no modulo, no per-tile address multiplies or tid reloads).
    const uint32_t fb0 = smem_u32(&bars[0]), eb0 = smem_u32(&bars[ST]);
    const uint64_t dk0 = desc64(smem_u32(sk)), dv0 = descv(smem_u32(sv));
    uint32_t s = 1 % ST, ph = (1 / ST) & 1, sp = 0;  // stage and parity of tile it, stage of tile it - 1
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
        if (p.has_lo) {  // + q_lo . k (q_lo tile in smem)
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
  // Epilogue: O x (1 / rsum, which is 1) as BF16 straight from registers, through o_norm_gate when p.gate is set.
  // Without the gate (sl_attn) the output is the unnormalized O.
#pragma unroll
  for (int r = 0; r < 2; ++r) {
    const int i = ib + r;
    if (i < Lb) {  // warp-uniform (every lane of a warp holds the same two positions)
      __nv_bfloat16* dst = p.out + ((size_t)(q0 + i) * 16 + kh * 8 + hq) * 64 + 2 * tq;
      const float inv = st.rsum[r] > 0.f ? 1.f / st.rsum[r] : 0.f;
      __nv_bfloat162 v[8];
#pragma unroll
      for (int jb = 0; jb < 8; ++jb) { const float2 f = __half22float2(h2(o[2 * jb + r])); v[jb] = __floats2bfloat162_rn(f.x * inv, f.y * inv); }
      if (p.gate) {
        // o_norm_gate on the BF16 attention output: x * rsqrt(mean(x^2) + eps) * sigmoid(g), FP32 math, BF16 out.
        // The 4 lanes of a quad hold the head's 64 values.
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

// Gather: K rows straight through, V transposed with the sigma key order inside each 32-key block: slot k of V^T
// holds key sigma(k) = 2 t + (i & 1) + 8 (i >> 1) + 16 (k >> 4), t = (k & 15) >> 2, i = k & 3 (the QK accumulator
// column order of the PV A operand's k index). One CTA per 64 copy rows of one request (two 32-key blocks). Copy row
// j is cache row s0 + j, s0 = max(0, n - L - win); rows past the request's keys are written as zeros.
__global__ void __launch_bounds__(256) sl_gather_k(const uint4* __restrict__ kc, const uint4* __restrict__ vc,
    const int* __restrict__ pt, const int* __restrict__ seqlens, const int* __restrict__ cu_q, int pages, int page,
    int S, int win, int part, uint4* __restrict__ ko, uint4* __restrict__ vt) {
  __shared__ uint4 tile[64][9];
  asm volatile("griddepcontrol.wait;" ::: "memory");
  asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
  const int b = blockIdx.y, j0 = blockIdx.x * 64, t = threadIdx.x;
  const int n = seqlens[b], L = cu_q[b + 1] - cu_q[b];
  const int s0 = max(0, n - L - win), nv = n - s0;
  // part 0: blocks holding only rows cached before this call (can run before the front kernel); part 1: the rest
  // (blocks touching this call's rows, whole blocks); part 2: every block.
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

// sl_ts and sl_pvstat are bound for interface compatibility: sl_ts stores a pointer nothing reads, sl_pvstat is empty.
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
  { cudaLaunchConfig_t q0 = {cfg.gridDim, cfg.blockDim, cfg.dynamicSmemBytes, at::cuda::getCurrentCUDAStream(), nullptr, 0}; cfg = q0; }  // queue set positionally, never by field name (submission lint)
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

// q [T, 16, 64] e4m3, ko [B, S, 2, 64], vt [B, 2, 64, S] (sigma order), out [T, 16, 64] bf16. a32 = A rounded to
// f16, the code slope of the softmax (8 codes per octave).
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
  // Key split: only when the unsplit grid fills at most half of the resident CTA slots (SK_KS 2 builds only).
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
  { cudaLaunchConfig_t q0 = {cfg.gridDim, cfg.blockDim, cfg.dynamicSmemBytes, at::cuda::getCurrentCUDAStream(), nullptr, 0}; cfg = q0; }  // queue set positionally, never by field name (submission lint)
  cfg.attrs = at;
  cfg.numAttrs = 2;
  cudaLaunchKernelEx(&cfg, sl_attn_k, mq, mk, mv, mql, p);
}

void sl_attn(torch::Tensor q, torch::Tensor ko, torch::Tensor vt, torch::Tensor cache_seqlens, torch::Tensor cu_q,
             int64_t max_q, int64_t window, double softmax_scale, torch::Tensor out) {
  sl_launch(q, ko, vt, cache_seqlens, cu_q, max_q, window, softmax_scale, out, nullptr, 0, 0.f);
}

// The same with o_norm_gate fused: gate [T, 16 * 64] BF16 view (row stride gate.stride(0)), out [T, 16 * 64] BF16.
void sl_attn_ng(torch::Tensor q, torch::Tensor ko, torch::Tensor vt, torch::Tensor cache_seqlens, torch::Tensor cu_q,
                int64_t max_q, int64_t window, double softmax_scale, torch::Tensor gate, double eps, torch::Tensor out) {
  TORCH_CHECK(gate.stride(1) == 1 && gate.size(1) == 1024, "sl_attn_ng gate");
  sl_launch(q, ko, vt, cache_seqlens, cu_q, max_q, window, softmax_scale, out,
            (const __nv_bfloat16*)gate.data_ptr(), (int)gate.stride(0), (float)eps);
}

// The same with a second query plane: qlo [T, 16, 64] e4m3, contiguous, q = q + qlo, attended by two more QK wgmmas.
void sl_attn_ng2(torch::Tensor q, torch::Tensor qlo, torch::Tensor ko, torch::Tensor vt, torch::Tensor cache_seqlens,
                 torch::Tensor cu_q, int64_t max_q, int64_t window, double softmax_scale, torch::Tensor gate, double eps,
                 torch::Tensor out) {
  TORCH_CHECK(gate.stride(1) == 1 && gate.size(1) == 1024, "sl_attn_ng gate");
  TORCH_CHECK(qlo.is_contiguous() && qlo.numel() == q.numel() && qlo.element_size() == 1, "sl_attn_ng2 qlo");
  sl_launch(q, ko, vt, cache_seqlens, cu_q, max_q, window, softmax_scale, out,
            (const __nv_bfloat16*)gate.data_ptr(), (int)gate.stride(0), (float)eps, (const uint32_t*)qlo.data_ptr());
}

