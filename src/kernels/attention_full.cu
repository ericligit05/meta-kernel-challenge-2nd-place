// Full-attention prefill kernel and its helper kernels, for attention_full.py.
//
// The attention kernel is derived from the FlashAttention-3 Hopper (SM90) forward kernel, Copyright (c) 2024,
// Jay Shah, Ganesh Bikshandi, Ying Zhang, Vijay Thakkar, Pradeep Ramani, Tri Dao (github.com/Dao-AILab/flash-attention,
// BSD 3-Clause License, full text in THIRD_PARTY_NOTICES.md), and includes CUTLASS / CuTe headers (BSD 3-Clause). It
// runs one case only: FP8 E4M3 q / K / V, head dim 64, 16 q heads on 2 KV heads (GQA packed into the tile rows), varlen
// queries, causal, contiguous K and a prepared V image, 192-row x 256-key tiles, 2 pipeline stages.
// Changes to FA3's forward, all in this file:
// - Softmax (SoftmaxL): S = q K^T accumulates in FP16, and P is exp-free. front() writes q prescaled by
//   log2(e) softmax_scale, so S is already the log2 of the unnormalized probability. One fma.rn.relu.f16x2 per two
//   scores, z = S 2^-21 + B, lands in the f16 subnormal range where the bit pattern is the integer 8 (S - max) + U_PB;
//   its low byte is the E4M3 code of P. An E4M3 code c reads as (1 + (c & 7) / 8) 2^((c >> 3) - 7), a piecewise-linear
//   2^(c / 8 - 7), so P is about 2^(S - max + 5): the row max gets code U_PB = 96 (32.0), and keys more than about 12
//   octaves below it get code 0 (relu).
// - P V accumulates in FP16 (fk_pv16, 16 half2 registers per thread) over the whole key range and is folded into the
//   FP32 output registers once at the end (u_fold). V comes from a prepared image (one 72 x 256-byte tile per request,
//   KV head and 256-key block, written by the gather kernels below) loaded with one bulk copy per tile.
// - No row sum: the output rows stay unnormalized (each off by a positive factor). finalize returns, in place of the
//   LSE, the log of the factor that maps a split's O back to sum exp(scale s) v, which is all the split merge needs.
//   The per-head RMSNorm of o_norm_gate after attention cancels the remaining factor.
// - o_norm_gate is fused into the epilogue (unsplit calls) or the split combine kernel (calls split by FA3's
//   heuristic, at most 16 splits).
// - An equal-work split schedule (SK2Scheduler, sk2_prepare): for long-context calls every tile's key blocks are laid
//   end to end and cut into one equal range per SM; a tile cut across CTAs is merged by the last CTA to finish it.
//
// Helper kernels: pv_gather_stage / z0_gp_k / z0_gp8_k copy the paged K / V cache into the contiguous K and the V
// image; at_front_k is the full-layer front (QK-norm, FP8 q / k / v, cache write).
//
// Settings: ATT_VARIANTS lists the compiled att_dispatch variants (bit 1 local, 2 paged, 4 split, 8 V column-major,
// 16 equal-work split; 20, 4 and 0 are all causal and unpaged). U_PB: E4M3 code of the row max. FK_PV72: V tiles in
// the 72-row image layout. FK_SUMFL: only read by the unused row-sum code. ATT_BN: keys per tile. ATT_SCHED 1: the
// FP8 path uses FA3's warpgroup scheduler barrier. AT_SUB: the subnormal code frame above. Z0_NOL: no row sum.
// Z0_GP / Z0_GP_MINB: z0_gp_k mode for gather part 2, used from Z0_GP_MINB requests up.
//
// The z0_ / Z0_ and x52 / X52 prefixes are naming tags with no meaning of their own, kept because they are part of the
// submitted code. Here they name the gather kernels and settings added to FA3 (z0_gp_k, Z0_NOL, Z0_GP), the fused
// o_norm_gate epilogue (x52_fuse, x52_gptr, x52_geps, x52_gstride, kX52Rows) and a value in the equal-work merge (x52z).
#define ATT_VARIANTS ATT_CASE(20) ATT_CASE(4) ATT_CASE(0)
#define U_PB 96
#define FK_PV72
#define FK_SUMFL 8
#define ATT_BN 256
#define ATT_SCHED 1
#define AT_SUB
#define Z0_NOL
#define Z0_GP 2
#define Z0_GP_MINB 2
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include "cute/tensor.hpp"
#include "cutlass/cutlass.h"
#include "cutlass/device_kernel.h"
#include <cutlass/kernel_hardware_info.h>
#include "cutlass/cluster_launch.hpp"
#include "cutlass/kernel_launch.h"
#define BOOL_SWITCH(COND, CONST_NAME, ...)                                                       \
  [&] {                                                                                          \
    if (COND) {                                                                                  \
      constexpr static bool CONST_NAME = true;                                                   \
      return __VA_ARGS__();                                                                      \
    } else {                                                                                     \
      constexpr static bool CONST_NAME = false;                                                  \
      return __VA_ARGS__();                                                                      \
    }                                                                                            \
  }()
  #define CAUSAL_LOCAL_SWITCH(CAUSAL_COND, LOCAL_COND, CAUSAL_CONST_NAME, LOCAL_CONST_NAME, ...) \
    [&] {                                                                                        \
      if (CAUSAL_COND) {                                                                         \
        constexpr static bool CAUSAL_CONST_NAME = true;                                          \
        constexpr static bool LOCAL_CONST_NAME = false;                                          \
        return __VA_ARGS__();                                                                    \
      } else if (LOCAL_COND) {                                                                   \
        constexpr static bool CAUSAL_CONST_NAME = false;                                         \
        constexpr static bool LOCAL_CONST_NAME = true;                                           \
        return __VA_ARGS__();                                                                    \
      } else {                                                                                   \
        constexpr static bool CAUSAL_CONST_NAME = false;                                         \
        constexpr static bool LOCAL_CONST_NAME = false;                                          \
        return __VA_ARGS__();                                                                    \
      }                                                                                          \
    }()
  #define SOFTCAP_SWITCH BOOL_SWITCH
  #define PAGEDKV_SWITCH BOOL_SWITCH
  #define SPLIT_SWITCH BOOL_SWITCH
  #define APPENDKV_SWITCH BOOL_SWITCH
  #define PACKGQA_SWITCH BOOL_SWITCH
  #define VARLEN_SWITCH BOOL_SWITCH
  #define CLUSTER_SWITCH BOOL_SWITCH
  #define ARCH_SWITCH(ARCH, ARCH_NAME, ...)                                                      \
  [&] {                                                                                          \
    if (ARCH == 86 || ARCH == 89) {                                                              \
      constexpr static int ARCH_NAME = 86;                                                       \
      return __VA_ARGS__();                                                                      \
    } else if (ARCH < 90) {                                                                      \
      constexpr static int ARCH_NAME = 80;                                                       \
      return __VA_ARGS__();                                                                      \
    } else {                                                                                     \
      constexpr static int ARCH_NAME = 90;                                                       \
      return __VA_ARGS__();                                                                      \
    }                                                                                            \
  }()
  #define VCOLMAJOR_SWITCH(COND, CONST_NAME, ...)                                                \
  [&] {                                                                                          \
    constexpr static bool CONST_NAME = false;                                                    \
    return __VA_ARGS__();                                                                        \
  }()
#define HEADDIM_SWITCH(HEADDIM, ...)                                                             \
  [&] {                                                                                          \
    if (HEADDIM == 64) {                                                                         \
      constexpr static int kHeadSize = 64;                                                       \
      return __VA_ARGS__();                                                                      \
    } else if (HEADDIM == 96) {                                                                  \
      constexpr static int kHeadSize = 96;                                                       \
      return __VA_ARGS__();                                                                      \
    } else if (HEADDIM == 128) {                                                                 \
      constexpr static int kHeadSize = 128;                                                      \
      return __VA_ARGS__();                                                                      \
    } else if (HEADDIM == 96) {                                                                  \
      constexpr static int kHeadSize = 96;                                                       \
      return __VA_ARGS__();                                                                      \
    } else if (HEADDIM == 256) {                                                                 \
      constexpr static int kHeadSize = 256;                                                      \
      return __VA_ARGS__();                                                                      \
    }                                                                                            \
  }()
#define NUM_WARP_SWITCH(VALUE, CONST_NAME, ...)                                                  \
  [&] {                                                                                          \
    if (VALUE <= 1) {                                                                            \
      constexpr static int CONST_NAME = 1;                                                       \
      return __VA_ARGS__();                                                                      \
    } else if (VALUE <= 2) {                                                                     \
      constexpr static int CONST_NAME = 2;                                                       \
      return __VA_ARGS__();                                                                      \
    } else if (VALUE <= 4) {                                                                     \
      constexpr static int CONST_NAME = 4;                                                       \
      return __VA_ARGS__();                                                                      \
    } else if (VALUE <= 8) {                                                                     \
      constexpr static int CONST_NAME = 8;                                                       \
      return __VA_ARGS__();                                                                      \
    } else if (VALUE <= 16) {                                                                    \
      constexpr static int CONST_NAME = 16;                                                      \
      return __VA_ARGS__();                                                                      \
    } else {                                                                                     \
      constexpr static int CONST_NAME = 32;                                                      \
      return __VA_ARGS__();                                                                      \
    }                                                                                            \
  }()
  #define SPARSE_MASK_SWITCH BOOL_SWITCH
#include <cuda.h>
#include <vector>
struct Qkv_params {
    using index_t = int64_t;
    void *__restrict__ q_ptr;
    void *__restrict__ k_ptr;
    void *__restrict__ v_ptr;
    index_t q_batch_stride;
    index_t k_batch_stride;
    index_t v_batch_stride;
    index_t q_row_stride;
    index_t k_row_stride;
    index_t v_row_stride;
    index_t q_head_stride;
    index_t k_head_stride;
    index_t v_head_stride;
    index_t v_dim_stride;
    int h, h_k;
};
struct Flash_fwd_params : public Qkv_params {
    using index_t = int64_t;
    void * __restrict__ o_ptr;
    void * __restrict__ oaccum_ptr;
    index_t o_batch_stride;
    index_t o_row_stride;
    index_t o_head_stride;
    void * __restrict__ softmax_lse_ptr;
    void * __restrict__ softmax_lseaccum_ptr;
    float * __restrict__ q_descale_ptr;
    float * __restrict__ k_descale_ptr;
    float * __restrict__ v_descale_ptr;
    index_t q_descale_batch_stride;
    index_t q_descale_head_stride;
    index_t k_descale_batch_stride;
    index_t k_descale_head_stride;
    index_t v_descale_batch_stride;
    index_t v_descale_head_stride;
    int b, seqlen_q, seqlen_k, seqlen_knew, d, seqlen_q_rounded, seqlen_k_rounded, d_rounded, rotary_dim;
    int total_q, total_k, total_knew;
    int b_k;
    int dv, dv_rounded;
    float scale_softmax;
    float softcap;
    int * __restrict__ cu_seqlens_q;
    int * __restrict__ cu_seqlens_k;
    int * __restrict__ cu_seqlens_knew;
    int * __restrict__ leftpad_k;
    int *__restrict__ seqused_q;
    int *__restrict__ seqused_k;
    index_t oaccum_split_stride;
    index_t oaccum_batch_stride;
    index_t oaccum_row_stride;
    index_t oaccum_head_stride;
    index_t lseaccum_split_stride;
    index_t lseaccum_batch_stride;
    index_t lseaccum_head_stride;
    void * __restrict__ knew_ptr;
    void * __restrict__ vnew_ptr;
    index_t knew_batch_stride;
    index_t vnew_batch_stride;
    index_t knew_row_stride;
    index_t vnew_row_stride;
    index_t knew_head_stride;
    index_t vnew_head_stride;
    void *__restrict__ qv_ptr;
    index_t qv_batch_stride;
    index_t qv_row_stride;
    index_t qv_head_stride;
    void * __restrict__ rotary_cos_ptr;
    void * __restrict__ rotary_sin_ptr;
    int *__restrict__ seqlens_rotary;
    int * __restrict__ kv_batch_idx;
    int * __restrict__ page_table;
    index_t page_table_batch_stride;
    int page_size;
    int num_pages;
    bool pagedkv_tma;
    float p_dropout;
    uint8_t p_dropout_in_uint8_t;
    float rp_dropout;
    int window_size_left, window_size_right;
    int attention_chunk;
    uint64_t * rng_state;
    bool is_bf16;
    bool is_fp32;
    bool is_e4m3;
    bool is_causal;
    bool is_local;
    bool is_rotary_interleaved;
    int num_splits;
    bool pack_gqa;
    int * __restrict__ tile_count_semaphore;
    int * __restrict__ num_m_blocks_ptr;
    int * __restrict__ num_splits_dynamic_ptr;
    int * __restrict__ varlen_batch_idx_ptr;
    int * __restrict__ num_nheads_in_l2_ptr;
    bool skip_scheduler_metadata_computation;
    bool varlen_sort_batches;
    int tile_count_semaphore_offset;
    bool head_swizzle;
    bool prepare_varlen_pdl;
    int arch;
    int num_sm;
    int sk2_k;
    int sk2_wave;
    bool only_qv;
    void *__restrict__ sink_ptr;
    int * __restrict__ sparse_mask_fine;
    int sparse_mask_k_block_size;
    int sparse_mask_q_tile_size;
    int sparse_mask_max_k_blocks;
    int sparse_mask_fine_k_stride;
    int sparse_mask_fine_q_stride;
    bool use_sparse_mask;
};
struct Flash_bwd_params : public Flash_fwd_params {
    using index_t = int64_t;
    void *__restrict__ do_ptr;
    void *__restrict__ dq_ptr;
    void *__restrict__ dk_ptr;
    void *__restrict__ dv_ptr;
    void *__restrict__ dq_accum_ptr;
    void *__restrict__ dk_accum_ptr;
    void *__restrict__ dv_accum_ptr;
    index_t do_batch_stride;
    index_t do_row_stride;
    index_t do_head_stride;
    index_t dq_batch_stride;
    index_t dk_batch_stride;
    index_t dv_batch_stride;
    index_t dq_row_stride;
    index_t dk_row_stride;
    index_t dv_row_stride;
    index_t dq_head_stride;
    index_t dk_head_stride;
    index_t dv_head_stride;
    void *__restrict__ dsoftmax_sum;
    void *__restrict__ softmax_lse_log2_ptr;
    int *__restrict__ dq_semaphore;
    int *__restrict__ dk_semaphore;
    int *__restrict__ dv_semaphore;
    bool deterministic;
    index_t dq_accum_split_stride;
};
template <int Arch, typename T, int kHeadDim, int kHeadDimV, bool Split, bool PagedKVNonTMA, bool Has_softcap, bool PackGQA>
void run_mha_fwd_(Flash_fwd_params &params, cudaStream_t strm_);
void prepare_varlen_num_blocks(Flash_fwd_params &params, cudaStream_t strm_, bool packgqa, int blockM, int blockN, bool enable_pdl);
void sk2_prepare(Flash_fwd_params &params, cudaStream_t strm_, int blockM, int blockN);
template <int Arch, typename T, int kHeadDim, bool Has_softcap>
void run_mha_bwd_(Flash_bwd_params &params, cudaStream_t strm_);
template <typename T, typename Tpartial, int kBlockK>
void run_mha_fwd_combine_(Flash_fwd_params &params, cudaStream_t strm_, bool enable_pdl);
#include <tuple>
#define ATT_BM 192
#define ATT_BN_LOCAL ATT_BN
#define ATT_STAGES 2
constexpr std::tuple<int, int, bool, bool> tile_size_fwd_sm90(
        int headdim, int headdim_v, bool is_causal, bool is_local, int element_size=2,
        bool v_colmajor=false, bool paged_kv_non_TMA=false, bool softcap=false, bool use_one_mma_wg=false) {
    if (element_size == 2) {
        if (headdim <= 64) {
            if (headdim_v == 512) {
                return {64, 64, false, false};
            } else if (headdim_v == 256) {
                return {128, 96, true, false};
            } else {
                if (use_one_mma_wg) {
                    return {64, 192, true, true};
                } else {
                    bool const use_blockN_128 = is_causal || (!is_local && paged_kv_non_TMA);
                    return {192, use_blockN_128 ? 128 : 192, use_blockN_128, true};
                }
            }
        } else if (headdim <= 96) {
            return {192, is_local || paged_kv_non_TMA ? 128 : 144, false, true};
        } else if (headdim <= 128) {
            if (use_one_mma_wg) {
                return {64, is_causal || is_local || paged_kv_non_TMA ? 128 : 176, true, true};
            } else {
                bool const use_blockN_128 = is_causal || is_local || paged_kv_non_TMA;
                return {128, use_blockN_128 ? 128 : 176, true, true};
            }
        } else if (headdim <= 192) {
            return {128, paged_kv_non_TMA || is_local ? 96 : (headdim_v <= 128 ? 128 : 112), true, true};
        } else {
            return {128, is_local ? 64 : 80, true, true};
        }
    } else {
        if (headdim <= 64) {
            if (use_one_mma_wg) {
                return {64, 128, true, true};
            } else {
                return {ATT_BM, is_local ? ATT_BN_LOCAL : ATT_BN, true, true};
            }
        } else if (headdim <= 96) {
            return {192, 128, true, true};
        } else if (headdim <= 128) {
            if (use_one_mma_wg) {
                return {64, 96, true, true};
            } else {
                return {128, paged_kv_non_TMA ? 160 : (v_colmajor || (softcap && is_local) ? 192 : 224), true, true};
            }
        } else if (headdim <= 192) {
            return {128, (paged_kv_non_TMA || softcap) && is_local ? 128 : 160, true, true};
        } else {
            return {128, is_local ? 64 : 128, true, !paged_kv_non_TMA};
        }
    }
}
constexpr std::tuple<int, int, int, int, bool> tile_size_fwd_sm8x(
        bool sm86_or_89, int headdim, int headdim_v, bool is_causal, bool is_local, int element_size=2,
        bool paged_kv=false, bool varlen_and_split=false,
        bool softcap=false, bool append_kv=false) {
    if (element_size == 2) {
        if (headdim <= 64) {
            return {128, varlen_and_split ? 80 : (is_local ? 96 : 112), 4, 1, false};
        } else if (headdim <= 96) {
            return {128, varlen_and_split || is_local ? 48 : 64, 4, 1, false};
        } else if (headdim <= 128) {
            bool const use_8_warps = sm86_or_89 | varlen_and_split;
            return {128, use_8_warps ? (varlen_and_split ? (is_local ? 96 : 112) : (is_local ? 96 : 128)) : (is_local ? 48 : 64), use_8_warps ? 8 : 4, 1, use_8_warps};
        } else if (headdim <= 192) {
            bool const kBlockN_64 = append_kv || is_local || varlen_and_split || paged_kv;
            return {128, kBlockN_64 ? 64 : 96, 8, sm86_or_89 ? 1 : 2, !kBlockN_64};
        } else {
            return {128, sm86_or_89 ? (append_kv ? 32 : (varlen_and_split || is_local ? 48 : 64)) : (append_kv ? 48 : (varlen_and_split || is_local ? 64 : 96)), 8, 1, sm86_or_89 && !append_kv};
        }
    } else {
        return {128, 64, 8, 2, false};
    }
}
#include "cutlass/fast_math.h"
#include "cutlass/arch/barrier.h"
#include "cutlass/arch/barrier.h"
namespace flash {
CUTLASS_DEVICE
static void named_barrier_sync(uint32_t num_threads, uint32_t barrier_id_) {
    static constexpr uint32_t ReservedNamedBarrierCount = static_cast<uint32_t>(cutlass::arch::ReservedNamedBarriers::FirstUserBarrier);
    uint32_t barrier_id = barrier_id_ + ReservedNamedBarrierCount;
    asm volatile("bar.sync %0, %1;" : : "r"(barrier_id), "r"(num_threads));
    cutlass::arch::synclog_emit_named_barrier_arrive_and_wait(__LINE__, num_threads, barrier_id);
}
CUTLASS_DEVICE
static void named_barrier_sync(uint32_t num_threads, cutlass::arch::ReservedNamedBarriers reserved_named_barriers) {
    uint32_t barrier_id = static_cast<uint32_t>(reserved_named_barriers);
    asm volatile("bar.sync %0, %1;" : : "r"(barrier_id), "r"(num_threads));
    cutlass::arch::synclog_emit_named_barrier_arrive_and_wait(__LINE__, num_threads, barrier_id);
}
CUTLASS_DEVICE
static void named_barrier_arrive(uint32_t num_threads, uint32_t barrier_id_) {
    static constexpr uint32_t ReservedNamedBarrierCount = static_cast<uint32_t>(cutlass::arch::ReservedNamedBarriers::FirstUserBarrier);
    uint32_t barrier_id = barrier_id_ + ReservedNamedBarrierCount;
    cutlass::arch::synclog_emit_named_barrier_arrive(__LINE__, num_threads, barrier_id);
    asm volatile("bar.arrive %0, %1;" : : "r"(barrier_id), "r"(num_threads));
}
CUTLASS_DEVICE
static void named_barrier_arrive(uint32_t num_threads, cutlass::arch::ReservedNamedBarriers reserved_named_barriers) {
    uint32_t barrier_id = static_cast<uint32_t>(reserved_named_barriers);
    cutlass::arch::synclog_emit_named_barrier_arrive(__LINE__, num_threads, barrier_id);
    asm volatile("bar.arrive %0, %1;" : : "r"(barrier_id), "r"(num_threads));
}
enum class FwdNamedBarriers {
    QueryEmpty = 0,
    WarpSchedulerWG1 = 1,
    WarpSchedulerWG2 = 2,
    WarpSchedulerWG3 = 3,
    AppendKV = 4,
    QueryRotated = 5,
    PFull = 6,
    PEmpty = 7,
};
enum class BwdNamedBarriers {
    KVEmpty = 0,
    PdS = 1,
    dQEmptyWG1 = 2,
    dQEmptyWG2 = 3,
    dQEmptyWG3 = 4,
    dQFullWG1 = 5,
    dQFullWG2 = 6,
    dQFullWG3 = 7,
};
}
#include <assert.h>
#include <stdint.h>
#include <stdlib.h>
#include <cuda_fp16.h>
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
#include <cuda_bf16.h>
#endif
#include <cute/tensor.hpp>
#include <cutlass/cutlass.h>
#include <cutlass/array.h>
#include <cutlass/numeric_conversion.h>
#include <cutlass/numeric_types.h>
#include <assert.h>
#include <stdlib.h>
#define CHECK_CUDA(call)                        \
    do {                                                                                                  \
        cudaError_t status_ = call;                                                                       \
        if (status_ != cudaSuccess) {                                                                     \
            fprintf(stderr, "CUDA error (%s:%d): %s\n", __FILE__, __LINE__, cudaGetErrorString(status_)); \
            exit(1);                                                                                      \
        }                                                                                                 \
    } while(0)
#define CHECK_CUDA_KERNEL_LAUNCH() CHECK_CUDA(cudaGetLastError())
namespace flash {
using namespace cute;
template <typename Kernel>
struct enable_sm90_or_later : Kernel {
    template <typename... Args>
    CUTLASS_DEVICE void operator()(Args&&... args) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
        Kernel::operator()(std::forward<Args>(args)...);
#endif
    }
};
template <typename Kernel>
struct enable_sm80_to_sm89 : Kernel {
    template <typename... Args>
    CUTLASS_DEVICE void operator()(Args&&... args) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800) && (__CUDA_ARCH__ <= 890)
        Kernel::operator()(std::forward<Args>(args)...);
#endif
    }
};
template<typename T>
struct MaxOp {
__device__ __forceinline__ T operator()(T const & x, T const & y) { return x > y ? x : y; }
};
template <>
struct MaxOp<float> {
__device__ __forceinline__ float operator()(float const &x, float const &y) { return max(x, y); }
};
template<typename T>
struct SumOp {
__device__ __forceinline__ T operator()(T const & x, T const & y) { return x + y; }
};
template<int THREADS>
struct Allreduce {
    static_assert(THREADS == 32 || THREADS == 16 || THREADS == 8 || THREADS == 4);
    template<typename T, typename Operator>
    static __device__ __forceinline__ T run(T x, Operator &op) {
        constexpr int OFFSET = THREADS / 2;
        x = op(x, __shfl_xor_sync(uint32_t(-1), x, OFFSET));
        return Allreduce<OFFSET>::run(x, op);
    }
};
template<>
struct Allreduce<2> {
template<typename T, typename Operator>
static __device__ __forceinline__ T run(T x, Operator &op) {
    x = op(x, __shfl_xor_sync(uint32_t(-1), x, 1));
    return x;
}
};
CUTLASS_HOST_DEVICE
int div_floor(cutlass::FastDivmod const& divmod, int dividend) {
    return dividend >= 0 ? divmod.divide(dividend) : -1 - divmod.divide(-1 - dividend);
}
CUTLASS_HOST_DEVICE
int round_down(cutlass::FastDivmod const& divmod, int dividend) {
    return div_floor(divmod, dividend) * divmod.divisor;
}
CUTLASS_HOST_DEVICE
int round_up(cutlass::FastDivmod const& divmod, int dividend) {
    return div_floor(divmod, dividend - 1) * divmod.divisor + divmod.divisor;
}
template<bool Transposed=false, typename Layout0>
CUTLASS_DEVICE auto convert_layout_acc_rowcol(Layout0 acc_layout) {
    if constexpr (decltype(rank<0>(acc_layout))::value == 3) {
        static_assert(decltype(size<0, 0>(acc_layout))::value == 2);
        static_assert(decltype(size<0, 1>(acc_layout))::value == 2);
        static_assert(decltype(rank(acc_layout))::value == 3);
        auto l = acc_layout;
        if constexpr (!Transposed) {
            return make_layout(make_layout(get<0, 1>(l), get<1>(l)), make_layout(get<0, 0>(l), get<0, 2>(l), get<2>(l)));
        } else {
             return make_layout(make_layout(get<0, 0>(l), get<0, 2>(l), get<2>(l)), make_layout(get<0, 1>(l), get<1>(l)));
        }
    } else {
        static_assert(decltype(size<0>(acc_layout))::value == 4);
        static_assert(decltype(rank(acc_layout))::value == 3);
        auto l = logical_divide(acc_layout, Shape<_2>{});
        if constexpr (!Transposed) {
            return make_layout(make_layout(get<0, 1>(l), get<1>(l)), make_layout(get<0, 0>(l), get<2>(l)));
        } else {
            return make_layout(make_layout(get<0, 0>(l), get<2>(l)), make_layout(get<0, 1>(l), get<1>(l)));
        }
    }
};
template<typename MMA_Traits, typename Layout0>
CUTLASS_DEVICE auto convert_layout_acc_Aregs(Layout0 acc_layout) {
    using X = Underscore;
    if constexpr (decltype(rank<0>(acc_layout))::value == 3) {
        static_assert(decltype(size<0, 0>(acc_layout))::value == 2);
        static_assert(decltype(size<0, 1>(acc_layout))::value == 2);
        static_assert(decltype(rank(acc_layout))::value == 3);
        static_assert(decltype(rank(get<0>(acc_layout)))::value == 3);
        if constexpr (sizeof(typename MMA_Traits::ValTypeA) == 2) {
            auto l = logical_divide(get<0, 2>(acc_layout), Tile<_2>{});
            return make_layout(make_layout(get<0, 0>(acc_layout), get<0, 1>(acc_layout), get<0, 0>(l)), get<1>(acc_layout), coalesce(make_layout(get<0, 1>(l), get<2>(acc_layout))));
        } else {
            static_assert(sizeof(typename MMA_Traits::ValTypeA) == 1);
            static_assert(decltype(stride<0, 0>(acc_layout))::value == 1);
            static_assert(decltype(stride<0, 1>(acc_layout))::value == 2);
            auto l = logical_divide(get<0, 2>(acc_layout), Tile<Layout<Shape<_2, _2>>>{});
            return make_layout(make_layout(Layout<_4>{}, get<0, 0, 0>(l), get<0, 0, 1>(l)),
                               get<1>(acc_layout),
                               coalesce(make_layout(get<0, 1>(l), get<2>(acc_layout))));
        }
    } else {
        static_assert(decltype(size<0>(acc_layout))::value == 4);
        static_assert(decltype(rank(acc_layout))::value == 3);
        constexpr int mma_shape_K = get<2>(typename MMA_Traits::Shape_MNK{});
        static_assert(mma_shape_K == 8 || mma_shape_K == 16);
        if constexpr (mma_shape_K == 8) {
            return acc_layout;
        } else {
            auto l = logical_divide(acc_layout, Shape<X, X, _2>{});
            return make_layout(make_layout(get<0>(l), get<2, 0>(l)), get<1>(l), get<2, 1>(l));
        }
    }
};
template <typename To_type, typename Engine, typename Layout>
CUTLASS_DEVICE auto convert_type_unsafe(Tensor<Engine, Layout> const &tensor) {
    using From_type = typename Engine::value_type;
    static constexpr int numel = decltype(size(tensor))::value;
    cutlass::NumericArrayConverter<To_type, From_type, numel> convert_op;
    auto frag = convert_op(*reinterpret_cast<const cutlass::Array<From_type, numel> *>(tensor.data()));
    return make_tensor(make_rmem_ptr<To_type>(&frag), tensor.layout());
}
template <typename Engine, typename Layout, typename EngineOut>
CUTLASS_DEVICE void convert_type_out(Tensor<Engine, Layout> const &tensor, Tensor<EngineOut, Layout> &out) {
    using From_type = typename Engine::value_type;
    using To_type = typename EngineOut::value_type;
    static constexpr int FragmentSize = std::max(sizeof(From_type) / sizeof(To_type), sizeof(To_type) / sizeof(From_type));
    static_assert(CUTE_STATIC_V(size(tensor)) % FragmentSize == 0, "Fragment size does not vectorize properly");
    Tensor frag = recast<cutlass::Array<From_type, FragmentSize> const>(tensor);
    Tensor out_frg = recast<cutlass::Array<To_type, FragmentSize>>(out);
    static_assert(size(frag) == size(out_frg));
    cutlass::NumericArrayConverter<To_type, From_type, FragmentSize> convert_op;
    #pragma unroll
    for (int i = 0; i < size(frag); ++i) { out_frg[i] = convert_op(frag[i]); }
}
template <int N>
CUTE_HOST_DEVICE
void cp_async_wait() {
#if defined(CUTE_ARCH_CP_ASYNC_SM80_ENABLED)
    asm volatile("cp.async.wait_group %0;\n" :: "n"(N));
#endif
}
template <typename TS, typename TD>
CUTE_HOST_DEVICE
void cp_async_ca_zfill(TD* smem_ptr, TS const* gmem_ptr, bool pred) {
    SM80_CP_ASYNC_CACHEALWAYS_ZFILL<TS, TD>::copy(*gmem_ptr, *smem_ptr, pred);
}
template <bool A, class Mma, class Tensor0>
CUTLASS_DEVICE
auto mma_partition_fragment_AB(Mma const& mma, Tensor0 const& tensor0) {
    if constexpr (A) {
        return mma.partition_fragment_A(tensor0);
    } else {
        return mma.partition_fragment_B(tensor0);
    }
}
template <bool zero_init=false, int wg_wait=0, bool SwapAB=false, int M_slice=-1,
        typename Tensor0, typename Tensor1, typename Tensor2, typename TiledMma>
CUTLASS_DEVICE void gemm(TiledMma& tiled_mma, Tensor0 const& tCrA, Tensor1 const& tCrB, Tensor2& tCrC) {
    if constexpr (M_slice >= 0) {
        static constexpr int MMA_M = decltype(size<1>(tCrC))::value;
        static_assert(M_slice < MMA_M);
        Tensor tCrC_slice = cute::logical_divide(tCrC, Shape<cute::Underscore, Int<MMA_M>>{})(_, make_coord(Int<M_slice>{}, _), _);
        if constexpr (!SwapAB) {
            Tensor tCrA_slice = cute::logical_divide(tCrA, Shape<cute::Underscore, Int<MMA_M>>{})(_, make_coord(Int<M_slice>{}, _), _);
            gemm<zero_init, wg_wait, SwapAB, -1>(tiled_mma, tCrA_slice, tCrB, tCrC_slice);
        } else {
            Tensor tCrB_slice = cute::logical_divide(tCrB, Shape<cute::Underscore, Int<MMA_M>>{})(_, make_coord(Int<M_slice>{}, _), _);
            gemm<zero_init, wg_wait, SwapAB, -1>(tiled_mma, tCrA, tCrB_slice, tCrC_slice);
        }
    } else {
        constexpr bool Is_RS = !cute::is_base_of<cute::GMMA::DescriptorIterator, typename TiledMma::FrgTypeA>::value;
        if constexpr (Is_RS) {
            if constexpr (!SwapAB) {
                warpgroup_fence_operand(const_cast<Tensor0 &>(tCrA));
            } else {
                warpgroup_fence_operand(const_cast<Tensor1 &>(tCrB));
            }
        }
        warpgroup_fence_operand(tCrC);
        warpgroup_arrive();
        if constexpr (zero_init) {
            tiled_mma.accumulate_ = GMMA::ScaleOut::Zero;
        }
        static constexpr int kNumKIters = CUTE_STATIC_V(size<2>(tCrA));
        static constexpr int kMaxKIters = 16;
        CUTLASS_PRAGMA_UNROLL
        for (int k_block = 0; k_block < std::min(kNumKIters, kMaxKIters); ++k_block) {
            if constexpr (!SwapAB) {
                cute::gemm(tiled_mma, tCrA(_,_,k_block), tCrB(_,_,k_block), tCrC);
            } else {
                cute::gemm(tiled_mma, tCrB(_,_,k_block), tCrA(_,_,k_block), tCrC);
            }
            tiled_mma.accumulate_ = GMMA::ScaleOut::One;
        }
        if constexpr (kNumKIters > kMaxKIters) {
            int const k_offset = cutlass::canonical_warp_group_idx() < 128 ? 0 : 1;
            CUTLASS_PRAGMA_UNROLL
            for (int k_block = kMaxKIters; k_block < kNumKIters; ++k_block) {
                if constexpr (!SwapAB) {
                    cute::gemm(tiled_mma, tCrA(_,_,k_block + k_offset), tCrB(_,_,k_block + k_offset), tCrC);
                } else {
                    cute::gemm(tiled_mma, tCrB(_,_,k_block + k_offset), tCrA(_,_,k_block + k_offset), tCrC);
                }
                tiled_mma.accumulate_ = GMMA::ScaleOut::One;
            }
        }
        warpgroup_commit_batch();
        if constexpr (wg_wait >= 0) { warpgroup_wait<wg_wait>(); }
        warpgroup_fence_operand(tCrC);
        if constexpr (Is_RS) {
            if constexpr (!SwapAB) {
                warpgroup_fence_operand(const_cast<Tensor0 &>(tCrA));
            } else {
                warpgroup_fence_operand(const_cast<Tensor1 &>(tCrB));
            }
        }
    }
}
template<bool A_in_regs=false, bool B_in_regs=false, bool SwapAB=false,
         typename Tensor0, typename Tensor1,
         typename Tensor2, typename Tensor3, typename Tensor4,
         typename TiledMma, typename TiledCopyA, typename TiledCopyB,
         typename ThrCopyA, typename ThrCopyB, typename Hook>
CUTLASS_DEVICE void gemm_sm80(Tensor0 &acc, Tensor1 &tCrA, Tensor2 &tCrB, Tensor3 const& tCsA,
                              Tensor4 const& tCsB, TiledMma tiled_mma,
                              TiledCopyA smem_tiled_copy_A, TiledCopyB smem_tiled_copy_B,
                              ThrCopyA smem_thr_copy_A, ThrCopyB smem_thr_copy_B, Hook fn) {
    if constexpr (SwapAB) {
        gemm_sm80<B_in_regs, A_in_regs>(acc, tCrB, tCrA, tCsB, tCsA, tiled_mma, smem_tiled_copy_B, smem_tiled_copy_A, smem_thr_copy_B, smem_thr_copy_A, fn);
    } else {
        CUTE_STATIC_ASSERT_V(size<1>(tCrA) == size<1>(acc));
        CUTE_STATIC_ASSERT_V(size<1>(tCrB) == size<2>(acc));
        CUTE_STATIC_ASSERT_V(size<2>(tCrA) == size<2>(tCrB));
        Tensor tCrA_copy_view = smem_thr_copy_A.retile_D(tCrA);
        CUTE_STATIC_ASSERT_V(size<1>(tCsA) == size<1>(tCrA_copy_view));
        Tensor tCrB_copy_view = smem_thr_copy_B.retile_D(tCrB);
        CUTE_STATIC_ASSERT_V(size<1>(tCsB) == size<1>(tCrB_copy_view));
        if (!A_in_regs) { cute::copy(smem_tiled_copy_A, tCsA(_, _, _0{}), tCrA_copy_view(_, _, _0{})); }
        if (!B_in_regs) { cute::copy(smem_tiled_copy_B, tCsB(_, _, _0{}), tCrB_copy_view(_, _, _0{})); }
        #pragma unroll
        for (int i = 0; i < size<2>(tCrA); ++i) {
            if (i < size<2>(tCrA) - 1) {
                if (!A_in_regs) { cute::copy(smem_tiled_copy_A, tCsA(_, _, i + 1), tCrA_copy_view(_, _, i + 1)); }
                if (!B_in_regs) { cute::copy(smem_tiled_copy_B, tCsB(_, _, i + 1), tCrB_copy_view(_, _, i + 1)); }
            }
            if constexpr (!std::is_same_v<Hook, std::nullptr_t>) {
                if (i == 0) { fn(); }
            }
            cute::gemm(tiled_mma, tCrA(_, _, i), tCrB(_, _, i), acc);
        }
    }
}
template<typename Tensor0, typename Tensor1, typename Tensor2, typename Tensor3,
         typename TiledMma, typename TiledCopy, typename ThrCopy>
CUTLASS_DEVICE void gemm_rs_sm80(Tensor0 &acc, Tensor1 &tCrA, Tensor2 &tCrB, Tensor3 const& tCsB,
                                 TiledMma tiled_mma, TiledCopy smem_tiled_copy_B,
                                 ThrCopy smem_thr_copy_B) {
    CUTE_STATIC_ASSERT_V(size<1>(tCrA) == size<1>(acc));
    CUTE_STATIC_ASSERT_V(size<1>(tCrB) == size<2>(acc));
    CUTE_STATIC_ASSERT_V(size<2>(tCrA) == size<2>(tCrB));
    Tensor tCrB_copy_view = smem_thr_copy_B.retile_D(tCrB);
    CUTE_STATIC_ASSERT_V(size<1>(tCsB) == size<1>(tCrB_copy_view));
    cute::copy(smem_tiled_copy_B, tCsB(_, _, _0{}), tCrB_copy_view(_, _, _0{}));
    #pragma unroll
    for (int i = 0; i < size<2>(tCrA); ++i) {
        if (i < size<2>(tCrA) - 1) {
            cute::copy(smem_tiled_copy_B, tCsB(_, _, i + 1), tCrB_copy_view(_, _, i + 1));
        }
        cute::gemm(tiled_mma, tCrA(_, _, i), tCrB(_, _, i), acc);
    }
}
template <bool zero_init=false, typename Atom, typename TA, typename TB, typename TC>
CUTLASS_DEVICE void gemm_sm100(Atom& atom, TA const& tA, TB const& tB, TC&& tC) {
    static constexpr int rA = decltype(rank(tA))::value;
    static constexpr int rB = decltype(rank(tB))::value;
    static constexpr int rC = decltype(rank(tC))::value;
    static_assert(rA == 3 && rB == 3 && rC == 3);
    if constexpr (zero_init) { atom.accumulate_ = decltype(atom.accumulate_)::Zero; }
    CUTLASS_PRAGMA_UNROLL
    for (int k_block = 0; k_block < size<2>(tA); k_block++) {
        cute::gemm(atom, tA(_,_,k_block), tB(_,_,k_block), tC);
        atom.accumulate_ = decltype(atom.accumulate_)::One;
    }
}
template <class a_type, class b_type, class c_type,
          int M, int N, UMMA::Major a_major, UMMA::Major b_major,
          UMMA::ScaleIn a_neg, UMMA::ScaleIn b_neg, class... TAs, class... TMs>
CUTE_HOST_DEVICE constexpr
auto
to_tiled_mma_sm100_ts(
    TiledMMA<MMA_Atom<
      MMA_Traits<SM100_MMA_F8F6F4_SS, a_type, b_type, c_type,
                    cute::C<M>, cute::C<N>,
                    cute::integral_constant<UMMA::Major, a_major>,
                    cute::integral_constant<UMMA::Major, b_major>,
                    cute::integral_constant<UMMA::ScaleIn, a_neg>,
                    cute::integral_constant<UMMA::ScaleIn, b_neg>>,
      TAs...>, TMs...>) {
  return TiledMMA<MMA_Atom<
    MMA_Traits<SM100_MMA_F8F6F4_TS<a_type, b_type, c_type,
                                M, N,
                                a_major, b_major,
                                a_neg, b_neg, UMMA::Saturate::False>>,
    TAs...>, TMs...>{};
}
template <class a_type, class b_type, class c_type,
          int M, int N, UMMA::Major a_major, UMMA::Major b_major,
          UMMA::ScaleIn a_neg, UMMA::ScaleIn b_neg, class... TAs, class... TMs>
CUTE_HOST_DEVICE constexpr
auto
to_tiled_mma_sm100_ts(
    TiledMMA<MMA_Atom<
      SM100_MMA_F16BF16_SS<a_type, b_type, c_type,
                    M, N,
                    a_major,
                    b_major,
                    a_neg,
                    b_neg>,
      TAs...>, TMs...>) {
  return TiledMMA<MMA_Atom<
    SM100_MMA_F16BF16_TS<a_type, b_type, c_type,
                                M, N,
                                a_major, b_major,
                                a_neg, b_neg, UMMA::Saturate::False>,
    TAs...>, TMs...>{};
}
template <bool Is_even_MN=true, bool Is_even_K=true, bool Clear_OOB_MN=false, bool Clear_OOB_K=true,
          class CopyAtom, class TV, class Tiler, typename Engine0, typename Layout0, typename Engine1, typename Layout1,
          typename Engine2, typename Layout2, typename Engine3, typename Layout3>
CUTLASS_DEVICE void copy(TiledCopy<CopyAtom, TV, Tiler> const &tiled_copy, Tensor<Engine0, Layout0> const &S,
                         Tensor<Engine1, Layout1> &D, Tensor<Engine2, Layout2> const &identity_MN,
                         Tensor<Engine3, Layout3> const &predicate_K, const int max_MN=0) {
    auto copy_atom = static_cast<CopyAtom const&>(tiled_copy);
    CUTE_STATIC_ASSERT_V(rank(S) == Int<3>{});
    CUTE_STATIC_ASSERT_V(rank(D) == Int<3>{});
    CUTE_STATIC_ASSERT_V(size<0>(S) == size<0>(D));
    CUTE_STATIC_ASSERT_V(size<1>(S) == size<1>(D));
    CUTE_STATIC_ASSERT_V(size<2>(S) == size<2>(D));
    static_assert(!(Clear_OOB_MN && !Clear_OOB_K));
    auto has_with_bool = cute::is_valid([](auto t)->void_t<decltype(declval<typename decltype(t)::Traits>().with(true))>{}, copy_atom);
    #pragma unroll
    for (int m = 0; m < size<1>(S); ++m) {
        bool predicate_mn = Is_even_MN || get<0>(identity_MN(_0{}, m, _0{})) < max_MN;
        if constexpr (Is_even_MN || !Clear_OOB_MN) {
            if (Is_even_MN || predicate_mn) {
                #pragma unroll
                for (int k = 0; k < size<2>(S); ++k) {
                    if constexpr (Is_even_K || !Clear_OOB_K) {
                        if (Is_even_K || predicate_K(k)) { cute::copy(copy_atom, S(_, m, k), D(_, m, k)); }
                    } else {
                        if constexpr (has_with_bool) {
                            cute::copy(copy_atom.with(predicate_K(k)), S(_, m, k), D(_, m, k));
                        } else {
                            if (predicate_K(k)) {
                                cute::copy(copy_atom, S(_, m, k), D(_, m, k));
                            } else {
                                cute::clear(D(_, m, k));
                            }
                        }
                    }
                }
            }
        } else {
            if constexpr (!has_with_bool) {
                if (predicate_mn) {
                    #pragma unroll
                    for (int k = 0; k < size<2>(S); ++k) {
                        if (Is_even_K || predicate_K(k)) {
                            cute::copy(copy_atom, S(_, m, k), D(_, m, k));
                        } else if (Clear_OOB_K) {
                            cute::clear(D(_, m, k));
                        }
                    }
                } else {
                    cute::clear(D(_, m, _));
                }
            } else {
                #pragma unroll
                for (int k = 0; k < size<2>(S); ++k) {
                    cute::copy(copy_atom.with(predicate_mn && (Is_even_K || predicate_K(k))), S(_, m, k), D(_, m, k));
                }
            }
        }
    }
}
template <typename Fragment>
CUTLASS_DEVICE void permute_Aregs_fp8(Fragment &frag) {
    static_assert(decltype(size<0, 0>(frag))::value == 4);
    static_assert(decltype(size<0, 1>(frag))::value == 2);
    static_assert(decltype(stride<0, 0>(frag))::value == 1);
    static_assert(decltype(stride<0, 1>(frag))::value == 4);
    static_assert(sizeof(typename Fragment::value_type) == 1);
    int quad_idx = threadIdx.x % 4;
    bool lane_03 = quad_idx == 0 || quad_idx == 3;
    int selector_upper = lane_03 ? 0x5410 : 0x1054;
    int selector_lower = lane_03 ? 0x7632 : 0x3276;
    static constexpr int upper_map[4] = {0, 3, 1, 2};
    Tensor frag_64b = recast<uint2>(frag);
    #pragma unroll
    for (int i = 0; i < size(frag_64b); ++i) {
        uint32_t upper = frag_64b[i].x;
        uint32_t lower = frag_64b[i].y;
        uint32_t upper0 = lane_03 ? upper : lower;
        uint32_t lower0 = lane_03 ? lower : upper;
        upper0 = __shfl_sync(uint32_t(-1), upper0, upper_map[quad_idx], 4);
        lower0 = __shfl_sync(uint32_t(-1), lower0, upper_map[quad_idx] ^ 1, 4);
        frag_64b[i].x = __byte_perm(upper0, lower0, selector_upper);
        frag_64b[i].y = __byte_perm(upper0, lower0, selector_lower);
    }
}
template <typename Fragment>
CUTLASS_DEVICE void permute_Cregs_fp8(Fragment &frag) {
    static_assert(decltype(size<0, 0>(frag))::value == 2);
    static_assert(decltype(size<0, 1>(frag))::value == 2);
    static_assert(decltype(size<0, 2>(frag))::value % 2 == 0);
    static_assert(decltype(stride<0, 0>(frag))::value == 1);
    static_assert(sizeof(typename Fragment::value_type) == 4);
    Tensor frag_64b = group_modes<1, 3>(recast<uint2>(frag));
    #pragma unroll
    for (int mi = 0; mi < size<1>(frag_64b); ++mi) {
        #pragma unroll
        for (int i = 0; i < size<0, 2>(frag_64b) / 2; ++i) {
            cutlass::swap(frag_64b(make_coord(_0{}, _1{}, 2 * i), mi), frag_64b(make_coord(_0{}, _0{}, 2 * i + 1), mi));
        }
    }
}
template <typename Fragment>
CUTLASS_DEVICE void permute_output_fp8(Fragment &out) {
    static_assert(decltype(size<0, 0>(out))::value == 2);
    static_assert(decltype(size<0, 1>(out))::value == 2);
    static_assert(decltype(size<0, 2>(out))::value % 2 == 0);
    static_assert(decltype(stride<0, 0>(out))::value == 1);
    static_assert(sizeof(typename Fragment::value_type) == 4);
    Tensor frag = group_modes<1, 3>(out);
    #pragma unroll
    for (int mi = 0; mi < size<1>(frag); ++mi) {
        #pragma unroll
        for (int j = 0; j < size<0, 1>(frag); ++j) {
            #pragma unroll
            for (int i = 0; i < size<0, 2>(frag) / 2; ++i) {
                cutlass::swap(frag(make_coord(_1{}, j, 2 * i), mi), frag(make_coord(_0{}, j, 2 * i + 1), mi));
            }
        }
    }
}
template <typename Fragment>
CUTLASS_DEVICE void permute_output_fp8_Vcolmajor(Fragment &frag) {
    static_assert(decltype(size<0, 0>(frag))::value == 2);
    static_assert(decltype(size<0, 1>(frag))::value == 2);
    static_assert(decltype(stride<0, 0>(frag))::value == 1);
    static_assert(sizeof(typename Fragment::value_type) == 2 || sizeof(typename Fragment::value_type) == 4);
    int quad_idx = threadIdx.x % 4;
    bool lane_03 = quad_idx == 0 || quad_idx == 3;
    static constexpr int upper_map[4] = {0, 2, 3, 1};
    using type2 = std::conditional_t<sizeof(typename Fragment::value_type) == 2, uint32_t, uint64_t>;
    Tensor frag_2 = group_modes<1, 3>(recast<type2>(frag));
    #pragma unroll
    for (int mi = 0; mi < size<1>(frag_2); ++mi) {
        #pragma unroll
        for (int j = 0; j < size<0, 1>(frag_2); ++j) {
            #pragma unroll
            for (int i = 0; i < size<0, 2>(frag_2) / 2; ++i) {
                type2 upper = frag_2(make_coord(_0{}, j, 2 * i), mi);
                type2 lower = frag_2(make_coord(_0{}, j, 2 * i + 1), mi);
                type2 upper0 = lane_03 ? upper : lower;
                type2 lower0 = lane_03 ? lower : upper;
                upper0 = __shfl_sync(uint32_t(-1), upper0, upper_map[quad_idx], 4);
                lower0 = __shfl_sync(uint32_t(-1), lower0, upper_map[quad_idx] ^ 2, 4);
                frag_2(make_coord(_0{}, j, 2 * i), mi) = lane_03 ? upper0 : lower0;
                frag_2(make_coord(_0{}, j, 2 * i + 1), mi) = lane_03 ? lower0 : upper0;
            }
        }
    }
}
template <typename Engine, typename Layout>
CUTLASS_DEVICE void apply_softcap(Tensor<Engine, Layout> &tensor, float const softcap){
    #pragma unroll
    for (int i = 0; i < size(tensor); ++i) {
        tensor(i) = cutlass::fast_tanh(tensor(i) * softcap);
    }
}
template <typename Engine, typename Layout>
CUTLASS_DEVICE auto calculate_dtanh(Tensor<Engine, Layout> &tensor){
    Tensor out = make_fragment_like<float>(tensor);
    #pragma unroll
    for (int i = 0; i < size(tensor); ++i) {
        out(i) = 1.f - (tensor(i) * tensor(i));
    }
    return out;
}
template<class T>
CUTE_DEVICE T warp_prefix_sum(T val) {
    int lane = threadIdx.x % cutlass::NumThreadsPerWarp;
    CUTLASS_PRAGMA_UNROLL
    for (int i = 1; i < cutlass::NumThreadsPerWarp; i <<= 1) {
        T partial_sum = __shfl_up_sync(0xffffffff, val, i);
        if (lane >= i) { val += partial_sum; }
    }
    return val;
}
template<class T>
CUTE_DEVICE T warp_uniform(T a) {
    return __shfl_sync(0xffffffff, a, 0);
}
CUTLASS_DEVICE
int canonical_warp_group_idx_nosync() {
    return threadIdx.x / cutlass::NumThreadsPerWarpGroup;
}
}
namespace flash {
struct TileSchedulerArguments {
    int const num_blocks, num_head, num_batch, num_splits;
    int const qhead_per_khead;
    int const seqlen;
    int const seqlen_k, headdim, headdim_v, element_size;
    int* const tile_count_semaphore = nullptr;
    int const* const cu_seqlens = nullptr;
    int const* const seqused = nullptr;
    int const* const num_splits_dynamic_ptr = nullptr;
    int const* const num_m_blocks_ptr = nullptr;
    int const* const varlen_batch_idx_ptr = nullptr;
    int const* const num_nheads_in_l2_ptr = nullptr;
};
template<bool Varlen=false, bool Split=false, bool PackGQA=false, int kBlock=128>
class SingleTileScheduler {
public:
    using SharedStorage = int;
    struct Params {
        int const num_blocks, num_head, num_batch, num_splits;
        int const qhead_per_khead;
        int const seqlen;
        cutlass::FastDivmod nsplits_divmod;
        int const* const cu_seqlens;
        int const* const seqused;
        int const* const num_splits_dynamic_ptr = nullptr;
    };
    static Params
    to_underlying_arguments(TileSchedulerArguments const& args) {
        assert(!Split || !Varlen || args.num_splits_dynamic_ptr != nullptr);
        assert(!Split || !Varlen || args.num_splits < (1 << 16));
        return {args.num_blocks, args.num_head, args.num_batch, !Split ? 1 : args.num_splits,
                args.qhead_per_khead, args.seqlen,
                cutlass::FastDivmod(!Split ? 1 : args.num_splits),
                !Varlen ? nullptr : args.cu_seqlens, !Varlen ? nullptr : args.seqused,
                args.num_splits_dynamic_ptr};
    }
    static dim3
    get_grid_shape(Params const& params, int num_sm) {
        return {uint32_t(params.num_blocks), uint32_t((!Split ? 1 : params.num_splits) * params.num_head), uint32_t(params.num_batch)};
    }
    struct WorkTileInfo {
        int block_idx = 0;
        int bidh = 0;
        int bidb = 0;
        int split_idx = 0;
        CUTLASS_DEVICE
        bool
        is_valid(Params const& params) const {
            return bidb >= 0;
        }
        CUTLASS_DEVICE
        cute::tuple<int32_t, int32_t, int32_t, int32_t>
        get_block_coord(Params const& params) const {
            return {block_idx, bidh, bidb, !Split ? 0 : split_idx};
        }
    };
    CUTLASS_DEVICE
    SingleTileScheduler(SharedStorage* const smem_scheduler) { }
    template<bool IsProducerWarp=false>
    CUTLASS_DEVICE
    WorkTileInfo
    get_initial_work(Params const& params) const {
        WorkTileInfo work_info {int(blockIdx.x), int(blockIdx.y), int(blockIdx.z), 0};
        if constexpr (Split) {
            int split_idx;
            work_info.bidh = params.nsplits_divmod.divmod(split_idx, work_info.bidh);
            work_info.split_idx = split_idx;
        }
        bool is_valid_tile = true;
        if constexpr (Varlen) {
            int seqlen = params.seqused
                ? params.seqused[work_info.bidb]
                : (params.cu_seqlens ? params.cu_seqlens[work_info.bidb + 1] - params.cu_seqlens[work_info.bidb] : params.seqlen);
            if constexpr (PackGQA) { seqlen *= params.qhead_per_khead; }
            is_valid_tile = work_info.block_idx * kBlock < seqlen;
        }
        if constexpr (Varlen && Split) {
            int num_splits_dynamic = params.num_splits_dynamic_ptr ? params.num_splits_dynamic_ptr[work_info.bidb] : params.num_splits;
            is_valid_tile &= work_info.split_idx < num_splits_dynamic;
            work_info.split_idx |= (num_splits_dynamic << 16);
        }
        work_info.bidb = is_valid_tile ? work_info.bidb : -1;
        return work_info;
    }
    CUTLASS_DEVICE
    void
    init_consumer() const {}
    CUTLASS_DEVICE
    void
    prefetch_next_work(Params const& params, WorkTileInfo& current_work) const {}
    template<bool IsProducerWarp=false>
    CUTLASS_DEVICE
    WorkTileInfo
    get_next_work(Params const& params, WorkTileInfo const& current_work) const {
        return {0, 0, -1, 0};
    }
};
template<bool Split=false>
class StaticPersistentTileScheduler {
public:
    using SharedStorage = int;
    struct Params {
        int total_blocks;
        cutlass::FastDivmod m_block_divmod, head_divmod;
        cutlass::FastDivmod nsplits_divmod;
    };
    static Params
    to_underlying_arguments(TileSchedulerArguments const& args) {
        return {args.num_blocks * args.num_head * args.num_batch * (!Split ? 1 : args.num_splits),
                cutlass::FastDivmod(args.num_blocks), cutlass::FastDivmod(args.num_head * (!Split ? 1 : args.num_splits)),
                cutlass::FastDivmod(!Split ? 1 : args.num_splits)};
    }
    static dim3
    get_grid_shape(Params const& params, int num_sm) {
        return {uint32_t(num_sm)};
    }
    struct WorkTileInfo {
        int tile_idx;
        CUTLASS_DEVICE
        bool
        is_valid(Params const& params) const {
            return tile_idx < params.total_blocks;
        }
        CUTLASS_DEVICE
        cute::tuple<int32_t, int32_t, int32_t, int32_t>
        get_block_coord(Params const& params) const {
            int block, bidh, bidb;
            bidb = params.head_divmod.divmod(bidh, params.m_block_divmod.divmod(block, tile_idx));
            int split_idx = 0;
            if constexpr (Split) {
                bidh = params.nsplits_divmod.divmod(split_idx, bidh);
            }
            return {block, bidh, bidb, split_idx};
        }
    };
    CUTLASS_DEVICE
    StaticPersistentTileScheduler(SharedStorage* const smem_scheduler) {};
    template<bool IsProducerWarp=false>
    CUTLASS_DEVICE
    WorkTileInfo
    get_initial_work(Params const& params) const {
        return {int(blockIdx.x)};
    }
    CUTLASS_DEVICE
    void
    init_consumer() const {}
    CUTLASS_DEVICE
    void
    prefetch_next_work(Params const& params, WorkTileInfo& current_work) const {}
    template<bool IsProducerWarp=false>
    CUTLASS_DEVICE
    WorkTileInfo
    get_next_work(Params const& params, WorkTileInfo const& current_work) const {
        return {current_work.tile_idx + int(gridDim.x)};
    }
};
template<int NumMmaThreads=2 * cutlass::NumThreadsPerWarpGroup, int NumProducerThreads=cutlass::NumThreadsPerWarp,
        bool Split=false, bool PackGQA=false, bool WarpSpecialized=true>
class DynamicPersistentTileScheduler {
    static_assert(WarpSpecialized || NumProducerThreads == NumMmaThreads);
    static constexpr int NumThreads = WarpSpecialized ? NumMmaThreads + NumProducerThreads : NumMmaThreads;
public:
    using SharedStorage = int;
protected:
    SharedStorage* const tile_count_smem;
public:
    struct Params {
        int const total_blocks;
        cutlass::FastDivmod const m_block_divmod, head_divmod;
        cutlass::FastDivmod const l2_minor_divmod, l2_major_divmod;
        cutlass::FastDivmod const l2_minor_residual_divmod;
        int const num_hb_quotient;
        int* const tile_count_semaphore;
    };
    static Params
    to_underlying_arguments(TileSchedulerArguments const& args) {
        long long const size_one_kv_head = long(args.seqlen_k) * long(args.headdim + args.headdim_v) * long(args.element_size);
        int const size_l2 = 32 * 1024 * 1024;
        auto find_log2_floor = [&](int n) { return 31 - cutlass::clz(n); };
        int const swizzle = (size_l2 < size_one_kv_head ? 1 : (1 << find_log2_floor(size_l2 / size_one_kv_head))) * (PackGQA ? 1 : args.qhead_per_khead);
        int const num_hb_remainder = (args.num_head * args.num_batch) % swizzle;
        int const num_split_blocks = args.num_blocks * (!Split ? 1 : args.num_splits);
        assert(args.tile_count_semaphore != nullptr);
        return {num_split_blocks * args.num_head * args.num_batch,
                cutlass::FastDivmod(args.num_blocks), cutlass::FastDivmod(args.num_head),
                cutlass::FastDivmod(swizzle), cutlass::FastDivmod(swizzle * num_split_blocks),
                cutlass::FastDivmod(num_hb_remainder > 0 ? num_hb_remainder : 1),
                (args.num_head * args.num_batch) / swizzle,
                args.tile_count_semaphore};
    }
    static dim3
    get_grid_shape(Params const& params, int num_sm) {
        return {uint32_t(num_sm)};
    }
    struct WorkTileInfo {
        int tile_idx;
        CUTLASS_DEVICE
        bool
        is_valid(Params const& params) const {
            return tile_idx < params.total_blocks;
        }
        CUTLASS_DEVICE
        cute::tuple<int32_t, int32_t, int32_t, int32_t>
        get_block_coord(Params const& params) const {
            int block, bidh, bidb;
            int l2_mod, bidhb, bidhb_residual;
            bidhb = params.l2_major_divmod.divmod(l2_mod, tile_idx);
            if (bidhb < params.num_hb_quotient) {
                block = params.l2_minor_divmod.divmod(bidhb_residual, l2_mod);
            } else {
                block = params.l2_minor_residual_divmod.divmod(bidhb_residual, l2_mod);
            }
            bidb = params.head_divmod.divmod(bidh, bidhb * params.l2_minor_divmod.divisor + bidhb_residual);
            int split_idx = 0;
            if constexpr (Split) {
                split_idx = params.m_block_divmod.divmod(block, block);
            }
            block = params.m_block_divmod.divisor - 1 - block;
            return {block, bidh, bidb, split_idx};
        }
    };
    CUTLASS_DEVICE
    DynamicPersistentTileScheduler(SharedStorage* const smem_scheduler) : tile_count_smem(smem_scheduler) {};
    template<bool IsProducerWarp=false>
    CUTLASS_DEVICE
    WorkTileInfo
    get_initial_work(Params const& params) const {
        return {int(blockIdx.x)};
    }
    CUTLASS_DEVICE
    void
    init_consumer() const {
        if (WarpSpecialized || cutlass::canonical_warp_idx_sync() > 0) {
            flash::named_barrier_arrive(NumThreads, cutlass::arch::ReservedNamedBarriers::StreamkBarrier0 );
        }
    }
    CUTLASS_DEVICE
    void
    prefetch_next_work(Params const& params, WorkTileInfo& current_work) const {
        if (threadIdx.x % NumProducerThreads == 0) {
            current_work.tile_idx = atomicAdd(params.tile_count_semaphore, 1) + int(gridDim.x);
        }
    }
    template<bool IsProducerWarp=false>
    CUTLASS_DEVICE
    WorkTileInfo
    get_next_work(Params const& params, WorkTileInfo const& current_work) const {
        if constexpr (IsProducerWarp) {
            int new_tile_idx = __shfl_sync(0xffffffff, current_work.tile_idx, 0 );
            flash::named_barrier_sync(NumThreads, cutlass::arch::ReservedNamedBarriers::StreamkBarrier0 );
            if (threadIdx.x % NumProducerThreads == 0) {
                *tile_count_smem = current_work.tile_idx;
            }
            flash::named_barrier_arrive(NumThreads, cutlass::arch::ReservedNamedBarriers::StreamkBarrier1 );
            return {new_tile_idx};
        } else {
            flash::named_barrier_sync(NumThreads, cutlass::arch::ReservedNamedBarriers::StreamkBarrier1 );
            int tile_idx = *tile_count_smem;
            flash::named_barrier_arrive(NumThreads, cutlass::arch::ReservedNamedBarriers::StreamkBarrier0 );
            return {tile_idx};
        }
    }
};
template <bool Varlen, int kBlock, bool SPT = false>
class SingleTileBwdLPTScheduler {
public:
    using SharedStorage = int;
    struct Params {
        int const total_blocks;
        cutlass::FastDivmod const block_divmod, head_divmod;
        cutlass::FastDivmod const l2_minor_divmod, l2_major_divmod;
        cutlass::FastDivmod const l2_minor_residual_divmod;
        int const num_hb_quotient;
        int const seqlen;
        int const* const cu_seqlens;
        int const* const seqused;
    };
    static Params
    to_underlying_arguments(TileSchedulerArguments const& args) {
        long long const size_one_qdo_head = long(args.seqlen_k) * long(args.headdim + args.headdim_v) * long(args.element_size);
        long long const size_one_dqaccum_head = long(args.seqlen_k) * long(args.headdim) * sizeof(float);
        long long const size_one_head = size_one_qdo_head + size_one_dqaccum_head;
        int const size_l2 = 40 * 1024 * 1024;
        auto find_log2_floor = [&](int n) { return 31 - cutlass::clz(n); };
        int const swizzle = size_l2 < size_one_head ? 1 : (1 << find_log2_floor(size_l2 / size_one_head));
        int const num_hb_remainder = (args.num_head * args.num_batch) % swizzle;
        assert(args.tile_count_semaphore != nullptr);
        return {args.num_blocks * args.num_head * args.num_batch,
                cutlass::FastDivmod(args.num_blocks), cutlass::FastDivmod(args.num_head),
                cutlass::FastDivmod(swizzle), cutlass::FastDivmod(swizzle * args.num_blocks),
                cutlass::FastDivmod(num_hb_remainder > 0 ? num_hb_remainder : 1),
                (args.num_head * args.num_batch) / swizzle,
                args.seqlen, !Varlen ? nullptr : args.cu_seqlens, !Varlen ? nullptr : args.seqused};
    }
    static dim3
    get_grid_shape(Params const& params, int num_sm) {
        return {uint32_t(params.total_blocks)};
    }
    struct WorkTileInfo {
        int block;
        int bidh;
        int bidb;
        CUTLASS_DEVICE
        bool
        is_valid(Params const& params) const {
            return bidb >= 0;
        }
        CUTLASS_DEVICE
        cute::tuple<int32_t, int32_t, int32_t, int32_t>
        get_block_coord(Params const& params) const {
            return {block, bidh, bidb, 0 };
        }
    };
    CUTLASS_DEVICE
    SingleTileBwdLPTScheduler(SharedStorage* const smem_scheduler) { }
    template<bool IsProducerWarp=false>
    CUTLASS_DEVICE
    WorkTileInfo
    get_initial_work(Params const& params) const {
        int tile_idx = blockIdx.x;
        int block, bidh, bidb;
        int l2_mod, bidhb, bidhb_residual;
        bidhb = params.l2_major_divmod.divmod(l2_mod, tile_idx);
        if (bidhb < params.num_hb_quotient) {
            block = params.l2_minor_divmod.divmod(bidhb_residual, l2_mod);
        } else {
            block = params.l2_minor_residual_divmod.divmod(bidhb_residual, l2_mod);
        }
        bidb = params.head_divmod.divmod(bidh, bidhb * params.l2_minor_divmod.divisor + bidhb_residual);
        bool is_valid_tile = true;
        int num_blocks;
        if constexpr (Varlen) {
            int seqlen = params.seqused
                ? params.seqused[bidb]
                : (params.cu_seqlens ? params.cu_seqlens[bidb + 1] - params.cu_seqlens[bidb] : params.seqlen);
            num_blocks = cute::ceil_div(seqlen, Int<kBlock>{});
            is_valid_tile = block < num_blocks;
        } else {
            num_blocks = params.block_divmod.divisor;
        }
        if constexpr (SPT) {
            block = num_blocks - block - 1;
        }
        return {block, bidh, is_valid_tile ? bidb : -1};
    }
    CUTLASS_DEVICE
    void
    init_consumer() const {}
    CUTLASS_DEVICE
    void
    prefetch_next_work(Params const& params, WorkTileInfo& current_work) const {}
    template<bool IsProducerWarp=false>
    CUTLASS_DEVICE
    WorkTileInfo
    get_next_work(Params const& params, WorkTileInfo const& current_work) const {
        return {0, 0, -1};
    }
};
template<int kBlockM, int kBlockN, int NumMmaThreads=2 * cutlass::NumThreadsPerWarpGroup, int NumProducerThreads=cutlass::NumThreadsPerWarp,
         bool Split=false, bool PackGQA=false, bool WarpSpecialized=true, bool LPT = false, bool Sort = false, bool Prepared = true>
class VarlenDynamicPersistentTileScheduler {
    static_assert(WarpSpecialized || NumProducerThreads == NumMmaThreads);
    static constexpr int NumThreads = WarpSpecialized ? NumMmaThreads + NumProducerThreads : NumMmaThreads;
public:
    using SharedStorage = int4;
protected:
    SharedStorage* const work_info_smem;
public:
    struct Params {
        int num_head, num_batch;
        int const qhead_per_khead;
        int const seqlen;
        cutlass::FastDivmod head_divmod;
        cutlass::FastDivmod nsplits_divmod;
        int* const tile_count_semaphore;
        int const* const cu_seqlens;
        int const* const seqused;
        int const* const num_splits_dynamic_ptr;
        int const* const num_m_blocks_ptr;
        int const* const varlen_batch_idx_ptr;
        int const* const num_nheads_in_l2_ptr;
    };
    static Params
    to_underlying_arguments(TileSchedulerArguments const& args) {
        assert(args.tile_count_semaphore != nullptr);
        assert(args.num_head < (1 << 16));
        assert(!Split || args.num_splits < (1 << 8));
        return {args.num_head, args.num_batch,
                args.qhead_per_khead, args.seqlen,
                cutlass::FastDivmod(args.num_head),
                cutlass::FastDivmod(!Split ? 1 : args.num_splits),
                args.tile_count_semaphore, args.cu_seqlens, args.seqused,
                args.num_splits_dynamic_ptr,
                args.num_m_blocks_ptr,
                args.varlen_batch_idx_ptr,
                args.num_nheads_in_l2_ptr};
    }
    static dim3
    get_grid_shape(Params const& params, int num_sm) {
        return {uint32_t(num_sm)};
    }
    struct WorkTileInfo {
        int tile_idx, block, bidh, bidb;
        CUTLASS_DEVICE
        bool
        is_valid(Params const& params) const {
            return bidb < params.num_batch;
        }
        CUTLASS_DEVICE
        cute::tuple<int32_t, int32_t, int32_t, int32_t>
        get_block_coord(Params const& params) const {
            auto get_actual_batch = [&](int virtual_batch) {
                if constexpr(Prepared && Sort) {
                    return params.varlen_batch_idx_ptr[virtual_batch];
                } else {
                    return virtual_batch;
                }
            };
            if constexpr (!Split) {
                return {block, bidh, get_actual_batch(bidb), 0 };
            } else {
                uint32_t bidh_packed = reinterpret_cast<uint32_t const&>(bidh);
                uint32_t bidh_actual_u = bidh_packed & 0x0000FFFF;
                int bidh_actual = reinterpret_cast<int&>(bidh_actual_u);
                uint32_t split_idx_u = ((bidh_packed & 0x00FF0000) >> 16) + ((bidh_packed & 0xFF000000) >> 8);
                int split_idx = reinterpret_cast<int&>(split_idx_u);
                return {block, bidh_actual, get_actual_batch(bidb), split_idx};
            }
        }
    };
    CUTLASS_DEVICE
    VarlenDynamicPersistentTileScheduler(SharedStorage* const smem_scheduler) : work_info_smem(smem_scheduler) {};
    CUTLASS_DEVICE
    WorkTileInfo
    tile_idx_to_work_tile(Params const& params, int next_tile_idx, WorkTileInfo const& current_work) const {
        int lane = threadIdx.x % cutlass::NumThreadsPerWarp;
        auto get_num_m_blocks = [&] (int bidb_start) {
            int batch_idx = lane + bidb_start;
            if constexpr (Prepared) {
                return batch_idx < params.num_batch && lane < cutlass::NumThreadsPerWarp - 1
                    ? params.num_m_blocks_ptr[batch_idx] : 0;
            } else {
                int seqlen = params.seqlen * (!PackGQA ? 1 : params.qhead_per_khead);
                if (seqlen > kBlockM) {
                    if (params.seqused) {
                        seqlen = batch_idx < params.num_batch ? params.seqused[batch_idx] : 0;
                    } else if (params.cu_seqlens) {
                        int cur_cu_seqlen = batch_idx <= params.num_batch ? params.cu_seqlens[batch_idx] : 0;
                        int next_cu_seqlen = __shfl_down_sync(0xffffffff, cur_cu_seqlen, 1);
                        seqlen = next_cu_seqlen - cur_cu_seqlen;
                    } else {
                        seqlen = params.seqlen;
                    }
                    if constexpr (PackGQA) { seqlen *= params.qhead_per_khead; }
                }
                return batch_idx < params.num_batch && lane < cutlass::NumThreadsPerWarp - 1
                    ? cute::ceil_div(seqlen, kBlockM) : 0;
            }
        };
        auto get_num_splits = [&] (int bidb_start) {
            int batch_idx = lane + bidb_start;
            bool is_valid = batch_idx < params.num_batch && lane < cutlass::NumThreadsPerWarp - 1;
            if constexpr (!Split) {
                return is_valid ? 1 : 0;
            } else if constexpr(Prepared) {
                return is_valid ? params.num_splits_dynamic_ptr[batch_idx] : 0;
            } else {
                return is_valid ? params.nsplits_divmod.divisor : 0;
            }
        };
        int num_m_blocks = get_num_m_blocks(current_work.bidb);
        int num_splits = get_num_splits(current_work.bidb);
        int num_split_m_blocks = !Split ? num_m_blocks : num_m_blocks * num_splits;
        int num_m_blocks_cumulative = warp_prefix_sum(num_split_m_blocks);
        int m_blocks_in_group = __shfl_sync(0xffffffff, num_m_blocks_cumulative, cutlass::NumThreadsPerWarp - 1);
        int group_end_tile = current_work.tile_idx + m_blocks_in_group * params.num_head;
        int bidb = current_work.bidb;
        while (group_end_tile <= next_tile_idx) {
            bidb += cutlass::NumThreadsPerWarp - 1;
            if (bidb >= params.num_batch) {
                return {next_tile_idx, 0, 0, params.num_batch};
            }
            num_m_blocks = get_num_m_blocks(bidb);
            num_splits = get_num_splits(bidb);
            num_split_m_blocks = !Split ? num_m_blocks : num_m_blocks * num_splits;
            num_m_blocks_cumulative = warp_prefix_sum(num_split_m_blocks);
            m_blocks_in_group = __shfl_sync(0xffffffff, num_m_blocks_cumulative, cutlass::NumThreadsPerWarp - 1);
            group_end_tile += m_blocks_in_group * params.num_head;
        }
        int group_start_tile = group_end_tile - m_blocks_in_group * params.num_head;
        int batch_idx_in_group = __popc(__ballot_sync(0xffffffff, group_start_tile + num_m_blocks_cumulative * params.num_head <= next_tile_idx));
        bidb += batch_idx_in_group;
        num_m_blocks = __shfl_sync(0xffffffff, num_m_blocks, batch_idx_in_group);
        if constexpr (Split) { num_splits = __shfl_sync(0xffffffff, num_splits, batch_idx_in_group); }
        group_start_tile += (batch_idx_in_group == 0 ? 0 : __shfl_sync(0xffffffff, num_m_blocks_cumulative, batch_idx_in_group - 1)) * params.num_head;
        int mh_block = next_tile_idx - group_start_tile;
        int block, bidh;
        if constexpr (LPT) {
            if (!Split || num_splits == 1) {
                auto get_nheads_in_l2 = [&](int batch_idx) {
                    if constexpr(Prepared) {
                        return params.num_nheads_in_l2_ptr[batch_idx];
                    } else {
                        return !PackGQA ? params.qhead_per_khead : 1;
                    }
                };
                int nheads_in_l2 = get_nheads_in_l2(bidb);
                int mh_in_l2 = nheads_in_l2 * num_m_blocks;
                int section_idx = mh_block / mh_in_l2;
                int l2_mod = mh_block - section_idx * mh_in_l2;
                int nheads_remainder = params.num_head - section_idx * nheads_in_l2;
                int nheads_in_this_section = nheads_in_l2 <= nheads_remainder ? nheads_in_l2 : nheads_remainder;
                block = l2_mod / nheads_in_this_section;
                int bidh_residual = l2_mod - block * nheads_in_this_section;
                bidh = section_idx * nheads_in_l2 + bidh_residual;
                if constexpr(Split) {
                    uint32_t bidh_packed = reinterpret_cast<uint32_t&>(bidh) + (reinterpret_cast<uint32_t&>(num_splits) << 24);
                    bidh = reinterpret_cast<int&>(bidh_packed);
                }
            } else {
                bidh = mh_block / num_m_blocks;
                block = mh_block - bidh * num_m_blocks;
                if constexpr (Split) {
                    int bidh_actual = bidh / num_splits;
                    int split_idx = bidh - bidh_actual * num_splits;
                    uint32_t bidh_packed = reinterpret_cast<uint32_t&>(bidh_actual) + (reinterpret_cast<uint32_t&>(split_idx) << 16) + (reinterpret_cast<uint32_t&>(num_splits) << 24);
                    bidh = reinterpret_cast<int&>(bidh_packed);
                }
            }
            block = num_m_blocks - 1 - block;
        } else {
            bidh = mh_block / num_m_blocks;
            block = mh_block - bidh * num_m_blocks;
            if constexpr (Split) {
                int bidh_actual = bidh / num_splits;
                int split_idx = bidh - bidh_actual * num_splits;
                uint32_t bidh_packed = reinterpret_cast<uint32_t&>(bidh_actual) + (reinterpret_cast<uint32_t&>(split_idx) << 16) + (reinterpret_cast<uint32_t&>(num_splits) << 24);
                bidh = reinterpret_cast<int&>(bidh_packed);
            }
        }
        return {group_start_tile, block, bidh, bidb};
    }
    template<bool IsProducerWarp=false>
    CUTLASS_DEVICE
    WorkTileInfo
    get_initial_work(Params const& params) const {
        if constexpr (IsProducerWarp) {
            WorkTileInfo work_info = tile_idx_to_work_tile(params, int(blockIdx.x), {0, 0, 0, 0});
            if (threadIdx.x % cutlass::NumThreadsPerWarp == 0) {
                *work_info_smem = make_int4(work_info.tile_idx, work_info.block, work_info.bidh, work_info.bidb);
            }
            flash::named_barrier_arrive(NumThreads, cutlass::arch::ReservedNamedBarriers::StreamkBarrier1 );
            return work_info;
        } else {
            return get_next_work<false>(params, {0, 0, 0, 0});
        }
    }
    CUTLASS_DEVICE
    void
    init_consumer() const {
    }
    CUTLASS_DEVICE
    void
    prefetch_next_work(Params const& params, WorkTileInfo& current_work) const {
        if (threadIdx.x % NumProducerThreads == 0) {
            current_work.tile_idx = atomicAdd(params.tile_count_semaphore, 1) + int(gridDim.x);
        }
    }
    template<bool IsProducerWarp=false>
    CUTLASS_DEVICE
    WorkTileInfo
    get_next_work(Params const& params, WorkTileInfo const& current_work) const {
        if constexpr (IsProducerWarp) {
            int new_tile_idx = __shfl_sync(0xffffffff, current_work.tile_idx, 0 );
            WorkTileInfo work_info = {__shfl_sync(0xffffffff, current_work.tile_idx, 1 ), current_work.block, current_work.bidh, current_work.bidb};
            work_info = tile_idx_to_work_tile(params, new_tile_idx, work_info);
            flash::named_barrier_sync(NumThreads, cutlass::arch::ReservedNamedBarriers::StreamkBarrier0 );
            if (threadIdx.x % cutlass::NumThreadsPerWarp == 0) {
                *work_info_smem = make_int4(work_info.tile_idx, work_info.block, work_info.bidh, work_info.bidb);
            }
            flash::named_barrier_arrive(NumThreads, cutlass::arch::ReservedNamedBarriers::StreamkBarrier1 );
            return work_info;
        } else {
            flash::named_barrier_sync(NumThreads, cutlass::arch::ReservedNamedBarriers::StreamkBarrier1 );
            int4 work_info = *work_info_smem;
            flash::named_barrier_arrive(NumThreads, cutlass::arch::ReservedNamedBarriers::StreamkBarrier0 );
            return WorkTileInfo{work_info.x, work_info.y, work_info.z, work_info.w};
        }
    }
};
template <class T, class = void> struct is_sk2_sched { static constexpr bool value = false; };
template <class T> struct is_sk2_sched<T, std::void_t<decltype(T::IsSK2)>> { static constexpr bool value = T::IsSK2; };
template <int NumMmaThreads, int NumProducerThreads, bool WarpSpecialized>
class SK2Scheduler {
public:
    static constexpr bool IsSK2 = true;
private:
    static constexpr int NumThreads = WarpSpecialized ? NumMmaThreads + NumProducerThreads : NumMmaThreads;
public:
    struct SharedStorage { int4 a; int4 b; int4 c; };
protected:
    SharedStorage* const work_info_smem;
public:
    struct Params {
        int num_batch;
        int const* tp;
        int const* desc;
        int const* st;
        int* cnt;
    };
    static Params to_underlying_arguments(TileSchedulerArguments const& args) {
        return {args.num_batch, args.num_m_blocks_ptr, args.varlen_batch_idx_ptr, args.num_splits_dynamic_ptr,
                args.tile_count_semaphore};
    }
    static dim3 get_grid_shape(Params const& params, int num_sm) { return {uint32_t(num_sm)}; }
    struct WorkTileInfo {
        int code, block, bidh, bidb;
        int t, np, slot, r1;
        CUTLASS_DEVICE bool is_valid(Params const& params) const { return bidb < params.num_batch; }
        CUTLASS_DEVICE cute::tuple<int32_t, int32_t, int32_t, int32_t> get_block_coord(Params const& params) const {
            return {block, bidh, bidb, code};
        }
    };
    CUTLASS_DEVICE SK2Scheduler(SharedStorage* const smem_scheduler) : work_info_smem(smem_scheduler) {};
    static CUTLASS_DEVICE int cta_of(long long x, long long W, int G) {
        return int(((x + 1) * G + W - 1) / W) - 1;
    }
    CUTLASS_DEVICE WorkTileInfo piece(Params const& params, int t, int r0, int r1, int slot) const {
        int const G = int(gridDim.x);
        int const W = params.st[G], K = params.st[G + 1];
        int const t0 = params.tp[t], t1 = params.tp[t + 1];
        int const d = params.desc[t];
        int const nb = t1 - t0 - K;
        int const lo = max(r0 - t0 - K, 0), hi = min(r1 - t0 - K, nb);
        if (hi <= lo) { return invalid(params); }
        int const cf = cta_of(t0 + K, W, G), cl = cta_of(t1 - 1, W, G);
        WorkTileInfo w;
        w.code = lo | (hi << 11) | (1 << 30);
        w.block = d >> 16; w.bidh = (d >> 8) & 0xFF; w.bidb = d & 0xFF;
        w.t = t; w.np = cl - cf + 1; w.slot = slot; w.r1 = r1;
        return w;
    }
    CUTLASS_DEVICE WorkTileInfo invalid(Params const& params) const {
        WorkTileInfo w{};
        w.bidb = params.num_batch;
        return w;
    }
    CUTLASS_DEVICE void publish(WorkTileInfo const& w) const {
        if (threadIdx.x % cutlass::NumThreadsPerWarp == 0) {
            work_info_smem->a = make_int4(w.code, w.block, w.bidh, w.bidb);
            work_info_smem->b = make_int4(w.t, w.np, w.slot, w.r1);
        }
    }
    template<bool IsProducerWarp=false>
    CUTLASS_DEVICE WorkTileInfo get_initial_work(Params const& params) const {
        if constexpr (IsProducerWarp) {
            int const G = int(gridDim.x), c = int(blockIdx.x);
            long long const W = params.st[G];
            int const r0 = int(c * W / G), r1 = int((c + 1) * W / G);
            WorkTileInfo w = r0 < r1 ? piece(params, params.st[c], r0, r1, 0) : invalid(params);
            publish(w);
            flash::named_barrier_arrive(NumThreads, cutlass::arch::ReservedNamedBarriers::StreamkBarrier1 );
            return w;
        } else {
            return get_next_work<false>(params, WorkTileInfo{});
        }
    }
    CUTLASS_DEVICE void init_consumer() const {}
    CUTLASS_DEVICE void prefetch_next_work(Params const& params, WorkTileInfo& current_work) const {}
    template<bool IsProducerWarp=false>
    CUTLASS_DEVICE WorkTileInfo get_next_work(Params const& params, WorkTileInfo const& cur) const {
        if constexpr (IsProducerWarp) {
            WorkTileInfo w = invalid(params);
            if (cur.bidb < params.num_batch) {
                int const nt = cur.t + 1;
                if (params.tp[nt] < cur.r1) { w = piece(params, nt, params.tp[nt], cur.r1, 1); }
            }
            flash::named_barrier_sync(NumThreads, cutlass::arch::ReservedNamedBarriers::StreamkBarrier0 );
            publish(w);
            flash::named_barrier_arrive(NumThreads, cutlass::arch::ReservedNamedBarriers::StreamkBarrier1 );
            return w;
        } else {
            flash::named_barrier_sync(NumThreads, cutlass::arch::ReservedNamedBarriers::StreamkBarrier1 );
            int4 a = work_info_smem->a, b = work_info_smem->b;
            flash::named_barrier_arrive(NumThreads, cutlass::arch::ReservedNamedBarriers::StreamkBarrier0 );
            return WorkTileInfo{a.x, a.y, a.z, a.w, b.x, b.y, b.z, b.w};
        }
    }
};
}
#include "cute/tensor.hpp"
#include <cutlass/cutlass.h>
#include <cutlass/arch/reg_reconfig.h>
#include <cutlass/array.h>
#include <cutlass/numeric_types.h>
#include <cutlass/numeric_conversion.h>
#include <cutlass/kernel_hardware_info.h>
#include "cutlass/pipeline/pipeline.hpp"
#include "cutlass/arch/grid_dependency_control.h"
namespace flash {
template <bool Varlen, int kBlock>
struct SeqlenInfo {
    int const offset, offset_padded;
    int const seqlen;
    CUTLASS_DEVICE
    SeqlenInfo(int const bidb, int const seqlen_static, int const* const cu_seqlens, int const* const seqused)
        : offset(!Varlen || cu_seqlens == nullptr ? 0 : cu_seqlens[bidb])
        , offset_padded(!Varlen || cu_seqlens == nullptr ? 0 : (cu_seqlens[bidb] + bidb * kBlock) / kBlock * kBlock)
        , seqlen(!Varlen
                 ? seqlen_static
                 : (seqused ? seqused[bidb] : (cu_seqlens ? cu_seqlens[bidb + 1] - cu_seqlens[bidb] : seqlen_static)))
    {
    }
};
template <bool Varlen, int kBlockM>
struct SeqlenInfoQK {
    int const offset_q, offset_k, offset_q_padded;
    int const seqlen_q, seqlen_k;
    CUTLASS_DEVICE
    SeqlenInfoQK(int const bidb, int const seqlen_q_static, int const seqlen_k_static,
                 int const* const cu_seqlens_q, int const* const cu_seqlens_k,
                 int const* const seqused_q, int const* const seqused_k
                 )
        : offset_q(!Varlen || cu_seqlens_q == nullptr ? 0 : cu_seqlens_q[bidb])
        , offset_k(!Varlen || cu_seqlens_k == nullptr ? 0 : cu_seqlens_k[bidb])
        , offset_q_padded(!Varlen || cu_seqlens_q == nullptr ? 0 : (cu_seqlens_q[bidb] + bidb * kBlockM) / kBlockM * kBlockM)
        , seqlen_q(!Varlen
                   ? seqlen_q_static
                   : (seqused_q ? seqused_q[bidb] : (cu_seqlens_q ? cu_seqlens_q[bidb + 1] - cu_seqlens_q[bidb] : seqlen_q_static)))
        , seqlen_k(!Varlen
                   ? seqlen_k_static
                   : (seqused_k ? seqused_k[bidb] : (cu_seqlens_k ? cu_seqlens_k[bidb + 1] - cu_seqlens_k[bidb] : seqlen_k_static)))
    {
    }
};
template <bool Varlen, bool AppendKV>
struct SeqlenInfoQKNewK {
    static_assert(!(AppendKV && !Varlen), "AppendKV is only supported with Varlen");
    int const leftpad_k;
    int const offset_q, offset_k, offset_k_new;
    int const seqlen_q, seqlen_k_og, seqlen_k_new, seqlen_k, seqlen_rotary;
    CUTLASS_DEVICE
    SeqlenInfoQKNewK(int const bidb, int const seqlen_q_static, int const seqlen_k_static, int const shape_K_new_0,
                     int const* const cu_seqlens_q, int const* const cu_seqlens_k, int const* const cu_seqlens_k_new,
                     int const* const seqused_q, int const* const seqused_k, int const* const ptr_leftpad_k,
                     int const* const seqlens_rotary
                     )
        : leftpad_k(ptr_leftpad_k ? ptr_leftpad_k[bidb] : 0)
        , offset_q(!Varlen || cu_seqlens_q == nullptr ? 0 : cu_seqlens_q[bidb])
        , offset_k(!Varlen ? 0 : (cu_seqlens_k ? cu_seqlens_k[bidb] : 0) + leftpad_k)
        , offset_k_new(!AppendKV || cu_seqlens_k_new == nullptr ? 0 : cu_seqlens_k_new[bidb])
        , seqlen_q(!Varlen
                   ? seqlen_q_static
                   : (seqused_q ? seqused_q[bidb] : (cu_seqlens_q ? cu_seqlens_q[bidb + 1] - cu_seqlens_q[bidb] : seqlen_q_static)))
        , seqlen_k_og(!Varlen
                      ? seqlen_k_static
                      : (seqused_k ? seqused_k[bidb] : (cu_seqlens_k ? cu_seqlens_k[bidb + 1] - cu_seqlens_k[bidb] : seqlen_k_static)) - leftpad_k)
        , seqlen_k_new(!AppendKV
                       ? 0
                       : (cu_seqlens_k_new ? cu_seqlens_k_new[bidb + 1] - cu_seqlens_k_new[bidb] : shape_K_new_0))
        , seqlen_k(!AppendKV ? seqlen_k_og : seqlen_k_og + seqlen_k_new)
        , seqlen_rotary(!AppendKV || !seqlens_rotary ? seqlen_k_og + leftpad_k : seqlens_rotary[bidb])
    {
    }
};
}
#include <cmath>
#include <cute/tensor.hpp>
#include <cutlass/numeric_types.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
namespace flash {
using namespace cute;
template<bool zero_init=true, typename Engine0, typename Layout0, typename Engine1, typename Layout1, typename Operator>
__device__ __forceinline__ void thread_reduce_(Tensor<Engine0, Layout0> const &tensor, Tensor<Engine1, Layout1> &summary, Operator &op) {
    static_assert(Layout0::rank == 2, "Only support 2D Tensor");
    static_assert(Layout1::rank == 1, "Only support 1D Tensor");
    CUTE_STATIC_ASSERT_V(size<0>(summary) == size<0>(tensor));
    #pragma unroll
    for (int ni = 0; ni < size<1>(tensor); ni++) {
        #pragma unroll
        for (int mi = 0; mi < size<0>(tensor); mi++) {
            summary(mi) = zero_init && ni == 0 ? tensor(mi, ni) : op(summary(mi), tensor(mi, ni));
        }
    }
}
template<typename Engine0, typename Layout0, typename Engine1, typename Layout1, typename Operator>
__device__ __forceinline__ void quad_allreduce_(Tensor<Engine0, Layout0> &dst, Tensor<Engine1, Layout1> &src, Operator &op) {
    CUTE_STATIC_ASSERT_V(size(dst) == size(src));
    #pragma unroll
    for (int i = 0; i < size(dst); i++) {
        dst(i) = Allreduce<4>::run(src(i), op);
    }
}
template<bool zero_init=true, typename Engine0, typename Layout0, typename Engine1, typename Layout1, typename Operator>
__device__ __forceinline__ void reduce_(Tensor<Engine0, Layout0> const& tensor, Tensor<Engine1, Layout1> &summary, Operator &op) {
    thread_reduce_<zero_init>(tensor, summary, op);
    quad_allreduce_(summary, summary, op);
}
template<bool zero_init=true, typename Engine0, typename Layout0, typename Engine1, typename Layout1>
__device__ __forceinline__ void reduce_max(Tensor<Engine0, Layout0> const& tensor, Tensor<Engine1, Layout1> &max){
    MaxOp<float> max_op;
    reduce_<zero_init>(tensor, max, max_op);
}
template<bool zero_init=true, bool warp_reduce=true, typename Engine0, typename Layout0, typename Engine1, typename Layout1>
__device__ __forceinline__ void reduce_sum(Tensor<Engine0, Layout0> const& tensor, Tensor<Engine1, Layout1> &sum){
    SumOp<float> sum_op;
    thread_reduce_<zero_init>(tensor, sum, sum_op);
    if constexpr (warp_reduce) { quad_allreduce_(sum, sum, sum_op); }
}
__device__ __forceinline__ float att_ex2_poly(float x) {
    x = fmaxf(x, -64.f);
    float const t = __fadd_rn(x, 12582912.f);
    float const f = __fsub_rn(x, __fsub_rn(t, 12582912.f));
    float const p = __fmaf_rn(__fmaf_rn(__fmaf_rn(5.521949008e-02f, f, 2.426151186e-01f), f, 6.932522058e-01f), f, 9.999275804e-01f);
    return __uint_as_float(__float_as_uint(t) * (1u << 23) + __float_as_uint(p));
}
template <bool Scale_max=true, bool Check_inf=true, int Max_offset=0,
        typename Engine0, typename Layout0, typename Engine1, typename Layout1>
__forceinline__ __device__ void scale_apply_exp2(Tensor<Engine0, Layout0> &tensor, Tensor<Engine1, Layout1> const &max, const float scale) {
    static constexpr float max_offset = float(Max_offset);
    static_assert(Layout0::rank == 2, "Only support 2D Tensor");
    static_assert(Layout1::rank == 1, "Only support 1D Tensor");
    CUTE_STATIC_ASSERT_V(size<0>(max) == size<0>(tensor));
    #pragma unroll
    for (int mi = 0; mi < size<0>(tensor); ++mi) {
        const float max_scaled = Check_inf
            ? (max(mi) == -INFINITY ? 0.f : (!Scale_max ? max(mi) : max(mi) * scale) - max_offset)
            : (!Scale_max ? max(mi) : max(mi) * scale) - max_offset;
        #pragma unroll
        for (int ni = 0; ni < size<1>(tensor); ++ni)  {
            tensor(mi, ni) = exp2f(tensor(mi, ni) * scale - max_scaled);
        }
    }
}
template<typename TensorP>
__forceinline__ __device__ void att_issue_sum(TensorP const &tOrP, uint64_t desc, float (&osum)[4]) {
    Tensor a = recast<uint32_t const>(tOrP);
    constexpr int NK = CUTE_STATIC_V(size(a)) / 4;
    asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory");
    #pragma unroll
    for (int k = 0; k < NK; ++k) {
        if (k == 0) {
            asm volatile("{\n.reg .pred p;\nsetp.ne.b32 p, 0, 0;\n"
                         "wgmma.mma_async.sync.aligned.m64n8k32.f32.e4m3.e4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, %8, p, 1, 1;\n}\n"
                         : "+f"(osum[0]), "+f"(osum[1]), "+f"(osum[2]), "+f"(osum[3])
                         : "r"(a(4 * k)), "r"(a(4 * k + 1)), "r"(a(4 * k + 2)), "r"(a(4 * k + 3)), "l"(desc));
        } else {
            asm volatile("{\n.reg .pred p;\nsetp.ne.b32 p, 1, 0;\n"
                         "wgmma.mma_async.sync.aligned.m64n8k32.f32.e4m3.e4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, %8, p, 1, 1;\n}\n"
                         : "+f"(osum[0]), "+f"(osum[1]), "+f"(osum[2]), "+f"(osum[3])
                         : "r"(a(4 * k)), "r"(a(4 * k + 1)), "r"(a(4 * k + 2)), "r"(a(4 * k + 3)), "l"(desc));
        }
    }
}
template <int kNRows, int Max_offset=0>
struct Softmax {
    using TensorT = decltype(make_tensor<float>(Shape<Int<kNRows>>{}));
    TensorT row_max, row_sum;
    float osum[4];
    float const softmax_scale_log2;
    bool has_sink = false;
    TensorT row_sink;
    CUTLASS_DEVICE Softmax(float const softmax_scale_log2_) : softmax_scale_log2(softmax_scale_log2_) {};
    template<bool Is_first, bool Check_inf=false, typename Tensor0>
    __forceinline__ __device__ TensorT max_get_scale(Tensor0 &acc_s) {
        Tensor scores = make_tensor(acc_s.data(), flash::convert_layout_acc_rowcol(acc_s.layout()));
        static_assert(CUTE_STATIC_V(size<0>(scores)) == kNRows);
        TensorT scores_scale;
        if constexpr (Is_first) {
            flash::template reduce_max<true>(scores, row_max);
            cute::fill(scores_scale, 1.f);
        } else {
            Tensor scores_max_prev = make_fragment_like(row_max);
            cute::copy(row_max, scores_max_prev);
            flash::template reduce_max<false>(scores, row_max);
            #pragma unroll
            for (int mi = 0; mi < size(row_max); ++mi) {
                float scores_max_cur = !Check_inf
                    ? row_max(mi)
                    : (row_max(mi) == -INFINITY ? 0.0f : row_max(mi));
                scores_scale(mi) = exp2f((scores_max_prev(mi) - scores_max_cur) * softmax_scale_log2);
                row_sum(mi) *= scores_scale(mi);
            }
        }
        return scores_scale;
    };
    template<bool Is_packGQA, typename Tensor0>
    __forceinline__ __device__ void set_row_sink(Tensor0 &sSink, int bidh, int qhead_per_khead = 0, int row_start = 0, int row_step = 0) {
        has_sink = true;
        if constexpr (!Is_packGQA) {
            #pragma unroll
            for(int mi = 0; mi < size(row_sink); ++mi) {
                row_sink(mi) = static_cast<float>(sSink(bidh));
            }
        } else {
            #pragma unroll
            for(int mi = 0; mi < size(row_sink); ++mi) {
                int row   = row_start + mi * row_step;
                int head  = bidh * qhead_per_khead + row % qhead_per_khead;
                row_sink(mi) = static_cast<float>(sSink(head));
            }
        }
    }
    template<bool Is_first, bool Check_inf=false, typename Tensor0>
    __forceinline__ __device__ void online_softmax(Tensor0 &acc_s) {
        Tensor scores = make_tensor(acc_s.data(), flash::convert_layout_acc_rowcol(acc_s.layout()));
        static_assert(CUTE_STATIC_V(size<0>(scores)) == kNRows);
        flash::template scale_apply_exp2<true, Check_inf, Max_offset>(scores, row_max, softmax_scale_log2);
        if constexpr (Is_first) { cute::fill(row_sum, 0.f); }
    };
    __forceinline__ __device__ void clear_sum() { osum[0] = osum[1] = osum[2] = osum[3] = 0.f; }
    __forceinline__ __device__ void add_sum(TensorT const &scores_scale) {
        row_sum(0) += osum[0] * scores_scale(0);
        row_sum(1) += osum[2] * scores_scale(1);
    }
    template<typename TensorP>
    __forceinline__ __device__ void issue_sum(TensorP const &tOrP, uint64_t desc) { att_issue_sum(tOrP, desc, osum); }
    __forceinline__ __device__ TensorT finalize(float const final_scale=1.f) {
        row_sum(0) += osum[0];
        row_sum(1) += osum[2];
        TensorT scores_scale;
        #pragma unroll
        for (int mi = 0; mi < size(row_sum); ++mi) {
            if (row_max(mi) == -INFINITY) { row_max(mi) = 0.f; }
            float sum = row_sum(mi);
            if (has_sink) {
                const float max_scaled = row_max(mi) * softmax_scale_log2 - Max_offset;
                float const extra = exp2f(float(M_LOG2E) * row_sink(mi) - max_scaled);
                sum += extra;
            }
            float inv_sum = (sum == 0.f || sum != sum) ? 0.f : 1.f / sum;
            scores_scale(mi) = inv_sum * final_scale;
            if constexpr (Max_offset != 0) {
                static constexpr float sum_scale = 1.f / float(1 << Max_offset);
                sum *= sum_scale;
            }
            row_sum(mi) = (sum == 0.f || sum != sum) ? -INFINITY : row_max(mi) * (softmax_scale_log2 * float(M_LN2)) + __logf(sum);
        }
        return scores_scale;
    };
    template<typename Tensor1>
    __forceinline__ __device__ void rescale_o(Tensor1 &acc_o, TensorT const &scores_scale) {
        Tensor acc_o_rowcol = make_tensor(acc_o.data(), flash::convert_layout_acc_rowcol(acc_o.layout()));
        static_assert(CUTE_STATIC_V(size<0>(acc_o_rowcol)) == kNRows);
        #pragma unroll
        for (int mi = 0; mi < size<0>(acc_o_rowcol); ++mi) {
            #pragma unroll
            for (int ni = 0; ni < size<1>(acc_o_rowcol); ++ni) { acc_o_rowcol(mi, ni) *= scores_scale(mi); }
        }
    };
};
#define ATT_POLY_MASK 0
#define ATT_MAX_SLACK 0.5f
__device__ __forceinline__ uint32_t att_cvt_h2(float lo, float hi) {
    uint32_t r; asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(r) : "f"(hi), "f"(lo)); return r;
}
__device__ __forceinline__ uint32_t att_ex2_h2(uint32_t x) {
    uint32_t y; asm("ex2.approx.f16x2 %0, %1;" : "=r"(y) : "r"(x)); return y;
}
__device__ __forceinline__ __half2 att_as_h2(uint32_t x) { return *reinterpret_cast<__half2*>(&x); }
__device__ __forceinline__ uint32_t att_as_u(__half2 x) { return *reinterpret_cast<uint32_t*>(&x); }
__device__ __forceinline__ uint32_t att_ex2_poly_h2(__half2 y) {
    __half2 const magic = __float2half2_rn(1024.f);
    __half2 t = __hadd2(y, magic);
    __half2 f = __hsub2(y, __hsub2(t, magic));
    __half2 p = __hfma2(__hfma2(__float2half2_rn(0.23734974f), f, __float2half2_rn(0.70128093f)), f,
                        __float2half2_rn(1.00035163f));
    uint32_t e = (att_as_u(t) & 0x001F001Fu) << 10;
    return att_as_u(__hmul2(p, att_as_h2(e)));
}
__device__ __forceinline__ uint16_t att_e4m3x2(uint32_t h) {
    uint16_t r; asm("cvt.rn.satfinite.e4m3x2.f16x2 %0, %1;" : "=h"(r) : "r"(h)); return r;
}
template <int kNRows, int Max_offset=0>
struct SoftmaxH {
    using TensorT = decltype(make_tensor<float>(Shape<Int<kNRows>>{}));
    TensorT row_max, row_sum;
    float const softmax_scale_log2;
    bool has_sink = false;
    TensorT row_sink;
    CUTLASS_DEVICE SoftmaxH(float const softmax_scale_log2_) : softmax_scale_log2(softmax_scale_log2_) {};
    template<bool Is_packGQA, typename Tensor0>
    __forceinline__ __device__ void set_row_sink(Tensor0 &sSink, int bidh, int qhead_per_khead = 0, int row_start = 0, int row_step = 0) {}
    template<bool Is_first, bool Check_inf, typename Tensor0, int NW>
    __forceinline__ __device__ TensorT step(Tensor0 const &acc_s, uint32_t (&h)[NW]) {
        static_assert(kNRows == 2);
        static_assert(CUTE_STATIC_V(size(acc_s)) == 2 * NW);
        if constexpr (sizeof(typename Tensor0::value_type) == 2) {
            Tensor acc_u = recast<uint32_t const>(acc_s);
            #pragma unroll
            for (int w = 0; w < NW; ++w) { h[w] = acc_u(w); }
        } else {
            #pragma unroll
            for (int w = 0; w < NW; ++w) { h[w] = att_cvt_h2(acc_s(2 * w), acc_s(2 * w + 1)); }
        }
#define ATT_ABL 0
        __half2 m2[2] = {att_as_h2(h[0]), att_as_h2(h[1])};
        if constexpr ((ATT_ABL & 2) == 0) {
        #pragma unroll
        for (int w = 2; w < NW; ++w) { m2[w & 1] = __hmax2(m2[w & 1], att_as_h2(h[w])); }
        } else { m2[0] = m2[1] = __float2half2_rn(30.f); }
        TensorT scores_scale;
        #pragma unroll
        for (int r = 0; r < 2; ++r) {
            float m = fmaxf(__low2float(m2[r]), __high2float(m2[r]));
            m = fmaxf(m, __shfl_xor_sync(0xffffffff, m, 2));
            m = fmaxf(m, __shfl_xor_sync(0xffffffff, m, 1));
            if constexpr (Is_first) {
                row_max(r) = m;
                scores_scale(r) = 1.f;
            } else {
                float const prev = row_max(r);
                float cur = fmaxf(prev, m);
                if ((cur - prev) * softmax_scale_log2 <= float(ATT_MAX_SLACK)) { cur = prev; }
                float const sc = cur == prev ? 1.f : exp2f((prev - cur) * softmax_scale_log2);
                scores_scale(r) = sc;
                row_max(r) = cur;
                row_sum(r) *= sc;
            }
        }
        __half2 const sc2 = __float2half2_rn(softmax_scale_log2);
        __half2 nms[2], nms15[2];
        #pragma unroll
        for (int r = 0; r < 2; ++r) {
            float const ms = Check_inf
                ? (row_max(r) == -INFINITY ? 0.f : row_max(r) * softmax_scale_log2 - float(Max_offset))
                : row_max(r) * softmax_scale_log2 - float(Max_offset);
            nms[r] = __float2half2_rn(-ms);
            nms15[r] = __float2half2_rn(15.f - ms);
        }
        __half2 sum2[2] = {__float2half2_rn(0.f), __float2half2_rn(0.f)};
        #pragma unroll
        for (int w = 0; w < NW; ++w) {
            uint32_t p;
            if (((ATT_POLY_MASK >> (w % 8)) & 1) != 0) {
                p = att_ex2_poly_h2(__hfma2_relu(att_as_h2(h[w]), sc2, nms15[w & 1]));
            } else if constexpr ((ATT_ABL & 1) != 0) {
                p = att_as_u(__hfma2(att_as_h2(h[w]), sc2, nms[w & 1]));
            } else {
                p = att_ex2_h2(att_as_u(__hfma2(att_as_h2(h[w]), sc2, nms[w & 1])));
            }
            sum2[w & 1] = __hadd2(sum2[w & 1], att_as_h2(p));
            h[w] = p;
        }
        #pragma unroll
        for (int r = 0; r < 2; ++r) {
            float const s = __low2float(sum2[r]) + __high2float(sum2[r]);
            if constexpr (Is_first) { row_sum(r) = s; } else { row_sum(r) += s; }
        }
        return scores_scale;
    };
    template<int NW, typename TensorP>
    __forceinline__ __device__ void pack_p(uint32_t const (&h)[NW], TensorP &tOrP) {
        Tensor a = recast<uint32_t>(tOrP);
        static_assert(CUTE_STATIC_V(size(a)) * 2 == NW);
        #pragma unroll
        for (int i = 0; i < NW / 4; ++i) {
            a(2 * i) = uint32_t(att_e4m3x2(h[4 * i])) | (uint32_t(att_e4m3x2(h[4 * i + 2])) << 16);
            a(2 * i + 1) = uint32_t(att_e4m3x2(h[4 * i + 1])) | (uint32_t(att_e4m3x2(h[4 * i + 3])) << 16);
        }
    }
    __forceinline__ __device__ TensorT finalize(float const final_scale=1.f) {
        SumOp<float> sum_op;
        quad_allreduce_(row_sum, row_sum, sum_op);
        TensorT scores_scale;
        #pragma unroll
        for (int mi = 0; mi < size(row_sum); ++mi) {
            if (row_max(mi) == -INFINITY) { row_max(mi) = 0.f; }
            float sum = row_sum(mi);
            float inv_sum = (sum == 0.f || sum != sum) ? 0.f : 1.f / sum;
            scores_scale(mi) = inv_sum * final_scale;
            if constexpr (Max_offset != 0) {
                static constexpr float sum_scale = 1.f / float(1 << Max_offset);
                sum *= sum_scale;
            }
            row_sum(mi) = (sum == 0.f || sum != sum) ? -INFINITY : row_max(mi) * (softmax_scale_log2 * float(M_LN2)) + __logf(sum);
        }
        return scores_scale;
    };
    template<typename Tensor1>
    __forceinline__ __device__ void rescale_o(Tensor1 &acc_o, TensorT const &scores_scale) {
        Tensor acc_o_rowcol = make_tensor(acc_o.data(), flash::convert_layout_acc_rowcol(acc_o.layout()));
        static_assert(CUTE_STATIC_V(size<0>(acc_o_rowcol)) == kNRows);
        #pragma unroll
        for (int mi = 0; mi < size<0>(acc_o_rowcol); ++mi) {
            if (scores_scale(mi) != 1.f) {
                #pragma unroll
                for (int ni = 0; ni < size<1>(acc_o_rowcol); ++ni) { acc_o_rowcol(mi, ni) *= scores_scale(mi); }
            }
        }
    };
};
__device__ __forceinline__ uint32_t att_e4m3x2_to_h2(uint16_t x) {
    uint32_t r; asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(r) : "h"(x)); return r;
}
__device__ __forceinline__ uint32_t att_prmt(uint32_t a, uint32_t b, uint32_t sel) {
    uint32_t r; asm("prmt.b32 %0, %1, %2, %3;" : "=r"(r) : "r"(a), "r"(b), "r"(sel)); return r;
}
__device__ __forceinline__ uint64_t att_ones_desc(void const* p) {
    uint32_t const a = static_cast<uint32_t>(__cvta_generic_to_shared(p));
    return uint64_t((a & 0x3FFFF) >> 4) | (uint64_t(128 >> 4) << 16) | (uint64_t(256 >> 4) << 32);
}
// The kernel's softmax (see the header). step() takes a tile of FP16 scores, updates the running max of the
// thread's two rows, and writes the E4M3 codes of P over them (h[]); B = U_PB 2^-24 - max 2^-21 places the row max at
// code U_PB. When the max moves, O is rescaled by exp2(dB 2^21) = exp2(max_old - max_new). The non-AT_SUB branches
// (f16 frame [1024, 2048), slope a32) and the row-sum members (osum, issue_sum, add_sum) are not used in this build.
template <int kNRows, int Max_offset=0>
struct SoftmaxL {
    using TensorT = decltype(make_tensor<float>(Shape<Int<kNRows>>{}));
    TensorT row_max, row_sum, row_b;
    uint32_t rmh, rbh;
    bool fx_chg = true;
    float osum[4];
    float const softmax_scale_log2;
    float const a32;
    bool has_sink = false;
    CUTLASS_DEVICE SoftmaxL(float const softmax_scale_log2_)
        : softmax_scale_log2(softmax_scale_log2_), a32(__half2float(__float2half_rn(8.f * softmax_scale_log2_))) {};
    template<bool Is_packGQA, typename Tensor0>
    __forceinline__ __device__ void set_row_sink(Tensor0 &sSink, int bidh, int qhead_per_khead = 0, int row_start = 0, int row_step = 0) {}
    template<bool Is_first, bool Check_inf, typename Tensor0, int NW>
    __forceinline__ __device__ TensorT step(Tensor0 const &acc_s, uint32_t (&h)[NW]) {
        static_assert(kNRows == 2);
        static_assert(CUTE_STATIC_V(size(acc_s)) == 2 * NW);
        static_assert(NW % 4 == 0);
        if constexpr (sizeof(typename Tensor0::value_type) == 2) {
            Tensor acc_u = recast<uint32_t const>(acc_s);
            #pragma unroll
            for (int w = 0; w < NW; ++w) { h[w] = acc_u(w); }
        } else {
            #pragma unroll
            for (int w = 0; w < NW; ++w) { h[w] = att_cvt_h2(acc_s(2 * w), acc_s(2 * w + 1)); }
        }
        __half2 mp[8];
        #pragma unroll
        for (int k = 0; k < 8; ++k) { mp[k] = att_as_h2(h[k]); }
        #pragma unroll
        for (int w = 8; w < NW; ++w) { mp[w % 8] = __hmax2(mp[w % 8], att_as_h2(h[w])); }
        #pragma unroll
        for (int k = 0; k < 4; ++k) { mp[k] = __hmax2(mp[k], mp[k + 4]); }
        __half2 const r0 = __hmax2(mp[0], mp[2]), r1 = __hmax2(mp[1], mp[3]);
        __half2 mm = __hmax2(__lows2half2(r0, r1), __highs2half2(r0, r1));
        uint32_t mu = att_as_u(mm);
        mm = __hmax2(mm, att_as_h2(__shfl_xor_sync(0xffffffff, mu, 2)));
        mu = att_as_u(mm);
        mm = __hmax2(mm, att_as_h2(__shfl_xor_sync(0xffffffff, mu, 1)));
        TensorT scores_scale;
        __half2 b2[2];
        __half2 const cur = Is_first ? mm : __hmax2(att_as_h2(rmh), mm);
#ifdef AT_SUB
        __half2 const bn = __hmin2(__hfma2(cur, __float2half2_rn(-0x1p-21f), __float2half2_rn(float(U_PB) * 0x1p-24f)), __float2half2_rn(65504.f));
#else
        __half2 const bn = __hmin2(__hfma2(cur, __hneg2(__float2half2_rn(a32)), __float2half2_rn(1144.f)), __float2half2_rn(65504.f));
#endif
        uint32_t const bnu = att_as_u(bn);
        scores_scale(0) = scores_scale(1) = 1.f;
        if constexpr (Is_first) {
            row_sum(0) = row_sum(1) = 0.f;
        } else {
            {
#ifdef AT_SUB
                scores_scale(0) = exp2f((__low2float(bn) - __low2float(att_as_h2(rbh))) * 0x1p21f);
                scores_scale(1) = exp2f((__high2float(bn) - __high2float(att_as_h2(rbh))) * 0x1p21f);
#else
                scores_scale(0) = exp2f((__low2float(bn) - __low2float(att_as_h2(rbh))) * 0.125f);
                scores_scale(1) = exp2f((__high2float(bn) - __high2float(att_as_h2(rbh))) * 0.125f);
#endif
            }
            row_sum(0) *= scores_scale(0);
            row_sum(1) *= scores_scale(1);
        }
        rmh = att_as_u(cur);
        rbh = bnu;
        b2[0] = __low2half2(bn);
        b2[1] = __high2half2(bn);
        __half2 const a2 = __float2half2_rn(a32);
        __half2 const lo2 = __float2half2_rn(1024.f);
        #pragma unroll
        for (int w = 0; w < NW; ++w) {
#if defined(AT_SUB)
            {
                uint32_t r;
                asm("fma.rn.relu.f16x2 %0, %1, %2, %3;" : "=r"(r) : "r"(h[w]), "r"(0x00080008u), "r"(att_as_u(b2[w & 1])));
                h[w] = r;
            }
#else
            h[w] = att_as_u(__hmax2(__hfma2(att_as_h2(h[w]), a2, b2[w & 1]), lo2));
#endif
        }
        return scores_scale;
    };
    template<int NW, typename TensorP>
    __forceinline__ __device__ void pack_p(uint32_t const (&h)[NW], TensorP &tOrP) {
        Tensor a = recast<uint32_t>(tOrP);
        static_assert(CUTE_STATIC_V(size(a)) * 2 == NW);
        #pragma unroll
        for (int i = 0; i < NW / 4; ++i) {
            uint32_t p0, p1;
            asm volatile("prmt.b32 %0, %1, %2, 0x6420;" : "=r"(p0) : "r"(h[4 * i]), "r"(h[4 * i + 2]));
            asm volatile("prmt.b32 %0, %1, %2, 0x6420;" : "=r"(p1) : "r"(h[4 * i + 1]), "r"(h[4 * i + 3]));
            a(2 * i) = p0;
            a(2 * i + 1) = p1;
        }
    }
    __forceinline__ __device__ void clear_sum() { osum[0] = osum[1] = osum[2] = osum[3] = 0.f; }
    __forceinline__ __device__ void add_sum(TensorT const &scores_scale) {
        row_sum(0) += osum[0] * scores_scale(0);
        row_sum(1) += osum[2] * scores_scale(1);
    }
    // No row sum in this build: returns at once.
    __forceinline__ __device__ void scale_sum(TensorT const &scores_scale) {
        return;
        bool const fl = ((++fl_cnt) & (FK_SUMFL - 1)) == 0;
        float const a0 = osum[0] * scores_scale(0), a2 = osum[2] * scores_scale(1);
        row_sum(0) += fl ? a0 : 0.f;
        row_sum(1) += fl ? a2 : 0.f;
        osum[0] = fl ? 0.f : a0; osum[1] = fl ? 0.f : osum[1] * scores_scale(0);
        osum[2] = fl ? 0.f : a2; osum[3] = fl ? 0.f : osum[3] * scores_scale(1);
    }
    int fl_cnt = 0;
    template<typename TensorP>
    __forceinline__ __device__ void issue_sum(TensorP const &tOrP, uint64_t desc) {
        Tensor a = recast<uint32_t const>(tOrP);
        constexpr int NK = CUTE_STATIC_V(size(a)) / 4;
        asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory");
        #pragma unroll
        for (int k = 0; k < NK; ++k) {
            if (k == 0) {
                asm volatile(
                    "{\n.reg .pred p;\nsetp.ne.b32 p, 0, 0;\n"
                    "wgmma.mma_async.sync.aligned.m64n8k32.f32.e4m3.e4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, %8, p, 1, 1;\n}\n"
                    : "+f"(osum[0]), "+f"(osum[1]), "+f"(osum[2]), "+f"(osum[3])
                    : "r"(a(4 * k)), "r"(a(4 * k + 1)), "r"(a(4 * k + 2)), "r"(a(4 * k + 3)), "l"(desc));
            } else {
                asm volatile(
                    "{\n.reg .pred p;\nsetp.ne.b32 p, 1, 0;\n"
                    "wgmma.mma_async.sync.aligned.m64n8k32.f32.e4m3.e4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, %8, p, 1, 1;\n}\n"
                    : "+f"(osum[0]), "+f"(osum[1]), "+f"(osum[2]), "+f"(osum[3])
                    : "r"(a(4 * k)), "r"(a(4 * k + 1)), "r"(a(4 * k + 2)), "r"(a(4 * k + 3)), "l"(desc));
            }
        }
    }
    // Returns the factor for O (final_scale, or 0 for a row that saw no key) and leaves in row_sum the log of the
    // factor that maps this O back to sum exp(scale s) v: ln 2 (-8 - (B - 1144) / 8) with B in the f16 frame of
    // 1024 + code. The split merges weight partials by it; the code after the return is the unused row-sum path.
    __forceinline__ __device__ TensorT finalize(float const final_scale=1.f) {
#ifdef AT_SUB
        row_b(0) = fmaf(__low2float(att_as_h2(rbh)), 0x1p24f, 1024.f);
        row_b(1) = fmaf(__high2float(att_as_h2(rbh)), 0x1p24f, 1024.f);
#else
        row_b(0) = __low2float(att_as_h2(rbh));
        row_b(1) = __high2float(att_as_h2(rbh));
#endif
        {
            static_assert(kNRows == 2, "Z0_NOL: two rows");
            TensorT nol_scale;
            #pragma unroll
            for (int mi = 0; mi < 2; ++mi) {
                uint32_t const mh = mi == 0 ? (rmh & 0xFFFFu) : (rmh >> 16);
                bool const empty = mh == 0xFC00u;  // running max -inf: no visible key, O is 0
                nol_scale(mi) = empty ? 0.f : final_scale;
                row_sum(mi) = empty ? -INFINITY : float(M_LN2) * (-8.f - (row_b(mi) - 1144.f) * 0.125f);
            }
            return nol_scale;
        }
        row_sum(0) += osum[0];
        row_sum(1) += osum[2];
        TensorT scores_scale;
        #pragma unroll
        for (int mi = 0; mi < size(row_sum); ++mi) {
            float const sum = row_sum(mi);
            bool const empty = sum == 0.f || sum != sum;
            scores_scale(mi) = empty ? 0.f : final_scale / sum;
            row_sum(mi) = empty ? -INFINITY : float(M_LN2) * (__log2f(sum) - 8.f - (row_b(mi) - 1144.f) * 0.125f);
        }
        return scores_scale;
    };
    template<typename Tensor1>
    __forceinline__ __device__ void rescale_o(Tensor1 &acc_o, TensorT const &scores_scale) {
        Tensor acc_o_rowcol = make_tensor(acc_o.data(), flash::convert_layout_acc_rowcol(acc_o.layout()));
        static_assert(CUTE_STATIC_V(size<0>(acc_o_rowcol)) == kNRows);
        #pragma unroll
        for (int mi = 0; mi < size<0>(acc_o_rowcol); ++mi) {
            if (scores_scale(mi) != 1.f) {
                #pragma unroll
                for (int ni = 0; ni < size<1>(acc_o_rowcol); ++ni) { acc_o_rowcol(mi, ni) *= scores_scale(mi); }
            }
        }
    };
};
}
namespace flash {
using namespace cute;
template <class CollectiveMainloop_, class CollectiveEpilogue_, class TileScheduler_>
class FlashAttnFwdSm90 {
public:
    using CollectiveMainloop = CollectiveMainloop_;
    using CollectiveEpilogue = CollectiveEpilogue_;
    static constexpr bool Is_causal = CollectiveMainloop::Is_causal;
    static constexpr bool Is_local = CollectiveMainloop::Is_local;
    static_assert(CollectiveMainloop::Varlen == CollectiveEpilogue::Varlen);
    static constexpr bool Has_softcap = CollectiveMainloop::Has_softcap;
    static constexpr bool Varlen = CollectiveMainloop::Varlen;
    static constexpr bool Split = CollectiveMainloop::Split;
    static constexpr bool Is_FP8 = CollectiveMainloop::Is_FP8;
    static constexpr bool Transpose_V = CollectiveMainloop::Transpose_V;
    static constexpr bool AppendKV = CollectiveMainloop::AppendKV;
    static constexpr bool HasQv = CollectiveMainloop::HasQv;
    static constexpr bool OnlyQv = CollectiveMainloop::OnlyQv;
    static constexpr bool Use_TMA_Q = CollectiveMainloop::Use_TMA_Q;
    static constexpr bool Use_TMA_KV = CollectiveMainloop::Use_TMA_KV;
    static constexpr bool Use_TMA_O = CollectiveEpilogue::Use_TMA_O;
    static constexpr bool PackGQA = CollectiveMainloop::PackGQA;
    static constexpr int NumProducerThreads = CollectiveMainloop::NumProducerThreads;
    static constexpr bool SameHeadDim = CollectiveMainloop::SameHeadDim;
    static constexpr bool LargeHeadDimV = CollectiveMainloop::LargeHeadDimV;
    static constexpr bool Has_sparse_mask = CollectiveMainloop::Has_sparse_mask;
    static constexpr bool SK = flash::is_sk2_sched<TileScheduler_>::value;
    static_assert(CollectiveMainloop::LargeHeadDimV == CollectiveEpilogue::LargeHeadDimV);
    using SeqlenInfo_t = typename CollectiveMainloop::SeqlenInfo_t;
    using TiledMmaQK = typename CollectiveMainloop::TiledMmaQK;
    using TileShape_MNK = typename CollectiveMainloop::TileShape_MNK;
    using TileShape_MNK_PV = typename CollectiveMainloop::TileShape_MNK_PV;
    using TiledMmaPV = typename CollectiveMainloop::TiledMmaPV;
    using ArchTag = typename CollectiveMainloop::ArchTag;
    using ClusterShape = typename CollectiveMainloop::ClusterShape;
    using MainloopArguments = typename CollectiveMainloop::Arguments;
    using MainloopParams = typename CollectiveMainloop::Params;
    using BarrierQ = std::conditional_t<Use_TMA_Q, cutlass::arch::ClusterTransactionBarrier, cutlass::arch::ClusterBarrier>;
    using SmemLayoutSink = typename CollectiveMainloop::SmemLayoutSink;
    using EpilogueArguments = typename CollectiveEpilogue::Arguments;
    using EpilogueParams = typename CollectiveEpilogue::Params;
    static_assert(ArchTag::kMinComputeCapability >= 90);
    using TileScheduler = TileScheduler_;
    using TileSchedulerArguments = typename flash::TileSchedulerArguments;
    using TileSchedulerParams = typename TileScheduler::Params;
    static constexpr uint32_t NumLoadWarpGroups = 1;
    static constexpr uint32_t NumMmaWarpGroups = CUTE_STATIC_V(size(TiledMmaPV{})) / cutlass::NumThreadsPerWarpGroup;
    static constexpr uint32_t MaxThreadsPerBlock = CUTE_STATIC_V(size(TiledMmaPV{})) + (NumLoadWarpGroups * cutlass::NumThreadsPerWarpGroup);
    static constexpr uint32_t MinBlocksPerMultiprocessor = 1;
    static_assert(NumMmaWarpGroups == 1 || NumMmaWarpGroups == 2 || NumMmaWarpGroups == 3 || NumMmaWarpGroups == 4);
#define ATT_REG4_LOAD 32
#define ATT_REG4_MMA 120
    static constexpr uint32_t LoadRegisterRequirement = (NumMmaWarpGroups == 1 ? 56 : (NumMmaWarpGroups == 2 ? ((Use_TMA_KV) ? 24 : 40) : (NumMmaWarpGroups == 3 ? 32 : ATT_REG4_LOAD)));
    static constexpr uint32_t MmaRegisterRequirement = (NumMmaWarpGroups == 1 ? 256 : (NumMmaWarpGroups == 2 ? ((Use_TMA_KV) ? 240 : 232) : (NumMmaWarpGroups == 3 ? 160 : ATT_REG4_MMA)));
    static constexpr int mainloop_smem_padding_ = int(sizeof(typename CollectiveEpilogue::TensorStorage)) - int(sizeof(decltype((typename CollectiveMainloop::TensorStorage{}).smem_v)));
    static constexpr int mainloop_smem_padding = mainloop_smem_padding_ < 0 ? 0 : mainloop_smem_padding_;
    struct SharedStorage {
        struct TensorStorage : cute::aligned_struct<128, _1> {
            union {
                struct {
                    cute::array<uint32_t, mainloop_smem_padding / sizeof(uint32_t)> padding_;
                    typename CollectiveMainloop::TensorStorage mainloop;
                };
                typename CollectiveEpilogue::TensorStorage epilogue;
            };
        } tensors;
        struct PipelineStorage : cute::aligned_struct<16, _1> {
            alignas(16) BarrierQ barrier_Q;
            alignas(16) BarrierQ barrier_Qv;
            alignas(16) cutlass::arch::ClusterBarrier barrier_O;
            alignas(16) typename CollectiveMainloop::MainloopPipelineK::SharedStorage pipeline_k;
            alignas(16) typename CollectiveMainloop::MainloopPipelineV::SharedStorage pipeline_v;
            alignas(16) typename CollectiveMainloop::MainloopPipelineVt::SharedStorage pipeline_vt;
            alignas(16) typename CollectiveMainloop::MainloopPipelineKVNew::SharedStorage pipeline_k_new;
            alignas(16) typename CollectiveMainloop::MainloopPipelineKVNew::SharedStorage pipeline_v_new;
            alignas(16) typename TileScheduler::SharedStorage smem_scheduler;
            alignas(16) typename CollectiveMainloop::MainloopPipelineMask::SharedStorage pipeline_mask;
        } pipelines;
    };
    static constexpr int SharedStorageSize = sizeof(SharedStorage);
    struct Arguments {
        MainloopArguments mainloop{};
        EpilogueArguments epilogue{};
        cutlass::KernelHardwareInfo hw_info{};
        TileSchedulerArguments scheduler{};
    };
    struct Params {
        MainloopParams mainloop{};
        EpilogueParams epilogue{};
        cutlass::KernelHardwareInfo hw_info{};
        TileSchedulerParams scheduler{};
    };
    static
    Params
    to_underlying_arguments(Arguments const& args) {
        CUTLASS_TRACE_HOST("to_underlying_arguments():");
        int sm_count = args.hw_info.sm_count;
        if (sm_count <= 0) {
            CUTLASS_TRACE_HOST("  WARNING: Arguments do not include a valid SM count.\n"
                "  For optimal performance, populate the arguments KernelHardwareInfo struct with the SM count.");
            sm_count = cutlass::KernelHardwareInfo::query_device_multiprocessor_count(args.hw_info.device_id);
        }
        CUTLASS_TRACE_HOST("to_underlying_arguments(): Setting persistent grid SM count to " << sm_count);
        cutlass::KernelHardwareInfo hw_info{args.hw_info.device_id, sm_count};
        return {
            CollectiveMainloop::to_underlying_arguments(args.mainloop),
            CollectiveEpilogue::to_underlying_arguments(args.epilogue),
            hw_info,
            TileScheduler::to_underlying_arguments(args.scheduler)
        };
    }
    static dim3
    get_grid_shape(Params const& params) {
        return TileScheduler::get_grid_shape(params.scheduler, params.hw_info.sm_count);
    }
    static dim3
    get_block_shape() {
        return dim3(MaxThreadsPerBlock, 1, 1);
    }
    CUTLASS_DEVICE
    void
    operator()(Params const& params, char* smem_buf) {
        static constexpr int NumMmaThreads = NumMmaWarpGroups * cutlass::NumThreadsPerWarpGroup;
        static constexpr int MmaThreadOffset = NumLoadWarpGroups * cutlass::NumThreadsPerWarpGroup;
        static constexpr int kBlockM = get<0>(TileShape_MNK_PV{});
        using MainloopPipelineK = typename CollectiveMainloop::MainloopPipelineK;
        using MainloopPipelineV = typename CollectiveMainloop::MainloopPipelineV;
        using MainloopPipelineVt = typename CollectiveMainloop::MainloopPipelineVt;
        using MainloopPipelineKVNew = typename CollectiveMainloop::MainloopPipelineKVNew;
        using MainloopPipelineMask = typename CollectiveMainloop::MainloopPipelineMask;
        using PipelineState = typename CollectiveMainloop::PipelineState;
        using PipelineParamsK = typename MainloopPipelineK::Params;
        using PipelineParamsV = typename MainloopPipelineV::Params;
        using PipelineParamsVt = typename MainloopPipelineVt::Params;
        using PipelineParamsKVNew = typename MainloopPipelineKVNew::Params;
        using PipelineParamsMask = typename MainloopPipelineMask::Params;
        SharedStorage& shared_storage = *reinterpret_cast<SharedStorage*>(smem_buf);
        int const lane_predicate = cute::elect_one_sync();
        int const warp_idx = cutlass::canonical_warp_idx_sync();
        if (warp_idx == 0 && lane_predicate) {
            CollectiveMainloop::prefetch_tma_descriptors(params.mainloop);
            CollectiveEpilogue::prefetch_tma_descriptors(params.epilogue);
        }
        int const warp_group_thread_idx = threadIdx.x % cutlass::NumThreadsPerWarpGroup;
        int warp_group_idx = cutlass::canonical_warp_group_idx();
        if (warp_idx == 0 && lane_predicate) {
            shared_storage.pipelines.barrier_Q.init(Use_TMA_Q ? 1 : NumProducerThreads );
            if constexpr (HasQv) {
                shared_storage.pipelines.barrier_Qv.init(Use_TMA_Q ? 1 : NumProducerThreads );
            }
            shared_storage.pipelines.barrier_O.init(size(ClusterShape{}) * (Use_TMA_O ? 1 : NumMmaThreads) );
        }
        PipelineParamsK pipeline_params_k;
        pipeline_params_k.role = warp_group_idx == 0
            ? MainloopPipelineK::ThreadCategory::Producer
            : MainloopPipelineK::ThreadCategory::Consumer;
        if constexpr (Use_TMA_KV) {
            pipeline_params_k.transaction_bytes = CollectiveMainloop::TmaTransactionBytesK;
            pipeline_params_k.is_leader = warp_group_thread_idx == 0;
            pipeline_params_k.num_consumers = !LargeHeadDimV ? NumMmaThreads : cutlass::NumThreadsPerWarpGroup;
        } else {
            pipeline_params_k.consumer_arv_count = !LargeHeadDimV ? NumMmaThreads : cutlass::NumThreadsPerWarpGroup;
            pipeline_params_k.producer_arv_count = NumProducerThreads;
        }
        PipelineParamsMask pipeline_params_mask;
        pipeline_params_mask.role = warp_group_idx == 0
            ? MainloopPipelineMask::ThreadCategory::Producer
            : MainloopPipelineMask::ThreadCategory::Consumer;
        if constexpr (CollectiveMainloop::Use_TMA_Mask) {
            pipeline_params_mask.transaction_bytes = CollectiveMainloop::TmaTransactionBytesMask;
            pipeline_params_mask.is_leader = warp_group_thread_idx == 0;
            pipeline_params_mask.num_consumers = !LargeHeadDimV ? NumMmaThreads : cutlass::NumThreadsPerWarpGroup;
        } else {
            pipeline_params_mask.consumer_arv_count = !LargeHeadDimV ? NumMmaThreads : cutlass::NumThreadsPerWarpGroup;
            pipeline_params_mask.producer_arv_count = NumProducerThreads;
        }
        static_assert(is_same_v<PipelineParamsK, PipelineParamsVt>);
        PipelineParamsVt pipeline_params_vt = pipeline_params_k;
        if constexpr (Use_TMA_KV) {
            pipeline_params_vt.transaction_bytes = CollectiveMainloop::TmaTransactionBytesV;
            if constexpr (LargeHeadDimV) { pipeline_params_vt.num_consumers = NumMmaThreads; }
        } else {
            if constexpr (LargeHeadDimV) { pipeline_params_vt.consumer_arv_count = NumMmaThreads; }
        }
        MainloopPipelineK pipeline_k = [&] {
            if constexpr (Use_TMA_KV) {
                return MainloopPipelineK(shared_storage.pipelines.pipeline_k, pipeline_params_k, ClusterShape{});
            } else {
                return MainloopPipelineK(shared_storage.pipelines.pipeline_k, pipeline_params_k);
            }
        }();
        MainloopPipelineMask pipeline_mask = [&] {
            if constexpr (CollectiveMainloop::Use_TMA_Mask) {
                return MainloopPipelineMask(shared_storage.pipelines.pipeline_mask, pipeline_params_mask, ClusterShape{});
            } else {
                return MainloopPipelineMask(shared_storage.pipelines.pipeline_mask, pipeline_params_mask);
            }
        }();
        MainloopPipelineV pipeline_v = [&] {
            if constexpr (!Transpose_V) {
                static_assert(is_same_v<PipelineParamsK, PipelineParamsV>);
                if constexpr (Use_TMA_KV) {
                    return MainloopPipelineV(shared_storage.pipelines.pipeline_v, pipeline_params_vt, ClusterShape{});
                } else {
                    return MainloopPipelineV(shared_storage.pipelines.pipeline_v, pipeline_params_vt);
                }
            } else {
                PipelineParamsV pipeline_params_v;
                pipeline_params_v.role = warp_group_idx == 0
                    ? MainloopPipelineV::ThreadCategory::Producer
                    : MainloopPipelineV::ThreadCategory::Consumer;
                pipeline_params_v.producer_arv_count = NumProducerThreads;
                pipeline_params_v.consumer_arv_count = NumMmaThreads;
                return MainloopPipelineV(shared_storage.pipelines.pipeline_v, pipeline_params_v);
            }
        }();
        MainloopPipelineVt pipeline_vt = [&] {
            if constexpr (Use_TMA_KV) {
                pipeline_params_vt.num_consumers = NumProducerThreads;
                return MainloopPipelineVt(shared_storage.pipelines.pipeline_vt, pipeline_params_vt, ClusterShape{});
            } else {
                pipeline_params_vt.consumer_arv_count = NumProducerThreads;
                return MainloopPipelineVt(shared_storage.pipelines.pipeline_vt, pipeline_params_vt);
            }
        }();
        PipelineParamsKVNew pipeline_params_kv_new;
        pipeline_params_kv_new.role = warp_group_idx == 0
            ? MainloopPipelineKVNew::ThreadCategory::Producer
            : MainloopPipelineKVNew::ThreadCategory::Consumer;
        pipeline_params_kv_new.transaction_bytes = CollectiveMainloop::TmaTransactionBytesK;
        pipeline_params_kv_new.is_leader = warp_group_thread_idx == 0;
        pipeline_params_kv_new.num_consumers = NumMmaThreads;
        auto pipeline_k_new = cute::conditional_return<AppendKV>(MainloopPipelineKVNew(shared_storage.pipelines.pipeline_k_new, pipeline_params_kv_new, ClusterShape{}), nullptr);
        if constexpr (!SameHeadDim) {
            pipeline_params_kv_new.transaction_bytes = CollectiveMainloop::TmaTransactionBytesV;
        }
        auto pipeline_v_new = cute::conditional_return<AppendKV>(MainloopPipelineKVNew(shared_storage.pipelines.pipeline_v_new, pipeline_params_kv_new, ClusterShape{}), nullptr);
        CollectiveMainloop mainloop;
        CollectiveEpilogue epilogue;
        const bool has_sink = params.mainloop.ptr_sink != nullptr;
        const int num_heads = get<2>(params.mainloop.shape_Q);
        Tensor gSink = make_tensor(make_gmem_ptr(params.mainloop.ptr_sink), make_shape(num_heads));
        Tensor sSink = make_tensor(make_smem_ptr(shared_storage.tensors.mainloop.smem_sink.data()), SmemLayoutSink{});
        if (has_sink) {
            #pragma unroll
            for (int i = threadIdx.x; i < num_heads; i += blockDim.x) {
                sSink(i) = gSink(i);
            }
        }
        {
            uint32_t* ones = reinterpret_cast<uint32_t*>(shared_storage.tensors.mainloop.smem_ones.data());
            for (int i = threadIdx.x; i < 128; i += blockDim.x) { ones[i] = 0x38383838u; }
            cutlass::arch::fence_view_async_shared();
        }
        if constexpr (size(ClusterShape{}) > 1) {
            cute::cluster_arrive_relaxed();
            cute::cluster_wait();
        } else {
            __syncthreads();
        }
        TileScheduler scheduler(reinterpret_cast<typename TileScheduler::SharedStorage*>(&shared_storage.pipelines.smem_scheduler));
        if (warp_group_idx == 0) {
            cutlass::arch::warpgroup_reg_dealloc<LoadRegisterRequirement>();
            PipelineState smem_pipe_write = cutlass::make_producer_start_state<MainloopPipelineK>();
            PipelineState smem_pipe_write_new = cutlass::make_producer_start_state<MainloopPipelineKVNew>();
            int work_idx = 0;
            int warp_idx_in_warpgroup = __shfl_sync(0xffffffff, (threadIdx.x / 32) % 4, 0);
            static constexpr bool SingleProducerWarp = NumProducerThreads == cutlass::NumThreadsPerWarp;
            if constexpr (SingleProducerWarp) {
                if (warp_idx_in_warpgroup != 0) { return; }
            }
            if (!SingleProducerWarp && warp_idx_in_warpgroup != 0) { scheduler.init_consumer(); }
            cutlass::arch::wait_on_dependent_grids();
            for (auto work_tile_info = SingleProducerWarp || warp_idx_in_warpgroup == 0 ? scheduler.template get_initial_work<true>(params.scheduler) : scheduler.template get_initial_work<false>(params.scheduler);
                 work_tile_info.is_valid(params.scheduler);
                 work_tile_info = SingleProducerWarp || warp_idx_in_warpgroup == 0 ? scheduler.template get_next_work<true>(params.scheduler, work_tile_info) : scheduler.template get_next_work<false>(params.scheduler, work_tile_info)) {
                auto block_coord = work_tile_info.get_block_coord(params.scheduler);
                SeqlenInfo_t seqlen_info{
                    get<2>(block_coord) ,
                    get<0>(params.mainloop.shape_Q),
                    !params.mainloop.ptr_pagetable ? size<0>(params.mainloop.shape_K) : size<0>(params.mainloop.shape_K) * size<1>(params.mainloop.shape_pagetable),
                    get<0>(params.mainloop.shape_K_new),
                    params.mainloop.cu_seqlens_q, params.mainloop.cu_seqlens_k, params.mainloop.cu_seqlens_k_new,
                    params.mainloop.seqused_q, params.mainloop.seqused_k, params.mainloop.leftpad_k,
                    params.mainloop.seqlens_rotary
                };
                if constexpr (AppendKV) {
                    bool tile_new_valid = mainloop.load_kv_new(
                        params.mainloop, pipeline_k_new, pipeline_v_new,
                        smem_pipe_write_new, shared_storage, seqlen_info, block_coord, work_idx);
                    if (tile_new_valid) {
                        cutlass::arch::NamedBarrier::sync(NumMmaThreads + NumProducerThreads, static_cast<uint32_t>(FwdNamedBarriers::AppendKV) );
                    }
                }
                auto scheduler_prefetch = [&scheduler, &params, &work_tile_info]() {
                    scheduler.prefetch_next_work(params.scheduler, work_tile_info);
                };
                mainloop.load(params.mainloop, pipeline_k, pipeline_v, pipeline_vt, pipeline_mask, smem_pipe_write,
                                         shared_storage, scheduler_prefetch, seqlen_info, block_coord, work_idx);
            }
            mainloop.load_tail(pipeline_k, pipeline_v, pipeline_vt, pipeline_mask, smem_pipe_write, shared_storage, work_idx);
        } else {
            cutlass::arch::warpgroup_reg_alloc<MmaRegisterRequirement>();
            TiledMmaPV tiled_mma_pv;
            PipelineState smem_pipe_read;
            PipelineState smem_pipe_read_new;
            scheduler.init_consumer();
            mainloop.mma_init();
            int work_idx = 0;
            CUTLASS_PRAGMA_NO_UNROLL
            for (auto work_tile_info = scheduler.template get_initial_work<false>(params.scheduler);
                 work_tile_info.is_valid(params.scheduler);
                 ) {
                auto block_coord = work_tile_info.get_block_coord(params.scheduler);
                int const bidb = get<2>(block_coord);
                int sk_t = 0, sk_np = 1, sk_slot = 0;
                if constexpr (SK) { sk_t = work_tile_info.t; sk_np = work_tile_info.np; sk_slot = work_tile_info.slot; }
                SeqlenInfo_t seqlen_info{
                    bidb,
                    get<0>(params.mainloop.shape_Q),
                    !params.mainloop.ptr_pagetable ? size<0>(params.mainloop.shape_K) : size<0>(params.mainloop.shape_K) * size<1>(params.mainloop.shape_pagetable),
                    get<0>(params.mainloop.shape_K_new),
                    params.mainloop.cu_seqlens_q, params.mainloop.cu_seqlens_k, params.mainloop.cu_seqlens_k_new,
                    params.mainloop.seqused_q, params.mainloop.seqused_k, params.mainloop.leftpad_k,
                    params.mainloop.seqlens_rotary
                };
                if constexpr (AppendKV) {
                    bool tile_new_valid = mainloop.store_kv_new(
                        params.mainloop, pipeline_k_new, pipeline_v_new, smem_pipe_read_new,
                        threadIdx.x - MmaThreadOffset, shared_storage, seqlen_info, block_coord);
                    if (tile_new_valid) {
                        asm volatile ("fence.proxy.async.global;");
                        cutlass::arch::NamedBarrier::arrive(NumMmaThreads + NumProducerThreads, static_cast<uint32_t>(FwdNamedBarriers::AppendKV) );
                    }
                }
                float softmax_scale_log2 = params.mainloop.softmax_scale_log2;
                if constexpr (Is_FP8 && !Has_softcap) {
                    int const bidh = get<1>(block_coord);
                    int const bidh_kv = !PackGQA ? params.mainloop.qhead_per_khead_divmod.divide(bidh) : bidh;
                    float const q_descale = params.mainloop.ptr_q_descale == nullptr ? 1.0f : params.mainloop.ptr_q_descale[bidb * get<0>(params.mainloop.stride_q_descale) + bidh_kv * get<1>(params.mainloop.stride_q_descale)];
                    float const k_descale = params.mainloop.ptr_k_descale == nullptr ? 1.0f : params.mainloop.ptr_k_descale[bidb * get<0>(params.mainloop.stride_k_descale) + bidh_kv * get<1>(params.mainloop.stride_k_descale)];
                    softmax_scale_log2 *= q_descale * k_descale;
                }
                flash::SoftmaxL<!LargeHeadDimV ? 2 * (2 * kBlockM / NumMmaThreads) : 2, !Is_FP8 ? 0 : 8> softmax(softmax_scale_log2);
                if (has_sink) {
                    int const split_idx = get<3>(block_coord);
                    if (!Split || (split_idx & 0x0000FFFF) == 0) {
                        int const bidh = get<1>(block_coord);
                        if (!PackGQA) {
                            softmax.template set_row_sink<PackGQA>(sSink, bidh);
                        } {
                            int const m_block = get<0>(block_coord);
                            int const qhead_per_khead = params.mainloop.qhead_per_khead_divmod.divisor;
                            TiledMmaQK tiled_mma_qk;
                            auto thread_mma_qk = tiled_mma_qk.get_thread_slice(threadIdx.x - MmaThreadOffset);
                            Tensor coord_S = cute::make_identity_tensor(select<0, 1>(TileShape_MNK{}));
                            Tensor tCoord_S = thread_mma_qk.partition_C(coord_S);
                            auto tCoord_rowcol = make_tensor(
                                tCoord_S.data(),
                                flash::convert_layout_acc_rowcol<false>(tCoord_S.layout())
                            );
                            const int row_start = m_block * kBlockM + get<0>(tCoord_rowcol(_0{}, _0{}));
                            const int row_step = get<0>(tCoord_rowcol(_1{}, _0{})) - get<0>(tCoord_rowcol(_0{}, _0{}));
                            softmax.template set_row_sink<PackGQA>(sSink, bidh, qhead_per_khead, row_start, row_step);
                        }
                    }
                }
                Tensor tOrO = partition_fragment_C(tiled_mma_pv, select<0, 1>(TileShape_MNK_PV{}));
                bool tile_valid;
                if constexpr (!LargeHeadDimV) {
                    tile_valid = mainloop.mma(
                        params.mainloop, pipeline_k, pipeline_v, pipeline_mask, smem_pipe_read,
                        tOrO, softmax, threadIdx.x - MmaThreadOffset, work_idx, seqlen_info, block_coord, shared_storage);
                } else {
                    if (warp_group_idx == 1) {
                        tile_valid = mainloop.mma(
                            params.mainloop, pipeline_k, pipeline_v, pipeline_mask, smem_pipe_read,
                            tOrO, softmax, threadIdx.x - MmaThreadOffset, work_idx, seqlen_info, block_coord, shared_storage);
                    } else {
                        tile_valid = mainloop.mma_pv(
                            params.mainloop, pipeline_v, smem_pipe_read,
                            tOrO, softmax, threadIdx.x - MmaThreadOffset, seqlen_info, block_coord, shared_storage);
                    }
                }
                work_tile_info = scheduler.template get_next_work<false>(params.scheduler, work_tile_info);
                if constexpr (Split && Varlen) {
                    if (!work_tile_info.is_valid(params.scheduler)) {
                        cutlass::arch::launch_dependent_grids();
                    }
                }
                if constexpr (SK) {
                    cute::tuple<int32_t, int32_t, int32_t, int32_t> const coord_f{get<0>(block_coord), get<1>(block_coord), get<2>(block_coord), int32_t(1 << 16)};
                    if (!tile_valid) {
                        cute::clear(tOrO);
                        #pragma unroll
                        for (int mi = 0; mi < size(softmax.row_sum); ++mi) { softmax.row_sum(mi) = -INFINITY; }
                    }
                    // Equal-work split: a tile whose key range sits in one CTA stores directly. Otherwise every
                    // part writes its FP32 O and log factor to the partial buffer, and the last part to finish
                    // (atomic count) merges the others in, weighting each by exp(log factor - max), and stores.
                    if (sk_np <= 1) {
                        epilogue.store(params.epilogue, tOrO, softmax.row_sum, shared_storage, tiled_mma_pv,
                                       threadIdx.x - MmaThreadOffset, coord_f);
                    } else {
                        int* const sk2_flag = reinterpret_cast<int*>(&shared_storage.pipelines.smem_scheduler) + 8;
                        static constexpr int FR = decltype(size(tOrO))::value;
                        static constexpr int NR = decltype(size(softmax.row_sum))::value;
                        static_assert(FR % 4 == 0);
                        int const tid = threadIdx.x - MmaThreadOffset;
                        int const G = int(gridDim.x);
                        int const t0 = params.scheduler.tp[sk_t], t1 = params.scheduler.tp[sk_t + 1];
                        float* const part = reinterpret_cast<float*>(params.epilogue.ptr_O_partial);
                        float* const lsep = params.epilogue.ptr_LSE_partial;
                        {
                            size_t const o = (size_t(blockIdx.x) * 2 + sk_slot) * NumMmaThreads + tid;
                            float4* dst = reinterpret_cast<float4*>(part + o * FR);
                            #pragma unroll
                            for (int i = 0; i < FR / 4; ++i) { dst[i] = make_float4(tOrO(4 * i), tOrO(4 * i + 1), tOrO(4 * i + 2), tOrO(4 * i + 3)); }
                            #pragma unroll
                            for (int r = 0; r < NR; ++r) { lsep[o * NR + r] = softmax.row_sum(r); }
                        }
                        __threadfence();
                        flash::named_barrier_sync(NumMmaThreads, cutlass::arch::ReservedNamedBarriers::EpilogueBarrier);
                        if (tid == 0) { *sk2_flag = atomicAdd(params.scheduler.cnt + sk_t, 1) == sk_np - 1; }
                        flash::named_barrier_sync(NumMmaThreads, cutlass::arch::ReservedNamedBarriers::EpilogueBarrier);
                        bool const last = *sk2_flag;
                        if (last) {
                            int const x52z = *sk2_flag - 1;
                            long long const W = params.scheduler.st[G + x52z];
                            __threadfence();
                            int const cf = int(((long long)(t0 + params.scheduler.st[G + 1 + x52z] + 1) * G + W - 1) / W) - 1;
                            int const cl = int(((long long)t1 * G + W - 1) / W) - 1;
                            Tensor acc = make_tensor(tOrO.data(), flash::convert_layout_acc_rowcol(tOrO.layout()));
                            for (int c2 = cf; c2 <= cl; ++c2) {
                                if (c2 == int(blockIdx.x)) { continue; }
                                int const sl = (long long)c2 * W / G >= t0 ? 0 : 1;
                                size_t const o = (size_t(c2) * 2 + sl) * NumMmaThreads + tid;
                                Tensor tOrO2 = make_fragment_like(tOrO);
                                float4 const* src = reinterpret_cast<float4 const*>(part + o * FR);
                                #pragma unroll
                                for (int i = 0; i < FR / 4; ++i) {
                                    float4 const v = __ldcg(src + i);
                                    tOrO2(4 * i) = v.x; tOrO2(4 * i + 1) = v.y; tOrO2(4 * i + 2) = v.z; tOrO2(4 * i + 3) = v.w;
                                }
                                Tensor acc2 = make_tensor(tOrO2.data(), flash::convert_layout_acc_rowcol(tOrO2.layout()));
                                #pragma unroll
                                for (int mi = 0; mi < NR; ++mi) {
                                    float const la = softmax.row_sum(mi), lb = __ldcg(lsep + o * NR + mi);
                                    float const mx = fmaxf(la, lb);
                                    if (mx == -INFINITY) { continue; }
                                    float const wa = expf(la - mx), wb = expf(lb - mx);
                                    float const inv = 1.f / (wa + wb);
                                    float const ia = wa * inv, ib = wb * inv;
                                    #pragma unroll
                                    for (int ni = 0; ni < size<1>(acc); ++ni) { acc(mi, ni) = acc(mi, ni) * ia + acc2(mi, ni) * ib; }
                                    softmax.row_sum(mi) = mx + logf(wa + wb);
                                }
                            }
                            epilogue.store(params.epilogue, tOrO, softmax.row_sum, shared_storage, tiled_mma_pv,
                                           threadIdx.x - MmaThreadOffset, coord_f);
                        } else {
                            epilogue.store_skip(shared_storage);
                        }
                    }
                } else
                {
                if (tile_valid) {
                    epilogue.store(params.epilogue, tOrO, softmax.row_sum, shared_storage, tiled_mma_pv,
                                   threadIdx.x - MmaThreadOffset, block_coord);
                } else {
                    epilogue.store_zero(params.epilogue, threadIdx.x - MmaThreadOffset, block_coord);
                }
                }
            }
            epilogue.store_tail();
        }
    }
};
}
#include "cute/tensor.hpp"
#include <cutlass/cutlass.h>
#include <cutlass/array.h>
#include <cutlass/numeric_types.h>
#include <cutlass/kernel_hardware_info.h>
namespace flash {
using namespace cute;
template <class CollectiveMainloop_, class CollectiveEpilogue_, class TileScheduler_>
class FlashAttnFwdSm80 {
public:
    using CollectiveMainloop = CollectiveMainloop_;
    using CollectiveEpilogue = CollectiveEpilogue_;
    static constexpr bool Is_causal = CollectiveMainloop::Is_causal;
    static constexpr bool Is_local = CollectiveMainloop::Is_local;
    static_assert(CollectiveMainloop::Varlen == CollectiveEpilogue::Varlen);
    static constexpr bool Has_softcap = CollectiveMainloop::Has_softcap;
    static constexpr bool Varlen = CollectiveMainloop::Varlen;
    static constexpr bool PagedKV = CollectiveMainloop::PagedKV;
    static constexpr bool Split = CollectiveMainloop::Split;
    static constexpr bool Is_FP8 = CollectiveMainloop::Is_FP8;
    static constexpr bool Transpose_V = CollectiveMainloop::Transpose_V;
    static constexpr bool AppendKV = CollectiveMainloop::AppendKV;
    static constexpr bool PackGQA = CollectiveMainloop::PackGQA;
    static constexpr int NumProducerThreads = CollectiveMainloop::NumProducerThreads;
    using SeqlenInfo_t = typename CollectiveMainloop::SeqlenInfo_t;
    using TileShape_MNK = typename CollectiveMainloop::TileShape_MNK;
    using TiledMma = typename CollectiveMainloop::TiledMma;
    using ArchTag = typename CollectiveMainloop::ArchTag;
    using MainloopArguments = typename CollectiveMainloop::Arguments;
    using MainloopParams = typename CollectiveMainloop::Params;
    using EpilogueArguments = typename CollectiveEpilogue::Arguments;
    using EpilogueParams = typename CollectiveEpilogue::Params;
    static_assert(ArchTag::kMinComputeCapability >= 80);
    using TileScheduler = TileScheduler_;
    using TileSchedulerArguments = typename flash::TileSchedulerArguments;
    using TileSchedulerParams = typename TileScheduler::Params;
    static constexpr uint32_t NumThreads = CUTE_STATIC_V(size(TiledMma{}));
    static constexpr uint32_t MaxThreadsPerBlock = CUTE_STATIC_V(size(TiledMma{}));
    static constexpr uint32_t MinBlocksPerMultiprocessor = NumThreads == 128 ? 2 : 1;
    static constexpr int mainloop_smem_padding_ = int(sizeof(typename CollectiveEpilogue::TensorStorage))
        - int(sizeof(decltype((typename CollectiveMainloop::TensorStorage{}).smem_v)))
        - int(sizeof(decltype((typename CollectiveMainloop::TensorStorage{}).smem_k)));
    static constexpr int mainloop_smem_padding = mainloop_smem_padding_ < 0 ? 0 : mainloop_smem_padding_;
    struct SharedStorage {
        struct TensorStorage : cute::aligned_struct<128> {
            union {
                struct {
                    cute::array<uint32_t, mainloop_smem_padding / sizeof(uint32_t)> padding_;
                    typename CollectiveMainloop::TensorStorage mainloop;
                };
                typename CollectiveEpilogue::TensorStorage epilogue;
            };
        } tensors;
        alignas(16) typename TileScheduler::SharedStorage smem_scheduler;
    };
    static constexpr int SharedStorageSize = sizeof(SharedStorage);
    struct Arguments {
        MainloopArguments mainloop{};
        EpilogueArguments epilogue{};
        cutlass::KernelHardwareInfo hw_info{};
        TileSchedulerArguments scheduler{};
    };
    struct Params {
        MainloopParams mainloop{};
        EpilogueParams epilogue{};
        cutlass::KernelHardwareInfo hw_info{};
        TileSchedulerParams scheduler{};
    };
    static
    Params
    to_underlying_arguments(Arguments const& args) {
        CUTLASS_TRACE_HOST("to_underlying_arguments():");
        int sm_count = args.hw_info.sm_count;
        if (sm_count <= 0) {
            CUTLASS_TRACE_HOST("  WARNING: Arguments do not include a valid SM count.\n"
                "  For optimal performance, populate the arguments KernelHardwareInfo struct with the SM count.");
            sm_count = cutlass::KernelHardwareInfo::query_device_multiprocessor_count(args.hw_info.device_id);
        }
        CUTLASS_TRACE_HOST("to_underlying_arguments(): Setting persistent grid SM count to " << sm_count);
        cutlass::KernelHardwareInfo hw_info{args.hw_info.device_id, sm_count};
        return {
            CollectiveMainloop::to_underlying_arguments(args.mainloop),
            CollectiveEpilogue::to_underlying_arguments(args.epilogue),
            hw_info,
            TileScheduler::to_underlying_arguments(args.scheduler)
        };
    }
    static dim3
    get_grid_shape(Params const& params) {
        return TileScheduler::get_grid_shape(params.scheduler, params.hw_info.sm_count * MinBlocksPerMultiprocessor);
    }
    static dim3
    get_block_shape() {
        return dim3(MaxThreadsPerBlock, 1, 1);
    }
    CUTLASS_DEVICE
    void
    operator()(Params const& params, char* smem_buf) {
        static constexpr int kBlockM = get<0>(TileShape_MNK{});
        SharedStorage& shared_storage = *reinterpret_cast<SharedStorage*>(smem_buf);
        CollectiveMainloop mainloop;
        CollectiveEpilogue epilogue;
        TileScheduler scheduler(reinterpret_cast<typename TileScheduler::SharedStorage*>(&shared_storage.smem_scheduler));
        TiledMma tiled_mma;
        scheduler.init_consumer();
        int warp_idx = cutlass::canonical_warp_idx_sync();
        CUTLASS_PRAGMA_NO_UNROLL
        for (auto work_tile_info = warp_idx == 0 ? scheduler.template get_initial_work<true>(params.scheduler) : scheduler.template get_initial_work<false>(params.scheduler);
             work_tile_info.is_valid(params.scheduler);
             work_tile_info = warp_idx == 0 ? scheduler.template get_next_work<true>(params.scheduler, work_tile_info) : scheduler.template get_next_work<false>(params.scheduler, work_tile_info)) {
            Tensor tOrO = partition_fragment_C(tiled_mma, select<0, 2>(TileShape_MNK{}));
            float softmax_scale_log2 = params.mainloop.softmax_scale_log2;
            auto block_coord = work_tile_info.get_block_coord(params.scheduler);
            int const bidb = get<2>(block_coord);
            if constexpr (Is_FP8 && !Has_softcap) {
                int const bidh = get<1>(block_coord);
                int const bidh_kv = !PackGQA ? params.mainloop.qhead_per_khead_divmod.divide(bidh) : bidh;
                float const q_descale = params.mainloop.ptr_q_descale == nullptr ? 1.0f : params.mainloop.ptr_q_descale[bidb * get<0>(params.mainloop.stride_q_descale) + bidh_kv * get<1>(params.mainloop.stride_q_descale)];
                float const k_descale = params.mainloop.ptr_k_descale == nullptr ? 1.0f : params.mainloop.ptr_k_descale[bidb * get<0>(params.mainloop.stride_k_descale) + bidh_kv * get<1>(params.mainloop.stride_k_descale)];
                softmax_scale_log2 *= q_descale * k_descale;
            }
            flash::Softmax<2 * (2 * kBlockM / NumThreads), !Is_FP8 ? 0 : 8> softmax(softmax_scale_log2);
            SeqlenInfo_t seqlen_info{
                bidb,
                get<0>(params.mainloop.shape_Q),
                !PagedKV ? size<0>(params.mainloop.shape_K) : size<0>(params.mainloop.shape_K) * size<1>(params.mainloop.shape_pagetable),
                get<0>(params.mainloop.shape_K_new),
                params.mainloop.cu_seqlens_q, params.mainloop.cu_seqlens_k, params.mainloop.cu_seqlens_k_new,
                params.mainloop.seqused_q, params.mainloop.seqused_k, params.mainloop.leftpad_k,
                params.mainloop.seqlens_rotary
            };
            if constexpr (AppendKV) {
                bool tile_new_valid = mainloop.store_kv_new(
                    params.mainloop, threadIdx.x, shared_storage, seqlen_info, block_coord);
                if (tile_new_valid) { __syncthreads(); }
            }
            bool tile_valid = mainloop.mma(
                params.mainloop, tOrO, softmax, threadIdx.x, seqlen_info, block_coord,
                shared_storage);
            scheduler.prefetch_next_work(params.scheduler, work_tile_info);
            if (tile_valid) {
                epilogue.store(params.epilogue, tOrO, softmax.row_sum, shared_storage, tiled_mma,
                               threadIdx.x, block_coord);
            } else {
                epilogue.store_zero(params.epilogue, threadIdx.x, block_coord);
            }
        }
    }
};
}
#include <cutlass/cutlass.h>
#include <cutlass/array.h>
#include <cutlass/numeric_types.h>
#include <cutlass/numeric_conversion.h>
#include "cutlass/pipeline/pipeline.hpp"
#include "cute/tensor.hpp"
#include "cutlass/gemm/collective/builders/sm90_common.inl"
namespace flash {
template <class SeqlenInfo_t, int kBlockM, int kBlockN, bool Is_causal, bool Is_local, bool PackGQA=false, bool Split=false>
struct BlockMN {
    static
    CUTLASS_DEVICE
    cute::tuple<int, int, int> get_n_block_min_max(
            SeqlenInfo_t const& seqlen_info,
            int const m_block, int const bidb, int const split_idx, int const num_splits,
            int const window_size_left, int const window_size_right,
            cutlass::FastDivmod const& attention_chunk_divmod,
            cutlass::FastDivmod const& qhead_per_khead_divmod) {
        int seqlen_k = seqlen_info.seqlen_k;
        int const seqlen_q = seqlen_info.seqlen_q;
        int n_offset = 0;
        if constexpr (Is_local) {
            int m_idx_min = m_block * kBlockM;
            if (PackGQA) { m_idx_min = qhead_per_khead_divmod.divide(m_idx_min); }
            int const n_idx = m_idx_min + seqlen_k - seqlen_q;
            int n_idx_left = n_idx - window_size_left;
            if (attention_chunk_divmod.divisor > 0) {
                n_idx_left = std::max(n_idx_left, flash::round_down(attention_chunk_divmod, n_idx));
            }
            n_offset = std::max(int(0), n_idx_left);
            seqlen_k -= n_offset;
        }
        int n_block_max = cute::ceil_div(seqlen_k, kBlockN);
        if constexpr (Is_causal || Is_local) {
            int m_idx_max = (m_block + 1) * kBlockM;
            if (PackGQA) { m_idx_max = qhead_per_khead_divmod.divide(m_idx_max - 1) + 1 ; }
            n_block_max = std::min(n_block_max,
                                   cute::ceil_div(m_idx_max + seqlen_k - seqlen_q + window_size_right, kBlockN));
        }
        int n_block_min = 0;
        if (Split && (split_idx & (1 << 30))) {
            n_block_min = split_idx & 0x7FF;
            n_block_max = std::min(n_block_max, (split_idx >> 11) & 0x7FF);
            return {n_block_min, n_block_max, n_offset};
        }
        if constexpr (Split) {
            uint32_t num_splits_dynamic_u = reinterpret_cast<uint32_t const&>(split_idx) >> 16;
            int num_splits_dynamic = reinterpret_cast<int&>(num_splits_dynamic_u);
            int split_idx_actual = split_idx & 0x0000FFFF;
            int num_splits_actual = num_splits_dynamic > 0 ? num_splits_dynamic : num_splits;
            int num_n_blocks_per_split = n_block_max <= n_block_min ? 0 : cute::ceil_div(n_block_max - n_block_min, num_splits_actual);
            n_block_min = n_block_min + split_idx_actual * num_n_blocks_per_split;
            n_block_max = std::min(n_block_min + num_n_blocks_per_split, n_block_max);
        }
        return {n_block_min, n_block_max, n_offset};
    }
    static
    CUTLASS_DEVICE
    cute::tuple<int, int> get_n_block_k_new_min_max(
            SeqlenInfo_t const& seqlen_info,
            int const m_block, int const bidb, int const split_idx, int const num_splits,
            int const window_size_left, int const window_size_right,
            cutlass::FastDivmod const& attention_chunk_divmod,
            cutlass::FastDivmod const& qhead_per_khead_divmod) {
        auto [n_block_min, n_block_max, n_offset] = get_n_block_min_max(
            seqlen_info, m_block, bidb, split_idx, num_splits,
            window_size_left, window_size_right, attention_chunk_divmod, qhead_per_khead_divmod);
        int const idx_k_new_min = std::max(n_block_min * kBlockN + n_offset - seqlen_info.seqlen_k_og, 0);
        int const idx_k_new_max = std::min(n_block_max * kBlockN + n_offset - seqlen_info.seqlen_k_og, seqlen_info.seqlen_k_new);
        int const n_block_new_min = idx_k_new_min / kBlockN;
        int const n_block_new_max = idx_k_new_max > idx_k_new_min ? cute::ceil_div(idx_k_new_max, kBlockN) : n_block_new_min;
        return {n_block_new_min, n_block_new_max};
    }
    static
    CUTLASS_DEVICE
    cute::tuple<int, int> get_m_block_min_max(
            SeqlenInfo_t const& seqlen_info,
            int const n_block, int const bidb,
            int const window_size_left, int const window_size_right, int const sink_token_length) {
        int const seqlen_q = seqlen_info.seqlen_q;
        int const seqlen_k = seqlen_info.seqlen_k;
        int m_block_max = cute::ceil_div(seqlen_q, kBlockM);
        if constexpr (Is_local) {
            if (n_block >= cute::ceil_div(sink_token_length, kBlockN)) {
                m_block_max = std::min(m_block_max, cute::ceil_div((n_block + 1) * kBlockN + seqlen_q - seqlen_k + window_size_left, kBlockM));
            }
        }
        int m_block_min = 0;
        if constexpr (Is_causal || Is_local) {
            m_block_min = std::max(m_block_min, (n_block * kBlockN + seqlen_q - seqlen_k - window_size_right) / kBlockM);
        }
        return {m_block_min, m_block_max};
    }
    static
    CUTLASS_DEVICE
    int get_n_block_min_causal_local_mask(
            SeqlenInfo_t const& seqlen_info,
            int const m_block, int const n_block_min, int const window_size_right,
            cutlass::FastDivmod const& attention_chunk_divmod,
            cutlass::FastDivmod const& qhead_per_khead_divmod) {
        int const m_idx_min = !PackGQA ? m_block * kBlockM : qhead_per_khead_divmod.divide(m_block * kBlockM);
        int const n_idx = m_idx_min + seqlen_info.seqlen_k - seqlen_info.seqlen_q;
        int n_idx_right = !Is_local ? n_idx : n_idx + window_size_right;
        if (Is_local && attention_chunk_divmod.divisor > 0) {
            n_idx_right = std::min(n_idx_right, flash::round_up(attention_chunk_divmod, n_idx));
        }
        return std::max(n_block_min, n_idx_right / kBlockN);
    }
    static
    CUTLASS_DEVICE
    int get_n_block_min_before_local_mask(
            SeqlenInfo_t const& seqlen_info,
            int const m_block, int const n_block_min, int const window_size_left,
            cutlass::FastDivmod const& attention_chunk_divmod,
            cutlass::FastDivmod const& qhead_per_khead_divmod) {
        int const m_idx_max = !PackGQA ? (m_block + 1) * kBlockM : qhead_per_khead_divmod.divide((m_block + 1) * kBlockM - 1) + 1;
        int const n_idx = m_idx_max + seqlen_info.seqlen_k - seqlen_info.seqlen_q;
        int n_idx_left = !Is_local ? n_idx : n_idx - window_size_left;
        if (Is_local && attention_chunk_divmod.divisor > 0) {
            n_idx_left = std::max(n_idx_left, flash::round_down(attention_chunk_divmod, n_idx));
        }
        return !Is_local ? n_block_min : std::max(n_block_min, cute::ceil_div(n_idx_left, kBlockN));
    }
};
}
#include <cute/tensor.hpp>
#include "cutlass/fast_math.h"
namespace flash {
using namespace cute;
template <int kBlockM, int kBlockN, bool PackGQA, typename TiledMma, bool SwapAB=false>
struct Mask {
    static_assert(!(PackGQA && SwapAB), "Cannot be both PackGQA and SwapAB");
    int const thread_idx;
    int const seqlen_q, seqlen_k;
    int const window_size_left, window_size_right, sink_token_length;
    cutlass::FastDivmod const attention_chunk_divmod;
    cutlass::FastDivmod const qhead_per_khead_divmod;
    CUTLASS_DEVICE
    Mask(const int thread_idx, const int seqlen_q, const int seqlen_k,
         const int window_size_left, const int window_size_right, const int sink_token_length,
         cutlass::FastDivmod const &attention_chunk_divmod,
         cutlass::FastDivmod const &qhead_per_khead_divmod)
        : thread_idx(thread_idx)
        , seqlen_q(seqlen_q)
        , seqlen_k(seqlen_k)
        , window_size_left(window_size_left)
        , window_size_right(window_size_right)
        , sink_token_length(sink_token_length)
        , attention_chunk_divmod(attention_chunk_divmod)
        , qhead_per_khead_divmod(qhead_per_khead_divmod)
    {
    };
    template <bool Seqlenk_mask=false, bool Causal_mask=false, bool Local_mask=false,
        typename Engine, typename Layout>
    CUTLASS_DEVICE
    void apply(Tensor<Engine, Layout> &tSrS, const int m_block, const int n_block) const {
        static_assert(!(Causal_mask && Local_mask), "Cannot be both causal and local");
        static_assert(Layout::rank == 3, "Only support 3D Tensor");
        if (!Seqlenk_mask && !Causal_mask && !Local_mask) { return; }
        auto thread_mma = TiledMma{}.get_thread_slice(thread_idx);
        auto thread0_mma = TiledMma{}.get_thread_slice(_0{});
        static constexpr int Row = !SwapAB ? 0 : 1, Col = !SwapAB ? 1 : 0;
        Tensor cS = cute::make_identity_tensor(Shape<Int<!SwapAB ? kBlockM : kBlockN>, Int<!SwapAB ? kBlockN : kBlockM>>{});
        Tensor tScS = thread_mma.partition_C(cS);
        Tensor tSrS_rowcol = make_tensor(tSrS.data(), flash::convert_layout_acc_rowcol<SwapAB>(tSrS.layout()));
        Tensor tScS_rowcol = make_tensor(tScS.data(), flash::convert_layout_acc_rowcol<SwapAB>(tScS.layout()));
        Tensor t0ScS = thread0_mma.partition_C(cS);
        Tensor t0ScS_rowcol = make_tensor(t0ScS.data(), flash::convert_layout_acc_rowcol<SwapAB>(t0ScS.layout()));
        int const thread_col_offset = get<Col>(tScS_rowcol(_0{}, _0{}));
        int const seqlenk_col_limit = seqlen_k - n_block * kBlockN - thread_col_offset;
        if constexpr (!Causal_mask && !Local_mask) {
            if constexpr (Seqlenk_mask) {
                #pragma unroll
                for (int n = 0; n < size<1>(tSrS_rowcol); ++n) {
                    if (int(get<Col>(t0ScS_rowcol(_0{}, n))) >= seqlenk_col_limit) {
                        #pragma unroll
                        for (int m = 0; m < size<0>(tSrS_rowcol); ++m) { tSrS_rowcol(m, n) = -INFINITY; }
                    }
                }
            }
        } else {
            if constexpr (!SwapAB) {
                static constexpr int kMmaThreadsPerRow = size<0, 0>(typename TiledMma::AtomLayoutC_TV{});
                static_assert(cutlass::NumThreadsPerWarp % kMmaThreadsPerRow == 0);
                static_assert(!PackGQA || CUTE_STATIC_V(size<0>(tSrS_rowcol)) <= kMmaThreadsPerRow);
                int mma_m_idx;
                if constexpr (PackGQA) {
                    mma_m_idx = qhead_per_khead_divmod.divide(m_block * kBlockM + get<Row>(tScS_rowcol(thread_idx % kMmaThreadsPerRow, _0{})));
                }
                int const causal_row_offset = 1 + seqlen_k - n_block * kBlockN - seqlen_q - thread_col_offset;
                if constexpr (Causal_mask) {
                    #pragma unroll
                    for (int m = 0; m < size<0>(tSrS_rowcol); ++m) {
                        int const row_idx = !PackGQA
                            ? get<Row>(tScS_rowcol(m, _0{})) + m_block * kBlockM
                            :  __shfl_sync(0xffffffff, mma_m_idx, m % kMmaThreadsPerRow, kMmaThreadsPerRow);
                        int const col_limit_right = !Seqlenk_mask
                            ? row_idx + causal_row_offset
                            : __viaddmin_s32(row_idx, causal_row_offset, seqlenk_col_limit);
                        #pragma unroll
                        for (int n = 0; n < size<1>(tSrS_rowcol); ++n) {
                            if (int(get<Col>(t0ScS_rowcol(_0{}, n))) >= col_limit_right) { tSrS_rowcol(m, n) = -INFINITY; }
                        }
                    }
                } else {
                    int const local_row_offset_right = causal_row_offset + window_size_right;
                    int const local_row_offset_left = causal_row_offset - 1 - window_size_left;
                    int const col_limit_sink = sink_token_length - n_block * kBlockN;
                    #pragma unroll
                    for (int m = 0; m < size<0>(tSrS_rowcol); ++m) {
                        int const row_idx = !PackGQA
                            ? get<Row>(tScS_rowcol(m, _0{})) + m_block * kBlockM
                            :  __shfl_sync(0xffffffff, mma_m_idx, m % kMmaThreadsPerRow, kMmaThreadsPerRow);
                        int col_limit_right = !Seqlenk_mask
                            ? row_idx + local_row_offset_right
                            : __viaddmin_s32(row_idx, local_row_offset_right, seqlenk_col_limit);
                        int col_limit_left = row_idx + local_row_offset_left;
                        if (attention_chunk_divmod.divisor > 0) {
                            int col_limit_left_chunk = flash::round_down(attention_chunk_divmod, row_idx + seqlen_k - seqlen_q) - n_block * kBlockN - thread_col_offset;
                            col_limit_left = std::max(col_limit_left, col_limit_left_chunk);
                            col_limit_right = std::min(col_limit_right, col_limit_left_chunk + attention_chunk_divmod.divisor);
                        }
                        #pragma unroll
                        for (int n = 0; n < size<1>(tSrS_rowcol); ++n) {
                            int const col_idx = int(get<Col>(t0ScS_rowcol(m, n)));
                            if (col_idx >= col_limit_right || (col_idx < col_limit_left && col_idx >= col_limit_sink)) { tSrS_rowcol(m, n) = -INFINITY; }
                        }
                    }
                }
            } else {
                int const thread_row_offset = get<Row>(tScS_rowcol(_0{}, _0{}));
                int const causal_row_offset = seqlenk_col_limit - seqlen_q + m_block * kBlockM + thread_row_offset;
                if constexpr (Causal_mask) {
                    #pragma unroll
                    for (int n = 0; n < size<1>(tSrS_rowcol); ++n) {
                        int const col0 = int(get<Col>(t0ScS_rowcol(_0{}, n)));
                        int const row_limit_top = col0 >= seqlenk_col_limit ? kBlockM : col0 - causal_row_offset;
                        #pragma unroll
                        for (int m = 0; m < size<0>(tSrS_rowcol); ++m) {
                            if (int(get<Row>(t0ScS_rowcol(m, _0{}))) < row_limit_top) { tSrS_rowcol(m, n) = -INFINITY; }
                        }
                    }
                } else {
                    int const col_limit_sink = sink_token_length - n_block * kBlockN - thread_col_offset;
                    #pragma unroll
                    for (int n = 0; n < size<1>(tSrS_rowcol); ++n) {
                        int const col0 = int(get<Col>(t0ScS_rowcol(_0{}, n)));
                        int const row_limit_top = col0 >= seqlenk_col_limit ? kBlockM : col0 - causal_row_offset - window_size_right;
                        int const row_limit_bot = col0 < col_limit_sink ? kBlockM : col0 - causal_row_offset + window_size_left;
                        #pragma unroll
                        for (int m = 0; m < size<0>(tSrS_rowcol); ++m) {
                            int const row_idx = int(get<Row>(t0ScS_rowcol(m, _0{})));
                            if (row_idx < row_limit_top || row_idx > row_limit_bot) { tSrS_rowcol(m, n) = -INFINITY; }
                        }
                    }
                }
            }
        }
    };
};
template <int kBlockM, int kBlockN, bool PackGQA, typename TiledMma, bool SwapAB=false>
struct SparseMask {
    static constexpr int kNumInt32PerBlock = (kBlockN + 31) / 32;
    int const thread_idx;
    int const seqlen_q, seqlen_k;
    int const max_k_blocks;
    cutlass::FastDivmod const qhead_per_khead_divmod;
    CUTLASS_DEVICE
    SparseMask(const int thread_idx, const int seqlen_q, const int seqlen_k,
               int const max_k_blocks,
               cutlass::FastDivmod const qhead_per_khead_divmod
            )
        : thread_idx(thread_idx)
        , seqlen_q(seqlen_q)
        , seqlen_k(seqlen_k)
        , max_k_blocks(max_k_blocks)
        , qhead_per_khead_divmod(qhead_per_khead_divmod)
    {
    }
    template <bool Seqlenk_mask=false, typename Engine, typename Layout>
    CUTLASS_DEVICE
    void apply(Tensor<Engine, Layout> &tSrS, const int m_block, const int n_block,
               uint32_t const* __restrict__ smem_mask_ptr) const {
        auto thread_mma = TiledMma{}.get_thread_slice(thread_idx);
        auto thread0_mma = TiledMma{}.get_thread_slice(_0{});
        static constexpr int Row = !SwapAB ? 0 : 1, Col = !SwapAB ? 1 : 0;
        Tensor cS = cute::make_identity_tensor(Shape<Int<!SwapAB ? kBlockM : kBlockN>, Int<!SwapAB ? kBlockN : kBlockM>>{});
        Tensor tScS = thread_mma.partition_C(cS);
        Tensor t0ScS = thread0_mma.partition_C(cS);
        Tensor tSrS_rowcol = make_tensor(tSrS.data(), flash::convert_layout_acc_rowcol<SwapAB>(tSrS.layout()));
        Tensor tScS_rowcol = make_tensor(tScS.data(), flash::convert_layout_acc_rowcol<SwapAB>(tScS.layout()));
        Tensor t0ScS_rowcol = make_tensor(t0ScS.data(), flash::convert_layout_acc_rowcol<SwapAB>(t0ScS.layout()));
        static constexpr int kMmaThreadsPerRow = size<0, 0>(typename TiledMma::AtomLayoutC_TV{});
        int const thread_col_offset = get<Col>(tScS_rowcol(_0{}, _0{}));
        int const quad_lane = thread_col_offset / 2;
        int const seqlenk_col_limit = seqlen_k - n_block * kBlockN - thread_col_offset;
        const bool is_block_safe = n_block < max_k_blocks;
        #pragma unroll
        for (int m = 0; m < size<0>(tSrS_rowcol); ++m) {
            int local_row = get<Row>(tScS_rowcol(m, _0{}));
            int global_row_idx;
            if constexpr (!PackGQA) {
                global_row_idx = local_row + m_block * kBlockM;
            } else {
                int mma_m_idx = qhead_per_khead_divmod.divide(m_block * kBlockM + local_row);
                global_row_idx = __shfl_sync(0xffffffff, mma_m_idx, m % kMmaThreadsPerRow, kMmaThreadsPerRow);
            }
            bool const is_row_safe = (global_row_idx < seqlen_q) && is_block_safe;
            if (!is_row_safe) {
                #pragma unroll
                for (int n = 0; n < size<1>(tSrS_rowcol); ++n) {
                    tSrS_rowcol(m, n) = -INFINITY;
                }
                continue;
            }
            if constexpr (kBlockN == 128) {
                uint32_t const mask_word = smem_mask_ptr[local_row * 4 + quad_lane];
                #pragma unroll
                for (int n = 0; n < size<1>(tSrS_rowcol); ++n) {
                    bool const mask_bit = (mask_word >> n) & 1;
                    if constexpr (Seqlenk_mask) {
                        int const col0 = int(get<Col>(t0ScS_rowcol(_0{}, n)));
                        bool const is_valid = col0 < seqlenk_col_limit;
                        bool const final_keep = is_valid && mask_bit;
                        if (!final_keep) {
                            tSrS_rowcol(m, n) = -INFINITY;
                        }
                    } else {
                        if (!mask_bit) {
                            tSrS_rowcol(m, n) = -INFINITY;
                        }
                    }
                }
            } else if constexpr (kBlockN == 64) {
                int const word_idx = quad_lane / 2;
                int const half_idx = quad_lane % 2;
                uint32_t const full_word = smem_mask_ptr[local_row * 2 + word_idx];
                uint32_t const mask_word = half_idx ? (full_word >> 16) : full_word;
                #pragma unroll
                for (int n = 0; n < size<1>(tSrS_rowcol); ++n) {
                    bool const mask_bit = (mask_word >> n) & 1;
                    if constexpr (Seqlenk_mask) {
                        int const col0 = int(get<Col>(t0ScS_rowcol(_0{}, n)));
                        bool const is_valid = col0 < seqlenk_col_limit;
                        bool const final_keep = is_valid && mask_bit;
                        if (!final_keep) {
                            tSrS_rowcol(m, n) = -INFINITY;
                        }
                    } else {
                        if (!mask_bit) {
                            tSrS_rowcol(m, n) = -INFINITY;
                        }
                    }
                }
            } else {
                #pragma unroll
                for (int n = 0; n < size<1>(tSrS_rowcol); ++n) {
                    int const token_in_block = int(get<Col>(tScS_rowcol(m, n)));
                    int const word_idx = token_in_block / 32;
                    int const bit_idx = token_in_block % 32;
                    bool const mask_bit = (smem_mask_ptr[local_row * kNumInt32PerBlock + word_idx] >> bit_idx) & 1;
                    if constexpr (Seqlenk_mask) {
                        bool const is_valid = (n_block * kBlockN + token_in_block < seqlen_k);
                        bool const final_keep = is_valid && mask_bit;
                        if (!final_keep) {
                            tSrS_rowcol(m, n) = -INFINITY;
                        }
                    } else {
                        if (!mask_bit) {
                            tSrS_rowcol(m, n) = -INFINITY;
                        }
                    }
                }
            }
        }
    }
};
}
#include <cute/tensor.hpp>
#include "cutlass/fast_math.h"
namespace flash {
using namespace cute;
template <int kBlockM, int kHeadDim, int NumThreads, typename Element>
struct PackGQAManager {
    static constexpr int kGmemElemsPerLoad = sizeof(cute::uint128_t) / sizeof(Element);
    static constexpr int kGmemElemsPerStore = kGmemElemsPerLoad;
    static_assert(kHeadDim % kGmemElemsPerLoad == 0, "Headdim must be a multiple of kGmemElemsPerLoad");
    static constexpr int kBytePerRow = kHeadDim * sizeof(Element);
    static constexpr int kBlockKGmem = (kBytePerRow % 128 == 0 ? 128 : (kBytePerRow % 64 == 0 ? 64 : 32)) / sizeof(Element);
    static constexpr int kGmemThreadsPerRow = kBlockKGmem / kGmemElemsPerLoad;
    static_assert(NumThreads % kGmemThreadsPerRow == 0, "NumThreads must be a multiple of kGmemThreadsPerRow");
    static_assert(cutlass::NumThreadsPerWarp % kGmemThreadsPerRow == 0, "kGmemThreadsPerRow must divide NumThreadsPerWarp");
    using GmemCopyAtomCpAsync = cute::Copy_Atom<SM80_CP_ASYNC_CACHEGLOBAL_ZFILL<uint128_t>, Element>;
    using GmemLayoutAtom = Layout<Shape <Int<NumThreads / kGmemThreadsPerRow>, Int<kGmemThreadsPerRow>>,
                                  Stride<Int<kGmemThreadsPerRow>, _1>>;
    using GmemTiledCopyQCpAsync = decltype(
        make_tiled_copy(GmemCopyAtomCpAsync{},
                        GmemLayoutAtom{},
                        Layout<Shape<_1, Int<kGmemElemsPerLoad>>>{}));
    using GmemTiledCopyO = decltype(
        make_tiled_copy(Copy_Atom<AutoVectorizingCopyWithAssumedAlignment<128>, Element>{},
                        GmemLayoutAtom{},
                        Layout<Shape<_1, Int<kGmemElemsPerStore>>>{}));
    template <int NumThreadsPerRow=kGmemThreadsPerRow, typename Engine, typename Layout, typename TensorC>
    CUTLASS_DEVICE
    static auto
    compute_ptr(Tensor<Engine, Layout> &tensor, TensorC const &tRows,
                cutlass::FastDivmod const &qhead_per_khead_divmod, int const thread_idx, int const m_block) {
        static constexpr int NumPtrPerThread = cute::ceil_div(CUTE_STATIC_V(cute::size(tRows)), NumThreadsPerRow);
        using TensorType = typename Engine::value_type;
        Tensor tPrPtr = make_tensor<TensorType const*>(Shape<Int<NumPtrPerThread>>{});
        #pragma unroll
        for (int i = 0; i < NumPtrPerThread; ++i) {
            int const row = i * NumThreads + get<0>(tRows(thread_idx % NumThreadsPerRow));
            int const idx = m_block * kBlockM + row;
            int m_idx, h_idx;
            m_idx = qhead_per_khead_divmod.divmod(h_idx, idx);
            tPrPtr[i] = &tensor(make_coord(make_coord(h_idx, m_idx)));
        }
        return tPrPtr;
    }
    template <typename TensormQ, typename TensorsQ>
    CUTLASS_DEVICE
    static void
    load_Q(TensormQ const &mQ,
           TensorsQ &sQ,
           cutlass::FastDivmod const &qhead_per_khead_divmod,
           int const thread_idx, int const seqlen_q, int const m_block
          )
    {
        GmemTiledCopyQCpAsync gmem_tiled_copy_Q_cp_async;
        auto gmem_thr_copy_Q_cp_async = gmem_tiled_copy_Q_cp_async.get_thread_slice(thread_idx);
        Tensor cQ = make_identity_tensor(make_shape(size<0>(sQ), size<1>(sQ)));
        Tensor tQcQ = gmem_thr_copy_Q_cp_async.partition_S(cQ);
        Tensor tQsQ = gmem_thr_copy_Q_cp_async.partition_D(sQ);
        Tensor tQpQ = make_tensor<bool>(make_shape(size<2>(tQsQ)));
        #pragma unroll
        for (int k = 0; k < size(tQpQ); ++k) { tQpQ(k) = get<1>(tQcQ(_0{}, _0{}, k)) < size<1>(mQ); }
        Tensor mQ_0 = mQ(_, _0{});
        Tensor tQcQ_row = tQcQ(_0{}, _, _0{});
        Tensor tPrQPtr = compute_ptr(mQ_0, tQcQ_row, qhead_per_khead_divmod, thread_idx, m_block);
        int const qhead_per_khead = qhead_per_khead_divmod.divisor;
        #pragma unroll
        for (int m = 0; m < size<1>(tQsQ); ++m) {
            int idx = m_block * kBlockM + get<0>(tQcQ(_0{}, m, _0{}));
            Element const* q_ptr = reinterpret_cast<Element const*>(__shfl_sync(0xffffffff, reinterpret_cast<uint64_t>(tPrQPtr(m / kGmemThreadsPerRow)), m % kGmemThreadsPerRow, kGmemThreadsPerRow));
            if (idx < seqlen_q * qhead_per_khead) {
                Tensor mQ_cur = make_tensor(make_gmem_ptr(q_ptr), Shape<Int<kHeadDim>>{});
                Tensor mQ_cur_copy = cute::tiled_divide(mQ_cur, Shape<Int<kGmemElemsPerLoad>>{});
                #pragma unroll
                for (int k = 0; k < size<2>(tQsQ); ++k) {
                    int ki = get<1>(tQcQ(_0{}, _0{}, k)) / kGmemElemsPerLoad;
                    cute::copy(gmem_tiled_copy_Q_cp_async.with(tQpQ(k)), mQ_cur_copy(_, ki), tQsQ(_, m, k));
                }
            }
        }
    };
    template <typename TensormLSE, typename TensorsLSE, typename TiledMma>
    CUTLASS_DEVICE
    static void
    store_LSE(TensormLSE &mLSE,
              TensorsLSE const &tLSErLSE,
              TiledMma tiled_mma,
              cutlass::FastDivmod const &qhead_per_khead_divmod,
              int const thread_idx, int const seqlen_o, int const m_block
             )
    {
        Tensor caccO = cute::make_identity_tensor(Shape<Int<kBlockM>, Int<kHeadDim>>{});
        auto thread_mma = tiled_mma.get_thread_slice(thread_idx);
        Tensor taccOcO = thread_mma.partition_C(caccO);
        Tensor taccOcO_row = make_tensor(taccOcO.data(), flash::convert_layout_acc_rowcol(taccOcO.layout()))(_, _0{});
        CUTE_STATIC_ASSERT_V(size(tLSErLSE) == size(taccOcO_row));
        static constexpr int kMmaThreadsPerRow = size<0, 0>(typename TiledMma::AtomLayoutC_TV{});
        static_assert(cutlass::NumThreadsPerWarp % kMmaThreadsPerRow == 0);
        static_assert(CUTE_STATIC_V(size(tLSErLSE)) <= kMmaThreadsPerRow);
        static_assert(CUTE_STATIC_V(size(taccOcO_row)) <= kMmaThreadsPerRow);
        Tensor tPrLSEPtr = compute_ptr<kMmaThreadsPerRow>(mLSE, taccOcO_row, qhead_per_khead_divmod, thread_idx, m_block);
        static_assert(CUTE_STATIC_V(size(tPrLSEPtr)) == 1);
        int const qhead_per_khead = qhead_per_khead_divmod.divisor;
        #pragma unroll
        for (int mi = 0; mi < size(tLSErLSE); ++mi) {
            int const row = m_block * kBlockM + get<0>(taccOcO_row(mi));
            float* ptr_LSE_cur = reinterpret_cast<float*>(__shfl_sync(0xffffffff, reinterpret_cast<uint64_t>(tPrLSEPtr[0]), mi % kMmaThreadsPerRow, kMmaThreadsPerRow));
            if (get<1>(taccOcO_row(_0{})) == 0 && row < seqlen_o * qhead_per_khead) {
                *ptr_LSE_cur = tLSErLSE(mi);
            }
        }
    };
    static constexpr int kX52Rows = kBlockM * kGmemThreadsPerRow / NumThreads;
    template <typename TensormO>
    CUTLASS_DEVICE
    static void
    load_gate(uint4 (&gpre)[kX52Rows], TensormO &mO, cutlass::FastDivmod const &qhead_per_khead_divmod,
              int const thread_idx, int const seqlen_o, int const m_block,
              Element const* obase, int64_t const orow, Element const* gate, int const gstride)
    {
        GmemTiledCopyO gmem_tiled_copy_O;
        auto gmem_thr_copy_O = gmem_tiled_copy_O.get_thread_slice(thread_idx);
        Tensor cO = make_identity_tensor(Shape<Int<kBlockM>, Int<kHeadDim>>{});
        Tensor tOcO = gmem_thr_copy_O.partition_D(cO);
        static_assert(CUTE_STATIC_V(size<1>(tOcO)) == kX52Rows && CUTE_STATIC_V(size<2>(tOcO)) == 1, "x52 gate rows");
        Tensor mO_0 = mO(_, _0{});
        Tensor tOcO_row = tOcO(_0{}, _, _0{});
        Tensor tPrOPtr = compute_ptr(mO_0, tOcO_row, qhead_per_khead_divmod, thread_idx, m_block);
        int const qhead_per_khead = qhead_per_khead_divmod.divisor;
        int const col = get<1>(tOcO(_0{}, _0{}, _0{}));
        #pragma unroll
        for (int m = 0; m < kX52Rows; ++m) {
            int idx = m_block * kBlockM + get<0>(tOcO(_0{}, m, _0{}));
            Element* o_ptr = reinterpret_cast<Element*>(__shfl_sync(0xffffffff, reinterpret_cast<uint64_t>(tPrOPtr(m / kGmemThreadsPerRow)), m % kGmemThreadsPerRow, kGmemThreadsPerRow));
            gpre[m] = make_uint4(0u, 0u, 0u, 0u);
            if (idx < seqlen_o * qhead_per_khead) {
                int64_t const off = o_ptr - obase;
                int64_t const tok = off / orow;
                gpre[m] = __ldg(reinterpret_cast<uint4 const*>(gate + tok * gstride + (off - tok * orow) + col));
            }
        }
    };
    template <typename TensormO, typename TensorrO>
    CUTLASS_DEVICE
    static void
    store_O_g(TensormO &mO, TensorrO &tOrO, cutlass::FastDivmod const &qhead_per_khead_divmod,
              int const thread_idx, int const seqlen_o, int const m_block,
              float const geps, uint4 const* gpre)
    {
        GmemTiledCopyO gmem_tiled_copy_O;
        auto gmem_thr_copy_O = gmem_tiled_copy_O.get_thread_slice(thread_idx);
        Tensor cO = make_identity_tensor(Shape<Int<kBlockM>, Int<kHeadDim>>{});
        Tensor tOcO = gmem_thr_copy_O.partition_D(cO);
        Tensor tOpO = make_tensor<bool>(make_shape(size<2>(tOcO)));
        #pragma unroll
        for (int k = 0; k < size(tOpO); ++k) { tOpO(k) = get<1>(tOcO(_0{}, _0{}, k)) < size<1>(mO); }
        Tensor mO_0 = mO(_, _0{});
        Tensor tOcO_row = tOcO(_0{}, _, _0{});
        Tensor tPrOPtr = compute_ptr(mO_0, tOcO_row, qhead_per_khead_divmod, thread_idx, m_block);
        int const qhead_per_khead = qhead_per_khead_divmod.divisor;
        static_assert(kHeadDim == 64 && kGmemThreadsPerRow == 8 && kGmemElemsPerStore == 8, "x52 gate layout");
        static_assert(CUTE_STATIC_V(size<2>(tOcO)) == 1, "x52 gate: one 8-element column group per thread");
        #pragma unroll
        for (int m = 0; m < size<1>(tOrO); ++m) {
            int idx = m_block * kBlockM + get<0>(tOcO(_0{}, m, _0{}));
            Element* o_ptr = reinterpret_cast<Element*>(__shfl_sync(0xffffffff, reinterpret_cast<uint64_t>(tPrOPtr(m / kGmemThreadsPerRow)), m % kGmemThreadsPerRow, kGmemThreadsPerRow));
            float ss = 0.f;
            #pragma unroll
            for (int k = 0; k < size<2>(tOrO); ++k) {
                #pragma unroll
                for (int i = 0; i < size<0>(tOrO); ++i) { float const xv = float(tOrO(i, m, k)); ss = fmaf(xv, xv, ss); }
            }
            #pragma unroll
            for (int o = 1; o < kGmemThreadsPerRow; o <<= 1) ss += __shfl_xor_sync(0xffffffffu, ss, o);
            float const rs = rsqrtf(ss * (1.f / 64.f) + geps);
            if (idx < seqlen_o * qhead_per_khead) {
                Element const* gv8 = reinterpret_cast<Element const*>(&gpre[m]);
                Tensor mO_cur = make_tensor(make_gmem_ptr(o_ptr), Shape<Int<kHeadDim>>{});
                Tensor mO_cur_copy = cute::tiled_divide(mO_cur, Shape<Int<kGmemElemsPerStore>>{});
                #pragma unroll
                for (int k = 0; k < size<2>(tOrO); ++k) {
                    int const col = get<1>(tOcO(_0{}, _0{}, k));
                    int ki = col / kGmemElemsPerStore;
                    if (tOpO(k)) {
                        #pragma unroll
                        for (int i = 0; i < size<0>(tOrO); ++i) {
                            float const gv = float(gv8[i]);
                            tOrO(i, m, k) = Element(float(tOrO(i, m, k)) * rs / (1.f + __expf(-gv)));
                        }
                        cute::copy(gmem_tiled_copy_O, tOrO(_, m, k), mO_cur_copy(_, ki));
                    }
                }
            }
        }
    };
    template <typename TensormO, typename TensorrO>
    CUTLASS_DEVICE
    static void
    store_O(TensormO &mO,
            TensorrO const &tOrO,
            cutlass::FastDivmod const &qhead_per_khead_divmod,
            int const thread_idx, int const seqlen_o, int const m_block
          )
    {
        GmemTiledCopyO gmem_tiled_copy_O;
        auto gmem_thr_copy_O = gmem_tiled_copy_O.get_thread_slice(thread_idx);
        Tensor cO = make_identity_tensor(Shape<Int<kBlockM>, Int<kHeadDim>>{});
        Tensor tOcO = gmem_thr_copy_O.partition_D(cO);
        Tensor tOpO = make_tensor<bool>(make_shape(size<2>(tOcO)));
        #pragma unroll
        for (int k = 0; k < size(tOpO); ++k) { tOpO(k) = get<1>(tOcO(_0{}, _0{}, k)) < size<1>(mO); }
        Tensor mO_0 = mO(_, _0{});
        Tensor tOcO_row = tOcO(_0{}, _, _0{});
        Tensor tPrOPtr = compute_ptr(mO_0, tOcO_row, qhead_per_khead_divmod, thread_idx, m_block);
        int const qhead_per_khead = qhead_per_khead_divmod.divisor;
        #pragma unroll
        for (int m = 0; m < size<1>(tOrO); ++m) {
            int idx = m_block * kBlockM + get<0>(tOcO(_0{}, m, _0{}));
            Element* o_ptr = reinterpret_cast<Element*>(__shfl_sync(0xffffffff, reinterpret_cast<uint64_t>(tPrOPtr(m / kGmemThreadsPerRow)), m % kGmemThreadsPerRow, kGmemThreadsPerRow));
            if (idx < seqlen_o * qhead_per_khead) {
                Tensor mO_cur = make_tensor(make_gmem_ptr(o_ptr), Shape<Int<kHeadDim>>{});
                Tensor mO_cur_copy = cute::tiled_divide(mO_cur, Shape<Int<kGmemElemsPerStore>>{});
                #pragma unroll
                for (int k = 0; k < size<2>(tOrO); ++k) {
                    int ki = get<1>(tOcO(_0{}, _0{}, k)) / kGmemElemsPerStore;
                    if (tOpO(k)) {
                        cute::copy(gmem_tiled_copy_O, tOrO(_, m, k), mO_cur_copy(_, ki));
                    }
                }
            }
        }
    };
    template <typename TensormO, typename TensorrO, typename TiledMma>
    CUTLASS_DEVICE
    static void
    store_O_direct(TensormO &mO,
                   TensorrO const &tOrO,
                   TiledMma tiled_mma,
                   cutlass::FastDivmod const &qhead_per_khead_divmod,
                   int const thread_idx, int const seqlen_o, int const m_block
                 )
    {
        static constexpr int kGmemElemsPerStoreDirect = 2;
        cute::Copy_Atom<AutoVectorizingCopyWithAssumedAlignment<128>, Element> gmem_copy_direct;
        Tensor tOrO_rowcol = make_tensor(tOrO.data(), flash::convert_layout_acc_rowcol(tOrO.layout()));
        Tensor tOrO_copy = cute::tiled_divide(tOrO_rowcol, Shape<_1, Int<kGmemElemsPerStoreDirect>>{});
        Tensor caccO = cute::make_identity_tensor(Shape<Int<kBlockM>, Int<kHeadDim>>{});
        auto thread_mma = tiled_mma.get_thread_slice(thread_idx);
        Tensor taccOcO = thread_mma.partition_C(caccO);
        Tensor taccOcO_rowcol = make_tensor(taccOcO.data(), flash::convert_layout_acc_rowcol(taccOcO.layout()));
        Tensor taccOcO_row = taccOcO_rowcol(_, _0{});
        Tensor taccOcO_col = taccOcO_rowcol(_0{}, _);
        static constexpr int kMmaThreadsPerRow = size<0, 0>(typename TiledMma::AtomLayoutC_TV{});
        static_assert(cutlass::NumThreadsPerWarp % kMmaThreadsPerRow == 0);
        static_assert(CUTE_STATIC_V(size(taccOcO_row)) <= kMmaThreadsPerRow);
        Tensor mO_0 = mO(_, _0{});
        Tensor tPrOPtr = compute_ptr<kMmaThreadsPerRow>(mO_0, taccOcO_row, qhead_per_khead_divmod, thread_idx, m_block);
        static_assert(CUTE_STATIC_V(size(tPrOPtr)) == 1);
        int const qhead_per_khead = qhead_per_khead_divmod.divisor;
        #pragma unroll
        for (int m = 0; m < size<1>(tOrO_copy); ++m) {
            int row = m_block * kBlockM + get<0>(taccOcO_row(m));
            Element* o_ptr = reinterpret_cast<Element*>(__shfl_sync(0xffffffff, reinterpret_cast<uint64_t>(tPrOPtr[0]), m % kMmaThreadsPerRow, kMmaThreadsPerRow));
            if (row < seqlen_o * qhead_per_khead) {
                Tensor mO_cur = make_tensor(make_gmem_ptr(o_ptr), Shape<Int<kHeadDim>>{});
                Tensor mO_cur_copy = cute::tiled_divide(mO_cur, Shape<Int<kGmemElemsPerStoreDirect>>{});
                #pragma unroll
                for (int k = 0; k < size<2>(tOrO_copy); ++k) {
                    int col = get<1>(taccOcO_col(k * kGmemElemsPerStoreDirect));
                    if (col < size<1>(mO)) {
                        cute::copy(gmem_copy_direct, tOrO_copy(_, m, k), mO_cur_copy(_, col / kGmemElemsPerStoreDirect));
                    }
                }
            }
        }
    };
};
}
#include <cute/tensor.hpp>
#include "cutlass/fast_math.h"
namespace flash {
using namespace cute;
template <int kBlockN, int kHeadDim, int kHeadDimV, int NumThreads, typename Element, bool KV_Same_Iter=false, int LoadsPerRow_LB=1>
struct PagedKVManager {
    static constexpr bool SameHeadDim = (kHeadDim == kHeadDimV);
    static constexpr int kHeadDimGCD = cute::gcd(kHeadDim, kHeadDimV);
    static constexpr int kGmemElemsPerLoad = sizeof(cute::uint128_t) / sizeof(Element);
    static_assert(kHeadDimGCD % kGmemElemsPerLoad == 0, "Headdim and HeaddimV must be a multiple of kGmemElemsPerLoad");
    static_assert(kHeadDimGCD % LoadsPerRow_LB == 0, "Headdim and HeaddimV must be a multiple of LoadsPerRow_LB");
    static constexpr int kBytePerRow = kHeadDimGCD / LoadsPerRow_LB * sizeof(Element);
    static constexpr int kBlockKGmem = (kBytePerRow % 128 == 0 ? 128 : (kBytePerRow % 64 == 0 ? 64 : 32)) / sizeof(Element);
    static constexpr int kGmemThreadsPerRow = kBlockKGmem / kGmemElemsPerLoad;
    static_assert(NumThreads % kGmemThreadsPerRow == 0, "NumThreads must be a multiple of kGmemThreadsPerRow");
    static_assert(cutlass::NumThreadsPerWarp % kGmemThreadsPerRow == 0, "kGmemThreadsPerRow must divide NumThreadsPerWarp");
    using GmemCopyAtomCpAsync = cute::Copy_Atom<SM80_CP_ASYNC_CACHEGLOBAL_ZFILL<uint128_t>, Element>;
    using GmemLayoutAtomKVCpAsync = Layout<Shape <Int<NumThreads / kGmemThreadsPerRow>, Int<kGmemThreadsPerRow>>,
                                           Stride<Int<kGmemThreadsPerRow>, _1>>;
    using GmemTiledCopyKVCpAsync = decltype(
        make_tiled_copy(GmemCopyAtomCpAsync{},
                        GmemLayoutAtomKVCpAsync{},
                        Layout<Shape<_1, Int<kGmemElemsPerLoad>>>{}));
    using GmemTiledCopyKVStore = decltype(
        make_tiled_copy(Copy_Atom<AutoVectorizingCopyWithAssumedAlignment<128>, Element>{},
                        GmemLayoutAtomKVCpAsync{},
                        Layout<Shape<_1, Int<kGmemElemsPerLoad>>>{}));
    using ShapeKV = cute::Shape<int32_t, int32_t, int32_t, int32_t>;
    using StrideKV = cute::Stride<int64_t, _1, int64_t, int64_t>;
    using ShapePageTable = cute::Shape<int32_t, int32_t>;
    using StridePageTable = cute::Stride<int64_t, _1>;
    using TensorPageTable = decltype(make_tensor(make_gmem_ptr(static_cast<int const*>(nullptr)), ShapePageTable{}, StridePageTable{})(int(0), _));
    using TensorKV = decltype(make_tensor(make_gmem_ptr(static_cast<Element*>(nullptr)), ShapeKV{}, StrideKV{})(_, _, int(0), _));
    using GmemThrCopyKVCpAsync = decltype(GmemTiledCopyKVCpAsync{}.get_thread_slice(int(0)));
    using TensortKcK = decltype(GmemTiledCopyKVCpAsync{}.get_thread_slice(int(0)).partition_D(cute::make_identity_tensor(Shape<Int<kBlockN>, Int<kHeadDim>>{})));
    using TensortKpK = decltype(make_tensor<bool>(make_shape(size<1>(TensortKcK{}), size<2>(TensortKcK{})), Stride<_0, _1>{}));
    using TensortVcV = decltype(GmemTiledCopyKVCpAsync{}.get_thread_slice(int(0)).partition_D(cute::make_identity_tensor(Shape<Int<kBlockN>, Int<kHeadDimV>>{})));
    using TensortVpV = decltype(make_tensor<bool>(make_shape(size<1>(TensortVcV{}), size<2>(TensortVcV{})), Stride<_0, _1>{}));
    static_assert(CUTE_STATIC_V(size<1>(TensortKcK{})) == CUTE_STATIC_V(size<1>(TensortVcV{})));
    static constexpr int kPageEntryPerThread = cute::ceil_div(size<1>(TensortKcK{}), kGmemThreadsPerRow);
    using TensorPageOffset = decltype(make_tensor<cute::tuple<int, int>>(Shape<Int<kPageEntryPerThread>>{}));
    using TensorKVPtr = decltype(make_tensor<Element*>(Shape<Int<kPageEntryPerThread>>{}));
    GmemTiledCopyKVCpAsync gmem_tiled_copy_kv;
    cutlass::FastDivmod const &page_size_divmod;
    cutlass::FastDivmod const &blockN_per_page_size_divmod;
    int const thread_idx;
    int const seqlen_k;
    int const leftpad_k;
    int const* const ptr_page_table;
    GmemThrCopyKVCpAsync const gmem_thr_copy_kv;
    TensorPageTable mPageTable;
    TensorKV mK_paged, mV_paged;
    TensortKpK tKpK;
    TensortVpV tVpV;
    TensorPageOffset tPrPageOffset;
    TensorKVPtr tPrVPtr;
    int bidb_kv_idx, bidb_kv_idx_prev, n_block_idx, n_block_idx_prev;
    CUTLASS_DEVICE
    PagedKVManager(int const* const ptr_page_table_,
                   ShapePageTable const &shape_pagetable, StridePageTable const &stride_pagetable,
                   Element* const ptr_K, ShapeKV const &shape_K, StrideKV const &stride_K,
                   Element* const ptr_V, int const headdim_v, StrideKV const &stride_V,
                   cutlass::FastDivmod const &page_size_divmod,
                   cutlass::FastDivmod const &blockN_per_page_size_divmod,
                   int const bidb, int const bidh, int const thread_idx, int const seqlen_k, int const leftpad_k,
                   int bidb_kv_idx
                   )
        : page_size_divmod(page_size_divmod)
        , blockN_per_page_size_divmod(blockN_per_page_size_divmod)
        , thread_idx(thread_idx)
        , seqlen_k(seqlen_k)
        , leftpad_k(leftpad_k)
        , ptr_page_table(ptr_page_table_)
        , gmem_thr_copy_kv(gmem_tiled_copy_kv.get_thread_slice(thread_idx))
        , bidb_kv_idx(bidb_kv_idx)
        , bidb_kv_idx_prev(bidb_kv_idx)
    {
        mPageTable = make_tensor(make_gmem_ptr(ptr_page_table), shape_pagetable, stride_pagetable)(bidb, _);
        mK_paged = make_tensor(make_gmem_ptr(ptr_K), shape_K, stride_K)(_, _, bidh, _);
        auto shape_V = make_shape(get<0>(shape_K), headdim_v, get<2>(shape_K), get<3>(shape_K));
        mV_paged = make_tensor(make_gmem_ptr(ptr_V), shape_V, stride_V)(_, _, bidh, _);
        tKpK = make_tensor<bool>(make_shape(size<1>(TensortKcK{}), size<2>(TensortKcK{})), Stride<_0, _1>{});
        Tensor cK = cute::make_identity_tensor(Shape<Int<kBlockN>, Int<kHeadDim>>{});
        Tensor tKcK = gmem_thr_copy_kv.partition_S(cK);
        #pragma unroll
        for (int k = 0; k < size<1>(tKpK); ++k) { tKpK(_0{}, k) = get<1>(tKcK(_0{}, _0{}, k)) < get<1>(shape_K); }
        Tensor tVpV_ = make_tensor<bool>(make_shape(size<1>(TensortVcV{}), size<2>(TensortVcV{})), Stride<_0, _1>{});
        Tensor cV = cute::make_identity_tensor(Shape<Int<kBlockN>, Int<kHeadDimV>>{});
        Tensor tVcV = gmem_thr_copy_kv.partition_S(cV);
        #pragma unroll
        for (int k = 0; k < size<1>(tVpV_); ++k) { tVpV_(_0{}, k) = get<1>(tVcV(_0{}, _0{}, k)) < get<1>(shape_V); }
        tVpV = cute::conditional_return<SameHeadDim>(tKpK, tVpV_);
    };
    template <bool Seqlenk_mask=false, bool First_iter=false>
    CUTLASS_DEVICE
    void load_page_table(const int n_block) {
        #pragma unroll
        for (int i = 0; i < kPageEntryPerThread; ++i) {
            int const row = i * NumThreads + (thread_idx % kGmemThreadsPerRow) * (NumThreads / kGmemThreadsPerRow) + (thread_idx / kGmemThreadsPerRow);
            int const row_idx = n_block * kBlockN + row;
            int page_idx, page_offset;
            page_idx = page_size_divmod.divmod(page_offset, row_idx + leftpad_k);
            int const page = ((i + 1) * NumThreads <= kBlockN || row < kBlockN) && (!Seqlenk_mask || row_idx < seqlen_k) ? mPageTable[page_idx] : 0;
            tPrPageOffset[i] = {page, page_offset};
        }
        if constexpr (First_iter && !KV_Same_Iter) { compute_V_ptr(); }
    };
    template <bool First_iter=false>
    CUTLASS_DEVICE
    void load_page_table_TMA(const int n_block) {
        if (ptr_page_table) {
            bidb_kv_idx = mPageTable[blockN_per_page_size_divmod.divmod(n_block_idx, n_block)];
        } else {
            n_block_idx = n_block;
        }
        if constexpr (First_iter && !KV_Same_Iter) {
            bidb_kv_idx_prev = bidb_kv_idx;
            n_block_idx_prev = n_block_idx;
        }
    };
    CUTLASS_DEVICE
    cute::tuple<int, int> get_indices_for_K_TMA() {
        return {n_block_idx, bidb_kv_idx};
    };
    CUTLASS_DEVICE
    cute::tuple<int, int> get_indices_for_V_TMA() {
        if constexpr (KV_Same_Iter) {
            return {n_block_idx, bidb_kv_idx};
        } else {
            cute::tuple<int, int> const indices = {n_block_idx_prev, bidb_kv_idx_prev};
            bidb_kv_idx_prev = bidb_kv_idx;
            n_block_idx_prev = n_block_idx;
            return indices;
        }
    };
    CUTLASS_DEVICE
    TensorKVPtr compute_K_ptr() {
        Tensor tPrKPtr = make_tensor<Element*>(Shape<Int<kPageEntryPerThread>>{});
        #pragma unroll
        for (int i = 0; i < kPageEntryPerThread; ++i) {
            auto [page, page_offset] = tPrPageOffset[i];
            tPrKPtr[i] = &mK_paged(page_offset, _0{}, page);
        }
        return tPrKPtr;
    };
    CUTLASS_DEVICE
    void compute_V_ptr() {
        #pragma unroll
        for (int i = 0; i < kPageEntryPerThread; ++i) {
            auto [page, page_offset] = tPrPageOffset[i];
            tPrVPtr[i] = &mV_paged(page_offset, _0{}, page);
        }
    };
    template <bool Seqlenk_mask=false, typename TensorK>
    CUTLASS_DEVICE
    void load_K(const int n_block, TensorK &&sK) {
        static constexpr bool EvenN = kBlockN % CUTE_STATIC_V(shape<0>(GmemLayoutAtomKVCpAsync{})) == 0;
        Tensor tPrKPtr = compute_K_ptr();
        auto gmem_thr0_copy_kv = gmem_tiled_copy_kv.get_thread_slice(_0{});
        Tensor tKsK = gmem_thr_copy_kv.partition_D(sK);
        Tensor cK = cute::make_identity_tensor(Shape<Int<kBlockN>, Int<kHeadDim>>{});
        Tensor tKcK = gmem_thr_copy_kv.partition_S(cK);
        Tensor t0KcK = gmem_thr0_copy_kv.partition_S(cK);
        int const seqlenk_row_limit = -int(get<0>(tKcK(_0{}, _0{}, _0{}))) + (EvenN
            ? seqlen_k - n_block * kBlockN
            : (!Seqlenk_mask ? kBlockN : std::min(seqlen_k - n_block * kBlockN, kBlockN)));
        #pragma unroll
        for (int m = 0; m < size<1>(tKsK); ++m) {
            bool const should_load = EvenN
                ? (!Seqlenk_mask || get<0>(t0KcK(_0{}, m, _0{})) < seqlenk_row_limit)
                : get<0>(t0KcK(_0{}, m, _0{})) < seqlenk_row_limit;
            Element const* k_ptr = reinterpret_cast<Element const*>(__shfl_sync(0xffffffff, reinterpret_cast<uint64_t>(tPrKPtr(m / kGmemThreadsPerRow)), (m % kGmemThreadsPerRow), kGmemThreadsPerRow));
            Tensor mK_paged_cur = make_tensor(make_gmem_ptr(k_ptr), Shape<Int<kHeadDim>>{});
            Tensor mK_paged_cur_copy = cute::tiled_divide(mK_paged_cur, Shape<Int<kGmemElemsPerLoad>>{});
            if (should_load) {
                #pragma unroll
                for (int k = 0; k < size<2>(tKsK); ++k) {
                    int const ki = get<1>(tKcK(_0{}, _0{}, k)) / kGmemElemsPerLoad;
                    cute::copy(gmem_tiled_copy_kv.with(tKpK(_0{}, k)), mK_paged_cur_copy(_, ki), tKsK(_, m, k));
                }
            }
        }
    };
    template <bool Seqlenk_mask=false, typename TensorV>
    CUTLASS_DEVICE
    void load_V(const int n_block, TensorV &&sV) {
        static constexpr bool EvenN = kBlockN % CUTE_STATIC_V(shape<0>(GmemLayoutAtomKVCpAsync{})) == 0;
        if constexpr (KV_Same_Iter) { compute_V_ptr(); }
        auto gmem_thr0_copy_kv = gmem_tiled_copy_kv.get_thread_slice(_0{});
        Tensor tVsV = gmem_thr_copy_kv.partition_D(sV);
        Tensor cV = cute::make_identity_tensor(Shape<Int<kBlockN>, Int<kHeadDimV>>{});
        Tensor tVcV = gmem_thr_copy_kv.partition_S(cV);
        Tensor t0VcV = gmem_thr0_copy_kv.partition_S(cV);
        int const seqlenk_row_limit = seqlen_k - n_block * kBlockN - get<0>(tVcV(_0{}, _0{}, _0{}));
        #pragma unroll
        for (int m = 0; m < size<1>(tVsV); ++m) {
            if (EvenN || m < size<1>(tVsV) - 1 || get<0>(tVcV(_0{}, m, _0{})) < kBlockN) {
                bool const should_load = !Seqlenk_mask || get<0>(t0VcV(_0{}, m, _0{})) < seqlenk_row_limit;
                Element const* v_ptr = reinterpret_cast<Element const*>(__shfl_sync(0xffffffff, reinterpret_cast<uint64_t>(tPrVPtr(m / kGmemThreadsPerRow)), m % kGmemThreadsPerRow, kGmemThreadsPerRow));
                Tensor mV_paged_cur = make_tensor(make_gmem_ptr(v_ptr), Shape<Int<kHeadDimV>>{});
                Tensor mV_paged_cur_copy = cute::tiled_divide(mV_paged_cur, Shape<Int<kGmemElemsPerLoad>>{});
                #pragma unroll
                for (int k = 0; k < size<2>(tVsV); ++k) {
                    int const ki = get<1>(tVcV(_0{}, _0{}, k)) / kGmemElemsPerLoad;
                    cute::copy(gmem_tiled_copy_kv.with(tVpV(_0{}, k) && should_load), mV_paged_cur_copy(_, ki), tVsV(_, m, k));
                }
            }
        }
        if constexpr (!KV_Same_Iter) { compute_V_ptr(); }
    };
    template <typename TensorK>
    CUTLASS_DEVICE
    void store_K(const int n_block, TensorK &&tKrK) {
        Tensor tPrKPtr = compute_K_ptr();
        auto gmem_thr0_copy_kv = gmem_tiled_copy_kv.get_thread_slice(_0{});
        Tensor cK = cute::make_identity_tensor(Shape<Int<kBlockN>, Int<kHeadDim>>{});
        Tensor tKcK = gmem_thr_copy_kv.partition_S(cK);
        Tensor t0KcK = gmem_thr0_copy_kv.partition_S(cK);
        GmemTiledCopyKVStore gmem_tiled_copy_kv_store;
        int const seqlenk_row_limit = std::min(seqlen_k - n_block * kBlockN, kBlockN) - get<0>(tKcK(_0{}, _0{}, _0{}));
        #pragma unroll
        for (int m = 0; m < size<1>(tKrK); ++m) {
            bool const should_load = get<0>(t0KcK(_0{}, m, _0{})) < seqlenk_row_limit;
            Element* k_ptr = reinterpret_cast<Element*>(__shfl_sync(0xffffffff, reinterpret_cast<uint64_t>(tPrKPtr(m / kGmemThreadsPerRow)), (m % kGmemThreadsPerRow), kGmemThreadsPerRow));
            Tensor mK_paged_cur = make_tensor(make_gmem_ptr(k_ptr), Shape<Int<kHeadDim>>{});
            Tensor mK_paged_cur_copy = cute::tiled_divide(mK_paged_cur, Shape<Int<kGmemElemsPerLoad>>{});
            if (should_load) {
                #pragma unroll
                for (int k = 0; k < size<2>(tKrK); ++k) {
                    int const ki = get<1>(tKcK(_0{}, _0{}, k)) / kGmemElemsPerLoad;
                    if (tKpK(_0{}, k)) {
                        cute::copy(gmem_tiled_copy_kv_store, tKrK(_, m, k), mK_paged_cur_copy(_, ki));
                    }
                }
            }
        }
    };
    template <typename TensorV>
    CUTLASS_DEVICE
    void store_V(const int n_block, TensorV &&tVrV) {
        if constexpr (KV_Same_Iter) { compute_V_ptr(); }
        auto gmem_thr0_copy_kv = gmem_tiled_copy_kv.get_thread_slice(_0{});
        Tensor cV = cute::make_identity_tensor(Shape<Int<kBlockN>, Int<kHeadDimV>>{});
        Tensor tVcV = gmem_thr_copy_kv.partition_S(cV);
        Tensor t0VcV = gmem_thr0_copy_kv.partition_S(cV);
        GmemTiledCopyKVStore gmem_tiled_copy_kv_store;
        int const seqlenk_row_limit = std::min(seqlen_k - n_block * kBlockN, kBlockN) - get<0>(tVcV(_0{}, _0{}, _0{}));
        #pragma unroll
        for (int m = 0; m < size<1>(tVrV); ++m) {
            bool const should_load = get<0>(t0VcV(_0{}, m, _0{})) < seqlenk_row_limit;
            Element* v_ptr = reinterpret_cast<Element*>(__shfl_sync(0xffffffff, reinterpret_cast<uint64_t>(tPrVPtr(m / kGmemThreadsPerRow)), m % kGmemThreadsPerRow, kGmemThreadsPerRow));
            Tensor mV_paged_cur = make_tensor(make_gmem_ptr(v_ptr), Shape<Int<kHeadDimV>>{});
            Tensor mV_paged_cur_copy = cute::tiled_divide(mV_paged_cur, Shape<Int<kGmemElemsPerLoad>>{});
            if (should_load) {
                #pragma unroll
                for (int k = 0; k < size<2>(tVrV); ++k) {
                    int const ki = get<1>(tVcV(_0{}, _0{}, k)) / kGmemElemsPerLoad;
                    if (tVpV(_0{}, k)) {
                        cute::copy(gmem_tiled_copy_kv_store, tVrV(_, m, k), mV_paged_cur_copy(_, ki));
                    }
                }
            }
        }
        if constexpr (!KV_Same_Iter) { compute_V_ptr(); }
    };
};
}
#include <cute/tensor.hpp>
namespace flash {
using namespace cute;
template <typename Engine1, typename Layout1, typename Engine2, typename Layout2>
CUTLASS_DEVICE void
apply_rotary_interleaved(Tensor<Engine1, Layout1> &rK,
                         Tensor<Engine2, Layout2> const &rCos,
                         Tensor<Engine2, Layout2> const &rSin) {
    CUTE_STATIC_ASSERT_V(rank(rK) == _1{});
    CUTE_STATIC_ASSERT_V(rank(rCos) == _1{});
    CUTE_STATIC_ASSERT_V(rank(rSin) == _1{});
    CUTE_STATIC_ASSERT_V(size<0>(rCos) == size<0>(rSin));
    static_assert(decltype(size<0>(rK))::value == decltype(size<0>(rCos))::value * 2);
    static_assert(decltype(size<0>(rCos))::value % 2 == 0);
    Tensor K_fp32 = make_tensor_like<float>(rK);
    convert_type_out(rK, K_fp32);
    Tensor cos_fp32 = make_tensor_like<float>(rCos);
    convert_type_out(rCos, cos_fp32);
    Tensor sin_fp32 = make_tensor_like<float>(rSin);
    convert_type_out(rSin, sin_fp32);
    #pragma unroll
    for (int i = 0; i < size<0>(K_fp32) / 2; ++i) {
        float real = K_fp32[2 * i] * cos_fp32[i] - K_fp32[2 * i + 1] * sin_fp32[i];
        float imag = K_fp32[2 * i] * sin_fp32[i] + K_fp32[2 * i + 1] * cos_fp32[i];
        K_fp32[2 * i] = real;
        K_fp32[2 * i + 1] = imag;
    }
    convert_type_out(K_fp32, rK);
}
template <typename Engine1, typename Layout1, typename Engine2, typename Layout2>
CUTLASS_DEVICE void
apply_rotary_contiguous(Tensor<Engine1, Layout1> &rK_left,
                        Tensor<Engine1, Layout1> &rK_right,
                        Tensor<Engine2, Layout2> const &rCos,
                        Tensor<Engine2, Layout2> const &rSin) {
    CUTE_STATIC_ASSERT_V(rank(rK_left) == _1{});
    CUTE_STATIC_ASSERT_V(rank(rK_right) == _1{});
    CUTE_STATIC_ASSERT_V(rank(rCos) == _1{});
    CUTE_STATIC_ASSERT_V(rank(rSin) == _1{});
    CUTE_STATIC_ASSERT_V(size<0>(rK_left) == size<0>(rK_right));
    CUTE_STATIC_ASSERT_V(size<0>(rK_left) == size<0>(rCos));
    CUTE_STATIC_ASSERT_V(size<0>(rCos) == size<0>(rSin));
    static_assert(decltype(size<0>(rCos))::value % 2 == 0);
    Tensor K_left_fp32 = make_tensor_like<float>(rK_left);
    convert_type_out(rK_left, K_left_fp32);
    Tensor K_right_fp32 = make_tensor_like<float>(rK_right);
    convert_type_out(rK_right, K_right_fp32);
    Tensor cos_fp32 = make_tensor_like<float>(rCos);
    convert_type_out(rCos, cos_fp32);
    Tensor sin_fp32 = make_tensor_like<float>(rSin);
    convert_type_out(rSin, sin_fp32);
    #pragma unroll
    for (int i = 0; i < size<0>(K_left_fp32); ++i) {
        float real = K_left_fp32[i] * cos_fp32[i] - K_right_fp32[i] * sin_fp32[i];
        float imag = K_left_fp32[i] * sin_fp32[i] + K_right_fp32[i] * cos_fp32[i];
        K_left_fp32[i] = real;
        K_right_fp32[i] = imag;
    }
    convert_type_out(K_left_fp32, rK_left);
    convert_type_out(K_right_fp32, rK_right);
}
template <int kBlockMN, int kHeadDim, int NumThreads, typename Element, bool FixedPosition=false>
struct Rotary {
    static constexpr int kGmemElemsPerLoad = sizeof(cute::uint128_t) / sizeof(Element);
    static_assert(kHeadDim % kGmemElemsPerLoad == 0, "Headdim must be a multiple of kGmemElemsPerLoad");
    static constexpr int kBytePerHalfRow = kHeadDim / 2 * sizeof(Element);
    static constexpr int kBlockKGmem = (kBytePerHalfRow % 128 == 0 ? 128 : (kBytePerHalfRow % 64 == 0 ? 64 : 32)) / sizeof(Element);
    static constexpr int kGmemThreadsPerRow = kBlockKGmem / kGmemElemsPerLoad;
    static_assert(NumThreads % kGmemThreadsPerRow == 0, "NumThreads must be a multiple of kGmemThreadsPerRow");
    static_assert(cutlass::NumThreadsPerWarp % kGmemThreadsPerRow == 0, "kGmemThreadsPerRow must divide NumThreadsPerWarp");
    using LayoutAtom = Layout<Shape <Int<NumThreads / kGmemThreadsPerRow>, Int<kGmemThreadsPerRow>>,
                                  Stride<Int<kGmemThreadsPerRow>, _1>>;
    using TiledCopyQK = decltype(
        make_tiled_copy(Copy_Atom<AutoVectorizingCopyWithAssumedAlignment<128>, Element>{},
                        LayoutAtom{},
                        Layout<Shape<_1, Int<kGmemElemsPerLoad>>>{}));
    using GmemTiledCopyRotary = decltype(
        make_tiled_copy(Copy_Atom<AutoVectorizingCopyWithAssumedAlignment<64>, Element>{},
                        LayoutAtom{},
                        Layout<Shape<_1, Int<kGmemElemsPerLoad / 2>>>{}));
    using GmemTiledCopyRotaryCont = decltype(
        make_tiled_copy(Copy_Atom<AutoVectorizingCopyWithAssumedAlignment<128>, Element>{},
                        LayoutAtom{},
                        Layout<Shape<_1, Int<kGmemElemsPerLoad>>>{}));
    using ShapeRotary = cute::Shape<int32_t, int32_t>;
    using StrideRotary = cute::Stride<int64_t, _1>;
    using GmemThrCopyRotary = decltype(GmemTiledCopyRotary{}.get_thread_slice(int(0)));
    using GmemThrCopyRotaryCont = decltype(GmemTiledCopyRotaryCont{}.get_thread_slice(int(0)));
    using TensortRcR = decltype(GmemTiledCopyRotary{}.get_thread_slice(int(0)).partition_D(cute::make_identity_tensor(Shape<Int<kBlockMN>, Int<kHeadDim / 2>>{})));
    using TensortRpR = decltype(make_tensor<bool>(make_shape(size<2>(TensortRcR{}))));
    using TensortRcRCont = decltype(GmemTiledCopyRotaryCont{}.get_thread_slice(int(0)).partition_D(cute::make_identity_tensor(Shape<Int<kBlockMN>, Int<kHeadDim / 2>>{})));
    using TensortRpRCont = decltype(make_tensor<bool>(make_shape(size<2>(TensortRcRCont{}))));
    using TensormR = decltype(make_tensor(
        make_gmem_ptr((Element const*)nullptr),
        ShapeRotary{},
        make_stride(cute::conditional_return<FixedPosition>(_0{}, int64_t(0)), _1{})));
    using TensortRgR = decltype(
        GmemTiledCopyRotary{}.get_thread_slice(int(0)).partition_S(make_tensor(
            make_gmem_ptr((Element const*)nullptr),
            make_shape(Int<kBlockMN>{}, Int<kHeadDim / 2>{}, int(0)),
            make_stride(cute::conditional_return<FixedPosition>(_0{}, int64_t(0)), _1{}, cute::conditional_return<FixedPosition>(_0{}, int64_t(0))))));
    using TensortRgRCont = decltype(
        GmemTiledCopyRotaryCont{}.get_thread_slice(int(0)).partition_S(make_tensor(
            make_gmem_ptr((Element const*)nullptr),
            make_shape(Int<kBlockMN>{}, Int<kHeadDim / 2>{}, int(0)),
            make_stride(cute::conditional_return<FixedPosition>(_0{}, int64_t(0)), _1{}, cute::conditional_return<FixedPosition>(_0{}, int64_t(0))))));
    GmemTiledCopyRotary gmem_tiled_copy_rotary;
    GmemTiledCopyRotaryCont gmem_tiled_copy_rotary_cont;
    bool const is_rotary_interleaved;
    int const rotary_dim;
    int const thread_idx;
    int const max_seqlen;
    GmemThrCopyRotary const gmem_thr_copy_rotary;
    GmemThrCopyRotaryCont const gmem_thr_copy_rotary_cont;
    TensortRpR tRpR;
    TensortRpRCont tRpRCont;
    TensormR mCos, mSin;
    TensortRgR tRgCos, tRgSin;
    TensortRgRCont tRgCosCont, tRgSinCont;
    CUTLASS_DEVICE
    Rotary(Element const* const ptr_rotary_cos, ShapeRotary const &shape_rotary, StrideRotary const &stride_rotary_cos_,
           Element const* const ptr_rotary_sin, StrideRotary const &stride_rotary_sin_,
           bool const is_rotary_interleaved, int const thread_idx, int const max_seqlen, int const start_idx)
        : is_rotary_interleaved(is_rotary_interleaved)
        , rotary_dim(get<1>(shape_rotary) * 2)
        , thread_idx(thread_idx)
        , max_seqlen(max_seqlen)
        , gmem_thr_copy_rotary(gmem_tiled_copy_rotary.get_thread_slice(thread_idx))
        , gmem_thr_copy_rotary_cont(gmem_tiled_copy_rotary_cont.get_thread_slice(thread_idx))
    {
        auto stride_rotary_cos = make_stride(cute::conditional_return<!FixedPosition>(get<0>(stride_rotary_cos_), _0{}), get<1>(stride_rotary_cos_));
        auto stride_rotary_sin = make_stride(cute::conditional_return<!FixedPosition>(get<0>(stride_rotary_sin_), _0{}), get<1>(stride_rotary_sin_));
        mCos = make_tensor(make_gmem_ptr(ptr_rotary_cos + start_idx * get<0>(stride_rotary_cos_)), shape_rotary, stride_rotary_cos);
        mSin = make_tensor(make_gmem_ptr(ptr_rotary_sin + start_idx * get<0>(stride_rotary_sin_)), shape_rotary, stride_rotary_sin);
        Tensor gCos = local_tile(mCos, Shape<Int<kBlockMN>, Int<kHeadDim / 2>>{}, make_coord(_, _0{}));
        Tensor gSin = local_tile(mSin, Shape<Int<kBlockMN>, Int<kHeadDim / 2>>{}, make_coord(_, _0{}));
        tRgCos = gmem_thr_copy_rotary.partition_S(gCos);
        tRgSin = gmem_thr_copy_rotary.partition_S(gSin);
        tRgCosCont = gmem_thr_copy_rotary_cont.partition_S(gCos);
        tRgSinCont = gmem_thr_copy_rotary_cont.partition_S(gSin);
        Tensor cR = cute::make_identity_tensor(Shape<Int<kBlockMN>, Int<kHeadDim / 2>>{});
        Tensor tRcR = gmem_thr_copy_rotary.partition_D(cR);
        tRpR = make_tensor<bool>(make_shape(size<2>(tRcR)));
        #pragma unroll
        for (int k = 0; k < size(tRpR); ++k) { tRpR(k) = get<1>(tRcR(_0{}, _0{}, k)) < get<1>(shape_rotary); }
        Tensor tRcRCont = gmem_thr_copy_rotary_cont.partition_D(cR);
        tRpRCont = make_tensor<bool>(make_shape(size<2>(tRcRCont)));
        #pragma unroll
        for (int k = 0; k < size(tRpRCont); ++k) { tRpRCont(k) = get<1>(tRcRCont(_0{}, _0{}, k)) < get<1>(shape_rotary); }
    };
    template <bool kInterleaved=true>
    CUTLASS_DEVICE
    auto load_cos_sin(int const block) {
        using GmemTiledCopyRo = std::conditional_t<kInterleaved, GmemTiledCopyRotary, GmemTiledCopyRotaryCont>;
        auto gmem_thr_copy_ro = cute::conditional_return<kInterleaved>(gmem_thr_copy_rotary, gmem_thr_copy_rotary_cont);
        Tensor tRpRCur = cute::conditional_return<kInterleaved>(tRpR, tRpRCont);
        Tensor tRgCosCur = cute::conditional_return<kInterleaved>(tRgCos, tRgCosCont)(_, _, _, block);
        Tensor tRgSinCur = cute::conditional_return<kInterleaved>(tRgSin, tRgSinCont)(_, _, _, block);
        Tensor tRrCos = make_tensor_like(tRgCosCur);
        Tensor tRrSin = make_tensor_like(tRgSinCur);
        Tensor cR = cute::make_identity_tensor(Shape<Int<kBlockMN>, Int<kHeadDim / 2>>{});
        Tensor tRcR = gmem_thr_copy_ro.partition_D(cR);
        #pragma unroll
        for (int m = 0; m < (!FixedPosition ? size<1>(tRrCos) : 1); ++m) {
            if (get<0>(tRcR(_0{}, m, _0{})) < std::min(max_seqlen - block * kBlockMN, kBlockMN)) {
                #pragma unroll
                for (int k = 0; k < size<2>(tRrCos); ++k) {
                    if (tRpRCur(k)) {
                        cute::copy(GmemTiledCopyRo{}, tRgCosCur(_, m, k), tRrCos(_, m, k));
                        cute::copy(GmemTiledCopyRo{}, tRgSinCur(_, m, k), tRrSin(_, m, k));
                    }
                }
            }
        }
        return cute::make_tuple(tRrCos, tRrSin);;
    }
    template <bool kInterleaved=true>
    CUTLASS_DEVICE
    auto load_cos_sin_packgqa(int const block, cutlass::FastDivmod const &qhead_per_khead_divmod) {
        static constexpr int kGmemElemsPerLoadCur = kInterleaved ? kGmemElemsPerLoad / 2 : kGmemElemsPerLoad;
        using GmemTiledCopyRo = std::conditional_t<kInterleaved, GmemTiledCopyRotary, GmemTiledCopyRotaryCont>;
        auto gmem_thr_copy_ro = cute::conditional_return<kInterleaved>(gmem_thr_copy_rotary, gmem_thr_copy_rotary_cont);
        Tensor tRpRCur = cute::conditional_return<kInterleaved>(tRpR, tRpRCont);
        Tensor tRrCos = make_tensor_like(cute::conditional_return<kInterleaved>(tRgCos, tRgCosCont)(_, _, _, _0{}));
        Tensor tRrSin = make_tensor_like(cute::conditional_return<kInterleaved>(tRgSin, tRgSinCont)(_, _, _, _0{}));
        int const qhead_per_khead = qhead_per_khead_divmod.divisor;
        Tensor cR = cute::make_identity_tensor(Shape<Int<kBlockMN>, Int<kHeadDim / 2>>{});
        Tensor tRcR = gmem_thr_copy_ro.partition_D(cR);
        static constexpr int NumPtrPerThread = cute::ceil_div(CUTE_STATIC_V(cute::size<1>(tRrCos)), kGmemThreadsPerRow);
        Tensor tPrCosPtr = make_tensor<Element const*>(Shape<Int<NumPtrPerThread>>{});
        Tensor tPrSinPtr = make_tensor<Element const*>(Shape<Int<NumPtrPerThread>>{});
        #pragma unroll
        for (int i = 0; i < NumPtrPerThread; ++i) {
            int const row = i * NumThreads + get<0>(tRcR(_0{}, thread_idx % kGmemThreadsPerRow, _0{}));
            int const idx = block * kBlockMN + row;
            int row_actual = qhead_per_khead_divmod.divide(idx);
            tPrCosPtr[i] = &mCos(row_actual, _0{});
            tPrSinPtr[i] = &mSin(row_actual, _0{});
        }
        #pragma unroll
        for (int m = 0; m < (!FixedPosition ? size<1>(tRgCos) : 1); ++m) {
            int const idx = block * kBlockMN + get<0>(tRcR(_0{}, m, _0{}));
            Element const* cos_ptr = reinterpret_cast<Element const*>(__shfl_sync(0xffffffff, reinterpret_cast<uint64_t>(tPrCosPtr(m / kGmemThreadsPerRow)), m % kGmemThreadsPerRow, kGmemThreadsPerRow));
            Element const* sin_ptr = reinterpret_cast<Element const*>(__shfl_sync(0xffffffff, reinterpret_cast<uint64_t>(tPrSinPtr(m / kGmemThreadsPerRow)), m % kGmemThreadsPerRow, kGmemThreadsPerRow));
            if (idx < max_seqlen * qhead_per_khead) {
                Tensor mCos_copy = cute::tiled_divide(make_tensor(make_gmem_ptr(cos_ptr), Shape<Int<kHeadDim / 2>>{}),
                                                    Shape<Int<kGmemElemsPerLoadCur>>{});
                Tensor mSin_copy = cute::tiled_divide(make_tensor(make_gmem_ptr(sin_ptr), Shape<Int<kHeadDim / 2>>{}),
                                                    Shape<Int<kGmemElemsPerLoadCur>>{});
                #pragma unroll
                for (int k = 0; k < size<2>(tRgCos); ++k) {
                    int const ki = get<1>(tRcR(_0{}, _0{}, k)) / (kGmemElemsPerLoadCur);
                    if (tRpRCur(k)) {
                        cute::copy(GmemTiledCopyRo{}, mCos_copy(_, ki), tRrCos(_, m, k));
                        cute::copy(GmemTiledCopyRo{}, mSin_copy(_, ki), tRrSin(_, m, k));
                    }
                }
            }
        }
        return cute::make_tuple(tRrCos, tRrSin);
    }
    template <typename TensorsQ, typename TensortRrR>
    CUTLASS_DEVICE
    void
    apply_Q_interleaved(TensorsQ &sQ,
                        TensortRrR const &tRrCos,
                        TensortRrR const &tRrSin,
                        int const m_block, int const qhead_per_khead=1)
    {
        TiledCopyQK tiled_copy_q;
        auto gmem_thr_copy_q = tiled_copy_q.get_thread_slice(thread_idx);
        Tensor tQsQ = gmem_thr_copy_q.partition_S(sQ);
        Tensor tQcQ = gmem_thr_copy_q.partition_S(cute::make_identity_tensor(Shape<Int<kBlockMN>, Int<kHeadDim>>{}));
        CUTE_STATIC_ASSERT_V(rank(tQsQ) == _3{});
        CUTE_STATIC_ASSERT_V(rank(tRrCos) == _3{});
        CUTE_STATIC_ASSERT_V(rank(tRrSin) == _3{});
        CUTE_STATIC_ASSERT_V(size<1>(tQsQ) == size<1>(tRrCos));
        CUTE_STATIC_ASSERT_V(size<2>(tQsQ) == size<2>(tRrCos));
        CUTE_STATIC_ASSERT_V(size<1>(tQsQ) == size<1>(tRrSin));
        CUTE_STATIC_ASSERT_V(size<2>(tQsQ) == size<2>(tRrSin));
        CUTE_STATIC_ASSERT_V(size<0>(tRrCos) == size<0>(tRrSin));
        static_assert(decltype(size<0>(tQsQ))::value == decltype(size<0>(tRrCos))::value * 2);
        static_assert(decltype(size<0>(tRrCos))::value % 2 == 0);
        #pragma unroll
        for (int m = 0; m < size<1>(tQsQ); ++m) {
            if (get<0>(tQcQ(_0{}, m, _0{})) < std::min(max_seqlen * qhead_per_khead - m_block * kBlockMN, kBlockMN)) {
                #pragma unroll
                for (int k = 0; k < size<2>(tQsQ); ++k) {
                    if (tRpR(k)) {
                        Tensor rQ = make_fragment_like(tQsQ(_, m, k));
                        cute::copy(tiled_copy_q, tQsQ(_, m, k), rQ);
                        apply_rotary_interleaved(rQ, tRrCos(_, m, k), tRrSin(_, m, k));
                        cute::copy(tiled_copy_q, rQ, tQsQ(_, m, k));
                    }
                }
            }
        }
    };
    template <typename TensorsQ, typename TensortRrR>
    CUTLASS_DEVICE
    void
    apply_Q_contiguous(TensorsQ &sQ,
                       TensortRrR const &tRrCosCont,
                       TensortRrR const &tRrSinCont,
                       int const m_block, int const qhead_per_khead=1)
    {
        TiledCopyQK tiled_copy_q;
        auto gmem_thr_copy_q = tiled_copy_q.get_thread_slice(thread_idx);
        Tensor sQ_copy = cute::tiled_divide(sQ, Shape<_1, Int<kGmemElemsPerLoad>>{});
        Tensor tQcQ = gmem_thr_copy_q.partition_S(cute::make_identity_tensor(Shape<Int<kBlockMN>, Int<kHeadDim / 2>>{}));
        CUTE_STATIC_ASSERT_V(rank(tQcQ) == _3{});
        CUTE_STATIC_ASSERT_V(rank(tRrCosCont) == _3{});
        CUTE_STATIC_ASSERT_V(rank(tRrSinCont) == _3{});
        CUTE_STATIC_ASSERT_V(size<1>(tQcQ) == size<1>(tRrCosCont));
        CUTE_STATIC_ASSERT_V(size<2>(tQcQ) == size<2>(tRrCosCont));
        CUTE_STATIC_ASSERT_V(size<1>(tQcQ) == size<1>(tRrSinCont));
        CUTE_STATIC_ASSERT_V(size<2>(tQcQ) == size<2>(tRrSinCont));
        CUTE_STATIC_ASSERT_V(size<0>(tRrCosCont) == size<0>(tRrSinCont));
        CUTE_STATIC_ASSERT_V(size<0>(tQcQ) == size<0>(tRrCosCont));
        static_assert(decltype(size<0>(tRrCosCont))::value % 2 == 0);
        #pragma unroll
        for (int m = 0; m < size<1>(tQcQ); ++m) {
            int const row = get<0>(tQcQ(_0{}, m, _0{}));
            if (row < std::min(max_seqlen * qhead_per_khead - m_block * kBlockMN, kBlockMN)) {
                #pragma unroll
                for (int k = 0; k < size<2>(tQcQ); ++k) {
                    int const col = get<1>(tQcQ(_0{}, _0{}, k));
                    if (col < rotary_dim / 2) {
                        int const col_idx_left = col / kGmemElemsPerLoad;
                        int const col_idx_right = col / kGmemElemsPerLoad + rotary_dim / (2 * kGmemElemsPerLoad);
                        Tensor rQ_left = make_fragment_like(sQ_copy(_, row, col_idx_left));
                        cute::copy(tiled_copy_q, sQ_copy(_, row, col_idx_left), rQ_left);
                        Tensor rQ_right = make_fragment_like(rQ_left);
                        cute::copy(tiled_copy_q, sQ_copy(_, row, col_idx_right), rQ_right);
                        apply_rotary_contiguous(rQ_left, rQ_right, tRrCosCont(_, m, k), tRrSinCont(_, m, k));
                        cute::copy(tiled_copy_q, rQ_left, sQ_copy(_, row, col_idx_left));
                        cute::copy(tiled_copy_q, rQ_right, sQ_copy(_, row, col_idx_right));
                    }
                }
            }
        }
    };
    template <bool PagedKVNonTMA=false, typename TensorsK, typename TensorgK, typename TensorpK, typename TensortRrR, typename TensorKPtr>
    CUTLASS_DEVICE
    void
    apply_K_interleaved(TensorsK const &sK,
                        TensorgK &gK,
                        TensorpK const &tKpK,
                        TensortRrR const &tRrCos,
                        TensortRrR const &tRrSin,
                        TensorKPtr const &tPrKPtr,
                        int const n_block)
    {
        TiledCopyQK tiled_copy_k;
        auto gmem_thr_copy_q = tiled_copy_k.get_thread_slice(thread_idx);
        Tensor tKsK = gmem_thr_copy_q.partition_S(sK);
        Tensor tKgK = gmem_thr_copy_q.partition_S(gK);
        Tensor tKcK = gmem_thr_copy_q.partition_S(cute::make_identity_tensor(Shape<Int<kBlockMN>, Int<kHeadDim>>{}));
        CUTE_STATIC_ASSERT_V(rank(tKsK) == _3{});
        CUTE_STATIC_ASSERT_V(rank(tRrCos) == _3{});
        CUTE_STATIC_ASSERT_V(rank(tRrSin) == _3{});
        CUTE_STATIC_ASSERT_V(size<1>(tKsK) == size<1>(tRrCos));
        CUTE_STATIC_ASSERT_V(size<2>(tKsK) == size<2>(tRrCos));
        CUTE_STATIC_ASSERT_V(size<1>(tKsK) == size<1>(tRrSin));
        CUTE_STATIC_ASSERT_V(size<2>(tKsK) == size<2>(tRrSin));
        CUTE_STATIC_ASSERT_V(size<0>(tRrCos) == size<0>(tRrSin));
        static_assert(decltype(size<0>(tKsK))::value == decltype(size<0>(tRrCos))::value * 2);
        static_assert(decltype(size<0>(tRrCos))::value % 2 == 0);
        if constexpr (PagedKVNonTMA) {
            static_assert(decltype(size(tPrKPtr))::value == cute::ceil_div(size<1>(tKcK), kGmemThreadsPerRow));
        }
        #pragma unroll
        for (int m = 0; m < size<1>(tKsK); ++m) {
            int const row = get<0>(tKcK(_0{}, m, _0{}));
            auto mK_cur_copy = [&] {
                if constexpr (PagedKVNonTMA) {
                    Element* k_ptr = reinterpret_cast<Element*>(__shfl_sync(0xffffffff, reinterpret_cast<uint64_t>(tPrKPtr(m / kGmemThreadsPerRow)), (m % kGmemThreadsPerRow), kGmemThreadsPerRow));
                    Tensor mK_cur = make_tensor(make_gmem_ptr(k_ptr), Shape<Int<kHeadDim>>{});
                    return cute::tiled_divide(mK_cur, Shape<Int<kGmemElemsPerLoad>>{});
                } else {
                    return nullptr;
                }
            }();
            if (row < std::min(max_seqlen - n_block * kBlockMN, kBlockMN)) {
                #pragma unroll
                for (int k = 0; k < size<2>(tKsK); ++k) {
                    if (tKpK(k)) {
                        Tensor rK = make_fragment_like(tKsK(_, m, k));
                        cute::copy(tiled_copy_k, tKsK(_, m, k), rK);
                        if (tRpR(k)) { apply_rotary_interleaved(rK, tRrCos(_, m, k), tRrSin(_, m, k)); }
                        if constexpr (!PagedKVNonTMA) {
                            cute::copy(tiled_copy_k, rK, tKgK(_, m, k));
                        } else {
                            int const ki = get<1>(tKcK(_0{}, _0{}, k)) / kGmemElemsPerLoad;
                            cute::copy(tiled_copy_k, rK, mK_cur_copy(_, ki));
                        }
                    }
                }
            }
        }
    };
    template <bool PagedKVNonTMA=false, typename TensorsK, typename TensorgK, typename TensorpK, typename TensortRrR, typename TensorKPtr>
    CUTLASS_DEVICE
    void
    apply_K_contiguous(TensorsK const &sK,
                       TensorgK &gK,
                       TensorpK const &tKpK,
                       TensortRrR const &tRrCosCont,
                       TensortRrR const &tRrSinCont,
                       TensorKPtr const &tPrKPtr,
                       int const n_block, int const max_k)
    {
        TiledCopyQK tiled_copy_k;
        auto gmem_thr_copy_q = tiled_copy_k.get_thread_slice(thread_idx);
        Tensor sK_copy = cute::tiled_divide(sK, Shape<_1, Int<kGmemElemsPerLoad>>{});
        Tensor gK_copy = cute::tiled_divide(gK, Shape<_1, Int<kGmemElemsPerLoad>>{});
        Tensor tKcK = gmem_thr_copy_q.partition_S(cute::make_identity_tensor(Shape<Int<kBlockMN>, Int<kHeadDim / 2>>{}));
        CUTE_STATIC_ASSERT_V(rank(tKcK) == _3{});
        CUTE_STATIC_ASSERT_V(rank(tRrCosCont) == _3{});
        CUTE_STATIC_ASSERT_V(rank(tRrSinCont) == _3{});
        CUTE_STATIC_ASSERT_V(size<1>(tKcK) == size<1>(tRrCosCont));
        CUTE_STATIC_ASSERT_V(size<2>(tKcK) == size<2>(tRrCosCont));
        CUTE_STATIC_ASSERT_V(size<1>(tKcK) == size<1>(tRrSinCont));
        CUTE_STATIC_ASSERT_V(size<2>(tKcK) == size<2>(tRrSinCont));
        CUTE_STATIC_ASSERT_V(size<0>(tRrCosCont) == size<0>(tRrSinCont));
        CUTE_STATIC_ASSERT_V(size<0>(tKcK) == size<0>(tRrCosCont));
        static_assert(decltype(size<0>(tRrCosCont))::value % 2 == 0);
        if constexpr (PagedKVNonTMA) {
            static_assert(decltype(size(tPrKPtr))::value == cute::ceil_div(size<1>(tKcK), kGmemThreadsPerRow));
        }
        const int ro_dim_vec = rotary_dim / kGmemElemsPerLoad;
        const int non_ro_dim_vec = (max_k - rotary_dim) / kGmemElemsPerLoad;
        #pragma unroll
        for (int m = 0; m < size<1>(tKcK); ++m) {
            int const row = get<0>(tKcK(_0{}, m, _0{}));
            Tensor gK_cur_copy = [&] {
                if constexpr (PagedKVNonTMA) {
                    Element* k_ptr = reinterpret_cast<Element*>(__shfl_sync(0xffffffff, reinterpret_cast<uint64_t>(tPrKPtr(m / kGmemThreadsPerRow)), (m % kGmemThreadsPerRow), kGmemThreadsPerRow));
                    Tensor mK_cur = make_tensor(make_gmem_ptr(k_ptr), Shape<Int<kHeadDim>>{});
                    return cute::tiled_divide(mK_cur, Shape<Int<kGmemElemsPerLoad>>{});
                } else {
                    return gK_copy(_, row, _);
                }
            }();
            if (row < std::min(max_seqlen - n_block * kBlockMN, kBlockMN)) {
                #pragma unroll
                for (int k = 0; k < size<2>(tKcK); ++k) {
                    if (tKpK(k)) {
                        int const col = get<1>(tKcK(_0{}, _0{}, k));
                        bool rotate = col < rotary_dim / 2;
                        int const col_idx_left = rotate ? col / kGmemElemsPerLoad : (col + rotary_dim / 2) / kGmemElemsPerLoad;
                        int const col_idx_right = col_idx_left + (rotate ? ro_dim_vec / 2 : non_ro_dim_vec / 2);
                        Tensor rK_left = make_fragment_like(sK_copy(_, row, col_idx_left));
                        cute::copy(tiled_copy_k, sK_copy(_, row, col_idx_left), rK_left);
                        Tensor rK_right = make_fragment_like(rK_left);
                        cute::copy(tiled_copy_k, sK_copy(_, row, col_idx_right), rK_right);
                        if (rotate) {
                            apply_rotary_contiguous(rK_left, rK_right, tRrCosCont(_, m, k), tRrSinCont(_, m, k));
                        }
                        cute::copy(tiled_copy_k, rK_left, gK_cur_copy(_, col_idx_left));
                        if (col_idx_right * kGmemElemsPerLoad < max_k) {
                            cute::copy(tiled_copy_k, rK_right, gK_cur_copy(_, col_idx_right));
                        }
                    }
                }
            }
        }
    };
};
}
#include<cutlass/pipeline/sm90_pipeline.hpp>
namespace cutlass {
using namespace cute;
template <int Stages_, class Base=cutlass::PipelineTmaAsync<Stages_>>
class PipelineTmaAsyncNoCluster: public Base {
public:
  using FullBarrier = typename Base::FullBarrier;
  using EmptyBarrier = typename Base::EmptyBarrier;
  static constexpr uint32_t Stages = Stages_;
  using PipelineState = typename Base::PipelineState;
  using SharedStorage = typename Base::SharedStorage;
  using ThreadCategory = typename Base::ThreadCategory;
  using Params = typename Base::Params;
  static
  CUTLASS_DEVICE
  void
  init_barriers(SharedStorage& storage, Params params) {
    int warp_idx = canonical_warp_idx_sync();
    bool is_initializing_warp = (warp_idx == 0);
    if (is_initializing_warp) {
      constexpr int producer_arv_cnt = 1;
      uint32_t const num_consumer_warpgroups_per_cluster = (params.num_consumers + NumThreadsPerWarpGroup - 1) / NumThreadsPerWarpGroup;
      uint32_t const multicast_consumer_arrival_count = num_consumer_warpgroups_per_cluster;
      cutlass::arch::detail::initialize_barrier_array_pair_aligned<decltype(storage.full_barrier_), decltype(storage.empty_barrier_), Stages>(
          storage.full_barrier_, storage.empty_barrier_, producer_arv_cnt, multicast_consumer_arrival_count);
    }
    cutlass::arch::fence_barrier_init();
  }
  template<class ClusterShape, class InitBarriers, class InitMasks>
  CUTLASS_DEVICE
  PipelineTmaAsyncNoCluster(SharedStorage& storage, Params params, ClusterShape cluster_shape, InitBarriers = {}, InitMasks = {})
      : Base(storage, params, make_shape(_1{}, _1{}, _1{}) , cute::false_type{} , cute::false_type{} )
      , empty_barrier_ptr_(&storage.empty_barrier_[0]) {
    int warp_idx = canonical_warp_idx_sync();
    int lane_predicate = cute::elect_one_sync();
    static_assert(cute::is_same_v<InitBarriers, cute::true_type> || cute::is_same_v<InitBarriers, cute::false_type>);
    static_assert(cute::is_same_v<InitMasks, cute::true_type> || cute::is_same_v<InitMasks, cute::false_type>);
    if constexpr (cute::is_same_v<InitBarriers, cute::true_type>) {
      init_barriers(storage, params);
    }
  }
  template<class ClusterShape>
  CUTLASS_DEVICE
  PipelineTmaAsyncNoCluster(SharedStorage& storage, Params params, ClusterShape cluster_shape)
      : PipelineTmaAsyncNoCluster(storage, params, cluster_shape, cute::true_type{}, cute::true_type{}) { }
  template<class ClusterShape, class InitBarriers>
  CUTLASS_DEVICE
  PipelineTmaAsyncNoCluster(SharedStorage& storage, Params params, ClusterShape cluster_shape, InitBarriers = {})
      : PipelineTmaAsyncNoCluster(storage, params, cluster_shape, InitBarriers{}, cute::true_type{}) { }
  CUTLASS_DEVICE
  void consumer_release(PipelineState state) {
    consumer_release(state.index());
  }
private:
  EmptyBarrier* const empty_barrier_ptr_ = nullptr;
  CUTLASS_DEVICE
  void consumer_release(uint32_t stage, uint32_t skip = false) {
    if ((threadIdx.x % cutlass::NumThreadsPerWarpGroup == 0) && !skip) {
      uint32_t const a = cute::cast_smem_ptr_to_uint(&empty_barrier_ptr_[stage]);
      asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" :: "r"(a) : "memory");
    }
  }
};
}
#define SK2_ABL 0
#define ATT_CABL (((SK2_ABL & 1) ? 2 : 0) | ((SK2_ABL & 2) ? 1 : 0))
#define ATT_PABL 0
#define ATT_STAMP(cond, role, it, ev) {}
namespace flash {
using namespace cute;
// P V for one tile into the FP16 accumulator o[16]: one m64n64k32 f16.e4m3.e4m3 wgmma per 32 keys, V read from the
// 72-row image tile at vaddr (the first 64 rows are V^T, 128 keys per 72 x 128-byte half).
template <int kBlockN, typename TP>
CUTLASS_DEVICE void fk_pv16(TP const& tOrP, uint32_t (&o)[16], uint32_t vaddr) {
    Tensor a = recast<uint32_t const>(tOrP);
    constexpr int NK = CUTE_STATIC_V(size(a)) / 4;
    static_assert(NK == kBlockN / 32);
    constexpr int CW = kBlockN % 128 == 0 ? 128 : 64;
    constexpr int KPC = CW / 32;
    constexpr uint32_t LT = CW == 128 ? 1 : 2;
    uint32_t const lo0 = ((vaddr & 0x3FFFF) >> 4) | (1u << 16);
    uint32_t const hi = uint32_t(8 * CW >> 4) | uint32_t(LT << 30);
    warpgroup_fence_operand(const_cast<TP&>(tOrP));
    #pragma unroll
    for (int j = 0; j < 16; ++j) { asm volatile("" : "+r"(o[j]) :: "memory"); }
    warpgroup_arrive();
    #pragma unroll
    for (int k = 0; k < NK; ++k) {
        uint64_t desc;
        asm("mov.b64 %0, {%1, %2};" : "=l"(desc) : "r"(lo0 + uint32_t(((k / KPC) * (72 * CW) + (k % KPC) * 32) >> 4)), "r"(hi));
        asm volatile(
            "{\n.reg .pred p;\nsetp.ne.b32 p, 1, 0;\n"
            "wgmma.mma_async.sync.aligned.m64n64k32.f16.e4m3.e4m3 "
            "{%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, {%16, %17, %18, %19}, %20, p, 1, 1;\n}\n"
            : "+r"(o[0]), "+r"(o[1]), "+r"(o[2]), "+r"(o[3]), "+r"(o[4]), "+r"(o[5]), "+r"(o[6]), "+r"(o[7]),
              "+r"(o[8]), "+r"(o[9]), "+r"(o[10]), "+r"(o[11]), "+r"(o[12]), "+r"(o[13]), "+r"(o[14]), "+r"(o[15])
            : "r"(a(4 * k)), "r"(a(4 * k + 1)), "r"(a(4 * k + 2)), "r"(a(4 * k + 3)), "l"(desc));
    }
    warpgroup_commit_batch();
    #pragma unroll
    for (int j = 0; j < 16; ++j) { asm volatile("" : "+r"(o[j]) :: "memory"); }
    warpgroup_fence_operand(const_cast<TP&>(tOrP));
}
// Rescale the FP16 accumulator by the softmax step's per-row factors (rows r0 and r0 + 8 alternate).
template <typename TS>
CUTLASS_DEVICE void u_rescale16(uint32_t (&o)[16], TS const& sc) {
    __half2 const s0 = __float2half2_rn(sc(0)), s1 = __float2half2_rn(sc(1));
    #pragma unroll
    for (int j = 0; j < 16; ++j) {
        __half2 v = *reinterpret_cast<__half2*>(&o[j]);
        v = __hmul2(v, (j & 1) ? s1 : s0);
        o[j] = *reinterpret_cast<uint32_t*>(&v);
    }
}
CUTLASS_DEVICE void u_stat16(uint32_t const (&o)[16]) {
}
// Fold the FP16 accumulator into the FP32 output registers (a copy when first) and clear it.
template <typename TO>
CUTLASS_DEVICE void u_fold(TO& acc, uint32_t (&o)[16], float (&usc)[2], bool first) {
    u_stat16(o);
    #pragma unroll
    for (int j = 0; j < 16; ++j) {
        __half2 v = *reinterpret_cast<__half2*>(&o[j]);
        float const f = usc[j & 1];
        if (first) {
            acc(2 * j) = __low2float(v);
            acc(2 * j + 1) = __high2float(v);
        } else {
            acc(2 * j) = fmaf(acc(2 * j), f, __low2float(v));
            acc(2 * j + 1) = fmaf(acc(2 * j + 1), f, __high2float(v));
        }
        o[j] = 0u;
    }
    usc[0] = usc[1] = 1.f;
}
// FP32-accumulator version of fk_pv16; not called in this build.
template <int kBlockN, typename TP, typename TO>
CUTLASS_DEVICE void fk_pv72(TP const& tOrP, TO& o, float (&os)[4], uint32_t vaddr) {
    Tensor a = recast<uint32_t const>(tOrP);
    constexpr int NK = CUTE_STATIC_V(size(a)) / 4;
    static_assert(NK == kBlockN / 32);
    static_assert(CUTE_STATIC_V(size(o)) == 32);
    constexpr int CW = kBlockN % 128 == 0 ? 128 : 64;
    constexpr int KPC = CW / 32;
    constexpr uint64_t LT = CW == 128 ? 1 : 2;
    uint32_t const fx_lo0 = ((vaddr & 0x3FFFF) >> 4) | (1u << 16);
    uint32_t const fx_hi = uint32_t(8 * CW >> 4) | uint32_t(LT << 30);
    warpgroup_fence_operand(const_cast<TP&>(tOrP));
    warpgroup_fence_operand(o);
    #pragma unroll
    for (int i = 0; i < 4; ++i) { asm volatile("" : "+f"(os[i]) :: "memory"); }
    warpgroup_arrive();
    #pragma unroll
    for (int k = 0; k < NK; ++k) {
        uint32_t const ad = vaddr + (k / KPC) * (72 * CW) + (k % KPC) * 32;
        uint64_t desc;
        asm("mov.b64 %0, {%1, %2};" : "=l"(desc) : "r"(fx_lo0 + uint32_t(((k / KPC) * (72 * CW) + (k % KPC) * 32) >> 4)), "r"(fx_hi));
        (void)ad;
        asm volatile(
            "{\n.reg .pred p;\nsetp.ne.b32 p, 1, 0;\n"
            "wgmma.mma_async.sync.aligned.m64n64k32.f32.e4m3.e4m3 "
            "{%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, {%32, %33, %34, %35}, %36, p, 1, 1;\n}\n"
            : "+f"(o(0)), "+f"(o(1)), "+f"(o(2)), "+f"(o(3)), "+f"(o(4)), "+f"(o(5)), "+f"(o(6)), "+f"(o(7)), "+f"(o(8)), "+f"(o(9)), "+f"(o(10)), "+f"(o(11)), "+f"(o(12)), "+f"(o(13)), "+f"(o(14)), "+f"(o(15)), "+f"(o(16)), "+f"(o(17)), "+f"(o(18)), "+f"(o(19)), "+f"(o(20)), "+f"(o(21)), "+f"(o(22)), "+f"(o(23)), "+f"(o(24)), "+f"(o(25)), "+f"(o(26)), "+f"(o(27)), "+f"(o(28)), "+f"(o(29)), "+f"(o(30)), "+f"(o(31))
            : "r"(a(4 * k)), "r"(a(4 * k + 1)), "r"(a(4 * k + 2)), "r"(a(4 * k + 3)), "l"(desc));
    }
    warpgroup_commit_batch();
    warpgroup_fence_operand(o);
    #pragma unroll
    for (int i = 0; i < 4; ++i) { asm volatile("" : "+f"(os[i]) :: "memory"); }
    warpgroup_fence_operand(const_cast<TP&>(tOrP));
}
template <int Stages, class ClusterShape_, class TileShape_MNK_, int kHeadDimV, class Element_, class ElementAccum_, class ArchTag_,
        bool Is_causal_, bool Is_local_, bool Has_softcap_, bool Varlen_, bool PagedKVNonTMA_, bool AppendKV_, bool HasQv_, bool OnlyQv_,
        bool MmaPV_is_RS, bool IntraWGOverlap, bool PackGQA_, bool Split_, bool V_colmajor_, class ElementSink_, bool Has_sparse_mask_=false, int kBlockH_=1>
struct CollectiveMainloopFwdSm90 {
    static constexpr int kStages = Stages;
    using ClusterShape = ClusterShape_;
    using TileShape_MNK = TileShape_MNK_;
    using TileShape_MNK_PV = Shape<decltype(get<0>(TileShape_MNK{})), Int<kHeadDimV>, decltype(get<1>(TileShape_MNK{}))>;
    using TileShape_MNK_QV = Shape<decltype(get<0>(TileShape_MNK{})), decltype(get<1>(TileShape_MNK{})), Int<kHeadDimV>>;
    using Element = Element_;
    using ElementAccum = ElementAccum_;
    using ElementSink = ElementSink_;
    using ArchTag = ArchTag_;
    static constexpr bool Is_FP8 = cute::is_same_v<Element, cutlass::float_e4m3_t> || cute::is_same_v<Element, cutlass::float_e5m2_t>;;
    static constexpr bool Is_causal = Is_causal_;
    static constexpr bool Is_local = Is_local_;
    static constexpr bool Has_softcap = Has_softcap_;
    static constexpr bool Varlen = Varlen_;
    static constexpr bool PagedKVNonTMA = PagedKVNonTMA_;
    static constexpr bool AppendKV = AppendKV_;
    static constexpr bool HasQv = HasQv_;
    static constexpr bool OnlyQv = OnlyQv_;
    static constexpr bool PackGQA = PackGQA_;
    static constexpr bool PackGQA_TMA = PackGQA && kBlockH_ > 1;
    static constexpr bool Split = Split_;
    static constexpr bool V_colmajor = V_colmajor_;
    static constexpr bool Has_sparse_mask = Has_sparse_mask_;
    static constexpr bool Transpose_V = false;
    static_assert(Is_FP8 && !V_colmajor && !PagedKVNonTMA && !AppendKV && !HasQv);
    static constexpr bool Use_TMA_Q = !PackGQA || PackGQA_TMA;
    static constexpr bool Use_TMA_KV = !PagedKVNonTMA;
    static_assert(Use_TMA_KV || CUTE_STATIC_V(size(ClusterShape{})) == 1, "If not using TMA for KV, ClusterShape must be 1");
    static_assert(Use_TMA_KV || !V_colmajor, "If not using TMA for KV, V_colmajor is not supported");
    static constexpr bool SameHeadDim = get<2>(TileShape_MNK{}) == kHeadDimV;
    static constexpr bool LargeHeadDimV = kHeadDimV > 256;
    static_assert(!OnlyQv || HasQv, "OnlyQv requires HasQv");
    static_assert(ArchTag::kMinComputeCapability >= 90);
    static constexpr cute::GMMA::Major MmaMajorV = !Is_FP8 && !V_colmajor ? GMMA::Major::MN : GMMA::Major::K;
    static constexpr cute::GMMA::Major TmaMajorV = !V_colmajor ? GMMA::Major::MN : GMMA::Major::K;
    static constexpr int kBlockM = get<0>(TileShape_MNK{});
    static constexpr int kBlockN = get<1>(TileShape_MNK{});
    static constexpr int kHeadDim = get<2>(TileShape_MNK{});
    static constexpr int kBlockH = kBlockH_;
    static constexpr int kNumInt32PerBlock = (kBlockN + 31) / 32;
    using SeqlenInfo_t = flash::SeqlenInfoQKNewK<Varlen, AppendKV>;
    using BlockMN_t = flash::BlockMN<SeqlenInfo_t, kBlockM, kBlockN, Is_causal, Is_local, PackGQA, Split>;
    static_assert(!LargeHeadDimV || kHeadDimV % 256 == 0);
    static_assert(!LargeHeadDimV || kBlockM <= 64, "kBlockM must be 64 or less for large Headdim_V");
    static_assert(!LargeHeadDimV || !MmaPV_is_RS, "MmaPV must be SS for large Headdim_V");
    static constexpr bool MmaQK_is_RS = false;
    static_assert(!(!MmaPV_is_RS && Is_FP8), "MmaPV must be RS if FP8");
    static_assert(!(!MmaPV_is_RS && Transpose_V), "MmaPV must be RS if Transpose_V");
    static constexpr bool MmaPV_use_RS_WG1 = !MmaPV_is_RS && kHeadDim == 64 && kHeadDimV == 512;
    using AtomLayoutQK = Layout<Shape<Int<kBlockM / 64>, _1, _1>>;
    using ElementAccumQK = std::conditional_t<Is_FP8, cutlass::half_t, ElementAccum>;
    using TiledMmaQK = decltype(cute::make_tiled_mma(
        std::conditional_t<
            !MmaQK_is_RS,
            decltype(cute::GMMA::ss_op_selector<Element, Element, ElementAccumQK, TileShape_MNK>()),
            decltype(cute::GMMA::rs_op_selector<Element, Element, ElementAccumQK, TileShape_MNK>())
        >{},
        AtomLayoutQK{}));
    using AtomLayoutPV = std::conditional_t<
        !LargeHeadDimV,
        AtomLayoutQK,
        Layout<Shape<_1, Int<kHeadDimV / 256>, _1>>
    >;
    using TiledMmaPV = decltype(cute::make_tiled_mma(
        std::conditional_t<
            !MmaPV_is_RS,
            decltype(cute::GMMA::ss_op_selector<Element, Element, ElementAccum,
                     TileShape_MNK_PV, GMMA::Major::K, MmaMajorV>()),
            decltype(cute::GMMA::rs_op_selector<Element, Element, ElementAccum,
                     TileShape_MNK_PV, GMMA::Major::K, MmaMajorV>())
        >{},
        AtomLayoutPV{}));
    using TiledMmaQV = decltype(cute::make_tiled_mma(
        cute::GMMA::ss_op_selector<Element, Element, ElementAccum, TileShape_MNK_QV>(),
        AtomLayoutQK{}));
    using TiledMmaPV_RS = decltype(cute::make_tiled_mma(
        cute::GMMA::rs_op_selector<Element, Element, ElementAccum, TileShape_MNK_PV, GMMA::Major::K, MmaMajorV>(),
        AtomLayoutPV{}));
    static constexpr int NumMmaThreadsQK = size(TiledMmaQK{});
    static constexpr int NumMmaThreads = size(TiledMmaPV{});
    static constexpr int NumProducerThreads = !Transpose_V && Use_TMA_KV && Use_TMA_Q ? cutlass::NumThreadsPerWarp : cutlass::NumThreadsPerWarpGroup;
    static_assert(NumMmaThreadsQK % cutlass::NumThreadsPerWarpGroup == 0);
    static_assert(NumMmaThreads % cutlass::NumThreadsPerWarpGroup == 0);
    static constexpr int NumMmaWarpGroups = NumMmaThreads / cutlass::NumThreadsPerWarpGroup;
    static_assert(NumMmaWarpGroups == 1 || NumMmaWarpGroups == 2 || NumMmaWarpGroups == 3 || NumMmaWarpGroups == 4);
    using SmemLayoutAtomQ = decltype(cutlass::gemm::collective::detail::ss_smem_selector<GMMA::Major::K, Element,
        decltype(cute::get<0>(TileShape_MNK{})), decltype(cute::get<2>(TileShape_MNK{}))>());
    using SmemLayoutQ = decltype(tile_to_shape(SmemLayoutAtomQ{}, select<0, 2>(TileShape_MNK{})));
    using SmemLayoutAtomK = decltype(cutlass::gemm::collective::detail::ss_smem_selector<GMMA::Major::K, Element,
        decltype(cute::get<1>(TileShape_MNK{})), decltype(cute::get<2>(TileShape_MNK{}))>());
    using SmemLayoutK = decltype(tile_to_shape(
        SmemLayoutAtomK{},
        make_shape(shape<1>(TileShape_MNK{}), shape<2>(TileShape_MNK{}), Int<kStages>{})));
    using SmemLayoutAtomVt = decltype(cutlass::gemm::collective::detail::ss_smem_selector<TmaMajorV, Element,
                                      Int<kHeadDimV>, decltype(cute::get<2>(TileShape_MNK_PV{}))>());
    using SmemLayoutVt = decltype(tile_to_shape(
        SmemLayoutAtomVt{},
        make_shape(Int<kHeadDimV>{}, shape<2>(TileShape_MNK_PV{}), Int<kStages>{}),
        std::conditional_t<TmaMajorV == GMMA::Major::K, cute::Step<_1, _2, _3>, cute::Step<_2, _1, _3>>{}));
    using SmemLayoutAtomVtMma = decltype(cutlass::gemm::collective::detail::ss_smem_selector<MmaMajorV, Element,
                                         Int<kHeadDimV>, decltype(cute::get<2>(TileShape_MNK_PV{}))>());
    static_assert(kHeadDimV == 64 && MmaMajorV == GMMA::Major::K && kBlockN % 64 == 0, "FK_PV72: hdim 64, BN % 64");
    using SmemLayoutVtMma72 = decltype(tile_to_shape(
        SmemLayoutAtomVtMma{},
        make_shape(Int<72>{}, shape<2>(TileShape_MNK_PV{}), Int<kStages>{}),
        cute::Step<_1, _2, _3>{}));
    using SmemLayoutVtMma = decltype(composition(SmemLayoutVtMma72{},
        Layout<Shape<Int<64>, Int<kBlockN>, Int<kStages>>, Stride<_1, Int<72>, Int<72 * kBlockN>>>{}));
    static constexpr int kSmemVSize = cute::cosize_v<SmemLayoutVtMma72>;
    using SmemLayoutAtomQv = decltype(cutlass::gemm::collective::detail::ss_smem_selector<GMMA::Major::K, Element,
        decltype(cute::get<0>(TileShape_MNK_QV{})), decltype(cute::get<2>(TileShape_MNK_QV{}))>());
    using SmemLayoutQv = decltype(tile_to_shape(SmemLayoutAtomQv{}, select<0, 2>(TileShape_MNK_QV{})));
    using SmemLayoutAtomVMmaQV = decltype(cutlass::gemm::collective::detail::ss_smem_selector<GMMA::Major::K, Element,
        decltype(cute::get<1>(TileShape_MNK_QV{})), decltype(cute::get<2>(TileShape_MNK_QV{}))>());
    using SmemLayoutVMmaQV = decltype(tile_to_shape(
        SmemLayoutAtomVMmaQV{},
        make_shape(shape<1>(TileShape_MNK_QV{}), Int<kHeadDimV>{}, Int<kStages>{})));
    static_assert(CUTE_STATIC_V(size(SmemLayoutVMmaQV{})) == size(SmemLayoutVtMma{}));
    using SmemLayoutAtomVCpAsync = decltype(cutlass::gemm::collective::detail::ss_smem_selector<GMMA::Major::K, Element,
        decltype(cute::get<1>(TileShape_MNK{})), Int<kHeadDimV>>());
    using SmemLayoutVCpAsync = decltype(tile_to_shape(
        SmemLayoutAtomVCpAsync{},
        make_shape(shape<1>(TileShape_MNK{}), Int<kHeadDimV>{}, Int<kStages>{})));
    using SmemLayoutAtomP = decltype(cutlass::gemm::collective::detail::ss_smem_selector<GMMA::Major::K, Element,
        decltype(cute::get<0>(TileShape_MNK{})), decltype(cute::get<1>(TileShape_MNK{}))>());
    using SmemLayoutP = decltype(tile_to_shape(SmemLayoutAtomP{}, select<0, 1>(TileShape_MNK{})));
    using SmemLayoutScale = cute::Layout<cute::Shape<Int<kBlockM>, Int<kStages>>>;
    using SmemCopyAtomP = Copy_Atom<cute::SM90_U32x4_STSM_N, Element>;
    using SmemLayoutSink = Layout<Shape<_64>>;
    static_assert(!Transpose_V || (kHeadDimV % 32 == 0 && kBlockN % 32 == 0));
    static constexpr bool kHeadDimV_multiple_64 = kHeadDimV % 64 == 0;
    static_assert(!Transpose_V || (kHeadDimV_multiple_64 || kBlockN % 64 == 0));
    using LDSM_thread_shape  = std::conditional_t<kHeadDimV_multiple_64, Shape<_32, _4, _1, _1>, Shape<_16, _4, _1, _2>>;
    using LDSM_thread_stride = std::conditional_t<kHeadDimV_multiple_64, Stride<_4, _1, _0, _0>, Stride<_4, _1, _0, _64>>;
    using LDSM_value_shape = Shape<_2, _2, _1, _4>;
    using LDSM_value_stride = Stride<_1, _2, _16, _4>;
    using LDSM_divide_shape = std::conditional_t<kHeadDimV_multiple_64, Shape<_64, _8>, Shape<_32, _8>>;
    using S2RTiledCopyVt = decltype(make_tiled_copy(
        Copy_Atom<SM75_U16x8_LDSM_T, Element>{}, Layout<LDSM_thread_shape, LDSM_thread_stride>{},
        Layout<LDSM_value_shape, LDSM_value_stride>{}));
    using STSM_thread_shape  = std::conditional_t<kHeadDimV_multiple_64, Shape<_8, _4, _4, _1>, Shape<_8, _4, _2, _2>>;
    using STSM_thread_stride = std::conditional_t<kHeadDimV_multiple_64, Stride<_4, _1, _32, _0>, Stride<_4, _1, _32, _64>>;
    using STSM_value_shape = Shape<_1, _4, _2, _2>;
    using STSM_value_stride = Stride<_0, _1, _4, _8>;
    using STSM_divide_shape = Shape<_8, _16>;
    using R2STiledCopyV = decltype(make_tiled_copy(
        Copy_Atom<SM90_U32x4_STSM_N, Element>{}, Layout<STSM_thread_shape, STSM_thread_stride>{},
        Layout<STSM_value_shape, STSM_value_stride>{}));
    using GmemTiledCopyQ = cute::SM90_TMA_LOAD;
    using GmemTiledCopyKV = decltype(cutlass::gemm::collective::detail::sm90_cluster_shape_to_tma_atom(shape<0>(ClusterShape{})));
    static constexpr int kHeadDimGCD = cute::gcd(kHeadDim, kHeadDimV);
    static constexpr int kGmemElemsPerLoad = sizeof(cute::uint128_t) / sizeof(Element);
    static_assert(kHeadDimGCD % kGmemElemsPerLoad == 0, "Headdim and HeaddimV must be a multiple of kGmemElemsPerLoad");
    static constexpr int kBytePerHalfRow = kHeadDimGCD / 2 * sizeof(Element);
    static constexpr int kBlockKGmem = (kBytePerHalfRow % 128 == 0 ? 128 : (kBytePerHalfRow % 64 == 0 ? 64 : 32)) / sizeof(Element);
    static constexpr int kGmemThreadsPerRow = kBlockKGmem / kGmemElemsPerLoad;
    static_assert(NumMmaThreads % kGmemThreadsPerRow == 0, "NumMmaThreads must be a multiple of kGmemThreadsPerRow");
    static_assert(cutlass::NumThreadsPerWarp % kGmemThreadsPerRow == 0, "kGmemThreadsPerRow must divide NumThreadsPerWarp");
    using GmemLayoutAtom = Layout<Shape <Int<NumMmaThreads / kGmemThreadsPerRow>, Int<kGmemThreadsPerRow>>,
                                  Stride<Int<kGmemThreadsPerRow>, _1>>;
    static_assert(!AppendKV || kBlockM % CUTE_STATIC_V(shape<0>(GmemLayoutAtom{})) == 0, "kBlockM must be a multiple of NumMmaThreads / kGmemThreadsPerRow");
    using GmemTiledCopyAppendKV = decltype(
        make_tiled_copy(Copy_Atom<AutoVectorizingCopyWithAssumedAlignment<128>, Element>{},
                        GmemLayoutAtom{},
                        Layout<Shape<_1, Int<kGmemElemsPerLoad>>>{}));
    using ShapeQKV = cute::Shape<int32_t, int32_t, int32_t, int32_t>;
    using StrideQK = cute::Stride<int64_t, _1, int64_t, int64_t>;
    using StrideV = std::conditional_t<!V_colmajor, StrideQK, cute::Stride<_1, int64_t, int64_t, int64_t>>;
    template <class P> static CUTLASS_DEVICE auto att_paged_stride_v(P const& p) {
        if constexpr (V_colmajor) { return p.stride_K; } else { return p.stride_V; }
    }
    using ShapeQPacked = std::conditional_t<!PackGQA, ShapeQKV, cute::Shape<cute::Shape<int32_t, int32_t>, int32_t, int32_t, int32_t>>;
    using StrideQPacked = std::conditional_t<!PackGQA, StrideQK, cute::Stride<cute::Stride<int64_t, int64_t>, _1, int64_t, int64_t>>;
    using ShapePageTable = cute::Shape<int32_t, int32_t>;
    using StridePageTable = cute::Stride<int64_t, _1>;
    using ShapeRotary = cute::Shape<int32_t, int32_t>;
    using StrideRotary = cute::Stride<int64_t, _1>;
    using StrideDescale = cute::Stride<int64_t, int64_t>;
    using TMA_Q = decltype(make_tma_copy_A_sm90(
        GmemTiledCopyQ{},
        make_tensor(make_gmem_ptr(static_cast<Element const*>(nullptr)), ShapeQKV{}, StrideQK{}),
        SmemLayoutQ{},
        TileShape_MNK{},
        ClusterShape{}));
    using TMA_K = decltype(make_tma_copy_B_sm90(
        GmemTiledCopyKV{},
        make_tensor(make_gmem_ptr(static_cast<Element const*>(nullptr)), ShapeQKV{}, StrideQK{}),
        take<0, 2>(SmemLayoutK{}),
        TileShape_MNK{},
        ClusterShape{}));
    using TMA_V = decltype(make_tma_copy(
        GmemTiledCopyKV{},
        make_tensor(make_gmem_ptr(static_cast<Element const*>(nullptr)), ShapeQKV{}, select<1, 0, 2, 3>(StrideV{})),
        take<0, 2>(SmemLayoutVt{}),
        select<1, 2>(TileShape_MNK_PV{}),
        size<0>(ClusterShape{})));
    using TMA_Qv_ = decltype(make_tma_copy_A_sm90(
        GmemTiledCopyQ{},
        make_tensor(make_gmem_ptr(static_cast<Element const*>(nullptr)), ShapeQKV{}, StrideQK{}),
        SmemLayoutQv{},
        TileShape_MNK_QV{},
        ClusterShape{}));
    using TMA_Qv = std::conditional_t<HasQv, TMA_Qv_, std::nullptr_t>;
    using ShapeMask = cute::Shape<int32_t, int32_t>;
    using StrideMask = cute::Stride<int64_t, _1>;
    using SmemLayoutMask = Layout<Shape<Int<kBlockM>, Int<kNumInt32PerBlock>, Int<kStages>>,
                                   Stride<Int<kNumInt32PerBlock>, _1, Int<kBlockM * kNumInt32PerBlock>>>;
    using TMA_Mask_ = decltype(make_tma_copy(
        SM90_TMA_LOAD{},
        make_tensor(make_gmem_ptr(static_cast<int32_t const*>(nullptr)), ShapeMask{}, StrideMask{}),
        take<0, 2>(SmemLayoutMask{}),
        Shape<Int<kBlockM>, Int<kNumInt32PerBlock>>{},
        _1{}));
    using TMA_Mask = std::conditional_t<Has_sparse_mask && Use_TMA_KV, TMA_Mask_, std::nullptr_t>;
    static constexpr uint32_t TmaTransactionBytesQ = static_cast<uint32_t>(size(SmemLayoutQ{}) * cutlass::sizeof_bits_v<Element> / 8);
    static constexpr uint32_t TmaTransactionBytesK = static_cast<uint32_t>(size(take<0, 2>(SmemLayoutK{})) * cutlass::sizeof_bits_v<Element> / 8);
    static constexpr uint32_t TmaTransactionBytesV = 72 * kBlockN;
    static constexpr uint32_t TmaTransactionBytesQv = static_cast<uint32_t>(size(SmemLayoutQv{}) * cutlass::sizeof_bits_v<Element> / 8);
    static constexpr uint32_t TmaTransactionBytesMask = static_cast<uint32_t>(kBlockM * kNumInt32PerBlock * sizeof(int));
    static constexpr bool Use_TMA_Mask = Use_TMA_KV && Has_sparse_mask;
    using PipelineTmaAsync = std::conditional_t<CUTE_STATIC_V(size(ClusterShape{})) == 1, typename cutlass::PipelineTmaAsyncNoCluster<kStages>, typename cutlass::PipelineTmaAsync<kStages>>;
    using PipelineTmaAsyncMask = typename cutlass::PipelineTmaAsyncNoCluster<kStages>;
    using MainloopPipelineK = std::conditional_t<Use_TMA_KV, PipelineTmaAsync, typename cutlass::PipelineAsync<kStages>>;
    using MainloopPipelineV = std::conditional_t<!Transpose_V && Use_TMA_KV, PipelineTmaAsync, typename cutlass::PipelineAsync<kStages>>;
    using MainloopPipelineVt = std::conditional_t<Use_TMA_KV, PipelineTmaAsync, typename cutlass::PipelineAsync<kStages>>;
    using MainloopPipelineMask = std::conditional_t<Use_TMA_Mask, PipelineTmaAsyncMask, typename cutlass::PipelineAsync<kStages>>;
    using MainloopPipelineKVNew = PipelineTmaAsync;
    using PipelineState = cutlass::PipelineState<kStages>;
    static constexpr size_t SmemAlignmentQ = Use_TMA_Q && !MmaQK_is_RS ? 128 : cutlass::detail::alignment_for_swizzle(SmemLayoutQ{});
    static constexpr size_t SmemAlignmentK = Use_TMA_KV && !AppendKV ? 128 : cutlass::detail::alignment_for_swizzle(SmemLayoutK{});
    static constexpr size_t SmemAlignmentVtNoTranspose = 1024;
    static constexpr size_t SmemAlignmentQv = Use_TMA_Q ? 128 : cutlass::detail::alignment_for_swizzle(SmemLayoutQv{});
    static_assert(SmemAlignmentQ >= 128 and SmemAlignmentK >= 128 && SmemAlignmentVtNoTranspose >= 128, "Require at least 128B alignment");
    static constexpr size_t SmemAlignmentP = cutlass::detail::alignment_for_swizzle(SmemLayoutP{});
    static_assert(SmemAlignmentP >= 128, "Require at least 128B alignment");
    using SmemP_t = std::conditional_t<MmaPV_is_RS, cute::array<Element, 0>, cute::array_aligned<Element, cute::cosize_v<SmemLayoutP>, SmemAlignmentP>>;
    using SmemScale_t = std::conditional_t<!LargeHeadDimV, cute::array<float, 0>, cute::array_aligned<float, cute::cosize_v<SmemLayoutScale>, 128>>;
    using SmemQv_t = std::conditional_t<!HasQv, cute::array<Element, 0>, cute::array_aligned<Element, cute::cosize_v<SmemLayoutQv>, SmemAlignmentQv>>;
    using SmemMask_t = std::conditional_t<!Has_sparse_mask, cute::array<int, 0>, cute::array_aligned<int, kStages * kBlockM * kNumInt32PerBlock, 128>>;
    struct TensorStorageWithoutPNoTranspose : cute::aligned_struct<cute::max(SmemAlignmentQ, SmemAlignmentK, SmemAlignmentVtNoTranspose), _0> {
        cute::array_aligned<Element, kSmemVSize, SmemAlignmentVtNoTranspose> smem_v;
        cute::array_aligned<Element, cute::cosize_v<SmemLayoutQ>, SmemAlignmentQ> smem_q;
        cute::array_aligned<Element, cute::cosize_v<SmemLayoutK>, SmemAlignmentK> smem_k;
        SmemQv_t smem_qv;
        cute::array_aligned<ElementSink, cute::cosize_v<SmemLayoutSink>, 128> smem_sink;
        cute::array_aligned<uint8_t, 512, 128> smem_ones;
        SmemMask_t smem_mask;
    };
    struct TensorStorageWithPNoTranspose : cute::aligned_struct<cute::max(SmemAlignmentQ, SmemAlignmentK, SmemAlignmentVtNoTranspose, SmemAlignmentP), _0> {
        cute::array_aligned<Element, kSmemVSize, SmemAlignmentVtNoTranspose> smem_v;
        cute::array_aligned<Element, cute::cosize_v<SmemLayoutQ>, SmemAlignmentQ> smem_q;
        cute::array_aligned<Element, cute::cosize_v<SmemLayoutK>, SmemAlignmentK> smem_k;
        SmemQv_t smem_qv;
        SmemP_t smem_p;
        cute::array_aligned<ElementSink, cute::cosize_v<SmemLayoutSink>, 128> smem_sink;
        cute::array_aligned<uint8_t, 512, 128> smem_ones;
        SmemMask_t smem_mask;
    };
    struct TensorStorageWithPScaleNoTranspose : cute::aligned_struct<cute::max(SmemAlignmentQ, SmemAlignmentK, SmemAlignmentVtNoTranspose, SmemAlignmentP), _0> {
        cute::array_aligned<Element, kSmemVSize, SmemAlignmentVtNoTranspose> smem_v;
        cute::array_aligned<Element, cute::cosize_v<SmemLayoutQ>, SmemAlignmentQ> smem_q;
        cute::array_aligned<Element, cute::cosize_v<SmemLayoutK>, SmemAlignmentK> smem_k;
        SmemQv_t smem_qv;
        SmemP_t smem_p;
        SmemScale_t smem_scale;
        cute::array_aligned<ElementSink, cute::cosize_v<SmemLayoutSink>, 128> smem_sink;
        cute::array_aligned<uint8_t, 512, 128> smem_ones;
        SmemMask_t smem_mask;
    };
    using TensorStorageNoTranspose = std::conditional_t<
        MmaPV_is_RS,
        TensorStorageWithoutPNoTranspose,
        std::conditional_t<!LargeHeadDimV, TensorStorageWithPNoTranspose, TensorStorageWithPScaleNoTranspose>
    >;
    static constexpr size_t SmemAlignmentVt = cutlass::detail::alignment_for_swizzle(SmemLayoutVt{});
    static constexpr size_t SmemAlignmentV = cutlass::detail::alignment_for_swizzle(SmemLayoutVtMma72{});
    static_assert(SmemAlignmentVt >= 128 and SmemAlignmentV >= 128, "Require at least 128B alignment");
    struct TensorStorageTransposeV : cute::aligned_struct<cute::max(SmemAlignmentQ, SmemAlignmentK, SmemAlignmentV), _0> {
        cute::array_aligned<Element, kSmemVSize, SmemAlignmentV> smem_v;
        cute::array_aligned<Element, cute::cosize_v<SmemLayoutVt>, SmemAlignmentVt> smem_vt;
        cute::array_aligned<Element, cute::cosize_v<SmemLayoutQ>, SmemAlignmentQ> smem_q;
        cute::array_aligned<Element, cute::cosize_v<SmemLayoutK>, SmemAlignmentK> smem_k;
        SmemQv_t smem_qv;
        SmemScale_t smem_scale;
        cute::array_aligned<ElementSink, cute::cosize_v<SmemLayoutSink>, 128> smem_sink;
        cute::array_aligned<uint8_t, 512, 128> smem_ones;
        SmemMask_t smem_mask;
    };
    using TensorStorage = std::conditional_t<!Transpose_V, TensorStorageNoTranspose, TensorStorageTransposeV>;
    static constexpr bool UseSchedulerBarrier = (IntraWGOverlap
        ? (NumMmaWarpGroups >= 2) && (!Is_FP8 ? kHeadDim <= 128 : (kHeadDim >= 128 || ATT_SCHED))
        : NumMmaWarpGroups == 2)
        && !LargeHeadDimV;
    static constexpr bool RescaleOBeforeGemm = kHeadDim > 128 && (!Is_FP8 || V_colmajor) && IntraWGOverlap;
    struct Arguments {
        Element const* const ptr_Q;
        ShapeQKV const shape_Q;
        StrideQK const stride_Q;
        Element* const ptr_K;
        ShapeQKV const shape_K;
        StrideQK const stride_K;
        Element* const ptr_V;
        int32_t const headdim_v;
        StrideV const stride_V;
        Element const* const ptr_K_new;
        ShapeQKV const shape_K_new;
        StrideQK const stride_K_new;
        Element const* const ptr_V_new;
        StrideV const stride_V_new;
        Element const* const ptr_Qv;
        StrideQK const stride_Qv;
        Element const* const ptr_rotary_cos;
        ShapeRotary const shape_rotary;
        StrideRotary const stride_rotary_cos;
        Element const* const ptr_rotary_sin;
        StrideRotary const stride_rotary_sin;
        bool const is_rotary_interleaved;
        int const* const ptr_pagetable;
        ShapePageTable const shape_pagetable;
        StridePageTable const stride_pagetable;
        float const softmax_scale;
        float const* ptr_q_descale, *ptr_k_descale, *ptr_v_descale;
        StrideDescale const stride_q_descale, stride_k_descale, stride_v_descale;
        int const window_size_left = -1, window_size_right = -1, attention_chunk = 0;
        float const softcap_val;
        int const num_splits;
        int const* const kv_batch_idx = nullptr;
        int const* const cu_seqlens_q = nullptr;
        int const* const cu_seqlens_k = nullptr;
        int const* const cu_seqlens_k_new = nullptr;
        int const* const seqused_q = nullptr;
        int const* const seqused_k = nullptr;
        int const* const leftpad_k = nullptr;
        int const* const seqlens_rotary = nullptr;
        ElementSink const* const ptr_sink = nullptr;
        int const* sparse_mask_fine = nullptr;
        int sparse_mask_max_k_blocks = 0;
        int sparse_mask_fine_q_stride = 0;
        int sparse_mask_fine_k_stride = 0;
        int total_q = 0;
    };
    struct Params {
        Element const* const ptr_Q;
        ShapeQKV const shape_Q;
        StrideQK const stride_Q;
        ShapeQPacked const shape_Q_packed;
        StrideQPacked const stride_Q_packed;
        Element* const ptr_K;
        ShapeQKV const shape_K;
        StrideQK const stride_K;
        Element* const ptr_V;
        int32_t const headdim_v;
        StrideV const stride_V;
        Element const* const ptr_K_new;
        ShapeQKV const shape_K_new;
        StrideQK const stride_K_new;
        Element const* const ptr_V_new;
        StrideV const stride_V_new;
        Element const* const ptr_Qv;
        StrideQK const stride_Qv;
        ShapeQPacked const shape_Qv_packed;
        StrideQPacked const stride_Qv_packed;
        Element const* const ptr_rotary_cos;
        ShapeRotary const shape_rotary;
        StrideRotary const stride_rotary_cos;
        Element const* const ptr_rotary_sin;
        StrideRotary const stride_rotary_sin;
        bool const is_rotary_interleaved;
        int const* const ptr_pagetable;
        ShapePageTable const shape_pagetable;
        StridePageTable const stride_pagetable;
        cutlass::FastDivmod page_size_divmod;
        cutlass::FastDivmod blockN_per_page_size_divmod;
        cutlass::FastDivmod qhead_per_khead_divmod;
        TMA_Q tma_load_Q;
        TMA_K tma_load_K;
        TMA_V tma_load_V;
        TMA_K tma_load_K_new;
        TMA_V tma_load_V_new;
        TMA_Qv tma_load_Qv;
        TMA_Mask tma_load_Mask;
        float const softmax_scale_log2;
        float const* ptr_q_descale, *ptr_k_descale, *ptr_v_descale;
        StrideDescale const stride_q_descale, stride_k_descale, stride_v_descale;
        float const softcap_val;
        int const window_size_left, window_size_right;
        cutlass::FastDivmod attention_chunk_divmod;
        int const num_splits;
        int const* const kv_batch_idx = nullptr;
        int const* const cu_seqlens_q = nullptr;
        int const* const cu_seqlens_k = nullptr;
        int const* const cu_seqlens_k_new = nullptr;
        int const* const seqused_q = nullptr;
        int const* const seqused_k = nullptr;
        int const* const leftpad_k = nullptr;
        int const *const seqlens_rotary = nullptr;
        ElementSink const* const ptr_sink = nullptr;
        int const* sparse_mask_fine = nullptr;
        int sparse_mask_max_k_blocks = 0;
        int sparse_mask_fine_q_stride = 0;
        int sparse_mask_fine_k_stride = 0;
        int total_q = 0;
    };
    static Params
    to_underlying_arguments(Arguments const& args) {
        Tensor mQ = make_tensor(make_gmem_ptr(args.ptr_Q), args.shape_Q, args.stride_Q);
        TMA_Q tma_load_Q = make_tma_copy_A_sm90(
            GmemTiledCopyQ{},
            mQ,
            SmemLayoutQ{},
            TileShape_MNK{},
            ClusterShape{});
        Tensor mK = make_tensor(make_gmem_ptr(args.ptr_K), args.shape_K, args.stride_K);
        TMA_K tma_load_K = make_tma_copy_B_sm90(
            GmemTiledCopyKV{},
            mK,
            take<0, 2>(SmemLayoutK{}),
            TileShape_MNK{},
            ClusterShape{});
        Tensor mV = make_tensor(make_gmem_ptr(args.ptr_V),
                                make_shape(args.headdim_v, get<0>(args.shape_K), get<2>(args.shape_K), get<3>(args.shape_K)),
                                select<1, 0, 2, 3>(args.stride_V));
        TMA_V tma_load_V = make_tma_copy(
            GmemTiledCopyKV{},
            mV,
            take<0, 2>(SmemLayoutVt{}),
            select<1, 2>(TileShape_MNK_PV{}),
            size<0>(ClusterShape{}));
        Tensor mKnew = make_tensor(make_gmem_ptr(args.ptr_K_new), args.shape_K_new, args.stride_K_new);
        TMA_K tma_load_K_new = make_tma_copy_B_sm90(
            GmemTiledCopyKV{},
            cute::conditional_return<AppendKV>(mKnew, mK),
            take<0, 2>(SmemLayoutK{}),
            TileShape_MNK{},
            ClusterShape{});
        Tensor mVnew = make_tensor(make_gmem_ptr(args.ptr_V_new),
                                   make_shape(args.headdim_v, get<0>(args.shape_K_new), get<2>(args.shape_K_new), get<3>(args.shape_K_new)),
                                   select<1, 0, 2, 3>(args.stride_V_new));
        TMA_V tma_load_V_new = make_tma_copy(
            GmemTiledCopyKV{},
            cute::conditional_return<AppendKV>(mVnew, mV),
            take<0, 2>(SmemLayoutVt{}),
            select<1, 2>(TileShape_MNK_PV{}),
            size<0>(ClusterShape{}));
        auto shape_Qv = make_shape(get<0>(args.shape_Q), args.headdim_v, get<2>(args.shape_Q), get<3>(args.shape_Q));
        Tensor mQv = make_tensor(make_gmem_ptr(args.ptr_Qv), shape_Qv, args.stride_Qv);
        TMA_Qv tma_load_Qv = [&] {
            if constexpr (HasQv) {
                return make_tma_copy_A_sm90(
                    GmemTiledCopyQ{},
                    mQv,
                    SmemLayoutQv{},
                    TileShape_MNK_QV{},
                    ClusterShape{});
            } else {
                return nullptr;
            }
        }();
        TMA_Mask tma_load_Mask = [&] {
            if constexpr (Has_sparse_mask && Use_TMA_KV) {
                auto shape_Mask = make_shape(
                    static_cast<int32_t>(args.total_q),
                    static_cast<int32_t>(args.sparse_mask_fine_q_stride)
                );
                auto stride_Mask = make_stride(args.sparse_mask_fine_q_stride, Int<1>{});
                Tensor mMask = make_tensor(make_gmem_ptr(args.sparse_mask_fine), shape_Mask, stride_Mask);
                return make_tma_copy(
                    SM90_TMA_LOAD{},
                    mMask,
                    take<0, 2>(SmemLayoutMask{}),
                    Shape<Int<kBlockM>, Int<kNumInt32PerBlock>>{},
                    _1{});
            } else {
                return nullptr;
            }
        }();
        int const qhead_per_khead = !PackGQA ? 1 : cute::ceil_div(get<2>(args.shape_Q), get<2>(args.shape_K));
        auto const shape_Q_packed = cute::conditional_return<!PackGQA>(
            args.shape_Q,
            make_shape(make_shape(qhead_per_khead, get<0>(args.shape_Q)), get<1>(args.shape_Q), get<2>(args.shape_K), get<3>(args.shape_Q))
        );
        auto const stride_Q_packed = cute::conditional_return<!PackGQA>(
            args.stride_Q,
            make_stride(make_stride(get<2>(args.stride_Q), get<0>(args.stride_Q)), get<1>(args.stride_Q), get<2>(args.stride_Q) * qhead_per_khead, get<3>(args.stride_Q))
        );
        auto const shape_Qv_packed = cute::conditional_return<!PackGQA>(
            shape_Qv,
            make_shape(make_shape(qhead_per_khead, get<0>(shape_Qv)), get<1>(shape_Qv), get<2>(args.shape_K), get<3>(shape_Qv))
        );
        auto const stride_Qv_packed = cute::conditional_return<!PackGQA>(
            args.stride_Qv,
            make_stride(make_stride(get<2>(args.stride_Qv), get<0>(args.stride_Qv)), get<1>(args.stride_Qv), get<2>(args.stride_Qv) * qhead_per_khead, get<3>(args.stride_Qv))
        );
        if (get<1>(args.shape_rotary) > 0) {
            assert(args.ptr_rotary_cos != nullptr && args.ptr_rotary_sin != nullptr);
        }
        assert(args.num_splits >= 1);
        int page_size = !args.ptr_pagetable ? 1 : get<0>(args.shape_K);
        if (!PagedKVNonTMA && args.ptr_pagetable != nullptr) {
            assert(page_size % kBlockN == 0);
            assert(!args.leftpad_k);
        }
        cutlass::FastDivmod attention_chunk_divmod(args.attention_chunk >= 1 ? args.attention_chunk : 1);
        attention_chunk_divmod.divisor = args.attention_chunk;
        return {args.ptr_Q, args.shape_Q, args.stride_Q, shape_Q_packed, stride_Q_packed,
                args.ptr_K, args.shape_K, args.stride_K, args.ptr_V, args.headdim_v, args.stride_V,
                args.ptr_K_new, args.shape_K_new, args.stride_K_new, args.ptr_V_new, args.stride_V_new,
                args.ptr_Qv, args.stride_Qv, shape_Qv_packed, stride_Qv_packed,
                args.ptr_rotary_cos, args.shape_rotary, args.stride_rotary_cos,
                args.ptr_rotary_sin, args.stride_rotary_sin, args.is_rotary_interleaved,
                args.ptr_pagetable, args.shape_pagetable, args.stride_pagetable,
                cutlass::FastDivmod(page_size),
                cutlass::FastDivmod(!args.ptr_pagetable ? 1 : cute::ceil_div(page_size, kBlockN)),
                cutlass::FastDivmod(cute::ceil_div(get<2>(args.shape_Q), get<2>(args.shape_K))),
                tma_load_Q, tma_load_K, tma_load_V, tma_load_K_new, tma_load_V_new, tma_load_Qv, tma_load_Mask,
                !Has_softcap ? float(args.softmax_scale * M_LOG2E) : float(args.softcap_val * M_LOG2E),
                args.ptr_q_descale, args.ptr_k_descale, args.ptr_v_descale,
                args.stride_q_descale, args.stride_k_descale, args.stride_v_descale,
                !Has_softcap ? 0.f : args.softmax_scale / args.softcap_val,
                args.window_size_left, args.window_size_right, attention_chunk_divmod,
                !Split ? 1 : args.num_splits,
                args.kv_batch_idx,
                args.cu_seqlens_q, args.cu_seqlens_k, args.cu_seqlens_k_new,
                args.seqused_q, args.seqused_k, args.leftpad_k, args.seqlens_rotary, args.ptr_sink,
                args.sparse_mask_fine,
                args.sparse_mask_max_k_blocks,
                args.sparse_mask_fine_q_stride, args.sparse_mask_fine_k_stride,
            };
    }
    CUTLASS_DEVICE
    static void prefetch_tma_descriptors(Params const& params) {
        if constexpr (Use_TMA_Q) {
            cute::prefetch_tma_descriptor(params.tma_load_Q.get_tma_descriptor());
            if constexpr (HasQv) {
                cute::prefetch_tma_descriptor(params.tma_load_Qv.get_tma_descriptor());
            }
        }
        if constexpr (Use_TMA_KV) {
            cute::prefetch_tma_descriptor(params.tma_load_K.get_tma_descriptor());
            cute::prefetch_tma_descriptor(params.tma_load_V.get_tma_descriptor());
        }
        if constexpr (AppendKV) {
            cute::prefetch_tma_descriptor(params.tma_load_K_new.get_tma_descriptor());
            cute::prefetch_tma_descriptor(params.tma_load_V_new.get_tma_descriptor());
        }
        if constexpr (Use_TMA_Mask) {
            cute::prefetch_tma_descriptor(params.tma_load_Mask.get_tma_descriptor());
        }
    }
    template <typename SchedulerPrefetch, typename SharedStorage>
    CUTLASS_DEVICE void
    load(Params const& params,
         MainloopPipelineK pipeline_k,
         MainloopPipelineV pipeline_v,
         MainloopPipelineVt pipeline_vt,
         MainloopPipelineMask pipeline_mask,
         PipelineState& smem_pipe_write,
         SharedStorage &shared_storage,
         SchedulerPrefetch const& scheduler_prefetch,
         SeqlenInfo_t const& seqlen_info,
         cute::tuple<int32_t, int32_t, int32_t, int32_t> block_coord,
         int &work_idx
         ) {
        int const m_block = get<0>(block_coord);
        int const bidh = get<1>(block_coord);
        int const bidb = get<2>(block_coord);
        int const split_idx = get<3>(block_coord);
        auto [n_block_min, n_block_max, n_offset] = BlockMN_t::get_n_block_min_max(
            seqlen_info, m_block, bidb, split_idx, params.num_splits,
            params.window_size_left, params.window_size_right, params.attention_chunk_divmod,
            params.qhead_per_khead_divmod);
        if constexpr (Is_causal || Is_local || Varlen || Split) {
            if (n_block_max <= n_block_min) {
                scheduler_prefetch();
                return;
            }
        }
        Tensor sQ = make_tensor(make_smem_ptr(shared_storage.tensors.mainloop.smem_q.data()), SmemLayoutQ{});
        Tensor sK = make_tensor(make_smem_ptr(shared_storage.tensors.mainloop.smem_k.data()), SmemLayoutK{});
        Tensor sK_pi = as_position_independent_swizzle_tensor(sK);
        Tensor sVt = [&] {
            if constexpr (!Transpose_V) {
                return make_tensor(make_smem_ptr(shared_storage.tensors.mainloop.smem_v.data()), SmemLayoutVt{});
            } else {
                return cute::as_position_independent_swizzle_tensor(make_tensor(make_smem_ptr(shared_storage.tensors.mainloop.smem_vt.data()), SmemLayoutVt{}));
            }
        }();
        Tensor sV = cute::as_position_independent_swizzle_tensor(make_tensor(make_smem_ptr(shared_storage.tensors.mainloop.smem_v.data()), SmemLayoutVtMma{}));
        Tensor sVcpasync = [&] {
            if constexpr (!Transpose_V) {
                return cute::as_position_independent_swizzle_tensor(make_tensor(make_smem_ptr(shared_storage.tensors.mainloop.smem_v.data()), SmemLayoutVCpAsync{}));
            } else {
                return cute::as_position_independent_swizzle_tensor(make_tensor(make_smem_ptr(shared_storage.tensors.mainloop.smem_vt.data()), SmemLayoutVCpAsync{}));
            }
        }();
        Tensor sQv = make_tensor(make_smem_ptr(shared_storage.tensors.mainloop.smem_qv.data()), SmemLayoutQv{});
        int const thread_idx = threadIdx.x % NumProducerThreads;
        int const bidh_kv = !PackGQA ? params.qhead_per_khead_divmod.divide(bidh) : bidh;
        int const bidb_kv = params.kv_batch_idx == nullptr ? bidb : params.kv_batch_idx[bidb];
        uint32_t block_rank_in_cluster = cute::block_rank_in_cluster();
        constexpr uint32_t cluster_shape_x = get<0>(ClusterShape());
        uint2 cluster_local_block_id = {block_rank_in_cluster % cluster_shape_x, block_rank_in_cluster / cluster_shape_x};
        bool const is_varlen_q = Varlen && params.cu_seqlens_q;
        bool const is_varlen_k = Varlen && params.cu_seqlens_k;
        Tensor mQ = params.tma_load_Q.get_tma_tensor(params.shape_Q)(_, _, bidh, !is_varlen_q ? bidb : 0);
        Tensor mK_TMA = params.tma_load_K.get_tma_tensor(params.shape_K)(_, _, bidh_kv, _);
        auto shape_V = make_shape(params.headdim_v, get<0>(params.shape_K), get<2>(params.shape_K), get<3>(params.shape_K));
        Tensor mVt_TMA = params.tma_load_V.get_tma_tensor(shape_V)(_, _, bidh_kv, _);
        Tensor gQ = local_tile(domain_offset(make_coord(seqlen_info.offset_q, _0{}), mQ), select<0, 2>(TileShape_MNK{}), make_coord(m_block, _0{}));
        Tensor gK_TMA = local_tile(domain_offset(make_coord(seqlen_info.offset_k + n_offset, _0{}, _0{}), mK_TMA), select<1, 2>(TileShape_MNK{}), make_coord(_, _0{}, _));
        Tensor gVt_TMA = local_tile(domain_offset(make_coord(_0{}, seqlen_info.offset_k + n_offset, _0{}), mVt_TMA), select<1, 2>(TileShape_MNK_PV{}), make_coord(_0{}, _, _));
        auto block_tma_Q = params.tma_load_Q.get_slice(_0{});
        Tensor tQgQ = group_modes<0, 3>(block_tma_Q.partition_S(gQ));
        Tensor tQsQ = group_modes<0, 3>(block_tma_Q.partition_D(sQ));
        if (Use_TMA_Q && thread_idx == 0) { prefetch(params.tma_load_Q, tQgQ); }
        auto block_tma_K = params.tma_load_K.get_slice(cluster_local_block_id.x);
        Tensor tKgK_TMA = group_modes<0, 3>(block_tma_K.partition_S(gK_TMA));
        Tensor tKsK_TMA = group_modes<0, 3>(block_tma_K.partition_D(sK));
        auto block_tma_V = params.tma_load_V.get_slice(cluster_local_block_id.x);
        Tensor tVgVt_TMA = group_modes<0, 3>(block_tma_V.partition_S(gVt_TMA));
        Tensor tVsVt_TMA = group_modes<0, 3>(block_tma_V.partition_D(sVt));
        auto [tQvgQv, tQvsQv] = [&] {
            if constexpr (HasQv) {
                auto shape_Qv = make_shape(get<0>(params.shape_Q), params.headdim_v, get<2>(params.shape_Q), get<3>(params.shape_Q));
                Tensor mQv = params.tma_load_Qv.get_tma_tensor(shape_Qv)(_, _, bidh, !is_varlen_q ? bidb : 0);
                Tensor gQv = local_tile(domain_offset(make_coord(seqlen_info.offset_q, _0{}), mQv), select<0, 2>(TileShape_MNK_QV{}), make_coord(m_block, _0{}));
                auto block_tma_Qv = params.tma_load_Qv.get_slice(_0{});
                Tensor tQvgQv = group_modes<0, 3>(block_tma_Qv.partition_S(gQv));
                Tensor tQvsQv = group_modes<0, 3>(block_tma_Qv.partition_D(sQv));
                return cute::make_tuple(tQvgQv, tQvsQv);
            } else {
                return cute::make_tuple(nullptr, nullptr);
            }
        }();
        int const bidb_kv_idx = !is_varlen_k && !params.ptr_pagetable ? bidb_kv : 0;
        using PagedKVManager_t = PagedKVManager<get<1>(TileShape_MNK{}), get<2>(TileShape_MNK{}), get<1>(TileShape_MNK_PV{}), NumProducerThreads, Element, Transpose_V || !IntraWGOverlap >;
        PagedKVManager_t paged_kv_manager(
            params.ptr_pagetable, params.shape_pagetable, params.stride_pagetable,
            params.ptr_K, params.shape_K, params.stride_K,
            params.ptr_V, params.headdim_v, att_paged_stride_v(params),
            params.page_size_divmod, params.blockN_per_page_size_divmod,
            bidb_kv, bidh_kv, thread_idx, seqlen_info.seqlen_k - n_offset, seqlen_info.leftpad_k + n_offset, bidb_kv_idx
        );
        S2RTiledCopyVt s2r_tiled_copy_vt;
        R2STiledCopyV r2s_tiled_copy_v;
        auto s2r_thr_copy_vt = s2r_tiled_copy_vt.get_thread_slice(thread_idx);
        auto r2s_thr_copy_v = r2s_tiled_copy_v.get_thread_slice(thread_idx);
        Tensor tTranssVt_ = s2r_thr_copy_vt.partition_S(flat_divide(sVt, LDSM_divide_shape{}));
        Tensor tTranssV_ = r2s_thr_copy_v.partition_D(flat_divide(sV, STSM_divide_shape{}));
        CUTE_STATIC_ASSERT_V(rank(tTranssVt_) == rank(tTranssV_));
        CUTE_STATIC_ASSERT_V(size<0>(tTranssVt_) == size<0>(tTranssV_));
        CUTE_STATIC_ASSERT_V(size<1>(tTranssVt_) == size<1>(tTranssV_));
        CUTE_STATIC_ASSERT_V(size<2>(tTranssVt_) == size<2>(tTranssV_));
        CUTE_STATIC_ASSERT_V(size<3>(tTranssVt_) == size<3>(tTranssV_));
        CUTE_STATIC_ASSERT_V(size<4>(tTranssVt_) == size<4>(tTranssV_));
        static constexpr int Transpose_ILP = (size<2>(tTranssVt_) * size<3>(tTranssVt_)) % 2 == 0 ? 2 : 1;
        Tensor tTranssVt = logical_divide(group_modes<1, rank(tTranssVt_) - 1>(tTranssVt_), Shape<Underscore, Int<Transpose_ILP>>{});
        Tensor tTranssV = logical_divide(group_modes<1, rank(tTranssV_) - 1>(tTranssV_), Shape<Underscore, Int<Transpose_ILP>>{});
        auto transpose_V = [&](int stage) {
            if constexpr (Transpose_V) {
                #pragma unroll
                for (int i = 0; i < size<1, 1>(tTranssVt); ++i) {
                    Tensor tTransrV = make_fragment_like(tTranssV(_, make_coord(_, _0{}), _0{}));
                    static_assert(size<0>(tTransrV) == 16);
                    Tensor tTransrV_64 = recast<uint2>(tTransrV);
                    cute::copy(s2r_tiled_copy_vt, tTranssVt(_, make_coord(_, i), stage), tTransrV);
                    #pragma unroll
                    for (int j = 0; j < size(tTransrV_64); ++j) {
                        uint32_t upper = tTransrV_64[j].x;
                        uint32_t lower = tTransrV_64[j].y;
                        tTransrV_64[j].x = __byte_perm(upper, lower, 0x6420);
                        tTransrV_64[j].y = __byte_perm(upper, lower, 0x7531);
                    }
                    cute::copy(r2s_tiled_copy_v, tTransrV, tTranssV(_, make_coord(_, i), stage));
                }
            }
        };
        uint16_t mcast_mask_kv = 0;
        if constexpr (cute::is_same_v<GmemTiledCopyKV, SM90_TMA_LOAD_MULTICAST>) {
            auto block_layout = Layout<ClusterShape>{};
            for (int m = 0; m < size<0>(block_layout); ++m) {
                mcast_mask_kv |= (uint16_t(1) << block_layout(m, cluster_local_block_id.y, _0{}));
            }
        }
        Tensor sMask = make_tensor(make_smem_ptr(shared_storage.tensors.mainloop.smem_mask.data()), SmemLayoutMask{});
        auto tMaskgMask_tMasksMask_tuple = [&] {
            if constexpr (Use_TMA_Mask) {
                auto shape_Mask_gmem = make_shape(
                    static_cast<int32_t>(params.total_q),
                    static_cast<int32_t>(params.sparse_mask_fine_q_stride)
                );
                Tensor mMask_TMA = params.tma_load_Mask.get_tma_tensor(shape_Mask_gmem);
                Tensor gMask_TMA = local_tile(
                    domain_offset(make_coord(seqlen_info.offset_q, _0{}), mMask_TMA),
                    Shape<Int<kBlockM>, Int<kNumInt32PerBlock>>{},
                    make_coord(_, _)
                );
                auto block_tma_Mask = params.tma_load_Mask.get_slice(_0{});
                Tensor tMaskgMask = group_modes<0, 3>(block_tma_Mask.partition_S(gMask_TMA));
                Tensor tMasksMask = group_modes<0, 3>(block_tma_Mask.partition_D(sMask));
                return cute::make_tuple(tMaskgMask, tMasksMask);
            } else {
                return cute::make_tuple(nullptr, nullptr);
            }
        }();
        auto tMaskgMask_TMA = cute::get<0>(tMaskgMask_tMasksMask_tuple);
        auto tMasksMask_TMA = cute::get<1>(tMaskgMask_tMasksMask_tuple);
        auto load_Mask = [&] (int const m_block, int const n_block, auto const& smem_pipe_write) {
            if constexpr (!Has_sparse_mask) {
                return;
            } else if constexpr (Use_TMA_Mask) {
                pipeline_mask.producer_acquire(smem_pipe_write);
                copy(params.tma_load_Mask.with(*pipeline_mask.producer_get_barrier(smem_pipe_write), 0 , TMA::CacheHintSm90::EVICT_FIRST),
                    tMaskgMask_TMA(_, m_block, n_block), tMasksMask_TMA(_, smem_pipe_write.index()));
            } else {
                pipeline_mask.producer_acquire(smem_pipe_write);
                int* stage_smem_base = shared_storage.tensors.mainloop.smem_mask.data() +
                                        smem_pipe_write.index() * (kBlockM * kNumInt32PerBlock);
                int const* src_base_common = params.sparse_mask_fine +
                                                n_block * params.sparse_mask_fine_k_stride +
                                                (seqlen_info.offset_q + m_block * kBlockM) * params.sparse_mask_fine_q_stride;
                constexpr int total_elems = kBlockM * kNumInt32PerBlock;
                int const block_row_start = m_block * kBlockM;
                #pragma unroll
                for (int i = thread_idx; i < total_elems; i += NumProducerThreads) {
                    int const row_offset = i / kNumInt32PerBlock;
                    int const word_idx   = i % kNumInt32PerBlock;
                    int const row_logic = block_row_start + row_offset;
                    bool const valid = (row_logic < seqlen_info.seqlen_q);
                    int const* src = src_base_common +
                                        row_offset * params.sparse_mask_fine_q_stride +
                                        word_idx;
                    flash::cp_async_ca_zfill(stage_smem_base + i, src, valid);
                }
                flash::cp_async_fence();
                pipeline_mask.producer_commit(smem_pipe_write, cutlass::arch::cpasync_barrier_arrive);
            }
        };
        auto load_K = [&] (int const n_block, auto const& smem_pipe_write, auto need_seqlenk_masking_type) {
            pipeline_k.producer_acquire(smem_pipe_write);
            if constexpr (!PagedKVNonTMA) {
                auto [n_block_idx, bidb_kv_idx] = paged_kv_manager.get_indices_for_K_TMA();
                copy(params.tma_load_K.with(*pipeline_k.producer_get_barrier(smem_pipe_write), mcast_mask_kv, TMA::CacheHintSm90::EVICT_LAST),
                    tKgK_TMA(_, n_block_idx, bidb_kv_idx), tKsK_TMA(_, smem_pipe_write.index()));
            } else {
                constexpr bool Seqlenk_mask = decltype(need_seqlenk_masking_type)::value;
                if constexpr ((ATT_PABL & 2) == 0) { paged_kv_manager.template load_K<Seqlenk_mask>(n_block, sK_pi(_, _, smem_pipe_write.index())); }
                pipeline_k.producer_commit(smem_pipe_write, cutlass::arch::cpasync_barrier_arrive);
            }
        };
        auto load_V = [&] (int const n_block, auto const& smem_pipe_write, auto need_seqlenk_masking_type) {
            auto pipeline_v_load = cute::conditional_return<!Transpose_V>(pipeline_v, pipeline_vt);
            pipeline_v_load.producer_acquire(smem_pipe_write);
            if constexpr (!PagedKVNonTMA) {
                auto [n_block_idx, bidb_kv_idx] = paged_kv_manager.get_indices_for_V_TMA();
                constexpr int TileBytes = 72 * ATT_BN;
                int const nblocks = (get<0>(params.shape_K) + ATT_BN - 1) / ATT_BN;
                int64_t const tile = (int64_t(bidb_kv_idx) * get<2>(params.shape_K) + bidh_kv) * nblocks + n_block;
                auto const* src = reinterpret_cast<uint8_t const*>(params.ptr_V) + tile * TileBytes;
                auto* dst = reinterpret_cast<uint8_t*>(shared_storage.tensors.mainloop.smem_v.data()) + smem_pipe_write.index() * TileBytes;
                uint32_t const da = cute::cast_smem_ptr_to_uint(dst);
                uint32_t const ba = cute::cast_smem_ptr_to_uint(pipeline_v_load.producer_get_barrier(smem_pipe_write));
                asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];" :: "r"(da), "l"(src), "r"(TileBytes), "r"(ba) : "memory");
            } else {
                constexpr bool Seqlenk_mask = decltype(need_seqlenk_masking_type)::value;
                if constexpr ((ATT_PABL & 2) == 0) { paged_kv_manager.template load_V<Seqlenk_mask>(n_block, sVcpasync(_, _, smem_pipe_write.index())); }
                pipeline_v_load.producer_commit(smem_pipe_write, cutlass::arch::cpasync_barrier_arrive);
            }
        };
        auto copy_Vt_to_V = [&] (auto const& smem_pipe_write) {
            PipelineState smem_pipe_read{smem_pipe_write.index(), smem_pipe_write.phase() ^ 1, smem_pipe_write.count()};
            pipeline_vt.consumer_wait(smem_pipe_read);
            pipeline_v.producer_acquire(smem_pipe_write);
                if constexpr ((ATT_PABL & 1) == 0 && (SK2_ABL & 16) == 0) { transpose_V(smem_pipe_write.index()); }
            {
                uint4* stg = reinterpret_cast<uint4*>(shared_storage.tensors.mainloop.smem_v.data() + smem_pipe_write.index() * (72 * kBlockN));
                constexpr int CW = kBlockN % 128 == 0 ? 128 : 64, PER = 8 * CW / 16;
                #pragma unroll
                for (int i = threadIdx.x % cutlass::NumThreadsPerWarpGroup; i < (kBlockN / CW) * PER; i += cutlass::NumThreadsPerWarpGroup) {
                    stg[(i / PER) * (72 * CW / 16) + 64 * CW / 16 + (i % PER)] = make_uint4(0x38383838u, 0x38383838u, 0x38383838u, 0x38383838u);
                }
            }
            cutlass::arch::fence_view_async_shared();
            pipeline_v.producer_commit(smem_pipe_write);
            cutlass::arch::NamedBarrier::sync(cutlass::NumThreadsPerWarpGroup, cutlass::arch::ReservedNamedBarriers::TransposeBarrier );
            pipeline_vt.consumer_release(smem_pipe_read);
        };
        int n_block = n_block_max - 1;
        int warp_idx_in_warpgroup = __shfl_sync(0xffffffff, (threadIdx.x / 32) % 4, 0);
        static constexpr bool SingleProducerWarp = NumProducerThreads == cutlass::NumThreadsPerWarp;
        bool should_load_KV = !Use_TMA_KV || ((SingleProducerWarp || warp_idx_in_warpgroup == 0) && cute::elect_one_sync());
        if (should_load_KV) {
            if constexpr (PagedKVNonTMA) {
                paged_kv_manager.template load_page_table<true , true >(n_block);
            } else {
                paged_kv_manager.template load_page_table_TMA<true >(n_block);
            }
            if constexpr (Transpose_V) { load_V(n_block, smem_pipe_write, cute::true_type{} ); }
            load_K(n_block, smem_pipe_write, cute::true_type{} );
        }
        if constexpr (Has_sparse_mask) {
            if constexpr (Use_TMA_Mask) {
                if (should_load_KV) { load_Mask(m_block, n_block, smem_pipe_write); }
            } else {
                load_Mask(m_block, n_block, smem_pipe_write);
            }
        }
        if constexpr (Use_TMA_Q) {
            if (SingleProducerWarp || warp_idx_in_warpgroup == 0) {
                cutlass::arch::NamedBarrier::sync(NumMmaThreadsQK + cutlass::NumThreadsPerWarp, static_cast<uint32_t>(FwdNamedBarriers::QueryEmpty) );
            }
            if ((SingleProducerWarp || warp_idx_in_warpgroup == 0) && cute::elect_one_sync()) {
                shared_storage.pipelines.barrier_Q.arrive_and_expect_tx(TmaTransactionBytesQ);
                copy(params.tma_load_Q.with(reinterpret_cast<typename cutlass::arch::ClusterTransactionBarrier::ValueType&>(shared_storage.pipelines.barrier_Q), 0 , !Split ? TMA::CacheHintSm90::EVICT_FIRST : TMA::CacheHintSm90::EVICT_LAST),
                    tQgQ, tQsQ);
                if constexpr (HasQv) {
                    shared_storage.pipelines.barrier_Qv.arrive_and_expect_tx(TmaTransactionBytesQv);
                    copy(params.tma_load_Qv.with(reinterpret_cast<typename cutlass::arch::ClusterTransactionBarrier::ValueType&>(shared_storage.pipelines.barrier_Qv), 0 , !Split ? TMA::CacheHintSm90::EVICT_FIRST : TMA::CacheHintSm90::EVICT_LAST),
                        tQvgQv, tQvsQv);
                }
            }
        } else {
            cutlass::arch::NamedBarrier::sync(NumMmaThreadsQK + NumProducerThreads, static_cast<uint32_t>(FwdNamedBarriers::QueryEmpty) );
            Tensor mQ = make_tensor(make_gmem_ptr(params.ptr_Q + seqlen_info.offset_q * get<0>(params.stride_Q)), params.shape_Q_packed, params.stride_Q_packed)(_, _, bidh, !is_varlen_q ? bidb : 0);
            Tensor sQ_pi = cute::as_position_independent_swizzle_tensor(sQ);
            using PackGQAt = flash::PackGQAManager<get<0>(TileShape_MNK{}), get<2>(TileShape_MNK{}), NumProducerThreads, Element>;
            PackGQAt::load_Q(mQ, sQ_pi, params.qhead_per_khead_divmod, thread_idx, seqlen_info.seqlen_q, m_block);
            auto &barrier_Q = shared_storage.pipelines.barrier_Q;
            cutlass::arch::cpasync_barrier_arrive(reinterpret_cast<uint64_t*>(&barrier_Q));
            barrier_Q.arrive();
            if constexpr (HasQv) {
                Tensor mQv = make_tensor(make_gmem_ptr(params.ptr_Qv + seqlen_info.offset_q * get<0>(params.stride_Qv)), params.shape_Qv_packed, params.stride_Qv_packed)(_, _, bidh, !is_varlen_q ? bidb : 0);
                Tensor sQv_pi = cute::as_position_independent_swizzle_tensor(sQv);
                using PackGQAt = flash::PackGQAManager<get<0>(TileShape_MNK_QV{}), get<2>(TileShape_MNK_QV{}), NumProducerThreads, Element>;
                PackGQAt::load_Q(mQv, sQv_pi, params.qhead_per_khead_divmod, thread_idx, seqlen_info.seqlen_q, m_block);
                auto &barrier_Qv = shared_storage.pipelines.barrier_Qv;
                cutlass::arch::cpasync_barrier_arrive(reinterpret_cast<uint64_t*>(&barrier_Qv));
                barrier_Qv.arrive();
            }
        }
        shared_storage.pipelines.barrier_O.wait((work_idx + 1) % 2);
        if constexpr (!Transpose_V && !IntraWGOverlap) {
            if (should_load_KV) { load_V(n_block, smem_pipe_write, cute::true_type{} ); }
        }
        int n_block_prev = n_block;
        --n_block;
        #pragma unroll (!Transpose_V && Use_TMA_KV && !Has_sparse_mask ? 2 : 1)
        for (; n_block >= n_block_min; --n_block) {
            PipelineState smem_pipe_write_v = smem_pipe_write;
            ++smem_pipe_write;
            if (should_load_KV) {
                if constexpr (PagedKVNonTMA) {
                if constexpr ((ATT_PABL & 4) == 0) { paged_kv_manager.template load_page_table<false >(n_block); }
                } else {
                    paged_kv_manager.load_page_table_TMA(n_block);
                }
                if constexpr (Transpose_V) { load_V(n_block, smem_pipe_write, cute::false_type{} ); }
                load_K(n_block, smem_pipe_write, cute::false_type{} );
                if constexpr (!Transpose_V) {
                    if constexpr (IntraWGOverlap) {
                        load_V(n_block_prev, smem_pipe_write_v, cute::true_type{} );
                    } else {
                        load_V(n_block, smem_pipe_write, cute::false_type{} );
                    }
                }
            }
            if constexpr (Has_sparse_mask) {
                if constexpr (Use_TMA_Mask) {
                    if (should_load_KV) { load_Mask(m_block, n_block, smem_pipe_write); }
                } else {
                    load_Mask(m_block, n_block, smem_pipe_write);
                }
            }
            n_block_prev = n_block;
            if constexpr (Transpose_V) { copy_Vt_to_V(smem_pipe_write_v); }
        }
        scheduler_prefetch();
        if constexpr (!Transpose_V && IntraWGOverlap) {
            if (should_load_KV) { load_V(n_block_prev, smem_pipe_write, cute::true_type{} ); }
        }
        if constexpr (Transpose_V) { copy_Vt_to_V(smem_pipe_write); }
        ++smem_pipe_write;
        ++work_idx;
    }
    template <typename SharedStorage>
    CUTLASS_DEVICE void
    load_tail(MainloopPipelineK pipeline_k, MainloopPipelineV pipeline_v, MainloopPipelineVt pipeline_vt, MainloopPipelineMask pipeline_mask,
              PipelineState& smem_pipe_write, SharedStorage &shared_storage, int const work_idx) {
        shared_storage.pipelines.barrier_O.wait((work_idx + 1) % 2);
        int warp_idx_in_warpgroup = __shfl_sync(0xffffffff, (threadIdx.x / 32) % 4, 0);
        if (warp_idx_in_warpgroup == 0 && cute::elect_one_sync()) {
            pipeline_k.producer_tail(smem_pipe_write);
            pipeline_v.producer_tail(smem_pipe_write);
            if constexpr (Has_sparse_mask) {pipeline_mask.producer_tail(smem_pipe_write);}
            if constexpr (Transpose_V) { pipeline_vt.producer_tail(smem_pipe_write); }
        }
    }
    CUTLASS_DEVICE void
    warp_scheduler_barrier_sync() {
        if constexpr (UseSchedulerBarrier) {
            cutlass::arch::NamedBarrier::sync(2 * cutlass::NumThreadsPerWarpGroup, static_cast<uint32_t>(FwdNamedBarriers::WarpSchedulerWG1) - 1 + flash::canonical_warp_group_idx_nosync() );
        }
    }
    CUTLASS_DEVICE void
    warp_scheduler_barrier_arrive() {
        if constexpr (UseSchedulerBarrier) {
            static_assert(NumMmaWarpGroups == 2 || NumMmaWarpGroups == 3);
            int const cur_WG = flash::canonical_warp_group_idx_nosync() - 1;
            int const next_WG = NumMmaWarpGroups == 2
                ? 1 - cur_WG
                : (cur_WG < NumMmaWarpGroups - 1 ? cur_WG + 1 : 0);
            cutlass::arch::NamedBarrier::arrive(2 * cutlass::NumThreadsPerWarpGroup, static_cast<uint32_t>(FwdNamedBarriers::WarpSchedulerWG1) + next_WG );
        }
    }
    CUTLASS_DEVICE void
    mma_init() {
        int warp_group_idx = flash::canonical_warp_group_idx_nosync();
        if (!LargeHeadDimV || warp_group_idx == 1) {
            cutlass::arch::NamedBarrier::arrive(NumMmaThreadsQK + (Use_TMA_Q ? cutlass::NumThreadsPerWarp : NumProducerThreads), static_cast<uint32_t>(FwdNamedBarriers::QueryEmpty) );
        }
        if (LargeHeadDimV && warp_group_idx > 1) {
            cutlass::arch::NamedBarrier::arrive(NumMmaThreads, static_cast<uint32_t>(FwdNamedBarriers::PEmpty) );
        }
        if constexpr (UseSchedulerBarrier) {
            static_assert(NumMmaWarpGroups == 2 || NumMmaWarpGroups == 3);
            if (warp_group_idx == 1) {
                cutlass::arch::NamedBarrier::arrive(2 * cutlass::NumThreadsPerWarpGroup, static_cast<uint32_t>(FwdNamedBarriers::WarpSchedulerWG1) );
            }
        }
    }
    CUTLASS_DEVICE void xu_sync() {
    }
    CUTLASS_DEVICE void xu_arrive() {
    }
    template <typename SharedStorage, typename FrgTensorO, typename Softmax>
    CUTLASS_DEVICE bool
    mma(Params const& params,
        MainloopPipelineK pipeline_k,
        MainloopPipelineV pipeline_v,
        MainloopPipelineMask pipeline_mask,
        PipelineState& smem_pipe_read,
        FrgTensorO& tOrO,
        Softmax& softmax,
        int const thread_idx,
        int &work_idx,
        SeqlenInfo_t const& seqlen_info,
        cute::tuple<int32_t, int32_t, int32_t, int32_t> block_coord,
        SharedStorage& shared_storage
        ) {
        static_assert(is_rmem<FrgTensorO>::value, "O tensor must be rmem resident.");
        static constexpr int kBlockM = get<0>(TileShape_MNK{});
        static constexpr int kBlockN = get<1>(TileShape_MNK{});
        int const m_block = get<0>(block_coord);
        int const bidh = get<1>(block_coord);
        int const bidb = get<2>(block_coord);
        int const split_idx = get<3>(block_coord);
        int const bidh_kv = !PackGQA ? params.qhead_per_khead_divmod.divide(bidh) : bidh;
        auto [n_block_min, n_block_max, n_offset] = BlockMN_t::get_n_block_min_max(
            seqlen_info, m_block, bidb, split_idx, params.num_splits,
            params.window_size_left, params.window_size_right, params.attention_chunk_divmod,
            params.qhead_per_khead_divmod);
        if constexpr (Is_causal || Is_local || Varlen || Split) {
            if (n_block_max <= n_block_min) { return false; }
        }
        Tensor sQ = make_tensor(make_smem_ptr(shared_storage.tensors.mainloop.smem_q.data()), SmemLayoutQ{});
        Tensor sK = make_tensor(make_smem_ptr(shared_storage.tensors.mainloop.smem_k.data()), SmemLayoutK{});
        Tensor sV = make_tensor(make_smem_ptr(shared_storage.tensors.mainloop.smem_v.data()), SmemLayoutVtMma{});
        Tensor sP = [&] {
            if constexpr (MmaPV_is_RS) {
                return make_tensor(make_smem_ptr(shared_storage.tensors.mainloop.smem_q.data()), SmemLayoutP{});
            } else {
                return make_tensor(make_smem_ptr(shared_storage.tensors.mainloop.smem_p.data()), SmemLayoutP{});
            }
        }();
        Tensor sScale = [&] {
            if constexpr (LargeHeadDimV) {
                return make_tensor(make_smem_ptr(shared_storage.tensors.mainloop.smem_scale.data()), SmemLayoutScale{});
            } else {
                return make_tensor(make_smem_ptr(static_cast<float*>(nullptr)), SmemLayoutScale{});
            }
        }();
        Tensor sQv = make_tensor(make_smem_ptr(shared_storage.tensors.mainloop.smem_qv.data()), SmemLayoutQv{});
        Tensor sVMmaQV = make_tensor(make_smem_ptr(shared_storage.tensors.mainloop.smem_v.data()), SmemLayoutVMmaQV{});
        if constexpr (!MmaQK_is_RS) {
            static_assert(stride<0>(typename TiledMmaQK::ALayout{}) == 0 and
                        stride<0>(typename TiledMmaQK::BLayout{}) == 0 and
                        size<0>(typename TiledMmaQK::ALayout{}) == cutlass::NumThreadsPerWarpGroup and
                        size<0>(typename TiledMmaQK::BLayout{}) == cutlass::NumThreadsPerWarpGroup,
                "Stride of the first mode must be 0 and the size of the mode must be NumThreadsPerWarpGroup");
        }
        static constexpr int MmaWarpGroups = size(TiledMmaPV{}) / cutlass::NumThreadsPerWarpGroup;
        Layout warp_group_thread_layout = make_layout(make_shape(Int<MmaWarpGroups>{}),
                                                      make_stride(Int<cutlass::NumThreadsPerWarpGroup>{}));
        int warp_group_idx = __shfl_sync(0xFFFFFFFF, thread_idx / cutlass::NumThreadsPerWarpGroup, 0);
        TiledMmaQK tiled_mma_qk;
        TiledMmaPV tiled_mma_pv;
        TiledMmaQV tiled_mma_qv;
        auto wg_mma_qk = tiled_mma_qk.get_slice(warp_group_thread_layout(warp_group_idx));
        auto wg_mma_pv = tiled_mma_pv.get_slice(warp_group_thread_layout(warp_group_idx));
        auto wg_mma_qv = tiled_mma_qv.get_slice(warp_group_thread_layout(warp_group_idx));
        auto smem_tiled_copy_P = make_tiled_copy_C(SmemCopyAtomP{}, tiled_mma_qk);
        auto smem_thr_copy_P = smem_tiled_copy_P.get_thread_slice(thread_idx);
        Tensor tSrQ = wg_mma_qk.partition_fragment_A(sQ);
        Tensor tSrK = wg_mma_qk.partition_fragment_B(sK);
        Tensor tOrV = wg_mma_pv.partition_fragment_B(sV);
        Tensor tOsP = wg_mma_pv.partition_fragment_A(sP);
        Tensor tSrQv = wg_mma_qv.partition_fragment_A(sQv);
        Tensor tSrV = wg_mma_qv.partition_fragment_B(sVMmaQV);
        Tensor tPsP = smem_thr_copy_P.partition_D(cute::as_position_independent_swizzle_tensor(sP));
        auto thread_mma_pv = tiled_mma_pv.get_thread_slice(thread_idx);
        Tensor taccOcO = thread_mma_pv.partition_C(cute::make_identity_tensor(select<0, 1>(TileShape_MNK_PV{})));
        Tensor taccOcO_rowcol = make_tensor(taccOcO.data(), flash::convert_layout_acc_rowcol(taccOcO.layout()));
        Tensor taccOcO_row = taccOcO_rowcol(_, _0{});
        auto store_scales = [&](auto& scales, int stage) {
            static_assert(CUTE_STATIC_V(size(scales)) == CUTE_STATIC_V(size(taccOcO_row)));
            #pragma unroll
            for (int mi = 0; mi < size(taccOcO_row); ++mi) {
                if (get<1>(taccOcO_row(_0{})) == 0) {
                    sScale(get<0>(taccOcO_row(mi)), stage) = scales(mi);
                }
            }
        };
        auto consumer_wait = [](auto& pipeline, auto& smem_pipe_read) {
            auto barrier_token = pipeline.consumer_try_wait(smem_pipe_read);
            pipeline.consumer_wait(smem_pipe_read, barrier_token);
        };
        int const seqlen_q = seqlen_info.seqlen_q;
        int const seqlen_k = seqlen_info.seqlen_k - n_offset;
        int const m_idx_min = !PackGQA ? m_block * kBlockM : params.qhead_per_khead_divmod.divide(m_block * kBlockM);
        int const m_idx_max = !PackGQA ? (m_block + 1) * kBlockM : params.qhead_per_khead_divmod.divide((m_block + 1) * kBlockM - 1) + 1;
        int n_block = n_block_max - 1;
        flash::Mask<kBlockM, kBlockN, PackGQA, TiledMmaQK> mask(
            thread_idx, seqlen_q, seqlen_k, params.window_size_left, params.window_size_right, 0 ,
            params.attention_chunk_divmod, params.qhead_per_khead_divmod
        );
        [[maybe_unused]] flash::SparseMask<kBlockM, kBlockN, PackGQA, TiledMmaQK> sparse_mask(
            thread_idx, seqlen_q, seqlen_k,
            params.sparse_mask_max_k_blocks,
            params.qhead_per_khead_divmod
        );
        float softcap_val = params.softcap_val;
        if constexpr (Has_softcap && Is_FP8) {
            float const q_descale = params.ptr_q_descale == nullptr ? 1.0f : params.ptr_q_descale[bidb * get<0>(params.stride_q_descale) + bidh_kv * get<1>(params.stride_q_descale)];
            float const k_descale = params.ptr_k_descale == nullptr ? 1.0f : params.ptr_k_descale[bidb * get<0>(params.stride_k_descale) + bidh_kv * get<1>(params.stride_k_descale)];
            softcap_val *= q_descale * k_descale;
        }
        auto scoremod_premask_fn = [&](auto& tSrS) {
            if constexpr (Has_softcap) { flash::apply_softcap(tSrS, softcap_val); }
        };
        auto write_P_to_smem = [&](auto& tOrP) {
            if constexpr (LargeHeadDimV) {
                cutlass::arch::NamedBarrier::sync(NumMmaThreads, static_cast<uint32_t>(FwdNamedBarriers::PEmpty) );
            }
            cute::copy(smem_tiled_copy_P, smem_thr_copy_P.retile_S(tOrP), tPsP);
        };
        auto arrive_on_P_write_barrier = [&] {
            cutlass::arch::fence_view_async_shared();
            __syncwarp();
            if constexpr (LargeHeadDimV) {
                cutlass::arch::NamedBarrier::arrive(NumMmaThreads, static_cast<uint32_t>(FwdNamedBarriers::PFull) );
            }
        };
        auto &barrier_Q = shared_storage.pipelines.barrier_Q;
        if constexpr (!AppendKV) {
            barrier_Q.wait(work_idx % 2);
        } else {
            if (get<1>(params.shape_rotary) > 0) {
                using Rotary_t = Rotary<kBlockM, kHeadDim, NumMmaThreadsQK, Element, !(Is_causal || Is_local) >;
                Rotary_t rotary(params.ptr_rotary_cos, params.shape_rotary, params.stride_rotary_cos,
                                params.ptr_rotary_sin, params.stride_rotary_sin,
                                params.is_rotary_interleaved, thread_idx, seqlen_q,
                                seqlen_info.seqlen_rotary);
                Tensor sQ_pi = cute::as_position_independent_swizzle_tensor(sQ);
                int const qhead_per_khead = !PackGQA ? 1 : params.qhead_per_khead_divmod.divisor;
                if (params.is_rotary_interleaved) {
                    auto [tRrCos, tRrSin] = cute::conditional_return<!PackGQA>(
                        rotary.template load_cos_sin<true >(m_block),
                        rotary.template load_cos_sin_packgqa<true >(m_block, params.qhead_per_khead_divmod)
                    );
                    barrier_Q.wait(work_idx % 2);
                    rotary.apply_Q_interleaved(sQ_pi, tRrCos, tRrSin, m_block, qhead_per_khead);
                } else {
                    auto [tRrCosCont, tRrSinCont] = cute::conditional_return<!PackGQA>(
                        rotary.template load_cos_sin<false >(m_block),
                        rotary.template load_cos_sin_packgqa<false >(m_block, params.qhead_per_khead_divmod)
                    );
                    barrier_Q.wait(work_idx % 2);
                    rotary.apply_Q_contiguous(sQ_pi, tRrCosCont, tRrSinCont, m_block, qhead_per_khead);
                }
                cutlass::arch::fence_view_async_shared();
                cutlass::arch::NamedBarrier::sync(NumMmaThreadsQK, static_cast<uint32_t>(FwdNamedBarriers::QueryRotated) );
            } else {
                barrier_Q.wait(work_idx % 2);
            }
        }
        if constexpr (MmaQK_is_RS) {
            using SmemCopyAtomQ = Copy_Atom<cute::SM75_U32x4_LDSM_N, Element>;
            auto smem_tiled_copy_Q = make_tiled_copy_A(SmemCopyAtomQ{}, tiled_mma_qk);
            auto smem_thr_copy_Q = smem_tiled_copy_Q.get_thread_slice(thread_idx);
            Tensor tSrQ_copy_view = smem_thr_copy_Q.retile_D(tSrQ);
            Tensor tSsQ_copy_view = smem_thr_copy_Q.partition_S(cute::as_position_independent_swizzle_tensor(sQ));
            cute::copy(smem_tiled_copy_Q, tSsQ_copy_view, tSrQ_copy_view);
        }
        if constexpr (IntraWGOverlap) {
          {
            Tensor tSrS = partition_fragment_C(tiled_mma_qk, select<0, 1>(TileShape_MNK{}));
            consumer_wait(pipeline_k, smem_pipe_read);
            if constexpr (!OnlyQv) {
                flash::gemm<true, -1>(tiled_mma_qk, tSrQ, tSrK(_, _, _, smem_pipe_read.index()), tSrS);
                warpgroup_wait<0>();
            }
            pipeline_k.consumer_release(smem_pipe_read);
            if constexpr (HasQv) {
                shared_storage.pipelines.barrier_Qv.wait(work_idx % 2);
                consumer_wait(pipeline_v, smem_pipe_read);
                flash::gemm<OnlyQv, 0>(tiled_mma_qv, tSrQv, tSrV(_, _, _, smem_pipe_read.index()), tSrS);
            }
            scoremod_premask_fn(tSrS);
            if constexpr (Has_sparse_mask) {
                consumer_wait(pipeline_mask, smem_pipe_read);
                uint32_t const* mask_ptr = reinterpret_cast<uint32_t const*>(
                    shared_storage.tensors.mainloop.smem_mask.data() +
                    smem_pipe_read.index() * (kBlockM * kNumInt32PerBlock)
                );
                sparse_mask.template apply<true >(tSrS, m_block, n_block, mask_ptr);
                pipeline_mask.consumer_release(smem_pipe_read);
            } else {
                mask.template apply<true , Is_causal, Is_local>(tSrS, m_block, n_block);
            }
            uint32_t hP[CUTE_STATIC_V(size(tSrS)) / 2];
            xu_sync();
            Tensor scores_scale = softmax.template step<true, true>(tSrS, hP);
            xu_arrive();
            Tensor tOrP_acc = make_tensor(tSrS.data(), flash::convert_layout_acc_Aregs<TiledMmaPV>(tSrS.layout()));
            Tensor tOrP = make_tensor_like<Element>(tOrP_acc);
            softmax.pack_p(hP, tOrP);
            if constexpr (Is_FP8 && V_colmajor) { flash::permute_Aregs_fp8(tOrP); }
            if constexpr (!MmaPV_is_RS) { write_P_to_smem(tOrP); }
            if constexpr (!MmaPV_is_RS) { arrive_on_P_write_barrier(); }
            --n_block;
            clear(tOrO);
            // FP16 P V accumulator for the whole key range, folded into tOrO after the last tile.
            uint32_t u_o16[16];
            #pragma unroll
            for (int j = 0; j < 16; ++j) { u_o16[j] = 0u; }
            float u_sc[2] = {1.f, 1.f};
            int u_cnt = 0;
            bool u_first = true;
            softmax.clear_sum();
            uint64_t const ones_desc = flash::att_ones_desc(shared_storage.tensors.mainloop.smem_ones.data());
            uint32_t const fk_vaddr = static_cast<uint32_t>(__cvta_generic_to_shared(shared_storage.tensors.mainloop.smem_v.data()));
            uint32_t fx_bsync, fx_barv;
            {
                int const cur_wg = flash::canonical_warp_group_idx_nosync() - 1;
                int const next_wg = NumMmaWarpGroups == 2 ? 1 - cur_wg : (cur_wg < NumMmaWarpGroups - 1 ? cur_wg + 1 : 0);
                asm volatile("mov.u32 %0, %1;" : "=r"(fx_bsync) : "r"(uint32_t(static_cast<uint32_t>(FwdNamedBarriers::WarpSchedulerWG1) + cur_wg)));
                asm volatile("mov.u32 %0, %1;" : "=r"(fx_barv) : "r"(uint32_t(static_cast<uint32_t>(FwdNamedBarriers::WarpSchedulerWG1) + next_wg)));
            }
            auto fwd_step2 = [&](int const n_block, auto mask_fn, auto check_inf_type, auto& P_use, auto& P_fill) {
                static constexpr bool Check_inf = decltype(check_inf_type)::value;
                PipelineState smem_pipe_read_v(smem_pipe_read.index(), smem_pipe_read.phase(), smem_pipe_read.count());
                ++smem_pipe_read;
                Tensor tSrS = partition_fragment_C(tiled_mma_qk, select<0, 1>(TileShape_MNK{}));
                ATT_STAMP(tl_on, warp_group_idx, tl_it, 0);
                if (!UseSchedulerBarrier || warp_group_idx == 0) { consumer_wait(pipeline_k, smem_pipe_read); }
                ATT_STAMP(tl_on, warp_group_idx, tl_it, 1);
                if constexpr (UseSchedulerBarrier) { cutlass::arch::NamedBarrier::sync(2 * cutlass::NumThreadsPerWarpGroup, fx_bsync); }
                if constexpr (!OnlyQv && (ATT_CABL & 2) == 0) {
                    flash::gemm<true, -1>(tiled_mma_qk, tSrQ, tSrK(_, _, _, smem_pipe_read.index()), tSrS);
                }
                ATT_STAMP(tl_on, warp_group_idx, tl_it, 7);
                if constexpr (RescaleOBeforeGemm) { softmax.rescale_o(tOrO, scores_scale); }
                if constexpr(!HasQv) {
                    if (!UseSchedulerBarrier || warp_group_idx == 0) { consumer_wait(pipeline_v, smem_pipe_read_v); }
                }
                if constexpr ((ATT_CABL & 1) == 0) {
                fk_pv16<kBlockN>(P_use, u_o16, fk_vaddr + smem_pipe_read_v.index() * (72 * kBlockN));
                }
                if constexpr (UseSchedulerBarrier) { cutlass::arch::NamedBarrier::arrive(2 * cutlass::NumThreadsPerWarpGroup, fx_barv); }
                ATT_STAMP(tl_on, warp_group_idx, tl_it, 2);
                warpgroup_wait<0>();
                ATT_STAMP(tl_on, warp_group_idx, tl_it, 3);
                pipeline_k.consumer_release(smem_pipe_read);
                if constexpr (HasQv) {
                    warpgroup_wait<0>();
                    pipeline_v.consumer_release(smem_pipe_read_v);
                    consumer_wait(pipeline_v, smem_pipe_read);
                    flash::gemm<OnlyQv, 0>(tiled_mma_qv, tSrQv, tSrV(_, _, _, smem_pipe_read.index()), tSrS);
                }
                scoremod_premask_fn(tSrS);
                if constexpr (Has_sparse_mask) {
                    consumer_wait(pipeline_mask, smem_pipe_read);
                }
                mask_fn(tSrS, n_block);
                if constexpr (Has_sparse_mask) {
                    pipeline_mask.consumer_release(smem_pipe_read);
                }
                uint32_t hP[CUTE_STATIC_V(size(tSrS)) / 2];
                xu_sync();
                cute::copy(softmax.template step<false, Check_inf>(tSrS, hP), scores_scale);
                xu_arrive();
                ATT_STAMP(tl_on, warp_group_idx, tl_it, 4);
                warpgroup_wait<0>();
                ATT_STAMP(tl_on, warp_group_idx, tl_it, 5);
                pipeline_v.consumer_release(smem_pipe_read_v);
                softmax.pack_p(hP, P_fill);
                if constexpr (Is_FP8 && V_colmajor) { flash::permute_Aregs_fp8(tOrP); }
                if constexpr (!MmaPV_is_RS) { write_P_to_smem(tOrP); }
                u_rescale16(u_o16, scores_scale);
                softmax.scale_sum(scores_scale);
                if constexpr (!MmaPV_is_RS) { arrive_on_P_write_barrier(); }
                ATT_STAMP(tl_on, warp_group_idx, tl_it, 6);
            };
            auto fwd_loop = [&](int const lo, auto mask_fn, auto check_inf_type) {
                #pragma unroll 1
                for (; n_block >= lo; --n_block) { fwd_step2(n_block, mask_fn, check_inf_type, tOrP, tOrP); }
            };
            auto fwd_step = [&](int const n_blk, auto mask_fn, auto check_inf_type) {
                fwd_step2(n_blk, mask_fn, check_inf_type, tOrP, tOrP);
            };
            if constexpr (Has_sparse_mask) {
                auto sparse_mask_fn = [&](auto& tSrS, int n_block) {
                    uint32_t const* mask_ptr = reinterpret_cast<uint32_t const*>(
                        shared_storage.tensors.mainloop.smem_mask.data() +
                        smem_pipe_read.index() * (kBlockM * kNumInt32PerBlock)
                    );
                    sparse_mask.template apply<false >(tSrS, m_block, n_block, mask_ptr);
                };
                #pragma unroll 1
                for (; n_block >= n_block_min; --n_block) {
                    fwd_step(n_block, sparse_mask_fn, cute::true_type{} );
                }
            } else {
                if constexpr (Is_causal || Is_local) {
                    auto mask_fn = [&](auto& tSrS, int n_block) { mask.template apply<false , Is_causal, Is_local>(tSrS, m_block, n_block); };
                    int n_block_min_causal_local_mask;
                    {
                        int const n_idx_cm = m_idx_min + seqlen_k - seqlen_q;
                        int n_idx_right = !Is_local ? n_idx_cm : n_idx_cm + params.window_size_right;
                        if constexpr (Is_local) { if (params.attention_chunk_divmod.divisor > 0) { n_idx_right = std::min(n_idx_right, flash::round_up(params.attention_chunk_divmod, n_idx_cm)); } }
                        n_block_min_causal_local_mask = std::max(n_block_min, n_idx_right / kBlockN);
                    }
                    fwd_loop(n_block_min_causal_local_mask, mask_fn, cute::true_type{} );
                }
                int n_block_min_before_local_mask;
                {
                    int const n_idx_bm = m_idx_max + seqlen_k - seqlen_q;
                    int n_idx_left = !Is_local ? n_idx_bm : n_idx_bm - params.window_size_left;
                    if constexpr (Is_local) { if (params.attention_chunk_divmod.divisor > 0) { n_idx_left = std::max(n_idx_left, flash::round_down(params.attention_chunk_divmod, n_idx_bm)); } }
                    n_block_min_before_local_mask = !Is_local ? n_block_min : std::max(n_block_min, cute::ceil_div(n_idx_left, kBlockN));
                }
                auto no_mask_fn = [](auto& tSrS, int n_block) { };
                fwd_loop(n_block_min_before_local_mask, no_mask_fn, cute::false_type{} );
                if constexpr (Is_local) {
                    auto local_mask_fn = [&](auto& tSrS, int n_block) { mask.template apply<false , false , Is_local>(tSrS, m_block, n_block); };
                    fwd_loop(n_block_min, local_mask_fn, cute::bool_constant<Is_local>{} );
                }
            }
            cutlass::arch::NamedBarrier::arrive(NumMmaThreadsQK + (Use_TMA_Q ? cutlass::NumThreadsPerWarp : NumProducerThreads), static_cast<uint32_t>(FwdNamedBarriers::QueryEmpty) );
            if constexpr (RescaleOBeforeGemm) { softmax.rescale_o(tOrO, scores_scale); }
            if constexpr (!HasQv) { consumer_wait(pipeline_v, smem_pipe_read); }
            fk_pv16<kBlockN>(tOrP, u_o16, fk_vaddr + smem_pipe_read.index() * (72 * kBlockN));
            float const v_descale = !Is_FP8 || params.ptr_v_descale == nullptr ? 1.0f : params.ptr_v_descale[bidb * get<0>(params.stride_v_descale) + bidh_kv * get<1>(params.stride_v_descale)];
            if constexpr (LargeHeadDimV) {
                cutlass::arch::NamedBarrier::sync(NumMmaThreads, static_cast<uint32_t>(FwdNamedBarriers::PEmpty) );
                store_scales(scores_scale, smem_pipe_read.index());
                cutlass::arch::NamedBarrier::arrive(NumMmaThreads, static_cast<uint32_t>(FwdNamedBarriers::PFull) );
            }
            warpgroup_wait<0>();
            pipeline_v.consumer_release(smem_pipe_read);
            cute::copy(softmax.finalize(v_descale), scores_scale);
            u_fold(tOrO, u_o16, u_sc, u_first);
            softmax.rescale_o(tOrO, scores_scale);
            if constexpr (Is_FP8 && !V_colmajor) { flash::permute_output_fp8(tOrO); }
            ++smem_pipe_read;
          }
        } else {
            warp_scheduler_barrier_sync();
            auto fwd_step = [&](int const n_block, auto mask_fn, auto is_first_iter_type, auto check_inf_type) {
                static constexpr bool Is_first_iter = decltype(is_first_iter_type)::value;
                static constexpr bool Check_inf = decltype(check_inf_type)::value;
                auto smem_pipe_read_prev = smem_pipe_read;
                if constexpr (!Is_first_iter) { ++smem_pipe_read; }
                Tensor tSrS = partition_fragment_C(tiled_mma_qk, select<0, 1>(TileShape_MNK{}));
                consumer_wait(pipeline_k, smem_pipe_read);
                if constexpr (!OnlyQv) {
                    flash::gemm<true, -1>(tiled_mma_qk, tSrQ, tSrK(_, _, _, smem_pipe_read.index()), tSrS);
                }
                if constexpr (!HasQv) {
                    warp_scheduler_barrier_arrive();
                    warpgroup_wait<0>();
                    pipeline_k.consumer_release(smem_pipe_read);
                } else {
                    if constexpr (Is_first_iter) {
                        shared_storage.pipelines.barrier_Qv.wait(work_idx % 2);
                    }
                    consumer_wait(pipeline_v, smem_pipe_read);
                    flash::gemm<OnlyQv, -1>(tiled_mma_qv, tSrQv, tSrV(_, _, _, smem_pipe_read.index()), tSrS);
                    warp_scheduler_barrier_arrive();
                    warpgroup_wait<1>();
                    pipeline_k.consumer_release(smem_pipe_read);
                    warpgroup_wait<0>();
                }
                scoremod_premask_fn(tSrS);
                if constexpr (Has_sparse_mask) {
                    consumer_wait(pipeline_mask, smem_pipe_read);
                }
                mask_fn(tSrS, n_block);
                if constexpr (Has_sparse_mask) {
                    pipeline_mask.consumer_release(smem_pipe_read);
                }
                Tensor scores_scale = softmax.template max_get_scale<Is_first_iter, Check_inf>(tSrS);
                if constexpr (LargeHeadDimV && !Is_first_iter) { store_scales(scores_scale, smem_pipe_read_prev.index()); }
                softmax.template online_softmax<Is_first_iter, Check_inf>(tSrS);
                if constexpr (Is_FP8 && !V_colmajor) { flash::permute_Cregs_fp8(tSrS); }
                Tensor tOrP_acc = make_tensor(tSrS.data(), flash::convert_layout_acc_Aregs<TiledMmaPV>(tSrS.layout()));
                Tensor tOrP = make_tensor_like<Element>(tOrP_acc);
                convert_type_out(tOrP_acc, tOrP);
                if constexpr (Is_FP8 && V_colmajor) { flash::permute_Aregs_fp8(tOrP); }
                if constexpr (!MmaPV_is_RS) { write_P_to_smem(tOrP); }
                if constexpr (!Is_first_iter) { softmax.rescale_o(tOrO, scores_scale); }
                if constexpr (!MmaPV_is_RS && !MmaPV_use_RS_WG1) { arrive_on_P_write_barrier(); }
                if constexpr (!HasQv) { consumer_wait(pipeline_v, smem_pipe_read); }
                warp_scheduler_barrier_sync();
                if constexpr (!MmaPV_use_RS_WG1) {
                    flash::gemm<Is_first_iter, -1>(tiled_mma_pv, cute::conditional_return<MmaPV_is_RS>(tOrP, tOsP), tOrV(_, _, _, smem_pipe_read.index()), tOrO);
                } else {
                    TiledMmaPV_RS tiled_mma_pv_rs;
                    flash::gemm<Is_first_iter, -1>(tiled_mma_pv_rs, tOrP, tOrV(_, _, _, smem_pipe_read.index()), tOrO);
                }
                if constexpr (!MmaPV_is_RS && MmaPV_use_RS_WG1) { arrive_on_P_write_barrier(); }
                warpgroup_wait<0>();
                pipeline_v.consumer_release(smem_pipe_read);
            };
            auto first_iter_mask_fn = [&](auto& tSrS, int n_block) {
                if constexpr (Has_sparse_mask) {
                    uint32_t const* mask_ptr = reinterpret_cast<uint32_t const*>(
                        shared_storage.tensors.mainloop.smem_mask.data() +
                        smem_pipe_read.index() * (kBlockM * kNumInt32PerBlock)
                    );
                    sparse_mask.template apply<true >(tSrS, m_block, n_block, mask_ptr);
                } else {
                    mask.template apply<true , Is_causal, Is_local>(tSrS, m_block, n_block);
                }
            };
            fwd_step(n_block, first_iter_mask_fn, cute::true_type{} , cute::true_type{} );
            --n_block;
            if constexpr (Is_causal || Is_local) {
                auto mask_fn = [&](auto& tSrS, int n_block) {
                    if constexpr (Has_sparse_mask) {
                        uint32_t const* mask_ptr = reinterpret_cast<uint32_t const*>(
                            shared_storage.tensors.mainloop.smem_mask.data() +
                            smem_pipe_read.index() * (kBlockM * kNumInt32PerBlock)
                        );
                        sparse_mask.template apply<false >(tSrS, m_block, n_block, mask_ptr);
                    } else {
                        mask.template apply<false , Is_causal, Is_local>(tSrS, m_block, n_block);
                    }
                };
                int n_block_min_causal_local_mask;
                {
                    int const n_idx_cm = m_idx_min + seqlen_k - seqlen_q;
                    int n_idx_right = !Is_local ? n_idx_cm : n_idx_cm + params.window_size_right;
                    if constexpr (Is_local) { if (params.attention_chunk_divmod.divisor > 0) { n_idx_right = std::min(n_idx_right, flash::round_up(params.attention_chunk_divmod, n_idx_cm)); } }
                    n_block_min_causal_local_mask = std::max(n_block_min, n_idx_right / kBlockN);
                }
                #pragma unroll 1
                for (; n_block >= n_block_min_causal_local_mask; --n_block) {
                    fwd_step(n_block, mask_fn, cute::false_type{} , cute::true_type{} );
                }
            }
            int n_block_min_before_local_mask;
            {
                int const n_idx_bm = m_idx_max + seqlen_k - seqlen_q;
                int n_idx_left = !Is_local ? n_idx_bm : n_idx_bm - params.window_size_left;
                if constexpr (Is_local) { if (params.attention_chunk_divmod.divisor > 0) { n_idx_left = std::max(n_idx_left, flash::round_down(params.attention_chunk_divmod, n_idx_bm)); } }
                n_block_min_before_local_mask = !Is_local ? n_block_min : std::max(n_block_min, cute::ceil_div(n_idx_left, kBlockN));
            }
            auto no_mask_fn = [](auto& tSrS, int n_block) { };
            #pragma unroll 1
            for (; n_block >= n_block_min_before_local_mask; --n_block) {
                fwd_step(n_block, no_mask_fn, cute::false_type{} , cute::false_type{} );
            }
            if constexpr (Is_local) {
                auto local_mask_fn = [&](auto& tSrS, int n_block) { mask.template apply<false , false , Is_local>(tSrS, m_block, n_block); };
                #pragma unroll 1
                for (; n_block >= n_block_min; --n_block) {
                    fwd_step(n_block, local_mask_fn, cute::false_type{} , cute::bool_constant<Is_local>{} );
                }
            }
            warp_scheduler_barrier_arrive();
            cutlass::arch::NamedBarrier::arrive(NumMmaThreadsQK + (Use_TMA_Q ? cutlass::NumThreadsPerWarp : NumProducerThreads), static_cast<uint32_t>(FwdNamedBarriers::QueryEmpty) );
            float const v_descale = !Is_FP8 || params.ptr_v_descale == nullptr ? 1.0f : params.ptr_v_descale[bidb * get<0>(params.stride_v_descale) + bidh_kv * get<1>(params.stride_v_descale)];
            Tensor scores_scale = softmax.finalize(v_descale);
            if constexpr (LargeHeadDimV) {
                cutlass::arch::NamedBarrier::sync(NumMmaThreads, static_cast<uint32_t>(FwdNamedBarriers::PEmpty) );
                store_scales(scores_scale, smem_pipe_read.index());
                cutlass::arch::NamedBarrier::arrive(NumMmaThreads, static_cast<uint32_t>(FwdNamedBarriers::PFull) );
            }
            softmax.rescale_o(tOrO, scores_scale);
            if constexpr (Is_FP8 && !V_colmajor) { flash::permute_output_fp8(tOrO); }
            ++smem_pipe_read;
        }
        ++work_idx;
        return true;
    }
    template <typename SharedStorage, typename FrgTensorO, typename Softmax>
    CUTLASS_DEVICE bool
    mma_pv(Params const& params,
           MainloopPipelineV pipeline_v,
           PipelineState& smem_pipe_read,
           FrgTensorO& tOrO,
           Softmax& softmax,
           int const thread_idx,
           SeqlenInfo_t const& seqlen_info,
           cute::tuple<int32_t, int32_t, int32_t, int32_t> block_coord,
           SharedStorage& shared_storage
           ) {
        static_assert(is_rmem<FrgTensorO>::value, "O tensor must be rmem resident.");
        int const m_block = get<0>(block_coord);
        int const bidb = get<2>(block_coord);
        int const split_idx = get<3>(block_coord);
        auto [n_block_min, n_block_max, n_offset_pv] = BlockMN_t::get_n_block_min_max(
            seqlen_info, m_block, bidb, split_idx, params.num_splits,
            params.window_size_left, params.window_size_right, params.attention_chunk_divmod,
            params.qhead_per_khead_divmod);
        if constexpr (Is_causal || Is_local || Varlen || Split) {
            if (n_block_max <= n_block_min) { return false; }
        }
        Tensor sV = make_tensor(make_smem_ptr(shared_storage.tensors.mainloop.smem_v.data()), SmemLayoutVtMma{});
        Tensor sP = make_tensor(make_smem_ptr(shared_storage.tensors.mainloop.smem_p.data()), SmemLayoutP{});
        Tensor sScale = make_tensor(make_smem_ptr(shared_storage.tensors.mainloop.smem_scale.data()), SmemLayoutScale{});
        static constexpr int MmaWarpGroups = size(TiledMmaPV{}) / cutlass::NumThreadsPerWarpGroup;
        Layout warp_group_thread_layout = make_layout(make_shape(Int<MmaWarpGroups>{}),
                                                      make_stride(Int<cutlass::NumThreadsPerWarpGroup>{}));
        int warp_group_idx = __shfl_sync(0xFFFFFFFF, thread_idx / cutlass::NumThreadsPerWarpGroup, 0);
        TiledMmaPV tiled_mma_pv;
        auto wg_mma_pv = tiled_mma_pv.get_slice(warp_group_thread_layout(warp_group_idx));
        Tensor tOrV = wg_mma_pv.partition_fragment_B(sV);
        Tensor tOsP = wg_mma_pv.partition_fragment_A(sP);
        auto thread_mma_pv = tiled_mma_pv.get_thread_slice(thread_idx % cutlass::NumThreadsPerWarpGroup);
        Tensor taccOcO = thread_mma_pv.partition_C(cute::make_identity_tensor(select<0, 1>(TileShape_MNK_PV{})));
        Tensor taccOcO_rowcol = make_tensor(taccOcO.data(), flash::convert_layout_acc_rowcol(taccOcO.layout()));
        Tensor taccOcO_row = taccOcO_rowcol(_, _0{});
        auto load_scales = [&](auto& scales, int stage) {
            static_assert(CUTE_STATIC_V(size(scales)) == CUTE_STATIC_V(size(taccOcO_row)));
            #pragma unroll
            for (int mi = 0; mi < size(taccOcO_row); ++mi) {
                scales(mi) = sScale(get<0>(taccOcO_row(mi)), stage);
            }
        };
        typename Softmax::TensorT scores_scale;
        int n_block = n_block_max - 1;
        if constexpr (!HasQv) { pipeline_v.consumer_wait(smem_pipe_read); }
        cutlass::arch::NamedBarrier::sync(NumMmaThreads, static_cast<uint32_t>(FwdNamedBarriers::PFull) );
        flash::gemm<true, 0>(tiled_mma_pv, tOsP, tOrV(_, _, _, smem_pipe_read.index()), tOrO);
        cutlass::arch::NamedBarrier::arrive(NumMmaThreads, static_cast<uint32_t>(FwdNamedBarriers::PEmpty) );
        pipeline_v.consumer_release(smem_pipe_read);
        --n_block;
        #pragma unroll 1
        for (; n_block >= n_block_min; --n_block) {
            cutlass::arch::NamedBarrier::sync(NumMmaThreads, static_cast<uint32_t>(FwdNamedBarriers::PFull) );
            load_scales(scores_scale, smem_pipe_read.index());
            softmax.rescale_o(tOrO, scores_scale);
            ++smem_pipe_read;
            if constexpr (!HasQv) {
                auto barrier_token = pipeline_v.consumer_try_wait(smem_pipe_read);
                pipeline_v.consumer_wait(smem_pipe_read, barrier_token);
            }
            flash::gemm<false, 0>(tiled_mma_pv, tOsP, tOrV(_, _, _, smem_pipe_read.index()), tOrO);
            cutlass::arch::NamedBarrier::arrive(NumMmaThreads, static_cast<uint32_t>(FwdNamedBarriers::PEmpty) );
            pipeline_v.consumer_release(smem_pipe_read);
        };
        cutlass::arch::NamedBarrier::sync(NumMmaThreads, static_cast<uint32_t>(FwdNamedBarriers::PFull) );
        load_scales(scores_scale, smem_pipe_read.index());
        cutlass::arch::NamedBarrier::arrive(NumMmaThreads, static_cast<uint32_t>(FwdNamedBarriers::PEmpty) );
        softmax.rescale_o(tOrO, scores_scale);
        if constexpr (Is_FP8 && !V_colmajor) { flash::permute_output_fp8(tOrO); }
        ++smem_pipe_read;
        return true;
    }
    template <typename SharedStorage>
    CUTLASS_DEVICE bool
    load_kv_new(Params const& params,
         MainloopPipelineKVNew pipeline_k_new,
         MainloopPipelineKVNew pipeline_v_new,
         PipelineState& smem_pipe_write,
         SharedStorage &shared_storage,
         SeqlenInfo_t const& seqlen_info,
         cute::tuple<int32_t, int32_t, int32_t, int32_t> block_coord,
         int const work_idx
         ) {
        auto [m_block, bidh, bidb, split_idx] = block_coord;
        auto [n_block_new_min, n_block_new_max] = BlockMN_t::get_n_block_k_new_min_max(
            seqlen_info, m_block, bidb, split_idx, params.num_splits,
            params.window_size_left, params.window_size_right, params.attention_chunk_divmod,
            params.qhead_per_khead_divmod);
        if (n_block_new_max <= n_block_new_min) { return false; }
        Tensor sK = make_tensor(make_smem_ptr(shared_storage.tensors.mainloop.smem_k.data()), SmemLayoutK{});
        Tensor sVt = [&] {
            if constexpr (!Transpose_V) {
                return make_tensor(make_smem_ptr(shared_storage.tensors.mainloop.smem_v.data()), SmemLayoutVt{});
            } else {
                return make_tensor(make_smem_ptr(shared_storage.tensors.mainloop.smem_vt.data()), SmemLayoutVt{});
            }
        }();
        int const bidh_kv = !PackGQA ? params.qhead_per_khead_divmod.divide(bidh) : bidh;
        uint32_t block_rank_in_cluster = cute::block_rank_in_cluster();
        constexpr uint32_t cluster_shape_x = get<0>(ClusterShape());
        uint2 cluster_local_block_id = {block_rank_in_cluster % cluster_shape_x, block_rank_in_cluster / cluster_shape_x};
        bool const is_varlen_k_new = Varlen && params.cu_seqlens_k_new;
        Tensor mKnew_TMA = params.tma_load_K_new.get_tma_tensor(params.shape_K_new)(_, _, bidh_kv, !is_varlen_k_new ? bidb : 0);
        auto shape_Vnew = make_shape(params.headdim_v, get<0>(params.shape_K_new), get<2>(params.shape_K_new), get<3>(params.shape_K_new));
        Tensor mVnewt_TMA = params.tma_load_V_new.get_tma_tensor(shape_Vnew)(_, _, bidh_kv, !is_varlen_k_new ? bidb : 0);
        Tensor gKnew_TMA = local_tile(domain_offset(make_coord(seqlen_info.offset_k_new, _0{}), mKnew_TMA), select<1, 2>(TileShape_MNK{}), make_coord(_, _0{}));
        Tensor gVnewt_TMA = local_tile(domain_offset(make_coord(_0{}, seqlen_info.offset_k_new), mVnewt_TMA), select<1, 2>(TileShape_MNK_PV{}), make_coord(_0{}, _));
        auto block_tma_K_new = params.tma_load_K_new.get_slice(cluster_local_block_id.x);
        Tensor tKgKnew_TMA = group_modes<0, 3>(block_tma_K_new.partition_S(gKnew_TMA));
        Tensor tKsK_TMA = group_modes<0, 3>(block_tma_K_new.partition_D(sK));
        auto block_tma_V_new = params.tma_load_V_new.get_slice(cluster_local_block_id.x);
        Tensor tVgVnewt_TMA = group_modes<0, 3>(block_tma_V_new.partition_S(gVnewt_TMA));
        Tensor tVsVt_TMA = group_modes<0, 3>(block_tma_V_new.partition_D(sVt));
        uint16_t mcast_mask_kv = 0;
        if constexpr (cute::is_same_v<GmemTiledCopyKV, SM90_TMA_LOAD_MULTICAST>) {
            auto block_layout = Layout<ClusterShape>{};
            for (int m = 0; m < size<0>(block_layout); ++m) {
                mcast_mask_kv |= (uint16_t(1) << block_layout(m, cluster_local_block_id.y, _0{}));
            }
        }
        auto load_K_new = [&] (int const n_block, auto const& smem_pipe_write) {
            pipeline_k_new.producer_acquire(smem_pipe_write);
            copy(params.tma_load_K_new.with(*pipeline_k_new.producer_get_barrier(smem_pipe_write), mcast_mask_kv, TMA::CacheHintSm90::EVICT_FIRST),
                tKgKnew_TMA(_, n_block), tKsK_TMA(_, smem_pipe_write.index()));
        };
        auto load_V_new = [&] (int const n_block, auto const& smem_pipe_write) {
            pipeline_v_new.producer_acquire(smem_pipe_write);
            copy(params.tma_load_V_new.with(*pipeline_v_new.producer_get_barrier(smem_pipe_write), mcast_mask_kv, TMA::CacheHintSm90::EVICT_FIRST),
                tVgVnewt_TMA(_, n_block), tVsVt_TMA(_, smem_pipe_write.index()));
        };
        int warp_idx_in_warpgroup = __shfl_sync(0xffffffff, (threadIdx.x / 32) % 4, 0);
        static constexpr bool SingleProducerWarp = NumProducerThreads == cutlass::NumThreadsPerWarp;
        bool should_load_KV = (SingleProducerWarp || warp_idx_in_warpgroup == 0) && cute::elect_one_sync();
        int n_block = n_block_new_max - 1;
        shared_storage.pipelines.barrier_O.wait((work_idx + 1) % 2);
        if (should_load_KV) { load_K_new(n_block, smem_pipe_write); }
        if (should_load_KV) { load_V_new(n_block, smem_pipe_write); }
        ++smem_pipe_write;
        --n_block;
        #pragma unroll 1
        for (; n_block >= n_block_new_min; --n_block) {
            if (should_load_KV) {
                load_K_new(n_block, smem_pipe_write);
                load_V_new(n_block, smem_pipe_write);
            }
            ++smem_pipe_write;
        }
        return true;
    }
    template <typename SharedStorage>
    CUTLASS_DEVICE bool
    store_kv_new(Params const& params,
                 MainloopPipelineKVNew pipeline_k_new,
                 MainloopPipelineKVNew pipeline_v_new,
                 PipelineState& smem_pipe_read,
                 int const thread_idx,
                 SharedStorage &shared_storage,
                 SeqlenInfo_t const& seqlen_info,
                 cute::tuple<int32_t, int32_t, int32_t, int32_t> block_coord
    ) {
        auto [m_block, bidh, bidb, split_idx] = block_coord;
        auto [n_block_new_min, n_block_new_max] = BlockMN_t::get_n_block_k_new_min_max(
            seqlen_info, m_block, bidb, split_idx, params.num_splits,
            params.window_size_left, params.window_size_right, params.attention_chunk_divmod,
            params.qhead_per_khead_divmod);
        if (n_block_new_max <= n_block_new_min) { return false; }
        Tensor sK = cute::as_position_independent_swizzle_tensor(make_tensor(make_smem_ptr(shared_storage.tensors.mainloop.smem_k.data()), SmemLayoutK{}));
        Tensor sV = [&] {
            if constexpr (!Transpose_V) {
                return cute::as_position_independent_swizzle_tensor(make_tensor(make_smem_ptr(shared_storage.tensors.mainloop.smem_v.data()), SmemLayoutVCpAsync{}));
            } else {
                return cute::as_position_independent_swizzle_tensor(make_tensor(make_smem_ptr(shared_storage.tensors.mainloop.smem_vt.data()), SmemLayoutVCpAsync{}));
            }
        }();
        int const bidh_kv = !PackGQA ? params.qhead_per_khead_divmod.divide(bidh) : bidh;
        int const bidb_kv = params.kv_batch_idx == nullptr ? bidb : params.kv_batch_idx[bidb];
        bool const is_varlen_k = Varlen && params.cu_seqlens_k;
        Tensor mK = make_tensor(make_gmem_ptr(params.ptr_K), params.shape_K, params.stride_K)(_, _, bidh_kv, !is_varlen_k ? bidb_kv : 0);
        auto shape_V = make_shape(params.headdim_v, get<0>(params.shape_K), get<2>(params.shape_K), get<3>(params.shape_K));
        Tensor mV = make_tensor(make_gmem_ptr(params.ptr_V), shape_V, params.stride_V)(_, _, bidh_kv, !is_varlen_k ? bidb_kv : 0);
        int const offset_k = seqlen_info.offset_k + seqlen_info.seqlen_k_og;
        Tensor gK = local_tile(domain_offset(make_coord(offset_k, _0{}), mK), select<1, 2>(TileShape_MNK{}), make_coord(_, _0{}));
        Tensor gV = local_tile(domain_offset(make_coord(offset_k, _0{}), mV), select<2, 1>(TileShape_MNK_PV{}), make_coord(_, _0{}));
        static constexpr int kBlockN = get<1>(TileShape_MNK{});
        static constexpr int kHeadDim = get<2>(TileShape_MNK{});
        int const seqlen_k_new = seqlen_info.seqlen_k_new;
        using Rotary_t = Rotary<kBlockN, kHeadDim, NumMmaThreads, Element>;
        Rotary_t rotary(params.ptr_rotary_cos, params.shape_rotary, params.stride_rotary_cos,
                        params.ptr_rotary_sin, params.stride_rotary_sin,
                        params.is_rotary_interleaved, thread_idx, seqlen_k_new,
                        seqlen_info.seqlen_rotary);
        int const bidb_kv_idx = !is_varlen_k && !params.ptr_pagetable ? bidb_kv : 0;
        using PagedKVManager_t = PagedKVManager<get<1>(TileShape_MNK{}), get<2>(TileShape_MNK{}), get<1>(TileShape_MNK_PV{}), NumMmaThreads, Element, true , 2 >;
        PagedKVManager_t paged_kv_manager(
            params.ptr_pagetable, params.shape_pagetable, params.stride_pagetable,
            params.ptr_K, params.shape_K, params.stride_K,
            params.ptr_V, params.headdim_v, att_paged_stride_v(params),
            params.page_size_divmod, params.blockN_per_page_size_divmod,
            bidb_kv, bidh_kv, thread_idx, seqlen_k_new, offset_k, bidb_kv_idx
        );
        if constexpr (UseSchedulerBarrier) {
            if (flash::canonical_warp_group_idx_nosync() == 1) {
                cutlass::arch::NamedBarrier::sync(2 * cutlass::NumThreadsPerWarpGroup, static_cast<uint32_t>(FwdNamedBarriers::WarpSchedulerWG1) );
            }
        }
        static_assert(std::is_same_v<GmemLayoutAtom, typename Rotary_t::LayoutAtom>);
        static_assert(!PagedKVNonTMA || std::is_same_v<GmemLayoutAtom, typename PagedKVManager_t::GmemLayoutAtomKVCpAsync>);
        GmemTiledCopyAppendKV gmem_tiled_copy_kv;
        auto gmem_thr_copy_kv = gmem_tiled_copy_kv.get_thread_slice(thread_idx);
        Tensor tKsK = gmem_thr_copy_kv.partition_S(sK);
        Tensor tKgK = gmem_thr_copy_kv.partition_D(gK);
        Tensor tVsV = gmem_thr_copy_kv.partition_S(sV);
        Tensor tVgV = gmem_thr_copy_kv.partition_D(gV);
        Tensor cK = cute::make_identity_tensor(select<1, 2>(TileShape_MNK{}));
        Tensor tKcK = gmem_thr_copy_kv.partition_D(cK);
        Tensor tKpK = make_tensor<bool>(make_shape(size<2>(tKsK)));
        #pragma unroll
        for (int k = 0; k < size(tKpK); ++k) { tKpK(k) = get<1>(tKcK(_0{}, _0{}, k)) < get<1>(params.shape_K); }
        Tensor cV = cute::make_identity_tensor(select<2, 1>(TileShape_MNK_PV{}));
        Tensor tVcV = cute::conditional_return<SameHeadDim>(tKcK, gmem_thr_copy_kv.partition_D(cV));
        Tensor tVpV_ = make_tensor<bool>(make_shape(size<2>(tVsV)));
        #pragma unroll
        for (int k = 0; k < size(tVpV_); ++k) { tVpV_(k) = get<1>(tVcV(_0{}, _0{}, k)) < params.headdim_v; }
        Tensor tVpV = cute::conditional_return<SameHeadDim>(tKpK, tVpV_);
        auto store_K = [&] (int const n_block, auto const& smem_pipe_read) {
            int const n_limit = std::min(seqlen_k_new - n_block * kBlockN, kBlockN);
            if (get<1>(params.shape_rotary) <= 0) {
                pipeline_k_new.consumer_wait(smem_pipe_read);
                Tensor tKsK_cur = tKsK(_, _, _, smem_pipe_read.index());
                if constexpr (!PagedKVNonTMA) {
                    Tensor tKgK_cur = tKgK(_, _, _, n_block);
                    flash::copy<false, false, false, false>(
                        gmem_tiled_copy_kv, tKsK_cur, tKgK_cur, tKcK, tKpK, std::min(seqlen_k_new - n_block * kBlockN, kBlockN)
                    );
                } else {
                    paged_kv_manager.store_K(n_block, tKsK_cur);
                }
            } else {
                Tensor gK_cur = gK(_, _, n_block);
                auto tPrKPtr = cute::conditional_return<PagedKVNonTMA>(paged_kv_manager.compute_K_ptr(), nullptr);
                if (params.is_rotary_interleaved) {
                    auto [tRrCos, tRrSin] = rotary.template load_cos_sin<true >(n_block);
                    pipeline_k_new.consumer_wait(smem_pipe_read);
                    rotary.template apply_K_interleaved<PagedKVNonTMA>(sK(_, _, smem_pipe_read.index()), gK_cur, tKpK, tRrCos, tRrSin, tPrKPtr, n_block);
                } else {
                    auto [tRrCosCont, tRrSinCont] = rotary.template load_cos_sin<false >(n_block);
                    pipeline_k_new.consumer_wait(smem_pipe_read);
                    rotary.template apply_K_contiguous<PagedKVNonTMA>(sK(_, _, smem_pipe_read.index()), gK_cur, tKpK, tRrCosCont, tRrSinCont, tPrKPtr, n_block, get<1>(params.shape_K));
                }
            }
            cutlass::arch::fence_view_async_shared();
            cutlass::arch::NamedBarrier::sync(cutlass::NumThreadsPerWarpGroup, static_cast<uint32_t>(FwdNamedBarriers::WarpSchedulerWG1) - 1 + flash::canonical_warp_group_idx_nosync() );
            pipeline_k_new.consumer_release(smem_pipe_read);
        };
        auto store_V = [&] (int const n_block, auto const& smem_pipe_read) {
            pipeline_v_new.consumer_wait(smem_pipe_read);
            int const n_limit = std::min(seqlen_k_new - n_block * kBlockN, kBlockN);
            Tensor tVsV_cur = tVsV(_, _, _, smem_pipe_read.index());
            if constexpr (!PagedKVNonTMA) {
                Tensor tVgV_cur = tVgV(_, _, _, n_block);
                flash::copy<false, false, false, false>(
                    gmem_tiled_copy_kv, tVsV_cur, tVgV_cur, tVcV, tVpV, n_limit);
            } else {
                paged_kv_manager.store_V(n_block, tVsV_cur);
            }
            cutlass::arch::fence_view_async_shared();
            cutlass::arch::NamedBarrier::sync(cutlass::NumThreadsPerWarpGroup, static_cast<uint32_t>(FwdNamedBarriers::WarpSchedulerWG1) - 1 + flash::canonical_warp_group_idx_nosync() );
            pipeline_v_new.consumer_release(smem_pipe_read);
        };
        #pragma unroll 1
        for (int n_block = n_block_new_max - 1; n_block >= n_block_new_min; --n_block) {
            if constexpr (PagedKVNonTMA) { paged_kv_manager.template load_page_table<true >(n_block); }
            store_K(n_block, smem_pipe_read);
            store_V(n_block, smem_pipe_read);
            ++smem_pipe_read;
        }
        if constexpr (UseSchedulerBarrier) {
            if (flash::canonical_warp_group_idx_nosync() == 1) {
                cutlass::arch::NamedBarrier::arrive(2 * cutlass::NumThreadsPerWarpGroup, static_cast<uint32_t>(FwdNamedBarriers::WarpSchedulerWG1) );
            }
        }
        return true;
    }
};
}
#include <cutlass/cutlass.h>
#include <cutlass/array.h>
#include <cutlass/numeric_types.h>
#include <cutlass/numeric_conversion.h>
#include "cute/tensor.hpp"
namespace flash {
using namespace cute;
template <int kNWarps, int Stages, bool Q_in_regs, class TileShape_MNK_, int kHeadDimV, class Element_, class ElementAccum_, class ArchTag_,
        bool Is_causal_, bool Is_local_, bool Has_softcap_, bool Varlen_, bool PagedKV_, bool AppendKV_,
        bool PackGQA_, bool Split_, class ElementSink_>
struct CollectiveMainloopFwdSm80 {
    static constexpr int kStages = Stages;
    static_assert(kStages > 0, "kStages must be greater than 0");
    using TileShape_MNK = TileShape_MNK_;
    using TileShape_MNK_PV = Shape<decltype(get<0>(TileShape_MNK{})), Int<kHeadDimV>, decltype(get<1>(TileShape_MNK{}))>;
    using Element = Element_;
    using ElementAccum = ElementAccum_;
    using ElementSink = ElementSink_;
    using ArchTag = ArchTag_;
    static constexpr bool Is_FP8 = cute::is_same_v<Element, cutlass::float_e4m3_t> || cute::is_same_v<Element, cutlass::float_e5m2_t>;;
    static constexpr bool Is_causal = Is_causal_;
    static constexpr bool Is_local = Is_local_;
    static constexpr bool Has_softcap = Has_softcap_;
    static constexpr bool Varlen = Varlen_;
    static constexpr bool PagedKV = PagedKV_;
    static constexpr bool AppendKV = AppendKV_;
    static constexpr bool PackGQA = PackGQA_;
    static constexpr bool Split = Split_;
    static constexpr bool Transpose_V = Is_FP8;
    static_assert(ArchTag::kMinComputeCapability >= 80);
    static constexpr bool Has_cp_async = ArchTag::kMinComputeCapability >= 80;
    static constexpr int kBlockM = get<0>(TileShape_MNK{});
    static constexpr int kBlockN = get<1>(TileShape_MNK{});
    static constexpr int kHeadDim = get<2>(TileShape_MNK{});
    using SeqlenInfo_t = flash::SeqlenInfoQKNewK<Varlen, AppendKV>;
    using BlockMN_t = flash::BlockMN<SeqlenInfo_t, kBlockM, kBlockN, Is_causal, Is_local, PackGQA, Split>;
    using MMA_Atom_Arch = std::conditional_t<
        ArchTag::kMinComputeCapability >= 80,
        std::conditional_t<
            std::is_same_v<Element, cutlass::half_t>,
            MMA_Atom<SM80_16x8x16_F32F16F16F32_TN>,
            MMA_Atom<SM80_16x8x16_F32BF16BF16F32_TN>
        >,
        MMA_Atom<SM75_16x8x8_F32F16F16F32_TN>
    >;
    using TiledMma = TiledMMA<
        MMA_Atom_Arch,
        Layout<Shape<Int<kNWarps>,_1,_1>>,
        Tile<Int<16 * kNWarps>, _16, _16>>;
    static constexpr int NumMmaThreads = size(TiledMma{});
    static constexpr int NumProducerThreads = NumMmaThreads;
    static constexpr int kGmemElemsPerLoad = sizeof(cute::uint128_t) / sizeof(Element);
    static_assert(kHeadDim % kGmemElemsPerLoad == 0, "Headdim must be a multiple of kGmemElemsPerLoad");
    static constexpr int kBytePerRow = kHeadDim * sizeof(Element);
    static constexpr int kBlockKGmem = (kBytePerRow % 128 == 0 ? 128 : (kBytePerRow % 64 == 0 ? 64 : 32)) / sizeof(Element);
    static constexpr int kSwizzle = kBlockKGmem == 128 ? 4 : (kBlockKGmem == 64 ? 3 : (kBlockKGmem == 32 ? 2 : 1));
    static constexpr int kSwizzleBase = sizeof(Element) == 4 ? 2 : (sizeof(Element) == 2 ? 3 : 4);
    using SmemLayoutAtomQKV = decltype(
        composition(Swizzle<kSwizzle, kSwizzleBase, kSwizzleBase>{},
                    Layout<Shape<_8, Int<kBlockKGmem>>,
                           Stride<Int<kBlockKGmem>, _1>>{}));
    using SmemLayoutQ = decltype(tile_to_shape(SmemLayoutAtomQKV{}, select<0, 2>(TileShape_MNK{})));
    using SmemLayoutK = decltype(tile_to_shape(
        SmemLayoutAtomQKV{},
        make_shape(shape<1>(TileShape_MNK{}), shape<2>(TileShape_MNK{}), Int<kStages>{})));
    using SmemLayoutV = decltype(tile_to_shape(
        SmemLayoutAtomQKV{},
        make_shape(shape<1>(TileShape_MNK{}), shape<2>(TileShape_MNK{}), Int<kStages>{})));
    using SmemLayoutVt = decltype(
        composition(SmemLayoutV{},
                    make_ordered_layout(make_shape(shape<2>(TileShape_MNK{}), shape<1>(TileShape_MNK{}), Int<kStages>{}),
                                        Step<_2, _1, _3>{})));
    using SmemCopyAtom = Copy_Atom<SM75_U32x4_LDSM_N, Element>;
    using SmemCopyAtomTransposed = Copy_Atom<SM75_U16x8_LDSM_T, Element>;
    using GmemCopyAtom = Copy_Atom<std::conditional_t<
        Has_cp_async,
        SM80_CP_ASYNC_CACHEGLOBAL_ZFILL<cute::uint128_t>,
        AutoVectorizingCopyWithAssumedAlignment<128>
    >, Element>;
    static constexpr int kGmemThreadsPerRow = kBlockKGmem / kGmemElemsPerLoad;
    static_assert(NumMmaThreads % kGmemThreadsPerRow == 0, "NumMmaThreads must be a multiple of kGmemThreadsPerRow");
    using GmemLayoutAtom = Layout<Shape <Int<NumMmaThreads / kGmemThreadsPerRow>, Int<kGmemThreadsPerRow>>,
                                  Stride<Int<kGmemThreadsPerRow>, _1>>;
    using GmemTiledCopyQKV = decltype(
        make_tiled_copy(GmemCopyAtom{},
                        GmemLayoutAtom{},
                        Layout<Shape<_1, Int<kGmemElemsPerLoad>>>{}));
    static_assert(kBlockM % CUTE_STATIC_V(shape<0>(GmemLayoutAtom{})) == 0);
    static constexpr int kBytePerHalfRow = kHeadDim / 2 * sizeof(Element);
    static constexpr int kBlockKGmemAppend = (kBytePerHalfRow % 128 == 0 ? 128 : (kBytePerHalfRow % 64 == 0 ? 64 : 32)) / sizeof(Element);
    static constexpr int kGmemThreadsPerRowAppend = kBlockKGmemAppend / kGmemElemsPerLoad;
    static_assert(NumMmaThreads % kGmemThreadsPerRowAppend == 0, "NumMmaThreads must be a multiple of kGmemThreadsPerRowAppend");
    static_assert(cutlass::NumThreadsPerWarp % kGmemThreadsPerRowAppend == 0, "kGmemThreadsPerRowAppend must divide NumThreadsPerWarp");
    using GmemLayoutAtomAppend = Layout<Shape <Int<NumMmaThreads / kGmemThreadsPerRowAppend>, Int<kGmemThreadsPerRowAppend>>,
                                        Stride<Int<kGmemThreadsPerRowAppend>, _1>>;
    static_assert(!AppendKV || kBlockM % CUTE_STATIC_V(shape<0>(GmemLayoutAtomAppend{})) == 0, "kBlockM must be a multiple of NumMmaThreads / kGmemThreadsPerRowAppend");
    using GmemTiledCopyAppendKV = decltype(
        make_tiled_copy(Copy_Atom<AutoVectorizingCopyWithAssumedAlignment<128>, Element>{},
                        GmemLayoutAtomAppend{},
                        Layout<Shape<_1, Int<kGmemElemsPerLoad>>>{}));
    using ShapeQKV = cute::Shape<int32_t, int32_t, int32_t, int32_t>;
    using StrideQK = cute::Stride<int64_t, _1, int64_t, int64_t>;
    using StrideV = StrideQK;
    using ShapeQPacked = std::conditional_t<!PackGQA, ShapeQKV, cute::Shape<cute::Shape<int32_t, int32_t>, int32_t, int32_t, int32_t>>;
    using StrideQPacked = std::conditional_t<!PackGQA, StrideQK, cute::Stride<cute::Stride<int64_t, int64_t>, _1, int64_t, int64_t>>;
    using ShapePageTable = cute::Shape<int32_t, int32_t>;
    using StridePageTable = cute::Stride<int64_t, _1>;
    using ShapeRotary = cute::Shape<int32_t, int32_t>;
    using StrideRotary = cute::Stride<int64_t, _1>;
    using StrideDescale = cute::Stride<int64_t, int64_t>;
    static constexpr bool Share_QV_Smem = Q_in_regs;
    struct TensorStorageSharedQV : cute::aligned_struct<128> {
        union {
            cute::array_aligned<Element, cute::cosize_v<SmemLayoutV>> smem_v;
            cute::array_aligned<Element, cute::cosize_v<SmemLayoutQ>> smem_q;
        };
        cute::array_aligned<Element, cute::cosize_v<SmemLayoutK>> smem_k;
    };
    struct TensorStorageSeparateQV : cute::aligned_struct<128> {
        cute::array_aligned<Element, cute::cosize_v<SmemLayoutV>> smem_v;
        cute::array_aligned<Element, cute::cosize_v<SmemLayoutK>> smem_k;
        cute::array_aligned<Element, cute::cosize_v<SmemLayoutQ>> smem_q;
    };
    using TensorStorage = std::conditional_t<Share_QV_Smem, TensorStorageSharedQV, TensorStorageSeparateQV>;
    struct Arguments {
        Element const* const ptr_Q;
        ShapeQKV const shape_Q;
        StrideQK const stride_Q;
        Element* const ptr_K;
        ShapeQKV const shape_K;
        StrideQK const stride_K;
        Element* const ptr_V;
        int32_t const headdim_v;
        StrideV const stride_V;
        Element const* const ptr_K_new;
        ShapeQKV const shape_K_new;
        StrideQK const stride_K_new;
        Element const* const ptr_V_new;
        StrideV const stride_V_new;
        Element const* const ptr_Qv;
        StrideQK const stride_Qv;
        Element const* const ptr_rotary_cos;
        ShapeRotary const shape_rotary;
        StrideRotary const stride_rotary_cos;
        Element const* const ptr_rotary_sin;
        StrideRotary const stride_rotary_sin;
        bool const is_rotary_interleaved;
        int const* const ptr_pagetable;
        ShapePageTable const shape_pagetable;
        StridePageTable const stride_pagetable;
        float const softmax_scale;
        float const* ptr_q_descale, *ptr_k_descale, *ptr_v_descale;
        StrideDescale const stride_q_descale, stride_k_descale, stride_v_descale;
        int const window_size_left = -1, window_size_right = -1, attention_chunk = 0;
        float const softcap_val;
        int const num_splits;
        int const* const kv_batch_idx = nullptr;
        int const* const cu_seqlens_q = nullptr;
        int const* const cu_seqlens_k = nullptr;
        int const* const cu_seqlens_k_new = nullptr;
        int const* const seqused_q = nullptr;
        int const* const seqused_k = nullptr;
        int const* const leftpad_k = nullptr;
        int const* const seqlens_rotary = nullptr;
        ElementSink const* const ptr_sink = nullptr;
    };
    struct Params {
        Element const* const ptr_Q;
        ShapeQKV const shape_Q;
        StrideQK const stride_Q;
        ShapeQPacked const shape_Q_packed;
        StrideQPacked const stride_Q_packed;
        Element* const ptr_K;
        ShapeQKV const shape_K;
        StrideQK const stride_K;
        Element* const ptr_V;
        int32_t const headdim_v;
        StrideV const stride_V;
        Element const* const ptr_K_new;
        ShapeQKV const shape_K_new;
        StrideQK const stride_K_new;
        Element const* const ptr_V_new;
        StrideV const stride_V_new;
        Element const* const ptr_rotary_cos;
        ShapeRotary const shape_rotary;
        StrideRotary const stride_rotary_cos;
        Element const* const ptr_rotary_sin;
        StrideRotary const stride_rotary_sin;
        bool const is_rotary_interleaved;
        int const* const ptr_pagetable;
        ShapePageTable const shape_pagetable;
        StridePageTable const stride_pagetable;
        cutlass::FastDivmod page_size_divmod;
        cutlass::FastDivmod qhead_per_khead_divmod;
        float const softmax_scale_log2;
        float const* ptr_q_descale, *ptr_k_descale, *ptr_v_descale;
        StrideDescale const stride_q_descale, stride_k_descale, stride_v_descale;
        float const softcap_val;
        int const window_size_left, window_size_right;
        cutlass::FastDivmod attention_chunk_divmod;
        int const num_splits;
        int const* const kv_batch_idx = nullptr;
        int const* const cu_seqlens_q = nullptr;
        int const* const cu_seqlens_k = nullptr;
        int const* const cu_seqlens_k_new = nullptr;
        int const* const seqused_q = nullptr;
        int const* const seqused_k = nullptr;
        int const* const leftpad_k = nullptr;
        int const* const seqlens_rotary = nullptr;
        ElementSink const* const ptr_sink = nullptr;
    };
    static Params
    to_underlying_arguments(Arguments const& args) {
        int const qhead_per_khead = !PackGQA ? 1 : cute::ceil_div(get<2>(args.shape_Q), get<2>(args.shape_K));
        auto const shape_Q_packed = cute::conditional_return<!PackGQA>(
            args.shape_Q,
            make_shape(make_shape(qhead_per_khead, get<0>(args.shape_Q)), get<1>(args.shape_Q), get<2>(args.shape_K), get<3>(args.shape_Q))
        );
        auto const stride_Q_packed = cute::conditional_return<!PackGQA>(
            args.stride_Q,
            make_stride(make_stride(get<2>(args.stride_Q), get<0>(args.stride_Q)), get<1>(args.stride_Q), get<2>(args.stride_Q) * qhead_per_khead, get<3>(args.stride_Q))
        );
        if (get<1>(args.shape_rotary) > 0) {
            assert(args.ptr_rotary_cos != nullptr && args.ptr_rotary_sin != nullptr);
        }
        assert(args.num_splits >= 1);
        cutlass::FastDivmod attention_chunk_divmod(args.attention_chunk >= 1 ? args.attention_chunk : 1);
        attention_chunk_divmod.divisor = args.attention_chunk;
        return {args.ptr_Q, args.shape_Q, args.stride_Q, shape_Q_packed, stride_Q_packed,
                args.ptr_K, args.shape_K, args.stride_K, args.ptr_V, args.headdim_v, args.stride_V,
                args.ptr_K_new, args.shape_K_new, args.stride_K_new, args.ptr_V_new, args.stride_V_new,
                args.ptr_rotary_cos, args.shape_rotary, args.stride_rotary_cos,
                args.ptr_rotary_sin, args.stride_rotary_sin, args.is_rotary_interleaved,
                args.ptr_pagetable, args.shape_pagetable, args.stride_pagetable,
                cutlass::FastDivmod(int(get<0>(args.shape_K))),
                cutlass::FastDivmod(cute::ceil_div(get<2>(args.shape_Q), get<2>(args.shape_K))),
                !Has_softcap ? float(args.softmax_scale * M_LOG2E) : float(args.softcap_val * M_LOG2E),
                args.ptr_q_descale, args.ptr_k_descale, args.ptr_v_descale,
                args.stride_q_descale, args.stride_k_descale, args.stride_v_descale,
                !Has_softcap ? 0.f : args.softmax_scale / args.softcap_val,
                args.window_size_left, args.window_size_right, attention_chunk_divmod,
                !Split ? 1 : args.num_splits,
                args.kv_batch_idx,
                args.cu_seqlens_q, args.cu_seqlens_k, args.cu_seqlens_k_new,
                args.seqused_q, args.seqused_k, args.leftpad_k, args.seqlens_rotary, args.ptr_sink};
    }
    template <typename SharedStorage, typename FrgTensorO, typename Softmax>
    CUTLASS_DEVICE bool
    mma(Params const& params,
        FrgTensorO& tOrO,
        Softmax& softmax,
        int const thread_idx,
        SeqlenInfo_t const& seqlen_info,
        cute::tuple<int32_t, int32_t, int32_t, int32_t> block_coord,
        SharedStorage& shared_storage
        ) {
        static_assert(is_rmem<FrgTensorO>::value, "O tensor must be rmem resident.");
        static constexpr int kBlockM = get<0>(TileShape_MNK{});
        static constexpr int kBlockN = get<1>(TileShape_MNK{});
        int const m_block = get<0>(block_coord);
        int const bidh = get<1>(block_coord);
        int const bidb = get<2>(block_coord);
        int const split_idx = get<3>(block_coord);
        int const bidh_kv = !PackGQA ? params.qhead_per_khead_divmod.divide(bidh) : bidh;
        auto n_block_min_max = BlockMN_t::get_n_block_min_max(
            seqlen_info, m_block, bidb, split_idx, params.num_splits,
            params.window_size_left, params.window_size_right, params.attention_chunk_divmod,
            params.qhead_per_khead_divmod);
        int const n_block_min = get<0>(n_block_min_max);
        int const n_block_max = get<1>(n_block_min_max);
        if constexpr (Is_causal || Is_local || Varlen || Split) {
            if (n_block_max <= n_block_min) { return false; }
        }
        Tensor sQ = make_tensor(make_smem_ptr(shared_storage.tensors.mainloop.smem_q.data()), SmemLayoutQ{});
        Tensor sK = make_tensor(make_smem_ptr(shared_storage.tensors.mainloop.smem_k.data()), SmemLayoutK{});
        Tensor sV = make_tensor(make_smem_ptr(shared_storage.tensors.mainloop.smem_v.data()), SmemLayoutV{});
        Tensor sVt = make_tensor(make_smem_ptr(shared_storage.tensors.mainloop.smem_v.data()), SmemLayoutVt{});
        bool const is_varlen_q = Varlen && params.cu_seqlens_q;
        bool const is_varlen_k = Varlen && params.cu_seqlens_k;
        int const bidb_kv = params.kv_batch_idx == nullptr ? bidb : params.kv_batch_idx[bidb];
        Tensor mQ = make_tensor(make_gmem_ptr(params.ptr_Q + seqlen_info.offset_q * get<0>(params.stride_Q)), params.shape_Q_packed, params.stride_Q_packed)(_, _, bidh, !is_varlen_q ? bidb : 0);
        Tensor gQ = local_tile(mQ, select<0, 2>(TileShape_MNK{}), make_coord(m_block, _0{}));
        Tensor mK = make_tensor(make_gmem_ptr(params.ptr_K + seqlen_info.offset_k * get<0>(params.stride_K)), params.shape_K, params.stride_K)(_, _, bidh_kv, !is_varlen_k ? bidb_kv : 0);
        Tensor gK = local_tile(mK, select<1, 2>(TileShape_MNK{}), make_coord(_, _0{}));
        Tensor mV = make_tensor(make_gmem_ptr(params.ptr_V + seqlen_info.offset_k * get<0>(params.stride_V)), params.shape_K, params.stride_V)(_, _, bidh_kv, !is_varlen_k ? bidb_kv : 0);
        Tensor gV = local_tile(mV, select<1, 2>(TileShape_MNK{}), make_coord(_, _0{}));
        GmemTiledCopyQKV gmem_tiled_copy_QKV;
        auto gmem_thr_copy_QKV = gmem_tiled_copy_QKV.get_thread_slice(thread_idx);
        auto gmem_thr0_copy_QKV = gmem_tiled_copy_QKV.get_thread_slice(_0{});
        Tensor tKgK = gmem_thr_copy_QKV.partition_S(gK);
        Tensor tKsK = gmem_thr_copy_QKV.partition_D(sK);
        Tensor tVgV = gmem_thr_copy_QKV.partition_S(gV);
        Tensor tVsV = gmem_thr_copy_QKV.partition_D(sV);
        TiledMma tiled_mma;
        auto thr_mma = tiled_mma.get_slice(thread_idx);
        Tensor tSrQ = thr_mma.partition_fragment_A(sQ);
        auto smem_tiled_copy_Q = make_tiled_copy_A(SmemCopyAtom{}, tiled_mma);
        auto smem_thr_copy_Q = smem_tiled_copy_Q.get_thread_slice(thread_idx);
        auto smem_tiled_copy_K = make_tiled_copy_B(SmemCopyAtom{}, tiled_mma);
        auto smem_thr_copy_K = smem_tiled_copy_K.get_thread_slice(thread_idx);
        auto smem_tiled_copy_V = make_tiled_copy_B(SmemCopyAtomTransposed{}, tiled_mma);
        auto smem_thr_copy_V = smem_tiled_copy_V.get_thread_slice(thread_idx);
        Tensor tSsQ = smem_thr_copy_Q.partition_S(sQ);
        Tensor tSsK = smem_thr_copy_K.partition_S(sK);
        Tensor tOsVt = smem_thr_copy_V.partition_S(sVt);
        Tensor cKV = cute::make_identity_tensor(select<1, 2>(TileShape_MNK{}));
        Tensor tKVcKV = gmem_thr_copy_QKV.partition_S(cKV);
        Tensor t0KVcKV = gmem_thr0_copy_QKV.partition_S(cKV);
        Tensor tKVpKV = make_tensor<bool>(make_shape(size<2>(tKsK)));
        #pragma unroll
        for (int k = 0; k < size(tKVpKV); ++k) { tKVpKV(k) = get<1>(tKVcKV(_0{}, _0{}, k)) < get<1>(params.shape_K); }
        int const seqlen_q = seqlen_info.seqlen_q;
        int const seqlen_k = seqlen_info.seqlen_k;
        int n_block = n_block_max - 1;
        if constexpr (Share_QV_Smem) { __syncthreads(); }
        if constexpr (!PackGQA) {
            Tensor tQgQ = gmem_thr_copy_QKV.partition_S(gQ);
            Tensor tQsQ = gmem_thr_copy_QKV.partition_D(sQ);
            Tensor cQ = cute::make_identity_tensor(select<0, 2>(TileShape_MNK{}));
            Tensor tQcQ = gmem_thr_copy_QKV.partition_S(cQ);
            Tensor t0QcQ = gmem_thr0_copy_QKV.partition_S(cQ);
            Tensor tQpQ = make_tensor<bool>(make_shape(size<2>(tQsQ)));
            #pragma unroll
            for (int k = 0; k < size(tQpQ); ++k) { tQpQ(k) = get<1>(tQcQ(_0{}, _0{}, k)) < get<1>(params.shape_Q); }
            flash::copy<false, false, false, true>(
                gmem_tiled_copy_QKV, tQgQ, tQsQ, t0QcQ, tQpQ, seqlen_info.seqlen_q - m_block * kBlockM - get<0>(tQcQ(_0{}, _0{}, _0{}))
            );
        } else {
            using PackGQAt = flash::PackGQAManager<get<0>(TileShape_MNK{}), get<2>(TileShape_MNK{}), NumMmaThreads, Element>;
            PackGQAt::load_Q(mQ, sQ, params.qhead_per_khead_divmod, thread_idx, seqlen_q, m_block);
        }
        cute::cp_async_fence();
        using PagedKVManager_t = PagedKVManager<get<1>(TileShape_MNK{}), get<2>(TileShape_MNK{}), get<1>(TileShape_MNK_PV{}), NumMmaThreads, Element, true >;
        PagedKVManager_t paged_kv_manager(
            params.ptr_pagetable, params.shape_pagetable, params.stride_pagetable,
            params.ptr_K, params.shape_K, params.stride_K,
            params.ptr_V, params.headdim_v, params.stride_V,
            params.page_size_divmod,
            params.page_size_divmod ,
            bidb_kv, bidh_kv, thread_idx, seqlen_info.seqlen_k, seqlen_info.leftpad_k,
            0
        );
        auto load_K = [&] (int const n_block, int const smem_pipe_write, auto need_seqlenk_masking_type) {
            static constexpr bool Seqlenk_mask = decltype(need_seqlenk_masking_type)::value;
            if constexpr (!PagedKV) {
                static constexpr bool EvenN = kBlockN % CUTE_STATIC_V(shape<0>(GmemLayoutAtom{})) == 0;
                Tensor tKsK_cur = tKsK(_, _, _, smem_pipe_write);
                int const seqlenk_row_limit = -int(get<0>(tKVcKV(_0{}, _0{}, _0{}))) + (EvenN
                    ? seqlen_info.seqlen_k - n_block * kBlockN
                    : (!Seqlenk_mask ? kBlockN : std::min(seqlen_info.seqlen_k - n_block * kBlockN, kBlockN)));
                flash::copy<!Seqlenk_mask && EvenN, false, false, true>(
                    gmem_tiled_copy_QKV, tKgK(_, _, _, n_block), tKsK_cur, t0KVcKV, tKVpKV, seqlenk_row_limit);
            } else {
                paged_kv_manager.template load_page_table<Seqlenk_mask>(n_block);
                paged_kv_manager.template load_K<Seqlenk_mask>(n_block, sK(_, _, smem_pipe_write));
            }
        };
        auto load_V = [&] (int const n_block, int const smem_pipe_write, auto need_seqlenk_masking_type) {
            static constexpr bool Seqlenk_mask = decltype(need_seqlenk_masking_type)::value;
            if constexpr (!PagedKV) {
                static constexpr bool EvenN = kBlockN % CUTE_STATIC_V(shape<0>(GmemLayoutAtom{})) == 0;
                Tensor tVsV_cur = tVsV(_, _, _, smem_pipe_write);
                Tensor tVgV_cur = tVgV(_, _, _, n_block);
                int const seqlenk_row_limit = seqlen_info.seqlen_k - n_block * kBlockN - get<0>(tKVcKV(_0{}, _0{}, _0{}));
                #pragma unroll
                for (int m = 0; m < size<1>(tVsV); ++m) {
                    if (EvenN || m < size<1>(tVsV) - 1 || get<0>(tKVcKV(_0{}, m, _0{})) < kBlockN) {
                        bool const predicate_n = !Seqlenk_mask || get<0>(t0KVcKV(_0{}, m, _0{})) < seqlenk_row_limit;
                        #pragma unroll
                        for (int k = 0; k < size<2>(tVsV); ++k) {
                            cute::copy(gmem_tiled_copy_QKV.with(tKVpKV(k) && predicate_n), tVgV_cur(_, m, k), tVsV_cur(_, m, k));
                        }
                    }
                }
            } else {
                paged_kv_manager.template load_V<Seqlenk_mask>(n_block, sV(_, _, smem_pipe_write));
            }
        };
        auto preprocess_Q = [&] {
            if constexpr (!AppendKV) {
                flash::cp_async_wait<Share_QV_Smem ? 1 : kStages * 2 - 1>();
            } else {
                if (get<1>(params.shape_rotary) > 0) {
                    using Rotary_t = Rotary<kBlockM, kHeadDim, NumMmaThreads, Element, !(Is_causal || Is_local) >;
                    Rotary_t rotary(params.ptr_rotary_cos, params.shape_rotary, params.stride_rotary_cos,
                                    params.ptr_rotary_sin, params.stride_rotary_sin,
                                    params.is_rotary_interleaved, thread_idx, seqlen_q,
                                    seqlen_info.seqlen_rotary);
                    int const qhead_per_khead = !PackGQA ? 1 : params.qhead_per_khead_divmod.divisor;
                    if (params.is_rotary_interleaved) {
                        auto [tRrCos, tRrSin] = cute::conditional_return<!PackGQA>(
                            rotary.template load_cos_sin<true >(m_block),
                            rotary.template load_cos_sin_packgqa<true >(m_block, params.qhead_per_khead_divmod)
                        );
                        flash::cp_async_wait<Share_QV_Smem ? 1 : kStages * 2 - 1>();
                        __syncthreads();
                        rotary.apply_Q_interleaved(sQ, tRrCos, tRrSin, m_block, qhead_per_khead);
                    } else {
                        auto [tRrCosCont, tRrSinCont] = cute::conditional_return<!PackGQA>(
                            rotary.template load_cos_sin<false >(m_block),
                            rotary.template load_cos_sin_packgqa<false >(m_block, params.qhead_per_khead_divmod)
                        );
                        flash::cp_async_wait<Share_QV_Smem ? 1 : kStages * 2 - 1>();
                        __syncthreads();
                        rotary.apply_Q_contiguous(sQ, tRrCosCont, tRrSinCont, m_block, qhead_per_khead);
                    }
                } else {
                    flash::cp_async_wait<Share_QV_Smem ? 1 : kStages * 2 - 1>();
                }
            }
            if constexpr (Q_in_regs) {
                __syncthreads();
                Tensor tSrQ_copy_view = smem_thr_copy_Q.retile_D(tSrQ);
                Tensor tSsQ_copy_view = smem_thr_copy_Q.partition_S(sQ);
                cute::copy(smem_tiled_copy_Q, tSsQ_copy_view, tSrQ_copy_view);
            }
        };
        if constexpr (Share_QV_Smem) {
            load_K(n_block, 0, cute::true_type{} );
            cute::cp_async_fence();
            preprocess_Q();
            __syncthreads();
        }
        if constexpr (!Share_QV_Smem) { __syncthreads(); }
        for_each(make_int_sequence<kStages>{}, [&] (auto stage) {
            static constexpr bool Is_first_stage = CUTE_STATIC_V(stage) == 0;
            static constexpr bool Is_last_stage = CUTE_STATIC_V(stage) == kStages - 1;
            if constexpr (!Share_QV_Smem || !Is_first_stage) {
                if (Is_first_stage || n_block - stage >= n_block_min) {
                    load_K(n_block - stage, stage, cute::bool_constant<Is_first_stage>{} );
                }
                cute::cp_async_fence();
            }
            if constexpr (!Is_last_stage) {
                if (Is_first_stage || n_block - stage >= n_block_min) {
                    load_V(n_block - stage, stage, cute::bool_constant<Is_first_stage>{} );
                }
                cute::cp_async_fence();
            }
        });
        if constexpr (!Share_QV_Smem) { preprocess_Q(); }
        flash::Mask<kBlockM, kBlockN, PackGQA, TiledMma> mask(
            thread_idx, seqlen_q, seqlen_k, params.window_size_left, params.window_size_right, 0 ,
            params.attention_chunk_divmod, params.qhead_per_khead_divmod
        );
        float softcap_val = params.softcap_val;
        if constexpr (Has_softcap && Is_FP8) {
            float const q_descale = params.ptr_q_descale == nullptr ? 1.0f : params.ptr_q_descale[bidb * get<0>(params.stride_q_descale) + bidh_kv * get<1>(params.stride_q_descale)];
            float const k_descale = params.ptr_k_descale == nullptr ? 1.0f : params.ptr_k_descale[bidb * get<0>(params.stride_k_descale) + bidh_kv * get<1>(params.stride_k_descale)];
            softcap_val *= q_descale * k_descale;
        }
        auto scoremod_premask_fn = [&](auto& tSrS) {
            if constexpr (Has_softcap) { flash::apply_softcap(tSrS, softcap_val); }
        };
        int smem_pipe_read = 0, smem_pipe_write = kStages - 1;
        auto load_K_next = [&] {
            if (n_block - kStages >= n_block_min) {
                load_K(n_block - kStages, kStages > 1 ? smem_pipe_write : 0, cute::false_type{} );
            }
            cute::cp_async_fence();
        };
        auto sync = [&] {
            flash::cp_async_wait<kStages * 2 - 2>();
            __syncthreads();
        };
        clear(tOrO);
        auto fwd_step = [&](int const n_block, auto mask_fn, auto is_first_iter_type, auto check_inf_type) {
            static constexpr bool Is_first_iter = decltype(is_first_iter_type)::value;
            static constexpr bool Check_inf = decltype(check_inf_type)::value;
            Tensor tSrS = partition_fragment_C(tiled_mma, select<0, 1>(TileShape_MNK{}));
            clear(tSrS);
            sync();
            auto load_V_next = [&] {
                if (n_block - kStages + 1 >= n_block_min) {
                    load_V(n_block - kStages + 1, kStages > 1 ? smem_pipe_write : 0, cute::bool_constant<Is_first_iter && kStages == 1>{} );
                }
                cute::cp_async_fence();
            };
            Tensor tSrQ_cur = cute::conditional_return<Q_in_regs>(tSrQ, thr_mma.partition_fragment_A(sQ));
            Tensor tSrK = thr_mma.partition_fragment_B(sK(_, _, _0{}));
            flash::gemm_sm80<Q_in_regs>(
                tSrS, tSrQ_cur, tSrK, tSsQ, tSsK(_, _, _, kStages > 1 ? smem_pipe_read : 0),
                tiled_mma, smem_tiled_copy_Q, smem_tiled_copy_K, smem_thr_copy_Q, smem_thr_copy_K, load_V_next
            );
            smem_pipe_write = smem_pipe_write < kStages - 1 ? smem_pipe_write + 1 : 0;
            scoremod_premask_fn(tSrS);
            if constexpr (kStages == 1) { sync(); load_K_next(); }
            mask_fn(tSrS, n_block);
            Tensor scores_scale = softmax.template max_get_scale<Is_first_iter, Check_inf>(tSrS);
            softmax.template online_softmax<Is_first_iter, Check_inf>(tSrS);
            if constexpr (Is_FP8) { flash::permute_Cregs_fp8(tSrS); }
            Tensor tOrP_acc = make_tensor(tSrS.data(), flash::convert_layout_acc_Aregs<TiledMma>(tSrS.layout()));
            Tensor tOrP = make_tensor_like<Element>(tOrP_acc);
            convert_type_out(tOrP_acc, tOrP);
            if constexpr (!Is_first_iter) { softmax.rescale_o(tOrO, scores_scale); }
            if constexpr (kStages > 1) { sync(); }
            Tensor tOrV = thr_mma.partition_fragment_B(sVt(_, _, _0{}));
            flash::gemm_rs_sm80(tOrO, tOrP, tOrV, tOsVt(_, _, _, kStages > 1 ? smem_pipe_read : 0), tiled_mma, smem_tiled_copy_V, smem_thr_copy_V);
            if constexpr (kStages > 1) { load_K_next(); }
            smem_pipe_read = smem_pipe_read < kStages - 1 ? smem_pipe_read + 1 : 0;
        };
        auto first_iter_mask_fn = [&](auto& tSrS, int n_block) { mask.template apply<true , Is_causal, Is_local>(tSrS, m_block, n_block); };
        fwd_step(n_block, first_iter_mask_fn, cute::true_type{} , cute::true_type{} );
        --n_block;
        if constexpr (Is_causal || Is_local) {
            auto mask_fn = [&](auto& tSrS, int n_block) { mask.template apply<false , Is_causal, Is_local>(tSrS, m_block, n_block); };
            int const n_block_min_causal_local_mask = BlockMN_t::get_n_block_min_causal_local_mask(
                seqlen_info, m_block, n_block_min, params.window_size_right,
                params.attention_chunk_divmod, params.qhead_per_khead_divmod);
            #pragma unroll 1
            for (; n_block >= n_block_min_causal_local_mask; --n_block) {
                fwd_step(n_block, mask_fn, cute::false_type{} , cute::true_type{} );
            }
        }
        int const n_block_min_before_local_mask = BlockMN_t::get_n_block_min_before_local_mask(
            seqlen_info, m_block, n_block_min, params.window_size_left,
            params.attention_chunk_divmod, params.qhead_per_khead_divmod);
        auto no_mask_fn = [](auto& tSrS, int n_block) { };
        #pragma unroll 1
        for (; n_block >= n_block_min_before_local_mask; --n_block) {
            fwd_step(n_block, no_mask_fn, cute::false_type{} , cute::false_type{} );
        }
        if constexpr (Is_local) {
            auto local_mask_fn = [&](auto& tSrS, int n_block) { mask.template apply<false , false , Is_local>(tSrS, m_block, n_block); };
            #pragma unroll 1
            for (; n_block >= n_block_min; --n_block) {
                fwd_step(n_block, local_mask_fn, cute::false_type{} , cute::bool_constant<Is_local>{} );
            }
        }
        float const v_descale = !Is_FP8 || params.ptr_v_descale == nullptr ? 1.0f : params.ptr_v_descale[bidb * get<0>(params.stride_v_descale) + bidh_kv * get<1>(params.stride_v_descale)];
        Tensor scores_scale = softmax.finalize(v_descale);
        softmax.rescale_o(tOrO, scores_scale);
        if constexpr (Is_FP8) { flash::permute_output_fp8(tOrO); }
        return true;
    }
    template <typename SharedStorage>
    CUTLASS_DEVICE bool
    store_kv_new(Params const& params,
                 int const thread_idx,
                 SharedStorage &shared_storage,
                 SeqlenInfo_t const& seqlen_info,
                 cute::tuple<int32_t, int32_t, int32_t, int32_t> block_coord
    ) {
        auto [m_block, bidh, bidb, split_idx] = block_coord;
        auto n_block_new_min_max = BlockMN_t::get_n_block_k_new_min_max(
            seqlen_info, m_block, bidb, split_idx, params.num_splits,
            params.window_size_left, params.window_size_right, params.attention_chunk_divmod,
            params.qhead_per_khead_divmod);
        int const n_block_new_min = get<0>(n_block_new_min_max);
        int const n_block_new_max = get<1>(n_block_new_min_max);
        if (n_block_new_max <= n_block_new_min) { return false; }
        Tensor sK = make_tensor(make_smem_ptr(shared_storage.tensors.mainloop.smem_k.data()), SmemLayoutK{});
        Tensor sV = make_tensor(make_smem_ptr(shared_storage.tensors.mainloop.smem_v.data()), SmemLayoutV{});
        int const bidh_kv = !PackGQA ? params.qhead_per_khead_divmod.divide(bidh) : bidh;
        int const bidb_kv = params.kv_batch_idx == nullptr ? bidb : params.kv_batch_idx[bidb];
        bool const is_varlen_k_new = Varlen && params.cu_seqlens_k_new;
        Tensor mKnew = make_tensor(make_gmem_ptr(params.ptr_K_new), params.shape_K_new, params.stride_K_new)(_, _, bidh_kv, !is_varlen_k_new ? bidb : 0);
        Tensor mVnew = make_tensor(make_gmem_ptr(params.ptr_V_new), params.shape_K_new, params.stride_V_new)(_, _, bidh_kv, !is_varlen_k_new ? bidb : 0);
        bool const is_varlen_k = Varlen && params.cu_seqlens_k;
        Tensor mK = make_tensor(make_gmem_ptr(params.ptr_K), params.shape_K, params.stride_K)(_, _, bidh_kv, !is_varlen_k ? bidb_kv : 0);
        Tensor mV = make_tensor(make_gmem_ptr(params.ptr_V), params.shape_K, params.stride_V)(_, _, bidh_kv, !is_varlen_k ? bidb_kv : 0);
        Tensor gKnew = local_tile(domain_offset(make_coord(seqlen_info.offset_k_new, _0{}), mKnew), select<1, 2>(TileShape_MNK{}), make_coord(_, _0{}));
        Tensor gVnew = local_tile(domain_offset(make_coord(seqlen_info.offset_k_new, _0{}), mVnew), select<1, 2>(TileShape_MNK{}), make_coord(_, _0{}));
        int const offset_k = seqlen_info.offset_k + seqlen_info.seqlen_k_og;
        Tensor gK = local_tile(domain_offset(make_coord(offset_k, _0{}), mK), select<1, 2>(TileShape_MNK{}), make_coord(_, _0{}));
        Tensor gV = local_tile(domain_offset(make_coord(offset_k, _0{}), mV), select<1, 2>(TileShape_MNK{}), make_coord(_, _0{}));
        static constexpr int kBlockN = get<1>(TileShape_MNK{});
        static constexpr int kHeadDim = get<2>(TileShape_MNK{});
        int const seqlen_k_new = seqlen_info.seqlen_k_new;
        using Rotary_t = Rotary<kBlockN, kHeadDim, NumMmaThreads, Element>;
        Rotary_t rotary(params.ptr_rotary_cos, params.shape_rotary, params.stride_rotary_cos,
                        params.ptr_rotary_sin, params.stride_rotary_sin,
                        params.is_rotary_interleaved, thread_idx, seqlen_k_new,
                        seqlen_info.seqlen_rotary);
        using PagedKVManager_t = PagedKVManager<get<1>(TileShape_MNK{}), get<2>(TileShape_MNK{}), get<1>(TileShape_MNK_PV{}), NumMmaThreads, Element, true , 2 >;
        PagedKVManager_t paged_kv_manager(
            params.ptr_pagetable, params.shape_pagetable, params.stride_pagetable,
            params.ptr_K, params.shape_K, params.stride_K,
            params.ptr_V, params.headdim_v, params.stride_V,
            params.page_size_divmod,
            params.page_size_divmod ,
            bidb_kv, bidh_kv, thread_idx, seqlen_k_new, offset_k,
            0
        );
        static_assert(std::is_same_v<GmemLayoutAtomAppend, typename Rotary_t::LayoutAtom>);
        static_assert(!PagedKV || std::is_same_v<GmemLayoutAtomAppend, typename PagedKVManager_t::GmemLayoutAtomKVCpAsync>);
        GmemTiledCopyQKV gmem_tiled_copy_kv_g2s;
        auto gmem_thr_copy_kv_g2s = gmem_tiled_copy_kv_g2s.get_thread_slice(thread_idx);
        auto gmem_thr0_copy_kv_g2s = gmem_tiled_copy_kv_g2s.get_thread_slice(_0{});
        GmemTiledCopyAppendKV gmem_tiled_copy_kv_s2g;
        auto gmem_thr_copy_kv_s2g = gmem_tiled_copy_kv_s2g.get_thread_slice(thread_idx);
        auto gmem_thr0_copy_kv_s2g = gmem_tiled_copy_kv_s2g.get_thread_slice(_0{});
        Tensor tKgKnew = gmem_thr_copy_kv_g2s.partition_S(gKnew);
        Tensor tKsKg2s = gmem_thr_copy_kv_g2s.partition_S(sK);
        Tensor tKsKs2g = gmem_thr_copy_kv_s2g.partition_S(sK);
        Tensor tKgK = gmem_thr_copy_kv_s2g.partition_D(gK);
        Tensor tVgVnew = gmem_thr_copy_kv_g2s.partition_S(gVnew);
        Tensor tVsVg2s = gmem_thr_copy_kv_g2s.partition_S(sV);
        Tensor tVsVs2g = gmem_thr_copy_kv_s2g.partition_S(sV);
        Tensor tVgV = gmem_thr_copy_kv_s2g.partition_D(gV);
        Tensor cK = cute::make_identity_tensor(select<1, 2>(TileShape_MNK{}));
        Tensor tKcKg2s = gmem_thr_copy_kv_g2s.partition_D(cK);
        Tensor t0KcKg2s = gmem_thr0_copy_kv_g2s.partition_D(cK);
        Tensor tKpKg2s = make_tensor<bool>(make_shape(size<2>(tKsKg2s)));
        Tensor tKcKs2g = gmem_thr_copy_kv_s2g.partition_D(cK);
        Tensor t0KcKs2g = gmem_thr0_copy_kv_s2g.partition_D(cK);
        Tensor tKpKs2g = make_tensor<bool>(make_shape(size<2>(tKsKs2g)));
        #pragma unroll
        for (int k = 0; k < size(tKpKg2s); ++k) { tKpKg2s(k) = get<1>(tKcKg2s(_0{}, _0{}, k)) < get<1>(params.shape_K); }
        #pragma unroll
        for (int k = 0; k < size(tKpKs2g); ++k) { tKpKs2g(k) = get<1>(tKcKs2g(_0{}, _0{}, k)) < get<1>(params.shape_K); }
        auto load_K_new = [&] (int const n_block, int const smem_pipe_write, auto need_seqlenk_masking_type) {
            static constexpr bool Seqlenk_mask = decltype(need_seqlenk_masking_type)::value;
            static constexpr bool EvenN = kBlockN % CUTE_STATIC_V(shape<0>(GmemLayoutAtom{})) == 0;
            Tensor tKsK_cur = tKsKg2s(_, _, _, smem_pipe_write);
            int const seqlenk_row_limit = -int(get<0>(tKcKg2s(_0{}, _0{}, _0{}))) + (EvenN
                ? seqlen_k_new - n_block * kBlockN
                : (!Seqlenk_mask ? kBlockN : std::min(seqlen_k_new - n_block * kBlockN, kBlockN)));
            flash::copy<!Seqlenk_mask && EvenN, false, false, true>(
                gmem_tiled_copy_kv_g2s, tKgKnew(_, _, _, n_block), tKsK_cur, t0KcKg2s, tKpKg2s, seqlenk_row_limit);
        };
        auto load_V_new = [&] (int const n_block, int const smem_pipe_write, auto need_seqlenk_masking_type) {
            static constexpr bool Seqlenk_mask = decltype(need_seqlenk_masking_type)::value;
            static constexpr bool EvenN = kBlockN % CUTE_STATIC_V(shape<0>(GmemLayoutAtom{})) == 0;
            Tensor tVsV_cur = tVsVg2s(_, _, _, smem_pipe_write);
            int const seqlenk_row_limit = -int(get<0>(tKcKg2s(_0{}, _0{}, _0{}))) + (EvenN
                ? seqlen_k_new - n_block * kBlockN
                : (!Seqlenk_mask ? kBlockN : std::min(seqlen_k_new - n_block * kBlockN, kBlockN)));
            flash::copy<!Seqlenk_mask && EvenN, false, false, true>(
                gmem_tiled_copy_kv_g2s, tVgVnew(_, _, _, n_block), tVsV_cur, t0KcKg2s, tKpKg2s, seqlenk_row_limit);
        };
        auto store_K = [&] (int const n_block, int const smem_pipe_read) {
            int const n_limit = std::min(seqlen_k_new - n_block * kBlockN, kBlockN);
            if (get<1>(params.shape_rotary) <= 0) {
                Tensor tKsK_cur = tKsKs2g(_, _, _, smem_pipe_read);
                if constexpr (!PagedKV) {
                    Tensor tKgK_cur = tKgK(_, _, _, n_block);
                    flash::copy<false, false, false, false>(
                        gmem_tiled_copy_kv_s2g, tKsK_cur, tKgK_cur, tKcKs2g, tKpKs2g, std::min(seqlen_k_new - n_block * kBlockN, kBlockN)
                    );
                } else {
                    paged_kv_manager.store_K(n_block, tKsK_cur);
                }
            } else {
                Tensor gK_cur = gK(_, _, n_block);
                auto tPrKPtr = cute::conditional_return<PagedKV>(paged_kv_manager.compute_K_ptr(), nullptr);
                if (params.is_rotary_interleaved) {
                    auto [tRrCos, tRrSin] = rotary.template load_cos_sin<true >(n_block);
                    rotary.template apply_K_interleaved<PagedKV>(sK(_, _, smem_pipe_read), gK_cur, tKpKs2g, tRrCos, tRrSin, tPrKPtr, n_block);
                } else {
                    auto [tRrCosCont, tRrSinCont] = rotary.template load_cos_sin<false >(n_block);
                    rotary.template apply_K_contiguous<PagedKV>(sK(_, _, smem_pipe_read), gK_cur, tKpKs2g, tRrCosCont, tRrSinCont, tPrKPtr, n_block, get<1>(params.shape_K));
                }
            }
        };
        auto store_V = [&] (int const n_block, int const smem_pipe_read) {
            int const n_limit = std::min(seqlen_k_new - n_block * kBlockN, kBlockN);
            Tensor tVsV_cur = tVsVs2g(_, _, _, smem_pipe_read);
            if constexpr (!PagedKV) {
                Tensor tVgV_cur = tVgV(_, _, _, n_block);
                flash::copy<false, false, false, false>(
                    gmem_tiled_copy_kv_s2g, tVsV_cur, tVgV_cur, tKcKs2g, tKpKs2g, n_limit);
            } else {
                paged_kv_manager.store_V(n_block, tVsV_cur);
            }
        };
        int n_block = n_block_new_max - 1;
        for_each(make_int_sequence<kStages>{}, [&] (auto stage) {
            static constexpr bool Is_first_stage = CUTE_STATIC_V(stage) == 0;
            static constexpr bool Is_last_stage = CUTE_STATIC_V(stage) == kStages - 1;
            if (Is_first_stage || n_block - stage >= n_block_new_min) {
                load_K_new(n_block - stage, stage, cute::bool_constant<Is_first_stage>{} );
            }
            cute::cp_async_fence();
            if constexpr (Is_first_stage) { __syncthreads(); }
            if constexpr (!Is_last_stage) {
                if (Is_first_stage || n_block - stage >= n_block_new_min) {
                    load_V_new(n_block - stage, stage, cute::bool_constant<Is_first_stage>{} );
                }
                cute::cp_async_fence();
            }
        });
        int smem_pipe_read = 0, smem_pipe_write = kStages - 1;
        #pragma unroll 1
        for (; n_block >= n_block_new_min; --n_block) {
            if constexpr (PagedKV) { paged_kv_manager.template load_page_table<true >(n_block); }
            flash::cp_async_wait<kStages * 2 - 2>();
            __syncthreads();
            store_K(n_block, kStages > 1 ? smem_pipe_read : 0);
            if (n_block - kStages + 1 >= n_block_new_min) {
                load_V_new(n_block - kStages + 1, kStages > 1 ? smem_pipe_write : 0, cute::bool_constant<kStages == 1>{} );
            }
            cute::cp_async_fence();
            smem_pipe_write = smem_pipe_write < kStages - 1 ? smem_pipe_write + 1 : 0;
            flash::cp_async_wait<kStages * 2 - 2>();
            __syncthreads();
            store_V(n_block, kStages > 1 ? smem_pipe_read : 0);
            smem_pipe_read = smem_pipe_read < kStages - 1 ? smem_pipe_read + 1 : 0;
            if (n_block - kStages >= n_block_new_min) {
                load_K_new(n_block - kStages, kStages > 1 ? smem_pipe_write : 0, cute::false_type{} );
            }
            cute::cp_async_fence();
        }
        return true;
    }
};
}
#include <cutlass/cutlass.h>
#include <cutlass/fast_math.h>
#include "cute/tensor.hpp"
#include "cutlass/gemm/collective/builders/sm90_common.inl"
#include "cutlass/epilogue/collective/builders/sm90_common.inl"
namespace flash {
using namespace cute;
template <class TileShape_MNK_PV_, class ClusterShape_, class Element_, class ArchTag_,
          int NumEpilogueThreads_, bool Varlen_, bool PackGQA_, bool Split_, bool FP8PermuteCol=false>
struct CollectiveEpilogueFwd {
    using TileShape_MNK_PV = TileShape_MNK_PV_;
    using ClusterShape = ClusterShape_;
    using Element = Element_;
    using ElementPartial = float;
    using ArchTag = ArchTag_;
    static constexpr int NumEpilogueThreads = NumEpilogueThreads_;
    static constexpr bool Varlen = Varlen_;
    static constexpr bool PackGQA = PackGQA_;
    static constexpr bool Split = Split_;
    static constexpr bool Use_smem = !(Split && !Varlen);
    static constexpr bool Use_TMA_O = ArchTag::kMinComputeCapability >= 90 && !Varlen && !Split && !PackGQA;
    static_assert(ArchTag::kMinComputeCapability >= 80);
    static_assert(ArchTag::kMinComputeCapability >= 90 || CUTE_STATIC_V(size(ClusterShape{})) == 1);
    static_assert(sizeof(Element) <= 2);
    static constexpr int kBlockM = get<0>(TileShape_MNK_PV{});
    static constexpr int kHeadDimV = get<1>(TileShape_MNK_PV{});
    static constexpr bool LargeHeadDimV = kHeadDimV > 256;
    using GmemTiledCopyOTMA = cute::SM90_TMA_STORE;
    static constexpr int kGmemElemsPerStore = sizeof(cute::uint128_t) / sizeof(Element);
    static_assert(kHeadDimV % kGmemElemsPerStore == 0, "Headdim must be a multiple of kGmemElemsPerStore");
    static constexpr int kBytePerRow = kHeadDimV * sizeof(Element);
    static constexpr int kBlockKGmem = (kBytePerRow % 128 == 0 ? 128 : (kBytePerRow % 64 == 0 ? 64 : 32)) / sizeof(Element);
    static constexpr int kGmemThreadsPerRow = kBlockKGmem / kGmemElemsPerStore;
    static_assert(cutlass::NumThreadsPerWarp % kGmemThreadsPerRow == 0);
    static_assert(NumEpilogueThreads % kGmemThreadsPerRow == 0, "NumEpilogueThreads must be a multiple of kGmemThreadsPerRow");
    using GmemLayoutAtom = Layout<Shape <Int<NumEpilogueThreads / kGmemThreadsPerRow>, Int<kGmemThreadsPerRow>>,
                                  Stride<Int<kGmemThreadsPerRow>, _1>>;
    static_assert(kBlockM % CUTE_STATIC_V(shape<0>(GmemLayoutAtom{})) == 0, "kBlockM must be a multiple of NumEpilogueThreads / kGmemThreadsPerRow");
    using GmemTiledCopyO = decltype(
        make_tiled_copy(Copy_Atom<AutoVectorizingCopyWithAssumedAlignment<128>, Element>{},
                        GmemLayoutAtom{},
                        Layout<Shape<_1, Int<kGmemElemsPerStore>>>{}));
    using SmemLayoutAtomOTMA = decltype(cutlass::gemm::collective::detail::ss_smem_selector<GMMA::Major::K, Element,
        decltype(cute::get<0>(TileShape_MNK_PV{})), decltype(cute::get<1>(TileShape_MNK_PV{}))>());
    using SmemLayoutOTMA = decltype(tile_to_shape(SmemLayoutAtomOTMA{}, select<0, 1>(TileShape_MNK_PV{})));
    static constexpr int kSwizzle = kBlockKGmem == 128 ? 4 : (kBlockKGmem == 64 ? 3 : (kBlockKGmem == 32 ? 2 : 1));
    static constexpr int kSwizzleBase = sizeof(Element) == 4 ? 2 : (sizeof(Element) == 2 ? 3 : 4);
    using SmemLayoutAtomO = decltype(
        composition(Swizzle<kSwizzle, kSwizzleBase, kSwizzleBase>{},
                    Layout<Shape<_8, Int<kBlockKGmem>>,
                           Stride<Int<kBlockKGmem>, _1>>{}));
    using SmemLayoutOSTS = decltype(tile_to_shape(SmemLayoutAtomO{}, select<0, 1>(TileShape_MNK_PV{})));
    using SmemLayoutO = std::conditional_t<ArchTag::kMinComputeCapability >= 90, SmemLayoutOTMA, SmemLayoutOSTS>;
    using ShapeO = cute::Shape<int32_t, int32_t, int32_t, int32_t, int32_t>;
    using StrideO = cute::Stride<int64_t, _1, int64_t, int64_t, int64_t>;
    using StrideLSE = cute::Stride<_1, int64_t, int64_t, int64_t>;
    using ShapeOPacked = std::conditional_t<!PackGQA, ShapeO, cute::Shape<cute::Shape<int32_t, int32_t>, int32_t, int32_t, int32_t, int32_t>>;
    using StrideOPacked = std::conditional_t<!PackGQA, StrideO, cute::Stride<cute::Stride<int64_t, int64_t>, _1, int64_t, int64_t, int64_t>>;
    using ShapeLSEPacked = std::conditional_t<!PackGQA, cute::Shape<int32_t, int32_t, int32_t, int32_t>, cute::Shape<cute::Shape<int32_t, int32_t>, int32_t, int32_t, int32_t>>;
    using StrideLSEPacked = std::conditional_t<!PackGQA, StrideLSE, cute::Stride<cute::Stride<int64_t, _1>, int64_t, int64_t, int64_t>>;
    using EpilogueTile_MN = decltype(select<0, 1>(TileShape_MNK_PV{}));
    using CopyOpR2S = std::conditional_t<
        ArchTag::kMinComputeCapability >= 90,
        decltype(cutlass::epilogue::collective::detail::sm90_get_smem_store_op_for_accumulator<StrideO, Element, EpilogueTile_MN>()),
        AutoVectorizingCopyWithAssumedAlignment<128>
    >;
    using SmemCopyAtomO = Copy_Atom<CopyOpR2S, Element>;
    struct TensorStorage : cute::aligned_struct<128> {
        cute::array_aligned<Element, Use_smem ? cute::cosize_v<SmemLayoutO> : 0> smem_o;
    };
    using TMA_O = std::conditional_t<
        Use_TMA_O,
        decltype(make_tma_copy(
            GmemTiledCopyOTMA{},
            make_tensor(make_gmem_ptr(static_cast<Element*>(nullptr)), ShapeO{}, StrideO{}),
            SmemLayoutOTMA{},
            select<0, 1>(TileShape_MNK_PV{}),
            _1{})),
        std::nullptr_t
    >;
    struct Arguments {
        Element* ptr_O;
        ShapeO const shape_O;
        StrideO const stride_O;
        ElementPartial* ptr_O_partial;
        StrideO const stride_O_partial;
        float* ptr_LSE;
        StrideLSE const stride_LSE;
        float* ptr_LSE_partial;
        StrideLSE const stride_LSE_partial;
        int32_t const nheads_kv;
        int const* cu_seqlens = nullptr;
        int const* seqused = nullptr;
        Element const* ptr_gate = nullptr;
        int gate_stride = 0;
        float gate_eps = 0.f;
    };
    struct Params {
        Element* ptr_O;
        ShapeO const shape_O;
        StrideO const stride_O;
        ShapeOPacked const shape_O_packed;
        StrideOPacked const stride_O_packed;
        ElementPartial* ptr_O_partial;
        StrideO const stride_O_partial;
        StrideOPacked const stride_O_partial_packed;
        float* ptr_LSE;
        StrideLSE const stride_LSE;
        ShapeLSEPacked const shape_LSE_packed;
        StrideLSEPacked const stride_LSE_packed;
        float* ptr_LSE_partial;
        StrideLSE const stride_LSE_partial;
        StrideLSEPacked const stride_LSE_partial_packed;
        cutlass::FastDivmod qhead_per_khead_divmod;
        TMA_O tma_store_O;
        int const* cu_seqlens = nullptr;
        int const* seqused = nullptr;
        Element const* ptr_gate = nullptr;
        int gate_stride = 0;
        float gate_eps = 0.f;
    };
    static Params
    to_underlying_arguments(Arguments const& args) {
        Tensor mO = make_tensor(make_gmem_ptr(args.ptr_O), args.shape_O, args.stride_O);
        TMA_O tma_store_O = [&]{
            if constexpr (Use_TMA_O) {
                return make_tma_copy(GmemTiledCopyOTMA{}, mO, SmemLayoutO{}, select<0, 1>(TileShape_MNK_PV{}), _1{});
            } else {
                return nullptr;
            }
        }();
        int const qhead_per_khead = !PackGQA ? 1 : cute::ceil_div(get<2>(args.shape_O), args.nheads_kv);
        auto const shape_O_packed = cute::conditional_return<!PackGQA>(
            args.shape_O,
            make_shape(make_shape(qhead_per_khead, get<0>(args.shape_O)), get<1>(args.shape_O), args.nheads_kv, get<3>(args.shape_O), get<4>(args.shape_O))
        );
        auto const stride_O_packed = cute::conditional_return<!PackGQA>(
            args.stride_O,
            make_stride(make_stride(get<2>(args.stride_O), get<0>(args.stride_O)), get<1>(args.stride_O), get<2>(args.stride_O) * qhead_per_khead, get<3>(args.stride_O), get<4>(args.stride_O))
        );
        auto const stride_O_partial_packed = cute::conditional_return<!PackGQA>(
            args.stride_O_partial,
            make_stride(make_stride(get<2>(args.stride_O_partial), get<0>(args.stride_O_partial)), get<1>(args.stride_O_partial), get<2>(args.stride_O_partial) * qhead_per_khead, get<3>(args.stride_O_partial), get<4>(args.stride_O_partial))
        );
        auto const shape_LSE_packed = cute::conditional_return<!PackGQA>(
            select<0, 2, 3, 4>(args.shape_O),
            make_shape(make_shape(qhead_per_khead, get<0>(args.shape_O)), args.nheads_kv, get<3>(args.shape_O), get<4>(args.shape_O))
        );
        auto const stride_LSE_packed = cute::conditional_return<!PackGQA>(
            args.stride_LSE,
            make_stride(make_stride(get<1>(args.stride_LSE), get<0>(args.stride_LSE)), get<1>(args.stride_LSE) * qhead_per_khead, get<2>(args.stride_LSE), get<3>(args.stride_LSE))
        );
        auto const stride_LSE_partial_packed = cute::conditional_return<!PackGQA>(
            args.stride_LSE_partial,
            make_stride(make_stride(get<1>(args.stride_LSE_partial), get<0>(args.stride_LSE_partial)), get<1>(args.stride_LSE_partial) * qhead_per_khead, get<2>(args.stride_LSE_partial), get<3>(args.stride_LSE_partial))
        );
        return {args.ptr_O, args.shape_O, args.stride_O, shape_O_packed, stride_O_packed,
                args.ptr_O_partial, args.stride_O_partial, stride_O_partial_packed,
                args.ptr_LSE, args.stride_LSE, shape_LSE_packed, stride_LSE_packed,
                args.ptr_LSE_partial, args.stride_LSE_partial, stride_LSE_partial_packed,
                cutlass::FastDivmod(qhead_per_khead),
                tma_store_O, args.cu_seqlens, args.seqused
                , args.ptr_gate, args.gate_stride, args.gate_eps
                };
    }
    CUTLASS_DEVICE
    static void prefetch_tma_descriptors(Params const& params) {
        if constexpr (Use_TMA_O) {
            cute::prefetch_tma_descriptor(params.tma_store_O.get_tma_descriptor());
        }
    }
    template <typename SharedStorage>
    CUTLASS_DEVICE void store_skip(SharedStorage& shared_storage) {
        flash::named_barrier_sync(NumEpilogueThreads, cutlass::arch::ReservedNamedBarriers::EpilogueBarrier);
        #pragma unroll
        for (uint32_t cta_id = 0; cta_id < size(ClusterShape{}); ++cta_id) {
            shared_storage.pipelines.barrier_O.arrive(cta_id);
        }
    }
    template <typename SharedStorage, typename FrgTensorO, typename FrgTensorLSE, typename TiledMma>
    CUTLASS_DEVICE void
    store(Params const& params,
          FrgTensorO& tOrO,
          FrgTensorLSE const& lse,
          SharedStorage& shared_storage,
          TiledMma tiled_mma,
          int thread_idx,
          cute::tuple<int32_t, int32_t, int32_t, int32_t> const& block_coord
          ) {
        auto [m_block, bidh, bidb, split_idx] = block_coord;
        int num_splits = get<4>(params.shape_O_packed);
        if constexpr (Split && Varlen) {
            uint32_t num_splits_dynamic_u = reinterpret_cast<uint32_t const&>(split_idx) >> 16;
            int num_splits_dynamic = reinterpret_cast<int&>(num_splits_dynamic_u);
            num_splits = num_splits_dynamic > 0 ? num_splits_dynamic : num_splits;
            split_idx &= 0x0000FFFF;
        }
        bool const is_split = !Split ? false : (!Varlen ? true : num_splits > 1);
        using X52P_t = flash::PackGQAManager<get<0>(TileShape_MNK_PV{}), get<1>(TileShape_MNK_PV{}), NumEpilogueThreads, Element>;
        uint4 x52g[X52P_t::kX52Rows];
        if constexpr (!Split && PackGQA && !Use_TMA_O) {
            if (params.ptr_gate != nullptr) {
                flash::SeqlenInfo<Varlen, kBlockM> x52si{bidb, size<0>(params.shape_O), params.cu_seqlens, params.seqused};
                bool const x52vl = Varlen && params.cu_seqlens;
                Tensor x52mO = make_tensor(make_gmem_ptr(params.ptr_O + x52si.offset * get<0>(params.stride_O)), params.shape_O_packed, params.stride_O_packed)(_, _, bidh, !x52vl ? bidb : 0, _0{});
                X52P_t::load_gate(x52g, x52mO, params.qhead_per_khead_divmod, thread_idx, x52si.seqlen, m_block,
                                  params.ptr_O, int64_t(get<0>(params.stride_O)), params.ptr_gate, params.gate_stride);
            }
        }
        Tensor sO = make_tensor(make_smem_ptr(shared_storage.tensors.epilogue.smem_o.data()), SmemLayoutO{});
        static constexpr bool NeedFP8Permute = FP8PermuteCol && (sizeof(Element) == 2 || sizeof(Element) == 4);
        if constexpr (NeedFP8Permute && Split) { flash::permute_output_fp8_Vcolmajor(tOrO); }
        Tensor tOrO_out = make_tensor_like<Element>(tOrO);
        flash::convert_type_out(tOrO, tOrO_out);
        if constexpr (NeedFP8Permute && !Split) { flash::permute_output_fp8_Vcolmajor(tOrO_out); }
        flash::named_barrier_sync(NumEpilogueThreads, cutlass::arch::ReservedNamedBarriers::EpilogueBarrier);
        if constexpr (Use_smem) {
            auto smem_tiled_copy_O = make_tiled_copy_C(SmemCopyAtomO{}, tiled_mma);
            auto smem_thr_copy_O = smem_tiled_copy_O.get_thread_slice(thread_idx);
            Tensor taccOrO = smem_thr_copy_O.retile_S(tOrO_out);
            Tensor taccOsO = smem_thr_copy_O.partition_D(sO);
            cute::copy(smem_tiled_copy_O, taccOrO, taccOsO);
            if constexpr (Use_TMA_O) {
                cutlass::arch::fence_view_async_shared();
                cutlass::arch::NamedBarrier::arrive(NumEpilogueThreads + cutlass::NumThreadsPerWarp,
                                                    cutlass::arch::ReservedNamedBarriers::EpilogueBarrier);
            } else {
                flash::named_barrier_sync(NumEpilogueThreads, cutlass::arch::ReservedNamedBarriers::EpilogueBarrier);
            }
        } else {
            if constexpr (ArchTag::kMinComputeCapability >= 90) {
                #pragma unroll
                for (uint32_t cta_id = 0; cta_id < size(ClusterShape{}); ++cta_id) {
                    shared_storage.pipelines.barrier_O.arrive(cta_id);
                }
            }
        }
        flash::SeqlenInfo<Varlen, kBlockM> seqlen_info{bidb, size<0>(params.shape_O), params.cu_seqlens, params.seqused};
        bool is_varlen = Varlen && params.cu_seqlens;
        int offset_o = seqlen_info.offset;
        int seqlen_o = seqlen_info.seqlen;
        int warp_group_idx = __shfl_sync(0xFFFFFFFF, thread_idx / cutlass::NumThreadsPerWarpGroup, 0);
        auto thread_mma = tiled_mma.get_thread_slice(thread_idx);
        Tensor taccOcO = thread_mma.partition_C(cute::make_identity_tensor(select<0, 1>(TileShape_MNK_PV{})));
        static_assert(decltype(size<0, 0>(taccOcO))::value == 2);
        static_assert(decltype(size<0, 1>(taccOcO))::value == 2);
        Tensor taccOcO_rowcol = make_tensor(taccOcO.data(), flash::convert_layout_acc_rowcol(taccOcO.layout()));
        Tensor taccOcO_row = taccOcO_rowcol(_, _0{});
        CUTE_STATIC_ASSERT_V(size(lse) == size(taccOcO_row));
        using PackGQA_t = flash::PackGQAManager<get<0>(TileShape_MNK_PV{}), get<1>(TileShape_MNK_PV{}), NumEpilogueThreads, Element>;
        using PackGQApartial_t = flash::PackGQAManager<get<0>(TileShape_MNK_PV{}), get<1>(TileShape_MNK_PV{}), NumEpilogueThreads, ElementPartial>;
        Tensor mLSE = make_tensor(make_gmem_ptr((!is_split ? params.ptr_LSE : params.ptr_LSE_partial) + offset_o * get<0>(!is_split ? params.stride_LSE : params.stride_LSE_partial)),
                                  params.shape_LSE_packed,
                                  !is_split ? params.stride_LSE_packed : params.stride_LSE_partial_packed)(_, bidh, !is_varlen ? bidb : 0, !is_split ? 0 : split_idx);
        if (!LargeHeadDimV || warp_group_idx == 0) {
            if constexpr (!PackGQA) {
                #pragma unroll
                for (int mi = 0; mi < size(lse); ++mi) {
                    int const row = m_block * kBlockM + get<0>(taccOcO_row(mi));
                    if (get<1>(taccOcO_row(_0{})) == 0 && row < seqlen_o) { mLSE(row) = lse(mi); }
                }
            } else {
                PackGQA_t::store_LSE(mLSE, lse, tiled_mma, params.qhead_per_khead_divmod, thread_idx, seqlen_o, m_block);
            }
        }
        if constexpr (Use_TMA_O) {
            Tensor mO = params.tma_store_O.get_tma_tensor(params.shape_O)(_, _, bidh, bidb, split_idx);
            Tensor gO = local_tile(mO, select<0, 1>(TileShape_MNK_PV{}), make_coord(m_block, _0{}));
            auto block_tma_O = params.tma_store_O.get_slice(_0{});
            Tensor tOgO = block_tma_O.partition_D(gO);
            Tensor tOsO = block_tma_O.partition_S(sO);
            int warp_idx_sync = __shfl_sync(0xffffffff, thread_idx / cutlass::NumThreadsPerWarp, 0);
            if (warp_idx_sync == NumEpilogueThreads / cutlass::NumThreadsPerWarp - 1) {
                cutlass::arch::NamedBarrier::sync(NumEpilogueThreads + cutlass::NumThreadsPerWarp,
                                                  cutlass::arch::ReservedNamedBarriers::EpilogueBarrier);
                if (cute::elect_one_sync()) {
                    cute::copy(params.tma_store_O, tOsO, tOgO);
                    tma_store_arrive();
                    tma_store_wait<0>();
                    #pragma unroll
                    for (uint32_t cta_id = 0; cta_id < size(ClusterShape{}); ++cta_id) {
                        shared_storage.pipelines.barrier_O.arrive(cta_id);
                    }
                }
            }
        } else {
            if (!is_split) {
                Tensor mO = make_tensor(make_gmem_ptr(params.ptr_O + offset_o * get<0>(params.stride_O)), params.shape_O_packed, params.stride_O_packed)(_, _, bidh, !is_varlen ? bidb : 0, _0{});
                Tensor gO = local_tile(mO, select<0, 1>(TileShape_MNK_PV{}), make_coord(m_block, _0{}));
                GmemTiledCopyO gmem_tiled_copy_O;
                auto gmem_thr_copy_O = gmem_tiled_copy_O.get_thread_slice(thread_idx);
                Tensor tOsO = gmem_thr_copy_O.partition_S(sO);
                Tensor tOrO = make_fragment_like(tOsO);
                cute::copy(gmem_tiled_copy_O, tOsO, tOrO);
                if constexpr (ArchTag::kMinComputeCapability >= 90) {
                    cutlass::arch::fence_view_async_shared();
                    #pragma unroll
                    for (uint32_t cta_id = 0; cta_id < size(ClusterShape{}); ++cta_id) {
                        shared_storage.pipelines.barrier_O.arrive(cta_id);
                    }
                }
                if constexpr (!PackGQA) {
                    Tensor tOcO = gmem_thr_copy_O.partition_D(cute::make_identity_tensor(select<0, 1>(TileShape_MNK_PV{})));
                    Tensor tOpO = make_tensor<bool>(make_shape(size<2>(tOsO)));
                    #pragma unroll
                    for (int k = 0; k < size(tOpO); ++k) { tOpO(k) = get<1>(tOcO(_0{}, _0{}, k)) < get<1>(params.shape_O); }
                    Tensor tOgO = gmem_thr_copy_O.partition_D(gO);
                    flash::copy<false, false, false, false>(
                        gmem_tiled_copy_O, tOrO, tOgO, tOcO, tOpO, seqlen_o - m_block * kBlockM
                    );
                } else {
                    if (params.ptr_gate != nullptr) {
                        PackGQA_t::store_O_g(mO, tOrO, params.qhead_per_khead_divmod, thread_idx, seqlen_o, m_block,
                                             params.gate_eps, x52g);
                    } else
                    PackGQA_t::store_O(mO, tOrO, params.qhead_per_khead_divmod, thread_idx, seqlen_o, m_block);
                }
            } else {
                Tensor mOpartial = make_tensor(make_gmem_ptr(params.ptr_O_partial + offset_o * get<0>(params.stride_O_partial)), params.shape_O_packed, params.stride_O_partial_packed)(_, _, bidh, !is_varlen ? bidb : 0, split_idx);
                Tensor gOpartial = local_tile(mOpartial, select<0, 1>(TileShape_MNK_PV{}), make_coord(m_block, _0{}));
                if constexpr (Use_smem) {
                    if constexpr (ArchTag::kMinComputeCapability >= 90) {
                        #pragma unroll
                        for (uint32_t cta_id = 0; cta_id < size(ClusterShape{}); ++cta_id) {
                            shared_storage.pipelines.barrier_O.arrive(cta_id);
                        }
                    }
                }
                if constexpr (!PackGQA) {
                    static constexpr int kGmemElemsPerStoreDirect = 2;
                    cute::Copy_Atom<AutoVectorizingCopyWithAssumedAlignment<128>, ElementPartial> gmem_copy_direct;
                    Tensor tOrO_rowcol = make_tensor(tOrO.data(), flash::convert_layout_acc_rowcol(tOrO.layout()));
                    Tensor tOrO_copy = cute::tiled_divide(tOrO_rowcol, Shape<_1, Int<kGmemElemsPerStoreDirect>>{});
                    Tensor tOgO = thread_mma.partition_C(gOpartial);
                    Tensor tOgO_rowcol = make_tensor(tOgO.data(), flash::convert_layout_acc_rowcol(tOgO.layout()));
                    Tensor tOgO_copy = cute::tiled_divide(tOgO_rowcol, Shape<_1, Int<kGmemElemsPerStoreDirect>>{});
                    Tensor taccOcO_col = taccOcO_rowcol(_0{}, _);
                    #pragma unroll
                    for (int m = 0; m < size(taccOcO_row); ++m) {
                        if (get<0>(taccOcO_row(m)) < seqlen_o - m_block * kBlockM) {
                            #pragma unroll
                            for (int k = 0; k < size(taccOcO_col) / kGmemElemsPerStoreDirect; ++k) {
                                if (get<1>(taccOcO_col(k * kGmemElemsPerStoreDirect)) < get<1>(params.shape_O)) {
                                    cute::copy(gmem_copy_direct, tOrO_copy(_, m, k), tOgO_copy(_, m, k));
                                }
                            }
                        }
                    }
                } else {
                    PackGQApartial_t::store_O_direct(mOpartial, tOrO, tiled_mma, params.qhead_per_khead_divmod, thread_idx, seqlen_o, m_block);
                }
            }
        }
    }
    CUTLASS_DEVICE void
    store_tail() {
    }
    CUTLASS_DEVICE void
    store_zero(
         Params const& params,
         int thread_idx,
         cute::tuple<int32_t, int32_t, int32_t, int32_t> const& block_coord
         ) {
        static constexpr int kBlockM = get<0>(TileShape_MNK_PV{});
        auto [m_block, bidh, bidb, split_idx] = block_coord;
        int num_splits = get<4>(params.shape_O_packed);
        if constexpr (Split && Varlen) {
            uint32_t num_splits_dynamic_u = reinterpret_cast<uint32_t const&>(split_idx) >> 16;
            int num_splits_dynamic = reinterpret_cast<int&>(num_splits_dynamic_u);
            num_splits = num_splits_dynamic > 0 ? num_splits_dynamic : num_splits;
            split_idx &= 0x0000FFFF;
        }
        bool const is_split = !Split ? false : (!Varlen ? true : num_splits > 1);
        flash::SeqlenInfo<Varlen, kBlockM> seqlen_info{bidb, size<0>(params.shape_O), params.cu_seqlens, params.seqused};
        bool const is_varlen = Varlen && params.cu_seqlens;
        int offset_o = seqlen_info.offset;
        int seqlen_o = seqlen_info.seqlen;
        int qhead_per_khead = !PackGQA ? 1 : params.qhead_per_khead_divmod.divisor;
        Tensor mLSE = make_tensor(make_gmem_ptr((!is_split ? params.ptr_LSE : params.ptr_LSE_partial) + offset_o * get<0>(!is_split ? params.stride_LSE : params.stride_LSE_partial)),
                                  params.shape_LSE_packed,
                                  !is_split ? params.stride_LSE_packed : params.stride_LSE_partial_packed)(_, bidh, !is_varlen ? bidb : 0, !is_split ? 0 : split_idx);
        Tensor gLSE = local_tile(mLSE, Shape<Int<kBlockM>>{}, make_coord(m_block));
        static_assert(kBlockM <= NumEpilogueThreads);
        if (thread_idx < kBlockM) {
            const int row = m_block * kBlockM + thread_idx;
            if constexpr (!PackGQA) {
                if (row < seqlen_o) { mLSE(row) = -INFINITY; }
            } else {
                if (row < seqlen_o * qhead_per_khead) {
                    int m_idx, h_idx;
                    m_idx = params.qhead_per_khead_divmod.divmod(h_idx, row);
                    mLSE(make_coord(make_coord(h_idx, m_idx))) = -INFINITY;
                }
            }
        }
        if (!is_split) {
            Tensor mO = make_tensor(make_gmem_ptr(params.ptr_O + offset_o * get<0>(params.stride_O)), params.shape_O_packed, params.stride_O_packed)(_, _, bidh, !is_varlen ? bidb : 0, _0{});
            GmemTiledCopyO gmem_tiled_copy_O;
            auto gmem_thr_copy_O = gmem_tiled_copy_O.get_thread_slice(thread_idx);
            Tensor tOcO = gmem_thr_copy_O.partition_D(cute::make_identity_tensor(select<0, 1>(TileShape_MNK_PV{})));
            if constexpr (!PackGQA) {
                Tensor tOpO = make_tensor<bool>(make_shape(size<2>(tOcO)));
                #pragma unroll
                for (int k = 0; k < size(tOpO); ++k) { tOpO(k) = get<1>(tOcO(_0{}, _0{}, k)) < get<1>(params.shape_O); }
                Tensor gO = local_tile(mO, select<0, 1>(TileShape_MNK_PV{}), make_coord(m_block, _0{}));
                Tensor tOgO = gmem_thr_copy_O.partition_D(gO);
                Tensor tOrO = make_fragment_like(tOgO);
                cute::clear(tOrO);
                flash::copy<false, false, false, false>(
                    gmem_tiled_copy_O, tOrO, tOgO, tOcO, tOpO, seqlen_o - m_block * kBlockM
                );
            } else {
                using PackGQA_t = flash::PackGQAManager<get<0>(TileShape_MNK_PV{}), get<1>(TileShape_MNK_PV{}), NumEpilogueThreads, Element>;
                Tensor tOrO = make_tensor<Element>(make_shape(Shape<_1, Int<kGmemElemsPerStore>>{}, size<1>(tOcO), size<2>(tOcO)));
                cute::clear(tOrO);
                PackGQA_t::store_O(mO, tOrO, params.qhead_per_khead_divmod, thread_idx, seqlen_o, m_block);
            }
        }
    }
};
}
#include <vector>
inline bool use_one_mma_wg(Flash_fwd_params const& params) {
    return params.arch >= 90 && (params.d == 128 || params.d == 64) &&
        params.seqlen_q * (params.h / params.h_k) <= 64;
};
inline bool should_pack_gqa(bool varlen_q, int seqlen_q, int qhead_per_khead, int blockM) {
    if (varlen_q) return true;
    auto round_up = [](int a, int b) { return (a + b - 1) / b * b; };
    float nopack_gqa_efficiency = float(seqlen_q) / float(round_up(seqlen_q, blockM));
    float pack_gqa_efficiency = float(seqlen_q * qhead_per_khead) / float(round_up(seqlen_q * qhead_per_khead, blockM));
    return nopack_gqa_efficiency < 0.9 * pack_gqa_efficiency;
};
inline int num_splits_heuristic(int total_mblocks, int num_SMs, int num_n_blocks, int num_m_blocks, int size_one_kv_head, bool is_causal_or_local, int max_splits) {
    if (total_mblocks >= 0.8f * num_SMs) {
        int const size_l2 = 50 * 1024 * 1024;
        if (size_one_kv_head > size_l2 && num_m_blocks >= num_SMs * 2 && !is_causal_or_local) {
            return std::min((size_one_kv_head + size_l2 - 1) / size_l2, max_splits);
        } else {
            return 1;
        }
    }
    if (num_n_blocks <= 4) { return 1; }
    max_splits = std::min({max_splits, num_SMs, num_n_blocks});
    float max_efficiency = 0.f;
    std::vector<float> efficiency;
    efficiency.reserve(max_splits);
    for (int num_splits = 1; num_splits <= max_splits; num_splits++) {
        float n_waves = float(total_mblocks * num_splits) / num_SMs;
        float eff = n_waves / ceil(n_waves);
        if (eff > max_efficiency) { max_efficiency = eff; }
        efficiency.push_back(eff);
    }
    for (int num_splits = 1; num_splits <= max_splits; num_splits++) {
        if (efficiency[num_splits - 1] >= 0.85 * max_efficiency) {
            return num_splits;
        }
    }
    return 1;
}
using namespace cute;
static void const* x52_gptr = nullptr;
static int x52_gstride = 0;
static float x52_geps = 0.f;
template <int Arch, int kHeadDim, int kHeadDimV, int ClusterM, typename Element, typename ElementOut,
          bool Is_causal, bool Is_local, bool Has_softcap, bool Varlen, bool PagedKVNonTMA, bool AppendKV, bool HasQv, bool OnlyQv,
          bool PackGQA, bool Split, bool V_colmajor, bool Use_one_mma_wg, bool Has_sparse_mask=false, bool SK=false>
void run_flash_fwd(Flash_fwd_params &params, cudaStream_t strm_) {
    static_assert(!(Is_causal && Is_local), "Causal and Local cannot be enabled at the same time");
    static_assert(!(AppendKV && V_colmajor), "AppendKV and V_colmajor cannot be enabled at the same time");
    static_assert(!(AppendKV && !Varlen), "AppendKV requires Varlen");
    static constexpr bool Is_FP8 = cute::is_same_v<Element, cutlass::float_e4m3_t> || cute::is_same_v<Element, cutlass::float_e5m2_t>;
    static constexpr bool FP8_TransposeV = Is_FP8 && !V_colmajor;
    using ArchTag = std::conditional_t<Arch >= 90, cutlass::arch::Sm90, cutlass::arch::Sm80>;
    using ElementSink=cutlass::bfloat16_t;
    static constexpr std::tuple<int, int, bool, bool> kBlockMN_RS_IntraWGOverlap = tile_size_fwd_sm90(kHeadDim, kHeadDimV, Is_causal, Is_local, sizeof(Element) , V_colmajor, PagedKVNonTMA, Has_softcap, Use_one_mma_wg);
    static constexpr std::tuple<int, int, int, int, bool> kBlockMN_kNWarps_Stages_RS = tile_size_fwd_sm8x(Arch == 86 || Arch == 89, kHeadDim, kHeadDimV, Is_causal, Is_local, sizeof(Element) , PagedKVNonTMA, Varlen && Split, Has_softcap, AppendKV);
    static constexpr int kBlockM = Arch >= 90 ? std::get<0>(kBlockMN_RS_IntraWGOverlap) : std::get<0>(kBlockMN_kNWarps_Stages_RS);
    static constexpr int kBlockN = Arch >= 90 ? std::get<1>(kBlockMN_RS_IntraWGOverlap) : std::get<1>(kBlockMN_kNWarps_Stages_RS);
    static constexpr bool MmaPV_is_RS = std::get<2>(kBlockMN_RS_IntraWGOverlap);
    static constexpr bool IntraWGOverlap = std::get<3>(kBlockMN_RS_IntraWGOverlap);
    static constexpr int kNWarps = std::get<2>(kBlockMN_kNWarps_Stages_RS);
    static constexpr int kStages = Arch >= 90 ? ATT_STAGES : std::get<3>(kBlockMN_kNWarps_Stages_RS);
    static constexpr bool Q_in_regs = Arch >= 90 ? false : std::get<4>(kBlockMN_kNWarps_Stages_RS);
    using TileShape_MNK = cute::Shape<Int<kBlockM>, Int<kBlockN>, Int<kHeadDim>>;
    using TileShape_MNK_PV = cute::Shape<Int<kBlockM>, Int<kHeadDimV>, Int<kBlockN>>;
    using ClusterShape = cute::Shape<Int<ClusterM>, _1, _1>;
    using CollectiveMainloop = std::conditional_t<
        Arch >= 90,
        flash::CollectiveMainloopFwdSm90<kStages, ClusterShape, TileShape_MNK, kHeadDimV, Element, float, cutlass::arch::Sm90, Is_causal, Is_local, Has_softcap, Varlen, PagedKVNonTMA, AppendKV, HasQv, OnlyQv, MmaPV_is_RS, IntraWGOverlap, PackGQA, Split, V_colmajor, ElementSink, Has_sparse_mask>,
        flash::CollectiveMainloopFwdSm80<kNWarps, kStages, Q_in_regs, TileShape_MNK, kHeadDimV, Element, float, cutlass::arch::Sm80, Is_causal, Is_local, Has_softcap, Varlen, PagedKVNonTMA, AppendKV, PackGQA, Split, ElementSink>
    >;
    using CollectiveEpilogue = flash::CollectiveEpilogueFwd<TileShape_MNK_PV, ClusterShape, ElementOut, ArchTag, CollectiveMainloop::NumMmaThreads, Varlen, PackGQA, Split, FP8_TransposeV>;
    static constexpr int NumProducerThreads = Arch >= 90 ? CollectiveMainloop::NumProducerThreads : CollectiveMainloop::NumMmaThreads;
    static constexpr bool LPT = Is_causal || Is_local;
    static constexpr bool Sort = !Is_local;
    using SchedulerPersistent0 = std::conditional_t<Varlen,
        flash::VarlenDynamicPersistentTileScheduler<kBlockM, kBlockN, CollectiveMainloop::NumMmaThreads, NumProducerThreads, Split, PackGQA, Arch >= 90 , LPT, Sort, true >,
        std::conditional_t<!Is_causal && !Is_local,
            flash::StaticPersistentTileScheduler<Split>,
            flash::DynamicPersistentTileScheduler<CollectiveMainloop::NumMmaThreads, NumProducerThreads, Split, PackGQA, Arch >= 90 >
        >
    >;
    using SchedulerPersistent = std::conditional_t<SK && Varlen && Split && !Is_local,
        flash::SK2Scheduler<CollectiveMainloop::NumMmaThreads, NumProducerThreads, Arch >= 90>, SchedulerPersistent0>;
    using SchedulerSingleTile = flash::SingleTileScheduler<Varlen, Split, PackGQA, kBlockM>;
    static constexpr bool UsePersistentScheduler = Arch >= 90 ? !(Split && !Varlen) : ((Is_causal && !Varlen) || (Varlen && Split));
    using Scheduler = std::conditional_t<!UsePersistentScheduler, SchedulerSingleTile, SchedulerPersistent>;
    using AttnKernel = std::conditional_t<
        Arch >= 90,
        flash::enable_sm90_or_later<flash::FlashAttnFwdSm90<CollectiveMainloop, CollectiveEpilogue, Scheduler>>,
        flash::enable_sm80_to_sm89<flash::FlashAttnFwdSm80<CollectiveMainloop, CollectiveEpilogue, Scheduler>>
    >;
    bool const is_varlen_q = params.cu_seqlens_q;
    bool const is_varlen_k = params.cu_seqlens_k;
    bool const is_varlen_k_new = params.cu_seqlens_knew;
    int seqlen_q = !is_varlen_q ? params.seqlen_q : params.total_q;
    int batch_q = !is_varlen_q ? params.b : 1;
    int batch_k = !is_varlen_k ? (params.kv_batch_idx ? params.b_k : params.b) : 1;
    using StrideV_t = typename CollectiveMainloop::StrideV;
    int64_t const v_bs = !is_varlen_k ? params.v_batch_stride : 0;
    StrideV_t v_strides = [&] {
        if constexpr (!V_colmajor) { return StrideV_t{params.v_row_stride, _1{}, params.v_head_stride, v_bs}; }
        else { return StrideV_t{_1{}, params.v_dim_stride, params.v_head_stride, v_bs}; }
    }();
    StrideV_t vnew_strides = [&] {
        if constexpr (!V_colmajor) {
            return StrideV_t{params.vnew_row_stride, _1{}, params.vnew_head_stride, !is_varlen_k_new ? int64_t(params.vnew_batch_stride) : int64_t(0)};
        } else { return StrideV_t{}; }
    }();
    typename CollectiveMainloop::Arguments mainloop_args = {
        static_cast<Element const*>(params.q_ptr),
        {seqlen_q, params.d, params.h, batch_q},
        {params.q_row_stride, _1{}, params.q_head_stride, !is_varlen_q ? params.q_batch_stride : 0},
        static_cast<Element*>(params.k_ptr),
        {!params.page_table ? (!is_varlen_k ? params.seqlen_k : params.total_k) : params.page_size,
            params.d, params.h_k, !params.page_table ? batch_k : params.num_pages},
        {params.k_row_stride, _1{}, params.k_head_stride, !is_varlen_k ? params.k_batch_stride : 0},
        static_cast<Element*>(params.v_ptr),
        params.dv,
        v_strides,
        static_cast<Element const*>(params.knew_ptr),
        {!is_varlen_k_new ? params.seqlen_knew : params.total_knew, params.d, params.h_k, !is_varlen_k_new ? params.b : 1},
        {params.knew_row_stride, _1{}, params.knew_head_stride, !is_varlen_k_new ? params.knew_batch_stride : 0},
        static_cast<Element const*>(params.vnew_ptr),
        vnew_strides,
        static_cast<Element const*>(params.qv_ptr),
        {params.qv_row_stride, _1{}, params.qv_head_stride, !is_varlen_q ? params.qv_batch_stride : 0},
        static_cast<Element const*>(params.rotary_cos_ptr),
        {params.seqlen_k, params.rotary_dim / 2},
        {params.rotary_dim / 2, _1{}},
        static_cast<Element const*>(params.rotary_sin_ptr),
        {params.rotary_dim / 2, _1{}},
        params.is_rotary_interleaved,
        params.page_table,
        {params.kv_batch_idx ? params.b_k : params.b, !params.page_table ? 0 : params.seqlen_k / params.page_size},
        {params.page_table_batch_stride, _1{}},
        params.scale_softmax,
        params.q_descale_ptr, params.k_descale_ptr, params.v_descale_ptr,
        {params.q_descale_batch_stride, params.q_descale_head_stride},
        {params.k_descale_batch_stride, params.k_descale_head_stride},
        {params.v_descale_batch_stride, params.v_descale_head_stride},
        params.window_size_left, params.window_size_right, params.attention_chunk,
        params.softcap,
        params.num_splits,
        params.kv_batch_idx,
        params.cu_seqlens_q, params.cu_seqlens_k, params.cu_seqlens_knew,
        params.seqused_q, params.seqused_k,
        params.leftpad_k, params.seqlens_rotary,
        static_cast<ElementSink const*>(params.sink_ptr)
    };
    if constexpr (Arch >= 90) {
        mainloop_args.sparse_mask_fine = params.sparse_mask_fine;
        mainloop_args.sparse_mask_max_k_blocks = params.sparse_mask_max_k_blocks;
        mainloop_args.sparse_mask_fine_q_stride = params.sparse_mask_fine_q_stride;
        mainloop_args.sparse_mask_fine_k_stride = params.sparse_mask_fine_k_stride;
        mainloop_args.total_q = params.total_q;
    }
    typename CollectiveEpilogue::Arguments epilogue_args {
        static_cast<ElementOut*>(params.o_ptr),
        {seqlen_q, params.dv, params.h, batch_q, params.num_splits},
        {params.o_row_stride, _1{}, params.o_head_stride, !is_varlen_q ? params.o_batch_stride : 0, 0},
        static_cast<float*>(params.oaccum_ptr),
        {params.oaccum_row_stride, _1{}, params.oaccum_head_stride, !is_varlen_q ? params.oaccum_batch_stride : 0, params.oaccum_split_stride},
        static_cast<float*>(params.softmax_lse_ptr),
        {_1{}, seqlen_q, !is_varlen_q ? params.h * seqlen_q : 0, 0},
        static_cast<float*>(params.softmax_lseaccum_ptr),
        {_1{}, seqlen_q, !is_varlen_q ? params.h * seqlen_q : 0, params.h * seqlen_q * batch_q},
        params.h_k,
        params.cu_seqlens_q, params.seqused_q
    };
    if constexpr (!Split) {
        epilogue_args.ptr_gate = static_cast<typename CollectiveEpilogue::Element const*>(x52_gptr);
        epilogue_args.gate_stride = x52_gstride; epilogue_args.gate_eps = x52_geps;
    }
    int qhead_per_khead = !PackGQA ? 1 : cutlass::ceil_div(params.h, params.h_k);
    int num_blocks_m = cutlass::ceil_div(params.seqlen_q * qhead_per_khead, get<0>(TileShape_MNK{}));
    num_blocks_m = cutlass::round_up(num_blocks_m, size<0>(ClusterShape{}));
    typename flash::TileSchedulerArguments scheduler_args {
        num_blocks_m, !PackGQA ? params.h : params.h_k, params.b, params.num_splits,
        params.h / params.h_k,
        params.seqlen_q,
        params.seqlen_k, params.d, params.dv, sizeof(Element),
        params.tile_count_semaphore, params.cu_seqlens_q, params.seqused_q,
        params.num_splits_dynamic_ptr,
        params.num_m_blocks_ptr,
        params.varlen_batch_idx_ptr,
        params.num_nheads_in_l2_ptr
    };
    if (Varlen && !params.skip_scheduler_metadata_computation) {
        if constexpr (SK && Split && !Is_local) {
            sk2_prepare(params, strm_, kBlockM, kBlockN);
        } else
        prepare_varlen_num_blocks(params, strm_, PackGQA, kBlockM, kBlockN, Arch >= 90 && params.prepare_varlen_pdl );
        CHECK_CUDA_KERNEL_LAUNCH();
    }
    int device;
    CHECK_CUDA(cudaGetDevice(&device));
    typename AttnKernel::Params kernel_params = AttnKernel::to_underlying_arguments({
        mainloop_args, epilogue_args, {device, params.num_sm}, scheduler_args
    });
    dim3 grid_dims = AttnKernel::get_grid_shape(kernel_params);
    dim3 block_dims = AttnKernel::get_block_shape();
    int smem_size = AttnKernel::SharedStorageSize;
    if constexpr (size(ClusterShape{}) > 1) {
        void const* kernel = (void const*) cutlass::device_kernel<AttnKernel>;
        if (smem_size >= 48 * 1024) {
            CHECK_CUDA(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
        }
        dim3 cluster_dims(size<0>(ClusterShape{}), size<1>(ClusterShape{}), size<2>(ClusterShape{}));
        cutlass::ClusterLaunchParams launch_params{grid_dims, block_dims, cluster_dims, smem_size, strm_};
        cutlass::launch_kernel_on_cluster(launch_params, kernel, kernel_params);
    } else {
        auto kernel = cutlass::device_kernel<AttnKernel>;
        if (smem_size >= 48 * 1024) {
            CHECK_CUDA(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
        }
        cutlass::kernel_launch<AttnKernel>(grid_dims, block_dims, smem_size, strm_, kernel_params,
                                           Arch >= 90 && Varlen && !params.skip_scheduler_metadata_computation && params.prepare_varlen_pdl );
    }
    CHECK_CUDA_KERNEL_LAUNCH();
}
template<int Arch, typename T, int kHeadDim, int kHeadDimV, bool Split, bool PagedKVNonTMA, bool Has_softcap, bool PackGQA>
void run_mha_fwd_(Flash_fwd_params &params, cudaStream_t strm_) {
    static_assert(sizeof(T) == 2 || sizeof(T) == 1, "Only 16bit and 8bit are supported");
    static constexpr bool Is_FP8 = cute::is_same_v<T, cutlass::float_e4m3_t> || cute::is_same_v<T, cutlass::float_e5m2_t>;
    using T_out = std::conditional_t<!Is_FP8, T, cutlass::bfloat16_t>;
    CAUSAL_LOCAL_SWITCH(params.is_causal, params.is_local, Is_causal, Is_local, [&] {
        VCOLMAJOR_SWITCH(params.v_dim_stride != 1, V_colmajor_, [&] {
            static constexpr bool V_colmajor = V_colmajor_ && sizeof(T) == 1;
            VARLEN_SWITCH(params.cu_seqlens_q || params.cu_seqlens_k || params.seqused_q || params.seqused_k || params.leftpad_k, Varlen, [&] {
                BOOL_SWITCH(use_one_mma_wg(params), Use_one_mma_wg_, [&] {
                    static constexpr bool Use_one_mma_wg = Use_one_mma_wg_ && Arch >= 90 && (kHeadDim == 128 || kHeadDim == 64) && (kHeadDimV == kHeadDim);
                    static constexpr int kBlockM = Arch >= 90 ? std::get<0>(tile_size_fwd_sm90(kHeadDim, kHeadDimV, Is_causal, Is_local, sizeof(T) , V_colmajor, PagedKVNonTMA, Has_softcap, Use_one_mma_wg)) : 128;
                    static constexpr bool Enable_cluster = Arch == 90 && (sizeof(T) == 2 ? (kHeadDim >= 128) : (kHeadDim == 192)) && !Is_causal && !Is_local && !Split && !PagedKVNonTMA && !Varlen && !Use_one_mma_wg;
                    BOOL_SWITCH(params.qv_ptr, HasQV_, [&] {
                        static constexpr bool HasQv = HasQV_ && Arch == 90 && !Is_FP8 && kHeadDim == 64 && kHeadDimV >= 256;
                        BOOL_SWITCH(params.only_qv, OnlyQv_, [&] {
                            static constexpr bool OnlyQv = OnlyQv_ && HasQv;
                            APPENDKV_SWITCH(params.knew_ptr, AppendKV, [&] {
                                CLUSTER_SWITCH(cutlass::ceil_div(params.seqlen_q * (!PackGQA ? 1 : params.h / params.h_k), kBlockM) % 2 == 0, Use_cluster, [&] {
                                    static constexpr int ClusterM = Enable_cluster && Use_cluster ? 2 : 1;
                                    SPARSE_MASK_SWITCH(params.use_sparse_mask && Arch >= 90 && Varlen, Has_sparse_mask, [&] {
                                        run_flash_fwd<Arch, kHeadDim, kHeadDimV, ClusterM, T, T_out, Is_causal, Is_local, Has_softcap, Varlen, PagedKVNonTMA, AppendKV && Varlen, HasQv, OnlyQv, PackGQA, Split, V_colmajor, Use_one_mma_wg, Has_sparse_mask>(params, strm_);
                                    });
                                });
                            });
                        });
                    });
                });
            });
        });
    });
}
#include "cute/tensor.hpp"
#include "cutlass/cutlass.h"
#include "cutlass/arch/arch.h"
#include "cutlass/device_kernel.h"
#include "cutlass/kernel_launch.h"
#include "cute/tensor.hpp"
#include <cutlass/cutlass.h>
#include <cutlass/arch/memory.h>
#include <cutlass/array.h>
#include <cutlass/numeric_types.h>
#include <cutlass/numeric_conversion.h>
#include "cutlass/arch/grid_dependency_control.h"
namespace flash {
using namespace cute;
template <class TileShape_MK_, int kLogMaxSplits_, int kNThreads, int AlignmentLSE_,
          bool Is_even_K, bool Varlen, class Element, class ElementPartial, class ArchTag_>
class FlashAttnFwdCombine {
public:
    using TileShape_MK = TileShape_MK_;
    using ArchTag = ArchTag_;
    static constexpr int kMaxSplits = 1 << kLogMaxSplits_;
    static constexpr int AlignmentLSE = std::min(AlignmentLSE_, int(128 / 8 / sizeof(float)));
    static_assert(AlignmentLSE >= 1);
    static constexpr int kStages = 4;
    static_assert(ArchTag::kMinComputeCapability >= 75);
    static constexpr bool Has_cp_async = ArchTag::kMinComputeCapability >= 80;
    static constexpr uint32_t MaxThreadsPerBlock = kNThreads;
    static constexpr uint32_t MinBlocksPerMultiprocessor = 2;
    static constexpr int kBlockM = get<0>(TileShape_MK{});
    static constexpr int kBlockK = get<1>(TileShape_MK{});
    static constexpr int kGmemElemsPerLoad = sizeof(cute::uint128_t) / sizeof(ElementPartial);
    static_assert(kBlockK % kGmemElemsPerLoad == 0, "kBlockK must be a multiple of kGmemElemsPerLoad");
    static constexpr int kBlockKGmem = kBlockK % 128 == 0 ? 128 : (kBlockK % 64 == 0 ? 64 : 32);
    static constexpr int kGmemThreadsPerRow = kBlockKGmem / kGmemElemsPerLoad;
    static_assert(MaxThreadsPerBlock % kGmemThreadsPerRow == 0, "MaxThreadsPerBlock must be a multiple of kGmemThreadsPerRow");
    using GmemCopyAtom = std::conditional_t<
        Has_cp_async,
        cute::Copy_Atom<SM80_CP_ASYNC_CACHEGLOBAL<uint128_t>, ElementPartial>,
        cute::Copy_Atom<AutoVectorizingCopyWithAssumedAlignment<128>, ElementPartial>
    >;
    using GmemLayoutAtom = Layout<Shape <Int<MaxThreadsPerBlock / kGmemThreadsPerRow>, Int<kGmemThreadsPerRow>>,
                                  Stride<Int<kGmemThreadsPerRow>, _1>>;
    static_assert(kBlockM % CUTE_STATIC_V(shape<0>(GmemLayoutAtom{})) == 0);
    using GmemTiledCopyAccum = decltype(
        make_tiled_copy(GmemCopyAtom{},
                        GmemLayoutAtom{},
                        Layout<Shape<_1, Int<kGmemElemsPerLoad>>>{}));
    using GmemTiledCopy = decltype(
        make_tiled_copy(Copy_Atom<AutoVectorizingCopyWithAssumedAlignment<128>, Element>{},
                        GmemLayoutAtom{},
                        Layout<Shape<_1, Int<kGmemElemsPerLoad>>>{}));
    using AlignmentTypeLSE = cute::uint_byte_t<static_cast<int>(sizeof(float)) * AlignmentLSE>;
    static constexpr int kGmemElemsPerLoadLSE = sizeof(AlignmentTypeLSE) / sizeof(float);
    static_assert(kBlockM % kGmemElemsPerLoadLSE == 0, "kBlockM must be a multiple of kGmemElemsPerLoadLSE");
    static_assert(kBlockM % 8 == 0, "kBlockM must be a multiple of 8");
    static constexpr int kBlockMSmem = kBlockM % 128 == 0 ? 128 : (kBlockM % 64 == 0 ? 64 : (kBlockM % 32 == 0 ? 32 : (kBlockM % 16 == 0 ? 16 : 8)));
    static constexpr int kGmemThreadsPerRowLSE = kBlockMSmem / kGmemElemsPerLoadLSE;
    static_assert(MaxThreadsPerBlock % kGmemThreadsPerRowLSE == 0, "MaxThreadsPerBlock must be a multiple of kGmemThreadsPerRowLSE");
    using GmemLayoutAtomLSE = Layout<Shape <Int<MaxThreadsPerBlock / kGmemThreadsPerRowLSE>, Int<kGmemThreadsPerRowLSE>>,
                                     Stride<Int<kGmemThreadsPerRowLSE>, _1>>;
    static_assert(kMaxSplits % CUTE_STATIC_V(shape<0>(GmemLayoutAtomLSE{})) == 0);
    using GmemCopyAtomLSE = std::conditional_t<
        Has_cp_async,
        cute::Copy_Atom<SM80_CP_ASYNC_CACHEALWAYS<AlignmentTypeLSE>, float>,
        cute::Copy_Atom<AutoVectorizingCopyWithAssumedAlignment<AlignmentLSE * sizeof(float) * 8>, float>
    >;
    using GmemTiledCopyLSE = decltype(
        make_tiled_copy(GmemCopyAtomLSE{},
                        GmemLayoutAtomLSE{},
                        Layout<Shape<_1, Int<kGmemElemsPerLoadLSE>>>{}));
    static_assert((kBlockM * kMaxSplits * AlignmentLSE) % kNThreads == 0, "kNThreads must divide kBlockM * kMaxSplits * AlignmentLSE");
    using SmemLSESwizzle = std::conditional_t<
        kBlockMSmem == 8,
        Swizzle<5, 0, 5>,
        std::conditional_t<kBlockMSmem == 16, Swizzle<4, 0, 4>, Swizzle<3, 2, 3>>
    >;
    using SmemLayoutAtomLSE =
        decltype(composition(SmemLSESwizzle{},
                 Layout<Shape<Int<8>, Int<kBlockMSmem>>,
                 Stride<Int<kBlockMSmem>, _1>>{}));
    using SmemLayoutLSE = decltype(tile_to_shape(SmemLayoutAtomLSE{}, Shape<Int<kMaxSplits>, Int<kBlockM>>{}));
    using SmemLayoutO = Layout<Shape<Int<kBlockM>, Int<kBlockK>, Int<kStages>>,
                               Stride<Int<kBlockK>, _1, Int<kBlockM * kBlockK>>>;
    static_assert(MaxThreadsPerBlock % kBlockMSmem == 0, "MaxThreadsPerBlock must be a multiple of kBlockMSmem");
    static constexpr int kSmemThreadsPerColLSEt = MaxThreadsPerBlock / kBlockMSmem;
    static_assert(cutlass::NumThreadsPerWarp % kSmemThreadsPerColLSEt == 0, "kSmemThreadsPerColLSEt must divide NumThreadsPerWarp");
    using S2RLayoutAtomLSE = Layout<Shape<Int<kSmemThreadsPerColLSEt>, Int<MaxThreadsPerBlock / kSmemThreadsPerColLSEt>>>;
    using S2RTiledCopyLSE = decltype(make_tiled_copy(cute::Copy_Atom<cute::DefaultCopy, float>{}, S2RLayoutAtomLSE{}, Layout<_1>{}));
    using ShapeOPartial = cute::Shape<int32_t, int32_t, int32_t, int32_t, int32_t>;
    using StrideOPartial = cute::Stride<int64_t, _1, int64_t, int64_t, int64_t>;
    using ShapeLSEPartial = cute::Shape<int32_t, int32_t, int32_t, int32_t>;
    using StrideLSEPartial = cute::Stride<_1, int64_t, int64_t, int64_t>;
    using ShapeO = cute::Shape<int32_t, int32_t, int32_t, int32_t>;
    using StrideO = cute::Stride<int64_t, _1, int64_t, int64_t>;
    using ShapeLSE = cute::Shape<int32_t, int32_t, int32_t>;
    using StrideLSE = cute::Stride<_1, int64_t, int64_t>;
    struct SharedStorage : cute::aligned_struct<128> {
        cute::array_aligned<float, cute::cosize_v<SmemLayoutLSE>> smem_lse_partial;
        cute::array_aligned<int, kBlockM> smem_max_valid_split;
        cute::array_aligned<ElementPartial, cute::cosize_v<SmemLayoutO>> smem_o_partial;
    };
    static constexpr int SharedStorageSize = sizeof(SharedStorage);
    struct Arguments {
        ElementPartial const* const ptr_O_partial;
        ShapeOPartial const shape_O_partial;
        StrideOPartial const stride_O_partial;
        float const* const ptr_LSE_partial;
        ShapeLSEPartial const shape_LSE_partial;
        StrideLSEPartial const stride_LSE_partial;
        Element* const ptr_O;
        StrideO const stride_O;
        float* const ptr_LSE;
        StrideLSE const stride_LSE;
        int const* const cu_seqlens = nullptr;
        int const* const seqused = nullptr;
        int const* const num_splits_dynamic_ptr = nullptr;
        int const* const varlen_batch_idx_ptr = nullptr;
        int* const semaphore_to_reset = nullptr;
        Element const* const ptr_gate = nullptr;
        int const gate_stride = 0;
        float const gate_eps = 0.f;
    };
    struct Params {
        ElementPartial const* const ptr_O_partial;
        ShapeOPartial const shape_O_partial;
        StrideOPartial const stride_O_partial;
        float const* const ptr_LSE_partial;
        ShapeLSEPartial const shape_LSE_partial;
        StrideLSEPartial const stride_LSE_partial;
        Element* const ptr_O;
        StrideO const stride_O;
        float* const ptr_LSE;
        StrideLSE const stride_LSE;
        cutlass::FastDivmod seqlen_divmod, head_divmod;
        int const* const cu_seqlens = nullptr;
        int const* const seqused = nullptr;
        int const* const num_splits_dynamic_ptr = nullptr;
        int const* const varlen_batch_idx_ptr = nullptr;
        int* const semaphore_to_reset = nullptr;
        Element const* const ptr_gate = nullptr;
        int const gate_stride = 0;
        float const gate_eps = 0.f;
    };
    static
    Params
    to_underlying_arguments(Arguments const& args) {
        assert(get<1>(args.shape_LSE_partial) <= kMaxSplits);
        return {
            args.ptr_O_partial,
            args.shape_O_partial,
            args.stride_O_partial,
            args.ptr_LSE_partial,
            args.shape_LSE_partial,
            args.stride_LSE_partial,
            args.ptr_O,
            args.stride_O,
            args.ptr_LSE,
            args.stride_LSE,
            cutlass::FastDivmod(get<0>(args.shape_LSE_partial)), cutlass::FastDivmod(get<2>(args.shape_LSE_partial)),
            args.cu_seqlens,
            args.seqused,
            args.num_splits_dynamic_ptr,
            args.varlen_batch_idx_ptr,
            args.semaphore_to_reset,
            args.ptr_gate,
            args.gate_stride,
            args.gate_eps,
        };
    }
    CUTLASS_DEVICE
    void
    operator()(Params const& params, char* smem_buf) {
        SharedStorage& shared_storage = *reinterpret_cast<SharedStorage*>(smem_buf);
        Tensor sLSE = make_tensor(make_smem_ptr(shared_storage.smem_lse_partial.data()), SmemLayoutLSE{});
        Tensor sMaxValidSplit = make_tensor(make_smem_ptr(shared_storage.smem_max_valid_split.data()), Shape<Int<kBlockM>>{});
        Tensor sO = make_tensor(make_smem_ptr(shared_storage.smem_o_partial.data()), SmemLayoutO{});
        int const thread_idx = threadIdx.x;
        int const m_block = blockIdx.x;
        int const k_block = blockIdx.y;
        int const maybe_virtual_batch = blockIdx.z;
        int const batch = params.varlen_batch_idx_ptr ? params.varlen_batch_idx_ptr[maybe_virtual_batch] : maybe_virtual_batch;
        int const num_splits = params.num_splits_dynamic_ptr ? params.num_splits_dynamic_ptr[maybe_virtual_batch] : get<1>(params.shape_LSE_partial);
        if (params.semaphore_to_reset && threadIdx.x == 0 && blockIdx.x == gridDim.x - 1 && blockIdx.y == gridDim.y - 1 && blockIdx.z == gridDim.z - 1) {
            cutlass::arch::wait_on_dependent_grids();
            *params.semaphore_to_reset = 0;
        }
        if (num_splits <= 1) {
            if (params.ptr_gate != nullptr && k_block == 0) {
                static_assert(kBlockM * 16 == MaxThreadsPerBlock && kBlockK == 64, "sw gate fallback layout");
                cutlass::arch::wait_on_dependent_grids();
                flash::SeqlenInfo<Varlen, kBlockM> si{batch, size<0>(params.shape_LSE_partial), params.cu_seqlens, params.seqused};
                int const r = thread_idx / 16, c = (thread_idx % 16) * 4;
                int const idx = m_block * kBlockM + r;
                bool const ok = idx < si.seqlen * get<2>(params.shape_LSE_partial);
                int m_idx = 0, bidh = 0;
                if (ok) { cutlass::FastDivmod dm(si.seqlen); bidh = dm.divmod(m_idx, idx); }
                __nv_bfloat16* po = reinterpret_cast<__nv_bfloat16*>(params.ptr_O) + (size_t)(si.offset + m_idx) * get<0>(params.stride_O)
                                    + (size_t)bidh * get<2>(params.stride_O) + c;
                __nv_bfloat16 const* pg = reinterpret_cast<__nv_bfloat16 const*>(params.ptr_gate) + (size_t)(si.offset + m_idx) * params.gate_stride + bidh * 64 + c;
                float x[4] = {0.f, 0.f, 0.f, 0.f}, g[4] = {0.f, 0.f, 0.f, 0.f};
                if (ok) {
                    #pragma unroll
                    for (int i = 0; i < 4; ++i) { x[i] = __bfloat162float(po[i]); g[i] = __bfloat162float(pg[i]); }
                }
                float ss = x[0] * x[0] + x[1] * x[1] + x[2] * x[2] + x[3] * x[3];
                #pragma unroll
                for (int o = 1; o < 16; o <<= 1) ss += __shfl_xor_sync(0xffffffffu, ss, o);
                float const rs = rsqrtf(ss * (1.f / 64.f) + params.gate_eps);
                if (ok) {
                    #pragma unroll
                    for (int i = 0; i < 4; ++i) po[i] = __float2bfloat16_rn(x[i] * rs / (1.f + __expf(-g[i])));
                }
            }
            return;
        }
        flash::SeqlenInfo<Varlen, kBlockM> seqlen_info{batch, size<0>(params.shape_LSE_partial), params.cu_seqlens, params.seqused};
        int const offset = seqlen_info.offset;
        int const seqlen = seqlen_info.seqlen;
        int max_idx = seqlen * get<2>(params.shape_LSE_partial);
        if constexpr (Varlen) {
            if (m_block * kBlockM >= max_idx) { return; }
        }
        cutlass::FastDivmod seqlen_divmod_dynamic(seqlen);
        Tensor mLSEpartial = make_tensor(make_gmem_ptr(params.ptr_LSE_partial + offset * get<0>(params.stride_LSE_partial)),
                                         select<1, 0, 2, 3>(params.shape_LSE_partial),
                                         select<1, 0, 2, 3>(params.stride_LSE_partial))(_, _, _, !Varlen ? batch : 0);
        Tensor mLSEpartial_copy = cute::tiled_divide(mLSEpartial, Shape<_1, Int<kGmemElemsPerLoadLSE>>{});
        GmemTiledCopyLSE gmem_tiled_copy_LSE;
        auto gmem_thr_copy_LSE = gmem_tiled_copy_LSE.get_thread_slice(thread_idx);
        Tensor tLSEsLSE = gmem_thr_copy_LSE.partition_D(sLSE);
        Tensor cLSE = make_identity_tensor(make_shape(size<0>(sLSE), size<1>(sLSE)));
        Tensor tLSEcLSE = gmem_thr_copy_LSE.partition_S(cLSE);
        GmemTiledCopyAccum gmem_tiled_copy_O_partial;
        auto gmem_thr_copy_O_partial = gmem_tiled_copy_O_partial.get_thread_slice(thread_idx);
        Tensor cO = cute::make_identity_tensor(TileShape_MK{});
        Tensor tOcO = gmem_thr_copy_O_partial.partition_D(cO);
        Tensor mOpartial = make_tensor(make_gmem_ptr(params.ptr_O_partial + offset * get<0>(params.stride_O_partial)),
                                       params.shape_O_partial, params.stride_O_partial)(_, _, _, _, !Varlen ? batch : 0);
        Tensor tOmidx = make_tensor<int>(make_shape(size<1>(tOcO)));
        Tensor tObidh = make_tensor<int>(make_shape(size<1>(tOcO)));
        Tensor tOrOptr = make_tensor<ElementPartial const*>(make_shape(size<1>(tOcO)));
        #pragma unroll
        for (int m = 0; m < size<1>(tOcO); ++m) {
            int mi = get<0>(tOcO(_0{}, m, _0{}));
            int idx = m_block * kBlockM + mi;
            if constexpr (!Varlen) {
                tObidh(m) = params.seqlen_divmod.divmod(tOmidx(m), idx);
            } else {
                tObidh[m] = seqlen_divmod_dynamic.divmod(tOmidx(m), idx);
            }
            tOrOptr[m] = &mOpartial(tOmidx(m), k_block * kBlockK, _0{}, tObidh(m));
            if (idx >= max_idx) {
                tObidh[m] = -1;
            }
        }
        constexpr int ZM = decltype(size<1>(tOcO))::value, ZK = decltype(size<2>(tOcO))::value;
        float z0g[ZM][ZK][kGmemElemsPerLoad];
        if (params.ptr_gate != nullptr) {
            #pragma unroll
            for (int m = 0; m < ZM; ++m) {
                #pragma unroll
                for (int k = 0; k < ZK; ++k) {
                    int const col = k_block * kBlockK + get<1>(tOcO(_0{}, m, k));
                    Element const* pg = params.ptr_gate + (size_t)(offset + tOmidx(m)) * params.gate_stride + (tObidh(m) < 0 ? 0 : tObidh(m)) * 64 + col;
                    #pragma unroll
                    for (int i = 0; i < kGmemElemsPerLoad; ++i) { z0g[m][k][i] = tObidh(m) >= 0 ? float(pg[i]) : 0.f; }
                }
            }
        }
        cutlass::arch::wait_on_dependent_grids();
        #pragma unroll
        for (int m = 0; m < size<2>(tLSEcLSE); ++m) {
            int mi = int(get<1>(tLSEcLSE(_0{}, _0{}, m)));
            int idx = m_block * kBlockM + mi;
            if (idx < max_idx) {
                int m_idx, bidh;
                if constexpr (!Varlen) {
                    bidh = params.seqlen_divmod.divmod(m_idx, idx);
                } else {
                    bidh = seqlen_divmod_dynamic.divmod(m_idx, idx);
                }
                Tensor mLSEpartial_cur_copy = mLSEpartial_copy(_, _, m_idx, bidh);
                #pragma unroll
                for (int s = 0; s < size<1>(tLSEcLSE); ++s) {
                    int si = get<0>(tLSEcLSE(_0{}, s, _0{}));
                    if (si < num_splits) {
                        cute::copy(gmem_tiled_copy_LSE, mLSEpartial_cur_copy(_, si), tLSEsLSE(_, s, m));
                    } else {
                        cute::fill(tLSEsLSE(_, s, m), -INFINITY);
                    }
                }
            } else {
            }
        }
        if constexpr (Has_cp_async) { cute::cp_async_fence(); }
        Tensor tOpO = make_tensor<bool>(make_shape(size<2>(tOcO)));
        if constexpr (!(Is_even_K)) {
            #pragma unroll
            for (int k = 0; k < size(tOpO); ++k) { tOpO(k) = get<1>(tOcO(_0{}, _0{}, k)) < get<1>(params.shape_O_partial) - k_block * kBlockK; }
        }
        Tensor tOsOpartial = gmem_thr_copy_O_partial.partition_D(sO);
        auto load_O_partial = [&] (int split, int stage) {
            Tensor tOsOpartial_cur = tOsOpartial(_, _, _, stage);
            #pragma unroll
            for (int m = 0; m < size<1>(tOcO); ++m) {
                if (tObidh(m) >= 0)  {
                    Tensor mOpartial_cur = make_tensor(make_gmem_ptr(tOrOptr[m]), mOpartial(_0{}, _, _, _0{}).layout());
                    Tensor mOpartial_cur_copy = cute::tiled_divide(mOpartial_cur, Shape<Int<kGmemElemsPerLoad>>{});
                    #pragma unroll
                    for (int k = 0; k < size<2>(tOcO); ++k) {
                        int k_idx = get<1>(tOcO(_0{}, _0{}, k)) / kGmemElemsPerLoad;
                        if (Is_even_K || tOpO(k)) {
                            cute::copy(gmem_tiled_copy_O_partial, mOpartial_cur_copy(_, k_idx, split), tOsOpartial_cur(_, m, k));
                        }
                    }
                }
            }
        };
        for (int s = 0; s < kStages - 1; ++s) {
            if (s < num_splits) { load_O_partial(s, s); }
            if constexpr (Has_cp_async) { cute::cp_async_fence(); }
        }
        if constexpr (Has_cp_async) { cutlass::arch::cp_async_wait<kStages - 1>(); }
        __syncthreads();
        S2RTiledCopyLSE s2r_tiled_copy_LSE;
        auto s2r_thr_copy_LSE = s2r_tiled_copy_LSE.get_thread_slice(thread_idx);
        Tensor ts2rsLSE = s2r_thr_copy_LSE.partition_S(sLSE);
        Tensor ts2rrLSE = make_fragment_like(ts2rsLSE);
        cute::copy(s2r_tiled_copy_LSE, ts2rsLSE, ts2rrLSE);
        Tensor lse_sum = make_tensor<float>(make_shape(size<2>(ts2rrLSE)));
        Tensor ts2rcLSE = s2r_thr_copy_LSE.partition_D(cLSE);
        Tensor max_valid_split = make_tensor<int>(make_shape(size<2>(ts2rrLSE)));
        static_assert(CUTE_STATIC_V(size<0>(ts2rrLSE)) == 1);
        #pragma unroll
        for (int m = 0; m < size<2>(ts2rrLSE); ++m) {
            float lse_max = ts2rrLSE(_0{}, _0{}, m);
            #pragma unroll
            for (int s = 1; s < size<1>(ts2rrLSE); ++s) { lse_max = max(lse_max, ts2rrLSE(_0{}, s, m)); }
            MaxOp<float> max_op;
            lse_max = Allreduce<kSmemThreadsPerColLSEt>::run(lse_max, max_op);
            int max_valid_idx = -1;
            #pragma unroll
            for (int s = 0; s < size<1>(ts2rrLSE); ++s) {
                if (ts2rrLSE(_0{}, s, m) != -INFINITY) { max_valid_idx = get<0>(ts2rcLSE(_0{}, s, _0{})); }
            }
            MaxOp<int> max_int_op;
            max_valid_split[m] = Allreduce<kSmemThreadsPerColLSEt>::run(max_valid_idx, max_int_op);
            float lse_max_cur = lse_max == -INFINITY ? 0.0f : lse_max;
            float lse_sum_cur = 0.f;
            #pragma unroll
            for (int s = 0; s < size<1>(ts2rrLSE); ++s) {
                float scale = expf(ts2rrLSE(_0{}, s, m) - lse_max_cur);
                lse_sum_cur += scale;
                ts2rrLSE(_0{}, s, m) = scale;
            }
            SumOp<float> sum_op;
            lse_sum_cur = Allreduce<kSmemThreadsPerColLSEt>::run(lse_sum_cur, sum_op);
            lse_sum(m) = logf(lse_sum_cur) + lse_max;
            float inv_sum = (lse_sum_cur == 0.f || lse_sum_cur != lse_sum_cur) ? 0.f : 1.f / lse_sum_cur;
            #pragma unroll
            for (int s = 0; s < size<1>(ts2rrLSE); ++s) { ts2rrLSE(_0{}, s, m) *= inv_sum; }
        }
        cute::copy(s2r_tiled_copy_LSE, ts2rrLSE, ts2rsLSE);
        #pragma unroll
        for (int m = 0; m < size<2>(ts2rrLSE); ++m) {
            if (get<0>(ts2rcLSE(_0{}, _0{}, m)) == 0) {
                int mi = int(get<1>(ts2rcLSE(_0{}, _0{}, m)));
                if (mi < kBlockM) { sMaxValidSplit[mi] = max_valid_split[m]; }
            }
        }
        if (k_block == 0) {
            auto shape_LSE = select<0, 2, 3>(params.shape_LSE_partial);
            Tensor mLSE = make_tensor(make_gmem_ptr(params.ptr_LSE + offset * get<0>(params.stride_LSE)), shape_LSE, params.stride_LSE)(_, _, !Varlen ? batch : 0);
            #pragma unroll
            for (int m = 0; m < size<2>(ts2rrLSE); ++m) {
                if (get<0>(ts2rcLSE(_0{}, _0{}, m)) == 0) {
                    int mi = int(get<1>(ts2rcLSE(_0{}, _0{}, m)));
                    int idx = m_block * kBlockM + mi;
                    if (idx < max_idx) {
                        int m_idx, bidh;
                        if constexpr (!Varlen) {
                            bidh = params.seqlen_divmod.divmod(m_idx, idx);
                        } else {
                            bidh = seqlen_divmod_dynamic.divmod(m_idx, idx);
                        }
                        mLSE(m_idx, bidh) = lse_sum(m);
                    }
                }
            }
        }
        __syncthreads();
        int thr_max_valid_split = sMaxValidSplit[get<0>(tOcO(_0{}, _0{}, _0{}))];
        #pragma unroll
        for (int m = 1; m < size<1>(tOcO); ++m) { thr_max_valid_split = max(thr_max_valid_split, sMaxValidSplit[get<0>(tOcO(_0{}, m, _0{}))]); }
        Layout tOrOpartial_layout = gmem_thr_copy_O_partial.partition_S(make_tensor<ElementPartial>(TileShape_MK{})).layout();
        Tensor tOrOpartial = make_fragment_like<ElementPartial>(tOrOpartial_layout);
        Tensor tOrO = make_fragment_like<float>(tOrOpartial);
        clear(tOrO);
        int stage_load = kStages - 1, stage_compute = 0;
        #pragma unroll 4
        for (int s = 0; s <= thr_max_valid_split; ++s) {
            Tensor scale = make_tensor<float>(make_shape(size<1>(tOrOpartial)));
            #pragma unroll
            for (int m = 0; m < size<1>(tOrOpartial); ++m) { scale(m) = sLSE(s, get<0>(tOcO(_0{}, m, _0{}))); }
            if (s + kStages - 1 <= thr_max_valid_split) { load_O_partial(s + kStages - 1, stage_load); }
            if constexpr (Has_cp_async) { cute::cp_async_fence(); }
            stage_load = stage_load < kStages - 1 ? stage_load + 1 : 0;
            if constexpr (Has_cp_async) { cutlass::arch::cp_async_wait<kStages - 1>(); }
            cute::copy(Copy_Atom<AutoVectorizingCopyWithAssumedAlignment<128>, ElementPartial>{},
                       tOsOpartial(_, _, _, stage_compute), tOrOpartial);
            stage_compute = stage_compute < kStages - 1 ? stage_compute + 1 : 0;
            #pragma unroll
            for (int m = 0; m < size<1>(tOrOpartial); ++m) {
                if (tObidh(m) >= 0 && scale(m) > 0.f) {
                    #pragma unroll
                    for (int k = 0; k < size<2>(tOrOpartial); ++k) {
                        if (Is_even_K || tOpO(k)) {
                            Tensor rOpartial = make_tensor_like<float>(tOrOpartial(_, m, k));
                            flash::convert_type_out(tOrOpartial(_, m, k), rOpartial);
                            #pragma unroll
                            for (int i = 0; i < size<0>(tOrOpartial); ++i) {
                                tOrO(i, m, k) += scale(m) * rOpartial[i];
                            }
                        }
                    }
                }
            }
        }
        Tensor rO = make_tensor_like<Element>(tOrO);
        flash::convert_type_out(tOrO, rO);
        if (params.ptr_gate != nullptr) {
            // o_norm_gate (SGLang's FusedRMSNormGate, as sl_attn's fused epilogue): x * rsqrt(mean(x^2) + eps) *
            // sigmoid(g) on the BF16 attention output, FP32 math. A row's 64 values sit in 16 consecutive lanes.
            static_assert(kBlockK == 64 && kBlockKGmem == 64 && kGmemThreadsPerRow == 16, "sw gate layout");
            #pragma unroll
            for (int m = 0; m < size<1>(rO); ++m) {
                float ss = 0.f;
                #pragma unroll
                for (int k = 0; k < size<2>(rO); ++k) {
                    #pragma unroll
                    for (int i = 0; i < size<0>(rO); ++i) { float const xv = float(rO(i, m, k)); ss = fmaf(xv, xv, ss); }
                }
                #pragma unroll
                for (int o = 1; o < 16; o <<= 1) ss += __shfl_xor_sync(0xffffffffu, ss, o);
                float const rs = rsqrtf(ss * (1.f / 64.f) + params.gate_eps);
                if (tObidh(m) >= 0) {
                    #pragma unroll
                    for (int k = 0; k < size<2>(rO); ++k) {
                        int const col = k_block * kBlockK + get<1>(tOcO(_0{}, m, k));
                        Element const* pg = params.ptr_gate + (size_t)(offset + tOmidx(m)) * params.gate_stride + tObidh(m) * 64 + col;
                        #pragma unroll
                        for (int i = 0; i < size<0>(rO); ++i) {
                            float const gv = z0g[m][k][i];
                            rO(i, m, k) = Element(float(rO(i, m, k)) * rs / (1.f + __expf(-gv)));
                        }
                    }
                }
            }
        }
        auto shape_O = make_shape(get<0>(params.shape_O_partial), get<1>(params.shape_O_partial) - k_block * kBlockK, get<3>(params.shape_O_partial), get<4>(params.shape_O_partial));
        Tensor mO = make_tensor(make_gmem_ptr(params.ptr_O + offset * get<0>(params.stride_O) + k_block * kBlockK * get<1>(params.stride_O)),
                                shape_O, params.stride_O)(_, _, _, !Varlen ? batch : 0);
        Tensor mO_copy = cute::tiled_divide(mO, Shape<_1, Int<kGmemElemsPerLoad>>{});
        GmemTiledCopy gmem_tiled_copy_O;
        auto gmem_thr_copy_O = gmem_tiled_copy_O.get_thread_slice(thread_idx);
        #pragma unroll
        for (int m = 0; m < size<1>(tOcO); ++m) {
            if (tObidh(m) >= 0)  {
                #pragma unroll
                for (int k = 0; k < size<2>(tOcO); ++k) {
                    int k_idx = get<1>(tOcO(_0{}, _0{}, k)) / kGmemElemsPerLoad;
                    if (Is_even_K || tOpO(k)) {
                        cute::copy(gmem_tiled_copy_O, rO(_, m, k), mO_copy(_, tOmidx(m), k_idx, tObidh(m)));
                    }
                }
            }
        }
    }
};
}
using namespace cute;
static void const* sw_gate_ptr = nullptr;
static int sw_gate_stride = 0;
static float sw_gate_eps = 0.f;
template <int Arch, int kBlockM, int kBlockK, int kLogMaxSplits, bool IsEvenK, bool Varlen, typename Element, typename ElementPartial>
void run_flash_fwd_combine(Flash_fwd_params &params, cudaStream_t strm_, bool enable_pdl) {
    using ArchTag = std::conditional_t<Arch >= 90, cutlass::arch::Sm90, cutlass::arch::Sm80>;
    using TileShape_MK = cute::Shape<Int<kBlockM>, Int<kBlockK>>;
    using CombineKernel = flash::FlashAttnFwdCombine<TileShape_MK, kLogMaxSplits, 256 , 1 ,
                                                     IsEvenK, Varlen, Element, ElementPartial, ArchTag>;
    typename CombineKernel::Arguments args {
        static_cast<ElementPartial const*>(params.oaccum_ptr),
        {!Varlen ? params.seqlen_q : params.total_q, params.dv, params.num_splits, params.h, !Varlen ? params.b : 1},
        {params.oaccum_row_stride, _1{}, params.oaccum_split_stride, params.oaccum_head_stride, !Varlen ? params.oaccum_batch_stride : 0},
        static_cast<float*>(params.softmax_lseaccum_ptr),
        {!Varlen ? params.seqlen_q : params.total_q, params.num_splits, params.h, !Varlen ? params.b : 1},
        {_1{}, params.lseaccum_split_stride, params.lseaccum_head_stride, !Varlen ? params.lseaccum_batch_stride : 0},
        static_cast<Element*>(params.o_ptr),
        {params.o_row_stride, _1{}, params.o_head_stride, !Varlen ? params.o_batch_stride : 0},
        static_cast<float*>(params.softmax_lse_ptr),
        {_1{}, !Varlen ? params.seqlen_q : params.total_q, !Varlen ? params.h * params.seqlen_q : 0},
        params.cu_seqlens_q, params.seqused_q, params.num_splits_dynamic_ptr, params.varlen_batch_idx_ptr, params.tile_count_semaphore,
        static_cast<Element const*>(sw_gate_ptr), sw_gate_stride, sw_gate_eps
    };
    typename CombineKernel::Params kernel_params = CombineKernel::to_underlying_arguments(args);
    int num_blocks_k = cute::ceil_div(params.dv, kBlockK);
    int num_blocks_m = cute::ceil_div(params.seqlen_q * params.h, kBlockM);
    dim3 grid_m(num_blocks_m, num_blocks_k, params.b);
    auto kernel = cutlass::device_kernel<CombineKernel>;
    int smem_size = CombineKernel::SharedStorageSize;
    if (smem_size >= 48 * 1024) {
        CHECK_CUDA(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
    }
    cutlass::kernel_launch<CombineKernel>(grid_m, CombineKernel::MaxThreadsPerBlock, smem_size, strm_, kernel_params, Arch >= 90 && enable_pdl );
    CHECK_CUDA_KERNEL_LAUNCH();
}
template<typename T, typename Tpartial, int kBlockK>
void run_mha_fwd_combine_(Flash_fwd_params &params, cudaStream_t strm_, bool enable_pdl) {
    static_assert(kBlockK % 32 == 0, "kBlockK must be a multiple of 32");
    static constexpr int kBlockM = kBlockK % 128 == 0 ? 8 : (kBlockK % 64 == 0 ? 16 : 32);
    ARCH_SWITCH(params.arch, Arch, [&] {
        BOOL_SWITCH(params.cu_seqlens_q || params.seqused_q, Varlen, [&] {
            if constexpr (kBlockM >= 16) {
                if (params.num_splits <= 16) {
                    run_flash_fwd_combine<Arch, kBlockM, kBlockK, 4, false , Varlen, T, Tpartial>(params, strm_, enable_pdl);
                    return;
                }
            }
            if (params.num_splits <= 32) {
                run_flash_fwd_combine<Arch, kBlockM, kBlockK, 5, false , Varlen, T, Tpartial>(params, strm_, enable_pdl);
            } else if (params.num_splits <= 64) {
                run_flash_fwd_combine<Arch, kBlockM, kBlockK, 6, false , Varlen, T, Tpartial>(params, strm_, enable_pdl);
            } else if (params.num_splits <= 128) {
                run_flash_fwd_combine<Arch, kBlockM, kBlockK, 7, false , Varlen, T, Tpartial>(params, strm_, enable_pdl);
            } else {
                run_flash_fwd_combine<Arch, kBlockM, kBlockK, 8, false , Varlen, T, Tpartial>(params, strm_, enable_pdl);
            }
        });
    });
}
#include <cub/cub.cuh>
#include "cutlass/fast_math.h"
#include "cutlass/barrier.h"
#include "cutlass/arch/barrier.h"
#include "cutlass/arch/grid_dependency_control.h"
namespace flash {
template <typename T>
struct PrepareSortOp
{
    __device__ __forceinline__ bool operator()(T const & lhs, T const & rhs)
    {
        return lhs > rhs;
    }
};
template <>
struct PrepareSortOp<int2> {
    __device__ __forceinline__ bool operator()(int2 const & lhs, int2 const & rhs) const {
        return lhs.x > rhs.x;
    }
};
template <>
struct PrepareSortOp<int4> {
    __device__ __forceinline__ bool operator()(int4 const & lhs, int4 const & rhs) const {
        return lhs.x > rhs.x;
    }
};
template <int NumWarps, bool Sort>
__global__ void prepare_varlen_num_blocks_kernel(
        int seqlen_q_static, int seqlen_k_static, int seqlen_k_new_static,
        int const* const cu_seqlens_q, int const* const cu_seqlens_k, int const* const cu_seqlens_k_new,
        int const* const seqused_q, int const* const seqused_k, int const* const leftpad_k_ptr,
        int num_batch, int num_head, int qhead_per_khead, int num_sm, int num_splits_static,
        cutlass::FastDivmod blockm_divmod, cutlass::FastDivmod blockn_divmod,
        int* const tile_count_semaphore,
        int* const num_m_blocks_ptr,
        int* const num_splits_dynamic_ptr,
        int* const varlen_batch_idx_ptr,
        int* const num_nheads_in_l2_ptr,
        bool enable_pdl,
        bool is_causal,
        bool packgqa,
        int max_kvblocks_in_l2) {
    static constexpr int kNumBatchPerWarp = cutlass::NumThreadsPerWarp - 1;
    static constexpr int kSmemSize = 1;
    static constexpr int BLOCK_DIM_X = NumWarps * 32;
    static constexpr int ITEMS_PER_THREAD = 1;
    static_assert(BLOCK_DIM_X * ITEMS_PER_THREAD == NumWarps * 32);
    using BlockMergeSort = cub::BlockMergeSort<int4, BLOCK_DIM_X, ITEMS_PER_THREAD>;
    __shared__ int total_blocks_smem[kSmemSize];
    __shared__ typename BlockMergeSort::TempStorage temp_storage;
    if (enable_pdl) { cutlass::arch::launch_dependent_grids(); }
    if (threadIdx.x < kSmemSize) { total_blocks_smem[threadIdx.x] = 0; }
    __syncthreads();
    if (threadIdx.x == 0 && tile_count_semaphore) { *tile_count_semaphore = 0; }
    int lane = threadIdx.x % cutlass::NumThreadsPerWarp;
    auto get_num_m_blocks = [&](int batch_idx) {
        int seqlen;
        if (seqused_q) {
            seqlen = batch_idx < num_batch ? seqused_q[batch_idx] : 0;
        } else if (cu_seqlens_q) {
            int cur_cu_seqlen = batch_idx <= num_batch ? cu_seqlens_q[batch_idx] : 0;
            int next_cu_seqlen = __shfl_down_sync(0xffffffff, cur_cu_seqlen, 1);
            seqlen = next_cu_seqlen - cur_cu_seqlen;
        } else {
            seqlen = seqlen_q_static;
        }
        if(packgqa) { seqlen *= qhead_per_khead; }
        return batch_idx < num_batch && lane < kNumBatchPerWarp
            ? blockm_divmod.div(seqlen + blockm_divmod.divisor - 1) : 0;
    };
    auto get_num_n_blocks = [&](int batch_idx) {
        int leftpad_k = batch_idx < num_batch && leftpad_k_ptr != nullptr ? leftpad_k_ptr[batch_idx] : 0;
        int seqlen;
        if (seqused_k) {
            seqlen = batch_idx < num_batch ? seqused_k[batch_idx] : 0;
        } else if (cu_seqlens_k) {
            int cur_cu_seqlen = batch_idx <= num_batch ? cu_seqlens_k[batch_idx] : 0;
            int next_cu_seqlen = __shfl_down_sync(0xffffffff, cur_cu_seqlen, 1);
            seqlen = next_cu_seqlen - cur_cu_seqlen;
        } else {
            seqlen = seqlen_k_static;
        }
        int seqlen_new;
        if (cu_seqlens_k_new) {
            int cur_cu_seqlen_new = batch_idx <= num_batch ? cu_seqlens_k_new[batch_idx] : 0;
            int next_cu_seqlen_new = __shfl_down_sync(0xffffffff, cur_cu_seqlen_new, 1);
            seqlen_new = next_cu_seqlen_new - cur_cu_seqlen_new;
        } else {
            seqlen_new = seqlen_k_new_static;
        }
        seqlen = seqlen - leftpad_k + seqlen_new;
        return batch_idx < num_batch && lane < kNumBatchPerWarp
            ? blockn_divmod.div(seqlen + blockn_divmod.divisor - 1) : 0;
    };
    int warp_idx = threadIdx.x / cutlass::NumThreadsPerWarp;
    int batch_cta_idx_offset = int(blockIdx.x) * 992;
    int bidb_start = batch_cta_idx_offset + kNumBatchPerWarp * warp_idx;
    int batch_idx = lane + bidb_start;
    int num_m_blocks = get_num_m_blocks(batch_idx);
    int num_n_blocks = get_num_n_blocks(batch_idx);
    auto get_nheads_in_l2 = [&](int n_blocks) {
        int nheads_in_l2 = n_blocks * 16 <= max_kvblocks_in_l2 ? 16
            : n_blocks * 8 <= max_kvblocks_in_l2 ? 8
            : n_blocks * 4 <= max_kvblocks_in_l2 ? 4
            : n_blocks * 2 <= max_kvblocks_in_l2 ? 2
            : 1;
        if(!packgqa) { nheads_in_l2 *= qhead_per_khead; }
        return min(nheads_in_l2, num_head);
    };
    int num_splits_dynamic;
    if (int(gridDim.x) > 1 || num_splits_static == 1) {
        num_splits_dynamic = 1;
    } else {
        int total_blocks = num_m_blocks * num_n_blocks;
        #pragma unroll
        for (int i = cutlass::NumThreadsPerWarp / 2; i >= 1; i /= 2) {
            total_blocks += __shfl_down_sync(0xffffffff, total_blocks, i);
        }
        if (lane == 0) { atomicAdd(total_blocks_smem, total_blocks); }
        __syncthreads();
        total_blocks = total_blocks_smem[0];
        int blocks_per_sm = static_cast<int>(ceilf(float(total_blocks) * 1.1f * float(num_head) / float(num_sm)));
        num_splits_dynamic = std::max(std::min((num_n_blocks + blocks_per_sm - 1) / blocks_per_sm, num_splits_static), 1);
        num_n_blocks = cutlass::ceil_div(num_n_blocks, num_splits_dynamic);
    }
    if constexpr (Sort) {
        if(lane == kNumBatchPerWarp || batch_idx >= num_batch) {
            num_n_blocks = INT_MIN;
        } else if (is_causal) {
            num_n_blocks = num_n_blocks * blockn_divmod.divisor - num_m_blocks * blockm_divmod.divisor;
        }
        int4 batch_coords[ITEMS_PER_THREAD];
        batch_coords[0] = make_int4(num_n_blocks, num_m_blocks, num_splits_dynamic, batch_idx);
        BlockMergeSort(temp_storage).Sort(batch_coords, PrepareSortOp<int4>());
        if (is_causal) {
            batch_coords[0].x = blockn_divmod.div(batch_coords[0].x + batch_coords[0].y * blockm_divmod.divisor);
        }
        batch_idx = batch_cta_idx_offset + threadIdx.x;
        if (batch_idx < num_batch && threadIdx.x < 992) {
            if(num_nheads_in_l2_ptr) { num_nheads_in_l2_ptr[batch_idx] = get_nheads_in_l2(max(batch_coords[0].x, 1)); }
            num_m_blocks_ptr[batch_idx] = batch_coords[0].y;
            num_splits_dynamic_ptr[batch_idx] = batch_coords[0].z;
            varlen_batch_idx_ptr[batch_idx] = batch_coords[0].w;
        }
    } else {
        if (batch_idx < num_batch && lane < kNumBatchPerWarp) {
            if(num_nheads_in_l2_ptr) { num_nheads_in_l2_ptr[batch_idx] = get_nheads_in_l2(max(num_n_blocks, 1)); }
            num_splits_dynamic_ptr[batch_idx] = num_splits_dynamic;
            num_m_blocks_ptr[batch_idx] = num_m_blocks;
        }
    }
    if (enable_pdl) { cutlass::arch::wait_on_dependent_grids(); }
}
}
void prepare_varlen_num_blocks(Flash_fwd_params &params, cudaStream_t strm_, bool packgqa,
                               int blockM, int blockN, bool enable_pdl) {
    int qhead_per_khead = cutlass::ceil_div(params.h, params.h_k);
    int num_warps = cutlass::ceil_div(params.b, 31);
    int num_ctas = cutlass::ceil_div(params.b, 31 * 32);
    int const size_l2 = 8 * 1024 * 1024;
    int const element_size = params.is_e4m3 ? 1 : 2;
    int const size_one_kvblock = blockN * (params.d + params.dv) * element_size;
    int const max_kvblocks_in_l2 = size_l2 / size_one_kvblock;
    BOOL_SWITCH(params.varlen_sort_batches, Sort, [&] {
        NUM_WARP_SWITCH(num_warps, NumWarps, [&] {
            cudaLaunchAttribute la[1];
            la[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
            la[0].val.programmaticStreamSerializationAllowed = 1;
            cudaLaunchConfig_t lc = {dim3(num_ctas), dim3(32 * NumWarps), 0, strm_, la, enable_pdl ? 1u : 0u};
            cudaLaunchKernelEx(&lc, flash::prepare_varlen_num_blocks_kernel<NumWarps, Sort>,
                params.seqlen_q, params.seqlen_k, params.seqlen_knew,
                params.cu_seqlens_q, params.cu_seqlens_k, params.cu_seqlens_knew,
                params.seqused_q, params.seqused_k, params.leftpad_k,
                params.b, !packgqa ? params.h : params.h_k, qhead_per_khead, params.num_sm, params.num_splits,
                cutlass::FastDivmod(blockM), cutlass::FastDivmod(blockN),
                params.tile_count_semaphore,
                params.num_m_blocks_ptr,
                params.num_splits_dynamic_ptr,
                params.varlen_batch_idx_ptr,
                params.num_nheads_in_l2_ptr,
                enable_pdl,
                params.is_causal,
                packgqa,
                max_kvblocks_in_l2);
        });
    });
}
// Schedule of the equal-work split, one block: per tile (request, KV head, 192-row block) its causal key-block
// count plus a fixed cost K, prefix-summed into tp; st[c] is the first tile of CTA c's equal share of the total.
#define SK2_PREP_THREADS 1024
#define SK2_MAXT 4096
__global__ void __launch_bounds__(SK2_PREP_THREADS) sk2_prepare_kernel(
        int const* __restrict__ cu_q, int const* __restrict__ seqused_k, int B, int h_k, int qpk, int bm, int bn, int G, int K, int wave,
        int* __restrict__ tp, int* __restrict__ desc, int* __restrict__ st, int* __restrict__ cnt) {
    cutlass::arch::launch_dependent_grids();
    __shared__ int s_nmb[64], s_base[65];
    __shared__ int s_tp[SK2_MAXT + 1];
    using BlockScan = cub::BlockScan<int, SK2_PREP_THREADS>;
    __shared__ typename BlockScan::TempStorage scan_tmp;
    int const tid = threadIdx.x;
    if (tid < B) { int const L = cu_q[tid + 1] - cu_q[tid]; s_nmb[tid] = (L * qpk + bm - 1) / bm; }
    __syncthreads();
    if (tid == 0) { int a = 0; for (int b = 0; b < B; ++b) { s_base[b] = a; a += s_nmb[b] * h_k; } s_base[B] = a; }
    __syncthreads();
    int const NT = s_base[B];
    int const per = (NT + SK2_PREP_THREADS - 1) / SK2_PREP_THREADS;
    int local = 0;
    for (int k = 0; k < per; ++k) {
        int const t = tid * per + k;
        if (t >= NT) break;
        int o = t;
        if (wave) {
            int const q = NT / G, rem = NT % G;
            int rr, j;
            if (t < rem * (q + 1)) { rr = t / (q + 1); j = t % (q + 1); }
            else { int const p2 = t - rem * (q + 1); rr = rem + p2 / max(q, 1); j = p2 % max(q, 1); }
            o = j * G + rr;
        }
        int b = 0;
        while (b + 1 < B && s_base[b + 1] <= o) ++b;
        int const r = o - s_base[b], nmb = s_nmb[b];
        int const h = r / nmb, m = r - h * nmb;
        int const Lq = cu_q[b + 1] - cu_q[b], Lk = seqused_k[b];
        int nb = (Lk + bn - 1) / bn;
        int const m_idx_max = ((m + 1) * bm - 1) / qpk + 1;
        int const ncaus = m_idx_max + Lk - Lq;
        nb = min(nb, ncaus <= 0 ? 0 : (ncaus + bn - 1) / bn);
        nb = max(nb, 1);
        desc[t] = b | (h << 8) | (m << 16);
        cnt[t] = 0;
        s_tp[t] = nb + K;
        local += nb + K;
    }
    int excl;
    BlockScan(scan_tmp).ExclusiveSum(local, excl);
    __syncthreads();
    for (int k = 0; k < per; ++k) {
        int const t = tid * per + k;
        if (t >= NT) break;
        int const nb = s_tp[t];
        s_tp[t] = excl;
        excl += nb;
    }
    if (tid == SK2_PREP_THREADS - 1) { s_tp[NT] = excl; }
    __syncthreads();
    for (int t = tid; t <= NT; t += SK2_PREP_THREADS) { tp[t] = s_tp[t]; }
    long long const W = s_tp[NT];
    for (int c = tid; c < G; c += SK2_PREP_THREADS) {
        long long const r0 = c * W / G;
        int lo = 0, hi = NT - 1;
        while (lo < hi) { int const mid = (lo + hi + 1) >> 1; if (s_tp[mid] <= r0) lo = mid; else hi = mid - 1; }
        st[c] = lo;
    }
    if (tid == 0) { st[G] = int(W); st[G + 1] = K; }
}
void sk2_prepare(Flash_fwd_params &params, cudaStream_t strm_, int blockM, int blockN) {
    int const qpk = params.h / params.h_k;
    sk2_prepare_kernel<<<1, SK2_PREP_THREADS, 0, strm_>>>(
        params.cu_seqlens_q, params.seqused_k, params.b, params.h_k, qpk, blockM, blockN, params.num_sm, params.sk2_k, params.sk2_wave,
        params.num_m_blocks_ptr, params.varlen_batch_idx_ptr, params.num_splits_dynamic_ptr, params.tile_count_semaphore);
}
void pv_gather_stage(torch::Tensor kc, torch::Tensor vc, torch::Tensor pt, torch::Tensor seqlens, torch::Tensor ko, torch::Tensor vo, torch::Tensor cu_q, int64_t part);
template <bool Is_causal, bool Is_local, bool Paged, bool Split, bool Vcol, bool SK>
static void att_launch(Flash_fwd_params& params, cudaStream_t s) {
  run_flash_fwd<90, 64, 64, 1, cutlass::float_e4m3_t, cutlass::bfloat16_t, Is_causal, Is_local, false ,
                true , Paged, false, false, false, true , Split, Vcol ,
                false , false, SK>(params, s);
}
static void att_dispatch(Flash_fwd_params& params, cudaStream_t s, int v) {
#define ATT_CASE(V) if (v == V) { att_launch<!((V) & 1), ((V) & 1) != 0, ((V) & 2) != 0, ((V) & 4) != 0, ((V) & 8) != 0, ((V) & 16) != 0>(params, s); return; }
  ATT_VARIANTS
#undef ATT_CASE
  TORCH_CHECK(false, "att: variant not compiled: ", v);
}
static int att_num_splits(Flash_fwd_params const& params) {
  int const kBlockM = ATT_BM, kBlockN = params.is_local ? ATT_BN_LOCAL : ATT_BN;
  int seqlen_k_loaded = !params.is_local ? params.seqlen_k
      : std::max(0, std::min(params.seqlen_k, params.window_size_right + params.window_size_left + 1 + kBlockM));
  int num_n_blocks = (seqlen_k_loaded + kBlockN - 1) / kBlockN;
  int num_m_blocks = (params.seqlen_q * (params.h / params.h_k) + kBlockM - 1) / kBlockM;
  int size_one_kv_head = params.seqlen_k * (params.d + params.dv);
  int total_mblocks = ((params.total_q * (params.h / params.h_k) + kBlockM - 1) / kBlockM) * params.h_k;
  return num_splits_heuristic(total_mblocks, params.num_sm, num_n_blocks, num_m_blocks, size_one_kv_head,
                              params.is_causal || params.is_local, 128);
}
torch::Tensor att_fwd(torch::Tensor q, torch::Tensor k, torch::Tensor v, torch::Tensor page_table,
                      torch::Tensor cache_seqlens, torch::Tensor cu_seqlens_q, int64_t max_seqlen_q,
                      double softmax_scale, int64_t window_left, int64_t num_splits_req, torch::Tensor out,
                      torch::Tensor gate, double gate_eps) {
  bool const paged = page_table.numel() > 0;
  int const batch = cu_seqlens_q.size(0) - 1;
  int const total_q = q.size(0), h = q.size(1), d = q.size(2);
  int const page_size = paged ? k.size(1) : 1;
  int const seqlen_k = paged ? page_table.size(1) * page_size : k.size(1);
  int const h_k = k.size(-2);
  int seqlen_q = max_seqlen_q;
  int wl = window_left, wr = 0;
  if (wl >= seqlen_k - 1) wl = -1;
  Flash_fwd_params params = {};
  params.is_bf16 = false;
  params.is_e4m3 = true;
  params.q_ptr = q.data_ptr(); params.k_ptr = k.data_ptr(); params.v_ptr = v.data_ptr();
  params.q_row_stride = q.stride(-3); params.k_row_stride = k.stride(-3); params.v_row_stride = v.stride(-3);
  params.q_head_stride = q.stride(-2); params.k_head_stride = k.stride(-2); params.v_head_stride = v.stride(-2);
  params.v_dim_stride = v.stride(-1);
  params.o_ptr = out.data_ptr(); params.o_row_stride = out.stride(-3); params.o_head_stride = out.stride(-2);
  params.k_batch_stride = k.stride(0); params.v_batch_stride = v.stride(0);
  params.v_row_stride=128; params.v_head_stride=64; params.v_dim_stride=1;
  params.v_batch_stride=int64_t(k.size(1))*128;
  params.cu_seqlens_q = cu_seqlens_q.data_ptr<int>();
  params.seqused_k = cache_seqlens.data_ptr<int>();
  auto opts = q.options();
  auto lse = torch::empty({h, total_q}, opts.dtype(torch::kFloat));
  params.softmax_lse_ptr = lse.data_ptr();
  params.b = batch; params.h = h; params.h_k = h_k;
  params.seqlen_q = seqlen_q; params.seqlen_k = seqlen_k;
  params.seqlen_q_rounded = (seqlen_q + 127) / 128 * 128; params.seqlen_k_rounded = (seqlen_k + 127) / 128 * 128;
  params.d = d; params.d_rounded = 64;
  params.scale_softmax = softmax_scale; params.softcap = 0.f;
  params.p_dropout = 1.f; params.p_dropout_in_uint8_t = 255; params.rp_dropout = 1.f;
  params.is_causal = wl < 0 && wr == 0;
  params.is_local = (wl >= 0) && !params.is_causal;
  if (wl < 0) wl = seqlen_k - 1;
  params.window_size_left = wl; params.window_size_right = wr; params.attention_chunk = 0;
  params.arch = 90; params.num_sm = at::cuda::getCurrentDeviceProperties()->multiProcessorCount;
  params.total_q = total_q; params.total_k = paged ? 0 : batch * k.size(1); params.b_k = paged ? page_table.size(0) : batch;
  params.dv = d; params.dv_rounded = 64;
  if (paged) { params.page_table = page_table.data_ptr<int>(); params.page_table_batch_stride = page_table.stride(0); }
  params.page_size = page_size; params.num_pages = paged ? k.size(0) : 0;
  params.prepare_varlen_pdl = batch <= 992;
  params.num_splits_dynamic_ptr = reinterpret_cast<int*>(1);
  params.pagedkv_tma = false;
  params.num_splits = std::min(num_splits_req <= 0 ? att_num_splits(params) : (int)num_splits_req, 16);  // the combine takes <= 16 splits
#define SK2_K_DEFAULT 2
  int const sk_qpk = h / h_k;
  int const sk_ntcap = ((total_q * sk_qpk) / ATT_BM + batch + 1) * h_k;
  int const sk_tiles = ((total_q * sk_qpk + ATT_BM - 1) / ATT_BM) * h_k;
  bool const sk_turn = false;
  // Equal-work split (variant bit 16) for unpaged causal calls with at least 4 tiles per SM and keys >= 2 x queries.
  // num_splits_req >= 100 / 10000 / 1000000 force it with a given per-tile cost K, SM count or wave order.
  bool const sk_want = num_splits_req >= 100 || sk_turn ||
                       (num_splits_req == 0 && sk_tiles >= 4 * params.num_sm && seqlen_k >= 2 * seqlen_q);
  bool const sk = sk_want && !params.is_local && !paged && sk_ntcap <= 4096 &&
                  (seqlen_k + ATT_BN - 1) / ATT_BN < 2047 && batch <= 64;
  if (num_splits_req == -1) params.num_splits = att_num_splits(params);
  if (sk) {
    params.num_splits = 2;
    params.sk2_wave = 0;
    params.sk2_k = num_splits_req >= 100 && num_splits_req < 10000 ? int(num_splits_req - 100) : SK2_K_DEFAULT;
    if (num_splits_req >= 10000) {
      params.num_sm = std::min(params.num_sm, int((num_splits_req - 10000) / 100));
      params.sk2_k = int((num_splits_req - 10000) % 100);
    }
    if (num_splits_req >= 1000000) {
      params.sk2_wave = 1;
      params.sk2_k = SK2_K_DEFAULT; params.num_sm = at::cuda::getCurrentDeviceProperties()->multiProcessorCount;
    }
  }
  params.pack_gqa = true;
  params.varlen_sort_batches = !params.is_local;
  params.head_swizzle = params.is_causal || params.is_local;
  int b_rounded = (batch + 3) / 4 * 4;
  int nvec = 2 + (params.varlen_sort_batches ? 1 : 0) + (params.head_swizzle ? 1 : 0);
  int head_swizzle_offset = b_rounded * (params.varlen_sort_batches ? 3 : 2);
  int sem_offset = b_rounded * nvec;
  auto meta = torch::empty({sem_offset + 1}, opts.dtype(torch::kInt32));
  params.skip_scheduler_metadata_computation = false;
  int* mp = meta.data_ptr<int>();
  params.num_splits_dynamic_ptr = mp;
  params.num_m_blocks_ptr = mp + b_rounded;
  params.varlen_batch_idx_ptr = params.varlen_sort_batches ? mp + b_rounded * 2 : nullptr;
  params.num_nheads_in_l2_ptr = params.head_swizzle ? mp + head_swizzle_offset : nullptr;
  params.tile_count_semaphore = mp + sem_offset;
  params.tile_count_semaphore_offset = sem_offset;
  torch::Tensor oacc, lseacc;
  torch::Tensor sk_meta;
  if (sk) {
    int const G = params.num_sm, NMMA = 3 * 128, FR = ATT_BM * 64 / NMMA, NR = 2;
    sk_meta = torch::empty({3 * sk_ntcap + 1 + G + 2}, opts.dtype(torch::kInt32));
    int* m = sk_meta.data_ptr<int>();
    params.num_m_blocks_ptr = m;
    params.varlen_batch_idx_ptr = m + sk_ntcap + 1;
    params.tile_count_semaphore = m + 2 * sk_ntcap + 1;
    params.num_splits_dynamic_ptr = m + 3 * sk_ntcap + 1;
    params.num_nheads_in_l2_ptr = nullptr;
    oacc = torch::empty({(long)G * 2 * NMMA * FR}, opts.dtype(torch::kFloat));
    lseacc = torch::empty({(long)G * 2 * NMMA * NR}, opts.dtype(torch::kFloat));
    params.is_fp32 = false;
    params.oaccum_ptr = oacc.data_ptr(); params.softmax_lseaccum_ptr = lseacc.data_ptr();
    params.oaccum_split_stride = 0; params.oaccum_row_stride = 0; params.oaccum_head_stride = 0;
    params.lseaccum_split_stride = 0; params.lseaccum_head_stride = 0;
  } else
  if (params.num_splits > 1) {
    oacc = torch::empty({params.num_splits, h, total_q, d}, opts.dtype(torch::kFloat));
    lseacc = torch::empty({params.num_splits, h, total_q}, opts.dtype(torch::kFloat));
    params.is_fp32 = false;
    params.oaccum_ptr = oacc.data_ptr(); params.softmax_lseaccum_ptr = lseacc.data_ptr();
    params.oaccum_split_stride = oacc.stride(0); params.oaccum_row_stride = oacc.stride(-2);
    params.oaccum_head_stride = oacc.stride(-3);
    params.lseaccum_split_stride = lseacc.stride(0); params.lseaccum_head_stride = lseacc.stride(-2);
  }
  cudaStream_t s = at::cuda::getCurrentCUDAStream();
  int var = (params.is_local ? 1 : 0) | (paged ? 2 : 0) | (params.num_splits > 1 ? 4 : 0) | (params.v_dim_stride != 1 ? 8 : 0);
  if (sk) var |= 16;
  // o_norm_gate in the attention epilogue: unsplit, unpaged calls with a gate view the epilogue can read.
  bool x52_fuse = gate.numel() > 0 && params.num_splits == 1 && !paged && d == 64 &&
                  gate.stride(-1) == 1 && gate.size(-1) == h * d && gate.size(0) == total_q && out.stride(-3) == h * d &&
                  gate.stride(0) % 8 == 0 && (reinterpret_cast<uintptr_t>(gate.data_ptr()) % 16) == 0;
  x52_fuse = x52_fuse && !sk;
  if (x52_fuse) { x52_gptr = gate.data_ptr(); x52_gstride = (int)gate.stride(0); x52_geps = (float)gate_eps; }
  att_dispatch(params, s, var);
  x52_gptr = nullptr; x52_gstride = 0; x52_geps = 0.f;
  if (x52_fuse) return out;
  if (sk) return torch::empty({0}, out.options());  // equal-work split merged in-kernel: the gate is not applied
  if (params.num_splits > 1) {
    params.is_bf16 = true;
    TORCH_CHECK(params.num_splits <= 16, "att: more than 16 splits");
    bool const fuse = gate.numel() > 0;
    if (fuse) {
      TORCH_CHECK(gate.stride(-1) == 1 && gate.size(-1) == h * d && d == 64 && gate.size(0) == total_q, "att: gate view");
      sw_gate_ptr = gate.data_ptr(); sw_gate_stride = (int)gate.stride(0); sw_gate_eps = (float)gate_eps;
    }
    run_flash_fwd_combine<90, 16, 64, 4, false, true, cutlass::bfloat16_t, float>(params, s, true);
    sw_gate_ptr = nullptr; sw_gate_stride = 0; sw_gate_eps = 0.f;
    if (fuse) return out;
  }
  return torch::empty({0}, out.options());
}
int64_t att_splits(torch::Tensor q, torch::Tensor k, torch::Tensor page_table, torch::Tensor cu_seqlens_q,
                   int64_t max_seqlen_q, int64_t window_left) {
  Flash_fwd_params params = {};
  bool const paged = page_table.numel() > 0;
  params.b = cu_seqlens_q.size(0) - 1; params.h = q.size(1); params.h_k = k.size(-2);
  params.seqlen_q = max_seqlen_q; params.seqlen_k = paged ? page_table.size(1) * k.size(1) : k.size(1);
  params.d = 64; params.dv = 64; params.total_q = q.size(0);
  params.num_sm = at::cuda::getCurrentDeviceProperties()->multiProcessorCount;
  int wl = window_left >= params.seqlen_k - 1 ? -1 : window_left;
  params.is_causal = wl < 0; params.is_local = wl >= 0;
  params.window_size_left = wl < 0 ? params.seqlen_k - 1 : wl; params.window_size_right = 0;
  return att_num_splits(params);
}
__global__ void att_gather_k(uint4 const* __restrict__ kc, uint4 const* __restrict__ vc, int const* __restrict__ pt,
                             int const* __restrict__ seqlens, int const* __restrict__ cu_q, int pages, int page, int S,
                             int rows_per_b, int part, int win, uint4* __restrict__ ko, uint4* __restrict__ vo) {
  int constexpr L16 = 8;
  long long const i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  int const b = blockIdx.y;
  if (i >= (long long)rows_per_b * L16) return;
  int const j = int(i / L16), u = int(i % L16);
  int const n = seqlens[b], L = cu_q[b + 1] - cu_q[b];
  int s, drow = -1;
  if (part == 1) { if (j >= L) return; s = n - L + j; }
  else if (part == 3) {
    int const s0 = max(0, n - L - win);
    if (j >= n - s0) {
      long long const z = ((long long)b * S + j) * L16 + u; ko[z] = make_uint4(0, 0, 0, 0); vo[z] = make_uint4(0, 0, 0, 0);
      return;
    }
    s = s0 + j; drow = j;
  } else {
    s = j;
    if (s >= n) {
      long long const z = ((long long)b * S + s) * L16 + u; ko[z] = make_uint4(0, 0, 0, 0); vo[z] = make_uint4(0, 0, 0, 0);
      return;
    }
    if (part == 0 && s >= n - L) return;
  }
  int p = pt[(long long)b * pages + s / page];
  p = p < 0 ? 0 : p;
  long long const src = ((long long)p * page + s % page) * L16 + u;
  long long const dst = ((long long)b * S + (drow >= 0 ? drow : s)) * L16 + u;
  uint4 const kv = __ldg(kc + src), vv = __ldg(vc + src);
  ko[dst] = kv;
  vo[dst] = vv;
}
void att_gather(torch::Tensor kc, torch::Tensor vc, torch::Tensor pt, torch::Tensor seqlens, torch::Tensor cu_q,
                int64_t max_q, int64_t part, torch::Tensor ko, torch::Tensor vo) {
  int const B = pt.size(0), pages = pt.size(1), page = kc.size(1);
  int const S = ko.size(1);
  TORCH_CHECK(kc.size(2) * kc.size(3) == 128 && kc.is_contiguous() && vc.is_contiguous() && ko.is_contiguous(), "att_gather");
  TORCH_CHECK((part == 2 || (part >= 20 && part <= 26)) && S % ATT_BN == 0, "pv_gather full block");
  // Every allowed part goes to pv_gather_stage; the att_gather_k launch after the return is unreachable.
  pv_gather_stage(kc,vc,pt,seqlens,ko,vo,cu_q,part);
  return;
  int const rows = part == 1 ? (int)max_q : S;
  int const win = part == 3 ? (int)max_q : 0;
  long long const n = (long long)rows * 8;
  dim3 grid((unsigned)((n + 255) / 256), B);
  att_gather_k<<<grid, 256, 0, at::cuda::getCurrentCUDAStream()>>>(
      (uint4 const*)kc.data_ptr(), (uint4 const*)vc.data_ptr(), pt.data_ptr<int>(), seqlens.data_ptr<int>(),
      cu_q.data_ptr<int>(), pages, page, S, rows, (int)part, win, (uint4*)ko.data_ptr(), (uint4*)vo.data_ptr());
}
// Gather with V column-major ([B, hkv, d, S] storage), for attention_full.gather_t. part 3: the window rows from
// s0 = max(0, n - L - win); otherwise all rows. Rows past the request's keys are zero.
__global__ void __launch_bounds__(256) att_gather_t_k(uint4 const* __restrict__ kc, uint4 const* __restrict__ vc,
    int const* __restrict__ pt, int const* __restrict__ seqlens, int const* __restrict__ cu_q, int pages, int page,
    int S, int part, int win, uint4* __restrict__ ko, uint4* __restrict__ vt) {
  __shared__ uint4 tile[64][9];
  int const b = blockIdx.y, j0 = blockIdx.x * 64, t = threadIdx.x;
  int const n = seqlens[b], L = cu_q[b + 1] - cu_q[b];
  int const s0 = part == 3 ? max(0, n - L - win) : 0;
  int const nv = n - s0;
  #pragma unroll
  for (int r = 0; r < 2; ++r) {
    int const idx = t + 256 * r, jr = idx >> 3, u = idx & 7, j = j0 + jr;
    uint4 kv = make_uint4(0, 0, 0, 0), vv = make_uint4(0, 0, 0, 0);
    if (j < nv) {
      int const s = s0 + j;
      int p = pt[(long long)b * pages + s / page];
      p = p < 0 ? 0 : p;
      long long const src = ((long long)p * page + s % page) * 8 + u;
      kv = __ldg(kc + src); vv = __ldg(vc + src);
    }
    ko[((long long)b * S + j) * 8 + u] = kv;
    tile[jr][u] = vv;
  }
  __syncthreads();
  unsigned char const* tb = reinterpret_cast<unsigned char const*>(&tile[0][0]);
  #pragma unroll
  for (int r = 0; r < 2; ++r) {
    int const idx = t + 256 * r, hd = idx >> 2, q = idx & 3;
    uint32_t w[4];
    #pragma unroll
    for (int k = 0; k < 4; ++k) {
      uint32_t x = 0;
      #pragma unroll
      for (int i = 0; i < 4; ++i) { x |= uint32_t(tb[(q * 16 + 4 * k + i) * 144 + hd]) << (8 * i); }
      w[k] = x;
    }
    vt[(((long long)b * 128 + hd) * S + j0) / 16 + q] = make_uint4(w[0], w[1], w[2], w[3]);
  }
}
void att_gather_t(torch::Tensor kc, torch::Tensor vc, torch::Tensor pt, torch::Tensor seqlens, torch::Tensor cu_q,
                  int64_t win, int64_t part, torch::Tensor ko, torch::Tensor vt) {
  int const B = pt.size(0), pages = pt.size(1), page = kc.size(1);
  int const S = ko.size(1);
  TORCH_CHECK(kc.size(2) * kc.size(3) == 128 && S % 64 == 0 && kc.is_contiguous() && vc.is_contiguous() &&
              ko.is_contiguous() && vt.is_contiguous() && vt.numel() == ko.numel(), "att_gather_t");
  dim3 grid(S / 64, B);
  att_gather_t_k<<<grid, 256, 0, at::cuda::getCurrentCUDAStream()>>>(
      (uint4 const*)kc.data_ptr(), (uint4 const*)vc.data_ptr(), pt.data_ptr<int>(), seqlens.data_ptr<int>(),
      cu_q.data_ptr<int>(), pages, page, S, (int)part, (int)win, (uint4*)ko.data_ptr(), (uint4*)vt.data_ptr());
}
using PvImageMainloop = flash::CollectiveMainloopFwdSm90<
    ATT_STAGES, cute::Shape<cute::_1,cute::_1,cute::_1>,
    cute::Shape<cute::Int<ATT_BM>,cute::Int<ATT_BN>,cute::Int<64>>,64,
    cutlass::float_e4m3_t,float,cutlass::arch::Sm90,
    true,false,false,true,false,false,false,false,true,true,true,false,false,cutlass::bfloat16_t,false>;
// Gather through FA3's own V transpose: one CTA per (KV head, 256-key block, request) copies the K rows and writes
// the V image tile (the 64 V^T rows in the mainloop's smem layout, then 8 rows of 0x38 that this build never reads).
// Handles any page size.
template<class Mainloop>
__global__ void __launch_bounds__(128) pv_gather_stage_k(const uint8_t* kc,const uint8_t* vc,const int* pt,
    const int* lengths,int S,int pages,int page,int physical_pages,uint8_t* ko,uint8_t* vo,const int* cu_q) {
    using namespace cute;
    using Element=typename Mainloop::Element;
    constexpr int BN=ATT_BN,TileBytes=72*BN;
    using SmemLayoutVt=decltype(tile_to_shape(typename Mainloop::SmemLayoutAtomVt{},make_shape(Int<64>{},Int<BN>{},_1{}),Step<_2,_1,_3>{}));
    using Pad=decltype(tile_to_shape(typename Mainloop::SmemLayoutAtomVtMma{},make_shape(Int<72>{},Int<BN>{},_1{}),Step<_1,_2,_3>{}));
    using SmemLayoutVtMma=decltype(composition(Pad{},Layout<Shape<Int<64>,Int<BN>,_1>,Stride<_1,Int<72>,Int<72*BN>>>{}));
    using S2RTiledCopyVt=typename Mainloop::S2RTiledCopyVt;
    using R2STiledCopyV=typename Mainloop::R2STiledCopyV;
    using LDSM_divide_shape=typename Mainloop::LDSM_divide_shape;
    using STSM_divide_shape=typename Mainloop::STSM_divide_shape;
    __shared__ __align__(1024) Element vt[cosize_v<SmemLayoutVt>];
    __shared__ __align__(1024) Element v[TileBytes];
    auto sVt=as_position_independent_swizzle_tensor(make_tensor(make_smem_ptr(vt),SmemLayoutVt{}));
    auto sV=as_position_independent_swizzle_tensor(make_tensor(make_smem_ptr(v),SmemLayoutVtMma{}));
    int const thread_idx=threadIdx.x,nb=blockIdx.y,head=blockIdx.x,batch=blockIdx.z;
    int const nblocks=S/BN,n=lengths[batch];
    asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
    constexpr int IT=BN*4/128;
    int64_t row[IT];
    #pragma unroll
    for(int i=0;i<IT;++i) {
        int const token=nb*BN+(thread_idx+128*i)/4;
        int64_t r=-1;
        if(page==1) {
            if(token<n && token<pages) {
                int pg=pt[(int64_t)batch*pages+token];pg=pg<0?0:pg;
                if(pg<physical_pages) r=pg;
            }
        } else if(token<n && token/page<pages) {
            int pg=pt[(int64_t)batch*pages+token/page];pg=pg<0?0:pg;
            if(pg<physical_pages) r=(int64_t)pg*page+token%page;
        }
        row[i]=r;
    }
    asm volatile("griddepcontrol.wait;" ::: "memory");
    uint4 kv[IT],vv[IT];
    #pragma unroll
    for(int i=0;i<IT;++i) {
        int const d=((thread_idx+128*i)%4)*16;
        kv[i]=make_uint4(0,0,0,0);vv[i]=kv[i];
        if(row[i]>=0) {
            int64_t const src=(row[i]*2+head)*64+d;
            kv[i]=__ldg(reinterpret_cast<const uint4*>(kc+src));
            vv[i]=__ldg(reinterpret_cast<const uint4*>(vc+src));
        }
    }
    #pragma unroll
    for(int i=0;i<IT;++i) {
        int const seg=thread_idx+128*i,j=seg/4,d=(seg%4)*16,token=nb*BN+j;
        *reinterpret_cast<uint4*>(ko+(((int64_t)batch*S+token)*2+head)*64+d)=kv[i];
        *reinterpret_cast<uint4*>(&sVt(d,j,0))=vv[i];
    }
    constexpr int CW=128,PER=8*CW/16;
    uint4* stg=reinterpret_cast<uint4*>(v);
    for(int i=thread_idx;i<(BN/CW)*PER;i+=128) {
        stg[(i/PER)*(72*CW/16)+64*CW/16+(i%PER)]=make_uint4(0x38383838u,0x38383838u,0x38383838u,0x38383838u);
    }
    __syncthreads();
    constexpr bool Transpose_V=true;
        S2RTiledCopyVt s2r_tiled_copy_vt;
        R2STiledCopyV r2s_tiled_copy_v;
        auto s2r_thr_copy_vt = s2r_tiled_copy_vt.get_thread_slice(thread_idx);
        auto r2s_thr_copy_v = r2s_tiled_copy_v.get_thread_slice(thread_idx);
        Tensor tTranssVt_ = s2r_thr_copy_vt.partition_S(flat_divide(sVt, LDSM_divide_shape{}));
        Tensor tTranssV_ = r2s_thr_copy_v.partition_D(flat_divide(sV, STSM_divide_shape{}));
        CUTE_STATIC_ASSERT_V(rank(tTranssVt_) == rank(tTranssV_));
        CUTE_STATIC_ASSERT_V(size<0>(tTranssVt_) == size<0>(tTranssV_));
        CUTE_STATIC_ASSERT_V(size<1>(tTranssVt_) == size<1>(tTranssV_));
        CUTE_STATIC_ASSERT_V(size<2>(tTranssVt_) == size<2>(tTranssV_));
        CUTE_STATIC_ASSERT_V(size<3>(tTranssVt_) == size<3>(tTranssV_));
        CUTE_STATIC_ASSERT_V(size<4>(tTranssVt_) == size<4>(tTranssV_));
        static constexpr int Transpose_ILP = (size<2>(tTranssVt_) * size<3>(tTranssVt_)) % 2 == 0 ? 2 : 1;
        Tensor tTranssVt = logical_divide(group_modes<1, rank(tTranssVt_) - 1>(tTranssVt_), Shape<Underscore, Int<Transpose_ILP>>{});
        Tensor tTranssV = logical_divide(group_modes<1, rank(tTranssV_) - 1>(tTranssV_), Shape<Underscore, Int<Transpose_ILP>>{});
        auto transpose_V = [&](int stage) {
            if constexpr (Transpose_V) {
                #pragma unroll
                for (int i = 0; i < size<1, 1>(tTranssVt); ++i) {
                    Tensor tTransrV = make_fragment_like(tTranssV(_, make_coord(_, _0{}), _0{}));
                    static_assert(size<0>(tTransrV) == 16);
                    Tensor tTransrV_64 = recast<uint2>(tTransrV);
                    cute::copy(s2r_tiled_copy_vt, tTranssVt(_, make_coord(_, i), stage), tTransrV);
                    #pragma unroll
                    for (int j = 0; j < size(tTransrV_64); ++j) {
                        uint32_t upper = tTransrV_64[j].x;
                        uint32_t lower = tTransrV_64[j].y;
                        tTransrV_64[j].x = __byte_perm(upper, lower, 0x6420);
                        tTransrV_64[j].y = __byte_perm(upper, lower, 0x7531);
                    }
                    cute::copy(r2s_tiled_copy_v, tTransrV, tTranssV(_, make_coord(_, i), stage));
                }
            }
        };
    transpose_V(0);
    __syncthreads();
    int64_t const dst=((int64_t(batch)*2+head)*nblocks+nb)*TileBytes;
    for(int i=thread_idx;i<TileBytes/16;i+=128) {
        reinterpret_cast<uint4*>(vo+dst)[i]=reinterpret_cast<const uint4*>(v)[i];
    }
}
__device__ __forceinline__ uint4 z0_gp_ld(const uint4* p, uint64_t pol, int ef) {
    uint4 r;
    if (ef) {
        asm("ld.global.nc.L1::no_allocate.L2::cache_hint.v4.u32 {%0,%1,%2,%3}, [%4], %5;"
            : "=r"(r.x), "=r"(r.y), "=r"(r.z), "=r"(r.w) : "l"(p), "l"(pol));
    } else {
        r = __ldg(p);
    }
    return r;
}
__device__ __forceinline__ uint32_t z0_gp_w(uint4 const& v, int w) { return w == 0 ? v.x : w == 1 ? v.y : w == 2 ? v.z : v.w; }
// Page-1 gather without shared memory: one CTA per (256-key block, request), both KV heads; each thread loads 16 keys
// of 16 V bytes and transposes them in registers into the image tile. MODE 0: with the 8 ones rows; 1: loads only
// (timing); 2: no ones rows (gather part 2 here, Z0_GP); 3: evict_first loads.
template<int MODE>
__global__ void __launch_bounds__(128) z0_gp_k(const uint4* __restrict__ kc, const uint4* __restrict__ vc,
    const int* __restrict__ pt, const int* __restrict__ lengths, int S, int pages, int physical_pages,
    uint4* __restrict__ ko, uint4* __restrict__ vo) {
    asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
    int const t = threadIdx.x, g = t >> 3, l8 = t & 7, a = l8 & 3, h = l8 >> 2;
    int const nb = blockIdx.x, batch = blockIdx.y, nblocks = S >> 8;
    int const n = lengths[batch];
    int const key0 = nb * 256 + g * 16;
    uint64_t pol = 0;
    if (MODE == 3) asm("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;" : "=l"(pol));
    int e[2];
    #pragma unroll
    for (int i = 0; i < 2; ++i) {
        int const token = key0 + 2 * l8 + i;
        int r = -1;
        if (token < n && token < pages) {
            int pg = pt[(int64_t)batch * pages + token];
            pg = pg < 0 ? 0 : pg;
            if (pg < physical_pages) r = pg;
        }
        e[i] = r;
    }
    asm volatile("griddepcontrol.wait;" ::: "memory");
    int const lane0 = (t & 31) & ~7;
    int src[16];
    #pragma unroll
    for (int k = 0; k < 16; ++k) src[k] = __shfl_sync(0xffffffffu, e[k & 1], lane0 | (k >> 1));
    uint4 v[16];
    #pragma unroll
    for (int k = 0; k < 16; ++k) {
        v[k] = make_uint4(0, 0, 0, 0);
        if (src[k] >= 0) v[k] = z0_gp_ld(vc + (int64_t)src[k] * 8 + l8, pol, MODE == 3);
    }
    uint4* const kdst = ko + ((int64_t)batch * S + key0) * 8 + l8;
    uint32_t acc = 0;
    #pragma unroll
    for (int half = 0; half < 2; ++half) {
        uint4 kv[8];
        #pragma unroll
        for (int k = 0; k < 8; ++k) {
            kv[k] = make_uint4(0, 0, 0, 0);
            int const s = src[half * 8 + k];
            if (s >= 0) kv[k] = z0_gp_ld(kc + (int64_t)s * 8 + l8, pol, MODE == 3);
        }
        #pragma unroll
        for (int k = 0; k < 8; ++k) {
            if (MODE == 1) acc ^= kv[k].x ^ kv[k].y ^ kv[k].z ^ kv[k].w;
            else kdst[(half * 8 + k) * 8] = kv[k];
        }
    }
    if (MODE == 1) {
        #pragma unroll
        for (int k = 0; k < 16; ++k) acc ^= v[k].x ^ v[k].y ^ v[k].z ^ v[k].w;
        if (acc == 0x9e3779b9u && n == -7) ko[t] = make_uint4(acc, 0, 0, 0);
        return;
    }
    // out[e][q]: output chunk of dim 16a + e, word q holds keys (2q, 2q+1, 2q+8, 2q+9) of the group.
    int64_t const tile = ((int64_t)(batch * 2 + h) * nblocks + nb) * 1152 + (g >> 3) * 576;
    int const gl = g & 7;
    #pragma unroll
    for (int w = 0; w < 4; ++w) {
        uint32_t o[4][4];  // [j][q]: dim 4w + j
        #pragma unroll
        for (int q = 0; q < 4; ++q) {
            uint32_t const A = z0_gp_w(v[2 * q], w), B = z0_gp_w(v[2 * q + 1], w);
            uint32_t const C = z0_gp_w(v[2 * q + 8], w), D = z0_gp_w(v[2 * q + 9], w);
            uint32_t const t0 = __byte_perm(A, B, 0x5140), t1 = __byte_perm(A, B, 0x7362);
            uint32_t const t2 = __byte_perm(C, D, 0x5140), t3 = __byte_perm(C, D, 0x7362);
            o[0][q] = __byte_perm(t0, t2, 0x5410);
            o[1][q] = __byte_perm(t0, t2, 0x7632);
            o[2][q] = __byte_perm(t1, t3, 0x5410);
            o[3][q] = __byte_perm(t1, t3, 0x7632);
        }
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            int const ed = 4 * w + j;
            int const r = 16 * a + 8 * (ed & 1) + (ed >> 1);
            int const s = gl ^ (ed >> 1);
            vo[tile + r * 8 + s] = make_uint4(o[j][0], o[j][1], o[j][2], o[j][3]);
        }
    }
    if (MODE != 2) {
        #pragma unroll
        for (int i = 0; i < 2; ++i) {
            int const idx = t + 128 * i, hd = idx >> 7, rem = idx & 127;
            int64_t const tb = ((int64_t)(batch * 2 + hd) * nblocks + nb) * 1152 + (rem >> 6) * 576;
            vo[tb + (64 + ((rem >> 3) & 7)) * 8 + (rem & 7)] = make_uint4(0x38383838u, 0x38383838u, 0x38383838u, 0x38383838u);
        }
    }
}
// 8 keys per thread (twice the warps): a CTA owns one 128-key half of a 256-key block, both heads; partner lanes
// (lane ^ 8) hold keys 0-7 / 8-15 of a 16-key group and swap 16-bit key pairs so each writes 8 of the 16 dims.
template<int ONES>
__global__ void __launch_bounds__(128) z0_gp8_k(const uint4* __restrict__ kc, const uint4* __restrict__ vc,
    const int* __restrict__ pt, const int* __restrict__ lengths, int S, int pages, int physical_pages,
    uint4* __restrict__ ko, uint4* __restrict__ vo) {
    asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
    int const t = threadIdx.x, gl = t >> 4, kh = (t >> 3) & 1, l8 = t & 7, a = l8 & 3, h = l8 >> 2;
    int const nb = blockIdx.x >> 1, hh = blockIdx.x & 1, batch = blockIdx.y, nblocks = S >> 8;
    int const n = lengths[batch];
    int const key0 = nb * 256 + hh * 128 + gl * 16 + kh * 8;
    int e = -1;
    {
        int const token = key0 + l8;
        if (token < n && token < pages) {
            int pg = pt[(int64_t)batch * pages + token];
            pg = pg < 0 ? 0 : pg;
            if (pg < physical_pages) e = pg;
        }
    }
    asm volatile("griddepcontrol.wait;" ::: "memory");
    int const lane0 = (t & 31) & ~7;
    int src[8];
    #pragma unroll
    for (int k = 0; k < 8; ++k) src[k] = __shfl_sync(0xffffffffu, e, lane0 | k);
    uint4 v[8], kv[8];
    #pragma unroll
    for (int k = 0; k < 8; ++k) {
        v[k] = make_uint4(0, 0, 0, 0);
        kv[k] = v[k];
        if (src[k] >= 0) {
            v[k] = __ldg(vc + (int64_t)src[k] * 8 + l8);
            kv[k] = __ldg(kc + (int64_t)src[k] * 8 + l8);
        }
    }
    uint4* const kdst = ko + ((int64_t)batch * S + key0) * 8 + l8;
    #pragma unroll
    for (int k = 0; k < 8; ++k) kdst[k * 8] = kv[k];
    int64_t const tile = ((int64_t)(batch * 2 + h) * nblocks + nb) * 1152 + hh * 576;
    #pragma unroll
    for (int wi = 0; wi < 2; ++wi) {
        uint32_t lo[4][2], hi[4][2];
        #pragma unroll
        for (int p = 0; p < 4; ++p) {
            int const wk = wi + 2 * kh, ws = wi + 2 * (kh ^ 1);
            uint32_t const A = z0_gp_w(v[2 * p], wk), B = z0_gp_w(v[2 * p + 1], wk);
            uint32_t const As = z0_gp_w(v[2 * p], ws), Bs = z0_gp_w(v[2 * p + 1], ws);
            uint32_t const k0 = __byte_perm(A, B, 0x5140), k1 = __byte_perm(A, B, 0x7362);
            uint32_t const s0 = __byte_perm(As, Bs, 0x5140), s1 = __byte_perm(As, Bs, 0x7362);
            uint32_t const r0 = __shfl_xor_sync(0xffffffffu, s0, 8), r1 = __shfl_xor_sync(0xffffffffu, s1, 8);
            lo[p][0] = kh ? r0 : k0; hi[p][0] = kh ? k0 : r0;
            lo[p][1] = kh ? r1 : k1; hi[p][1] = kh ? k1 : r1;
        }
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            int const ed = 4 * (wi + 2 * kh) + j;
            uint32_t const sel = (j & 1) ? 0x7632 : 0x5410;
            uint32_t o[4];
            #pragma unroll
            for (int q = 0; q < 4; ++q) o[q] = __byte_perm(lo[q][j >> 1], hi[q][j >> 1], sel);
            int const r = 16 * a + 8 * (ed & 1) + (ed >> 1);
            int const s = gl ^ (ed >> 1);
            vo[tile + r * 8 + s] = make_uint4(o[0], o[1], o[2], o[3]);
        }
    }
    if (ONES) {
        int const hd = t >> 6;
        int64_t const tb = ((int64_t)(batch * 2 + hd) * nblocks + nb) * 1152 + hh * 576;
        vo[tb + (64 + ((t >> 3) & 7)) * 8 + (t & 7)] = make_uint4(0x38383838u, 0x38383838u, 0x38383838u, 0x38383838u);
    }
}
void z0_gp_launch(torch::Tensor kc, torch::Tensor vc, torch::Tensor pt, torch::Tensor lengths, torch::Tensor ko,
                  torch::Tensor vo, int mode) {
    int const B = pt.size(0), S = ko.size(1), pages = pt.size(1);
    TORCH_CHECK(S % 256 == 0 && ATT_BN == 256 && kc.size(1) == 1, "z0_gp");
    dim3 grid(S / 256, B);
    auto st = at::cuda::getCurrentCUDAStream();
    const uint4* k = reinterpret_cast<const uint4*>(kc.data_ptr());
    const uint4* v = reinterpret_cast<const uint4*>(vc.data_ptr());
    uint4* o1 = reinterpret_cast<uint4*>(ko.data_ptr());
    uint4* o2 = reinterpret_cast<uint4*>(vo.data_ptr());
    int const* p = pt.data_ptr<int>();
    int const* l = lengths.data_ptr<int>();
    int const ph = (int)kc.size(0);
    if (mode == 4) z0_gp8_k<1><<<dim3(S / 128, B), 128, 0, st>>>(k, v, p, l, S, pages, ph, o1, o2);
    else if (mode == 5) z0_gp8_k<0><<<dim3(S / 128, B), 128, 0, st>>>(k, v, p, l, S, pages, ph, o1, o2);
    else if (mode == 1) z0_gp_k<1><<<grid, 128, 0, st>>>(k, v, p, l, S, pages, ph, o1, o2);
    else if (mode == 3) z0_gp_k<3><<<grid, 128, 0, st>>>(k, v, p, l, S, pages, ph, o1, o2);
    else if (mode == 2) {
        cudaLaunchAttribute la[1];
        la[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
        la[0].val.programmaticStreamSerializationAllowed = 1;
        cudaLaunchConfig_t lc = {grid, dim3(128), 0, st, la, 1};
        cudaLaunchKernelEx(&lc, z0_gp_k<2>, k, v, p, l, S, pages, ph, o1, o2);
    }
    else z0_gp_k<0><<<grid, 128, 0, st>>>(k, v, p, l, S, pages, ph, o1, o2);
}
// Page-1 caches: part 2 runs z0_gp_k in mode Z0_GP from Z0_GP_MINB requests up, parts 21..26 run z0_gp_launch modes
// 0..5; part 2 below Z0_GP_MINB requests and part 20 run pv_gather_stage_k. Larger pages always run pv_gather_stage_k.
void pv_gather_stage(torch::Tensor kc,torch::Tensor vc,torch::Tensor pt,torch::Tensor lengths,torch::Tensor ko,torch::Tensor vo,torch::Tensor cu_q,int64_t part) {
    if(kc.size(1)==1 && part!=20 && (part!=2 || pt.size(0)>=Z0_GP_MINB)) { TORCH_CHECK(vo.numel()==(int64_t)pt.size(0)*ko.size(1)*2*72,"pv_gather storage"); z0_gp_launch(kc,vc,pt,lengths,ko,vo,part==2?Z0_GP:(int)part-21); return; }
    int const B=pt.size(0),S=ko.size(1),pages=pt.size(1),page=kc.size(1);
    TORCH_CHECK(vo.numel()==(int64_t)B*S*2*72,"pv_gather storage");
    {
    cudaLaunchAttribute la[1];
    la[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    la[0].val.programmaticStreamSerializationAllowed = 1;
    unsigned const nla = 1;
    cudaLaunchConfig_t lc = {dim3(2,S/ATT_BN,B), dim3(128), 0, at::cuda::getCurrentCUDAStream(), la, nla};
    cudaLaunchKernelEx(&lc, pv_gather_stage_k<PvImageMainloop>, (const uint8_t*)kc.data_ptr(), (const uint8_t*)vc.data_ptr(),
        (const int*)pt.data_ptr<int>(), (const int*)lengths.data_ptr<int>(), S, pages, page, (int)kc.size(0),
        (uint8_t*)ko.data_ptr(), (uint8_t*)vo.data_ptr(), (const int*)cu_q.data_ptr<int>());
    }
}
#ifdef AT_SUB
__device__ __forceinline__ float at_bfr(float x) { return __bfloat162float(__float2bfloat16(x)); }
__device__ __forceinline__ uint16_t at_fp8x2(float a, float b) {
  return (uint16_t)__nv_cvt_float2_to_fp8x2(make_float2(a, b), __NV_SATFINITE, __NV_E4M3);
}
// Full-layer front, one warp per token: RMSNorm (no weight) of the 16 q and 2 k heads, BF16 rounding, then FP8
// (saturating) of q x qmul into qo and of k and the unnormalized v into the cache row loc[t]. qkv rows are
// [16 q | 2 k | 2 v | gate] x 64 BF16 (2304 values); the gate is not read here.
__global__ void __launch_bounds__(256) at_front_k(const __nv_bfloat162* __restrict__ qkv, const long* __restrict__ loc,
    uint16_t* __restrict__ qo, uint8_t* __restrict__ kc, uint8_t* __restrict__ vc, int T, float eps, float qmul) {
  asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
  asm volatile("griddepcontrol.wait;" ::: "memory");
  const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
  const int t = blockIdx.x * 8 + warp;
  if (t >= T) return;
  const __nv_bfloat162* row = qkv + (size_t)t * 1152;
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
      x0 = at_bfr(x0); x1 = at_bfr(x1);
    }
    if (h < 16) qo[((size_t)t * 16 + h) * 32 + lane] = at_fp8x2(x0 * qmul, x1 * qmul);
    else if (h < 18) *reinterpret_cast<uint16_t*>(kc + L * 128 + (h - 16) * 64 + 2 * lane) = at_fp8x2(x0, x1);
    else *reinterpret_cast<uint16_t*>(vc + L * 128 + (h - 18) * 64 + 2 * lane) = at_fp8x2(x0, x1);
  }
}
void att_front(torch::Tensor qkv, torch::Tensor loc, torch::Tensor qo, torch::Tensor kc, torch::Tensor vc, double eps,
               double qmul) {
  const int T = (int)qkv.size(0);
  TORCH_CHECK(qkv.size(1) == 2304 && qkv.stride(0) == 2304, "att_front: qkv layout");
  cudaLaunchAttribute la[1];
  la[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  la[0].val.programmaticStreamSerializationAllowed = 1;
  cudaLaunchConfig_t lc = {dim3((T + 7) / 8), dim3(256), 0, at::cuda::getCurrentCUDAStream(), la, 1};
  cudaLaunchKernelEx(&lc, at_front_k, (const __nv_bfloat162*)qkv.data_ptr(), (const long*)loc.data_ptr<long>(),
                     (uint16_t*)qo.data_ptr(), (uint8_t*)kc.data_ptr(), (uint8_t*)vc.data_ptr(), T, (float)eps,
                     (float)qmul);
}
#endif
