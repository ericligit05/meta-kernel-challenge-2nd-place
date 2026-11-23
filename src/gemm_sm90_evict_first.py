"""SM90 BF16 GEMM out = x @ w.T compiled through CuTe DSL, for the unpooler in layer.py.

The same kernel as gemm_sm90.py with the weight (B) TMA loads under an L2 evict_first cache policy, on QuACK's
static tile scheduler. layer.py compiles it in prepare (compile_gemm) and calls gemm() for 8 < T < UNP_MAX tokens.
AVAILABLE is False when QuACK / CuTe DSL cannot be imported, and layer.py then uses cuBLASLt.
"""
# The device kernel below is a trimmed copy of QuACK's GemmSm90.kernel (quack-kernels 0.6.5, quack/gemm_sm90.py,
# Copyright (c) 2025-2026 QuACK team, Apache License 2.0; itself based on the CUTLASS CuTe DSL hopper dense_gemm
# example). Changes against QuACK's kernel: only the dense path is kept (no varlen, gather, aux operand, A transform,
# C operand or split-K); the producer issues the B (weight) loads of the first ab_stage k-tiles of its first tile
# BEFORE griddepcontrol.wait (weights do not depend on the previous kernel), and the A loads of those stages after
# it; the consumers fire griddepcontrol.launch_dependents at the end of the kernel instead of right after the
# mainloop. Nothing in QuACK is modified: the class below subclasses QuACK's GemmDefaultSm90 in this module and
# QuACK's own host helpers build, compile and launch it. Math (wgmma SS, FP32 accumulation in K order, BF16 store
# through smem + TMA) is QuACK's, so the output equals QuACK's and cuBLAS's bit for bit.
from functools import partial

import torch

try:  # QuACK and CuTe DSL ship in the judge image; without them the layer keeps cuBLASLt
    import cutlass
    import cutlass.cute as cute
    import cutlass.pipeline as pipeline
    from cutlass import Boolean, Float32, Int32, Int64, const_expr
    from cutlass.cute.nvgpu import cpasync
    from cutlass.experimental import primitives as prims
    from cutlass.pipeline import pipeline_init_arrive, pipeline_init_wait
    from cutlass.utils import SmemPartition

    import quack.copy_utils as copy_utils
    import quack.sm90_utils as quack_sm90_utils
    from quack.cute_dsl_utils import get_device_capacity, get_max_active_clusters
    from quack.gemm_default_epi import GemmDefaultEpiMixin, GemmDefaultSm90
    from quack.gemm_tvm_ffi_utils import (compile_gemm_kernel, get_dtypes, get_majors, make_fake_gemm_tensors,
                                          make_fake_scheduler_args, make_scheduler_args)
    from quack.pipeline import make_pipeline_state
    from quack.rounding import RoundingMode
    from quack.varlen_utils import VarlenManager

    AVAILABLE = True
except Exception:  # noqa: BLE001
    AVAILABLE = False


# L2 eviction priority of the weight (B) TMA loads, CUTLASS's CacheHintSm90::EVICT_FIRST policy word. The weights
# stream through L2 once per cluster column, so evicting them first keeps the A rows resident, and with them the FP32
# residual that boundary_norm.cu's attention boundary stored evict_last for the FFN boundary after this GEMM. Hint
# only: no math or data change. (X45_ is a naming tag of the submitted code.)
X45_B_POLICY = 0x12F0000000000000

if AVAILABLE:
    # The class name differs from the other GEMM file's on purpose: the build merges every component into one
    # namespace and does not prefix this indented class, so a shared name would rebind the other file's class (Ef:
    # evict_first).
    class PdlGemmSm90Ef(GemmDefaultSm90):
        """QuACK's SM90 GEMM with weights loaded before the PDL wait and the dependents' trigger at kernel end."""

        @cute.kernel
        def kernel(self, tiled_mma, tma_atom_a, mA_mkl, tma_atom_b, mB_nkl, tma_atom_sfa, mSFA_mkl, tma_atom_sfb,
                   mSFB_nkl, tma_atom_d, mD_mnl, tma_atom_c, mC_mnl, epilogue_params, varlen_params, cluster_layout_mnk,
                   a_smem_layout, b_smem_layout, sfa_smem_layout, sfb_smem_layout, epi_smem_layout, epi_c_smem_layout,
                   tma_atom_aux_a, mAuxA_mkl, aux_a_smem_layout, tile_sched_params,
                   TileSchedulerCls: cutlass.Constexpr):
            assert varlen_params.cu_seqlens_m is None and varlen_params.cu_seqlens_k is None
            assert not self.gather_A and self.transform_a is None and self.aux_a is None and self.split_k == 1
            assert mC_mnl is None and mD_mnl is not None and self.epi_c_stage == 0 and self.use_pdl
            warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
            if warp_idx == self.ab_load_warp_id:
                for tma_atom in (tma_atom_a, tma_atom_b, tma_atom_d):
                    cpasync.prefetch_descriptor(tma_atom)
            smem = cutlass.utils.SmemAllocator()
            storage = smem.allocate(self.shared_storage)
            ab_pipeline = self.make_ab_pipeline(tiled_mma=tiled_mma,
                                                cluster_layout_vmnk=cute.make_layout((1, *cluster_layout_mnk.shape)))
            sched_pipeline = None
            sched_data = None
            if const_expr(self.is_persistent):
                sched_pipeline = self.make_sched_pipeline(cluster_layout_mnk, varlen_k=False)
                sched_data = smem.allocate_tensor(Int32, cute.make_layout((4, self.sched_stage)), byte_alignment=16,
                                                  partition=SmemPartition.RESERVED)
            pipeline_init_arrive(cluster_shape_mn=self.cluster_shape_mnk[:-1], is_relaxed=True)
            sA = storage.sA.get_tensor(a_smem_layout.outer, swizzle=a_smem_layout.inner)
            sB = storage.sB.get_tensor(b_smem_layout.outer, swizzle=b_smem_layout.inner)
            sD = storage.sD.get_tensor(epi_smem_layout.outer, swizzle=epi_smem_layout.inner)
            epi_smem_tensors = self.epi_get_smem_tensors(epilogue_params, storage)
            varlen_manager = VarlenManager.create(varlen_params, len_m_static=Int32(cute.size(mA_mkl, mode=[0])),
                                                  len_k_static=Int32(cute.size(mB_nkl, mode=[1])),
                                                  len_n_static=Int32(cute.size(mB_nkl, mode=[0])))
            TileSchedulerCls = partial(TileSchedulerCls.create, tile_sched_params, sched_data, sched_pipeline)
            pipeline_init_wait(cluster_shape_mn=self.cluster_shape_mnk[:-1])
            k_tile_cnt_static = cute.ceil_div(cute.size(mA_mkl, mode=[1]), self.cta_tile_shape_mnk[2])

            if warp_idx >= self.ab_load_warp_id:
                cute.arch.setmaxregister_decrease(self.num_regs_load)
                if warp_idx >= self.ab_load_warp_id and warp_idx < self.ab_load_warp_id + self.num_ab_load_warps:
                    a_tma_multicast = {"cluster_shape": self.cluster_shape_mnk[:2], "multicast_dim": "M"}
                    b_tma_multicast = {"cluster_shape": self.cluster_shape_mnk[:2], "multicast_dim": "N"}
                    is_scheduler_warp = self.num_ab_load_warps == 1 or warp_idx == self.ab_load_warp_id
                    if const_expr(cute.size(cluster_layout_mnk) > 1):
                        is_scheduler_warp = is_scheduler_warp and cute.arch.block_idx_in_cluster() == 0
                    tile_scheduler = TileSchedulerCls()
                    work_tile = tile_scheduler.initial_work_tile_info()
                    ab_producer_state = make_pipeline_state(pipeline.PipelineUserType.Producer, self.ab_stage)
                    # The weights of the first stages of the first tile go out before the PDL wait.
                    n_pre = Int32(0)
                    if work_tile.is_valid_tile:
                        gB0 = cute.local_tile(varlen_manager.offset_batch_B(mB_nkl, work_tile.tile_idx[3]),
                                              cute.select(self.cta_tile_shape_mnk, [1, 2]), (work_tile.tile_idx[1], None))
                        copy_B0 = copy_utils.tma_get_block_copy_fn(tma_atom_b, src_tensor=gB0, dst_tensor=sB,
                                                                   tma_multicast=b_tma_multicast,
                                                                   cache_policy=Int64(X45_B_POLICY))
                        n_pre = Int32(k_tile_cnt_static)
                        if n_pre > self.ab_stage:
                            n_pre = Int32(self.ab_stage)
                        pre_state = ab_producer_state.clone()
                        for k_tile in cutlass.range(n_pre, unroll=1):
                            ab_pipeline.producer_acquire(pre_state)
                            copy_B0(k_tile, pre_state.index, tma_bar_ptr=ab_pipeline.producer_get_barrier(pre_state))
                            pre_state.advance()
                    prims.griddepcontrol(prims.GridDepAction.WAIT)
                    while work_tile.is_valid_tile:
                        tile_coord_mnkl = work_tile.tile_idx
                        batch_idx = tile_coord_mnkl[3]
                        gA_mk = cute.local_tile(varlen_manager.offset_batch_A(mA_mkl, batch_idx),
                                                cute.select(self.cta_tile_shape_mnk, [0, 2]), (tile_coord_mnkl[0], None))
                        copy_A = copy_utils.tma_get_block_copy_fn(tma_atom_a, src_tensor=gA_mk, dst_tensor=sA,
                                                                  tma_multicast=a_tma_multicast)
                        gB_nk = cute.local_tile(varlen_manager.offset_batch_B(mB_nkl, batch_idx),
                                                cute.select(self.cta_tile_shape_mnk, [1, 2]), (tile_coord_mnkl[1], None))
                        copy_B = copy_utils.tma_get_block_copy_fn(tma_atom_b, src_tensor=gB_nk, dst_tensor=sB,
                                                                  tma_multicast=b_tma_multicast,
                                                                  cache_policy=Int64(X45_B_POLICY))
                        k_tile_cnt = cute.ceil_div(varlen_manager.len_k(batch_idx), self.cta_tile_shape_mnk[2])
                        # A of the stages whose B went out before the wait (their barriers already expect A + B)
                        for k_tile in cutlass.range(n_pre, unroll=1):
                            copy_A(k_tile, ab_producer_state.index,
                                   tma_bar_ptr=ab_pipeline.producer_get_barrier(ab_producer_state))
                            ab_pipeline.producer_commit(ab_producer_state)
                            ab_producer_state.advance()
                        ab_producer_state = self.load_tma(ab_pipeline, ab_producer_state, [copy_A, copy_B, None],
                                                          k_tile_cnt - n_pre, k_tile_start=n_pre)
                        n_pre = Int32(0)
                        tile_scheduler.advance_to_next_work(is_scheduler_warp=is_scheduler_warp)
                        work_tile = tile_scheduler.get_current_work()
                    if const_expr(self.pingpong):
                        if is_scheduler_warp:
                            tile_scheduler.write_work_tile_to_smem(work_tile)
                        work_tile = tile_scheduler.get_current_work()
                    if warp_idx == self.ab_load_warp_id:
                        ab_pipeline.producer_tail(ab_producer_state)
                    if is_scheduler_warp:
                        tile_scheduler.producer_tail()

            if warp_idx < self.ab_load_warp_id:
                cute.arch.setmaxregister_increase(self.num_regs_mma)
                is_tma_warp = Boolean((not self.pingpong and warp_idx == 0)
                                      or (self.pingpong and (warp_idx == 0 or warp_idx == 4)))
                tidx, _, _ = cute.arch.thread_idx()
                warp_group_idx = cute.arch.make_warp_uniform(tidx // self.num_threads_per_warp_group)
                if const_expr(self.pingpong):
                    tidx = tidx % self.num_threads_per_warp_group
                warp_group_thread_layout = cute.make_layout(
                    self.mma_warp_groups if const_expr(not self.pingpong) else 1, stride=self.num_threads_per_warp_group)
                thr_mma = tiled_mma.get_slice(warp_group_thread_layout(warp_group_idx if not self.pingpong else 0))
                acc, tCrA, tCrB = quack_sm90_utils.partition_fragment_ABC(thr_mma, self.cta_tile_shape_mnk, sA, sB)
                assert not self.mma_is_rs and not self.fp8_slow_accum
                mma_fn = partial(quack_sm90_utils.gemm_w_idx, tiled_mma, acc, tCrA, tCrB)
                if const_expr(self.pingpong):
                    if warp_group_idx == 0:
                        self.pingpong_barrier_arrive(warp_group_idx=0, stage="mma")
                        self.pingpong_barrier_arrive(warp_group_idx=0, stage="epi")
                c_tile_cnt = cute.size(self.epi_tile_shape)
                ab_read_state = make_pipeline_state(pipeline.PipelineUserType.Consumer, self.ab_stage)
                epi_store_pipeline = self.make_epi_store_pipeline()
                epi_read_state = make_pipeline_state(pipeline.PipelineUserType.Consumer, self.epi_c_stage)
                epi_producer_state = make_pipeline_state(pipeline.PipelineUserType.Producer, self.epi_c_stage)
                tile_scheduler = TileSchedulerCls()
                work_tile = tile_scheduler.initial_work_tile_info()
                if const_expr(self.pingpong):
                    if warp_idx >= 4:
                        epi_read_state.advance_iters(c_tile_cnt)
                        epi_producer_state.advance_iters(c_tile_cnt)
                        ab_read_state.advance_iters(k_tile_cnt_static)
                        tile_scheduler.advance_to_next_work()
                        work_tile = tile_scheduler.get_current_work()
                while work_tile.is_valid_tile:
                    tile_coord_mnkl = work_tile.tile_idx
                    batch_idx = tile_coord_mnkl[3]
                    k_tile_cnt = cute.ceil_div(varlen_manager.len_k(batch_idx), self.cta_tile_shape_mnk[2])
                    if const_expr(self.pingpong):
                        self.pingpong_barrier_sync(warp_group_idx, stage="mma")
                    ab_read_state = self.mma(ab_pipeline, ab_read_state, mma_fn, acc, None, k_tile_cnt, warp_group_idx)
                    if const_expr(self.pingpong):
                        self.pingpong_barrier_sync(warp_group_idx, "epi")
                    copy_D, _, _ = self.epilog_gmem_copy_and_partition(
                        tma_atom_d, varlen_manager.offset_batch_epi(mD_mnl, batch_idx), self.cta_tile_shape_mnk[:2],
                        self.epi_tile, sD, tile_coord_mnkl)
                    tiled_copy_r2s, tRS_rD, tRS_sD = self.epilog_smem_store_and_partition(
                        tiled_mma, self.d_layout, self.d_dtype, sD, tidx)
                    tRS_rAcc = self.epi_retile_acc(acc, tRS_rD, tiled_copy_r2s)
                    load_acc_subtile = partial(self.epi_load_acc_subtile, tRS_rAcc)
                    self.epi_visit_acc(epilogue_params, acc, tiled_mma, tile_coord_mnkl, tidx)
                    epi_fn = partial(
                        self.epilogue, epilogue_params, epi_smem_tensors, None, epi_store_pipeline, epi_read_state,
                        epi_producer_state, self.epi_tile, tRS_rD=tRS_rD, tRS_rC=None, tiled_copy_t2r=None,
                        tiled_copy_r2s=tiled_copy_r2s, tRS_sD=tRS_sD, tiled_copy_s2r=None, tSR_rC=None, tSR_sC=None,
                        copy_D=copy_D, copy_C=None, tile_coord_mnkl=tile_coord_mnkl, varlen_manager=varlen_manager,
                        epilogue_barrier=self.epilogue_barrier, tile_scheduler=tile_scheduler, tidx=tidx,
                        is_tma_warp=is_tma_warp)
                    epi_read_state, epi_producer_state = self.epilogue_split_k(
                        epilogue_params, epi_fn, load_acc_subtile, tRS_rD, self.epi_tile, epi_read_state,
                        epi_producer_state, epi_store_pipeline, tile_coord_mnkl, self.epilogue_barrier, tidx, is_tma_warp)
                    if const_expr(self.pingpong):
                        if is_tma_warp:
                            epi_store_pipeline.producer_tail()
                        self.pingpong_barrier_arrive(1 - warp_group_idx, stage="epi")
                    if const_expr(not self.pingpong):
                        tile_scheduler.advance_to_next_work()
                        work_tile = tile_scheduler.get_current_work()
                    else:
                        epi_read_state.advance_iters(c_tile_cnt)
                        epi_producer_state.advance_iters(c_tile_cnt)
                        ab_read_state.advance_iters(k_tile_cnt_static)
                        tile_scheduler.advance_to_next_work(advance_count=self.mma_warp_groups)
                        work_tile = tile_scheduler.get_current_work()
                if const_expr(not self.pingpong):
                    if is_tma_warp:
                        epi_store_pipeline.producer_tail()
                # Dependents launch only once this CTA's output tiles are stored (QuACK: after the mainloop).
                prims.griddepcontrol(prims.GridDepAction.LAUNCH_DEPENDENTS)


_CACHE = {}


def compile_gemm(x, w, tile_m=128, tile_n=192, pingpong=True, cluster_m=1, cluster_n=1, max_swizzle_size=8):
    """Compile (once per configuration and operand layout) out = x @ w.T for BF16 x (M, K) and w (N, K)."""
    out = torch.empty((x.shape[0], w.shape[0]), dtype=torch.bfloat16, device=x.device)
    B = w.t()  # (K, N) view, passed as QuACK's b_kn operand (the kernel reads it k-major)
    key = (tile_m, tile_n, pingpong, cluster_m, cluster_n, max_swizzle_size, x.stride(), B.stride(), x.device.index)
    if key in _CACHE:
        return key
    cap = get_device_capacity(x.device)
    assert cap[0] == 9, "PdlGemmSm90Ef is an SM90 kernel"
    a_major, b_major, d_major, c_major = get_majors(x, B, out, None)
    b_major = "n" if B.stride(-1) == 1 else "k"
    a_dt, b_dt, d_dt, c_dt = get_dtypes(x, B, out, None)
    mA, mB, mD, mC, m, n, k, l = make_fake_gemm_tensors(a_dt, b_dt, d_dt, c_dt, a_major, b_major, d_major, c_major,
                                                        batched=False, b_kn=True)
    epi = GemmDefaultEpiMixin.EpilogueArguments(add_to_output=False, rounding_mode=RoundingMode.RN)
    fn = compile_gemm_kernel(PdlGemmSm90Ef, a_dt, (tile_m, tile_n), (cluster_m, cluster_n, 1), pingpong, True, False,
                             False, cap, mA, mB, mD, mC, epi, make_fake_scheduler_args(False, False, l), None,
                             b_transposed=True)
    epi_run = GemmDefaultEpiMixin.EpilogueArguments(alpha=None, beta=None, mRowVecBroadcast=None,
                                                    mColVecBroadcast=None, add_to_output=None, rounding_mode=None,
                                                    sr_seed=None)
    sched = make_scheduler_args(get_max_active_clusters(cluster_m * cluster_n, device_capacity=cap),
                                max_swizzle_size, None)
    _CACHE[key] = (fn, epi_run, sched)
    return key


def gemm(x, w, key):
    """out = x @ w.T (BF16 out, FP32 accumulation) on the compiled kernel `key` (from compile_gemm)."""
    if key not in _CACHE:  # a process that did not run prepare (the key holds the whole configuration)
        compile_gemm(x, w, *key[:6])
    fn, epi_run, sched = _CACHE[key]
    out = torch.empty((x.shape[0], w.shape[0]), dtype=torch.bfloat16, device=x.device)
    fn(x, w.t(), out, None, epi_run, sched, None, None, None)
    return out
