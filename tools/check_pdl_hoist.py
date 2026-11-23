"""check_pdl_hoist: SASS-level guard against loads of a predecessor's output issued before the PDL wait.

Hazard: with a pointer to the producer's output marked const __restrict__, nvcc may treat the load as invariant and
hoist it above griddepcontrol.wait. Under PDL the kernel then reads the PREVIOUS replay's data (or a half-written buffer): invisible when every replay gets the same
inputs, a wrong answer on fresh inputs. This tool reads the SASS ptxas emitted and proves, per kernel, that every
memory read issued before the wait reads a buffer that no kernel of the layer writes (weights / tables written in
prepare(), kernel parameters, layer inputs that nothing in the call overwrites).

Model:
  - griddepcontrol.wait lowers to ACQBULK on sm_90a (test kernels, CUDA 13.0.88); cudaGridDependencySynchronize and
    cutlass::arch::wait_on_dependent_grids lower to the same instruction; launch_dependents lowers to PREEXIT.
  - Over the CFG of check_rs_inflight.link, per function with a wait:
      W  reachable from the entry over forward edges without an ACQBULK, and able to reach one: the waiting thread's
         own code before its wait. A nvcc hoist of a producer load lands here.
      N  reachable that way but never reaching a wait: a role that does not wait (consumer warpgroups fed by a
         waiting producer, early-exit lanes).
      L  pre-wait only through a backward branch (a loop that runs the wait on an earlier iteration): reported only.
  - Checked accesses: LDG / LD / LDGSTS / UBLKCP / UTMALDG / atomics (enforced) and prefetches / stores (reported).
  - Each access's address is traced back through R and UR registers to its roots: kernel parameters c[0x0][0x210 +
    off] (ordinals from EIATTR_KPARAM_INFO, by-value struct fields as "N+0xOFF"), the parameter-space base
    c[0x0][0x198] + immediate (grid_constant tensor maps used by TMA), other constant banks, special registers,
    shared-memory loads (lds@) and other global loads (via:).
  - TABLE (kernel name substring -> {param / field: class}) classes: prep (weights, prepare-time tables), input (a
    layer input no kernel of the call writes), scalar, prior (written in this call by a kernel that must be complete
    before this one can launch; the note says why), self (this kernel's own workspace, own synchronization), prod
    (a predecessor's output). Verdict: W and N accesses of a prod buffer FAIL, except N accesses whose address
    depends on a shared-memory value (ordered after the waiting producer's hand-off); unknown kernels / params, TMA
    descriptors and constant-bank pointers that cannot be resolved FAIL unless the kernel's TABLE entry
    acknowledges them (tma_ack / cbank_ack / loop_wait_ack, each with its reason).
Per kernel it prints the waits, the W / N / L sizes and every checked access with its roots and verdict.

TABLE is specific to this layer: it names this submission's kernels and classifies each of their parameters from the
host code that launches them. Another layer needs its own TABLE (--survey lists what to classify), and a change here
to a kernel's parameter list or to a by-value struct's layout needs its TABLE entry updated, or the check fails.

    python3 tools/check_pdl_hoist.py submission/submission.py [--only SUBSTR] [--survey] [--no-dsl]
        on an H200: import the file, check every torch extension .so it loads, and compile and check its
        CuTe DSL GEMMs (QuACK o projection, src/gemm_sm90.py and src/gemm_sm90_evict_first.py) from cubins dumped
        with CUTE_DSL_KEEP=cubin (QuACK cache off)
    python3 tools/check_pdl_hoist.py --so A.so [...] / --sass DUMP.sass [...]
    python3 tools/check_pdl_hoist.py --selftest       synthetic positive and negative controls (CPU)
Prints PDL_HOIST_PASS or PDL_HOIST_FAIL at the end; exit 1 on any hazard, unclassified access or setup error.
--survey prints every pre-wait access with its roots and never fails on classification (for building TABLE). Reading
SASS needs cuobjdump (and cu++filt) or nvdisasm from the CUDA toolkit; --selftest needs neither. Importing the
submission needs the problem files on sys.path: set MKC_PROBLEM_DIR, default /okc/problems/muse_prefill_layer, the
challenge container's own path.

Result: every candidate build went through this check before it was submitted, the submitted one included; the
recorded runs cover 53 kernels with a PDL wait and 0 failures. It never caught a real hoist. Its one failure during
development was a stale TABLE entry: a by-value epilogue struct grew by 16 bytes, which moved two of the FA3
scheduler's own workspace fields to offsets TABLE did not list, so the check took them for a predecessor's output.
The fix kept the struct layout unchanged.
"""
import argparse
import collections
import os
import re
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import check_rs_inflight as rsi  # noqa: E402  (SASS parser, CFG link, bundle import, cuobjdump wrapper)

WAIT_OPS = ("ACQBULK",)
LOAD_OPS = ("LDG", "LD", "LDGSTS", "UBLKCP", "UTMALDG", "ATOMG", "ATOM", "RED", "REDG", "SULD")
PF_OPS = ("UTMAPF", "CCTL", "UBLKPF")
STORE_OPS = ("STG", "ST", "UTMASTG", "UBLKRED", "UTMAREDG")
PARAM_BASE = 0x210
RREF = re.compile(r"(?<![A-Z_.])(U?R)(\d+)(\.64|\.128)?(?![0-9])")
CPARAM = re.compile(r"c\[0x0\]\[(0x[0-9a-f]+)\]")
CBANK = re.compile(r"c\[(0x[0-9a-f]+)\]\[")


def regs_in(s):
    out = set()
    for m in RREF.finditer(s):
        if m.group(1) == "R" and m.group(2) == "Z":
            continue
        r = int(m.group(2))
        w = 4 if m.group(3) == ".128" else 2 if m.group(3) == ".64" else 1
        out.update((m.group(1), r + k) for k in range(w))
    return out


def width_of(op):
    return 4 if ".128" in op else 2 if (".64" in op or "WIDE" in op) else 1


def dst_of(ins):
    """(kind, number) registers an instruction writes: its first operand when that is a register (second operand
    when the first is a predicate). Conservative for tracing: unknown writers are treated as defining nothing."""
    if not ins.ops:
        return set()
    first = ins.ops[0]
    if re.match(r"^!?U?P[T0-9]$", first) and len(ins.ops) > 1:
        first = ins.ops[1]
    m = re.match(r"^(U?R)(\d+)$", first)
    if not m:
        return set()
    base = ins.op.split(".")[0]
    if base in ("STG", "ST", "STS", "STL", "RED", "REDG", "UTMASTG", "UTMALDG", "UBLKCP", "UTMAPF", "BAR", "SYNCS",
                "MEMBAR", "ACQBULK", "PREEXIT"):
        return set()
    w = width_of(ins.op) if base in ("LDG", "LD", "LDS", "LDL", "LDC", "ULDC", "IMAD", "UIMAD", "MOV", "UMOV", "CS2R",
                                      "S2UR", "IADD3", "UIADD3", "LEA", "ULEA") else 1
    if base in ("IMAD", "UIMAD", "IADD3", "UIADD3", "LEA", "ULEA", "MOV", "UMOV") and "WIDE" not in ins.op and ".64" not in ins.op:
        w = 1
    return {(m.group(1), int(m.group(2)) + k) for k in range(w)}


def addr_operands(ins):
    """The operand strings that hold the global address of a memory instruction."""
    base = ins.op.split(".")[0]
    ops = ins.ops
    if base in ("LDG", "LD", "SULD"):
        return [o for o in ops if "[" in o][-1:]
    if base == "LDGSTS":
        return [o for o in ops if "[" in o][1:2]
    if base in ("ATOMG", "ATOM", "RED", "REDG", "STG", "ST"):
        return [o for o in ops if "[" in o][:1]
    if base in ("UBLKCP",):
        return [o for o in ops if "[" in o][1:2]
    if base in ("UTMALDG", "UTMAPF", "UTMASTG"):
        br = [o for o in ops if o.startswith("[")]
        return br[1:2] if base == "UTMALDG" else br[:1]
    return [o for o in ops if "[" in o][:1]


def strip_desc(s):
    return re.sub(r"desc\[[^\]]*\]", "", s)


def kparams(text_elf):
    """{function: {param byte offset: (ordinal, size)}} from cuobjdump -elf text."""
    out, cur = collections.defaultdict(dict), None
    for line in text_elf.split("\n"):
        m = re.match(r"^\s*\.nv\.info\.(\S+)", line)
        if m:
            cur = m.group(1)
            continue
        m = re.search(r"Ordinal\s*:\s*(0x[0-9a-f]+)\s+Offset\s*:\s*(0x[0-9a-f]+)\s+Size\s*:\s*(0x[0-9a-f]+)", line)
        if m and cur:
            out[cur][int(m.group(2), 16)] = (int(m.group(1), 16), int(m.group(3), 16))
    return out


def ordinal_of(kp, off):
    for o, (ordn, size) in kp.items():
        if o <= off < o + size:
            return ordn, off - o
    return None, None


def fwd_succ(ins_list, i):
    """Successors of instruction i without backward branches (target address <= own address)."""
    return [j for j in ins_list[i].succ if ins_list[j].addr > ins_list[i].addr]


def prewait(ins_list, forward_only=False):
    """Indices reachable from the entry without passing an ACQBULK (forward edges only if forward_only)."""
    seen, stack = set(), [0] if ins_list else []
    while stack:
        i = stack.pop()
        if i in seen:
            continue
        seen.add(i)
        if ins_list[i].op.split(".")[0] in WAIT_OPS:
            continue
        stack.extend(fwd_succ(ins_list, i) if forward_only else ins_list[i].succ)
    return seen


def reaches_wait(ins_list, region):
    """Members of region from which a wait is reachable over forward edges inside region (the waiting thread's
    code before its wait)."""
    rev = collections.defaultdict(list)
    for i in region:
        for j in fwd_succ(ins_list, i):
            rev[j].append(i)
    out, stack = set(), [i for i in region if ins_list[i].op.split(".")[0] in WAIT_OPS]
    while stack:
        i = stack.pop()
        if i in out:
            continue
        out.add(i)
        stack.extend(k for k in rev[i] if k in region)
    return out


def preds_of(ins_list):
    pr = collections.defaultdict(list)
    for i, ins in enumerate(ins_list):
        for j in ins.succ:
            pr[j].append(i)
    return pr


PSPACE = re.compile(r"c\[0x0\]\[0x198\]")  # generic address of the kernel's parameter space (grid_constant operands)


def reaching_defs(ins_list, pr, idx, reg):
    seen, stack, defs = set(), list(pr[idx]), []
    while stack:
        j = stack.pop()
        if j in seen:
            continue
        seen.add(j)
        if reg in dst_of(ins_list[j]):
            defs.append(j)
            continue
        stack.extend(pr[j])
    return defs


def pspace_params(ins_list, pr, idx, regs, kp, depth=0):
    """Parameter ordinals a register addresses as (parameter-space base c[0x0][0x198] + immediate): how nvcc and the
    CuTe DSL form the generic address of a __grid_constant__ tensor map for TMA. Empty when the form is not matched."""
    out = set()
    if depth > 12:
        return out
    for reg in regs:
        for j in reaching_defs(ins_list, pr, idx, reg):
            ins = ins_list[j]
            txt = ", ".join(ins.ops[1:])
            if PSPACE.search(txt):
                out.add(("base", 0))
                continue
            base = ins.op.split(".")[0]
            imm = [int(o, 16) for o in ins.ops[1:] if re.match(r"^0x[0-9a-f]+$", o)]
            sub = regs_in(strip_desc(txt))
            for kind, off in pspace_params(ins_list, pr, j, sub, kp, depth + 1):
                if kind == "base" and base in ("UIADD3", "IADD3", "VIADD", "UIADD3.64") and imm:
                    out.add(("off", off + imm[0]))
                else:
                    out.add((kind, off))
    return out


def roots(ins_list, pr, region, idx, regs, kp, depth=0, memo=None):
    """Roots of a register set used at instruction idx: walk predecessors to the nearest definitions of each register
    (all reaching definitions), recurse into their sources. Returns a set of root tags."""
    if memo is None:
        memo = {}
    out = set()
    for reg in regs:
        key = (idx, reg)
        if key in memo:
            out |= memo[key]
            continue
        memo[key] = set()
        got = set()
        seen, stack, defs = set(), list(pr[idx]), []
        while stack:
            j = stack.pop()
            if j in seen:
                continue
            seen.add(j)
            if reg in dst_of(ins_list[j]):
                defs.append(j)
                continue
            stack.extend(pr[j])
            if not pr[j] and j == 0:
                pass
        if not defs:
            got.add(f"undef:{reg[0]}{reg[1]}")
        for j in defs:
            ins = ins_list[j]
            base = ins.op.split(".")[0]
            srcs_txt = ins.ops[1:] if ins.ops and re.match(r"^!?U?P[T0-9]$", ins.ops[0]) is None else ins.ops[2:]
            txt = ", ".join(srcs_txt)
            for m in CPARAM.finditer(txt):
                off = int(m.group(1), 16)
                if off >= PARAM_BASE:
                    o, sub = ordinal_of(kp, off - PARAM_BASE)
                    got.add(f"param{o}" + (f"+{sub:#x}" if sub else "") if o is not None else f"cparam{off:#x}")
                else:
                    got.add("driver_cbank")
            for m in CBANK.finditer(txt):
                if m.group(1) != "0x0":
                    got.add(f"cbank{m.group(1)}")
            if base in ("S2R", "S2UR", "CS2R"):
                got.add("sreg")
            if base in ("LDG", "LD", "LDS", "LDL", "ATOMG", "ATOM", "ATOMS", "SHFL"):
                if base in ("LDS", "LDL", "ATOMS"):
                    got.add(f"{base.lower()}@{ins.addr:#x}")
                elif base == "SHFL":
                    sub = regs_in(strip_desc(txt))
                    got |= roots(ins_list, pr, region, j, sub, kp, depth + 1, memo) if depth < 40 else {"deep"}
                else:
                    got.add(f"load@{ins.addr:#x}")
                    sub = regs_in(strip_desc(", ".join(addr_operands(ins))))
                    if depth < 40:
                        got |= {"via:" + t for t in roots(ins_list, pr, region, j, sub, kp, depth + 1, memo)}
                continue
            sub = regs_in(strip_desc(txt))
            if sub and depth < 40:
                got |= roots(ins_list, pr, region, j, sub, kp, depth + 1, memo)
            elif depth >= 40:
                got.add("deep")
        memo[key] = got
        out |= got
    return out


# Kernel name substring (demangled or mangled) -> {parameter ordinal: class}. Classes: prep / input / prod / param.
# Built from the host launch code of each component.
P, I, S, R, Q, X = "prep", "input", "scalar", "prior", "prod", "self"
TABLE = {
    # frontg_k(qkv, pos, cs, loc, qo, kc, vc, eps, rope, ql, pt, seqlens, cu_q, pages, page, S, win,
    # part, ko, vt): qkv is the qkv GEMM's output; kc / vc get this call's rows from this kernel.
    "frontg_k": {"role": "front + window gather",
                 "params": {"0": Q, "1": I, "2": P, "3": I, "4": Q, "5": Q, "6": Q, "7": S, "8": S, "9": Q, "10": I,
                            "11": I, "12": I, "13": S, "14": S, "15": S, "16": S, "17": S, "18": Q, "19": Q}},
    "at_front_k": {"role": "full-attention front", "params": {}},
    "front_k": {"role": "front (no window gather)", "params": {"0": Q, "1": I, "2": P, "3": I}},
    "rng3y_k": {"role": "routing chain", "params": {}},
    "rng2_k": {"role": "routing (cat path)", "params": {}},
    "normgather_k": {"role": "pre-expert norm + gather", "params": {}},
    # moe_mixed(W1a, W2a, W1b, W2b, bmap1, bmap2, bmx, meta, gs1, gs2, out1, out2, C, slot_of, cnt2)
    "moe_mixed": {"role": "experts (grouped)", "params": {"0": P, "1": P, "2": P, "3": P, "7": Q, "8": P, "9": P, "13": Q,
                                                      "14": Q},
                  "tma_ack": "producer TMA after the k == 0 wait: pre-wait only through the loop back edge / the "
                             "skip of the k == 0 block, an infeasible path (k starts at 0); consumers / copiers wait",
                  "loop_wait_ack": "the producer warp's claim loop executes GDC_WAIT inside `if (k == 0)` on its first "
                                   "iteration (k starts at 0), so code after that block that the CFG also reaches "
                                   "through the k != 0 skip is post-wait (the claim atomicAdd sits on the k != 0 "
                                   "side of the same loop)"},
    # sl_attn_k(mq, mk, mv, [mql,] SlParams{out, cu_q +0x8, cache_seqlens +0x10, S +0x18, G +0x1c, W, a32, gate +0x28})
    "sl_attn_k": {"role": "sliding attention", "params": {
        "3+0x8": I, "3+0x10": I, "3+0x18": S, "3+0x1c": S, "3+0x20": S, "3+0x28": Q, "3": Q,
        "4+0x8": I, "4+0x10": I, "4+0x18": S, "4+0x1c": S, "4+0x20": S, "4+0x28": Q, "4": Q}},
    "sl_gather_k": {"role": "window gather", "params": {}},
    # nr2 / nr3 / remat: (stream, attn, ffn, out, nst, rps, aa, ba, ab, bb, postw, prew, eps, eps, T, inv, ids_in, ids_out)
    "nr2_k": {"role": "boundary", "params": {}},
    "nr3_k": {"role": "boundary A (remat path)", "params": {}},
    "remat": {"role": "boundary B (remat path)", "params": {
        "0": I, "1": Q, "2": Q, "5": Q, "6": P, "7": P, "8": P, "9": P, "10": P, "11": P, "12": S, "13": S, "14": S,
        "15": R, "16": Q},
              "note": "15 inv: token order from order2_k (side queue, joined before nr3_k; nr3 and every GEMM after it "
                      "complete before remat can launch). 6-11 weights: nvcc hoists these above the wait (const "
                      "__restrict__ loads written after the wait in source)."},
    # FA3 port (xt, src/attention_full.py)
    "prepare_varlen_num_blocks_kernel": {"role": "FA3 scheduler metadata", "params": {
        "0": S, "1": S, "2": S, "3": I, "4": I, "5": I, "6": I, "7": I, "8": I}},
    "FlashAttnFwdCombine": {"role": "FA3 split combine", "params": {
        "0+0xc0": I, "0+0xc8": I, "0+0xd0": R, "0+0xd8": R, "0+0xe8": R, "0+0xf0": S},
        "note": "+0xd0 num_splits_dynamic / +0xd8 varlen_batch_idx: prepare_varlen's outputs (two launches back; FA3 "
                "main triggers its dependents only after its consumers consumed tiles its producer loaded after "
                "waiting on prepare). +0xe8 gate: qkv GEMM output (first kernel of the call). Read before the wait "
                "in source, as upstream FA3."},
    "FlashAttnFwdSm90": {"role": "FA3 main", "params": {
        "0+0x7c0": P, "0+0x880": X, "0+0x928": X, "0+0x9d0": X, "0+0x9d8": X, "0": Q},
        "cbank_ack": "consumer read of a module-scope read-only table (no kernel writes a __device__ symbol: the stamp "
                     "symbols are compiled out)",
        "note": "+0x7c0 ptr_sink (attention sink weights, read by every thread before the role split); +0x880 / +0x928 "
                "/ +0x9d0 / +0x9d8 the in-kernel split-KV partials, LSE partials and their counters (written and read "
                "only by this kernel, ordered by its own atomics). Every other consumer read is data-dependent on the "
                "work tile the producer hands over in shared memory after its wait."},
    "pv_gather_stage_k": {"role": "FA3 K / V staging", "params": {"0": Q, "1": Q, "2": I, "3": I, "4": S, "5": S, "6": S,
                                                                  "7": S, "10": I}},
    "z0_gp8_k": {"role": "K / V gather", "params": {"0": Q, "1": Q, "2": I, "3": I, "4": S, "5": S, "6": S}},
    "z0_gp_k": {"role": "K / V gather", "params": {"0": Q, "1": Q, "2": I, "3": I, "4": S, "5": S, "6": S}},
    # CuTe DSL GEMMs (QuACK GemmSm90.kernel / xg PdlGemmSm90): (tiled_mma, tma_atom_a, mA, tma_atom_b, mB, ...)
    "PdlGemmSm90": {"role": "xg GEMM (unpooler / o projection)", "params": {"1": Q, "2": Q, "3": P, "4": P},
                    "note": "B (weights) TMA of the first stages is issued before the wait by design."},
    "quackgemm": {"role": "QuACK o GEMM", "params": {"1": Q, "2": Q, "3": P, "4": P}},
}


def lookup(fname):
    for k, v in TABLE.items():
        if k in fname:
            return v
    return None


def classify(fname, tags, setname, base="LDG"):
    """Verdict of one access from its root tags. setname: 'W' (the waiting thread, before its wait), 'N' (a warp /
    path that never reaches a wait), 'L' (pre-wait only through a loop back edge)."""
    tab = lookup(fname)
    direct = sorted({t for t in tags if t.startswith("param")})
    via = any(t.startswith("via:param") or t.startswith("via:load") for t in tags)
    if tab is None:
        return "UNCLASSIFIED-KERNEL", direct
    cls = []
    for t in direct:
        num = t[5:]
        cls.append(tab["params"].get(num, tab["params"].get(num.split("+")[0], "?")))
    if not direct:
        if base.startswith("UTMA") or base == "UBLKCP":
            return ("ok-acknowledged" if tab.get("tma_ack") else "UNRESOLVED"), direct
        if any(t.startswith("cbank") for t in tags):
            return ("ok-acknowledged" if tab.get("cbank_ack") else "UNRESOLVED"), direct
        if any(t.startswith("undef") for t in tags) or via:
            return "UNRESOLVED", direct
        return "ok-no-buffer", direct  # thread / CTA ids and shared-window constants only: shared or local memory
    if "prod" in cls:
        if setname == "N" and any(t.startswith("lds@") or t.startswith("via:lds@") for t in tags):
            return "ok-ordered", direct  # address depends on shared memory written after a waiting thread's barrier
        return "HAZARD", direct
    if "?" in cls:
        return "UNRESOLVED", direct
    return "ok", direct


def check_text(label, text, elf, only="", survey=False, kp_override=None, quiet=False):
    funcs = rsi.parse_sass(text)
    kps = kparams(elf or "")
    nbad = nwait = 0
    for fname, ins_list in funcs.items():
        if only and only not in fname:
            continue
        rsi.link(ins_list)
        waits = [ins for ins in ins_list if ins.op.split(".")[0] in WAIT_OPS]
        if not waits:
            continue
        nwait += 1
        region = prewait(ins_list)
        fwd = prewait(ins_list, forward_only=True)
        wset = reaches_wait(ins_list, fwd)
        pr0 = preds_of(ins_list)
        post = set()  # forward-reachable from a wait: code that runs after a wait on some path
        stack = [i for i, x in enumerate(ins_list) if x.op.split(".")[0] in WAIT_OPS]
        while stack:
            i = stack.pop()
            if i in post:
                continue
            post.add(i)
            stack.extend(fwd_succ(ins_list, i))
        loopw = set()  # in a loop (strongly connected with) a wait: the loop's body executes the wait
        for w in [i for i, x in enumerate(ins_list) if x.op.split(".")[0] in WAIT_OPS]:
            fw, st = set(), [w]
            while st:
                i = st.pop()
                if i not in fw:
                    fw.add(i)
                    st.extend(ins_list[i].succ)
            bw, st = set(), [w]
            while st:
                i = st.pop()
                if i not in bw:
                    bw.add(i)
                    st.extend(pr0[i])
            if len(fw & bw) > 1:
                loopw |= fw & bw
        pr = preds_of(ins_list)
        kp = kp_override if kp_override is not None else kps.get(fname, {})
        dem = fname if kp_override is not None else (
            subprocess.run(["cu++filt", fname], capture_output=True, text=True).stdout.strip() or fname)
        tab = lookup(fname + " " + dem)
        if not quiet:
            print(f"PDL_HOIST_FN {label} {dem[:150]}: waits at {[hex(w.addr) for w in waits]}, before the wait "
              f"{len(wset)} / never waiting {len(fwd - wset)} / loop-only {len(region - fwd)} of {len(ins_list)}"
              f"{'' if tab is None else ' [' + tab['role'] + ']'}", flush=True)
        for i in sorted(region):
            ins = ins_list[i]
            base = ins.op.split(".")[0]
            kind = "load" if base in LOAD_OPS else "prefetch" if base in PF_OPS else "store" if base in STORE_OPS else None
            if kind is None:
                continue
            setname = "W" if i in wset else "N" if i in fwd else "L"
            regs = regs_in(strip_desc(", ".join(addr_operands(ins))))
            tags = roots(ins_list, pr, region, i, regs, kp)
            if base.startswith("UTMA") or base == "UBLKCP":
                for kind_, off in pspace_params(ins_list, pr, i, regs, kp):
                    if kind_ == "off":
                        o, sub = ordinal_of(kp, off)
                        tags.add(f"param{o}" if o is not None else f"pspace{off:#x}")
            verdict, params = classify(fname + " " + dem, tags, setname, base)
            if (setname == "N" and (i in post or i in loopw) and tab is not None and tab.get("loop_wait_ack")
                    and verdict in ("HAZARD", "UNRESOLVED")):
                verdict = "ok-first-iteration-wait"  # see the kernel's loop_wait_ack in TABLE
            enforce = kind == "load" and setname in ("W", "N") and not survey
            bad = enforce and verdict in ("HAZARD", "UNCLASSIFIED-KERNEL", "UNRESOLVED")
            nbad += bad
            if not quiet:
                print(f"  {'FAIL' if bad else 'ok  '} {setname} {kind:8s} {ins.addr:#07x} {ins.text[:70]:70s} "
                  f"roots {sorted(tags)[:6]} -> {verdict}{'' if enforce else ' (reported)'}")
    return nbad, nwait, len(funcs)


DSL_CODE = r"""
import importlib.util, os, sys, torch
sys.path.insert(0, os.environ.get("MKC_PROBLEM_DIR", "/okc/problems/muse_prefill_layer"))
spec = importlib.util.spec_from_file_location("pdl_hoist_dsl", sys.argv[1])
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
dense = {"o": torch.zeros(8192, 1024, dtype=torch.bfloat16, device="cuda"),
         "unpooler": torch.zeros(8192, 4096, dtype=torch.bfloat16, device="cuda")}
fn = getattr(m, sys.argv[2], None)
print("DSL_COMPILE", sys.argv[2], None if fn is None else fn(dense))
"""
DSL_FNS = ("_quack_compile", "_unp_compile", "_og_compile")


def dsl_cubins(bundle):
    """Compile the bundle's CuTe DSL GEMMs (QuACK o, xg unpooler, xg o) in child processes with CUTE_DSL_KEEP=cubin and
    the QuACK cache off, one dump directory per kernel; returns [(label, cubin path)]."""
    import tempfile
    out = []
    for fn in DSL_FNS:
        d = tempfile.mkdtemp(prefix="pdl_hoist_dsl_")
        env = dict(os.environ, CUTE_DSL_KEEP="cubin", CUTE_DSL_DUMP_DIR=d, QUACK_CACHE_ENABLED="0")
        r = subprocess.run([sys.executable, "-c", DSL_CODE, bundle, fn], capture_output=True, text=True, env=env)
        line = [x for x in r.stdout.split("\n") if x.startswith("DSL_COMPILE")]
        print(f"PDL_HOIST_DSL {fn}: rc {r.returncode} {line[-1] if line else r.stderr.strip()[-300:]}", flush=True)
        for root, _, files in os.walk(d):
            out += [(f"{fn}:{f[:60]}", os.path.join(root, f)) for f in sorted(files) if f.endswith(".cubin")]
    return out


def elf_of(path):
    p = subprocess.run(["cuobjdump", "-elf", path], capture_output=True, text=True)
    return p.stdout if p.returncode == 0 else ""


SELFTEST = {
    # name: (expected failures, SASS). TABLE 'selfprod': param0 prep, param1 prod; 'selfloop' adds loop_wait_ack.
    "selfprod_load_after_wait": (0, """
        /*0000*/  ACQBULK ;
        /*0010*/  LDC.64 R2, c[0x0][0x218] ;
        /*0020*/  LDG.E R4, desc[UR4][R2.64] ;
        /*0030*/  EXIT ;"""),
    "selfprod_prod_load_hoisted": (1, """
        /*0000*/  LDC.64 R2, c[0x0][0x218] ;
        /*0010*/  LDG.E.CONSTANT R4, desc[UR4][R2.64] ;
        /*0020*/  ACQBULK ;
        /*0030*/  EXIT ;"""),
    "selfprod_weight_load_hoisted": (0, """
        /*0000*/  ULDC.64 UR6, c[0x0][0x210] ;
        /*0010*/  LEA R2, P1, R3, UR6, 0x2 ;
        /*0020*/  LEA.HI.X R3, R3, UR7, R4, 0x2, P1 ;
        /*0030*/  LDG.E.CONSTANT R6, desc[UR4][R2.64] ;
        /*0040*/  ACQBULK ;
        /*0050*/  EXIT ;"""),
    "selfprod_prod_on_skip_path": (1, """
        /*0000*/  @P0 BRA 0x30 ;
        /*0010*/  ACQBULK ;
        /*0020*/  BRA 0x40 ;
        /*0030*/  NOP ;
        /*0040*/  LDC.64 R2, c[0x0][0x218] ;
        /*0050*/  LDG.E R4, desc[UR4][R2.64] ;
        /*0060*/  EXIT ;"""),
    "selfprod_tma_prod_hoisted": (1, """
        /*0000*/  UMOV UR8, 0x400 ;
        /*0010*/  ULDC.64 UR4, c[0x0][0x218] ;
        /*0020*/  UTMALDG.2D [UR8], [UR4], desc[UR10] ;
        /*0030*/  ACQBULK ;
        /*0040*/  EXIT ;"""),
    "selfprod_tma_pspace_weights": (0, """
        /*0000*/  ULDC.64 UR16, c[0x0][0x198] ;
        /*0010*/  UIADD3 UR16, UP0, UR16, 0x0, URZ ;
        /*0020*/  UTMALDG.2D [UR20], [UR16], desc[URZ] ;
        /*0030*/  ACQBULK ;
        /*0040*/  EXIT ;"""),
    "selfprod_tma_pspace_prod": (1, """
        /*0000*/  ULDC.64 UR16, c[0x0][0x198] ;
        /*0010*/  UIADD3 UR16, UP0, UR16, 0x8, URZ ;
        /*0020*/  UTMALDG.2D [UR20], [UR16], desc[URZ] ;
        /*0030*/  ACQBULK ;
        /*0040*/  EXIT ;"""),
    "selfprod_loop_first_iter_wait": (1, """
        /*0000*/  MOV R0, RZ ;
        /*0010*/  ISETP.NE.AND P0, PT, R0, RZ, PT ;
        /*0020*/  @P0 BRA 0x40 ;
        /*0030*/  ACQBULK ;
        /*0040*/  LDC.64 R2, c[0x0][0x218] ;
        /*0050*/  @P0 LDG.E R4, desc[UR4][R2.64] ;
        /*0060*/  IADD3 R0, R0, 0x1, RZ ;
        /*0070*/  BRA 0x10 ;"""),
    "selfloop_first_iter_wait_acked": (0, """
        /*0000*/  MOV R0, RZ ;
        /*0010*/  ISETP.NE.AND P0, PT, R0, RZ, PT ;
        /*0020*/  @P0 BRA 0x40 ;
        /*0030*/  ACQBULK ;
        /*0040*/  LDC.64 R2, c[0x0][0x218] ;
        /*0050*/  @P0 LDG.E R4, desc[UR4][R2.64] ;
        /*0060*/  IADD3 R0, R0, 0x1, RZ ;
        /*0070*/  BRA 0x10 ;"""),
    "selfprod_consumer_reads_prod_after_smem": (0, """
        /*0000*/  @P1 BRA 0x50 ;
        /*0010*/  ACQBULK ;
        /*0020*/  EXIT ;
        /*0050*/  LDS R6, [R1] ;
        /*0060*/  LDC.64 R2, c[0x0][0x218] ;
        /*0070*/  IMAD.WIDE R2, R6, 0x4, R2 ;
        /*0080*/  LDG.E R4, desc[UR4][R2.64] ;
        /*0090*/  EXIT ;"""),
    "selfprod_consumer_reads_prod_blind": (1, """
        /*0000*/  @P1 BRA 0x50 ;
        /*0010*/  ACQBULK ;
        /*0020*/  EXIT ;
        /*0050*/  LDC.64 R2, c[0x0][0x218] ;
        /*0060*/  LDG.E.CONSTANT R4, desc[UR4][R2.64] ;
        /*0070*/  EXIT ;"""),
}


def selftest():
    TABLE["selfprod"] = {"role": "selftest", "params": {"0": "prep", "1": "prod"}}
    TABLE["selfloop"] = {"role": "selftest", "params": {"0": "prep", "1": "prod"}, "loop_wait_ack": "selftest"}
    kp = {0x0: (0, 8), 0x8: (1, 8)}
    bad = 0
    for name, (want, body) in SELFTEST.items():
        got, _, _ = check_text("selftest", "\t\tFunction : " + name + "\n" + body, "", kp_override=kp, quiet=True)
        ok = got == want
        bad += not ok
        print(f"  SELFTEST {name}: failures {got} want {want} {'ok' if ok else 'WRONG'}")
    del TABLE["selfprod"], TABLE["selfloop"]
    return bad


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("bundle", nargs="?")
    ap.add_argument("--so", nargs="+", default=[])
    ap.add_argument("--sass", nargs="+", default=[])
    ap.add_argument("--only", default="")
    ap.add_argument("--survey", action="store_true")
    ap.add_argument("--no-dsl", action="store_true", help="skip compiling and checking the CuTe DSL GEMMs")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    setup_bad = selftest()
    if a.selftest:
        print("PDL_HOIST_" + ("FAIL selftest" if setup_bad else "PASS selftest"))
        sys.exit(1 if setup_bad else 0)
    items = []
    if a.bundle:
        objs, names = rsi.bundle_objects(a.bundle)
        print(f"PDL_HOIST bundle {a.bundle}: {len(names)} load_inline names, {len(objs)} objects found", flush=True)
        for n in sorted(names - set(objs)):
            print(f"  MISSING build for {n}")
            setup_bad += 1
        items += [(k, v) for k, v in sorted(objs.items())]
        if not a.no_dsl:
            dsl = dsl_cubins(a.bundle)
            if not dsl:
                print("  MISSING CuTe DSL cubins (QuACK / xg kernels not dumped)")
                setup_bad += 1
            items += dsl
    items += [(os.path.basename(p), p) for p in a.so]
    tb = tw = 0
    for label, path in items:
        text = rsi.sass_of(path)
        if text is None:
            print(f"PDL_HOIST_OBJ {label}: UNREADABLE")
            setup_bad += 1
            continue
        b, w, nf = check_text(label, text, elf_of(path), a.only, a.survey)
        print(f"PDL_HOIST_OBJ {label}: functions {nf} with a PDL wait {w} failures {b}", flush=True)
        tb += b
        tw += w
    for p in a.sass:
        b, w, nf = check_text(os.path.basename(p), open(p).read(), "", a.only, a.survey)
        tb += b
        tw += w
    verdict = "FAIL" if tb or setup_bad else "PASS"
    print(f"PDL_HOIST_{verdict} objects {len(items)} kernels with a wait {tw} failures {tb} setup_errors {setup_bad}"
          f"{' (survey mode: classification not enforced)' if a.survey else ''}")
    sys.exit(1 if verdict == "FAIL" else 0)


if __name__ == "__main__":
    main()
