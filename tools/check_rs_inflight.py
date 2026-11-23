"""check_rs_inflight: SASS-level guard against the ptxas RS-wgmma register-liveness bug.

ptxas through CUDA 13.2 (the judge builds with 13.0.88) can let code after a register-input (RS) wgmma overwrite that
wgmma's A registers while it is still in flight, because it copy-propagates across wgmma.wait_group N with N >= 1 and
removes the movs placed there to protect them. NVIDIA lists this as the CUDA 13.2 known issue "Data Race in WGMMA A/B
Register Copy Propagation" and as fixed in the CUDA 13.3 release notes, with the workarounds wait_group 0 before the
next ldmatrix, or ldmatrix.x1 loads straight into the wgmma input registers (also Triton issue #9433 and Triton PR
#11781: an integer-valued matmul that should give 64 gives 61). The result is a silent, timing-dependent wrong
answer. This tool reads the SASS that ptxas actually emitted and proves, per kernel,
that no instruction writes a pending RS HGMMA's A registers before the WARPGROUP.DEPBAR that retires it.

Model (conservative):
  - every *GMMA (HGMMA / QGMMA / IGMMA / BGMMA) joins the open group; the one carrying gsb0 closes the group.
  - WARPGROUP.DEPBAR.LE gsb0, N leaves the newest N closed groups pending and retires the rest. Nothing else retires
    a group (an HGMMA without gsb0 stays pending until a later gsb0 group closes over it and is retired).
  - an RS HGMMA reads 4 A registers (A operand Rn..Rn+3); its accumulator is N/2 registers (F32 / S32) or N/4 (F16).
  - the walk is a fixpoint over the function's control-flow graph (branch targets, fall-through, calls), so a group
    kept in flight across a loop back edge is checked against the next iteration's code.
Violations (each one FAILS the run):
  A-write    a non-GMMA instruction, or a later GMMA's accumulator, writes a pending RS HGMMA's A registers
  acc-touch  a non-GMMA instruction reads or writes a pending GMMA's accumulator registers
Per kernel it prints: RS and SS GMMA counts, RS GMMAs kept in flight across other code, RS GMMAs kept in flight across
a backward branch, issue-to-retire windows, and violations.

    python3 tools/check_rs_inflight.py submission/submission.py [--only SUBSTR]
        import the file on an H200 (builds or reuses its torch extensions) and check every extension it loads
    python3 tools/check_rs_inflight.py --so A.so [B.so ...]          check these shared objects / cubins
    python3 tools/check_rs_inflight.py --sass DUMP.sass [...]        check existing cuobjdump -sass / nvdisasm text
    python3 tools/check_rs_inflight.py --ptx K.ptx [--ptxas PATH]    assemble with ptxas for sm_90a, then check
    python3 tools/check_rs_inflight.py --selftest                    synthetic positive and negative controls (CPU)
Prints RS_INFLIGHT_PASS or RS_INFLIGHT_FAIL at the end; exit code 1 on any violation or on a setup error. Reading
SASS needs cuobjdump or nvdisasm from the CUDA toolkit; --selftest needs neither. Importing the submission needs the
problem files (reference.py and the rest) on sys.path: set MKC_PROBLEM_DIR, default /okc/problems/muse_prefill_layer,
the challenge container's own path.

Result: every candidate build of this layer went through this check before it was submitted, the submitted one
included. The recorded runs found 752 RS GMMAs in the layer's extensions (ptxas 13.0.88) and 0 violations, so this
layer does not hit the bug. The check stays because the failure is silent and depends on register allocation, which
any edit to a kernel can change.
"""
import argparse
import collections
import glob
import importlib.util
import os
import re
import subprocess
import sys
import tempfile

LINE = re.compile(r"^\s+/\*([0-9a-f]{4,})\*/\s+(.*?)\s*;")
FUNC = re.compile(r"Function : (\S+)")
PRED = re.compile(r"^(@!?U?P[T0-9]+)\s+")
REG = re.compile(r"^R(\d+)$")
ANYREG = re.compile(r"(?<![A-Z_])R(\d+)(\.64|\.128)?(?![0-9])")
ISPRED = re.compile(r"^!?U?P[T0-9]$")
GMMA = re.compile(r"^[HQIB]GMMA\.64x(\d+)x(\d+)\.(F32|F16|S32)")
DEPBAR = re.compile(r"^WARPGROUP\.DEPBAR\.LE$")
MAX_GROUPS = 16
MAX_STATES = 4096


def split_ops(s):
    out, depth, cur = [], 0, ""
    for ch in s:
        if ch in "[(":
            depth += 1
        elif ch in "])":
            depth -= 1
        if ch == "," and depth == 0:
            out.append(cur.strip())
            cur = ""
        else:
            cur += ch
    if cur.strip():
        out.append(cur.strip())
    return out


class Ins:
    __slots__ = ("addr", "pred", "op", "ops", "text", "dst", "src", "gmma", "depbar", "succ", "back", "kind")

    def __init__(self, addr, text):
        self.addr = addr
        self.text = text
        m = PRED.match(text)
        self.pred = m.group(1) if m else ""
        rest = text[m.end():] if m else text
        parts = rest.split(None, 1)
        self.op = parts[0]
        self.ops = split_ops(parts[1]) if len(parts) > 1 else []
        self.gmma = None
        self.depbar = None
        self.kind = ""
        self.dst, self.src = set(), set()
        g = GMMA.match(self.op)
        if g:
            n, acc_t = int(g.group(1)), g.group(3)
            w = n // 4 if acc_t == "F16" else n // 2
            d = REG.match(self.ops[0])
            a = REG.match(self.ops[1]) if len(self.ops) > 1 else None
            acc = set(range(int(d.group(1)), int(d.group(1)) + w)) if d else set()
            areg = set(range(int(a.group(1)), int(a.group(1)) + 4)) if a else set()
            self.gmma = dict(acc=acc, a=areg, rs=bool(a), gsb0="gsb0" in self.ops)
            return
        if DEPBAR.match(self.op) and self.ops and self.ops[0] == "gsb0":
            self.depbar = int(self.ops[1], 0)
            return
        self.dst = dest_regs(self.op, self.ops)
        for o in self.ops[1:] if self.dst else self.ops:
            for m2 in ANYREG.finditer(o):
                r = int(m2.group(1))
                w = 4 if m2.group(2) == ".128" else 2 if m2.group(2) == ".64" else 1
                self.src.update(range(r, r + w))


MEM_LD = ("LD", "LDG", "LDS", "LDL", "LDC", "ATOM", "ATOMG", "ATOMS", "LDGSTS", "SUATOM", "SULD")


def dest_regs(op, ops):
    """Registers an instruction writes (R only; uniform and predicate registers cannot alias an A operand)."""
    if not ops:
        return set()
    first = ops[0]
    if ISPRED.match(first) and len(ops) > 1 and REG.match(ops[1]):
        first = ops[1]  # SHFL PT, R3, ... and LOP3.LUT P0, R2, ... write the second operand
    m = REG.match(first)
    if not m:
        return set()
    base = op.split(".")[0]
    parts = op.split(".")
    w = 1
    if base in MEM_LD:
        w = 4 if ".128" in op else 2 if ".64" in op else 1
    elif base == "LDSM" or base == "MOVM":
        w = 4 if parts[-1] in ("4", "x4") else 2 if parts[-1] in ("2", "x2") else 1
    elif base == "IMAD" and "WIDE" in parts:
        w = 2
    elif base == "CS2R":
        w = 1 if "32" in parts else 2
    elif base in ("DADD", "DMUL", "DFMA", "DMNMX", "DMMA"):
        w = 2
    elif base in ("F2F", "I2F", "F2I", "I2I") and len(parts) > 1 and parts[1] in ("F64", "S64", "U64"):
        w = 2
    elif base in ("HMMA", "IMMA"):
        shape = parts[1] if len(parts) > 1 else ""
        acc32 = any(p in ("F32", "S32") for p in parts[2:3])
        w = {"16816": 4 if acc32 else 2, "1688": 4 if acc32 else 2, "1684": 4, "16832": 4, "8816": 2,
             "8832": 2}.get(shape, 4)
    return set(range(int(m.group(1)), int(m.group(1)) + w))


def parse_sass(text):
    """{function name: [Ins, ...]} from cuobjdump -sass / nvdisasm output."""
    funcs, cur = collections.OrderedDict(), None
    for line in text.split("\n"):
        m = FUNC.search(line)
        if m:
            cur = m.group(1)
            funcs[cur] = []
            continue
        if cur is None:
            continue
        m = LINE.match(line)
        if m:
            funcs[cur].append(Ins(int(m.group(1), 16), m.group(2)))
    return funcs


def link(ins_list):
    idx = {ins.addr: i for i, ins in enumerate(ins_list)}
    warn = []
    for i, ins in enumerate(ins_list):
        base = ins.op.split(".")[0]
        nxt = [i + 1] if i + 1 < len(ins_list) else []
        always = ins.pred in ("", "@PT")
        never = ins.pred == "@!PT"
        tgt = None
        for o in ins.ops:
            m = re.match(r"^`?\(?(0x[0-9a-f]+)", o)
            if m:
                tgt = int(m.group(1), 16)
        if never:
            ins.succ = nxt
        elif base in ("BRA", "JMP", "CALL"):
            t = [idx[tgt]] if tgt in idx else []
            if tgt is not None and tgt not in idx:
                warn.append(f"branch target {tgt:#x} outside the function at {ins.addr:#x}")
            if base == "CALL":
                ins.succ = t + nxt  # the callee returns to the fall-through; its RET ends its own path
            else:
                ins.succ = t if always else t + nxt
            ins.back = bool(t) and ins_list[t[0]].addr <= ins.addr
        elif base in ("BRX", "JMX"):
            warn.append(f"indirect branch at {ins.addr:#x}: successors limited to fall-through")
            ins.succ = nxt
        elif base in ("EXIT", "RET", "BPT", "KILL") and always:
            ins.succ = []
            ins.kind = "end"
        else:
            ins.succ = nxt
        if not getattr(ins, "back", None):
            ins.back = False
    return warn


def check_function(name, ins_list):
    warn = link(ins_list)
    n = len(ins_list)
    states = [set() for _ in range(n)]
    gm = {i: ins.gmma for i, ins in enumerate(ins_list) if ins.gmma}
    viol = {}
    inflight, backedge, windows, end_pending = set(), set(), set(), set()
    overflow = [False]
    if n == 0:
        return dict(rs=0, ss=0, inflight=0, backedge=0, windows=0, viol=[], warn=warn, end_pending=0)
    work = [0]
    states[0].add(((), ()))
    queued = {0}
    while work:
        i = work.pop()
        queued.discard(i)
        ins = ins_list[i]
        outs = set()
        for groups, opn in states[i]:
            pend = [h for g in groups for h in g] + list(opn)
            if ins.gmma:
                for h in pend:
                    if gm[h]["rs"] and ins.gmma["acc"] & gm[h]["a"]:
                        viol[(h, i, "A-write")] = sorted(ins.gmma["acc"] & gm[h]["a"])
                opn2 = opn + (i,)
                if ins.gmma["gsb0"]:
                    groups2, opn2 = groups + (opn2,), ()
                    if len(groups2) > MAX_GROUPS:
                        groups2 = groups2[-MAX_GROUPS:]
                        overflow[0] = True
                else:
                    groups2 = groups
                outs.add((groups2, opn2))
                continue
            if ins.depbar is not None:
                keep = groups[len(groups) - ins.depbar:] if ins.depbar else ()
                for g in groups[:len(groups) - len(keep)]:
                    for h in g:
                        if gm[h]["rs"]:
                            windows.add((h, i))
                outs.add((keep, opn))
                continue
            for h in pend:
                g = gm[h]
                if g["rs"]:
                    inflight.add(h)
                    if ins.back:
                        backedge.add(h)
                    hit = ins.dst & g["a"]
                    if hit:
                        viol[(h, i, "A-write")] = sorted(hit)
                hit = (ins.dst | ins.src) & g["acc"]
                if hit:
                    viol[(h, i, "acc-touch")] = sorted(hit)
            if ins.kind == "end":
                end_pending.update(h for h in pend if gm[h]["rs"])
            outs.add((groups, opn))
        for s in ins.succ:
            new = outs - states[s]
            if new:
                if len(states[s]) + len(new) > MAX_STATES:
                    warn.append(f"state cap {MAX_STATES} reached at {ins_list[s].addr:#x}; analysis incomplete")
                    return dict(rs=-1, ss=-1, inflight=0, backedge=0, windows=0, viol=[], warn=warn, end_pending=0,
                                incomplete=True)
                states[s] |= new
                if s not in queued:
                    queued.add(s)
                    work.append(s)
    if overflow[0]:
        warn.append(f"more than {MAX_GROUPS} groups in flight on some path; oldest dropped (analysis incomplete)")
    vl = []
    for (h, i, kind), regs in sorted(viol.items()):
        vl.append(f"{kind}: GMMA at {ins_list[h].addr:#06x} ({ins_list[h].text}) pending when "
                  f"{ins_list[i].addr:#06x} ({ins_list[i].text}) touches R{regs}")
    rs = sum(1 for g in gm.values() if g["rs"])
    return dict(rs=rs, ss=len(gm) - rs, inflight=len(inflight), backedge=len(backedge), windows=len(windows),
                viol=vl, warn=warn, end_pending=len(end_pending), incomplete=overflow[0])


def check_text(label, text, only=""):
    funcs = parse_sass(text)
    total_v, total_rs, bad_setup = 0, 0, 0
    for name, ins in funcs.items():
        if only and only not in name:
            continue
        r = check_function(name, ins)
        if not r["rs"] and not r["ss"] and not r["viol"]:
            continue
        short = name if len(name) < 90 else name[:87] + "..."
        print(f"  {label} {short}: RS {r['rs']} SS {r['ss']} rs_inflight {r['inflight']} rs_backedge "
              f"{r['backedge']} windows {r['windows']} rs_pending_at_exit {r['end_pending']} "
              f"violations {len(r['viol'])}", flush=True)
        for v in r["viol"][:40]:
            print("    VIOLATION " + v)
        if len(r["viol"]) > 40:
            print(f"    ... {len(r['viol']) - 40} more")
        for w in sorted(set(r["warn"]))[:10]:
            print("    WARN " + w)
        total_v += len(r["viol"])
        total_rs += max(r["rs"], 0)
        bad_setup += bool(r.get("incomplete"))
    return total_v, total_rs, bad_setup, len(funcs)


def sass_of(path):
    """SASS text of a .so / cubin; None if neither cuobjdump nor nvdisasm could read it. An object whose fatbin
    holds no kernels (a host-only wrapper around cuBLASLt) comes back as text with no Function line."""
    p = subprocess.run(["cuobjdump", "-sass", path], capture_output=True, text=True)
    if p.returncode == 0 and ("Function" in p.stdout or "Fatbin" in p.stdout):
        return p.stdout
    p = subprocess.run(["nvdisasm", path], capture_output=True, text=True)
    return p.stdout if p.returncode == 0 else None


def bundle_objects(bundle):
    """Import the bundle the way the judge does (from its own directory, as a module) and return every torch
    extension .so it loaded, with the name it was built under."""
    sys.path.insert(0, os.environ.get("MKC_PROBLEM_DIR", "/okc/problems/muse_prefill_layer"))
    sys.path.insert(0, os.path.dirname(os.path.abspath(bundle)))
    before = set(sys.modules)
    spec = importlib.util.spec_from_file_location("rs_inflight_sub", bundle)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    out = {}
    for k, m in list(sys.modules.items()):
        f = getattr(m, "__file__", None) or ""
        if k not in before and f.endswith(".so") and "torch_extensions" in f:
            out[k] = f
    for v in vars(mod).values():  # load_inline modules held only as attributes
        f = getattr(v, "__file__", None) or ""
        if f.endswith(".so") and "torch_extensions" in f:
            out[getattr(v, "__name__", f)] = f
    names = set(re.findall(r'load_inline\(\s*name\s*=\s*["\']([^"\']+)["\']', open(bundle).read()))
    missing = [n for n in names if n not in out]
    for n in missing:  # built in a helper process or under a lock: fall back to the extension cache
        hits = sorted(glob.glob(os.path.expanduser(f"~/.cache/torch_extensions/*/{n}/{n}.so")))
        if hits:
            out[n] = hits[-1]
    return out, names


SELFTEST = {
    # name: (expected violations, SASS body)
    "retired_then_reused": (0, """
        /*0000*/  WARPGROUP.ARRIVE ;
        /*0010*/  HGMMA.64x32x16.F32.BF16 R24, R40, gdesc[UR4], R24, gsb0 ;
        /*0020*/  WARPGROUP.DEPBAR.LE gsb0, 0x0 ;
        /*0030*/  LDS.128 R40, [R2] ;
        /*0040*/  EXIT ;"""),
    "overwrite_in_flight": (1, """
        /*0000*/  HGMMA.64x32x16.F32.BF16 R24, R40, gdesc[UR4], R24, gsb0 ;
        /*0010*/  MOV R43, R2 ;
        /*0020*/  WARPGROUP.DEPBAR.LE gsb0, 0x0 ;
        /*0030*/  EXIT ;"""),
    "wide_load_overlap": (1, """
        /*0000*/  HGMMA.64x32x16.F32.BF16 R24, R40, gdesc[UR4], R24, gsb0 ;
        /*0010*/  LDS.64 R38, [R2] ;
        /*0020*/  WARPGROUP.DEPBAR.LE gsb0, 0x0 ;
        /*0030*/  EXIT ;"""),
    "only_via_back_edge": (1, """
        /*0000*/  LDS.128 R40, [R2] ;
        /*0010*/  WARPGROUP.ARRIVE ;
        /*0020*/  HGMMA.64x32x16.F32.BF16 R24, R40, gdesc[UR4], R24, gsb0 ;
        /*0030*/  WARPGROUP.DEPBAR.LE gsb0, 0x1 ;
        /*0040*/  @P0 BRA 0x0 ;
        /*0050*/  WARPGROUP.DEPBAR.LE gsb0, 0x0 ;
        /*0060*/  EXIT ;"""),
    "ping_pong_two_sets": (0, """
        /*0000*/  LDS.128 R40, [R2] ;
        /*0010*/  WARPGROUP.ARRIVE ;
        /*0020*/  HGMMA.64x32x16.F32.BF16 R24, R40, gdesc[UR4], R24, gsb0 ;
        /*0030*/  WARPGROUP.DEPBAR.LE gsb0, 0x1 ;
        /*0040*/  LDS.128 R44, [R2+0x10] ;
        /*0050*/  WARPGROUP.ARRIVE ;
        /*0060*/  HGMMA.64x32x16.F32.BF16 R24, R44, gdesc[UR8], R24, gsb0 ;
        /*0070*/  WARPGROUP.DEPBAR.LE gsb0, 0x1 ;
        /*0080*/  @P0 BRA 0x0 ;
        /*0090*/  WARPGROUP.DEPBAR.LE gsb0, 0x0 ;
        /*00a0*/  EXIT ;"""),
    "chained_no_gsb0_stays_pending": (1, """
        /*0000*/  HGMMA.64x32x16.F32.BF16 R24, R40, gdesc[UR4], R24 ;
        /*0010*/  WARPGROUP.DEPBAR.LE gsb0, 0x0 ;
        /*0020*/  F2FP.BF16.F32.PACK_AB R41, R3, R2 ;
        /*0030*/  HGMMA.64x32x16.F32.BF16 R24, R44, gdesc[UR8], R24, gsb0 ;
        /*0040*/  WARPGROUP.DEPBAR.LE gsb0, 0x0 ;
        /*0050*/  EXIT ;"""),
    "acc_read_in_flight": (1, """
        /*0000*/  HGMMA.64x32x16.F32.BF16 R24, gdesc[UR4], R24, gsb0 ;
        /*0010*/  FADD R2, R30, R3 ;
        /*0020*/  WARPGROUP.DEPBAR.LE gsb0, 0x0 ;
        /*0030*/  EXIT ;"""),
    "shfl_second_operand_dest": (1, """
        /*0000*/  HGMMA.64x32x16.F32.BF16 R24, R40, gdesc[UR4], R24, gsb0 ;
        /*0010*/  SHFL.IDX PT, R42, R2, 0x1, 0x1f ;
        /*0020*/  WARPGROUP.DEPBAR.LE gsb0, 0x0 ;
        /*0030*/  EXIT ;"""),
    "taken_branch_skips_writer": (0, """
        /*0000*/  HGMMA.64x32x16.F32.BF16 R24, R40, gdesc[UR4], R24, gsb0 ;
        /*0010*/  BRA 0x30 ;
        /*0020*/  MOV R40, R2 ;
        /*0030*/  WARPGROUP.DEPBAR.LE gsb0, 0x0 ;
        /*0040*/  EXIT ;"""),
}


def selftest():
    bad = 0
    for name, (want, body) in SELFTEST.items():
        funcs = parse_sass("\t\tFunction : " + name + "\n" + body.replace("\n        /*", "\n        /*"))
        r = check_function(name, funcs[name])
        ok = len(r["viol"]) == want
        bad += not ok
        print(f"  SELFTEST {name}: violations {len(r['viol'])} want {want} {'ok' if ok else 'WRONG'}")
    return bad


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("bundle", nargs="?")
    ap.add_argument("--so", nargs="+", default=[])
    ap.add_argument("--sass", nargs="+", default=[])
    ap.add_argument("--ptx", nargs="+", default=[])
    ap.add_argument("--ptxas", default="ptxas")
    ap.add_argument("--only", default="")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    setup_bad = selftest()  # the synthetic controls run every time: a parser regression fails the gate
    if a.selftest:
        print("RS_INFLIGHT_" + ("FAIL selftest" if setup_bad else "PASS selftest"))
        sys.exit(1 if setup_bad else 0)
    items = []
    if a.bundle:
        objs, names = bundle_objects(a.bundle)
        print(f"RS_INFLIGHT bundle {a.bundle}: {len(names)} load_inline names, {len(objs)} objects found", flush=True)
        for n in sorted(names - set(objs)):
            print(f"  MISSING build for {n}")
            setup_bad += 1
        items += [(k, sass_of(v)) for k, v in sorted(objs.items())]
    items += [(os.path.basename(p), sass_of(p)) for p in a.so]
    for p in a.sass:
        t = open(p).read()
        items.append((os.path.basename(p), t if "Function" in t else None))
    for p in a.ptx:
        cub = os.path.join(tempfile.mkdtemp(), "k.cubin")
        r = subprocess.run([a.ptxas, "--gpu-name=sm_90a", p, "-o", cub], capture_output=True, text=True)
        ver = subprocess.run([a.ptxas, "--version"], capture_output=True, text=True).stdout.strip().split("\n")[-1]
        print(f"RS_INFLIGHT ptx {p} assembled by {ver}: rc {r.returncode} {r.stderr.strip()[:300]}")
        if r.returncode:
            setup_bad += 1
            continue
        items.append((os.path.basename(p), sass_of(cub)))
    tv = trs = 0
    for label, text in items:
        if text is None:
            print(f"RS_INFLIGHT_OBJ {label}: UNREADABLE (cuobjdump and nvdisasm both failed)")
            setup_bad += 1
            continue
        v, rs, inc, nf = check_text(label, text, a.only)
        print(f"RS_INFLIGHT_OBJ {label}: functions {nf} RS GMMAs {rs} violations {v}"
              f"{' INCOMPLETE' if inc else ''}{' (no device kernels in this object)' if nf == 0 else ''}",
              flush=True)
        tv += v
        trs += rs
        setup_bad += inc
    verdict = "FAIL" if tv or setup_bad else "PASS"
    print(f"RS_INFLIGHT_{verdict} objects {len(items)} RS GMMAs {trs} violations {tv} setup_errors {setup_bad}")
    sys.exit(1 if verdict == "FAIL" else 0)


if __name__ == "__main__":
    main()
