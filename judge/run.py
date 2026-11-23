"""Time a submission against SGLang's baseline the way the official judge does, on one H200 (an H100 SXM should work but
is untested).

    python3 judge/run.py submission/submission.py
    python3 judge/run.py A.py B.py --control          # A/B: files alternated in one process, deltas in us vs A
    python3 judge/run.py A.py --cases 0,1 --reps 20   # some shapes, more repetitions

It needs the judge's software stack and the problem files, as docs/SETUP.md describes; point --problem-dir (or
MKC_PROBLEM_DIR) at them. See docs/JUDGE.md for what it replicates, what it does not, and how close it lands to
official runs.

The steps, in order:
1. Case seeds: drawn fresh each run (as the official judge draws them), or replayed from an official record
   (judge/seeds.py reads them).
2. Router: reference.weights_fields picks the router seed that gives each load its target expert work. The routes
   target per load is backed out of TASK.md's floors; the distinct-experts target (not published) is the value that
   fits official runs best (14.5, 14, 16 for turn 1, turn 4, 16K prompt).
   The challenge's files and records call the 16K prompt load "first turn"; that stays the data key.
3. prepare() for SGLang and every submission, each on its own copy of the weights.
4. The five TASK.md tests, eager and as a replayed graph, on every submission.
5. Per shape: one CUDA graph per side, on its own input buffers. Repetition r makes fresh inputs (seed + 1 + r) and
   copies them into both sides; SGLang goes first on even r and second on odd r. Before every timed replay a ~300 us
   spin and a read of a 256 MB buffer leave L2 cold. Every repetition's output is checked.
6. judge/score.py turns the times into the per-shape, per-load and overall speedups.
"""
import argparse
import importlib.util
import json
import math
import os
import pathlib
import random
import re
import shutil
import statistics
import sys
import tempfile

import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from judge import score, seeds  # noqa: E402

HW = {"hbm_bytes_per_s": 4.8e12, "bf16_dense_flops_per_s": 989e12}  # the hardware TASK.md's floors use
FLUSH_BYTES = 256 << 20  # larger than the H100 / H200 L2 (50 MB)
GATE_US = 300.0  # the official record's replay gate (replay_overhead.gate_ns, about 298 us)
GATE_CYCLES = int(GATE_US * 1980)  # spin length at the H200's 1980 MHz boost clock
DISTINCT = {"turn 1": 14.5, "turn 4": 14.0, "first turn": 16.0}
LABEL = {"first turn": "16K prompt"}  # display name of a load key
TEST_SEED = 901
CASE_KEYS = ("batch", "ctx", "layer", "q_len", "page", "dist", "spread")


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def clone(x):
    if isinstance(x, torch.Tensor):
        return x.clone()
    if isinstance(x, dict):
        return {k: clone(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return type(x)(clone(v) for v in x)
    return x


def copy_into(dst, src):
    if isinstance(dst, torch.Tensor):
        dst.copy_(src)
    elif isinstance(dst, dict):
        for k in dst:
            copy_into(dst[k], src[k])
    elif isinstance(dst, (list, tuple)):
        for a, b in zip(dst, src):
            copy_into(a, b)


def parse_case(line):
    case = {}
    for part in line.split(","):
        k, v = part.strip().split(" ", 1)
        case[k] = int(v) if v.strip().isdigit() else v.strip()
    return case


def renamed_control(path, out_dir):
    """A copy of `path` whose compiled extensions get new names (every name="..." gains a _ctl suffix): the same code
    under a new extension name, the control an A/B needs (docs/JUDGE.md)."""
    text = re.sub(r"\bname=([\"'])([A-Za-z_]\w*)\1", r"name=\1\2_ctl\1", pathlib.Path(path).read_text())
    out = pathlib.Path(out_dir) / (pathlib.Path(path).stem + "_ctl.py")
    out.write_text(text)
    return str(out)


class Problem:
    """The checker (reference.py, checks.py) loaded apart from the code under test, SGLang's baseline, the weights."""

    def __init__(self, problem_dir, seeds_from=None, seed_draw=None, seeds_job=None):
        src = pathlib.Path(problem_dir)
        chk = tempfile.mkdtemp()
        for f in ("reference.py", "checks.py", "nvfp4.py"):
            shutil.copy(src / f, chk)
        sys.path.insert(0, chk)
        self.ref = load_module("judge_reference", pathlib.Path(chk) / "reference.py")
        sys.path.remove(chk)
        for m in ("reference", "checks"):
            sys.modules.pop(m, None)  # the code under test must not find the checker's modules by import
        self.src = src
        self.task = (src / "TASK.md").read_text()
        self.meta = json.loads((src / "meta.json").read_text())
        self.cases = [dict(c) for c in self.meta["benchmarks"]]
        if seeds_from:  # replay one official run's case seeds (and so its router)
            rec = seeds.case_seeds(seeds_from, seeds_job)
            for c in self.cases:
                c["seed"] = rec[(c["load"], c["layer"], c["ctx"])]
        else:
            rng = random.Random(seed_draw)
            for c in self.cases:
                c["seed"] = rng.getrandbits(63)
        self._router_targets()
        base = self.ref.generate_weights(self.ref.WEIGHTS_SEED)
        fields = self.ref.weights_fields(base, self.cases, self.ref.WEIGHTS_SEED)
        del base
        self.router_seed = fields.get("router_seed")
        self.weights = self.ref.generate_weights(self.ref.WEIGHTS_SEED, **fields)
        torch.cuda.empty_cache()

    def _router_targets(self):
        """router_load per shape from its floor in TASK.md (prefill floors are flops-bound, so the floor fixes the
        routes term: invert speed_of_light_ns by bisection), and the fitted router_distinct per load."""
        floors = {}
        for m in re.finditer(r"^\| ([a-z ]+?\d*) \| (.+?) \| (\d+) \| ([\d,.]+) µs \|$", self.task, re.M):
            c = parse_case(m.group(2))
            floors[(m.group(1), c["batch"], c["ctx"], c["layer"], c["q_len"])] = float(m.group(4).replace(",", ""))
        for c in self.cases:
            floor_ns = 1e3 * floors[(c["load"], c["batch"], c["ctx"], c["layer"], c["q_len"])]
            shape = {k: c[k] for k in CASE_KEYS if k in c}
            lo, hi = 0.0, float(c["batch"] * c["q_len"] * self.ref.TOP_K)
            for _ in range(80):
                mid = (lo + hi) / 2
                lo, hi = (mid, hi) if self.ref.speed_of_light_ns(**shape, hw=HW, router_load=mid) < floor_ns else (lo, mid)
            c["router_load"] = (lo + hi) / 2
            c["router_distinct"] = DISTINCT[c["load"]]

    def inputs(self, case, seed):
        c = {k: case[k] for k in CASE_KEYS if k in case}
        c["seed"] = seed
        c.update(self.ref.case_fields(c, self.weights))  # the 32-expert range this input sends the most routes to
        return self.ref.generate_input(**c)

    def check(self, data, out):
        return self.ref.check_implementation(data, self.ref.collect(data, out), self.weights)

    def side(self, path, tag):
        """Import a submission (or the baseline) from a directory holding only it, baseline.py and nvfp4.py, as the
        judge gives a submission, and prepare it on its own copy of the weights."""
        d = pathlib.Path(tempfile.mkdtemp())
        for f in ("baseline.py", "nvfp4.py"):
            shutil.copy(self.src / f, d)
        shutil.copy(path, d / "submission.py")
        sys.path.insert(0, str(d))
        mod = load_module(f"judge_side_{tag}", d / "submission.py")
        state = mod.prepare(clone(self.weights))
        torch.cuda.synchronize()
        return lambda data: mod.kernel_fn(data, state)


def graph(fn):
    """Two warmup calls on a side queue, then capture."""
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(2):
            fn()
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        out = fn()
    torch.cuda.synchronize()
    return g, out


class Timer:
    """One timed replay: a ~300 us spin, an L2 flush that reads 256 MB, then the replay between two events."""

    def __init__(self):
        self.buf = torch.ones(FLUSH_BYTES // 4, dtype=torch.float32, device="cuda")
        self.sink = torch.empty((), dtype=torch.float32, device="cuda")

    def __call__(self, g):
        torch.cuda._sleep(GATE_CYCLES)
        torch.sum(self.buf, dim=0, out=self.sink)
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        g.replay()
        b.record()
        b.synchronize()
        return a.elapsed_time(b) * 1e6  # ns

    def overhead(self, n=60):
        """The replay fixed cost, as the official record computes it: 2 * one - two, from graphs of one and two
        tiny kernels (3.6 to 4.7us in the official records)."""
        t = torch.zeros(1, device="cuda")
        g1, _ = graph(lambda: t.add_(1))
        g2, _ = graph(lambda: (t.add_(1), t.add_(1)))
        one = statistics.median(self(g1) for _ in range(n))
        two = statistics.median(self(g2) for _ in range(n))
        return 2 * one - two


def run_tests(p, fn, name):
    lines = re.search(r"must pass these \d+ tests.*?```\n(.*?)```", p.task, re.S).group(1).strip().splitlines()
    for i, line in enumerate(lines):
        case = parse_case(line)
        data = p.inputs(case, TEST_SEED + i)
        ok, msg = p.check(data, fn(data))
        if ok:  # again as a captured graph, replayed on another fresh input
            g, out = graph(lambda d=data: fn(d))
            copy_into(data, p.inputs(case, TEST_SEED + i + 7919))
            g.replay()
            torch.cuda.synchronize()
            ok, msg = p.check(data, out)
            del g, out
        print(f"test {i} {name}: {'pass' if ok else 'FAIL ' + str(msg)}  ({line})", flush=True)
        if not ok:
            sys.exit(f"{name} fails test {i}")
        del data
        torch.cuda.empty_cache()


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("files", nargs="+", help="one submission, or several for an A/B in one process")
    ap.add_argument("--problem-dir", default=os.environ.get("MKC_PROBLEM_DIR", "/okc/problems/muse_prefill_layer"))
    ap.add_argument("--cases", default="", help="comma-separated indices into meta.json's shapes (default all)")
    ap.add_argument("--reps", type=int, default=0, help="repetitions per shape (default meta.json's, 6)")
    ap.add_argument("--seed", type=int, default=None, help="draw the case seeds from this seed (default: fresh)")
    ap.add_argument("--seeds-from", default=None, help="replay the case seeds of an official run: `okc status <job> "
                    "--json` output, or tests/data/official_runs.json with --job")
    ap.add_argument("--job", default=None, help="the run in tests/data/official_runs.json whose seeds --seeds-from uses")
    ap.add_argument("--control", action="store_true", help="add the first file again under renamed extensions")
    ap.add_argument("--skip-tests", action="store_true")
    ap.add_argument("--out", default=None, help="write the per-repetition times and results here as JSON")
    a = ap.parse_args()

    files = list(a.files)
    if a.control:
        files.append(renamed_control(files[0], tempfile.mkdtemp()))
    names = [pathlib.Path(f).name for f in files]
    p = Problem(a.problem_dir, a.seeds_from, a.seed, a.job)
    print(f"router seed {p.router_seed}; targets (routes, distinct) per shape "
          f"{[(round(c['router_load'], 1), c['router_distinct']) for c in p.cases]}", flush=True)
    base = p.side(p.src / "baseline.py", "sglang")
    subs = [p.side(f, i) for i, f in enumerate(files)]
    if not a.skip_tests:
        for fn, name in zip(subs, names):
            run_tests(p, fn, name)
    timer = Timer()
    ov = timer.overhead()
    print(f"graph replay fixed cost {ov / 1e3:.2f} us", flush=True)

    idx = [int(x) for x in a.cases.split(",")] if a.cases else range(len(p.cases))
    rows = []  # per shape: case, SGLang times, each file's times (ns, fixed cost included)
    for ci in idx:
        case = p.cases[ci]
        n = a.reps or case.get("repeats", 6)
        n += n % 2  # ratios come in pairs
        first = p.inputs(case, case["seed"])
        bufs = [clone(first) for _ in range(len(files) + 1)]  # SGLang's buffers, then each file's
        graphs = [graph(lambda d=bufs[0]: base(d))] + [graph(lambda d=d, fn=fn: fn(d)) for d, fn in zip(bufs[1:], subs)]
        bt = [[] for _ in files]  # each file gets its own SGLang replays, so each file's speedup is the official form
        st = [[] for _ in files]
        fails = []
        for r in range(n):
            fresh = p.inputs(case, case["seed"] + 1 + r)
            for j in [(r // 2 + k) % len(files) for k in range(len(files))]:  # rotate which file goes first
                for side in ("b", "s") if r % 2 == 0 else ("s", "b"):
                    k = 0 if side == "b" else j + 1
                    copy_into(bufs[k], fresh)
                    t = timer(graphs[k][0])
                    if side == "b":
                        bt[j].append(t)
                    else:
                        st[j].append(t)
                        ok, msg = p.check(bufs[k], graphs[k][1])
                        if not ok:
                            fails.append(f"{names[j]} rep {r}: {msg}")
            del fresh
        shape = {k: case[k] for k in ("load", "weight", "batch", "ctx", "layer", "q_len")}
        rows.append({"case": shape, "sglang_ns": bt, "yours_ns": st, "fails": fails})
        res = [score.shape(bt[j], st[j], ov) for j in range(len(files))]
        line = f"{ci} {LABEL.get(case['load'], case['load']):10s} {case['layer']:7s} ctx {case['ctx']:5d}  sglang {res[0]['sglang_ns'] / 1e3:9.1f} us"
        for j, r_ in enumerate(res):
            line += f" | {names[j]} {r_['yours_ns'] / 1e3:9.1f} us {r_['speedup']:.4f}x"
            if j:  # same inputs, same repetition: the per-repetition difference to the first file
                d = [(s - s0) / 1e3 for s, s0 in zip(st[j], st[0])]
                se = statistics.stdev(d) / math.sqrt(len(d))
                line += f" ({statistics.mean(d):+.2f} +- {se:.2f} us vs {names[0]})"
        print(line + ("  CHECK FAILED: " + "; ".join(fails[:3]) if fails else ""), flush=True)
        del graphs, bufs, first
        torch.cuda.empty_cache()

    results = []
    for j, name in enumerate(names):
        shapes = [(row["case"], score.shape(row["sglang_ns"][j], row["yours_ns"][j], ov)) for row in rows]
        res = score.score(shapes, p.meta.get("sglang_us") if len(shapes) == len(p.cases) else None)
        loads = "  ".join(f"{LABEL.get(k, k)} {v:.4f}x" for k, v in res["loads"].items())
        shown = f"  time {res['time_us'] / 1e3:.2f} ms" if "time_us" in res else ""
        # With --cases, the loads sum only the shapes that ran, so the geometric mean is not a leaderboard score.
        tag = ("SCORE" if len(shapes) == len(p.cases)
               else f"PARTIAL (not a score; {len(shapes)} of {len(p.cases)} shapes)")
        print(f"{tag} {name} {res['speedup']:.4f}x  95% [{res['ci'][0]:.4f}, {res['ci'][1]:.4f}]{shown}  | {loads}")
        results.append({"file": name, "speedup": res["speedup"], "ci": res["ci"], "loads": res["loads"],
                        "shapes": [{"case": c, "speedup": s["speedup"], "sglang_ns": s["sglang_ns"],
                                    "yours_ns": s["yours_ns"]} for c, s in shapes]})
    if a.out:
        rec = {"files": files, "router_seed": p.router_seed, "case_seeds": [c["seed"] for c in p.cases],
               "overhead_ns": ov, "rows": rows, "results": results}
        pathlib.Path(a.out).write_text(json.dumps(rec))
    if any(row["fails"] for row in rows):
        sys.exit("a timed output failed its check")


if __name__ == "__main__":
    main()
