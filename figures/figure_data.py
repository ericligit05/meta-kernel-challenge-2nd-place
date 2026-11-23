"""Every number drawn in Figures 2 and 3, recomputed from the repo's data.

    python3 figures/figure_data.py            # prints the numbers and rewrites figures/data/figure_numbers.json
    python3 figures/figure_data.py --check    # fails if figure_numbers.json is stale

Inputs:
- tests/data/official_runs.json: per-repetition times of the 19 official runs, scored with judge/score.py;
- figures/data/phase_profile.json: kernel time per phase for each shape, from one profiled call per side.
- figures/data/kernel_timeline.json: per-kernel start and end times of our layer at the Figure 2 shape, four replays.

Figure 3 uses the official run whose score is the median of the 19. Its per-shape judged times come from judge/score.py.
The judge times whole layers, not phases, so each shape's judged time is split across phases in proportion to that
shape's profiled kernel time per phase (the same split for every run). A load's phase time is the weighted sum over its
shapes, as the judge sums load times. The waterfall switches phases from SGLang's time to ours in the order
PHASE_ORDER and rescores; the last step equals the run's score by construction. Its intermediate values depend on that
order.

Figure 2 draws the summed kernel time of the turn-1 sliding shape straight from the profile (no scaling). Its timeline
of our kernels comes from figures/data/kernel_timeline.json, an Nsight Systems trace of four replays: each kernel's start
and end are the medians over the replays, and the overlap intervals are where two or more kernels are on the GPU at once.

The SVGs themselves are hand-built; this script is the source of their numbers.
"""
import json
import math
import os
import statistics
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from judge import score as S  # noqa: E402

PHASE_ORDER = ["routed experts", "attention", "dense GEMMs", "other"]
LOADS = [("turn 1", "Turn 1"), ("turn 4", "Turn 4"), ("first turn", "16K prompt")]
OUT = os.path.join(ROOT, "figures", "data", "figure_numbers.json")


def key(case):
    return (case["load"], case["batch"], case["ctx"], case["layer"], case["q_len"])


def run_shapes(run):
    return [(s["case"], S.shape(s["baseline_runs_ns"], s["runs_ns"], s["overhead_ns"])) for s in run["shapes"]]


def median_run(runs):
    scored = sorted((S.geomean(b / y for b, y in S.loads(run_shapes(r)).values()), r["job_id"], r) for r in runs)
    assert len(scored) % 2 == 1
    return scored[len(scored) // 2], [round(s, 4) for s, _, _ in scored]


def phase_shares(profile):
    shares = {}
    for c in profile["cases"]:
        per_side = {}
        for side in ("sglang", "ours"):
            ph = c[side]["phases_us"]
            tot = sum(ph.values())
            per_side[side] = {p: ph[p] / tot for p in PHASE_ORDER}
        shares[key(c["case"])] = per_side
    return shares


def timeline():
    tl = json.load(open(os.path.join(ROOT, "figures", "data", "kernel_timeline.json")))
    reps = tl["replays_us"]
    kernels = []
    for i, label in enumerate(tl["labels"]):
        kernels.append({"label": label, "start_us": round(statistics.median(r[i][0] for r in reps), 2),
                        "end_us": round(statistics.median(r[i][1] for r in reps), 2)})
    # intervals covered by two or more kernels
    ev = sorted([(k["start_us"], 1) for k in kernels] + [(k["end_us"], -1) for k in kernels], key=lambda e: (e[0], e[1]))
    overlaps, depth, t_in = [], 0, None
    for t, d in ev:
        depth += d
        if depth >= 2 and t_in is None:
            t_in = t
        elif depth < 2 and t_in is not None:
            if t > t_in:
                overlaps.append([round(t_in, 2), round(t, 2)])
            t_in = None
    span = round(max(k["end_us"] for k in kernels) - min(k["start_us"] for k in kernels), 2)
    return {"kernels": kernels, "overlaps_us": overlaps, "span_us": span,
            "overlap_total_us": round(sum(b - a for a, b in overlaps), 1),
            "replay_spans_us": [round(r[-1][1] - r[0][0], 2) for r in reps]}


def compute():
    runs = json.load(open(os.path.join(ROOT, "tests", "data", "official_runs.json")))["runs"]
    profile = json.load(open(os.path.join(ROOT, "figures", "data", "phase_profile.json")))
    (sc, job, run), all_scores = median_run(runs)
    shapes = run_shapes(run)
    shares = phase_shares(profile)

    # per load, per phase, per side: weighted judged time (ms)
    def load_phase_times(switched):
        out = {}
        for case, res in shapes:
            sh = shares[key(case)]
            t = out.setdefault(case["load"], [0.0, 0.0])
            t[0] += case["weight"] * res["sglang_ns"]
            for p in PHASE_ORDER:
                if p in switched:
                    t[1] += case["weight"] * res["yours_ns"] * sh["ours"][p]
                else:
                    t[1] += case["weight"] * res["sglang_ns"] * sh["sglang"][p]
        return out

    loads = {}
    for case, res in shapes:
        sh = shares[key(case)]
        L = loads.setdefault(case["load"], {"sglang_ms": 0.0, "ours_ms": 0.0,
                                            "sglang_phase_ms": dict.fromkeys(PHASE_ORDER, 0.0),
                                            "ours_phase_ms": dict.fromkeys(PHASE_ORDER, 0.0)})
        L["sglang_ms"] += case["weight"] * res["sglang_ns"] / 1e6
        L["ours_ms"] += case["weight"] * res["yours_ns"] / 1e6
        for p in PHASE_ORDER:
            L["sglang_phase_ms"][p] += case["weight"] * res["sglang_ns"] * sh["sglang"][p] / 1e6
            L["ours_phase_ms"][p] += case["weight"] * res["yours_ns"] * sh["ours"][p] / 1e6
    fig3_loads = []
    for k, label in LOADS:
        L = loads[k]
        fig3_loads.append({"load": k, "label": label, "speedup": round(L["sglang_ms"] / L["ours_ms"], 4),
                           "sglang_ms": round(L["sglang_ms"], 2), "ours_ms": round(L["ours_ms"], 2),
                           "sglang_phase_pct_of_sglang": {p: round(100 * v / L["sglang_ms"], 2) for p, v in L["sglang_phase_ms"].items()},
                           "ours_phase_pct_of_sglang": {p: round(100 * v / L["sglang_ms"], 2) for p, v in L["ours_phase_ms"].items()}})
    waterfall, switched = [], []
    for p in PHASE_ORDER:
        switched.append(p)
        t = load_phase_times(set(switched))
        waterfall.append({"switched": p, "score": round(S.geomean(b / y for b, y in t.values()), 4)})
    assert abs(waterfall[-1]["score"] - sc) < 1e-3

    c0 = next(c for c in profile["cases"] if c["case"]["load"] == "turn 1" and c["case"]["layer"] == "sliding")
    fig2_rows = {}
    for side in ("sglang", "ours"):
        ph = c0[side]["phases_us"]
        fig2_rows[side] = {"routed experts": ph["routed experts"], "dense GEMMs": ph["dense GEMMs"],
                           "other (attention included)": round(ph["attention"] + ph["other"], 1),
                           "total": round(sum(ph.values()), 1)}
    fig2_rows["ratio"] = {k: round(fig2_rows["sglang"][k] / fig2_rows["ours"][k], 3) for k in fig2_rows["ours"]}
    ours = c0["ours"]
    judged = [statistics.median(v) for v in zip(*[
        [(s["official"]["baseline_ns"] / 1e3, s["official"]["median_ns"] / 1e3)
         for s in r["shapes"] if s["case"]["load"] == "turn 1" and s["case"]["layer"] == "sliding"][0]
        for r in runs])]
    return {
        "median_run": {"job_id": job, "score": round(sc, 4), "all_scores_sorted": all_scores},
        "figure3": {"phase_order": PHASE_ORDER, "loads": fig3_loads, "waterfall": waterfall},
        "figure2": {"shape": "turn 1, sliding layer, 753 tokens on an 80K context",
                    "summed_kernel_us": fig2_rows,
                    "ours_gpu_busy_us": ours["gpu_busy_us"],
                    "ours_pdl_overlap_us": round(ours["kernel_sum_us"] - ours["gpu_busy_us"], 1),
                    "judged_median_us_over_19_runs": {"sglang": round(judged[0], 0), "ours": round(judged[1], 0)},
                    "timeline": timeline()},
    }


def main():
    out = compute()
    text = json.dumps(out, indent=1) + "\n"
    if "--check" in sys.argv:
        if open(OUT).read() != text:
            sys.exit("figures/data/figure_numbers.json is stale: run python3 figures/figure_data.py")
        print("figure_numbers.json is current")
        return
    open(OUT, "w").write(text)
    print(text)


if __name__ == "__main__":
    main()
