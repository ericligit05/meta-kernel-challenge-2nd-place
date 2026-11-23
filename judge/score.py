"""The judge's scoring, as pure Python: per-repetition times in, per-shape, per-load and overall speedups out.

No GPU and no torch. judge/run.py feeds it the times it measures; tests/test_judge_score.py feeds it the times of 19
official runs and requires their official numbers back.

The official rules, as reproduced here:
- the graph replay fixed cost (overhead) is subtracted from every replay time, on both sides;
- repetition r gives the ratio r_i = (sglang_i - overhead) / (yours_i - overhead);
- repetitions come in pairs (0, 1), (2, 3), ... whose run order is swapped, and a pair's ratio is the GEOMETRIC mean
  sqrt(r_i * r_i+1) of its two ratios; a shape's speedup is the median over pairs;
- SGLang's time on a shape is the median of its replay times minus the overhead; yours is that over the speedup;
- a load's time is the weighted sum of its shapes' times, a load's speedup is SGLang's load time over yours, and the
  score is the geometric mean of the load speedups;
- the time shown for a run is SGLang's reference time per load (meta.json's sglang_us) over the score;
- the 95% interval is at least +-1.5% of the score.
"""
import math
import random
import statistics

CI_FLOOR = 0.015


def pair_speedups(sglang_ns, yours_ns, overhead_ns):
    """The per-pair ratios of one shape: the geometric mean of each pair's two per-repetition ratios."""
    assert len(sglang_ns) == len(yours_ns) and len(sglang_ns) >= 2
    r = [(b - overhead_ns) / (s - overhead_ns) for b, s in zip(sglang_ns, yours_ns)]
    return [math.sqrt(r[i] * r[i + 1]) for i in range(0, len(r) - 1, 2)]


def shape(sglang_ns, yours_ns, overhead_ns):
    """One shape's result: speedup (median over pairs), SGLang's time and yours, in ns."""
    pairs = pair_speedups(sglang_ns, yours_ns, overhead_ns)
    sp = statistics.median(pairs)
    base = statistics.median(sglang_ns) - overhead_ns
    return {"speedup": sp, "sglang_ns": base, "yours_ns": base / sp, "pairs": pairs}


def loads(shapes):
    """{load: (SGLang's weighted time, yours)} from [(case, shape result)]; case needs 'load' and 'weight'."""
    out = {}
    for case, res in shapes:
        t = out.setdefault(case["load"], [0.0, 0.0])
        t[0] += case["weight"] * res["sglang_ns"]
        t[1] += case["weight"] * res["yours_ns"]
    return {k: tuple(v) for k, v in out.items()}


def geomean(xs):
    xs = list(xs)
    return math.exp(sum(math.log(x) for x in xs) / len(xs))


def _score_of(shapes):
    return geomean(b / y for b, y in loads(shapes).values())


def score(shapes, sglang_reference_us=None, boot=2000, seed=0):
    """The run's result from [(case, shape result)].

    Returns per-load speedups, the score, the shown time (if the SGLang reference time is given) and a 95% interval:
    a bootstrap over each shape's pairs, floored at +-1.5% as the official interval is."""
    per_load = {k: b / y for k, (b, y) in loads(shapes).items()}
    sp = geomean(per_load.values())
    rng = random.Random(seed)
    draws = []
    for _ in range(boot):
        resampled = []
        for case, res in shapes:
            m = statistics.median(rng.choices(res["pairs"], k=len(res["pairs"])))
            resampled.append((case, {"sglang_ns": res["sglang_ns"], "yours_ns": res["sglang_ns"] / m}))
        draws.append(_score_of(resampled))
    draws.sort()
    lo, hi = draws[int(0.025 * boot)], draws[int(0.975 * boot) - 1]
    lo, hi = min(lo, sp * (1 - CI_FLOOR)), max(hi, sp * (1 + CI_FLOOR))
    out = {"loads": per_load, "speedup": sp, "ci": (lo, hi)}
    if sglang_reference_us:
        out["time_us"] = sglang_reference_us / sp
    return out
