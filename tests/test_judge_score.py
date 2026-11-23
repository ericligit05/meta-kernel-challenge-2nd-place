"""CPU: judge/score.py reproduces the official numbers of all 19 official runs of the submitted code.

tests/data/official_runs.json holds, for each official run, every shape's per-repetition replay times on both
sides and the graph fixed cost, exactly as the official result record gives them, next to the record's own speedups.
score.py must turn the times into the official per-shape speedup, SGLang's per-shape time, the run's speedup, its
95% interval and its shown time, each to 4 decimals.

What this catches: a wrong pair formula (the sum form (b1 + b2) / (s1 + s2) misses the official per-shape speedup by
up to 2.9e-3 here), a wrong overhead subtraction, unweighted or arithmetic-mean loads, or the wrong interval floor.
The test proves the first claim on the data rather than asserting it. The last tests check that judge/seeds.py reads
the same case seeds from the fixture and from the two official record formats, so --seeds-from can replay any run.
"""
import json
import math
import pathlib
import statistics
import sys

import pytest

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from judge import score, seeds  # noqa: E402

DATA = json.loads((REPO / "tests" / "data" / "official_runs.json").read_text())
RUNS = DATA["runs"]


def _shapes(run):
    return [(s["case"], score.shape(s["baseline_runs_ns"], s["runs_ns"], s["overhead_ns"])) for s in run["shapes"]]


def test_fixture_has_19_runs_of_8_shapes():
    assert len(RUNS) == 19 and all(len(r["shapes"]) == 8 for r in RUNS)


@pytest.mark.parametrize("run", RUNS, ids=[r["job_id"] for r in RUNS])
def test_every_shape_matches_the_official_record(run):
    for s, (_, res) in zip(run["shapes"], _shapes(run)):
        off = s["official"]
        assert res["speedup"] == pytest.approx(off["speedup"], abs=5e-5), s["case"]
        assert res["sglang_ns"] == pytest.approx(off["baseline_ns"], rel=5e-5), s["case"]


@pytest.mark.parametrize("run", RUNS, ids=[r["job_id"] for r in RUNS])
def test_run_score_interval_and_time_match_the_official_record(run):
    res = score.score(_shapes(run), DATA["sglang_reference_us"], boot=200)
    off = run["official"]
    assert res["speedup"] == pytest.approx(off["speedup"], abs=5e-5)
    assert res["ci"][0] == pytest.approx(off["speedup_ci"][0], abs=5e-5)
    assert res["ci"][1] == pytest.approx(off["speedup_ci"][1], abs=5e-5)
    assert res["time_us"] == pytest.approx(off["score_us"], rel=5e-5)


def test_the_sum_form_of_the_pair_ratio_does_not_match():
    """The sum form (b1 + b2) / (s1 + s2) must fail the same 4-decimal bar somewhere, or the test above would not be
    able to tell the two formulas apart."""
    worst = 0.0
    for run in RUNS:
        for s in run["shapes"]:
            b, y, ov = s["baseline_runs_ns"], s["runs_ns"], s["overhead_ns"]
            pairs = [(b[i] + b[i + 1] - 2 * ov) / (y[i] + y[i + 1] - 2 * ov) for i in range(0, len(b) - 1, 2)]
            worst = max(worst, abs(statistics.median(pairs) - s["official"]["speedup"]))
    assert worst > 5e-5


def test_median_of_the_19_official_scores():
    """The number docs/JUDGE.md compares local runs with."""
    med = statistics.median(score.score(_shapes(r), boot=1)["speedup"] for r in RUNS)
    assert med == pytest.approx(2.0875, abs=5e-5)
    assert math.isclose(med, statistics.median(r["official"]["speedup"] for r in RUNS), abs_tol=1e-6)



def test_every_run_records_one_seed_per_shape():
    for run in RUNS:
        got = seeds.case_seeds(REPO / "tests" / "data" / "official_runs.json", run["job_id"])
        assert len(got) == 8 and len(set(got.values())) == 8, run["job_id"]


def test_seeds_read_the_same_from_every_record_format(tmp_path):
    """The `okc status <job> --json` form (benchmark events under result.records) and the plain "benchmarks" form
    must give the fixture's seeds, keyed the way judge/run.py looks them up."""
    run = RUNS[1]
    want = seeds.case_seeds(REPO / "tests" / "data" / "official_runs.json", run["job_id"])
    status = {"result": {"records": [{"event": "setup"}] + [{"event": "benchmark", "case": s["case"]}
                                                           for s in run["shapes"]]}}
    plain = {"benchmarks": [{"case": s["case"]} for s in run["shapes"]]}
    for name, d in (("status.json", status), ("plain.json", plain)):
        (tmp_path / name).write_text(json.dumps(d))
        assert seeds.case_seeds(tmp_path / name) == want
    assert set(want) == {(s["case"]["load"], s["case"]["layer"], s["case"]["ctx"]) for s in run["shapes"]}
    with pytest.raises(SystemExit, match="pass --job"):
        seeds.case_seeds(REPO / "tests" / "data" / "official_runs.json")
