"""The case seeds of an official run, read from any of the three files that record them. No GPU and no torch.

- the JSON that `okc status <job> --json` prints for any official run, yours included;
- tests/data/official_runs.json, which holds the 19 official runs of the submitted code, with `job` naming one;
- a JSON with a top-level "benchmarks" list of {"case": {...}} records.

judge/run.py --seeds-from uses it to replay a run's inputs and, through them, its router.
"""
import json
import pathlib


def _key(case):
    return case["load"], case["layer"], case["ctx"]


def case_seeds(path, job=None):
    """{(load, layer, ctx): seed} of one official run."""
    d = json.loads(pathlib.Path(path).read_text())
    if "runs" in d:
        runs = [r for r in d["runs"] if r["job_id"] == job]
        if len(runs) != 1:
            raise SystemExit(f"{path} holds {len(d['runs'])} runs: pass --job with one of "
                             + ", ".join(r["job_id"] for r in d["runs"]))
        cases = [s["case"] for s in runs[0]["shapes"]]
    elif "benchmarks" in d:
        cases = [b["case"] for b in d["benchmarks"]]
    else:
        cases = [r["case"] for r in d["result"]["records"] if r.get("event") == "benchmark"]
    if not cases or any("seed" not in c for c in cases):
        raise SystemExit(f"{path} records no case seeds")
    return {_key(c): c["seed"] for c in cases}
