"""GPU: three processes replaying the submission on one GPU at the same time, every replay checked.

The judge may share a GPU between processes, and the final reruns run on other machines. A kernel with a
shared-memory or barrier ordering bug can pass every serialized run and still fault when other work on the GPU
changes its timing: a weighted-sum kernel whose mbarriers shared memory with another array passed alone and
failed with 'unspecified launch failure' on every 16K-token case once three processes ran together. This test runs
three replay_stress.py workers at once over all scored shapes (the 16K shapes in two of them) and fails on any check
failure, CUDA error or timeout.

    MKC_PROBLEM_DIR=... python3 -m pytest tests/gpu/test_concurrent_processes.py -q
"""
import os
import pathlib
import subprocess
import sys

import pytest

pytest.importorskip("torch", reason="the GPU tests need torch and the judge's stack")
import harness  # noqa: E402

OK, WHY = harness.available()
pytestmark = pytest.mark.skipif(not OK, reason=WHY)
HERE = pathlib.Path(__file__).resolve().parent
REPS = os.environ.get("MKC_STRESS_REPS", "40")
GROUPS = [("4,5", 7600000), ("6,7", 7700000), ("0,1,2,3", 7800000)]


def test_three_concurrent_processes_pass_every_replay():
    env = dict(os.environ, PYTHONUNBUFFERED="1")
    procs = [subprocess.Popen([sys.executable, str(HERE / "replay_stress.py"), "--cases", c, "--reps", REPS,
                               "--seed-base", str(s)], cwd=HERE, env=env, stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT, text=True) for c, s in GROUPS]
    results = []
    for (cases, _), pr in zip(GROUPS, procs):
        try:
            out, _ = pr.communicate(timeout=1500)
        except subprocess.TimeoutExpired:
            pr.kill()
            out, _ = pr.communicate()
            results.append((cases, "timeout", out))
            continue
        results.append((cases, pr.returncode, out))
    bad = [(c, rc, out[-1500:]) for c, rc, out in results if rc != 0 or "STRESS_DONE PASS" not in out]
    assert not bad, "\n\n".join(f"cases {c}: rc {rc}\n{tail}" for c, rc, tail in bad)
