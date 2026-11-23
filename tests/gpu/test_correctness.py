"""GPU: the problem's own correctness check on every TASK.md test, every scored shape and the nearby shapes below.

Each case runs once eagerly, then as a captured CUDA graph replayed on REPLAYS fresh inputs. Before every replay the
captured outputs are poisoned (NaN / -17), so a kernel that skips writing some rows, or a graph that captured a
host-side decision about the first input, fails the check instead of passing on stale values.

NEARBY covers the two edges where a development version of this layer failed: short calls on a long context, where
full attention crashed with more than 16 key splits, and the 200-token switch below which full layers run SGLang's
FA3, checked on both sides. They run on the same check and tolerance as the scored shapes.

    MKC_PROBLEM_DIR=/path/to/okc-init-dir python3 -m pytest tests/gpu/test_correctness.py -q
"""
import os

import pytest

torch = pytest.importorskip("torch", reason="the GPU tests need torch and the judge's stack")
import harness  # noqa: E402

OK, WHY = harness.available()
pytestmark = pytest.mark.skipif(not OK, reason=WHY)
REPLAYS = int(os.environ.get("MKC_REPLAYS", "3"))


@pytest.fixture(scope="module")
def problem():
    return harness.Problem()


def _cases():
    if not OK:
        return []
    import json
    import pathlib
    import re
    d = pathlib.Path(harness.PROBLEM_DIR)
    m = re.search(r"must pass these \d+ tests.*?```\n(.*?)```", (d / "TASK.md").read_text(), re.S)
    tests = [harness.parse_case(l) for l in m.group(1).strip().splitlines()]
    scored = json.loads((d / "meta.json").read_text())["benchmarks"]
    nearby = [harness.parse_case(l) for l in NEARBY]
    return [(f"test{i}", c, 9000 + i) for i, c in enumerate(tests)] + \
           [(f"scored{i}", c, 1000 + i) for i, c in enumerate(scored)] + \
           [(f"nearby{i}", c, 5000 + i) for i, c in enumerate(nearby)]


NEARBY = [
    "batch 1, ctx 80000, layer full, q_len 9, page 1",
    "batch 1, ctx 80000, layer full, q_len 65, page 1",
    "batch 1, ctx 80000, layer full, q_len 199, page 1",
    "batch 1, ctx 80000, layer full, q_len 200, page 1",
    "batch 1, ctx 80000, layer full, q_len 215, page 1",
    "batch 1, ctx 80000, layer sliding, q_len 9, page 1",
    "batch 1, ctx 80000, layer sliding, q_len 199, page 1",
    "batch 2, ctx 65536, layer full, q_len 64, page 1",
]


CASES = _cases()


@pytest.mark.parametrize("tag,case,seed", CASES, ids=[c[0] for c in CASES])
def test_case_passes_the_reference_check(problem, tag, case, seed):
    data = problem.inputs(case, seed)
    out = problem.call(data)
    torch.cuda.synchronize()
    ok, msg = problem.check(data, out)
    assert ok, f"{tag} eager: {msg}"
    del out
    g, gout = harness.graph(lambda: problem.call(data))
    for r in range(REPLAYS):
        harness.poison(gout)
        harness.copy_into(data, problem.inputs(case, seed + 7919 * (r + 1)))
        g.replay()
        torch.cuda.synchronize()
        ok, msg = problem.check(data, gout)
        assert ok, f"{tag} graph replay {r}: {msg}"
    del g, gout, data
    torch.cuda.empty_cache()
