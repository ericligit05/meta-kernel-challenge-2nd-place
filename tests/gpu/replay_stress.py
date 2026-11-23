"""Replay the submission's captured graph on many fresh inputs, checking every replay (worker for the GPU tests).

    MKC_PROBLEM_DIR=... python tests/gpu/replay_stress.py --cases 4,5 --reps 40 --seed-base 7600000

--cases are indices into meta.json's scored shapes. For each case: capture once, then per repetition make a fresh
input, poison the captured outputs, copy the input in, replay, and run the reference check. Prints one STRESS_CASE
line per case and STRESS_DONE PASS or FAIL; exits 1 on any failed check or CUDA error.
"""
import argparse
import sys
import time

import torch

import harness


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", required=True)
    ap.add_argument("--reps", type=int, default=40)
    ap.add_argument("--seed-base", type=int, default=7300000)
    a = ap.parse_args()
    p = harness.Problem()
    fails = 0
    for wi, idx in enumerate(int(x) for x in a.cases.split(",")):
        case = p.cases[idx]
        t0 = time.time()
        seed0 = a.seed_base + 100000 * wi
        data = p.inputs(case, seed0)
        g, out = harness.graph(lambda: p.call(data))
        bad = []
        for r in range(a.reps):
            fresh = p.inputs(case, seed0 + 1 + r)
            harness.poison(out)
            harness.copy_into(data, fresh)
            g.replay()
            torch.cuda.synchronize()
            ok, msg = p.check(data, out)  # data holds the cache the replay wrote its new entries into
            if not ok:
                bad.append(f"rep {r}: {msg[:300]}")
        fails += len(bad)
        print(f"STRESS_CASE scored{idx} reps {a.reps} fails {len(bad)} [{time.time() - t0:.0f}s]", flush=True)
        for b in bad[:5]:
            print("  FAIL", b, flush=True)
        del g, out, data
        torch.cuda.empty_cache()
    print(f"STRESS_DONE {'FAIL' if fails else 'PASS'} fails {fails}", flush=True)
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
