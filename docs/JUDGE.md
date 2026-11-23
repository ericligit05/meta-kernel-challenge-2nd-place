# Local judge

`judge/run.py` times a submission against SGLang's baseline the way the official judge does, on one H200.
`judge/score.py` implements the scoring alone and needs no GPU. On the submitted file, three local runs average 2.096x
against an official median of 2.088x over 19 runs.

## Running it

It needs the judge's software stack and the problem files, both described in [SETUP.md](SETUP.md).

    export MKC_PROBLEM_DIR=/path/to/problem
    python3 judge/run.py submission/submission.py                 # one full run, about 2 minutes once compiled
    python3 judge/run.py A.py B.py --control --reps 20 --cases 0,2,4,5    # A/B in one process
    python3 judge/run.py A.py --seeds-from tests/data/official_runs.json --job 09ce8d3608884bdd  # replay a run's seeds

A run prints the five TASK.md tests, each shape's SGLang time, the submission's time and speedup, and then the
per-load speedups and the score, as the official result does. With `--cases`, the loads sum only the shapes that
ran, so the last line reads PARTIAL instead of SCORE. `--out` writes every repetition's times as JSON.
With several files, each shape line also gives each file's time minus the first file's, averaged over repetitions
that ran on the same inputs, with its standard error. `--control` adds the first file again under new extension
names (see the first pitfall). `--seeds-from` replays the case seeds of one official run, read from
`tests/data/official_runs.json` (the 19 runs of the submitted code, one chosen with `--job`) or from the JSON that
`okc status <job> --json` prints for any run.

## What it replicates

- Fresh case seeds every run, and the router chosen from them by the problem's own `reference.weights_fields`, so
  every run times a different routing, as official runs do.
- The routing targets. We back the per-load routes target out of the TASK.md floors, which are flops-bound for
  prefill. The distinct-experts target is not published, so the defaults are the values that fit the official runs
  best, 14.5, 14 and 16 experts for turn 1, turn 4 and the 16K prompt.
- prepare() on separate copies of the weights, the submission imported from a directory that holds only its file,
  `baseline.py` and `nvfp4.py`, and the checker loaded apart from it.
- The five TASK.md tests, eagerly and as a replayed graph, before any timing.
- The timing contract in TASK.md: one CUDA graph per side, SGLang and the submission alternating on the same fresh
  inputs with the order swapped every repetition, a ~300us spin and a 256MB read before every replay so L2 is
  cold, the replay fixed cost (2 x one kernel minus two kernels, about 3.7us on our H200 and 3.6 to 4.7us in the
  official records) subtracted from both sides, every repetition's output checked, and 6 repetitions per shape.
- The scoring: each repetition's ratio, the geometric mean of each pair's two ratios, the median over pairs, loads
  as weighted sums, and the geometric mean of the three loads. `tests/test_judge_score.py` feeds `judge/score.py`
  the raw times of all 19 official runs and gets every official per-shape speedup, score, interval and shown time
  back to 4 decimals. The simpler pair ratio (b1 + b2) / (s1 + s2) misses the official per-shape speedup by up to
  0.003 on the same data, which the test also checks.

## What it does not replicate

- The official GPU. Each official run gets a machine from a pool, and the board shows one run. Run-to-run spread
  on the official side is 0.95% (one sd over the 19 runs, 2.042x to 2.129x).
- The exact router. The official per-repetition seeds and the distinct-experts target are inferred, not known.
- The 16 resident copies of the prepared state that the official record lists, the anti-cheat timings it records
  around each replay, the static checks on the source, and the 15-minute budget.

## Agreement with official runs

The submitted file, three local runs on one H200 against the 19 official runs of the same code:

| shape | official median | local runs | local mean / official |
|---|---|---|---|
| turn 1, sliding | 2.153 | 2.217, 2.226, 2.215 | 1.031 |
| turn 1, full | 2.249 | 2.263, 2.299, 2.269 | 1.012 |
| turn 4, sliding | 2.016 | 2.009, 2.012, 2.020 | 0.999 |
| turn 4, full | 2.168 | 2.154, 2.148, 2.161 | 0.994 |
| 16K prompt, full, ctx 0 | 2.004 | 1.985, 1.975, 2.051 | 1.000 |
| 16K prompt, sliding, ctx 32K | 1.974 | 1.960, 1.952, 1.980 | 0.995 |
| 16K prompt, full, ctx 32K | 2.093 | 2.090, 2.076, 2.076 | 0.994 |
| 16K prompt, full, ctx 64K | 2.187 | 2.166, 2.163, 2.180 | 0.992 |
| **score** | **2.088** | **2.093, 2.094, 2.102** | **1.004** |

The score lands 0.4% above the official median, well inside one official run's spread. Individual shapes agree less
well. Turn 1 reads 1 to 3% high, because on this H200 our layer ran faster and SGLang slower than on the official
machines, and the 16K shapes read 0 to 0.8% low. Treat a local score as an estimate of the official median within
about 1%, and a single shape as within about 3%.

## Two pitfalls

**Compare against a renamed control.** A new CUDA extension name alone, with identical code and identical SASS, read 1
to 2.5us slower at some shapes during development, which is the size of many real gains. An A/B of a change against
the original file therefore needs a third file, the original under new extension names. `--control` adds it, and a
change counts only if it beats the control, not just the original. On one H200, with 20 repetitions per shape, the
renamed copy read 1.4 +- 0.6us slower at turn 4 sliding, 0.1 +- 0.4us at turn 1 sliding, and 7.7 +- 4.2us (0.09%) at
both 16K shapes it ran.

**Alternate A/B in one process.** Two separate runs of the same file drift by several percent. The three runs above
span 0.5% in score and up to 3.8% on one shape, and the 19 official runs span 4.3%. A 1% gain between two separate
runs is noise. `run.py A.py B.py` alternates the files repetition by repetition on the same inputs and reports the
per-repetition difference in us with its standard error, which is the number to act on.
