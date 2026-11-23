# Tests

Each test here catches a failure class this layer has actually had. For each one, we give the bug it catches and the
evidence that a realistic version of that bug makes it fail. We left out tests that only re-ran a passing path.

The CPU tests run anywhere with Python 3.9+ and pytest, in a few seconds:

    python3 -m pytest tests/test_build_identity.py -q
    python3 -m pytest tests/test_judge_score.py -q

The GPU tests need an H200, the judge's software stack and the problem files, as
[docs/SETUP.md](../docs/SETUP.md) describes. They skip themselves without torch, a GPU or the problem files. Both pass
on an H200 in the challenge's container image, where test_correctness.py runs its 21 cases in 3 minutes and
test_concurrent_processes.py takes 3 minutes.

    export MKC_PROBLEM_DIR=/path/to/problem          # reference.py, checks.py, nvfp4.py, baseline.py, TASK.md, meta.json
    export MKC_SUBMISSION=submission/submission.py   # default; build/out/submission.py is the same code
    python3 -m pytest tests/gpu/test_correctness.py -q
    python3 -m pytest tests/gpu/test_concurrent_processes.py -q

`tests/gpu/harness.py` is a small stand-in for the judge's steps (separate checker modules, two warmup calls on a
side queue, graph capture, replays on fresh inputs). It does not time anything and does not pick a router seed per
load, so the expert load per GPU is the reference's default rather than a real run's.

## test_build_identity.py (CPU)

**Catches:** src/ drifting from the submitted code. src/ is the submitted layer with rewritten comments. A slip while
editing a comment (a deleted line, a changed constant, a `#define` joined to the next line, a comment opened inside
code) changes what runs, and nothing else would notice it before a GPU run.

**What it checks:** build/build_submission.py rebuilds the file from src/ and build/compare_tokens.py finds the same
code tokens as `submission/submission.py` (comments, docstrings and whitespace ignored). Both files must also match
the code-token fingerprint of the file behind our leaderboard row, recorded in `tests/data/leaderboard_file.json`.

**Evidence it fails on the bug:** the test applies seven realistic slips to the rebuilt file and requires the proof to
detect each one: a CUDA constant (`NW = 8` to 9), a CUDA call line commented out, a `#define` line joined to the next
line (a newline that is code), a Python threshold, a renamed function, an extension name, and a docstring turned into
an assignment. It also checks the other direction: a Python comment, a CUDA comment, a docstring edit and CUDA
re-indentation must not count as changes. The proof has caught a real slip. An intermediate src/ kept several
`#define` lines (`FT_ARV`, `FT_XB`, ...) that the submitted file does not have, because the build that produced it
dropped macros only comments named, and compare_tokens.py stopped at that exact token.

The same file also checks the build's guard against two components that define the same class name. The build
prefixes module-level names with the component alias but not classes defined inside an `if AVAILABLE:` block, so a
shared name makes the later class replace the earlier one and the first component silently runs the other's GEMM.
This happened during development, when both GEMM components defined `PdlGemmSm90` and the o projection ran with the
unpooler's evict_first weight policy (3.7us slower on one shape). The test copies src/, gives the second GEMM file
the first one's class name, and requires the build to fail. The same check finds the duplicate in that development file.

## gpu/test_correctness.py (GPU)

**Catches:** wrong outputs. It runs the problem's own check (reference.check_implementation: residual update and next
attention input against the FP32 reference, per-row bounds, routing rules, new cache entries) on the five TASK.md
tests and the eight scored shapes, eagerly and then as a CUDA graph replayed on three fresh inputs. Before each
replay it fills the captured outputs with NaN / -17, so rows a kernel skips, or a host-side choice captured from the
first input, show up as failures instead of passing on stale values.

It also runs eight nearby shapes at the two edges where a development version of this layer failed: short calls on
an 80K context, where full attention crashed with more than 16 key splits (q_len 9 to 65), and both sides of the
200-token switch below which full layers run SGLang's FA3.

**Evidence:** this is the judge's own acceptance check, run the way the judge runs it, so any output the judge would
reject fails here. Three mutated copies of the submission, each run on test0, scored0 and scored4, fail all three
cases where the real file passes. The first returns the last 64 rows without their residual update, and fails eagerly.
The second caches the experts of the first input and returns them for later inputs. It passes eagerly and fails the
first graph replay on routing. The third leaves the last 64 rows of one output unwritten only inside a captured graph.
It passes eagerly and fails the first graph replay. The last two are why the test replays graphs on fresh inputs
instead of stopping at the eager call. A milder mutation, the last 3 rows of the stream 5% too large, passes, because
it stays inside the check's own row bound, so the judge would accept it too.

## gpu/test_concurrent_processes.py (GPU)

**Catches:** ordering and memory faults that only appear when other work shares the GPU. Three processes replay the
submission at the same time (the 16K-token shapes split over two of them), 40 fresh inputs per shape, each replay
checked; any failed check, CUDA error or timeout fails the test.

**Evidence:** we found a real fault this way. In a development version, a weighted-sum kernel reached a `__shared__`
array only by an offset from another one, so the compiler dropped the array and live mbarriers shared memory with
other data. That version passed every serialized run and failed every 16K-token case in this three-process form. That
version is not in this repository, so we have not shown that this test catches the fault again. On the submitted file
it passes all 320 checked replays. The build now refuses that layout before any GPU run, and
test_build_identity.py shows it on the CPU by putting the old `full[NS], empty[NS]` declaration back into a copy of
`src/` and requiring the build to fail.

The worker checks the captured buffers, which the replay writes its new cache entries into, not the fresh input it
copied from. Checking the input would compare cache entries the kernel never touched and fail every replay
(relative error 1.4, the value for unrelated data).

## test_judge_score.py (CPU)

**Catches:** a local judge that scores differently from the official one. tests/data/official_runs.json stores every
shape's per-repetition replay times from all 19 official runs of the submitted code, as the official result records
give them. judge/score.py must turn them into each run's official per-shape speedups, SGLang times, score, 95%
interval and shown time, to 4 decimals.

**Evidence it fails on the bug:** the test also computes the pair ratio in its plausible wrong form, (b1 + b2) / (s1 +
s2), and requires that form to miss an official per-shape speedup at 4 decimals (it misses by up to 0.003).
[docs/JUDGE.md](../docs/JUDGE.md) compares the times judge/run.py measures with the official runs.
