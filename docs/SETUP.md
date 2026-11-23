# Setup

## Without a GPU

The CPU checks need Python 3.9 or later and pytest (`python3 -m pip install pytest`), and each finishes in seconds:

    python3 build/build_submission.py                  # rebuild the submission from src/
    python3 build/compare_tokens.py submission/submission.py build/out/submission.py
    python3 -m pytest tests/test_build_identity.py -q
    python3 -m pytest tests/test_judge_score.py -q     # every official speedup and score from the raw times
    python3 figures/figure_data.py --check             # the figures' numbers from figures/data and tests/data
    python3 figures/check_animation.py
    python3 tools/check_rs_inflight.py --selftest
    python3 tools/check_pdl_hoist.py --selftest

The GPU tests skip themselves on a machine without torch or without a GPU.

## The problem files

The local judge, the GPU tests and the SASS checks import the challenge's problem files, which are not redistributed
here. `okc` is the challenge's command-line client, installed with

    curl -fsSL https://openkernelchallenge.dev/install.sh | sh     # installs okc into ~/.okc/bin

Two commands write the problem files, and neither needs an account:

    okc init muse_prefill_layer /path/to/problem                   # reference.py, checks.py, nvfp4.py, baseline.py, TASK.md
    okc show muse_prefill_layer --json > /path/to/problem/meta.json  # the scored shapes and SGLang's reference times
    export MKC_PROBLEM_DIR=/path/to/problem

`okc init` does not write `meta.json`, which `judge/run.py` and the GPU tests read, and the JSON that `okc show`
prints carries every field they use. Both commands download from openkernelchallenge.dev, so they work only while the
site serves the problem. Running on the official judge (`okc check`, `okc submit`) also needs `okc login`, which
shows the rules and signs in with GitHub, and the kernel track stopped taking submissions on 2026-11-04.

## The GPU stack

We measured everything on one H200. The code fixes its persistent grids at 132 CTAs, one per SM of an H200 or an H100
SXM, so an H100 SXM should work but is untested, and a GPU with another SM count is unsupported. The kernels build for
sm_90a only.

The judge ran the challenge's container image `lmsysorg/sglang:dev-muse-spark-open`. That tag is mutable, so pin it
by the digest it has carried since 2026-10-15, before the challenge opened:

    lmsysorg/sglang:dev-muse-spark-open@sha256:8cfc5a72cb296771f5f8a4a4688cb6d05685610a830cbb616fa4ab0eb7bb2b0a

The image supplies torch, sgl_kernel, triton and the rest. Outside it, match the versions the code depends on:

| dependency | version | where the code depends on it |
|---|---|---|
| CUDA toolkit | 13.0.88, the judge's build | `tools/check_rs_inflight.py`. ptxas through CUDA 13.2 has a silent wrong-answer bug in register-input wgmma, fixed in 13.3, and that tool checks every build for it |
| SGLang | commit `a55ab7d5ae1bba59ac351e55026fe7f7644ecc5f` | `src/layer.py` imports `sglang.kernels.ops.muse_spark_v12`, which only that development branch has |
| quack-kernels | 0.6.5 | `src/gemm_sm90.py`, `src/gemm_sm90_evict_first.py` and the o GEMM in `src/layer.py` |
| flashinfer | any release that ships CUTLASS headers under `flashinfer/data/cutlass/include` | `src/attention_full.py` imports it when it loads, to find the CUTLASS headers |
| Python | 3.10 or later | `src/moe_block.py` annotates `torch.Tensor \| None` at module level |

## Running a change to src/

`src/layer.py` loads its components from its own directory, while `judge/run.py` copies only the file it is given, so
`judge/run.py src/layer.py` cannot find them. Rebuild first and judge the single file:

    python3 build/build_submission.py
    python3 judge/run.py build/out/submission.py
