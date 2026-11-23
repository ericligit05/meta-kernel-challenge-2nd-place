"""A small stand-in for the judge's steps, enough to run the submission against the problem's own checks.

It needs the problem files that `okc init muse_prefill_layer` writes (reference.py, checks.py, nvfp4.py, baseline.py,
TASK.md, meta.json): point MKC_PROBLEM_DIR at that directory. MKC_SUBMISSION picks the file under test (default: the
submitted file, submission/submission.py; build/out/submission.py is the same code rebuilt from src/).

As the judge does: the checker's modules are loaded from their own directory and removed from sys.modules before the
submission is imported; the submission's directory holds only its file, baseline.py and nvfp4.py; prepare() gets its
own copy of the weights; each case is captured as a CUDA graph after two warmup calls on a fresh side queue, and
replayed on fresh inputs copied into the captured buffers. What it does not do: time anything, or pick a router seed
per load (the weights use the reference's default seed), so the expert load per GPU differs from a real run.
"""
import importlib.util
import json
import os
import pathlib
import re
import shutil
import sys
import tempfile

import torch

REPO = pathlib.Path(__file__).resolve().parents[2]
SUBMISSION = pathlib.Path(os.environ.get("MKC_SUBMISSION", REPO / "submission" / "submission.py"))
PROBLEM_DIR = os.environ.get("MKC_PROBLEM_DIR", "")
CASE_KEYS_DROP = ("load", "weight", "repeats", "seed")


def available():
    """(True, '') when a GPU and the problem files are present, else (False, why)."""
    if not torch.cuda.is_available():
        return False, "needs a CUDA GPU (H200)"
    if not PROBLEM_DIR or not (pathlib.Path(PROBLEM_DIR) / "reference.py").exists():
        return False, "set MKC_PROBLEM_DIR to an `okc init muse_prefill_layer` directory"
    return True, ""


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def clone_data(x):
    if isinstance(x, torch.Tensor):
        return x.clone()
    if isinstance(x, dict):
        return {k: clone_data(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return type(x)(clone_data(v) for v in x)
    return x


def copy_into(dst, src):
    if isinstance(dst, torch.Tensor):
        dst.copy_(src)
    elif isinstance(dst, dict):
        for k in dst:
            if isinstance(dst[k], (torch.Tensor, dict, list, tuple)):
                copy_into(dst[k], src[k])
    elif isinstance(dst, (list, tuple)):
        for a, b in zip(dst, src):
            copy_into(a, b)


def parse_case(line):
    case = {}
    for part in line.split(","):
        k, v = part.strip().split(" ", 1)
        v = v.strip()
        for cast in (int, float):
            try:
                v = cast(v)
                break
            except ValueError:
                pass
        case[k] = v
    return case


class Problem:
    """The reference, the weights and one prepared copy of the submission (tag names its module)."""

    def __init__(self, tag="sub"):
        src = pathlib.Path(PROBLEM_DIR)
        chk, sub = tempfile.mkdtemp(), tempfile.mkdtemp()
        for f in ("reference.py", "checks.py", "nvfp4.py"):
            if (src / f).exists():
                shutil.copy(src / f, chk)
        for f in ("baseline.py", "nvfp4.py"):
            if (src / f).exists():
                shutil.copy(src / f, sub)
        shutil.copy(SUBMISSION, pathlib.Path(sub) / "submission.py")
        sys.path.insert(0, chk)
        self.ref = _load(f"reference_{tag}", pathlib.Path(chk) / "reference.py")
        sys.path.remove(chk)
        for m in ("reference", "checks"):
            sys.modules.pop(m, None)  # the submission must not find the checker's modules by import
        sys.path.insert(0, sub)
        self.sub = _load(f"submission_{tag}", pathlib.Path(sub) / "submission.py")
        task = (src / "TASK.md").read_text()
        m = re.search(r"must pass these \d+ tests.*?```\n(.*?)```", task, re.S)
        self.tests = [parse_case(l) for l in m.group(1).strip().splitlines()] if m else []
        self.cases = [dict(c) for c in json.loads((src / "meta.json").read_text())["benchmarks"]]
        self.weights = self.ref.generate_weights(self.ref.WEIGHTS_SEED)
        self.state = self.sub.prepare(clone_data(self.weights))
        torch.cuda.synchronize()

    def inputs(self, case, seed):
        c = {k: v for k, v in case.items() if k not in CASE_KEYS_DROP}
        c["seed"] = seed
        if hasattr(self.ref, "case_fields"):
            c.update(self.ref.case_fields(c, self.weights))
        return self.ref.generate_input(**c)

    def call(self, data):
        return self.sub.kernel_fn(data, self.state)

    def check(self, data, out):
        return self.ref.check_implementation(data, self.ref.collect(data, out), self.weights)


def graph(fn, warmup_stream=None):
    """Two warmup calls on a side queue, then capture, as the judge does."""
    s = warmup_stream or torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(2):
            fn()
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        out = fn()
    torch.cuda.synchronize()
    return g, out


def poison(outputs):
    """Fill the captured outputs with values a skipped write would leave visible."""
    for t in outputs:
        t.fill_(float("nan") if t.is_floating_point() else -17)
