"""Read a CUDA source file from src/kernels/.

Each component builds its extension with load_inline from a CUDA source string. In src/ those sources live in
src/kernels/*.cu so they read as CUDA; build/build_submission.py replaces every cuda_source("x.cu") call with the
file's text as a raw string literal, so the submission is one self-contained file.
"""
import pathlib

KERNELS = pathlib.Path(__file__).resolve().parent / "kernels"


def cuda_source(name):
    return (KERNELS / name).read_text()
