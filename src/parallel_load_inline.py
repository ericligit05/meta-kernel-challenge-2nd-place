"""Build every load_inline extension of the submission concurrently.

build/build_submission.py pastes this right after the submission's one `load_inline` import. It rebinds
load_inline so each call starts the build on a daemon thread and returns a proxy whose first attribute read waits
for that build. nvcc runs in subprocesses, so the GIL is free while it runs and the extensions compile side by
side: a cold import costs about the slowest single build instead of the sum of all of them. The kernels and their
compiler flags are unchanged. A build error is raised again at the extension's first use.
"""
from torch.utils.cpp_extension import load_inline
import threading as _par_threading
_par_load_inline = load_inline


class _ParExt:
    """An extension module still compiling on a daemon thread: the first attribute read waits for the build."""

    def __init__(self, args, kwargs):
        box = {}

        def build():
            try:
                box["mod"] = _par_load_inline(*args, **kwargs)
            except BaseException as e:  # raised again at first use
                box["err"] = e

        th = _par_threading.Thread(target=build, daemon=True)
        th.start()
        object.__setattr__(self, "_box", box)
        object.__setattr__(self, "_th", th)

    def __getattr__(self, name):
        object.__getattribute__(self, "_th").join()
        box = object.__getattribute__(self, "_box")
        if "err" in box:
            raise box["err"]
        return getattr(box["mod"], name)


def load_inline(*args, **kwargs):  # noqa: F811 (concurrent builds, same arguments)
    return _ParExt(args, kwargs)
