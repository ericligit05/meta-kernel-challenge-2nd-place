# Static checks

Two scripts read the SASS that ptxas emitted for every kernel of the submission, so they need cuobjdump or nvdisasm
from the CUDA toolkit and, to build the extensions, an H200 with the judge's stack.
[`check_rs_inflight.py`](check_rs_inflight.py) proves that no instruction overwrites the A registers of a
register-input wgmma still in flight. ptxas through CUDA 13.2 can produce exactly that silent wrong answer (NVIDIA's
CUDA 13.2 known issue "Data Race in WGMMA A/B Register Copy Propagation", fixed in CUDA 13.3, and Triton issue #9433),
and the judge builds with 13.0.88. [`check_pdl_hoist.py`](check_pdl_hoist.py) proves that no kernel loads a
predecessor's output before its PDL wait, using a table of this layer's kernel parameters. Run either as
`python3 tools/<script> submission/submission.py`, or with `--selftest` on any CPU for their synthetic positive and
negative controls. The candidate builds we checked before submission passed both, with 0 violations over 752
register-input wgmmas and 0 failures over 53 kernels with a PDL wait. The PDL check's only failure in development was
a stale table entry, not a real hoist.
