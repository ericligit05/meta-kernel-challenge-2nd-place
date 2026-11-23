# Rebuilding the submission from src/

`submission/submission.py` is the submitted layer as one file. Because the judge accepts a single file, it merges
every component into one namespace and stays under the 1,000,000-byte limit. It has the same code tokens as the file
behind our leaderboard row (sha256 `9ac94705434fcee2530e7d22e048cf373ae537791b695fae09e2ddd7ee53466c`), with comments
removed and the docstrings of `src/`. `build/build_submission.py` rebuilds it byte for byte, and
`tests/data/leaderboard_file.json` records that file's code-token fingerprint. `src/` is the same code as a normal
source tree: one Python file per component, the CUDA of each extension in `src/kernels/*.cu`, and comments that
explain the code. This page shows how to turn `src/` back into one file and how to check that it is the submitted
code.

## Build

    python3 build/build_submission.py            # writes build/out/submission.py

It needs only the Python standard library (3.9 or later) and runs in a few seconds. As the top of
`build/build_submission.py` describes, it merges the components with their alias prefixes, inlines the `.cu` files as
raw strings, strips comments and squeezes blanks inside the CUDA strings, pastes in `src/parallel_load_inline.py`,
adds the extension name suffix, and runs the judge's known lint checks and a duplicate-class check.

## Proof that the code is the submitted code

    python3 build/compare_tokens.py submission/submission.py build/out/submission.py

Output:

    EQUAL: 25417 Python tokens, 15 CUDA strings with 195809 C tokens, all identical

With one argument it checks a file against the leaderboard file's recorded fingerprint:

    python3 build/compare_tokens.py build/out/submission.py
    MATCHES the leaderboard file's code tokens (file sha256 9ac94705...)

`compare_tokens.py` tokenizes both files with Python's `tokenize`, drops comments, blank lines and docstrings (a
string literal that is a statement on its own), and compares every other token, including NEWLINE, INDENT and DEDENT
and every string literal by value. It lexes each of the 15 CUDA source strings with its own C lexer, which drops `//`
and `/* */` comments and whitespace but keeps string and character literals whole and marks the end of every
preprocessor line, because a line break that ends a `#define` is code. That lexer is separate from the build's comment
stripper, so a stripping bug cannot hide itself. Names, numbers, operators, compiler flags and the load_inline
extension names all have to match. By default the build appends the submitted file's extension suffix (`_b460010`), so
the names match exactly. The suffix is only a cache key, since torch caches each compiled extension under its name.

`tests/test_build_identity.py` runs this proof and also shows that it fails on realistic slips: a changed CUDA
constant, a commented-out CUDA line, a `#define` joined to the next line, a changed Python threshold, a renamed
function, a changed extension name, and a docstring turned into an assignment. It also shows that comments,
docstrings and CUDA indentation do not count as changes.

## What differs, and why it is not byte equality

The result is token equality, not byte equality. `build/out/submission.py` has 899,649 bytes and 19,114 lines, and the
leaderboard file has 915,717 bytes and 23,336 lines. The differences are only of these kinds:

- Docstrings. The leaderboard file carries the docstrings written during development, which refer to files that are
  not in this repository; the rebuilt file carries the docstrings of `src/`, written for readers of this repository.
  Byte equality would require restoring the development text.
- Comment lines: line 2 of the leaderboard file, a comment that is not code, and the `# okc:` lines at the top of each
  component. The build keeps the first line, `# okc: problem=muse_prefill_layer`, because `okc submit` reads it.
- Blank lines and indentation. The script that produced the leaderboard file blanked dead configuration branches and
  comment-only lines in place to keep line numbers; `src/` deletes them instead.
- Line numbers inside the CUDA strings, as a consequence. The only code that reads a line number is `__LINE__` in
  error messages (`TORCH_CHECK`, a `fprintf` in an error macro) and in CUTLASS's synclog calls, which compile to
  nothing unless CUTLASS_ENABLE_SYNCLOG is defined. The compiled kernels are otherwise the same. Because the CUDA
  source text differs, torch's extension cache treats the rebuilt extensions as new builds and compiles them once.

We made `src/` from the development sources the submitted file was built from, with exactly the code changes that
build applied (the dead `#if` branches removed, the word `stream` renamed to `resid` where it is a name), and then
rewrote every comment and docstring. The proof above shows that nothing else changed.

## Running it

The rebuilt file is a drop-in submission. `okc check build/out/submission.py` runs it on the official judge after
`okc login`, for as long as the challenge accepts runs. To run the judge and the GPU tests locally, see
[SETUP.md](SETUP.md) and `tests/README.md`.
