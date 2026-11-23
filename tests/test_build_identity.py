"""CPU: src/ rebuilds into a file with exactly the leaderboard file's code tokens, and the proof detects real changes."""
import importlib.util
import json
import pathlib
import shutil

import pytest

REPO = pathlib.Path(__file__).resolve().parent.parent
SUBMITTED = REPO / "submission" / "submission.py"
LEADERBOARD = json.loads((REPO / "tests" / "data" / "leaderboard_file.json").read_text())


def _module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


build = _module("build_submission", REPO / "build" / "build_submission.py")
compare = _module("compare_tokens", REPO / "build" / "compare_tokens.py")


@pytest.fixture(scope="module")
def rebuilt():
    return build.build()


@pytest.fixture(scope="module")
def submitted_tokens():
    return compare.code_tokens(SUBMITTED.read_text())


def test_rebuilt_file_has_the_submitted_code_tokens(rebuilt, submitted_tokens):
    assert len(rebuilt.encode()) < build.LIMIT
    assert compare.code_tokens(rebuilt) == submitted_tokens


def test_both_files_match_the_leaderboard_file(rebuilt):
    """submission/ is the build of src/: comments removed, src/'s docstrings kept, the leaderboard file's code tokens."""
    assert compare.fingerprint(rebuilt) == LEADERBOARD["code_token_sha256"]
    assert compare.fingerprint(SUBMITTED.read_text()) == LEADERBOARD["code_token_sha256"]


def test_f_strings_are_one_token_on_every_python():
    """Python 3.12+ tokenizes f-strings in parts (PEP 701); the proof must see each one as its source text."""
    src = 'x = f"a{b!r:>{w}}c" + f\'{d}\'\ny = f"""\n{e}\n"""\n'
    toks = [t for t in compare.code_tokens(src) if t[0] == "FSTRING"]
    assert toks == [("FSTRING", 'f"a{b!r:>{w}}c"'), ("FSTRING", "f'{d}'"), ("FSTRING", 'f"""\n{e}\n"""')]


# Realistic slips while editing comments in src/, each applied to the rebuilt file. Every one changes what runs.
MUTATIONS = [
    ("a CUDA constant", "constexpr int NW = 8,", "constexpr int NW = 9,"),
    ("a CUDA line commented out", "  mbar_init(", "  //mbar_init("),
    ("a #define joined to the next line", "#define WSC_NS 3\n", "#define WSC_NS 3 "),
    ("a Python threshold", "ORDER_MIN = 2048", "ORDER_MIN = 4096"),
    ("a renamed Python name", "def xa__moe_tile(", "def xa__moe_tiles("),
    ("an extension name", '"okc_prefill_moe_x46001_b460010"', '"okc_prefill_moe_x46001_b460011"'),
    ("a docstring turned into code", '    """Compile the QuACK kernel once', '    QO_DOC = """Compile the QuACK kernel once'),
]


@pytest.mark.parametrize("what,old,new", MUTATIONS, ids=[m[0] for m in MUTATIONS])
def test_proof_detects_code_changes(rebuilt, submitted_tokens, what, old, new):
    assert old in rebuilt, f"mutation anchor for {what} not found"
    assert compare.code_tokens(rebuilt.replace(old, new, 1)) != submitted_tokens


# Changes the proof must ignore: comments, docstrings and whitespace.
NEUTRAL = [
    ("a Python comment", "ORDER_MIN = 2048", "ORDER_MIN = 2048  # a comment"),
    ("a CUDA comment", "constexpr int NW = 8,", "constexpr int NW = 8, /* a comment */"),
    ("a docstring", '"""Compile the QuACK kernel once', '"""Compile it once'),
    ("CUDA indentation", "\n  mbar_init(", "\n        mbar_init("),
]


@pytest.mark.parametrize("what,old,new", NEUTRAL, ids=[m[0] for m in NEUTRAL])
def test_proof_ignores_comments_and_whitespace(rebuilt, submitted_tokens, what, old, new):
    assert old in rebuilt, f"anchor for {what} not found"
    assert compare.code_tokens(rebuilt.replace(old, new, 1)) == submitted_tokens


def test_build_rejects_a_class_name_two_components_share(tmp_path):
    """Classes inside `if AVAILABLE:` are not prefixed, so a shared name would silently rebind (wrong GEMM runs)."""
    src = tmp_path / "src"
    shutil.copytree(REPO / "src", src)
    name = (src / "gemm_sm90.py").read_text().split("    class ", 1)[1].split("(", 1)[0]
    f = src / "gemm_sm90_evict_first.py"
    b = f.read_text()
    other = b.split("    class ", 1)[1].split("(", 1)[0]
    f.write_text(b.replace(other, name))
    with pytest.raises(AssertionError, match="defined twice"):
        build.build(root=src)


def test_build_rejects_a_shared_array_reached_only_by_offset(tmp_path):
    """The weighted-sum producer once declared `full[NS], empty[NS]` and reached `empty` only as fb + 8 * NS. Never
    named, `empty` was dropped by the compiler and s_live was placed over the live mbarriers, which faulted when two
    16K-token processes shared the GPU. The build must refuse that layout."""
    src = tmp_path / "src"
    shutil.copytree(REPO / "src", src)
    f = src / "kernels" / "weighted_sum_producer.cu"
    good = "__shared__ __align__(8) uint64_t full[2 * NS];"
    assert good in f.read_text()
    f.write_text(f.read_text().replace(good, "__shared__ __align__(8) uint64_t full[NS], empty[NS];"))
    with pytest.raises(AssertionError, match="empty.*never named"):
        build.build(root=src)
