"""Prove two submission files run the same code: their code tokens must be equal.

    python3 build/compare_tokens.py submission/submission.py build/out/submission.py

Python side (tokenize): comments, blank lines and docstrings (a string literal that is a statement by itself) are
dropped; every other token is compared, NEWLINE / INDENT / DEDENT included, and string literals by value (so
quote style may differ). CUDA side: every raw triple-quoted string that holds CUDA source (#include, #define or
__global__) is lexed by its own C lexer that drops // and /* */ comments and whitespace, keeps string and
character literals whole, and marks the end of every preprocessor directive line (a directive ends at its line
break, so a newline there is code). Everything else, including names, numbers, operators and extension names,
must match exactly. The C lexer here is independent of the build script's comment stripper.

Prints EQUAL and exits 0, or prints the first difference and exits 1. With one file, it checks that file against
the recorded code-token fingerprint of the leaderboard file (tests/data/leaderboard_file.json):

    python3 build/compare_tokens.py build/out/submission.py
"""
import ast
import hashlib
import io
import re
import sys
import tokenize

C_TOKEN = re.compile(r"""
    (?P<lc>//[^\n]*)
  | (?P<bc>/\*.*?\*/)
  | (?P<str>"(?:[^"\\\n]|\\.)*")
  | (?P<chr>'(?:[^'\\\n]|\\.)*')
  | (?P<nl>\n)
  | (?P<cont>\\\n)
  | (?P<ws>[ \t\r\f\v]+)
  | (?P<word>[A-Za-z0-9_.]+)
  | (?P<punct>.)
""", re.S | re.X)


def c_tokens(src):
    """C tokens without comments or whitespace; '<EOD>' closes each preprocessor directive."""
    toks, pos, at_line_start, in_directive = [], 0, True, False
    for m in C_TOKEN.finditer(src):
        assert m.start() == pos, ("lexer gap", src[pos:pos + 40])
        pos = m.end()
        kind, text = m.lastgroup, m.group()
        if kind == "nl":
            if in_directive:
                toks.append("<EOD>")
            in_directive, at_line_start = False, True
            continue
        if kind == "bc" and "\n" in text:
            # a multi-line block comment ends a directive line just as a newline does
            if in_directive:
                toks.append("<EOD>")
            in_directive, at_line_start = False, True
            continue
        if kind in ("lc", "bc", "ws", "cont"):
            continue
        if at_line_start and text == "#":
            in_directive = True
        at_line_start = False
        toks.append(text)
    assert pos == len(src)
    if in_directive:
        toks.append("<EOD>")
    return toks


def is_cuda_string(s):
    m = re.match(r"([rRbBuUfF]*)('''|\"\"\")", s)
    return bool(m and "r" in m.group(1).lower() and not set("bBfF") & set(m.group(1))
                and re.search(r"#include|#define|__global__", s))


def docstring_lines(src):
    """Start lines of string literals that are statements on their own (docstrings and bare strings)."""
    rows = set()
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
            rows.add(node.lineno)
    return rows


def python_tokens(src):
    """tokenize's tokens, with each f-string as one STRING token on every Python version.

    From Python 3.12 (PEP 701) tokenize splits an f-string into FSTRING_START, its parts and FSTRING_END; before
    3.12 it is one STRING token. The parts are joined back into one STRING token holding the f-string's exact source
    text, so the token stream, and the fingerprint, are the same on every version.
    """
    start_tok = getattr(tokenize, "FSTRING_START", None)
    end_tok = getattr(tokenize, "FSTRING_END", None)
    lines = src.splitlines(keepends=True)
    depth, first = 0, None
    for t in tokenize.generate_tokens(io.StringIO(src).readline):
        if start_tok is not None and t.type == start_tok:
            depth += 1
            if depth == 1:
                first = t
            continue
        if depth:
            if t.type == end_tok:
                depth -= 1
                if depth == 0:
                    (r0, c0), (r1, c1) = first.start, t.end
                    text = lines[r0 - 1][c0:] if r0 != r1 else lines[r0 - 1][c0:c1]
                    if r0 != r1:
                        text += "".join(lines[r0:r1 - 1]) + lines[r1 - 1][:c1]
                    yield tokenize.TokenInfo(tokenize.STRING, text, first.start, t.end, first.line)
            continue
        yield t


def code_tokens(src):
    skip_rows = docstring_lines(src)
    out, skipping = [], False
    for t in python_tokens(src):
        if t.type in (tokenize.COMMENT, tokenize.NL, tokenize.ENDMARKER):
            continue
        if t.type == tokenize.STRING and t.start[0] in skip_rows and (not out or out[-1][0] in
                                                                       ("NEWLINE", "INDENT", "DEDENT")):
            skipping = True  # the docstring and the NEWLINE that ends it
            continue
        if skipping and t.type == tokenize.NEWLINE:
            skipping = False
            continue
        skipping = False
        if t.type == tokenize.STRING:
            if is_cuda_string(t.string):
                out.append(("CUDA", tuple(c_tokens(ast.literal_eval(t.string)))))
            elif re.match(r"[rRbBuU]*['\"]", t.string):
                out.append(("STRING", repr(ast.literal_eval(t.string))))
            else:  # f-string: compared as written
                out.append(("FSTRING", t.string))
        else:
            out.append((tokenize.tok_name[t.type], t.string if t.type != tokenize.DEDENT else ""))
    return out


def fingerprint(src):
    """SHA-256 of a file's code token stream: equal for two files exactly when code_tokens() is equal."""
    return hashlib.sha256(repr(code_tokens(src)).encode()).hexdigest()


def first_diff(a, b):
    for k, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return k
    return min(len(a), len(b))


def main(pa, pb):
    a, b = code_tokens(open(pa).read()), code_tokens(open(pb).read())
    ncuda = sum(1 for t in a if t[0] == "CUDA")
    nc = sum(len(t[1]) for t in a if t[0] == "CUDA")
    if a == b:
        print(f"EQUAL: {len(a)} Python tokens, {ncuda} CUDA strings with {nc} C tokens, all identical")
        return 0
    k = first_diff(a, b)
    print(f"DIFFERENT at Python token {k} of {len(a)} / {len(b)}")
    for side, toks in (("A", a), ("B", b)):
        ctx = toks[max(0, k - 5):k + 5]
        for t in ctx:
            if t[0] == "CUDA":
                print(f"  {side} CUDA string ({len(t[1])} tokens)")
            else:
                print(f"  {side} {t}")
    if k < len(a) and k < len(b) and a[k][0] == b[k][0] == "CUDA":
        j = first_diff(a[k][1], b[k][1])
        print("  first CUDA token difference at", j)
        print("  A:", " ".join(a[k][1][max(0, j - 15):j + 15]))
        print("  B:", " ".join(b[k][1][max(0, j - 15):j + 15]))
    return 1


def check_leaderboard(path):
    import json
    import pathlib
    rec = json.loads((pathlib.Path(__file__).resolve().parent.parent / "tests" / "data" / "leaderboard_file.json").read_text())
    ok = fingerprint(open(path).read()) == rec["code_token_sha256"]
    print(("MATCHES" if ok else "DOES NOT MATCH") + f" the leaderboard file's code tokens (file sha256 {rec['sha256']})")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(check_leaderboard(sys.argv[1]) if len(sys.argv) == 2 else main(sys.argv[1], sys.argv[2]))
