"""Turn src/ into the single-file submission.

    python3 build/build_submission.py [--out build/out/submission.py] [--ext-suffix _b460010] [--src src]

The judge takes one Python file of at most 1,000,000 bytes, and its lint rejects exec / eval and a few words
(network-like names such as "requests", and "stream"). src/ is a normal multi-file tree, so this script merges it:

1. src/layer.py names its components with `alias = _load("alias", ROOT / "file.py")`. Each component's top-level
   names get the prefix `alias_` wherever the component uses them (tokenize-based: attribute accesses and string
   contents are untouched), and the layer's `alias.name` references become `alias_name`. Module-level imports of
   every component are hoisted, deduplicated and sorted into one header.
2. Every `cuda_source("x.cu")` call becomes the raw string literal of src/kernels/x.cu.
3. Comments are removed: Python comments (except the `# okc:` header line the judge reads) and C / C++ comments
   inside the CUDA strings. Inside each CUDA string, runs of blanks collapse to one and leading indentation is
   halved. Every line keeps its tokens, so the compiled code is the same; this only keeps the file under the limit.
4. src/parallel_load_inline.py is pasted right after the one `load_inline` import, so every extension compiles on
   its own thread and the cold import costs about the slowest single build.
5. Every load_inline extension name gets the suffix --ext-suffix (default: _b460010, the suffix the submitted file
   uses, so the names match it exactly). torch caches a compiled extension under its name, so a new suffix forces
   a fresh compile; the suffix has no other meaning.
6. Checks before writing: the judge's known lint words, the size limit, no class name defined twice (classes
   inside an `if AVAILABLE:` block are not prefixed, so two components with one class name would rebind it), and no
   static __shared__ variable that its kernel never names (storage reached only by an offset from another array,
   which the compiler may drop and place other data over).

build/compare_tokens.py proves the result has the same code tokens as submission/.
"""
import argparse
import ast
import io
import keyword
import pathlib
import re
import sys
import tokenize

REPO = pathlib.Path(__file__).resolve().parent.parent
SRC = REPO / "src"
LIMIT = 1_000_000
LOAD_RE = re.compile(r'^(\w+) = _load\("(\w+)", ROOT / "([\w.]+)"\)\s*$', re.M)
CUDA_CALL = re.compile(r'\bcuda_source\("([\w.]+)"\)')
ANCHOR = "from torch.utils.cpp_extension import load_inline\n"
SUBMITTED_SUFFIX = "_b460010"  # the extension name suffix in submission/submission.py


# ------------------------------------------------------------------------------------------- component merge
def top_level_names(src):
    """Names a component defines at module level (def, class, assignment targets), minus imported names."""
    names = set()
    for line in src.splitlines():
        m = re.match(r"^(?:def|class)\s+(\w+)", line) or re.match(r"^(\w+)\s*(?::[^=]*)?=", line)
        if m:
            names.add(m.group(1))
        m = re.match(r"^(\w+)\s*,\s*(\w+)(?:\s*,\s*(\w+))?\s*=", line)
        if m:
            names.update(n for n in m.groups() if n)
        m = re.match(r"^from\s+\S+\s+import\s+(.+)$", line)
        if m and "(" not in m.group(1):
            for part in m.group(1).split(","):
                part = part.strip()
                if " as " in part:
                    part = part.split(" as ")[1]
                names.discard(part)
    return {n for n in names if not keyword.iskeyword(n) and not n.startswith("__")}


def rename(src, names, prefix):
    """Prefix every NAME token in `names` that is not an attribute access."""
    out, prev = [], None
    for tok in tokenize.generate_tokens(io.StringIO(src).readline):
        if tok.type == tokenize.NAME and tok.string in names and not (prev and prev.string == "."):
            tok = tok._replace(string=prefix + tok.string)
        out.append(tok)
        if tok.type not in (tokenize.NL, tokenize.NEWLINE, tokenize.COMMENT, tokenize.INDENT, tokenize.DEDENT):
            prev = tok
    return tokenize.untokenize(out)


def inline_cuda(src, root):
    def lit(m):
        body = (root / "kernels" / m.group(1)).read_text()
        assert '"""' not in body and not body.endswith("\\"), m.group(1)
        return 'r"""' + body + '"""'
    return CUDA_CALL.sub(lit, src)


def strip_header(src):
    """Drop a component's leading comment lines and its module docstring (the layer keeps its own)."""
    lines = src.split("\n")
    while lines and (lines[0].startswith("#") or not lines[0].strip()):
        lines.pop(0)
    src = "\n".join(lines)
    tree = ast.parse(src)
    if tree.body and isinstance(tree.body[0], ast.Expr) and isinstance(getattr(tree.body[0], "value", None),
                                                                        ast.Constant):
        end = tree.body[0].end_lineno
        src = "\n".join(src.split("\n")[end:])
    return src


def split_imports(src):
    """Module-level import lines (multi-line parenthesized ones joined) and the rest of the component."""
    body, headers, pending = [], [], None
    for line in src.splitlines():
        if pending is not None:
            pending.append(line)
            if ")" in line:
                headers.append("\n".join(pending))
                pending = None
        elif re.match(r"^(import |from )", line):
            if "(" in line and ")" not in line:
                pending = [line]
            else:
                headers.append(line)
        else:
            body.append(line)
    return headers, "\n".join(body) + "\n"


def drop_comment(line):
    return re.sub(r"\s+#.*$", "", line) if line.lstrip().startswith(("import ", "from ")) else line


def merge(root):
    main_src = (root / "layer.py").read_text()
    loads = LOAD_RE.findall(main_src)
    assert loads, "no components found in src/layer.py"
    parts, headers, comp_names = [], set(), {}
    for var, name, fname in loads:
        assert var == name, (var, name)
        csrc = strip_header(inline_cuda((root / fname).read_text(), root))
        hdr, body = split_imports(csrc)
        headers.update(drop_comment(h) for h in hdr if "import cuda_source" not in h)
        names = top_level_names(body)
        comp_names[var] = names
        parts.append(f"# ---- component {var} (src/{fname}) ----\n" + rename(body, names, var + "_"))
    main_body = LOAD_RE.sub("", main_src)
    for var, _, _ in loads:
        main_body = re.sub(rf"\b{var}\.(\w+)",
                           lambda m, v=var: (v + "_" + m.group(1)) if m.group(1) in comp_names[v] else m.group(1),
                           main_body)
    # The development loader goes: its helper, ROOT, the sys.path line and importlib.
    main_body = re.sub(r"\ndef _load\(name, path\):\n(?:    .*\n)+", "\n", main_body)
    main_body = re.sub(r"^(ROOT = .*|sys\.path\.insert\(0, str\(ROOT.*|import importlib\.util)\n", "", main_body,
                       flags=re.M)
    first, rest = main_body.split("\n", 1)
    assert first.startswith("# okc: problem="), first
    return first + "\n" + "\n".join(sorted(headers)) + "\n\n" + "\n\n".join(parts) + "\n# ---- main ----\n" + rest


# ------------------------------------------------------------------------------------------- comment removal
def strip_c(src):
    """C / C++ comments out, every newline kept; string and character literals untouched."""
    out, i, n = [], 0, len(src)
    while i < n:
        c = src[i]
        if src.startswith("//", i):
            j = src.find("\n", i)
            j = n if j < 0 else j
            assert not src[i:j].rstrip(" \t").endswith("\\"), ("// comment ends in a backslash", src[i:j])
            i = j
            continue
        if src.startswith("/*", i):
            j = src.find("*/", i + 2)
            assert j >= 0, "unterminated block comment"
            lines = src[i:j + 2].count("\n")
            out.append("\n" * lines if lines else " ")
            i = j + 2
            continue
        if c in "\"'":
            j = i + 1
            while True:
                assert j < n and src[j] != "\n", ("unterminated literal", src[i:i + 60])
                if src[j] == "\\":
                    j += 2
                    continue
                if src[j] == c:
                    break
                j += 1
            out.append(src[i:j + 1])
            i = j + 1
            continue
        out.append(c)
        i += 1
    return "\n".join(line.rstrip(" \t") for line in "".join(out).split("\n"))


def squeeze_line(line):
    n = len(line) - len(line.lstrip(" "))
    body, out, i, quote = line[n:], [], 0, None
    while i < len(body):
        c = body[i]
        if quote:
            out.append(c)
            if c == "\\" and i + 1 < len(body):
                out.append(body[i + 1])
                i += 2
                continue
            if c == quote:
                quote = None
            i += 1
            continue
        if c in "\"'":
            quote = c
        if c in " \t" and out and out[-1] == " ":
            i += 1
            continue
        out.append(" " if c == "\t" else c)
        i += 1
    return " " * ((n + 1) // 2) + "".join(out).rstrip(" ")


def is_cuda_string(s):
    m = re.match(r"([rRbBuUfF]*)('''|\"\"\")", s)
    return bool(m and "r" in m.group(1).lower() and not set("bBfF") & set(m.group(1))
                and re.search(r"#include|#define|__global__", s))


def offsets(text):
    starts, acc = [0], 0
    for line in text.splitlines(keepends=True):
        acc += len(line)
        starts.append(acc)
    return starts


def compact(src):
    # CUDA strings: comments out, blanks squeezed (line structure kept).
    starts, edits = offsets(src), []
    for t in tokenize.generate_tokens(io.StringIO(src).readline):
        if t.type == tokenize.STRING and is_cuda_string(t.string):
            a = starts[t.start[0] - 1] + t.start[1]
            b = starts[t.end[0] - 1] + t.end[1]
            p = len(t.string) - len(t.string.lstrip("rRuU"))
            body = t.string[p + 3:-3]
            new = "\n".join(squeeze_line(x) for x in strip_c(body).split("\n"))
            edits.append((a, b, t.string[:p + 3] + new + t.string[-3:]))
    for a, b, new in reversed(edits):
        src = src[:a] + new + src[b:]
    # Python comments.
    lines = src.splitlines(keepends=True)
    drop, cuts = set(), {}
    for t in tokenize.generate_tokens(io.StringIO(src).readline):
        if t.type == tokenize.COMMENT and not t.string.startswith(("# okc:", "#!")):
            row, col = t.start
            if lines[row - 1][:col].strip() == "":
                drop.add(row - 1)
            else:
                cuts[row - 1] = col
    out = []
    for k, line in enumerate(lines):
        if k in drop:
            continue
        if k in cuts:
            line = line[:cuts[k]].rstrip(" \t") + ("\n" if line.endswith("\n") else "")
        out.append(line)
    return "".join(out)


# ------------------------------------------------------------------------------------------------ finishing
def parallel_shim(root):
    """src/parallel_load_inline.py without its docstring and its own load_inline import."""
    s = strip_header((root / "parallel_load_inline.py").read_text())
    s = s.replace(ANCHOR, "", 1).lstrip("\n")
    return s


def ext_names(code):
    names = []
    for node in ast.walk(ast.parse(code)):
        if isinstance(node, ast.Call) and getattr(node.func, "id", None) == "load_inline":
            for kw in node.keywords:
                if kw.arg == "name" and isinstance(kw.value, ast.Constant):
                    names.append(kw.value.value)
    return names


def finish(s, suffix, root):
    lines, seen = [], False
    for line in s.splitlines(keepends=True):
        if line == ANCHOR:
            if seen:
                continue
            seen = True
            lines.append(line)
            lines.append(parallel_shim(root))
            continue
        lines.append(line)
    assert seen
    s = "".join(lines)
    names = ext_names(s)
    assert names and len(names) == len(set(names)), names
    for name in sorted(names, key=len, reverse=True):
        s = re.sub(r"([\'\"])" + re.escape(name) + r"\1", lambda m: m.group(1) + name + suffix + m.group(1), s)
    return s


def lint(s):
    """Checks the judge's lint is known to make, so a bad build fails here instead of on the site."""
    assert "_load(" not in s and "importlib" not in s, "development loader left in the output"
    for pat in (r"\b(subprocess|multiprocessing|socket|urllib|requests|http\.client)\b",
                r"os\.(system|popen|fork|exec)", r"\b(exec|eval)\s*\(", r"__globals__|__code__\s*=|sys\.modules\["):
        m = re.search(pat, s)
        assert not m, ("lint word in the output", pat, s[max(0, m.start() - 60):m.end() + 20])
    # "stream" may appear only as an attribute (torch.cuda.stream) or inside strings (the input key data["stream"]).
    prev = None
    for t in tokenize.generate_tokens(io.StringIO(s).readline):
        assert not (t.type == tokenize.NAME and t.string == "stream" and not (prev and prev.string == ".")), t
        if t.type not in (tokenize.NL, tokenize.NEWLINE, tokenize.COMMENT, tokenize.INDENT, tokenize.DEDENT):
            prev = t
    assert chr(0x2013) not in s and chr(0x2014) not in s
    tree = ast.parse(s)
    # Classes are not prefixed when they are defined inside a block (if AVAILABLE: ...), so two components that
    # define the same class name would rebind it, and the first component would silently run the second one's class.
    seen = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            assert node.name not in seen, f"class {node.name} defined twice (lines {seen[node.name]} and {node.lineno})"
            seen[node.name] = node.lineno
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str) and re.search(r"__global__", node.value):
            unnamed_shared_arrays(node.value)


SHARED_DECL = re.compile(r"(?<!extern )__shared__\s+(?:__align__\(\d+\)\s+)?(?:(?:unsigned|signed|const|volatile)\s+)*"
                         r"[\w:]+\s+([^;]+);")


def unnamed_shared_arrays(cuda):
    """Refuse a static __shared__ variable that its function never names after declaring it.

    A weighted-sum kernel once declared `__shared__ uint64_t full[NS], empty[NS];` and reached the empty mbarriers
    only as &full + 8 * NS. `empty` was never named, so the compiler dropped it and placed the next shared array over
    those barriers; stores to that array corrupted live mbarriers, which faulted only when two 16K-token processes
    shared the GPU. Storage reached by an offset from another array must be part of that array (`full[2 * NS]`).
    Runs on the comment-free CUDA string; a function body ends at the next closing brace in column 0.
    """
    for m in SHARED_DECL.finditer(cuda):
        end = cuda.find("\n}", m.end())
        body = cuda[m.end():end if end > 0 else len(cuda)]
        for part in m.group(1).split(","):
            nm = re.match(r"\s*\*?\s*(\w+)", part)
            if nm and not re.search(r"\b" + re.escape(nm.group(1)) + r"\b", body):
                line = cuda[m.start():cuda.find("\n", m.start())].strip()
                raise AssertionError(f"__shared__ {nm.group(1)} is declared and never named in its function ({line}); "
                                     "storage reached only by an offset from another array may be dropped and aliased")


def build(suffix=SUBMITTED_SUFFIX, root=SRC):
    root = pathlib.Path(root)
    s = finish(compact(merge(root)), suffix, root)
    lint(s)
    assert len(s.encode()) < LIMIT, len(s.encode())
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(REPO / "build" / "out" / "submission.py"))
    ap.add_argument("--ext-suffix", default=SUBMITTED_SUFFIX)
    ap.add_argument("--src", default=str(SRC))
    args = ap.parse_args()
    s = build(args.ext_suffix, args.src)
    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(s)
    print(f"wrote {out}: {len(s.encode())} bytes, {len(ext_names(s))} extensions")


if __name__ == "__main__":
    sys.exit(main())
