#!/usr/bin/env python3
"""Generate `crates/rmda-core/src/analysis/ptm/quat_tables.rs` from `ptm_quat.cpp`.

Provenance
----------
The orientation fundamental zones are described by three tables of symmetry
generators in upstream ``ptm_quat.cpp``::

    generator_cubic[24]              SC, FCC, BCC (and cubic diamond)
    generator_icosahedral[60]        ICO
    generator_hcp[6]                 HCP
    generator_hcp_conventional[12]   HCP orientation output, hexagonal diamond, graphene

``rotate_quaternion_into_fundamental_zone`` picks whichever generator has the
largest ``|q . g|`` and rotates ``q`` by it, so the table size is the order of
the point group and nothing else. Transcribing 90 quaternions by hand goes
wrong quietly, so they are generated from the C++ instead.

These four tables are emitted. ``HCP_CONVENTIONAL`` holds upstream's
``generator_hcp_conventional`` rows in a fixed order: the six ``generator_hcp``
rows, then each of them times the 60 degree rotation about c. The order decides
ties in the fundamental-zone reduction, so ``--check`` asserts the rule as well
as the values. Upstream's ``diamond_cubic`` and ``diamond_hexagonal`` subsets
are not needed: the port reduces those structures over ``CUBIC`` and
``HCP_CONVENTIONAL``. Every table is ``#[rustfmt::skip]``, so a row stays on
one line with its upstream expressions.

The entries are written in upstream as exact surds (``sqrt(2)/2``,
``(1+sqrt(5))/4``, ``sqrt(25-10*sqrt(5))/10``, ...). Following the same
convention as ``gen_ptm_tables.py``, they are emitted as exact *decimals*,
because ``f64::sqrt`` is not a ``const fn`` and a Rust ``const`` cannot call
it, with the upstream expression kept as a trailing comment on every row. The
``--check`` mode proves the decimals are right: it evaluates the C++ source
expressions with the host's own ``sqrt`` and asserts bit-identical output
against the values parsed back out of the generated file.

Usage
-----
    python3 tools/gen_ptm_quat_tables.py \\
        --source <upstream>/ptm_quat.cpp \\
        --out crates/rmda-core/src/analysis/ptm/quat_tables.rs

    python3 tools/gen_ptm_quat_tables.py \\
        --source <upstream>/ptm_quat.cpp \\
        --table crates/rmda-core/src/analysis/ptm/quat_tables.rs --check
"""

import argparse
import math
import re
import sys

# The point groups the ported structures need, in file order.
WANTED = [
    ("generator_cubic", 24),
    ("generator_icosahedral", 60),
    ("generator_hcp", 6),
    ("generator_hcp_conventional", 12),
]

RUST_NAMES = {
    "generator_cubic": "CUBIC",
    "generator_icosahedral": "ICOSAHEDRAL",
    "generator_hcp": "HCP",
    "generator_hcp_conventional": "HCP_CONVENTIONAL",
}

DOCS = {
    "generator_cubic": (
        "Order-24 generators of the cubic point group `O`, shared by SC, FCC and\n"
        "/// BCC.",
        "SC, FCC, BCC, cubic diamond",
    ),
    "generator_icosahedral": (
        "Order-60 generators of the icosahedral point group `I`, for ICO.",
        "ICO",
    ),
    "generator_hcp": (
        "Six generators of the `D3d` point group, for HCP; the first half of\n"
        "/// [`HCP_CONVENTIONAL`], which the orientation output reduces over.",
        "HCP",
    ),
    "generator_hcp_conventional": (
        "The order-12 hexagonal group: the six [`HCP`] generators, then each\n"
        "/// times the 60° rotation about c (upstream's `hcp_conventional` set). PTM's\n"
        "/// two HCP templates differ by that 60° turn, which with a c/2 shift maps the\n"
        "/// crystal onto itself; reducing over all twelve gives OVITO's orientation\n"
        "/// (its `ptm_remap_template` to the identity), reducing over six leaves\n"
        "/// half of the atoms 60° away from it.",
        "HCP (and dHCP), hexagonal diamond and graphene orientation output\n"
        "/// and comparison",
    ),
}


def to_python_expr(expr):
    """Rewrite a C++ surd expression into an evaluable Python one.

    Only `+ - * /`, parentheses, unary minus and `sqrt` appear, which is a
    Python subset once `sqrt` is bound, so the host's own `float` arithmetic
    reproduces the correctly-rounded `sqrt` the C++ compiler would use.
    """
    e = expr.replace("sqrt", "sqrt")
    e = re.sub(r"(?<![\w.])(\d+)(?![\w.])", r"\1.0", e)
    return e


def evaluate(expr):
    return float(
        eval(to_python_expr(expr), {"__builtins__": {}}, {"sqrt": math.sqrt})
    )


def extract(src, name, count):
    """Return `count` rows of 4 expression strings from `const double name[N][4]`."""
    m = re.search(
        r"const\s+double\s+" + re.escape(name) + r"\s*\[\s*(\d+)\s*\]\s*\[\s*4\s*\]\s*=\s*\{",
        src,
    )
    if not m:
        raise SystemExit(f"could not find `{name}` in the source")
    if int(m.group(1)) != count:
        raise SystemExit(f"{name}: expected [{count}][4], source says [{m.group(1)}][4]")
    body = src[m.end() :]
    rows = []
    for rm in re.finditer(r"\{([^{}]*)\}", body):
        cells = [c.strip() for c in rm.group(1).split(",")]
        if len(cells) != 4:
            raise SystemExit(f"{name}: row has {len(cells)} cells, expected 4")
        rows.append(cells)
        if len(rows) == count:
            return rows
    raise SystemExit(f"{name}: found only {len(rows)} rows, expected {count}")


def qmul(a, b):
    """Hamilton product `a * b` of (w, x, y, z) quaternions."""
    return (
        a[0] * b[0] - a[1] * b[1] - a[2] * b[2] - a[3] * b[3],
        a[0] * b[1] + a[1] * b[0] + a[2] * b[3] - a[3] * b[2],
        a[0] * b[2] - a[1] * b[3] + a[2] * b[0] + a[3] * b[1],
        a[0] * b[3] + a[1] * b[2] - a[2] * b[1] + a[3] * b[0],
    )


# The 60 degree rotation about c: (cos 30, 0, 0, sin 30).
R60 = (math.sqrt(3) / 2, 0.0, 0.0, 0.5)


def conventional_order(src):
    """Upstream's `generator_hcp_conventional` rows, ordered as the port
    uses them: the `generator_hcp` rows, then each times the 60 degree turn.
    Each target is matched to the one upstream row within 1e-12, so the
    emitted expressions are upstream's own."""
    conv = extract(src, "generator_hcp_conventional", 12)
    hcp = [[evaluate(e) for e in row] for row in extract(src, "generator_hcp", 6)]
    targets = hcp + [list(qmul(g, R60)) for g in hcp]
    out = []
    for t in targets:
        hits = [
            row for row in conv
            if all(abs(evaluate(e) - v) < 1e-12 for e, v in zip(row, t))
        ]
        if len(hits) != 1:
            raise SystemExit(f"hcp_conventional: {len(hits)} upstream rows match {t}")
        out.append(hits[0])
    if len({tuple(r) for r in out}) != 12:
        raise SystemExit("hcp_conventional: the ordered rows are not all distinct")
    return out


def rows_for(src, name, count):
    if name == "generator_hcp_conventional":
        return conventional_order(src)
    return extract(src, name, count)


def fmt(value):
    """Render a float so it round-trips exactly, and readably."""
    return repr(value)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", help="path to upstream ptm_quat.cpp")
    ap.add_argument("--out", help="path to the generated Rust file")
    ap.add_argument("--table", help="generated file to verify (with --check)")
    ap.add_argument("--check", action="store_true", help="verify instead of generate")
    args = ap.parse_args()

    if args.check:
        return check(args.source, args.table)
    if not (args.source and args.out):
        raise SystemExit("generation needs --source and --out")

    src = open(args.source).read()
    out = []
    out.append("//! Symmetry generators for the PTM orientation fundamental zones.")
    out.append("//!")
    out.append("//! GENERATED FILE -- do not edit by hand. Regenerate with:")
    out.append("//!")
    out.append("//! ```text")
    out.append("//! python3 tools/gen_ptm_quat_tables.py \\")
    out.append(f"//!     --source <upstream>/ptm_quat.cpp \\")
    out.append(f"//!     --out {args.out}")
    out.append("//! ```")
    out.append("//!")
    out.append("//! `rotate_quaternion_into_fundamental_zone` selects the generator")
    out.append("//! with the largest `|q . g|` and rotates `q` by it, so each table's")
    out.append("//! length is exactly the order of the corresponding point group.")
    out.append("//!")
    out.append("//! Entries are exact decimals rather than the surd expressions upstream")
    out.append("//! writes, because `f64::sqrt` is not `const` and a Rust `const` cannot")
    out.append("//! call it. The upstream expression is kept as a trailing comment on")
    out.append("//! every element, and `tools/gen_ptm_quat_tables.py --check` asserts both")
    out.append("//! that the comment round-trips and that the decimal is bit-identical to")
    out.append("//! what C++ computes.")
    out.append("//!")
    out.append("//! DCUB reduces over [`CUBIC`]; DHEX and GRAPHENE over the order-12")
    out.append("//! conventional set [`HCP_CONVENTIONAL`].")
    out.append("#![allow(clippy::approx_constant)]")
    out.append("")

    for name, count in WANTED:
        doc, used_by = DOCS[name]
        rust = RUST_NAMES[name]
        out.append(f"/// {doc}")
        out.append(f"///")
        out.append(f"/// Used by: {used_by}.")
        out.append("#[rustfmt::skip]")
        out.append(
            f"pub const {rust}: [[f64; 4]; {count}] = ["
        )
        for row in rows_for(src, name, count):
            # One trailing comment per row, holding all four upstream
            # expressions, so each generated row maps 1:1 onto the C++ line it
            # came from. (A per-element `//` comment cannot work: the comment
            # would swallow the following separator.)
            vals = ", ".join(fmt(evaluate(c)) for c in row)
            exprs = ", ".join(row)
            out.append(f"    [ {vals} ],  // {{ {exprs} }}")
        out.append("];")
        out.append("")

    text = "\n".join(out).rstrip("\n") + "\n"
    open(args.out, "w").write(text)
    total = sum(c for _, c in WANTED)
    print(f"wrote {args.out}: {len(WANTED)} tables, {total} generators")
    return 0


def check(source, table):
    """Assert every generated decimal is bit-identical to the C++ expression."""
    if not (source and table):
        raise SystemExit("--check needs --source and --table")
    src = open(source).read()
    gen = open(table).read()

    rust_names = {v: k for k, v in RUST_NAMES.items()}
    ok = True
    checked = 0
    for m in re.finditer(
        r"pub const (\w+): \[\[f64; 4\]; (\d+)\] = \[(.*?)\n\];", gen, re.S
    ):
        rust, count, body = m.group(1), int(m.group(2)), m.group(3)
        name = rust_names.get(rust)
        if name is None:
            print(f"unexpected table `{rust}` in {table}", file=sys.stderr)
            ok = False
            continue
        expected = rows_for(src, name, count)
        rows = re.findall(r"^\s*\[.*$", body, re.M)
        if len(rows) != count:
            print(f"{rust}: parsed {len(rows)} rows, expected {count}", file=sys.stderr)
            ok = False
            continue
        for i, row in enumerate(rows):
            # Layout is `[ v0, v1, v2, v3 ],  // { e0, e1, e2, e3 }`
            bracket = re.search(r"\[([^\]]*)\]", row)
            if not bracket or "//" not in row:
                print(f"{rust}[{i}]: cannot split row/comment", file=sys.stderr)
                ok = False
                continue
            values = [v.strip() for v in bracket.group(1).split(",")]
            comment = row.split("//", 1)[1].strip()
            if comment.startswith("{") and comment.endswith("}"):
                comment = comment[1:-1]
            comments = [c.strip() for c in comment.split(",")]
            if len(values) != 4 or len(comments) != 4:
                print(
                    f"{rust}[{i}]: parsed {len(values)} values / "
                    f"{len(comments)} comments, expected 4 each",
                    file=sys.stderr,
                )
                ok = False
                continue
            for j in range(4):
                literal, comment_j = values[j], comments[j]
                upstream = expected[i][j]
                if comment_j != upstream:
                    print(
                        f"{rust}[{i}][{j}]: comment `{comment_j}` != upstream "
                        f"`{upstream}`",
                        file=sys.stderr,
                    )
                    ok = False
                want = evaluate(upstream)
                got = float(literal)
                checked += 1
                if want.hex() != got.hex():
                    print(
                        f"{rust}[{i}][{j}]: {got.hex()} != C++ {want.hex()} "
                        f"({literal} vs `{upstream}`)",
                        file=sys.stderr,
                    )
                    ok = False
    if not ok:
        print("quat table check FAILED", file=sys.stderr)
        return 1
    print(f"quat table check OK: {checked} values bit-identical to {source}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
