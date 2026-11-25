#!/usr/bin/env python3
"""Generate `crates/rmda-core/src/analysis/ptm/remap_tables.rs` from upstream PTM.

The deformation gradient, the template index and the alloy ordering types
(design §14.2) need four more pieces of upstream's hand-maintained data:

* the Penrose pseudo-inverses of every template   <- ``ptm_deformation_gradient.h``
* the fundamental (point-group) point mappings     <- ``ptm_fundamental_mappings.h``
* the conventional mappings and template indices   <- ``ptm_fundamental_mappings.h``
* the order-12 hexagonal generators, upstream order <- ``ptm_quat.cpp``

Values are taken from a compiled C++ probe that includes the upstream headers
and prints every double as a C99 hex float, so the emitted decimals are the
doubles C++ computes, bit for bit; the upstream expression is kept as a comment
on every row so the table reads line for line against the source. The Python
evaluation of each expression is cross-checked against the probe as well.

Usage
-----
    python3 tools/gen_ptm_remap_tables.py --upstream <dir with ptm_*.h> \\
        --out crates/rmda-core/src/analysis/ptm/remap_tables.rs
"""

import argparse
import os
import re
import subprocess
import sys
import tempfile
import importlib.util

_spec = importlib.util.spec_from_file_location(
    "gen", os.path.join(os.path.dirname(os.path.abspath(__file__)), "gen_ptm_tables.py")
)
gen = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(gen)

MAX_POINTS = 17

# (name, number of points, penrose variants)
PENROSE = [
    ("sc", 7, [""]),
    ("fcc", 13, [""]),
    ("hcp", 13, ["", "_alt1"]),
    ("ico", 13, [""]),
    ("bcc", 15, [""]),
    ("dcub", 17, ["", "_alt1"]),
    ("dhex", 17, ["", "_alt1", "_alt2", "_alt3"]),
    ("graphene", 10, ["", "_alt1"]),
]

# (C symbol, rows, points)
MAPPINGS = [
    ("mapping_sc", 24, 7),
    ("mapping_fcc", 24, 13),
    ("mapping_bcc", 24, 15),
    ("mapping_ico", 60, 13),
    ("mapping_hcp", 6, 13),
    ("mapping_dcub", 12, 17),
    ("mapping_dhex", 3, 17),
    ("mapping_graphene", 6, 10),
    ("mapping_hcp_conventional", 12, 13),
    ("mapping_dcub_conventional", 24, 17),
    ("mapping_dhex_conventional", 12, 17),
    ("mapping_graphene_conventional", 12, 10),
]

TEMPLATE_INDICES = [
    ("template_indices_hcp", 12),
    ("template_indices_dcub", 24),
    ("template_indices_dhex", 12),
    ("template_indices_graphene", 12),
]


def parse_rows(body):
    return [m.group(1) for m in re.finditer(r"\{([^{}]*)\}", body)]


def parse_penrose(src, name, npts):
    body = gen.extract_braced(src, "const double penrose_%s[" % name)
    rows = [[p.strip() for p in r.split(",")] for r in parse_rows(body)]
    assert len(rows) == npts, (name, len(rows), npts)
    for r in rows:
        assert len(r) == 3, (name, r)
    return rows


def parse_generators(src):
    body = gen.extract_braced(src, "const double generator_hcp_conventional[")
    rows = [[p.strip() for p in r.split(",")] for r in parse_rows(body)]
    assert len(rows) == 12 and all(len(r) == 4 for r in rows)
    return rows


def probe(upstream, exprs):
    """Evaluate every expression in C++ (with the upstream headers in scope) and
    return the doubles as Python floats."""
    lines = ['#include <cmath>', '#include <cstdio>', 'using std::sqrt;', 'int main() {']
    for e in exprs:
        lines.append('  printf("%%a\\n", (double)(%s));' % e)
    lines.append('  return 0;\n}')
    with tempfile.TemporaryDirectory() as d:
        src = os.path.join(d, "p.cpp")
        exe = os.path.join(d, "p")
        open(src, "w").write("\n".join(lines))
        subprocess.check_call(["g++", "-O0", "-I", upstream, src, "-o", exe])
        out = subprocess.check_output([exe]).decode().split()
    assert len(out) == len(exprs)
    return [float.fromhex(v) for v in out]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--upstream", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    up = args.upstream
    dsrc = open(os.path.join(up, "ptm_deformation_gradient.h")).read()
    msrc = open(os.path.join(up, "ptm_fundamental_mappings.h")).read()
    qsrc = open(os.path.join(up, "ptm_quat.cpp")).read()

    penrose = {}
    exprs = []
    for name, npts, variants in PENROSE:
        for v in variants:
            rows = parse_penrose(dsrc, name + v, npts)
            penrose[name + v] = rows
            exprs.extend(x for r in rows for x in r)
    gens = parse_generators(qsrc)
    exprs.extend(x for r in gens for x in r)

    values = probe(up, exprs)
    for e, v in zip(exprs, values):
        py = gen.eval_cpp_expr(e)
        assert py == v or (py == 0.0 and v == 0.0), (e, py, v)
    it = iter(values)

    out = [
        "//! Penrose inverses, point mappings and conventional-template data for PTM's\n"
        "//! deformation gradient, template index and alloy ordering (design §14.2).\n"
        "//!\n"
        "//! GENERATED FILE -- do not edit by hand. Regenerate with:\n"
        "//!\n"
        "//! ```text\n"
        "//! python3 tools/gen_ptm_remap_tables.py --upstream <ptm sources> \\\n"
        "//!     --out crates/rmda-core/src/analysis/ptm/remap_tables.rs\n"
        "//! ```\n"
        "//!\n"
        "//! Sources: `ptm_deformation_gradient.h`, `ptm_fundamental_mappings.h` and\n"
        "//! `ptm_quat.cpp` of OVITO's bundled PTM. Every double is the value a C++\n"
        "//! probe computes from the upstream expression (kept as a comment), so the\n"
        "//! table is bit-identical to upstream. Rows are zero-padded to `MAX_POINTS`.\n"
        "#![allow(clippy::approx_constant)]\n"
        "#![allow(clippy::excessive_precision)]\n"
        "#![allow(clippy::unreadable_literal)]\n"
    ]

    for name, npts, variants in PENROSE:
        out.append(
            "\n/// Penrose inverse of the %s template(s): variant `k` is upstream's\n"
            "/// `penrose_%s%s`.\n"
            "pub static PENROSE_%s: [[[f64; 3]; %d]; %d] = [\n"
            % (name.upper(), name, " (then `_alt1`..)" if len(variants) > 1 else "",
               name.upper(), MAX_POINTS, len(variants))
        )
        for v in variants:
            rows = penrose[name + v]
            out.append("    [\n")
            for r in rows:
                vals = [next(it) for _ in r]
                out.append("        [%s], // %s\n" % (", ".join(repr(x) for x in vals),
                                                     ", ".join(r)))
            for _ in range(MAX_POINTS - npts):
                out.append("        [0.0, 0.0, 0.0],\n")
            out.append("    ],\n")
        out.append("];\n")

    out.append(
        "\n/// Upstream's order-12 hexagonal generators (`generator_hcp_conventional`),\n"
        "/// in upstream's order: `mapping_*_conventional[i]` and\n"
        "/// `template_indices_*[i]` are indexed by position in this table.\n"
        "pub static GENERATOR_HCP_CONVENTIONAL: [[f64; 4]; 12] = [\n"
    )
    for r in gens:
        vals = [next(it) for _ in r]
        out.append("    [%s], // %s\n" % (", ".join(repr(x) for x in vals), ", ".join(r)))
    out.append("];\n")

    for sym, nrows, npts in MAPPINGS:
        body = gen.extract_braced(msrc, "const int8_t %s[" % sym)
        rows = [[int(x) for x in r.split(",") if x.strip()] for r in parse_rows(body)]
        assert len(rows) == nrows and all(len(r) == npts for r in rows), sym
        out.append(
            "\n/// Upstream `%s` (%d points per row).\n"
            "pub static %s: [[i8; %d]; %d] = [\n" % (sym, npts, sym.upper(), MAX_POINTS, nrows)
        )
        for r in rows:
            out.append("    [%s],\n" % ", ".join(str(x) for x in r + [0] * (MAX_POINTS - npts)))
        out.append("];\n")

    for sym, n in TEMPLATE_INDICES:
        m = re.search(r"const int8_t %s\[[^\]]*\]\s*=\s*\{([^}]*)\}" % sym, msrc)
        vals = [int(x) for x in m.group(1).split(",") if x.strip()]
        assert len(vals) == n, sym
        out.append(
            "\n/// Upstream `%s`.\npub static %s: [u8; %d] = [%s];\n"
            % (sym, sym.upper(), n, ", ".join(str(v) for v in vals))
        )

    with open(args.out, "w") as f:
        f.write("".join(out))
    # Committed formatted, so `cargo fmt --check` passes on a regenerated file.
    subprocess.check_call(["rustfmt", "--edition", "2021", args.out])
    print("wrote %s" % args.out)


if __name__ == "__main__":
    sys.exit(main())
