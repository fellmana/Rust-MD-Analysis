#!/usr/bin/env python3
"""Verify `analysis/ptm/tables.rs` against the upstream C++ it was generated from.

Two independent checks, because they fail in different ways:

1. **Textual** -- every coordinate expression emitted into `tables.rs` must
   appear verbatim (modulo the `7.` -> `7.0` numeric rewrite) in the upstream
   template header. This catches transcription and translation bugs, which are
   the kind that produce a subtly wrong table that still "looks" plausible.

2. **Numerical** -- compiles a C++ probe that evaluates the upstream expressions
   and prints each value as a C99 hex float, and does the same from Rust, then
   compares the bit patterns. A Rust `f64` expression and the equivalent C++
   expression must agree bit-for-bit because both use correctly-rounded `sqrt`;
   this check proves that for the actual generated table rather than assuming it.

Usage
-----
    python3 tools/check_ptm_tables.py \\
        --templates <OVITO PTM copy>/ptm_templates.h \\
        --graphs <upstream>/ptm_graph_data.cpp \\
        --table crates/rmda-core/src/analysis/ptm/tables.rs
"""

import argparse
import re
import subprocess
import sys
import tempfile
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import importlib.util

_spec = importlib.util.spec_from_file_location(
    "gen", os.path.join(os.path.dirname(os.path.abspath(__file__)), "gen_ptm_tables.py")
)
gen = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(gen)

CPP_PROBE = r"""
#include <cstdio>
#include <cmath>
%(decl)s
int main() {
%(body)s
        return 0;
}
"""


def parse_rust_templates(table, name, nvar):
    """Brace-balanced parse of a generated `*_TEMPLATES` static.

    A regex is not usable here: the generated rows are nested two deep and the
    block width varies with the structure, so walk the bracket depth instead.
    Each row comes back as `(literals, upstream_comment)`.
    """
    m = re.search(
        r"pub static %s_TEMPLATES: \[\[\[f64; 3\]; 17\]; %d\] = \[" % (name, nvar),
        table,
    )
    if not m:
        raise AssertionError("could not locate %s_TEMPLATES" % name)
    # Find the matching close of the outer array.
    start = table.index("[", m.end() - 1)
    depth = 0
    k = start
    while True:
        if table[k] == "[":
            depth += 1
        elif table[k] == "]":
            depth -= 1
            if depth == 0:
                body = table[start + 1 : k]
                break
        k += 1

    variants = []
    for variant in gen.split_top_level(body, "[", "]"):
        rows = []
        for line in variant.splitlines():
            line = line.strip()
            if not line or not line.startswith("["):
                continue
            lit, _, comment = line.partition("//")
            nums = [e.strip() for e in gen.split_top_level(lit.strip(), "[", "]")[0].split(",")]
            rows.append((nums, comment.strip()))
        variants.append(rows)
    assert len(variants) == nvar, (name, len(variants), nvar)
    return variants


def check_textual(tsrc, table):
    """Each row's comment must be an exact round-trip of the upstream expression.

    Upstream declares `[n][PTM_MAX_POINTS][3]` and leaves the tail zero-filled, so
    only the first `nnbrs + 1` rows carry an expression. The generated table
    spells that padding out as 17 rows, so the tail is compared against the
    expected zero row instead of against upstream.
    """
    total = 0
    for name, nvar, nnbrs, _nf, _md, _ng in gen.STRUCTURES:
        upstream = gen.parse_templates(tsrc, name, nvar)
        emitted = parse_rust_templates(table, name.upper(), nvar)
        assert len(upstream) == len(emitted), (name, len(upstream), len(emitted))
        for vi, (u, e) in enumerate(zip(upstream, emitted)):
            # Every emitted variant is padded out to PTM_MAX_POINTS rows.
            assert len(e) == 17, (name, vi, len(e))
            assert len(u) == nnbrs + 1, (name, vi, len(u), nnbrs + 1)
            for pi, (ut, (lits, comment)) in enumerate(zip(u, e)):
                want = ", ".join(x.strip() for x in ut)
                if comment != want:
                    print(
                        "%s[%d][%d]: comment %r, upstream %r"
                        % (name, vi, pi, comment, want)
                    )
                    return False
                if len(lits) != 3:
                    print("%s[%d][%d]: expected 3 literals, got %r" % (name, vi, pi, lits))
                    return False
                total += len(ut)
            # The zero padding upstream leaves implicit, spelled out here. The
            # comment still round-trips the upstream literal text.
            for pi, (lits, comment) in enumerate(e[nnbrs + 1 :], start=nnbrs + 1):
                if lits != ["0.0", "0.0", "0.0"] or comment != "0, 0, 0":
                    print(
                        "%s[%d][%d]: padding row is %r / %r, want zeros and comment %r"
                        % (name, vi, pi, lits, comment, "0, 0, 0")
                    )
                    return False
    print("textual: %d coordinates round-trip to the upstream expression" % total)
    return True


def build_probe(tsrc, structures):
    """C++ probe printing every template coordinate as a hex float.

    Only the first `num_nbrs + 1` points are printed: upstream declares the
    arrays `[n][PTM_MAX_POINTS][3]` and leaves the tail zero-filled, and the
    generated Rust table relies on the same implicit zero padding.
    """
    decls, prints = [], []
    for name, nvar, nnbrs, _nf, _md, _ng in structures:
        variants = gen.parse_templates(tsrc, name, nvar)
        for vi, coords in enumerate(variants):
            nm = "%s_%d" % (name, vi)
            decls.append("static const double t_%s[17][3] = {" % nm)
            decls.append(",".join("{%s}" % ",".join(c) for c in coords))
            decls.append("};")
        for vi in range(nvar):
            nm = "%s_%d" % (name, vi)
            for p in range(nnbrs + 1):
                for c in range(3):
                    prints.append('    printf("%%a\\n", t_%s[%d][%d]);' % (nm, p, c))
    src = CPP_PROBE % {"decl": "\n".join(decls), "body": "\n".join(prints)}
    return src


def check_numerical(tsrc, structures, table):
    src = build_probe(tsrc, structures)
    with tempfile.TemporaryDirectory() as td:
        probe = os.path.join(td, "probe.cpp")
        binp = os.path.join(td, "probe")
        with open(probe, "w") as f:
            f.write(src)
        r = subprocess.run(
            ["g++", "-O0", "-std=c++11", "-o", binp, probe],
            capture_output=True,
            text=True,
        )
        if r.returncode != 0:
            print("probe failed to compile:\n" + r.stderr)
            return False
        cpp = subprocess.run([binp], capture_output=True, text=True).stdout.split()

    # Pull the same values back out of the generated Rust table.
    rust_lits = []
    for name, nvar, nnbrs, _nf, _md, _ng in structures:
        for rows in parse_rust_templates(table, name.upper(), nvar):
            # 17 rows: the real points first, then the zero padding upstream
            # also has. Only the first `nnbrs + 1` carry data.
            assert len(rows) == 17, (name, len(rows))
            assert all(
                lits == ["0.0", "0.0", "0.0"] for lits, _ in rows[nnbrs + 1 :]
            ), name
            for lits, _comment in rows[: nnbrs + 1]:
                for lit in lits:
                    rust_lits.append(lit)

    if len(rust_lits) != len(cpp):
        print("count mismatch: rust=%d cpp=%d" % (len(rust_lits), len(cpp)))
        return False

    # Compare raw bits rather than the printed form: C's `%a` and Python's
    # `float.hex()` spell the mantissa differently (`0x0p+0` vs `0x0.0p+0`) but
    # agree bitwise.
    import struct

    def bits(x):
        return struct.pack(">d", x).hex()

    bad = 0
    for i, (lit, want) in enumerate(zip(rust_lits, cpp)):
        got = bits(float(lit))
        want = bits(float.fromhex(want))
        if got != want:
            bad += 1
            if bad <= 10:
                print(
                    "  value %d: %-24s rust=%s cpp=%s"
                    % (i, lit, got, float.fromhex(want).hex())
                )
    if bad:
        print("numerical: %d/%d values differ" % (bad, len(cpp)))
        return False
    print("numerical: all %d literals bit-identical to C++" % len(cpp))
    return True


def check_graphs(gsrc, table):
    """Facet lists and automorphism windows must match upstream exactly."""
    ok = True
    for name, _nvar, _n, nfacets, _md, ngraphs in gen.STRUCTURES:
        graphs = gen.parse_graphs(gsrc, "graphs_%s" % name, ngraphs, nfacets)
        m = re.search(
            r"pub static %s_GRAPHS: \[HullGraph; %d\] = \[(.*?)\n\];"
            % (name.upper(), ngraphs),
            table,
            re.S,
        )
        if not m:
            print("could not locate %s_GRAPHS" % name.upper())
            return False
        found = re.findall(
            r"automorphism_index: (\d+),\s*\n\s*num_automorphisms: (\d+),\s*\n"
            r"\s*num_facets: (\d+),\s*\n\s*facets: \[(.*?)\],\n",
            m.group(1),
            re.S,
        )
        if len(found) != ngraphs:
            print(
                "%s: parsed %d graph entries, expected %d"
                % (name.upper(), len(found), ngraphs)
            )
            ok = False
            continue
        for gi, (ai, na, nf, facets) in enumerate(found):
            want_ai, want_na, want_facets = graphs[gi]
            got_facets = [
                tuple(int(x) for x in f)
                for f in re.findall(r"\[(-?\d+), (-?\d+), (-?\d+)\]", facets)
            ]
            # Every facet slot is emitted, so the real facets must be followed
            # by zero padding out to PTM_MAX_FACETS.
            padded = want_facets + [(0, 0, 0)] * (gen.MAX_FACETS - nfacets)
            if int(nf) != nfacets:
                print(
                    "%s[%d]: num_facets %s, want %d"
                    % (name.upper(), gi, nf, nfacets)
                )
                ok = False
            elif (int(ai), int(na)) != (want_ai, want_na):
                print("%s[%d]: automorphism header mismatch" % (name.upper(), gi))
                ok = False
            elif got_facets != padded:
                print(
                    "%s[%d]: facet list mismatch (%d entries, want %d)"
                    % (name.upper(), gi, len(got_facets), gen.MAX_FACETS)
                )
                ok = False
    if ok:
        print(
            "graphs: %d hull graphs match upstream"
            % sum(s[5] for s in gen.STRUCTURES)
        )
    return ok


def check_automorphisms(gsrc, table):
    upstream = gen.parse_int_matrix(gsrc, "int8_t automorphisms[", 65, 17)
    m = re.search(
        r"pub static AUTOMORPHISMS: \[\[i8; 17\]; 65\] = \[(.*?)\n\];", table, re.S
    )
    if not m:
        print("could not locate AUTOMORPHISMS")
        return False
    found = [
        [int(x) for x in re.findall(r"-?\d+", row)]
        for row in re.findall(r"\[(.*?)\],", m.group(1), re.S)
    ]
    if found != upstream:
        print("automorphisms mismatch (%d rows vs %d)" % (len(found), len(upstream)))
        return False
    print("automorphisms: 65 rows match upstream")
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--templates", required=True)
    ap.add_argument("--graphs", required=True)
    ap.add_argument("--table", required=True)
    args = ap.parse_args()

    tsrc = open(args.templates).read()
    gsrc = open(args.graphs).read()
    table = open(args.table).read()

    results = [
        check_textual(tsrc, table),
        check_numerical(tsrc, gen.STRUCTURES, table),
        check_graphs(gsrc, table),
        check_automorphisms(gsrc, table),
    ]
    if all(results):
        print("OK: generated table is faithful to upstream")
        return 0
    print("FAILED")
    return 1


if __name__ == "__main__":
    sys.exit(main())
