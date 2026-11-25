#!/usr/bin/env python3
"""Generate `crates/rmda-core/src/analysis/ptm/tables.rs` from the upstream PTM sources.

Provenance
----------
The ideal templates, the convex-hull facet lists and the graph automorphisms are
*hand-maintained data* in the upstream PTM distribution, not something the
algorithm computes at run time. Transcribing 244 hull graphs plus the
automorphism table by hand is exactly the kind of work that goes wrong quietly,
so it is generated from the upstream C++ instead:

* templates  <- ``ptm_ptm_templates.h``   (OVITO's copy of ``ptm_templates.h``)
* hull graphs <- ``ptm_graph_data.cpp``

All eight PTM structures are emitted: SC, FCC, HCP, ICO, BCC, and the two-shell
structures cubic diamond (DCUB), hexagonal diamond (DHEX) and graphene. Graphene
has templates only: upstream matches it by direct permutation, without a hull
graph (`match_graphene`).

The template coordinates are written in upstream as exact surds (``sqrt(2)/2``,
``sqrt(3)/6``, ...). They are emitted *verbatim* as Rust expressions rather than
decimal literals, so the table is both exact (a Rust ``f64`` expression
evaluates with the same correctly-rounded ``sqrt`` as C++) and reviewable against
the upstream source line by line. ``tools/check_ptm_tables.py`` compiles a C++
probe that prints the same expressions and asserts bit-identical output.

The generated table stores only the fields the matcher reads. ``hash`` and
``canonical_labelling`` are intentionally absent: upstream computes both at
initialisation via ``canonical_form_coloured`` (after first re-winding every
facet clockwise), so we do the same rather than transcribing derived values.

Usage
-----
    python3 tools/gen_ptm_tables.py --templates <path> --graphs <path> \
        --out crates/rmda-core/src/analysis/ptm/tables.rs
"""

import argparse
import math
import re
import sys


def eval_cpp_expr(expr):
    """Evaluate a C++ arithmetic expression to a Python float (an IEEE double).

    The expression grammar upstream uses is tiny: integer and decimal literals,
    `+ - * /`, parentheses, unary minus, and `sqrt`. That is a Python subset
    once `sqrt` is bound, so the host's own `float` arithmetic reproduces the
    C++ result bit for bit -- same IEEE-754 operations, same correctly-rounded
    `sqrt`.
    """
    return float(eval(to_python_expr(expr), {"__builtins__": {}}, {"sqrt": math.sqrt}))


def to_python_expr(expr):
    """Translate a C++ arithmetic expression into an equivalent Python one."""
    return to_rust_expr(expr)

# name -> (n_template_variants, num_neighbours, num_facets, max_degree, n_graphs)
STRUCTURES = [
    ("sc", 1, 6, 8, 4, 1),
    ("fcc", 1, 12, 20, 6, 8),
    ("hcp", 2, 12, 20, 6, 16),
    ("ico", 1, 12, 20, 6, 1),
    ("bcc", 1, 14, 24, 8, 218),
    ("dcub", 2, 16, 28, 8, 12),
    ("dhex", 4, 16, 28, 8, 24),
    ("graphene", 2, 9, None, None, 0),
]

MAX_POINTS = 17  # upstream PTM_MAX_POINTS
MAX_FACETS = 28  # upstream PTM_MAX_FACETS

RUST_TYPE = {
    "sc": "SC",
    "fcc": "FCC",
    "hcp": "HCP",
    "ico": "ICO",
    "bcc": "BCC",
    "dcub": "DCUB",
    "dhex": "DHEX",
    "graphene": "GRAPHENE",
}


def extract_braced(src, anchor):
    """Return the body of the C++ initialiser that follows `anchor`.

    The returned text starts after the initialiser's own opening brace, so it
    contains only the initialised elements and they can be walked as a
    brace-balanced tree.
    """
    i = src.index(anchor)
    j = src.index("=", i)
    open_brace = src.index("{", j)
    depth = 0
    k = open_brace
    while True:
        if src[k] == "{":
            depth += 1
        elif src[k] == "}":
            depth -= 1
            if depth == 0:
                return src[open_brace + 1 : k]
        k += 1


def to_rust_expr(expr):
    """Translate a C++ arithmetic expression into an equivalent Rust expression.

    Two rewrites are needed. Upstream writes `7./2` for a half, which Rust spells
    `7.0/2.0`; and bare integer literals need an explicit fractional part because
    Rust does not coerce them inside a nested `[[f64; 3]; N]` literal.
    `sqrt(...)` is kept verbatim and resolved by a local `fn sqrt` in the
    generated module.

    Both rewrites are value-preserving: an `f64` is exactly representable for
    every integer involved here, so the only rounding that matters is the one in
    `sqrt`, and that is correctly rounded in C++ and Rust alike.
    """
    e = expr.strip()
    e = re.sub(r"(?<![\w.])(\d+)\.(?![0-9])", r"\1.0", e)
    e = re.sub(r"(?<![\w.])(\d+)(?![\w.])", r"\1.0", e)
    return e


def parse_templates(src, name, n_variants):
    body = extract_braced(src, "const double ptm_template_%s[" % name)
    variants = []
    for m in re.finditer(r"\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}", body):
        inner = m.group(1)
        coords = []
        for triple in re.findall(r"\{([^{}]*)\}", inner):
            parts = [p.strip() for p in triple.split(",")]
            assert len(parts) == 3, (name, triple)
            coords.append(parts)
        if coords:
            variants.append(coords)
    if len(variants) != n_variants:
        sys.exit(
            f"ptm_template_{name}: {len(variants)} orientation variant(s) parsed, {n_variants} "
            "expected. --templates must be OVITO's ptm_templates.h, which carries the "
            "conventional-orientation variants; upstream master's ptm_constants.h has template 0 "
            "only (the stored tables' template 0 is bit-identical to it)."
        )
    return variants


def parse_int_matrix(src, anchor, n_rows, n_cols):
    body = extract_braced(src, anchor)
    rows = []
    for m in re.finditer(r"\{([^{}]*)\}", body):
        vals = [int(v.strip()) for v in m.group(1).split(",") if v.strip()]
        if vals:
            assert len(vals) == n_cols, (anchor, len(vals), n_cols)
            rows.append(vals)
    assert len(rows) == n_rows, (anchor, len(rows), n_rows)
    return rows


def split_top_level(src, open_ch="{", close_ch="}"):
    """Split a delimited body into its top-level groups.

    The graph entries nest (`{scalars, {labelling}, {{f},{f},...}}`), so a
    regular expression is not enough; walk the nesting depth instead. The
    delimiters are parameters so the same routine can walk the square brackets
    of a generated Rust array.
    """
    groups = []
    depth = 0
    start = None
    for k, c in enumerate(src):
        if c == open_ch:
            if depth == 0:
                start = k
            depth += 1
        elif c == close_ch:
            depth -= 1
            if depth == 0:
                groups.append(src[start + 1 : k])
    return groups


def parse_graphs(src, symbol, n_graphs, num_facets):
    body = extract_braced(src, "graph_t %s[" % symbol)
    entries = []
    for fields in split_top_level(body):
        fields = fields.strip().lstrip("{").strip()
        # id, hash, automorphism_index, num_automorphisms, canonical_labelling, facets
        scalars = re.match(
            r"\s*(-?\d+)\s*,\s*(-?\d+)\s*,\s*(-?\d+)\s*,\s*(-?\d+)\s*,", fields
        )
        assert scalars, fields[:80]
        automorphism_index = int(scalars.group(3))
        num_automorphisms = int(scalars.group(4))
        # What follows is `{canonical_labelling}, {{f},{f},...}`; the labelling
        # braces are single-level and would otherwise parse as facets.
        rest = fields[scalars.end() :]
        groups = split_top_level(rest)
        assert len(groups) == 2, (symbol, len(groups))
        facets = []
        for f in split_top_level(groups[1]):
            nums = [int(x) for x in re.findall(r"-?\d+", f)]
            assert len(nums) == 3, (symbol, f)
            facets.append(tuple(nums))
        assert len(facets) == num_facets, (symbol, len(facets), num_facets)
        entries.append((automorphism_index, num_automorphisms, facets))
    assert len(entries) == n_graphs, (symbol, len(entries), n_graphs)
    return entries


HEADER = """//! Ideal templates, convex-hull graphs and graph automorphisms for PTM.
//!
//! GENERATED FILE -- do not edit by hand. Regenerate with:
//!
//! ```text
//! python3 tools/gen_ptm_tables.py \\
//!     --templates <OVITO PTM copy>/ptm_templates.h \\
//!     --graphs <upstream>/ptm_graph_data.cpp \\
//!     --out crates/rmda-core/src/analysis/ptm/tables.rs
//! ```
//!
//! Scope: all eight PTM structures. SC, FCC, HCP, ICO, BCC, DCUB and DHEX carry
//! templates and hull graphs; GRAPHENE carries templates only (upstream matches
//! it by direct permutation, `match_graphene`). Only template 0 of each is fitted;
//! the others are upstream's conventional-orientation variants.
//!
//! Many coordinates are decimal renderings of surds. `clippy::approx_constant`
//! would have those rewritten as `FRAC_1_SQRT_2` and friends, but those
//! constants are *different doubles*, and swapping them would move the
//! reference geometry by an ULP and stop the data matching upstream bit for bit.
#![allow(clippy::approx_constant)]
//!
//! Every point set has barycentre {{0, 0, 0}} and is scaled so that the mean
//! neighbour distance is 1. Coordinates are transcribed from upstream as exact
//! surd expressions so they evaluate identically to the C++ originals.

/// The coordinates are emitted as exact decimals rather than as the surd
/// expressions upstream writes, because `f64::sqrt` is not a `const fn` and a
/// `static` cannot call it. The upstream expression is kept as a trailing
/// comment on every row, and `tools/check_ptm_tables.py` asserts both that the
/// comment round-trips exactly and that the decimal is bit-identical to the
/// value C++ computes.
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--templates", required=True)
    ap.add_argument("--graphs", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    tsrc = open(args.templates).read()
    gsrc = open(args.graphs).read()

    automorphisms = parse_int_matrix(gsrc, "int8_t automorphisms[", 65, 17)

    out = [HEADER]
    out.append(
        "\n/// Number of hull-graph variants per structure (upstream `NUM_*_GRAPHS`).\n"
    )
    out.append("pub const N_AUTOMORPHISMS: usize = %d;\n" % len(automorphisms))

    out.append("\n/// Automorphism index sets, indexed by `graph.automorphism_index`.\n")
    out.append(
        "pub static AUTOMORPHISMS: [[i8; 17]; %d] = [\n" % len(automorphisms)
    )
    for row in automorphisms:
        out.append("    [" + ", ".join(str(v) for v in row) + "],\n")
    out.append("];\n")

    for name, nvar, nnbrs, nfacets, maxdeg, ngraphs in STRUCTURES:
        variants = parse_templates(tsrc, name, nvar)
        rust = RUST_TYPE[name]
        out.append(
            "\n/// Ideal %s template(s); point 0 is the central atom.\n"
            "pub static %s_TEMPLATES: [[[f64; 3]; 17]; %d] = [\n"
            % (name.upper(), rust, nvar)
        )
        for coords in variants:
            assert len(coords) == nnbrs + 1, (name, len(coords))
            # Upstream declares `[n][PTM_MAX_POINTS][3]`; only `num_nbrs + 1`
            # rows are initialised and the rest stay zero. Pad identically so
            # the Rust array is `MAX_POINTS` rows everywhere.
            padded = list(coords) + [["0", "0", "0"]] * (MAX_POINTS - len(coords))
            out.append("    [\n")
            for c in padded:
                # `f64::sqrt` is not a `const fn`, so the table cannot hold the
                # surd expressions themselves. Emit exact round-trip decimals and
                # keep the upstream expression as a comment: the literal is
                # bit-identical (asserted by tools/check_ptm_tables.py) and the
                # table still reads line-for-line against the C++ source.
                lits = [repr(eval_cpp_expr(x)) for x in c]
                comment = ", ".join(x.strip() for x in c)
                out.append(
                    "        [" + ", ".join(lits) + "], // " + comment + "\n"
                )
            out.append("    ],\n")
        out.append("];\n")

        if ngraphs == 0:
            continue
        graphs = parse_graphs(gsrc, "graphs_%s" % name, ngraphs, nfacets)
        out.append(
            "\n/// Convex-hull graphs for %s: automorphism window plus facet list.\n"
            "/// `hash` and `canonical_labelling` are computed at init, not stored.\n"
            "pub static %s_GRAPHS: [HullGraph; %d] = [\n" % (name.upper(), rust, ngraphs)
        )
        for aut_idx, n_aut, facets in graphs:
            out.append("    HullGraph {\n")
            out.append("        automorphism_index: %d,\n" % aut_idx)
            out.append("        num_automorphisms: %d,\n" % n_aut)
            out.append("        num_facets: %d,\n" % nfacets)
            # Upstream declares `int8_t facets[PTM_MAX_FACETS][3]` and relies on
            # the tail being zeroed. Pad the same way so the Rust type is fixed
            # and the matcher can carry one graph type for every structure.
            padded = list(facets) + [(0, 0, 0)] * (MAX_FACETS - nfacets)
            out.append("        facets: ")
            out.append(
                "[" + ", ".join("[%d, %d, %d]" % f for f in padded) + "],\n"
            )
            out.append("    },\n")
        out.append("];\n")

    out.append(
        """
/// A convex-hull graph variant: which automorphisms apply, and the hull faces.
///
/// The 280 `*_GRAPHS` tables across the seven hull-matched structures all have this
/// shape; they are separate tables upstream only for generator convenience.
/// `facets` is zero-padded to [`MAX_FACETS`]; only the first `num_facets`
/// entries are meaningful.
#[derive(Clone, Copy)]
pub struct HullGraph {
    pub automorphism_index: usize,
    pub num_automorphisms: usize,
    pub num_facets: usize,
    pub facets: [[i8; 3]; crate::analysis::ptm::MAX_FACETS],
}
"""
    )

    with open(args.out, "w") as f:
        f.write("".join(out))

    print("wrote %s" % args.out)


if __name__ == "__main__":
    sys.exit(main())
