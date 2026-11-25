//! Runtime reference-graph table (upstream `ptm_initialize_data.cpp`).
//!
//! The hardcoded tables in [`tables`] hold *source* graph data: facet lists
//! written in whatever winding was convenient, with no hash or canonical
//! labelling. Upstream derives those on first use in
//! `ptm_initialize_global`, and the derived values are what the matcher keys on:
//! `check_graphs` finds candidate reference graphs by hash, then uses the
//! stored canonical labelling to build the point mapping.
//!
//! Deriving them here rather than baking them into the generated table keeps
//! [`tables`] a literal transcription of upstream's data file, and means the
//! canonical-form code is genuinely exercised on every run instead of being
//! trusted from a precomputed blob.
//!
//! The derivation is deterministic, so it is done once per process behind a
//! [`OnceLock`].

use std::sync::OnceLock;

use crate::analysis::ptm::graph::{canonical_form_coloured, graph_degree, CanonicalForm};
use crate::analysis::ptm::hull::add_facet;
use crate::analysis::ptm::structure;
use crate::analysis::ptm::tables;
use crate::analysis::ptm::tables::HullGraph;
use crate::analysis::ptm::{MAX_NBRS, MAX_POINTS};

/// The structures in scope, matching upstream's `refdata_t` type codes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Structure {
    Sc,
    Fcc,
    Hcp,
    Ico,
    Bcc,
    /// Cubic diamond: 4 nearest + 12 second neighbours (two-shell).
    Dcub,
    /// Hexagonal diamond (lonsdaleite): 4 + 12, two-shell.
    Dhex,
    /// Graphene: 3 nearest + 6 second neighbours, matched by direct
    /// permutation rather than a hull graph.
    Graphene,
}

impl Structure {
    pub const ALL: [Structure; 8] = [
        Structure::Sc,
        Structure::Fcc,
        Structure::Hcp,
        Structure::Ico,
        Structure::Bcc,
        Structure::Dcub,
        Structure::Dhex,
        Structure::Graphene,
    ];

    /// Upstream `PTM_MATCH_*` code, which is also our reported structure type.
    pub fn type_code(self) -> i32 {
        match self {
            Structure::Sc => structure::SC,
            Structure::Fcc => structure::FCC,
            Structure::Hcp => structure::HCP,
            Structure::Ico => structure::ICO,
            Structure::Bcc => structure::BCC,
            Structure::Dcub => structure::DCUB,
            Structure::Dhex => structure::DHEX,
            Structure::Graphene => structure::GRAPHENE,
        }
    }

    pub fn from_type_code(code: i32) -> Option<Structure> {
        match code {
            structure::SC => Some(Structure::Sc),
            structure::FCC => Some(Structure::Fcc),
            structure::HCP => Some(Structure::Hcp),
            structure::ICO => Some(Structure::Ico),
            structure::BCC => Some(Structure::Bcc),
            structure::DCUB => Some(Structure::Dcub),
            structure::DHEX => Some(Structure::Dhex),
            structure::GRAPHENE => Some(Structure::Graphene),
            _ => None,
        }
    }

    /// `check` bit for this structure, used to honour the requested check set.
    pub fn check_bit(self) -> u32 {
        match self {
            Structure::Sc => crate::analysis::ptm::check::SC,
            Structure::Fcc => crate::analysis::ptm::check::FCC,
            Structure::Hcp => crate::analysis::ptm::check::HCP,
            Structure::Ico => crate::analysis::ptm::check::ICO,
            Structure::Bcc => crate::analysis::ptm::check::BCC,
            Structure::Dcub => crate::analysis::ptm::check::DCUB,
            Structure::Dhex => crate::analysis::ptm::check::DHEX,
            Structure::Graphene => crate::analysis::ptm::check::GRAPHENE,
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Structure::Sc => "sc",
            Structure::Fcc => "fcc",
            Structure::Hcp => "hcp",
            Structure::Ico => "ico",
            Structure::Bcc => "bcc",
            Structure::Dcub => "dcub",
            Structure::Dhex => "dhex",
            Structure::Graphene => "graphene",
        }
    }
}

/// Upstream `dcolours`: inner (first-shell) neighbours 0-3 coloured 1.
pub const DIAMOND_COLOURS: [i8; MAX_POINTS] = [1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];

/// One reference graph, with the values derived at initialisation.
#[derive(Debug, Clone, Copy)]
pub struct GraphInfo {
    /// Canonical-form hash, used to find candidate graphs cheaply.
    pub hash: u64,
    /// Canonical labelling: `canonical_labelling[i + 1]` is the 1-based
    /// canonical rank of node `i`; entry 0 (the central atom) is 0.
    pub canonical_labelling: [i8; MAX_POINTS],
    /// Facet-incidence degree per node.
    pub degree: [i8; MAX_NBRS],
    /// Index of this graph's first automorphism in `tables::AUTOMORPHISMS`.
    pub automorphism_index: usize,
    pub num_automorphisms: usize,
}

impl GraphInfo {
    /// The subset of a [`CanonicalForm`] this module keeps.
    pub fn form(&self) -> CanonicalForm {
        CanonicalForm {
            hash: self.hash,
            canonical_labelling: self.canonical_labelling,
        }
    }
}

/// A reference structure: the ideal templates plus the derived graph table.
pub struct RefData {
    pub structure: Structure,
    /// Neighbours of the central atom.
    pub num_nbrs: usize,
    /// Facets of the reference hull, used as a cheap pre-filter.
    pub num_facets: usize,
    /// Upper bound on node degree; a candidate above this is rejected before
    /// any expensive work.
    pub max_degree: i8,
    /// Ideal point sets. Index 0 is the one used for RMSD/scale/orientation;
    /// the alternative templates only matter for the conventional orientation
    /// (the template remap).
    pub templates: &'static [[[f64; 3]; MAX_POINTS]],
    /// Source facet lists, as transcribed from upstream.
    source_graphs: &'static [HullGraph],
    derived: OnceLock<Vec<GraphInfo>>,
}

macro_rules! refdata {
    ($name:ident, $structure:expr, $num_nbrs:expr, $num_facets:expr, $max_degree:expr,
     $templates:expr, $graphs:expr) => {
        pub static $name: RefData = RefData {
            structure: $structure,
            num_nbrs: $num_nbrs,
            num_facets: $num_facets,
            max_degree: $max_degree,
            templates: $templates,
            source_graphs: $graphs,
            derived: OnceLock::new(),
        };
    };
}

refdata!(
    SC,
    Structure::Sc,
    6,
    8,
    4,
    &[tables::SC_TEMPLATES[0]],
    &tables::SC_GRAPHS
);
refdata!(
    FCC,
    Structure::Fcc,
    12,
    20,
    6,
    &[tables::FCC_TEMPLATES[0]],
    &tables::FCC_GRAPHS
);
refdata!(
    ICO,
    Structure::Ico,
    12,
    20,
    6,
    &[tables::ICO_TEMPLATES[0]],
    &tables::ICO_GRAPHS
);
refdata!(
    BCC,
    Structure::Bcc,
    14,
    24,
    8,
    &[tables::BCC_TEMPLATES[0]],
    &tables::BCC_GRAPHS
);
refdata!(
    DCUB,
    Structure::Dcub,
    16,
    28,
    8,
    &[tables::DCUB_TEMPLATES[0], tables::DCUB_TEMPLATES[1]],
    &tables::DCUB_GRAPHS
);
refdata!(
    DHEX,
    Structure::Dhex,
    16,
    28,
    8,
    &[
        tables::DHEX_TEMPLATES[0],
        tables::DHEX_TEMPLATES[1],
        tables::DHEX_TEMPLATES[2],
        tables::DHEX_TEMPLATES[3],
    ],
    &tables::DHEX_GRAPHS
);
// Graphene has no hull graphs (upstream `num_facets = num_graphs = -1`):
// `match_graphene` fits the template directly over eight permutations.
refdata!(
    GRAPHENE,
    Structure::Graphene,
    9,
    0,
    0,
    &[tables::GRAPHENE_TEMPLATES[0], tables::GRAPHENE_TEMPLATES[1]],
    &[]
);
refdata!(
    HCP,
    Structure::Hcp,
    12,
    20,
    6,
    &[tables::HCP_TEMPLATES[0], tables::HCP_TEMPLATES[1],],
    &tables::HCP_GRAPHS
);

/// Every structure, in a stable order.
pub fn structures() -> [&'static RefData; 8] {
    [&SC, &FCC, &HCP, &ICO, &BCC, &DCUB, &DHEX, &GRAPHENE]
}

impl RefData {
    /// Look up a structure by its reported type code.
    pub fn for_type(code: i32) -> Option<&'static RefData> {
        Some(match Structure::from_type_code(code)? {
            Structure::Sc => &SC,
            Structure::Fcc => &FCC,
            Structure::Hcp => &HCP,
            Structure::Ico => &ICO,
            Structure::Bcc => &BCC,
            Structure::Dcub => &DCUB,
            Structure::Dhex => &DHEX,
            Structure::Graphene => &GRAPHENE,
        })
    }

    /// Number of source graphs transcribed for this structure.
    pub fn source_graph_count(&self) -> usize {
        self.source_graphs.len()
    }

    /// Ideal points for the central atom and its `num_nbrs` neighbours.
    pub fn points(&self, variant: usize) -> &[[f64; 3]; MAX_POINTS] {
        &self.templates[variant.min(self.templates.len() - 1)]
    }

    /// The derived graph table, initialising it on first use.
    pub fn graphs(&self) -> &[GraphInfo] {
        self.derived.get_or_init(|| self.derive_graphs())
    }

    /// Derive degree, hash and canonical labelling for every reference graph.
    ///
    /// Mirrors upstream's `initialize_graphs`, including the step that rewrites
    /// each source facet list into clockwise winding: the canonical form is
    /// defined over directed edges, so winding has to be normalised first.
    fn derive_graphs(&self) -> Vec<GraphInfo> {
        let template = self.points(0);
        // Upstream passes `&s->points[1]`: the graph's node indices address the
        // neighbours, not the central atom.
        let neighbours = &template[1..=self.num_nbrs];
        // Upstream `dcolours`: the diamonds colour their four inner neighbours
        // 1 so the canonical form cannot swap a first-shell node for a
        // second-shell one; every other structure is uncoloured.
        let colours = match self.structure {
            Structure::Dcub | Structure::Dhex => DIAMOND_COLOURS,
            _ => [0i8; MAX_POINTS],
        };

        let mut out = Vec::with_capacity(self.source_graphs.len());
        for g in self.source_graphs {
            let mut facets = [[0i8; 3]; crate::analysis::ptm::MAX_FACETS];
            facets[..self.num_facets].copy_from_slice(&g.facets[..self.num_facets]);

            let (degree, max_degree) = graph_degree(self.num_facets, &facets, self.num_nbrs);
            debug_assert!(
                max_degree <= self.max_degree,
                "{}: template degree {max_degree} exceeds bound {}",
                self.structure.name(),
                self.max_degree
            );

            // Normalise the winding: `add_facet` flips the facet if its normal
            // points at the origin, so the stored copy ends up clockwise.
            for f in facets.iter_mut().take(self.num_facets) {
                let (a, b, c) = (f[0] as usize, f[1] as usize, f[2] as usize);
                *f = add_facet(neighbours, a, b, c, [0.0; 3]).0;
            }

            let form =
                canonical_form_coloured(self.num_facets, &facets, self.num_nbrs, &degree, &colours)
                    .unwrap_or_else(|_| {
                        panic!(
                            "{}: reference graph is not a closed manifold",
                            self.structure.name()
                        )
                    });

            out.push(GraphInfo {
                hash: form.hash,
                canonical_labelling: form.canonical_labelling,
                degree,
                automorphism_index: g.automorphism_index,
                num_automorphisms: g.num_automorphisms,
            });
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn descriptor_fields_match_upstream() {
        for (s, nn, nf, md, ng) in [
            (&SC, 6usize, 8usize, 4i8, 1usize),
            (&FCC, 12, 20, 6, 8),
            (&HCP, 12, 20, 6, 16),
            (&ICO, 12, 20, 6, 1),
            (&BCC, 14, 24, 8, 218),
            (&DCUB, 16, 28, 8, 12),
            (&DHEX, 16, 28, 8, 24),
            (&GRAPHENE, 9, 0, 0, 0),
        ] {
            assert_eq!(s.num_nbrs, nn, "{}", s.structure.name());
            assert_eq!(s.num_facets, nf, "{}", s.structure.name());
            assert_eq!(s.max_degree, md, "{}", s.structure.name());
            assert_eq!(s.source_graphs.len(), ng, "{}", s.structure.name());
        }
    }

    #[test]
    fn every_structure_derives_the_expected_number_of_graphs() {
        assert_eq!(SC.graphs().len(), 1);
        assert_eq!(FCC.graphs().len(), 8);
        assert_eq!(HCP.graphs().len(), 16);
        assert_eq!(ICO.graphs().len(), 1);
        assert_eq!(BCC.graphs().len(), 218);
        assert_eq!(DCUB.graphs().len(), 12);
        assert_eq!(DHEX.graphs().len(), 24);
        assert_eq!(GRAPHENE.graphs().len(), 0);
    }

    #[test]
    fn type_codes_round_trip() {
        for s in Structure::ALL {
            assert_eq!(
                RefData::for_type(s.type_code()).map(|r| r.structure),
                Some(s)
            );
        }
        assert!(RefData::for_type(structure::OTHER).is_none());
    }

    #[test]
    fn check_bits_cover_every_structure() {
        // No two in-scope structures may share a check bit, and the default set
        // must be exactly FCC|HCP|ICO|BCC.
        let mut seen = 0u32;
        for s in Structure::ALL {
            let bit = s.check_bit();
            assert_eq!(seen & bit, 0, "{} reuses a check bit", s.name());
            seen |= bit;
        }
        use crate::analysis::ptm::check;
        assert_eq!(seen, check::ALL);
        assert_eq!(
            check::DEFAULT,
            seen & !(check::SC | check::DCUB | check::DHEX | check::GRAPHENE)
        );
    }

    #[test]
    fn hash_collisions_are_expected() {
        // The canonical hash is a lossy digest: distinct graphs do collide
        // (upstream has 7 distinct hashes for FCC's 8 graphs, 199 for BCC's
        // 218). `check_graphs` therefore scans *every* graph with a matching
        // hash and keeps the lowest RMSD, so a collision costs time, not
        // correctness. This test pins the counts so a change in the hash
        // function shows up here rather than as a silent accuracy change.
        for (s, graphs, distinct) in [
            (&SC, 1usize, 1usize),
            (&FCC, 8, 7),
            (&HCP, 16, 13),
            (&ICO, 1, 1),
            (&BCC, 218, 199),
        ] {
            let set: std::collections::HashSet<u64> =
                s.graphs().iter().take(graphs).map(|g| g.hash).collect();
            assert_eq!(
                set.len(),
                distinct,
                "{}: distinct hash count changed",
                s.structure.name()
            );
        }
    }

    #[test]
    fn automorphism_windows_are_in_range() {
        for s in structures() {
            for (gi, g) in s.graphs().iter().enumerate() {
                assert!(
                    g.automorphism_index + g.num_automorphisms <= tables::N_AUTOMORPHISMS,
                    "{}: graph {gi} automorphism window out of range",
                    s.structure.name()
                );
            }
        }
    }

    #[test]
    fn hcp_variants_are_distinct() {
        // Variant 0 is what matching uses; variant 1 exists for conventional
        // orientation. They must not be the same point set.
        assert_ne!(HCP.points(0), HCP.points(1));
    }
}
