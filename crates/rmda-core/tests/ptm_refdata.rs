//! Differential test of the derived PTM reference-graph table against upstream.
//!
//! `tests/data/ptm_refdata_graphs.txt` holds, for each of the 280 reference
//! graphs (SC, FCC, HCP, ICO, BCC, DCUB, DHEX), the canonical hash and canonical labelling that upstream's
//! `ptm_initialize_global` derives from the hardcoded facet lists.
//!
//! These values are what `check_graphs` matches on, so a divergence here means
//! either a wrong hash (wrong candidate graphs selected) or a wrong labelling
//! (wrong automorphism, wrong RMSD). Everything in the file was produced by
//! compiling the upstream C++ and running the same derivation; regenerate with
//! `tools/gen_ptm_refdata.py`.

use rmda_core::analysis::ptm::refdata::{self, RefData};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::OnceLock;

fn data_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tests/data")
}

type PtmRef = BTreeMap<(String, usize), (u64, Vec<i8>)>;

/// `(structure, graph index) -> (hash, canonical labelling)`
fn reference() -> &'static PtmRef {
    static CELL: OnceLock<PtmRef> = OnceLock::new();
    CELL.get_or_init(|| {
        let path = data_dir().join("ptm_refdata_graphs.txt");
        let raw = std::fs::read_to_string(&path).unwrap_or_else(|e| {
            panic!(
                "{}: {e} (run tools/gen_ptm_refdata.py to regenerate)",
                path.display()
            )
        });
        let mut map = BTreeMap::new();
        for (lineno, line) in raw.lines().enumerate() {
            let line = line.trim();
            if line.is_empty() || line.starts_with('#') {
                continue;
            }
            let mut fields = line.split_whitespace();
            let name = fields.next().expect("structure name");
            let index: usize = fields.next().expect("graph index").parse().expect("index");
            let hash = u64::from_str_radix(fields.next().expect("hash"), 16).expect("hex hash");
            let labelling: Vec<i8> = fields
                .map(|f| f.parse().expect("labelling entry"))
                .collect();
            assert!(!labelling.is_empty(), "line {}: no labelling", lineno + 1);
            assert!(
                map.insert((name.to_string(), index), (hash, labelling))
                    .is_none(),
                "line {}: duplicate entry for {} graph {}",
                lineno + 1,
                name,
                index
            );
        }
        assert_eq!(
            map.len(),
            280,
            "expected all 280 reference graphs, got {}",
            map.len()
        );
        map
    })
}

#[test]
fn derived_graph_table_matches_upstream() {
    let reference = reference();
    let mut compared = 0usize;

    for s in refdata::structures() {
        let name = s.structure.name().to_string();
        let graphs = s.graphs();
        assert_eq!(graphs.len(), s.source_graph_count(), "{name}: graph count");
        for (i, g) in graphs.iter().enumerate() {
            let (want_hash, want_labelling) = reference
                .get(&(name.clone(), i))
                .unwrap_or_else(|| panic!("{name}: no reference row for graph {i}"));
            assert_eq!(
                g.hash, *want_hash,
                "{name}[{i}]: hash {} != upstream {want_hash:016x}",
                g.hash
            );
            assert_eq!(
                &g.canonical_labelling[..s.num_nbrs],
                &want_labelling[..],
                "{name}[{i}]: canonical labelling differs from upstream"
            );
            compared += 1;
        }
    }

    assert_eq!(compared, 280, "not every reference graph was compared");
}

#[test]
fn no_reference_row_is_left_uncompared() {
    // Guards against a structure silently dropping out of the loop above.
    let mut from_code = 0usize;
    for s in refdata::structures() {
        from_code += RefData::for_type(s.structure.type_code())
            .expect("structure lookup")
            .graphs()
            .len();
    }
    assert_eq!(from_code, reference().len());
}
