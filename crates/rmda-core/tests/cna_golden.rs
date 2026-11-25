//! Golden-file parity for CNA against OVITO 3.15 (`tools/gen_fixtures.py`).
//!
//! `tests/data/structure_expected.txt` holds, per fixture, the per-atom
//! `StructureType` that OVITO's `CommonNeighborAnalysisModifier` reports in
//! both adaptive mode and fixed-cutoff (3.5 Å) mode. rmda must agree atom for
//! atom on the OVITO-verified structure fixtures plus the basic lattices.
//!
//! Codes: 0=OTHER 1=FCC 2=HCP 3=BCC 4=ICO 5=SC.

use rmda_core::analysis::cna::{Cna, CnaMode};
use rmda_core::analysis::{Analysis, AnalysisContext, NeighborhoodContext};
use rmda_core::data::property::PropertyStore;
use rmda_core::exec::Par;
use rmda_core::io::read_frames;
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::OnceLock;

fn data_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tests/data")
}

#[derive(Debug, Default)]
struct Fixture {
    adaptive_cutoff: f64,
    adaptive: Vec<u8>,
    fixed_cutoff: f64,
    fixed: Vec<u8>,
}

fn expected() -> &'static BTreeMap<String, Fixture> {
    static CELL: OnceLock<BTreeMap<String, Fixture>> = OnceLock::new();
    CELL.get_or_init(|| {
        let raw = std::fs::read_to_string(data_dir().join("structure_expected.txt"))
            .expect("tests/data/structure_expected.txt (run tools/gen_fixtures.py)");
        let mut map = BTreeMap::new();
        let mut lines = raw.lines().peekable();
        while let Some(line) = lines.next() {
            let Some(name) = line.strip_prefix("fixture ") else {
                continue;
            };
            let mut fx = Fixture::default();
            while let Some(next) = lines.peek() {
                if next.starts_with("fixture ") || next.trim().is_empty() {
                    break;
                }
                let cur = lines.next().unwrap().trim();
                let (key, value) = cur.split_once(' ').expect("key value");
                match key {
                    "cna_adaptive_cutoff" => fx.adaptive_cutoff = value.parse().unwrap(),
                    "cna_fixed_cutoff" => fx.fixed_cutoff = value.parse().unwrap(),
                    "cna_adaptive" => {
                        fx.adaptive = value
                            .split_whitespace()
                            .map(|v| v.parse().unwrap())
                            .collect()
                    }
                    "cna_fixed" => {
                        fx.fixed = value
                            .split_whitespace()
                            .map(|v| v.parse().unwrap())
                            .collect()
                    }
                    _ => {}
                }
            }
            map.insert(name.trim().to_string(), fx);
        }
        map
    })
}

fn run_cna(path: &Path, mode: CnaMode) -> Vec<u8> {
    let frames = read_frames(path).unwrap();
    let frame = &frames[0];
    let cna = Cna::new(mode);
    // CNA builds its own k-NN index.
    let nkeys = NeighborhoodContext { cell: None };
    let par = Par::new(1);
    let ctx = AnalysisContext {
        frame,
        nkeys: &nkeys,
        par: &par,
    };
    let mut out = PropertyStore::new();
    let mut summary = rmda_core::analysis::Summary::default();
    cna.run(&ctx, &mut out, &mut summary).unwrap();
    out.get("StructureType").unwrap().as_u8().unwrap().clone()
}

fn check(name: &str) {
    let fx = expected()
        .get(name)
        .unwrap_or_else(|| panic!("no expectation for {name}"));
    let path = data_dir().join(name);

    let got_adaptive = run_cna(&path, CnaMode::Adaptive);
    assert_eq!(
        got_adaptive, fx.adaptive,
        "{name}: adaptive CNA mismatch (cutoff {})",
        fx.adaptive_cutoff
    );

    let got_fixed = run_cna(
        &path,
        CnaMode::Fixed {
            cutoff: fx.fixed_cutoff,
        },
    );
    assert_eq!(
        got_fixed, fx.fixed,
        "{name}: fixed CNA mismatch (cutoff {})",
        fx.fixed_cutoff
    );
}

#[test]
fn cna_fcc_orthorhombic() {
    check("fcc_ortho.xyz");
}

#[test]
fn cna_hcp_hexagonal() {
    check("hcp_ortho.xyz");
}

#[test]
fn cna_bcc() {
    check("m2_bcc.xyz");
}

#[test]
fn cna_icosahedron() {
    check("m2_ico.xyz");
}

#[test]
fn cna_simple_cubic() {
    check("m2_sc.xyz");
}

#[test]
fn cna_fcc_thermal() {
    check("m2_fcc_thermal.xyz");
}
