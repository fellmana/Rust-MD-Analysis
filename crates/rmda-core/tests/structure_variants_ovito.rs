//! PTM two-shell structures and the CNA variants against OVITO 3.15, atom for
//! atom (`tools/gen_structure_variants.py`).
//!
//! `tests/data/structure_variants_expected.txt` holds, per small thermal
//! fixture in `tests/data/variants/`, OVITO's per-atom `Structure Type` for
//! PTM with the diamonds and graphene enabled (`ptm6`: FCC, HCP, BCC, DCUB,
//! DHEX, graphene; `ptm8`: all eight), interval CNA, fixed-cutoff CNA,
//! bond-based CNA (`CreateBondsModifier` at the same cutoff) and
//! identify-diamond. PTM is run with OVITO's Voronoi neighbour order.

use rmda_core::analysis::cna::{Cna, CnaMode};
use rmda_core::analysis::ptm::{check, NeighbourOrder, Ptm};
use rmda_core::analysis::{Analysis, AnalysisContext, NeighborhoodContext, Summary};
use rmda_core::data::property::PropertyStore;
use rmda_core::exec::Par;
use rmda_core::io::read_frames;
use std::path::{Path, PathBuf};

fn data_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tests/data")
}

/// `(fixture, cutoff, analysis, labels)` rows of the oracle file.
fn expected() -> Vec<(String, f64, String, Vec<u8>)> {
    let raw = std::fs::read_to_string(data_dir().join("structure_variants_expected.txt"))
        .expect("tests/data/structure_variants_expected.txt (run tools/gen_structure_variants.py)");
    raw.lines()
        .filter(|l| !l.starts_with('#') && !l.trim().is_empty())
        .map(|l| {
            let mut it = l.split_whitespace();
            let name = it.next().unwrap().to_string();
            let cutoff = it.next().unwrap().parse().unwrap();
            let kind = it.next().unwrap().to_string();
            let labels = it.next().unwrap().bytes().map(|b| b - b'0').collect();
            (name, cutoff, kind, labels)
        })
        .collect()
}

fn run(name: &str, cutoff: f64, kind: &str, threads: usize) -> Vec<u8> {
    let path = data_dir().join("variants").join(format!("{name}.xyz"));
    let frames = read_frames(&path).unwrap();
    let frame = &frames[0];
    let nkeys = NeighborhoodContext { cell: None };
    let par = Par::new(threads);
    let ctx = AnalysisContext {
        frame,
        nkeys: &nkeys,
        par: &par,
    };
    let mut out = PropertyStore::new();
    let mut summary = Summary::default();
    match kind {
        "ptm6" | "ptm8" => {
            let flags = if kind == "ptm6" {
                check::FCC | check::HCP | check::BCC | check::TWO_SHELL
            } else {
                check::ALL
            };
            Ptm {
                rmsd_cutoff: 0.1,
                flags,
            }
            .run_ordered(&ctx, &mut out, &mut summary, NeighbourOrder::Voronoi)
            .unwrap();
        }
        _ => {
            let mode = match kind {
                "interval" => CnaMode::Interval,
                "fixed" => CnaMode::Fixed { cutoff },
                "bond" => CnaMode::Bond { cutoff },
                "diamond" => CnaMode::Diamond,
                other => panic!("unknown analysis {other}"),
            };
            Cna::new(mode).run(&ctx, &mut out, &mut summary).unwrap();
        }
    }
    out.get("StructureType").unwrap().as_u8().unwrap().clone()
}

#[test]
fn every_variant_matches_ovito_atom_for_atom() {
    let rows = expected();
    assert!(rows.len() >= 50, "oracle has {} rows", rows.len());
    let mut failures = Vec::new();
    for (name, cutoff, kind, want) in &rows {
        let got = run(name, *cutoff, kind, 1);
        assert_eq!(got.len(), want.len(), "{name} {kind}: atom count");
        let bad: Vec<usize> = (0..got.len()).filter(|&i| got[i] != want[i]).collect();
        if !bad.is_empty() {
            failures.push(format!(
                "{name} {kind}: {} of {} differ (first: atom {} rmda {} ovito {})",
                bad.len(),
                got.len(),
                bad[0],
                got[bad[0]],
                want[bad[0]]
            ));
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

/// The new modes are deterministic across thread counts.
#[test]
fn variants_are_thread_count_independent() {
    for (name, cutoff, kind) in [
        ("dcub_si_vac_surface", 2.8, "ptm8"),
        ("fcc_t15", 3.0, "interval"),
        ("bcc_t15", 3.4, "bond"),
        ("dcub_si_t20", 2.8, "diamond"),
    ] {
        assert_eq!(
            run(name, cutoff, kind, 1),
            run(name, cutoff, kind, 4),
            "{name} {kind}"
        );
    }
}
