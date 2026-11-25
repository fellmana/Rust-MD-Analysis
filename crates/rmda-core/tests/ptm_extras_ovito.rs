//! PTM's opt-in outputs against OVITO 3.15, atom for atom
//! (`tools/gen_ptm_extras.py`): alloy ordering type, elastic deformation
//! gradient, interatomic distance and the encoded correspondences on B2, L1₂,
//! L1₀, zincblende, wurtzite and hexagonal-BN fixtures (thermal noise plus
//! antisite swaps) and on strained, rotated FCC and HCP. PTM runs with OVITO's
//! Voronoi neighbour order. The upstream-only outputs (F residual, polar
//! decomposition), which OVITO lacks, are checked by their defining identities
//! on the same atoms.

use rmda_core::analysis::ptm::{output, NeighbourOrder, Ptm};
use rmda_core::analysis::{AnalysisContext, NeighborhoodContext, Summary};
use rmda_core::data::property::PropertyStore;
use rmda_core::exec::Par;
use rmda_core::io::read_frames;
use std::collections::HashMap;
use std::io::Read;
use std::path::{Path, PathBuf};

fn data_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tests/data")
}

/// fixture -> (field -> raw value text), in file order.
fn expected() -> Vec<(String, HashMap<String, String>)> {
    let gz = std::fs::read(data_dir().join("ptm_extras_expected.txt.gz"))
        .expect("tests/data/ptm_extras_expected.txt.gz (run tools/gen_ptm_extras.py)");
    let mut raw = String::new();
    flate2::read::GzDecoder::new(&gz[..])
        .read_to_string(&mut raw)
        .unwrap();
    let mut out: Vec<(String, HashMap<String, String>)> = Vec::new();
    for l in raw
        .lines()
        .filter(|l| !l.starts_with('#') && !l.trim().is_empty())
    {
        let mut it = l.splitn(3, ' ');
        let (name, field, val) = (it.next().unwrap(), it.next().unwrap(), it.next().unwrap());
        if out.last().map_or(true, |(n, _)| n != name) {
            out.push((name.to_string(), HashMap::new()));
        }
        out.last_mut()
            .unwrap()
            .1
            .insert(field.to_string(), val.to_string());
    }
    out
}

fn digits(s: &str) -> Vec<u8> {
    s.bytes().map(|b| b - b'0').collect()
}

fn floats(s: &str) -> Vec<f64> {
    s.split_whitespace().map(|v| v.parse().unwrap()).collect()
}

fn run(name: &str, flags: u32, threads: usize) -> PropertyStore {
    let path = data_dir().join("ptm_extras").join(format!("{name}.xyz"));
    let frame = read_frames(&path).unwrap().into_iter().next().unwrap();
    let nkeys = NeighborhoodContext { cell: None };
    let par = Par::new(threads);
    let ctx = AnalysisContext {
        frame: &frame,
        nkeys: &nkeys,
        par: &par,
    };
    let mut out = PropertyStore::new();
    let mut summary = Summary::default();
    Ptm {
        rmsd_cutoff: 0.1,
        flags,
    }
    .run_extended(
        &ctx,
        &mut out,
        &mut summary,
        NeighbourOrder::Voronoi,
        output::ALL,
    )
    .unwrap();
    out
}

#[test]
fn ordering_and_deformation_gradient_match_ovito() {
    let cases = expected();
    assert_eq!(cases.len(), 8);
    for (name, f) in &cases {
        let flags = f["structures"]
            .split(',')
            .map(|s| 1u32 << (s.parse::<u32>().unwrap() - 1))
            .fold(0, |a, b| a | b);
        // OVITO's structure ids are 1-based PTM_MATCH codes; the check bits
        // follow the same order (FCC=1<<0 ... GRAPHENE=1<<7).
        let out = run(name, flags, 2);
        let types = out.get("StructureType").unwrap().as_u8().unwrap();
        let ord = out.get("StructureOrdering").unwrap().as_u8().unwrap();
        let fgrad = out
            .get("StructureDeformationGradient")
            .unwrap()
            .as_f64()
            .unwrap();
        let ia = out
            .get("StructureInteratomicDistance")
            .unwrap()
            .as_f64()
            .unwrap();
        let tpl = out.get("StructureTemplate").unwrap().as_u8().unwrap();
        let corr = out.get("Correspondences").unwrap().as_i64().unwrap();

        let want_types = digits(&f["types"]);
        let want_ord = digits(&f["ordering"]);
        let want_f = floats(&f["F"]);
        let want_ia = floats(&f["ia"]);
        let want_corr: Vec<i64> = f["corr"]
            .split_whitespace()
            .map(|v| u64::from_str_radix(v, 16).unwrap() as i64)
            .collect();
        let n = want_types.len();
        assert_eq!(types.len(), n, "{name}");
        assert_eq!(types[..], want_types[..], "{name}: structure types");
        let differ = (0..n).filter(|&i| ord[i] != want_ord[i]).count();
        // OVITO's raw component array is upstream's row-major F[3 i + j],
        // which is ours (its labels XX YX ZX ... read column-major, so the
        // matrix OVITO *labels* is the transpose of F; the strained fixtures
        // confirm the row-major reading is the applied deformation).
        // Relative to max(|F_ij|, 1): the oracle stores 13 significant digits.
        let worst_f = (0..9 * n)
            .map(|k| (fgrad[k] - want_f[k]).abs() / want_f[k].abs().max(1.0))
            .fold(0.0f64, f64::max);
        let corr_differ = (0..n).filter(|&i| corr[i] != want_corr[i]).count();
        let worst_ia = (0..n)
            .map(|i| (ia[i] - want_ia[i]).abs() / want_ia[i].max(1e-12))
            .fold(0.0f64, f64::max);
        let mut tcount = [0usize; 4];
        for &t in tpl {
            tcount[t as usize] += 1;
        }
        println!(
            "{name}: {n} atoms, ordering differs on {differ}, correspondences on \
             {corr_differ}, worst relative dF {worst_f:.1e}, worst relative ia {worst_ia:.1e}, \
             templates {tcount:?}"
        );
        // Upstream-only outputs (bitwise equal to upstream's
        // `polar_decomposition_3x3` on these atoms, checked with a C++ probe;
        // design §14.2): F = P U with U a proper rotation and P symmetric,
        // and a non-negative residual.
        let u = out.get("StructurePolarU").unwrap().as_f64().unwrap();
        let p = out.get("StructurePolarP").unwrap().as_f64().unwrap();
        let res = out
            .get("StructureDeformationResidual")
            .unwrap()
            .as_f64()
            .unwrap();
        let mut worst_polar = 0.0f64;
        for i in 0..n {
            let (fi, ui, pi) = (&fgrad[9 * i..], &u[9 * i..], &p[9 * i..]);
            if fi[..9].iter().all(|&v| v == 0.0) {
                continue;
            }
            for r in 0..3 {
                for c in 0..3 {
                    let pu: f64 = (0..3).map(|k| pi[3 * r + k] * ui[3 * k + c]).sum();
                    let uut: f64 = (0..3).map(|k| ui[3 * r + k] * ui[3 * c + k]).sum();
                    let id = if r == c { 1.0 } else { 0.0 };
                    worst_polar = worst_polar
                        .max((pu - fi[3 * r + c]).abs())
                        .max((uut - id).abs())
                        .max((pi[3 * r + c] - pi[3 * c + r]).abs());
                }
            }
            let det = ui[0] * (ui[4] * ui[8] - ui[5] * ui[7])
                - ui[1] * (ui[3] * ui[8] - ui[5] * ui[6])
                + ui[2] * (ui[3] * ui[7] - ui[4] * ui[6]);
            worst_polar = worst_polar.max((det - 1.0).abs());
            assert!(res[3 * i..3 * i + 3].iter().all(|&r| r >= 0.0), "{name}");
        }
        assert!(
            worst_polar < 1e-12,
            "{name}: polar identities off by {worst_polar:e}"
        );
        assert_eq!(
            corr_differ, 0,
            "{name}: correspondences differ from OVITO on {corr_differ} atoms"
        );
        assert_eq!(
            differ, 0,
            "{name}: ordering differs from OVITO on {differ} atoms"
        );
        assert!(worst_f < 1e-12, "{name}: F differs by {worst_f:e}");
        assert!(
            worst_ia < 1e-12,
            "{name}: interatomic distance differs by {worst_ia:e}"
        );
    }
}

/// The opt-in columns do not depend on the thread count, and leave the
/// default columns exactly as a default run writes them.
#[test]
fn extras_are_deterministic_and_leave_defaults_alone() {
    let name = "l12_t10";
    let a = run(name, 0b111, 1);
    let b = run(name, 0b111, 4);
    for key in [
        "StructureOrdering",
        "StructureTemplate",
        "StructureDeformationGradient",
    ] {
        assert!(same(a.get(key).unwrap(), b.get(key).unwrap()), "{key}");
    }
    let path = data_dir().join("ptm_extras").join(format!("{name}.xyz"));
    let frame = read_frames(&path).unwrap().into_iter().next().unwrap();
    let nkeys = NeighborhoodContext { cell: None };
    let par = Par::new(2);
    let ctx = AnalysisContext {
        frame: &frame,
        nkeys: &nkeys,
        par: &par,
    };
    let mut plain = PropertyStore::new();
    Ptm::default()
        .run_ordered(
            &ctx,
            &mut plain,
            &mut Summary::default(),
            NeighbourOrder::Voronoi,
        )
        .unwrap();
    assert!(plain.get("StructureOrdering").is_none());
    for p in plain.iter() {
        assert!(same(p, a.get(&p.name).unwrap()), "{}", p.name);
    }
}

/// Bitwise equality of two u8 or f64 columns.
fn same(p: &rmda_core::data::property::Property, q: &rmda_core::data::property::Property) -> bool {
    if let (Some(x), Some(y)) = (p.as_u8(), q.as_u8()) {
        return x == y;
    }
    match (p.as_f64(), q.as_f64()) {
        (Some(x), Some(y)) => {
            x.len() == y.len() && x.iter().zip(y).all(|(a, b)| a.to_bits() == b.to_bits())
        }
        _ => false,
    }
}
