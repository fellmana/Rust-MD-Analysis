//! Index-for-index parity with OVITO's PTM modifier.
//!
//! `tools/capture_ptm.py` runs `PolyhedralTemplateMatchingModifier` (OVITO
//! 3.15.0, default settings) over the four `reference_structures/` snapshots
//! and writes the per-atom `Structure Type` and `Structure RMSD` vectors to
//! `target/refcheck/<name>.ptm`: a one-line header, then the types, then the
//! RMSDs. Those oracles are a build artefact rather than a committed fixture,
//! so the test skips when they are absent.
//!
//! Both files are run at the shipped defaults, so this is a direct check that
//! `Ptm::default()` means the same thing here as it does in OVITO.
//!
//! # What is and is not expected to match
//!
//! The RMSDs of atoms that *did* match a structure agree essentially exactly -
//! in practice to the last bit, or off by a few ULP from the order of a
//! floating-point dot product. That is the strong statement in this file, and
//! it is what pins the RMSD convention: upstream normalises by the reference's
//! own point count (13 for FCC/HCP/ICO, 15 for BCC), not by the width of the
//! gathered environment. Dividing by 19 instead scales every RMSD by
//! `sqrt(13/19)` and silently loosens the cutoff by the same factor.
//!
//! **Resolved (see `voronoi_order_matches_ovito_exactly`):** OVITO hands
//! `ptm_index` the 18 nearest neighbours in `ptm_preorder_neighbours` order
//! (decreasing Voronoi solid angle), so it matches the 12 (14) largest
//! Voronoi faces rather than the 12 (14) nearest. With that order every type
//! and RMSD agrees. The analysis below is of the default distance order.
//!
//! The *type* agreement is high but not exact: 99.6-99.997% depending on the
//! snapshot. The residual is not noise at the cutoff - no disagreeing atom sits
//! within 1e-6 of 0.1. Instead, in every sampled case OVITO finds a *better*
//! fit than we do (e.g. 0.089 where we get 0.099), which is the opposite of
//! what a tie-break or precision difference would produce.
//!
//! That residual is a difference in the *input environment*, not in this port,
//! and it was pinned down rather than guessed at:
//!
//!   1. Our gathered neighbours are exactly right. For a disagreeing atom the
//!      19 nearest agree with an independent brute force over periodic images
//!      in atom id, image shift, and distance, to the last digit.
//!   2. This port matches compiled upstream on that environment. Feeding the
//!      same 13 points to upstream's own `get_convex_hull` +
//!      `canonical_form_coloured` + `match_fcc_hcp_ico` reproduces our hull
//!      (20 facets), graph hash (10757376221875612728), canonical labelling,
//!      and the same "no FCC/HCP/ICO match" verdict.
//!   3. No choice of 13 from our 19 can reproduce OVITO's answer. Enumerating
//!      all 27132 subsets of the 19 gathered points and running the full
//!      upstream dispatch on each produces no subset matching OVITO's
//!      structure and RMSD. Normalising over 13..19 points instead of 19 does
//!      not produce it either.
//!
//! So for these atoms OVITO feeds `ptm_index` a different point set than
//! "the 19 nearest, distance-ordered". `lib_ptm` puts that requirement on the
//! host rather than enforcing it, and this port meets it; OVITO evidently does
//! not meet it for a small fraction of atoms. Chasing the last fraction of a
//! percent would mean replicating OVITO's internal particle reordering rather
//! than implementing PTM, so it is documented instead.
//!
//! The clearest corroboration is a synthetic perfect simple-cubic lattice,
//! where the 6 nearest neighbours are unambiguous and every RMSD is 0. This
//! port returns SC for 64/64 atoms at RMSD 0.0. OVITO, on the same input,
//! reports no SC at all - not even when the cutoff is raised to 100, where it
//! splits the box into 47 Other, 7 HCP and 10 BCC. A host that passed the
//! 6 nearest would have to find SC there, so this is the same defect wearing a
//! more obvious costume. None of the four shipped snapshots contain SC atoms,
//! which is why it never showed up in the histograms above.
//!
//! Note also that OVITO writes `RMSD = 0` for *some* unmatched atoms and a real
//! best-fit value for others, with no rule we could find that predicts which.
//! This test therefore asserts on types, and on RMSD only for matched atoms.

use rmda_core::analysis::ptm::{check, Ptm};
use rmda_core::analysis::{Analysis, AnalysisContext, NeighborhoodContext, Summary};
use rmda_core::data::property::PropertyStore;
use rmda_core::exec::Par;
use rmda_core::io::read_frames;
use rmda_core::neighbor::cell_list::CellList;
use std::path::{Path, PathBuf};

mod common;

fn root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

struct Oracle {
    types: Vec<i32>,
    rmsd: Vec<f64>,
}

/// Parse a cached `*.ptm` capture: header, then types, then RMSDs.
fn read_oracle(path: &Path) -> Option<Oracle> {
    let text = std::fs::read_to_string(path).ok()?;
    let mut lines = text.lines();
    lines.next()?;
    let types = lines
        .next()?
        .split_whitespace()
        .map(|v| v.parse().ok())
        .collect::<Option<Vec<i32>>>()?;
    let rmsd = lines
        .next()?
        .split_whitespace()
        .map(|v| v.parse().ok())
        .collect::<Option<Vec<f64>>>()?;
    Some(Oracle { types, rmsd })
}

fn classify(frame_path: &Path) -> Option<(Vec<u8>, Vec<f64>)> {
    let frame = read_frames(frame_path).ok()?.into_iter().next()?;
    let cl = CellList::build(&frame.cell, &frame.particles.positions, 4.0).ok()?;
    let nkeys = NeighborhoodContext { cell: Some(&cl) };
    let par = Par::default();
    let ctx = AnalysisContext {
        frame: &frame,
        nkeys: &nkeys,
        par: &par,
    };
    let mut out = PropertyStore::new();
    let mut summary = Summary::default();
    Ptm {
        rmsd_cutoff: 0.1,
        flags: check::OVITO_DEFAULT,
    }
    .run(&ctx, &mut out, &mut summary)
    .ok()?;
    let types = out.get("StructureType")?.as_u8()?.clone();
    let rmsd = out.get("StructureRmsd")?.as_f64()?.clone();
    Some((types, rmsd))
}

#[test]
#[ignore = "reference-data oracle: set RMDA_REFERENCE_DIR, run with --include-ignored"]
fn matches_ovito_atom_by_atom() {
    let root = root();
    // (file, minimum type agreement as a percentage)
    let cases: &[(&str, f64)] = &[
        ("data.final-lastframe-Al-2000", 99.9),
        ("data.final-lastframe-W-2000", 99.9),
        ("data.final-lastframe-Zr-2000", 99.9),
        ("lastframe-HEA-poly.data", 99.5),
    ];
    let mut ran = 0usize;
    for (name, min_agree) in cases {
        let oracle_path = root.join("target/refcheck").join(format!("{name}.ptm"));
        let Some(oracle) = read_oracle(&oracle_path) else {
            eprintln!(
                "skipping {name}: no oracle at {} (run tools/capture_ptm.py)",
                oracle_path.display()
            );
            continue;
        };
        let frame_path = common::reference_dir().join(name);
        if !frame_path.exists() {
            eprintln!("skipping {name}: {}", frame_path.display());
            continue;
        }
        let Some((types, rmsd)) = classify(&frame_path) else {
            panic!("{name}: failed to classify");
        };
        assert_eq!(types.len(), oracle.types.len(), "{name}: atom count");
        ran += 1;

        // Structure type: the headline agreement.
        let agree = types
            .iter()
            .zip(oracle.types.iter())
            .filter(|(a, b)| **a as i32 == **b)
            .count();
        let pct = 100.0 * agree as f64 / types.len() as f64;
        println!("{name}: type agreement {agree}/{} ({pct:.4}%)", types.len());
        assert!(
            pct >= *min_agree,
            "{name}: type agreement {pct:.4}% below {min_agree}%"
        );

        // RMSD, for atoms both sides matched. These should be bit-identical
        // up to a few ULP; anything larger means the fit convention drifted.
        let mut worst: f64 = 0.0;
        let mut compared = 0usize;
        let mut exact = 0usize;
        let mut outliers: Vec<(usize, f64)> = Vec::new();
        for i in 0..types.len() {
            if types[i] == 0 || oracle.types[i] == 0 {
                continue;
            }
            let d = (rmsd[i] - oracle.rmsd[i]).abs();
            if d > worst {
                worst = d;
            }
            if d < 1e-9 {
                exact += 1;
            } else {
                outliers.push((i, d));
            }
            compared += 1;
        }
        outliers.sort_by(|a, b| b.1.total_cmp(&a.1));
        let agree_pct = 100.0 * exact as f64 / compared as f64;
        println!(
            "  {name}: rmsd over {compared} jointly matched atoms: {exact} agree <1e-9 \
             ({agree_pct:.5}%), {} outliers, worst {worst:.3e}",
            outliers.len()
        );
        // Every observed outlier is also a type disagreement - the same
        // environment difference - and there are 3 across all four snapshots,
        // so this is a tight bound rather than a loose one.
        for (i, d) in outliers.iter().take(4) {
            println!(
                "     outlier atom {i}: our={} {:.6e} ovo={} {:.6e} (diff {d:.3e})",
                types[*i], rmsd[*i], oracle.types[*i], oracle.rmsd[*i]
            );
        }
        assert!(compared > 0, "{name}: no jointly matched atoms");
        assert!(
            agree_pct >= 99.99,
            "{name}: only {agree_pct:.5}% of jointly matched RMSDs agree to 1e-9"
        );
    }
    if ran == 0 {
        eprintln!("skipping: no oracles or snapshots present");
    }
}

/// OVITO's own neighbour order (`NeighbourOrder::Voronoi`: decreasing Voronoi
/// solid angle over the 18 nearest, as `PTMAlgorithm::Kernel::cacheNeighbors`
/// preorders them) removes the residual described above: every atom's type
/// agrees with OVITO, and every matched RMSD to f32 precision (the compact
/// record).
#[test]
#[ignore = "reference-data oracle: set RMDA_REFERENCE_DIR, run with --include-ignored"]
fn voronoi_order_matches_ovito_exactly() {
    let root = root();
    for name in [
        "data.final-lastframe-Al-2000",
        "data.final-lastframe-W-2000",
        "data.final-lastframe-Zr-2000",
        "lastframe-HEA-poly.data",
    ] {
        let oracle_path = root.join("target/refcheck").join(format!("{name}.ptm"));
        let frame_path = common::reference_dir().join(name);
        let (Some(oracle), true) = (read_oracle(&oracle_path), frame_path.exists()) else {
            eprintln!("skipping {name}: oracle or snapshot absent");
            continue;
        };
        let frame = read_frames(&frame_path)
            .unwrap()
            .into_iter()
            .next()
            .unwrap();
        let ptm = Ptm {
            rmsd_cutoff: 0.1,
            flags: check::OVITO_DEFAULT,
        }
        .compute_ordered(
            &frame.cell,
            &frame.particles.positions,
            &Par::default(),
            rmda_core::analysis::ptm::NeighbourOrder::Voronoi,
        )
        .unwrap();
        let differ = ptm
            .iter()
            .zip(&oracle.types)
            .filter(|(p, &t)| p.structure as i32 != t)
            .count();
        let worst = ptm
            .iter()
            .zip(&oracle.rmsd)
            .filter(|(p, _)| p.structure != 0)
            .map(|(p, &r)| (p.rmsd as f64 - r).abs() / r.max(1e-3))
            .fold(0.0f64, f64::max);
        println!("{name}: voronoi order, {differ} types differ, worst relative rmsd {worst:.2e}");
        assert_eq!(differ, 0, "{name}: {differ} atoms differ from OVITO");
        assert!(worst < 1e-5, "{name}: rmsd differs by {worst:e}");
    }
}

/// With OVITO's neighbour order every matched atom's orientation is OVITO's
/// quaternion, component for component (to the f32 of the compact record).
/// HCP needs the order-12 reduction (`HCP_CONVENTIONAL`): over the six D3d
/// generators alone about half of the HCP atoms came out 60° about c from
/// OVITO's, which picks the representative nearest the identity over all
/// twelve. Cubic structures already agreed. The oracle's fourth line (w x y z
/// per atom) is written by `tools/capture_ptm.py`.
#[test]
#[ignore = "reference-data oracle: set RMDA_REFERENCE_DIR, run with --include-ignored"]
fn voronoi_order_matches_ovito_orientations_exactly() {
    let root = root();
    for name in [
        "data.final-lastframe-Al-2000",
        "data.final-lastframe-W-2000",
        "data.final-lastframe-Zr-2000",
        "lastframe-HEA-poly.data",
    ] {
        let oracle_path = root.join("target/refcheck").join(format!("{name}.ptm"));
        let frame_path = common::reference_dir().join(name);
        let q_oracle: Option<Vec<f64>> = std::fs::read_to_string(&oracle_path).ok().and_then(|t| {
            t.lines()
                .nth(3)?
                .split_whitespace()
                .map(|v| v.parse().ok())
                .collect()
        });
        let (Some(q_oracle), true) = (q_oracle, frame_path.exists()) else {
            eprintln!("skipping {name}: orientation oracle or snapshot absent");
            continue;
        };
        let frame = read_frames(&frame_path)
            .unwrap()
            .into_iter()
            .next()
            .unwrap();
        let ptm = Ptm {
            rmsd_cutoff: 0.1,
            flags: check::OVITO_DEFAULT,
        }
        .compute_ordered(
            &frame.cell,
            &frame.particles.positions,
            &Par::default(),
            rmda_core::analysis::ptm::NeighbourOrder::Voronoi,
        )
        .unwrap();
        assert_eq!(q_oracle.len(), 4 * ptm.len(), "{name}");
        let mut worst = 0.0f64;
        let mut matched = 0;
        for (i, p) in ptm.iter().enumerate().filter(|(_, p)| p.structure != 0) {
            let q = p.q();
            for c in 0..4 {
                worst = worst.max((q[c] - q_oracle[4 * i + c]).abs());
            }
            matched += 1;
        }
        println!("{name}: {matched} orientations, worst component difference {worst:.1e}");
        assert!(
            worst < 1e-6,
            "{name}: orientation differs from OVITO by {worst:e}"
        );
    }
}

/// OVITO's `Ordering Type`, `Elastic Deformation Gradient`, `Interatomic
/// Distance` and `Correspondences` (the last only in oracles that
/// include it; `tools/gen_ptm_extras.py --snapshots` writes
/// `target/refcheck/<name>.ptmx`) equal the opt-in columns of
/// `Ptm::run_extended` on every atom, with OVITO's neighbour order. OVITO
/// reports them for the best fit even where the RMSD cutoff rejects it, as we
/// do; all three are zero where nothing matched.
#[test]
#[ignore = "reference-data oracle: set RMDA_REFERENCE_DIR, run with --include-ignored"]
fn voronoi_order_matches_ovito_ordering_and_deformation_gradient() {
    use rmda_core::analysis::ptm::{output, NeighbourOrder};
    let root = root();
    for name in [
        "data.final-lastframe-Al-2000",
        "data.final-lastframe-W-2000",
        "data.final-lastframe-Zr-2000",
        "lastframe-HEA-poly.data",
    ] {
        let oracle_path = root.join("target/refcheck").join(format!("{name}.ptmx"));
        let frame_path = common::reference_dir().join(name);
        let (Ok(raw), true) = (std::fs::read(&oracle_path), frame_path.exists()) else {
            eprintln!("skipping {name}: .ptmx oracle or snapshot absent");
            continue;
        };
        let n = u64::from_le_bytes(raw[..8].try_into().unwrap()) as usize;
        let types = &raw[8..8 + n];
        let ordering = &raw[8 + n..8 + 2 * n];
        let f64s = |off: usize, len: usize| -> Vec<f64> {
            raw[off..off + 8 * len]
                .chunks_exact(8)
                .map(|c| f64::from_le_bytes(c.try_into().unwrap()))
                .collect()
        };
        let want_f = f64s(8 + 2 * n, 9 * n);
        let want_ia = f64s(8 + 2 * n + 72 * n, n);
        let corr_off = 8 + 2 * n + 80 * n;
        let want_corr: Option<Vec<i64>> = (raw.len() >= corr_off + 8 * n).then(|| {
            raw[corr_off..corr_off + 8 * n]
                .chunks_exact(8)
                .map(|c| i64::from_le_bytes(c.try_into().unwrap()))
                .collect()
        });

        let frame = read_frames(&frame_path)
            .unwrap()
            .into_iter()
            .next()
            .unwrap();
        assert_eq!(frame.natoms(), n, "{name}");
        let nkeys = NeighborhoodContext { cell: None };
        let par = Par::default();
        let ctx = AnalysisContext {
            frame: &frame,
            nkeys: &nkeys,
            par: &par,
        };
        let mut out = PropertyStore::new();
        Ptm::default()
            .run_extended(
                &ctx,
                &mut out,
                &mut Summary::default(),
                NeighbourOrder::Voronoi,
                output::ALL,
            )
            .unwrap();
        let st = out.get("StructureType").unwrap().as_u8().unwrap();
        let ord = out.get("StructureOrdering").unwrap().as_u8().unwrap();
        let f = out
            .get("StructureDeformationGradient")
            .unwrap()
            .as_f64()
            .unwrap();
        let ia = out
            .get("StructureInteratomicDistance")
            .unwrap()
            .as_f64()
            .unwrap();
        let corr = out.get("Correspondences").unwrap().as_i64().unwrap();
        let corr_diff = want_corr
            .as_ref()
            .map(|w| (0..n).filter(|&i| corr[i] != w[i]).count());
        let type_diff = (0..n).filter(|&i| st[i] != types[i]).count();
        let ord_diff = (0..n).filter(|&i| ord[i] != ordering[i]).count();
        let mut hist = [0usize; 8];
        for &o in ord {
            hist[o as usize] += 1;
        }
        // Absolute in F (entries are O(1)); relative in the distance.
        let worst_f = (0..9 * n)
            .map(|k| (f[k] - want_f[k]).abs())
            .fold(0.0f64, f64::max);
        let worst_ia = (0..n)
            .map(|i| (ia[i] - want_ia[i]).abs() / want_ia[i].max(1e-3))
            .fold(0.0f64, f64::max);
        println!(
            "{name}: types differ {type_diff}, ordering differs {ord_diff} (ours {hist:?}), \
             worst |dF| {worst_f:.1e}, worst relative ia {worst_ia:.1e}, \
             correspondences differ {corr_diff:?}"
        );
        assert!(
            corr_diff.map_or(true, |d| d == 0),
            "{name}: correspondences differ on {corr_diff:?} atoms"
        );
        assert_eq!(type_diff, 0, "{name}");
        assert_eq!(ord_diff, 0, "{name}: ordering differs on {ord_diff} atoms");
        assert!(worst_f < 1e-9, "{name}: F differs by {worst_f:e}");
        assert!(
            worst_ia < 1e-9,
            "{name}: interatomic distance differs by {worst_ia:e}"
        );
    }
}
