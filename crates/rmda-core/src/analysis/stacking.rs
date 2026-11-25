//! Double-HCP (dHCP, ABAC stacking) from PTM labels (design §28.1).
//!
//! PTM has no dHCP template: in ABAC every close-packed layer is locally FCC
//! or HCP. A layer whose neighbours on both sides are the same letter is
//! hexagonal (PTM: HCP), one between two different letters cubic (PTM: FCC),
//! so in dHCP the A layers are FCC and the B and C layers HCP, alternating
//! (Jagodzinski `hc`). An atom is dHCP when
//!
//! 1. its in-plane neighbours carry its own label and its neighbours in the
//!    layers above and below the other one, and
//! 2. it belongs to a run of at least `2·ERODE + 1` such layers: the set from
//!    (1) is eroded [`ERODE`] times along the stacking (an atom stays while
//!    most of its out-of-plane neighbours do) and grown back as many times
//!    within (1), so a run either survives whole or not at all.
//!
//! (2) is the minimum number of periods: a lone HCP twin plane of an FCC
//! crystal passes (1), and so does a short alternating run in FCC with
//! dense stacking faults (30 % of layers faulted gave one such layer), but
//! neither is a dHCP crystal. FCC with an intrinsic stacking fault (two HCP
//! layers) fails (1), as do FCC and HCP themselves. Only close-packed
//! neighbours count, so the layers next to a free surface are not eroded.
//!
//! Atoms that pass are relabelled [`structure::DHCP`] (a derived code, never
//! produced by PTM itself), with their orientation in the HCP template frame:
//! an HCP atom keeps its own, an FCC atom's is mapped through the
//! close-packed frame map onto the c axis of its layers. The A layers'
//! two FCC twin variants are 60° apart about c, which the HCP group holds,
//! so the whole crystal shares one orientation and segments as one grain.

use crate::analysis::orient;
use crate::analysis::ptm::{structure, PtmAtom};
use crate::cell::{Coord3, UnitCell};
use crate::error::Result;
use crate::exec::Par;
use crate::lattice::fit::quat_to_mat;
use crate::neighbor::{KnnIndex, KnnScratch};

/// Close-packed first shell.
const SHELL: usize = 12;
/// A neighbour is out of an atom's layer when the cosine of its bond to the
/// layer normal exceeds this (in-plane 0, out-of-plane 0.816).
const OUT_OF_PLANE: f64 = 0.4;
/// Neighbours within this many bonds count; at a surface the missing ones
/// are missing, not replaced by second neighbours.
const REACH: f64 = 1.2;
/// Of the neighbours present in-plane (6 in bulk) and out of plane (6), all
/// but this many must agree, and at least `MIN_IN` / `MIN_OUT` be present:
/// a layer next to a basal surface has its 6 in-plane neighbours but only
/// 3 out of plane.
const SLACK: usize = 1;
const MIN_IN: usize = 4;
const MIN_OUT: usize = 3;
/// Erosion steps of (2): a run of `2·ERODE + 1` alternating layers
/// survives, about two ABAC periods.
const ERODE: usize = 3;

/// Relabel the dHCP atoms of `ptm` in place; returns how many.
pub fn relabel_dhcp(
    cell: &UnitCell,
    positions: &[Coord3],
    ptm: &mut [PtmAtom],
    par: &Par,
) -> Result<usize> {
    let n = positions.len();
    let close_packed = |s: i32| s == structure::FCC || s == structure::HCP;
    if !ptm.iter().any(|p| p.structure() == structure::HCP)
        || !ptm.iter().any(|p| p.structure() == structure::FCC)
    {
        return Ok(0);
    }
    let index = KnnIndex::build(cell, positions, par)?;
    // A close-packed atom's layer normal: the c axis of the HCP atoms among
    // it and its neighbours.
    let c_axis = |i: usize| -> Coord3 {
        let r = quat_to_mat(ptm[i].q());
        [r.m[0][2], r.m[1][2], r.m[2][2]]
    };
    struct Local {
        /// Neighbours (atom, unit bond) and the layer normal.
        nbrs: Vec<(u32, Coord3)>,
        normal: Coord3,
        pass: bool,
    }
    let mut local: Vec<Option<Local>> = (0..n).map(|_| None).collect();
    par.fill(&mut local, |i, o| {
        let s = ptm[i].structure();
        if !close_packed(s) {
            return;
        }
        let mut sc = KnnScratch::default();
        let mut nb = Vec::new();
        index.knn(i, positions[i], SHELL, &mut sc, &mut nb);
        if nb.is_empty() {
            return;
        }
        let reach2 = (REACH * f64::from(ptm[i].interatomic_distance)).powi(2);
        // Only close-packed neighbours speak: a surface layer's atoms (OTHER)
        // are absent from the layer under it, not against it.
        let nbrs: Vec<(u32, Coord3)> = nb
            .iter()
            .filter(|e| e.2 < reach2 && close_packed(ptm[e.0 as usize].structure()))
            .map(|e| {
                let l = (e.1[0] * e.1[0] + e.1[1] * e.1[1] + e.1[2] * e.1[2]).sqrt();
                (e.0, [e.1[0] / l, e.1[1] / l, e.1[2] / l])
            })
            .collect();
        let mut normal = [0.0; 3];
        let mut count = 0;
        for j in std::iter::once(i).chain(nbrs.iter().map(|e| e.0 as usize)) {
            if ptm[j].structure() != structure::HCP {
                continue;
            }
            let c = c_axis(j);
            let sgn = if normal[0] * c[0] + normal[1] * c[1] + normal[2] * c[2] < 0.0 {
                -1.0
            } else {
                1.0
            };
            for a in 0..3 {
                normal[a] += sgn * c[a];
            }
            count += 1;
        }
        let len = (normal[0] * normal[0] + normal[1] * normal[1] + normal[2] * normal[2]).sqrt();
        if count == 0 || len < 1e-9 {
            return;
        }
        let normal = normal.map(|v| v / len);
        let (mut n_in, mut same_in, mut n_out, mut other_out) = (0, 0, 0, 0);
        for &(j, u) in &nbrs {
            let t = ptm[j as usize].structure();
            let out = (u[0] * normal[0] + u[1] * normal[1] + u[2] * normal[2]).abs() > OUT_OF_PLANE;
            if out {
                n_out += 1;
                other_out += usize::from(close_packed(t) && t != s);
            } else {
                n_in += 1;
                same_in += usize::from(t == s);
            }
        }
        let pass = n_in >= MIN_IN
            && n_out >= MIN_OUT
            && same_in + SLACK >= n_in
            && other_out + SLACK >= n_out;
        *o = Some(Local { nbrs, normal, pass });
    });
    // Opening along the stacking: erode, then grow back within (1).
    let out_of_plane: Vec<Vec<u32>> = local
        .iter()
        .map(|l| match l {
            Some(l) => l
                .nbrs
                .iter()
                .filter(|&&(_, u)| {
                    (u[0] * l.normal[0] + u[1] * l.normal[1] + u[2] * l.normal[2]).abs()
                        > OUT_OF_PLANE
                })
                .map(|e| e.0)
                .collect(),
            None => Vec::new(),
        })
        .collect();
    let passed: Vec<bool> = local
        .iter()
        .map(|l| l.as_ref().is_some_and(|l| l.pass))
        .collect();
    let mut set = passed.clone();
    for _ in 0..ERODE {
        let prev = set.clone();
        par.fill(&mut set, |i, o| {
            if !prev[i] {
                *o = false;
                return;
            }
            let on = out_of_plane[i]
                .iter()
                .filter(|&&j| prev[j as usize])
                .count();
            // Most of them: 4 of 6 in bulk, 2 of 3 by a basal surface.
            *o = 3 * on >= 2 * out_of_plane[i].len();
        });
    }
    for _ in 0..ERODE {
        let prev = set.clone();
        par.fill(&mut set, |i, o| {
            *o = prev[i] || passed[i] && out_of_plane[i].iter().any(|&j| prev[j as usize]);
        });
    }
    let dhcp = set;
    // Relabel, with orientations in the HCP template frame.
    let cubic = orient::group(structure::FCC).expect("cubic group");
    let m = orient::hcp_to_fcc();
    let mut relabelled = 0;
    for i in 0..n {
        if !dhcp[i] {
            continue;
        }
        let p = &mut ptm[i];
        if p.structure() == structure::FCC {
            let normal = local[i].as_ref().unwrap().normal;
            p.orientation = fcc_in_layer_frame(p.q(), normal, &cubic, m).map(|v| v as f32);
        }
        p.structure = structure::DHCP as u8;
        relabelled += 1;
    }
    Ok(relabelled)
}

/// An FCC orientation in the HCP template frame of its close-packed layers:
/// the cubic variant whose mapped c axis lies along the layers' `normal`.
fn fcc_in_layer_frame(q: [f64; 4], normal: Coord3, cubic: &[[f64; 4]], m: [f64; 4]) -> [f64; 4] {
    cubic
        .iter()
        .map(|g| orient::qmul(orient::qmul(q, *g), m))
        .max_by(|a, b| {
            let along = |x: &[f64; 4]| {
                let r = quat_to_mat(*x);
                (r.m[0][2] * normal[0] + r.m[1][2] * normal[1] + r.m[2][2] * normal[2]).abs()
            };
            along(a).total_cmp(&along(b))
        })
        .unwrap()
}

/// Longest run of non-dHCP close-packed layers between dHCP layers that is a
/// stacking fault of the dHCP rather than another phase: one layer short of
/// the shortest dHCP run [`relabel_dhcp`] accepts, so the two definitions
/// meet.
pub const MAX_FAULT_LAYERS: usize = 2 * ERODE;

/// Close-packed atoms (FCC or HCP labelled) in a stacking fault of a dHCP
/// crystal — a run of at most [`MAX_FAULT_LAYERS`] non-dHCP layers with dHCP
/// layers on both sides — with their orientation in the dHCP (HCP template)
/// frame: an HCP atom's own, an FCC atom's mapped onto the c axis of its
/// layers as [`relabel_dhcp`] maps dHCP's FCC layers. Grain segmentation
/// (`SegConfig::dhcp_faults`) treats these, and only these, as dHCP atoms; a
/// thicker FCC or HCP region next to dHCP is another phase and keeps its own
/// grain.
///
/// The distance in layers to dHCP along each sense of the layer normal is a
/// shortest path over close-packed out-of-plane bonds (dHCP at 0, every other
/// atom blocks), so the run is measured however the stacking is shifted. An
/// atom's normal is that of the HCP and dHCP atoms among it and its
/// neighbours, as in [`relabel_dhcp`], spread in a few rounds to FCC atoms
/// with none (the middle of a cubic fault).
pub fn dhcp_faults(
    cell: &UnitCell,
    positions: &[Coord3],
    ptm: &[PtmAtom],
    par: &Par,
) -> Result<Vec<Option<[f64; 4]>>> {
    let n = positions.len();
    if !ptm.iter().any(|p| p.structure() == structure::DHCP) {
        return Ok(vec![None; n]);
    }
    let packed = |s: i32| s == structure::FCC || s == structure::HCP || s == structure::DHCP;
    let index = KnnIndex::build(cell, positions, par)?;
    // Close-packed neighbours within REACH bonds, with unit bond vectors.
    let mut nbrs: Vec<Vec<(u32, Coord3)>> = vec![Vec::new(); n];
    par.fill(&mut nbrs, |i, o| {
        if !packed(ptm[i].structure()) {
            return;
        }
        let mut sc = KnnScratch::default();
        let mut nb = Vec::new();
        index.knn(i, positions[i], SHELL, &mut sc, &mut nb);
        let reach2 = (REACH * f64::from(ptm[i].interatomic_distance)).powi(2);
        *o = nb
            .iter()
            .filter(|e| e.2 < reach2 && packed(ptm[e.0 as usize].structure()))
            .map(|e| {
                let l = (e.1[0] * e.1[0] + e.1[1] * e.1[1] + e.1[2] * e.1[2]).sqrt();
                (e.0, [e.1[0] / l, e.1[1] / l, e.1[2] / l])
            })
            .collect();
    });
    let dot = |a: Coord3, b: Coord3| a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
    // Normals: the c axis of hexagonal atoms (HCP, and dHCP in the HCP
    // frame), then averaged into atoms without one.
    let mut normal: Vec<Option<Coord3>> = (0..n)
        .map(|i| {
            let s = ptm[i].structure();
            (s == structure::HCP || s == structure::DHCP).then(|| {
                let r = quat_to_mat(ptm[i].q());
                [r.m[0][2], r.m[1][2], r.m[2][2]]
            })
        })
        .collect();
    for _ in 0..=MAX_FAULT_LAYERS {
        let prev = normal.clone();
        par.fill(&mut normal, |i, o| {
            if prev[i].is_some() || nbrs[i].is_empty() {
                *o = prev[i];
                return;
            }
            let mut acc = [0.0; 3];
            for &(j, _) in &nbrs[i] {
                if let Some(c) = prev[j as usize] {
                    let s = if dot(acc, c) < 0.0 { -1.0 } else { 1.0 };
                    for a in 0..3 {
                        acc[a] += s * c[a];
                    }
                }
            }
            let l = dot(acc, acc).sqrt();
            *o = (l > 1e-9).then(|| acc.map(|v| v / l));
        });
    }
    // Layers to dHCP along +normal and −normal.
    const FAR: u32 = u32::MAX / 4;
    let init: Vec<[u32; 2]> = (0..n)
        .map(|i| {
            if ptm[i].structure() == structure::DHCP {
                [0, 0]
            } else {
                [FAR, FAR]
            }
        })
        .collect();
    let mut dist = init.clone();
    for _ in 0..=MAX_FAULT_LAYERS {
        let prev = dist.clone();
        par.fill(&mut dist, |i, o| {
            *o = prev[i];
            let s = ptm[i].structure();
            if s == structure::DHCP || !packed(s) {
                return;
            }
            let Some(ni) = normal[i] else { return };
            for &(j, u) in &nbrs[i] {
                let along = dot(u, ni);
                if along.abs() <= OUT_OF_PLANE {
                    continue;
                }
                let j = j as usize;
                // Continue in the same sense: j's +normal may point the
                // other way.
                let flip = normal[j].map_or(false, |nj| dot(nj, ni) < 0.0);
                let (mine, theirs) = if along > 0.0 { (0, 0) } else { (1, 1) };
                let theirs = if flip { 1 - theirs } else { theirs };
                o[mine] = o[mine].min(prev[j][theirs].saturating_add(1));
            }
        });
    }
    let cubic = orient::group(structure::FCC).expect("cubic group");
    let m = orient::hcp_to_fcc();
    Ok((0..n)
        .map(|i| {
            let s = ptm[i].structure();
            if !((s == structure::FCC || s == structure::HCP)
                && dist[i][0] + dist[i][1] <= MAX_FAULT_LAYERS as u32 + 1)
            {
                return None;
            }
            Some(if s == structure::HCP {
                ptm[i].q()
            } else {
                fcc_in_layer_frame(ptm[i].q(), normal[i]?, &cubic, m)
            })
        })
        .collect())
}
