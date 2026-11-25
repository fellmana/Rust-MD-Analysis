//! Antisites in multi-species frames (design §16.1a).
//!
//! An antisite is a **singly occupied** site whose one atom is not of the
//! site's species: OVITO's `WignerSeitzAnalysisModifier` with
//! `per_type_occupancies`, read as "total occupancy 1, and the occupancy of
//! the site's own type 0". Vacant and over-occupied sites are vacancies and
//! interstitials and are never antisites, so antisites are counted beside
//! V and I without touching them: the assignment is the census's own.
//!
//! The site's species comes from
//! - an explicit reference: the type of the reference atom that is the site;
//! - a fitted lattice (global or patch): the majority type of the occupants of
//!   the singly occupied sites on the same motif basis site `k`, per grain.
//!   This is sound only when every chemical sublattice is a union of the
//!   motif's basis sublattices (B2 on BCC's two, L1₂ and L1₀ on FCC's four);
//!   a chemical cell larger than the conventional cell (D0₃, L2₁, D0₁₉) or an
//!   antiphase boundary inside the grain is not representable, and shows as a
//!   large antisite count. The species per basis site are reported, so a
//!   mixed sublattice is visible.
//! - the local census: the same majority vote per vote region (a cell of
//!   about ten bonds with a margin), each labelled by its own fitted lattice
//!   (`local/antisites.rs`).

use crate::cell::Coord3;
use crate::data::elements::type_names;
use crate::data::frame::Frame;
use std::collections::BTreeMap;

/// One grain's antisites.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct GrainAntisites {
    /// Antisites on interior and on boundary sites.
    pub interior: u64,
    pub boundary: u64,
    /// `(site type, occupant type)` → `[interior, boundary]`.
    pub pairs: BTreeMap<(i32, i32), [u64; 2]>,
    /// Fitted reference: the species of each motif basis site (−1 for a basis
    /// site with no singly occupied site). Empty for an explicit reference.
    pub sublattice: Vec<i32>,
    /// Fitted reference: the chemical order the count rests on (Cramér's V,
    /// [`chemical_order`]); 1 for an explicit reference.
    pub order: f64,
    /// Fitted reference: the smallest majority share of a basis site
    /// ([`majority_share`]); 1 for an explicit reference.
    pub share: f64,
}

impl GrainAntisites {
    pub(crate) fn add(&mut self, site_type: i32, occupant: i32, boundary: bool) {
        let e = self.pairs.entry((site_type, occupant)).or_insert([0; 2]);
        if boundary {
            self.boundary += 1;
            e[1] += 1;
        } else {
            self.interior += 1;
            e[0] += 1;
        }
    }
}

/// An antisite atom of a measured grain.
#[derive(Debug, Clone, Copy)]
pub struct Antisite {
    pub atom: u32,
    /// The species the site should hold.
    pub site_type: i32,
    /// The site's position (the reference atom's for an explicit reference,
    /// reported like a vacancy's).
    pub site_position: Coord3,
    pub grain: i32,
    pub boundary: bool,
}

/// Antisites are counted only in a chemically ordered crystal: Cramér's V
/// of its (basis site, type) occupancy at least this. A random alloy (an
/// HEA) has the same composition on every basis site (V ~ 0), where an
/// "antisite" is any atom not of its basis site's majority type: 80 % of a
/// random five-species FCC block. Ordered alloys read near 1 (B2 with 10 %
/// antisites ~0.8).
pub(crate) const ORDER_MIN: f64 = 0.3;

/// ... or every basis site has a dominant species: its majority type holds
/// at least this share of its singly occupied sites. A dilute solid solution
/// (W with 5 % Re) has no chemical order but a host species on every site,
/// and its solutes on host sites are antisites; an equiatomic random alloy
/// (majority ~1/k) has neither.
pub(crate) const SHARE_MIN: f64 = 0.75;

/// The smallest share any basis site's majority type has of its singly
/// occupied sites (1 when there are none).
pub(crate) fn majority_share(occupants: impl Iterator<Item = (u8, i32)>) -> f64 {
    let mut counts: BTreeMap<(u8, i32), f64> = BTreeMap::new();
    for key in occupants {
        *counts.entry(key).or_insert(0.0) += 1.0;
    }
    let mut per: BTreeMap<u8, (f64, f64)> = BTreeMap::new();
    for (&(k, _), &c) in &counts {
        let e = per.entry(k).or_insert((0.0, 0.0));
        e.0 = e.0.max(c);
        e.1 += c;
    }
    per.values().map(|&(m, n)| m / n).fold(1.0, f64::min)
}

/// Whether antisites are defined: chemical order, or a dominant species on
/// every basis site.
pub(crate) fn antisites_defined(order: f64, share: f64) -> bool {
    order >= ORDER_MIN || share >= SHARE_MIN
}

/// Cramér's V of the occupancy (basis site, type): 0 for the same
/// composition on every basis site, 1 for one type per basis site. One type
/// present: 1 (nothing can be an antisite); one basis site: 0.
pub(crate) fn chemical_order(occupants: impl Iterator<Item = (u8, i32)>) -> f64 {
    let mut counts: BTreeMap<(u8, i32), f64> = BTreeMap::new();
    for key in occupants {
        *counts.entry(key).or_insert(0.0) += 1.0;
    }
    let mut rows: BTreeMap<u8, f64> = BTreeMap::new();
    let mut cols: BTreeMap<i32, f64> = BTreeMap::new();
    for (&(k, t), &c) in &counts {
        *rows.entry(k).or_insert(0.0) += c;
        *cols.entry(t).or_insert(0.0) += c;
    }
    let n: f64 = rows.values().sum();
    if cols.len() <= 1 {
        return 1.0;
    }
    if rows.len() <= 1 || n == 0.0 {
        return 0.0;
    }
    let mut chi2 = 0.0;
    for (&k, &r) in &rows {
        for (&t, &c) in &cols {
            let e = r * c / n;
            let o = counts.get(&(k, t)).copied().unwrap_or(0.0);
            chi2 += (o - e) * (o - e) / e;
        }
    }
    let m = (rows.len().min(cols.len()) - 1) as f64;
    (chi2 / (n * m)).sqrt()
}

/// The species of each of `nb` basis sites: the majority occupant type over
/// `(basis site, type)` pairs of singly occupied sites, ties to the lower
/// type; −1 for a basis site with none.
pub(crate) fn sublattice_species(
    nb: usize,
    occupants: impl Iterator<Item = (u8, i32)>,
) -> Vec<i32> {
    let mut counts: BTreeMap<(u8, i32), u64> = BTreeMap::new();
    for key in occupants {
        *counts.entry(key).or_insert(0) += 1;
    }
    let mut best = vec![(0u64, -1i32); nb];
    // Ascending (k, type): a later type replaces only on a strictly larger
    // count, so ties go to the lower type.
    for (&(k, t), &c) in &counts {
        let b = &mut best[k as usize];
        if c > b.0 {
            *b = (c, t);
        }
    }
    best.into_iter().map(|b| b.1).collect()
}

/// Per-atom types (as `i32`) of a frame for antisites, with a name per type
/// (index type − 1): `elements`, else the file's labels, else by mass.
pub fn frame_types(f: &Frame, elements: &[String]) -> (Vec<i32>, Vec<String>) {
    let t: Vec<i32> = f.particles.types.iter().map(|&t| t as i32).collect();
    let ntypes = t.iter().copied().max().unwrap_or(0).max(0) as usize;
    let names = type_names(elements, &f.type_labels, &f.type_masses, ntypes);
    (t, names)
}

/// Types of an explicit reference's sites (`rf`'s atoms) and of the current
/// atoms (`f`) in one numbering, with a name per type. When both frames name
/// their types and the names differ in order, the current types are matched
/// to the reference's by name (names the reference lacks are numbered after
/// its own); otherwise type numbers are compared as they are.
pub fn matched_types(
    rf: &Frame,
    f: &Frame,
    elements: &[String],
) -> (Vec<i32>, Vec<i32>, Vec<String>) {
    let sites: Vec<i32> = rf.particles.types.iter().map(|&t| t as i32).collect();
    let (rl, fl) = (&rf.type_labels, &f.type_labels);
    if !rl.is_empty() && !fl.is_empty() && rl != fl {
        let mut names = rl.clone();
        let map: Vec<i32> = fl
            .iter()
            .map(|l| match names.iter().position(|n| n == l) {
                Some(k) => k as i32 + 1,
                None => {
                    names.push(l.clone());
                    names.len() as i32
                }
            })
            .collect();
        let atoms = f
            .particles
            .types
            .iter()
            .map(|&t| {
                (t as usize)
                    .checked_sub(1)
                    .and_then(|k| map.get(k).copied())
                    .unwrap_or(t as i32)
            })
            .collect();
        let names = type_names(elements, &names, &[], names.len());
        return (sites, atoms, names);
    }
    let atoms: Vec<i32> = f.particles.types.iter().map(|&t| t as i32).collect();
    let ntypes = sites
        .iter()
        .chain(&atoms)
        .copied()
        .max()
        .unwrap_or(0)
        .max(0) as usize;
    let (labels, masses) = if rl.is_empty() {
        (fl, &f.type_masses)
    } else {
        (rl, &rf.type_masses)
    };
    let names = type_names(elements, labels, masses, ntypes);
    (sites, atoms, names)
}

/// A type's label in tables and keys: its name when every name is known and
/// unique, else its number.
pub fn type_label(names: &[String], t: i32) -> String {
    let named = names.iter().all(|n| !n.is_empty() && n != "X")
        && names
            .iter()
            .collect::<std::collections::BTreeSet<_>>()
            .len()
            == names.len();
    match usize::try_from(t - 1).ok().and_then(|k| names.get(k)) {
        Some(n) if named => n.clone(),
        _ => t.to_string(),
    }
}

/// The species of each motif basis site, space separated (`-` for none).
pub fn sublattice_label(names: &[String], s: &[i32]) -> String {
    s.iter()
        .map(|&t| {
            if t < 0 {
                "-".to_owned()
            } else {
                type_label(names, t)
            }
        })
        .collect::<Vec<_>>()
        .join(" ")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn majority_per_basis_site() {
        let occ = [(0u8, 2), (0, 2), (0, 1), (1, 1), (1, 1), (1, 2), (1, 2)];
        assert_eq!(sublattice_species(3, occ.into_iter()), vec![2, 1, -1]);
    }

    #[test]
    fn counts_by_class_and_pair() {
        let mut a = GrainAntisites::default();
        a.add(1, 2, false);
        a.add(1, 2, true);
        a.add(2, 1, false);
        assert_eq!((a.interior, a.boundary), (2, 1));
        assert_eq!(a.pairs[&(1, 2)], [1, 1]);
        assert_eq!(a.pairs[&(2, 1)], [1, 0]);
    }

    #[test]
    fn chemical_order_tells_an_ordered_alloy_from_a_random_one() {
        // Two basis sites, two types. Random: the same composition on both.
        let random = (0..4000u32).map(|i| ((i % 2) as u8, ((i / 2 * 7919) % 2) as i32 + 1));
        assert!(chemical_order(random) < 0.05);
        // Ordered (B2): one type per basis site, then 10 % antisites.
        let ordered = (0..4000u32).map(|i| ((i % 2) as u8, (i % 2) as i32 + 1));
        assert!((chemical_order(ordered) - 1.0).abs() < 1e-12);
        let ten = (0..4000u32).map(|i| {
            let k = (i % 2) as u8;
            let t = if i % 20 < 2 { 1 - k as i32 } else { k as i32 };
            (k, t + 1)
        });
        assert!(chemical_order(ten) > 0.7);
        // One type: nothing can be an antisite.
        assert_eq!(chemical_order((0..10u32).map(|i| ((i % 2) as u8, 1))), 1.0);
        // A dilute random solution: no order, but a host on every site.
        let dilute = |i: u32| ((i % 2) as u8, if i % 9 == 0 { 2 } else { 1 });
        let v = chemical_order((0..4000u32).map(dilute));
        let sh = majority_share((0..4000u32).map(dilute));
        assert!(v < 0.05 && sh > 0.85 && antisites_defined(v, sh));
        // Equiatomic five-species random alloy: neither.
        let hea = |i: u32| ((i % 4) as u8, ((i / 4 * 7919) % 5) as i32);
        let v = chemical_order((0..20000u32).map(hea));
        let sh = majority_share((0..20000u32).map(hea));
        assert!(!antisites_defined(v, sh), "{v} {sh}");
    }
}
