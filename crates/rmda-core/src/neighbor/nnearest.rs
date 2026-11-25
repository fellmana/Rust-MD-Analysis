//! Fixed-k nearest neighbours (design §9.2).
//!
//! Doubles a probe radius over the cell list's build-time stencil until it
//! holds at least `k` candidates (or a wider probe adds none), then returns the
//! `k` closest, sorted by distance (closest first).

use crate::error::Result;
use crate::neighbor::cell_list::CellList;

/// Return up to `k` nearest neighbours of atom `i` as `(atom_id, min_image
/// displacement, squared distance)` in ascending distance order.
///
/// `k==0` returns an empty list. Doubles the probe radius from the bin size
/// until at least `k` image-pair candidates fall inside it (or a wider probe
/// adds none), then sorts them.
pub fn nearest_k(
    cl: &CellList,
    i: usize,
    k: usize,
    out: &mut Vec<(u32, [f64; 3], f64)>,
) -> Result<()> {
    out.clear();
    if k == 0 {
        return Ok(());
    }
    // Grow a search radius until k *image-pair* candidates are inside it.
    // Using `neighbor_entries` (rather than a single minimum image per atom)
    // is what keeps this correct when the cutoff exceeds half a cell width:
    // a neighbour can then be reached through several images, and each is a
    // genuine k-NN candidate. Since the returned set is exactly the pairs
    // within the radius, the k smallest are the true k nearest as soon as
    // there are at least k of them.
    let mut radius = cl.binsize().max(f64::MIN_POSITIVE);
    let mut entries: Vec<(usize, [f64; 3], [i64; 3])> = Vec::new();
    let mut last_seen = 0usize;
    loop {
        entries.clear();
        cl.neighbor_entries(i, radius, &mut entries)?;
        if entries.len() >= k {
            break;
        }
        // Stop when a wider probe adds no new candidates. `neighbor_entries`
        // filters a build-time stencil by radius, so its output is monotone in
        // the radius *and bounded by the stencil*: once two probes return the
        // same set, no larger radius can ever add more — they would only
        // re-walk the same bins (doubling on to 1e12 would be ~38 identical
        // stencil walks per starved atom).
        if entries.len() == last_seen {
            break;
        }
        last_seen = entries.len();
        if !radius.is_finite() || radius > 1e12 {
            break;
        }
        radius *= 2.0;
    }
    entries.sort_by(|a, b| {
        let da = a.1[0] * a.1[0] + a.1[1] * a.1[1] + a.1[2] * a.1[2];
        let db = b.1[0] * b.1[0] + b.1[1] * b.1[1] + b.1[2] * b.1[2];
        da.total_cmp(&db)
            .then_with(|| a.0.cmp(&b.0))
            .then_with(|| a.2[0].cmp(&b.2[0]))
    });
    out.reserve(k.min(entries.len()));
    for (j, dr, _) in entries.iter().take(k) {
        out.push((
            *j as u32,
            *dr,
            dr[0] * dr[0] + dr[1] * dr[1] + dr[2] * dr[2],
        ));
    }
    Ok(())
}

/// Like [`nearest_k`], but never returns the query atom itself.
///
/// [`nearest_k`] counts every image of atom `i` as a candidate, including the
/// zero-shift copy of itself and any periodic images of itself. Those are not
/// neighbours of anything, and in a cell small enough to bring an image back
/// close they sort to the front: a 2x2x2 hexagonal cell puts images of the
/// reference atom at distance ~0, so a plain k-NN is filled with ghosts.
///
/// CNA only needs a rough 12/14 and works around this by testing in larger
/// cells. PTM cannot: its references read a fixed-length prefix of the nearest
/// neighbours, so a single ghost at distance 0 both displaces a real neighbour
/// out of the prefix and makes the match meaningless. Hence this variant,
/// which filters the query atom out of the candidate set *before* selecting.
pub fn nearest_k_excluding_self(
    cl: &CellList,
    i: usize,
    k: usize,
    out: &mut Vec<(u32, [f64; 3], f64)>,
) -> Result<()> {
    out.clear();
    if k == 0 {
        return Ok(());
    }
    // One probe at `2 * binsize`, then fall back to doubling if that is still
    // not enough.
    //
    // `neighbor_entries` fixes its bin stencil at build time and uses the cutoff
    // only to filter what that stencil already returned, so every extra probe
    // re-walks identical bins and redoes identical per-candidate minimum-image
    // work in order to loosen a filter. Doubling from `binsize` would make the
    // 500k Al snapshot pay two stencil walks for every atom at k=18:
    // 14 neighbours at binsize, then 97 candidates at 8.0065 A, ~80 of them
    // discarded after full price. The walk dominates, not the sort, so the win is
    // in taking one probe instead of two.
    //
    // Starting at `2 * binsize` rather than at a density-derived radius is
    // deliberate. `r = (3k / 4*pi*rho)^(1/3)` predicts where the k-th neighbour
    // sits, but it assumes an ideal gas and so undercounts against a
    // close-packed lattice; tuning its margin only traded one guess for another
    // and never beat simply skipping the probe that was going to be thrown away.
    // `2 * binsize` is also the second probe of a doubling from `binsize`, so
    // every cutoff reachable here is one that doubling would reach too.
    let mut radius = cl.binsize().max(f64::MIN_POSITIVE) * 2.0;

    let mut entries: Vec<(usize, [f64; 3], [i64; 3])> = Vec::new();
    let mut last_seen = 0usize;
    loop {
        entries.clear();
        cl.neighbor_entries(i, radius, &mut entries)?;
        // Drop every image of the query atom, then judge sufficiency on what is
        // left rather than on the raw pair count.
        entries.retain(|e| e.0 != i);
        if entries.len() >= k {
            break;
        }
        // Same monotone-stencil stop as [`nearest_k`]: the build-time stencil
        // bounds what any probe can return, so a probe that gains nothing never
        // will, and doubling from `2 * binsize` to 1e12 would be ~38 re-walks of
        // the same bins per starved atom.
        if entries.len() == last_seen {
            break;
        }
        last_seen = entries.len();
        if !radius.is_finite() || radius > 1e12 {
            break;
        }
        radius *= 2.0;
    }

    // Sort on a precomputed squared distance: recomputing both sides in the
    // comparator would pay ~6 multiplies per comparison over the whole
    // candidate set.
    let mut keyed: Vec<(f64, usize, [f64; 3], [i64; 3])> = entries
        .into_iter()
        .map(|(j, dr, img)| {
            let d2 = dr[0] * dr[0] + dr[1] * dr[1] + dr[2] * dr[2];
            (d2, j, dr, img)
        })
        .collect();
    // Select the k smallest, then sort only those. The candidate set is ~25x k
    // at the default probe (k = 18, ~450 candidates), so a full sort spent most
    // of its time ordering entries that are thrown away.
    //
    // The order is the one a stable `sort_by` on (d2, j, image x) produces:
    // entries tied on all three are images of one atom differing in y/z
    // shift, which `scan_within` emits in ascending (y, z) shift order, so a
    // stable sort leaves them in that order. Comparing the y and z shifts too
    // makes the key a strict total order that reproduces it, and with a strict
    // total order an unstable selection gives exactly the same prefix.
    let key = |a: &(f64, usize, [f64; 3], [i64; 3]), b: &(f64, usize, [f64; 3], [i64; 3])| {
        a.0.total_cmp(&b.0)
            .then_with(|| a.1.cmp(&b.1))
            .then_with(|| a.3.cmp(&b.3))
    };
    if keyed.len() > k {
        keyed.select_nth_unstable_by(k - 1, key);
        keyed.truncate(k);
    }
    keyed.sort_unstable_by(key);
    out.reserve(k.min(keyed.len()));
    for (d2, j, dr, _) in keyed.iter().take(k) {
        out.push((*j as u32, *dr, *d2));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cell::UnitCell;

    fn simple_grid(n: usize, a: f64) -> Vec<[f64; 3]> {
        let mut v = Vec::new();
        for i in 0..n {
            for j in 0..n {
                for k in 0..n {
                    v.push([i as f64 * a, j as f64 * a, k as f64 * a]);
                }
            }
        }
        v
    }

    #[test]
    fn finds_k_nearest_in_grid() {
        let n = 4;
        let a = 2.0;
        let cell = UnitCell::orthorhombic(n as f64 * a, n as f64 * a, n as f64 * a);
        let pos = simple_grid(n, a);
        let cl = CellList::build(&cell, &pos, 2.5).unwrap();
        let mut out = Vec::new();
        nearest_k(&cl, 0, 6, &mut out).unwrap();
        // atom 0 at a grid corner: 6 neighbours at distance 2.0
        assert_eq!(out.len(), 6);
        for (_, _, d2) in &out {
            assert!((d2 - 4.0).abs() < 1e-9, "expected d2=4, got {d2}");
        }
        assert!(out.windows(2).all(|w| w[0].2 <= w[1].2));
    }

    #[test]
    fn k_exceeds_available_counts_images() {
        // A 2 Å box with two atoms at (0,0,0) and (1,0,0). The neighbour is
        // reachable through two images at 1 Å, then eight at sqrt(5) Å and two
        // at 3 Å. Counting one entry per *atom* would stop at 1 and silently
        // under-fill the k-list for small cells, so k=12 must be satisfiable.
        let cell = UnitCell::orthorhombic(2.0, 2.0, 2.0);
        let pos = vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]];
        let cl = CellList::build(&cell, &pos, 1.5).unwrap();
        let mut out = Vec::new();
        nearest_k(&cl, 0, 12, &mut out).unwrap();
        assert_eq!(out.len(), 12);
        let dists: Vec<f64> = out
            .iter()
            .map(|(_, _, d2)| (d2.sqrt() * 1e6).round() / 1e6)
            .collect();
        let want: [f64; 12] = [
            1.0,
            1.0, // two images at 1 A
            5f64.sqrt(),
            5f64.sqrt(),
            5f64.sqrt(),
            5f64.sqrt(), // (x,y) images
            5f64.sqrt(),
            5f64.sqrt(),
            5f64.sqrt(),
            5f64.sqrt(), // (x,z) images
            3.0,
            3.0,
        ];
        for (n, (got, exp)) in dists.iter().zip(want.iter()).enumerate() {
            assert!(
                (got - exp).abs() < 1e-6,
                "entry {n}: {got} vs {exp} in {dists:?}"
            );
        }
        assert!(
            out.windows(2).all(|w| w[0].2 <= w[1].2),
            "not sorted ascending"
        );
    }

    #[test]
    fn k_counts_distinct_atoms_when_images_are_far() {
        // With a roomy cell the k nearest are distinct atoms, and the closest
        // is the real neighbour, not one of its far images.
        let cell = UnitCell::orthorhombic(20.0, 20.0, 20.0);
        let pos = vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.2, 0.0, 0.0]];
        let cl = CellList::build(&cell, &pos, 1.5).unwrap();
        let mut out = Vec::new();
        nearest_k(&cl, 0, 2, &mut out).unwrap();
        assert_eq!(out.len(), 2);
        assert!(
            (out[0].2.sqrt() - 1.0).abs() < 1e-9,
            "closest is not the 1 A neighbour"
        );
        assert!(
            (out[1].2.sqrt() - 1.2).abs() < 1e-9,
            "second is not the 1.2 A neighbour"
        );
    }

    /// The selection-based [`nearest_k_excluding_self`] must return exactly
    /// what a full stable sort does, entry for entry and bit for bit,
    /// including tiny cells where one atom appears through several images at
    /// equal distance (the ties the extended key exists for).
    #[test]
    fn selection_reproduces_the_full_stable_sort() {
        type Entry = (u32, [f64; 3], f64);
        fn reference(cl: &CellList, i: usize, k: usize) -> Vec<Entry> {
            let mut radius = cl.binsize().max(f64::MIN_POSITIVE) * 2.0;
            let mut entries: Vec<(usize, [f64; 3], [i64; 3])> = Vec::new();
            let mut last_seen = 0usize;
            loop {
                entries.clear();
                cl.neighbor_entries(i, radius, &mut entries).unwrap();
                entries.retain(|e| e.0 != i);
                if entries.len() >= k || entries.len() == last_seen || radius > 1e12 {
                    break;
                }
                last_seen = entries.len();
                radius *= 2.0;
            }
            let mut keyed: Vec<(f64, usize, [f64; 3], [i64; 3])> = entries
                .into_iter()
                .map(|(j, dr, img)| (dr[0] * dr[0] + dr[1] * dr[1] + dr[2] * dr[2], j, dr, img))
                .collect();
            keyed.sort_by(|a, b| {
                a.0.total_cmp(&b.0)
                    .then_with(|| a.1.cmp(&b.1))
                    .then_with(|| a.3[0].cmp(&b.3[0]))
            });
            keyed
                .iter()
                .take(k)
                .map(|(d2, j, dr, _)| (*j as u32, *dr, *d2))
                .collect()
        }
        fn bits(v: &[Entry]) -> Vec<(u32, [u64; 3], u64)> {
            v.iter()
                .map(|(j, d, d2)| {
                    (
                        *j,
                        [d[0].to_bits(), d[1].to_bits(), d[2].to_bits()],
                        d2.to_bits(),
                    )
                })
                .collect()
        }
        let mut s = 99u64;
        let mut next = || {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (s >> 11) as f64 / (1u64 << 53) as f64
        };
        // A lattice (exact ties) and a jittered one, in a roomy cell and in
        // cells small enough that images of one atom tie with each other.
        for (ncell, a, jitter, cutoff) in [
            (4, 2.0, 0.0, 2.5),
            (4, 2.0, 0.2, 2.5),
            (2, 1.5, 0.0, 2.5),
            (3, 1.0, 0.0, 2.5),
        ] {
            let mut pos = simple_grid(ncell, a);
            for p in pos.iter_mut() {
                for v in p.iter_mut() {
                    *v += jitter * (next() - 0.5);
                }
            }
            let l = ncell as f64 * a;
            let cell = UnitCell::orthorhombic(l, l, l);
            let cl = CellList::build(&cell, &pos, cutoff).unwrap();
            let mut out = Vec::new();
            for k in [1, 6, 12, 18, 26] {
                for i in 0..pos.len() {
                    nearest_k_excluding_self(&cl, i, k, &mut out).unwrap();
                    assert_eq!(
                        bits(&out),
                        bits(&reference(&cl, i, k)),
                        "ncell {ncell} k {k} atom {i}"
                    );
                }
            }
        }
    }
}
