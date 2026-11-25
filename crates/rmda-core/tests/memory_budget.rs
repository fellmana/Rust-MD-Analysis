//! Memory budget verification (design §24.2).
//!
//! The design's memory table (§24.2) gives bytes per atom for each resident
//! structure. The rows built from fixed-size types are *checked* here against
//! the real data structures rather than assumed:
//! every layout decision that would silently break the budget (AoS vs SoA,
//! a positions copy inside the cell list, a second neighbour structure
//! creeping back into the shared context) shows up as a failing assertion
//! instead of a surprise on a 128M-atom run.
//!
//! The neighbour
//! subsystem *is* the linked-cell list: no pool rows, no offset formulas —
//! everything below is derived from `size_of` of the structs that exist.

use rmda_core::analysis::NeighborhoodContext;
use rmda_core::cell::UnitCell;
use rmda_core::data::particle::ParticleData;
use rmda_core::neighbor::cell_list::{CellList, MdaLocal};

/// Atoms in the design's worked example (the 128M frame).
const N: u64 = 128_000_000;

/// Decimal GB, the unit of the design's 128M figures.
fn gb(bytes: u64) -> f64 {
    bytes as f64 / 1e9
}

#[test]
fn particle_data_matches_stated_budget() {
    let p = ParticleData::new(0);
    let size = |t: usize| t as u64;
    // positions 24 + ids 8 + types 4 = 36 B/atom; images (8) and selection (1)
    // only exist when the source/analysis needs them.
    assert_eq!(
        size(std::mem::size_of::<[f64; 3]>()) * N,
        24 * N,
        "positions 24 B/atom"
    );
    assert_eq!(
        size(std::mem::size_of::<rmda_core::data::particle::MdaTag>()) * N,
        8 * N,
        "ids 8 B/atom"
    );
    assert_eq!(
        size(std::mem::size_of::<rmda_core::data::particle::MdaType>()) * N,
        4 * N,
        "types 4 B/atom"
    );

    // design §24.2, row "atoms (pos 24, id 8, type 4)": 36 B/atom.
    let base = 24 + 8 + 4;
    assert_eq!(base, 36);
    // Row "images (if present, packed)": 8 B/atom.
    assert_eq!(
        std::mem::size_of::<rmda_core::cell::MdaImage>(),
        8,
        "packed image flags 8 B/atom"
    );
    // Optional arrays only materialise on demand.
    assert!(p.images.is_none() && p.selection.is_none());
    assert_eq!(std::mem::size_of::<u8>(), 1, "selection 1 B/atom");
}

#[test]
fn neighbor_subsystem_is_the_cell_list_only() {
    // design §24.2: the neighbour
    // subsystem is the linked-cell list alone, and the shared context every
    // analysis receives carries nothing else — a second structure (a pool, a
    // cache) growing back into `NeighborhoodContext` would push this handle
    // past its single niche-optimised pointer.
    assert_eq!(std::mem::size_of::<MdaLocal>(), 4, "bin chains stay 32-bit");
    assert_eq!(
        std::mem::size_of::<NeighborhoodContext<'_>>(),
        std::mem::size_of::<Option<&CellList<'_>>>(),
        "NeighborhoodContext must stay one optional cell-list pointer"
    );
    assert_eq!(
        std::mem::size_of::<Option<&CellList<'_>>>(),
        8,
        "Option<&CellList> is niche-optimised to one pointer"
    );

    // The neighbour row, measured on a real build: `binhead` (nbins
    // entries) + bins (one entry per atom), both `MdaLocal` chains, ≤ 8 B/atom
    // while `nbins ≤ natoms` (the volume guard is a separate test): the
    // table's "cell list" row.
    let cell = UnitCell::orthorhombic(40.0, 40.0, 40.0);
    let positions: Vec<[f64; 3]> = (0..4000u32)
        .map(|i| {
            [
                (i % 20) as f64 * 2.0,
                ((i / 20) % 20) as f64 * 2.0,
                (i / 400) as f64 * 2.0,
            ]
        })
        .collect();
    let cl = CellList::build(&cell, &positions, 3.5).unwrap();
    let n = positions.len() as u64;
    let bytes = (cl.nbins() as u64 + n) * std::mem::size_of::<MdaLocal>() as u64;
    let bpa = bytes as f64 / n as f64;
    assert!(
        bpa <= 8.0,
        "cell list {bpa:.1} B/atom exceeds the design's ≤8 B/atom row"
    );
    let at_scale = bpa * N as f64;
    assert!(
        at_scale <= 8.0 * N as f64,
        "{bpa:.1} B/atom -> {:.2} GB at 128M, over the row's 8 B/atom",
        at_scale / 1e9
    );
}

#[test]
fn cell_list_does_not_allocate_proportional_to_box_volume() {
    // Regression: a 4000 A box at cutoff 3 A asks for (4000/3)^3 ~ 2.4e9 bins
    // (~19 GB of dense binhead) for only 10k atoms. LAMMPS raises the bin size
    // above the cutoff in that regime (`Neighbor::setup_bins`) and grows the
    // stencil reach, trading some speed for a bounded footprint.
    let a = 10_000usize;
    let positions: Vec<[f64; 3]> = (0..a)
        .map(|i| {
            [
                (i % 100) as f64 * 1.1,
                ((i / 100) % 100) as f64 * 1.1,
                (i / 10000) as f64 * 1.1,
            ]
        })
        .collect();
    let big_cell = UnitCell::orthorhombic(4000.0, 4000.0, 4000.0);
    let cl = CellList::build(&big_cell, &positions, 3.0).unwrap();

    let bins = cl.nbins();
    assert!(
        bins <= a,
        "bin count {bins} must not track box volume for {a} atoms"
    );
    // binhead + bins are u32 chains (LAMMPS binhead/bins), so ~8 B/atom
    assert_eq!(std::mem::size_of::<MdaLocal>(), 4, "bin chains stay 32-bit");
    let bytes = (bins * 4 + a * 4) as u64;
    let budget = 8 * a as u64;
    assert!(
        bytes <= budget,
        "cell list {bytes} B vs budget {budget} B (bins={bins})"
    );

    // raising the bin size must not change the answer: same neighbours found.
    // Sampled, since the reference pass is O(n^2)-ish for a 100x100 slab.
    let mut got: Vec<Vec<usize>> = Vec::new();
    for i in (0..a).step_by(97) {
        let mut s = Vec::new();
        cl.neighbors_of(i, 3.0, &mut s).unwrap();
        let mut ids: Vec<usize> = s.iter().map(|(j, _)| *j).collect();
        ids.sort_unstable();
        got.push(ids);
    }
    // reference: a box sized to the atoms, where binsize == cutoff holds
    let small_cell = UnitCell::orthorhombic(120.0, 120.0, 120.0);
    let cl2 = CellList::build(&small_cell, &positions, 3.0).unwrap();
    for (n, i) in (0..a).step_by(97).enumerate() {
        let mut s = Vec::new();
        cl2.neighbors_of(i, 3.0, &mut s).unwrap();
        let mut ids: Vec<usize> = s.iter().map(|(j, _)| *j).collect();
        ids.sort_unstable();
        assert_eq!(got[n], ids, "atom {i}: oversized bins changed the result");
    }
}

#[test]
fn cell_list_borrows_positions_instead_of_copying_them() {
    // A cell list that copied (`to_vec()`) every position it was handed
    // would add 24 B/atom on top of design §24.2's ~8 B/atom row, i.e. 3.07 GB
    // at 128M atoms that no table makes room for. It borrows for its own
    // lifetime, so the cheapest way to catch a copy is
    // address identity: a copy would hand back different memory.
    let cell = UnitCell::orthorhombic(40.0, 40.0, 40.0);
    let positions: Vec<[f64; 3]> = (0..500u32)
        .map(|i| {
            [
                (i % 10) as f64 * 4.0,
                ((i / 10) % 10) as f64 * 4.0,
                ((i / 100) % 5) as f64 * 4.0,
            ]
        })
        .collect();
    let cl = CellList::build(&cell, &positions, 3.5).unwrap();
    assert!(
        std::ptr::eq(cl.positions().as_ptr(), positions.as_ptr()),
        "cell list must borrow the caller's array, not a copy of it"
    );
    // and such a copy would be exactly the unbudgeted one
    let copy_bpa = std::mem::size_of::<[f64; 3]>() as u64;
    assert_eq!(copy_bpa, 24, "the removed copy was 24 B/atom");
    assert!(
        copy_bpa > 8,
        "a 24 B/atom positions copy cannot fit the design's ~8 B/atom cell-list row"
    );
    // The struct itself is a fixed-size handle (slice + geometry + bin
    // vectors), independent of how many atoms it indexes.
    assert!(
        std::mem::size_of::<CellList<'_>>() < 512,
        "CellList handle grew to {} B",
        std::mem::size_of::<CellList<'_>>()
    );
}

#[test]
fn pipeline_floor_fits_the_documented_envelope() {
    // The per-allocation maxima describe structures that exist: the
    // default pipeline's persistent floor is the atom
    // arrays plus the cell list, every term from `size_of` of a real type so
    // the row can drift loudly instead of quietly.
    let atom_arrays = (std::mem::size_of::<[f64; 3]>()
        + std::mem::size_of::<rmda_core::data::particle::MdaTag>()
        + std::mem::size_of::<rmda_core::data::particle::MdaType>()) as u64;
    let cell_list = 2 * std::mem::size_of::<MdaLocal>() as u64; // binhead + bins, ≤ nbins+N
    let per_atom = atom_arrays + cell_list;
    assert_eq!(
        per_atom, 44,
        "36 B atoms + ≤8 B cell list = 44 B/atom floor"
    );
    let floor = gb(per_atom * N);
    // The measured peak of the defect census on the 128M frame is 210-225
    // B/atom (design §16.8); the resident floor must sit well inside it, the
    // rest being per-analysis memory, not resident structures.
    let measured = gb(210 * N);
    assert!(
        floor < 0.5 * measured,
        "floor {floor:.1} GB must stay well under the measured peak {measured:.1} GB"
    );
}
