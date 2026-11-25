//! Connected components over a pair stream (design §18.3, and the grain
//! labelling of §15.2).
//!
//! Shared by grain segmentation and cluster analysis because both reduce to the
//! same two operations over the same data: union a bond stream, then hand out
//! contiguous ids in first-atom order. Neither owns anything the other needs, so
//! the union-find lives here rather than being written out twice — a second copy
//! is a second chance to get the `rank`/`size` bookkeeping subtly wrong, and at
//! 128M atoms that costs an order of magnitude without changing a single answer.
//!
//! [`DisjointSet`] is deliberately small. Everything interesting about how the
//! pairs were generated lives in the caller; this only answers "are these two in
//! the same set yet".

/// Label for an atom that belongs to no set.
///
/// `u32::MAX`, so it is not a valid set index and indexing a per-set array with
/// it is a bounds check rather than a silent read of the last element. Callers
/// that keep per-set side tables must test against this first: a `u32::MAX` atom
/// index lands in the last bin, which is usually in range and usually wrong.
pub const NO_SET: u32 = u32::MAX;

/// Union-find with path halving and union by size.
///
/// Written out rather than pulled in: this is all the crate needs from the
/// category, and getting the `rank`/`size` bookkeeping subtly wrong costs an
/// order of magnitude at 128M atoms without changing a single answer.
#[derive(Clone)]
pub struct DisjointSet {
    parent: Vec<u32>,
    size: Vec<u32>,
}

impl DisjointSet {
    pub fn new(n: usize) -> Self {
        assert!(
            n < NO_SET as usize,
            "a union-find this large cannot be indexed by u32 set ids"
        );
        Self {
            parent: (0..n as u32).collect(),
            size: vec![1; n],
        }
    }

    /// Number of elements, `n`; unions do not change it.
    pub fn len(&self) -> usize {
        self.parent.len()
    }

    pub fn is_empty(&self) -> bool {
        self.parent.is_empty()
    }

    /// Total set size of the set containing `x`, as of the last union touching
    /// it. Only exact after every union has been applied.
    pub fn size_of(&mut self, x: u32) -> u32 {
        let r = self.find(x) as usize;
        self.size[r]
    }

    #[inline]
    pub fn find(&mut self, mut x: u32) -> u32 {
        while self.parent[x as usize] != x {
            // Path halving: point x at its grandparent and continue from there,
            // so a long chain flattens in one pass instead of one step per lookup.
            let p = self.parent[self.parent[x as usize] as usize];
            self.parent[x as usize] = p;
            x = p;
        }
        x
    }

    /// Merge two sets. Returns false when they were already one.
    #[inline]
    pub fn union(&mut self, a: u32, b: u32) -> bool {
        let (mut ra, mut rb) = (self.find(a), self.find(b));
        if ra == rb {
            return false;
        }
        // Union by size keeps the tree shallow; without it a bad order can build
        // a linear chain and turn `find` into a scan.
        if self.size[ra as usize] < self.size[rb as usize] {
            std::mem::swap(&mut ra, &mut rb);
        }
        self.parent[rb as usize] = ra;
        self.size[ra as usize] += self.size[rb as usize];
        true
    }
}

/// Relabel per-atom set ids to contiguous ids handed out in **first-atom order**.
///
/// `sets[i]` is the set of atom `i`, or [`NO_SET`] for an atom that belongs to
/// none. Returns the new labels and how many ids were used.
///
/// The ordering is the whole point. Not discovery order — that depends on the
/// neighbour iteration order, which depends on the cell list's bin layout and so
/// on the frame's origin. Not size order — that renumbers a cluster when an
/// unrelated one grows. Both produce output that looks stable and is not, which
/// is worse than an obviously unstable labelling because it survives a casual
/// eyeball check. First-atom order depends on nothing but the atom order.
///
/// `NO_SET` atoms neither consume an id nor are relabelled, so an id space with
/// holes cannot appear: a run of `n` atoms in `k` sets gives ids `0..k`, and the
/// largest id is `k - 1`.
pub fn canonical_labels(ds: &mut DisjointSet, sets: &[u32]) -> (Vec<u32>, usize) {
    let mut label_of_root = vec![NO_SET; ds.len()];
    let mut next = 0u32;
    let mut out = vec![NO_SET; sets.len()];
    for (i, &s) in sets.iter().enumerate() {
        if s == NO_SET {
            continue;
        }
        let r = ds.find(s) as usize;
        if label_of_root[r] == NO_SET {
            label_of_root[r] = next;
            next += 1;
        }
        out[i] = label_of_root[r];
    }
    (out, next as usize)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_fresh_set_is_its_own_root() {
        let mut ds = DisjointSet::new(4);
        for i in 0..4 {
            assert_eq!(ds.find(i), i);
        }
        assert_eq!(ds.size_of(2), 1);
    }

    #[test]
    fn union_reports_whether_it_actually_merged() {
        let mut ds = DisjointSet::new(3);
        assert!(ds.union(0, 1));
        assert!(!ds.union(1, 0), "already one set");
        assert_eq!(ds.size_of(1), 2);
        assert_eq!(ds.size_of(2), 1);
    }

    #[test]
    fn union_by_size_keeps_the_larger_set_as_the_root() {
        let mut ds = DisjointSet::new(4);
        ds.union(0, 1);
        ds.union(1, 2); // size 3 set
        let big = ds.find(0);
        let singleton = ds.find(3);
        // Joining a singleton must not make the singleton the root, or the tree
        // grows a new spine every time a size-1 set is absorbed.
        ds.union(big, singleton);
        assert_eq!(ds.find(0), big);
        assert_eq!(ds.find(3), big, "the size-1 set must not become the root");
        assert_eq!(ds.size_of(0), 4);
    }

    #[test]
    fn a_long_chain_does_not_degenerate() {
        // The pathological order for path halving: always attach the newest
        // singleton to the growing root, which is what makes find() a scan when
        // union by size is missing.
        let n = 4096;
        let mut ds = DisjointSet::new(n);
        for i in 1..n {
            ds.union(0, i as u32);
        }
        assert_eq!(ds.size_of(0), n as u32);
        assert!(ds.find((n - 1) as u32) < n as u32);
    }

    #[test]
    fn canonical_labels_follow_first_atom_order_not_root_order() {
        // Atom 0's singleton gets id 0 and the {1, 2, 3} set id 1, whichever of
        // its atoms is the union-find root.
        let mut ds = DisjointSet::new(4);
        ds.union(1, 2);
        ds.union(3, 1);
        let sets = vec![0u32, 3, 3, 3];
        let (labels, count) = canonical_labels(&mut ds, &sets);
        assert_eq!(labels, vec![0, 1, 1, 1]);
        assert_eq!(count, 2);
    }

    #[test]
    fn canonical_labels_leave_no_holes_for_unset_atoms() {
        let mut ds = DisjointSet::new(4);
        ds.union(1, 2);
        let sets = vec![NO_SET, 1, 2, NO_SET];
        let (labels, count) = canonical_labels(&mut ds, &sets);
        assert_eq!(labels, vec![NO_SET, 0, 0, NO_SET]);
        assert_eq!(count, 1, "unset atoms must not consume an id");
    }

    #[test]
    fn canonical_labels_are_independent_of_the_input_numbering() {
        // Same partition, different set ids: the output must be identical, since
        // the ids it produces are a function of atom order alone.
        let mut a = DisjointSet::new(4);
        a.union(0, 1);
        let mut b = DisjointSet::new(4);
        b.union(1, 0);
        let (la, _) = canonical_labels(&mut a, &[0, 1, 2, 3]);
        let (lb, _) = canonical_labels(&mut b, &[1, 0, 2, 3]);
        assert_eq!(la, lb);
        assert_eq!(la, vec![0, 0, 1, 2]);
    }

    #[test]
    fn an_empty_frame_labels_to_nothing() {
        let mut ds = DisjointSet::new(0);
        let (labels, count) = canonical_labels(&mut ds, &[]);
        assert!(labels.is_empty());
        assert_eq!(count, 0);
    }
}
