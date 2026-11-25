//! Grain tracking across frames (design §15.4).
//!
//! A grain keeps its stable id from one frame to the next when it and its
//! predecessor share more than half of each other's atoms (matched by atom
//! id, so frames may list atoms in any order). Pairs are matched greedily by
//! overlap, largest first, ties by lower ids. An unmatched grain is new, or a
//! split when most of its atoms came from one previous grain; a previous
//! grain that did not continue and whose atoms mostly went to one grain
//! merged into it.

use std::collections::HashMap;

/// What happened to a grain between two frames.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Event {
    /// First seen; no predecessor holds most of its atoms.
    New { stable: u32 },
    /// Matched to its predecessor.
    Continued { stable: u32 },
    /// Unmatched; most of its atoms came from previous grain `from`.
    Split { stable: u32, from: u32 },
    /// Previous grain `stable` ended; most of its atoms are now in `into`.
    Merged { stable: u32, into: u32 },
}

/// Tracker state: the previous frame's `atom id → stable id`.
#[derive(Debug, Default)]
pub struct Tracker {
    /// Sorted by atom id.
    prev: Vec<(i64, u32)>,
    prev_size: HashMap<u32, usize>,
    next_id: u32,
}

impl Tracker {
    pub fn new() -> Self {
        Self::default()
    }

    /// Stable ids for this frame's grains (`grain[i]` in `0..n_grains` or
    /// negative), and the events since the previous frame.
    pub fn update(
        &mut self,
        ids: &[i64],
        grain: &[i32],
        n_grains: usize,
    ) -> (Vec<u32>, Vec<Event>) {
        let size = grain_sizes(grain, n_grains);
        let mut overlap: HashMap<(u32, u32), usize> = HashMap::new(); // (grain, prev stable)
        for (&id, &g) in ids.iter().zip(grain) {
            if g < 0 {
                continue;
            }
            if let Ok(k) = self.prev.binary_search_by_key(&id, |e| e.0) {
                *overlap.entry((g as u32, self.prev[k].1)).or_insert(0) += 1;
            }
        }
        let (out, events) = self.step(&size, &overlap);
        self.remember(ids, grain, &out);
        (out, events)
    }

    /// The tracking decision alone, from this frame's grain sizes and the
    /// atoms each grain shares with each previous stable id (`(grain, prev
    /// stable) → count`). [`Tracker::update`] counts those from atom ids; a
    /// sharded series records them per frame and replays the decisions when
    /// merging (`rmda merge`), so the two give the same ids and events.
    pub fn step(
        &mut self,
        size: &[usize],
        overlap: &HashMap<(u32, u32), usize>,
    ) -> (Vec<u32>, Vec<Event>) {
        let n_grains = size.len();
        let mut pairs: Vec<((u32, u32), usize)> = overlap.iter().map(|(&k, &v)| (k, v)).collect();
        pairs.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));

        let mut stable: Vec<Option<u32>> = vec![None; n_grains];
        let mut taken: HashMap<u32, u32> = HashMap::new(); // prev stable -> grain
        for &((g, s), c) in &pairs {
            if stable[g as usize].is_some() || taken.contains_key(&s) {
                continue;
            }
            let ps = self.prev_size.get(&s).copied().unwrap_or(0);
            if 2 * c > size[g as usize] && 2 * c > ps {
                stable[g as usize] = Some(s);
                taken.insert(s, g);
            }
        }
        let mut events = Vec::new();
        let mut out = vec![0u32; n_grains];
        for g in 0..n_grains {
            match stable[g] {
                Some(s) => {
                    out[g] = s;
                    events.push(Event::Continued { stable: s });
                }
                None => {
                    let s = self.next_id;
                    self.next_id += 1;
                    out[g] = s;
                    // Majority source among previous grains.
                    let src = pairs
                        .iter()
                        .find(|((gg, _), c)| *gg as usize == g && 2 * c > size[g])
                        .map(|((_, ps), _)| *ps);
                    events.push(match src {
                        Some(from) => Event::Split { stable: s, from },
                        None => Event::New { stable: s },
                    });
                }
            }
        }
        // Previous grains that did not continue.
        let mut ended: Vec<u32> = self
            .prev_size
            .keys()
            .copied()
            .filter(|s| !taken.contains_key(s))
            .collect();
        ended.sort_unstable();
        for s in ended {
            let ps = self.prev_size[&s];
            if let Some(((g, _), _)) = pairs.iter().find(|((_, pp), c)| *pp == s && 2 * c > ps) {
                events.push(Event::Merged {
                    stable: s,
                    into: out[*g as usize],
                });
            }
        }

        self.prev_size = out.iter().copied().zip(size.iter().copied()).collect();
        (out, events)
    }

    /// Remember this frame's atoms under their stable ids, for the next
    /// [`Tracker::update`].
    fn remember(&mut self, ids: &[i64], grain: &[i32], out: &[u32]) {
        self.prev = ids
            .iter()
            .zip(grain)
            .filter(|(_, &g)| g >= 0)
            .map(|(&id, &g)| (id, out[g as usize]))
            .collect();
        self.prev.sort_unstable_by_key(|e| e.0);
    }

    /// Restart tracking after a recorded frame: its grain label per atom, the
    /// stable id recorded for each label, and the next id the run would have
    /// assigned. A resumed series continues with the ids it wrote before,
    /// rather than renumbering from 0 (which a fresh tracker would).
    pub fn seed(&mut self, ids: &[i64], grain: &[i32], stable: &[u32], next_id: u32) {
        let size = grain_sizes(grain, stable.len());
        self.remember(ids, grain, stable);
        self.prev_size = stable.iter().copied().zip(size).collect();
        self.next_id = next_id;
    }
}

/// Atoms per grain label (`grain[i]` in `0..n_grains`, negative for none).
pub fn grain_sizes(grain: &[i32], n_grains: usize) -> Vec<usize> {
    let mut size = vec![0usize; n_grains];
    for &g in grain {
        if g >= 0 {
            size[g as usize] += 1;
        }
    }
    size
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ids_survive_reordering_growth_split_and_merge() {
        let mut t = Tracker::new();
        // Frame 0: grains 0 (ids 0..100) and 1 (ids 100..200).
        let ids: Vec<i64> = (0..200).collect();
        let g: Vec<i32> = (0..200).map(|i| i32::from(i >= 100)).collect();
        let (s0, _) = t.update(&ids, &g, 2);
        assert_eq!(s0, vec![0, 1]);

        // Frame 1: rows reversed and the grain *labels* swapped; grain 1 grew
        // by 20 atoms taken from grain 0.
        let ids1: Vec<i64> = (0..200).rev().collect();
        let g1: Vec<i32> = ids1.iter().map(|&i| if i >= 80 { 0 } else { 1 }).collect();
        let (s1, ev) = t.update(&ids1, &g1, 2);
        assert_eq!(s1, vec![1, 0], "{ev:?}");

        // Frame 2: old grain 0 (stable 0) splits in two halves of 40.
        let g2: Vec<i32> = ids1
            .iter()
            .map(|&i| {
                if i >= 80 {
                    0
                } else if i >= 40 {
                    1
                } else {
                    2
                }
            })
            .collect();
        let (s2, ev) = t.update(&ids1, &g2, 3);
        assert_eq!(s2, vec![1, 2, 3]);
        assert!(ev.contains(&Event::Split { stable: 2, from: 0 }), "{ev:?}");
        assert!(ev.contains(&Event::Split { stable: 3, from: 0 }), "{ev:?}");

        // Frame 3: everything is one grain: the big one continues and both
        // halves merged into it.
        let (s3, ev) = t.update(&ids1, &vec![0; 200], 1);
        assert_eq!(s3, vec![1]);
        assert!(ev.contains(&Event::Merged { stable: 2, into: 1 }), "{ev:?}");
        assert!(ev.contains(&Event::Merged { stable: 3, into: 1 }), "{ev:?}");
    }

    /// Frames of the test above: (atom ids, labels, label count), with the
    /// split (frame 2) and the merge (frame 3).
    fn frames() -> Vec<(Vec<i64>, Vec<i32>, usize)> {
        let ids: Vec<i64> = (0..200).collect();
        let rev: Vec<i64> = (0..200).rev().collect();
        let split = |i: i64| -> i32 {
            if i >= 80 {
                0
            } else if i >= 40 {
                1
            } else {
                2
            }
        };
        vec![
            (
                ids.clone(),
                ids.iter().map(|&i| i32::from(i >= 100)).collect(),
                2,
            ),
            (
                rev.clone(),
                rev.iter().map(|&i| if i >= 80 { 0 } else { 1 }).collect(),
                2,
            ),
            (rev.clone(), rev.iter().map(|&i| split(i)).collect(), 3),
            (rev.clone(), vec![0; 200], 1),
            // A new grain from nowhere: atoms no previous frame had.
            (
                (0..260).collect(),
                (0..260).map(|i| if i >= 200 { 1 } else { 0 }).collect(),
                2,
            ),
        ]
    }

    #[test]
    fn a_seeded_tracker_continues_like_an_uninterrupted_one() {
        let fr = frames();
        let mut whole = Tracker::new();
        let all: Vec<_> = fr.iter().map(|(i, g, n)| whole.update(i, g, *n)).collect();
        // Resume after each frame k: seed from frame k's labels and recorded
        // ids, with the next id the run had reached.
        for k in 0..fr.len() - 1 {
            let next = all[..=k]
                .iter()
                .flat_map(|(s, _)| s.iter().copied())
                .max()
                .map_or(0, |m| m + 1);
            let mut t = Tracker::new();
            t.seed(&fr[k].0, &fr[k].1, &all[k].0, next);
            for j in k + 1..fr.len() {
                let (i, g, n) = &fr[j];
                assert_eq!(t.update(i, g, *n), all[j], "resumed after {k}, frame {j}");
            }
        }
    }

    #[test]
    fn decisions_from_label_overlaps_equal_the_atom_level_ones() {
        // What a sharded series records per frame: each label's size and its
        // overlap with the previous frame's labels; `rmda merge` maps those
        // labels to the stable ids it has assigned and calls `step`.
        let fr = frames();
        let mut atoms = Tracker::new();
        let mut labels = Tracker::new();
        let mut prev: Option<(&[i64], &[i32], Vec<u32>)> = None;
        for (ids, g, n) in &fr {
            let want = atoms.update(ids, g, *n);
            let mut overlap: HashMap<(u32, u32), usize> = HashMap::new();
            if let Some((pi, pg, ps)) = &prev {
                let mut by_id: Vec<(i64, i32)> =
                    pi.iter().copied().zip(pg.iter().copied()).collect();
                by_id.sort_unstable();
                for (&id, &gi) in ids.iter().zip(g.iter()) {
                    if let Ok(k) = by_id.binary_search_by_key(&id, |e| e.0) {
                        let pl = by_id[k].1;
                        if gi >= 0 && pl >= 0 {
                            *overlap.entry((gi as u32, ps[pl as usize])).or_insert(0) += 1;
                        }
                    }
                }
            }
            let got = labels.step(&grain_sizes(g, *n), &overlap);
            assert_eq!(got, want);
            prev = Some((ids, g, got.0));
        }
    }
}
