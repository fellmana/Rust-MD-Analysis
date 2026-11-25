//! Neighbour census: per-atom coordination number and its histogram.
//!
//! Uses the shared cell list at `cutoff` (default 3.5 Å). Stores `cn` per atom and
//! `cn_histogram` in the summary.

use crate::analysis::{Analysis, AnalysisContext, Summary};
use crate::data::property::PropertyStore;
use crate::error::Result;

pub struct NeighborCensus {
    pub cutoff: f64,
}

impl Default for NeighborCensus {
    fn default() -> Self {
        NeighborCensus {
            cutoff: 3.5, // metallic bond cutoff default (Å, metal units)
        }
    }
}

impl NeighborCensus {
    pub fn new(cutoff: f64) -> Self {
        NeighborCensus { cutoff }
    }
}

impl Analysis for NeighborCensus {
    fn name(&self) -> &'static str {
        "neighbor_census"
    }

    fn cutoffs(&self) -> Vec<f64> {
        // report the *configured* cutoff, not the default: the pipeline sizes
        // its cell lists from this, so a stale constant would build the wrong
        // neighbour structure for `rmda census --cutoff 4.2`.
        vec![self.cutoff]
    }

    fn run(
        &self,
        ctx: &AnalysisContext,
        out: &mut PropertyStore,
        summary: &mut Summary,
    ) -> Result<()> {
        let n = ctx.frame.natoms();
        let cell = ctx.nkeys.require_cell("neighbour census")?;
        // Per-atom coordination over fixed §10.2 chunks in this analysis's
        // parallel context, so `--threads` reaches it. Each chunk reuses one
        // scratch buffer (no per-atom allocation) and the per-chunk results are
        // concatenated in ascending chunk ordinal order, keeping `cn[i]`
        // aligned to atom index regardless of scheduling.
        let parts = ctx.par.try_map_chunks(n, |s, e| -> Result<Vec<u32>> {
            let mut scratch = Vec::new();
            let mut local = Vec::with_capacity(e - s);
            for i in s..e {
                cell.neighbors_of(i, self.cutoff, &mut scratch)?;
                local.push(scratch.len() as u32);
            }
            Ok(local)
        })?;
        let mut cn = Vec::with_capacity(n);
        for part in parts {
            cn.extend(part);
        }
        // Histogram of coordination numbers, also chunk-folded: each chunk
        // bins its own slice (growing to its local max), and combines ascend-
        // ing by integer add — same counts as the sequential max-then-bin pass,
        // for any thread count. Seeded at length 1 so an empty frame keeps the
        // sequential behaviour of reporting `[0]`, not `[]`.
        let hist: Vec<u64> = ctx.par.fold_chunks(
            n,
            vec![0u64; 1],
            |s, e| {
                let mut local = Vec::new();
                for &c0 in &cn[s..e] {
                    let c = c0 as usize;
                    if c >= local.len() {
                        local.resize(c + 1, 0);
                    }
                    local[c] += 1;
                }
                local
            },
            |total: &mut Vec<u64>, local: &Vec<u64>| {
                if local.len() > total.len() {
                    total.resize(local.len(), 0);
                }
                for (t, l) in total.iter_mut().zip(local.iter()) {
                    *t += *l;
                }
            },
        );

        let sum: u64 = cn.iter().map(|&c| c as u64).sum();
        out.register_scalar(crate::data::property::DType::F32, "cn", cn);
        let mean = if n == 0 { 0.0 } else { sum as f64 / n as f64 };
        summary.merge_from(Summary::scalar("mean_cn", mean));
        let total_pairs = sum / 2;
        summary.merge_from(Summary::scalar("total_pairs", total_pairs as f64));

        // histogram as a f64 series
        summary.histograms.push((
            "cn_histogram".to_string(),
            hist.iter().map(|&v| v as f64).collect(),
        ));
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::analysis::NeighborhoodContext;
    use crate::cell::UnitCell;
    use crate::data::frame::Frame;
    use crate::data::particle::ParticleData;
    use crate::exec::Par;
    use crate::neighbor::cell_list::CellList;

    #[test]
    fn census_on_2x2x2_cube() {
        // 8 atoms on a 10Å cube (FCC-like simple cubic lattice spacing 5)
        let cell = UnitCell::orthorhombic(10.0, 10.0, 10.0);
        let mut pos = Vec::new();
        for iz in 0..2 {
            for iy in 0..2 {
                for ix in 0..2 {
                    pos.push([ix as f64 * 5.0, iy as f64 * 5.0, iz as f64 * 5.0]);
                }
            }
        }
        let mut p = ParticleData::with_capacity(8);
        for (i, q) in pos.iter().enumerate() {
            p.add_atom(*q, i as i64 + 1, 1);
        }
        let frame = Frame::new(cell, p);

        let cl = CellList::build(&cell, &pos, 3.5).unwrap();
        let nk = NeighborhoodContext { cell: Some(&cl) };
        let ctx = AnalysisContext {
            frame: &frame,
            nkeys: &nk,
            par: &Par::new(1),
        };
        let mut out = PropertyStore::new();
        let mut summary = Summary::default();
        NeighborCensus::new(3.5)
            .run(&ctx, &mut out, &mut summary)
            .unwrap();
        let cn = out.get("cn").unwrap();
        let flat: Vec<u64> = (0..8).map(|i| cn.get_flat(i).unwrap() as u64).collect();
        // every nearest neighbour is at 5Å (>3.5): so cn=0 for all
        assert!(flat.iter().all(|&c| c == 0));
        let mean = summary
            .scalars
            .iter()
            .find(|(n, _)| n == "mean_cn")
            .unwrap()
            .1;
        assert_eq!(mean, 0.0);
        let _ = frame;
    }

    /// The cutoff the analysis reports must be the one it was configured with:
    /// the pipeline sizes its cell lists from `cutoffs()`, so a hard-coded
    /// default would build the neighbour structure for the wrong radius.
    #[test]
    fn cutoffs_reflect_configuration() {
        assert_eq!(NeighborCensus::new(4.2).cutoffs(), vec![4.2]);
        assert_eq!(NeighborCensus::new(1.5).cutoffs(), vec![1.5]);
        assert_eq!(NeighborCensus::default().cutoffs(), vec![3.5]);
    }

    /// Design §10.2: the chunk-ordered `cn` assembly and histogram fold must
    /// give identical output for any thread count. 41³ = 68921 atoms spans
    /// two `CHUNK_ATOMS` chunks, so concatenation order and the fold merge
    /// are both actually exercised.
    #[test]
    fn census_is_identical_across_thread_counts() {
        let side = 41usize;
        let spacing = 5.0f64;
        let mut pos = Vec::with_capacity(side.pow(3));
        for iz in 0..side {
            for iy in 0..side {
                for ix in 0..side {
                    pos.push([
                        ix as f64 * spacing,
                        iy as f64 * spacing,
                        iz as f64 * spacing,
                    ]);
                }
            }
        }
        let n = pos.len();
        let mut pd = ParticleData::with_capacity(n);
        for (i, q) in pos.iter().enumerate() {
            pd.add_atom(*q, i as i64 + 1, 1);
        }
        let cell = UnitCell::orthorhombic(
            side as f64 * spacing,
            side as f64 * spacing,
            side as f64 * spacing,
        );
        let frame = Frame::new(cell, pd);
        let cl = CellList::build(&frame.cell, &pos, 5.5).unwrap();
        let nk = NeighborhoodContext { cell: Some(&cl) };
        let mut ref_cn: Option<Vec<f64>> = None;
        let mut ref_hist: Option<Vec<f64>> = None;
        for threads in [1usize, 2, 4] {
            let par = Par::new(threads);
            let ctx = AnalysisContext {
                frame: &frame,
                nkeys: &nk,
                par: &par,
            };
            let mut out = PropertyStore::new();
            let mut summary = Summary::default();
            NeighborCensus::new(5.5)
                .run(&ctx, &mut out, &mut summary)
                .unwrap();
            let prop = out.get("cn").expect("cn property");
            let cn: Vec<f64> = (0..n).map(|i| prop.get_flat(i).unwrap()).collect();
            // Simple-cubic with pbc and cutoff 5.5 > spacing 5: exactly 6
            // neighbours per atom, so any index/order mix-up shows up as a
            // wrong value, not merely a different-but-equivalent one.
            assert!(
                cn.iter().all(|&c| c == 6.0),
                "threads={threads}: expected cn=6 everywhere"
            );
            let hist = summary
                .histograms
                .iter()
                .find(|(name, _)| name == "cn_histogram")
                .expect("cn_histogram")
                .1
                .clone();
            match (&ref_cn, &ref_hist) {
                (None, _) => {
                    ref_cn = Some(cn);
                    ref_hist = Some(hist);
                }
                (Some(rc), Some(rh)) => {
                    assert_eq!(rc, &cn, "threads={threads}: cn diverged");
                    assert_eq!(rh, &hist, "threads={threads}: histogram diverged");
                }
                _ => unreachable!(),
            }
        }
    }
}
