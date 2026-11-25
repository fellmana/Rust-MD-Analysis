//! Defect clusters (design §16.5).
//!
//! Vacancy sites and interstitial atoms are joined when within the cutoff
//! (default: halfway between the first and second neighbour shell of each
//! one's grain; two defects of different grains join within the smaller of
//! their grains' cutoffs). A
//! cluster may mix both kinds and may cross grains; its net count `n_i − n_v`
//! is what a mixed cluster amounts to.
//!
//! **Antisites** (with `--antisites`, [`attach_antisites`]) never change the
//! vacancy–interstitial clusters: their membership, ids, centres and counts
//! are the same with and without. Each antisite joins the V/I cluster of its
//! nearest vacancy or interstitial within the cutoff (counted in that
//! cluster's `antisites`); antisites near no V/I defect are clustered among
//! themselves by the same cutoff into antisite-only clusters, numbered after
//! the V/I clusters. An antisite never bridges two V/I clusters, and
//! antisites do not chain one another onto a V/I cluster.

use super::{DefectsResult, Status};
use crate::analysis::ptm::structure;
use crate::cell::{Coord3, UnitCell};
use crate::error::Result;
use crate::exec::Par;
use crate::neighbor::{KnnIndex, KnnScratch};

/// One cluster.
#[derive(Debug, Clone)]
pub struct Cluster {
    pub id: u32,
    pub vacancies: u32,
    pub interstitials: u32,
    /// Centre of the members, unwrapped about the first member.
    pub centre: Coord3,
    /// Grain of the first member.
    pub grain: i32,
    /// Members belong to more than one grain.
    pub multi_grain: bool,
    /// Antisites attached to the cluster ([`attach_antisites`]); not part
    /// of `size`, `net` or the centre.
    pub antisites: u32,
}

impl Cluster {
    pub fn size(&self) -> u32 {
        self.vacancies + self.interstitials
    }
    pub fn net(&self) -> i64 {
        self.interstitials as i64 - self.vacancies as i64
    }
}

/// Clusters and per-defect labels (`vacancy_cluster[k]` for the k-th vacancy).
#[derive(Debug, Clone, Default)]
pub struct Clusters {
    pub clusters: Vec<Cluster>,
    pub vacancy_cluster: Vec<u32>,
    pub interstitial_cluster: Vec<u32>,
    /// With antisites: each antisite's cluster (a V/I cluster's id, or an
    /// antisite-only cluster's, numbered from `clusters.len()`).
    pub antisite_cluster: Vec<u32>,
    /// Antisite-only clusters (no vacancy or interstitial within the cutoff
    /// of any member); their `antisites` is their size.
    pub antisite_only: Vec<Cluster>,
}

/// How far apart two defects may be and still join.
#[derive(Debug, Clone)]
pub enum Cutoff {
    /// One distance for every grain, Å.
    Uniform(f64),
    /// Each grain's own, Å; `largest` is the largest measured grain's, used
    /// for a grain not listed and reported as the run's cutoff.
    PerGrain {
        by_grain: std::collections::HashMap<i32, f64>,
        largest: f64,
    },
}

impl Cutoff {
    /// The cutoff of a defect of grain `g`.
    pub fn of(&self, g: i32) -> f64 {
        match self {
            Cutoff::Uniform(c) => *c,
            Cutoff::PerGrain { by_grain, largest } => by_grain.get(&g).copied().unwrap_or(*largest),
        }
    }

    /// The single figure a table reports: the uniform cutoff, or the largest
    /// measured grain's.
    pub fn reported(&self) -> f64 {
        match self {
            Cutoff::Uniform(c) => *c,
            Cutoff::PerGrain { largest, .. } => *largest,
        }
    }
}

/// Halfway between the first and second neighbour shells, in bonds: BCC
/// `(1 + 2/√3)/2`, FCC, HCP, dHCP and SC `(1 + √2)/2`.
pub fn shell_cutoff(st: i32) -> f64 {
    if st == structure::BCC {
        (1.0 + 2.0 / 3.0f64.sqrt()) / 2.0
    } else {
        (1.0 + 2.0f64.sqrt()) / 2.0
    }
}

/// Each measured grain's cutoff from its structure and fitted bond; 3 Å
/// when none is measured.
pub fn default_cutoff(r: &DefectsResult) -> Cutoff {
    let measured = r.grains.iter().filter(|g| g.status == Status::Measured);
    let by_grain = measured
        .clone()
        .map(|g| (g.grain, shell_cutoff(g.structure) * g.bond))
        .collect();
    let largest = measured
        .max_by_key(|g| g.n_atoms)
        .map_or(3.0, |g| shell_cutoff(g.structure) * g.bond);
    Cutoff::PerGrain { by_grain, largest }
}

/// Union-find with path halving; roots are component minima.
fn find(p: &mut [u32], mut x: u32) -> u32 {
    while p[x as usize] != x {
        p[x as usize] = p[p[x as usize] as usize];
        x = p[x as usize];
    }
    x
}

/// Cluster defects. `*_grain` give each defect's grain.
pub fn cluster(
    cell: &UnitCell,
    vacancies: &[Coord3],
    vacancy_grain: &[i32],
    interstitials: &[Coord3],
    interstitial_grain: &[i32],
    cutoff: &Cutoff,
    par: &Par,
) -> Result<Clusters> {
    let pts: Vec<Coord3> = vacancies.iter().chain(interstitials).copied().collect();
    let grains: Vec<i32> = vacancy_grain
        .iter()
        .chain(interstitial_grain)
        .copied()
        .collect();
    let n = pts.len();
    if n == 0 {
        return Ok(Clusters::default());
    }
    let nv = vacancies.len();
    let index = KnnIndex::build(cell, &pts, par)?;
    // Neighbours within the largest cutoff (k grows until the k-th lies
    // outside it), then each pair within the smaller of its own two.
    let own: Vec<f64> = grains.iter().map(|&g| cutoff.of(g)).collect();
    let c2 = own.iter().copied().fold(0.0, f64::max).powi(2);
    let mut links: Vec<Vec<(u32, Coord3)>> = vec![Vec::new(); n];
    par.fill(&mut links, |i, out| {
        let mut s = KnnScratch::default();
        let mut nb = Vec::new();
        let mut k = 16;
        loop {
            index.knn(i, pts[i], k, &mut s, &mut nb);
            if nb.len() < k || nb.last().map_or(true, |e| e.2 > c2) || k >= 64 {
                break;
            }
            k *= 2;
        }
        *out = nb
            .iter()
            .filter(|e| e.2 <= own[i].min(own[e.0 as usize]).powi(2))
            .map(|e| (e.0, e.1))
            .collect();
    });
    let mut parent: Vec<u32> = (0..n as u32).collect();
    for (i, l) in links.iter().enumerate() {
        for &(j, _) in l {
            let (a, b) = (find(&mut parent, i as u32), find(&mut parent, j));
            if a != b {
                let (lo, hi) = if a < b { (a, b) } else { (b, a) };
                parent[hi as usize] = lo;
            }
        }
    }
    // Dense ids by root order (component minima).
    let roots: Vec<u32> = (0..n as u32).map(|i| find(&mut parent, i)).collect();
    let mut id_of = vec![u32::MAX; n];
    let mut out: Vec<Cluster> = Vec::new();
    for i in 0..n {
        let r = roots[i] as usize;
        if id_of[r] == u32::MAX {
            id_of[r] = out.len() as u32;
            out.push(Cluster {
                id: out.len() as u32,
                vacancies: 0,
                interstitials: 0,
                centre: pts[r],
                grain: grains[r],
                multi_grain: false,
                antisites: 0,
            });
        }
        let c = &mut out[id_of[r] as usize];
        if i < nv {
            c.vacancies += 1;
        } else {
            c.interstitials += 1;
        }
        if grains[i] != c.grain {
            c.multi_grain = true;
        }
    }
    // Centres: unwrap each cluster breadth-first along its own links, whose
    // displacements are image-true, so a cluster across a periodic face
    // centres at the face.
    let mut offset: Vec<Option<Coord3>> = vec![None; n];
    let mut sums: Vec<(Coord3, u32)> = vec![([0.0; 3], 0); out.len()];
    for r in 0..n {
        if roots[r] as usize != r {
            continue;
        }
        offset[r] = Some([0.0; 3]);
        let mut queue = std::collections::VecDeque::from([r]);
        while let Some(i) = queue.pop_front() {
            let oi = offset[i].unwrap();
            let c = id_of[r] as usize;
            for (s, v) in sums[c].0.iter_mut().zip(oi) {
                *s += v;
            }
            sums[c].1 += 1;
            for &(j, dr) in &links[i] {
                let j = j as usize;
                if offset[j].is_none() {
                    offset[j] = Some([oi[0] + dr[0], oi[1] + dr[1], oi[2] + dr[2]]);
                    queue.push_back(j);
                }
            }
        }
    }
    for (c, (s, k)) in out.iter_mut().zip(&sums) {
        for (x, v) in c.centre.iter_mut().zip(s) {
            *x += v / (*k).max(1) as f64;
        }
    }
    let label: Vec<u32> = roots.iter().map(|&r| id_of[r as usize]).collect();
    Ok(Clusters {
        clusters: out,
        vacancy_cluster: label[..nv].to_vec(),
        interstitial_cluster: label[nv..].to_vec(),
        ..Default::default()
    })
}

/// Add antisites to clusters `cl` made by [`cluster`] from the same
/// vacancies and interstitials (see the module notes): each antisite joins
/// the cluster of its nearest V/I defect within the cutoff (the smaller of
/// the two defects' grains', as in [`cluster`]; ties to the lower defect),
/// and the others form antisite-only clusters among themselves. The V/I
/// clusters keep their members, ids and centres.
#[allow(clippy::too_many_arguments)]
pub fn attach_antisites(
    cell: &UnitCell,
    cl: &mut Clusters,
    vacancies: &[Coord3],
    vacancy_grain: &[i32],
    interstitials: &[Coord3],
    interstitial_grain: &[i32],
    antisites: &[Coord3],
    antisite_grain: &[i32],
    cutoff: &Cutoff,
    par: &Par,
) -> Result<()> {
    let pts: Vec<Coord3> = vacancies.iter().chain(interstitials).copied().collect();
    let grains: Vec<i32> = vacancy_grain
        .iter()
        .chain(interstitial_grain)
        .copied()
        .collect();
    let label: Vec<u32> = cl
        .vacancy_cluster
        .iter()
        .chain(&cl.interstitial_cluster)
        .copied()
        .collect();
    let mut attached: Vec<Option<u32>> = vec![None; antisites.len()];
    if !pts.is_empty() && !antisites.is_empty() {
        let index = KnnIndex::build(cell, &pts, par)?;
        let own: Vec<f64> = grains.iter().map(|&g| cutoff.of(g)).collect();
        let c2 = own.iter().copied().fold(0.0, f64::max).powi(2);
        par.fill(&mut attached, |a, out| {
            let ca = cutoff.of(antisite_grain[a]);
            let mut s = KnnScratch::default();
            let mut nb = Vec::new();
            let mut k = 16;
            loop {
                index.knn_point(antisites[a], k, &mut s, &mut nb);
                if nb.len() < k || nb.last().map_or(true, |e| e.2 > c2) || k >= 1024 {
                    break;
                }
                k *= 2;
            }
            *out = nb
                .iter()
                .filter(|e| e.2 <= ca.min(own[e.0 as usize]).powi(2))
                .min_by(|x, y| x.2.total_cmp(&y.2).then(x.0.cmp(&y.0)))
                .map(|e| label[e.0 as usize]);
        });
    }
    for &c in attached.iter().flatten() {
        cl.clusters[c as usize].antisites += 1;
    }
    // The rest among themselves: `cluster` with them as its only kind.
    let rest: Vec<usize> = (0..antisites.len())
        .filter(|&a| attached[a].is_none())
        .collect();
    let rp: Vec<Coord3> = rest.iter().map(|&a| antisites[a]).collect();
    let rg: Vec<i32> = rest.iter().map(|&a| antisite_grain[a]).collect();
    let own = cluster(cell, &rp, &rg, &[], &[], cutoff, par)?;
    let base = cl.clusters.len() as u32;
    cl.antisite_only = own
        .clusters
        .into_iter()
        .map(|c| Cluster {
            id: base + c.id,
            vacancies: 0,
            interstitials: 0,
            antisites: c.vacancies,
            ..c
        })
        .collect();
    cl.antisite_cluster = attached.iter().map(|c| c.unwrap_or(u32::MAX)).collect();
    for (&a, &c) in rest.iter().zip(&own.vacancy_cluster) {
        cl.antisite_cluster[a] = base + c;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn neighbours_join_and_far_defects_stay_apart_across_the_boundary() {
        let cell = UnitCell::orthorhombic(20.0, 20.0, 20.0);
        // A divacancy straddling the periodic x face, one interstitial next to
        // it, and an isolated vacancy.
        let v = [[0.2, 5.0, 5.0], [19.5, 5.0, 5.0], [10.0, 10.0, 10.0]];
        let i = [[1.5, 5.5, 5.0]];
        let c = cluster(
            &cell,
            &v,
            &[0, 0, 0],
            &i,
            &[1],
            &Cutoff::Uniform(2.5),
            &Par::new(2),
        )
        .unwrap();
        assert_eq!(c.clusters.len(), 2);
        let big = &c.clusters[c.vacancy_cluster[0] as usize];
        assert_eq!((big.vacancies, big.interstitials, big.net()), (2, 1, -1));
        assert!(big.multi_grain);
        assert_eq!(c.vacancy_cluster[0], c.vacancy_cluster[1]);
        assert_ne!(c.vacancy_cluster[2], c.vacancy_cluster[0]);
        // Antisites: one beside the divacancy joins it; two near each other
        // and far from any V/I defect form their own cluster; one alone
        // forms another. The V/I clusters do not change.
        let mut ca = c.clone();
        let a = [
            [1.0, 6.5, 5.0],
            [15.0, 15.0, 15.0],
            [16.5, 15.0, 15.0],
            [5.0, 15.0, 5.0],
        ];
        attach_antisites(
            &cell,
            &mut ca,
            &v,
            &[0, 0, 0],
            &i,
            &[1],
            &a,
            &[0, 0, 0, 0],
            &Cutoff::Uniform(2.5),
            &Par::new(2),
        )
        .unwrap();
        assert_eq!(ca.vacancy_cluster, c.vacancy_cluster);
        assert_eq!(ca.interstitial_cluster, c.interstitial_cluster);
        for (x, y) in ca.clusters.iter().zip(&c.clusters) {
            assert_eq!(
                (x.id, x.vacancies, x.interstitials),
                (y.id, y.vacancies, y.interstitials)
            );
            assert_eq!(x.centre, y.centre);
        }
        assert_eq!(ca.clusters[c.vacancy_cluster[0] as usize].antisites, 1);
        assert_eq!(ca.clusters[c.vacancy_cluster[2] as usize].antisites, 0);
        assert_eq!(ca.antisite_cluster[0], c.vacancy_cluster[0]);
        assert_eq!(ca.antisite_only.len(), 2);
        assert_eq!(ca.antisite_cluster[1], ca.antisite_cluster[2]);
        assert_eq!(ca.antisite_cluster[1], 2);
        assert_eq!(ca.antisite_cluster[3], 3);
        let pair = &ca.antisite_only[0];
        assert_eq!((pair.id, pair.antisites, pair.size()), (2, 2, 0));
        assert!((pair.centre[0] - 15.75).abs() < 1e-9, "{:?}", pair.centre);
        // The divacancy's centre is at the face, not mid-box.
        assert!(
            big.centre[0] < 1.5 || big.centre[0] > 18.5,
            "{:?}",
            big.centre
        );
    }
}
