//! One frame of the per-grain defect census, end to end: PTM, grain labels,
//! optional segmentation splitting each region (with tracking), the census
//! against a fitted or an explicit reference, and defect clusters.
//!
//! This is the per-frame glue `rmda defects` (single file and series) and the
//! Python `rmda.defects` share, so the two surfaces run one code path rather
//! than two copies kept equal by a parity test (design §22). It prints
//! nothing: what the CLI reports while a frame runs (dHCP relabelling, the
//! segmentation summary, the grain count after splitting) goes through a
//! [`FrameObserver`], whose default methods do nothing.

use crate::analysis::defects::antisite;
use crate::analysis::defects::clusters::{self, Clusters};
use crate::analysis::defects::explicit::{self, AffineMapping, ExplicitReference};
use crate::analysis::defects::{self, DefectsConfig, DefectsResult};
use crate::analysis::grain_seg::{self, SegConfig, Segmentation};
use crate::analysis::grain_track::{Event, Tracker};
use crate::analysis::ptm::{check, NeighbourOrder, Ptm, PtmAtom};
use crate::data::frame::Frame;
use crate::data::select::{label_regions, Selection};
use crate::error::{Error, Result};
use crate::exec::Par;

/// Where a frame's grain labels come from before any segmentation.
#[derive(Debug, Clone)]
pub enum GrainSource {
    /// Selection expressions; an atom takes the first region it matches.
    Regions(Vec<Selection>),
    /// An input column of the frame, read per frame.
    Column(String),
    /// Labels already resolved by the caller, one per atom.
    Labels(Vec<i32>),
    /// The whole frame is one grain.
    Whole,
}

impl GrainSource {
    /// Grain label per atom of `f`.
    pub fn labels(&self, f: &Frame) -> Result<Vec<i32>> {
        match self {
            GrainSource::Regions(sel) => label_regions(
                &f.cell,
                &f.particles.positions,
                &f.particles.types,
                &f.particles.ids,
                sel,
            ),
            GrainSource::Column(name) => column_labels(f, name),
            GrainSource::Labels(v) => Ok(v.clone()),
            GrainSource::Whole => Ok(vec![0; f.natoms()]),
        }
    }
}

/// An input column of `f` as integer grain labels (`--grain-column`): i32 as
/// is, i64 truncated, floats rounded.
pub fn column_labels(f: &Frame, name: &str) -> Result<Vec<i32>> {
    let p = f
        .particles
        .props
        .get(name)
        .ok_or_else(|| Error::invalid(format!("input has no '{name}' column")))?;
    if let Some(v) = p.as_i32() {
        return Ok(v.clone());
    }
    if let Some(v) = p.as_i64() {
        return Ok(v.iter().map(|&g| g as i32).collect());
    }
    if let Some(v) = p.as_f64() {
        return Ok(v.iter().map(|&g| g.round() as i32).collect());
    }
    if let Some(v) = p.as_f32() {
        return Ok(v.iter().map(|&g| g.round() as i32).collect());
    }
    if let Some(v) = p.as_u8() {
        return Ok(v.iter().map(|&g| g as i32).collect());
    }
    Err(Error::invalid(format!("column '{name}' is not numeric")))
}

/// Progress a frame reports while it runs. Every method defaults to nothing.
pub trait FrameObserver {
    /// `dhcp` relabelled `n` close-packed atoms as double HCP.
    fn dhcp(&mut self, _n: usize) {}
    /// Segmentation ran and split the regions into `grains` grains.
    fn segmented(&mut self, _s: &Segmentation, _grains: usize) {}
}

/// The silent observer.
impl FrameObserver for () {}

/// PTM settings of the defect census: OVITO's default structures, with
/// `dhcp` double HCP derived from the FCC and HCP labels.
#[derive(Debug, Clone, Copy)]
pub struct PtmSettings {
    pub rmsd_cutoff: f64,
    pub dhcp: bool,
    pub order: NeighbourOrder,
}

/// PTM with OVITO's default structures, one compact record per atom, and with
/// `dhcp` double HCP relabelled from it (reported through `obs`).
pub fn run_ptm(
    f: &Frame,
    s: &PtmSettings,
    par: &Par,
    obs: &mut dyn FrameObserver,
) -> Result<Vec<PtmAtom>> {
    let _t = crate::prof::StageTimer::start("ptm");
    let mut ptm = Ptm {
        rmsd_cutoff: s.rmsd_cutoff,
        flags: check::OVITO_DEFAULT,
    }
    .compute_ordered(&f.cell, &f.particles.positions, par, s.order)?;
    if s.dhcp {
        let _s = crate::prof::StageTimer::start("ptm.dhcp");
        let n = crate::analysis::stacking::relabel_dhcp(
            &f.cell,
            &f.particles.positions,
            &mut ptm,
            par,
        )?;
        obs.dhcp(n);
    }
    Ok(ptm)
}

/// Everything that decides one frame's census.
#[derive(Debug, Clone)]
pub struct FrameSettings {
    pub source: GrainSource,
    /// Split each region by segmented grains.
    pub segment: Option<SegConfig>,
    pub ptm: PtmSettings,
    pub defects: DefectsConfig,
    /// Defect cluster cutoff, Å; `None` derives it from the measured grains.
    pub cluster_cutoff: Option<f64>,
    /// Count antisites.
    pub antisites: bool,
    /// Element of each atom type, in order (`--elements`).
    pub elements: Vec<String>,
    /// Under an explicit reference, also run PTM on the current frame (it
    /// feeds nothing in the census; the CLI's non-lattice file reads it).
    pub current_ptm: bool,
}

/// Grain labels after segmentation, with what produced them.
pub struct FrameGrains {
    pub grains: Vec<i32>,
    /// Stable id per grain label, when segmented and tracked.
    pub stable: Option<Vec<u32>>,
    pub events: Vec<Event>,
    pub segmentation: Option<Segmentation>,
}

/// Grain label per atom of `f` (regions, a column or the whole frame, split
/// by segmentation when asked), with the tracker's stable ids and events when
/// segmented and tracked.
pub fn frame_grains(
    f: &Frame,
    ptm: &[PtmAtom],
    s: &FrameSettings,
    par: &Par,
    tracker: Option<&mut Tracker>,
    obs: &mut dyn FrameObserver,
) -> Result<FrameGrains> {
    let mut grains = s.source.labels(f)?;
    let mut stable = None;
    let mut events = Vec::new();
    let mut segmentation = None;
    if let Some(cfg) = &s.segment {
        let seg = {
            let _t = crate::prof::StageTimer::start("segment");
            grain_seg::segment(&f.cell, &f.particles.positions, ptm, cfg, par)?
        };
        // Split each region by its segmented grains, then number densely.
        let g_max = seg.grains.len() as i64 + 1;
        let combined: Vec<i64> = grains
            .iter()
            .zip(&seg.grain)
            .map(|(&r, &g)| {
                if r < 0 || g < 0 {
                    -1
                } else {
                    r as i64 * g_max + g as i64
                }
            })
            .collect();
        let mut ids: Vec<i64> = combined.iter().copied().filter(|&c| c >= 0).collect();
        ids.sort_unstable();
        ids.dedup();
        grains = combined
            .iter()
            .map(|&c| {
                if c < 0 {
                    -1
                } else {
                    ids.binary_search(&c).map_or(-1, |i| i as i32)
                }
            })
            .collect();
        obs.segmented(&seg, ids.len());
        if let Some(t) = tracker {
            let (st, ev) = t.update(&f.particles.ids, &grains, ids.len());
            stable = Some(st);
            events = ev;
        }
        segmentation = Some(seg);
    }
    Ok(FrameGrains {
        grains,
        stable,
        events,
        segmentation,
    })
}

/// One frame's grain label per atom, as [`analyse`] assigns them (PTM, then
/// the grain source and segmentation), without the defect census or any
/// tracking: what a resumed or sharded series needs of the frame before its
/// first one (design §20), at the cost of PTM and segmentation alone.
pub fn grain_labels(f: &Frame, s: &FrameSettings, par: &Par) -> Result<Vec<i32>> {
    let ptm = run_ptm(f, &s.ptm, par, &mut ())?;
    Ok(frame_grains(f, &ptm, s, par, None, &mut ())?.grains)
}

/// An explicit Wigner–Seitz reference prepared once: PTM of the reference
/// frame, its grains (labels and segmentation evaluated on the reference,
/// never tracked) and their boundary classes.
pub fn prepare_reference(
    rf: &Frame,
    s: &FrameSettings,
    par: &Par,
    obs: &mut dyn FrameObserver,
) -> Result<(ExplicitReference, Option<Segmentation>)> {
    let ptm = run_ptm(rf, &s.ptm, par, obs)?;
    let g = frame_grains(rf, &ptm, s, par, None, obs)?;
    let r = ExplicitReference::new(
        &rf.cell,
        &rf.particles.positions,
        &g.grains,
        &ptm,
        s.defects.boundary_width,
        par,
    )?;
    Ok((r, g.segmentation))
}

/// An explicit reference and how the current frame maps onto it.
#[derive(Clone, Copy)]
pub struct WsReference<'a> {
    pub reference: &'a ExplicitReference,
    /// The reference frame (its atoms are the sites).
    pub frame: &'a Frame,
    pub affine: AffineMapping,
}

/// The types antisites are counted in.
#[derive(Debug, Clone)]
pub struct AntisiteTypes {
    /// Name per type (index type − 1): the frame's, or under an explicit
    /// reference the reference's (current types matched to it by label).
    pub names: Vec<String>,
    /// Per atom: its type in that numbering.
    pub types: Vec<i32>,
    /// Under an explicit reference: the reference atom id of each antisite's
    /// site.
    pub site_ids: Option<Vec<i64>>,
}

impl AntisiteTypes {
    pub fn label(&self, t: i32) -> String {
        antisite::type_label(&self.names, t)
    }
}

/// Everything one frame produces.
pub struct FrameResult {
    pub result: DefectsResult,
    pub clusters: Clusters,
    pub cluster_cutoff: f64,
    /// The grain label each atom was measured in.
    pub grains: Vec<i32>,
    /// Stable id per grain label (index = label), when tracked.
    pub stable: Option<Vec<u32>>,
    pub events: Vec<Event>,
    /// PTM of the frame: always under a fitted reference, under an explicit
    /// one only with [`FrameSettings::current_ptm`].
    pub ptm: Option<Vec<PtmAtom>>,
    /// The segmentation that split the grains (fitted reference only; an
    /// explicit reference's comes from [`prepare_reference`]).
    pub segmentation: Option<Segmentation>,
    /// Under an explicit reference: the reference atom id of each vacancy's
    /// site, and of the site each interstitial sits on.
    pub ws_sites: Option<(Vec<i64>, Vec<i64>)>,
    pub antisites: Option<AntisiteTypes>,
}

impl FrameResult {
    /// The id a grain label is reported under: its stable id when tracked.
    pub fn stable_id(&self, grain: i32) -> i64 {
        match &self.stable {
            Some(s) if grain >= 0 => s[grain as usize] as i64,
            _ => grain as i64,
        }
    }
}

/// One frame of the census. Under an explicit reference (`ws`) the grains are
/// the reference's and nothing is tracked; otherwise PTM, labels and
/// segmentation run on `f`, with `tracker` (when given) assigning stable ids.
pub fn analyse(
    f: &Frame,
    s: &FrameSettings,
    par: &Par,
    tracker: Option<&mut Tracker>,
    ws: Option<WsReference<'_>>,
    obs: &mut dyn FrameObserver,
) -> Result<FrameResult> {
    let mut antisites = None;
    let (result, grains, stable, events, ptm, segmentation, ws_sites) = match ws {
        Some(w) => {
            let types = s
                .antisites
                .then(|| antisite::matched_types(w.frame, f, &s.elements));
            let o = {
                let _t = crate::prof::StageTimer::start("defects");
                explicit::run_typed(
                    w.reference,
                    &f.cell,
                    &f.particles.positions,
                    w.affine,
                    types
                        .as_ref()
                        .map(|(s, t, _)| explicit::Types { sites: s, atoms: t }),
                    par,
                )?
            };
            let ptm = if s.current_ptm {
                Some(run_ptm(f, &s.ptm, par, obs)?)
            } else {
                None
            };
            let ids = &w.frame.particles.ids;
            if let Some((_, t, names)) = types {
                antisites = Some(AntisiteTypes {
                    names,
                    types: t,
                    site_ids: Some(o.antisite_sites.iter().map(|&s| ids[s as usize]).collect()),
                });
            }
            let sites = (
                o.vacancy_sites.iter().map(|&s| ids[s as usize]).collect(),
                o.interstitial_sites
                    .iter()
                    .map(|&s| ids[s as usize])
                    .collect(),
            );
            (
                o.result,
                o.atom_grain,
                None,
                Vec::new(),
                ptm,
                None,
                Some(sites),
            )
        }
        None => {
            let ptm = run_ptm(f, &s.ptm, par, obs)?;
            let g = frame_grains(f, &ptm, s, par, tracker, obs)?;
            let typed = s.antisites.then(|| antisite::frame_types(f, &s.elements));
            let types = typed.as_ref().map(|t| t.0.as_slice());
            let result = {
                let _t = crate::prof::StageTimer::start("defects");
                defects::run_typed(
                    &f.cell,
                    &f.particles.positions,
                    &g.grains,
                    &ptm,
                    types,
                    &s.defects,
                    par,
                )?
            };
            if let Some((types, names)) = typed {
                antisites = Some(AntisiteTypes {
                    names,
                    types,
                    site_ids: None,
                });
            }
            (
                result,
                g.grains,
                g.stable,
                g.events,
                Some(ptm),
                g.segmentation,
                None,
            )
        }
    };

    // Clusters over measured grains' defects.
    let cutoff = match s.cluster_cutoff {
        Some(c) => clusters::Cutoff::Uniform(c),
        None => clusters::default_cutoff(&result),
    };
    let cluster_cutoff = cutoff.reported();
    let clusters = {
        let _t = crate::prof::StageTimer::start("defects.clusters");
        let vp: Vec<_> = result.vacancies.iter().map(|v| v.position).collect();
        let vg: Vec<i32> = result.vacancies.iter().map(|v| v.grain).collect();
        let ip: Vec<_> = result
            .interstitials
            .iter()
            .map(|s| f.particles.positions[s.atom as usize])
            .collect();
        let ig: Vec<i32> = result.interstitials.iter().map(|s| s.grain).collect();
        let mut c = clusters::cluster(&f.cell, &vp, &vg, &ip, &ig, &cutoff, par)?;
        // Antisites join (never merge) the V/I clusters, at their sites (as
        // vacancies are).
        if antisites.is_some() {
            let ap: Vec<_> = result.antisites.iter().map(|a| a.site_position).collect();
            let ag: Vec<i32> = result.antisites.iter().map(|a| a.grain).collect();
            clusters::attach_antisites(
                &f.cell, &mut c, &vp, &vg, &ip, &ig, &ap, &ag, &cutoff, par,
            )?;
        }
        c
    };
    Ok(FrameResult {
        result,
        clusters,
        cluster_cutoff,
        grains,
        stable,
        events,
        ptm,
        segmentation,
        ws_sites,
        antisites,
    })
}
