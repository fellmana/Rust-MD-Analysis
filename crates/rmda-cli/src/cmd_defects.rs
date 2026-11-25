//! `rmda defects`: per-grain Wigner–Seitz census (design §16), for one
//! file or a time series of files (design §20).
//!
//! Series mode (`--out DIR`) streams one frame at a time and appends long-format
//! tables keyed by file index: `summary.csv` (one row per frame), `grains.csv`
//! (one row per grain per frame), `clusters.csv` (one row per defect cluster
//! per frame) and, with `--segment`, `grain_events.csv`. A rerun skips frames
//! already in `summary.csv`; `run.txt` records the settings so a rerun with
//! different settings is refused instead of mixing results.

use crate::cmd_segment;
use rmda_core::analysis::defects::antisite;
use rmda_core::analysis::defects::clusters::{self, Clusters};
use rmda_core::analysis::defects::explicit::{AffineMapping, ExplicitReference};
use rmda_core::analysis::defects::frame::{
    self, AntisiteTypes, FrameObserver, FrameSettings, GrainSource, PtmSettings, WsReference,
};
use rmda_core::analysis::defects::{
    ClassCount, DefectsConfig, DefectsResult, ReferenceMode, Status,
};
use rmda_core::analysis::grain_seg::{SegConfig, Segmentation};
use rmda_core::analysis::grain_track::{Event, Tracker};
use rmda_core::analysis::ptm::{structure, PtmAtom};
use rmda_core::data::frame::Frame;
use rmda_core::data::select::Selection;
use rmda_core::error::{Error, Result};
use rmda_core::exec::Par;
use std::collections::BTreeSet;
use std::io::Write;
use std::path::{Path, PathBuf};

/// `--ws-reference`: explicit-reference Wigner–Seitz.
pub struct WsArgs {
    pub path: PathBuf,
    /// The reference frame (its atoms are the sites).
    pub frame: Frame,
    pub affine: AffineMapping,
}

pub struct DefectsArgs {
    /// Explicit reference frame instead of a fitted lattice.
    pub ws: Option<WsArgs>,
    pub regions: Vec<String>,
    pub grain_column: Option<String>,
    pub boundary_width: Option<f64>,
    pub min_atoms: usize,
    pub rmsd_cutoff: f64,
    pub table: Option<PathBuf>,
    pub defects_dir: Option<PathBuf>,
    pub reference: ReferenceMode,
    pub patch_size: Option<f64>,
    /// Local census merge rule (`--local-merge`).
    pub local_merge: rmda_core::analysis::defects::LocalMerge,
    /// `--resolve-ambiguous`.
    pub resolve_ambiguous: bool,
    pub dhcp: bool,
    /// PTM neighbour order (`--ptm-order`).
    pub ptm_order: rmda_core::analysis::ptm::NeighbourOrder,
    /// Segment grains (within each region, when regions are given).
    pub segment: Option<SegConfig>,
    /// Defect cluster cutoff, Å; `None` is halfway between the first and
    /// second neighbour shells of the largest measured grain.
    pub cluster_cutoff: Option<f64>,
    /// Also write `non_lattice.xyz` beside the defect files.
    pub non_lattice: bool,
    /// Count antisites (`--antisites`).
    pub antisites: bool,
    /// Element of each atom type, in order (`--elements`); types beyond it
    /// take the file's labels, then the element of their mass.
    pub elements: Vec<String>,
}

/// Series-mode settings.
pub struct SeriesArgs {
    pub out: PathBuf,
    /// Write per-frame defect files under `defects/`, laid out by
    /// `defects_layout`.
    pub defects_xyz: bool,
    pub defects_layout: DefectsLayout,
    pub overwrite: bool,
    /// `--shard I/N`: this run is shard `I` of `N` (design §20).
    pub shard: Option<(usize, usize)>,
}

/// Frames `lo..hi` of `len` that shard `i` of `n` takes: contiguous blocks,
/// sizes differing by at most one, so a block's frames are consecutive and
/// grain tracking needs only the frame before the block.
pub(crate) fn shard_block(len: usize, i: usize, n: usize) -> (usize, usize) {
    (i * len / n, (i + 1) * len / n)
}

/// The frames a series run takes and where it writes them.
pub(crate) struct ShardPlan {
    pub items: Vec<(i64, PathBuf)>,
    /// The frame before a shard's block (none for shard 0 or no shard): grain
    /// tracking needs its grains.
    pub lead_in: Option<(i64, PathBuf)>,
    pub out: PathBuf,
}

/// Every frame into `out`, or shard `I` of `N`: its contiguous block of the
/// ordered frames into `out/shard-I-of-N/`, with the frame before it.
pub(crate) fn plan_shard(
    all: &[(i64, PathBuf)],
    shard: Option<(usize, usize)>,
    out: &Path,
) -> ShardPlan {
    match shard {
        None => ShardPlan {
            items: all.to_vec(),
            lead_in: None,
            out: out.to_path_buf(),
        },
        Some((i, n)) => {
            let (lo, hi) = shard_block(all.len(), i, n);
            ShardPlan {
                items: all[lo..hi].to_vec(),
                lead_in: lo.checked_sub(1).map(|p| all[p].clone()),
                out: shard_dir(out, i, n),
            }
        }
    }
}

/// A shard's `shard.txt` (its index, the frame before its block, its
/// frames), which `rmda merge` checks the shards against.
pub(crate) fn write_shard_file(
    dir: &Path,
    shard: Option<(usize, usize)>,
    lead_in: Option<&(i64, PathBuf)>,
    items: &[(i64, PathBuf)],
    total: usize,
) -> Result<()> {
    let Some((i, n)) = shard else {
        return Ok(());
    };
    let mut t = format!("shard {i} of {n}\n");
    if let Some((k, p)) = lead_in {
        t += &format!("lead_in {k} {}\n", p.display());
    }
    for (k, p) in items {
        t += &format!("frame {k} {}\n", p.display());
    }
    std::fs::write(dir.join("shard.txt"), t)?;
    eprintln!(
        "series: shard {i} of {n}: {} of {total} frame(s) into {}",
        items.len(),
        dir.display()
    );
    Ok(())
}

/// A shard's directory under the series' `--out`.
pub(crate) fn shard_dir(out: &Path, i: usize, n: usize) -> PathBuf {
    out.join(format!("shard-{i}-of-{n}"))
}

/// Where a series writes each frame's defect files.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DefectsLayout {
    /// `defects/<index>/vacancies.xyz`, one directory per frame.
    Dirs,
    /// `defects/vacancies-<index>.xyz`, all in one directory.
    Files,
    /// `defects/vacancies.xyz` holding every frame in turn, each with
    /// `index=<index>` in its comment line.
    Append,
}

/// The defect files a frame writes.
const DEFECT_FILES: [&str; 4] = ["vacancies", "interstitials", "non_lattice", "antisites"];

/// The shared per-frame settings (`rmda_core::analysis::defects::frame`) of
/// these arguments: grains from regions, else a column, else one grain.
pub(crate) fn frame_settings(a: &DefectsArgs) -> Result<FrameSettings> {
    let source = if !a.regions.is_empty() {
        GrainSource::Regions(
            a.regions
                .iter()
                .map(|r| Selection::parse(r))
                .collect::<Result<_>>()?,
        )
    } else if let Some(name) = &a.grain_column {
        GrainSource::Column(name.clone())
    } else {
        GrainSource::Whole
    };
    Ok(FrameSettings {
        source,
        segment: a.segment.clone(),
        ptm: PtmSettings {
            rmsd_cutoff: a.rmsd_cutoff,
            dhcp: a.dhcp,
            order: a.ptm_order,
        },
        defects: DefectsConfig {
            boundary_width: a.boundary_width,
            min_atoms: a.min_atoms,
            reference: a.reference,
            patch_size: a.patch_size,
            local_merge: a.local_merge,
            resolve_ambiguous: a.resolve_ambiguous,
            ..Default::default()
        },
        cluster_cutoff: a.cluster_cutoff,
        antisites: a.antisites,
        elements: a.elements.clone(),
        // PTM of the current frame under an explicit reference only feeds
        // the non-lattice file.
        current_ptm: a.non_lattice,
    })
}

/// What `rmda defects` reports while a frame runs.
struct Report {
    regions: usize,
}

impl FrameObserver for Report {
    fn dhcp(&mut self, n: usize) {
        eprintln!("dhcp: {n} close-packed atoms in ABAC stacking");
    }

    fn segmented(&mut self, s: &Segmentation, grains: usize) {
        cmd_segment::print_summary(s);
        eprintln!(
            "defects: {grains} grain(s) after splitting {} region(s) by segmentation",
            self.regions.max(1)
        );
    }
}

/// Everything one frame produces: the shared census result plus what only
/// the CLI's files need.
struct FrameOutput {
    result: DefectsResult,
    clusters: Clusters,
    cluster_cutoff: f64,
    /// The grain label of each atom.
    grains: Vec<i32>,
    /// Stable id per grain label (index = label), when tracked.
    stable: Option<Vec<u32>>,
    events: Vec<Event>,
    /// With `--non-lattice-atoms`: (atom, PTM structure, grain label) of every
    /// atom not on its grain's lattice.
    non_lattice: Option<Vec<(u32, u8, i32)>>,
    /// With `--non-lattice-atoms`: per vacancy, how many of its first-shell
    /// neighbour positions hold a non-lattice atom.
    vacancy_shell: Option<Vec<u8>>,
    /// With `--ws-reference`: the reference atom id of each vacancy's site,
    /// and of the site each interstitial sits on.
    ws_sites: Option<(Vec<i64>, Vec<i64>)>,
    /// With `--antisites`.
    antisites: Option<AntisiteOut>,
}

/// What `--antisites` adds to a frame's output beyond the result.
type AntisiteOut = AntisiteTypes;

impl FrameOutput {
    fn stable_id(&self, grain: i32) -> i64 {
        match &self.stable {
            Some(s) if grain >= 0 => s[grain as usize] as i64,
            _ => grain as i64,
        }
    }
}

/// The explicit WS reference of `--ws-reference`, prepared once per run:
/// PTM of the reference frame, its grains (regions, column or segmentation,
/// all evaluated on the reference) and their boundary classes.
pub(crate) fn prepare_ws(a: &DefectsArgs, par: &Par) -> Result<Option<ExplicitReference>> {
    let Some(ws) = &a.ws else {
        return Ok(None);
    };
    let _t = rmda_core::prof::StageTimer::start("defects.ws_reference");
    let s = frame_settings(a)?;
    let mut report = Report {
        regions: a.regions.len(),
    };
    let (r, _) = frame::prepare_reference(&ws.frame, &s, par, &mut report)?;
    eprintln!(
        "ws reference: {} site(s), affine mapping {}",
        r.len(),
        ws.affine.label()
    );
    Ok(Some(r))
}

fn analyse_frame(
    f: &Frame,
    a: &DefectsArgs,
    par: &Par,
    tracker: Option<&mut Tracker>,
    ws: Option<&ExplicitReference>,
) -> Result<FrameOutput> {
    let reference = match (ws, &a.ws) {
        (Some(r), Some(wa)) => {
            if wa.affine == AffineMapping::Off {
                let (h, g) = (f.cell.h_matrix().m, r.cell().h_matrix().m);
                let scale = g.iter().flatten().fold(0.0f64, |m, v| m.max(v.abs()));
                let diff = h
                    .iter()
                    .flatten()
                    .zip(g.iter().flatten())
                    .fold(0.0f64, |m, (x, y)| m.max((x - y).abs()));
                if diff > 1e-6 * scale {
                    eprintln!(
                        "defects: warning: the current cell differs from the WS reference \
                         cell by up to {diff:.4} Å and --ws-affine is off; give \
                         --ws-affine to-reference if the box was deformed"
                    );
                }
            }
            Some(WsReference {
                reference: r,
                frame: &wa.frame,
                affine: wa.affine,
            })
        }
        _ => None,
    };
    let s = frame_settings(a)?;
    let mut report = Report {
        regions: a.regions.len(),
    };
    let out = frame::analyse(f, &s, par, tracker, reference, &mut report)?;
    let non_lattice = match &out.ptm {
        Some(ptm) if a.non_lattice => Some(non_lattice_atoms(ptm, &out.grains, &out.result)),
        _ => None,
    };
    let vacancy_shell = match &non_lattice {
        Some(atoms) => Some(vacancy_shell(f, atoms, &out.result, par)?),
        None => None,
    };
    Ok(FrameOutput {
        result: out.result,
        clusters: out.clusters,
        cluster_cutoff: out.cluster_cutoff,
        grains: out.grains,
        stable: out.stable,
        events: out.events,
        non_lattice,
        vacancy_shell,
        ws_sites: out.ws_sites,
        antisites: out.antisites,
    })
}

/// Atoms whose PTM structure is not their grain's reference structure, and
/// atoms in no grain. A grain without a crystalline reference (`OTHER`)
/// contributes all of its atoms: none of them is on a lattice.
fn non_lattice_atoms(ptm: &[PtmAtom], grains: &[i32], r: &DefectsResult) -> Vec<(u32, u8, i32)> {
    let n = r
        .grains
        .iter()
        .map(|g| g.grain + 1)
        .max()
        .unwrap_or(0)
        .max(0) as usize;
    let mut lattice = vec![structure::OTHER; n];
    for g in r.grains.iter().filter(|g| g.grain >= 0) {
        lattice[g.grain as usize] = g.structure;
    }
    ptm.iter()
        .zip(grains)
        .enumerate()
        .filter(|(_, (p, &g))| {
            let own = usize::try_from(g)
                .ok()
                .and_then(|g| lattice.get(g).copied())
                .unwrap_or(structure::OTHER);
            own == structure::OTHER || p.structure() != own
        })
        .map(|(i, (p, &g))| (i as u32, p.structure, g))
        .collect()
}

/// Per vacancy, the non-lattice atoms within its grain's first shell: a
/// vacancy in a lattice is surrounded by atoms PTM cannot match (8 in BCC, 12
/// in FCC/HCP), so a count near zero marks a vacancy worth looking at.
fn vacancy_shell(
    f: &Frame,
    atoms: &[(u32, u8, i32)],
    r: &DefectsResult,
    par: &Par,
) -> Result<Vec<u8>> {
    if atoms.is_empty() || r.vacancies.is_empty() {
        return Ok(vec![0; r.vacancies.len()]);
    }
    let pos: Vec<_> = atoms
        .iter()
        .map(|&(i, _, _)| f.particles.positions[i as usize])
        .collect();
    let index = rmda_core::neighbor::KnnIndex::build(&f.cell, &pos, par)?;
    let reach = |g: i32| -> f64 {
        r.grains
            .iter()
            .find(|x| x.grain == g)
            .filter(|x| x.bond.is_finite())
            .map_or(0.0, |x| clusters::shell_cutoff(x.structure) * x.bond)
    };
    let mut scratch = rmda_core::neighbor::KnnScratch::default();
    let mut nb = Vec::new();
    Ok(r.vacancies
        .iter()
        .map(|v| {
            let rc = reach(v.grain);
            index.knn_point(v.position, 16.min(pos.len()), &mut scratch, &mut nb);
            nb.iter().filter(|&&(_, _, d2)| d2 <= rc * rc).count() as u8
        })
        .collect())
}

/// The census of one frame, exactly as `rmda defects` counts it (for `rmda
/// void`): nothing is printed or written. The explicit reference is prepared by the caller ([`prepare_ws`]), so a
/// series prepares it once; each frame is counted on its own, exactly as a
/// single-file run counts it.
pub(crate) fn census_with(
    f: &Frame,
    a: &DefectsArgs,
    par: &Par,
    ws: Option<&ExplicitReference>,
) -> Result<DefectsResult> {
    let mut tracker = Tracker::new();
    Ok(analyse_frame(f, a, par, Some(&mut tracker), ws)?.result)
}

/// One frame's census for a tracked series (`rmda void --out`): the result,
/// each atom's grain label, the stable id per label when `tracker` is given
/// (`None` writes labels, for a shard's `rmda merge` to replay), and the
/// tracker's events.
pub(crate) struct TrackedCensus {
    pub result: DefectsResult,
    pub grains: Vec<i32>,
    pub stable: Option<Vec<u32>>,
    pub events: Vec<Event>,
}

impl TrackedCensus {
    pub fn stable_id(&self, grain: i32) -> i64 {
        match &self.stable {
            Some(s) if grain >= 0 => s[grain as usize] as i64,
            _ => grain as i64,
        }
    }
}

pub(crate) fn census_tracked(
    f: &Frame,
    a: &DefectsArgs,
    par: &Par,
    tracker: Option<&mut Tracker>,
    ws: Option<&ExplicitReference>,
) -> Result<TrackedCensus> {
    let out = analyse_frame(f, a, par, tracker, ws)?;
    Ok(TrackedCensus {
        result: out.result,
        grains: out.grains,
        stable: out.stable,
        events: out.events,
    })
}

/// Single file: summary to stdout, optional table and defect files.
/// One frame's census, with its outputs written. Returns whether the frame
/// has grains and none of them was measured (every one refused by a gate,
/// too small or not crystalline), for the CLI's exit code 5.
pub fn run(f: &Frame, a: &DefectsArgs, par: &Par) -> Result<bool> {
    let mut tracker = Tracker::new();
    let ws = prepare_ws(a, par)?;
    let out = analyse_frame(f, a, par, Some(&mut tracker), ws.as_ref())?;
    print_summary(&out);
    if let Some(p) = &a.table {
        let mut w = std::io::BufWriter::new(std::fs::File::create(p)?);
        writeln!(w, "{}", grain_header(a.antisites))?;
        write_grain_rows(&mut w, "", f, &out)?;
        eprintln!("defects: per-grain table written to {}", p.display());
    }
    if let Some(d) = &a.defects_dir {
        std::fs::create_dir_all(d)?;
        for (n, text) in defect_xyz(&out, f, None, &a.elements) {
            std::fs::write(d.join(format!("{n}.xyz")), text)?;
        }
        eprintln!(
            "defects: vacancies.xyz{}{} and interstitials.xyz written to {}",
            if out.non_lattice.is_some() {
                ", non_lattice.xyz"
            } else {
                ""
            },
            if out.antisites.is_some() {
                ", antisites.xyz"
            } else {
                ""
            },
            d.display()
        );
    }
    let grains = &out.result.grains;
    Ok(!grains.is_empty() && grains.iter().all(|g| g.status != Status::Measured))
}

/// The trailing integer of a file name (`data.lastframe-999` → 999).
pub fn file_index(p: &Path) -> Option<i64> {
    let name = p.file_name()?.to_string_lossy();
    let digits: String = name
        .chars()
        .rev()
        .skip_while(|c| !c.is_ascii_digit())
        .take_while(|c| c.is_ascii_digit())
        .collect();
    digits.chars().rev().collect::<String>().parse().ok()
}

/// A time series over `files`, one frame each, read by `read`.
pub fn run_series(
    files: &[PathBuf],
    a: &DefectsArgs,
    s: &SeriesArgs,
    read: &dyn Fn(&Path) -> Result<Frame>,
    par: &Par,
) -> Result<()> {
    let all = series_items(files)?;
    let ShardPlan {
        items,
        lead_in,
        out: out_dir,
    } = plan_shard(&all, s.shard, &s.out);
    let s = &SeriesArgs {
        out: out_dir,
        defects_xyz: s.defects_xyz,
        defects_layout: s.defects_layout,
        overwrite: s.overwrite,
        shard: s.shard,
    };
    let tracking = a.segment.is_some() && a.ws.is_none();
    // Sharded, a series does not track: it writes grain labels where stable
    // ids go and records in track.csv what the tracker decides from (each
    // label's size, its overlap with the previous frame's labels); `rmda
    // merge` replays the decisions in order and writes the stable ids.
    let record_tracks = tracking && s.shard.is_some();

    let done = open_series(
        &s.out,
        &settings_signature(a),
        &[
            "summary.csv",
            "grains.csv",
            "clusters.csv",
            "grain_events.csv",
            "antisites.csv",
            "track.csv",
        ],
        s.overwrite,
    )?;
    write_shard_file(&s.out, s.shard, lead_in.as_ref(), &items, all.len())?;
    if s.overwrite && s.defects_layout == DefectsLayout::Append {
        for n in DEFECT_FILES {
            let _ = std::fs::remove_file(s.out.join("defects").join(format!("{n}.xyz")));
        }
    }

    let open = |name: &str, header: &str| open_table(&s.out, name, header);
    let mut summary = open(
        "summary.csv",
        &format!(
            "file_index,file,timestep,natoms,grains,grains_measured,sites,atoms,vacancies,interstitials,\
         gb_vacancies,gb_interstitials,balance_residual,clusters,largest_cluster,cluster_cutoff,\
         wall_s,rss_mb,peak_rss_mb,net_vacancies,gb_net_vacancies{}",
        if a.antisites {
            ",antisites,gb_antisites"
        } else {
            ""
        }
    ))?;
    let mut grains_w = open(
        "grains.csv",
        &format!("file_index,file,{}", grain_header(a.antisites)),
    )?;
    let mut antisites_w = if a.antisites {
        Some(open(
            "antisites.csv",
            "file_index,file,timestep,grain,stable_grain,site_type,occupant_type,site_species,\
             occupant_species,antisites,gb_antisites",
        )?)
    } else {
        None
    };
    let mut clusters_w = open(
        "clusters.csv",
        &format!(
            "file_index,file,timestep,cluster,grain,multi_grain,size,vacancies,interstitials,net,\
             cx,cy,cz{}",
            if a.antisites { ",antisites" } else { "" }
        ),
    )?;
    let mut events_w = if tracking && !record_tracks {
        Some(open(
            "grain_events.csv",
            "file_index,file,timestep,event,stable,other",
        )?)
    } else {
        None
    };
    let mut track_w = if record_tracks {
        Some(open("track.csv", "file_index,kind,label,other,count")?)
    } else {
        None
    };

    if s.defects_xyz && s.defects_layout == DefectsLayout::Append {
        // A frame interrupted after its defect frames were appended but
        // before its summary row is redone; drop what it appended.
        for n in DEFECT_FILES {
            prune_appended(&s.out.join("defects").join(format!("{n}.xyz")), &done)?;
        }
    }

    let mut tracker = Tracker::new();
    // An explicit reference is read, labelled and indexed once for the series.
    let ws = prepare_ws(a, par)?;
    let settings = frame_settings(a)?;
    // The frame before the first one to do: its grains seed the tracking.
    // Resuming, the last finished frame (its stable ids are in grains.csv);
    // a shard starting fresh, the frame before its block. Only PTM and
    // segmentation run on it. An explicit reference's grains are its own and
    // need no tracking.
    let mut prev_labels: Option<Vec<(i64, i32)>> = None;
    if tracking {
        let last_done = items.iter().rev().find(|(k, _)| done.contains(k));
        if let Some((k, p)) = last_done.or(lead_in.as_ref()) {
            eprintln!(
                "series: segmenting {} to continue grain tracking",
                p.display()
            );
            let f = read(p)?;
            let labels = frame::grain_labels(&f, &settings, par)?;
            if !record_tracks {
                let (stable, next) = recorded_ids(&s.out, *k, &labels)?;
                tracker.seed(&f.particles.ids, &labels, &stable, next);
            }
            prev_labels = Some(by_id(&f.particles.ids, &labels));
        }
    }

    let total = items.len();
    for (n, (k, p)) in items.iter().enumerate() {
        if done.contains(k) {
            eprintln!("series [{}/{total}] {k}: already done, skipping", n + 1);
            continue;
        }
        let t0 = std::time::Instant::now();
        eprintln!("series [{}/{total}] {k}: {}", n + 1, p.display());
        let f = read(p)?;
        let track = if record_tracks {
            None
        } else {
            Some(&mut tracker)
        };
        let out = analyse_frame(&f, a, par, track, ws.as_ref())?;
        let prefix = series_prefix(*k, p);
        if let Some(w) = track_w.as_mut() {
            write_track_rows(w, *k, &f.particles.ids, &out.grains, prev_labels.as_deref())?;
            prev_labels = Some(by_id(&f.particles.ids, &out.grains));
        }
        write_grain_rows(&mut grains_w, &prefix, &f, &out)?;
        // With antisites: their count per V/I cluster at the end, then the
        // antisite-only clusters (size 0: size counts V and I).
        for c in out
            .clusters
            .clusters
            .iter()
            .chain(&out.clusters.antisite_only)
        {
            let centre = f.cell.wrap(c.centre);
            write!(
                clusters_w,
                "{prefix}{},{},{},{},{},{},{},{},{:.4},{:.4},{:.4}",
                f.meta.timestep,
                c.id,
                out.stable_id(c.grain),
                u8::from(c.multi_grain),
                c.size(),
                c.vacancies,
                c.interstitials,
                c.net(),
                centre[0],
                centre[1],
                centre[2]
            )?;
            if out.antisites.is_some() {
                write!(clusters_w, ",{}", c.antisites)?;
            }
            writeln!(clusters_w)?;
        }
        if let (Some(w), Some(an)) = (antisites_w.as_mut(), &out.antisites) {
            for g in out
                .result
                .grains
                .iter()
                .filter(|g| g.status == Status::Measured)
            {
                for (&(st, ot), c) in g.antisites.iter().flat_map(|x| &x.pairs) {
                    writeln!(
                        w,
                        "{prefix}{},{},{},{st},{ot},{},{},{},{}",
                        f.meta.timestep,
                        g.grain,
                        out.stable_id(g.grain),
                        an.label(st),
                        an.label(ot),
                        c[0],
                        c[1]
                    )?;
                }
            }
        }
        if let Some(w) = events_w.as_mut() {
            for e in &out.events {
                let (kind, st, other) = match *e {
                    Event::New { stable } => ("new", stable as i64, -1),
                    Event::Continued { stable } => ("continued", stable as i64, -1),
                    Event::Split { stable, from } => ("split", stable as i64, from as i64),
                    Event::Merged { stable, into } => ("merged", stable as i64, into as i64),
                };
                writeln!(w, "{prefix}{},{kind},{st},{other}", f.meta.timestep)?;
            }
        }
        if s.defects_xyz {
            let base = s.out.join("defects");
            for (n, text) in defect_xyz(&out, &f, Some(*k), &a.elements) {
                let (path, append) = match s.defects_layout {
                    DefectsLayout::Dirs => {
                        (base.join(k.to_string()).join(format!("{n}.xyz")), false)
                    }
                    DefectsLayout::Files => (base.join(format!("{n}-{k}.xyz")), false),
                    DefectsLayout::Append => (base.join(format!("{n}.xyz")), true),
                };
                std::fs::create_dir_all(path.parent().unwrap())?;
                let mut w = std::fs::OpenOptions::new()
                    .create(true)
                    .write(true)
                    .append(append)
                    .truncate(!append)
                    .open(&path)?;
                w.write_all(&text)?;
            }
        }
        let (i, b) = out.result.totals();
        let measured = out
            .result
            .grains
            .iter()
            .filter(|g| g.status == Status::Measured)
            .count();
        let largest = out
            .clusters
            .clusters
            .iter()
            .map(|c| c.size())
            .max()
            .unwrap_or(0);
        let mb = |b: u64| b as f64 / 1048576.0;
        // The summary row goes last, so a frame interrupted mid-way is redone.
        writeln!(
            summary,
            "{prefix}{},{},{},{},{},{},{},{},{},{},{},{},{},{:.4},{:.2},{:.1},{:.1},{},{}{}",
            f.meta.timestep,
            f.natoms(),
            out.result.grains.len(),
            measured,
            i.sites,
            i.atoms,
            i.vacancies,
            i.interstitials,
            b.vacancies,
            b.interstitials,
            i.residual() + b.residual(),
            out.clusters.clusters.len(),
            largest,
            out.cluster_cutoff,
            t0.elapsed().as_secs_f64(),
            mb(rmda_core::prof::current_rss_bytes()),
            mb(rmda_core::prof::peak_rss_bytes()),
            i.net_vacancies(),
            b.net_vacancies(),
            match (&out.antisites, out.result.antisite_totals()) {
                (Some(_), Some((ai, ab))) => format!(",{ai},{ab}"),
                (Some(_), None) => ",,".to_owned(),
                (None, _) => String::new(),
            },
        )?;
        for w in [&mut grains_w, &mut clusters_w, &mut summary] {
            w.flush()?;
        }
        if let Some(w) = antisites_w.as_mut() {
            w.flush()?;
        }
        if let Some(w) = events_w.as_mut() {
            w.flush()?;
        }
        if let Some(w) = track_w.as_mut() {
            w.flush()?;
        }
        println!(
            "{k}: grains={} measured={measured} V={} I={} gb_V={} gb_I={} clusters={} wall={:.1}s",
            out.result.grains.len(),
            i.vacancies,
            i.interstitials,
            b.vacancies,
            b.interstitials,
            out.clusters.clusters.len(),
            t0.elapsed().as_secs_f64()
        );
    }
    eprintln!("series: tables in {}", s.out.display());
    Ok(())
}

/// `(atom id, grain label)` sorted by id.
pub(crate) fn by_id(ids: &[i64], labels: &[i32]) -> Vec<(i64, i32)> {
    let mut v: Vec<(i64, i32)> = ids.iter().copied().zip(labels.iter().copied()).collect();
    v.sort_unstable();
    v
}

/// One frame's tracking input (track.csv): a `size` row per grain label, and
/// an `overlap` row per (label, previous frame's label) sharing atoms, with
/// the count. Atoms are matched by id; labels below 0 (no grain) are left out,
/// as `Tracker::update` leaves them out.
pub(crate) fn write_track_rows(
    w: &mut dyn Write,
    k: i64,
    ids: &[i64],
    labels: &[i32],
    prev: Option<&[(i64, i32)]>,
) -> Result<()> {
    let n = labels
        .iter()
        .copied()
        .max()
        .map_or(0, |m| (m + 1).max(0) as usize);
    for (g, size) in rmda_core::analysis::grain_track::grain_sizes(labels, n)
        .into_iter()
        .enumerate()
    {
        writeln!(w, "{k},size,{g},-1,{size}")?;
    }
    if let Some(prev) = prev {
        let mut overlap: std::collections::BTreeMap<(i32, i32), usize> = Default::default();
        for (&id, &g) in ids.iter().zip(labels) {
            if g < 0 {
                continue;
            }
            if let Ok(j) = prev.binary_search_by_key(&id, |e| e.0) {
                if prev[j].1 >= 0 {
                    *overlap.entry((g, prev[j].1)).or_insert(0) += 1;
                }
            }
        }
        for ((g, pg), c) in overlap {
            writeln!(w, "{k},overlap,{g},{pg},{c}")?;
        }
    }
    Ok(())
}

/// The stable id grains.csv recorded for each label of frame `k` (labels
/// `0..=max(labels)`), and the next id the run would assign: one past every
/// stable id in grains.csv and grain_events.csv.
pub(crate) fn recorded_ids(out: &Path, k: i64, labels: &[i32]) -> Result<(Vec<u32>, u32)> {
    let n = labels
        .iter()
        .copied()
        .max()
        .map_or(0, |m| (m + 1).max(0) as usize);
    let grains = std::fs::read_to_string(out.join("grains.csv"))?;
    let mut lines = grains.lines();
    let header: Vec<&str> = lines.next().unwrap_or("").split(',').collect();
    let col = |name: &str| {
        header.iter().position(|&h| h == name).ok_or_else(|| {
            Error::invalid(format!(
                "{}: no `{name}` column",
                out.join("grains.csv").display()
            ))
        })
    };
    let (gc, sc) = (col("grain")?, col("stable_grain")?);
    let mut stable: Vec<Option<u32>> = vec![None; n];
    let mut next = 0u32;
    for l in lines {
        let f: Vec<&str> = l.split(',').collect();
        let (Some(g), Some(st)) = (f.get(gc), f.get(sc)) else {
            continue;
        };
        let st: u32 = st
            .parse()
            .map_err(|_| Error::invalid(format!("grains.csv: stable_grain `{st}` is not an id")))?;
        next = next.max(st + 1);
        if f[0].parse::<i64>().ok() == Some(k) {
            if let Some(slot) = g.parse::<usize>().ok().and_then(|g| stable.get_mut(g)) {
                *slot = Some(st);
            }
        }
    }
    if let Ok(ev) = std::fs::read_to_string(out.join("grain_events.csv")) {
        for l in ev.lines().skip(1) {
            let f: Vec<&str> = l.split(',').collect();
            for x in f.iter().rev().take(2) {
                if let Ok(v) = x.parse::<i64>() {
                    if v >= 0 {
                        next = next.max(v as u32 + 1);
                    }
                }
            }
        }
    }
    let stable = stable
        .into_iter()
        .enumerate()
        .map(|(g, s)| {
            s.ok_or_else(|| {
                Error::invalid(format!(
                    "resuming: grains.csv has no stable id for grain {g} of frame {k}; \
                     rerun with --overwrite"
                ))
            })
        })
        .collect::<Result<Vec<u32>>>()?;
    Ok((stable, next))
}

/// Series files ordered by file index. Files without one (a baseline such as
/// `data.firstframe`) come first, numbered -n..-1 in the order given; two
/// files with the same index are refused. Shared by `rmda defects` and `rmda
/// void` series.
pub(crate) fn series_items(files: &[PathBuf]) -> Result<Vec<(i64, PathBuf)>> {
    let unnumbered = files.iter().filter(|p| file_index(p).is_none()).count() as i64;
    let mut next_unnumbered = -unnumbered;
    let mut items: Vec<(i64, PathBuf)> = files
        .iter()
        .map(|p| {
            let k = file_index(p).unwrap_or_else(|| {
                next_unnumbered += 1;
                next_unnumbered - 1
            });
            (k, p.clone())
        })
        .collect();
    items.sort();
    for w in items.windows(2) {
        if w[0].0 == w[1].0 {
            return Err(Error::invalid(format!(
                "files {} and {} have the same index {}",
                w[0].1.display(),
                w[1].1.display(),
                w[0].0
            )));
        }
    }

    Ok(items)
}

/// Prepare a series directory: create it, drop `tables` under `overwrite`,
/// refuse results from other settings (`run.txt` holds `signature`), record
/// the signature, and return the file indices already in `summary.csv`.
pub(crate) fn open_series(
    out: &Path,
    signature: &str,
    tables: &[&str],
    overwrite: bool,
) -> Result<BTreeSet<i64>> {
    std::fs::create_dir_all(out)?;
    let sig_path = out.join("run.txt");
    let summary_path = out.join("summary.csv");
    if overwrite {
        for t in tables {
            let _ = std::fs::remove_file(out.join(t));
        }
    } else if let Ok(old) = std::fs::read_to_string(&sig_path) {
        if old.trim() != signature.trim() && summary_path.exists() {
            return Err(Error::usage(format!(
                "{} holds results from different settings; use --overwrite or another --out\n  \
                 before: {}\n  now:    {}",
                out.display(),
                old.trim(),
                signature.trim()
            )));
        }
    }
    std::fs::write(&sig_path, format!("{signature}\n"))?;
    let done: BTreeSet<i64> = std::fs::read_to_string(&summary_path)
        .map(|t| {
            t.lines()
                .skip(1)
                .filter_map(|l| l.split(',').next()?.parse().ok())
                .collect()
        })
        .unwrap_or_default();
    // The summary row is a frame's last; rows another table holds for a frame
    // without one are from an interrupted frame, which is redone. Drop them,
    // or the redone frame's rows would follow them.
    for t in tables.iter().filter(|&&t| t != "summary.csv") {
        prune_unfinished(&out.join(t), &done)?;
    }
    Ok(done)
}

/// Keep the header and the rows whose first field (the file index) is in
/// `done`; leave the file untouched when it has nothing else.
fn prune_unfinished(path: &Path, done: &BTreeSet<i64>) -> Result<()> {
    let Ok(text) = std::fs::read_to_string(path) else {
        return Ok(());
    };
    let keep = |l: &&str| {
        l.split(',')
            .next()
            .and_then(|k| k.parse::<i64>().ok())
            .map_or(false, |k| done.contains(&k))
    };
    let mut lines = text.lines();
    let header = lines.next().unwrap_or("");
    let rows: Vec<&str> = lines.collect();
    if rows.iter().all(keep) {
        return Ok(());
    }
    let mut out = String::with_capacity(text.len());
    out.push_str(header);
    out.push('\n');
    for l in rows.iter().filter(|l| keep(l)) {
        out.push_str(l);
        out.push('\n');
    }
    std::fs::write(path, out)?;
    Ok(())
}

/// Open a long-format table of a series for appending, writing `header` to a
/// new file and refusing an existing file with other columns.
pub(crate) fn open_table(
    out: &Path,
    name: &str,
    header: &str,
) -> Result<std::io::BufWriter<std::fs::File>> {
    let p = out.join(name);
    let fresh = !p.exists();
    let mut w = std::io::BufWriter::new(
        std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&p)?,
    );
    if fresh {
        writeln!(w, "{header}")?;
    } else {
        // Rows of one layout only: a table from another version is not
        // appended to.
        let old = std::fs::read_to_string(&p)?;
        let have = old.lines().next().unwrap_or("");
        if have != header {
            return Err(Error::invalid(format!(
                "{} has other columns than this version writes; use --overwrite or \
                     another --out\n  file: {have}\n  now:  {header}",
                p.display()
            )));
        }
    }
    Ok(w)
}

/// `<file_index>,<file name>,`: the key every series row starts with.
pub(crate) fn series_prefix(k: i64, p: &Path) -> String {
    let name = p
        .file_name()
        .map(|n| n.to_string_lossy().replace(',', "_"))
        .unwrap_or_default();
    format!("{k},{name},")
}

/// The settings that change results, one line.
pub(crate) fn settings_signature(a: &DefectsArgs) -> String {
    format!(
        "regions={:?} grain_column={:?} boundary_width={:?} min_atoms={} rmsd_cutoff={} \
         reference={:?} patch_size={:?}{} segment={:?} cluster_cutoff={:?}",
        a.regions,
        a.grain_column,
        a.boundary_width,
        a.min_atoms,
        a.rmsd_cutoff,
        a.reference,
        a.patch_size,
        // Only when on, so a default run's signature is unchanged.
        if a.dhcp { " dhcp=true" } else { "" }.to_owned()
            + if a.local_merge == rmda_core::analysis::defects::LocalMerge::Partial {
                " local_merge=partial"
            } else {
                ""
            }
            + if a.resolve_ambiguous {
                " resolve_ambiguous=true"
            } else {
                ""
            },
        a.segment
            .as_ref()
            .map(|s| (s.threshold, s.min_size, s.core_angle, s.stacking_faults)),
        a.cluster_cutoff
    ) + &match &a.ws {
        // Only when on, so a default run's signature is unchanged.
        Some(w) => format!(" ws_reference={:?} ws_affine={}", w.path, w.affine.label()),
        None => String::new(),
    } + if a.antisites { " antisites=true" } else { "" }
}

fn print_summary(out: &FrameOutput) {
    let r = &out.result;
    let (i, b) = r.totals();
    let measured = r
        .grains
        .iter()
        .filter(|g| g.status == Status::Measured)
        .count();
    let largest = out
        .clusters
        .clusters
        .iter()
        .map(|c| c.size())
        .max()
        .unwrap_or(0);
    for (k, v) in [
        ("defects_grains", r.grains.len() as i64),
        ("defects_grains_measured", measured as i64),
        ("defects_sites", i.sites as i64),
        ("defects_atoms", i.atoms as i64),
        ("defects_vacancies", i.vacancies as i64),
        ("defects_interstitials", i.interstitials as i64),
        ("defects_boundary_sites", b.sites as i64),
        ("defects_boundary_atoms", b.atoms as i64),
        ("defects_boundary_vacancies", b.vacancies as i64),
        ("defects_boundary_interstitials", b.interstitials as i64),
        ("defects_net_vacancies", i.net_vacancies()),
        ("defects_boundary_net_vacancies", b.net_vacancies()),
        ("defects_balance_residual", i.residual() + b.residual()),
        ("defects_clusters", out.clusters.clusters.len() as i64),
        ("defects_largest_cluster", largest as i64),
    ] {
        println!("{k}: {v}");
    }
    // A withheld grain adds nothing to the totals, so a frame where no grain
    // was measured prints the zeros of a perfect crystal: say so.
    let unmeasured: Vec<_> = r
        .grains
        .iter()
        .filter(|g| g.status != Status::Measured)
        .collect();
    if !unmeasured.is_empty() {
        let atoms: usize = unmeasured.iter().map(|g| g.n_atoms).sum();
        let all: usize = r.grains.iter().map(|g| g.n_atoms).sum();
        eprintln!(
            "defects: warning: {} of {} grain(s) not measured ({} of {} atoms); their \
             defects are not in the totals{}",
            unmeasured.len(),
            r.grains.len(),
            atoms,
            all,
            if measured == 0 {
                " (no grain measured: the zero counts are not a defect-free crystal)"
            } else {
                ""
            }
        );
    }
    if let Some(an) = &out.antisites {
        // Totals, then per (site, occupant) pair in both classes. A grain
        // without a count adds nothing.
        // No grain counted (each refused for want of chemical order, or
        // none measured): not a zero.
        match r.antisite_totals() {
            Some((ai, ab)) => {
                println!("defects_antisites: {ai}");
                println!("defects_boundary_antisites: {ab}");
            }
            None => {
                println!("defects_antisites: nan");
                println!("defects_boundary_antisites: nan");
            }
        }
        for (&(st, ot), c) in &r.antisite_pairs() {
            let (st, ot) = (an.label(st), an.label(ot));
            println!("defects_antisites_{ot}_on_{st}: {}", c[0]);
            println!("defects_boundary_antisites_{ot}_on_{st}: {}", c[1]);
        }
        // Clusters (§16.5): antisites on V/I clusters, and antisite-only
        // clusters.
        let cl = &out.clusters;
        let joined: u32 = cl.clusters.iter().map(|c| c.antisites).sum();
        let largest = cl.antisite_only.iter().map(|c| c.antisites).max();
        println!("defects_clustered_antisites: {joined}");
        println!("defects_antisite_clusters: {}", cl.antisite_only.len());
        println!("defects_largest_antisite_cluster: {}", largest.unwrap_or(0));
    }
    for g in &r.grains {
        let why = match &g.status {
            Status::FitFailed(e) => format!(" ({e})"),
            _ => String::new(),
        };
        let how = match &g.local {
            Some(s) => format!("regions={} extended={}", s.regions, s.extended),
            None => format!(
                "patches={}/{} phase={:.3} faces={:.3}",
                g.n_patches_fitted, g.n_patches, g.phase_agreement, g.face_agreement
            ),
        };
        eprintln!(
            "grain {} (stable {}): {}{} ref={} {how} atoms={} structure={} \
             bond={:.4} pairs={:.3} rms_disp={:.3} offlattice={:.4} snapped={:?} V={} I={} gb_V={} \
             gb_I={} net_V={}{}",
            g.grain,
            out.stable_id(g.grain),
            g.status.label(),
            why,
            g.reference,
            g.n_atoms,
            g.structure,
            g.bond,
            g.pair_fraction,
            g.rms_displacement,
            g.offlattice_fraction,
            g.snapped,
            g.interior.vacancies,
            g.interior.interstitials,
            g.boundary.vacancies,
            g.boundary.interstitials,
            g.interior.net_vacancies(),
            match (&out.antisites, &g.antisites) {
                (Some(an), Some(x)) => format!(
                    " antisites={} gb_antisites={}{}",
                    x.interior,
                    x.boundary,
                    if x.sublattice.is_empty() {
                        String::new()
                    } else {
                        format!(" sublattices={}", sublattice_label(an, &x.sublattice))
                    }
                ),
                (Some(_), None) => " antisites=n/a".to_owned(),
                _ => String::new(),
            },
        );
    }
}

fn sublattice_label(an: &AntisiteOut, s: &[i32]) -> String {
    antisite::sublattice_label(&an.names, s)
}

fn grain_header(antisites: bool) -> String {
    let base =
        "timestep,grain,stable_grain,status,reference,n_patches,n_patches_fitted,phase_agreement,\
     n_atoms,n_crystalline,structure,bond,pair_fraction,pair_rms,rms_displacement,\
     offlattice_fraction,snapped_x,snapped_y,snapped_z,domain_converged,sites,atoms,vacancies,\
     interstitials,gb_sites,gb_atoms,gb_vacancies,gb_interstitials,clusters,largest_cluster,\
     t_xx,t_xy,t_xz,t_yx,t_yy,t_yz,t_zx,t_zy,t_zz,net_vacancies,gb_net_vacancies,local_chunks,\
     local_defect_atoms,local_unresolved,local_regions,local_region_atoms,local_extended_atoms,\
     local_fit_rms,face_agreement";
    if antisites {
        format!("{base},antisites,gb_antisites,sublattice_types")
    } else {
        base.to_owned()
    }
}

fn write_grain_rows(w: &mut dyn Write, prefix: &str, f: &Frame, out: &FrameOutput) -> Result<()> {
    for g in &out.result.grains {
        let measured = g.status == Status::Measured;
        // Counts of an unmeasured grain are withheld as -1, never printed as data.
        let c = |v: u64| if measured { v as i64 } else { -1 };
        let mine = out.clusters.clusters.iter().filter(|k| k.grain == g.grain);
        let (n_cl, largest) = mine.fold((0u64, 0u32), |(n, m), k| (n + 1, m.max(k.size())));
        let t = g.t;
        // Patch figures mean nothing for a local grain, and local ones
        // nothing for the others: left empty, never printed as data.
        let (patches, fitted, phase, faces) = match g.local {
            Some(_) => (String::new(), String::new(), String::new(), String::new()),
            None => (
                g.n_patches.to_string(),
                g.n_patches_fitted.to_string(),
                format!("{:.6}", g.phase_agreement),
                format!("{:.6}", g.face_agreement),
            ),
        };
        let local = match &g.local {
            Some(s) => format!(
                "{},{},{},{},{},{},{:.6}",
                s.chunks,
                s.defect_atoms,
                s.unresolved,
                s.regions,
                s.region_atoms,
                s.extended,
                s.frame_rms
            ),
            None => ",,,,,,".to_owned(),
        };
        let net = |c: &ClassCount| {
            if measured {
                c.net_vacancies().to_string()
            } else {
                "-1".to_owned()
            }
        };
        write!(
            w,
            "{prefix}{},{},{},{},{},{},{},{},{},{},{},{:.6},{:.6},{:.6},{:.6},{:.6},{},{},{},{},\
             {},{},{},{},{},{},{},{},{},{},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},\
             {},{},{},{}",
            f.meta.timestep,
            g.grain,
            out.stable_id(g.grain),
            g.status.label(),
            g.reference,
            patches,
            fitted,
            phase,
            g.n_atoms,
            g.n_crystalline,
            g.structure,
            g.bond,
            g.pair_fraction,
            g.pair_rms,
            g.rms_displacement,
            g.offlattice_fraction,
            u8::from(g.snapped[0]),
            u8::from(g.snapped[1]),
            u8::from(g.snapped[2]),
            u8::from(g.domain_converged),
            c(g.interior.sites),
            c(g.interior.atoms),
            c(g.interior.vacancies),
            c(g.interior.interstitials),
            c(g.boundary.sites),
            c(g.boundary.atoms),
            c(g.boundary.vacancies),
            c(g.boundary.interstitials),
            if measured { n_cl as i64 } else { -1 },
            if measured { largest as i64 } else { -1 },
            t[0][0],
            t[0][1],
            t[0][2],
            t[1][0],
            t[1][1],
            t[1][2],
            t[2][0],
            t[2][1],
            t[2][2],
            net(&g.interior),
            net(&g.boundary),
            local,
            faces,
        )?;
        // Antisites: -1 withheld (unmeasured), empty for a grain without a
        // count.
        if let Some(an) = &out.antisites {
            match (&g.antisites, measured) {
                (_, false) => write!(w, ",-1,-1,")?,
                (Some(x), true) => write!(
                    w,
                    ",{},{},{}",
                    x.interior,
                    x.boundary,
                    sublattice_label(an, &x.sublattice)
                )?,
                (None, true) => write!(w, ",,,")?,
            }
        }
        writeln!(w)?;
    }
    Ok(())
}

pub(crate) fn lattice_header(f: &Frame) -> String {
    let h = f.cell.h_matrix();
    let (a, b, c) = (h.row(0), h.row(1), h.row(2));
    let o = f.cell.origin;
    let pbc: Vec<&str> = f
        .cell
        .pbc
        .iter()
        .map(|&p| if p { "T" } else { "F" })
        .collect();
    format!(
        "Lattice=\"{} {} {} {} {} {} {} {} {}\" Origin=\"{} {} {}\" pbc=\"{}\" Timestep={}",
        a[0],
        a[1],
        a[2],
        b[0],
        b[1],
        b[2],
        c[0],
        c[1],
        c[2],
        o[0],
        o[1],
        o[2],
        pbc.join(" "),
        f.meta.timestep
    )
}

/// The text of each defect file of a frame, by name (`vacancies`,
/// `interstitials`, and `non_lattice` when asked for). A series frame
/// carries `index=<index>` in its comment line.
fn defect_xyz(
    out: &FrameOutput,
    f: &Frame,
    index: Option<i64>,
    elements: &[String],
) -> Vec<(&'static str, Vec<u8>)> {
    let r = &out.result;
    let ntypes = f.particles.types.iter().copied().max().unwrap_or(0) as usize;
    let names =
        rmda_core::data::elements::type_names(elements, &f.type_labels, &f.type_masses, ntypes);
    let mut head = lattice_header(f);
    if let Some(k) = index {
        head = format!("index={k} {head}");
    }
    let mut files = Vec::new();
    let mut w: Vec<u8> = Vec::new();
    writeln!(w, "{}", r.vacancies.len()).unwrap();
    writeln!(
        w,
        "{head} Properties=species:S:1:pos:R:3:grain:I:1:boundary:I:1:cluster:I:1{}{}",
        if out.vacancy_shell.is_some() {
            ":non_lattice_shell:I:1"
        } else {
            ""
        },
        if out.ws_sites.is_some() {
            ":site_id:I:1"
        } else {
            ""
        }
    )
    .unwrap();
    for (k, v) in r.vacancies.iter().enumerate() {
        let p = f.cell.wrap(v.position);
        write!(
            w,
            "Va {} {} {} {} {} {}",
            p[0],
            p[1],
            p[2],
            out.stable_id(v.grain),
            u8::from(v.boundary),
            out.clusters.vacancy_cluster[k]
        )
        .unwrap();
        if let Some(s) = &out.vacancy_shell {
            write!(w, " {}", s[k]).unwrap();
        }
        if let Some((s, _)) = &out.ws_sites {
            write!(w, " {}", s[k]).unwrap();
        }
        writeln!(w).unwrap();
    }
    files.push(("vacancies", std::mem::take(&mut w)));
    writeln!(w, "{}", r.interstitials.len()).unwrap();
    writeln!(
        w,
        "{head} Properties=species:S:1:pos:R:3:id:I:1:type:I:1:grain:I:1:boundary:I:1:cluster:I:1{}",
        if out.ws_sites.is_some() {
            ":site_id:I:1"
        } else {
            ""
        }
    )
    .unwrap();
    for (k, s) in r.interstitials.iter().enumerate() {
        let i = s.atom as usize;
        let p = f.particles.positions[i];
        let t = f.particles.types[i];
        let label = names
            .get((t as usize).saturating_sub(1))
            .map_or("X", String::as_str);
        write!(
            w,
            "{label} {} {} {} {} {} {} {} {}",
            p[0],
            p[1],
            p[2],
            f.particles.ids[i],
            t,
            out.stable_id(s.grain),
            u8::from(s.boundary),
            out.clusters.interstitial_cluster[k]
        )
        .unwrap();
        if let Some((_, s)) = &out.ws_sites {
            write!(w, " {}", s[k]).unwrap();
        }
        writeln!(w).unwrap();
    }
    files.push(("interstitials", std::mem::take(&mut w)));
    if let Some(an) = &out.antisites {
        writeln!(w, "{}", r.antisites.len()).unwrap();
        writeln!(
            w,
            "{head} Properties=species:S:1:pos:R:3:id:I:1:type:I:1:site_type:I:1:\
             site_species:S:1:site_pos:R:3:grain:I:1:boundary:I:1{}:cluster:I:1",
            if an.site_ids.is_some() {
                ":site_id:I:1"
            } else {
                ""
            }
        )
        .unwrap();
        for (k, x) in r.antisites.iter().enumerate() {
            let i = x.atom as usize;
            let p = f.particles.positions[i];
            let q = f.cell.wrap(x.site_position);
            let t = an.types[i];
            let name = |t: i32| {
                usize::try_from(t - 1)
                    .ok()
                    .and_then(|k| an.names.get(k))
                    .map_or("X", String::as_str)
                    .to_owned()
            };
            write!(
                w,
                "{} {} {} {} {} {} {} {} {} {} {} {} {}",
                name(t),
                p[0],
                p[1],
                p[2],
                f.particles.ids[i],
                t,
                x.site_type,
                name(x.site_type),
                q[0],
                q[1],
                q[2],
                out.stable_id(x.grain),
                u8::from(x.boundary)
            )
            .unwrap();
            if let Some(s) = &an.site_ids {
                write!(w, " {}", s[k]).unwrap();
            }
            writeln!(w, " {}", out.clusters.antisite_cluster[k]).unwrap();
        }
        files.push(("antisites", std::mem::take(&mut w)));
    }
    if let Some(atoms) = &out.non_lattice {
        writeln!(w, "{}", atoms.len()).unwrap();
        writeln!(
            w,
            "{head} Properties=species:S:1:pos:R:3:id:I:1:type:I:1:StructureType:I:1:grain:I:1"
        )
        .unwrap();
        for &(i, s, g) in atoms {
            let i = i as usize;
            let p = f.particles.positions[i];
            let t = f.particles.types[i];
            let label = names
                .get((t as usize).saturating_sub(1))
                .map_or("X", String::as_str);
            writeln!(
                w,
                "{label} {} {} {} {} {} {} {}",
                p[0],
                p[1],
                p[2],
                f.particles.ids[i],
                t,
                s,
                out.stable_id(g)
            )
            .unwrap();
        }
        files.push(("non_lattice", w));
    }
    files
}

/// Keep only the frames of an appended extended-XYZ file whose `index=` is in
/// `done`.
fn prune_appended(path: &Path, done: &BTreeSet<i64>) -> Result<()> {
    let Ok(text) = std::fs::read_to_string(path) else {
        return Ok(());
    };
    let mut keep = String::with_capacity(text.len());
    let mut lines = text.split_inclusive('\n');
    let mut changed = false;
    while let Some(count) = lines.next() {
        let Ok(n) = count.trim().parse::<usize>() else {
            changed = true;
            break;
        };
        let Some(comment) = lines.next() else {
            changed = true;
            break;
        };
        let body: Vec<&str> = lines.by_ref().take(n).collect();
        let index = comment
            .split_whitespace()
            .find_map(|t| t.strip_prefix("index="))
            .and_then(|v| v.parse::<i64>().ok());
        if body.len() == n && index.is_some_and(|k| done.contains(&k)) {
            keep.push_str(count);
            keep.push_str(comment);
            body.iter().for_each(|l| keep.push_str(l));
        } else {
            changed = true;
        }
    }
    if changed {
        std::fs::write(path, keep)?;
    }
    Ok(())
}

use crate::prelude::*;

pub(crate) fn command(ctx: &Ctx) -> Result<()> {
    let cli = ctx.cli;
    let par = ctx.par;
    let path = ctx.path.clone();
    let load_frame = || ctx.load_frame();
    let Cmd::Defects {
        regions,
        grain_column,
        boundary_width,
        min_atoms,
        rmsd_cutoff,
        table,
        defects,
        segment,
        seg_threshold,
        seg_min_size,
        seg_stacking_faults,
        seg_mode,
        ptm_order: ptm_order_arg,
        reference,
        patch_size,
        local_merge,
        resolve_ambiguous,
        ws_reference,
        ws_affine,
        antisites,
        dhcp,
        files,
        out,
        defects_xyz,
        defects_layout,
        elements,
        non_lattice_atoms,
        overwrite,
        shard,
        cluster_cutoff,
        ..
    } = &cli.cmd
    else {
        unreachable!("dispatched by command");
    };
    // `-o` is the global per-atom output, which this command does not
    // write; ignoring it silently would leave users looking for a file.
    if let Some(o) = &cli.output {
        return Err(rmda_core::error::Error::invalid(format!(
            "rmda defects does not write -o ({}): use --table FILE for the per-grain \
                 table, --defects DIR for vacancies.xyz/interstitials.xyz, or --out DIR \
                 for a time series",
            o.display()
        )));
    }
    let defects_xyz = *defects_xyz || defects_layout.is_some();
    if defects_layout.is_some() && out.is_none() {
        return Err(rmda_core::error::Error::invalid(
            "--defects-layout lays out a series' defect files: give --out DIR",
        ));
    }
    let writes_defect_files = defects.is_some() || (out.is_some() && defects_xyz);
    if *non_lattice_atoms && !writes_defect_files {
        return Err(rmda_core::error::Error::invalid(
            "--non-lattice-atoms writes beside the defect files: give --defects DIR, \
                 or --out DIR with --defects-xyz",
        ));
    }
    let ws = match ws_reference {
        Some(p) => {
            let _t = rmda_core::prof::StageTimer::start("io.read_reference_frame");
            let mut rf = rmda_core::io::read_frame_windowed_with(
                p,
                None,
                None,
                1,
                &read_options(cli, true),
            )?;
            apply_stated_cell(cli, &mut rf)?;
            eprintln!("ws reference: {} atom(s) from {}", rf.natoms(), p.display());
            Some(cmd_defects::WsArgs {
                path: p.clone(),
                frame: rf,
                affine: ws_affine.to_mapping(),
            })
        }
        None => None,
    };
    let args = cmd_defects::DefectsArgs {
        ws,
        regions: regions.clone(),
        grain_column: grain_column.clone(),
        boundary_width: *boundary_width,
        min_atoms: *min_atoms,
        rmsd_cutoff: *rmsd_cutoff,
        table: table.clone(),
        defects_dir: defects.clone(),
        reference: reference.to_mode(),
        patch_size: *patch_size,
        local_merge: local_merge.to_mode(),
        resolve_ambiguous: *resolve_ambiguous,
        dhcp: *dhcp,
        ptm_order: ptm_order(*ptm_order_arg, *segment && *seg_mode == SegModeArg::Auto),
        segment: segment.then(|| rmda_core::analysis::grain_seg::SegConfig {
            mode: seg_mode.to_mode(),
            threshold: *seg_threshold,
            min_size: *seg_min_size,
            stacking_faults: *seg_stacking_faults == StackingFaults::Merge,
            ..Default::default()
        }),
        cluster_cutoff: *cluster_cutoff,
        non_lattice: *non_lattice_atoms,
        antisites: *antisites,
        elements: elements.clone(),
    };
    match out {
        Some(dir) => {
            let mut all = files.clone();
            if all.is_empty() {
                all.push(path.clone());
            }
            let read = |p: &std::path::Path| -> Result<Frame> {
                let mut f = rmda_core::io::read_frame_windowed_with(
                    p,
                    cli.start.or(cli.frame),
                    cli.end,
                    cli.stride,
                    &read_options(cli, true),
                )?;
                apply_stated_cell(cli, &mut f)?;
                Ok(f)
            };
            cmd_defects::run_series(
                &all,
                &args,
                &cmd_defects::SeriesArgs {
                    out: dir.clone(),
                    defects_xyz,
                    defects_layout: match defects_layout.unwrap_or(DefectsLayoutArg::Dirs) {
                        DefectsLayoutArg::Dirs => cmd_defects::DefectsLayout::Dirs,
                        DefectsLayoutArg::Files => cmd_defects::DefectsLayout::Files,
                        DefectsLayoutArg::Append => cmd_defects::DefectsLayout::Append,
                    },
                    overwrite: *overwrite,
                    shard: *shard,
                },
                &read,
                &par,
            )?;
        }
        None => {
            if files.len() > 1 {
                eprintln!("error: several input files need --out DIR (time series mode)");
                std::process::exit(2);
            }
            let frame = load_frame()?;
            if cmd_defects::run(&frame, &args, &par)? {
                // design §21: results written, but no grain was
                // measured. A series records this per frame instead.
                eprintln!("defects: exit 5: no grain was measured");
                std::process::exit(5);
            }
        }
    }
    Ok(())
}
