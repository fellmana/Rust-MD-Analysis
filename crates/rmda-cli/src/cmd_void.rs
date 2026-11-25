//! `rmda void`, method A (design §17): voids as clusters of the vacancy sites
//! of the defects census (`rmda defects`, same references), for one file or
//! a time series of files.
//!
//! Series mode (`--out DIR`) uses the `rmda defects` series machinery
//! (`cmd_defects::{series_items, open_series, open_table}`): files ordered by
//! their trailing index, long-format `summary.csv` (one row per frame) and
//! `voids.csv` (one row per void per frame), resume by skipping frames already
//! in `summary.csv`, and `run.txt` refusing a rerun with other settings. Each
//! frame is counted exactly as a single-file `rmda void` counts it. With
//! `--segment` the grains are tracked across frames as `rmda defects --out`
//! tracks them: `voids.csv` and `grains.csv` carry each grain's stable id,
//! `grain_events.csv` the tracker's events, and a shard records `track.csv`
//! for `rmda merge` to replay. `--sites-xyz` writes each frame's void sites to
//! `sites/<index>.xyz`.

use crate::cmd_defects::{self, DefectsArgs};
use rmda_core::analysis::defects::explicit::ExplicitReference;
use rmda_core::analysis::defects::voids::{self, VoidClass, VoidConfig, Voids};
use rmda_core::analysis::defects::{DefectsResult, Status};
use rmda_core::analysis::grain_track::{Event, Tracker};
use rmda_core::data::frame::Frame;
use rmda_core::error::Result;
use rmda_core::exec::Par;
use std::io::Write;
use std::path::{Path, PathBuf};

pub struct VoidArgs {
    pub min_sites: usize,
    pub cutoff: Option<f64>,
    pub table: Option<PathBuf>,
    pub sites: Option<PathBuf>,
}

/// Series-mode settings.
pub struct VoidSeriesArgs {
    pub out: PathBuf,
    pub overwrite: bool,
    /// `--shard I/N`: this run is shard `I` of `N` (as `rmda defects`).
    pub shard: Option<(usize, usize)>,
    /// `--sites-xyz`: each frame's void vacancy sites in `sites/<index>.xyz`.
    pub sites_xyz: bool,
}

/// Columns of the per-void table after any series key.
const VOID_COLUMNS: &str = "void,sites,volume,radius,x,y,z,grain,multi_grain,class";

/// The census of one frame and its voids.
fn analyse(
    f: &Frame,
    a: &DefectsArgs,
    v: &VoidArgs,
    par: &Par,
    ws: Option<&ExplicitReference>,
) -> Result<(DefectsResult, Voids)> {
    let result = cmd_defects::census_with(f, a, par, ws)?;
    let found = find_voids(f, &result, v, par)?;
    report_unmeasured(&result);
    Ok((result, found))
}

/// The voids of a frame's census.
fn find_voids(f: &Frame, result: &DefectsResult, v: &VoidArgs, par: &Par) -> Result<Voids> {
    let _t = rmda_core::prof::StageTimer::start("void.clusters");
    voids::find(
        &f.cell,
        &f.particles.positions,
        result,
        &VoidConfig {
            min_sites: v.min_sites,
            cutoff: v.cutoff,
        },
        par,
    )
}

fn report_unmeasured(result: &DefectsResult) {
    for g in result
        .grains
        .iter()
        .filter(|g| g.status != Status::Measured)
    {
        eprintln!(
            "void: grain {} not measured ({}); its vacancies are not counted",
            g.grain,
            g.status.label()
        );
    }
}

fn measured(result: &DefectsResult) -> usize {
    result
        .grains
        .iter()
        .filter(|g| g.status == Status::Measured)
        .count()
}

/// One per-void row, `prefix` (the series key, or empty) first; with
/// `stable`, the grain's stable id follows its label (`stable_grain`).
fn write_void_rows(
    w: &mut dyn Write,
    prefix: &str,
    found: &Voids,
    stable: Option<&dyn Fn(i32) -> i64>,
) -> Result<()> {
    for d in &found.voids {
        let st = match stable {
            Some(f) => format!(",{}", f(d.grain)),
            None => String::new(),
        };
        writeln!(
            w,
            "{prefix}{},{},{:.6},{:.6},{:.6},{:.6},{:.6},{}{st},{},{}",
            d.id,
            d.n_sites,
            d.volume,
            d.radius,
            d.centre[0],
            d.centre[1],
            d.centre[2],
            d.grain,
            d.multi_grain as u8,
            d.class.label()
        )?;
    }
    Ok(())
}

/// The void vacancy sites as extended XYZ (`--sites`, and a series'
/// `sites/<index>.xyz` with `index=K` first in the comment line); `grain`
/// maps a label to the id written.
fn sites_xyz(
    f: &Frame,
    result: &DefectsResult,
    found: &Voids,
    index: Option<i64>,
    grain: &dyn Fn(i32) -> i64,
) -> (usize, String) {
    let mut rows = String::new();
    let mut n = 0usize;
    for (vac, id) in result.vacancies.iter().zip(&found.vacancy_void) {
        if let Some(id) = id {
            n += 1;
            rows.push_str(&format!(
                "VAC {:.6} {:.6} {:.6} {} {} {}\n",
                vac.position[0],
                vac.position[1],
                vac.position[2],
                id,
                grain(vac.grain),
                found.voids[*id as usize].class.label()
            ));
        }
    }
    let index = index.map_or(String::new(), |k| format!("index={k} "));
    let text = format!(
        "{n}\n{index}{} Properties=species:S:1:pos:R:3:void_id:I:1:grain:I:1:class:S:1\n{rows}",
        cmd_defects::lattice_header(f)
    );
    (n, text)
}

pub fn run(f: &Frame, a: &DefectsArgs, v: &VoidArgs, par: &Par) -> Result<()> {
    let ws = cmd_defects::prepare_ws(a, par)?;
    let (result, found) = analyse(f, a, v, par, ws.as_ref())?;
    println!("void_grains: {}", result.grains.len());
    println!("void_grains_measured: {}", measured(&result));
    println!("void_vacancies: {}", result.vacancies.len());
    println!("void_min_sites: {}", found.min_sites);
    println!("void_cutoff: {:.4}", found.cutoff);
    println!("voids: {}", found.voids.len());
    for c in [VoidClass::Bulk, VoidClass::Boundary, VoidClass::Surface] {
        println!("voids_{}: {}", c.label(), found.count(c));
    }
    println!("void_sites: {}", found.n_sites());
    println!("void_isolated_vacancies: {}", found.n_isolated);
    println!("void_volume: {:.4}", found.volume());
    println!(
        "void_largest_sites: {}",
        found.voids.first().map_or(0, |w| w.n_sites)
    );
    println!(
        "void_largest_radius: {:.4}",
        found.voids.first().map_or(0.0, |w| w.radius)
    );

    if let Some(p) = &v.table {
        let mut w = std::io::BufWriter::new(std::fs::File::create(p)?);
        writeln!(w, "{VOID_COLUMNS}")?;
        write_void_rows(&mut w, "", &found, None)?;
        eprintln!("void: per-void table written to {}", p.display());
    }
    if let Some(p) = &v.sites {
        let (n, text) = sites_xyz(f, &result, &found, None, &|g| g as i64);
        std::fs::write(p, text)?;
        eprintln!("void: {n} void site(s) written to {}", p.display());
    }
    Ok(())
}

/// The settings that change a void series' results, one line: the census's
/// (`rmda defects`) and the clustering's, marked as a void run so a defects
/// directory is not appended to.
fn settings_signature(a: &DefectsArgs, v: &VoidArgs) -> String {
    format!(
        "void min_sites={} cutoff={:?} {}",
        v.min_sites,
        v.cutoff,
        cmd_defects::settings_signature(a)
    )
}

/// A time series over `files`, one frame each, read by `read`.
pub fn run_series(
    files: &[PathBuf],
    a: &DefectsArgs,
    v: &VoidArgs,
    s: &VoidSeriesArgs,
    read: &dyn Fn(&Path) -> Result<Frame>,
    par: &Par,
) -> Result<()> {
    let all = cmd_defects::series_items(files)?;
    // A shard takes its contiguous block into its own directory, as `rmda
    // defects --shard`: tracked, it writes grain labels where stable ids go
    // and records what the tracker decides from in track.csv, for `rmda
    // merge` to replay.
    let plan = cmd_defects::plan_shard(&all, s.shard, &s.out);
    let items = plan.items.clone();
    let lead_in = plan.lead_in.clone();
    let s = &VoidSeriesArgs {
        out: plan.out.clone(),
        overwrite: s.overwrite,
        shard: s.shard,
        sites_xyz: s.sites_xyz,
    };
    let tracking = a.segment.is_some() && a.ws.is_none();
    let record_tracks = tracking && s.shard.is_some();
    let done = cmd_defects::open_series(
        &s.out,
        &settings_signature(a, v),
        &[
            "summary.csv",
            "voids.csv",
            "grains.csv",
            "grain_events.csv",
            "track.csv",
        ],
        s.overwrite,
    )?;
    if s.overwrite {
        let _ = std::fs::remove_dir_all(s.out.join("sites"));
    }
    cmd_defects::write_shard_file(&s.out, s.shard, lead_in.as_ref(), &items, all.len())?;
    let mut summary = cmd_defects::open_table(
        &s.out,
        "summary.csv",
        "file_index,file,timestep,natoms,grains,grains_measured,vacancies,min_sites,cutoff,\
         voids,voids_bulk,voids_boundary,voids_surface,void_sites,isolated_vacancies,volume,\
         largest_sites,largest_radius,wall_s",
    )?;
    let void_columns = VOID_COLUMNS.replace(",grain,", ",grain,stable_grain,");
    let mut voids_w = cmd_defects::open_table(
        &s.out,
        "voids.csv",
        &format!("file_index,file,timestep,{void_columns}"),
    )?;
    let mut grains_w = cmd_defects::open_table(
        &s.out,
        "grains.csv",
        "file_index,file,timestep,grain,stable_grain,status,reference,n_atoms,vacancies,voids,\
         void_sites",
    )?;
    let mut events_w = if tracking && !record_tracks {
        Some(cmd_defects::open_table(
            &s.out,
            "grain_events.csv",
            "file_index,file,timestep,event,stable,other",
        )?)
    } else {
        None
    };
    let mut track_w = if record_tracks {
        Some(cmd_defects::open_table(
            &s.out,
            "track.csv",
            "file_index,kind,label,other,count",
        )?)
    } else {
        None
    };
    // An explicit reference is read, labelled and indexed once for the series.
    let ws = cmd_defects::prepare_ws(a, par)?;
    let mut tracker = Tracker::new();
    // The frame before the first one to do seeds the tracking, as in `rmda
    // defects`: resuming, the last finished frame (its stable ids are in
    // grains.csv); a shard starting fresh, the frame before its block.
    let mut prev_labels: Option<Vec<(i64, i32)>> = None;
    if tracking {
        let last_done = items.iter().rev().find(|(k, _)| done.contains(k));
        if let Some((k, p)) = last_done.or(lead_in.as_ref()) {
            eprintln!(
                "series: segmenting {} to continue grain tracking",
                p.display()
            );
            let f = read(p)?;
            let settings = cmd_defects::frame_settings(a)?;
            let labels = rmda_core::analysis::defects::frame::grain_labels(&f, &settings, par)?;
            if !record_tracks {
                let (stable, next) = cmd_defects::recorded_ids(&s.out, *k, &labels)?;
                tracker.seed(&f.particles.ids, &labels, &stable, next);
            }
            prev_labels = Some(cmd_defects::by_id(&f.particles.ids, &labels));
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
        let track = if tracking && !record_tracks {
            Some(&mut tracker)
        } else {
            None
        };
        let c = cmd_defects::census_tracked(&f, a, par, track, ws.as_ref())?;
        let found = find_voids(&f, &c.result, v, par)?;
        report_unmeasured(&c.result);
        let result = &c.result;
        let stable = |g: i32| c.stable_id(g);
        let prefix = cmd_defects::series_prefix(*k, p);
        if let Some(w) = track_w.as_mut() {
            cmd_defects::write_track_rows(
                w,
                *k,
                &f.particles.ids,
                &c.grains,
                prev_labels.as_deref(),
            )?;
            prev_labels = Some(cmd_defects::by_id(&f.particles.ids, &c.grains));
        }
        write_void_rows(
            &mut voids_w,
            &format!("{prefix}{},", f.meta.timestep),
            &found,
            Some(&stable),
        )?;
        for g in &result.grains {
            let (nv, ns) = found
                .voids
                .iter()
                .filter(|d| d.grain == g.grain)
                .fold((0usize, 0usize), |(nv, ns), d| {
                    (nv + 1, ns + d.n_sites as usize)
                });
            writeln!(
                grains_w,
                "{prefix}{},{},{},{},{},{},{},{},{}",
                f.meta.timestep,
                g.grain,
                stable(g.grain),
                g.status.label(),
                g.reference,
                g.n_atoms,
                result
                    .vacancies
                    .iter()
                    .filter(|x| x.grain == g.grain)
                    .count(),
                nv,
                ns
            )?;
        }
        if let Some(w) = events_w.as_mut() {
            for e in &c.events {
                let (kind, st, other) = match *e {
                    Event::New { stable } => ("new", stable as i64, -1),
                    Event::Continued { stable } => ("continued", stable as i64, -1),
                    Event::Split { stable, from } => ("split", stable as i64, from as i64),
                    Event::Merged { stable, into } => ("merged", stable as i64, into as i64),
                };
                writeln!(w, "{prefix}{},{kind},{st},{other}", f.meta.timestep)?;
            }
        }
        if s.sites_xyz {
            let (_, text) = sites_xyz(&f, result, &found, Some(*k), &stable);
            let dir = s.out.join("sites");
            std::fs::create_dir_all(&dir)?;
            std::fs::write(dir.join(format!("{k}.xyz")), text)?;
        }
        for w in [&mut voids_w, &mut grains_w] {
            w.flush()?;
        }
        for w in [events_w.as_mut(), track_w.as_mut()].into_iter().flatten() {
            w.flush()?;
        }
        let largest = found.voids.first();
        // The summary row goes last, so a frame interrupted mid-way is redone.
        writeln!(
            summary,
            "{prefix}{},{},{},{},{},{},{:.4},{},{},{},{},{},{},{:.4},{},{:.4},{:.2}",
            f.meta.timestep,
            f.natoms(),
            result.grains.len(),
            measured(result),
            result.vacancies.len(),
            found.min_sites,
            found.cutoff,
            found.voids.len(),
            found.count(VoidClass::Bulk),
            found.count(VoidClass::Boundary),
            found.count(VoidClass::Surface),
            found.n_sites(),
            found.n_isolated,
            found.volume(),
            largest.map_or(0, |w| w.n_sites),
            largest.map_or(0.0, |w| w.radius),
            t0.elapsed().as_secs_f64(),
        )?;
        summary.flush()?;
        println!(
            "{k}: vacancies={} voids={} void_sites={} volume={:.4} wall={:.1}s",
            result.vacancies.len(),
            found.voids.len(),
            found.n_sites(),
            found.volume(),
            t0.elapsed().as_secs_f64()
        );
    }
    eprintln!("series: tables in {}", s.out.display());
    Ok(())
}

use crate::prelude::*;

/// `rmda void`: method A on the defect census, or method B with
/// `--experimental`.
pub(crate) fn command(ctx: &Ctx) -> Result<()> {
    let Cmd::Void { experimental, .. } = &ctx.cli.cmd else {
        unreachable!("dispatched by command");
    };
    if *experimental {
        crate::cmd_void_b::command(ctx)
    } else {
        method_a(ctx)
    }
}

pub(crate) fn method_a(ctx: &Ctx) -> Result<()> {
    let cli = ctx.cli;
    let par = ctx.par;
    let path = ctx.path.clone();
    let load_frame = || ctx.load_frame();
    let Cmd::Void {
        min_sites,
        cutoff,
        table,
        sites,
        site_reference,
        patch_size,
        ws_reference,
        ws_affine,
        grain_regions,
        grain_column,
        segment,
        boundary_width,
        min_atoms,
        rmsd_cutoff,
        dhcp,
        files,
        out,
        overwrite,
        shard,
        sites_xyz,
        ..
    } = &cli.cmd
    else {
        unreachable!("dispatched by command");
    };
    if let Some(o) = &cli.output {
        return Err(rmda_core::error::Error::invalid(format!(
            "rmda void does not write -o ({}): use --table FILE for the per-void \
                 table or --sites FILE for the void vacancy sites",
            o.display()
        )));
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
        regions: grain_regions.clone(),
        grain_column: grain_column.clone(),
        boundary_width: *boundary_width,
        min_atoms: *min_atoms,
        rmsd_cutoff: *rmsd_cutoff,
        table: None,
        defects_dir: None,
        reference: site_reference.to_mode(),
        patch_size: *patch_size,
        local_merge: rmda_core::analysis::defects::LocalMerge::Union,
        resolve_ambiguous: false,
        dhcp: *dhcp,
        ptm_order: ptm_order(None, false),
        segment: segment.then(rmda_core::analysis::grain_seg::SegConfig::default),
        cluster_cutoff: None,
        non_lattice: false,
        antisites: false,
        elements: Vec::new(),
    };
    let vargs = cmd_void::VoidArgs {
        min_sites: *min_sites,
        cutoff: *cutoff,
        table: table.clone(),
        sites: sites.clone(),
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
            cmd_void::run_series(
                &all,
                &args,
                &vargs,
                &cmd_void::VoidSeriesArgs {
                    out: dir.clone(),
                    overwrite: *overwrite,
                    shard: *shard,
                    sites_xyz: *sites_xyz,
                },
                &read,
                &par,
            )?;
        }
        None => {
            if files.len() > 1 {
                return Err(rmda_core::error::Error::usage(
                    "several input files need --out DIR (time series mode)",
                ));
            }
            cmd_void::run(&load_frame()?, &args, &vargs, &par)?;
        }
    }
    Ok(())
}
