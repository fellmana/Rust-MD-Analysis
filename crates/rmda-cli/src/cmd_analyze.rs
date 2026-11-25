//! `rmda analyze`.

use crate::prelude::*;

pub(crate) fn command(ctx: &Ctx) -> Result<()> {
    let cli = ctx.cli;
    let par = ctx.par;
    let path = ctx.path.clone();
    let Cmd::Analyze {
        ptm,
        ptm_rmsd,
        ptm_structures,
        rdf,
        rdf_rmax,
        rdf_dr,
        rdf_no_partials,
        rdf_no_coordination,
        rdf_volume,
        rdf_edge_correct,
        wc,
        wc_shells,
        cluster,
        cluster_cutoff,
        cluster_knear,
        cluster_min_size,
        ..
    } = &cli.cmd
    else {
        unreachable!("dispatched by command");
    };
    run_analyze(
        cli,
        &path,
        &AnalyzeArgs {
            ptm: *ptm,
            ptm_rmsd: *ptm_rmsd,
            ptm_structures: ptm_structures.clone(),
            rdf: *rdf,
            rdf_rmax: *rdf_rmax,
            rdf_dr: *rdf_dr,
            rdf_no_partials: *rdf_no_partials,
            rdf_no_coordination: *rdf_no_coordination,
            rdf_volume: *rdf_volume,
            rdf_edge_correct: *rdf_edge_correct,
            wc: *wc,
            wc_shells: wc_shells.clone(),
            cluster: *cluster,
            cluster_cutoff: *cluster_cutoff,
            cluster_knear: *cluster_knear,
            cluster_min_size: *cluster_min_size,
        },
        &par,
    )?;
    Ok(())
}

/// Arguments of the `analyze` subcommand, grouped so the frame loop and the
/// per-frame measurement live in functions a test can call without a CLI.
pub(crate) struct AnalyzeArgs {
    pub(crate) ptm: bool,
    pub(crate) ptm_rmsd: f64,
    pub(crate) ptm_structures: Vec<PtmTypeArg>,
    pub(crate) rdf: bool,
    pub(crate) rdf_rmax: Option<f64>,
    pub(crate) rdf_dr: f64,
    pub(crate) rdf_no_partials: bool,
    pub(crate) rdf_no_coordination: bool,
    pub(crate) rdf_volume: RdfVolume,
    pub(crate) rdf_edge_correct: bool,
    pub(crate) wc: bool,
    pub(crate) wc_shells: String,
    pub(crate) cluster: bool,
    pub(crate) cluster_cutoff: Option<f64>,
    pub(crate) cluster_knear: Option<usize>,
    pub(crate) cluster_min_size: usize,
}

/// The `analyze` subcommand: read the window once, run the requested analyses
/// over each frame with a single per-frame neighbour index, and print every
/// frame's scalars and histograms in a diffable order.
pub(crate) fn run_analyze(cli: &Cli, path: &Path, a: &AnalyzeArgs, par: &Par) -> Result<()> {
    let ptm_on = a.ptm;
    if !(ptm_on || a.rdf || a.wc || a.cluster) {
        return Err(rmda_core::error::Error::invalid(
            "analyze needs at least one analysis: --ptm, --rdf, --wc, --cluster",
        ));
    }

    // Validate the whole request before spending time building anything: a bad
    // flag reported as a failed neighbour search points at the wrong thing.
    let mut flags = 0u32;
    for t in &a.ptm_structures {
        flags |= t.to_flag();
    }
    if ptm_on && flags == 0 {
        return Err(rmda_core::error::Error::invalid(
            "--ptm-structures selected no structures",
        ));
    }
    if !a.ptm_rmsd.is_finite() || a.ptm_rmsd <= 0.0 {
        return Err(rmda_core::error::Error::invalid(format!(
            "--ptm-rmsd must be positive and finite, got {}",
            a.ptm_rmsd
        )));
    }
    let rdf = if a.rdf {
        let rmax = match a.rdf_rmax {
            Some(r) if r.is_finite() && r > 0.0 => r,
            Some(r) => {
                return Err(rmda_core::error::Error::invalid(format!(
                    "--rdf-rmax must be positive and finite, got {r}"
                )))
            }
            None => {
                return Err(rmda_core::error::Error::invalid(
                    "--rdf needs --rdf-rmax: a largest pair distance is what defines the range",
                ))
            }
        };
        if !a.rdf_dr.is_finite() || a.rdf_dr <= 0.0 {
            return Err(rmda_core::error::Error::invalid(format!(
                "--rdf-dr must be positive and finite, got {}",
                a.rdf_dr
            )));
        }
        Some(Rdf {
            rmax,
            dr: a.rdf_dr,
            partials: !a.rdf_no_partials,
            coordination: !a.rdf_no_coordination,
            volume: a.rdf_volume,
            edge_correct: a.rdf_edge_correct,
        })
    } else {
        None
    };
    let wc = if a.wc {
        // Parsed before the loop: a malformed rank list is a bad request, not a
        // failure inside the frame measurement.
        let nneigh = parse_shells(&a.wc_shells)?;
        Some(WarrenCowley {
            nneigh,
            only_selected: false,
            only_same_type: false,
            edge_correct: false,
            per_particle: false,
        })
    } else {
        None
    };
    let cluster = if !a.cluster {
        None
    } else {
        let ms = a.cluster_min_size;
        match (a.cluster_cutoff, a.cluster_knear) {
            (Some(c), _) if c.is_finite() && c > 0.0 => Some(ClusterAnalysis {
                mode: ClusterMode::Cutoff { cutoff: c },
                min_size: ms,
                ..Default::default()
            }),
            (Some(c), _) => {
                return Err(rmda_core::error::Error::invalid(format!(
                    "--cluster-cutoff must be positive and finite, got {c}"
                )))
            }
            (None, Some(k)) if k > 0 => Some(ClusterAnalysis {
                mode: ClusterMode::KNear { k },
                min_size: ms,
                ..Default::default()
            }),
            (None, Some(_)) => {
                return Err(rmda_core::error::Error::invalid(
                    "--cluster-knear must be at least 1",
                ))
            }
            // `conflicts_with` already refuses both flags together at parse time.
            (None, None) => {
                return Err(rmda_core::error::Error::invalid(
                    "--cluster needs --cluster-cutoff or --cluster-knear",
                ))
            }
        }
    };

    eprintln!(
        "analyze: {}",
        [
            if ptm_on {
                Some(format!(
                    "ptm(rmsd={}, types={})",
                    a.ptm_rmsd,
                    a.ptm_structures
                        .iter()
                        .map(|t| t.name())
                        .collect::<Vec<_>>()
                        .join(",")
                ))
            } else {
                None
            },
            rdf.as_ref()
                .map(|r| format!("rdf(rmax={}, dr={})", r.rmax, r.dr)),
            if a.wc {
                Some(format!("wc(shells={})", a.wc_shells))
            } else {
                None
            },
            cluster.as_ref().map(|c| match &c.mode {
                ClusterMode::Cutoff { cutoff } => format!("cluster(cutoff={cutoff})"),
                ClusterMode::KNear { k } => format!("cluster(knear={k})"),
                m @ ClusterMode::PairCutoffs { .. } => format!("cluster({})", m.describe()),
            }),
        ]
        .into_iter()
        .flatten()
        .collect::<Vec<_>>()
        .join(" + ")
    );

    let _t_read = rmda_core::prof::StageTimer::start("io.read_frames");
    let it = rmda_core::io::FrameReader::open(path)?
        .with_options(read_options(cli, true))
        .into_frames()?;
    drop(_t_read);
    let mut n_frames = 0usize;
    let mut first_atoms = 0usize;
    // Pull the window by hand so the parse (which `for` would perform in its own
    // hidden `next()` call) lands inside the `io.read_frames` timer; a body-side
    // timer around `res?` measures only the unwrap and reports zero.
    let mut wit = rmda_core::io::windowed(it, cli.start.or(cli.frame), cli.end, cli.stride);
    loop {
        let _t_read = rmda_core::prof::StageTimer::start("io.read_frames");
        let res = wit.next();
        drop(_t_read);
        let Some(frame) = res else {
            break;
        };
        let mut frame = frame?;
        apply_stated_cell(cli, &mut frame)?;
        if n_frames == 0 {
            first_atoms = frame.natoms();
        }
        analyze_frame(
            cli,
            par,
            a,
            ptm_on,
            flags,
            rdf.as_ref(),
            wc.as_ref(),
            cluster.as_ref(),
            n_frames,
            &frame,
        )?;
        n_frames += 1;
    }
    eprintln!("analyze: read {n_frames} frame(s), {first_atoms} atoms in frame 0");
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn analyze_frame(
    cli: &Cli,
    par: &Par,
    a: &AnalyzeArgs,
    ptm_on: bool,
    flags: u32,
    rdf: Option<&Rdf>,
    wc: Option<&WarrenCowley>,
    cluster: Option<&ClusterAnalysis>,
    idx: usize,
    frame: &Frame,
) -> Result<()> {
    use std::io::Write;

    let n = frame.natoms();

    // One neighbour index per frame, wide enough for every analysis that reads
    // it. PTM sizes its radius from the frame density (`Ptm::search_radius`);
    // cluster derives its own per frame (KNear has no radius until it asks).
    // `--rdf` and `--wc` build their own: their rmax / k-th shell exceed what
    // the shared cell list covers, deliberately.
    // PTM builds its own k-NN index, so it adds nothing here.
    let mut radius = 0.0f64;
    if let Some(cl) = cluster {
        radius = radius.max(cl.cutoffs_for(frame).into_iter().fold(0.0, f64::max));
    }
    let owned = if radius > 0.0 {
        let _t = rmda_core::prof::StageTimer::start("cell_list.build");
        Some(CellList::build(
            &frame.cell,
            &frame.particles.positions[..n],
            radius,
        )?)
    } else {
        None
    };
    let nk = NeighborhoodContext {
        cell: owned.as_ref(),
    };
    let ctx = AnalysisContext {
        frame,
        nkeys: &nk,
        par,
    };

    let mut out = PropertyStore::new();
    let mut summary = Summary::default();

    if ptm_on {
        let _t = rmda_core::prof::StageTimer::start("ptm.run");
        Ptm {
            rmsd_cutoff: a.ptm_rmsd,
            flags,
        }
        .run(&ctx, &mut out, &mut summary)?;
    }
    if let Some(cl) = cluster {
        let _t = rmda_core::prof::StageTimer::start("cluster.run");
        cl.run(&ctx, &mut out, &mut summary)?;
    }
    if let Some(r) = rdf {
        let _t = rmda_core::prof::StageTimer::start("rdf.run");
        r.validate(Some(&frame.cell))?;
        r.run(&ctx, &mut out, &mut summary)?;
    }
    if let Some(w) = wc {
        let _t = rmda_core::prof::StageTimer::start("wc.run");
        w.validate(frame)?;
        w.run(&ctx, &mut out, &mut summary)?;
    }

    print_analyze_frame(idx, &summary, rdf);

    let Some(prefix) = cli.output.clone() else {
        return Ok(());
    };
    if let Some(r) = rdf {
        let path = PathBuf::from(format!("{}.rdf.{idx}.dat", prefix.display()));
        let mut file = std::io::BufWriter::new(std::fs::File::create(&path)?);
        writeln!(
            file,
            "# r= {}",
            r.edges()
                .iter()
                .map(|v| format!("{v:.5}"))
                .collect::<Vec<_>>()
                .join(" ")
        )?;
        for (name, series) in &summary.histograms {
            if name.starts_with("rdf_") {
                writeln!(
                    file,
                    "{name}: {}",
                    series
                        .iter()
                        .map(|v| format!("{v:.6}"))
                        .collect::<Vec<_>>()
                        .join(" ")
                )?;
            }
        }
        eprintln!("analyze: wrote g(r) table to {}", path.display());
    }
    // `-o` means a file was expected; a run that produced no per-atom property
    // and no rdf table would silently write nothing, so refuse it out loud.
    if out.is_empty() && rdf.is_none() {
        return Err(rmda_core::error::Error::invalid(format!(
            "rmda analyze cannot write {}: none of the requested analyses produced \
             per-atom properties (--wc and --rdf are frame-level; per-atom columns \
             come from --ptm or --cluster)",
            prefix.display()
        )));
    }
    if !out.is_empty() {
        let path = PathBuf::from(format!("{}.{idx}.xyz", prefix.display()));
        let mut file = std::io::BufWriter::new(std::fs::File::create(&path)?);
        rmda_core::io::XyzWriter::write_frame_with_origin(
            &mut file,
            frame,
            &out,
            cli.xyz_origin.into(),
        )?;
        let names: Vec<&str> = out
            .iter()
            .filter(|p| p.per_particle)
            .map(|p| p.name.as_ref())
            .collect();
        eprintln!(
            "analyze: wrote {} propert{} to {}",
            names.len(),
            if names.len() == 1 { "y" } else { "ies" },
            path.display()
        );
        eprintln!("  {}", names.join(", "));
    }
    Ok(())
}

/// Print one frame's results on stdout: every scalar sorted by name, then the
/// histograms. Bin edges are emitted once, ahead of the `rdf_*` arrays, because
/// a histogram without its abscissa is a list of numbers.
pub(crate) fn print_analyze_frame(idx: usize, summary: &Summary, rdf: Option<&Rdf>) {
    println!("frame {idx}:");
    let mut scalars: Vec<&(String, f64)> = summary.scalars.iter().collect();
    scalars.sort_by(|x, y| x.0.cmp(&y.0));
    for (name, v) in scalars {
        println!("  {name}: {v}");
    }
    let mut hists: Vec<&(String, Vec<f64>)> = summary.histograms.iter().collect();
    hists.sort_by(|x, y| x.0.cmp(&y.0));
    if let Some(r) = rdf {
        println!(
            "  # r= {}",
            r.edges()
                .iter()
                .map(|v| format!("{v:.5}"))
                .collect::<Vec<_>>()
                .join(" ")
        );
    }
    for (name, series) in hists {
        println!(
            "  {name}: {}",
            series
                .iter()
                .map(|v| format!("{v:.6}"))
                .collect::<Vec<_>>()
                .join(" ")
        );
    }
}
