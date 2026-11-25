//! `rmda cluster`.

use crate::prelude::*;

pub(crate) fn command(ctx: &Ctx) -> Result<()> {
    let cli = ctx.cli;
    let par = ctx.par;
    let load_frame = || ctx.load_frame();
    let Cmd::Cluster {
        cutoff,
        knear,
        pair_cutoff,
        min_size,
        select,
        only_selected,
        sort_by_size,
        unwrap,
        table,
        ..
    } = &cli.cmd
    else {
        unreachable!("dispatched by command");
    };
    let mut frame = load_frame()?;
    let select = select
        .as_deref()
        .map(rmda_core::data::select::Selection::parse)
        .transpose()?;
    let options = ClusterAnalysis {
        select,
        only_selected: *only_selected,
        sort_by_size: *sort_by_size,
        unwrap: *unwrap,
        properties: table.is_some(),
        ..Default::default()
    };
    let f = &frame;
    // Neither mode has a usable default. A cutoff that is too large
    // percolates the whole frame into one cluster and too small splits it
    // into one cluster per atom; both are plausible-looking answers to a
    // question nobody asked. Requiring one of the two flags is the same
    // call `Grains::resolve_cutoff` makes when PTM has not run.
    let analysis = match (cutoff, knear) {
        (None, None) if !pair_cutoff.is_empty() => ClusterAnalysis {
            mode: ClusterMode::PairCutoffs {
                pairs: pair_cutoff
                    .iter()
                    .map(|s| rmda_core::analysis::cluster::PairCutoff::parse(s))
                    .collect::<rmda_core::error::Result<Vec<_>>>()?,
            },
            min_size: *min_size,
            ..options
        },
        (Some(c), None) => ClusterAnalysis {
            mode: ClusterMode::Cutoff { cutoff: *c },
            min_size: *min_size,
            ..options
        },
        (None, Some(k)) => ClusterAnalysis {
            mode: ClusterMode::KNear { k: *k },
            min_size: *min_size,
            ..options
        },
        // `conflicts_with` should have caught this, but a subcommand that
        // silently picks one of two contradictory requests is worse than
        // one that refuses.
        _ => {
            return Err(rmda_core::error::Error::usage(
                "cluster needs --cutoff, --knear or --pair-cutoff",
            ));
        }
    };
    // KNear has no configured radius, so it reports a frame-derived one;
    // taking the max of `cutoffs_for` is what lets the pipeline build a
    // list wide enough without the analysis re-deriving it.
    //
    // Only the cutoff mode over every atom reads the shared list: the
    // k-nearest mode and a selection index their own atoms, so building
    // one for them would be a wasted pass.
    let radius = analysis.cutoffs_for(f).into_iter().fold(0.0_f64, f64::max);
    let shared = matches!(analysis.mode, ClusterMode::Cutoff { .. })
        && analysis.select.is_none()
        && !analysis.only_selected;
    let (mut out, summary) = {
        let owned = if radius > 0.0 && shared {
            Some({
                let _t = rmda_core::prof::StageTimer::start("cell_list.build");
                CellList::build(&f.cell, &f.particles.positions, radius)?
            })
        } else {
            None
        };
        let nk = NeighborhoodContext {
            cell: owned.as_ref(),
        };
        let ctx = AnalysisContext {
            frame: f,
            nkeys: &nk,
            par: &par,
        };
        let mut out = PropertyStore::new();
        let mut summary = Summary::default();
        analysis.run(&ctx, &mut out, &mut summary)?;
        (out, summary)
    };
    eprintln!(
        "cluster: mode={} min_size={min_size}",
        analysis.mode.describe()
    );
    for (name, v) in &summary.scalars {
        println!("{name}: {v}");
    }
    for (name, series) in &summary.histograms {
        println!("{name}:");
        for (i, v) in series.iter().enumerate() {
            if *v != 0.0 {
                println!("  size={i} clusters={v}");
            }
        }
    }
    if let Some(path) = table {
        write_cluster_table(path, &out)?;
        eprintln!("cluster: wrote the per-cluster table to {}", path.display());
    }
    // OVITO's `unwrap_particles` rewrites the positions; so does this, in
    // the written file only, and the column it came from is not written
    // a second time.
    if *unwrap {
        let u = out
            .get("UnwrappedPosition")
            .and_then(|p| p.as_f64())
            .expect("unwrap registers UnwrappedPosition")
            .clone();
        for (p, c) in frame.particles.positions.iter_mut().zip(u.chunks_exact(3)) {
            *p = [c[0], c[1], c[2]];
        }
        let mut kept = PropertyStore::new();
        for p in out.iter() {
            if p.name.as_ref() as &str != "UnwrappedPosition" {
                kept.register(p.clone());
            }
        }
        out = kept;
    }
    // `Cluster` and `ParticleCluster` are per-atom and useless without
    // them on disk; `ClusterSize` is per cluster and `write_props` filters
    // it on `per_particle`, same as the grain table's columns.
    write_props(cli, &frame, &out)?;
    Ok(())
}
