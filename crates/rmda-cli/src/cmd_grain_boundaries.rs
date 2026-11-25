//! `rmda grain-boundaries`.

use crate::prelude::*;

pub(crate) fn command(ctx: &Ctx) -> Result<()> {
    let cli = ctx.cli;
    let par = ctx.par;
    let load_frame = || ctx.load_frame();
    let Cmd::GrainBoundaries {
        w_db,
        unassigned_is_boundary,
        min_grain_size,
        no_adopt_orphans,
        grain_table,
        ..
    } = &cli.cmd
    else {
        unreachable!("dispatched by command");
    };
    let frame = load_frame()?;
    let f = &frame;
    let a = GrainBoundaries {
        w_db: *w_db,
        unassigned_is_boundary: *unassigned_is_boundary,
    };
    // Validated before anything is built, so a bad `--w-db` reports itself
    // rather than failing later inside a cell-list constructor.
    let width = a
        .w_db
        .unwrap_or(rmda_core::analysis::grain_boundaries::DEFAULT_W_DB);
    if !width.is_finite() || width <= 0.0 {
        eprintln!("error: --w-db must be positive and finite");
        std::process::exit(2);
    }
    // PTM and segmentation run into one store:
    // the pass reads the `Grain` column grains writes, and re-deriving it here
    // would mean a second classifier that could disagree with the first.
    let ptm = Ptm {
        rmsd_cutoff: 0.1,
        flags: PtmTypeArg::all_flags(),
    };
    let radius = ptm.cutoffs_for(f).into_iter().fold(0.0_f64, f64::max);
    let cl = {
        let _t = rmda_core::prof::StageTimer::start("cell_list.build");
        CellList::build(&f.cell, &f.particles.positions, radius.max(width))?
    };
    let nk = NeighborhoodContext { cell: Some(&cl) };
    let ctx = AnalysisContext {
        frame: f,
        nkeys: &nk,
        par: &par,
    };
    let mut out = PropertyStore::new();
    let mut summary = Summary::default();
    ptm.run(&ctx, &mut out, &mut summary)?;
    let grains = Grains {
        misorientation_threshold: 10.0,
        cutoff: None,
        min_size: 1,
        min_grain_size: *min_grain_size,
        stacking_faults: false,
        // An `OTHER` region has no
        // point group and no reference lattice, so it is not a grain,
        // and the `Grain` column must mean the same thing in every
        // command that writes it.
        include_other: false,
        merge_threshold: Some(10.0),
        orphan_adoption: !*no_adopt_orphans,
    };
    grains.run(&ctx, &mut out, &mut summary)?;

    let _t = rmda_core::prof::StageTimer::start("grain_boundaries.scan");
    a.run(&ctx, &mut out, &mut summary)?;

    eprintln!("grain_boundaries: w_db={width} unassigned_is_boundary={unassigned_is_boundary}");
    for key in [
        "gb_atoms_boundary",
        "gb_atoms_interior",
        "gb_fraction_boundary",
        "gb_atoms_assigned",
        "gb_atoms_unassigned",
        "gb_grains",
        "gb_slab_thickness",
        "gb_w_db",
        "gb_unassigned_is_boundary",
    ] {
        if let Some((_, v)) = summary.scalars.iter().find(|(k, _)| *k == key) {
            println!("{key}: {v}");
        }
    }
    // The seam counts are per grain, so the atom table cannot carry them; a
    // table run is the only place they are readable one grain at a time.
    if grain_table.is_none() {
        if let Some(p) = out.get("GrainBoundaryCount").and_then(|p| p.as_i32()) {
            let mut hist: std::collections::BTreeMap<i32, usize> = Default::default();
            for &c in p.iter() {
                *hist.entry(c).or_default() += 1;
            }
            println!("grain_boundary_histogram:");
            for (count, grains) in hist {
                println!("  boundary_atoms={count} grains={grains}");
            }
        }
    }
    if let Some(path) = grain_table {
        let rep = write_grain_table(
            &out,
            &summary,
            &GrainTableOptions {
                path: path.clone(),
                ..Default::default()
            },
        )?;
        eprintln!("grain_table: {}", rep.summary());
    }
    write_props(cli, f, &out)?;
    Ok(())
}
