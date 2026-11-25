//! `rmda ws`.

use crate::prelude::*;

pub(crate) fn command(ctx: &Ctx) -> Result<()> {
    let cli = ctx.cli;
    let par = ctx.par;
    let load_frame = || ctx.load_frame();
    let Cmd::Ws {
        lattice,
        nn_distance,
        fit_self,
        reference,
        map,
        cutoff,
        orientation,
        tolerance,
        void_min_sites,
        defects,
        search_sample,
        ..
    } = &cli.cmd
    else {
        unreachable!("dispatched by command");
    };
    let frame = load_frame()?;
    let f = &frame;
    // The reference lattice is anchored at the box corner (`with_origin`
    // below), not at absolute coordinate zero. Every LAMMPS `write_data` file
    // has nonzero bounds (the HEA fixtures sit at xlo = 2.3213...), and the
    // reader stores positions unwrapped *and* `cell.origin`, so sites
    // generated from [0,0,0] would sit -xlo off the atoms: 2.32 A of offset
    // on a 2.49 A nearest-neighbour spacing reports ~64k vacancies in a
    // *perfect* synthetic crystal.
    // The two analyses share the assignment and occupancy core; they differ
    // only in where the reference lattice comes from, so the dispatch is on
    // `Analysis` rather than on a separate code path per mode.
    if *fit_self && reference.is_some() {
        return Err(rmda_core::error::Error::invalid(
            "--self and --reference ask for two different references: \
                 --self fits a lattice, --reference uses a frame's sites",
        ));
    }
    let ws: Box<dyn Analysis> = if *fit_self {
        let seed = orientation.as_ref().map(|q| [q[0], q[1], q[2], q[3]]);
        let mut opts = rmda_core::lattice::fit::FitOptions {
            seed_orientation: seed,
            search_sample: *search_sample,
            ..Default::default()
        };
        if let Some(t) = *tolerance {
            opts.tolerance = Some(t);
        }
        Box::new(
            rmda_core::analysis::ws::WsSelf::new(lattice.build(*nn_distance))
                .with_options(opts)
                .with_void_min_sites(*void_min_sites),
        )
    } else if let Some(refpath) = reference {
        if orientation.is_some() {
            return Err(rmda_core::error::Error::invalid(
                "--orientation only applies to --self",
            ));
        }
        if *void_min_sites != 1 {
            return Err(rmda_core::error::Error::invalid(
                "--void-min-sites only applies to --self: a reference frame \
                     has no fitted region to cluster",
            ));
        }
        if *search_sample != 100_000 {
            return Err(rmda_core::error::Error::invalid(
                "--search-sample only applies to --self: a reference frame's \
                     sites are given, not searched",
            ));
        }
        let _t = rmda_core::prof::StageTimer::start("io.read_reference_frame");
        let rframes = read_frames(refpath)?;
        drop(_t);
        let rf = rframes.first().ok_or_else(|| {
            rmda_core::error::Error::invalid(format!(
                "{} contains no frames to use as a reference",
                refpath.display()
            ))
        })?;
        eprintln!(
            "reference: {} atom(s) from {}",
            rf.natoms(),
            refpath.display()
        );
        // The explicit-reference census (pure WS, OVITO's rule);
        // `--tolerance` only sets the `ws_config_unmatched` diagnostic.
        let mut cfg = rmda_core::analysis::defects::explicit_ws::ExplicitWs::new(
            rf.cell,
            rf.particles.positions[..rf.natoms()].to_vec(),
        )
        .with_affine(map.build());
        if let Some(t) = *tolerance {
            cfg = cfg.with_tolerance(t);
        }
        if let Some(c) = *cutoff {
            if !c.is_finite() || c <= 0.0 {
                return Err(rmda_core::error::Error::invalid(
                    "cutoff must be positive and finite",
                ));
            }
            eprintln!(
                "warning: --cutoff is ignored: every atom is assigned to its \
                     nearest reference site, with no search radius (OVITO's rule)"
            );
        }
        Box::new(cfg)
    } else {
        if orientation.is_some() {
            return Err(rmda_core::error::Error::invalid(
                "--orientation only applies to --self",
            ));
        }
        if tolerance.is_some() {
            return Err(rmda_core::error::Error::invalid(
                "--tolerance only applies to --self or --reference",
            ));
        }
        if cutoff.is_some() {
            return Err(rmda_core::error::Error::invalid(
                "--cutoff only applies to --reference: a lattice reference \
                     has a known spacing",
            ));
        }
        if *map != MapArg::None {
            return Err(rmda_core::error::Error::invalid(
                "--map only applies to --reference: a lattice reference is \
                     already in the frame's cell",
            ));
        }
        if *search_sample != 100_000 {
            return Err(rmda_core::error::Error::invalid(
                "--search-sample only applies to --self: a given reference \
                     lattice is not searched",
            ));
        }
        if *void_min_sites != 1 {
            return Err(rmda_core::error::Error::invalid(
                "--void-min-sites only applies to --self: a given reference \
                     lattice has no fitted region to cluster",
            ));
        }
        Box::new(
            rmda_core::analysis::ws::WsRef::new(lattice.build(*nn_distance))
                .with_origin(f.cell.origin),
        )
    };
    // Every Wigner-Seitz engine indexes its own reference data, so none of
    // them read the shared list. Building it here would allocate a whole
    // neighbour index over the input frame for nothing, which at 128M atoms
    // is not a rounding error — so it is built only if one is asked for.
    let shared = if ws.cutoffs().is_empty() {
        None
    } else {
        let _t = rmda_core::prof::StageTimer::start("cell_list.build");
        Some(CellList::build(&f.cell, &f.particles.positions, 1.0)?)
    };
    let nk = NeighborhoodContext {
        cell: shared.as_ref(),
    };
    let ctx = AnalysisContext {
        frame: f,
        nkeys: &nk,
        par: &par,
    };
    let mut out = PropertyStore::new();
    let mut summary = Summary::default();
    Analysis::run(ws.as_ref(), &ctx, &mut out, &mut summary)?;
    for (name, v) in &summary.scalars {
        println!("{name}: {v}");
    }
    write_props(cli, f, &out)?;
    if let Some(dir) = defects {
        let Some(dump) = ws.take_defects() else {
            return Err(rmda_core::error::Error::invalid(
                "--defects was given but this analysis kept no site detail; \\
                     nothing to write",
            ));
        };
        let report = rmda_core::analysis::defect_dump::write_defect_files(
            f,
            &dump,
            &rmda_core::analysis::defect_dump::DefectDumpOptions {
                dir: dir.clone(),
                write_empty: true,
                origin: cli.xyz_origin.into(),
            },
        )?;
        println!("defect sites: {}", report.summary());
    }
    Ok(())
}
