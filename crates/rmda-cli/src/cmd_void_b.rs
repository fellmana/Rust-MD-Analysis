//! `rmda void --experimental`: method B, the phi field.

use crate::prelude::*;

pub(crate) fn command(ctx: &Ctx) -> Result<()> {
    let cli = ctx.cli;
    let load_frame = || ctx.load_frame();
    let Cmd::Void {
        experimental,
        lattice,
        nn_distance,
        reference,
        map,
        region,
        samples,
        seed,
        epsilon,
        epsilons,
        strain,
        strain_spacing,
        strain_sigma,
        excess,
        thermal_window,
        files,
        ..
    } = &cli.cmd
    else {
        unreachable!("dispatched by command");
    };
    if files.len() > 1 {
        return Err(rmda_core::error::Error::invalid(
            "method B (--experimental) measures one file; series mode (--out) is method A",
        ));
    }
    run_void_b(
        cli,
        &VoidArgs {
            experimental: *experimental,
            lattice: *lattice,
            nn_distance: *nn_distance,
            reference: reference.clone(),
            map: *map,
            region: region.clone(),
            samples: *samples,
            seed: *seed,
            epsilon: *epsilon,
            epsilons: epsilons.clone(),
            strain: *strain,
            strain_spacing: *strain_spacing,
            strain_sigma: *strain_sigma,
            excess: *excess,
            thermal_window: *thermal_window,
        },
        &load_frame()?,
    )?;
    Ok(())
}

/// Arguments of the `void` subcommand, grouped so the gate checks and the
/// measurement live in one function that a test can call without a CLI.
pub(crate) struct VoidArgs {
    pub(crate) experimental: bool,
    pub(crate) lattice: LatticeArg,
    pub(crate) nn_distance: f64,
    pub(crate) reference: Option<PathBuf>,
    pub(crate) map: MapArg,
    pub(crate) region: Option<String>,
    pub(crate) samples: usize,
    pub(crate) seed: u64,
    pub(crate) epsilon: f64,
    pub(crate) epsilons: Option<String>,
    pub(crate) strain: bool,
    pub(crate) strain_spacing: Option<f64>,
    pub(crate) strain_sigma: Option<f64>,
    pub(crate) excess: bool,
    pub(crate) thermal_window: bool,
}

/// Method B, the phi field (design §17.2), from the command line.
///
/// The reference modes mirror `rmda ws` so both commands agree on what a lattice is: a
/// lattice given on the command line is taken as exact, and `--reference` takes the ideal
/// configuration from a fit to a reference frame's own atoms; the sample itself is never
/// fitted. What differs is what the reference is *for*. Here it supplies the ideal
/// neighbour directions that become the cell face normals, not a set of sites to
/// compare against, so no site assignment and no defect count happen at all -- which is
/// also why `--self` has no equivalent here: a phi field needs an ideal configuration,
/// and there is no such thing as "the configuration this frame would have if it were
/// perfect" once the frame is the thing being measured.
pub(crate) fn run_void_b(cli: &Cli, a: &VoidArgs, f: &rmda_core::data::frame::Frame) -> Result<()> {
    use rmda_core::analysis::roto_strain::{RotoStrainField, RotoStrainOptions};
    use rmda_core::analysis::void_b::{
        circumradius, estimate_void_volume, excess_void_volume, extrapolate_shell_free,
        extrapolate_to_phi_one, face_normals, max_face_gap, window_epsilons, PhiField,
    };
    use rmda_core::cell::Coord3;
    use rmda_core::lattice::fit::{fit_region, FitOptions};

    check_void_args(a)?;

    // Refuse rather than quietly do nothing. `-o` is a global flag, so a user who
    // passes it expects a file; silently producing none is how someone concludes the
    // void region is empty when in fact nothing was measured per atom. Method B's
    // answer is a region integral, and there is no per-atom quantity to write --
    // which is a fact about the method, and worth saying out loud.
    if let Some(p) = &cli.output {
        return Err(rmda_core::error::Error::invalid(format!(
            "rmda void cannot write {}: method B reports a region integral, not a \
             per-atom property, so there is nothing to put in an atom file. The \
             scalars above are the result. For per-atom defect sites use method A: \
             `rmda void FILE --sites OUT.xyz` (method A).",
            p.display()
        )));
    }

    let nn = a.nn_distance;
    let atoms = &f.particles.positions[..f.natoms()];

    let lattice = a.lattice.build(nn);

    // Three things are kept deliberately separate here, because conflating any two of them
    // produces a plausible number that measures the wrong thing:
    //
    //   * `pc` -- the *ideal* configuration, and the only source of the face normals.
    //   * the *sample*: the frame being measured, whose atoms and whose cell are what the
    //     field is evaluated around.
    //   * the *reference* frame, under `--reference`, which supplies `pc` and (under
    //     `--excess`) a second field to difference against.
    //
    // Taking the sample's own atoms for the ideal directions would make phi measure every
    // atom against its own displacement, and taking the reference's atoms for the sample
    // field would make `--excess` difference the reference against itself -- which is
    // exactly the bug this split exists to make impossible.

    // The ideal configuration. A reference frame supplies its scale and orientation by a fit
    // over its own atoms, the same public route `rmda ws --self` uses, so `pc` is a measured
    // primitive cell rather than the declared lattice assumed onto someone else's box.
    let ref_frame = match &a.reference {
        Some(p) => {
            let rframes = read_frames(p)?;
            let rf = rframes.into_iter().next().ok_or_else(|| {
                rmda_core::error::Error::invalid(format!(
                    "{} contains no frames to use as a reference",
                    p.display()
                ))
            })?;
            eprintln!("reference: {} atom(s) from {}", rf.natoms(), p.display());
            Some(rf)
        }
        None => None,
    };
    let pc = match &ref_frame {
        Some(rf) => {
            let _t = rmda_core::prof::StageTimer::start("void.fit_reference");
            let fit = fit_region(
                &rf.cell,
                &rf.particles.positions[..rf.natoms()],
                lattice.as_ref(),
                nn,
                &FitOptions::default(),
            )?;
            println!(
                "ideal configuration: fitted to the reference frame, a = {:.6} A, {} orientation",
                fit.cell.h.col_norm(0),
                fit.orientation
                    .iter()
                    .map(|q| format!("{q:.4}"))
                    .collect::<Vec<_>>()
                    .join(" ")
            );
            fit.cell
        }
        None => lattice.primitive(),
    };
    let normals = face_normals(&pc)?;
    println!(
        "phi field: lattice {:?}, nn {nn:.4} A, {} face normal(s), circumradius {:.4} A",
        a.lattice,
        normals.len(),
        circumradius(&normals)?
    );

    // Measure what the atoms are actually doing and compare. This is not a nicety:
    // a declared `nn` that disagrees with the sample does not error, it returns a field
    // with the wrong cell size, and the sign of the error decides what a user sees. Too
    // large and the ideal cells swallow the sample, so a perfect crystal and a crater
    // both read as zero void; too small and every interstice reads as void. Both look
    // like a result. The check costs one neighbour search and turns the silent case into
    // a line of output.
    let measured = rmda_core::lattice::fit::estimate_nn(&f.cell, atoms, nn)?;
    let drift = (measured - nn) / nn;
    println!(
        "observed nn: {measured:.4} A ({:+.1} % vs --nn-distance)",
        100.0 * drift
    );
    if drift.abs() > 0.10 {
        eprintln!(
            "warning: --nn-distance {nn:.4} A differs from the sample's measured {:.4} A by \
             {:+.1} %. The ideal cells will not match the crystal, and the void volume below \
             measures that mismatch rather than void. Pass the lattice's real nearest-neighbour \
             distance, or give --reference to take it from a reference configuration.",
            measured,
            100.0 * drift
        );
    }

    // Which cell and which positions the sample's field is built on. `--map to-reference`
    // is the same operation it is in `rmda ws`, on the same term: the field is periodic
    // through the cell it is given, so when the cell has strained between frames the
    // positions have to be carried into the reference cell or every distance measured is
    // the wrong one.
    let (field_cell, field_atoms): (rmda_core::cell::UnitCell, Vec<Coord3>) =
        match (ref_frame.as_ref(), a.map.build()) {
            (Some(rf), rmda_core::analysis::ws::AffineMapping::ToReferenceCell) => {
                let lam = atoms
                    .iter()
                    .map(|&p| f.cell.x2lamda(p))
                    .collect::<Result<Vec<_>>>()?;
                let mapped = lam.iter().map(|&l| rf.cell.lamda2x(l)).collect();
                (rf.cell, mapped)
            }
            (_, rmda_core::analysis::ws::AffineMapping::ToReferenceCell) => {
                return Err(rmda_core::error::Error::invalid(
                    "--map to-reference needs --reference: there is no reference cell to map into",
                ))
            }
            _ => (f.cell, atoms.to_vec()),
        };
    println!(
        "atoms: {} in the sample, {} ideal face normal(s), epsilon {:.3e}",
        field_atoms.len(),
        normals.len(),
        a.epsilon
    );

    let strain_field = if a.strain {
        let a0 = pc.h.col_norm(0);
        let opts = RotoStrainOptions {
            node_spacing: a.strain_spacing.unwrap_or(a0 / 2.0),
            sigma: a.strain_sigma.unwrap_or(a0),
            refine_passes: 1,
            max_deviation: None,
            fill_unsolved: true,
        };
        println!(
            "roto-strain: mesh {:.3} A, sigma {:.3} A, {} pass(es)",
            opts.node_spacing, opts.sigma, opts.refine_passes
        );
        let _t = rmda_core::prof::StageTimer::start("void.roto_strain");
        let s = RotoStrainField::fit(&f.cell, atoms, &pc, &opts)?;
        eprintln!(
            "roto-strain: {} nodes, {} pairs, max |T - I| {:.4}, unsolved {}",
            s.len_nodes(),
            s.n_pairs(),
            s.max_deviation_from_identity(),
            s.unsolved()
        );
        Some(s)
    } else {
        None
    };

    // Searched for the largest threshold measured: a field built for a smaller one misses atoms
    // and counts the points they hold as void (`PhiField::max_epsilon`).
    let face_gap = if a.thermal_window {
        let g = max_face_gap(&field_cell, &field_atoms, &pc)?;
        println!(
            "largest face gap: {:.4} over {} matched neighbour pair(s)",
            g.max, g.bonds
        );
        Some(g)
    } else {
        None
    };
    let eps_field = match (&a.epsilons, face_gap) {
        (Some(spec), _) => parse_floats(spec, "--epsilons")?
            .into_iter()
            .fold(a.epsilon, f64::max),
        (None, Some(g)) => window_epsilons(g.max)?
            .into_iter()
            .fold(a.epsilon, f64::max),
        (None, None) => a.epsilon,
    };
    let field = {
        let _t = rmda_core::prof::StageTimer::start("void.build_field");
        PhiField::for_threshold_strained(&field_cell, &field_atoms, &pc, eps_field, strain_field)?
    };
    let region = match &a.region {
        Some(spec) => parse_region(spec)?,
        None => default_region(&field_cell, nn),
    };
    let region_vol: f64 = region
        .iter()
        .map(|(lo, hi)| (hi - lo).max(0.0))
        .product::<f64>();
    println!("region: [{region_vol:.0} A^3]");

    let _t = rmda_core::prof::StageTimer::start("void.integrate");
    if a.excess {
        let rf = ref_frame
            .as_ref()
            .ok_or_else(|| rmda_core::error::Error::invalid("--excess needs --reference"))?;
        let ref_positions = rf.particles.positions[..rf.natoms()].to_vec();
        let ref_field = PhiField::for_threshold(&rf.cell, &ref_positions, &pc, eps_field)?;
        let eps = match &a.epsilons {
            Some(spec) => parse_floats(spec, "--epsilons")?,
            None => vec![a.epsilon],
        };
        let x = excess_void_volume(&field, &ref_field, &region, &eps, a.samples, a.seed)?;
        println!("excess over reference: {}", x.summary());
        for (e, m) in eps.iter().zip(&x.measurements) {
            println!(
                "  epsilon {e:.4}: {:>12.2} +/- {:>8.2} A^3 ({} / {} hits)",
                m.volume, m.standard_error, m.hits, m.samples
            );
        }
        return Ok(());
    }

    let est = estimate_void_volume(&field, &region, a.epsilon, a.samples, a.seed)?;
    println!("void volume: {}", est.summary());
    println!(
        "  {:>12.2} A^3 of {:>12.0} A^3 = {:.4} % of the region",
        est.volume,
        est.region_volume,
        100.0 * est.volume / est.region_volume
    );

    if let Some(g) = face_gap {
        let x = extrapolate_shell_free(&field, &region, g, a.samples, a.seed)?;
        println!("extrapolated to phi = 1: {}", x.summary());
        for (e, m) in x.epsilons.iter().zip(&x.measurements) {
            println!(
                "  epsilon {e:.4}: {:>12.2} +/- {:>8.2} A^3",
                m.volume, m.standard_error
            );
        }
    }
    if let Some(spec) = &a.epsilons {
        let eps = parse_floats(spec, "--epsilons")?;
        let x = extrapolate_to_phi_one(&field, &region, &eps, a.samples, a.seed)?;
        println!("extrapolated to phi = 1: {}", x.summary());
        println!(
            "  falling in epsilon: {}, fit trustworthy: {}",
            x.slope_negative(),
            x.is_trustworthy()
        );
        for (e, m) in eps.iter().zip(&x.measurements) {
            println!(
                "  epsilon {e:.4}: {:>12.2} +/- {:>8.2} A^3",
                m.volume, m.standard_error
            );
        }
    }
    let _ = cli;
    Ok(())
}

/// Everything the `void` subcommand will refuse, refused before it spends any time.
///
/// These are argument relationships rather than measurements, so they are checked up
/// front: a run that computes a 256k-atom field and then discovers `--map` was
/// meaningless has wasted the user's minute and taught them nothing.
pub(crate) fn check_void_args(a: &VoidArgs) -> Result<()> {
    let bad = rmda_core::error::Error::invalid;
    if !a.experimental {
        return Err(bad(
            "method B is experimental and needs --experimental to run.\n\
             It is validated on synthetic void geometry, and measured NOT to recover the void \
             fraction of a relaxed crystal: on a defect-free 256k-atom alloy it reports ~2.9% of \
             the box as void, and the roto-strain field removes only ~5% of that, because an \
             affine strain cannot represent non-affine thermal displacement (design §17.2).\n\
             For a void census that is trustworthy, use method A instead:\n    \
             rmda void FILE   (method A, without --experimental)"
                .to_owned(),
        ));
    }
    if a.samples == 0 {
        return Err(bad(
            "--samples must be at least 1: the integral has nothing to average otherwise"
                .to_owned(),
        ));
    }
    if !a.epsilon.is_finite() || a.epsilon < 0.0 {
        return Err(bad(format!(
            "--epsilon must be finite and non-negative, got {}",
            a.epsilon
        )));
    }
    if a.excess && a.reference.is_none() {
        return Err(bad(
            "--excess needs --reference: an excess is a difference between two fields, and \
             without a reference there is no second field to difference against"
                .to_owned(),
        ));
    }
    if a.reference.is_none() && a.map != MapArg::None {
        return Err(bad("--map only applies with --reference".to_owned()));
    }
    if (a.strain_spacing.is_some() || a.strain_sigma.is_some()) && !a.strain {
        return Err(bad(
            "--strain-spacing/--strain-sigma only apply with --strain".to_owned(),
        ));
    }
    if let Some(spec) = &a.epsilons {
        let eps = parse_floats(spec, "--epsilons")?;
        // A non-finite threshold cannot be ordered, so the extrapolation
        // cannot run: an invalid request (exit 2), not a panic.
        if eps.iter().any(|e| !e.is_finite()) {
            return Err(rmda_core::error::Error::usage(format!(
                "--epsilons must be finite, got {eps:?}"
            )));
        }
        if eps.len() < 3 {
            return Err(bad(format!(
                "--epsilons needs at least 3 values to fit a quadratic, got {}. With fewer it \
                 degenerates to a straight line, which overstates a defect-free crystal by orders \
                 of magnitude; use --epsilon for a single-threshold measurement instead.",
                eps.len()
            )));
        }
        let mut sorted = eps.clone();
        sorted.sort_by(|x, y| x.total_cmp(y));
        sorted.dedup_by(|x, y| x == y);
        if sorted.len() != eps.len() {
            return Err(bad(format!(
                "--epsilons must be distinct, got {eps:?}: three copies of one epsilon give \
                 three rows of the same design matrix",
                eps = eps
            )));
        }
    }
    if let Some(spec) = &a.region {
        parse_region(spec)?;
    }
    Ok(())
}
