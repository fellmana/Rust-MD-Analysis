//! `rmda wc`.

use crate::prelude::*;

pub(crate) fn command(ctx: &Ctx) -> Result<()> {
    let cli = ctx.cli;
    let par = ctx.par;
    let load_frame = || ctx.load_frame();
    let Cmd::Wc {
        wc_shells,
        only_selected,
        only_same_type,
        per_particle,
        edge_correct,
        ..
    } = &cli.cmd
    else {
        unreachable!("dispatched by command");
    };
    let frame = load_frame()?;
    let f = &frame;
    // Parse before building anything: a malformed rank list is a bad
    // request, and reporting it as a failure of the neighbour search would
    // point at the wrong thing.
    let nneigh = parse_shells(wc_shells)?;
    let wc = WarrenCowley {
        nneigh: nneigh.clone(),
        only_selected: *only_selected,
        only_same_type: *only_same_type,
        per_particle: *per_particle,
        edge_correct: *edge_correct,
    };
    wc.validate(f)?;
    let nk = NeighborhoodContext { cell: None };
    let ctx = AnalysisContext {
        frame: f,
        nkeys: &nk,
        par: &par,
    };
    let mut summary = Summary::default();
    let mut store = PropertyStore::new();
    wc.run(&ctx, &mut store, &mut summary)?;

    // The scalar names carry type *ids*, so the mapping has to be printed
    // or `wc_m1_7_2` is only interpretable by guessing.
    let mut types: Vec<u32> = f.particles.types.clone();
    types.sort_unstable();
    types.dedup();
    let stat = |k: &str| {
        summary
            .scalars
            .iter()
            .find(|(name, _)| name == k)
            .map(|(_, v)| *v)
            .unwrap_or(f64::NAN)
    };
    eprintln!(
        "wc: ranks={} types={} central={:.0} skipped={:.0} probe={:.3} A",
        nneigh
            .iter()
            .map(|k| k.to_string())
            .collect::<Vec<_>>()
            .join(","),
        types
            .iter()
            .map(|t| t.to_string())
            .collect::<Vec<_>>()
            .join(","),
        stat("wc_atoms_central"),
        stat("wc_atoms_skipped"),
        stat("wc_probe_radius"),
    );
    for (name, v) in &summary.scalars {
        println!("{name}: {v}");
    }
    // A species with no central atoms has no `alpha` at all -- matching the
    // reference, which averages an empty axis into `NaN`. Saying so on
    // stderr is the difference between "this alloy has no type-3 order" and
    // "the run was broken".
    let empty: Vec<&str> = summary
        .scalars
        .iter()
        .filter(|(k, v)| k.starts_with("wc_m") && v.is_nan())
        .map(|(k, _)| k.as_str())
        .collect();
    if !empty.is_empty() {
        eprintln!(
            "wc: unmeasured, because no central atom is that type: {}",
            empty.join(" ")
        );
    }
    if *per_particle {
        write_props(cli, f, &store)?;
    } else if let Some(p) = &cli.output {
        return Err(rmda_core::error::Error::invalid(format!(
            "rmda wc would write nothing to {}: the per-atom matrix is behind \
                 --per-particle, and the global numbers above are not per-atom \
                 properties. Rerun with --per-particle, or keep the output with your \
                 shell's redirect.",
            p.display()
        )));
    }
    Ok(())
}
