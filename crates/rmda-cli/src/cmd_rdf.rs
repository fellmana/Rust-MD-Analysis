//! `rmda rdf`.

use crate::prelude::*;

pub(crate) fn command(ctx: &Ctx) -> Result<()> {
    let cli = ctx.cli;
    let par = ctx.par;
    let load_frame = || ctx.load_frame();
    let Cmd::Rdf {
        rmax,
        dr,
        no_partials,
        no_coordination,
        volume,
        edge_correct,
        ..
    } = &cli.cmd
    else {
        unreachable!("dispatched by command");
    };
    let frame = load_frame()?;
    let f = &frame;
    let rdf = Rdf {
        rmax: *rmax,
        dr: *dr,
        partials: !no_partials,
        coordination: !no_coordination,
        volume: *volume,
        edge_correct: *edge_correct,
    };
    // Validate before allocating anything for the query, or a bad range is
    // reported as a failure of the neighbour list rather than of the request.
    rdf.validate(Some(&f.cell))?;
    let cl = {
        let _t = rmda_core::prof::StageTimer::start("cell_list.build");
        CellList::build(&f.cell, &f.particles.positions, *rmax)?
    };
    let nk = NeighborhoodContext { cell: Some(&cl) };
    let ctx = AnalysisContext {
        frame: f,
        nkeys: &nk,
        par: &par,
    };
    eprintln!(
        "rdf: rmax={rmax} dr={dr} partials={} coordination={}",
        rdf.partials, rdf.coordination
    );
    let mut summary = Summary::default();
    let mut store = PropertyStore::new();
    rdf.run(&ctx, &mut store, &mut summary)?;
    // Bin edges first: a histogram without its abscissa is a list of numbers,
    // and the caller would have to know `dr` to use it. `rmax` is printed
    // rather than `nbins` because it is the number the user typed.
    println!(
        "# r={}",
        rdf.edges()
            .iter()
            .map(|r| format!("{r:.5}"))
            .collect::<Vec<_>>()
            .join(" ")
    );
    for (name, series) in &summary.histograms {
        println!(
            "{name}: {}",
            series
                .iter()
                .map(|v| format!("{v:.6}"))
                .collect::<Vec<_>>()
                .join(" ")
        );
    }
    for (name, v) in &summary.scalars {
        println!("{name}: {v}");
    }
    // Refuse `-o` rather than quietly produce no file, for the same reason
    // `rmda void` does: `-o` is a global flag, so passing it means a file was
    // expected, and silently writing nothing is how someone concludes a frame
    // had no pairs in range when in fact nothing was measured per atom. The
    // refusal says where the numbers went.
    if let Some(p) = &cli.output {
        return Err(rmda_core::error::Error::invalid(format!(
            "rmda rdf cannot write {}: g(r) is a frame-level histogram, not a \
                 per-atom property, so there is nothing to put in an atom file. The \
                 histogram above is the result; `> file.dat` or your shell's redirect \
                 keeps it as a table.",
            p.display()
        )));
    }
    Ok(())
}
