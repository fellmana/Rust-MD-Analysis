//! `rmda structure`.

use crate::prelude::*;

pub(crate) fn command(ctx: &Ctx) -> Result<()> {
    let cli = ctx.cli;
    let par = ctx.par;
    let load_frame = || ctx.load_frame();
    let Cmd::Structure {
        mode,
        cutoff,
        low,
        full,
        ..
    } = &cli.cmd
    else {
        unreachable!("dispatched by command");
    };
    let frame = load_frame()?;
    let f = &frame;
    let cna = Cna::new(mode.to_mode(*cutoff, *low)).with_full(*full);
    let radius = cna.cutoffs_for(f).into_iter().fold(0.0_f64, f64::max);
    let cl = shared_cell_list(f, radius)?;
    let nk = NeighborhoodContext { cell: cl.as_ref() };
    let ctx = AnalysisContext {
        frame: f,
        nkeys: &nk,
        par: &par,
    };
    let mut out = PropertyStore::new();
    let mut summary = Summary::default();
    cna.run(&ctx, &mut out, &mut summary)?;
    for (name, v) in &summary.scalars {
        println!("{name}: {v}");
    }
    // `cna_*_total` comes out of the loop above: `Cna` reports the
    // whole-frame sum of each neighbour count alongside the per-atom
    // array, so the printer does not have to re-derive it.
    write_props(cli, f, &out)?;
    Ok(())
}
