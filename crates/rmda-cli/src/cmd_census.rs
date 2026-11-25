//! `rmda census`.

use crate::prelude::*;

pub(crate) fn command(ctx: &Ctx) -> Result<()> {
    let cli = ctx.cli;
    let par = ctx.par;
    let load_frame = || ctx.load_frame();
    let Cmd::Census { cutoff, .. } = &cli.cmd else {
        unreachable!("dispatched by command");
    };
    let frame = load_frame()?;
    let f = &frame;
    let cl = {
        let _t = rmda_core::prof::StageTimer::start("cell_list.build");
        CellList::build(&f.cell, &f.particles.positions, *cutoff)?
    };
    let nk = NeighborhoodContext { cell: Some(&cl) };
    let ctx = AnalysisContext {
        frame: f,
        nkeys: &nk,
        par: &par,
    };
    let mut out = PropertyStore::new();
    let mut summary = Summary::default();
    NeighborCensus::new(*cutoff).run(&ctx, &mut out, &mut summary)?;
    for (name, v) in &summary.scalars {
        println!("{name}: {v}");
    }
    Ok(())
}
