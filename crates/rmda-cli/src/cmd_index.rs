//! `rmda index`.

use crate::prelude::*;

pub(crate) fn command(ctx: &Ctx) -> Result<()> {
    let cli = ctx.cli;
    let path = ctx.path.clone();
    let Cmd::Index { .. } = &cli.cmd else {
        unreachable!("dispatched by command");
    };
    let _t = rmda_core::prof::StageTimer::start("io.index");
    if rmda_core::io::format::is_gzip(&path) {
        let idx = rmda_core::io::index::write_gz(&path)?;
        println!(
            "indexed gzip: {} access point(s), {} bytes decompressed -> {}",
            idx.points.len(),
            idx.total_out,
            rmda_core::io::index::index_path(&path).display()
        );
        return Ok(());
    }
    let entries = rmda_core::io::index::write(&path)?;
    let atoms: usize = entries.iter().map(|e| e.natoms).sum();
    println!(
        "indexed {} frame(s), {atoms} atom rows -> {}",
        entries.len(),
        rmda_core::io::index::index_path(&path).display()
    );
    Ok(())
}
