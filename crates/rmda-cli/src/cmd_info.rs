//! `rmda info`.

use crate::prelude::*;

pub(crate) fn command(ctx: &Ctx) -> Result<()> {
    let cli = ctx.cli;
    let path = ctx.path.clone();
    let Cmd::Info { .. } = &cli.cmd else {
        unreachable!("dispatched by command");
    };
    // Frame-at-a-time, with the per-frame lines buffered only so the
    // `read N frame(s)` summary can still lead the output exactly as
    // documented. That buffer grows by a few dozen bytes per frame;
    // the frames themselves are dropped as they are printed.
    let _t_read = rmda_core::prof::StageTimer::start("io.read_frames");
    let it = rmda_core::io::FrameReader::open(&path)?
        .with_options(read_options(cli, false))
        .into_frames()?;
    let mut body = String::new();
    let mut n = 0usize;
    let mut first_atoms = 0usize;
    for res in rmda_core::io::windowed(it, cli.start.or(cli.frame), cli.end, cli.stride) {
        let mut f = res?;
        apply_stated_cell(cli, &mut f)?;
        if n == 0 {
            first_atoms = f.natoms();
        }
        let (xlo, xhi, ylo, yhi, zlo, zhi) = {
            let b = f.cell.to_box_bounds();
            (b.0, b.1, b.3, b.4, b.6, b.7)
        };
        body.push_str(&format!(
            "frame {n}: t={} natoms={} ntypes={} cell={} pbc={:?} volume={:.3} Å^3\n",
            f.meta.timestep,
            f.natoms(),
            f.ntypes(),
            if f.cell.is_triclinic() {
                "triclinic"
            } else {
                "orthorhombic"
            },
            f.cell.pbc,
            f.cell.volume()
        ));
        body.push_str(&format!(
            "  lengths: {:.4} {:.4} {:.4}\n",
            xhi - xlo,
            yhi - ylo,
            zhi - zlo
        ));
        let labels: Vec<&str> = f.type_labels.iter().map(|s| s.as_str()).collect();
        if !labels.is_empty() {
            body.push_str(&format!("  types: {}\n", labels.join(" ")));
        }
        n += 1;
    }
    drop(_t_read);
    eprintln!("read {n} frame(s), {first_atoms} atoms in frame 0");
    eprint!("{body}");
    Ok(())
}
