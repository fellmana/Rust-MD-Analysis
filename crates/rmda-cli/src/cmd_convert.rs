//! `rmda convert`.

use crate::prelude::*;

pub(crate) fn command(ctx: &Ctx) -> Result<()> {
    let cli = ctx.cli;
    let path = ctx.path.clone();
    let Cmd::Convert {
        format,
        props,
        precision,
        prop_precision,
        reduce_tilt,
        out_atom_style,
        ..
    } = &cli.cmd
    else {
        unreachable!("dispatched by command");
    };
    use rmda_core::error::Error;
    use rmda_core::io::writer::{planned_props, PropSelect, WriteOptions};
    let fmt = match (format.as_deref(), &cli.output) {
        (Some("xyz"), _) => Format::Xyz,
        (Some("dump"), _) => Format::LammpsDump,
        (Some("data"), _) => Format::LammpsData,
        (Some("bin"), _) => Format::LammpsDumpBinary,
        (Some(f), _) => return Err(Error::invalid(format!("unknown --format {f}"))),
        (None, Some(o)) => Format::from_extension(o).ok_or_else(|| {
            Error::invalid(format!(
                "cannot tell the output format from {}; give --format \
                     (xyz, dump, data, bin) or an -o ending in .xyz, .dump, .data or .bin",
                o.display()
            ))
        })?,
        (None, None) => {
            return Err(Error::invalid(
                "convert needs --format or an -o FILE with a known suffix",
            ))
        }
    };
    let out = cli.output.clone().unwrap_or_else(|| {
        PathBuf::from(match fmt {
            Format::Xyz => "out.xyz",
            Format::LammpsDump => "out.dump",
            Format::LammpsData => "out.data",
            Format::LammpsDumpBinary => "out.bin",
        })
    });
    let gz = rmda_core::io::format::is_gzip(&out);
    if gz && fmt == Format::LammpsDumpBinary {
        return Err(Error::invalid(format!(
            "{}: gzipped binary dumps cannot be read back (by RMDA or LAMMPS); \
                 write {} instead",
            out.display(),
            out.display().to_string().trim_end_matches(".gz")
        )));
    }
    let opts = WriteOptions {
        extended: props.is_some(),
        props: props.as_deref().map_or(PropSelect::None, PropSelect::parse),
        precision: *precision,
        prop_precision: *prop_precision,
        origin: cli.xyz_origin.into(),
        atom_style: out_atom_style.clone(),
    };
    if opts.atom_style.is_some() && fmt != Format::LammpsData {
        return Err(Error::invalid(
            "--out-atom-style applies to data-file output only",
        ));
    }
    let mut file = ConvertOut::create(&out, gz)?;
    // Stream the window, so a trajectory is never materialised before a
    // single byte reaches the output file.
    let _t_read = rmda_core::prof::StageTimer::start("io.read_frames");
    let it = rmda_core::io::FrameReader::open(&path)?
        .with_options(read_options(cli, false))
        .into_frames()?;
    let mut n = 0usize;
    let mut first_atoms = 0usize;
    for res in rmda_core::io::windowed(it, cli.start.or(cli.frame), cli.end, cli.stride) {
        let mut f = res?;
        apply_stated_cell(cli, &mut f)?;
        if *reduce_tilt {
            f.reduce_tilts()?;
        } else if n == 0 && fmt == Format::LammpsData && f.cell.reduce_tilts().is_some() {
            eprintln!(
                "note: a box tilt exceeds half the box length; LAMMPS before \
                     2023 refuses such a data file (--reduce-tilt flips it)"
            );
        }
        if n == 0 {
            first_atoms = f.natoms();
            if opts.props != PropSelect::None || opts.atom_style.is_some() {
                let (written, skipped) = planned_props(fmt, &f, &opts)?;
                eprintln!(
                    "writing {} propert{}: {}",
                    written.len(),
                    if written.len() == 1 { "y" } else { "ies" },
                    if written.is_empty() {
                        "-".to_string()
                    } else {
                        written.join(", ")
                    }
                );
                if !skipped.is_empty() {
                    eprintln!(
                        "note: not written (the format cannot carry them): {}",
                        skipped.join(", ")
                    );
                }
            }
        } else if fmt == Format::LammpsData {
            file.finish()?;
            return Err(Error::invalid(format!(
                "a LAMMPS data file holds one frame and the window has more; \
                     pick one with --frame N ({} holds only the first)",
                out.display()
            )));
        }
        // Every writer is a per-frame loop underneath, so writing one
        // frame at a time is byte-identical to writing the slice.
        rmda_core::io::Writer::write_with(&mut file, fmt, std::slice::from_ref(&f), &opts)?;
        n += 1;
    }
    file.finish()?;
    drop(_t_read);
    eprintln!("read {n} frame(s), {first_atoms} atoms in frame 0");
    eprintln!("wrote {}", out.display());
    Ok(())
}
