//! Helpers shared by several commands.

use crate::prelude::*;

/// The cell list an analysis asks for through `cutoffs_for`, or none when it
/// asks for none (CNA and PTM build their own k-NN index).
pub(crate) fn shared_cell_list(
    f: &Frame,
    radius: f64,
) -> rmda_core::error::Result<Option<CellList<'_>>> {
    if radius > 0.0 {
        let _t = rmda_core::prof::StageTimer::start("cell_list.build");
        Ok(Some(CellList::build(
            &f.cell,
            &f.particles.positions,
            radius,
        )?))
    } else {
        Ok(None)
    }
}

/// The box the user stated, applied to a frame read from a file.
///
/// A file that says `Lattice=` carries its own box and a file that does not is
/// read as the isolated system it says it is. `--cell A B C` is how a box is
/// stated for a file that has none — the edges are taken as given, the origin
/// is the one the file already had, so stating a box does not move the atoms,
/// and the box is periodic unless `--pbc false` says otherwise.
///
/// `--pbc` on its own overrides the periodicity a file carries, which is how a
/// file with a box is read as the free system it may also be. Neither is an
/// error on a file that already agrees: the file's own answer simply stands.
pub(crate) fn apply_stated_cell(cli: &Cli, f: &mut Frame) -> Result<()> {
    let Some(edges) = cli.cell.as_deref() else {
        if let Some(p) = cli.pbc {
            f.cell = f.cell.with_pbc([p; 3]);
        }
        return Ok(());
    };
    if edges.len() != 3 {
        return Err(rmda_core::error::Error::invalid(format!(
            "--cell takes three edges, got {}",
            edges.len()
        )));
    }
    for (d, e) in edges.iter().enumerate() {
        if !e.is_finite() || *e <= 0.0 {
            return Err(rmda_core::error::Error::invalid(format!(
                "--cell edge {} must be positive and finite, got {e}",
                ['x', 'y', 'z'][d]
            )));
        }
    }
    let pbc = cli.pbc.unwrap_or(true);
    let origin = f.cell.origin;
    f.cell =
        rmda_core::cell::UnitCell::orthorhombic(edges[0], edges[1], edges[2]).with_pbc([pbc; 3]);
    f.cell.origin = origin;
    // The box is stated now, so it is not inferred any more.
    f.meta.cell_inferred = false;
    Ok(())
}

pub(crate) fn parse_region(spec: &str) -> Result<[(f64, f64); 3]> {
    let v = parse_floats(spec, "--region")?;
    if v.len() != 6 {
        return Err(rmda_core::error::Error::invalid(format!(
            "--region needs 6 values, xlo,xhi,ylo,yhi,zlo,zhi; got {} in {spec:?}",
            v.len()
        )));
    }
    let r = [(v[0], v[1]), (v[2], v[3]), (v[4], v[5])];
    for (lo, hi) in r {
        if !(lo.is_finite() && hi.is_finite()) || hi <= lo {
            return Err(rmda_core::error::Error::invalid(format!(
                "--region needs lo < hi on every axis, got [{lo}, {hi}]"
            )));
        }
    }
    Ok(r)
}

/// Comma-separated numbers, named in the error so the user knows which flag to fix.
pub(crate) fn parse_floats(spec: &str, what: &str) -> Result<Vec<f64>> {
    let mut out = Vec::new();
    for (i, part) in spec.split(',').enumerate() {
        let t = part.trim();
        let v = t.parse::<f64>().map_err(|e| {
            rmda_core::error::Error::invalid(format!(
                "{what} value {} ({t:?}) is not a number: {e}",
                i + 1
            ))
        })?;
        out.push(v);
    }
    Ok(out)
}

/// Box interior inset by one nearest-neighbour distance.
///
/// The inset is not cosmetic. The field is periodic through the cell, so a region
/// touching a face integrates across the wrap, where a point's neighbours are the ones
/// the periodic images make and not the ones inside the crystal. A shell of that kind
/// is thin, and on a relaxed fixture it is most of the answer.
pub(crate) fn default_region(cell: &rmda_core::cell::UnitCell, nn: f64) -> [(f64, f64); 3] {
    let (lo, hi) = cell.bounds();
    (0..3)
        .map(|d| {
            // Half a box when the box is smaller than two bonds, so a tiny cell is
            // still measurable rather than inverting.
            let inset = (nn).min((hi[d] - lo[d]) / 4.0);
            (lo[d] + inset, hi[d] - inset)
        })
        .collect::<Vec<_>>()
        .try_into()
        .unwrap()
}

/// `--precision` / `--prop-precision`: `full` or a number of decimals.
pub(crate) fn parse_float_fmt(
    s: &str,
) -> std::result::Result<rmda_core::io::writer::FloatFmt, String> {
    rmda_core::io::writer::FloatFmt::parse(s)
}

/// `rmda convert`'s output file: plain, or gzip-compressed (by the `.gz`
/// suffix) through the parallel member writer.
pub(crate) enum ConvertOut {
    Plain(std::io::BufWriter<std::fs::File>),
    Gz(rmda_core::io::writer::GzWriter<std::fs::File>),
}

impl ConvertOut {
    pub(crate) fn create(path: &Path, gz: bool) -> Result<Self> {
        let f = std::fs::File::create(path)?;
        Ok(if gz {
            ConvertOut::Gz(rmda_core::io::writer::GzWriter::new(f))
        } else {
            ConvertOut::Plain(std::io::BufWriter::new(f))
        })
    }

    /// Flush everything; for gzip, compress and write the tail.
    pub(crate) fn finish(self) -> Result<()> {
        use std::io::Write;
        match self {
            ConvertOut::Plain(mut w) => w.flush()?,
            ConvertOut::Gz(w) => {
                w.finish()?;
            }
        }
        Ok(())
    }
}

impl std::io::Write for ConvertOut {
    fn write(&mut self, b: &[u8]) -> std::io::Result<usize> {
        match self {
            ConvertOut::Plain(w) => w.write(b),
            ConvertOut::Gz(w) => w.write(b),
        }
    }
    fn write_all(&mut self, b: &[u8]) -> std::io::Result<()> {
        match self {
            ConvertOut::Plain(w) => w.write_all(b),
            ConvertOut::Gz(w) => w.write_all(b),
        }
    }
    fn flush(&mut self) -> std::io::Result<()> {
        match self {
            ConvertOut::Plain(w) => w.flush(),
            ConvertOut::Gz(w) => w.flush(),
        }
    }
}

/// The per-cluster table of `rmda cluster --table`, one CSV row per cluster id.
///
/// Values are written with Rust's shortest round-trip formatting, so the file
/// carries every bit the analysis computed.
pub(crate) fn write_cluster_table(path: &std::path::Path, out: &PropertyStore) -> Result<()> {
    use std::io::Write;
    let col = |name: &str| -> Result<Vec<f64>> {
        let p = out.get(name).ok_or_else(|| {
            rmda_core::error::Error::invalid(format!("cluster table: {name} was not computed"))
        })?;
        Ok((0..p.len() * p.components)
            .map(|i| p.get_flat(i).unwrap_or(f64::NAN))
            .collect())
    };
    let size = col("ClusterSize")?;
    let mass = col("ClusterMass")?;
    let com = col("ClusterCenterOfMass")?;
    let rg = col("ClusterRadiusOfGyration")?;
    let g = col("ClusterGyrationTensor")?;
    let perc = col("ClusterPercolates")?;
    let mut w = std::io::BufWriter::new(std::fs::File::create(path)?);
    writeln!(
        w,
        "cluster,size,mass,com_x,com_y,com_z,rg,g_xx,g_yy,g_zz,g_xy,g_xz,g_yz,percolates"
    )?;
    for k in 0..size.len() {
        // Row `k` is cluster id `k` (OVITO's 0-based ids).
        write!(w, "{k},{},{}", size[k], mass[k])?;
        for v in com[3 * k..3 * k + 3]
            .iter()
            .chain(std::iter::once(&rg[k]))
            .chain(&g[6 * k..6 * k + 6])
        {
            write!(w, ",{v}")?;
        }
        writeln!(w, ",{}", perc[k])?;
    }
    w.flush()?;
    Ok(())
}

/// Attach the analysis' per-atom properties to the frame and write extended XYZ.
///
/// Without this the computed properties only ever exist as the histograms on
/// stdout, even though the engines register every one of them — including the
/// PTM orientation quaternion, which grain segmentation (design §15) needs per atom.
pub(crate) fn write_props(
    cli: &Cli,
    f: &rmda_core::data::frame::Frame,
    out: &PropertyStore,
) -> Result<()> {
    let Some(path) = cli.output.clone() else {
        return Ok(());
    };
    if out.is_empty() {
        eprintln!("note: analysis registered no per-atom properties; nothing written");
        return Ok(());
    }
    let mut file = std::io::BufWriter::new(std::fs::File::create(&path)?);
    // Analyses run on frame 0 only, so writing later frames would attach frame
    // 0's classifications to the wrong atoms.
    rmda_core::io::XyzWriter::write_frame_with_origin(&mut file, f, out, cli.xyz_origin.into())?;
    let names: Vec<&str> = out
        .iter()
        .filter(|p| p.per_particle)
        .map(|p| p.name.as_ref())
        .collect();
    eprintln!(
        "wrote {} propert{} to {}",
        names.len(),
        if names.len() == 1 { "y" } else { "ies" },
        path.display()
    );
    eprintln!("  {}", names.join(", "));
    Ok(())
}

/// The columns a read keeps: `--skip-columns` when given, else the lean set
/// for analyses and everything for the commands that pass data through.
pub(crate) fn read_options(cli: &Cli, analysis: bool) -> rmda_core::io::ReadOptions {
    use rmda_core::io::ReadOptions;
    match &cli.skip_columns {
        Some(list) => {
            ReadOptions::skipping(list.split(',').map(str::trim).filter(|s| !s.is_empty()))
        }
        None if analysis => ReadOptions::lean(),
        None => ReadOptions::all(),
    }
    .with_atom_style(cli.atom_style.clone())
    .with_keep_general(cli.keep_general)
}

/// The input files of a command: a series' files (or `--input`), else the one
/// positional file or `--input` (design §21: the input may be positional or
/// given with -i/--input). Empty when none is given.
pub(crate) fn input_files(cli: &Cli) -> Vec<PathBuf> {
    let positional: Vec<PathBuf> = match &cli.cmd {
        Cmd::Info { file }
        | Cmd::Index { file }
        | Cmd::Census { file, .. }
        | Cmd::Structure { file, .. }
        | Cmd::Ptm { file, .. }
        | Cmd::GrainBoundaries { file, .. }
        | Cmd::Segment { file, .. }
        | Cmd::Cluster { file, .. }
        | Cmd::Rdf { file, .. }
        | Cmd::Wc { file, .. }
        | Cmd::Ws { file, .. }
        | Cmd::Analyze { file, .. }
        | Cmd::Convert { file, .. } => file.iter().cloned().collect(),
        Cmd::Defects { files, .. } | Cmd::Void { files, .. } => files.clone(),
        Cmd::Run { .. } | Cmd::Merge { .. } | Cmd::Explain { .. } | Cmd::Bench { .. } => Vec::new(),
    };
    match &cli.input {
        Some(i) if positional.is_empty() => vec![i.clone()],
        Some(i) => std::iter::once(i.clone()).chain(positional).collect(),
        None => positional,
    }
}

type FatalHook = Box<dyn Fn(&str) + Send + Sync>;
static FATAL_HOOK: std::sync::OnceLock<FatalHook> = std::sync::OnceLock::new();

/// What to do before the process is stopped by `--mem-limit` (`rmda run`
/// writes its report). Set once.
pub(crate) fn set_fatal_hook(f: FatalHook) {
    let _ = FATAL_HOOK.set(f);
}

/// Watch the run's anonymous memory every 100 ms; past `limit` bytes, say so
/// (with the stages running), call the fatal hook and exit with code 4
/// (design §21). Anonymous memory is what the run allocated; the mapped
/// input is file-backed and droppable, so it is not counted.
///
/// The first check runs here, before the thread starts: a run shorter than
/// the thread's first wake-up is still checked once, whatever the scheduler.
pub(crate) fn start_mem_watchdog(limit: u64) {
    check_mem_limit(limit);
    let _ = std::thread::Builder::new()
        .name("rmda-mem-limit".into())
        .spawn(move || loop {
            std::thread::sleep(std::time::Duration::from_millis(100));
            check_mem_limit(limit);
        });
}

/// Past `limit` bytes of anonymous memory: report, call the fatal hook, exit 4.
fn check_mem_limit(limit: u64) {
    let (anon, _) = rmda_core::prof::rss_anon_file();
    if anon > limit {
        let stages = rmda_core::prof::active_stages();
        let why = format!(
            "memory limit: {} allocated, over --mem-limit {}{}",
            rmda_core::prof::fmt_bytes(anon),
            rmda_core::prof::fmt_bytes(limit),
            if stages.is_empty() {
                String::new()
            } else {
                format!(", in {}", stages.join(", "))
            }
        );
        eprintln!("error: {why}");
        if let Some(h) = FATAL_HOOK.get() {
            h(&why);
        }
        std::process::exit(4);
    }
}
