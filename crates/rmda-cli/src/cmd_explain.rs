//! `rmda explain CONFIG`: what `rmda run CONFIG` would do, without running it
//! (design §21): the resolved command, the input files with their atom
//! counts (read from each file's first megabyte, so a multi-GB `.gz` costs
//! nothing), the frames a `--shard` would take, and a memory estimate per
//! frame from measured runs. It is for sizing a job, not a prediction.

use crate::prelude::*;
use std::io::Read;

/// Measured peak anonymous memory per atom of a defect census (design
/// §16.8, §24): 193 B/atom for thin film frame 999 under `auto` (1.11 GB,
/// 5.76M atoms), 245 B/atom under `local` (1.41 GB); the 128M frame peaks
/// near 170 B/atom. Small frames add a few hundred MB of fixed cost.
const DEFECTS_BYTES_PER_ATOM: (f64, f64) = (190.0, 250.0);
const FIXED_BYTES: f64 = 0.3e9;

/// The atom count a frame file declares, from its first megabyte (inflated
/// when gzipped): a dump's NUMBER OF ATOMS, a data file's `N atoms`, an xyz
/// file's first line. `None` for a binary dump or an unreadable head.
pub(crate) fn peek_natoms(path: &Path) -> Option<u64> {
    let fmt = rmda_core::io::format::guess_format(path).ok()?;
    let file = std::fs::File::open(path).ok()?;
    let mut head = Vec::new();
    if rmda_core::io::format::is_gzip(path) {
        flate2::read::MultiGzDecoder::new(file)
            .take(1 << 20)
            .read_to_end(&mut head)
            .ok()?;
    } else {
        file.take(1 << 20).read_to_end(&mut head).ok()?;
    }
    let text = String::from_utf8_lossy(&head);
    let mut lines = text.lines();
    match fmt {
        Format::LammpsDump => {
            while let Some(l) = lines.next() {
                if l.trim_start().starts_with("ITEM: NUMBER OF ATOMS") {
                    return lines.next()?.trim().parse().ok();
                }
            }
            None
        }
        Format::LammpsData => lines.find_map(|l| {
            let mut w = l.split_whitespace();
            match (w.next(), w.next(), w.next()) {
                (Some(n), Some("atoms"), None) => n.parse().ok(),
                _ => None,
            }
        }),
        Format::Xyz => lines.find(|l| !l.trim().is_empty())?.trim().parse().ok(),
        Format::LammpsDumpBinary => None,
    }
}

/// `rmda explain CONFIG` (dispatched in main: it runs nothing).
pub(crate) fn command(config: &Path, sets: &[String]) -> Result<()> {
    let plan = crate::cmd_run::plan(config, sets, &Cli::command())?;
    let line = crate::cmd_run::shell_line(&plan.args);
    let argv = std::iter::once(std::ffi::OsString::from("rmda")).chain(plan.args.iter().cloned());
    let inner = Cli::try_parse_from(argv).map_err(|e| {
        rmda_core::error::Error::usage(format!(
            "rmda explain: the config does not make a valid command:\n  {line}\n{e}"
        ))
    })?;
    // Relative paths in the config are relative to the config file.
    std::env::set_current_dir(&plan.base).map_err(|e| {
        rmda_core::error::Error::usage(format!("rmda explain: {}: {e}", plan.base.display()))
    })?;
    println!("config:  {}", plan.config_path.display());
    println!("command: {line}");

    let files = input_files(&inner);
    let shard = match &inner.cmd {
        Cmd::Defects { shard, .. } => *shard,
        Cmd::Void { shard, .. } => *shard,
        _ => None,
    };
    let ordered: Vec<(i64, PathBuf)> = if files.len() > 1 {
        cmd_defects::series_items(&files)?
    } else {
        files.iter().cloned().map(|p| (0, p)).collect()
    };
    let (lo, hi) = match shard {
        Some((i, n)) => cmd_defects::shard_block(ordered.len(), i, n),
        None => (0, ordered.len()),
    };
    println!(
        "inputs:  {} file(s){}",
        ordered.len(),
        match shard {
            Some((i, n)) => format!(", shard {i} of {n} takes {} of them", hi - lo),
            None => String::new(),
        }
    );
    let mut largest = 0u64;
    let mut missing = 0usize;
    for (k, (idx, p)) in ordered.iter().enumerate() {
        let mine = k >= lo && k < hi;
        let size = std::fs::metadata(p).map(|m| m.len()).ok();
        let atoms = peek_natoms(p);
        if size.is_none() {
            missing += 1;
        }
        if mine {
            largest = largest.max(atoms.unwrap_or(0));
        }
        println!(
            "  {} {:>8} {:>12} {:>14}  {}",
            if mine { "*" } else { " " },
            idx,
            size.map_or("missing".to_owned(), |b| format!(
                "{:.1} MB",
                b as f64 / 1e6
            )),
            atoms.map_or("? atoms".to_owned(), |n| format!("{n} atoms")),
            p.display()
        );
    }
    if missing > 0 {
        println!("  {missing} input file(s) missing: the run would exit 3 at the first");
    }
    if matches!(inner.cmd, Cmd::Defects { .. } | Cmd::Void { .. }) && largest > 0 {
        let (lo_b, hi_b) = DEFECTS_BYTES_PER_ATOM;
        let est = |b: f64| (largest as f64 * b + FIXED_BYTES) / 1e9;
        println!(
            "memory:  ~{:.1}-{:.1} GB peak per frame for {largest} atoms (measured {:.0}-{:.0} \
             B/atom on 5.8M- and 128M-atom frames, plus ~{:.1} GB fixed); one frame is held \
             at a time{}",
            est(lo_b),
            est(hi_b),
            lo_b,
            hi_b,
            FIXED_BYTES / 1e9,
            match &inner.cmd {
                Cmd::Defects {
                    ws_reference: Some(_),
                    ..
                } => "; --ws-reference holds the reference frame as well",
                _ => "",
            }
        );
        println!(
            "time:    no prediction; measured: thin film 5.76M atoms, auto, ~50 s per frame on \
             12 threads; a 128M-atom polycrystal on an 80-thread node: ~49 min with the default \
             `--local-merge union` (its merge is mostly in order), ~7.5 min with `partial` \
             (level-parallel; totals within 0.2 % on that frame)"
        );
    }
    if let Some(limit) = inner.mem_limit {
        println!(
            "limit:   --mem-limit {:.1} GB (exit 4 when exceeded)",
            limit as f64 / 1e9
        );
    }
    Ok(())
}
