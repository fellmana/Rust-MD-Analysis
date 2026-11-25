//! Frame index (design §8.3): `FILE.rmdaidx` records where each frame
//! starts, so `--frame k` seeks instead of scanning `k` frames.
//!
//! The index is plain text:
//!
//! ```text
//! rmdaidx 1 <format> <file bytes> <file mtime, ns since epoch>
//! <byte offset> <line number before the frame> <timestep> <natoms>
//! ...
//! ```
//!
//! It is used only while the file's size and modification time still match
//! the ones recorded, so a rewritten trajectory never seeks by a stale index.
//!
//! Gzipped input cannot seek to a frame. For a `.gz` file, `FILE.gz.rmdaidx`
//! is instead a binary gzip index ([`write_gz`], `io::gzpar`): access points
//! from which the file inflates in parallel, under the same size and mtime
//! rule.

use crate::error::{Error, Result};
use crate::io::engine::Cursor;
use crate::io::format::{guess_format, is_gzip, Format};
use std::path::{Path, PathBuf};

/// Where one frame starts.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct IndexEntry {
    pub offset: usize,
    /// Lines before the frame (text formats; 0 for binary).
    pub line: usize,
    pub timestep: i64,
    pub natoms: usize,
}

/// The index path of `path`: `traj.dump` → `traj.dump.rmdaidx`.
pub fn index_path(path: &Path) -> PathBuf {
    let mut s = path.as_os_str().to_os_string();
    s.push(".rmdaidx");
    PathBuf::from(s)
}

fn fmt_name(f: Format) -> &'static str {
    match f {
        Format::Xyz => "xyz",
        Format::LammpsDump => "dump",
        Format::LammpsData => "data",
        Format::LammpsDumpBinary => "bin",
    }
}

/// `(bytes, mtime ns)` of a file.
pub(crate) fn stamp(path: &Path) -> Result<(u64, u128)> {
    let m = std::fs::metadata(path)
        .map_err(|e| Error::Io(format!("cannot stat {}: {e}", path.display())))?;
    let t = m
        .modified()
        .ok()
        .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
        .map_or(0, |d| d.as_nanos());
    Ok((m.len(), t))
}

/// Scan `path` once and list its frames, parsing headers only.
pub fn build(path: &Path) -> Result<Vec<IndexEntry>> {
    if is_gzip(path) {
        return Err(Error::Unsupported(format!(
            "{}: gzipped input cannot be indexed (it cannot seek)",
            path.display()
        )));
    }
    let fmt = guess_format(path)?;
    let cur = Cursor::from_file(path)?;
    scan(&cur, fmt)
}

/// List the frames of a cursor positioned at the start of its input.
pub(crate) fn scan(cur: &Cursor, fmt: Format) -> Result<Vec<IndexEntry>> {
    let mut out = Vec::new();
    loop {
        // Leading blank lines belong to no frame: record where the frame
        // itself starts.
        while matches!(fmt, Format::LammpsDump | Format::Xyz)
            && cur
                .peek_line()
                .map_or(false, |l| l.iter().all(|b| b.is_ascii_whitespace()))
            && !cur.eof()
        {
            cur.next_line();
        }
        let (offset, line) = (cur.pos(), cur.line_no());
        let got = match fmt {
            Format::LammpsDump => crate::io::lammps::skip_next_dump(cur)?,
            Format::LammpsDumpBinary => crate::io::binary::scan_next(cur)?,
            Format::Xyz => {
                let n = cur
                    .peek_line()
                    .and_then(crate::io::engine::fast_i64)
                    .unwrap_or(0)
                    .max(0) as usize;
                crate::io::xyz::skip_next(cur)?.then_some((0, n))
            }
            Format::LammpsData => {
                if out.is_empty() {
                    let f = crate::io::lammps::parse_data_frame(cur)?;
                    Some((0, f.natoms()))
                } else {
                    None
                }
            }
        };
        match got {
            Some((timestep, natoms)) => out.push(IndexEntry {
                offset,
                line,
                timestep,
                natoms,
            }),
            None => return Ok(out),
        }
    }
}

/// Build the gzip index of a `.gz` file and write it next to the file.
pub fn write_gz(path: &Path) -> Result<crate::io::gzpar::GzIndex> {
    let stamp0 = stamp(path)?;
    let idx = crate::io::gzpar::index_file(path)?;
    let ip = index_path(path);
    std::fs::write(&ip, idx.to_bytes(stamp0))
        .map_err(|e| Error::Io(format!("cannot write {}: {e}", ip.display())))?;
    Ok(idx)
}

/// The gzip index of a `.gz` file, when one exists and still describes it.
pub fn load_gz(path: &Path) -> Option<crate::io::gzpar::GzIndex> {
    let b = std::fs::read(index_path(path)).ok()?;
    crate::io::gzpar::GzIndex::from_bytes(&b, stamp(path).ok()?)
}

/// Build the index of `path` and write it next to the file.
pub fn write(path: &Path) -> Result<Vec<IndexEntry>> {
    let entries = build(path)?;
    let (len, mtime) = stamp(path)?;
    let mut text = format!(
        "rmdaidx 1 {} {len} {mtime}\n",
        fmt_name(guess_format(path)?)
    );
    for e in &entries {
        text.push_str(&format!(
            "{} {} {} {}\n",
            e.offset, e.line, e.timestep, e.natoms
        ));
    }
    let ip = index_path(path);
    std::fs::write(&ip, text)
        .map_err(|e| Error::Io(format!("cannot write {}: {e}", ip.display())))?;
    Ok(entries)
}

/// The index of `path`, when one exists and still describes the file.
pub fn load(path: &Path) -> Option<Vec<IndexEntry>> {
    if is_gzip(path) {
        return None;
    }
    let text = std::fs::read_to_string(index_path(path)).ok()?;
    let mut lines = text.lines();
    let head: Vec<&str> = lines.next()?.split_whitespace().collect();
    let (len, mtime) = stamp(path).ok()?;
    if head.len() != 5
        || head[0] != "rmdaidx"
        || head[1] != "1"
        || head[2] != fmt_name(guess_format(path).ok()?)
        || head[3].parse::<u64>().ok()? != len
        || head[4].parse::<u128>().ok()? != mtime
    {
        return None;
    }
    lines
        .map(|l| {
            let v: Vec<&str> = l.split_whitespace().collect();
            Some(IndexEntry {
                offset: v.first()?.parse().ok()?,
                line: v.get(1)?.parse().ok()?,
                timestep: v.get(2)?.parse().ok()?,
                natoms: v.get(3)?.parse().ok()?,
            })
        })
        .collect()
}
