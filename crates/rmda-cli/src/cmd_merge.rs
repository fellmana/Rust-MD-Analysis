//! `rmda merge OUT`: join the shards of a sharded series (`rmda defects --out
//! OUT --shard I/N`, design §20) into the tables an unsharded run writes.
//!
//! Shards are contiguous blocks of the ordered frames, so their rows are
//! concatenated in shard order. A tracked series (`--segment`) is tracked
//! here: each shard recorded, per frame, its grains' sizes and their atom
//! overlaps with the previous frame's grains (`track.csv`, the previous frame
//! of a shard's first being the shard before's last), and wrote grain labels
//! where stable ids go. The merge replays the tracker's decisions over all
//! frames in order (`Tracker::step`, the code an unsharded series runs), so
//! the stable ids and `grain_events.csv` are the sequential run's, and
//! rewrites the stable-id columns of the tables and the `grain` column of
//! the defect files.

use crate::cmd_defects::series_prefix;
use rmda_core::analysis::grain_track::{Event, Tracker};
use rmda_core::error::{Error, Result};
use std::collections::{BTreeMap, HashMap};
use std::path::{Path, PathBuf};

/// The tables to merge: every `.csv` shard 0 holds (a defects or a void
/// series, whichever ran) except the tracker's input, `summary.csv` first.
fn tables(dir: &Path) -> Result<Vec<String>> {
    let mut v: Vec<String> = std::fs::read_dir(dir)?
        .flatten()
        .map(|e| e.file_name().to_string_lossy().into_owned())
        .filter(|n| n.ends_with(".csv") && n != "track.csv")
        .collect();
    v.sort_by_key(|n| (n != "summary.csv", n.clone()));
    Ok(v)
}

struct Shard {
    dir: PathBuf,
    /// The block's frames: (file index, path).
    frames: Vec<(i64, PathBuf)>,
}

/// Merge the shards under `out` into `out`'s own tables.
pub fn merge(out: &Path, overwrite: bool) -> Result<()> {
    let shards = find_shards(out)?;
    let n = shards.len();

    // One run: same settings, every block finished.
    let sig = std::fs::read_to_string(shards[0].dir.join("run.txt"))?;
    for s in &shards[1..] {
        if std::fs::read_to_string(s.dir.join("run.txt"))? != sig {
            return Err(Error::invalid(format!(
                "{} was run with other settings than {}",
                s.dir.display(),
                shards[0].dir.display()
            )));
        }
    }
    for s in &shards {
        let done = summary_indices(&s.dir)?;
        let missing: Vec<i64> = s
            .frames
            .iter()
            .map(|f| f.0)
            .filter(|k| !done.contains(k))
            .collect();
        if !missing.is_empty() {
            return Err(Error::invalid(format!(
                "{} is not finished: frames {missing:?} have no summary row (rerun that shard; \
                 it resumes)",
                s.dir.display()
            )));
        }
    }
    if out.join("summary.csv").exists() && !overwrite {
        return Err(Error::invalid(format!(
            "{} already holds merged (or unsharded) tables; use --overwrite",
            out.display()
        )));
    }

    // Tracking: replay the decisions over every frame in order.
    let tracked = shards[0].dir.join("track.csv").exists();
    let mut stable: HashMap<i64, Vec<u32>> = HashMap::new();
    let mut events: Vec<(i64, Vec<Event>)> = Vec::new();
    if tracked {
        let mut tracker = Tracker::new();
        let mut prev: Option<Vec<u32>> = None;
        for s in &shards {
            let rows = track_rows(&s.dir)?;
            for (k, _) in &s.frames {
                let t = rows.get(k).ok_or_else(|| {
                    Error::invalid(format!(
                        "{}: track.csv has no rows for frame {k}",
                        s.dir.display()
                    ))
                })?;
                // Overlaps by previous label, as the tracker's previous ids.
                let mut overlap: HashMap<(u32, u32), usize> = HashMap::new();
                if let Some(p) = &prev {
                    for &(g, pg, c) in &t.overlaps {
                        let ps = *p.get(pg as usize).ok_or_else(|| {
                            Error::invalid(format!(
                                "track.csv frame {k}: previous label {pg} does not exist"
                            ))
                        })?;
                        overlap.insert((g, ps), c);
                    }
                }
                let (st, ev) = tracker.step(&t.sizes, &overlap);
                stable.insert(*k, st.clone());
                events.push((*k, ev));
                prev = Some(st);
            }
        }
    }
    let to_stable = |k: i64, label: &str| -> Result<String> {
        if !tracked {
            return Ok(label.to_owned());
        }
        let g: i64 = label
            .parse()
            .map_err(|_| Error::invalid(format!("frame {k}: grain `{label}` is not a label")))?;
        if g < 0 {
            return Ok(label.to_owned());
        }
        stable
            .get(&k)
            .and_then(|s| s.get(g as usize))
            .map(|s| s.to_string())
            .ok_or_else(|| Error::invalid(format!("frame {k}: no grain {g} in track.csv")))
    };

    // Tables: concatenated in shard order; tracked, the stable-id columns
    // from the replay.
    for t in tables(&shards[0].dir)? {
        let t = t.as_str();
        let mut header: Option<String> = None;
        let mut body = String::new();
        for s in &shards {
            let Ok(text) = std::fs::read_to_string(s.dir.join(t)) else {
                continue;
            };
            let mut lines = text.lines();
            let h = lines.next().unwrap_or("").to_owned();
            match &header {
                None => header = Some(h.clone()),
                Some(x) if *x != h => {
                    return Err(Error::invalid(format!(
                        "{}: other columns than shard 0's",
                        s.dir.join(t).display()
                    )))
                }
                _ => {}
            }
            let cols: Vec<&str> = h.split(',').collect();
            // (column to write, column holding the label).
            let remap: Option<(usize, usize)> = match t {
                "grains.csv" | "antisites.csv" if tracked => {
                    Some((col(&cols, "stable_grain", t)?, col(&cols, "grain", t)?))
                }
                "voids.csv" if tracked => {
                    Some((col(&cols, "stable_grain", t)?, col(&cols, "grain", t)?))
                }
                "clusters.csv" if tracked => {
                    let c = col(&cols, "grain", t)?;
                    Some((c, c))
                }
                _ => None,
            };
            for l in lines {
                match remap {
                    None => body.push_str(l),
                    Some((dst, src)) => {
                        let mut f: Vec<String> = l.split(',').map(str::to_owned).collect();
                        let k: i64 = f[0].parse().map_err(|_| {
                            Error::invalid(format!("{}: bad row `{l}`", s.dir.join(t).display()))
                        })?;
                        f[dst] = to_stable(k, &f[src])?;
                        body.push_str(&f.join(","));
                    }
                }
                body.push('\n');
            }
        }
        if let Some(h) = header {
            std::fs::write(out.join(t), format!("{h}\n{body}"))?;
        }
    }

    // Grain events, as an unsharded tracked series writes them.
    if tracked {
        let timestep = timesteps(&shards)?;
        let path_of: HashMap<i64, &PathBuf> = shards
            .iter()
            .flat_map(|s| s.frames.iter().map(|(k, p)| (*k, p)))
            .collect();
        let mut text = String::from("file_index,file,timestep,event,stable,other\n");
        for (k, ev) in &events {
            let prefix = series_prefix(*k, path_of[k]);
            for e in ev {
                let (kind, st, other) = match *e {
                    Event::New { stable } => ("new", stable as i64, -1),
                    Event::Continued { stable } => ("continued", stable as i64, -1),
                    Event::Split { stable, from } => ("split", stable as i64, from as i64),
                    Event::Merged { stable, into } => ("merged", stable as i64, into as i64),
                };
                text.push_str(&format!("{prefix}{},{kind},{st},{other}\n", timestep[k]));
            }
        }
        std::fs::write(out.join("grain_events.csv"), text)?;
    }

    // Defect files (and a void series' sites): per-frame files copied,
    // appended ones concatenated in shard order; tracked, the `grain` column
    // rewritten.
    let mut written: BTreeMap<PathBuf, Vec<u8>> = BTreeMap::new();
    for (s, sub) in shards
        .iter()
        .flat_map(|s| ["defects", "sites"].map(|d| (s, d)))
    {
        let base = s.dir.join(sub);
        if !base.is_dir() {
            continue;
        }
        for file in walk(&base)? {
            let rel = Path::new(sub).join(file.strip_prefix(&base).unwrap());
            let text = std::fs::read_to_string(&file)?;
            let fixed = if tracked {
                remap_xyz(&text, &to_stable)
                    .map_err(|e| Error::invalid(format!("{}: {e}", file.display())))?
            } else {
                text
            };
            written
                .entry(rel)
                .or_default()
                .extend_from_slice(fixed.as_bytes());
        }
    }
    for (rel, bytes) in &written {
        let p = out.join(rel);
        std::fs::create_dir_all(p.parent().unwrap())?;
        std::fs::write(&p, bytes)?;
    }

    std::fs::write(out.join("run.txt"), &sig)?;
    let frames: usize = shards.iter().map(|s| s.frames.len()).sum();
    eprintln!(
        "merge: {n} shard(s), {frames} frame(s){} into {}",
        if tracked { ", grains tracked" } else { "" },
        out.display()
    );
    Ok(())
}

/// The shards under `out`: `shard-I-of-N` for every `I` of one `N`.
fn find_shards(out: &Path) -> Result<Vec<Shard>> {
    let mut found: BTreeMap<usize, (usize, PathBuf)> = BTreeMap::new();
    for e in std::fs::read_dir(out)
        .map_err(|e| Error::invalid(format!("{}: {e}", out.display())))?
        .flatten()
    {
        let name = e.file_name().to_string_lossy().into_owned();
        let Some(rest) = name.strip_prefix("shard-") else {
            continue;
        };
        let Some((i, n)) = rest.split_once("-of-") else {
            continue;
        };
        let (Ok(i), Ok(n)) = (i.parse::<usize>(), n.parse::<usize>()) else {
            continue;
        };
        found.insert(i, (n, e.path()));
    }
    let Some((_, (n, _))) = found.iter().next() else {
        return Err(Error::invalid(format!(
            "{}: no shard-I-of-N directories (run `rmda defects ... --out {} --shard I/N`)",
            out.display(),
            out.display()
        )));
    };
    let n = *n;
    if found.values().any(|(m, _)| *m != n) {
        return Err(Error::invalid(format!(
            "{}: shards of different counts (N); remove the stale ones",
            out.display()
        )));
    }
    let missing: Vec<usize> = (0..n).filter(|i| !found.contains_key(i)).collect();
    if !missing.is_empty() {
        return Err(Error::invalid(format!(
            "{}: shard(s) {missing:?} of {n} have not run",
            out.display()
        )));
    }
    let mut shards = Vec::with_capacity(n);
    let mut last: Option<i64> = None;
    for (i, (_, dir)) in found {
        let text = std::fs::read_to_string(dir.join("shard.txt"))
            .map_err(|e| Error::invalid(format!("{}: {e}", dir.join("shard.txt").display())))?;
        let mut frames = Vec::new();
        let mut lead_in = None;
        for l in text.lines() {
            let mut w = l.splitn(3, ' ');
            match (w.next(), w.next(), w.next()) {
                (Some("frame"), Some(k), Some(p)) => {
                    frames.push((k.parse().map_err(|_| bad(&dir, l))?, PathBuf::from(p)))
                }
                (Some("lead_in"), Some(k), Some(_)) => {
                    lead_in = Some(k.parse::<i64>().map_err(|_| bad(&dir, l))?)
                }
                _ => {}
            }
        }
        // Consecutive blocks: a shard's lead-in is the frame before it, the
        // previous non-empty shard's last.
        if !frames.is_empty() {
            if lead_in != last {
                return Err(Error::invalid(format!(
                    "{}: starts after frame {lead_in:?}, but the shards before it end at {last:?}; \
                     were they run over other files?",
                    dir.display()
                )));
            }
            last = frames.last().map(|f| f.0);
        }
        let _ = i;
        shards.push(Shard { dir, frames });
    }
    Ok(shards)
}

fn bad(dir: &Path, l: &str) -> Error {
    Error::invalid(format!(
        "{}: bad line `{l}`",
        dir.join("shard.txt").display()
    ))
}

fn col(cols: &[&str], name: &str, table: &str) -> Result<usize> {
    cols.iter()
        .position(|&c| c == name)
        .ok_or_else(|| Error::invalid(format!("{table}: no `{name}` column")))
}

fn summary_indices(dir: &Path) -> Result<std::collections::BTreeSet<i64>> {
    Ok(std::fs::read_to_string(dir.join("summary.csv"))
        .unwrap_or_default()
        .lines()
        .skip(1)
        .filter_map(|l| l.split(',').next()?.parse().ok())
        .collect())
}

/// Each frame's timestep, from the summary rows.
fn timesteps(shards: &[Shard]) -> Result<HashMap<i64, String>> {
    let mut m = HashMap::new();
    for s in shards {
        let text = std::fs::read_to_string(s.dir.join("summary.csv"))?;
        let mut lines = text.lines();
        let cols: Vec<&str> = lines.next().unwrap_or("").split(',').collect();
        let t = col(&cols, "timestep", "summary.csv")?;
        for l in lines {
            let f: Vec<&str> = l.split(',').collect();
            if let (Some(k), Some(ts)) = (f.first().and_then(|k| k.parse().ok()), f.get(t)) {
                m.insert(k, (*ts).to_owned());
            }
        }
    }
    Ok(m)
}

/// One frame's tracking input: size per label, (label, previous label, count).
#[derive(Default)]
struct Track {
    sizes: Vec<usize>,
    overlaps: Vec<(u32, u32, usize)>,
}

fn track_rows(dir: &Path) -> Result<HashMap<i64, Track>> {
    let text = std::fs::read_to_string(dir.join("track.csv"))?;
    let mut m: HashMap<i64, Track> = HashMap::new();
    for l in text.lines().skip(1) {
        let f: Vec<&str> = l.split(',').collect();
        let parse = |i: usize| -> Result<i64> {
            f.get(i).and_then(|x| x.parse().ok()).ok_or_else(|| {
                Error::invalid(format!(
                    "{}: bad row `{l}`",
                    dir.join("track.csv").display()
                ))
            })
        };
        let (k, label, other, count) = (parse(0)?, parse(2)?, parse(3)?, parse(4)?);
        let t = m.entry(k).or_default();
        match f.get(1).copied() {
            Some("size") => {
                let g = label as usize;
                if t.sizes.len() <= g {
                    t.sizes.resize(g + 1, 0);
                }
                t.sizes[g] = count as usize;
            }
            Some("overlap") => t
                .overlaps
                .push((label as u32, other as u32, count as usize)),
            _ => return Err(Error::invalid(format!("track.csv: bad row `{l}`"))),
        }
    }
    Ok(m)
}

/// Every file under `dir`, sorted.
fn walk(dir: &Path) -> Result<Vec<PathBuf>> {
    let mut out = Vec::new();
    let mut stack = vec![dir.to_path_buf()];
    while let Some(d) = stack.pop() {
        for e in std::fs::read_dir(&d)?.flatten() {
            let p = e.path();
            if p.is_dir() {
                stack.push(p);
            } else {
                out.push(p);
            }
        }
    }
    out.sort();
    Ok(out)
}

/// An extended-XYZ text (one or more frames, each with `index=K` in its
/// comment line) with its `grain` column mapped through `to_stable(K, _)`;
/// rows are single-space separated, as the series writes them.
fn remap_xyz(
    text: &str,
    to_stable: &dyn Fn(i64, &str) -> Result<String>,
) -> std::result::Result<String, String> {
    let mut out = String::with_capacity(text.len());
    let mut lines = text.lines();
    while let Some(count) = lines.next() {
        let n: usize = count
            .trim()
            .parse()
            .map_err(|_| format!("bad atom count `{count}`"))?;
        let comment = lines.next().ok_or("missing comment line")?;
        let k: i64 = comment
            .split_whitespace()
            .find_map(|t| t.strip_prefix("index="))
            .and_then(|v| v.parse().ok())
            .ok_or("a frame without index= in its comment line")?;
        let props = comment
            .split_whitespace()
            .find_map(|t| t.strip_prefix("Properties="))
            .ok_or("a frame without Properties=")?;
        // Column of `grain`: the sum of the counts of the fields before it.
        let spec: Vec<&str> = props.split(':').collect();
        let mut column = None;
        let mut at = 0usize;
        for f in spec.chunks(3) {
            let width: usize = f
                .get(2)
                .and_then(|w| w.parse().ok())
                .ok_or("bad Properties")?;
            if f[0] == "grain" {
                column = Some(at);
                break;
            }
            at += width;
        }
        out.push_str(count);
        out.push('\n');
        out.push_str(comment);
        out.push('\n');
        for _ in 0..n {
            let row = lines.next().ok_or("fewer rows than the atom count")?;
            match column {
                Some(c) => {
                    let mut t: Vec<String> = row.split(' ').map(str::to_owned).collect();
                    let g = t.get(c).ok_or("a row without a grain column")?.clone();
                    t[c] = to_stable(k, &g).map_err(|e| e.to_string())?;
                    out.push_str(&t.join(" "));
                }
                None => out.push_str(row),
            }
            out.push('\n');
        }
    }
    Ok(out)
}
