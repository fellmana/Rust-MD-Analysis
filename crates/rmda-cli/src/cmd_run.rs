//! `rmda run CONFIG.toml`: a whole run from a declarative config (design
//! §12.3, §20, §23.3).
//!
//! The config is a command line written down, not a second schema. Its
//! sections are:
//!
//! - `[input]`: `files`, the input files (paths or `*`/`?` patterns);
//! - `[run]`: global options (`threads`, `atom_style`, `start`, `pbc`, ...);
//! - exactly one command section, `[defects]`, `[void]`, ...: that command's
//!   options.
//!
//! A key is the option's long name with `_` or `-` (`rmsd_cutoff` is
//! `--rmsd-cutoff`), and the CLI's own parser reads the result, so every
//! default, check and option name is the CLI's: what `rmda defects --help`
//! documents is what the section takes. Each key is looked up in the CLI's
//! definition before anything runs; one it does not know is an error naming
//! it. Values: a string or number is the option's value; `true` sets a flag
//! (`false` leaves it off); an array is one value per occurrence for a
//! repeatable option (`region = ["z <= 500", "z > 500"]`) and the values of
//! one occurrence for a multi-value option (`cell = [100, 100, 100]`).
//! Relative paths resolve against the config file's directory.
//!
//! Every run writes a `RunReport` (`report.json`, in the command's `out`
//! directory when it has one): the version, the resolved command line, the
//! config, host, threads, wall time, peak memory, the per-stage profile and,
//! for a series, the per-frame summary rows. It is written when the run
//! fails, too, with the error.

use rmda_core::error::{Error, Result};
use std::ffi::OsString;
use std::path::{Path, PathBuf};

/// A config, read and resolved into a command line.
#[derive(Clone)]
pub struct RunPlan {
    /// The `rmda ...` arguments (without the program name).
    pub args: Vec<OsString>,
    /// The command (`defects`, ...).
    pub command: String,
    /// The config file and its directory (relative paths resolve there).
    pub config_path: PathBuf,
    pub base: PathBuf,
    /// The config as read, with `--set` applied.
    pub config: toml::Table,
}

/// Read `path`, apply `--set` overrides and turn it into `rmda` arguments,
/// checked against the CLI definition `cli`.
pub fn plan(path: &Path, sets: &[String], cli: &clap::Command) -> Result<RunPlan> {
    let text = std::fs::read_to_string(path)
        .map_err(|e| Error::usage(format!("rmda run: cannot read {}: {e}", path.display())))?;
    let mut config: toml::Table = toml::from_str(&text)
        .map_err(|e| Error::usage(format!("rmda run: {}: {e}", path.display())))?;
    for s in sets {
        apply_set(&mut config, s)?;
    }
    let config_path = path
        .canonicalize()
        .map_err(|e| Error::usage(format!("rmda run: {}: {e}", path.display())))?;
    let base = config_path
        .parent()
        .map(Path::to_path_buf)
        .unwrap_or_else(|| PathBuf::from("."));

    let commands: Vec<&String> = config
        .keys()
        .filter(|k| k.as_str() != "input" && k.as_str() != "run")
        .collect();
    let command = match commands.as_slice() {
        [c] => (*c).clone(),
        [] => {
            return Err(Error::usage(
                "rmda run: the config names no command: add a section such as [defects]",
            ))
        }
        many => {
            return Err(Error::usage(format!(
                "rmda run: the config names {} commands ({}); one run runs one",
                many.len(),
                many.iter()
                    .map(|s| format!("[{s}]"))
                    .collect::<Vec<_>>()
                    .join(", ")
            )))
        }
    };
    let sub = cli
        .find_subcommand(&command)
        .filter(|_| command != "run")
        .ok_or_else(|| {
            Error::usage(format!(
                "rmda run: [{command}] is not an rmda command (see `rmda --help`)"
            ))
        })?;

    let mut args: Vec<OsString> = Vec::new();
    if let Some(run) = config.get("run") {
        let t = section(run, "run")?;
        if let Some(key) = options(&mut args, t, cli, true)? {
            let long = key.replace('_', "-");
            return Err(Error::usage(
                if cli.get_subcommands().any(|c| {
                    c.get_arguments()
                        .any(|a| a.get_long() == Some(long.as_str()))
                }) {
                    format!("rmda run: run.{key} is a command option: put it under [{command}]")
                } else {
                    format!("rmda run: unknown key run.{key}: rmda has no global option --{long}")
                },
            ));
        }
    }
    args.push(command.clone().into());
    if let Some(key) = options(&mut args, section(&config[&command], &command)?, sub, false)? {
        // Not the command's: a global option in the wrong section?
        let long = key.replace('_', "-");
        return Err(Error::usage(
            if cli
                .get_arguments()
                .any(|a| a.is_global_set() && a.get_long() == Some(long.as_str()))
            {
                format!("rmda run: {command}.{key} is a global option: put it under [run]")
            } else {
                format!(
                    "rmda run: unknown key {command}.{key}: rmda {command} has no option --{long}"
                )
            },
        ));
    }

    // Positional inputs last, after `--`, so no file name reads as an option.
    let mut files: Vec<PathBuf> = Vec::new();
    if let Some(input) = config.get("input") {
        let t = section(input, "input")?;
        for (k, v) in t {
            if k != "files" {
                return Err(Error::usage(format!(
                    "rmda run: unknown key input.{k} ([input] takes `files`)"
                )));
            }
            let pats: Vec<String> = match v {
                toml::Value::String(s) => vec![s.clone()],
                toml::Value::Array(a) => a
                    .iter()
                    .map(|x| {
                        x.as_str().map(str::to_owned).ok_or_else(|| {
                            Error::usage("rmda run: input.files holds file names (strings)")
                        })
                    })
                    .collect::<Result<_>>()?,
                _ => {
                    return Err(Error::usage(
                        "rmda run: input.files is a file name or a list of them",
                    ))
                }
            };
            for p in pats {
                files.extend(expand(&base, &p)?);
            }
        }
    }
    if !files.is_empty() {
        args.push("--".into());
        args.extend(files.into_iter().map(OsString::from));
    }
    Ok(RunPlan {
        args,
        command,
        config_path,
        base,
        config,
    })
}

fn section<'a>(v: &'a toml::Value, name: &str) -> Result<&'a toml::Table> {
    v.as_table()
        .ok_or_else(|| Error::usage(format!("rmda run: `{name}` must be a section ([{name}])")))
}

/// `--set section.key=value`: the value is read as TOML (`0.15`, `true`,
/// `["a", "b"]`), and as a plain string when it does not parse.
fn apply_set(config: &mut toml::Table, s: &str) -> Result<()> {
    let (path, value) = s
        .split_once('=')
        .ok_or_else(|| Error::usage(format!("rmda run: --set {s}: expected section.key=value")))?;
    let (sec, key) = path.trim().split_once('.').ok_or_else(|| {
        Error::usage(format!(
            "rmda run: --set {s}: name the section, e.g. defects.reference=local"
        ))
    })?;
    let value = value.trim();
    let parsed: toml::Value = toml::from_str::<toml::Table>(&format!("v = {value}"))
        .ok()
        .and_then(|mut t| t.remove("v"))
        .unwrap_or_else(|| toml::Value::String(value.to_owned()));
    let t = config
        .entry(sec.trim().to_owned())
        .or_insert_with(|| toml::Value::Table(toml::Table::new()));
    let t = t
        .as_table_mut()
        .ok_or_else(|| Error::usage(format!("rmda run: --set {s}: `{sec}` is not a section")))?;
    t.insert(key.trim().to_owned(), parsed);
    Ok(())
}

/// Append the options of section `t`, checked against `cmd`'s arguments:
/// the global ones (`globals`, the `[run]` section) or the command's own.
/// Returns the first key `cmd` has no such option for, for the caller to
/// report.
fn options(
    out: &mut Vec<OsString>,
    t: &toml::Table,
    cmd: &clap::Command,
    globals: bool,
) -> Result<Option<String>> {
    let name = if globals { "run" } else { cmd.get_name() };
    for (key, v) in t {
        let long = key.replace('_', "-");
        let Some(arg) = cmd
            .get_arguments()
            .find(|a| a.get_long() == Some(long.as_str()))
            .filter(|a| a.is_global_set() == globals)
        else {
            return Ok(Some(key.clone()));
        };
        let flag = OsString::from(format!("--{long}"));
        let takes_values = arg.get_action().takes_values();
        let multi = arg.get_num_args().map_or(false, |r| r.max_values() > 1);
        let scalar = |x: &toml::Value| -> Result<String> {
            match x {
                toml::Value::String(s) => Ok(s.clone()),
                toml::Value::Integer(i) => Ok(i.to_string()),
                toml::Value::Float(f) => Ok(f.to_string()),
                toml::Value::Boolean(b) => Ok(b.to_string()),
                _ => Err(Error::usage(format!(
                    "rmda run: {name}.{key}: expected a string, number or boolean"
                ))),
            }
        };
        match v {
            toml::Value::Boolean(b) if !takes_values => {
                if *b {
                    out.push(flag);
                }
            }
            _ if !takes_values => {
                return Err(Error::usage(format!(
                    "rmda run: {name}.{key} is a switch: write {key} = true"
                )))
            }
            toml::Value::Array(items) if multi => {
                out.push(flag);
                for x in items {
                    out.push(scalar(x)?.into());
                }
            }
            toml::Value::Array(items) => {
                for x in items {
                    out.push(format!("--{long}={}", scalar(x)?).into());
                }
            }
            x => out.push(format!("--{long}={}", scalar(x)?).into()),
        }
    }
    Ok(None)
}

/// `pattern` relative to `base`, with `*` and `?` in its file name (not in
/// its directories) expanded, sorted; a plain path is kept even when it does
/// not exist yet, so the reader reports it.
fn expand(base: &Path, pattern: &str) -> Result<Vec<PathBuf>> {
    let p = base.join(pattern);
    let name = p
        .file_name()
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_default();
    if !name.contains(['*', '?']) {
        return Ok(vec![p]);
    }
    let dir = p.parent().unwrap_or(base);
    if dir.to_string_lossy().contains(['*', '?']) {
        return Err(Error::usage(format!(
            "rmda run: {pattern}: patterns are allowed in file names, not directories"
        )));
    }
    let mut found: Vec<PathBuf> = std::fs::read_dir(dir)
        .map_err(|e| Error::usage(format!("rmda run: {}: {e}", dir.display())))?
        .filter_map(|e| e.ok())
        .filter(|e| e.file_type().map_or(false, |t| !t.is_dir()))
        .filter(|e| wildcard(&name, &e.file_name().to_string_lossy()))
        .map(|e| e.path())
        .collect();
    if found.is_empty() {
        return Err(Error::usage(format!(
            "rmda run: input file pattern {pattern} matches nothing in {}",
            dir.display()
        )));
    }
    found.sort();
    Ok(found)
}

/// Shell-style match of `*` (any run) and `?` (one character).
fn wildcard(pat: &str, s: &str) -> bool {
    let (p, t): (Vec<char>, Vec<char>) = (pat.chars().collect(), s.chars().collect());
    let (mut i, mut j) = (0usize, 0usize);
    let (mut star, mut mark) = (None, 0usize);
    while j < t.len() {
        if i < p.len() && (p[i] == '?' || p[i] == t[j]) {
            i += 1;
            j += 1;
        } else if i < p.len() && p[i] == '*' {
            star = Some(i);
            mark = j;
            i += 1;
        } else if let Some(si) = star {
            i = si + 1;
            mark += 1;
            j = mark;
        } else {
            return false;
        }
    }
    while i < p.len() && p[i] == '*' {
        i += 1;
    }
    i == p.len()
}

/// Shell-quoted, for showing the resolved command line.
pub fn shell_line(args: &[OsString]) -> String {
    std::iter::once("rmda".to_owned())
        .chain(args.iter().map(|a| {
            let s = a.to_string_lossy();
            if !s.is_empty()
                && s.chars()
                    .all(|c| c.is_ascii_alphanumeric() || "-_=./,:+@%".contains(c))
            {
                s.into_owned()
            } else {
                format!("'{}'", s.replace('\'', "'\\''"))
            }
        }))
        .collect::<Vec<_>>()
        .join(" ")
}

/// What a finished (or failed) run reports.
pub struct Outcome<'a> {
    pub plan: &'a RunPlan,
    pub threads: usize,
    pub started: std::time::SystemTime,
    pub wall: f64,
    /// CPU seconds the run used (all threads), when known.
    pub cpu: Option<f64>,
    pub error: Option<String>,
    /// The command's output directory, when it has one.
    pub out_dir: Option<&'a Path>,
}

/// The `RunReport` as JSON (design §23.3).
pub fn report_json(o: &Outcome) -> serde_json::Value {
    use serde_json::{json, Value};
    let stages: Vec<Value> = rmda_core::prof::snapshot()
        .into_iter()
        .map(|s| {
            json!({
                "name": s.name,
                // Summed over every thread that ran the stage (and over the
                // calls), so it can exceed the run's wall time.
                "thread_s": s.wall,
                "calls": s.calls,
                "peak_anon_bytes": s.peak_anon,
            })
        })
        .collect();
    let frames = o
        .out_dir
        .map(|d| d.join("summary.csv"))
        .filter(|p| p.exists())
        .and_then(|p| csv_rows(&p).ok());
    let host = std::fs::read_to_string("/proc/sys/kernel/hostname")
        .map(|s| s.trim().to_owned())
        .unwrap_or_default();
    let started = o
        .started
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs_f64())
        .unwrap_or(0.0);
    json!({
        "report_version": 1,
        "rmda_version": env!("CARGO_PKG_VERSION"),
        "analysis": o.plan.command,
        "status": if o.error.is_some() { "error" } else { "ok" },
        "error": o.error,
        "config_file": o.plan.config_path,
        "config": toml_to_json(&toml::Value::Table(o.plan.config.clone())),
        "command": shell_line(&o.plan.args),
        "host": host,
        "threads": o.threads,
        "started_unix": started,
        "wall_s": o.wall,
        "cpu_s": o.cpu,
        // CPU over wall over threads: 1 when every thread was busy throughout.
        "parallel_efficiency": o.cpu.map(|c| c / (o.wall * o.threads as f64).max(1e-9)),
        "peak_rss_bytes": rmda_core::prof::peak_rss_bytes(),
        "stages": stages,
        "frames": frames,
    })
}

fn toml_to_json(v: &toml::Value) -> serde_json::Value {
    use serde_json::Value;
    match v {
        toml::Value::String(s) => Value::String(s.clone()),
        toml::Value::Integer(i) => Value::from(*i),
        toml::Value::Float(f) => {
            serde_json::Number::from_f64(*f).map_or(Value::Null, Value::Number)
        }
        toml::Value::Boolean(b) => Value::Bool(*b),
        toml::Value::Datetime(d) => Value::String(d.to_string()),
        toml::Value::Array(a) => Value::Array(a.iter().map(toml_to_json).collect()),
        toml::Value::Table(t) => Value::Object(
            t.iter()
                .map(|(k, v)| (k.clone(), toml_to_json(v)))
                .collect(),
        ),
    }
}

/// A CSV file (as the series writes them: no quoted commas) as objects, with
/// numbers as numbers.
fn csv_rows(path: &Path) -> std::io::Result<serde_json::Value> {
    use serde_json::Value;
    let text = std::fs::read_to_string(path)?;
    let mut lines = text.lines();
    let header: Vec<&str> = lines.next().unwrap_or("").split(',').collect();
    let rows = lines
        .filter(|l| !l.is_empty())
        .map(|l| {
            Value::Object(
                header
                    .iter()
                    .zip(l.split(','))
                    .map(|(k, v)| {
                        let val = if let Ok(i) = v.parse::<i64>() {
                            Value::from(i)
                        } else if let Some(n) =
                            v.parse::<f64>().ok().and_then(serde_json::Number::from_f64)
                        {
                            Value::Number(n)
                        } else {
                            Value::String(v.to_owned())
                        };
                        ((*k).to_owned(), val)
                    })
                    .collect(),
            )
        })
        .collect();
    Ok(Value::Array(rows))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wildcards_match_like_a_shell() {
        assert!(wildcard("data.lastframe-*", "data.lastframe-999"));
        assert!(wildcard("dump.*.gz", "dump.100.gz"));
        assert!(!wildcard("dump.*.gz", "dump.100"));
        assert!(wildcard("a?c", "abc"));
        assert!(!wildcard("a?c", "ac"));
        assert!(wildcard("*", ""));
        assert!(wildcard("*x*y", "axbxy"));
    }

    #[test]
    fn set_reads_toml_values_and_falls_back_to_strings() {
        let mut t = toml::Table::new();
        apply_set(&mut t, "defects.rmsd_cutoff=0.15").unwrap();
        apply_set(&mut t, "defects.reference=local").unwrap();
        apply_set(&mut t, "defects.region=[\"z <= 5\", \"z > 5\"]").unwrap();
        let d = t["defects"].as_table().unwrap();
        assert_eq!(d["rmsd_cutoff"].as_float(), Some(0.15));
        assert_eq!(d["reference"].as_str(), Some("local"));
        assert_eq!(d["region"].as_array().unwrap().len(), 2);
        assert!(apply_set(&mut t, "noequals").is_err());
        assert!(apply_set(&mut t, "nosection=1").is_err());
    }
}
