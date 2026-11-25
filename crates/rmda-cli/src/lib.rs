//! `rmda` command-line tool: argument parsing (`cli`), one module per
//! subcommand (`cmd_*::command`), shared helpers (`util`).
//!
//! A library as well as the `rmda` binary, so the Python package can carry
//! the command line: its `rmda` console script calls [`main_with_args`]
//! through the extension module, and `pip install rmda` gives both.

mod cli;
mod cmd_analyze;
mod cmd_bench;
mod cmd_census;
mod cmd_cluster;
mod cmd_convert;
mod cmd_defects;
mod cmd_explain;
mod cmd_grain_boundaries;
mod cmd_index;
mod cmd_info;
mod cmd_merge;
mod cmd_ptm;
mod cmd_rdf;
mod cmd_run;
mod cmd_segment;
mod cmd_structure;
mod cmd_void;
mod cmd_void_b;
mod cmd_wc;
mod cmd_ws;
mod prelude;
mod util;

use crate::prelude::*;

/// Run the command line on `args` (the program name first, as
/// `std::env::args_os` gives it) and exit with its status: 0 on success, the
/// error's exit code otherwise (design §21).
pub fn main_with_args<I, T>(args: I) -> !
where
    I: IntoIterator<Item = T>,
    T: Into<std::ffi::OsString> + Clone,
{
    let cli = Cli::parse_from(args);
    let result = match &cli.cmd {
        Cmd::Run {
            config,
            sets,
            dry_run,
            shard,
        } => run_config(&cli, config, sets, *dry_run, *shard),
        Cmd::Merge { out, overwrite } => cmd_merge::merge(out, *overwrite),
        Cmd::Explain { config, sets } => cmd_explain::command(config, sets),
        Cmd::Bench { .. } => {
            let threads = setup(&cli, true);
            cmd_bench::command(&cli, Par::new(threads))
        }
        _ => {
            let threads = setup(&cli, false);
            run(&cli, threads)
        }
    };
    if let Err(e) = result {
        eprintln!("error: {e}");
        // design §21: 3 = the input is unusable (unreadable, unparsable or
        // corrupt), 2 = an invalid request, 1 = anything else; clap's own
        // argument errors are 2 as well. 5 (no grain measured) is set by
        // `rmda defects` itself, with its results written.
        std::process::exit(e.exit_code());
    }
    std::process::exit(0);
}
/// Profiling and the thread count for a run of `cli`; `profile` switches
/// the stage profiler on regardless (`rmda run` records it in its report).
fn setup(cli: &Cli, profile: bool) -> usize {
    // Must happen before any stage runs: a timer reads the flag when created, and
    // a timer created while disabled records nothing at all.
    // --mem-limit names the stages running when it stops a run, which the
    // profiler tracks (its timers cost nothing measurable).
    if profile || cli.profile || cli.memreport || cli.mem_limit.is_some() {
        rmda_core::prof::Profile::set_enabled(true);
    }
    let threads = if cli.threads == 0 {
        rmda_core::exec::default_threads()
    } else {
        cli.threads
    };
    // Publish the count so `configured_threads()` (and hence every
    // `Par::default()` context) sees `--threads`, not just the contexts this
    // run builds explicitly.
    rmda_core::exec::set_threads(threads);
    // The readers parse rows on rayon's global pool; `--threads` bounds it too.
    rmda_core::exec::init_global_pool(threads);
    if let Some(limit) = cli.mem_limit {
        start_mem_watchdog(limit);
    }
    threads
}
/// `rmda run CONFIG`: resolve the config into a command line, run it, and
/// write the run report (also when the run fails).
fn run_config(
    outer: &Cli,
    config: &std::path::Path,
    sets: &[String],
    dry_run: bool,
    shard: Option<(usize, usize)>,
) -> Result<()> {
    // Options given to `rmda run` itself win over the config's.
    let mut sets = sets.to_vec();
    if outer.threads != 0 {
        sets.push(format!("run.threads={}", outer.threads));
    }
    if outer.profile {
        sets.push("run.profile=true".to_owned());
    }
    if let Some(m) = outer.mem_limit {
        sets.push(format!("run.mem_limit=\"{m}\""));
    }
    let mut plan = cmd_run::plan(config, &sets, &Cli::command())?;
    if let Some((i, n)) = shard {
        sets.push(format!("{}.shard=\"{i}/{n}\"", plan.command));
        plan = cmd_run::plan(config, &sets, &Cli::command())?;
    }
    let line = cmd_run::shell_line(&plan.args);
    let argv = std::iter::once(std::ffi::OsString::from("rmda")).chain(plan.args.iter().cloned());
    let inner = Cli::try_parse_from(argv).map_err(|e| {
        rmda_core::error::Error::usage(format!(
            "rmda run: the config does not make a valid command:\n  {line}\n{e}"
        ))
    })?;
    if dry_run {
        println!("{line}");
        return Ok(());
    }
    if matches!(inner.cmd, Cmd::Run { .. }) {
        return Err(rmda_core::error::Error::invalid(
            "rmda run: a config cannot run another config",
        ));
    }
    // Relative paths in the config are relative to the config file.
    std::env::set_current_dir(&plan.base).map_err(|e| {
        rmda_core::error::Error::invalid(format!("rmda run: {}: {e}", plan.base.display()))
    })?;
    eprintln!("rmda run {}: {line}", plan.config_path.display());
    let started = std::time::SystemTime::now();
    let clock = std::time::Instant::now();
    let cpu0 = rmda_core::prof::cpu_seconds();
    let out_dir = match &inner.cmd {
        Cmd::Defects {
            out: Some(o),
            shard: Some((i, n)),
            ..
        }
        | Cmd::Void {
            out: Some(o),
            shard: Some((i, n)),
            ..
        } => Some(cmd_defects::shard_dir(o, *i, *n)),
        Cmd::Defects { out, .. } | Cmd::Void { out, .. } => out.clone(),
        _ => None,
    };
    let threads = if inner.threads == 0 {
        rmda_core::exec::default_threads()
    } else {
        inner.threads
    };
    // A run stopped by --mem-limit still leaves its report.
    {
        let (plan, out_dir) = (plan.clone(), out_dir.clone());
        set_fatal_hook(Box::new(move |why: &str| {
            let cpu = rmda_core::prof::cpu_seconds().zip(cpu0).map(|(b, a)| b - a);
            write_run_report(
                &plan,
                threads,
                started,
                clock,
                cpu,
                Some(why.to_owned()),
                out_dir.as_deref(),
            );
        }));
    }
    let threads = setup(&inner, true);
    let result = run(&inner, threads);
    let cpu = rmda_core::prof::cpu_seconds().zip(cpu0).map(|(b, a)| b - a);
    let error = result.as_ref().err().map(|e| e.to_string());
    write_run_report(
        &plan,
        threads,
        started,
        clock,
        cpu,
        error,
        out_dir.as_deref(),
    );
    result
}

/// `report.json` of an `rmda run`: in the command's output directory when it
/// exists, else beside the config.
fn write_run_report(
    plan: &cmd_run::RunPlan,
    threads: usize,
    started: std::time::SystemTime,
    clock: std::time::Instant,
    cpu: Option<f64>,
    error: Option<String>,
    out_dir: Option<&std::path::Path>,
) {
    let report = cmd_run::report_json(&cmd_run::Outcome {
        plan,
        threads,
        started,
        wall: clock.elapsed().as_secs_f64(),
        cpu,
        error,
        out_dir,
    });
    let path = match out_dir {
        Some(d) if d.is_dir() => d.join("report.json"),
        _ => {
            let stem = plan
                .config_path
                .file_stem()
                .map(|s| s.to_string_lossy().into_owned())
                .unwrap_or_else(|| "run".to_owned());
            plan.base.join(format!("{stem}.report.json"))
        }
    };
    let text = serde_json::to_string_pretty(&report).unwrap_or_default();
    match std::fs::write(&path, text + "\n") {
        Ok(()) => eprintln!("rmda run: report written to {}", path.display()),
        Err(e) => eprintln!("rmda run: could not write {}: {e}", path.display()),
    }
}

/// What every command gets: the parsed command line, the thread pool, the
/// input file, and the single-frame reader most of them use.
pub(crate) struct Ctx<'a> {
    pub(crate) cli: &'a Cli,
    pub(crate) par: Par,
    pub(crate) path: PathBuf,
}

impl Ctx<'_> {
    /// The window's first frame (design §2: analyses read one frame).
    pub(crate) fn load_frame(&self) -> Result<Frame> {
        let (cli, path) = (self.cli, &self.path);
        let _t_read = rmda_core::prof::StageTimer::start("io.read_frames");
        let mut f = rmda_core::io::read_frame_windowed_with(
            path,
            cli.start.or(cli.frame),
            cli.end,
            cli.stride,
            &read_options(cli, true),
        )?;
        drop(_t_read);
        apply_stated_cell(cli, &mut f)?;
        eprintln!("read 1 frame(s), {} atoms in frame 0", f.natoms());
        Ok(f)
    }
}

fn run(cli: &Cli, threads: usize) -> Result<()> {
    let par = Par::new(threads);
    let positional = input_files(cli).into_iter().next();
    let path = match positional {
        Some(p) => p,
        None => {
            eprintln!("no input given; pass a file or use --input");
            std::process::exit(2);
        }
    };
    // Design §2: analyses run on a single frame, so they read a single frame.
    // The commands that want the whole window (`info`, `convert`, `analyze`)
    // stream it instead; nothing below materialises a trajectory.
    let ctx = Ctx { cli, par, path };
    match &cli.cmd {
        Cmd::Run { .. } | Cmd::Merge { .. } | Cmd::Explain { .. } | Cmd::Bench { .. } => {
            unreachable!("dispatched in main")
        }
        Cmd::Index { .. } => cmd_index::command(&ctx)?,
        Cmd::Info { .. } => cmd_info::command(&ctx)?,
        Cmd::Census { .. } => cmd_census::command(&ctx)?,
        Cmd::Cluster { .. } => cmd_cluster::command(&ctx)?,
        Cmd::Rdf { .. } => cmd_rdf::command(&ctx)?,
        Cmd::Wc { .. } => cmd_wc::command(&ctx)?,
        Cmd::Structure { .. } => cmd_structure::command(&ctx)?,
        Cmd::Ptm { .. } => cmd_ptm::command(&ctx)?,
        Cmd::GrainBoundaries { .. } => cmd_grain_boundaries::command(&ctx)?,
        Cmd::Defects { .. } => cmd_defects::command(&ctx)?,
        Cmd::Segment { .. } => cmd_segment::command(&ctx)?,
        Cmd::Ws { .. } => cmd_ws::command(&ctx)?,
        Cmd::Analyze { .. } => cmd_analyze::command(&ctx)?,
        Cmd::Convert { .. } => cmd_convert::command(&ctx)?,
        Cmd::Void { .. } => cmd_void::command(&ctx)?,
    }
    if cli.memreport {
        eprintln!(
            "memreport: peak RSS {}",
            rmda_core::prof::fmt_bytes(rmda_core::prof::peak_rss_bytes())
        );
    }
    if cli.profile {
        rmda_core::prof::report();
    }
    Ok(())
}
