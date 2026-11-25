//! Profiling and memory monitoring (design §23).
//!
//! Named, nested stage timers + RSS sampling. Intended to be cheap (~ns) when
//! disabled; all timing behind an enabled flag.
//!
//! Timers write into a small process-global table of atomics rather than into a
//! `Profiler` instance, because the work being timed is parallel: a per-instance
//! mutex would either serialise the workers or need draining from threads the
//! caller cannot reach. A global `AtomicU64` add per *stage exit* costs nothing
//! next to the stage itself, and aggregation is the same whether the stage ran
//! on one thread or twelve thousand.

use std::cell::Cell;
use std::fmt;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, RwLock};
use std::time::Instant;

/// Upper bound on distinct stage names. Registering past this is a no-op rather
/// than a panic: profiling must never be able to fail a production run.
const MAX_STAGES: usize = 64;

/// One named stage's accumulators. Leaked so timers can hold `&'static`.
struct StageCell {
    name: &'static str,
    nanos: AtomicU64,
    calls: AtomicU64,
    /// Stage timers of this name currently running.
    active: AtomicU64,
    /// Highest anonymous RSS seen while one was running.
    peak_anon: AtomicU64,
}

/// Registry of known stages. Names are interned on first use; measurement never
/// touches this lock.
static REGISTRY: RwLock<Vec<&'static StageCell>> = RwLock::new(Vec::new());

/// Set when the registry is full, so we stop trying to register.
static REGISTRY_FULL: AtomicBool = AtomicBool::new(false);

fn intern(name: &'static str) -> Option<&'static StageCell> {
    {
        // Fast path: already registered. Read lock, no allocation.
        let reg = REGISTRY.read().unwrap_or_else(|e| e.into_inner());
        if let Some(c) = reg.iter().find(|c| c.name == name) {
            return Some(c);
        }
    }
    let mut reg = REGISTRY.write().unwrap_or_else(|e| e.into_inner());
    if let Some(c) = reg.iter().find(|c| c.name == name) {
        return Some(c);
    }
    if reg.len() >= MAX_STAGES || REGISTRY_FULL.load(Ordering::Relaxed) {
        REGISTRY_FULL.store(true, Ordering::Relaxed);
        return None;
    }
    let cell: &'static StageCell = Box::leak(Box::new(StageCell {
        name,
        nanos: AtomicU64::new(0),
        calls: AtomicU64::new(0),
        active: AtomicU64::new(0),
        peak_anon: AtomicU64::new(0),
    }));
    reg.push(cell);
    Some(cell)
}

/// A named stage timer.
///
/// Records on drop, so a stage cannot be left unstopped by an early return.
/// When profiling is disabled the timer holds no `Instant` and drop is a no-op,
/// which keeps the disabled cost to one atomic load.
///
/// `Instant::now()` is a `vDSO` call rather than a syscall, but it is not free
/// and it is not on the hot path for free: keep timers around whole loops, not
/// around individual atoms.
pub struct StageTimer {
    cell: Option<&'static StageCell>,
    start: Option<Instant>,
}

impl StageTimer {
    /// Start timing `name`. Does nothing observable when profiling is disabled.
    #[inline]
    pub fn start(name: &'static str) -> Self {
        if !profiling_enabled() {
            return StageTimer {
                cell: None,
                start: None,
            };
        }
        let cell = intern(name);
        // Read the clock only when we have somewhere to put the answer.
        let start = if let Some(c) = cell {
            c.active.fetch_add(1, Ordering::Relaxed);
            sample_memory();
            Some(Instant::now())
        } else {
            None
        };
        StageTimer { cell, start }
    }

    pub fn is_enabled(&self) -> bool {
        self.start.is_some()
    }

    /// Finish the stage, returning the elapsed seconds.
    ///
    /// Clears the timer's own state first so the `Drop` impl sees an empty timer
    /// and does not record the same stage a second time.
    pub fn stop(mut self) -> f64 {
        let elapsed = match self.start {
            Some(t) => t.elapsed().as_secs_f64(),
            None => 0.0,
        };
        if let Some(c) = self.cell {
            c.nanos.fetch_add((elapsed * 1e9) as u64, Ordering::Relaxed);
            c.calls.fetch_add(1, Ordering::Relaxed);
            sample_memory();
            c.active.fetch_sub(1, Ordering::Relaxed);
        }
        self.cell = None;
        self.start = None;
        elapsed
    }
}

impl Drop for StageTimer {
    fn drop(&mut self) {
        if let (Some(c), Some(t)) = (self.cell, self.start) {
            let secs = t.elapsed().as_secs_f64();
            c.nanos.fetch_add((secs * 1e9) as u64, Ordering::Relaxed);
            c.calls.fetch_add(1, Ordering::Relaxed);
            sample_memory();
            c.active.fetch_sub(1, Ordering::Relaxed);
        }
    }
}

/// Aggregated timings for one stage across every thread that ran it.
#[derive(Debug, Clone)]
pub struct StageSummary {
    pub name: String,
    /// Total wall seconds summed across all threads that entered this stage.
    pub wall: f64,
    pub calls: u64,
    /// Peak anonymous RSS while the stage ran, bytes (0 if never sampled).
    pub peak_anon: u64,
}

impl StageSummary {
    /// Mean wall seconds per call. Zero calls reports 0.0, not NaN.
    pub fn mean(&self) -> f64 {
        if self.calls == 0 {
            0.0
        } else {
            self.wall / self.calls as f64
        }
    }
}

/// Global enable/disable switch for profiling overhead.
pub struct Profile;

static PROFILE_ENABLED: AtomicBool = AtomicBool::new(false);

/// When profiling was switched on: the start of the elapsed time `report`
/// prints.
static PROFILE_START: std::sync::OnceLock<Instant> = std::sync::OnceLock::new();

impl Profile {
    pub fn set_enabled(yes: bool) {
        PROFILE_ENABLED.store(yes, Ordering::Relaxed);
        if yes {
            PROFILE_START.get_or_init(Instant::now);
            start_memory_sampler();
        }
    }
}

/// Highest anonymous RSS seen by [`sample_memory`] since the last
/// [`reset_peak_anon`].
static PEAK_ANON: AtomicU64 = AtomicU64::new(0);

/// `(RssAnon, RssFile)` in bytes from `/proc/self/status` on Linux.
///
/// Anonymous memory is the heap the run allocated. File-backed pages are the
/// mmapped input, which the kernel can drop at will, so they are reported
/// apart from it rather than folded into one "peak RSS".
pub fn rss_anon_file() -> (u64, u64) {
    #[cfg(target_os = "linux")]
    {
        let mut anon = 0;
        let mut file = 0;
        if let Ok(s) = std::fs::read_to_string("/proc/self/status") {
            let kb = |rest: &str| -> u64 {
                rest.trim()
                    .trim_end_matches("kB")
                    .trim()
                    .parse::<u64>()
                    .unwrap_or(0)
                    * 1024
            };
            for line in s.lines() {
                if let Some(r) = line.strip_prefix("RssAnon:") {
                    anon = kb(r);
                } else if let Some(r) = line.strip_prefix("RssFile:") {
                    file = kb(r);
                }
            }
        }
        (anon, file)
    }
    #[cfg(not(target_os = "linux"))]
    {
        (0, 0)
    }
}

/// Sample anonymous RSS into the global peak and every running stage's peak.
pub fn sample_memory() {
    let (anon, _) = rss_anon_file();
    PEAK_ANON.fetch_max(anon, Ordering::Relaxed);
    let reg = REGISTRY.read().unwrap_or_else(|e| e.into_inner());
    for c in reg.iter() {
        if c.active.load(Ordering::Relaxed) > 0 {
            c.peak_anon.fetch_max(anon, Ordering::Relaxed);
        }
    }
}

/// Peak anonymous RSS since the last reset, bytes.
pub fn peak_anon_bytes() -> u64 {
    sample_memory();
    PEAK_ANON.load(Ordering::Relaxed)
}

/// Restart the anonymous-RSS peak (e.g. per frame of a series).
pub fn reset_peak_anon() {
    PEAK_ANON.store(rss_anon_file().0, Ordering::Relaxed);
}

/// Start the 20 ms memory sampler thread (once per process).
pub fn start_memory_sampler() {
    static STARTED: std::sync::Once = std::sync::Once::new();
    STARTED.call_once(|| {
        let _ = std::thread::Builder::new()
            .name("rmda-mem-sampler".into())
            .spawn(|| loop {
                sample_memory();
                std::thread::sleep(std::time::Duration::from_millis(20));
            });
    });
}

#[inline]
pub fn profiling_enabled() -> bool {
    PROFILE_ENABLED.load(Ordering::Relaxed)
}

/// Snapshot every registered stage, slowest first.
pub fn snapshot() -> Vec<StageSummary> {
    let reg = REGISTRY.read().unwrap_or_else(|e| e.into_inner());
    let mut out: Vec<StageSummary> = reg
        .iter()
        .map(|c| StageSummary {
            name: c.name.to_string(),
            wall: c.nanos.load(Ordering::Relaxed) as f64 / 1e9,
            calls: c.calls.load(Ordering::Relaxed),
            peak_anon: c.peak_anon.load(Ordering::Relaxed),
        })
        .filter(|s| s.calls > 0)
        .collect();
    // Sort by wall descending, then by name so equal-cost stages are stable.
    out.sort_by(|a, b| {
        b.wall
            .partial_cmp(&a.wall)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.name.cmp(&b.name))
    });
    out
}

/// The stages running right now (entered and not yet left), by name: what a
/// memory-limit report names as the culprit. Empty unless profiling is on.
pub fn active_stages() -> Vec<String> {
    let reg = REGISTRY.read().unwrap_or_else(|e| e.into_inner());
    reg.iter()
        .filter(|c| c.active.load(Ordering::Relaxed) > 0)
        .map(|c| c.name.to_string())
        .collect()
}

/// Reset all accumulators, keeping names registered.
pub fn reset() {
    let reg = REGISTRY.read().unwrap_or_else(|e| e.into_inner());
    for c in reg.iter() {
        c.nanos.store(0, Ordering::Relaxed);
        c.calls.store(0, Ordering::Relaxed);
    }
}

/// Print the stage table plus peak RSS to stderr.
///
/// Goes to stderr so it can be combined with a run whose real output is on
/// stdout, and so `--profile` does not corrupt a redirected histogram.
pub fn report() {
    // Drain the workers first: after a parallel region joins their un-flushed
    // residue is still recoverable, and skipping this leaves the call counts
    // short.
    flush_all_threads();
    let rows = snapshot();
    if rows.is_empty() {
        eprintln!("profile: no stages recorded (was --profile passed?)");
        return;
    }
    eprintln!("profile (wall seconds summed across threads; peak heap while the stage ran):");
    let w = rows.iter().map(|r| r.name.len()).max().unwrap_or(8).max(8);
    for r in &rows {
        let heap = if r.peak_anon > 0 {
            fmt_bytes(r.peak_anon)
        } else {
            "-".into()
        };
        eprintln!(
            "  {name:w$}  {wall:9.3} s  {calls:>10} calls  {mean:>9.6} s/call  {heap:>10}",
            name = r.name,
            wall = r.wall,
            calls = r.calls,
            mean = r.mean(),
        );
    }
    // Sum of thread-time exceeds elapsed time whenever stages ran in parallel,
    // so this is a work measure, not a latency one. Say so rather than let it be
    // read as a percentage of wall clock.
    let total: f64 = rows.iter().map(|r| r.wall).sum();
    eprintln!("  {:w$}  {:9.3} s  (total thread-time)", "SUM", total);
    if let Some(t0) = PROFILE_START.get() {
        eprintln!(
            "  {:w$}  {:9.3} s  (elapsed since --profile started, reading included)",
            "WALL",
            t0.elapsed().as_secs_f64()
        );
    }
    let (_, file) = rss_anon_file();
    eprintln!(
        "  peak heap (anonymous RSS): {}   file-backed now: {}   peak RSS incl. mapped files: {}",
        fmt_bytes(peak_anon_bytes()),
        fmt_bytes(file),
        fmt_bytes(peak_rss_bytes())
    );
}

/// A recorded stage timing.
pub struct StageRecord {
    pub name: &'static str,
    pub wall: f64,
    pub cpu: f64,
    pub rss_delta: i64,
}

impl fmt::Display for StageRecord {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}: {:.3} s", self.name, self.wall)
    }
}

/// Peak RSS in bytes via `/proc/self/status VmHWM` on Linux; 0 elsewhere.
pub fn peak_rss_bytes() -> u64 {
    #[cfg(target_os = "linux")]
    {
        if let Ok(s) = std::fs::read_to_string("/proc/self/status") {
            for line in s.lines() {
                if let Some(rest) = line.strip_prefix("VmHWM:") {
                    let kb: u64 = rest
                        .trim()
                        .trim_end_matches("kB")
                        .trim()
                        .parse()
                        .unwrap_or(0);
                    return kb * 1024;
                }
            }
        }
        0
    }
    #[cfg(not(target_os = "linux"))]
    {
        0
    }
}

/// CPU time this process has used so far, user and system, summed over all
/// its threads, in seconds: `utime + stime` of `/proc/self/stat` on Linux
/// (clock ticks of `USER_HZ`, which the kernel fixes at 100 for user space,
/// so 10 ms resolution). `None` elsewhere or when it cannot be read.
pub fn cpu_seconds() -> Option<f64> {
    #[cfg(target_os = "linux")]
    {
        let s = std::fs::read_to_string("/proc/self/stat").ok()?;
        // The command name (field 2) is in parentheses and may hold spaces:
        // count fields after its closing parenthesis. utime and stime are
        // fields 14 and 15, so the 12th and 13th after it.
        let rest = &s[s.rfind(')')? + 1..];
        let f: Vec<&str> = rest.split_whitespace().collect();
        let utime: u64 = f.get(11)?.parse().ok()?;
        let stime: u64 = f.get(12)?.parse().ok()?;
        Some((utime + stime) as f64 / 100.0)
    }
    #[cfg(not(target_os = "linux"))]
    {
        None
    }
}

/// Current RSS in bytes via `/proc/self/status VmRSS` on Linux; 0 elsewhere.
pub fn current_rss_bytes() -> u64 {
    #[cfg(target_os = "linux")]
    {
        if let Ok(s) = std::fs::read_to_string("/proc/self/status") {
            for line in s.lines() {
                if let Some(rest) = line.strip_prefix("VmRSS:") {
                    let kb: u64 = rest
                        .trim()
                        .trim_end_matches("kB")
                        .trim()
                        .parse()
                        .unwrap_or(0);
                    return kb * 1024;
                }
            }
        }
        0
    }
    #[cfg(not(target_os = "linux"))]
    {
        0
    }
}

/// A named allocation counter (design §23.2).
#[derive(Default)]
pub struct MemCounters {
    labels: Mutex<Vec<(&'static str, AtomicU64)>>,
}

impl MemCounters {
    pub fn new() -> Self {
        Default::default()
    }

    pub fn add(&self, label: &'static str, bytes: u64) {
        if let Ok(mut l) = self.labels.lock() {
            if let Some(e) = l.iter_mut().find(|(n, _)| *n == label) {
                e.1.fetch_add(bytes, Ordering::Relaxed);
            } else {
                l.push((label, AtomicU64::new(bytes)));
            }
        }
    }

    pub fn report(&self) -> Vec<(String, u64)> {
        let mut out = Vec::new();
        if let Ok(l) = self.labels.lock() {
            for (n, v) in l.iter() {
                out.push((n.to_string(), v.load(Ordering::Relaxed)));
            }
        }
        out.sort_by_key(|e| std::cmp::Reverse(e.1));
        out
    }
}

pub fn fmt_bytes(b: u64) -> String {
    if b >= 1 << 30 {
        format!("{:.2} GB", b as f64 / (1u64 << 30) as f64)
    } else if b >= 1 << 20 {
        format!("{:.2} MB", b as f64 / (1u64 << 20) as f64)
    } else if b >= 1 << 10 {
        format!("{:.2} KB", b as f64 / (1u64 << 10) as f64)
    } else {
        format!("{b} B")
    }
}

// ---------------------------------------------------------------------------
// Per-thread accumulation for inside a parallel loop.
//
// A `StageTimer` around a parallel region measures the region's wall time and
// says nothing about how the time split across its phases. Putting a timer
// inside the per-atom closure would work but costs two `Instant::now()` calls
// per atom and, worse, serialises on one global atomic per phase per atom --
// at 128M atoms that contention is itself a distortion of what is being
// measured.
//
// So per-phase timing accumulates in thread-local counters and is flushed to
// the global table once every `FLUSH_EVERY` atoms. The flush cost is then
// amortised over thousands of atoms and independent of thread count.
// ---------------------------------------------------------------------------

/// Phase counters for the thread currently running.
///
/// Backed by an `Arc` that is also in a process-global registry, so the thread
/// that owns the region cannot reach the other workers' thread-locals but the
/// main thread *can* drain everyone at the end. Thread-locals alone would lose
/// up to `FLUSH_EVERY` counts per worker (a call count of 491,524 for a
/// 500,000-atom run): the wall time is within ~1%, but a visibly wrong
/// denominator is not something to ship in a diagnostic.
#[derive(Default)]
pub struct PhaseAcc {
    nanos: Vec<u64>,
    calls: Vec<u64>,
}

/// Every thread that has ever accumulated, so `flush_all_threads` can drain it.
static ACCS: RwLock<Vec<Arc<Mutex<PhaseAcc>>>> = RwLock::new(Vec::new());

fn register_acc() -> Arc<Mutex<PhaseAcc>> {
    let acc = Arc::new(Mutex::new(PhaseAcc::default()));
    ACCS.write()
        .unwrap_or_else(|e| e.into_inner())
        .push(Arc::clone(&acc));
    acc
}

thread_local! {
    static MY_ACC: Arc<Mutex<PhaseAcc>> = register_acc();
}

/// Local slot index for `name` within this thread's counters, if enabled.
#[inline]
fn phase_slot(name: &'static str) -> Option<usize> {
    if !profiling_enabled() {
        return None;
    }
    let cell = intern(name)?;
    // The slot index is the position in REGISTRY, which can grow between calls;
    // re-read it so a late-registered stage cannot alias another.
    REGISTRY
        .read()
        .unwrap_or_else(|e| e.into_inner())
        .iter()
        .position(|c| std::ptr::eq(*c, cell))
}

/// How often per-thread phase counters are flushed to the global table. Only
/// bounds memory and lock traffic, since the residue is recoverable at the
/// end, so it can be generous.
const FLUSH_EVERY: u64 = 4096;

/// Add `secs` to this thread's counter for `name`.
#[inline]
pub fn accumulate(name: &'static str, secs: f64) {
    let Some(idx) = phase_slot(name) else { return };
    MY_ACC.with(|a| {
        let mut a = a.lock().unwrap_or_else(|e| e.into_inner());
        if a.nanos.len() <= idx {
            a.nanos.resize(idx + 1, 0);
            a.calls.resize(idx + 1, 0);
        }
        a.nanos[idx] += (secs * 1e9) as u64;
        a.calls[idx] += 1;
    });
    maybe_flush();
}

thread_local! {
    static SINCE_FLUSH: Cell<u64> = const { Cell::new(0) };
}

#[inline]
fn maybe_flush() {
    let due = SINCE_FLUSH.with(|c| {
        let n = c.get() + 1;
        if n >= FLUSH_EVERY {
            c.set(0);
            true
        } else {
            c.set(n);
            false
        }
    });
    if due {
        flush_phases();
    }
}

/// Move one accumulator's totals into the global table, emptying it.
fn drain_into_global(acc: &Arc<Mutex<PhaseAcc>>) {
    let drained: Vec<(usize, u64, u64)> = {
        let mut a = acc.lock().unwrap_or_else(|e| e.into_inner());
        let out: Vec<(usize, u64, u64)> = a
            .nanos
            .iter()
            .enumerate()
            .filter(|(_, &n)| n > 0)
            .map(|(i, &n)| (i, n, a.calls[i]))
            .collect();
        a.nanos.iter_mut().for_each(|x| *x = 0);
        a.calls.iter_mut().for_each(|x| *x = 0);
        out
    };
    if drained.is_empty() {
        return;
    }
    let reg = REGISTRY.read().unwrap_or_else(|e| e.into_inner());
    for (idx, nanos, calls) in drained {
        if let Some(c) = reg.get(idx) {
            c.nanos.fetch_add(nanos, Ordering::Relaxed);
            c.calls.fetch_add(calls, Ordering::Relaxed);
        }
    }
}

/// Move the calling thread's phase counters into the global table.
pub fn flush_phases() {
    MY_ACC.with(drain_into_global);
}

/// Drain every thread's counters, then snapshot.
///
/// Call after a parallel region has joined: workers are finished by then, so
/// their residue is recoverable and the call counts are exact.
pub fn flush_all_threads() {
    let accs: Vec<Arc<Mutex<PhaseAcc>>> = ACCS.read().unwrap_or_else(|e| e.into_inner()).clone();
    for a in accs {
        drain_into_global(&a);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The stage table and enable flag are process-global by design, so these
    /// tests cannot run concurrently: each one resets the accumulators the
    /// others are asserting on. Serialise rather than give up the global table.
    static SERIAL: Mutex<()> = Mutex::new(());

    #[test]
    fn disabled_timers_record_nothing_and_cost_nothing() {
        let _guard = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
        Profile::set_enabled(false);
        reset();
        let t = StageTimer::start("unit.test.disabled");
        assert!(!t.is_enabled());
        assert_eq!(t.stop(), 0.0);
        accumulate("unit.test.disabled.accum", 1e-3);
        flush_phases();
        assert!(snapshot().iter().all(|s| s.name != "unit.test.disabled"));
        assert!(snapshot()
            .iter()
            .all(|s| s.name != "unit.test.disabled.accum"));
    }

    #[test]
    fn enabled_timer_records_and_aggregates() {
        let _guard = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
        Profile::set_enabled(true);
        reset();
        for _ in 0..3 {
            let t = StageTimer::start("unit.test.aggregate");
            std::hint::black_box(t.is_enabled());
            let secs = t.stop();
            assert!(secs >= 0.0);
        }
        let rows = snapshot();
        let row = rows
            .iter()
            .find(|r| r.name == "unit.test.aggregate")
            .expect("stage missing from snapshot");
        assert_eq!(row.calls, 3);
        assert!(row.wall >= 0.0);
        Profile::set_enabled(false);
    }

    #[test]
    fn drop_records_without_an_explicit_stop() {
        let _guard = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
        Profile::set_enabled(true);
        reset();
        {
            let _t = StageTimer::start("unit.test.drop");
        }
        let row = snapshot();
        assert!(row
            .iter()
            .any(|r| r.name == "unit.test.drop" && r.calls == 1));
        Profile::set_enabled(false);
    }

    #[test]
    fn accumulate_flushes_into_the_global_table() {
        let _guard = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
        Profile::set_enabled(true);
        reset();
        // Push well past FLUSH_EVERY so a flush is guaranteed to happen.
        for _ in 0..(FLUSH_EVERY + 16) {
            accumulate("unit.test.accumulate", 1e-9);
        }
        flush_phases();
        let row = snapshot();
        let row = row
            .iter()
            .find(|r| r.name == "unit.test.accumulate")
            .expect("accumulated stage missing");
        assert_eq!(row.calls, FLUSH_EVERY + 16);
        Profile::set_enabled(false);
    }

    #[test]
    fn flush_all_threads_recovers_every_worker_s_residue() {
        let _guard = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
        Profile::set_enabled(true);
        reset();
        // A residue smaller than FLUSH_EVERY is exactly what a worker leaves
        // behind: it has not flushed, and the owning thread cannot see it.
        const RESIDUE: u64 = 100;
        let handles: Vec<_> = (0..4)
            .map(|_| {
                std::thread::spawn(|| {
                    for _ in 0..RESIDUE {
                        accumulate("unit.test.residue", 1e-9);
                    }
                })
            })
            .collect();
        for h in handles {
            h.join().unwrap();
        }
        flush_all_threads();
        let row = snapshot()
            .into_iter()
            .find(|r| r.name == "unit.test.residue")
            .expect("worker residue missing after flush_all_threads");
        assert_eq!(row.calls, RESIDUE * 4);
        Profile::set_enabled(false);
    }

    #[test]
    fn summary_sorts_slowest_first_and_reports_mean() {
        let _guard = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
        Profile::set_enabled(true);
        reset();
        accumulate("unit.test.fast", 1e-6);
        accumulate("unit.test.slow", 5e-3);
        flush_phases();
        let rows = snapshot();
        let slow = rows
            .iter()
            .position(|r| r.name == "unit.test.slow")
            .unwrap();
        let fast = rows
            .iter()
            .position(|r| r.name == "unit.test.fast")
            .unwrap();
        assert!(slow < fast, "slowest stage must come first");
        assert!(rows[slow].mean() > rows[fast].mean());
        Profile::set_enabled(false);
    }
}
