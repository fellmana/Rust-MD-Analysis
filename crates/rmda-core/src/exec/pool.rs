//! Thread-pool wrapper (design §10.2).

use rayon::prelude::*;
use std::sync::atomic::{AtomicUsize, Ordering};

/// Atoms per fixed chunk when an `0..n` atom range is split (design §10.2).
pub const CHUNK_ATOMS: usize = 65536;

/// Integer ceiling division (`usize::div_ceil` is not available on every
/// supported toolchain).
#[inline]
pub fn ceil_div(a: usize, b: usize) -> usize {
    if b == 0 {
        return a;
    }
    (a + b - 1) / b
}

/// A rayon pool with a given thread count, cached for the process.
///
/// Building a `ThreadPool` spawns OS threads, so `Par` cannot build one per
/// call without paying for that on every analysis. The cache is keyed by thread
/// count and the pools are leaked deliberately: they must stay alive for the
/// life of the process because `Par` is `Copy` and holds borrows into them.
fn pool_for(threads: usize) -> Option<&'static rayon::ThreadPool> {
    use std::collections::HashMap;
    use std::sync::{Mutex, OnceLock};

    static CACHE: OnceLock<Mutex<HashMap<usize, &'static rayon::ThreadPool>>> = OnceLock::new();
    let cache = CACHE.get_or_init(|| Mutex::new(HashMap::new()));
    let mut map = cache.lock().expect("pool cache poisoned");
    if let Some(p) = map.get(&threads) {
        return Some(*p);
    }
    let pool: &'static rayon::ThreadPool = Box::leak(Box::new(
        rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .expect("failed to build rayon pool"),
    ));
    map.insert(threads, pool);
    Some(pool)
}

/// An execution context that selects sequential or parallel iteration.
///
/// All hot loops go through this so that `--threads 1` gives a sequential path
/// (avoiding any rayon overhead and guaranteeing single-thread semantics).
///
/// `pool` is the rayon pool the parallel paths install into, sized to `threads`.
/// This matters: rayon's *global* pool is built lazily at whatever size the
/// runtime defaults to, so a plain `into_par_iter()` would silently ignore
/// `threads` entirely: `--threads 1` and `--threads 12` would both run on
/// every core.
#[derive(Clone, Copy, Debug)]
pub struct Par {
    pub threads: usize,
    /// `None` for the sequential path, where no pool is needed.
    pool: Option<&'static rayon::ThreadPool>,
}

impl Default for Par {
    fn default() -> Self {
        // Consult the configured count (CLI `--threads`, `rmda.set_threads`)
        // rather than only rayon's global default, so the setting reaches
        // every context built as `Par::default()`.
        Par::new(configured_threads())
    }
}

impl Par {
    pub fn new(threads: usize) -> Self {
        let threads = threads.max(1);
        Par {
            threads,
            pool: if threads > 1 { pool_for(threads) } else { None },
        }
    }

    pub fn is_parallel(&self) -> bool {
        self.threads > 1
    }

    /// Run `f` in this context's pool, or inline when sequential.
    ///
    /// Falling through to `f()` directly for the single-threaded case is what
    /// keeps `--threads 1` off rayon's scheduler entirely.
    #[inline]
    pub(crate) fn install<F, R>(&self, f: F) -> R
    where
        F: FnOnce() -> R + Send,
        R: Send,
    {
        match self.pool {
            Some(p) => p.install(f),
            None => f(),
        }
    }

    /// Splits `0..n` into chunk ranges, indexed by ordinal `0..nchunks`.
    pub fn chunk_ranges(&self, n: usize) -> Vec<(usize, usize)> {
        let nchunks = ceil_div(n, CHUNK_ATOMS).max(1);
        let mut out = Vec::with_capacity(nchunks);
        for c in 0..nchunks {
            let start = c * CHUNK_ATOMS;
            let end = (start + CHUNK_ATOMS).min(n);
            if start >= end {
                break;
            }
            out.push((start, end));
        }
        out
    }

    /// Run `f(range)` for each chunk, in parallel or sequentially.
    pub fn for_each_chunk<F>(&self, n: usize, f: F)
    where
        F: Fn((usize, usize)) + Sync,
    {
        let ranges = self.chunk_ranges(n);
        if self.is_parallel() {
            let f = &f;
            self.install(move || ranges.par_iter().for_each(|&r| f(r)));
        } else {
            for r in ranges {
                f(r);
            }
        }
    }

    /// Map every fixed atom chunk to a value, collected **in ascending chunk
    /// ordinal order** — the primitive of the deterministic reduction pattern
    /// (design §10.2). Runs in this context's pool, or sequentially when the
    /// context is single-threaded, so `--threads` reaches histogram
    /// reductions instead of only the per-atom phases.
    pub fn map_chunks<T, F>(&self, n: usize, f: F) -> Vec<T>
    where
        T: Send,
        F: Fn(usize, usize) -> T + Sync,
    {
        let ranges = self.chunk_ranges(n);
        if self.is_parallel() {
            let f = &f;
            self.install(move || ranges.par_iter().map(|&(s, e)| f(s, e)).collect())
        } else {
            ranges.iter().map(|&(s, e)| f(s, e)).collect()
        }
    }

    /// Fallible [`Par::map_chunks`]: results keep chunk-ordinal order and the
    /// error returned is the one from the **lowest failing chunk**, not
    /// whichever worker failed first — scheduling must not change which
    /// error a run reports (same rule as [`Par::map_collect`]).
    pub fn try_map_chunks<T, E, F>(&self, n: usize, f: F) -> std::result::Result<Vec<T>, E>
    where
        T: Send,
        E: Send,
        F: Fn(usize, usize) -> std::result::Result<T, E> + Sync,
    {
        let ranges = self.chunk_ranges(n);
        let results: Vec<std::result::Result<T, E>> = if self.is_parallel() {
            let f = &f;
            self.install(move || ranges.par_iter().map(|&(s, e)| f(s, e)).collect::<Vec<_>>())
        } else {
            ranges.iter().map(|&(s, e)| f(s, e)).collect()
        };
        let mut out = Vec::with_capacity(results.len());
        for r in results {
            out.push(r?);
        }
        Ok(out)
    }

    /// Fold every fixed atom chunk into one accumulator, combining **in
    /// ascending chunk ordinal order** (§10.2): with an order-fixed combine and
    /// integer accumulation the result is bit-reproducible across any thread
    /// count.
    pub fn fold_chunks<T, U, F, C>(&self, n: usize, zero: U, f: F, combine: C) -> U
    where
        T: Send,
        U: Clone + Send,
        F: Fn(usize, usize) -> T + Sync,
        C: Fn(&mut U, &T) + Sync,
    {
        let mut total = zero;
        for acc in self.map_chunks(n, f) {
            combine(&mut total, &acc);
        }
        total
    }

    /// Parallel map over `0..n` calling `f(i)`.
    pub fn for_each<F>(&self, n: usize, f: F)
    where
        F: Fn(usize) + Sync + Send,
    {
        if self.is_parallel() {
            let f = &f;
            self.install(move || (0..n).into_par_iter().for_each(f));
        } else {
            (0..n).for_each(f);
        }
    }

    /// Write `out[i] = f(i)` in place, in parallel chunks of [`CHUNK_ATOMS`]
    /// (design §10.2). The output is preallocated by the caller, so nothing
    /// per index is buffered or copied.
    pub fn fill<T, F>(&self, out: &mut [T], f: F)
    where
        T: Send,
        F: Fn(usize, &mut T) + Sync + Send,
    {
        self.try_fill_with(
            out,
            || (),
            |_, i, o| {
                f(i, o);
                Ok::<(), std::convert::Infallible>(())
            },
        )
        .unwrap_or_else(|e| match e {});
    }

    /// [`Par::fill`] with per-chunk scratch from `init` and fallible items.
    ///
    /// The returned error is the one at the lowest index: chunks are disjoint
    /// ascending ranges, each stops at its own first error, and the errors are
    /// scanned in chunk order. Memory is one `Result` per chunk, not per item.
    pub fn try_fill_with<T, S, E, I, F>(&self, out: &mut [T], init: I, f: F) -> Result<(), E>
    where
        T: Send,
        E: Send,
        I: Fn() -> S + Sync + Send,
        F: Fn(&mut S, usize, &mut T) -> Result<(), E> + Sync + Send,
    {
        let run = |(c, chunk): (usize, &mut [T])| -> Result<(), E> {
            let mut scratch = init();
            let base = c * CHUNK_ATOMS;
            for (k, o) in chunk.iter_mut().enumerate() {
                f(&mut scratch, base + k, o)?;
            }
            Ok(())
        };
        let results: Vec<Result<(), E>> = if self.is_parallel() {
            let run = &run;
            self.install(move || {
                out.par_chunks_mut(CHUNK_ATOMS)
                    .enumerate()
                    .map(run)
                    .collect()
            })
        } else {
            out.chunks_mut(CHUNK_ATOMS).enumerate().map(run).collect()
        };
        results.into_iter().collect()
    }

    /// Parallel map over `0..n` collecting one `R` per index, in index order.
    ///
    /// `map` must be pure with respect to `i`: rayon collects an *indexed*
    /// iterator into a `Vec` in index order, so the output order — and hence
    /// the whole analysis — is deterministic regardless of thread scheduling
    /// (design §10.2). Errors from any item are collected and the first one in
    /// *index* order is returned, again deterministically.
    ///
    /// A `Vec<Mutex<Option<R>>>` with one lock per atom would give the same
    /// ordering at the cost of a mutex construction plus a lock per atom (500k
    /// of each on the 500k-atom reference). Collecting gets the ordering from
    /// rayon directly: the mutex would buy nothing that `collect` does not
    /// already guarantee.
    ///
    /// Note this deliberately does *not* use rayon's `FromParallelIterator for
    /// Result`, which short-circuits and surfaces whichever error a worker
    /// happened to hit first. That is not index-ordered, so it would make the
    /// reported error depend on scheduling.
    pub fn map_collect<R, E, F>(&self, n: usize, map: F) -> std::result::Result<Vec<R>, E>
    where
        R: Send,
        E: Send,
        F: Fn(usize) -> std::result::Result<R, E> + Sync + Send,
    {
        if self.is_parallel() {
            // Collect per chunk rather than per item: a `Result<R, E>` per atom
            // costs `size_of::<E>()` (~80 B) on top of every `R`, which at 128M
            // atoms is ~10 GB of transient. Each chunk stops at its
            // own first error, and chunks are scanned in ascending order, so
            // the reported error is still the lowest-index one.
            let map = &map;
            let ranges = self.chunk_ranges(n);
            let chunks: Vec<std::result::Result<Vec<R>, E>> = self.install(move || {
                ranges
                    .par_iter()
                    .map(|&(a, b)| (a..b).map(map).collect())
                    .collect()
            });
            let mut out = Vec::with_capacity(n);
            for c in chunks {
                out.extend(c?);
            }
            Ok(out)
        } else {
            let mut out = Vec::with_capacity(n);
            for i in 0..n {
                out.push(map(i)?);
            }
            Ok(out)
        }
    }
}

/// Global thread count. Defaults to logical cores, clamped by rayon.
pub fn num_threads() -> usize {
    rayon::current_num_threads()
}

/// The thread count when none is asked for: the CPUs SLURM gave the task
/// (`SLURM_CPUS_PER_TASK`), else every core. Under SLURM every core is the
/// node's, not the job's, and oversubscribing a shared node slows the job.
pub fn default_threads() -> usize {
    threads_from(
        std::env::var("SLURM_CPUS_PER_TASK").ok().as_deref(),
        num_threads(),
    )
}

/// [`default_threads`] from the variable's value and the core count.
fn threads_from(slurm_cpus_per_task: Option<&str>, cores: usize) -> usize {
    slurm_cpus_per_task
        .and_then(|v| v.trim().parse::<usize>().ok())
        .filter(|&n| n > 0)
        .unwrap_or(cores)
}

/// Process-wide requested thread count; 0 = unset. One shared static: a
/// function-local `static` in each of `set_threads` and `configured_threads`
/// would be two *different* variables, and the reader would never see the
/// writer's value.
static CONFIGURED: AtomicUsize = AtomicUsize::new(0);

/// Set the process-wide thread count. Called once at startup from the CLI
/// (`--threads`) and from the `rmda.set_threads` binding.
pub fn set_threads(n: usize) {
    CONFIGURED.store(n.max(1), Ordering::SeqCst);
}

/// Size rayon's global pool, which the I/O row parsers and writers run on,
/// to `n` threads. Only the first call in a process takes effect (rayon
/// builds the global pool once), so the CLI calls it at startup.
pub fn init_global_pool(n: usize) {
    let _ = rayon::ThreadPoolBuilder::new()
        .num_threads(n.max(1))
        .build_global();
}

/// Requested thread count (`num_threads()` when never configured).
pub fn configured_threads() -> usize {
    let c = CONFIGURED.load(Ordering::SeqCst);
    if c == 0 {
        num_threads()
    } else {
        c
    }
}

/// A persistent parallel pool handle to pass through analysis contexts.
pub struct ParPool {
    pool: Option<rayon::ThreadPool>,
    threads: usize,
}

impl ParPool {
    pub fn new(threads: usize) -> Self {
        let threads = threads.max(1);
        let pool = if threads > 1 {
            Some(
                rayon::ThreadPoolBuilder::new()
                    .num_threads(threads)
                    .build()
                    .expect("failed to build rayon pool"),
            )
        } else {
            None
        };
        ParPool { pool, threads }
    }

    pub fn threads(&self) -> usize {
        self.threads
    }

    pub fn par(&self) -> Par {
        Par::new(self.threads)
    }

    /// Run `f` inside this pool (inline when the handle is
    /// single-threaded), so callers get the configured thread count
    /// without installing globally.
    pub fn install<F, R>(&self, f: F) -> R
    where
        F: FnOnce() -> R + Send,
        R: Send,
    {
        match &self.pool {
            Some(p) => p.install(f),
            None => f(),
        }
    }
}

impl Default for ParPool {
    fn default() -> Self {
        ParPool::new(num_threads())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn chunk_ranges_cover() {
        let p = Par::new(4);
        let ranges = p.chunk_ranges(150_000);
        let total: usize = ranges.iter().map(|(a, b)| b - a).sum();
        assert_eq!(total, 150_000);
        // contiguous, ascending
        for w in ranges.windows(2) {
            assert_eq!(w[0].1, w[1].0);
        }
    }

    #[test]
    fn chunk_ordinal_order() {
        let p = Par::new(1);
        let seen = std::sync::Mutex::new(Vec::new());
        p.for_each_chunk(70_000, |r| seen.lock().unwrap().push(r));
        // sequential must be in ascending chunk order
        assert_eq!(*seen.lock().unwrap(), p.chunk_ranges(70_000));
    }

    #[test]
    fn map_collect_is_index_ordered() {
        for threads in [1usize, 2, 4, 8] {
            let p = Par::new(threads);
            let out = p
                .map_collect::<_, std::convert::Infallible, _>(10_000, |i| Ok(i * 3))
                .unwrap();
            for (i, v) in out.iter().enumerate() {
                assert_eq!(*v, i * 3, "threads={threads}: out of order at {i}");
            }
        }
    }

    /// The first error must be the lowest-index one, not whichever worker
    /// finished first. This is the property that rules out rayon's
    /// short-circuiting `Result` collector.
    #[test]
    fn map_collect_reports_the_lowest_index_error() {
        for threads in [1usize, 2, 4, 8] {
            let p = Par::new(threads);
            let err = p
                .map_collect::<(), usize, _>(10_000, |i| if i % 97 == 3 { Err(i) } else { Ok(()) })
                .unwrap_err();
            assert_eq!(err, 3, "threads={threads}: wrong error reported");
        }
    }

    /// Errors in several chunks, so the lowest index has to win across chunks,
    /// not just within one.
    #[test]
    fn map_collect_and_try_fill_report_the_lowest_index_error_across_chunks() {
        let n = 3 * CHUNK_ATOMS + 17;
        let bad = |i: usize| i == 2 * CHUNK_ATOMS + 5 || i == CHUNK_ATOMS + 9;
        for threads in [1usize, 2, 4] {
            let p = Par::new(threads);
            let e = p
                .map_collect::<usize, usize, _>(n, |i| if bad(i) { Err(i) } else { Ok(i) })
                .unwrap_err();
            assert_eq!(e, CHUNK_ATOMS + 9, "map_collect threads={threads}");
            let mut out = vec![0usize; n];
            let e = p
                .try_fill_with(
                    &mut out,
                    || (),
                    |_, i, o| {
                        if bad(i) {
                            return Err(i);
                        }
                        *o = i;
                        Ok(())
                    },
                )
                .unwrap_err();
            assert_eq!(e, CHUNK_ATOMS + 9, "try_fill_with threads={threads}");
        }
    }

    #[test]
    fn fill_writes_every_slot_for_any_thread_count() {
        let n = 2 * CHUNK_ATOMS + 3;
        for threads in [1usize, 3, 8] {
            let mut out = vec![0u64; n];
            Par::new(threads).fill(&mut out, |i, o| *o = (i as u64) * 7 + 1);
            assert!(out
                .iter()
                .enumerate()
                .all(|(i, &v)| v == (i as u64) * 7 + 1));
            let collected = Par::new(threads)
                .map_collect::<u64, (), _>(n, |i| Ok(i as u64 * 7 + 1))
                .unwrap();
            assert_eq!(collected, out);
        }
    }

    #[test]
    fn slurm_cpus_per_task_sets_the_default_thread_count() {
        assert_eq!(threads_from(Some("64"), 128), 64);
        assert_eq!(threads_from(Some(" 8\n"), 128), 8);
        // Unset, empty, zero or malformed: every core.
        for v in [None, Some(""), Some("0"), Some("4(x2)")] {
            assert_eq!(threads_from(v, 128), 128, "{v:?}");
        }
    }

    /// `set_threads` and `configured_threads` share one static, so the write
    /// is visible to the reader.
    #[test]
    fn set_threads_round_trips_to_configured_threads() {
        set_threads(3);
        assert_eq!(configured_threads(), 3);
        set_threads(0); // clamped to 1
        assert_eq!(configured_threads(), 1);
    }

    /// Chunk reductions must be bit-identical for any thread count: fixed
    /// chunk ordinals + ascending combine (design §10.2).
    #[test]
    fn fold_chunks_is_thread_count_independent() {
        for threads in [1usize, 2, 4, 8] {
            let p = Par::new(threads);
            let n = 2 * CHUNK_ATOMS + 1234;
            let got = p.fold_chunks(
                n,
                vec![0u64; 100],
                |s, e| {
                    let mut local = vec![0u64; 100];
                    for i in s..e {
                        local[i % 100] += 1;
                    }
                    local
                },
                |total: &mut Vec<u64>, local: &Vec<u64>| {
                    for (t, l) in total.iter_mut().zip(local) {
                        *t += *l;
                    }
                },
            );
            let mut expect = vec![0u64; 100];
            for i in 0..n {
                expect[i % 100] += 1;
            }
            assert_eq!(got, expect, "threads={threads} diverged");
        }
    }

    /// `try_map_chunks` reports the lowest failing chunk's error, not the
    /// first worker to fail (same scheduling rule as `map_collect`).
    #[test]
    fn try_map_chunks_reports_the_lowest_chunk_error() {
        for threads in [1usize, 2, 4, 8] {
            let p = Par::new(threads);
            let n = 3 * CHUNK_ATOMS;
            let err = p
                .try_map_chunks::<(), usize, _>(
                    n,
                    |s, _| {
                        if s == CHUNK_ATOMS {
                            Err(s)
                        } else {
                            Ok(())
                        }
                    },
                )
                .unwrap_err();
            assert_eq!(err, CHUNK_ATOMS, "threads={threads}: wrong error");
        }
    }
}
