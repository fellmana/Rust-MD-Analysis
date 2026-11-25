//! Deterministic parallel reductions (design §10.2).
//!
//! Partial buffers are indexed by **fixed chunk ordinal `c`**, not rayon thread
//! id (thread ids are not a stable 0..T enumeration and depend on pool layout).
//! Combining in ascending chunk order makes output independent of scheduling
//! and, for integer sums, bit-reproducible across any thread count.

/// Number of atoms per fixed chunk. Single source of truth with the pool's
/// chunk size — both refer to the same constant (design §10.2).
pub const CHUNK: usize = super::pool::CHUNK_ATOMS;

/// Ordinal of the chunk owning atom `i`.
#[inline]
pub fn chunk_ordinal(i: usize) -> usize {
    i / CHUNK
}

/// Fixed chunk ranges for `0..n`, in ascending ordinal order.
#[inline]
pub fn chunk_ranges(n: usize) -> Vec<(usize, usize)> {
    let nchunks = super::pool::ceil_div(n, CHUNK).max(1);
    (0..nchunks)
        .map(|c| (c * CHUNK, ((c + 1) * CHUNK).min(n)))
        .filter(|(s, e)| s < e)
        .collect()
}

/// Number of fixed chunks for `n` atoms.
#[inline]
pub fn nchunks_for(n: usize) -> usize {
    super::pool::ceil_div(n, CHUNK).max(1)
}

/// Map every fixed atom chunk to a value, collected **in ascending chunk
/// ordinal order**. Thin wrapper over [`Par::map_chunks`] for the default
/// context; hot paths should use their `ctx.par` so `--threads` is honoured.
pub fn map_chunks<T, F>(n: usize, f: F) -> Vec<T>
where
    T: Send,
    F: Fn(usize, usize) -> T + Sync,
{
    super::pool::Par::default().map_chunks(n, f)
}

/// Fold every fixed atom chunk into a single accumulator, combining the per
/// chunk values **in ascending chunk ordinal order**. Wrapper over
/// [`Par::fold_chunks`] on the default context.
pub fn fold_chunks<T, U, F, C>(n: usize, zero: U, f: F, combine: C) -> U
where
    T: Send,
    U: Clone + Send,
    F: Fn(usize, usize) -> T + Sync,
    C: Fn(&mut U, &T) + Sync,
{
    super::pool::Par::default().fold_chunks(n, zero, f, combine)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn chunk_mapping() {
        assert_eq!(chunk_ordinal(0), 0);
        assert_eq!(chunk_ordinal(CHUNK - 1), 0);
        assert_eq!(chunk_ordinal(CHUNK), 1);
        assert_eq!(nchunks_for(0), 1);
        assert_eq!(nchunks_for(1), 1);
        assert_eq!(nchunks_for(CHUNK), 1);
        assert_eq!(nchunks_for(CHUNK + 1), 2);
    }

    #[test]
    fn ranges_cover_contiguous() {
        let n = 2 * CHUNK + 5;
        let ranges = chunk_ranges(n);
        assert!(ranges.iter().map(|(s, e)| e - s).sum::<usize>() == n);
        assert!(ranges.windows(2).all(|w| w[0].1 == w[1].0));
    }

    #[test]
    fn map_keeps_ordinal_order() {
        let n = 3 * CHUNK + 17;
        let parts = map_chunks(n, |s, _| s);
        assert_eq!(parts, vec![0, CHUNK, 2 * CHUNK, 3 * CHUNK]);
        let parts = map_chunks(3 * CHUNK, |s, _| s);
        assert_eq!(parts, vec![0, CHUNK, 2 * CHUNK]);
    }

    #[test]
    fn histogram_sum_is_order_independent() {
        // Binning by index modulo nbins must be identical regardless of how
        // many chunks (i.e. how much parallelism) was used.
        let nbins = 100usize;
        let expect: Vec<u64> = {
            let mut e = vec![0u64; nbins];
            for i in 0..(2 * CHUNK + 1234) {
                e[i % nbins] += 1;
            }
            e
        };
        let got = fold_chunks(
            2 * CHUNK + 1234,
            vec![0u64; nbins],
            |s, e| {
                let mut local = vec![0u64; nbins];
                for i in s..e {
                    local[i % nbins] += 1;
                }
                local
            },
            |total: &mut Vec<u64>, local: &Vec<u64>| {
                for (t, l) in total.iter_mut().zip(local.iter()) {
                    *t += *l;
                }
            },
        );
        assert_eq!(got, expect);
    }
}
