//! Parallelism wrappers (design §10).
//!
//! Data-parallel over atom-index ranges with `rayon`; reductions are keyed by
//! **fixed chunk ordinals** (not rayon thread ids) for determinism.

pub mod checkpoints;
pub mod pool;

pub use checkpoints::{chunk_ordinal, chunk_ranges, fold_chunks, map_chunks, nchunks_for, CHUNK};
pub use pool::{
    ceil_div, configured_threads, default_threads, init_global_pool, num_threads, set_threads, Par,
    ParPool,
};
