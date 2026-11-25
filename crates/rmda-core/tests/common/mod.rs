//! Shared helpers for the reference-data oracles: per-atom comparisons with
//! OVITO 3.15 and other checks on real MD snapshots (2000 K Al, W and Zr
//! frames, relaxed HEA pairs, an HEA polycrystal; ~240 MB of research data that
//! is not distributed with the code).
//!
//! The tests are `#[ignore]`d: each costs minutes in a debug build and needs
//! the snapshots. `cargo test --release -- --include-ignored` runs them, from
//! `RMDA_REFERENCE_DIR` (default: `reference_structures/` in the checkout);
//! a test whose snapshot is missing prints a skip line and passes.

use std::path::PathBuf;

/// Where the reference snapshots are: `RMDA_REFERENCE_DIR`, else
/// `reference_structures/` at the repository root.
pub fn reference_dir() -> PathBuf {
    match std::env::var_os("RMDA_REFERENCE_DIR") {
        Some(d) => PathBuf::from(d),
        None => PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../reference_structures"),
    }
}
