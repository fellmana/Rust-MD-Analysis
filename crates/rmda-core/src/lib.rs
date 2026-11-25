//! # RMDA core library
//!
//! Fundaments: cell geometry/PBC/MIC, particle data model, lattice registry,
//! spatial indexing, analysis framework, parallelism, and profiling.
//!
//! Design document: `design.md` in the repository root.

pub mod analysis;
pub mod cell;
pub mod data;
pub mod error;
pub mod exec;
pub mod graph;
pub mod io;
pub mod lattice;
pub mod neighbor;
pub mod prof;
pub mod units;

pub use error::{Error, Result};
pub use units::Units;
