//! Spatial indexing and neighbor finding (design §9).
//!
//! Spatial indexes are built per frame and are immutable: `KnnIndex` (exact
//! k-nearest, design §9.1) and `CellList` (within a cutoff, design §9.2).
//! Every analysis queries them in place — no neighbour table is materialised
//! for the frame (design §9.3).

pub mod cell_list;
pub mod knn;
pub mod nnearest;

pub use cell_list::CellList;
pub use knn::{KnnIndex, KnnScratch};
pub use nnearest::{nearest_k, nearest_k_excluding_self};
