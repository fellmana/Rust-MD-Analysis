//! Particle data model: property store, SoA particle arrays, frames.
//!
//! Design §7.2: OVITO-inspired typed SoA property model.

pub mod elements;
pub mod frame;
pub mod particle;
pub mod property;
pub mod select;

pub use frame::{Frame, Metadata};
pub use particle::ParticleData;
pub use property::{DType, Property, PropertyStore};
