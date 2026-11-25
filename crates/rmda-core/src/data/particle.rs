//! `ParticleData`: flat SoA core particle arrays (design §7.1).
//!
//! Positions are stored **wrapped** (in-cell); image flags separate; unwrapped
//! positions are computed on demand. Types/ids are fixed-width per §5.

use crate::cell::{Coord3, MdaImage};
use crate::data::property::{DType, Property, PropertyStore};

/// Per-atom integer id (LAMMPS `tagint`).
pub type MdaTag = i64;
/// Per-atom type index.
pub type MdaType = u32;

/// Core SoA particle data.
#[derive(Debug, Clone)]
pub struct ParticleData {
    /// Number of atoms.
    pub natoms: usize,
    /// Wrapped positions (in-cell), AoS.
    pub positions: Vec<Coord3>,
    /// Global atom ids.
    pub ids: Vec<MdaTag>,
    /// Atom types (1-based, LAMMPS style).
    pub types: Vec<MdaType>,
    /// Optional image flags (same length as positions when present).
    pub images: Option<Vec<MdaImage>>,
    /// Optional selection mask (0/1 per atom).
    pub selection: Option<Vec<u8>>,
    /// Analysis properties.
    pub props: PropertyStore,
}

impl ParticleData {
    pub fn new(natoms: usize) -> Self {
        ParticleData {
            natoms,
            positions: Vec::with_capacity(natoms),
            ids: Vec::with_capacity(natoms),
            types: Vec::with_capacity(natoms),
            images: None,
            selection: None,
            props: PropertyStore::new(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.natoms == 0
    }

    /// Pre-allocated, still empty; fill with [`ParticleData::add_atom`].
    pub fn with_capacity(natoms: usize) -> Self {
        let mut p = ParticleData::new(0);
        p.positions.reserve(natoms);
        p.ids.reserve(natoms);
        p.types.reserve(natoms);
        p
    }

    pub fn add_atom(&mut self, pos: Coord3, id: MdaTag, ty: MdaType) {
        self.natoms += 1;
        self.positions.push(pos);
        self.ids.push(id);
        self.types.push(ty);
    }

    /// Ensure a per-atom property of a given dtype exists (resizing to natoms).
    pub fn ensure_property(
        &mut self,
        name: &str,
        dtype: DType,
        components: usize,
    ) -> &mut Property {
        if self.props.get(name).is_none() {
            let mut p = Property::new(name.to_owned(), dtype, true, components);
            p.reserve(self.natoms);
            self.props.register(p);
        }
        self.props.get_mut(name).unwrap()
    }

    /// Add a per-atom property from a computed Vec<f32> (components=1 convenience).
    pub fn add_scalar_f32(&mut self, name: &str, vals: Vec<f32>) {
        let mut p = Property::new(name.to_owned(), DType::F32, true, 1);
        p.reserve(vals.len());
        for v in vals {
            p.push(v as f64);
        }
        self.props.register(p);
    }

    /// Add a per-atom property from a computed Vec<u8>.
    pub fn add_scalar_u8(&mut self, name: &str, vals: Vec<u8>) {
        let mut p = Property::new(name.to_owned(), DType::U8, true, 1);
        p.reserve(vals.len());
        for v in vals {
            p.push(v as f64);
        }
        self.props.register(p);
    }
}

impl Default for ParticleData {
    fn default() -> Self {
        ParticleData::new(0)
    }
}
