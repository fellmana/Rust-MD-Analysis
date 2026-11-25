//! Typed, named, SoA property store (design §7.2).
//!
//! Mirror of OVITO's `ParticleData`/`Property` but implemented as typed SoA
//! arrays. Names are `Cow<'static, str>`: `'static` for built-ins, owned for
//! user/CLI properties.

use crate::error::{Error, Result};
use std::borrow::Cow;

/// Element type of a property.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DType {
    U8,
    I32,
    I64,
    F32,
    F64,
}

impl DType {
    pub fn size_of(&self) -> usize {
        match self {
            DType::U8 => 1,
            DType::I32 => 4,
            DType::I64 => 8,
            DType::F32 => 4,
            DType::F64 => 8,
        }
    }

    pub fn name(&self) -> &'static str {
        match self {
            DType::U8 => "u8",
            DType::I32 => "i32",
            DType::I64 => "i64",
            DType::F32 => "f32",
            DType::F64 => "f64",
        }
    }
}

/// Typed payload, stored **atom-major**: `data[atom*components + component]`,
/// i.e. one value's components are consecutive. This matches `push_value`,
/// extended XYZ column layout, and every producer and test in the tree.
#[derive(Debug, Clone)]
pub enum TypedVec {
    U8(Vec<u8>),
    I32(Vec<i32>),
    I64(Vec<i64>),
    F32(Vec<f32>),
    F64(Vec<f64>),
}

/// A named property array.
#[derive(Debug, Clone)]
pub struct Property {
    pub name: Cow<'static, str>,
    pub dtype: DType,
    /// Whether this is a per-particle (len == n) or per-frame (len == components) value.
    pub per_particle: bool,
    /// Number of components per value (3 for Position, 1 for StructureType...).
    pub components: usize,
    data: TypedVec,
}

impl Property {
    pub fn new(
        name: impl Into<Cow<'static, str>>,
        dtype: DType,
        per_particle: bool,
        components: usize,
    ) -> Self {
        Property {
            name: name.into(),
            dtype,
            per_particle,
            components,
            data: TypedVec::U8(Vec::new()),
        }
    }

    pub fn with_capacity(
        name: impl Into<Cow<'static, str>>,
        dtype: DType,
        per_particle: bool,
        components: usize,
        n: usize,
    ) -> Self {
        let mut p = Property::new(name, dtype, per_particle, components);
        p.reserve(n);
        p
    }

    /// A per-particle property that takes ownership of finished data
    /// (`components` values per atom, atom-major). The dtype follows the
    /// payload, so a reader hands over its column without a copy.
    pub fn from_data(
        name: impl Into<Cow<'static, str>>,
        components: usize,
        data: TypedVec,
    ) -> Self {
        let dtype = match &data {
            TypedVec::U8(_) => DType::U8,
            TypedVec::I32(_) => DType::I32,
            TypedVec::I64(_) => DType::I64,
            TypedVec::F32(_) => DType::F32,
            TypedVec::F64(_) => DType::F64,
        };
        Property {
            name: name.into(),
            dtype,
            per_particle: true,
            components,
            data,
        }
    }

    /// Reserve space for `n` values (n*components element slots).
    pub fn reserve(&mut self, n: usize) {
        let cap = n.saturating_mul(self.components);
        self.data = match self.dtype {
            DType::U8 => TypedVec::U8(Vec::with_capacity(cap)),
            DType::I32 => TypedVec::I32(Vec::with_capacity(cap)),
            DType::I64 => TypedVec::I64(Vec::with_capacity(cap)),
            DType::F32 => TypedVec::F32(Vec::with_capacity(cap)),
            DType::F64 => TypedVec::F64(Vec::with_capacity(cap)),
        };
    }

    pub fn len(&self) -> usize {
        let raw = match &self.data {
            TypedVec::U8(v) => v.len(),
            TypedVec::I32(v) => v.len(),
            TypedVec::I64(v) => v.len(),
            TypedVec::F32(v) => v.len(),
            TypedVec::F64(v) => v.len(),
        };
        raw.checked_div(self.components).unwrap_or(raw)
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    fn ensure_typed(&mut self) {
        if matches!(&self.data, TypedVec::U8(v) if v.is_empty()) && !matches!(self.dtype, DType::U8)
        {
            self.data = match self.dtype {
                DType::U8 => TypedVec::U8(Vec::new()),
                DType::I32 => TypedVec::I32(Vec::new()),
                DType::I64 => TypedVec::I64(Vec::new()),
                DType::F32 => TypedVec::F32(Vec::new()),
                DType::F64 => TypedVec::F64(Vec::new()),
            };
        }
    }

    /// Push one scalar value (for per_particle, components == 1 this appends one
    /// entry; for components > 1 call per component or use the vector push).
    pub fn push(&mut self, val: f64) {
        self.ensure_typed();
        let data = std::mem::replace(&mut self.data, TypedVec::U8(Vec::new()));
        self.data = match data {
            TypedVec::U8(mut v) => {
                v.push(val as u8);
                TypedVec::U8(v)
            }
            TypedVec::I32(mut v) => {
                v.push(val as i32);
                TypedVec::I32(v)
            }
            TypedVec::I64(mut v) => {
                v.push(val as i64);
                TypedVec::I64(v)
            }
            TypedVec::F32(mut v) => {
                v.push(val as f32);
                TypedVec::F32(v)
            }
            TypedVec::F64(mut v) => {
                v.push(val);
                TypedVec::F64(v)
            }
        };
    }

    /// Push a whole value of `components` scalars.
    pub fn push_value(&mut self, vals: &[f64]) {
        for v in vals {
            self.push(*v);
        }
    }

    /// Get scalar at flat index `flat` (data[flat]).
    pub fn get_flat(&self, flat: usize) -> Option<f64> {
        match &self.data {
            TypedVec::U8(v) => v.get(flat).map(|x| *x as f64),
            TypedVec::I32(v) => v.get(flat).map(|x| *x as f64),
            TypedVec::I64(v) => v.get(flat).map(|x| *x as f64),
            TypedVec::F32(v) => v.get(flat).map(|x| *x as f64),
            TypedVec::F64(v) => v.get(flat).copied(),
        }
    }

    /// Set scalar at flat index.
    pub fn set_flat(&mut self, flat: usize, val: f64) {
        self.ensure_typed();
        let data = std::mem::replace(&mut self.data, TypedVec::U8(Vec::new()));
        self.data = match data {
            TypedVec::U8(mut v) => {
                if let Some(x) = v.get_mut(flat) {
                    *x = val as u8;
                }
                TypedVec::U8(v)
            }
            TypedVec::I32(mut v) => {
                if let Some(x) = v.get_mut(flat) {
                    *x = val as i32;
                }
                TypedVec::I32(v)
            }
            TypedVec::I64(mut v) => {
                if let Some(x) = v.get_mut(flat) {
                    *x = val as i64;
                }
                TypedVec::I64(v)
            }
            TypedVec::F32(mut v) => {
                if let Some(x) = v.get_mut(flat) {
                    *x = val as f32;
                }
                TypedVec::F32(v)
            }
            TypedVec::F64(mut v) => {
                if let Some(x) = v.get_mut(flat) {
                    *x = val;
                }
                TypedVec::F64(v)
            }
        };
    }

    /// Borrow the raw typed vector for fast indexed access.
    pub fn as_f64(&self) -> Option<&Vec<f64>> {
        match &self.data {
            TypedVec::F64(v) => Some(v),
            _ => None,
        }
    }

    pub fn as_i64(&self) -> Option<&Vec<i64>> {
        match &self.data {
            TypedVec::I64(v) => Some(v),
            _ => None,
        }
    }

    pub fn as_u8(&self) -> Option<&Vec<u8>> {
        match &self.data {
            TypedVec::U8(v) => Some(v),
            _ => None,
        }
    }

    pub fn as_i32(&self) -> Option<&Vec<i32>> {
        match &self.data {
            TypedVec::I32(v) => Some(v),
            _ => None,
        }
    }

    pub fn as_f32(&self) -> Option<&Vec<f32>> {
        match &self.data {
            TypedVec::F32(v) => Some(v),
            _ => None,
        }
    }

    pub fn as_mut_f64(&mut self) -> Option<&mut Vec<f64>> {
        match &mut self.data {
            TypedVec::F64(v) => Some(v),
            _ => None,
        }
    }
}

/// A set of named [`Property`] arrays (design §7.2 `PropertyStore`).
#[derive(Debug, Clone, Default)]
pub struct PropertyStore {
    props: Vec<Property>,
}

impl PropertyStore {
    pub fn new() -> Self {
        Default::default()
    }

    pub fn register(&mut self, p: Property) {
        if let Some(existing) = self.props.iter_mut().find(|e| e.name == p.name) {
            *existing = p;
        } else {
            self.props.push(p);
        }
    }

    pub fn get(&self, name: &str) -> Option<&Property> {
        self.props.iter().find(|p| p.name.as_ref() == name)
    }

    pub fn get_mut(&mut self, name: &str) -> Option<&mut Property> {
        self.props.iter_mut().find(|p| p.name.as_ref() == name)
    }

    pub fn ensure(
        &mut self,
        name: &str,
        dtype: DType,
        per_particle: bool,
        components: usize,
    ) -> Result<&Property> {
        self.register(Property::new(
            name.to_owned(),
            dtype,
            per_particle,
            components,
        ));
        self.get(name)
            .ok_or_else(|| Error::invalid("property registration failed"))
    }

    /// Register a completed scalar per-atom array (components=1) from any
    /// numeric source via `Into<f64>`.
    pub fn register_scalar<V: Copy + Into<f64>>(
        &mut self,
        dtype: DType,
        name: &str,
        values: Vec<V>,
    ) {
        let mut p = Property::new(name.to_owned(), dtype, true, 1);
        p.reserve(values.len());
        for v in values {
            p.push(v.into());
        }
        self.register(p);
    }

    /// Register a completed N-component per-atom array from flattened data.
    pub fn register_flat(&mut self, dtype: DType, name: &str, components: usize, values: Vec<f64>) {
        let mut p = Property::with_capacity(
            name.to_owned(),
            dtype,
            true,
            components,
            values.len() / components.max(1),
        );
        for v in values {
            p.push(v);
        }
        self.register(p);
    }

    pub fn iter(&self) -> impl Iterator<Item = &Property> {
        self.props.iter()
    }

    pub fn len(&self) -> usize {
        self.props.len()
    }

    pub fn is_empty(&self) -> bool {
        self.props.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn property_push_get() {
        let mut p = Property::new("StructureType", DType::U8, true, 1);
        p.reserve(4);
        p.push(1.0);
        p.push(2.0);
        p.push(0.0);
        assert_eq!(p.len(), 3);
        assert_eq!(p.get_flat(1), Some(2.0));
        let v = p.as_u8().unwrap();
        assert_eq!(v, &[1, 2, 0]);
    }

    #[test]
    fn store_dedup() {
        let mut s = PropertyStore::new();
        let mut p1 = Property::new("A", DType::F64, true, 1);
        p1.push(5.0);
        s.register(p1);
        let mut p2 = Property::new("A", DType::F64, true, 1);
        p2.push(9.0);
        s.register(p2);
        assert_eq!(s.len(), 1);
        assert_eq!(s.get("A").unwrap().get_flat(0), Some(9.0));
    }
}
