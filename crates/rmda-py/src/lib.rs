//! Python bindings for RMDA (design §22): the `rmda` module.
//!
//! A thin wrapper over `rmda-core` — every analysis function builds the same
//! `CellList` / `NeighborhoodContext` / `AnalysisContext` the corresponding CLI
//! subcommand builds, so Python and CLI results are identical by construction
//! rather than by re-implementation.
//!
//! Per-atom data crosses the boundary as numpy arrays (design §22): the
//! frame's positions, types and ids are read-only zero-copy views owned by the
//! `Frame`, analysis outputs are typed arrays copied once, and ASE input is read
//! through numpy without per-atom Python objects. Every analysis releases the
//! GIL while it runs.
//!
//! ASE interop (design §22): `read`/`iread` accept an ASE `Atoms` wherever
//! they accept a path, every analysis coerces its frame argument from one, and
//! `from_ase`/`to_ase` make the conversion explicit in both directions.

use numpy::ndarray::{Array2, Array3, ArrayView1, ArrayView2};
use numpy::{
    AllowTypeChange, Element, IntoPyArray, PyArray1, PyArray2, PyArrayLike1, PyArrayLike2,
};
use pyo3::exceptions::{
    PyIOError, PyImportError, PyKeyError, PyRuntimeError, PyTypeError, PyValueError,
};
use pyo3::prelude::*;
use pyo3::types::PyDict;
use pyo3::IntoPyObjectExt;

use rmda_core::analysis::cluster::{ClusterAnalysis, ClusterMode, PairCutoff, TypeKey};
use rmda_core::analysis::cna::{Cna, CnaMode};
use rmda_core::analysis::defects::antisite;
use rmda_core::analysis::defects::clusters as defect_clusters;
use rmda_core::analysis::defects::explicit::AffineMapping as WsAffine;
use rmda_core::analysis::defects::explicit_ws::ExplicitWs;
use rmda_core::analysis::defects::frame::{
    self as defect_frame, AntisiteTypes, FrameResult, FrameSettings, GrainSource, PtmSettings,
    WsReference,
};
use rmda_core::analysis::defects::{self, DefectsConfig, ReferenceMode, Status};
use rmda_core::analysis::grain_boundaries::GrainBoundaries;
use rmda_core::analysis::grain_seg::{self, SegConfig, SegMode, Segmentation};
use rmda_core::analysis::grain_track::Tracker;
use rmda_core::analysis::grains::Grains;
use rmda_core::analysis::ptm::{check, NeighbourOrder, Ptm};
use rmda_core::analysis::rdf::Rdf;
use rmda_core::analysis::warren_cowley::{parse_shells, WarrenCowley};
use rmda_core::analysis::ws::{AffineMapping, WsRef, WsSelf};
use rmda_core::analysis::{Analysis, AnalysisContext, NeighborhoodContext, Summary};
use rmda_core::cell::{Mat3, UnitCell};
use rmda_core::data::particle::ParticleData;
use rmda_core::data::property::{DType, Property, PropertyStore};
use rmda_core::data::select::Selection;
use rmda_core::error::Error as CoreError;
use rmda_core::exec::Par;
use rmda_core::io::xyz::{XyzOrigin, XyzWriter};
use rmda_core::lattice::fit::FitOptions;
use rmda_core::neighbor::cell_list::CellList;

use std::collections::{BTreeMap, HashMap};
use std::sync::{Mutex, MutexGuard};

/// The execution context for one call: the process-wide thread count that
/// `rmda.set_threads` (and the CLI's `--threads`) configure in `rmda_core::exec`.
fn par() -> Par {
    Par::default()
}

fn to_io_py(e: std::io::Error) -> PyErr {
    PyIOError::new_err(e.to_string())
}

fn to_py(e: CoreError) -> PyErr {
    match e {
        CoreError::Io(_) => PyIOError::new_err(e.to_string()),
        // Unusable input (a corrupt or truncated frame, a non-finite
        // coordinate) and an invalid request are bad values.
        CoreError::Input(_) | CoreError::Usage(_) | CoreError::Parse { .. } => {
            PyValueError::new_err(e.to_string())
        }
        _ => PyRuntimeError::new_err(e.to_string()),
    }
}

/// A missing input file is Python's `FileNotFoundError`, before the reader
/// (which reports it as unusable input, exit code 3 in the CLI) sees it.
fn input_path(p: &str) -> PyResult<&std::path::Path> {
    let path = std::path::Path::new(p);
    if !path.exists() {
        return Err(pyo3::exceptions::PyFileNotFoundError::new_err(format!(
            "no such file: {p}"
        )));
    }
    Ok(path)
}

/// Largest value of a cutoff list, or 0.0 when a frame derives no radius.
fn max_cut(v: Vec<f64>) -> f64 {
    v.into_iter().fold(0.0f64, f64::max)
}

/// Per-atom result copied out of a [`PropertyStore`] so a `Frame` can serve it
/// back through `f[name]`. Owned and typed: each column keeps the dtype the
/// analysis wrote, and comes back to Python as a numpy array of that dtype.
enum PropValues {
    U8(Vec<u8>),
    I32(Vec<i32>),
    I64(Vec<i64>),
    F32(Vec<f32>),
    F64(Vec<f64>),
}

/// One attached property: values plus the components-per-atom count the store
/// carried (4 for `StructureOrientation`, 1 for everything else), so the
/// property survives a round trip through `f.write` (design §22) and comes
/// back as an `(N, components)` array.
struct AbsorbedProp {
    values: PropValues,
    components: usize,
}

/// Copy every property an analysis produced onto the frame, so `f["StructureType"]`
/// works after `rmda.cna(f)` the way the design (§22) shows.
///
/// Per-atom properties become `f[name]`. Per-group ones (`per_particle ==
/// false`: one row per grain or per cluster) are never per-atom columns
/// (design §7.3): they go to `f.tables`, grouped by [`table_of`], and each
/// table a store provides replaces that table whole, so a column of an
/// earlier run with another number of rows cannot linger beside it.
fn absorb(src: &PropertyStore, f: &Frame) {
    let mut groups: BTreeMap<String, Vec<(String, AbsorbedProp)>> = BTreeMap::new();
    {
        let mut dst = f.attached();
        for p in src.iter() {
            let key = p.name.to_string();
            let val = match p.dtype {
                DType::U8 => p.as_u8().map(|v| PropValues::U8(v.clone())),
                DType::I32 => p.as_i32().map(|v| PropValues::I32(v.clone())),
                DType::I64 => p.as_i64().map(|v| PropValues::I64(v.clone())),
                DType::F32 => p.as_f32().map(|v| PropValues::F32(v.clone())),
                DType::F64 => p.as_f64().map(|v| PropValues::F64(v.clone())),
            };
            if let Some(values) = val {
                let prop = AbsorbedProp {
                    values,
                    components: p.components,
                };
                if p.per_particle {
                    dst.insert(key, prop);
                } else {
                    groups
                        .entry(table_of(&key).to_string())
                        .or_default()
                        .push((key, prop));
                }
            }
        }
    }
    let mut tables = f.tables();
    for (name, cols) in groups {
        tables.insert(name, Table::Columns(cols));
    }
}

/// The `f.tables` entry a per-group property belongs to: `grains` for the
/// per-grain `Grain*` properties (grains, grain boundaries), `clusters` for
/// the per-cluster `Cluster*` ones, else a table of its own name.
fn table_of(name: &str) -> &str {
    if name.starts_with("Grain") {
        "grains"
    } else if name.starts_with("Cluster") {
        "clusters"
    } else {
        name
    }
}

/// One entry of `f.tables`: per-group columns absorbed from a store, or a
/// table an analysis built as a dict of numpy columns (`segment`, `defects`).
enum Table {
    Columns(Vec<(String, AbsorbedProp)>),
    Dict(Py<PyDict>),
}

impl Table {
    /// A fresh dict of fresh columns, so editing what `f.tables` returned
    /// never edits the frame's copy (as `f[name]` returns a copy).
    fn to_dict<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let d = PyDict::new(py);
        match self {
            Table::Columns(cols) => {
                for (k, v) in cols {
                    d.set_item(k, prop_array(py, v)?)?;
                }
            }
            Table::Dict(src) => {
                for (k, v) in src.bind(py).iter() {
                    d.set_item(k, v.call_method0("copy")?)?;
                }
            }
        }
        Ok(d)
    }
}

/// Attach the tables of an analysis result (`res[name]` for each name, a dict
/// of columns) to `f.tables` as `"{prefix}.{name}"`, first dropping every
/// table under that prefix so a table the last call made and this one did not
/// (`defects.antisites`) does not linger.
fn attach_tables(f: &Frame, res: &Bound<'_, PyDict>, prefix: &str, names: &[&str]) -> PyResult<()> {
    let mut tables = f.tables();
    let dotted = format!("{prefix}.");
    tables.retain(|k, _| !k.starts_with(&dotted));
    for name in names {
        if let Some(t) = res.get_item(name)? {
            let t = t.downcast::<PyDict>()?;
            // A copy, so editing the returned result leaves `f.tables` alone.
            tables.insert(format!("{dotted}{name}"), Table::Dict(t.copy()?.unbind()));
        }
    }
    Ok(())
}

/// Rebuild a [`PropertyStore`] from the props a [`Frame`] carries, so an
/// analysis that reads earlier steps' output (grains reads PTM's columns) and
/// `f.write` get the same per-atom data the frame hands back.
///
/// Columns are registered in sorted name order for deterministic extended-XYZ
/// headers (`io/xyz.rs`); the value push goes through the dtype the
/// prop was absorbed as, which is exact at every magnitude these analyses emit.
fn store_of(props: &HashMap<String, AbsorbedProp>, only: Option<&Vec<String>>) -> PropertyStore {
    let mut names: Vec<&String> = props.keys().collect();
    names.sort();
    let mut out = PropertyStore::new();
    for name in names {
        if let Some(allowed) = only {
            if !allowed.iter().any(|n| n == name) {
                continue;
            }
        }
        let ap = &props[name];
        let dtype = match &ap.values {
            PropValues::U8(_) => DType::U8,
            PropValues::I32(_) => DType::I32,
            PropValues::I64(_) => DType::I64,
            PropValues::F32(_) => DType::F32,
            PropValues::F64(_) => DType::F64,
        };
        let mut p = Property::new(name.clone(), dtype, true, ap.components);
        match &ap.values {
            PropValues::U8(v) => v.iter().for_each(|&x| p.push(x as f64)),
            PropValues::I32(v) => v.iter().for_each(|&x| p.push(x as f64)),
            PropValues::I64(v) => v.iter().for_each(|&x| p.push(x as f64)),
            PropValues::F32(v) => v.iter().for_each(|&x| p.push(x as f64)),
            PropValues::F64(v) => v.iter().for_each(|&x| p.push(x)),
        }
        out.register(p);
    }
    out
}

/// A typed numpy array of `v`: 1-D for a scalar column, `(N, c)` otherwise.
fn shaped_array<T: Element + Copy>(py: Python<'_>, v: &[T], c: usize) -> PyResult<PyObject> {
    if c > 1 {
        Array2::from_shape_vec((v.len() / c, c), v.to_vec())
            .map_err(|e| PyValueError::new_err(e.to_string()))?
            .into_pyarray(py)
            .into_py_any(py)
    } else {
        PyArray1::from_slice(py, v).into_py_any(py)
    }
}

/// A core per-atom [`Property`] as a typed numpy array.
fn core_prop_array(py: Python<'_>, p: &Property) -> PyResult<PyObject> {
    let c = p.components;
    match p.dtype {
        DType::U8 => shaped_array(py, p.as_u8().map_or(&[][..], |v| v), c),
        DType::I32 => shaped_array(py, p.as_i32().map_or(&[][..], |v| v), c),
        DType::I64 => shaped_array(py, p.as_i64().map_or(&[][..], |v| v), c),
        DType::F32 => shaped_array(py, p.as_f32().map_or(&[][..], |v| v), c),
        DType::F64 => shaped_array(py, p.as_f64().map_or(&[][..], |v| v), c),
    }
}

fn prop_array(py: Python<'_>, ap: &AbsorbedProp) -> PyResult<PyObject> {
    let c = ap.components;
    match &ap.values {
        PropValues::U8(v) => shaped_array(py, v, c),
        PropValues::I32(v) => shaped_array(py, v, c),
        PropValues::I64(v) => shaped_array(py, v, c),
        PropValues::F32(v) => shaped_array(py, v, c),
        PropValues::F64(v) => shaped_array(py, v, c),
    }
}

/// `arr` with numpy's `writeable` flag cleared: a view of frame data must not
/// be written through, since the frame is shared by every analysis run on it.
fn read_only<T>(arr: Bound<'_, T>) -> PyResult<Bound<'_, T>> {
    arr.as_any().call_method1("setflags", (false,))?;
    Ok(arr)
}

/// One keyword argument as handed to `f.run("ptm", ...)` / `Pipeline.add(...)`.
/// Values are narrowed to the scalar kinds the analysis entry points take, so
/// the dispatch below never re-parses `PyObject`s.
#[derive(Clone)]
enum Kw {
    None,
    Bool(bool),
    Str(String),
    StrList(Vec<String>),
    I64(i64),
    F64(f64),
    I64List(Vec<i64>),
    F64List(Vec<f64>),
}

/// `**kwargs` -> typed map. Conversion order matters: `bool` must be tried
/// before `i64` (a Python bool *is* an int), and `str` before the numeric
/// types.
fn collect_kwargs(kwargs: Option<&Bound<'_, PyDict>>) -> PyResult<HashMap<String, Kw>> {
    let mut m = HashMap::new();
    let Some(d) = kwargs else {
        return Ok(m);
    };
    for (k, v) in d.iter() {
        let key: String = k.extract()?;
        let val = if v.is_none() {
            Kw::None
        } else if let Ok(b) = v.extract::<bool>() {
            Kw::Bool(b)
        } else if let Ok(s) = v.extract::<String>() {
            Kw::Str(s)
        } else if let Ok(i) = v.extract::<i64>() {
            Kw::I64(i)
        } else if let Ok(f) = v.extract::<f64>() {
            Kw::F64(f)
        } else if let Ok(l) = v.extract::<Vec<String>>() {
            Kw::StrList(l)
        } else if let Ok(l) = v.extract::<Vec<i64>>() {
            Kw::I64List(l)
        } else if let Ok(l) = v.extract::<Vec<f64>>() {
            Kw::F64List(l)
        } else {
            return Err(PyValueError::new_err(format!(
                "unsupported value for `{key}` in **kwargs"
            )));
        };
        m.insert(key, val);
    }
    Ok(m)
}

fn kw_f64(kw: &HashMap<String, Kw>, key: &str, def: f64) -> PyResult<f64> {
    match kw.get(key) {
        None => Ok(def),
        Some(Kw::F64(v)) => Ok(*v),
        Some(Kw::I64(v)) => Ok(*v as f64),
        Some(Kw::Bool(b)) => Ok(if *b { 1.0 } else { 0.0 }),
        Some(_) => Err(PyValueError::new_err(format!("`{key}` must be a number"))),
    }
}

fn kw_opt_f64(kw: &HashMap<String, Kw>, key: &str) -> PyResult<Option<f64>> {
    match kw.get(key) {
        None | Some(Kw::None) => Ok(None),
        Some(Kw::F64(v)) => Ok(Some(*v)),
        Some(Kw::I64(v)) => Ok(Some(*v as f64)),
        Some(_) => Err(PyValueError::new_err(format!("`{key}` must be a number"))),
    }
}

fn kw_required_f64(kw: &HashMap<String, Kw>, key: &str) -> PyResult<f64> {
    kw_opt_f64(kw, key)?
        .ok_or_else(|| PyValueError::new_err(format!("`{key}` is required for this analysis")))
}

fn kw_usize(kw: &HashMap<String, Kw>, key: &str, def: usize) -> PyResult<usize> {
    let v = match kw.get(key) {
        None => return Ok(def),
        Some(Kw::I64(i)) => *i,
        Some(Kw::F64(f)) if *f >= 0.0 => *f as i64,
        Some(_) => {
            return Err(PyValueError::new_err(format!(
                "`{key}` must be a non-negative integer"
            )))
        }
    };
    if v < 0 {
        return Err(PyValueError::new_err(format!(
            "`{key}` must be a non-negative integer"
        )));
    }
    Ok(v as usize)
}

fn kw_opt_usize(kw: &HashMap<String, Kw>, key: &str) -> PyResult<Option<usize>> {
    match kw.get(key) {
        None | Some(Kw::None) => Ok(None),
        Some(Kw::I64(i)) if *i >= 0 => Ok(Some(*i as usize)),
        Some(_) => Err(PyValueError::new_err(format!(
            "`{key}` must be a non-negative integer"
        ))),
    }
}

fn kw_bool(kw: &HashMap<String, Kw>, key: &str, def: bool) -> PyResult<bool> {
    match kw.get(key) {
        None => Ok(def),
        Some(Kw::Bool(b)) => Ok(*b),
        Some(Kw::I64(v)) => Ok(*v != 0),
        Some(_) => Err(PyValueError::new_err(format!("`{key}` must be a bool"))),
    }
}

fn kw_str(kw: &HashMap<String, Kw>, key: &str, def: &str) -> PyResult<String> {
    match kw.get(key) {
        None => Ok(def.to_string()),
        Some(Kw::Str(s)) => Ok(s.clone()),
        Some(_) => Err(PyValueError::new_err(format!("`{key}` must be a string"))),
    }
}

fn kw_str_list(kw: &HashMap<String, Kw>, key: &str) -> PyResult<Vec<String>> {
    match kw.get(key) {
        None => Ok(Vec::new()),
        Some(Kw::StrList(v)) => Ok(v.clone()),
        Some(Kw::Str(s)) => Ok(vec![s.clone()]),
        Some(_) => Err(PyValueError::new_err(format!(
            "`{key}` must be a list of strings"
        ))),
    }
}

fn kw_opt_str(kw: &HashMap<String, Kw>, key: &str) -> PyResult<Option<String>> {
    match kw.get(key) {
        None | Some(Kw::None) => Ok(None),
        Some(Kw::Str(s)) => Ok(Some(s.clone())),
        Some(_) => Err(PyValueError::new_err(format!("`{key}` must be a string"))),
    }
}

fn kw_f64_list(kw: &HashMap<String, Kw>, key: &str) -> PyResult<Option<Vec<f64>>> {
    match kw.get(key) {
        None | Some(Kw::None) => Ok(None),
        Some(Kw::F64List(v)) => Ok(Some(v.clone())),
        Some(Kw::I64List(v)) => Ok(Some(v.iter().map(|&x| x as f64).collect())),
        Some(_) => Err(PyValueError::new_err(format!(
            "`{key}` must be a list of numbers"
        ))),
    }
}

/// A list of numbers or a comma string (`region`, `epsilons` of `void`).
fn kw_floats(kw: &HashMap<String, Kw>, key: &str) -> PyResult<Option<Floats>> {
    match kw.get(key) {
        None | Some(Kw::None) => Ok(None),
        Some(Kw::Str(s)) => Ok(Some(Floats::Text(s.clone()))),
        Some(Kw::F64List(v)) => Ok(Some(Floats::List(v.clone()))),
        Some(Kw::I64List(v)) => Ok(Some(Floats::List(v.iter().map(|&x| x as f64).collect()))),
        Some(_) => Err(PyValueError::new_err(format!(
            "`{key}` must be a list of numbers or a comma string"
        ))),
    }
}

/// `f.run("warren_cowley", shells=[0,12])` / `shells=12` / the CLI's
/// comma string. Rendered to the comma-separated text `parse_shells` expects.
fn kw_shells_text(kw: &HashMap<String, Kw>) -> PyResult<String> {
    match kw.get("shells") {
        None => Ok("0,12".to_string()),
        Some(Kw::Str(s)) => Ok(s.clone()),
        Some(Kw::I64(i)) if *i > 0 => Ok(format!("0,{i}")),
        Some(Kw::I64List(v)) if !v.is_empty() && v.iter().all(|&x| x > 0) => Ok(v
            .iter()
            .map(|x| x.to_string())
            .collect::<Vec<_>>()
            .join(",")),
        Some(_) => Err(PyValueError::new_err(
            "`shells` must be a comma string, a positive int, or a list of positive ints",
        )),
    }
}

/// Every analysis name `f.run` and `Pipeline.add` accept.
const STEPS: &[&str] = &[
    "ptm",
    "cna",
    "rdf",
    "cluster",
    "clusters",
    "warren_cowley",
    "wc",
    "ws",
    "segment",
    "defects",
    "grain_boundaries",
    "void",
];

/// Run one analysis by name on a frame, forwarding the typed kwargs to the
/// module-level function of the same name. Shared by `Frame::run` and
/// `Pipeline`.
fn apply_step(py: Python, f: &Frame, name: &str, kw: &HashMap<String, Kw>) -> PyResult<PyObject> {
    match name {
        "ptm" => ptm(
            py,
            FrameArg::Ref(f),
            kw_f64(kw, "rmsd_cutoff", 0.1)?,
            kw_str_list(kw, "types")?,
            &kw_str(kw, "order", "distance")?,
            kw_str_list(kw, "outputs")?,
        ),
        "cna" => cna(
            py,
            FrameArg::Ref(f),
            &kw_str(kw, "mode", "adaptive")?,
            kw_f64(kw, "cutoff", 3.5)?,
            kw_f64(kw, "low", 0.0)?,
            kw_bool(kw, "full", false)?,
        ),
        "rdf" => rdf(
            py,
            FrameArg::Ref(f),
            kw_required_f64(kw, "rmax")?,
            kw_opt_f64(kw, "dr")?,
            kw_opt_f64(kw, "nbins")?,
            kw_bool(kw, "partials", true)?,
            kw_bool(kw, "coordination", true)?,
            &kw_str(kw, "volume", "auto")?,
            kw_bool(kw, "edge_correct", false)?,
        ),
        "cluster" | "clusters" => cluster(
            py,
            FrameArg::Ref(f),
            kw_opt_f64(kw, "cutoff")?,
            kw_opt_usize(kw, "knear")?,
            kw_usize(kw, "min_size", 1)?,
            kw_opt_str(kw, "select")?.as_deref(),
            kw_bool(kw, "only_selected", false)?,
            kw_bool(kw, "sort_by_size", false)?,
            kw_bool(kw, "unwrap", false)?,
            kw_bool(kw, "properties", false)?,
            // A dict kwarg does not fit the `Kw` scalars: `rmda.cluster` only.
            None,
        ),
        "warren_cowley" | "wc" => warren_cowley_text(
            py,
            f,
            &kw_shells_text(kw)?,
            kw_bool(kw, "edge_correct", false)?,
        ),
        "ws" => ws(
            py,
            FrameArg::Ref(f),
            kw_opt_str(kw, "reference")?.map(WsRefArg::Path),
            &kw_str(kw, "lattice", "fcc")?,
            kw_f64(kw, "nn_distance", 2.86)?,
            kw_bool(kw, "fit_self", false)?,
            &kw_str(kw, "affine_map", "none")?,
            kw_opt_f64(kw, "tolerance")?,
            kw_opt_f64(kw, "cutoff")?,
            kw_f64_list(kw, "orientation")?,
            kw_usize(kw, "search_sample", 100_000)?,
            kw_usize(kw, "void_min_sites", 1)?,
        ),
        "segment" => segment(
            py,
            FrameArg::Ref(f),
            kw_f64(kw, "threshold", 5.0)?,
            kw_f64(kw, "core_angle", 2.0)?,
            kw_usize(kw, "min_size", 100)?,
            kw_bool(kw, "adopt", true)?,
            kw_f64(kw, "rmsd_cutoff", 0.1)?,
            kw_bool(kw, "dhcp", false)?,
            &kw_str(kw, "mode", "crystallite")?,
            kw_opt_str(kw, "ptm_order")?.as_deref(),
            kw_bool(kw, "dhcp_merge_faults", false)?,
        ),
        "defects" => defects_py(
            py,
            FrameArg::Ref(f),
            kw_str_list(kw, "regions")?,
            kw_opt_str(kw, "grain_column")?,
            &kw_str(kw, "reference", "auto")?,
            kw_bool(kw, "segment", false)?,
            kw_f64(kw, "seg_threshold", 5.0)?,
            kw_usize(kw, "seg_min_size", 100)?,
            kw_opt_f64(kw, "boundary_width")?,
            kw_usize(kw, "min_atoms", 100)?,
            kw_f64(kw, "rmsd_cutoff", 0.1)?,
            kw_opt_f64(kw, "patch_size")?,
            kw_opt_f64(kw, "cluster_cutoff")?,
            kw_bool(kw, "dhcp", false)?,
            &kw_str(kw, "seg_mode", "crystallite")?,
            kw_opt_str(kw, "ptm_order")?.as_deref(),
            kw_opt_str(kw, "ws_reference")?.map(WsRefArg::Path),
            &kw_str(kw, "ws_affine", "off")?,
            kw_bool(kw, "antisites", false)?,
            &kw_str(kw, "local_merge", "union")?,
            None,
            kw_bool(kw, "resolve_ambiguous", false)?,
        ),
        "grain_boundaries" => grain_boundaries(
            py,
            FrameArg::Ref(f),
            kw_opt_f64(kw, "w_db")?,
            kw_bool(kw, "unassigned_is_boundary", false)?,
            kw_usize(kw, "min_grain_size", 100)?,
            kw_bool(kw, "no_adopt_orphans", false)?,
        ),
        "void" => void(
            py,
            FrameArg::Ref(f),
            kw_bool(kw, "experimental", false)?,
            &kw_str(kw, "lattice", "fcc")?,
            kw_f64(kw, "nn_distance", 2.86)?,
            kw_opt_str(kw, "reference")?,
            &kw_str(kw, "map", "none")?,
            kw_floats(kw, "region")?,
            kw_usize(kw, "samples", 40_000)?,
            kw_usize(kw, "seed", 11)? as u64,
            kw_f64(kw, "epsilon", 0.01)?,
            kw_floats(kw, "epsilons")?,
            kw_bool(kw, "strain", false)?,
            kw_opt_f64(kw, "strain_spacing")?,
            kw_opt_f64(kw, "strain_sigma")?,
            kw_bool(kw, "excess", false)?,
            kw_bool(kw, "thermal_window", false)?,
            &kw_str(kw, "site_reference", "auto")?,
            kw_usize(kw, "min_sites", 2)?,
            kw_opt_f64(kw, "cutoff")?,
            kw_str_list(kw, "regions")?,
            kw_opt_str(kw, "grain_column")?,
            kw_bool(kw, "segment", false)?,
            None,
        ),
        other => Err(PyValueError::new_err(format!(
            "unknown analysis '{other}' (expected one of {STEPS:?})"
        ))),
    }
}

/// A single frame, wrapping `rmda_core::data::Frame` plus any per-atom
/// properties produced by analyses run on it.
///
/// `frozen`: the core frame is never mutated after construction, which is what
/// lets `positions`/`types`/`ids` hand out zero-copy views of it. Attached
/// properties live behind a mutex so analyses can add to them through `&self`.
#[pyclass(frozen)]
struct Frame {
    inner: rmda_core::data::Frame,
    props: Mutex<HashMap<String, AbsorbedProp>>,
    /// Per-group tables (`f.tables`), by name.
    tables: Mutex<BTreeMap<String, Table>>,
}

impl Frame {
    fn from_core(inner: rmda_core::data::Frame) -> Self {
        Frame {
            inner,
            props: Mutex::new(HashMap::new()),
            tables: Mutex::new(BTreeMap::new()),
        }
    }

    /// The attached tables; poisoning is harmless as for [`Frame::attached`].
    fn tables(&self) -> MutexGuard<'_, BTreeMap<String, Table>> {
        self.tables.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// The input column `name` of the frame itself, per atom.
    fn input_column(&self, name: &str) -> Option<&Property> {
        self.inner
            .particles
            .props
            .get(name)
            .filter(|p| p.per_particle)
    }

    /// The attached properties. A poisoned lock only means an earlier caller
    /// panicked mid-insert of a whole column; the map itself is still valid.
    fn attached(&self) -> MutexGuard<'_, HashMap<String, AbsorbedProp>> {
        self.props.lock().unwrap_or_else(|e| e.into_inner())
    }
}

#[pymethods]
impl Frame {
    #[getter]
    fn natoms(&self) -> usize {
        self.inner.natoms()
    }

    #[getter]
    fn ntypes(&self) -> usize {
        self.inner.ntypes()
    }

    /// The 3x3 cell (row) matrix, matching the core's `h_matrix()`, as a
    /// float64 numpy array.
    #[getter]
    fn cell<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
        Array2::from(self.inner.cell.h_matrix().m.to_vec()).into_pyarray(py)
    }

    /// Per-atom positions: a read-only `(natoms, 3)` float64 view of the
    /// frame's own storage (no copy; `np.array(f.positions)` to own one).
    #[getter]
    fn positions<'py>(slf: &Bound<'py, Self>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let f = slf.get();
        let view = ArrayView2::from(&f.inner.particles.positions[..f.inner.natoms()]);
        // SAFETY: the frame is `frozen` and its core data is never mutated or
        // reallocated after construction; the array keeps `slf` alive as its
        // base object, so the borrowed buffer outlives every view of it.
        let arr = unsafe { PyArray2::borrow_from_array(&view, slf.clone().into_any()) };
        read_only(arr)
    }

    /// Per-atom type ids: a read-only uint32 view (never f64 — design §5).
    #[getter]
    fn types<'py>(slf: &Bound<'py, Self>) -> PyResult<Bound<'py, PyArray1<u32>>> {
        let f = slf.get();
        let view = ArrayView1::from(&f.inner.particles.types[..f.inner.natoms()]);
        // SAFETY: as for `positions`.
        let arr = unsafe { PyArray1::borrow_from_array(&view, slf.clone().into_any()) };
        read_only(arr)
    }

    /// Per-atom tag ids: a read-only int64 view.
    #[getter]
    fn ids<'py>(slf: &Bound<'py, Self>) -> PyResult<Bound<'py, PyArray1<i64>>> {
        let f = slf.get();
        let view = ArrayView1::from(&f.inner.particles.ids[..f.inner.natoms()]);
        // SAFETY: as for `positions`.
        let arr = unsafe { PyArray1::borrow_from_array(&view, slf.clone().into_any()) };
        read_only(arr)
    }

    /// Names of the per-atom properties currently attached to this frame.
    fn properties(&self) -> Vec<String> {
        let mut names: Vec<String> = self.attached().keys().cloned().collect();
        names.sort();
        names
    }

    /// Per-group tables of the analyses run on this frame (design §7.3), as
    /// `{name: {column: numpy array}}` (copies): `grains` (per-grain
    /// properties of `grains`/`grain_boundaries`, by property name, one row
    /// per grain id), `clusters` (per-cluster properties of `cluster`),
    /// `segment.grains`/`segment.pairs`, and `defects.grains`,
    /// `defects.vacancies`, `defects.interstitials`, `defects.clusters` (and
    /// `defects.antisites`) -- the tables those calls also return.
    #[getter(tables)]
    fn tables_py<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let d = PyDict::new(py);
        for (name, t) in self.tables().iter() {
            d.set_item(name, t.to_dict(py)?)?;
        }
        Ok(d)
    }

    /// Names of the per-atom input columns the frame was read or built with
    /// (a file's extra columns, `from_arrays(properties=...)`, an `Atoms`'
    /// extra arrays) -- the columns `grain_column=` and region expressions
    /// read. Attached properties are [`Frame::properties`].
    #[getter]
    fn columns(&self) -> Vec<String> {
        let mut names: Vec<String> = self
            .inner
            .particles
            .props
            .iter()
            .filter(|p| p.per_particle)
            .map(|p| p.name.to_string())
            .collect();
        names.sort();
        names
    }

    fn __len__(&self) -> usize {
        self.inner.natoms()
    }

    fn __contains__(&self, name: &str) -> bool {
        self.attached().contains_key(name) || self.input_column(name).is_some()
    }

    /// A per-atom column as a typed numpy array (a copy): `(N,)` for a
    /// scalar column, `(N, c)` for a `c`-component one. An attached property
    /// (an analysis output, or one set with `f[name] = ...`) first, else an
    /// input column of that name.
    fn __getitem__(&self, py: Python, name: &str) -> PyResult<PyObject> {
        if let Some(p) = self.attached().get(name) {
            return prop_array(py, p);
        }
        match self.input_column(name) {
            Some(p) => core_prop_array(py, p),
            None => Err(PyKeyError::new_err(name.to_string())),
        }
    }

    /// Attach a per-atom column: `f["Grain"] = labels`, then
    /// `rmda.defects(f, grain_column="Grain")`. Any numeric `(N,)` or
    /// `(N, c)` array; it keeps its kind (bool, integer, float32, float64).
    /// It shadows an input column of the same name and is written by
    /// `f.write` and `to_ase` like an analysis output.
    fn __setitem__(&self, name: &str, value: &Bound<'_, PyAny>) -> PyResult<()> {
        let ap = numeric_column(value, self.inner.natoms(), name)?
            .ok_or_else(|| PyTypeError::new_err(format!("column '{name}' must be numeric")))?;
        self.attached().insert(name.to_string(), ap);
        Ok(())
    }

    /// Build a frame from arrays, with the checks a file read gets.
    ///
    /// - `positions`: `(N, 3)`.
    /// - `cell`: 3x3, one cell vector per row; `None` (or all zeros) is the
    ///   box a cell-less file is read in: the atoms' bounding box,
    ///   non-periodic.
    /// - `pbc`, `origin`: three booleans, three floats.
    /// - `types`: per atom, either positive integer type ids or names
    ///   (`["W", "W", "Re", ...]`, coded 1.. in sorted order as the
    ///   extended-XYZ reader codes species); `None` makes every atom type 1.
    /// - `type_names`: names of integer types, a sequence (`type_names[i]`
    ///   names type `i + 1`) or a `{type: name}` dict.
    /// - `ids`: unique integer ids; `None` numbers the atoms 1..N.
    /// - `properties`: `{name: (N,) or (N, c) array}`, input columns as a
    ///   file's extra columns (`grain_column=`, region expressions).
    #[staticmethod]
    #[pyo3(signature = (positions, cell = None, *, pbc = None, origin = None, types = None,
                        type_names = None, ids = None, properties = None))]
    #[allow(clippy::too_many_arguments)]
    fn from_arrays(
        positions: &Bound<'_, PyAny>,
        cell: Option<&Bound<'_, PyAny>>,
        pbc: Option<&Bound<'_, PyAny>>,
        origin: Option<&Bound<'_, PyAny>>,
        types: Option<&Bound<'_, PyAny>>,
        type_names: Option<&Bound<'_, PyAny>>,
        ids: Option<&Bound<'_, PyAny>>,
        properties: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<Frame> {
        let positions = coords_of(positions, "positions")?;
        let n = positions.len();
        let cell = match cell {
            None => {
                if pbc.is_some() || origin.is_some() {
                    return Err(PyValueError::new_err(
                        "pbc and origin need a cell (without one the box is inferred, \
                         non-periodic)",
                    ));
                }
                None
            }
            Some(c) => Some((
                cell_rows_of(c)?,
                match pbc {
                    Some(p) => three_bools(p, "pbc")?,
                    None => [true; 3],
                },
                match origin {
                    Some(o) => three_floats(o, "origin")?,
                    None => [0.0; 3],
                },
            )),
        };
        let (codes, mut labels) = match types {
            None => (vec![1; n], Vec::new()),
            Some(t) => match t.extract::<Vec<String>>() {
                Ok(names) if t.extract::<String>().is_err() => {
                    if type_names.is_some() {
                        return Err(PyValueError::new_err(
                            "type_names only names integer types; these types are names already",
                        ));
                    }
                    codes_of_names(&names)
                }
                _ => (type_codes(t, "types")?, Vec::new()),
            },
        };
        if let Some(names) = type_names {
            apply_type_names(&mut labels, names)?;
        }
        let ids = ids.map(|v| whole_numbers(v, "ids")).transpose()?;
        let mut columns = Vec::new();
        if let Some(props) = properties {
            for (k, v) in props.iter() {
                let name: String = k
                    .extract()
                    .map_err(|_| PyTypeError::new_err("property names must be strings"))?;
                let ap = numeric_column(&v, n, &name)?.ok_or_else(|| {
                    PyTypeError::new_err(format!("property '{name}' must be numeric"))
                })?;
                columns.push((name, ap));
            }
            columns.sort_by(|a, b| a.0.cmp(&b.0));
        }
        build_frame(FrameParts {
            positions,
            cell,
            codes,
            labels,
            ids,
            columns,
        })
    }

    /// The cell origin, three floats.
    #[getter]
    fn origin(&self) -> [f64; 3] {
        self.inner.cell.origin
    }

    /// Periodicity along the three cell vectors.
    #[getter]
    fn pbc(&self) -> [bool; 3] {
        self.inner.cell.pbc
    }

    /// Type names: `type_names[t - 1]` names type `t` ("" when unnamed).
    #[getter]
    fn type_names(&self) -> Vec<String> {
        self.inner.type_labels.clone()
    }

    /// Run a single analysis on this frame, the design §22 `f.run("ptm")`
    /// surface. `kwargs` forward to the module-level function of the same name;
    /// the returned `Stats` dict is what that function returns.
    #[pyo3(signature = (name, **kwargs))]
    fn run(
        &self,
        py: Python,
        name: &str,
        kwargs: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<PyObject> {
        apply_step(py, self, name, &collect_kwargs(kwargs)?)
    }

    /// Write the frame — and any per-atom properties attached to it — as an
    /// extended-XYZ file (design §22 `f.write`). `props` selects a subset of
    /// the attached properties by name; the default writes every one.
    #[pyo3(signature = (path, format = "extxyz", props = None))]
    fn write(
        &self,
        py: Python,
        path: &str,
        format: &str,
        props: Option<Vec<String>>,
    ) -> PyResult<()> {
        if format != "extxyz" {
            return Err(PyValueError::new_err(format!(
                "write only supports format='extxyz', got {format}"
            )));
        }
        let store = store_of(&self.attached(), props.as_ref());
        let inner = &self.inner;
        py.allow_threads(|| -> PyResult<()> {
            let mut w = std::io::BufWriter::new(std::fs::File::create(path).map_err(to_io_py)?);
            XyzWriter::write_frame_with_origin(&mut w, inner, &store, XyzOrigin::Native)
                .map_err(to_py)?;
            std::io::Write::flush(&mut w).map_err(to_io_py)
        })
    }

    fn __repr__(&self) -> String {
        format!(
            "<rmda.Frame natoms={} ntypes={}>",
            self.inner.natoms(),
            self.inner.ntypes()
        )
    }
}

/// The type name used in coercion errors, e.g. `int` for `rmda.read(42)`.
fn type_name_of(ob: &Bound<'_, PyAny>) -> String {
    ob.get_type()
        .name()
        .map(|s| s.to_string())
        .unwrap_or_else(|_| "object".to_string())
}

/// The getter an ASE `Atoms`-shaped object has to provide for
/// [`frame_from_ase`]. Duck-typed: nothing here imports ASE, so anything with
/// the four `Atoms` getters converts.
fn ase_getter<'py>(obj: &Bound<'py, PyAny>, method: &str) -> PyResult<Bound<'py, PyAny>> {
    obj.getattr(method).map_err(|_| {
        PyTypeError::new_err(format!(
            "expected an ASE Atoms-like object (a `.{method}()` method), got {}",
            type_name_of(obj)
        ))
    })
}

/// Does `obj` look like an ASE `Atoms`? Checked before anything treats it as
/// one — and before the generic iteration in `iread`, because an `Atoms` is
/// itself iterable (over its `Atom`s, which are not frames).
fn is_ase_like(obj: &Bound<'_, PyAny>) -> bool {
    obj.hasattr("get_positions").unwrap_or(false)
}

/// `(n, 3)` rows of f64 from anything numpy can turn into a float64 array —
/// an ndarray of any numeric dtype passes through without per-element Python
/// work. A wrong-shaped position or cell array is a mistake to report, not to
/// squeeze into place.
fn coords_of(ob: &Bound<'_, PyAny>, what: &str) -> PyResult<Vec<[f64; 3]>> {
    let arr: PyArrayLike2<'_, f64, AllowTypeChange> = ob.extract().map_err(|_| {
        PyTypeError::new_err(format!(
            "{what} must be an (n, 3) array of floats, got {}",
            type_name_of(ob)
        ))
    })?;
    let a = arr.as_array();
    if a.ncols() != 3 {
        return Err(PyValueError::new_err(format!(
            "{what}: expected shape (n, 3), got {:?}",
            a.shape()
        )));
    }
    Ok(a.rows().into_iter().map(|r| [r[0], r[1], r[2]]).collect())
}

/// Per-atom type codes (1-based into the returned labels) and the sorted set of
/// chemical symbols present — the codes the extended-XYZ reader assigns when
/// there is no `type` column.
///
/// Through `get_atomic_numbers()` and `ase.data.chemical_symbols` when both
/// exist, so only the distinct elements are looked up in Python; otherwise
/// through `get_chemical_symbols()`, which any duck-typed `Atoms` provides.
fn species_of(atoms: &Bound<'_, PyAny>, natoms: usize) -> PyResult<(Vec<u32>, Vec<String>)> {
    let py = atoms.py();
    if let (Ok(getter), Ok(data)) = (
        atoms.getattr("get_atomic_numbers"),
        PyModule::import(py, "ase.data"),
    ) {
        let numbers: PyArrayLike1<'_, i64, AllowTypeChange> =
            getter.call0()?.extract().map_err(|_| {
                PyTypeError::new_err("get_atomic_numbers() must yield an array of integers")
            })?;
        let numbers = numbers.as_array();
        if numbers.len() != natoms {
            return Err(PyValueError::new_err(format!(
                "get_atomic_numbers() gave {} numbers for {natoms} positions",
                numbers.len()
            )));
        }
        let mut zs: Vec<i64> = numbers.iter().copied().collect();
        zs.sort_unstable();
        zs.dedup();
        let table = data.getattr("chemical_symbols")?;
        let symbols: Vec<String> = zs
            .iter()
            .map(|&z| table.get_item(z)?.extract::<String>())
            .collect::<PyResult<_>>()?;
        let mut labels = symbols.clone();
        labels.sort();
        labels.dedup();
        let code: HashMap<i64, u32> = zs
            .iter()
            .zip(&symbols)
            .map(|(&z, s)| (z, labels.binary_search(s).map_or(0, |i| i as u32 + 1)))
            .collect();
        return Ok((numbers.iter().map(|z| code[z]).collect(), labels));
    }

    let symbols: Vec<String> = ase_getter(atoms, "get_chemical_symbols")?
        .call0()?
        .extract()
        .map_err(|_| PyTypeError::new_err("get_chemical_symbols() must yield a list of strings"))?;
    if symbols.len() != natoms {
        return Err(PyValueError::new_err(format!(
            "get_chemical_symbols() gave {} symbols for {natoms} positions",
            symbols.len()
        )));
    }
    let mut labels = symbols.clone();
    labels.sort();
    labels.dedup();
    let type_map: HashMap<&str, u32> = labels
        .iter()
        .enumerate()
        .map(|(i, s)| (s.as_str(), i as u32 + 1))
        .collect();
    Ok((
        symbols.iter().map(|s| type_map[s.as_str()]).collect(),
        labels,
    ))
}

/// A cell as it arrives from Python: rows (cell vectors), pbc, origin.
type CellParts = ([[f64; 3]; 3], [bool; 3], [f64; 3]);

/// Everything a frame is built from, however it arrived: plain arrays
/// (`Frame.from_arrays`), an ASE `Atoms` (`from_ase`) or an OVITO data
/// collection (`rmda.ovito`, through `from_arrays`). [`build_frame`] turns it
/// into a frame by one set of rules, so the three routes cannot drift apart.
struct FrameParts {
    positions: Vec<[f64; 3]>,
    /// Cell rows, pbc, origin; `None` or an all-zero matrix is the inferred
    /// box of a cell-less file (the atoms' bounding box, non-periodic).
    cell: Option<CellParts>,
    /// 1-based type code per atom.
    codes: Vec<u32>,
    /// `labels[t - 1]` names type `t` ("" when unnamed).
    labels: Vec<String>,
    /// Atom ids; `None` numbers them 1..n.
    ids: Option<Vec<i64>>,
    /// Further per-atom input columns, as a file's extra columns.
    columns: Vec<(String, AbsorbedProp)>,
}

/// A core [`Property`] (per atom) holding `ap`'s values in its own dtype.
fn core_property(name: &str, ap: &AbsorbedProp) -> Property {
    let (dtype, vals): (DType, Vec<f64>) = match &ap.values {
        PropValues::U8(v) => (DType::U8, v.iter().map(|&x| x as f64).collect()),
        PropValues::I32(v) => (DType::I32, v.iter().map(|&x| x as f64).collect()),
        PropValues::I64(v) => (DType::I64, v.iter().map(|&x| x as f64).collect()),
        PropValues::F32(v) => (DType::F32, v.iter().map(|&x| x as f64).collect()),
        PropValues::F64(v) => (DType::F64, v.clone()),
    };
    let mut p = Property::with_capacity(
        name.to_owned(),
        dtype,
        true,
        ap.components,
        vals.len() / ap.components.max(1),
    );
    vals.into_iter().for_each(|x| p.push(x));
    p
}

/// Check and assemble [`FrameParts`] into a frame: the lengths agree, ids are
/// unique, and the result passes the checks every file read passes
/// (`Frame::check_input`: finite coordinates, a box with volume).
fn build_frame(p: FrameParts) -> PyResult<Frame> {
    let n = p.positions.len();
    if p.codes.len() != n {
        return Err(PyValueError::new_err(format!(
            "{} types for {n} positions",
            p.codes.len()
        )));
    }
    let ids = match p.ids {
        None => (1..=n as i64).collect(),
        Some(ids) => {
            if ids.len() != n {
                return Err(PyValueError::new_err(format!(
                    "{} ids for {n} positions",
                    ids.len()
                )));
            }
            let mut sorted = ids.clone();
            sorted.sort_unstable();
            if let Some(w) = sorted.windows(2).find(|w| w[0] == w[1]) {
                return Err(PyValueError::new_err(format!(
                    "atom ids are not unique: id {} appears more than once",
                    w[0]
                )));
            }
            ids
        }
    };
    let mut parts = ParticleData::with_capacity(n);
    for ((pos, &id), &ty) in p.positions.iter().zip(&ids).zip(&p.codes) {
        parts.add_atom(*pos, id, ty);
    }
    for (name, ap) in &p.columns {
        parts.props.register(core_property(name, ap));
    }
    let cell = match p.cell {
        Some((rows, pbc, origin)) if rows.iter().any(|r| r.iter().any(|&x| x != 0.0)) => {
            UnitCell::from_matrix(Mat3::from_rows(rows[0], rows[1], rows[2]), pbc, origin)
        }
        // No cell of its own: the box the reader gives a cell-less file, which
        // is the atoms' own bounding box, cut to them and non-periodic -- so
        // the numbers in it mean something, where a 1e30 cube has a density
        // of 1e-90.
        _ => rmda_core::io::xyz::infer_cell(&p.positions),
    };
    let mut inner = rmda_core::data::Frame::new(cell, parts);
    inner.type_labels = p.labels;
    inner.check_input().map_err(to_py)?;
    Ok(Frame::from_core(inner))
}

/// A per-atom numeric column from anything numpy can turn into an `(n,)` or
/// `(n, c)` array, keeping its kind: bool as u8, a signed integer of up to 32 bits as i32, any other integer as i64, float32
/// as f32, other floats as f64. `Ok(None)` when it is not numeric at all
/// (strings, objects), for callers that skip such columns.
fn numeric_column(ob: &Bound<'_, PyAny>, n: usize, name: &str) -> PyResult<Option<AbsorbedProp>> {
    let py = ob.py();
    let arr = PyModule::import(py, "numpy")?.call_method1("asarray", (ob,))?;
    let dtype = arr.getattr("dtype")?;
    let kind: String = dtype.getattr("kind")?.extract()?;
    if !matches!(kind.as_str(), "b" | "i" | "u" | "f") {
        return Ok(None);
    }
    let shape: Vec<usize> = arr.getattr("shape")?.extract()?;
    let components = match shape.as_slice() {
        [m] if *m == n => 1,
        [m, c] if *m == n && *c >= 1 => *c,
        _ => {
            return Err(PyValueError::new_err(format!(
                "column '{name}': expected shape ({n},) or ({n}, c), got {shape:?}"
            )))
        }
    };
    // C order: one atom's components consecutive, the store's layout.
    let flat = arr.call_method0("ravel")?;
    let values = match kind.as_str() {
        "b" => {
            let v: PyArrayLike1<'_, u8, AllowTypeChange> = flat.extract()?;
            PropValues::U8(v.as_array().to_vec())
        }
        "i" | "u" if kind == "i" && dtype.getattr("itemsize")?.extract::<usize>()? <= 4 => {
            let v: PyArrayLike1<'_, i32, AllowTypeChange> = flat.extract()?;
            PropValues::I32(v.as_array().to_vec())
        }
        "i" | "u" => {
            let v: PyArrayLike1<'_, i64, AllowTypeChange> = flat.extract()?;
            PropValues::I64(v.as_array().to_vec())
        }
        _ if dtype.getattr("itemsize")?.extract::<usize>()? == 4 => {
            let v: PyArrayLike1<'_, f32, AllowTypeChange> = flat.extract()?;
            PropValues::F32(v.as_array().to_vec())
        }
        _ => {
            let v: PyArrayLike1<'_, f64, AllowTypeChange> = flat.extract()?;
            PropValues::F64(v.as_array().to_vec())
        }
    };
    Ok(Some(AbsorbedProp { values, components }))
}

/// Integers given as any numeric array, checked to be whole numbers the way
/// the readers check a `type` or `id` column (`1.5` is an error, not `1`).
fn whole_numbers(ob: &Bound<'_, PyAny>, what: &str) -> PyResult<Vec<i64>> {
    let v: PyArrayLike1<'_, f64, AllowTypeChange> = ob
        .extract()
        .map_err(|_| PyTypeError::new_err(format!("{what} must be a sequence of integers")))?;
    v.as_array()
        .iter()
        .map(|&x| {
            if x.is_finite() && x.fract() == 0.0 && x.abs() <= 9.007_199_254_740_992e15 {
                Ok(x as i64)
            } else {
                Err(PyValueError::new_err(format!(
                    "{what} holds {x}, not an integer"
                )))
            }
        })
        .collect()
}

/// Type codes from declared integer types: each must be ≥ 1.
fn type_codes(ob: &Bound<'_, PyAny>, what: &str) -> PyResult<Vec<u32>> {
    whole_numbers(ob, what)?
        .into_iter()
        .map(|t| {
            if (1..=u32::MAX as i64).contains(&t) {
                Ok(t as u32)
            } else {
                Err(PyValueError::new_err(format!(
                    "{what} holds {t}, not a positive integer"
                )))
            }
        })
        .collect()
}

/// Codes 1.. for the sorted set of names, and that set: the codes the
/// extended-XYZ reader assigns from a species column.
fn codes_of_names(names: &[String]) -> (Vec<u32>, Vec<String>) {
    let mut labels = names.to_vec();
    labels.sort();
    labels.dedup();
    let code: HashMap<&str, u32> = labels
        .iter()
        .enumerate()
        .map(|(i, s)| (s.as_str(), i as u32 + 1))
        .collect();
    (names.iter().map(|s| code[s.as_str()]).collect(), labels)
}

/// Apply type names given as a sequence (`names[i]` names type `i + 1`) or a
/// mapping `{type: name}` to `labels`, growing it to the largest type named.
fn apply_type_names(labels: &mut Vec<String>, names: &Bound<'_, PyAny>) -> PyResult<()> {
    let mut set = |t: i64, name: String| -> PyResult<()> {
        if t < 1 || t > u32::MAX as i64 {
            return Err(PyValueError::new_err(format!(
                "type_names: {t} is not a type (types are 1-based)"
            )));
        }
        let i = t as usize - 1;
        if labels.len() <= i {
            labels.resize(i + 1, String::new());
        }
        labels[i] = name;
        Ok(())
    };
    if let Ok(d) = names.downcast::<PyDict>() {
        for (k, v) in d.iter() {
            set(k.extract()?, v.extract()?)?;
        }
    } else {
        let list: Vec<String> = names.extract().map_err(|_| {
            PyTypeError::new_err("type_names must be a sequence of names or a {type: name} dict")
        })?;
        for (i, name) in list.into_iter().enumerate() {
            set(i as i64 + 1, name)?;
        }
    }
    Ok(())
}

/// `[[f64; 3]; 3]` rows of a cell given as anything numpy reads as 3x3.
fn cell_rows_of(ob: &Bound<'_, PyAny>) -> PyResult<[[f64; 3]; 3]> {
    let rows = coords_of(ob, "cell")?;
    if rows.len() != 3 {
        return Err(PyValueError::new_err(format!(
            "cell must be 3x3, got {} rows",
            rows.len()
        )));
    }
    Ok([rows[0], rows[1], rows[2]])
}

/// Three booleans from any iterable of truthy values (a numpy bool array's
/// elements are `numpy.bool_`, not Python `bool`).
fn three_bools(ob: &Bound<'_, PyAny>, what: &str) -> PyResult<[bool; 3]> {
    let bad = || PyTypeError::new_err(format!("{what} must be three booleans"));
    let items: Vec<Bound<'_, PyAny>> = ob
        .try_iter()
        .map_err(|_| bad())?
        .collect::<PyResult<_>>()
        .map_err(|_| bad())?;
    if items.len() != 3 {
        return Err(PyValueError::new_err(format!(
            "{what} must be three booleans, got {}",
            items.len()
        )));
    }
    let mut out = [false; 3];
    for (i, it) in items.iter().enumerate() {
        out[i] = it.is_truthy().map_err(|_| bad())?;
    }
    Ok(out)
}

/// Three floats (an origin).
fn three_floats(ob: &Bound<'_, PyAny>, what: &str) -> PyResult<[f64; 3]> {
    let v: Vec<f64> = ob
        .extract()
        .map_err(|_| PyValueError::new_err(format!("{what} must be three floats")))?;
    <[f64; 3]>::try_from(v.as_slice())
        .map_err(|_| PyValueError::new_err(format!("{what} must be three floats, got {}", v.len())))
}

/// ASE array names that are the frame itself, not extra columns.
const ASE_RESERVED: [&str; 4] = ["positions", "numbers", "type", "id"];

/// Build a frame from an ASE `Atoms`-shaped object — the conversion behind
/// `rmda.from_ase`, the polymorphic `rmda.read`/`rmda.iread`, and every analysis
/// that accepts an `Atoms` in place of a `Frame`.
///
/// The conventions mirror the extended-XYZ reader so that
/// `rmda.read(ase.io.read(path))` is field-for-field `rmda.read(path)`: type
/// codes come from the sorted symbols present unless a declared `type` array
/// wins (as the file column does), ids come from `arrays["id"]` when present
/// (else 1..n), every other numeric array is an input column, a zero cell
/// becomes the reader's *inferred* box -- the atoms' own bounding box,
/// non-periodic -- and `atoms.info["cell_origin"]` carries the cell origin
/// through (the field OVITO's `ase_to_ovito` uses too). `info["type_names"]`
/// (`{type: name}`, written by `to_ase` when a name is not an element) names
/// declared types.
///
/// Arrays are read through numpy; nothing here loops over atoms in Python.
fn frame_from_ase(atoms: &Bound<'_, PyAny>) -> PyResult<Frame> {
    let positions = coords_of(&ase_getter(atoms, "get_positions")?.call0()?, "positions")?;
    let n = positions.len();

    let cell_ob = ase_getter(atoms, "get_cell")?.call0()?;
    // `ase.cell.Cell` keeps its matrix in `.array`; a bare 3x3 (ndarray or a
    // list of lists) passes through untouched.
    let cell_src = cell_ob.getattr("array").unwrap_or(cell_ob);
    let rows = cell_rows_of(&cell_src)?;
    let pbc = three_bools(&ase_getter(atoms, "get_pbc")?.call0()?, "get_pbc()")?;

    let info = atoms.getattr("info").ok();
    let info_get = |key: &str| {
        info.as_ref()
            .and_then(|i| i.call_method1("get", (key,)).ok())
            .filter(|v| !v.is_none())
    };
    // `info["cell_origin"]`, when present, offsets the cell the way a
    // 12-number Lattice= does.
    let origin = match info_get("cell_origin") {
        Some(v) => three_floats(&v, "info['cell_origin']")?,
        None => [0.0; 3],
    };

    let arrays = atoms.getattr("arrays").ok();
    let array = |key: &str| {
        arrays
            .as_ref()
            .and_then(|a| a.call_method1("get", (key,)).ok())
            .filter(|v| !v.is_none())
    };

    // Type codes: the sorted set of symbols present, as the reader assigns
    // when there is no `type` column...
    let (mut codes, mut labels) = species_of(atoms, n)?;
    // ...unless a declared `type` array wins, exactly as the column does.
    if let Some(v) = array("type") {
        let declared = type_codes(&v, "arrays['type']")?;
        if declared.len() != n {
            return Err(PyValueError::new_err(format!(
                "arrays['type'] has {} entries for {n} positions",
                declared.len()
            )));
        }
        // The labels then follow the reader's rule for a `type` column
        // without `species=`: each type its atoms' symbol, when it owns one.
        let symbols: Vec<&str> = codes
            .iter()
            .map(|&c| labels.get(c as usize - 1).map_or("", String::as_str))
            .collect();
        labels = rmda_core::io::xyz::labels_for_typed_symbols(&declared, &symbols);
        codes = declared;
        if let Some(names) = info_get("type_names") {
            apply_type_names(&mut labels, &names)?;
        }
    }

    let ids = array("id")
        .map(|v| whole_numbers(&v, "arrays['id']"))
        .transpose()?;

    let mut columns = Vec::new();
    if let Some(arrays) = &arrays {
        let d = arrays.downcast::<PyDict>()?;
        let mut keys: Vec<String> = d
            .keys()
            .iter()
            .filter_map(|k| k.extract::<String>().ok())
            .filter(|k| !ASE_RESERVED.contains(&k.as_str()))
            .collect();
        keys.sort();
        for k in keys {
            if let Some(v) = d.get_item(&k)? {
                if let Some(ap) = numeric_column(&v, n, &k)? {
                    columns.push((k, ap));
                }
            }
        }
    }

    build_frame(FrameParts {
        positions,
        cell: Some((rows, pbc, origin)),
        codes,
        labels,
        ids,
        columns,
    })
}

/// Convert an ASE `Atoms` into a frame (design §22). `rmda.read(atoms)`
/// does the same thing; this spelling makes the conversion explicit when a
/// script holds both an `Atoms` and a path.
#[pyfunction]
fn from_ase(atoms: &Bound<'_, PyAny>) -> PyResult<Frame> {
    frame_from_ase(atoms)
}

/// What an analysis function takes in place of the frame: a `Frame` pulled
/// out of the Python argument, an internal borrow (the `apply_step` path
/// already holds one), or an ASE `Atoms` converted on the spot — so
/// `rmda.rdf(atoms, rmax=6.0)` works wherever `rmda.rdf(rmda.read(atoms), ...)`
/// would, without every function re-checking the argument's type.
enum FrameArg<'py> {
    Borrowed(PyRef<'py, Frame>),
    Ref(&'py Frame),
    Owned(Box<Frame>),
}

impl<'py> FromPyObject<'py> for FrameArg<'py> {
    fn extract_bound(ob: &Bound<'py, PyAny>) -> PyResult<Self> {
        if let Ok(f) = ob.extract::<PyRef<'py, Frame>>() {
            return Ok(FrameArg::Borrowed(f));
        }
        if is_ase_like(ob) {
            return Ok(FrameArg::Owned(Box::new(frame_from_ase(ob)?)));
        }
        Err(PyTypeError::new_err(format!(
            "expected an rmda.Frame or an ASE Atoms object, got {}",
            type_name_of(ob)
        )))
    }
}

impl<'py> std::ops::Deref for FrameArg<'py> {
    type Target = Frame;
    fn deref(&self) -> &Frame {
        match self {
            FrameArg::Borrowed(f) => f,
            FrameArg::Ref(f) => f,
            FrameArg::Owned(f) => f,
        }
    }
}

/// Read the first frame of `path` (format autodetected), as `read_frame`
/// does — or convert an ASE `Atoms` directly, so `rmda.read(atoms)` works
/// wherever `rmda.read("file.xyz")` does (design §22).
#[pyfunction]
fn read(py: Python, path: &Bound<'_, PyAny>) -> PyResult<Frame> {
    if let Ok(p) = path.extract::<String>() {
        let path = input_path(&p)?;
        let f = py
            .allow_threads(|| rmda_core::io::read_frame(path))
            .map_err(to_py)?;
        return Ok(Frame::from_core(f));
    }
    if is_ase_like(path) {
        return frame_from_ase(path);
    }
    Err(PyTypeError::new_err(
        "read() expects a file path (str) or an ASE Atoms object",
    ))
}

/// The frame stream behind a `PyFrameIter`: the core's file reader, a Python
/// iterator of ASE `Atoms`-like objects (the stream `ase.io.iread` yields),
/// or one such object converted to a single frame.
enum FrameStream {
    Core(rmda_core::io::FrameIter),
    Objects(Py<PyAny>),
    One(Option<Box<Frame>>),
}

/// Streaming iterator over frames (design §22 `iread`): of `path`, or of
/// an ASE `Atoms`-like iterable's members, or the one frame an `Atoms`
/// itself holds. `unsendable`: the file cursor belongs to the thread that
/// opened it.
#[pyclass(unsendable)]
struct PyFrameIter {
    stream: FrameStream,
}

#[pymethods]
impl PyFrameIter {
    fn __iter__(slf: PyRef<Self>) -> PyRef<Self> {
        slf
    }

    fn __next__(&mut self, py: Python) -> PyResult<Option<Frame>> {
        match &mut self.stream {
            FrameStream::Core(it) => match it.next() {
                None => Ok(None),
                Some(Ok(f)) => Ok(Some(Frame::from_core(f))),
                Some(Err(e)) => Err(to_py(e)),
            },
            FrameStream::Objects(obj) => {
                // Re-`iter()`ing an iterator yields itself (the language
                // requires it of every conformant iterator), so the stored
                // object keeps its position between calls, and the C-API
                // iterator behind `next` folds StopIteration into `None`.
                let mut it = obj.bind(py).try_iter()?;
                match it.next() {
                    None => Ok(None),
                    Some(Err(e)) => Err(e),
                    Some(Ok(item)) => Ok(Some(frame_from_ase(&item)?)),
                }
            }
            FrameStream::One(slot) => Ok(slot.take().map(|f| *f)),
        }
    }
}

/// Streaming iterator over the frames of `path` (format autodetected), or —
/// when given an ASE `Atoms` — a one-frame stream, or — when given any
/// iterable of such objects, e.g. `ase.io.iread(...)` — a stream over them
/// (design §22).
#[pyfunction]
fn iread(py: Python, path: &Bound<'_, PyAny>) -> PyResult<Py<PyFrameIter>> {
    let stream = if let Ok(p) = path.extract::<String>() {
        let r = rmda_core::io::FrameReader::open(input_path(&p)?).map_err(to_py)?;
        FrameStream::Core(r.into_frames().map_err(to_py)?)
    } else if is_ase_like(path) {
        FrameStream::One(Some(Box::new(frame_from_ase(path)?)))
    } else if let Ok(it) = path.try_iter() {
        FrameStream::Objects(it.into_any().unbind())
    } else {
        return Err(PyTypeError::new_err(
            "iread() expects a file path (str), an ASE Atoms object, \
             or an iterable of ASE Atoms objects",
        ));
    };
    Py::new(py, PyFrameIter { stream })
}

/// Set the thread count used by analyses (0 = the job's CPUs under SLURM,
/// else all cores). This is the same process-wide setting the CLI's
/// `--threads` writes in `rmda_core::exec`.
#[pyfunction]
fn set_threads(n: usize) {
    let n = if n == 0 {
        rmda_core::exec::default_threads()
    } else {
        n
    };
    rmda_core::exec::set_threads(n);
}

/// The thread count analyses currently run with.
#[pyfunction]
fn get_threads() -> usize {
    rmda_core::exec::configured_threads()
}

/// The `Stats` dict the CLI prints, as design §22 requires: every scalar the
/// summary carries, plus `histograms` whenever the analysis produced a
/// distribution (only RDF and cluster do), so a command that reports a curve in
/// the terminal is not silent about it in Python.
fn summary_dict<'py>(py: Python<'py>, sum: &Summary) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    for (k, v) in &sum.scalars {
        d.set_item(k, *v)?;
    }
    if !sum.histograms.is_empty() {
        let hist = PyDict::new(py);
        for (k, series) in &sum.histograms {
            hist.set_item(k, series.clone())?;
        }
        d.set_item("histograms", hist)?;
    }
    Ok(d)
}

/// Run `a` on `f` with the GIL released, into `out`. `radius > 0` builds the
/// cell list the analysis reads its neighbours from (inside the released
/// section too: on a large frame that build is a large part of the cost).
fn run_released<A: Analysis + Sync>(
    py: Python,
    f: &Frame,
    a: &A,
    radius: f64,
    out: &mut PropertyStore,
) -> PyResult<Summary> {
    let inner = &f.inner;
    let par = par();
    py.allow_threads(|| {
        let cl = if radius > 0.0 {
            Some(CellList::build(
                &inner.cell,
                &inner.particles.positions,
                radius,
            )?)
        } else {
            None
        };
        let nk = NeighborhoodContext { cell: cl.as_ref() };
        let ctx = AnalysisContext {
            frame: inner,
            nkeys: &nk,
            par: &par,
        };
        let mut sum = Summary::default();
        a.run(&ctx, out, &mut sum)?;
        Ok(sum)
    })
    .map_err(to_py)
}

/// [`run_released`] into a fresh store, then onto the frame: the shape of
/// every single-step analysis below.
fn run_absorbed<A: Analysis + Sync>(
    py: Python,
    f: &Frame,
    a: &A,
    radius: f64,
) -> PyResult<Summary> {
    let mut out = PropertyStore::new();
    let sum = run_released(py, f, a, radius, &mut out)?;
    absorb(&out, f);
    Ok(sum)
}

/// Common-neighbour analysis (design §13); writes `f["StructureType"]`.
#[pyfunction]
#[pyo3(signature = (f, mode = "adaptive", cutoff = 3.5, low = 0.0, full = false))]
fn cna(
    py: Python,
    f: FrameArg<'_>,
    mode: &str,
    cutoff: f64,
    low: f64,
    full: bool,
) -> PyResult<PyObject> {
    let f: &Frame = &f;
    let m = match mode {
        "adaptive" => CnaMode::Adaptive,
        "fixed" => CnaMode::Fixed { cutoff },
        "interval" => CnaMode::Interval,
        "bond" => CnaMode::Bond { cutoff },
        "shell" => CnaMode::Shell { low, high: cutoff },
        "diamond" => CnaMode::Diamond,
        other => {
            return Err(PyValueError::new_err(format!(
                "unknown cna mode '{other}' (expected adaptive|fixed|interval|bond|shell|diamond)"
            )))
        }
    };
    let a = Cna::new(m).with_full(full);
    // CNA builds its own k-NN index.
    let sum = run_absorbed(py, f, &a, 0.0)?;
    summary_dict(py, &sum)?.into_py_any(py)
}

/// The eight PTM structure selectors `rmda ptm --types` accepts, in the CLI's
/// spelling, mapped to the same `check` flags the CLI maps them to.
const PTM_TYPES: [(&str, u32); 8] = [
    ("fcc", check::FCC),
    ("hcp", check::HCP),
    ("bcc", check::BCC),
    ("ico", check::ICO),
    ("sc", check::SC),
    ("dcub", check::DCUB),
    ("dhex", check::DHEX),
    ("graphene", check::GRAPHENE),
];

/// The flag mask a `types` selector means. No selector is OVITO's default set,
/// matching `rmda ptm`'s `default_value = "fcc,hcp,bcc"`.
fn ptm_flags(types: &[String]) -> PyResult<u32> {
    if types.is_empty() {
        return Ok(check::OVITO_DEFAULT);
    }
    let mut flags = 0u32;
    for t in types {
        match PTM_TYPES.iter().find(|(name, _)| name == t) {
            Some((_, f)) => flags |= f,
            None => {
                return Err(PyValueError::new_err(format!(
                    "unknown structure type '{t}' (expected one of \
                     fcc, hcp, bcc, ico, sc, dcub, dhex, graphene)"
                )))
            }
        }
    }
    Ok(flags)
}

/// [`Ptm`] run with a chosen neighbour order (`rmda ptm --ptm-order`).
struct PtmOrdered {
    ptm: Ptm,
    order: rmda_core::analysis::ptm::NeighbourOrder,
    /// Opt-in columns (`rmda ptm --outputs`), a `ptm::output` bitmask.
    outputs: u32,
}

impl rmda_core::analysis::Analysis for PtmOrdered {
    fn name(&self) -> &'static str {
        "ptm"
    }
    fn cutoffs(&self) -> Vec<f64> {
        Vec::new()
    }
    fn run(
        &self,
        ctx: &rmda_core::analysis::AnalysisContext,
        out: &mut PropertyStore,
        summary: &mut Summary,
    ) -> rmda_core::error::Result<()> {
        self.ptm
            .run_extended(ctx, out, summary, self.order, self.outputs)
    }
}

/// Polyhedral template matching (design §14); writes `f["StructureType"]`.
/// `types` selects the structures to test for, exactly like `rmda ptm --types`,
/// and `order` is `--ptm-order` (`"distance"` or `"voronoi"`). `outputs`
/// adds `rmda ptm --outputs` columns: `"ordering"`, `"F"`, `"template"`,
/// `"correspondences"`, `"residual"`, `"polar"` (or `"all"`).
#[pyfunction]
#[pyo3(signature = (f, rmsd_cutoff = 0.1, types = Vec::new(), order = "distance", outputs = Vec::new()))]
fn ptm(
    py: Python,
    f: FrameArg<'_>,
    rmsd_cutoff: f64,
    types: Vec<String>,
    order: &str,
    outputs: Vec<String>,
) -> PyResult<PyObject> {
    use rmda_core::analysis::ptm::NeighbourOrder;
    let f: &Frame = &f;
    let order = match order {
        "distance" => NeighbourOrder::Distance,
        "voronoi" => NeighbourOrder::Voronoi,
        other => {
            return Err(PyValueError::new_err(format!(
                "unknown ptm order '{other}' (expected distance|voronoi)"
            )))
        }
    };
    let a = PtmOrdered {
        ptm: Ptm {
            rmsd_cutoff,
            flags: ptm_flags(&types)?,
        },
        order,
        outputs: outputs.iter().try_fold(0u32, |acc, o| {
            rmda_core::analysis::ptm::output::from_name(o)
                .map(|b| acc | b)
                .ok_or_else(|| {
                    PyValueError::new_err(format!(
                        "unknown ptm output '{o}' (expected ordering|F|template|all)"
                    ))
                })
        })?,
    };
    // PTM builds its own k-NN index.
    let sum = run_absorbed(py, f, &a, 0.0)?;
    summary_dict(py, &sum)?.into_py_any(py)
}

/// Radial distribution function g(r) (design §18.2); returns scalars plus the
/// `histograms` and `edges` needed to plot the curve.
#[pyfunction]
#[pyo3(signature = (f, rmax, dr = None, nbins = None, partials = true, coordination = true, volume = "auto", edge_correct = false))]
#[allow(clippy::too_many_arguments)]
fn rdf(
    py: Python,
    f: FrameArg<'_>,
    rmax: f64,
    dr: Option<f64>,
    nbins: Option<f64>,
    partials: bool,
    coordination: bool,
    volume: &str,
    edge_correct: bool,
) -> PyResult<PyObject> {
    let f: &Frame = &f;
    // The CLI spells the bin width `--dr`; `nbins` is the alternative
    // spelling (`dr = rmax / nbins`). Both together are an error; neither
    // gives the CLI's default dr = 0.02.
    let dr = match (dr, nbins) {
        (Some(_), Some(_)) => {
            return Err(PyValueError::new_err(
                "rdf: give one of dr or nbins, not both",
            ))
        }
        (None, None) => 0.02,
        (Some(d), None) => d,
        (None, Some(n)) => {
            if !n.is_finite() || n <= 0.0 {
                return Err(PyValueError::new_err(format!(
                    "rdf: nbins must be positive, got {n}"
                )));
            }
            rmax / n
        }
    };
    let a = Rdf {
        rmax,
        dr,
        partials,
        coordination,
        volume: volume.parse().map_err(PyValueError::new_err)?,
        edge_correct,
    };
    a.validate(Some(&f.inner.cell)).map_err(to_py)?;
    let sum = run_absorbed(py, f, &a, rmax)?;
    let d = summary_dict(py, &sum)?;
    // Bin edges are what turns the histograms back into g(r); the CLI prints
    // them as the `# r=` line above the curves.
    d.set_item("edges", a.edges())?;
    d.into_py_any(py)
}

/// Cluster analysis (design §18.3); needs exactly one of `cutoff` / `knear` /
/// `pair_cutoffs`.
///
/// `pair_cutoffs` is `--pair-cutoff` as a dict `{(a, b): cutoff}`, types by
/// number or name; a pair not listed never bonds.
///
/// Mirrors `rmda cluster`: `select` is its `--select` expression,
/// `only_selected` its `--only-selected`, `sort_by_size` and `unwrap` its
/// flags, and `properties=True` computes what `--table` writes. Per-atom
/// `Cluster`, `ParticleCluster` (OVITO's ids: 0-based, -1 for no cluster; and
/// `UnwrappedPosition`) and the per-cluster `ClusterIdentifier`,
/// `ClusterSize`, `ClusterMass`, `ClusterCenterOfMass`,
/// `ClusterRadiusOfGyration`, `ClusterGyrationTensor` and `ClusterPercolates`
/// are attached to the frame.
#[pyfunction]
#[pyo3(signature = (f, cutoff = None, knear = None, min_size = 1, select = None,
                    only_selected = false, sort_by_size = false, unwrap = false,
                    properties = false, pair_cutoffs = None))]
#[allow(clippy::too_many_arguments)]
fn cluster(
    py: Python,
    f: FrameArg<'_>,
    cutoff: Option<f64>,
    knear: Option<usize>,
    min_size: usize,
    select: Option<&str>,
    only_selected: bool,
    sort_by_size: bool,
    unwrap: bool,
    properties: bool,
    pair_cutoffs: Option<Bound<'_, PyDict>>,
) -> PyResult<PyObject> {
    let f: &Frame = &f;
    let mode = match (cutoff, knear, pair_cutoffs) {
        (Some(c), None, None) => ClusterMode::Cutoff { cutoff: c },
        (None, Some(k), None) => ClusterMode::KNear { k },
        (None, None, Some(p)) => ClusterMode::PairCutoffs {
            pairs: p
                .iter()
                .map(|(k, v)| {
                    let (a, b): (PairType, PairType) = k.extract()?;
                    Ok(PairCutoff {
                        a: a.key()?,
                        b: b.key()?,
                        cutoff: v.extract()?,
                    })
                })
                .collect::<PyResult<_>>()?,
        },
        _ => {
            return Err(PyValueError::new_err(
                "cluster needs exactly one of cutoff, knear or pair_cutoffs",
            ))
        }
    };
    let select = select
        .map(Selection::parse)
        .transpose()
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    let a = ClusterAnalysis {
        mode,
        min_size,
        select,
        only_selected,
        sort_by_size,
        unwrap,
        properties,
    };
    let radius = max_cut(a.cutoffs_for(&f.inner));
    let sum = run_absorbed(py, f, &a, radius)?;
    summary_dict(py, &sum)?.into_py_any(py)
}

/// A particle type in `cluster(pair_cutoffs=)`: its number or its name.
#[derive(FromPyObject)]
enum PairType {
    #[pyo3(annotation = "int")]
    Id(u32),
    #[pyo3(annotation = "str")]
    Name(String),
}

impl PairType {
    fn key(self) -> PyResult<TypeKey> {
        match self {
            PairType::Id(0) => Err(PyValueError::new_err(
                "particle type 0 does not exist; type numbers start at 1",
            )),
            PairType::Id(t) => Ok(TypeKey::Id(t)),
            PairType::Name(n) => TypeKey::parse(&n).map_err(to_py),
        }
    }
}

/// `shells` as handed to `rmda.warren_cowley` / `f.run`: the CLI's comma string
/// (`"0,12"`), a single cumulative end (`12`), or a list of ends (`[0,12]`).
#[derive(FromPyObject)]
enum Shells {
    #[pyo3(annotation = "str")]
    Text(String),
    #[pyo3(annotation = "int")]
    Count(i64),
    #[pyo3(annotation = "list[int]")]
    Ends(Vec<i64>),
}

impl Shells {
    /// Rendered to the comma-separated text `parse_shells` expects.
    fn text(self) -> PyResult<String> {
        match self {
            Shells::Text(s) => Ok(s),
            Shells::Count(i) => {
                if i <= 0 {
                    return Err(PyValueError::new_err(
                        "warren_cowley: shells must be a positive cumulative neighbour count",
                    ));
                }
                Ok(format!("0,{i}"))
            }
            Shells::Ends(v) => {
                if v.is_empty() || !v.iter().all(|&x| x >= 0) || !v.iter().any(|&x| x > 0) {
                    return Err(PyValueError::new_err(
                        "warren_cowley: shells must be cumulative neighbour counts (e.g. [0, 12])",
                    ));
                }
                Ok(v.iter()
                    .map(|x| x.to_string())
                    .collect::<Vec<_>>()
                    .join(","))
            }
        }
    }
}

fn warren_cowley_text(
    py: Python,
    f: &Frame,
    shells: &str,
    edge_correct: bool,
) -> PyResult<PyObject> {
    let nneigh = parse_shells(shells).map_err(to_py)?;
    let a = WarrenCowley {
        nneigh,
        only_selected: false,
        only_same_type: false,
        edge_correct,
        per_particle: false,
    };
    a.validate(&f.inner).map_err(to_py)?;
    let sum = run_absorbed(py, f, &a, 0.0)?;
    summary_dict(py, &sum)?.into_py_any(py)
}

/// Warren-Cowley short-range-order parameters (design §18.1).
/// `edge_correct` is `rmda wc --edge-correct`: only atoms deeper below a free
/// surface than their own k-th neighbour are measured.
#[pyfunction]
#[pyo3(signature = (f, shells = None, edge_correct = false))]
fn warren_cowley(
    py: Python,
    f: FrameArg<'_>,
    shells: Option<Shells>,
    edge_correct: bool,
) -> PyResult<PyObject> {
    let f: &Frame = &f;
    let text = match shells {
        None => "0,12".to_string(),
        Some(s) => s.text()?,
    };
    warren_cowley_text(py, f, &text, edge_correct)
}

/// Build the reference lattice `name` at nearest-neighbour distance `nn`,
/// exactly as `rmda ws` builds `LatticeArg` — both go through
/// `rmda_core::lattice::from_nn`, so the conversion table has one copy.
fn lattice_box(name: &str, nn: f64) -> PyResult<Box<dyn rmda_core::lattice::Lattice>> {
    rmda_core::lattice::from_nn(name, nn).ok_or_else(|| {
        PyValueError::new_err(format!(
            "ws: unknown lattice '{name}' (expected fcc|bcc|hcp|sc|dia|dhcp)"
        ))
    })
}

/// Which reference `rmda.ws` measures against, with the inputs each one needs.
/// The engine itself is built inside the GIL-released section, from these.
enum WsMode {
    Fit(FitOptions, usize),
    Reference(Box<ExplicitWs>),
    Lattice,
}

/// Wigner-Seitz defect analysis (design §16.3, §22 `rmda.ws`).
///
/// Three reference modes, mirroring `rmda ws`:
/// - `reference='file.xyz'` (or a `Frame`, or an ASE `Atoms`): the sites of that
///   file's first frame (or of that frame), mapping the
///   input through `affine_map` (`none` when both share a box, `reference`
///   otherwise), by the explicit-reference census (pure WS, as OVITO);
///   `tolerance` only sets the `ws_config_unmatched` diagnostic and `cutoff`
///   is accepted but ignored.
/// - `fit_self=True`: fit `lattice` to the frame (design §16.3). `nn_distance`
///   seeds the search, `orientation` is a `(w,x,y,z)` seed quaternion,
///   `search_sample` bounds the search's cost, `void_min_sites` is the smallest
///   cluster of empty sites to report as a void, and `tolerance` is how far an
///   atom may sit from a lattice point and still count as on it.
/// - otherwise: take `lattice` as exact at `nn_distance`, anchored at the box
///   corner.
#[allow(clippy::too_many_arguments)]
#[pyfunction]
#[pyo3(signature = (f, reference = None, lattice = "fcc", nn_distance = 2.86, fit_self = false,
                    affine_map = "none", tolerance = None, cutoff = None,
                    orientation = None, search_sample = 100_000, void_min_sites = 1))]
fn ws(
    py: Python,
    f: FrameArg<'_>,
    reference: Option<WsRefArg<'_>>,
    lattice: &str,
    nn_distance: f64,
    fit_self: bool,
    affine_map: &str,
    tolerance: Option<f64>,
    cutoff: Option<f64>,
    orientation: Option<Vec<f64>>,
    search_sample: usize,
    void_min_sites: usize,
) -> PyResult<PyObject> {
    let f: &Frame = &f;
    if fit_self && reference.is_some() {
        return Err(PyValueError::new_err(
            "ws: give one of reference or fit_self=True, not both",
        ));
    }
    if !nn_distance.is_finite() || nn_distance <= 0.0 {
        return Err(PyValueError::new_err(format!(
            "ws: nn_distance must be positive and finite, got {nn_distance}"
        )));
    }
    if orientation.is_some() && !fit_self {
        return Err(PyValueError::new_err(
            "ws: `orientation` only applies to fit_self=True",
        ));
    }
    if void_min_sites != 1 && !fit_self {
        return Err(PyValueError::new_err(
            "ws: `void_min_sites` only applies to fit_self=True",
        ));
    }
    if search_sample != 100_000 && !fit_self {
        return Err(PyValueError::new_err(
            "ws: `search_sample` only applies to fit_self=True",
        ));
    }
    if affine_map != "none" && reference.is_none() {
        return Err(PyValueError::new_err(
            "ws: `affine_map` only applies to reference mode",
        ));
    }
    if cutoff.is_some() && reference.is_none() {
        return Err(PyValueError::new_err(
            "ws: `cutoff` only applies to reference mode",
        ));
    }
    if fit_self && (affine_map != "none" || cutoff.is_some()) {
        return Err(PyValueError::new_err(
            "ws: `affine_map`/`cutoff` only apply to a reference frame, not a fitted lattice",
        ));
    }
    let map = match affine_map {
        "none" => AffineMapping::None,
        "reference" => AffineMapping::ToReferenceCell,
        other => {
            return Err(PyValueError::new_err(format!(
                "unknown affine_map '{other}' (expected none|reference)"
            )))
        }
    };
    let seed: Option<[f64; 4]> = match orientation {
        None => None,
        Some(v) => Some(TryInto::<[f64; 4]>::try_into(v).map_err(|_: Vec<f64>| {
            PyValueError::new_err("ws: `orientation` must be a (w, x, y, z) quaternion")
        })?),
    };
    // Validates the name; the engine builds its own copy below.
    lattice_box(lattice, nn_distance)?;
    let mode = if fit_self {
        let mut opts = FitOptions {
            seed_orientation: seed,
            search_sample,
            ..Default::default()
        };
        if let Some(t) = tolerance {
            opts.tolerance = Some(t);
        }
        WsMode::Fit(opts, void_min_sites)
    } else if let Some(reference) = reference {
        let read;
        let rf: &rmda_core::data::Frame = match &reference {
            WsRefArg::Path(refpath) => {
                let path = input_path(refpath)?;
                read = py
                    .allow_threads(|| rmda_core::io::read_frames(path))
                    .map_err(to_py)?;
                read.first()
                    .ok_or_else(|| PyValueError::new_err(format!("{refpath} contains no frames")))?
            }
            WsRefArg::Frame(fr) => &fr.inner,
        };
        // Explicit-reference census; `cutoff` (a search radius) has no
        // meaning in pure WS and is only validated.
        let mut cfg = ExplicitWs::new(rf.cell, rf.particles.positions[..rf.natoms()].to_vec())
            .with_affine(map);
        if let Some(t) = tolerance {
            cfg = cfg.with_tolerance(t);
        }
        if cutoff.is_some_and(|c| !c.is_finite() || c <= 0.0) {
            return Err(PyValueError::new_err(
                "ws: cutoff must be positive and finite",
            ));
        }
        WsMode::Reference(Box::new(cfg))
    } else {
        WsMode::Lattice
    };
    let inner = &f.inner;
    let par = par();
    let mut out = PropertyStore::new();
    let sum = py
        .allow_threads(|| {
            let lt = || {
                rmda_core::lattice::from_nn(lattice, nn_distance)
                    .ok_or_else(|| CoreError::invalid(format!("ws: unknown lattice '{lattice}'")))
            };
            let engine: Box<dyn Analysis> = match mode {
                WsMode::Fit(opts, vms) => Box::new(
                    WsSelf::new(lt()?)
                        .with_options(opts)
                        .with_void_min_sites(vms),
                ),
                WsMode::Reference(cfg) => cfg,
                WsMode::Lattice => Box::new(WsRef::new(lt()?).with_origin(inner.cell.origin)),
            };
            let ctx = AnalysisContext {
                frame: inner,
                nkeys: &NeighborhoodContext { cell: None },
                par: &par,
            };
            let mut sum = Summary::default();
            engine.run(&ctx, &mut out, &mut sum)?;
            Ok(sum)
        })
        .map_err(to_py)?;
    absorb(&out, f);
    summary_dict(py, &sum)?.into_py_any(py)
}

/// The five scalars `rmda segment` prints, which `rmda defects --segment`
/// prints too.
fn segment_scalars(d: &Bound<'_, PyDict>, s: &Segmentation) -> PyResult<()> {
    let assigned = s.grain.iter().filter(|&&g| g >= 0).count();
    d.set_item("segment_grains", s.grains.len())?;
    d.set_item("segment_crystallites", s.n_crystallites)?;
    d.set_item("segment_atoms_assigned", assigned)?;
    d.set_item("segment_atoms_unassigned", s.grain.len() - assigned)?;
    d.set_item("segment_atoms_adopted", s.adopted)?;
    if let Some(t) = s.merge_threshold {
        d.set_item("segment_merge_threshold", t)?;
    }
    Ok(())
}

/// `mode` ("crystallite" | "auto") and `ptm_order` ("distance" | "voronoi",
/// default voronoi for auto), as `--mode` / `--ptm-order` of the CLI.
fn seg_mode_and_order(mode: &str, ptm_order: Option<&str>) -> PyResult<(SegMode, NeighbourOrder)> {
    let mode = match mode {
        "crystallite" => SegMode::Crystallite,
        "auto" => SegMode::Auto,
        other => {
            return Err(PyValueError::new_err(format!(
                "unknown segmentation mode '{other}' (expected crystallite|auto)"
            )))
        }
    };
    let order = match ptm_order {
        Some("distance") => NeighbourOrder::Distance,
        Some("voronoi") => NeighbourOrder::Voronoi,
        None if mode == SegMode::Auto => NeighbourOrder::Voronoi,
        None => NeighbourOrder::Distance,
        Some(other) => {
            return Err(PyValueError::new_err(format!(
                "unknown ptm_order '{other}' (expected distance|voronoi)"
            )))
        }
    };
    Ok((mode, order))
}

/// `rmda segment`'s stdout scalars, plus the grain and pair tables its
/// `--table` writes, as dicts of numpy columns.
fn segmentation_dict<'py>(py: Python<'py>, s: &Segmentation) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    segment_scalars(&d, s)?;

    let g = PyDict::new(py);
    let col =
        |f: &dyn Fn(&grain_seg::GrainInfo) -> i64| -> Vec<i64> { s.grains.iter().map(f).collect() };
    g.set_item("grain", PyArray1::from_vec(py, col(&|x| x.id as i64)))?;
    g.set_item("size", PyArray1::from_vec(py, col(&|x| x.size as i64)))?;
    g.set_item(
        "structure",
        PyArray1::from_vec(py, col(&|x| x.structure as i64)),
    )?;
    g.set_item(
        "crystallites",
        PyArray1::from_vec(py, col(&|x| x.crystallites as i64)),
    )?;
    let q: Vec<[f64; 4]> = s.grains.iter().map(|x| x.orientation).collect();
    g.set_item("orientation", Array2::from(q).into_pyarray(py))?;
    d.set_item("grains", g)?;

    let p = PyDict::new(py);
    let pcol =
        |f: &dyn Fn(&grain_seg::GrainPair) -> i64| -> Vec<i64> { s.pairs.iter().map(f).collect() };
    p.set_item("grain_a", PyArray1::from_vec(py, pcol(&|x| x.a as i64)))?;
    p.set_item("grain_b", PyArray1::from_vec(py, pcol(&|x| x.b as i64)))?;
    p.set_item("bonds", PyArray1::from_vec(py, pcol(&|x| x.bonds as i64)))?;
    let dis: Vec<f64> = s.pairs.iter().map(|x| x.disorientation).collect();
    p.set_item("disorientation_deg", PyArray1::from_vec(py, dis))?;
    d.set_item("pairs", p)?;
    Ok(d)
}

/// Grain segmentation (design §15), as `rmda segment` runs it: PTM, then
/// crystallites of coherent orientation merged on their mean orientations.
///
/// Attaches `f["Grain"]` (int32, -1 unassigned) and `f["StructureType"]`, the
/// columns `rmda segment -o` writes. Returns the CLI's scalars plus `grains`
/// (one row per grain) and `pairs` (one row per touching grain pair), each a
/// dict of numpy columns.
#[pyfunction]
#[pyo3(signature = (f, threshold = 5.0, core_angle = 2.0, min_size = 100, adopt = true,
                    rmsd_cutoff = 0.1, dhcp = false, mode = "crystallite", ptm_order = None,
                    dhcp_merge_faults = false))]
#[allow(clippy::too_many_arguments)]
fn segment(
    py: Python,
    f: FrameArg<'_>,
    threshold: f64,
    core_angle: f64,
    min_size: usize,
    adopt: bool,
    rmsd_cutoff: f64,
    dhcp: bool,
    mode: &str,
    ptm_order: Option<&str>,
    dhcp_merge_faults: bool,
) -> PyResult<PyObject> {
    let f: &Frame = &f;
    let (mode, order) = seg_mode_and_order(mode, ptm_order)?;
    if dhcp_merge_faults && !dhcp {
        return Err(PyValueError::new_err(
            "segment: dhcp_merge_faults needs dhcp=True",
        ));
    }
    let cfg = SegConfig {
        mode,
        threshold,
        core_angle,
        min_size,
        adopt,
        dhcp_faults: dhcp_merge_faults,
        ..Default::default()
    };
    let inner = &f.inner;
    let par = par();
    let (ptm, s) = py
        .allow_threads(|| -> rmda_core::error::Result<_> {
            let ptm = defect_frame::run_ptm(
                inner,
                &PtmSettings {
                    rmsd_cutoff,
                    dhcp,
                    order,
                },
                &par,
                &mut (),
            )?;
            let s = grain_seg::segment(&inner.cell, &inner.particles.positions, &ptm, &cfg, &par)?;
            Ok((ptm, s))
        })
        .map_err(to_py)?;
    {
        let mut props = f.attached();
        props.insert(
            "Grain".to_string(),
            AbsorbedProp {
                values: PropValues::I32(s.grain.clone()),
                components: 1,
            },
        );
        props.insert(
            "StructureType".to_string(),
            AbsorbedProp {
                values: PropValues::U8(ptm.iter().map(|p| p.structure).collect()),
                components: 1,
            },
        );
    }
    let res = segmentation_dict(py, &s)?;
    attach_tables(f, &res, "segment", &["grains", "pairs"])?;
    res.into_py_any(py)
}

/// An attached property, else an input column of the frame itself, as integer
/// grain labels — `rmda defects --grain-column`. Attached first, so
/// `rmda.segment(f); rmda.defects(f, grain_column="Grain")` uses the labels
/// that call just made.
fn column_labels(f: &Frame, name: &str) -> PyResult<Vec<i32>> {
    if let Some(p) = f.attached().get(name) {
        if p.components != 1 {
            return Err(PyValueError::new_err(format!(
                "column '{name}' has {} components, not one grain label per atom",
                p.components
            )));
        }
        return Ok(match &p.values {
            PropValues::U8(v) => v.iter().map(|&g| g as i32).collect(),
            PropValues::I32(v) => v.clone(),
            PropValues::I64(v) => v.iter().map(|&g| g as i32).collect(),
            PropValues::F32(v) => v.iter().map(|&g| g.round() as i32).collect(),
            PropValues::F64(v) => v.iter().map(|&g| g.round() as i32).collect(),
        });
    }
    // The input column, read as `rmda defects --grain-column` reads it.
    defect_frame::column_labels(&f.inner, name).map_err(|e| match e {
        CoreError::InvalidData(m) => PyValueError::new_err(m),
        other => to_py(other),
    })
}

/// `ws_reference`: a path (first frame read) or a frame.
enum WsRefArg<'py> {
    Path(String),
    Frame(FrameArg<'py>),
}

impl<'py> FromPyObject<'py> for WsRefArg<'py> {
    fn extract_bound(ob: &Bound<'py, PyAny>) -> PyResult<Self> {
        if let Ok(p) = ob.extract::<String>() {
            return Ok(WsRefArg::Path(p));
        }
        Ok(WsRefArg::Frame(ob.extract()?))
    }
}

/// Grain tracking across frames, for a series in Python: pass the same
/// tracker to `rmda.defects(f, segment=True, tracker=t)` (or `rmda.void`) for
/// each frame in order, and grains keep their ids (`stable_grain`) from frame
/// to frame as `rmda defects FILES... --out` keeps them; each result's
/// `grain_events` says what happened (new, continued, split, merged).
/// Atoms are matched by id.
#[pyclass(name = "GrainTracker")]
struct GrainTracker {
    inner: Tracker,
}

#[pymethods]
impl GrainTracker {
    #[new]
    fn new() -> Self {
        GrainTracker {
            inner: Tracker::new(),
        }
    }
}

/// The tracker's events as a table: `event` (new|continued|split|merged),
/// `stable`, `other` (the grain split from or merged into, else -1).
fn events_dict<'py>(
    py: Python<'py>,
    events: &[rmda_core::analysis::grain_track::Event],
) -> PyResult<Bound<'py, PyDict>> {
    use rmda_core::analysis::grain_track::Event;
    let rows: Vec<(&str, i64, i64)> = events
        .iter()
        .map(|e| match *e {
            Event::New { stable } => ("new", stable as i64, -1),
            Event::Continued { stable } => ("continued", stable as i64, -1),
            Event::Split { stable, from } => ("split", stable as i64, from as i64),
            Event::Merged { stable, into } => ("merged", stable as i64, into as i64),
        })
        .collect();
    let d = PyDict::new(py);
    d.set_item("event", rows.iter().map(|r| r.0).collect::<Vec<_>>())?;
    d.set_item(
        "stable",
        PyArray1::from_vec(py, rows.iter().map(|r| r.1).collect()),
    )?;
    d.set_item(
        "other",
        PyArray1::from_vec(py, rows.iter().map(|r| r.2).collect()),
    )?;
    Ok(d)
}

/// One frame of `rmda defects`, through the per-frame path the CLI runs
/// (`rmda_core::analysis::defects::frame`): PTM, grain labels, optional
/// segmentation splitting each region (with `tracker`'s stable ids, or a fresh
/// tracker's),
/// the per-grain census, then defect clusters. Under an explicit reference
/// the reference is prepared first, and its segmentation is the one reported.
fn defects_frame(
    f: &rmda_core::data::Frame,
    s: &FrameSettings,
    ws: Option<(&rmda_core::data::Frame, WsAffine)>,
    tracker: Option<&mut Tracker>,
    par: &Par,
) -> rmda_core::error::Result<FrameResult> {
    match ws {
        Some((rf, affine)) => {
            let (reference, seg) = defect_frame::prepare_reference(rf, s, par, &mut ())?;
            let mut out = defect_frame::analyse(
                f,
                s,
                par,
                None,
                Some(WsReference {
                    reference: &reference,
                    frame: rf,
                    affine,
                }),
                &mut (),
            )?;
            out.segmentation = seg;
            Ok(out)
        }
        None => {
            let mut fresh = Tracker::new();
            let t = tracker.unwrap_or(&mut fresh);
            defect_frame::analyse(f, s, par, Some(t), None, &mut ())
        }
    }
}

/// `rmda defects`' stdout scalars and every table its files hold, as Python:
/// the scalars at the top level, `grains` (the `--table` CSV), `vacancies`,
/// `interstitials` (the `--defects` xyz files) and `clusters` (the series
/// `clusters.csv`) as dicts of numpy columns, plus per-atom `atom_grain` and
/// `displacement` arrays.
fn defects_dict<'py>(
    py: Python<'py>,
    f: &rmda_core::data::Frame,
    out: &FrameResult,
) -> PyResult<Bound<'py, PyDict>> {
    let r = &out.result;
    let (i, b) = r.totals();
    let measured = r
        .grains
        .iter()
        .filter(|g| g.status == Status::Measured)
        .count();
    let largest = out
        .clusters
        .clusters
        .iter()
        .map(|c| c.size())
        .max()
        .unwrap_or(0);
    // As on the CLI's stderr: a withheld grain adds nothing to the totals, so a
    // frame where no grain was measured returns the zeros of a perfect crystal.
    let unmeasured: Vec<_> = r
        .grains
        .iter()
        .filter(|g| g.status != Status::Measured)
        .collect();
    if !unmeasured.is_empty() {
        let atoms: usize = unmeasured.iter().map(|g| g.n_atoms).sum();
        let all: usize = r.grains.iter().map(|g| g.n_atoms).sum();
        let msg = format!(
            "defects: {} of {} grain(s) not measured ({} of {} atoms); their defects are not \
             in the totals{}",
            unmeasured.len(),
            r.grains.len(),
            atoms,
            all,
            if measured == 0 {
                " (no grain measured: the zero counts are not a defect-free crystal)"
            } else {
                ""
            }
        );
        PyErr::warn(
            py,
            &py.get_type::<pyo3::exceptions::PyUserWarning>(),
            &std::ffi::CString::new(msg).expect("no NUL in the message"),
            1,
        )?;
    }
    let d = PyDict::new(py);
    for (k, v) in [
        ("defects_grains", r.grains.len() as i64),
        ("defects_grains_measured", measured as i64),
        ("defects_sites", i.sites as i64),
        ("defects_atoms", i.atoms as i64),
        ("defects_vacancies", i.vacancies as i64),
        ("defects_interstitials", i.interstitials as i64),
        ("defects_boundary_sites", b.sites as i64),
        ("defects_boundary_atoms", b.atoms as i64),
        ("defects_boundary_vacancies", b.vacancies as i64),
        ("defects_boundary_interstitials", b.interstitials as i64),
        ("defects_net_vacancies", i.net_vacancies()),
        ("defects_boundary_net_vacancies", b.net_vacancies()),
        ("defects_balance_residual", i.residual() + b.residual()),
        ("defects_clusters", out.clusters.clusters.len() as i64),
        ("defects_largest_cluster", largest as i64),
    ] {
        d.set_item(k, v)?;
    }
    if let Some(AntisiteTypes { names, .. }) = &out.antisites {
        // As `rmda defects --antisites` prints them.
        // NaN when no grain is counted (no chemical order, or none measured).
        match r.antisite_totals() {
            Some((ai, ab)) => {
                d.set_item("defects_antisites", ai as i64)?;
                d.set_item("defects_boundary_antisites", ab as i64)?;
            }
            None => {
                d.set_item("defects_antisites", f64::NAN)?;
                d.set_item("defects_boundary_antisites", f64::NAN)?;
            }
        }
        for (&(st, ot), c) in &r.antisite_pairs() {
            let (st, ot) = (
                antisite::type_label(names, st),
                antisite::type_label(names, ot),
            );
            d.set_item(format!("defects_antisites_{ot}_on_{st}"), c[0] as i64)?;
            d.set_item(
                format!("defects_boundary_antisites_{ot}_on_{st}"),
                c[1] as i64,
            )?;
        }
        let cl = &out.clusters;
        let joined: u32 = cl.clusters.iter().map(|c| c.antisites).sum();
        let largest = cl.antisite_only.iter().map(|c| c.antisites).max();
        d.set_item("defects_clustered_antisites", joined as i64)?;
        d.set_item("defects_antisite_clusters", cl.antisite_only.len() as i64)?;
        d.set_item(
            "defects_largest_antisite_cluster",
            largest.unwrap_or(0) as i64,
        )?;
    }
    d.set_item("cluster_cutoff", out.cluster_cutoff)?;
    if let Some(s) = &out.segmentation {
        segment_scalars(&d, s)?;
    }

    // Per grain. Counts of an unmeasured grain are withheld as -1, as in the
    // CLI table, never handed out as data.
    let gs = &r.grains;
    let ints = |f: &dyn Fn(&defects::GrainReport) -> i64| -> Bound<'py, PyArray1<i64>> {
        PyArray1::from_vec(py, gs.iter().map(f).collect())
    };
    let floats = |f: &dyn Fn(&defects::GrainReport) -> f64| -> Bound<'py, PyArray1<f64>> {
        PyArray1::from_vec(py, gs.iter().map(f).collect())
    };
    let counted = |g: &defects::GrainReport, v: u64| {
        if g.status == Status::Measured {
            v as i64
        } else {
            -1
        }
    };
    let cluster_stats = |g: &defects::GrainReport| {
        let mine = out.clusters.clusters.iter().filter(|k| k.grain == g.grain);
        mine.fold((0i64, 0i64), |(n, m), k| (n + 1, m.max(k.size() as i64)))
    };
    let t = PyDict::new(py);
    t.set_item("grain", ints(&|g| g.grain as i64))?;
    t.set_item("stable_grain", ints(&|g| out.stable_id(g.grain)))?;
    t.set_item(
        "status",
        gs.iter()
            .map(|g| g.status.label().to_string())
            .collect::<Vec<_>>(),
    )?;
    t.set_item(
        "status_detail",
        gs.iter()
            .map(|g| match &g.status {
                Status::FitFailed(e) => e.clone(),
                _ => String::new(),
            })
            .collect::<Vec<_>>(),
    )?;
    t.set_item(
        "reference",
        gs.iter().map(|g| g.reference).collect::<Vec<_>>(),
    )?;
    // Patch figures mean nothing for a local grain (-1, NaN), and local ones
    // nothing for the others.
    let patch = |g: &defects::GrainReport, v: usize| if g.local.is_some() { -1 } else { v as i64 };
    t.set_item("n_patches", ints(&|g| patch(g, g.n_patches)))?;
    t.set_item("n_patches_fitted", ints(&|g| patch(g, g.n_patches_fitted)))?;
    t.set_item(
        "phase_agreement",
        floats(&|g| {
            if g.local.is_some() {
                f64::NAN
            } else {
                g.phase_agreement
            }
        }),
    )?;
    t.set_item(
        "face_agreement",
        floats(&|g| {
            if g.local.is_some() {
                f64::NAN
            } else {
                g.face_agreement
            }
        }),
    )?;
    t.set_item("n_atoms", ints(&|g| g.n_atoms as i64))?;
    t.set_item("n_crystalline", ints(&|g| g.n_crystalline as i64))?;
    t.set_item("structure", ints(&|g| g.structure as i64))?;
    t.set_item("bond", floats(&|g| g.bond))?;
    t.set_item("pair_fraction", floats(&|g| g.pair_fraction))?;
    t.set_item("pair_rms", floats(&|g| g.pair_rms))?;
    t.set_item("rms_displacement", floats(&|g| g.rms_displacement))?;
    t.set_item("offlattice_fraction", floats(&|g| g.offlattice_fraction))?;
    t.set_item(
        "snapped",
        Array2::from(gs.iter().map(|g| g.snapped).collect::<Vec<_>>()).into_pyarray(py),
    )?;
    t.set_item(
        "domain_converged",
        PyArray1::from_vec(py, gs.iter().map(|g| g.domain_converged).collect()),
    )?;
    t.set_item("sites", ints(&|g| counted(g, g.interior.sites)))?;
    t.set_item("atoms", ints(&|g| counted(g, g.interior.atoms)))?;
    t.set_item("vacancies", ints(&|g| counted(g, g.interior.vacancies)))?;
    t.set_item(
        "interstitials",
        ints(&|g| counted(g, g.interior.interstitials)),
    )?;
    t.set_item("gb_sites", ints(&|g| counted(g, g.boundary.sites)))?;
    t.set_item("gb_atoms", ints(&|g| counted(g, g.boundary.atoms)))?;
    t.set_item("gb_vacancies", ints(&|g| counted(g, g.boundary.vacancies)))?;
    t.set_item(
        "gb_interstitials",
        ints(&|g| counted(g, g.boundary.interstitials)),
    )?;
    let net = |g: &defects::GrainReport, c: &defects::ClassCount| {
        if g.status == Status::Measured {
            c.net_vacancies()
        } else {
            -1
        }
    };
    t.set_item("net_vacancies", ints(&|g| net(g, &g.interior)))?;
    t.set_item("gb_net_vacancies", ints(&|g| net(g, &g.boundary)))?;
    let local = |f: &dyn Fn(&defects::LocalStats) -> usize| {
        ints(&|g| g.local.as_ref().map_or(-1, |s| f(s) as i64))
    };
    t.set_item("local_chunks", local(&|s| s.chunks))?;
    t.set_item("local_defect_atoms", local(&|s| s.defect_atoms))?;
    t.set_item("local_unresolved", local(&|s| s.unresolved))?;
    t.set_item("local_regions", local(&|s| s.regions))?;
    t.set_item("local_region_atoms", local(&|s| s.region_atoms))?;
    t.set_item("local_extended_atoms", local(&|s| s.extended))?;
    t.set_item(
        "local_fit_rms",
        floats(&|g| g.local.as_ref().map_or(f64::NAN, |s| s.frame_rms)),
    )?;
    t.set_item("clusters", ints(&|g| counted(g, cluster_stats(g).0 as u64)))?;
    t.set_item(
        "largest_cluster",
        ints(&|g| counted(g, cluster_stats(g).1 as u64)),
    )?;
    let tm: Vec<f64> = gs
        .iter()
        .flat_map(|g| g.t.iter().flatten().copied())
        .collect();
    t.set_item(
        "t",
        Array3::from_shape_vec((gs.len(), 3, 3), tm)
            .map_err(|e| PyValueError::new_err(e.to_string()))?
            .into_pyarray(py),
    )?;
    t.set_item(
        "origin",
        Array2::from(gs.iter().map(|g| g.origin).collect::<Vec<_>>()).into_pyarray(py),
    )?;
    if let Some(AntisiteTypes { names, .. }) = &out.antisites {
        // -1 withheld (unmeasured), as the CLI table; a grain without a
        // count has -1 and an empty label here, empty cells there.
        let an = |g: &defects::GrainReport, f: &dyn Fn(&defects::GrainAntisites) -> u64| match (
            &g.antisites,
            g.status == Status::Measured,
        ) {
            (Some(a), true) => f(a) as i64,
            _ => -1,
        };
        t.set_item("antisites", ints(&|g| an(g, &|a| a.interior)))?;
        t.set_item("gb_antisites", ints(&|g| an(g, &|a| a.boundary)))?;
        t.set_item(
            "sublattice_types",
            gs.iter()
                .map(|g| match (&g.antisites, g.status == Status::Measured) {
                    (Some(a), true) => antisite::sublattice_label(names, &a.sublattice),
                    _ => String::new(),
                })
                .collect::<Vec<_>>(),
        )?;
    }
    d.set_item("grains", t)?;

    // Vacancies: wrapped into the box, as `vacancies.xyz` writes them.
    let vs = &r.vacancies;
    let v = PyDict::new(py);
    v.set_item(
        "position",
        Array2::from(
            vs.iter()
                .map(|x| f.cell.wrap(x.position))
                .collect::<Vec<_>>(),
        )
        .into_pyarray(py),
    )?;
    v.set_item(
        "grain",
        PyArray1::from_vec(py, vs.iter().map(|x| out.stable_id(x.grain)).collect()),
    )?;
    v.set_item(
        "boundary",
        PyArray1::from_vec(py, vs.iter().map(|x| x.boundary).collect()),
    )?;
    v.set_item(
        "cluster",
        PyArray1::from_slice(py, &out.clusters.vacancy_cluster),
    )?;
    if let Some((vs, _)) = &out.ws_sites {
        v.set_item("site_id", PyArray1::from_slice(py, vs))?;
    }
    d.set_item("vacancies", v)?;

    // Interstitials: the atom itself, at its own position.
    let is = &r.interstitials;
    let it = PyDict::new(py);
    let atom = |s: &defects::Interstitial| s.atom as usize;
    it.set_item(
        "atom",
        PyArray1::from_vec(py, is.iter().map(|s| s.atom as i64).collect()),
    )?;
    it.set_item(
        "id",
        PyArray1::from_vec(py, is.iter().map(|s| f.particles.ids[atom(s)]).collect()),
    )?;
    it.set_item(
        "type",
        PyArray1::from_vec(py, is.iter().map(|s| f.particles.types[atom(s)]).collect()),
    )?;
    it.set_item(
        "position",
        Array2::from(
            is.iter()
                .map(|s| f.particles.positions[atom(s)])
                .collect::<Vec<_>>(),
        )
        .into_pyarray(py),
    )?;
    it.set_item(
        "grain",
        PyArray1::from_vec(py, is.iter().map(|s| out.stable_id(s.grain)).collect()),
    )?;
    it.set_item(
        "boundary",
        PyArray1::from_vec(py, is.iter().map(|s| s.boundary).collect()),
    )?;
    it.set_item(
        "cluster",
        PyArray1::from_slice(py, &out.clusters.interstitial_cluster),
    )?;
    if let Some((_, is)) = &out.ws_sites {
        it.set_item("site_id", PyArray1::from_slice(py, is))?;
    }
    d.set_item("interstitials", it)?;

    // Antisites: the atom, at its own position, and its site.
    if let Some(AntisiteTypes {
        types, site_ids, ..
    }) = &out.antisites
    {
        let xs = &r.antisites;
        let a = PyDict::new(py);
        let atom = |x: &defects::Antisite| x.atom as usize;
        a.set_item(
            "atom",
            PyArray1::from_vec(py, xs.iter().map(|x| x.atom as i64).collect()),
        )?;
        a.set_item(
            "id",
            PyArray1::from_vec(py, xs.iter().map(|x| f.particles.ids[atom(x)]).collect()),
        )?;
        a.set_item(
            "type",
            PyArray1::from_vec(py, xs.iter().map(|x| types[atom(x)]).collect()),
        )?;
        a.set_item(
            "site_type",
            PyArray1::from_vec(py, xs.iter().map(|x| x.site_type).collect()),
        )?;
        a.set_item(
            "position",
            Array2::from(
                xs.iter()
                    .map(|x| f.particles.positions[atom(x)])
                    .collect::<Vec<_>>(),
            )
            .into_pyarray(py),
        )?;
        a.set_item(
            "site_position",
            Array2::from(
                xs.iter()
                    .map(|x| f.cell.wrap(x.site_position))
                    .collect::<Vec<_>>(),
            )
            .into_pyarray(py),
        )?;
        a.set_item(
            "grain",
            PyArray1::from_vec(py, xs.iter().map(|x| out.stable_id(x.grain)).collect()),
        )?;
        a.set_item(
            "boundary",
            PyArray1::from_vec(py, xs.iter().map(|x| x.boundary).collect()),
        )?;
        if let Some(s) = site_ids {
            a.set_item("site_id", PyArray1::from_slice(py, s))?;
        }
        a.set_item(
            "cluster",
            PyArray1::from_slice(py, &out.clusters.antisite_cluster),
        )?;
        d.set_item("antisites", a)?;
    }

    // Clusters, as the series `clusters.csv` rows: with antisites, the
    // antisite-only clusters after the V/I ones, and an `antisites` column.
    let cs: Vec<&defect_clusters::Cluster> = out
        .clusters
        .clusters
        .iter()
        .chain(&out.clusters.antisite_only)
        .collect();
    let cs = &cs;
    let c = PyDict::new(py);
    let ccol = |f: &dyn Fn(&defect_clusters::Cluster) -> i64| -> Bound<'py, PyArray1<i64>> {
        PyArray1::from_vec(py, cs.iter().map(|k| f(k)).collect())
    };
    c.set_item("cluster", ccol(&|k| k.id as i64))?;
    c.set_item("grain", ccol(&|k| out.stable_id(k.grain)))?;
    c.set_item(
        "multi_grain",
        PyArray1::from_vec(py, cs.iter().map(|k| k.multi_grain).collect()),
    )?;
    c.set_item("size", ccol(&|k| k.size() as i64))?;
    c.set_item("vacancies", ccol(&|k| k.vacancies as i64))?;
    c.set_item("interstitials", ccol(&|k| k.interstitials as i64))?;
    c.set_item("net", ccol(&|k| k.net()))?;
    c.set_item(
        "centre",
        Array2::from(cs.iter().map(|k| f.cell.wrap(k.centre)).collect::<Vec<_>>()).into_pyarray(py),
    )?;
    if out.antisites.is_some() {
        c.set_item("antisites", ccol(&|k| k.antisites as i64))?;
    }
    d.set_item("clusters", c)?;

    d.set_item("atom_grain", PyArray1::from_slice(py, &out.grains))?;
    d.set_item("displacement", PyArray1::from_slice(py, &r.displacement))?;
    Ok(d)
}

/// Per-grain Wigner–Seitz defect census (design §16), as `rmda defects`
/// runs it on one frame.
///
/// Grains come from `regions` (selection expressions such as `"z <= 500"`;
/// atoms take the first region they match), else `grain_column` (an attached
/// property such as the `Grain` that `rmda.segment` writes, or an input
/// column), else the whole frame as one grain. `segment=True` splits each of
/// those by segmented grains (`seg_threshold` degrees, `seg_min_size` atoms).
/// `reference` is `"auto"`, `"global"` (one affine lattice per grain),
/// `"patch"` (patch-local lattices sharing one index space), `"local"` or
/// `"periodic"`, as `rmda defects --reference`.
///
/// `ws_reference` (a path, or a frame) switches to explicit-reference
/// Wigner–Seitz (`rmda defects --ws-reference`, OVITO's
/// WignerSeitzAnalysisModifier): the sites are the reference's atoms, and
/// `regions`, `grain_column` and `segment` are evaluated on the reference.
/// `ws_affine` is `"off"`, `"to-reference"` or `"to-current"`
/// (`--ws-affine`). The vacancy and interstitial tables then carry the
/// reference atom id of their site (`site_id`).
///
/// `antisites=True` (`--antisites`) also counts antisites: singly occupied
/// sites holding an atom of another type than the site's. With
/// `ws_reference` a site's type is its reference atom's (current types are
/// matched to the reference's by name when both frames name them); with a
/// fitted reference it is the majority type on each motif basis site of the
/// grain, and under `reference="local"` the majority per basis site of each
/// vote region. Adds the scalars `defects_antisites`,
/// `defects_boundary_antisites` and one per (occupant, site) pair,
/// `defects_clustered_antisites`, `defects_antisite_clusters` and
/// `defects_largest_antisite_cluster`, the grain columns `antisites`,
/// `gb_antisites` and `sublattice_types`, an `antisites` table (with each
/// antisite's `cluster`), and in the `clusters` table an `antisites` column
/// and the antisite-only clusters after the vacancy/interstitial ones.
///
/// `resolve_ambiguous=True` (`--resolve-ambiguous`): under the global and
/// patch references an atom more than 0.3 bonds from its nearest site takes
/// the site its neighbours point to, instead of the nearest (Wigner–Seitz,
/// OVITO's, the default), which removes the spurious Frenkel pairs inside
/// large loops. `tracker` (a `GrainTracker`) keeps grain ids across calls.
///
/// Returns the CLI's scalars plus the `grains`, `vacancies`, `interstitials`
/// and `clusters` tables (dicts of numpy columns) and per-atom `atom_grain`
/// and `displacement` arrays.
#[allow(clippy::too_many_arguments)]
#[pyfunction]
#[pyo3(name = "defects",
       signature = (f, regions = Vec::new(), grain_column = None, reference = "auto",
                    segment = false, seg_threshold = 5.0, seg_min_size = 100,
                    boundary_width = None, min_atoms = 100, rmsd_cutoff = 0.1,
                    patch_size = None, cluster_cutoff = None, dhcp = false,
                    seg_mode = "crystallite", ptm_order = None, ws_reference = None,
                    ws_affine = "off", antisites = false, local_merge = "union",
                    tracker = None, resolve_ambiguous = false))]
fn defects_py(
    py: Python,
    f: FrameArg<'_>,
    regions: Vec<String>,
    grain_column: Option<String>,
    reference: &str,
    segment: bool,
    seg_threshold: f64,
    seg_min_size: usize,
    boundary_width: Option<f64>,
    min_atoms: usize,
    rmsd_cutoff: f64,
    patch_size: Option<f64>,
    cluster_cutoff: Option<f64>,
    dhcp: bool,
    seg_mode: &str,
    ptm_order: Option<&str>,
    ws_reference: Option<WsRefArg<'_>>,
    ws_affine: &str,
    antisites: bool,
    local_merge: &str,
    tracker: Option<Bound<'_, GrainTracker>>,
    resolve_ambiguous: bool,
) -> PyResult<PyObject> {
    let f: &Frame = &f;
    if tracker.is_some() && ws_reference.is_some() {
        return Err(PyValueError::new_err(
            "defects: an explicit ws_reference's grains are its own; drop tracker",
        ));
    }
    let local_merge = match local_merge {
        "union" => rmda_core::analysis::defects::LocalMerge::Union,
        "partial" => rmda_core::analysis::defects::LocalMerge::Partial,
        other => {
            return Err(PyValueError::new_err(format!(
                "defects: unknown local_merge '{other}' (expected union|partial)"
            )))
        }
    };
    let ws_affine = match ws_affine {
        "off" => WsAffine::Off,
        "to-reference" | "to_reference" => WsAffine::ToReference,
        "to-current" | "to_current" => WsAffine::ToCurrent,
        other => {
            return Err(PyValueError::new_err(format!(
                "defects: unknown ws_affine '{other}' (expected off|to-reference|to-current)"
            )))
        }
    };
    let ws_frame: Option<Frame> = match &ws_reference {
        Some(WsRefArg::Path(p)) => Some(Frame::from_core(
            py.allow_threads(|| rmda_core::io::read_frame(std::path::Path::new(p)))
                .map_err(to_py)?,
        )),
        _ => None,
    };
    let ws_ref: Option<&Frame> = match &ws_reference {
        Some(WsRefArg::Frame(fr)) => Some(fr),
        Some(WsRefArg::Path(_)) => ws_frame.as_ref(),
        None => None,
    };
    if ws_ref.is_none() && ws_affine != WsAffine::Off {
        return Err(PyValueError::new_err(
            "defects: ws_affine needs ws_reference",
        ));
    }
    if ws_ref.is_some() && (reference != "auto" || patch_size.is_some()) {
        return Err(PyValueError::new_err(
            "defects: ws_reference replaces the fitted reference; drop reference/patch_size",
        ));
    }
    // Grain labels come from the frame they are evaluated on.
    let labelled: &Frame = ws_ref.unwrap_or(f);
    let (seg_mode, order) = seg_mode_and_order(seg_mode, ptm_order)?;
    // Voronoi PTM by default only when auto segmentation actually runs.
    let order = match ptm_order {
        None if !segment => NeighbourOrder::Distance,
        _ => order,
    };
    let reference = match reference {
        "auto" => ReferenceMode::Auto,
        "global" => ReferenceMode::Global,
        "patch" => ReferenceMode::Patch,
        "local" => ReferenceMode::Local,
        "periodic" => ReferenceMode::Periodic,
        other => {
            return Err(PyValueError::new_err(format!(
                "defects: unknown reference '{other}' (expected auto|global|patch|local|periodic)"
            )))
        }
    };
    let source = match (regions.is_empty(), &grain_column) {
        (false, Some(_)) => {
            return Err(PyValueError::new_err(
                "defects: give one of regions or grain_column, not both",
            ))
        }
        (false, None) => GrainSource::Regions(
            regions
                .iter()
                .map(|r| Selection::parse(r).map_err(|e| PyValueError::new_err(e.to_string())))
                .collect::<PyResult<_>>()?,
        ),
        (true, Some(name)) => GrainSource::Labels(column_labels(labelled, name)?),
        (true, None) => GrainSource::Whole,
    };
    let settings = FrameSettings {
        source,
        segment: segment.then(|| SegConfig {
            mode: seg_mode,
            threshold: seg_threshold,
            min_size: seg_min_size,
            ..Default::default()
        }),
        ptm: PtmSettings {
            rmsd_cutoff,
            dhcp,
            order,
        },
        defects: DefectsConfig {
            boundary_width,
            min_atoms,
            reference,
            patch_size,
            local_merge,
            resolve_ambiguous,
            ..Default::default()
        },
        cluster_cutoff,
        antisites,
        elements: Vec::new(),
        current_ptm: false,
    };
    let inner = &f.inner;
    let ws = ws_ref.map(|r| (&r.inner, ws_affine));
    let par = par();
    let mut guard = tracker.as_ref().map(|t| t.borrow_mut());
    let tr: Option<&mut Tracker> = guard.as_mut().map(|g| &mut g.inner);
    let out = py
        .allow_threads(|| defects_frame(inner, &settings, ws, tr, &par))
        .map_err(to_py)?;
    drop(guard);
    let res = defects_dict(py, inner, &out)?;
    res.set_item("grain_events", events_dict(py, &out.events)?)?;
    attach_tables(
        f,
        &res,
        "defects",
        &[
            "grains",
            "vacancies",
            "interstitials",
            "clusters",
            "antisites",
        ],
    )?;
    res.into_py_any(py)
}

/// The scalars `rmda grain-boundaries` prints, in its order.
const GB_KEYS: &[&str] = &[
    "gb_atoms_boundary",
    "gb_atoms_interior",
    "gb_fraction_boundary",
    "gb_atoms_assigned",
    "gb_atoms_unassigned",
    "gb_grains",
    "gb_slab_thickness",
    "gb_w_db",
    "gb_unassigned_is_boundary",
];

/// Grain-boundary (seam) atoms, as `rmda grain-boundaries` finds them: PTM
/// (FCC, HCP, BCC, ICO, SC at rmsd 0.1) and grain segmentation into one fresh
/// store (misorientation 10°, `min_grain_size`, orphan adoption unless
/// `no_adopt_orphans`), then an atom is on the boundary when the grain ids
/// among itself and its neighbours within `w_db` Å (default 4.0) hold two or
/// more values. `unassigned_is_boundary` puts atoms with no grain in the
/// boundary class too.
///
/// Attaches the per-atom columns the CLI's `-o` writes (`GrainBoundary`,
/// `Grain`, PTM's) and puts the per-grain properties, `GrainBoundaryCount`
/// among them, in `f.tables["grains"]`. Returns the `gb_*` scalars the CLI
/// prints plus that `grains` table (the `--grain-table` content, by property
/// name).
#[pyfunction]
#[pyo3(signature = (f, w_db = None, unassigned_is_boundary = false, min_grain_size = 100,
                    no_adopt_orphans = false))]
fn grain_boundaries(
    py: Python,
    f: FrameArg<'_>,
    w_db: Option<f64>,
    unassigned_is_boundary: bool,
    min_grain_size: usize,
    no_adopt_orphans: bool,
) -> PyResult<PyObject> {
    let f: &Frame = &f;
    let a = GrainBoundaries {
        w_db,
        unassigned_is_boundary,
    };
    // Validated before anything is built, as the CLI does.
    let width = w_db.unwrap_or(rmda_core::analysis::grain_boundaries::DEFAULT_W_DB);
    if !width.is_finite() || width <= 0.0 {
        return Err(PyValueError::new_err(
            "grain_boundaries: w_db must be positive and finite",
        ));
    }
    let ptm = Ptm {
        rmsd_cutoff: 0.1,
        flags: check::FCC | check::HCP | check::BCC | check::ICO | check::SC,
    };
    let grains = Grains {
        misorientation_threshold: 10.0,
        cutoff: None,
        min_size: 1,
        min_grain_size,
        stacking_faults: false,
        include_other: false,
        merge_threshold: Some(10.0),
        orphan_adoption: !no_adopt_orphans,
    };
    let inner = &f.inner;
    let par = par();
    let (out, summary) = py
        .allow_threads(|| -> rmda_core::error::Result<_> {
            let radius = max_cut(ptm.cutoffs_for(inner));
            let cl = CellList::build(&inner.cell, &inner.particles.positions, radius.max(width))?;
            let nk = NeighborhoodContext { cell: Some(&cl) };
            let ctx = AnalysisContext {
                frame: inner,
                nkeys: &nk,
                par: &par,
            };
            let mut out = PropertyStore::new();
            let mut summary = Summary::default();
            ptm.run(&ctx, &mut out, &mut summary)?;
            grains.run(&ctx, &mut out, &mut summary)?;
            a.run(&ctx, &mut out, &mut summary)?;
            Ok((out, summary))
        })
        .map_err(to_py)?;
    absorb(&out, f);
    let d = PyDict::new(py);
    for key in GB_KEYS {
        if let Some((_, v)) = summary.scalars.iter().find(|(k, _)| k == key) {
            d.set_item(*key, *v)?;
        }
    }
    if let Some(t) = f.tables().get("grains") {
        d.set_item("grains", t.to_dict(py)?)?;
    }
    d.into_py_any(py)
}

/// `region=` as `rmda void --region` takes it: six numbers (a sequence or the
/// CLI's comma string), `xlo,xhi,ylo,yhi,zlo,zhi`, with lo < hi on each axis.
fn void_region(v: &[f64]) -> PyResult<[(f64, f64); 3]> {
    if v.len() != 6 {
        return Err(PyValueError::new_err(format!(
            "void: region needs 6 values, xlo,xhi,ylo,yhi,zlo,zhi; got {}",
            v.len()
        )));
    }
    let r = [(v[0], v[1]), (v[2], v[3]), (v[4], v[5])];
    for (lo, hi) in r {
        if !(lo.is_finite() && hi.is_finite()) || hi <= lo {
            return Err(PyValueError::new_err(format!(
                "void: region needs lo < hi on every axis, got [{lo}, {hi}]"
            )));
        }
    }
    Ok(r)
}

/// A list of numbers, or the CLI's comma-separated string of them.
#[derive(FromPyObject)]
enum Floats {
    #[pyo3(annotation = "str")]
    Text(String),
    #[pyo3(annotation = "list[float]")]
    List(Vec<f64>),
}

impl Floats {
    fn values(self, what: &str) -> PyResult<Vec<f64>> {
        match self {
            Floats::List(v) => Ok(v),
            Floats::Text(s) => s
                .split(',')
                .enumerate()
                .map(|(i, part)| {
                    let t = part.trim();
                    t.parse::<f64>().map_err(|e| {
                        PyValueError::new_err(format!(
                            "void: {what} value {} ({t:?}) is not a number: {e}",
                            i + 1
                        ))
                    })
                })
                .collect(),
        }
    }
}

/// Box interior inset by one nearest-neighbour distance (half a box when the
/// box is smaller than two bonds) -- `rmda void`'s default region.
fn void_default_region(cell: &UnitCell, nn: f64) -> [(f64, f64); 3] {
    let (lo, hi) = cell.bounds();
    let axis = |d: usize| {
        let inset = nn.min((hi[d] - lo[d]) / 4.0);
        (lo[d] + inset, hi[d] - inset)
    };
    [axis(0), axis(1), axis(2)]
}

/// Method A of `rmda.void`: the defects census with default settings, then
/// voids as vacancy clusters (design §17), as `rmda void` prints and tables them.
#[allow(clippy::too_many_arguments)]
fn void_a(
    py: Python,
    f: &Frame,
    site_reference: &str,
    min_sites: usize,
    cutoff: Option<f64>,
    regions: Vec<String>,
    grain_column: Option<String>,
    segment: bool,
    tracker: Option<Bound<'_, GrainTracker>>,
) -> PyResult<PyObject> {
    use rmda_core::analysis::defects::voids::{self, VoidClass, VoidConfig};
    let reference = match site_reference {
        "auto" => ReferenceMode::Auto,
        "global" => ReferenceMode::Global,
        "patch" => ReferenceMode::Patch,
        "local" => ReferenceMode::Local,
        "periodic" => ReferenceMode::Periodic,
        other => {
            return Err(PyValueError::new_err(format!(
                "void: unknown site_reference '{other}' (expected auto|global|patch|local|periodic)"
            )))
        }
    };
    let source = match (regions.is_empty(), &grain_column) {
        (false, Some(_)) => {
            return Err(PyValueError::new_err(
                "void: give one of regions or grain_column, not both",
            ))
        }
        (false, None) => GrainSource::Regions(
            regions
                .iter()
                .map(|r| Selection::parse(r).map_err(|e| PyValueError::new_err(e.to_string())))
                .collect::<PyResult<_>>()?,
        ),
        (true, Some(name)) => GrainSource::Labels(column_labels(f, name)?),
        (true, None) => GrainSource::Whole,
    };
    let settings = FrameSettings {
        source,
        segment: segment.then(SegConfig::default),
        ptm: PtmSettings {
            rmsd_cutoff: 0.1,
            dhcp: false,
            order: NeighbourOrder::Distance,
        },
        defects: DefectsConfig {
            reference,
            ..Default::default()
        },
        cluster_cutoff: None,
        antisites: false,
        elements: Vec::new(),
        current_ptm: false,
    };
    let inner = &f.inner;
    let par = par();
    let cfg = VoidConfig { min_sites, cutoff };
    let mut guard = tracker.as_ref().map(|t| t.borrow_mut());
    let tr: Option<&mut Tracker> = guard.as_mut().map(|g| &mut g.inner);
    let (out, found) = py
        .allow_threads(|| -> rmda_core::error::Result<_> {
            let out = defects_frame(inner, &settings, None, tr, &par)?;
            let v = voids::find(
                &inner.cell,
                &inner.particles.positions,
                &out.result,
                &cfg,
                &par,
            )?;
            Ok((out, v))
        })
        .map_err(to_py)?;
    let r = &out.result;
    let d = PyDict::new(py);
    let measured = r
        .grains
        .iter()
        .filter(|g| g.status == Status::Measured)
        .count();
    // As on the CLI's stderr: an unmeasured grain's vacancies are not counted.
    for g in r.grains.iter().filter(|g| g.status != Status::Measured) {
        PyErr::warn(
            py,
            &py.get_type::<pyo3::exceptions::PyUserWarning>(),
            &std::ffi::CString::new(format!(
                "void: grain {} not measured ({}); its vacancies are not counted",
                g.grain,
                g.status.label()
            ))
            .expect("no NUL in the message"),
            1,
        )?;
    }
    d.set_item("void_grains", r.grains.len() as i64)?;
    d.set_item("void_grains_measured", measured as i64)?;
    d.set_item("void_vacancies", r.vacancies.len() as i64)?;
    d.set_item("void_min_sites", found.min_sites as i64)?;
    d.set_item("void_cutoff", found.cutoff)?;
    d.set_item("voids", found.voids.len() as i64)?;
    for c in [VoidClass::Bulk, VoidClass::Boundary, VoidClass::Surface] {
        d.set_item(format!("voids_{}", c.label()), found.count(c) as i64)?;
    }
    d.set_item("void_sites", found.n_sites() as i64)?;
    d.set_item("void_isolated_vacancies", found.n_isolated as i64)?;
    d.set_item("void_volume", found.volume())?;
    let vs = &found.voids;
    d.set_item(
        "void_largest_sites",
        vs.first().map_or(0, |w| w.n_sites as i64),
    )?;
    d.set_item("void_largest_radius", vs.first().map_or(0.0, |w| w.radius))?;
    let t = PyDict::new(py);
    t.set_item(
        "void",
        PyArray1::from_vec(py, vs.iter().map(|v| v.id as i64).collect()),
    )?;
    t.set_item(
        "sites",
        PyArray1::from_vec(py, vs.iter().map(|v| v.n_sites as i64).collect()),
    )?;
    t.set_item(
        "volume",
        PyArray1::from_vec(py, vs.iter().map(|v| v.volume).collect()),
    )?;
    t.set_item(
        "radius",
        PyArray1::from_vec(py, vs.iter().map(|v| v.radius).collect()),
    )?;
    for (k, name) in ["x", "y", "z"].iter().enumerate() {
        t.set_item(
            *name,
            PyArray1::from_vec(py, vs.iter().map(|v| v.centre[k]).collect()),
        )?;
    }
    t.set_item(
        "grain",
        PyArray1::from_vec(py, vs.iter().map(|v| v.grain as i64).collect()),
    )?;
    t.set_item(
        "stable_grain",
        PyArray1::from_vec(py, vs.iter().map(|v| out.stable_id(v.grain)).collect()),
    )?;
    t.set_item(
        "multi_grain",
        PyArray1::from_vec(py, vs.iter().map(|v| v.multi_grain).collect()),
    )?;
    t.set_item(
        "class",
        vs.iter().map(|v| v.class.label()).collect::<Vec<_>>(),
    )?;
    d.set_item("voids_table", t)?;
    d.set_item("grain_events", events_dict(py, &out.events)?)?;
    d.into_py_any(py)
}

/// `rmda void` (design §17). Without `experimental=True` this is **method A**:
/// voids are clusters of the vacancy sites of the `rmda.defects` census, with
/// `site_reference` (`auto|global|patch|local|periodic`), `regions`/`grain_column`/
/// `segment` for grains, `min_sites` (default 2) and `cutoff` (Å, default each
/// grain's shell cutoff). It returns the CLI's `void_*`/`voids*` scalars and a
/// `voids_table` dict of numpy columns (`void`, `sites`, `volume`, `radius`, `x`,
/// `y`, `z`, `grain`, `multi_grain`, `class`), as `rmda void --table` writes.
///
/// With `experimental=True` it is method B, the phi-field volume -- still
/// unvalidated: on a thermal crystal it measures a shell around every
/// displaced atom rather than porosity. The rest of this describes method B.
///
/// Keywords are the CLI flags: `lattice`/`nn_distance` give the ideal cells,
/// or `reference` (a path) fits them to a reference frame; `map`
/// (`"none"`/`"to-reference"`) carries positions into the reference cell;
/// `region` (six numbers or `"xlo,xhi,..."`, default the box inset by one
/// bond), `samples`, `seed`, `epsilon`; `epsilons` (3+ distinct values)
/// extrapolates to phi = 1; `strain` with `strain_spacing`/`strain_sigma`
/// fits design §17.2's roto-strain field; `excess` reports the excess over the
/// reference's own field.
///
/// Returns what the CLI prints, as numbers: `face_normals`, `circumradius`,
/// `observed_nn`, `nn_drift`, `atoms`, `epsilon`, `region`, `region_volume`,
/// with `reference` the fitted `reference_a`/`reference_orientation`, with
/// `strain` the `strain_*` mesh figures, then either `void_volume`,
/// `void_standard_error`, `void_hits`, `void_samples`, `void_fraction` (plus
/// `extrapolated_*`, `slope_negative`, `trustworthy` and the per-epsilon
/// `epsilon_*` lists with `epsilons`), or with `excess` the `excess_*` figures
/// and per-epsilon lists. With `thermal_window` (the CLI's `--thermal-window`)
/// the volume is also measured across the shell-free window above the frame's
/// largest face gap and extrapolated to phi = 1: `face_gap`, `face_gap_bonds`,
/// `window_volume`, `window_standard_error`, `window_coefficients` (alpha, beta),
/// `window_residual` and the per-epsilon `epsilons`/`window_*` lists. A
/// sample/nn mismatch over 10 % is a Python warning, as it is a warning on the
/// CLI's stderr.
#[allow(clippy::too_many_arguments)]
#[pyfunction]
#[pyo3(signature = (f, experimental = false, lattice = "fcc", nn_distance = 2.86,
                    reference = None, map = "none", region = None, samples = 40_000,
                    seed = 11, epsilon = 0.01, epsilons = None, strain = false,
                    strain_spacing = None, strain_sigma = None, excess = false,
                    thermal_window = false, site_reference = "auto", min_sites = 2, cutoff = None,
                    regions = Vec::new(), grain_column = None, segment = false,
                    tracker = None))]
fn void(
    py: Python,
    f: FrameArg<'_>,
    experimental: bool,
    lattice: &str,
    nn_distance: f64,
    reference: Option<String>,
    map: &str,
    region: Option<Floats>,
    samples: usize,
    seed: u64,
    epsilon: f64,
    epsilons: Option<Floats>,
    strain: bool,
    strain_spacing: Option<f64>,
    strain_sigma: Option<f64>,
    excess: bool,
    thermal_window: bool,
    site_reference: &str,
    min_sites: usize,
    cutoff: Option<f64>,
    regions: Vec<String>,
    grain_column: Option<String>,
    segment: bool,
    tracker: Option<Bound<'_, GrainTracker>>,
) -> PyResult<PyObject> {
    use rmda_core::analysis::roto_strain::{RotoStrainField, RotoStrainOptions};
    use rmda_core::analysis::void_b::{
        circumradius, estimate_void_volume, excess_void_volume, extrapolate_shell_free,
        extrapolate_to_phi_one, face_normals, max_face_gap, window_epsilons, PhiField,
    };
    use rmda_core::lattice::fit::fit_region;

    let f: &Frame = &f;
    if !experimental {
        let b_given = lattice != "fcc"
            || nn_distance != 2.86
            || reference.is_some()
            || map != "none"
            || region.is_some()
            || samples != 40_000
            || seed != 11
            || epsilon != 0.01
            || epsilons.is_some()
            || strain
            || strain_spacing.is_some()
            || strain_sigma.is_some()
            || excess
            || thermal_window;
        if b_given {
            return Err(PyValueError::new_err(
                "void: lattice/nn_distance/reference/map/region/samples/seed/epsilon(s)/\
                 strain*/excess/thermal_window are method B's and need experimental=True",
            ));
        }
        return void_a(
            py,
            f,
            site_reference,
            min_sites,
            cutoff,
            regions,
            grain_column,
            segment,
            tracker,
        );
    }
    if site_reference != "auto"
        || min_sites != 2
        || cutoff.is_some()
        || !regions.is_empty()
        || grain_column.is_some()
        || segment
        || tracker.is_some()
    {
        return Err(PyValueError::new_err(
            "void: site_reference/min_sites/cutoff/regions/grain_column/segment/tracker are \
             method A's \
             and do not apply with experimental=True",
        ));
    }
    // The CLI's argument checks, before any time is spent.
    if samples == 0 {
        return Err(PyValueError::new_err(
            "void: samples must be at least 1: the integral has nothing to average otherwise",
        ));
    }
    if !epsilon.is_finite() || epsilon < 0.0 {
        return Err(PyValueError::new_err(format!(
            "void: epsilon must be finite and non-negative, got {epsilon}"
        )));
    }
    if excess && reference.is_none() {
        return Err(PyValueError::new_err(
            "void: excess needs reference: an excess is a difference between two fields",
        ));
    }
    let to_reference = match map {
        "none" => false,
        "to-reference" | "to_reference" => true,
        other => {
            return Err(PyValueError::new_err(format!(
                "void: unknown map '{other}' (expected none|to-reference)"
            )))
        }
    };
    if reference.is_none() && to_reference {
        return Err(PyValueError::new_err(
            "void: map only applies with reference",
        ));
    }
    if (strain_spacing.is_some() || strain_sigma.is_some()) && !strain {
        return Err(PyValueError::new_err(
            "void: strain_spacing/strain_sigma only apply with strain",
        ));
    }
    if thermal_window && (epsilons.is_some() || excess || strain) {
        return Err(PyValueError::new_err(
            "void: thermal_window chooses its own thresholds and cannot be combined with \
             epsilons, excess or strain",
        ));
    }
    let eps_list = match epsilons {
        Some(e) => {
            let eps = e.values("epsilons")?;
            if eps.iter().any(|e| !e.is_finite()) {
                return Err(PyValueError::new_err(format!(
                    "void: epsilons must be finite, got {eps:?}"
                )));
            }
            if eps.len() < 3 {
                return Err(PyValueError::new_err(format!(
                    "void: epsilons needs at least 3 values to fit a quadratic, got {}",
                    eps.len()
                )));
            }
            let mut sorted = eps.clone();
            sorted.sort_by(|x, y| x.total_cmp(y));
            sorted.dedup_by(|x, y| x == y);
            if sorted.len() != eps.len() {
                return Err(PyValueError::new_err(format!(
                    "void: epsilons must be distinct, got {eps:?}"
                )));
            }
            Some(eps)
        }
        None => None,
    };
    let region = match region {
        Some(r) => Some(void_region(&r.values("region")?)?),
        None => None,
    };
    let nn = nn_distance;
    let lat = rmda_core::lattice::from_nn(lattice, nn).ok_or_else(|| {
        PyValueError::new_err(format!(
            "void: unknown lattice '{lattice}' (expected fcc|bcc|hcp|sc|dia|dhcp)"
        ))
    })?;
    let ref_frame = match &reference {
        Some(p) => Some(
            py.allow_threads(|| rmda_core::io::read_frame(std::path::Path::new(p)))
                .map_err(to_py)?,
        ),
        None => None,
    };

    let d = PyDict::new(py);
    let inner = &f.inner;
    /// What one measurement found, before it becomes Python.
    enum Measured {
        Void(
            rmda_core::analysis::void_b::VoidEstimate,
            Option<rmda_core::analysis::void_b::ExtrapolatedVoid>,
            Option<rmda_core::analysis::void_b::WindowVoid>,
        ),
        Excess(rmda_core::analysis::void_b::ExcessVoid, Vec<f64>),
    }
    type StrainFigures = (f64, f64, usize, usize, f64, usize);
    let (fit, n_normals, rcirc, measured_nn, n_atoms, reg, strain_figs, m) = py
        .allow_threads(|| -> rmda_core::error::Result<_> {
            let atoms = &inner.particles.positions[..inner.natoms()];
            // The ideal configuration: fitted to the reference, else the
            // declared lattice.
            let (pc, fit) = match &ref_frame {
                Some(rf) => {
                    let fit = fit_region(
                        &rf.cell,
                        &rf.particles.positions[..rf.natoms()],
                        lat.as_ref(),
                        nn,
                        &FitOptions::default(),
                    )?;
                    let a = fit.cell.h.col_norm(0);
                    let q = fit.orientation.to_vec();
                    (fit.cell, Some((a, q)))
                }
                None => (lat.primitive(), None),
            };
            let normals = face_normals(&pc)?;
            let rcirc = circumradius(&normals)?;
            let measured = rmda_core::lattice::fit::estimate_nn(&inner.cell, atoms, nn)?;
            let (field_cell, field_atoms): (UnitCell, Vec<[f64; 3]>) =
                match (ref_frame.as_ref(), to_reference) {
                    (Some(rf), true) => {
                        let lam = atoms
                            .iter()
                            .map(|&p| inner.cell.x2lamda(p))
                            .collect::<rmda_core::error::Result<Vec<_>>>()?;
                        (rf.cell, lam.iter().map(|&l| rf.cell.lamda2x(l)).collect())
                    }
                    _ => (inner.cell, atoms.to_vec()),
                };
            let n_atoms = field_atoms.len();
            let (strain_field, strain_figs): (Option<RotoStrainField>, Option<StrainFigures>) =
                if strain {
                    let a0 = pc.h.col_norm(0);
                    let opts = RotoStrainOptions {
                        node_spacing: strain_spacing.unwrap_or(a0 / 2.0),
                        sigma: strain_sigma.unwrap_or(a0),
                        refine_passes: 1,
                        max_deviation: None,
                        fill_unsolved: true,
                    };
                    let s = RotoStrainField::fit(&inner.cell, atoms, &pc, &opts)?;
                    let figs = (
                        opts.node_spacing,
                        opts.sigma,
                        s.len_nodes(),
                        s.n_pairs(),
                        s.max_deviation_from_identity(),
                        s.unsolved(),
                    );
                    (Some(s), Some(figs))
                } else {
                    (None, None)
                };
            // Searched for the largest threshold measured (`PhiField::max_epsilon`).
            let face_gap = if thermal_window {
                Some(max_face_gap(&field_cell, &field_atoms, &pc)?)
            } else {
                None
            };
            let window = match face_gap {
                Some(g) => window_epsilons(g.max)?,
                None => Vec::new(),
            };
            let eps_field = eps_list
                .iter()
                .flatten()
                .chain(&window)
                .copied()
                .fold(epsilon, f64::max);
            let field = PhiField::for_threshold_strained(
                &field_cell,
                &field_atoms,
                &pc,
                eps_field,
                strain_field,
            )?;
            let reg = region.unwrap_or_else(|| void_default_region(&field_cell, nn));
            let m = if excess {
                let rf = ref_frame.as_ref().expect("checked: excess needs reference");
                let ref_positions = rf.particles.positions[..rf.natoms()].to_vec();
                let ref_field = PhiField::for_threshold(&rf.cell, &ref_positions, &pc, eps_field)?;
                let eps = eps_list.clone().unwrap_or_else(|| vec![epsilon]);
                let x = excess_void_volume(&field, &ref_field, &reg, &eps, samples, seed)?;
                Measured::Excess(x, eps)
            } else {
                let est = estimate_void_volume(&field, &reg, epsilon, samples, seed)?;
                let x = match &eps_list {
                    Some(eps) => Some(extrapolate_to_phi_one(&field, &reg, eps, samples, seed)?),
                    None => None,
                };
                let w = match face_gap {
                    Some(g) => Some(extrapolate_shell_free(&field, &reg, g, samples, seed)?),
                    None => None,
                };
                Measured::Void(est, x, w)
            };
            Ok((
                fit,
                normals.len(),
                rcirc,
                measured,
                n_atoms,
                reg,
                strain_figs,
                m,
            ))
        })
        .map_err(to_py)?;

    if let Some((a, q)) = fit {
        d.set_item("reference_a", a)?;
        d.set_item("reference_orientation", q)?;
    }
    d.set_item("face_normals", n_normals)?;
    d.set_item("circumradius", rcirc)?;
    d.set_item("observed_nn", measured_nn)?;
    let drift = (measured_nn - nn) / nn;
    d.set_item("nn_drift", drift)?;
    if drift.abs() > 0.10 {
        let warning = py.get_type::<pyo3::exceptions::PyUserWarning>();
        PyErr::warn(
            py,
            &warning,
            &std::ffi::CString::new(format!(
                "void: nn_distance {nn:.4} A differs from the sample's measured {measured_nn:.4} A \
                 by {:+.1} %; the void volume then measures that mismatch rather than void",
                100.0 * drift
            ))
            .expect("no NUL in the message"),
            1,
        )?;
    }
    d.set_item("atoms", n_atoms)?;
    d.set_item("epsilon", epsilon)?;
    d.set_item(
        "region",
        reg.iter()
            .flat_map(|&(lo, hi)| [lo, hi])
            .collect::<Vec<_>>(),
    )?;
    d.set_item(
        "region_volume",
        reg.iter()
            .map(|(lo, hi)| (hi - lo).max(0.0))
            .product::<f64>(),
    )?;
    if let Some((spacing, sigma, nodes, pairs, dev, unsolved)) = strain_figs {
        d.set_item("strain_spacing", spacing)?;
        d.set_item("strain_sigma", sigma)?;
        d.set_item("strain_nodes", nodes)?;
        d.set_item("strain_pairs", pairs)?;
        d.set_item("strain_max_deviation", dev)?;
        d.set_item("strain_unsolved", unsolved)?;
    }
    let per_eps = |prefix: &str,
                   eps: &[f64],
                   ms: &[rmda_core::analysis::void_b::VoidEstimate]|
     -> PyResult<()> {
        d.set_item("epsilons", eps.to_vec())?;
        d.set_item(
            format!("{prefix}_volumes"),
            ms.iter().map(|m| m.volume).collect::<Vec<_>>(),
        )?;
        d.set_item(
            format!("{prefix}_standard_errors"),
            ms.iter().map(|m| m.standard_error).collect::<Vec<_>>(),
        )?;
        d.set_item(
            format!("{prefix}_hits"),
            ms.iter().map(|m| m.hits).collect::<Vec<_>>(),
        )?;
        Ok(())
    };
    match m {
        Measured::Excess(x, eps) => {
            d.set_item("excess_volume", x.volume_at_one)?;
            d.set_item("excess_standard_error", x.standard_error)?;
            d.set_item(
                "excess_coefficients",
                vec![x.coefficients.0, x.coefficients.1, x.coefficients.2],
            )?;
            per_eps("excess", &eps, &x.measurements)?;
        }
        Measured::Void(est, x, w) => {
            d.set_item("void_volume", est.volume)?;
            d.set_item("void_standard_error", est.standard_error)?;
            d.set_item("void_hits", est.hits)?;
            d.set_item("void_samples", est.samples)?;
            d.set_item("void_fraction", est.volume / est.region_volume)?;
            if let Some(x) = x {
                d.set_item("extrapolated_volume", x.volume_at_one)?;
                d.set_item("extrapolated_standard_error", x.standard_error)?;
                d.set_item("extrapolated_residual", x.residual)?;
                d.set_item(
                    "extrapolated_coefficients",
                    vec![x.coefficients.0, x.coefficients.1, x.coefficients.2],
                )?;
                d.set_item("slope_negative", x.slope_negative())?;
                d.set_item("trustworthy", x.is_trustworthy())?;
                per_eps(
                    "epsilon",
                    eps_list.as_deref().unwrap_or_default(),
                    &x.measurements,
                )?;
            }
            if let Some(w) = w {
                d.set_item("face_gap", w.face_gap.max)?;
                d.set_item("face_gap_bonds", w.face_gap.bonds)?;
                d.set_item("window_volume", w.volume_at_one)?;
                d.set_item("window_standard_error", w.standard_error)?;
                d.set_item(
                    "window_coefficients",
                    vec![w.coefficients.0, w.coefficients.1],
                )?;
                d.set_item("window_residual", w.residual)?;
                per_eps("window", &w.epsilons, &w.measurements)?;
            }
        }
    }
    d.into_py_any(py)
}

/// Convert a frame into an ASE `Atoms` (design §22) — the mirror of
/// `rmda.from_ase`, the direction OVITO spells `ovito_to_ase`. This one does
/// import ASE: a `Frame` is only built back into Python objects.
///
/// Type ids become atomic numbers through `type_labels` and
/// `ase.data.atomic_numbers`; an id with no label, or a label that is not an
/// element, falls back to the raw type id (what OVITO does for non-symbol
/// type names). Nothing is lost by that: when it happens, the type ids go to
/// `arrays["type"]` and the names to `info["type_names"]`, which `from_ase`
/// reads back. Ids other than 1..N go to `arrays["id"]`. The cell, its pbc
/// flags and a non-zero origin go through as they are; every input column
/// and attached per-atom property (an attached one wins on a shared name)
/// becomes an `atoms.arrays` entry. Arrays go across as numpy arrays.
#[pyfunction]
fn to_ase(py: Python, f: &Frame) -> PyResult<PyObject> {
    let ase = PyModule::import(py, "ase").map_err(|_| {
        PyImportError::new_err("rmda.to_ase needs the ASE package (pip install ase)")
    })?;
    let data = PyModule::import(py, "ase.data").map_err(|_| {
        PyImportError::new_err("rmda.to_ase needs the ASE package (pip install ase)")
    })?;
    let atomic_numbers = data.getattr("atomic_numbers")?;

    let inner = &f.inner;
    // One lookup per type, not per atom; `None` where the label is no element.
    let z_lookup: Vec<Option<i64>> = (0..=inner.ntypes() as u32)
        .map(|ty| {
            (ty >= 1)
                .then(|| inner.type_labels.get(ty as usize - 1))
                .flatten()
                .and_then(|label| atomic_numbers.get_item(label.as_str()).ok())
                .and_then(|z| z.extract::<i64>().ok())
        })
        .collect();
    let types = &inner.particles.types[..inner.natoms()];
    let mut used = vec![false; z_lookup.len()];
    types.iter().for_each(|&t| used[t as usize] = true);
    let lossy = used
        .iter()
        .zip(&z_lookup)
        .skip(1)
        .any(|(&u, z)| u && z.is_none());
    let z_of_type: Vec<i64> = z_lookup
        .iter()
        .enumerate()
        .map(|(ty, z)| z.unwrap_or(ty as i64))
        .collect();
    let numbers: Vec<i64> = inner.particles.types[..inner.natoms()]
        .iter()
        .map(|&ty| z_of_type.get(ty as usize).copied().unwrap_or(ty as i64))
        .collect();

    let kwargs = PyDict::new(py);
    kwargs.set_item("numbers", PyArray1::from_vec(py, numbers))?;
    kwargs.set_item(
        "positions",
        Array2::from(inner.particles.positions[..inner.natoms()].to_vec()).into_pyarray(py),
    )?;
    kwargs.set_item(
        "cell",
        Array2::from(inner.cell.h_matrix().m.to_vec()).into_pyarray(py),
    )?;
    kwargs.set_item("pbc", inner.cell.pbc.to_vec())?;
    let info = PyDict::new(py);
    if inner.cell.origin != [0.0, 0.0, 0.0] {
        info.set_item("cell_origin", inner.cell.origin.to_vec())?;
    }
    if lossy {
        let names = PyDict::new(py);
        for (i, label) in inner.type_labels.iter().enumerate() {
            if !label.is_empty() {
                names.set_item(i + 1, label)?;
            }
        }
        info.set_item("type_names", names)?;
    }
    if !info.is_empty() {
        kwargs.set_item("info", info)?;
    }
    let atoms = ase.getattr("Atoms")?.call((), Some(&kwargs))?;
    if lossy {
        atoms.call_method1(
            "new_array",
            (
                "type",
                PyArray1::from_iter(py, types.iter().map(|&t| t as i64)),
            ),
        )?;
    }
    let ids = &inner.particles.ids[..inner.natoms()];
    if ids.iter().enumerate().any(|(i, &id)| id != i as i64 + 1) {
        atoms.call_method1("new_array", ("id", PyArray1::from_slice(py, ids)))?;
    }

    // Input columns, then attached per-atom properties (which win on a
    // shared name), as numpy arrays under their own names.
    let props = f.attached();
    let mut columns: BTreeMap<String, PyObject> = BTreeMap::new();
    for p in inner.particles.props.iter().filter(|p| p.per_particle) {
        columns.insert(p.name.to_string(), core_prop_array(py, p)?);
    }
    for (name, ap) in props.iter() {
        columns.insert(name.clone(), prop_array(py, ap)?);
    }
    for (name, arr) in columns {
        if !ASE_RESERVED.contains(&name.as_str()) {
            atoms.call_method1("new_array", (name, arr))?;
        }
    }
    atoms.into_py_any(py)
}

/// An ordered, reusable analysis pipeline over a frame stream (design §22:
/// `p = rmda.Pipeline(rmda.iread(path)); p.add("ptm"); p.add("segment"); p.run()`;
/// `add` returns `None`, so calls do not chain). Steps run
/// in the order added, each absorbing its per-atom properties into the frame,
/// so a later step sees an earlier step's output the way `rmda analyze` does.
#[pyclass]
struct Pipeline {
    frames: Py<PyFrameIter>,
    steps: Vec<(String, HashMap<String, Kw>)>,
}

#[pymethods]
impl Pipeline {
    #[new]
    fn new(frames: Py<PyFrameIter>) -> Self {
        Pipeline {
            frames,
            steps: Vec::new(),
        }
    }

    #[pyo3(signature = (name, kwargs = None))]
    fn add(&mut self, name: &str, kwargs: Option<&Bound<'_, PyDict>>) -> PyResult<()> {
        if !STEPS.contains(&name) {
            return Err(PyValueError::new_err(format!(
                "unknown analysis step '{name}' (expected one of {STEPS:?})"
            )));
        }
        self.steps.push((name.to_string(), collect_kwargs(kwargs)?));
        Ok(())
    }

    /// Run every step on every frame, lazily: returns an iterator that reads
    /// the next frame, runs the steps on it and yields it, so only the frame
    /// in hand is resident (design §22). `list(pipe.run())` collects them.
    /// The frame stream is shared: a second `run()` continues where the first
    /// stopped.
    fn run(&self, py: Python) -> PipelineRun {
        PipelineRun {
            frames: self.frames.clone_ref(py),
            steps: self.steps.clone(),
        }
    }
}

/// The generator `Pipeline.run` returns: one frame per `next()`, with every
/// step's properties and tables attached.
#[pyclass]
struct PipelineRun {
    frames: Py<PyFrameIter>,
    steps: Vec<(String, HashMap<String, Kw>)>,
}

#[pymethods]
impl PipelineRun {
    fn __iter__(slf: PyRef<Self>) -> PyRef<Self> {
        slf
    }

    fn __next__(&self, py: Python) -> PyResult<Option<Frame>> {
        let next = self.frames.borrow_mut(py).__next__(py)?;
        let Some(f) = next else {
            return Ok(None);
        };
        for (name, kw) in &self.steps {
            apply_step(py, &f, name, kw)?;
        }
        Ok(Some(f))
    }
}

/// The `rmda` command line, run in this process on `argv` (the program name
/// first); it exits the interpreter with the command's status. The `rmda`
/// console script a pip install puts on `PATH` calls it (`rmda/__init__.py`).
#[pyfunction]
#[pyo3(name = "_cli_main")]
fn cli_main(py: Python<'_>, argv: Vec<String>) -> PyResult<()> {
    py.allow_threads(|| rmda_cli::main_with_args(argv))
}

/// The extension module, `rmda._rmda`; the `rmda` package re-exports it.
#[pymodule]
#[pyo3(name = "_rmda")]
fn rmda(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    m.add_function(wrap_pyfunction!(cli_main, m)?)?;
    m.add_class::<Frame>()?;
    m.add_class::<PyFrameIter>()?;
    m.add_class::<Pipeline>()?;
    m.add_class::<GrainTracker>()?;
    m.add_function(wrap_pyfunction!(read, m)?)?;
    m.add_function(wrap_pyfunction!(iread, m)?)?;
    m.add_function(wrap_pyfunction!(from_ase, m)?)?;
    m.add_function(wrap_pyfunction!(to_ase, m)?)?;
    m.add_function(wrap_pyfunction!(set_threads, m)?)?;
    m.add_function(wrap_pyfunction!(get_threads, m)?)?;
    m.add_function(wrap_pyfunction!(cna, m)?)?;
    m.add_function(wrap_pyfunction!(ptm, m)?)?;
    m.add_function(wrap_pyfunction!(rdf, m)?)?;
    m.add_function(wrap_pyfunction!(cluster, m)?)?;
    // `rmda.clusters` is an alias of `rmda.cluster`, the CLI's subcommand
    // name.
    m.add("clusters", m.getattr("cluster")?)?;
    m.add_function(wrap_pyfunction!(warren_cowley, m)?)?;
    m.add_function(wrap_pyfunction!(ws, m)?)?;
    m.add_function(wrap_pyfunction!(segment, m)?)?;
    m.add_function(wrap_pyfunction!(defects_py, m)?)?;
    m.add_function(wrap_pyfunction!(grain_boundaries, m)?)?;
    m.add_function(wrap_pyfunction!(void, m)?)?;
    m.add_class::<PipelineRun>()?;
    Ok(())
}
