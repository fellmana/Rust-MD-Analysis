//! The per-grain table of `rmda grain-boundaries --grain-table`, as CSV.
//!
//! Grain segmentation keeps its per-grain quantities in the [`PropertyStore`]
//! alongside the per-atom ones, which works for the library but not for output:
//! those properties are `per_particle = false`, one row per *grain*, and every
//! atom writer filters on that flag. So an atom file can only carry the
//! per-atom `Grain` label, and a caller who wants the grain inventory has to
//! re-derive it from a size histogram.
//!
//! This module writes that inventory out. One row per grain, ordered by grain id,
//! which is the same numbering the per-atom `Grain` column uses -- so a row here
//! and a label there mean the same grain.
//!
//! # Why CSV
//!
//! Because the consumer is a person or a spreadsheet, not another frame reader.
//! Every other format in the crate is a positional atom format where a column
//! has a fixed meaning and rows are atoms; a grain table has neither property,
//! and re-expressing it as an extended XYZ with one "atom" per grain would put
//! a 3-component fake position where a real centre of mass belongs.
//!
//! # Self-describing header
//!
//! The `misorientation` column is degrees relative to *one* grain, so a file
//! without that grain's id is not interpretable. Rather than repeat the
//! reference in every row, the scalars the columns depend on are written as
//! `#` comment lines above the header. `pandas.read_csv(..., comment='#')`
//! skips them; a hand edit does not.
//!
//! ```
//! # use rmda_core::analysis::grain_table::*;
//! # use rmda_core::data::property::PropertyStore;
//! # use rmda_core::analysis::Summary;
//! let store = PropertyStore::new();
//! let summary = Summary::default();
//! // A store with no `GrainSize` is an error, not an empty file.
//! let e = write_grain_table(&store, &summary, &GrainTableOptions::default()).unwrap_err();
//! assert!(e.to_string().contains("GrainSize"));
//! ```

use std::io::Write;
use std::path::PathBuf;

use crate::analysis::ptm::structure;
use crate::analysis::Summary;
use crate::data::property::{Property, PropertyStore};
use crate::error::{Error, Result};

/// Column names, in file order. Quaternions are `(w, x, y, z)`.
pub const COLUMNS: &[&str] = &[
    "grain",
    "size",
    "structure",
    "lattice",
    "cx",
    "cy",
    "cz",
    "qw",
    "qx",
    "qy",
    "qz",
    "misorientation",
];

/// Columns appended only when the analysis that produces them has run.
///
/// A column is here rather than in [`COLUMNS`] because its presence is a fact about
/// the *run*, not about the format: a file written straight after segmentation has no
/// seam information, and writing an empty or zero column for it would assert that the
/// grain has no boundary atoms, which is a different and wrong claim. Omitting it
/// leaves the absence of the measurement visible.
///
/// The order is the file's column order. `rmda grain-boundaries --grain-table`
/// writes `boundary_atoms`. (Per-grain defect counts are `rmda defects
/// --table`, a table of its own.)
pub const OPTIONAL_COLUMNS: &[(&str, &str, ColKind)] = &[
    // (property name in the store, column header, how to render it)
    ("GrainBoundaryCount", "boundary_atoms", ColKind::I32),
];

/// How an optional column's value is rendered.
#[derive(Debug, Clone, Copy)]
pub enum ColKind {
    /// An integer count, written as stored (`-1` for unmeasured).
    I32,
    /// A float at a fixed precision, or `nan` for unmeasured.
    F64(usize),
}

/// How to write the table.
#[derive(Debug, Clone)]
pub struct GrainTableOptions {
    /// File to write. Parent directories are created if absent.
    pub path: PathBuf,
    /// Digits after the decimal point for the float columns.
    ///
    /// Defaults to 6. Coordinates are Å and misorientations are degrees, so 6
    /// digits is well past any physical resolution here while keeping two runs
    /// of the same analysis byte-identical.
    pub float_precision: usize,
    /// Write a file even when there are no grains.
    ///
    /// On by default, for the same reason `DefectDumpOptions::write_empty` is:
    /// a fixed filename is easier to script against than one whose existence
    /// depends on the analysis finding anything.
    pub write_empty: bool,
}

impl Default for GrainTableOptions {
    fn default() -> Self {
        Self {
            path: PathBuf::from("grains.csv"),
            float_precision: 6,
            write_empty: true,
        }
    }
}

/// What the writer produced.
#[derive(Debug, Clone, Default)]
pub struct GrainTableReport {
    pub path: PathBuf,
    /// Rows written, i.e. grains. Zero is legitimate for a fully unassigned frame.
    pub grains: usize,
    /// Grain the `misorientation` column is measured against, if any.
    pub reference: Option<i64>,
    /// The file was deliberately not written because it would have been empty.
    pub skipped_empty: bool,
}

impl GrainTableReport {
    /// One line for the CLI to echo.
    pub fn summary(&self) -> String {
        if self.skipped_empty {
            return format!("{} (no grains, not written)", self.path.display());
        }
        match self.reference {
            Some(r) => format!(
                "{}: {} grain(s), misorientation relative to grain {r}",
                self.path.display(),
                self.grains
            ),
            None => format!(
                "{}: {} grain(s), no misorientation reference",
                self.path.display(),
                self.grains
            ),
        }
    }
}

/// Human-readable name for a PTM structure id, or `"?"` if unrecognised.
///
/// Kept next to the writer rather than pushed onto `structure::` because a
/// numeric id with no name is a property of the *table*, not of the
/// classification: `GrainStructureType` holds whatever ids PTM assigned.
fn lattice_name(kind: i32) -> &'static str {
    match kind {
        structure::OTHER => "OTHER",
        structure::FCC => "FCC",
        structure::HCP => "HCP",
        structure::BCC => "BCC",
        structure::ICO => "ICO",
        structure::SC => "SC",
        _ => "?",
    }
}

/// Format a float for CSV, or `nan` when undefined.
///
/// `nan` rather than an empty field: an empty cell reads as "not measured" and
/// `nan` reads as "measured, and the answer does not exist", which is the
/// distinction that matters here -- a grain of non-crystalline atoms has an
/// undefined orientation, and reporting `0` there would claim it is aligned
/// with the reference.
/// One optional column's values, read once per file.
enum Column {
    Ints(Vec<i32>),
    Floats { v: Vec<f64>, prec: usize },
}

fn num(v: f64, prec: usize) -> String {
    if !v.is_finite() {
        return "nan".to_owned();
    }
    let s = format!("{v:.prec$}");
    // Quaternion components come out of the sum as tiny residuals (~1e-17) with
    // either sign, which format as "-0.000000". The sign is numerical noise, but
    // it reaches a file people read and diff, so drop it once the digits are all
    // zero. Anything with a nonzero digit keeps its sign.
    if s.starts_with('-') && s[1..].bytes().all(|b| b == b'0' || b == b'.') {
        s[1..].to_owned()
    } else {
        s
    }
}

fn missing(name: &str) -> Error {
    Error::InvalidData(format!(
        "grain table needs the per-grain property `{name}`, which is not in the store; \
         the grains analysis did not run"
    ))
}

/// Write the per-grain table for the grains in `store` to `opts.path`.
///
/// `summary` supplies the scalars the columns are relative to -- currently the
/// misorientation reference grain and the threshold that produced the grain
/// boundaries. They are recorded but not required: a store without them still
/// writes a table, minus the lines describing them.
///
/// Fails if the per-grain properties are absent, rather than writing an empty
/// file that looks like a frame with no grains. The two are very different
/// problems and a header is the only place to say which happened.
pub fn write_grain_table(
    store: &PropertyStore,
    summary: &Summary,
    opts: &GrainTableOptions,
) -> Result<GrainTableReport> {
    let per_grain = |name: &str| -> Result<&Property> {
        store
            .get(name)
            .filter(|p| !p.per_particle)
            .ok_or_else(|| missing(name))
    };
    let size_p = per_grain("GrainSize")?;
    let centre_p = per_grain("GrainCenter")?;
    let quat_p = per_grain("GrainOrientation")?;
    let kind_p = per_grain("GrainStructureType")?;
    let miso_p = per_grain("GrainMisorientation")?;

    let n = size_p.len();
    // Every per-grain property must have exactly one row per grain. A mismatch
    // means the store was assembled by something other than `Grains::run`, and
    // silently zipping to the shortest would drop grains without saying so.
    for (name, p) in [
        ("GrainCenter", centre_p),
        ("GrainOrientation", quat_p),
        ("GrainStructureType", kind_p),
        ("GrainMisorientation", miso_p),
    ] {
        if p.len() != n {
            return Err(Error::InvalidData(format!(
                "grain table: `{name}` has {} row(s) but `GrainSize` has {n}",
                p.len()
            )));
        }
    }

    let reference = summary
        .scalars
        .iter()
        .find(|(k, _)| k == "grain_reference")
        .map(|(_, v)| *v as i64);
    let prec = opts.float_precision;

    let mut report = GrainTableReport {
        path: opts.path.clone(),
        grains: n,
        reference,
        skipped_empty: false,
    };
    if n == 0 && !opts.write_empty {
        report.skipped_empty = true;
        return Ok(report);
    }

    // Build the body before touching the file, and note why the typed accessors
    // are only reached when there is something to read: a property with zero rows
    // has no typed backing array at all -- `Property::new` starts every dtype as
    // U8 and only `push_value` swaps in the real one -- so on a frame that
    // legitimately has no grains they would report "the grains analysis did not
    // run", which is the opposite of what happened.
    let mut body = String::new();
    if n > 0 {
        let sizes = size_p.as_i32().ok_or_else(|| missing("GrainSize"))?;
        let centre = centre_p
            .as_f64()
            .ok_or_else(|| missing("GrainCenter"))?
            .as_slice();
        let quat = quat_p.as_f64().ok_or_else(|| missing("GrainOrientation"))?;
        let kinds = kind_p
            .as_i32()
            .ok_or_else(|| missing("GrainStructureType"))?;
        let miso = miso_p
            .as_f64()
            .ok_or_else(|| missing("GrainMisorientation"))?;
        // Read each present optional column once, up front, so the row loop below is
        // a format and not a lookup. A column of the wrong dtype is treated as
        // absent rather than fatal: the header would not be written for it either,
        // so writing rows without it keeps the file consistent.
        let optional: Vec<Option<Column>> = OPTIONAL_COLUMNS
            .iter()
            .map(|(prop, _, kind)| match kind {
                ColKind::I32 => store
                    .get(prop)
                    .and_then(|p| p.as_i32())
                    .map(|v| Column::Ints(v.clone())),
                ColKind::F64(prec) => {
                    store
                        .get(prop)
                        .and_then(|p| p.as_f64())
                        .map(|v| Column::Floats {
                            v: v.clone(),
                            prec: *prec,
                        })
                }
            })
            .collect();
        for g in 0..n {
            body.push_str(&format!(
                "{g},{},{},{},{},{},{},{},{},{},{},{}",
                sizes[g],
                kinds[g],
                lattice_name(kinds[g]),
                num(centre[3 * g], prec),
                num(centre[3 * g + 1], prec),
                num(centre[3 * g + 2], prec),
                num(quat[4 * g], prec),
                num(quat[4 * g + 1], prec),
                num(quat[4 * g + 2], prec),
                num(quat[4 * g + 3], prec),
                num(miso[g], prec),
            ));
            // Only the columns that are actually there, and only if they have a row
            // for this grain: a short column would misalign every field after it.
            for col in &optional {
                match col {
                    Some(Column::Ints(v)) if g < v.len() => {
                        body.push(',');
                        body.push_str(&v[g].to_string());
                    }
                    Some(Column::Floats { v, prec }) if g < v.len() => {
                        body.push(',');
                        body.push_str(&num(v[g], *prec));
                    }
                    _ => {}
                }
            }
            body.push('\n');
        }
    }

    if let Some(dir) = opts.path.parent() {
        if !dir.as_os_str().is_empty() {
            std::fs::create_dir_all(dir)
                .map_err(|e| Error::Io(format!("cannot create {}: {e}", dir.display())))?;
        }
    }
    let mut f = std::fs::File::create(&opts.path)
        .map_err(|e| Error::Io(format!("cannot write {}: {e}", opts.path.display())))?;

    writeln!(
        f,
        "# rmda per-grain table (rmda grain-boundaries --grain-table)"
    )?;
    writeln!(f, "# grains: {n}")?;
    match reference {
        Some(r) => writeln!(f, "# misorientation_reference: {r}")?,
        None => writeln!(f, "# misorientation_reference: none")?,
    }
    for key in [
        "grain_misorientation_threshold",
        "grain_cutoff",
        "grain_min_size",
    ] {
        if let Some((_, v)) = summary.scalars.iter().find(|(k, _)| k == key) {
            writeln!(f, "# {key}: {v}")?;
        }
    }
    writeln!(
        f,
        "# nan means undefined: a grain whose atoms carried no orientation, or a type with no point group"
    )?;
    // The header has to say which optional columns are present, since their presence
    // is a property of the run rather than of the format.
    let mut header = COLUMNS.join(",");
    if n > 0 {
        for (prop, col, _) in OPTIONAL_COLUMNS {
            if store.get(prop).is_some() {
                header.push(',');
                header.push_str(col);
            }
        }
    }
    writeln!(f, "{header}")?;
    f.write_all(body.as_bytes())
        .map_err(|e| Error::Io(format!("cannot write {}: {e}", opts.path.display())))?;
    f.flush()
        .map_err(|e| Error::Io(format!("cannot write {}: {e}", opts.path.display())))?;
    Ok(report)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::property::{DType, Property};

    /// The CSV rows, skipping the `#` preamble and the header.
    fn data_rows(text: &str) -> Vec<&str> {
        text.lines()
            .filter(|l| !l.starts_with('#') && !l.starts_with("grain,") && !l.is_empty())
            .collect()
    }

    fn tmp(name: &str) -> PathBuf {
        let d = std::env::temp_dir().join("rmda-grain-table-tests");
        std::fs::create_dir_all(&d).unwrap();
        d.join(name)
    }

    /// A store shaped like the one `Grains::run` leaves behind: two grains, the
    /// first well defined and the reference, the second with no orientation.
    fn store_two_grains() -> (PropertyStore, Summary) {
        let mut s = PropertyStore::new();
        let mut size = Property::new("GrainSize", DType::I32, false, 1);
        size.push_value(&[10.0]);
        size.push_value(&[20.0]);
        s.register(size);

        let mut centre = Property::new("GrainCenter", DType::F64, false, 3);
        centre.push_value(&[1.0, 2.0, 3.0]);
        centre.push_value(&[4.0, 5.0, 6.0]);
        s.register(centre);

        let mut quat = Property::new("GrainOrientation", DType::F64, false, 4);
        quat.push_value(&[1.0, 0.0, 0.0, 0.0]);
        // No orientation at all: grains.rs leaves this NaN rather than zeros.
        quat.push_value(&[f64::NAN; 4]);
        s.register(quat);

        let mut kind = Property::new("GrainStructureType", DType::I32, false, 1);
        kind.push_value(&[structure::FCC as f64]);
        kind.push_value(&[structure::BCC as f64]);
        s.register(kind);

        let mut miso = Property::new("GrainMisorientation", DType::F64, false, 1);
        miso.push_value(&[0.0]);
        miso.push_value(&[f64::NAN]);
        s.register(miso);

        let summary = Summary {
            scalars: vec![
                ("grain_reference".to_owned(), 0.0),
                ("grain_misorientation_threshold".to_owned(), 4.0),
            ],
            ..Default::default()
        };
        (s, summary)
    }

    #[test]
    fn writes_one_row_per_grain_with_a_header() {
        let (store, summary) = store_two_grains();
        let path = tmp("basic.csv");
        let rep = write_grain_table(
            &store,
            &summary,
            &GrainTableOptions {
                path: path.clone(),
                ..Default::default()
            },
        )
        .unwrap();
        assert_eq!(rep.grains, 2);
        assert_eq!(rep.reference, Some(0));
        assert!(rep.summary().contains("relative to grain 0"), "{rep:?}");

        let text = std::fs::read_to_string(&path).unwrap();
        let rows: Vec<&str> = text.lines().filter(|l| !l.starts_with('#')).collect();
        assert_eq!(rows[0], COLUMNS.join(","));
        assert_eq!(rows.len(), 3, "header plus two grains:\n{text}");
        // Grain 0 is well defined; grain 1 has no orientation and must say so.
        assert!(rows[1].starts_with("0,10,1,FCC,"), "{}", rows[1]);
        assert!(rows[1].ends_with(",0.000000"), "{}", rows[1]);
        assert!(rows[2].contains("nan"), "{}", rows[2]);
        assert!(rows[2].contains(",3,BCC,"), "{}", rows[2]);
    }

    #[test]
    fn the_reference_grain_is_recorded_in_the_file() {
        let (store, summary) = store_two_grains();
        let path = tmp("reference.csv");
        write_grain_table(
            &store,
            &summary,
            &GrainTableOptions {
                path: path.clone(),
                ..Default::default()
            },
        )
        .unwrap();
        let text = std::fs::read_to_string(&path).unwrap();
        assert!(
            text.lines().any(|l| l == "# misorientation_reference: 0"),
            "{text}"
        );
    }

    #[test]
    fn a_missing_reference_is_stated_not_left_blank() {
        let (store, _) = store_two_grains();
        let path = tmp("noref.csv");
        write_grain_table(
            &store,
            &Summary::default(),
            &GrainTableOptions {
                path: path.clone(),
                ..Default::default()
            },
        )
        .unwrap();
        let text = std::fs::read_to_string(&path).unwrap();
        assert!(
            text.lines()
                .any(|l| l == "# misorientation_reference: none"),
            "{text}"
        );
    }

    #[test]
    fn a_store_without_grains_is_an_error_not_an_empty_file() {
        let path = tmp("absent.csv");
        let e = write_grain_table(
            &PropertyStore::new(),
            &Summary::default(),
            &GrainTableOptions {
                path: path.clone(),
                ..Default::default()
            },
        )
        .unwrap_err();
        assert!(e.to_string().contains("GrainSize"), "{e}");
        assert!(!path.exists(), "no file should have been created");
    }

    #[test]
    fn ragged_rows_are_refused_rather_than_zipped() {
        let (mut store, summary) = store_two_grains();
        // One row short of the two grains.
        let mut kind = Property::new("GrainStructureType", DType::I32, false, 1);
        kind.push_value(&[structure::FCC as f64]);
        store.register(kind);
        let path = tmp("ragged.csv");
        let e = write_grain_table(
            &store,
            &summary,
            &GrainTableOptions {
                path: path.clone(),
                ..Default::default()
            },
        )
        .unwrap_err();
        assert!(e.to_string().contains("GrainStructureType"), "{e}");
        assert!(!path.exists());
    }

    #[test]
    fn a_per_atom_grain_size_is_refused() {
        let (mut store, summary) = store_two_grains();
        // The per-atom flag is the whole reason this table needs its own writer,
        // so a per-atom property of the same name must not be picked up.
        let mut size = Property::new("GrainSize", DType::I32, true, 1);
        size.push_value(&[10.0]);
        store.register(size);
        let path = tmp("peratom.csv");
        let e = write_grain_table(
            &store,
            &summary,
            &GrainTableOptions {
                path: path.clone(),
                ..Default::default()
            },
        )
        .unwrap_err();
        assert!(e.to_string().contains("GrainSize"), "{e}");
    }

    #[test]
    fn zero_grains_writes_a_header_only_file_by_default() {
        let mut store = PropertyStore::new();
        for (n, c, d) in [
            ("GrainSize", 1usize, DType::I32),
            ("GrainCenter", 3, DType::F64),
            ("GrainOrientation", 4, DType::F64),
            ("GrainStructureType", 1, DType::I32),
            ("GrainMisorientation", 1, DType::F64),
        ] {
            store.register(Property::new(n, d, false, c));
        }
        let path = tmp("empty.csv");
        let rep = write_grain_table(
            &store,
            &Summary::default(),
            &GrainTableOptions {
                path: path.clone(),
                ..Default::default()
            },
        )
        .unwrap();
        assert_eq!(rep.grains, 0);
        assert!(!rep.skipped_empty);
        let text = std::fs::read_to_string(&path).unwrap();
        assert_eq!(
            text.lines().filter(|l| !l.starts_with('#')).count(),
            1,
            "header only: {text}"
        );
        assert!(rep.summary().contains("no misorientation reference"));
    }

    #[test]
    fn zero_grains_can_skip_the_file_entirely() {
        let mut store = PropertyStore::new();
        for (n, c, d) in [
            ("GrainSize", 1usize, DType::I32),
            ("GrainCenter", 3, DType::F64),
            ("GrainOrientation", 4, DType::F64),
            ("GrainStructureType", 1, DType::I32),
            ("GrainMisorientation", 1, DType::F64),
        ] {
            store.register(Property::new(n, d, false, c));
        }
        let path = tmp("skipped.csv");
        let _ = std::fs::remove_file(&path);
        let rep = write_grain_table(
            &store,
            &Summary::default(),
            &GrainTableOptions {
                path: path.clone(),
                write_empty: false,
                ..Default::default()
            },
        )
        .unwrap();
        assert!(rep.skipped_empty);
        assert!(!path.exists());
        assert!(rep.summary().contains("not written"), "{rep:?}");
    }

    #[test]
    fn float_precision_is_honoured() {
        let (store, summary) = store_two_grains();
        let path = tmp("prec.csv");
        write_grain_table(
            &store,
            &summary,
            &GrainTableOptions {
                path: path.clone(),
                float_precision: 2,
                ..Default::default()
            },
        )
        .unwrap();
        let text = std::fs::read_to_string(&path).unwrap();
        assert!(text.contains("1.00,2.00,3.00"), "{text}");
        assert!(!text.contains("1.000000"), "{text}");
    }

    #[test]
    fn negative_zero_never_reaches_the_file() {
        // A quaternion component summed over a symmetric grain lands on a tiny
        // residual of either sign, not on exact -0.0, so this has to be caught
        // after formatting rather than by comparing the float.
        for v in [-0.0f64, -1e-17, -1e-9] {
            assert_eq!(num(v, 6), "0.000000", "for {v}");
        }
        for v in [0.0f64, 1e-17, 1e-9] {
            assert_eq!(num(v, 6), "0.000000", "for {v}");
        }
        // A genuinely negative value keeps its sign.
        assert_eq!(num(-1.5, 6), "-1.500000");
        assert_eq!(num(-1.5, 1), "-1.5");
        // A value that rounds to zero at the requested precision is zero in the
        // output whatever its sign, so the sign goes with it -- otherwise the
        // precision you asked for silently decides how often this appears.
        assert_eq!(num(-1e-7, 3), "0.000");
    }

    #[test]
    fn non_finite_values_are_nan_not_a_number() {
        assert_eq!(num(f64::NAN, 6), "nan");
        assert_eq!(num(f64::INFINITY, 6), "nan");
        assert_eq!(num(f64::NEG_INFINITY, 6), "nan");
    }

    #[test]
    fn every_lattice_id_has_a_name_and_unknown_ones_say_so() {
        assert_eq!(lattice_name(structure::OTHER), "OTHER");
        assert_eq!(lattice_name(structure::FCC), "FCC");
        assert_eq!(lattice_name(structure::HCP), "HCP");
        assert_eq!(lattice_name(structure::BCC), "BCC");
        assert_eq!(lattice_name(structure::ICO), "ICO");
        assert_eq!(lattice_name(structure::SC), "SC");
        assert_eq!(
            lattice_name(99),
            "?",
            "an unrecognised id must not be guessed"
        );
    }

    #[test]
    fn missing_parent_directories_are_created() {
        let (store, summary) = store_two_grains();
        let path = tmp("nested/deeper/table.csv");
        let _ = std::fs::remove_dir_all(tmp("nested"));
        write_grain_table(
            &store,
            &summary,
            &GrainTableOptions {
                path: path.clone(),
                ..Default::default()
            },
        )
        .unwrap();
        assert!(path.exists());
    }

    #[test]
    fn the_same_store_writes_byte_identical_twice() {
        let (store, summary) = store_two_grains();
        let a = tmp("det1.csv");
        let b = tmp("det2.csv");
        for p in [&a, &b] {
            write_grain_table(
                &store,
                &summary,
                &GrainTableOptions {
                    path: p.clone(),
                    ..Default::default()
                },
            )
            .unwrap();
        }
        assert_eq!(
            std::fs::read_to_string(&a).unwrap(),
            std::fs::read_to_string(&b).unwrap()
        );
    }

    /// The defect columns appear only when the census ran, and are appended after
    /// the seam column.
    ///
    /// The unmeasured grain is the point of the test. Grain 1 is reported
    /// unmeasured (`-1`, `nan`) because its reference could not seat it, and the
    /// requirement is that this reads as *"we did not measure this"* rather than
    /// as *"this grain has no defects"*. A `0` there would be a false clean bill,
    /// and a blank cell would break the row alignment every later column depends on.
    #[test]
    fn defect_columns_appear_only_when_the_census_ran() {
        let (mut store, summary) = store_two_grains();
        let mut b = Property::new("GrainBoundaryCount", DType::I32, false, 1);
        b.push_value(&[3.0]);
        b.push_value(&[5.0]);
        store.register(b);

        let path = tmp("defects.csv");
        write_grain_table(
            &store,
            &summary,
            &GrainTableOptions {
                path: path.clone(),
                ..Default::default()
            },
        )
        .expect("write");
        let text = std::fs::read_to_string(&path).unwrap();
        assert!(
            !text.contains("vacancies"),
            "no census ran, so no defect column may be written: {text}"
        );
        let header = text
            .lines()
            .find(|l| l.starts_with("grain,"))
            .expect("a header line");
        assert_eq!(
            header,
            "grain,size,structure,lattice,cx,cy,cz,qw,qx,qy,qz,misorientation,boundary_atoms"
        );
        let rows: Vec<&str> = data_rows(&text);
        assert_eq!(
            rows[0],
            "0,10,1,FCC,1.000000,2.000000,3.000000,1.000000,0.000000,0.000000,0.000000,0.000000,3"
        );
    }
}
