//! Format detection (design §8.1).

use crate::error::{Error, Result};
use std::path::Path;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Format {
    Xyz,
    LammpsDump,
    LammpsData,
    /// LAMMPS binary dump (`dump ... FILE.bin`).
    LammpsDumpBinary,
}

impl Format {
    pub fn from_extension(path: &Path) -> Option<Format> {
        // `.gz` is transparent to the table: `frame.data.gz` is a data file, and
        // the table consults the name *under* the suffix. Lowercased first, so
        // `FRAME.DATA.GZ` resolves like every other extension here.
        let name = path.file_name()?.to_str()?.to_ascii_lowercase();
        let name = name.strip_suffix(".gz").unwrap_or(&name);
        let ext = Path::new(name).extension()?.to_str()?;
        match ext {
            "xyz" => Some(Format::Xyz),
            "dump" | "lammpstrj" | "d" => Some(Format::LammpsDump),
            "data" => Some(Format::LammpsData),
            // LAMMPS `Dump` switches to binary on exactly these suffixes.
            "bin" | "lammpsbin" => Some(Format::LammpsDumpBinary),
            // `.lmp`/`.lammps` name dumps in some tools and data files in
            // others (atomsk `writeout.f90`), so they are sniffed.
            _ => None,
        }
    }
}

/// True for a gzip-compressed input path (`frame.data.gz`, `traj.dump.gz`, …).
///
/// Keyed on the `.gz` suffix alone, not on magic bytes: the suffix is what
/// every gzip pipeline produces, and a magic check here would accept files the
/// loader — which keys off the same suffix — then refuses, so detection and
/// loading must agree on one rule. A gzipped file *without* the suffix is
/// therefore undetectable, and says so through the sniff error.
pub fn is_gzip(path: &Path) -> bool {
    path.file_name()
        .and_then(|n| n.to_str())
        .map_or(false, |n| n.to_ascii_lowercase().ends_with(".gz"))
}

/// Extension→format map shown in design §8.1. `.lmp` and `.lammps` are not in
/// it: their content decides.
pub const BY_EXTENSION: &[(&str, Format)] = &[
    ("xyz", Format::Xyz),
    ("dump", Format::LammpsDump),
    ("lammpstrj", Format::LammpsDump),
    ("data", Format::LammpsData),
    ("bin", Format::LammpsDumpBinary),
    ("lammpsbin", Format::LammpsDumpBinary),
];

/// Detect format: by extension first, then by header sniffing.
pub fn guess_format(path: &Path) -> Result<Format> {
    if let Some(f) = Format::from_extension(path) {
        return Ok(f);
    }
    // sniff: read first bytes (through the gunzipper when gzipped)
    let bytes = read_head(path)?;
    if crate::io::binary::sniff(&bytes) {
        return Ok(Format::LammpsDumpBinary);
    }
    let head = &bytes;
    // The head may end inside a multi-byte character: decode the valid prefix.
    let head = match std::str::from_utf8(head) {
        Ok(h) => h,
        Err(e) => std::str::from_utf8(&head[..e.valid_up_to()]).unwrap_or(""),
    };
    // Order matters. A LAMMPS *data* file also opens with a line starting
    // "LAMMPS" ("LAMMPS data file via write_data, version 3 ..."), so a generic
    // `starts_with("LAMMPS")` test would claim data files as dumps and shadow the
    // data-file checks below it entirely. Data files with a `.data` extension are
    // resolved before sniffing, but one with no recognised extension falls through
    // to here, and the dump reader would fail on it with "unexpected content
    // outside frame". Match on the `ITEM:` prefix instead, which only dumps have.
    if head.starts_with("ITEM:") {
        return Ok(Format::LammpsDump);
    }
    if head.contains("LAMMPS data file") || head.contains(" atoms") || head.contains("atom types") {
        return Ok(Format::LammpsData);
    }
    if let Some(line) = head.lines().next() {
        // first line is a bare count -> xyz
        if !line.trim().contains(' ') && line.trim().parse::<usize>().is_ok() {
            return Ok(Format::Xyz);
        }
    }
    Err(Error::Unsupported(format!(
        "cannot detect format of {} (ext {:?})",
        path.display(),
        path.extension()
    )))
}

/// First 256 bytes of the file, gunzipped when the path ends in `.gz`.
///
/// The sniff never looks past those bytes, so only that many are pulled —
/// a multi-gigabyte trajectory is not decompressed end to end to answer one
/// question about its first line.
fn read_head(path: &Path) -> Result<Vec<u8>> {
    use std::io::Read;
    const HEAD: u64 = 256;
    let file = std::fs::File::open(path)
        .map_err(|e| Error::Io(format!("cannot read {}: {e}", path.display())))?;
    let reader: Box<dyn Read> = if is_gzip(path) {
        Box::new(flate2::read::GzDecoder::new(file))
    } else {
        Box::new(file)
    };
    let mut head = Vec::new();
    reader.take(HEAD).read_to_end(&mut head).map_err(|e| {
        if is_gzip(path) {
            Error::Io(format!("cannot decompress {}: {e}", path.display()))
        } else {
            Error::Io(format!("cannot read {}: {e}", path.display()))
        }
    })?;
    Ok(head)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    fn write_temp(name: &str, body: &str) -> std::path::PathBuf {
        let p = std::env::temp_dir().join(name);
        let mut f = std::fs::File::create(&p).unwrap();
        f.write_all(body.as_bytes()).unwrap();
        p
    }

    /// Same, gzipped: a `.gz` sibling of `name`.
    fn write_gzip(name: &str, body: &str) -> std::path::PathBuf {
        let p = std::env::temp_dir().join(format!("{name}.gz"));
        let f = std::fs::File::create(&p).unwrap();
        let mut enc = flate2::write::GzEncoder::new(f, flate2::Compression::default());
        enc.write_all(body.as_bytes()).unwrap();
        enc.finish().unwrap();
        p
    }

    /// The real header `write_data` emits, with no `.data` extension on the file.
    const DATA_HEAD: &str = "LAMMPS data file via write_data, version 3 Aug 2022, timestep = 3333\n\n256000 atoms\n4 atom types\n\n2.3213 143.3586 xlo xhi\n";

    /// This file has no recognised extension, so detection falls through to
    /// sniffing, which must not call it a dump.
    #[test]
    fn lammps_data_file_without_a_data_extension_is_not_misdetected_as_a_dump() {
        let p = write_temp("rmda_fmt_data_no_ext", DATA_HEAD);
        assert_eq!(
            Format::from_extension(&p),
            None,
            "must fall through to sniffing"
        );
        assert_eq!(guess_format(&p).unwrap(), Format::LammpsData);
        let _ = std::fs::remove_file(p);
    }

    #[test]
    fn lammps_dump_is_still_detected_from_its_item_header() {
        let p = write_temp(
            "rmda_fmt_dump_no_ext",
            "ITEM: TIMESTEP\n0\nITEM: NUMBER OF ATOMS\n1\nITEM: BOX BOUNDS pp pp pp\n",
        );
        assert_eq!(guess_format(&p).unwrap(), Format::LammpsDump);
        let _ = std::fs::remove_file(p);
    }

    #[test]
    fn xyz_is_still_detected_from_a_bare_count() {
        let p = write_temp("rmda_fmt_xyz_no_ext", "2\ncomment\nX 0 0 0\nX 1 1 1\n");
        assert_eq!(guess_format(&p).unwrap(), Format::Xyz);
        let _ = std::fs::remove_file(p);
    }

    #[test]
    fn extension_still_wins_when_it_is_recognised() {
        let p = write_temp("rmda_fmt_explicit.dump", DATA_HEAD);
        assert_eq!(guess_format(&p).unwrap(), Format::LammpsDump);
        let _ = std::fs::remove_file(p);
    }

    /// The `.gz` suffix is stripped before the table is consulted, so every
    /// compressed sibling resolves to exactly what its uncompressed name does.
    #[test]
    fn the_gz_suffix_is_transparent_to_the_extension_table() {
        let p = |s: &str| std::path::PathBuf::from(s);
        assert_eq!(
            Format::from_extension(&p("frame.data.gz")),
            Some(Format::LammpsData)
        );
        assert_eq!(
            Format::from_extension(&p("traj.dump.gz")),
            Some(Format::LammpsDump)
        );
        assert_eq!(
            Format::from_extension(&p("traj.lammpstrj.gz")),
            Some(Format::LammpsDump)
        );
        assert_eq!(
            Format::from_extension(&p("frame.xyz.gz")),
            Some(Format::Xyz)
        );
        assert_eq!(
            Format::from_extension(&p("FRAME.DATA.GZ")),
            Some(Format::LammpsData)
        );
        // without the suffix the table behaves as usual
        assert_eq!(
            Format::from_extension(&p("frame.data")),
            Some(Format::LammpsData)
        );
        assert_eq!(
            Format::from_extension(&p("data.relaxed-alloy-reference")),
            None
        );
    }

    #[test]
    fn sniffing_reads_through_gzip() {
        let p = write_gzip("rmda_fmt_sniff_gz", DATA_HEAD);
        assert_eq!(
            Format::from_extension(&p),
            None,
            "must fall through to sniffing"
        );
        assert_eq!(guess_format(&p).unwrap(), Format::LammpsData);
        let _ = std::fs::remove_file(p);
    }

    #[test]
    fn a_gzipped_xyz_is_detected_by_its_decompressed_first_line() {
        let p = write_gzip("rmda_fmt_sniff_gz_xyz", "2\ncomment\nX 0 0 0\nX 1 1 1\n");
        assert_eq!(guess_format(&p).unwrap(), Format::Xyz);
        let _ = std::fs::remove_file(p);
    }

    #[test]
    fn is_gzip_keys_off_the_suffix_only() {
        let p = |s: &str| std::path::PathBuf::from(s);
        assert!(is_gzip(&p("frame.data.gz")));
        assert!(is_gzip(&p("FRAME.DUMP.GZ")));
        assert!(!is_gzip(&p("frame.data")));
        assert!(!is_gzip(&p("fog.dat")));
        assert!(!is_gzip(&p("archive.gzip")));
    }

    /// `.lmp` is a dump in some tools and a data file in others
    /// (atomsk writes data files as `.lmp`), so its content decides.
    #[test]
    fn lmp_files_are_sniffed() {
        let data = write_temp("rmda_fmt_atomsk.lmp", DATA_HEAD);
        assert_eq!(guess_format(&data).unwrap(), Format::LammpsData);
        let dump = write_temp("rmda_fmt_dump.lmp", "ITEM: TIMESTEP\n0\n");
        assert_eq!(guess_format(&dump).unwrap(), Format::LammpsDump);
        let lammps = write_temp("rmda_fmt_data.lammps", DATA_HEAD);
        assert_eq!(guess_format(&lammps).unwrap(), Format::LammpsData);
        let _ = std::fs::remove_file(data);
        let _ = std::fs::remove_file(dump);
        let _ = std::fs::remove_file(lammps);
    }

    #[test]
    fn binary_dumps_are_detected_by_suffix_and_magic() {
        let p = |s: &str| std::path::PathBuf::from(s);
        assert_eq!(
            Format::from_extension(&p("t.bin")),
            Some(Format::LammpsDumpBinary)
        );
        assert_eq!(
            Format::from_extension(&p("t.lammpsbin")),
            Some(Format::LammpsDumpBinary)
        );
        let mut head = (-10i64).to_le_bytes().to_vec();
        head.extend_from_slice(b"DUMPCUSTOM\x01\x00\x00\x00");
        let f = std::env::temp_dir().join("rmda_fmt_bin_noext");
        std::fs::write(&f, head).unwrap();
        assert_eq!(guess_format(&f).unwrap(), Format::LammpsDumpBinary);
        let _ = std::fs::remove_file(f);
    }
}
