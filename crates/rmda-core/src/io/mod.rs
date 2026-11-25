//! IO: readers, writers, and the fast-parsing engine (design §8).
//!
//! Optimisation target (design §8.1–§8.2): no heap allocations in the inner
//! atom loop — the file is mapped (or read) once into a `&[u8]` buffer and every line is
//! a zero-copy byte slice produced by `Cursor`; numeric fields are parsed from
//! plain byte slices.

pub mod binary;
pub mod engine;
pub mod format;
pub mod gzpar;
pub mod index;
pub mod lammps;
#[cfg(test)]
mod legacy;
#[cfg(test)]
mod roundtrip_tests;
pub mod stream;
pub mod writer;
pub mod xyz;

pub use engine::{fast_f64, fast_i64, fast_u32, Cursor};
pub use format::{guess_format, Format, BY_EXTENSION};
pub use lammps::DumpReader;
pub use writer::Writer;
pub use xyz::{XyzOrigin, XyzReader, XyzWriter};

use crate::data::Frame;
use crate::error::{Error, Result};
use std::path::Path;

/// Which input columns a read keeps (design §8.1 `Sink::Skip`).
///
/// A skipped column is stepped over in every row and never allocated, so a
/// run that does not need, say, velocities pays neither their memory (24
/// B/atom) nor their parse time. Names are the LAMMPS column names, used the
/// same way for every format: dump columns (`vx`, `c_pe`, `ix`, ...), the data
/// file's `Velocities` (`vx vy vz`), trailing image flags (`ix iy iz`) and
/// per-style columns (`q`, `mol`, ...), and extended-XYZ properties. Naming
/// any of `ix iy iz` drops the image flags as a whole. Atom ids, types and the
/// position columns are always read.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ReadOptions {
    pub skip: Vec<String>,
    /// The LAMMPS atom style of a data file's `Atoms` section (`--atom-style`),
    /// for a section line without a `# style` comment; needed when the column
    /// count is ambiguous (6 or 9: `charge` or `molecular`). A comment naming
    /// another style is an error rather than silently overridden.
    pub atom_style: Option<String>,
    /// Keep a general-triclinic box (data `avec bvec cvec`, dump `abc
    /// origin`, binary `triclinic 2`) as given (`--keep-general`): the cell
    /// rows are the file's `a b c` and positions and vector columns stay in
    /// the file's frame, instead of being rotated into LAMMPS' restricted
    /// frame. Writers then emit the general form again.
    pub keep_general: bool,
}

impl ReadOptions {
    /// Keep every column (the default).
    pub fn all() -> Self {
        Self::default()
    }

    /// What a structural analysis needs: everything except velocities and
    /// image flags.
    pub fn lean() -> Self {
        Self::skipping(["vx", "vy", "vz", "ix", "iy", "iz"])
    }

    pub fn skipping<I: IntoIterator<Item = S>, S: Into<String>>(names: I) -> Self {
        ReadOptions {
            skip: names.into_iter().map(Into::into).collect(),
            atom_style: None,
            keep_general: false,
        }
    }

    /// With the data-file atom style `style` (see [`ReadOptions::atom_style`]).
    pub fn with_atom_style(mut self, style: Option<String>) -> Self {
        self.atom_style = style;
        self
    }

    /// With [`ReadOptions::keep_general`].
    pub fn with_keep_general(mut self, keep: bool) -> Self {
        self.keep_general = keep;
        self
    }

    /// Whether column `name` is skipped.
    pub fn skips(&self, name: &str) -> bool {
        self.skip.iter().any(|s| s == name)
    }

    /// Whether the image flags are skipped.
    pub fn skips_images(&self) -> bool {
        ["ix", "iy", "iz"].iter().any(|n| self.skips(n))
    }
}

/// Read every frame from a file, dispatching on format detected from the
/// extension (or sniffed header) exactly as the design §8.1 dispatch table.
pub fn read_frames(path: &Path) -> Result<Vec<Frame>> {
    read_frames_range(path, None, None, 1)
}

/// How a `start`/`end`/`stride` window treats one absolute frame index.
///
/// The three readers (`read_frames_range`, [`read_frame_windowed`] and
/// [`WindowedIter`]) all funnel through [`slot`] so they cannot drift apart:
/// a window that keeps frame 5 in one of them keeps it in all of them.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Slot {
    /// Before `start`, or a hole in the stride: parse the frame only because
    /// later frames sit behind it, and keep nothing.
    Skip,
    /// Inside the window: this frame is wanted.
    Take,
    /// Past `end`: the scan stops here regardless of stride.
    Stop,
}

fn slot(idx: usize, start: Option<usize>, end: Option<usize>, stride: usize) -> Slot {
    if let Some(s) = start {
        if idx < s {
            return Slot::Skip;
        }
    }
    if let Some(e) = end {
        if idx >= e {
            return Slot::Stop;
        }
    }
    // `stride == 0` is the documented "every frame" value and must not panic.
    if stride != 0 && idx % stride != 0 {
        return Slot::Skip;
    }
    Slot::Take
}

pub fn read_frames_range(
    path: &Path,
    start: Option<usize>,
    end: Option<usize>,
    stride: usize,
) -> Result<Vec<Frame>> {
    let mut it = FrameReader::open(path)?.into_frames()?;
    let mut frames = Vec::new();
    // Frames before `start` are stepped over (by the index when there is
    // one), never parsed.
    let mut idx = it.fast_forward(start.unwrap_or(0).min(end.unwrap_or(usize::MAX)))?;
    loop {
        // Decide *before* reading: a frame the window excludes is never even
        // parsed, so a corrupt frame just past `end` cannot fail a run that
        // did not ask for it.
        let s = slot(idx, start, end, stride);
        if s == Slot::Stop {
            break;
        }
        if s == Slot::Skip {
            match it.skip_frame() {
                Some(r) => r?,
                None => break,
            }
            idx += 1;
            continue;
        }
        match it.next() {
            Some(res) => frames.push(res?),
            None => break,
        };
        idx += 1;
    }
    Ok(frames)
}

/// Read only the first frame a window selects.
///
/// The scan still parses the frames in front of it — a sequential file gives
/// no cheaper way to reach frame N — but nothing behind it is ever materialised,
/// so the default single-frame run over a multi-frame trajectory costs one
/// frame of memory instead of the whole trajectory.
pub fn read_frame_windowed(
    path: &Path,
    start: Option<usize>,
    end: Option<usize>,
    stride: usize,
) -> Result<Frame> {
    read_frame_windowed_with(path, start, end, stride, &ReadOptions::all())
}

/// [`read_frame_windowed`] keeping only the columns `opts` allows.
pub fn read_frame_windowed_with(
    path: &Path,
    start: Option<usize>,
    end: Option<usize>,
    stride: usize,
    opts: &ReadOptions,
) -> Result<Frame> {
    let mut it = FrameReader::open(path)?
        .with_options(opts.clone())
        .into_frames()?;
    let mut idx = it.fast_forward(start.unwrap_or(0).min(end.unwrap_or(usize::MAX)))?;
    let mut seen = idx;
    loop {
        let s = slot(idx, start, end, stride);
        if s == Slot::Stop {
            break;
        }
        if s == Slot::Skip {
            match it.skip_frame() {
                Some(r) => r?,
                None => break,
            }
            seen += 1;
            idx += 1;
            continue;
        }
        match it.next() {
            Some(res) => return res,
            None => break,
        }
    }
    Err(if seen == 0 {
        crate::error::Error::Io(format!("no frames in {}", path.display()))
    } else {
        crate::error::Error::Io(format!(
            "no frame of {} falls inside the requested window",
            path.display()
        ))
    })
}

/// A frame file restricted to a `start`/`end`/`stride` window, held one frame
/// at a time.
///
/// This is the streaming counterpart of [`read_frames_range`] for the commands
/// that genuinely need every frame in the window (`info`, `convert`): the same
/// [`slot`] filter, never a `Vec<Frame>`.
pub struct WindowedIter {
    inner: FrameIter,
    start: Option<usize>,
    end: Option<usize>,
    stride: usize,
    idx: usize,
    done: bool,
    /// Frames before `start` not yet stepped over.
    pending_start: bool,
}

/// Restrict `inner` to the frames `read_frames_range` would have kept.
pub fn windowed(
    inner: FrameIter,
    start: Option<usize>,
    end: Option<usize>,
    stride: usize,
) -> WindowedIter {
    WindowedIter {
        inner,
        start,
        end,
        stride,
        idx: 0,
        done: false,
        pending_start: true,
    }
}

impl Iterator for WindowedIter {
    type Item = Result<Frame>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.done {
            return None;
        }
        if self.pending_start {
            self.pending_start = false;
            let k = self.start.unwrap_or(0).min(self.end.unwrap_or(usize::MAX));
            match self.inner.fast_forward(k) {
                Ok(n) => self.idx = n,
                Err(e) => {
                    self.done = true;
                    return Some(Err(e));
                }
            }
        }
        loop {
            // Same hard boundary as the materialising reader: the first index
            // past the window stops the scan before its bytes are touched.
            let s = slot(self.idx, self.start, self.end, self.stride);
            if s == Slot::Stop {
                self.done = true;
                return None;
            }
            if s == Slot::Skip {
                match self.inner.skip_frame() {
                    Some(Ok(())) => {
                        self.idx += 1;
                        continue;
                    }
                    Some(Err(e)) => {
                        self.done = true;
                        return Some(Err(e));
                    }
                    None => {
                        self.done = true;
                        return None;
                    }
                }
            }
            return match self.inner.next() {
                None => {
                    self.done = true;
                    None
                }
                Some(Err(e)) => {
                    // An unreadable frame ends the scan the way `?` does in the
                    // materialising reader: report it once, then stop.
                    self.done = true;
                    Some(Err(e))
                }
                Some(Ok(f)) => {
                    self.idx += 1;
                    Some(Ok(f))
                }
            };
        }
    }
}

/// A streaming frame reader.
///
/// The iterator holds a cursor and the format. It reads frames one at a time
/// without accumulating all frames into a `Vec`.
pub struct FrameReader<'a> {
    path: &'a Path,
    fmt: Format,
    opts: ReadOptions,
}

impl<'a> FrameReader<'a> {
    pub fn open(path: &'a Path) -> Result<Self> {
        let fmt = guess_format(path).map_err(Error::into_input)?;
        Ok(FrameReader {
            path,
            fmt,
            opts: ReadOptions::all(),
        })
    }

    /// Keep only the columns `opts` allows.
    pub fn with_options(mut self, opts: ReadOptions) -> Self {
        self.opts = opts;
        self
    }

    pub fn into_frames(self) -> Result<FrameIter> {
        // Gzip is streamed a frame at a time; everything else is mapped.
        // A file that cannot be opened is the input's fault (exit 3).
        let src = if format::is_gzip(self.path) {
            Source::Gz(stream::GzFrames::open(self.path).map_err(Error::into_input)?)
        } else {
            Source::Mem(engine::Cursor::from_file(self.path).map_err(Error::into_input)?)
        };
        Ok(FrameIter {
            fmt: self.fmt,
            src,
            index: index::load(self.path),
            opts: self.opts,
            done: false,
        })
    }

    pub fn format(&self) -> Format {
        self.fmt
    }
}

enum Source {
    /// The whole file, memory-mapped.
    Mem(engine::Cursor),
    /// A gzip stream, one frame resident at a time.
    Gz(stream::GzFrames),
}

/// Frames of one file, in order. After an error the iterator is finished.
pub struct FrameIter {
    fmt: Format,
    src: Source,
    /// The file's frame index, when a current one exists.
    index: Option<Vec<index::IndexEntry>>,
    opts: ReadOptions,
    done: bool,
}

/// Parse the next frame of `fmt` at the cursor.
fn read_next(fmt: Format, cur: &engine::Cursor, opts: &ReadOptions) -> Result<Option<Frame>> {
    match fmt {
        Format::Xyz => xyz::read_next_with(cur, opts),
        Format::LammpsDump => lammps::read_next_dump_with(cur, opts),
        Format::LammpsDumpBinary => binary::read_next_with(cur, opts),
        // A LAMMPS data file holds exactly one frame.
        Format::LammpsData => lammps::parse_data_frame_with(cur, opts).map(Some),
    }
}

impl FrameIter {
    /// Step over the next frame without parsing its atoms. `None` at
    /// the end of the input.
    pub fn skip_frame(&mut self) -> Option<Result<()>> {
        if self.done {
            return None;
        }
        let r: Result<bool> = match &mut self.src {
            Source::Mem(cur) => match self.fmt {
                Format::LammpsDump => lammps::skip_next_dump(cur).map(|o| o.is_some()),
                Format::Xyz => xyz::skip_next(cur),
                Format::LammpsDumpBinary => binary::skip_next(cur),
                Format::LammpsData => {
                    self.done = true;
                    Ok(true)
                }
            },
            // Decompressed but not parsed.
            Source::Gz(gz) => gz.next_frame(self.fmt).map(|o| o.is_some()),
        };
        match r {
            Ok(true) => {
                if self.fmt == Format::LammpsData {
                    self.done = true;
                }
                Some(Ok(()))
            }
            Ok(false) => {
                self.done = true;
                None
            }
            Err(e) => {
                self.done = true;
                Some(Err(e))
            }
        }
    }

    /// Step over the first `k` frames, by seeking through the frame index
    /// when there is one; returns how many frames were passed (fewer than
    /// `k` when the input is shorter).
    pub fn fast_forward(&mut self, k: usize) -> Result<usize> {
        if k == 0 {
            return Ok(0);
        }
        if let (Some(ix), Source::Mem(cur)) = (&self.index, &self.src) {
            if cur.pos() == 0 {
                return Ok(match ix.get(k) {
                    Some(e) => {
                        cur.seek(e.offset, e.line);
                        k
                    }
                    None => {
                        self.done = true;
                        ix.len()
                    }
                });
            }
        }
        let mut n = 0;
        while n < k {
            match self.skip_frame() {
                Some(r) => r?,
                None => break,
            }
            n += 1;
        }
        Ok(n)
    }

    /// Whether a frame index is in use.
    pub fn is_indexed(&self) -> bool {
        self.index.is_some()
    }
}

impl Iterator for FrameIter {
    type Item = Result<Frame>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.done {
            return None;
        }
        let r = match &mut self.src {
            Source::Mem(cur) => read_next(self.fmt, cur, &self.opts),
            Source::Gz(gz) => match gz.next_frame(self.fmt) {
                Ok(Some((bytes, first_line))) => {
                    let cur = engine::Cursor::from_vec(bytes, &gz.path, first_line);
                    read_next(self.fmt, &cur, &self.opts)
                }
                Ok(None) => Ok(None),
                Err(e) => Err(e),
            },
        };
        if self.fmt == Format::LammpsData {
            self.done = true;
        }
        match r {
            // Every frame a reader returns is checked once, here: what the
            // analyses assume (counts agree, everything finite, a real box).
            Ok(Some(f)) => match f.check_input() {
                Ok(()) => Some(Ok(f)),
                Err(e) => {
                    self.done = true;
                    Some(Err(e))
                }
            },
            Ok(None) => {
                self.done = true;
                None
            }
            Err(e) => {
                // An iterator that keeps going after a parse error would
                // resume mid-frame. Whatever the reader refused is the input's
                // fault (exit code 3).
                self.done = true;
                Some(Err(e.into_input()))
            }
        }
    }
}

/// Read the first frame from a file, useful for single-frame analyses.
pub fn read_frame(path: &Path) -> Result<Frame> {
    let mut it = FrameReader::open(path)?.into_frames()?;
    match it.next() {
        Some(Ok(f)) => Ok(f),
        Some(Err(e)) => Err(e),
        None => Err(crate::error::Error::Io(format!(
            "no frames in {}",
            path.display()
        ))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tempdir(name: &str) -> std::path::PathBuf {
        let dir = std::env::temp_dir().join(name);
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    /// A gzipped sibling of `path` holding `body`, e.g. `three.xyz` →
    /// `three.xyz.gz`.
    fn gz_of(path: &std::path::Path, body: &str) -> std::path::PathBuf {
        use std::io::Write;
        let name = path.file_name().unwrap().to_string_lossy();
        let gzpath = path.with_file_name(format!("{name}.gz"));
        let f = std::fs::File::create(&gzpath).unwrap();
        let mut enc = flate2::write::GzEncoder::new(f, flate2::Compression::default());
        enc.write_all(body.as_bytes()).unwrap();
        enc.finish().unwrap();
        gzpath
    }

    const DATA: &str = "(written by ASE)\n\
                        \n\
                        2 atoms\n\
                        1 atom types\n\
                        \n\
                        0.0 4.0 xlo xhi\n\
                        0.0 4.0 ylo yhi\n\
                        0.0 4.0 zlo zhi\n\
                        \n\
                        Atoms # atomic\n\
                        \n\
                        1 1 0.0 0.0 0.0\n\
                        2 1 2.0 2.0 2.0\n";

    /// `read_frame` is the entry point every single-frame analysis (and the
    /// Python bindings) goes through, so a LAMMPS data file must come back as
    /// its one frame, not as "no frames in <file>".
    #[test]
    fn read_frame_reads_lammps_data_file() {
        let p = tempdir("rmda_io_mod_tests").join("read_frame.data");
        std::fs::write(&p, DATA).unwrap();
        let f = read_frame(&p).expect("data file must yield a frame");
        assert_eq!(f.natoms(), 2);
        assert_eq!(f.particles.positions[1], [2.0, 2.0, 2.0]);
        assert_eq!(f.particles.ids, vec![1, 2]);
    }

    /// A data file holds exactly one frame: the iterator must stop after it,
    /// and the `start`/`end` window must apply to it like any other format.
    #[test]
    fn data_file_yields_one_frame_and_honours_the_window() {
        let p = tempdir("rmda_io_mod_tests").join("window.data");
        std::fs::write(&p, DATA).unwrap();
        assert_eq!(read_frames(&p).unwrap().len(), 1);
        assert_eq!(read_frames_range(&p, Some(0), Some(1), 1).unwrap().len(), 1);
        assert!(read_frames_range(&p, Some(1), None, 1).unwrap().is_empty());
        assert!(read_frames_range(&p, None, Some(0), 1).unwrap().is_empty());
    }

    /// Three 2-atom frames whose first x coordinate carries the frame index
    /// (`0.0`, `10.0`, `20.0`), so any window can be checked by inspection.
    fn three_frame_xyz() -> String {
        let mut src = String::new();
        for i in 0..3 {
            src.push_str("2\n");
            src.push_str(
                "Lattice=\"4.0 0.0 0.0 0.0 4.0 0.0 0.0 0.0 5.0\" pbc=\"T T T\" \
                 Properties=species:S:1:pos:R:3\n",
            );
            let x = 10.0 * i as f64;
            src.push_str(&format!("Cu {x} 0.0 0.0\n"));
            src.push_str(&format!("Cu {} 2.0 2.5\n", x + 2.0));
        }
        src
    }

    /// Every window the CLI can build from `--frame/--start/--end/--stride`,
    /// including the awkward ones (odd start under an even stride, a stride of
    /// 0, an empty window).
    fn window_matrix() -> Vec<(Option<usize>, Option<usize>, usize)> {
        let mut v = Vec::new();
        for start in [None, Some(0), Some(1), Some(2), Some(3)] {
            for end in [None, Some(0), Some(1), Some(2), Some(3)] {
                for stride in [0, 1, 2, 3] {
                    v.push((start, end, stride));
                }
            }
        }
        v
    }

    /// `windowed` and `read_frames_range` are two readings of the same window;
    /// the CLI picks between them per command, so a divergence would silently
    /// give `info` and `structure` different frame sets.
    #[test]
    fn windowed_iter_agrees_with_read_frames_range_on_every_window() {
        let p = tempdir("rmda_io_mod_tests").join("window_matrix.xyz");
        std::fs::write(&p, three_frame_xyz()).unwrap();

        for (start, end, stride) in window_matrix() {
            let want = read_frames_range(&p, start, end, stride).unwrap();
            let it = FrameReader::open(&p).unwrap().into_frames().unwrap();
            let got = windowed(it, start, end, stride)
                .collect::<Result<Vec<_>>>()
                .unwrap();
            assert_eq!(
                got.len(),
                want.len(),
                "start={start:?} end={end:?} stride={stride}"
            );
            for (g, w) in got.iter().zip(want.iter()) {
                assert_eq!(
                    g.particles.positions, w.particles.positions,
                    "start={start:?} end={end:?} stride={stride}"
                );
            }
        }
    }

    /// `read_frame_windowed` must land on exactly the frame the materialising
    /// reader would have returned first — the analysis commands depend on it
    /// for which frame they analyse.
    #[test]
    fn read_frame_windowed_returns_the_first_frame_the_window_keeps() {
        let p = tempdir("rmda_io_mod_tests").join("window_first.xyz");
        std::fs::write(&p, three_frame_xyz()).unwrap();

        for (start, end, stride) in window_matrix() {
            let want = read_frames_range(&p, start, end, stride).unwrap();
            let got = read_frame_windowed(&p, start, end, stride);
            assert_eq!(
                got.is_ok(),
                !want.is_empty(),
                "start={start:?} end={end:?} stride={stride}"
            );
            if let (Ok(g), Some(w)) = (got.as_ref(), want.first()) {
                assert_eq!(
                    g.particles.positions, w.particles.positions,
                    "start={start:?} end={end:?} stride={stride}"
                );
            }
        }

        // An empty file still reports the classic message, while a file that
        // simply has no frame inside the window says which problem it hit.
        let empty = tempdir("rmda_io_mod_tests").join("window_empty.xyz");
        std::fs::write(&empty, "").unwrap();
        let e = read_frame_windowed(&empty, None, None, 1).unwrap_err();
        assert!(e.to_string().contains("no frames in"), "{e}");
        let e = read_frame_windowed(&p, Some(3), None, 1).unwrap_err();
        assert!(e.to_string().contains("requested window"), "{e}");
    }

    /// The streaming window stops at `end` without parsing what lies past it,
    /// and a broken frame inside the window still surfaces as an error rather
    /// than as a silently short frame list. The file here has a corrupt second
    /// frame on purpose: `end = 1` must read cleanly, the full window must not.
    #[test]
    fn windowed_iter_stops_at_end_without_touching_what_lies_past_it() {
        let p = tempdir("rmda_io_mod_tests").join("window_corrupt_behind.xyz");
        let mut src = three_frame_xyz();
        // Keep frame 0 only: cut at frame 1's atom-count line (the first `2\n`
        // after the one that opens frame 0), then put a broken frame there.
        let frame1_at = src[2..].find("2\n").unwrap() + 2;
        src.truncate(frame1_at);
        src.push_str("not-a-number\ncomment\nCu 10 0.0 0.0\n");
        std::fs::write(&p, &src).unwrap();

        // Stop before the corrupt frame: it is never parsed, so no error.
        let it = FrameReader::open(&p).unwrap().into_frames().unwrap();
        let got = windowed(it, None, Some(1), 1)
            .collect::<Result<Vec<_>>>()
            .unwrap();
        assert_eq!(got.len(), 1);
        assert_eq!(got[0].particles.positions[0][0], 0.0);

        // The same file with no `end`: the corrupt frame is in the window, so
        // it must be reported.
        let it = FrameReader::open(&p).unwrap().into_frames().unwrap();
        let res: Vec<_> = windowed(it, None, None, 1).collect();
        assert!(res.iter().any(|r| r.is_err()), "expected an error");
        assert!(res[0].is_ok(), "frame 0 must still be delivered first");

        // The single-frame reader only wants frame 0 and never sees the rest.
        let f = read_frame_windowed(&p, None, None, 1).unwrap();
        assert_eq!(f.particles.positions[0][0], 0.0);
    }

    /// Start/end/stride over a multi-frame file, through the one windowing loop
    /// every format shares.
    #[test]
    fn read_frames_range_applies_window_and_stride_over_xyz() {
        let p = tempdir("rmda_io_mod_tests").join("three.xyz");
        std::fs::write(&p, three_frame_xyz()).unwrap();

        let all = read_frames_range(&p, None, None, 1).unwrap();
        assert_eq!(all.len(), 3);
        assert_eq!(all[0].particles.positions[0][0], 0.0);

        let win = read_frames_range(&p, Some(1), Some(3), 1).unwrap();
        assert_eq!(win.len(), 2);
        assert_eq!(win[0].particles.positions[0][0], 10.0);

        let strided = read_frames_range(&p, None, None, 2).unwrap();
        assert_eq!(strided.len(), 2);
        assert_eq!(strided[0].particles.positions[0][0], 0.0);
        assert_eq!(strided[1].particles.positions[0][0], 20.0);

        // Window plus stride: the window drops index 0, then stride keeps only
        // the even absolute indices, so frame 2 is what survives.
        let both = read_frames_range(&p, Some(1), Some(3), 2).unwrap();
        assert_eq!(both.len(), 1);
        assert_eq!(both[0].particles.positions[0][0], 20.0);
    }

    /// A dump file is read through the same iterator; data-file handling must
    /// not disturb the multi-frame paths.
    #[test]
    fn read_frame_reads_first_dump_frame() {
        let dir = tempdir("rmda_io_mod_tests");
        let p = dir.join("two.dump");
        std::fs::write(
            &p,
            "ITEM: TIMESTEP\n1\nITEM: NUMBER OF ATOMS\n2\n\
             ITEM: BOX BOUNDS pp pp pp\n0 10\n0 10\n0 10\n\
             ITEM: ATOMS id type x y z\n1 1 1 2 3\n2 1 8 9 10\n\
             ITEM: TIMESTEP\n2\nITEM: NUMBER OF ATOMS\n2\n\
             ITEM: BOX BOUNDS pp pp pp\n0 10\n0 10\n0 10\n\
             ITEM: ATOMS id type x y z\n1 1 4 5 6\n2 1 7 8 9\n",
        )
        .unwrap();
        let f = read_frame(&p).unwrap();
        assert_eq!(f.natoms(), 2);
        assert_eq!(f.particles.positions[0], [1.0, 2.0, 3.0]);
        assert_eq!(read_frames(&p).unwrap().len(), 2);
    }

    /// The gzip path through the loader: a gzipped LAMMPS data file (the
    /// format the pipeline actually ships compressed) must yield exactly the
    /// frame its uncompressed twin yields — same positions, same ids.
    #[test]
    fn a_gzipped_data_file_reads_exactly_like_the_plain_one() {
        let dir = tempdir("rmda_io_mod_tests");
        let plain = dir.join("gz_equiv.data");
        std::fs::write(&plain, DATA).unwrap();
        let gz = gz_of(&plain, DATA);

        let want = read_frame(&plain).expect("data file must yield a frame");
        let got = read_frame(&gz).expect("gzipped data file must yield a frame");
        assert_eq!(got.natoms(), want.natoms());
        assert_eq!(got.particles.positions, want.particles.positions);
        assert_eq!(got.particles.ids, want.particles.ids);
        assert_eq!(read_frames(&gz).unwrap().len(), 1);
    }

    /// Windowing and streaming over gzip: the format resolves from the name
    /// under `.gz`, the bytes come back inflated, and every window must be
    /// identical frame for frame to the plain file.
    #[test]
    fn a_gzipped_xyz_windows_identically_to_the_plain_file() {
        let dir = tempdir("rmda_io_mod_tests");
        let plain = dir.join("gz_equiv.xyz");
        let body = three_frame_xyz();
        std::fs::write(&plain, &body).unwrap();
        let gz = gz_of(&plain, &body);

        assert_eq!(read_frames(&gz).unwrap().len(), 3);
        for (start, end, stride) in [(None, None, 1), (Some(1), None, 2), (Some(1), Some(3), 1)] {
            let want = read_frames_range(&plain, start, end, stride).unwrap();
            let got = read_frames_range(&gz, start, end, stride).unwrap();
            assert_eq!(
                got.len(),
                want.len(),
                "start={start:?} end={end:?} stride={stride}"
            );
            for (g, w) in got.iter().zip(want.iter()) {
                assert_eq!(g.particles.positions, w.particles.positions);
            }
        }

        // The streaming reader (what the CLI's convert/info use) too.
        let frames = FrameReader::open(&gz)
            .unwrap()
            .into_frames()
            .unwrap()
            .collect::<Result<Vec<_>>>()
            .unwrap();
        assert_eq!(frames.len(), 3);
    }

    /// A multi-frame dump compressed whole: both frames, first frame's
    /// coordinates intact.
    #[test]
    fn a_gzipped_dump_reads_like_the_plain_one() {
        const TWO_DUMP: &str = "ITEM: TIMESTEP\n1\nITEM: NUMBER OF ATOMS\n2\n\
             ITEM: BOX BOUNDS pp pp pp\n0 10\n0 10\n0 10\n\
             ITEM: ATOMS id type x y z\n1 1 1 2 3\n2 1 8 9 10\n\
             ITEM: TIMESTEP\n2\nITEM: NUMBER OF ATOMS\n2\n\
             ITEM: BOX BOUNDS pp pp pp\n0 10\n0 10\n0 10\n\
             ITEM: ATOMS id type x y z\n1 1 4 5 6\n2 1 7 8 9\n";
        let dir = tempdir("rmda_io_mod_tests");
        let plain = dir.join("gz_equiv.dump");
        std::fs::write(&plain, TWO_DUMP).unwrap();
        let gz = gz_of(&plain, TWO_DUMP);

        let want = read_frame(&plain).unwrap();
        let got = read_frame(&gz).unwrap();
        assert_eq!(got.natoms(), 2);
        assert_eq!(got.particles.positions, want.particles.positions);
        assert_eq!(read_frames(&gz).unwrap().len(), 2);
    }

    /// Lying extension: plain bytes under a `.gz` name must fail with the
    /// decompression error, not with a format-detection mystery.
    #[test]
    fn a_non_gzip_file_with_a_gz_suffix_reports_a_decompression_error() {
        let p = tempdir("rmda_io_mod_tests").join("lying.data.gz");
        std::fs::write(&p, DATA).unwrap();
        let e = read_frame(&p).unwrap_err();
        assert!(e.to_string().contains("cannot decompress"), "{e}");
    }

    /// N-frame dump where frame k's first x is `k`, `natoms` atoms each.
    fn dump_frames(nframes: usize, natoms: usize) -> String {
        let mut s = String::new();
        for k in 0..nframes {
            s.push_str(&format!(
                "ITEM: TIMESTEP\n{}\nITEM: NUMBER OF ATOMS\n{natoms}\n\
                 ITEM: BOX BOUNDS pp pp pp\n0 100\n0 100\n0 100\nITEM: ATOMS id type x y z\n",
                k * 10
            ));
            for i in 0..natoms {
                s.push_str(&format!("{} 1 {k} {}.5 2.25\n", i + 1, i % 90));
            }
        }
        s
    }

    /// Concatenated gzip members (what `dump *.gz` and parallel writers
    /// produce) read as one stream, frame by frame, with identical frames.
    #[test]
    fn concatenated_gzip_members_stream_frame_by_frame() {
        use std::io::Write;
        let dir = tempdir("rmda_io_mod_tests");
        let body = dump_frames(5, 300);
        let plain = dir.join("members.dump");
        std::fs::write(&plain, &body).unwrap();
        let gz = dir.join("members.dump.gz");
        let mut out = Vec::new();
        // Cut mid-frame on purpose: members need not align with frames.
        for part in [&body[..1000], &body[1000..20_000], &body[20_000..]] {
            let mut enc = flate2::write::GzEncoder::new(Vec::new(), flate2::Compression::fast());
            enc.write_all(part.as_bytes()).unwrap();
            out.extend(enc.finish().unwrap());
        }
        std::fs::write(&gz, out).unwrap();
        let a = read_frames(&plain).unwrap();
        let b = read_frames(&gz).unwrap();
        assert_eq!(b.len(), 5);
        for (x, y) in a.iter().zip(&b) {
            assert_eq!(x.particles.positions, y.particles.positions);
            assert_eq!(x.meta.timestep, y.meta.timestep);
        }
        // Windows over the stream skip without parsing, same frames (the
        // stride keeps even absolute indices).
        let w = read_frames_range(&gz, Some(1), None, 2).unwrap();
        assert_eq!(w.len(), 2);
        assert_eq!(w[0].particles.positions[0][0], 2.0);
        assert_eq!(w[1].particles.positions[0][0], 4.0);
    }

    /// The gzip source holds about one frame, not the file.
    #[test]
    fn gzip_stream_keeps_one_frame_resident() {
        use std::io::Write;
        let body = dump_frames(12, 40_000); // ~1 MB per frame
        let mut enc = flate2::write::GzEncoder::new(Vec::new(), flate2::Compression::fast());
        enc.write_all(body.as_bytes()).unwrap();
        let bytes = enc.finish().unwrap();
        let mut g = stream::GzFrames::from_reader(
            Box::new(flate2::read::MultiGzDecoder::new(std::io::Cursor::new(
                bytes,
            ))),
            "mem.gz",
        );
        let mut n = 0;
        let mut biggest = 0;
        while let Some((frame, _)) = g.next_frame(Format::LammpsDump).unwrap() {
            biggest = biggest.max(frame.len());
            n += 1;
        }
        assert_eq!(n, 12);
        assert!(
            biggest < body.len() / 6,
            "a frame cut held {biggest} of {} bytes",
            body.len()
        );
    }

    /// Frames outside the window are stepped over by counting rows, so a
    /// malformed atom row in a skipped frame is never parsed; the frame index
    /// then seeks straight to the frame, and a stale index is ignored.
    #[test]
    fn skipped_frames_are_not_parsed_and_the_index_seeks() {
        let dir = tempdir("rmda_io_mod_tests");
        let p = dir.join("skip.dump");
        let body = dump_frames(4, 50).replacen("1 1 1 0.5 2.25", "1 1 BROKEN 0.5 2.25", 1);
        std::fs::write(&p, body).unwrap();
        let _ = std::fs::remove_file(index::index_path(&p));
        assert!(read_frames(&p).is_err(), "frame 1 is broken");
        let f = read_frame_windowed(&p, Some(2), None, 1).unwrap();
        assert_eq!(f.meta.timestep, 20);
        assert_eq!(read_frames_range(&p, None, None, 2).unwrap().len(), 2);

        let entries = index::write(&p).unwrap();
        assert_eq!(entries.len(), 4);
        assert_eq!(entries[3].timestep, 30);
        assert_eq!(entries[3].natoms, 50);
        let it = FrameReader::open(&p).unwrap().into_frames().unwrap();
        assert!(it.is_indexed());
        let f = read_frame_windowed(&p, Some(3), None, 1).unwrap();
        assert_eq!(f.meta.timestep, 30);
        assert_eq!(f.particles.positions[0][0], 3.0);
        let e = read_frame_windowed(&p, Some(9), None, 1).unwrap_err();
        assert!(e.to_string().contains("requested window"), "{e}");
        // An error in a seeked frame still names the right file line.
        let mut it = FrameReader::open(&p).unwrap().into_frames().unwrap();
        it.fast_forward(1).unwrap();
        let e = it.next().unwrap().unwrap_err().to_string();
        // 59 lines of frame 0, 9 header lines, then the broken row.
        assert!(e.contains("line 69"), "line of the broken row: {e}");

        // Rewrite the file: the index no longer matches and is ignored.
        std::thread::sleep(std::time::Duration::from_millis(20));
        std::fs::write(&p, dump_frames(2, 50)).unwrap();
        let it = FrameReader::open(&p).unwrap().into_frames().unwrap();
        assert!(!it.is_indexed(), "stale index must not be used");
        assert_eq!(read_frames(&p).unwrap().len(), 2);
    }

    /// The index covers every format that can seek.
    #[test]
    fn index_scans_xyz_and_binary() {
        let dir = tempdir("rmda_io_mod_tests");
        let p = dir.join("idx.xyz");
        std::fs::write(&p, three_frame_xyz()).unwrap();
        let e = index::build(&p).unwrap();
        assert_eq!(e.len(), 3);
        assert_eq!(e[0].offset, 0);
        let f = read_frame_windowed(&p, Some(2), None, 1).unwrap();
        assert_eq!(f.particles.positions[0][0], 20.0);

        let frames = read_frames(&dir.join("idx.xyz")).unwrap();
        let mut out = Vec::new();
        binary::write_dump_binary(&mut out, &frames).unwrap();
        let b = dir.join("idx.bin");
        std::fs::write(&b, out).unwrap();
        index::write(&b).unwrap();
        let f = read_frame_windowed(&b, Some(1), None, 1).unwrap();
        assert_eq!(f.particles.positions, frames[1].particles.positions);
    }

    /// `ReadOptions` drops the named columns in every format, never touches
    /// ids, types or positions, and `lean` drops exactly velocities and image
    /// flags.
    #[test]
    fn read_options_skip_columns_in_every_format() {
        let dir = tempdir("rmda_io_mod_tests");
        let lean = ReadOptions::lean();
        let read =
            |p: &Path, o: &ReadOptions| read_frame_windowed_with(p, None, None, 1, o).unwrap();

        let dump = dir.join("opts.dump");
        std::fs::write(
            &dump,
            "ITEM: TIMESTEP\n1\nITEM: NUMBER OF ATOMS\n2\nITEM: BOX BOUNDS pp pp pp\n0 9\n0 9\n0 9\n\
             ITEM: ATOMS id type x y z vx vy vz ix iy iz c_pe\n\
             1 1 1 2 3 0.1 0.2 0.3 0 0 1 -1.5\n2 2 4 5 6 0.4 0.5 0.6 1 0 0 -2.5\n",
        )
        .unwrap();
        let all = read(&dump, &ReadOptions::all());
        assert!(all.particles.images.is_some() && all.particles.props.get("vx").is_some());
        let f = read(&dump, &lean);
        assert!(f.particles.images.is_none());
        assert!(f.particles.props.get("vx").is_none() && f.particles.props.get("vz").is_none());
        assert!(
            f.particles.props.get("c_pe").is_some(),
            "only the named columns go"
        );
        assert_eq!(f.particles.positions, all.particles.positions);
        assert_eq!(f.particles.ids, all.particles.ids);
        assert_eq!(f.particles.types, all.particles.types);
        // Naming one image column drops the flags whole, even a partial set.
        let partial = dir.join("opts_partial.dump");
        std::fs::write(
            &partial,
            std::fs::read_to_string(&dump)
                .unwrap()
                .replace(" iy iz c_pe", " iy c_a c_pe"),
        )
        .unwrap();
        assert!(FrameReader::open(&partial)
            .unwrap()
            .into_frames()
            .unwrap()
            .next()
            .unwrap()
            .is_err());
        let f = read(&partial, &ReadOptions::skipping(["ix"]));
        assert!(f.particles.images.is_none());
        // Skipping does not excuse a malformed row.
        let short = dir.join("opts_short.dump");
        std::fs::write(
            &short,
            std::fs::read_to_string(&dump)
                .unwrap()
                .replace(" -2.5\n", "\n"),
        )
        .unwrap();
        assert!(read_frame_windowed_with(&short, None, None, 1, &lean).is_err());

        let data = dir.join("opts.data");
        std::fs::write(
            &data,
            "x\n\n2 atoms\n2 atom types\n\n0 9 xlo xhi\n0 9 ylo yhi\n0 9 zlo zhi\n\n\
             Atoms # charge\n\n1 1 0.5 1 2 3 0 0 1\n2 2 -0.5 4 5 6 1 0 0\n\n\
             Velocities\n\n1 0.1 0.2 0.3\n2 0.4 0.5 0.6\n",
        )
        .unwrap();
        let all = read(&data, &ReadOptions::all());
        assert!(all.particles.images.is_some() && all.particles.props.get("vy").is_some());
        let f = read(&data, &lean);
        assert!(f.particles.images.is_none() && f.particles.props.get("vy").is_none());
        assert_eq!(f.particles.positions, all.particles.positions);
        assert!(f.particles.props.get("q").is_some());
        let f = read(&data, &ReadOptions::skipping(["q", "vy"]));
        assert!(f.particles.props.get("q").is_none());
        assert!(f.particles.props.get("vx").is_some() && f.particles.props.get("vy").is_none());

        let xyz = dir.join("opts.xyz");
        std::fs::write(
            &xyz,
            "2\nProperties=species:S:1:pos:R:3:vx:R:1:c:I:1\nCu 1 2 3 0.5 7\nAl 4 5 6 0.25 8\n",
        )
        .unwrap();
        let f = read(&xyz, &lean);
        assert!(f.particles.props.get("vx").is_none() && f.particles.props.get("c").is_some());
        assert_eq!(f.particles.positions[1], [4.0, 5.0, 6.0]);

        let frames = read_frames(&dump).unwrap();
        let mut out = Vec::new();
        binary::write_dump_binary(&mut out, &frames).unwrap();
        let bin = dir.join("opts.bin");
        std::fs::write(&bin, out).unwrap();
        let f = read(&bin, &lean);
        assert!(f.particles.images.is_none());
        assert_eq!(f.particles.positions, frames[0].particles.positions);
    }
}
