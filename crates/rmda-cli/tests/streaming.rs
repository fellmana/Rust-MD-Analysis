//! Frame-at-a-time input.
//!
//! Every analysis here runs on one frame, so the CLI must not read *every*
//! frame the window selects before starting: materialising a trajectory is
//! the difference between one frame of memory and tens of gigabytes at the
//! scale this project claims to support.
//!
//! The tests come in two flavours because the property has two faces. Behaviourally,
//! a file whose later frames are unreadable must still be analysable up to the
//! boundary of the window — anything that has parsed frame 5 cannot claim to
//! have read only frame 0. Quantitatively, `--memreport` reports the process
//! peak, so a twelve-frame trajectory must cost about what one frame costs.
//!
//! Fixtures are built here rather than stored: the point is which frames were
//! touched, and a stored file could not show that.

use std::path::PathBuf;
use std::process::Command;

fn tempdir() -> PathBuf {
    let dir = std::env::temp_dir().join("rmda-cli-streaming-tests");
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

fn rmda(args: &[&str]) -> (String, String, bool) {
    let out = Command::new(env!("CARGO_BIN_EXE_rmda"))
        .args(args)
        .output()
        .expect("failed to run rmda");
    (
        String::from_utf8_lossy(&out.stdout).into_owned(),
        String::from_utf8_lossy(&out.stderr).into_owned(),
        out.status.success(),
    )
}

/// One 2-atom frame whose first atom carries `x`, so any frame in a file can be
/// told apart from every other one by a single number.
fn frame(x: f64) -> String {
    format!(
        "2\n\
         Lattice=\"4.0 0.0 0.0 0.0 4.0 0.0 0.0 0.0 5.0\" pbc=\"T T T\" \
         Properties=species:S:1:pos:R:3\n\
         Cu {x} 0.0 0.0\n\
         Cu {x2} 2.0 2.5\n",
        x2 = x + 2.0
    )
}

fn three_frames() -> String {
    format!("{}{}{}", frame(0.0), frame(10.0), frame(20.0))
}

/// Frame 0 followed by one whose atom-count line is not a number: reading past
/// frame 0 hits a parse error, reading only frame 0 does not.
fn second_frame_corrupt() -> String {
    format!("{}not-a-number\ncomment\nCu 99 0.0 0.0\n", frame(0.0))
}

fn stderr_lines_starting_with(stderr: &str, prefix: &str) -> Vec<String> {
    stderr
        .lines()
        .filter(|l| l.starts_with(prefix))
        .map(str::to_string)
        .collect()
}

/// The headline number `rmda info` and friends print before their output, with
/// the same wording under streaming, including for multi-frame files.
fn summary_line(stderr: &str) -> String {
    stderr
        .lines()
        .find(|l| l.starts_with("read "))
        .unwrap_or_else(|| panic!("no `read N frame(s)` summary in:\n{stderr}"))
        .to_string()
}

/// `structure` analyses frame 0 and must not even parse the frame behind it:
/// on a file whose second frame is unreadable it still reports CNA counts, and
/// asking for that second frame explicitly fails.
#[test]
fn structure_reads_only_the_frame_it_analyses() {
    let p = tempdir().join("structure_corrupt_behind.xyz");
    std::fs::write(&p, second_frame_corrupt()).unwrap();
    let path = p.to_str().unwrap();

    let (out, err, ok) = rmda(&["structure", path]);
    assert!(ok, "frame 0 must be analysable alone:\n{err}");
    assert!(out.contains("cna_fcc:"), "no CNA counts in:\n{out}");
    assert_eq!(summary_line(&err), "read 1 frame(s), 2 atoms in frame 0");

    // The window is honoured in the other direction too: selecting the broken
    // frame means reading it, so this must fail rather than silently analyse
    // frame 0 instead.
    let (_, err, ok) = rmda(&["structure", "--frame", "1", path]);
    assert!(!ok, "asking for the corrupt frame must fail");
    assert!(err.contains("bad atom count"), "{err}");
}

/// `info` is the one command that wants every frame, and it streams them: the
/// per-frame lines come out in order behind the summary, exactly as documented,
/// while only one frame is ever resident.
#[test]
fn info_streams_every_frame_behind_the_documented_summary() {
    let p = tempdir().join("info_three.xyz");
    std::fs::write(&p, three_frames()).unwrap();
    let path = p.to_str().unwrap();

    let (_, err, ok) = rmda(&["info", path]);
    assert!(ok, "info failed:\n{err}");
    assert_eq!(summary_line(&err), "read 3 frame(s), 2 atoms in frame 0");
    let frames = stderr_lines_starting_with(&err, "frame ");
    assert_eq!(
        frames,
        vec![
            "frame 0: t=0 natoms=2 ntypes=1 cell=orthorhombic pbc=[true, true, true] volume=80.000 Å^3".to_string(),
            "frame 1: t=0 natoms=2 ntypes=1 cell=orthorhombic pbc=[true, true, true] volume=80.000 Å^3".to_string(),
            "frame 2: t=0 natoms=2 ntypes=1 cell=orthorhombic pbc=[true, true, true] volume=80.000 Å^3".to_string(),
        ],
        "stderr was:\n{err}"
    );
    // The summary leads, as every documented example shows it.
    let summary_at = err.find("read 3 frame(s)").unwrap();
    let first_frame_at = err.find("frame 0:").unwrap();
    assert!(
        summary_at < first_frame_at,
        "summary must come first:\n{err}"
    );

    // A window selects a subset and the summary counts that subset.
    let (_, err, ok) = rmda(&["info", "--start", "1", "--end", "3", path]);
    assert!(ok, "windowed info failed:\n{err}");
    assert_eq!(summary_line(&err), "read 2 frame(s), 2 atoms in frame 0");
    assert_eq!(stderr_lines_starting_with(&err, "frame ").len(), 2);

    let (_, err, ok) = rmda(&["info", "--stride", "2", path]);
    assert!(ok, "strided info failed:\n{err}");
    assert_eq!(summary_line(&err), "read 2 frame(s), 2 atoms in frame 0");
}

/// Gzip input is transparent to the subcommands: `info` and `convert` on a
/// `.xyz.gz` must report and produce exactly what they do on the plain file —
/// the suffix only changes how the bytes arrive, never what they mean.
#[test]
fn gzipped_input_reads_like_the_plain_file() {
    use std::io::Write;

    let plain = tempdir().join("gz_input.xyz");
    std::fs::write(plain, three_frames()).unwrap();
    let gz = tempdir().join("gz_input.xyz.gz");
    {
        let f = std::fs::File::create(&gz).unwrap();
        let mut enc = flate2::write::GzEncoder::new(f, flate2::Compression::default());
        enc.write_all(three_frames().as_bytes()).unwrap();
        enc.finish().unwrap();
    }
    let path = gz.to_str().unwrap();

    let (_, err, ok) = rmda(&["info", path]);
    assert!(ok, "info on .gz failed:\n{err}");
    assert_eq!(summary_line(&err), "read 3 frame(s), 2 atoms in frame 0");

    let out = tempdir().join("gz_input_out.dump");
    let (_, err, ok) = rmda(&[
        "convert",
        "--format",
        "dump",
        "-o",
        out.to_str().unwrap(),
        path,
    ]);
    assert!(ok, "convert from .gz failed:\n{err}");
    assert_eq!(summary_line(&err), "read 3 frame(s), 2 atoms in frame 0");
    let got = rmda_core::io::read_frames(&out).unwrap();
    assert_eq!(got.len(), 3);
    assert_eq!(got[0].particles.positions[0][0], 0.0);
    assert_eq!(got[2].particles.positions[0][0], 20.0);
}

/// The window is a hard boundary: with `--end 1` the corrupt second frame is
/// never parsed, and without an `end` it is, and the error says which.
#[test]
fn info_stops_at_the_window_before_reading_what_lies_past_it() {
    let p = tempdir().join("info_corrupt_behind.xyz");
    std::fs::write(&p, second_frame_corrupt()).unwrap();
    let path = p.to_str().unwrap();

    let (_, err, ok) = rmda(&["info", "--end", "1", path]);
    assert!(ok, "a frame outside the window must not be parsed:\n{err}");
    assert_eq!(summary_line(&err), "read 1 frame(s), 2 atoms in frame 0");
    assert_eq!(stderr_lines_starting_with(&err, "frame ").len(), 1);

    let (_, err, ok) = rmda(&["info", path]);
    assert!(!ok, "with no window the corrupt frame is in scope");
    assert!(err.contains("bad atom count"), "{err}");
}

/// `convert` writes frames as it reads them, so converting a window touches
/// only that window and still produces byte-identical frames.
#[test]
fn convert_streams_the_window_it_was_asked_for() {
    let p = tempdir().join("convert_three.xyz");
    std::fs::write(&p, three_frames()).unwrap();
    let path = p.to_str().unwrap();

    let out = tempdir().join("convert_window.xyz");
    let (_, err, ok) = rmda(&[
        "convert",
        "--format",
        "xyz",
        "--start",
        "1",
        "--end",
        "3",
        "-o",
        out.to_str().unwrap(),
        path,
    ]);
    assert!(ok, "convert failed:\n{err}");
    assert_eq!(summary_line(&err), "read 2 frame(s), 2 atoms in frame 0");
    let got = rmda_core::io::read_frames(&out).unwrap();
    assert_eq!(got.len(), 2, "window 1..3 is two frames");
    assert_eq!(got[0].particles.positions[0][0], 10.0, "first kept frame");
    assert_eq!(got[1].particles.positions[0][0], 20.0, "second kept frame");

    // The whole window to another format: still one frame in memory at a time,
    // and the result reads back as every frame.
    let out = tempdir().join("convert_all.dump");
    let (_, err, ok) = rmda(&[
        "convert",
        "--format",
        "dump",
        "-o",
        out.to_str().unwrap(),
        path,
    ]);
    assert!(ok, "convert to dump failed:\n{err}");
    assert_eq!(summary_line(&err), "read 3 frame(s), 2 atoms in frame 0");
    let got = rmda_core::io::read_frames(&out).unwrap();
    assert_eq!(got.len(), 3);
    assert_eq!(got[2].particles.positions[0][0], 20.0);
}

/// Without `-o`, `convert` names its output after the format it writes: a
/// dump in a file called `out.xyz` would be read back as extxyz.
#[test]
fn convert_without_output_names_the_file_after_its_format() {
    let dir = tempdir().join("convert_default_name");
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let p = dir.join("in.xyz");
    std::fs::write(&p, three_frames()).unwrap();
    for (format, name) in [("dump", "out.dump"), ("xyz", "out.xyz")] {
        let out = Command::new(env!("CARGO_BIN_EXE_rmda"))
            .args(["convert", "--format", format, p.to_str().unwrap()])
            .current_dir(&dir)
            .output()
            .expect("failed to run rmda");
        assert!(
            out.status.success(),
            "{}",
            String::from_utf8_lossy(&out.stderr)
        );
        let got = rmda_core::io::read_frames(&dir.join(name)).unwrap();
        assert_eq!(got.len(), 3, "{format} written to {name}");
    }
}

/// The quantitative half: a twelve-frame trajectory analysed for one frame must
/// peak near one frame's worth of memory.
///
/// Measured on the fixture below with this build: 13.62 MB while streaming,
/// 62.29 MB for the same data held as twelve frames (the `ws --reference` path
/// still reads a reference trajectory whole). The threshold sits between the
/// two with room on both sides, and the fixture size assertion keeps a future
/// edit of this test from quietly shrinking the signal.
///
/// Pinned to one thread: the parallel reader reads ahead by a bounded amount
/// per thread (debug build, 48 frames: ~22 MB at 1 thread, ~35 MB at 4, ~67 MB
/// at 12, flat in the frame count), so with every core the peak measured the
/// machine's core count rather than whether frames are retained.
#[cfg(target_os = "linux")]
#[test]
fn peak_rss_stays_near_one_frame_over_a_twelve_frame_trajectory() {
    const FRAMES: usize = 12;
    const PER_FRAME: usize = 48_668; // 23^3 FCC conventional cells
    const A: f64 = 3.615;
    const BASIS: [[f64; 3]; 4] = [
        [0.0, 0.0, 0.0],
        [0.5, 0.5, 0.0],
        [0.5, 0.0, 0.5],
        [0.0, 0.5, 0.5],
    ];
    // Anything materialising the input holds every frame plus the whole mapped
    // file: ~62 MB against a ~19 MB file. Streaming holds one frame: ~14 MB.
    const THRESHOLD: u64 = 30 * 1024 * 1024;

    let p = tempdir().join("rss_twelve_frames.xyz");
    {
        use std::io::Write;
        let mut w = std::io::BufWriter::new(std::fs::File::create(&p).unwrap());
        let n = (PER_FRAME as f64 / 4.0).cbrt().floor() as usize;
        let box_len = n as f64 * A;
        for _ in 0..FRAMES {
            writeln!(w, "{PER_FRAME}").unwrap();
            writeln!(
                w,
                "Lattice=\"{box_len} 0.0 0.0 0.0 {box_len} 0.0 0.0 0.0 {box_len}\" \
                 pbc=\"T T T\" Properties=species:S:1:pos:R:3"
            )
            .unwrap();
            let mut written = 0usize;
            'outer: for i in 0..n {
                for j in 0..n {
                    for k in 0..n {
                        for b in BASIS {
                            writeln!(
                                w,
                                "Cu {:.6} {:.6} {:.6}",
                                (i as f64 + b[0]) * A,
                                (j as f64 + b[1]) * A,
                                (k as f64 + b[2]) * A
                            )
                            .unwrap();
                            written += 1;
                            if written == PER_FRAME {
                                break 'outer;
                            }
                        }
                    }
                }
            }
        }
    }
    let file_bytes = std::fs::metadata(&p).unwrap().len();
    assert!(
        file_bytes > 15_000_000,
        "fixture is {file_bytes} B; the threshold assumes ~19 MB"
    );

    let (out, err, ok) = rmda(&[
        "--threads",
        "1",
        "--memreport",
        "structure",
        p.to_str().unwrap(),
    ]);
    assert!(ok, "structure failed:\n{err}");
    assert!(out.contains("cna_fcc:"), "no CNA counts in:\n{out}");
    let peak = parse_peak_rss(&err);
    assert!(
        peak < THRESHOLD,
        "peak RSS {} is above {} MB: the CLI is holding more than one frame of a \
         {FRAMES}-frame trajectory ({file_bytes} B on disk)",
        rmda_core::prof::fmt_bytes(peak),
        THRESHOLD / (1024 * 1024)
    );
}

/// `memreport: peak RSS 13.62 MB` -> bytes, in the units `fmt_bytes` prints.
#[cfg(target_os = "linux")]
fn parse_peak_rss(stderr: &str) -> u64 {
    let line = stderr
        .lines()
        .find(|l| l.starts_with("memreport: peak RSS"))
        .unwrap_or_else(|| panic!("no memreport line in:\n{stderr}"));
    let rest = line.trim_start_matches("memreport: peak RSS").trim();
    let mut parts = rest.split_whitespace();
    let value: f64 = parts
        .next()
        .unwrap_or_else(|| panic!("no value in {rest:?}"))
        .parse()
        .unwrap_or_else(|e| panic!("{rest:?} is not a number: {e}"));
    let unit = parts.next().unwrap_or("B");
    let scale = match unit {
        "GB" => 1024u64.pow(3),
        "MB" => 1024u64.pow(2),
        "KB" => 1024,
        "B" => 1,
        other => panic!("unexpected unit {other:?} in {rest:?}"),
    };
    (value * scale as f64) as u64
}
