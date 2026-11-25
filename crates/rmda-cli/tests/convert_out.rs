//! `rmda convert` outputs (design §8.6): format from the `-o` suffix, gzip
//! by suffix, binary and data output, `--props`, `--precision`,
//! `--prop-precision`, and that the default output is unchanged.

use std::io::Read;
use std::path::PathBuf;
use std::process::Command;

fn tempdir(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join("rmda-cli-convert-out").join(name);
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

fn rmda(args: &[&str]) -> (String, bool) {
    let out = Command::new(env!("CARGO_BIN_EXE_rmda"))
        .args(args)
        .output()
        .expect("failed to run rmda");
    (
        String::from_utf8_lossy(&out.stderr).into_owned(),
        out.status.success(),
    )
}

/// A dump with velocities, a compute column and full-precision positions.
fn input(dir: &std::path::Path, frames: usize) -> PathBuf {
    let mut s = String::new();
    for t in 0..frames {
        s.push_str(&format!(
            "ITEM: TIMESTEP\n{t}\nITEM: NUMBER OF ATOMS\n3\nITEM: BOX BOUNDS pp pp pp\n\
             0 10.5\n0 11\n-1 12\nITEM: ATOMS id type x y z vx vy vz c_pe element\n"
        ));
        s.push_str("7 2 1.2345678901234567 2.5 3.0000000000000004 0.1 -0.2 1e-7 -3.25 Ni\n");
        s.push_str("3 1 4.000000000000001 5.123456789012345 6 1.5 2.5 3.5 -4.125 Cu\n");
        s.push_str("9 2 7.7 8.8 9.9 0 0 0 0.3333333333333333 Ni\n");
    }
    let p = dir.join("in.dump");
    std::fs::write(&p, s).unwrap();
    p
}

fn read(p: &std::path::Path) -> Vec<rmda_core::data::Frame> {
    rmda_core::io::read_frames(p).unwrap()
}

fn gunzip(p: &std::path::Path) -> Vec<u8> {
    let mut v = Vec::new();
    flate2::read::MultiGzDecoder::new(std::fs::File::open(p).unwrap())
        .read_to_end(&mut v)
        .unwrap();
    v
}

#[test]
fn the_format_comes_from_the_suffix_and_gz_compresses_the_same_bytes() {
    let dir = tempdir("suffix");
    let inp = input(&dir, 2);
    let i = inp.to_str().unwrap();
    for ext in ["dump", "xyz"] {
        let plain = dir.join(format!("o.{ext}"));
        let gz = dir.join(format!("o.{ext}.gz"));
        let (e, ok) = rmda(&["convert", "-o", plain.to_str().unwrap(), i]);
        assert!(ok, "{e}");
        let (e, ok) = rmda(&["convert", "-o", gz.to_str().unwrap(), i]);
        assert!(ok, "{e}");
        assert!(gunzip(&gz) == std::fs::read(&plain).unwrap(), "{ext}");
        // and the same bytes as an explicit --format
        let explicit = dir.join(format!("x.{ext}"));
        let (_, ok) = rmda(&[
            "convert",
            "--format",
            ext,
            "-o",
            explicit.to_str().unwrap(),
            i,
        ]);
        assert!(ok);
        assert!(std::fs::read(&explicit).unwrap() == std::fs::read(&plain).unwrap());
        assert_eq!(read(&gz).len(), 2);
    }
    let (e, ok) = rmda(&["convert", "-o", dir.join("o.weird").to_str().unwrap(), i]);
    assert!(!ok && e.contains("--format"), "{e}");
    let (e, ok) = rmda(&["convert", "-o", dir.join("o.bin.gz").to_str().unwrap(), i]);
    assert!(!ok && e.contains("gzipped binary"), "{e}");
}

#[test]
fn full_precision_props_round_trip_through_every_output_format() {
    let dir = tempdir("props");
    let inp = input(&dir, 1);
    let want = &read(&inp)[0];
    let i = inp.to_str().unwrap();
    for name in ["o.dump", "o.xyz", "o.bin", "o.dump.gz", "o.xyz.gz"] {
        let out = dir.join(name);
        let (e, ok) = rmda(&[
            "convert",
            "--props",
            "all",
            "--precision",
            "full",
            "-o",
            out.to_str().unwrap(),
            i,
        ]);
        assert!(ok, "{name}: {e}");
        assert!(e.contains("writing 4 properties: vx, vy, vz, c_pe"), "{e}");
        let got = &read(&out)[0];
        let bits = |f: &rmda_core::data::Frame| -> Vec<u64> {
            f.particles
                .positions
                .iter()
                .flatten()
                .map(|v| v.to_bits())
                .collect()
        };
        assert!(bits(got) == bits(want), "{name}: positions");
        assert_eq!(got.particles.ids, want.particles.ids, "{name}");
        assert_eq!(got.particles.types, want.particles.types, "{name}");
        if !name.contains("bin") {
            assert_eq!(got.type_labels, want.type_labels, "{name}");
        }
        for p in ["vx", "vy", "vz", "c_pe"] {
            assert_eq!(
                got.particles.props.get(p).unwrap().as_f64(),
                want.particles.props.get(p).unwrap().as_f64(),
                "{name}: {p}"
            );
        }
    }
}

#[test]
fn data_output_carries_velocities_and_refuses_a_second_frame() {
    let dir = tempdir("data");
    let one = input(&dir, 1);
    let out = dir.join("o.data");
    let (e, ok) = rmda(&[
        "convert",
        "--props",
        "all",
        "-o",
        out.to_str().unwrap(),
        one.to_str().unwrap(),
    ]);
    assert!(ok, "{e}");
    assert!(
        e.contains("not written (the format cannot carry them): c_pe"),
        "{e}"
    );
    let got = &read(&out)[0];
    let want = &read(&one)[0];
    assert_eq!(got.particles.ids, want.particles.ids);
    assert_eq!(got.type_labels, want.type_labels);
    assert_eq!(
        got.particles.props.get("vz").unwrap().as_f64(),
        want.particles.props.get("vz").unwrap().as_f64()
    );
    let (e, ok) = rmda(&[
        "convert",
        "--props",
        "c_pe",
        "-o",
        out.to_str().unwrap(),
        one.to_str().unwrap(),
    ]);
    assert!(!ok && e.contains("c_pe"), "{e}");
    let two = input(&dir, 2);
    let (e, ok) = rmda(&[
        "convert",
        "-o",
        out.to_str().unwrap(),
        two.to_str().unwrap(),
    ]);
    assert!(!ok && e.contains("one frame"), "{e}");
}

#[test]
fn prop_precision_rounds_float_columns_only() {
    let dir = tempdir("precision");
    let inp = input(&dir, 1);
    let out = dir.join("o.dump");
    let (e, ok) = rmda(&[
        "convert",
        "--props",
        "c_pe",
        "--prop-precision",
        "2",
        "-o",
        out.to_str().unwrap(),
        inp.to_str().unwrap(),
    ]);
    assert!(ok, "{e}");
    let text = std::fs::read_to_string(&out).unwrap();
    assert!(
        text.contains("ITEM: ATOMS id type x y z element c_pe\n"),
        "{text}"
    );
    assert!(
        text.contains("\n7 2 1.2345678901 2.5000000000 3.0000000000 Ni -3.25\n"),
        "{text}"
    );
    assert!(text.contains(" Ni 0.33\n"), "{text}");
    // without --props it is refused rather than ignored
    let (_, ok) = rmda(&[
        "convert",
        "--prop-precision",
        "2",
        "-o",
        out.to_str().unwrap(),
        inp.to_str().unwrap(),
    ]);
    assert!(!ok);
}

#[test]
fn without_new_options_the_output_is_unchanged() {
    let dir = tempdir("legacy");
    let inp = input(&dir, 1);
    let out = dir.join("o.dump");
    let (_, ok) = rmda(&[
        "convert",
        "--format",
        "dump",
        "-o",
        out.to_str().unwrap(),
        inp.to_str().unwrap(),
    ]);
    assert!(ok);
    let text = std::fs::read_to_string(&out).unwrap();
    assert_eq!(
        text,
        "ITEM: TIMESTEP\n0\nITEM: NUMBER OF ATOMS\n3\nITEM: BOX BOUNDS pp pp pp\n\
         0.0000000000 10.5000000000\n0.0000000000 11.0000000000\n-1.0000000000 12.0000000000\n\
         ITEM: ATOMS id type x y z\n\
         7 2 1.2345678901 2.5000000000 3.0000000000\n\
         3 1 4.0000000000 5.1234567890 6.0000000000\n\
         9 2 7.7000000000 8.8000000000 9.9000000000\n"
    );
}

/// `--reduce-tilt` flips an over-tilted box LAMMPS-style before writing; the
/// atoms stay where they are (wrapped into the reduced box), and an extended
/// XYZ written from a dump keeps the dump's timestep.
#[test]
fn reduce_tilt_flips_the_box_and_keeps_the_atoms() {
    let dir = tempdir("reduce_tilt");
    // xy = 7 > lx/2 = 5; the bound rows are as LAMMPS writes them
    // (xhi_bound = xhi + max(0, xy, xz, xy + xz) = 17).
    let s = "ITEM: TIMESTEP\n4200\nITEM: NUMBER OF ATOMS\n2\n\
             ITEM: BOX BOUNDS xy xz yz pp pp pp\n0 17 7\n0 10 0\n0 10 0\n\
             ITEM: ATOMS id type x y z\n1 1 1 1 1\n2 1 15 9 5\n";
    let inp = dir.join("tilt.dump");
    std::fs::write(&inp, s).unwrap();
    let i = inp.to_str().unwrap();
    let plain = dir.join("plain.xyz");
    let red = dir.join("red.dump");
    assert!(rmda(&["convert", i, "-o", plain.to_str().unwrap()]).1);
    let (err, ok) = rmda(&["convert", i, "-o", red.to_str().unwrap(), "--reduce-tilt"]);
    assert!(ok, "{err}");
    let (a, b) = (&read(&plain)[0], &read(&red)[0]);
    assert_eq!(a.meta.timestep, 4200, "extxyz Timestep= is read back");
    assert_eq!(b.meta.timestep, 4200);
    assert_eq!(a.cell.h_matrix().m[1][0], 7.0);
    assert_eq!(b.cell.h_matrix().m[1][0], -3.0);
    // Atom 1 at (1, 1, 1) is inside both boxes and keeps its position; atom 2
    // at (15, 9, 5) is outside the flipped box and comes back one `a` lower.
    assert_eq!(b.particles.positions[0], [1.0, 1.0, 1.0]);
    let p = b.particles.positions[1];
    assert!(
        (p[0] - 5.0).abs() < 1e-9 && (p[1] - 9.0).abs() < 1e-9 && (p[2] - 5.0).abs() < 1e-9,
        "{p:?}"
    );
    // Data output of the over-tilted box notes the flag.
    let data = dir.join("tilt.data");
    let (err, ok) = rmda(&["convert", i, "-o", data.to_str().unwrap()]);
    assert!(ok && err.contains("--reduce-tilt"), "{err}");
}

/// `--atom-style` reads a data file whose 6-column `Atoms` rows are ambiguous.
#[test]
fn atom_style_reads_an_ambiguous_data_file() {
    let dir = tempdir("atom_style");
    let s = "LAMMPS data file\n\n2 atoms\n2 atom types\n\n0 10 xlo xhi\n0 10 ylo yhi\n\
             0 10 zlo zhi\n\nAtoms\n\n1 1 0.5 1 2 3\n2 2 -0.5 4 5 6\n";
    let inp = dir.join("six.data");
    std::fs::write(&inp, s).unwrap();
    let out = dir.join("out.dump");
    let (i, o) = (inp.to_str().unwrap(), out.to_str().unwrap());
    let (err, ok) = rmda(&["convert", i, "-o", o]);
    assert!(!ok && err.contains("--atom-style"), "{err}");
    let (err, ok) = rmda(&[
        "convert",
        i,
        "-o",
        o,
        "--atom-style",
        "charge",
        "--props",
        "q",
    ]);
    assert!(ok, "{err}");
    let f = &read(&out)[0];
    assert_eq!(f.particles.positions[1], [4.0, 5.0, 6.0]);
    assert_eq!(
        f.particles.props.get("q").unwrap().as_f64().unwrap(),
        &vec![0.5, -0.5]
    );
}

/// A `.gz` input reads the same through every inflate path: serial
/// (`--threads 1`), parallel members (`convert -o x.gz`), one member, and one
/// member with a gzip index from `rmda index`.
#[test]
fn gzip_input_reads_the_same_serial_parallel_and_indexed() {
    use std::io::Write;
    let dir = tempdir("gz_index");
    // ~14 MB of dump text: several 4 MB members when written by convert,
    // two access points in the index.
    let mut s = String::new();
    for t in 0..6 {
        s.push_str(&format!(
            "ITEM: TIMESTEP\n{t}\nITEM: NUMBER OF ATOMS\n40000\nITEM: BOX BOUNDS pp pp pp\n\
             0 100\n0 100\n0 100\nITEM: ATOMS id type x y z vx\n"
        ));
        for i in 0..40000u64 {
            let h = i.wrapping_mul(0x9e3779b97f4a7c15).wrapping_add(t);
            s.push_str(&format!(
                "{} {} {:.10} {:.10} {:.10} {:.6}\n",
                i + 1,
                1 + (h >> 63),
                (h >> 11) as f64 / 9.1e13,
                (h >> 13) as f64 / 2.3e13,
                (h >> 17) as f64 / 1.4e12,
                (h >> 40) as f64 / 1e6
            ));
        }
    }
    let plain = dir.join("t.dump");
    std::fs::write(&plain, &s).unwrap();
    let members = dir.join("members.dump.gz");
    let (e, ok) = rmda(&[
        "convert",
        plain.to_str().unwrap(),
        "-o",
        members.to_str().unwrap(),
    ]);
    assert!(ok, "{e}");
    let one = dir.join("one.dump.gz");
    let mut enc = flate2::write::GzEncoder::new(
        std::fs::File::create(&one).unwrap(),
        flate2::Compression::default(),
    );
    enc.write_all(s.as_bytes()).unwrap();
    enc.finish().unwrap();
    let info = |p: &std::path::Path, threads: &str| {
        let out = Command::new(env!("CARGO_BIN_EXE_rmda"))
            .args(["info", "--threads", threads, p.to_str().unwrap()])
            .output()
            .unwrap();
        assert!(
            out.status.success(),
            "{}",
            String::from_utf8_lossy(&out.stderr)
        );
        out.stdout
    };
    let want = info(&plain, "4");
    for p in [&members, &one] {
        assert!(info(p, "1") == want, "{}", p.display());
        assert!(info(p, "4") == want, "{}", p.display());
    }
    let out = Command::new(env!("CARGO_BIN_EXE_rmda"))
        .args(["index", one.to_str().unwrap()])
        .output()
        .unwrap();
    let text = String::from_utf8_lossy(&out.stdout);
    assert!(
        out.status.success() && text.contains("indexed gzip:"),
        "{text}"
    );
    assert!(dir.join("one.dump.gz.rmdaidx").exists());
    assert!(info(&one, "4") == want);
    // convert from the indexed file gives the plain file's bytes
    let back = dir.join("back.dump");
    let (e, ok) = rmda(&[
        "convert",
        one.to_str().unwrap(),
        "-o",
        back.to_str().unwrap(),
    ]);
    assert!(ok, "{e}");
    let direct = dir.join("direct.dump");
    let (e, ok) = rmda(&[
        "convert",
        plain.to_str().unwrap(),
        "-o",
        direct.to_str().unwrap(),
    ]);
    assert!(ok, "{e}");
    assert!(std::fs::read(&back).unwrap() == std::fs::read(&direct).unwrap());
}

/// `--out-atom-style` picks the data-file style on output; its columns
/// come from the frame's properties and must exist.
#[test]
fn out_atom_style_sets_the_data_style() {
    let dir = tempdir("out_atom_style");
    let s = "LAMMPS data file\n\n2 atoms\n2 atom types\n\n0 10 xlo xhi\n0 10 ylo yhi\n\
             0 10 zlo zhi\n\nAtoms # full\n\n1 7 1 0.5 1 2 3\n2 8 2 -0.5 4 5 6\n";
    let inp = dir.join("full.data");
    std::fs::write(&inp, s).unwrap();
    let i = inp.to_str().unwrap();
    let out = dir.join("out.data");
    let o = out.to_str().unwrap();
    // default: no --props, so atomic (unchanged)
    let (err, ok) = rmda(&["convert", i, "-o", o]);
    assert!(ok, "{err}");
    let text = std::fs::read_to_string(&out).unwrap();
    assert!(text.contains("Atoms # atomic\n"), "{text}");
    for (style, row) in [
        ("full", "1 7 1 0.5 1e0 2e0 3e0"),
        ("molecular", "1 7 1 1e0 2e0 3e0"),
        ("charge", "1 1 0.5 1e0 2e0 3e0"),
    ] {
        let (err, ok) = rmda(&["convert", i, "-o", o, "--out-atom-style", style]);
        assert!(ok, "{err}");
        let text = std::fs::read_to_string(&out).unwrap();
        assert!(
            text.contains(&format!("Atoms # {style}\n\n{row}\n")),
            "{text}"
        );
    }
    let (err, ok) = rmda(&["convert", i, "-o", o, "--out-atom-style", "sphere"]);
    assert!(!ok && err.contains("diameter"), "{err}");
    let (err, ok) = rmda(&[
        "convert",
        i,
        "-o",
        o,
        "--out-atom-style",
        "charge",
        "--props",
        "mol",
    ]);
    assert!(!ok && err.contains("no column"), "{err}");
    let dump = dir.join("out.dump");
    let (err, ok) = rmda(&[
        "convert",
        i,
        "-o",
        dump.to_str().unwrap(),
        "--out-atom-style",
        "full",
    ]);
    assert!(!ok && err.contains("data-file output only"), "{err}");
}

/// `--keep-general` carries a general-triclinic data file through
/// `convert` in its own frame; without it the box is restricted.
#[test]
fn keep_general_keeps_the_general_box() {
    let dir = tempdir("keep_general");
    let s = "LAMMPS data file\n\n2 atoms\n1 atom types\n\n\
             4 1.25 0.5 avec\n0 4.5 0.75 bvec\n0.25 0 5 cvec\n1 2 3 abc origin\n\n\
             Atoms # atomic\n\n1 1 1.5 2.5 3.5\n2 1 3.25 4.75 6.125\n";
    let inp = dir.join("gen.data");
    std::fs::write(&inp, s).unwrap();
    let i = inp.to_str().unwrap();
    let out = dir.join("out.data");
    let o = out.to_str().unwrap();
    let (err, ok) = rmda(&["convert", i, "-o", o, "--keep-general"]);
    assert!(ok, "{err}");
    let text = std::fs::read_to_string(&out).unwrap();
    assert!(
        text.contains("4 1.25 0.5 avec\n0 4.5 0.75 bvec\n0.25 0 5 cvec\n1 2 3 abc origin\n"),
        "{text}"
    );
    assert!(text.contains("1 1 1.5e0 2.5e0 3.5e0\n"), "{text}");
    let (err, ok) = rmda(&["convert", i, "-o", o]);
    assert!(ok, "{err}");
    let text = std::fs::read_to_string(&out).unwrap();
    assert!(text.contains("xlo xhi") && !text.contains("avec"), "{text}");
}
