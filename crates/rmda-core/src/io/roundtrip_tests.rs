//! Writer round trips (design §8.6): read(write(frame)) gives back the frame
//! -- positions bit for bit at full precision, properties exactly, and with a
//! reduced precision exactly the printed decimals -- in every output format,
//! plain and gzipped.

use crate::cell::{image_ipart, image_jpart, image_kpart, pack_image, Mat3, UnitCell};
use crate::data::particle::ParticleData;
use crate::data::property::{DType, Property, TypedVec};
use crate::data::Frame;
use crate::io::engine::{parse_f64, Cursor};
use crate::io::writer::{FloatFmt, GzWriter, PropSelect, WriteOptions, Writer, GZ_BLOCK};
use crate::io::Format;
use std::io::{Read, Write};

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
    fn unit(&mut self) -> f64 {
        (self.next() >> 11) as f64 / (1u64 << 53) as f64
    }
    fn below(&mut self, n: u64) -> u64 {
        self.next() % n
    }
}

/// A frame with everything the writers can carry, and values chosen to break
/// a lossy writer: full 53-bit mantissas, -0.0, tiny and huge magnitudes.
fn rich_frame(n: usize, seed: u64) -> Frame {
    let mut rng = Rng(seed);
    let cell = UnitCell::from_matrix(
        Mat3::from_rows([31.25, 0.0, 0.0], [0.0, 27.5, 0.0], [0.0, 0.0, 40.0]),
        [true, true, false],
        [-3.0, 0.5, 1.25],
    );
    let mut p = ParticleData::new(n);
    for i in 0..n {
        let mut x = [0.0; 3];
        for (d, v) in x.iter_mut().enumerate() {
            *v = cell.origin[d] + rng.unit() * 27.0;
        }
        if i == 0 {
            x = [-0.0, 1e-9, 26.999999999999996];
        }
        p.positions.push(x);
        p.ids.push(1000 + (n - i) as i64 * 7); // not 1..n, not sorted
        p.types.push(1 + rng.below(3) as u32);
    }
    // every type present, so labels read back from a dump are complete
    p.types[0] = 1;
    p.types[n - 1] = 2;
    p.types[n / 2] = 3;
    p.images = Some(
        (0..n)
            .map(|i| pack_image(i as i64 % 3 - 1, -(i as i64 % 5), 2))
            .collect(),
    );
    p.selection = Some((0..n).map(|i| (i % 2) as u8).collect());
    let f64s = |rng: &mut Rng, k: usize| -> Vec<f64> {
        (0..n * k)
            .map(|j| match j % 7 {
                0 => (rng.unit() - 0.5) * 1e-12,
                1 => (rng.unit() - 0.5) * 3e17,
                _ => rng.unit() * 20.0 - 10.0,
            })
            .collect()
    };
    let vx = f64s(&mut rng, 1);
    let vy = f64s(&mut rng, 1);
    let vz = f64s(&mut rng, 1);
    p.props
        .register(Property::from_data("vx", 1, TypedVec::F64(vx)));
    p.props
        .register(Property::from_data("vy", 1, TypedVec::F64(vy)));
    p.props
        .register(Property::from_data("vz", 1, TypedVec::F64(vz)));
    let pe = f64s(&mut rng, 1);
    p.props
        .register(Property::from_data("c_pe", 1, TypedVec::F64(pe)));
    let q = f64s(&mut rng, 1);
    p.props
        .register(Property::from_data("q", 1, TypedVec::F64(q)));
    let mol: Vec<i64> = (0..n).map(|i| (i / 4) as i64 + 1).collect();
    p.props
        .register(Property::from_data("mol", 1, TypedVec::I64(mol)));
    let st: Vec<u8> = (0..n).map(|_| rng.below(6) as u8).collect();
    p.props
        .register(Property::from_data("ptm.structure", 1, TypedVec::U8(st)));
    let quat = f64s(&mut rng, 4);
    p.props.register(Property::from_data(
        "ptm.orientation",
        4,
        TypedVec::F64(quat),
    ));
    let disp: Vec<f32> = (0..n * 3)
        .map(|_| (rng.unit() as f32 - 0.5) * 3.0)
        .collect();
    p.props
        .register(Property::from_data("defect.disp", 3, TypedVec::F32(disp)));
    let big: Vec<i64> = (0..n).map(|i| (1i64 << 52) + i as i64 * 977).collect();
    p.props
        .register(Property::from_data("i_tag", 1, TypedVec::I64(big)));
    let mut f = Frame::new(cell, p);
    f.meta.timestep = 1234;
    f.type_labels = vec!["Cu".into(), "Ni".into(), "Al".into()];
    f.type_masses = vec![63.546, 58.6934, 26.9815385];
    f
}

fn write(fmt: Format, f: &Frame, opts: &WriteOptions) -> Vec<u8> {
    let mut b = Vec::new();
    Writer::write_with(&mut b, fmt, std::slice::from_ref(f), opts).unwrap();
    b
}

fn read(fmt: Format, b: &[u8]) -> Frame {
    let cur = Cursor::from_bytes(b, "rt");
    match fmt {
        Format::Xyz => crate::io::xyz::read_next_from_cursor(&cur)
            .unwrap()
            .unwrap(),
        Format::LammpsDump => crate::io::lammps::read_next_dump_from_cursor(&cur)
            .unwrap()
            .unwrap(),
        Format::LammpsData => crate::io::lammps::parse_data_frame(&cur).unwrap(),
        Format::LammpsDumpBinary => crate::io::binary::read_next(&cur).unwrap().unwrap(),
    }
}

fn full(props: PropSelect) -> WriteOptions {
    WriteOptions {
        extended: true,
        props,
        precision: Some(FloatFmt::Shortest),
        ..WriteOptions::default()
    }
}

fn bits(f: &Frame) -> Vec<u64> {
    f.particles
        .positions
        .iter()
        .flat_map(|p| p.iter().map(|c| c.to_bits()))
        .collect()
}

/// Component `c` of atom `i` of `p`, as the f64 a reader stores.
fn val(p: &Property, i: usize, c: usize) -> f64 {
    p.get_flat(i * p.components + c).unwrap()
}

/// `got` holds `want`'s values (as f64, bit for bit) under `name`, or under
/// `name[1] ..` when `split` (dump formats).
fn same_prop(got: &Frame, want: &Property, split: bool, ctx: &str) {
    let n = got.natoms();
    for c in 0..want.components {
        let (q, qc) = if split && want.components > 1 {
            let name = format!("{}[{}]", want.name, c + 1);
            (got.particles.props.get(&name), 0)
        } else {
            (got.particles.props.get(&want.name), c)
        };
        let q = q.unwrap_or_else(|| panic!("{ctx}: {} not read back", want.name));
        for i in 0..n {
            let (a, b) = (val(q, i, qc), val(want, i, c));
            assert_eq!(
                a.to_bits(),
                b.to_bits(),
                "{ctx}: {} atom {i} comp {c}",
                want.name
            );
        }
    }
}

fn same_atoms(got: &Frame, want: &Frame, ctx: &str) {
    assert_eq!(got.natoms(), want.natoms(), "{ctx}");
    assert!(bits(got) == bits(want), "{ctx}: positions differ");
    assert_eq!(got.particles.types, want.particles.types, "{ctx}: types");
}

fn images(f: &Frame) -> Vec<[i64; 3]> {
    f.particles
        .images
        .as_ref()
        .unwrap()
        .iter()
        .map(|&m| [image_ipart(m), image_jpart(m), image_kpart(m)])
        .collect()
}

#[test]
fn shortest_spelling_round_trips_every_double() {
    let mut rng = Rng(0x9E37_79B9_7F4A_7C15);
    let mut buf = Vec::new();
    let check = |v: f64, buf: &mut Vec<u8>| {
        buf.clear();
        crate::io::writer::write_shortest(buf, v).unwrap();
        let back = parse_f64(buf).unwrap();
        assert!(
            back.to_bits() == v.to_bits() || (v.is_nan() && back.is_nan()),
            "{v:e} -> {}",
            String::from_utf8_lossy(buf)
        );
        assert_eq!(
            std::str::from_utf8(buf)
                .unwrap()
                .parse::<f64>()
                .unwrap()
                .to_bits(),
            back.to_bits()
        );
    };
    for v in [
        0.0,
        -0.0,
        1e-5,
        9.999999999999999e-6,
        1e16,
        9999999999999998.0,
        f64::MIN_POSITIVE,
        5e-324,
        f64::MAX,
        -f64::MAX,
        0.1,
        1.0 / 3.0,
    ] {
        check(v, &mut buf);
    }
    for _ in 0..200_000 {
        let v = f64::from_bits(rng.next());
        if v.is_finite() {
            check(v, &mut buf);
        }
    }
}

#[test]
fn text_dump_round_trips_positions_labels_images_and_properties() {
    let f = rich_frame(3000, 11);
    let b = write(Format::LammpsDump, &f, &full(PropSelect::All));
    let g = read(Format::LammpsDump, &b);
    same_atoms(&g, &f, "dump");
    assert_eq!(g.particles.ids, f.particles.ids);
    assert_eq!(images(&g), images(&f));
    assert_eq!(g.type_labels, f.type_labels);
    assert_eq!(g.meta.timestep, 1234);
    for p in f.particles.props.iter() {
        same_prop(&g, p, true, "dump");
    }
    // dump columns carry no type: LAMMPS-named integer columns read as i64,
    // every other column as f64 holding the same value
    assert_eq!(g.particles.props.get("mol").unwrap().dtype, DType::I64);
    assert_eq!(g.particles.props.get("i_tag").unwrap().dtype, DType::I64);
    assert_eq!(
        g.particles.props.get("ptm.structure").unwrap().dtype,
        DType::F64
    );
    // the box: the bounds as printed, so the cell to rounding of xhi - xlo
    for (a, b) in [
        (g.cell.origin, f.cell.origin),
        (g.cell.h_matrix().m[0], f.cell.h_matrix().m[0]),
    ] {
        for d in 0..3 {
            assert!((a[d] - b[d]).abs() <= 1e-12 * (1.0 + b[d].abs()));
        }
    }
    assert_eq!(g.cell.pbc, f.cell.pbc);
}

#[test]
fn reduced_property_precision_reads_back_as_exactly_the_printed_decimals() {
    let f = rich_frame(500, 12);
    for fmt in [Format::LammpsDump, Format::Xyz] {
        for p in [0usize, 3, 7] {
            let opts = WriteOptions {
                prop_precision: Some(FloatFmt::Fixed(p)),
                ..full(PropSelect::Names(vec![
                    "c_pe".into(),
                    "defect.disp".into(),
                    "mol".into(),
                ]))
            };
            let g = read(fmt, &write(fmt, &f, &opts));
            same_atoms(&g, &f, "positions stay exact");
            let split = fmt == Format::LammpsDump;
            for name in ["c_pe", "defect.disp"] {
                let want = f.particles.props.get(name).unwrap();
                for c in 0..want.components {
                    let (q, qc) = if split && want.components > 1 {
                        (
                            g.particles
                                .props
                                .get(&format!("{name}[{}]", c + 1))
                                .unwrap(),
                            0,
                        )
                    } else {
                        (g.particles.props.get(name).unwrap(), c)
                    };
                    for i in 0..f.natoms() {
                        let printed = format!("{:.p$}", val(want, i, c));
                        let expect: f64 = printed.parse().unwrap();
                        assert_eq!(
                            val(q, i, qc).to_bits(),
                            expect.to_bits(),
                            "{fmt:?} {name} p={p}"
                        );
                    }
                }
            }
            // integers are never rounded
            same_prop(&g, f.particles.props.get("mol").unwrap(), split, "mol");
        }
    }
}

#[test]
fn reduced_position_precision_reads_back_as_the_printed_decimals() {
    let f = rich_frame(300, 13);
    for fmt in [Format::LammpsDump, Format::Xyz, Format::LammpsData] {
        let opts = WriteOptions {
            precision: Some(FloatFmt::Fixed(4)),
            ..WriteOptions::default()
        };
        let g = read(fmt, &write(fmt, &f, &opts));
        for (a, b) in g.particles.positions.iter().zip(&f.particles.positions) {
            for d in 0..3 {
                let expect: f64 = format!("{:.4}", b[d]).parse().unwrap();
                assert_eq!(a[d].to_bits(), expect.to_bits(), "{fmt:?}");
            }
        }
    }
}

#[test]
fn extxyz_round_trips_ids_selection_labels_and_properties() {
    let f = rich_frame(2500, 14);
    let b = write(Format::Xyz, &f, &full(PropSelect::All));
    let head = String::from_utf8_lossy(&b[..400.min(b.len())]).into_owned();
    assert!(head.contains(":id:I:1"), "{head}");
    assert!(head.contains("species=\"Cu Ni Al\""), "{head}");
    let g = read(Format::Xyz, &b);
    same_atoms(&g, &f, "xyz");
    assert_eq!(
        g.particles.ids, f.particles.ids,
        "the id column is the atom id"
    );
    assert!(g.particles.props.get("id").is_none());
    assert_eq!(g.particles.selection, f.particles.selection);
    assert_eq!(
        g.type_labels, f.type_labels,
        "labels in type order, not sorted"
    );
    for p in f.particles.props.iter() {
        same_prop(&g, p, false, "xyz");
        let q = g.particles.props.get(&p.name).unwrap();
        assert_eq!(q.components, p.components);
    }
    // the cell, printed exactly, reads back exactly
    assert_eq!(g.cell.h_matrix().m, f.cell.h_matrix().m);
    assert_eq!(g.cell.origin, f.cell.origin);
}

#[test]
fn data_file_round_trips_styles_velocities_labels_and_images() {
    let f = rich_frame(1500, 15);
    for (sel, style) in [
        (vec!["q", "mol", "vx", "vy", "vz"], "full"),
        (vec!["q"], "charge"),
        (vec!["mol"], "molecular"),
        (vec![], "atomic"),
    ] {
        let sel: Vec<String> = sel.into_iter().map(String::from).collect();
        let opts = WriteOptions {
            extended: true,
            props: PropSelect::Names(sel.clone()),
            ..WriteOptions::default()
        };
        let b = write(Format::LammpsData, &f, &opts);
        let text = String::from_utf8_lossy(&b);
        assert!(text.contains(&format!("Atoms # {style}\n")), "{style}");
        let g = read(Format::LammpsData, &b);
        same_atoms(&g, &f, style);
        assert_eq!(g.particles.ids, f.particles.ids);
        assert_eq!(images(&g), images(&f));
        assert_eq!(g.type_labels, f.type_labels);
        assert_eq!(g.type_masses, f.type_masses);
        for name in &sel {
            same_prop(&g, f.particles.props.get(name).unwrap(), false, style);
        }
        // the box is printed exactly
        assert_eq!(g.cell.origin, f.cell.origin);
    }
    // `all` writes what fits and names the rest
    let (written, skipped) =
        crate::io::writer::planned_props(Format::LammpsData, &f, &full(PropSelect::All)).unwrap();
    assert_eq!(written, ["vx", "vy", "vz", "q", "mol"]);
    assert!(skipped.contains(&"c_pe".to_string()));
    // naming one that does not fit is an error
    let mut sink = Vec::new();
    let err = Writer::write_with(
        &mut sink,
        Format::LammpsData,
        std::slice::from_ref(&f),
        &full(PropSelect::Names(vec!["c_pe".into()])),
    )
    .unwrap_err();
    assert!(err.to_string().contains("c_pe"), "{err}");
    let err = Writer::write_with(
        &mut sink,
        Format::LammpsData,
        std::slice::from_ref(&f),
        &full(PropSelect::Names(vec!["vx".into()])),
    )
    .unwrap_err();
    assert!(err.to_string().contains("vx, vy, vz"), "{err}");
}

/// `WriteOptions::atom_style` (`convert --out-atom-style`): the style's
/// columns come from the frame whether selected or not, a missing one is an
/// error, selected style properties outside it are refused when named and
/// skipped under `all`, and `atomic` without properties is the default bytes.
#[test]
fn data_file_atom_style_can_be_forced() {
    let f = rich_frame(1200, 21);
    let style = |s: &str, props: PropSelect| WriteOptions {
        atom_style: Some(s.into()),
        props,
        ..WriteOptions::default()
    };
    for s in ["full", "charge", "molecular", "bond", "angle", "atomic"] {
        let b = write(Format::LammpsData, &f, &style(s, PropSelect::None));
        let text = String::from_utf8_lossy(&b);
        assert!(text.contains(&format!("Atoms # {s}\n")), "{s}");
        let g = read(Format::LammpsData, &b);
        same_atoms(&g, &f, s);
        let cols: &[&str] = match s {
            "full" => &["q", "mol"],
            "charge" => &["q"],
            "molecular" | "bond" | "angle" => &["mol"],
            _ => &[],
        };
        for name in cols {
            same_prop(&g, f.particles.props.get(name).unwrap(), false, s);
        }
        for name in ["q", "mol"] {
            assert_eq!(
                g.particles.props.get(name).is_some(),
                cols.contains(&name),
                "{s} {name}"
            );
        }
    }
    assert_eq!(
        write(Format::LammpsData, &f, &style("atomic", PropSelect::None)),
        write(Format::LammpsData, &f, &WriteOptions::default())
    );
    // `charge` with q selected is what the automatic choice writes.
    let q = PropSelect::Names(vec!["q".into()]);
    assert_eq!(
        write(Format::LammpsData, &f, &style("charge", q.clone())),
        write(
            Format::LammpsData,
            &f,
            &WriteOptions {
                props: q,
                ..WriteOptions::default()
            }
        )
    );
    let mut sink = Vec::new();
    let mut try_write = |o: &WriteOptions| {
        Writer::write_with(&mut sink, Format::LammpsData, std::slice::from_ref(&f), o)
            .map_err(|e| e.to_string())
    };
    // sphere needs diameter and density, which the frame lacks
    let e = try_write(&style("sphere", PropSelect::None)).unwrap_err();
    assert!(e.contains("\"diameter\""), "{e}");
    // a named property the style has no column for
    let e = try_write(&style("charge", PropSelect::Names(vec!["mol".into()]))).unwrap_err();
    assert!(e.contains("mol") && e.contains("charge"), "{e}");
    assert!(try_write(&style("frobnicate", PropSelect::None))
        .unwrap_err()
        .contains("unknown atom style"));
    // under `all`, mol is skipped by `charge`; velocities still go
    let (written, skipped) =
        crate::io::writer::planned_props(Format::LammpsData, &f, &style("charge", PropSelect::All))
            .unwrap();
    assert_eq!(written, ["q", "vx", "vy", "vz"]);
    assert!(skipped.contains(&"mol".to_string()), "{skipped:?}");
    // sphere from properties of a frame that has them
    let mut s = rich_frame(300, 22);
    let n = s.natoms();
    for name in ["diameter", "density"] {
        s.particles.props.register(Property::from_data(
            name,
            1,
            TypedVec::F64((0..n).map(|i| 1.0 + i as f64 / 7.0).collect()),
        ));
    }
    let g = read(
        Format::LammpsData,
        &write(Format::LammpsData, &s, &style("sphere", PropSelect::None)),
    );
    same_atoms(&g, &s, "sphere");
    for name in ["diameter", "density"] {
        same_prop(&g, s.particles.props.get(name).unwrap(), false, "sphere");
    }
}

#[test]
fn binary_dump_round_trips_properties_exactly() {
    let f = rich_frame(4000, 16);
    let g = read(
        Format::LammpsDumpBinary,
        &write(Format::LammpsDumpBinary, &f, &full(PropSelect::All)),
    );
    same_atoms(&g, &f, "bin");
    assert_eq!(g.particles.ids, f.particles.ids);
    assert_eq!(images(&g), images(&f));
    for p in f.particles.props.iter() {
        same_prop(&g, p, true, "bin");
    }
}

#[test]
fn a_missing_or_reserved_property_name_is_refused_when_named() {
    let mut f = rich_frame(10, 17);
    let mut sink = Vec::new();
    for fmt in [Format::LammpsDump, Format::Xyz, Format::LammpsDumpBinary] {
        let err = Writer::write_with(
            &mut sink,
            fmt,
            std::slice::from_ref(&f),
            &full(PropSelect::Names(vec!["nope".into()])),
        )
        .unwrap_err();
        assert!(err.to_string().contains("nope"), "{err}");
    }
    f.particles
        .props
        .register(Property::from_data("id", 1, TypedVec::F64(vec![0.0; 10])));
    for fmt in [Format::LammpsDump, Format::Xyz] {
        assert!(Writer::write_with(
            &mut sink,
            fmt,
            std::slice::from_ref(&f),
            &full(PropSelect::Names(vec!["id".into()]))
        )
        .is_err());
        // under `all` it is skipped, and the atoms' own ids are written
        let g = read(fmt, &write(fmt, &f, &full(PropSelect::All)));
        assert_eq!(g.particles.ids, f.particles.ids, "{fmt:?}");
    }
}

#[test]
fn default_options_write_what_the_plain_writers_write() {
    let f = rich_frame(200, 18);
    for fmt in [
        Format::Xyz,
        Format::LammpsDump,
        Format::LammpsData,
        Format::LammpsDumpBinary,
    ] {
        let mut plain = Vec::new();
        Writer::write(&mut plain, fmt, std::slice::from_ref(&f)).unwrap();
        assert!(plain == write(fmt, &f, &WriteOptions::default()), "{fmt:?}");
    }
    // and the default spellings: 10 decimals, no properties
    let text = String::from_utf8(write(Format::LammpsDump, &f, &WriteOptions::default())).unwrap();
    assert!(text.contains("ITEM: ATOMS id type x y z ix iy iz\n"));
    assert!(text.contains("\n2400 1 -0.0000000000 0.0000000010 27.0000000000 -1 0 2\n"));
    let xyz = String::from_utf8(write(Format::Xyz, &f, &WriteOptions::default())).unwrap();
    assert!(xyz.contains("Properties=species:S:1:pos:R:3:type:I:1 Timestep=1234\n"));
}

fn gunzip(b: &[u8]) -> Vec<u8> {
    let mut out = Vec::new();
    flate2::read::MultiGzDecoder::new(b)
        .read_to_end(&mut out)
        .unwrap();
    out
}

#[test]
fn parallel_gzip_is_the_plain_bytes_and_independent_of_threads() {
    let mut rng = Rng(77);
    let data: Vec<u8> = (0..GZ_BLOCK * 3 + 12345)
        .map(|i| {
            if i % 9 == 0 {
                b'\n'
            } else {
                b'0' + rng.below(10) as u8
            }
        })
        .collect();
    let compress = |threads: usize| -> Vec<u8> {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .unwrap();
        pool.install(|| {
            let mut gz = GzWriter::new(Vec::new());
            // odd-sized writes, as the row formatter makes them
            for piece in data.chunks(1_000_003) {
                gz.write_all(piece).unwrap();
            }
            gz.finish().unwrap()
        })
    };
    let one = compress(1);
    assert!(gunzip(&one) == data);
    assert!(
        compress(4) == one,
        "member layout must not depend on threads"
    );
    assert!(compress(3) == one);
    // empty input is still a valid (empty) stream
    assert!(gunzip(&GzWriter::new(Vec::new()).finish().unwrap()).is_empty());
}

#[test]
fn gzipped_outputs_read_back_through_the_file_readers() {
    let dir = std::env::temp_dir().join(format!("rmda_rt_gz_{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let f = rich_frame(800, 19);
    for (name, fmt) in [
        ("o.dump.gz", Format::LammpsDump),
        ("o.xyz.gz", Format::Xyz),
        ("o.data.gz", Format::LammpsData),
    ] {
        let opts = if fmt == Format::LammpsData {
            full(PropSelect::Names(vec![
                "vx".into(),
                "vy".into(),
                "vz".into(),
            ]))
        } else {
            full(PropSelect::All)
        };
        let path = dir.join(name);
        let plain = write(fmt, &f, &opts);
        let mut gz = GzWriter::new(std::fs::File::create(&path).unwrap());
        gz.write_all(&plain).unwrap();
        gz.finish().unwrap();
        assert!(gunzip(&std::fs::read(&path).unwrap()) == plain, "{name}");
        let g = crate::io::read_frames(&path).unwrap();
        assert_eq!(g.len(), 1);
        same_atoms(&g[0], &f, name);
        same_prop(&g[0], f.particles.props.get("vx").unwrap(), false, name);
    }
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn a_binary_frame_too_large_for_one_chunk_is_split() {
    // The same code as for 2^31 values, with a small chunk: 1000 values is
    // 43 rows of 23 columns, so 1234 atoms take 29 chunks, the last short.
    let f = rich_frame(1234, 20);
    let opts = full(PropSelect::All);
    let mut b = Vec::new();
    crate::io::binary::write_dump_binary_chunked(&mut b, std::slice::from_ref(&f), &opts, 1000)
        .unwrap();
    let g = read(Format::LammpsDumpBinary, &b);
    same_atoms(&g, &f, "bin");
    assert_eq!(g.particles.ids, f.particles.ids);
    for p in f.particles.props.iter() {
        same_prop(&g, p, true, "bin chunks");
    }
    assert!(
        b.len() > write(Format::LammpsDumpBinary, &f, &opts).len(),
        "more chunk headers"
    );
}

/// A frame in a rotated (not LAMMPS-restricted) cell, as an extxyz `Lattice`
/// can give: every off-diagonal term of `h` live.
fn rotated_frame(n: usize, seed: u64) -> Frame {
    let mut rng = Rng(seed);
    let r = crate::lattice::fit::axis_angle([0.3, -0.5, 0.81], 0.6);
    let h0 = Mat3::from_rows([12.0, 0.0, 0.0], [2.5, 11.0, 0.0], [-1.5, 3.0, 13.0]);
    // Rotate each edge vector: rows of h0 · Rᵀ.
    let rows: Vec<[f64; 3]> = (0..3).map(|i| r.mul_v(*h0.row(i))).collect();
    let cell = UnitCell::from_matrix(
        Mat3::from_rows(rows[0], rows[1], rows[2]),
        [true, true, true],
        [-2.0, 1.0, 0.5],
    );
    assert!(!cell.is_restricted());
    let mut p = ParticleData::new(n);
    for i in 0..n {
        let lam = [rng.unit(), rng.unit(), rng.unit()];
        p.positions.push(cell.lamda2x(lam));
        p.ids.push(i as i64 + 1);
        p.types.push(1 + (i % 2) as u32);
    }
    for (k, name) in ["vx", "vy", "vz"].iter().enumerate() {
        let v: Vec<f64> = (0..n)
            .map(|i| (i as f64 + 1.0) * (k as f64 - 0.7))
            .collect();
        p.props
            .register(Property::from_data(*name, 1, TypedVec::F64(v)));
    }
    let mut f = Frame::new(cell, p);
    f.meta.timestep = 77;
    f
}

/// `ReadOptions::keep_general` (`--keep-general`) reads a general-triclinic
/// box as written: the same cell rows, origin, position bits and velocities
/// as the frame that was written, and writing it again gives the same bytes.
#[test]
fn keep_general_reads_the_general_form_as_written() {
    let f = rotated_frame(40, 12);
    let opts = WriteOptions {
        precision: Some(FloatFmt::Shortest),
        props: PropSelect::parse("vx,vy,vz"),
        extended: true,
        ..WriteOptions::default()
    };
    let keep = crate::io::ReadOptions::all().with_keep_general(true);
    for fmt in [
        Format::LammpsDump,
        Format::LammpsDumpBinary,
        Format::LammpsData,
    ] {
        let b = write(fmt, &f, &opts);
        let cur = Cursor::from_bytes(&b, "rt");
        let g = match fmt {
            Format::LammpsDump => crate::io::lammps::read_next_dump_with(&cur, &keep)
                .unwrap()
                .unwrap(),
            Format::LammpsDumpBinary => crate::io::binary::read_next_with(&cur, &keep)
                .unwrap()
                .unwrap(),
            _ => crate::io::lammps::parse_data_frame_with(&cur, &keep).unwrap(),
        };
        let ctx = format!("{fmt:?}");
        assert!(!g.cell.is_restricted(), "{ctx}");
        assert_eq!(g.cell.h_matrix().m, f.cell.h_matrix().m, "{ctx}");
        assert_eq!(g.cell.origin, f.cell.origin, "{ctx}");
        assert_eq!(g.cell.pbc, f.cell.pbc, "{ctx}");
        assert_eq!(g.particles.positions, f.particles.positions, "{ctx}");
        for n in ["vx", "vy", "vz"] {
            assert_eq!(
                g.particles.props.get(n).unwrap().as_f64(),
                f.particles.props.get(n).unwrap().as_f64(),
                "{ctx} {n}"
            );
        }
        assert!(write(fmt, &g, &opts) == b, "{ctx}: not byte-stable");
        // without the option the same file still reads restricted
        assert!(read(fmt, &b).cell.is_restricted(), "{ctx}");
    }
    // A restricted box is unaffected.
    let r = rich_frame(50, 3);
    let b = write(Format::LammpsData, &r, &opts);
    let cur = Cursor::from_bytes(&b, "rt");
    let g = crate::io::lammps::parse_data_frame_with(&cur, &keep).unwrap();
    assert_eq!(
        g.cell.h_matrix().m,
        read(Format::LammpsData, &b).cell.h_matrix().m
    );
}

/// A cell that is not restricted is written in LAMMPS' general triclinic form
/// (dump `BOX BOUNDS abc origin`, binary `triclinic 2`, data `avec bvec cvec
/// abc origin`), not from the lower triangle of `h` alone (a different box),
/// and reads back as the LAMMPS-restricted
/// rotation of the same cell: the same metric, the same fractional
/// coordinates, and velocities rotated with the atoms.
#[test]
fn a_general_cell_is_written_in_the_general_triclinic_form() {
    let f = rotated_frame(40, 11);
    let (rot, ..) = {
        let h = f.cell.h_matrix();
        crate::cell::general_to_restricted_rotation(*h.row(0), *h.row(1), *h.row(2)).unwrap()
    };
    let shortest = WriteOptions {
        precision: Some(FloatFmt::Shortest),
        props: PropSelect::parse("vx,vy,vz"),
        extended: true,
        ..WriteOptions::default()
    };
    for (fmt, opts) in [
        (Format::LammpsDump, WriteOptions::default()),
        (Format::LammpsDump, shortest.clone()),
        (Format::LammpsDumpBinary, shortest.clone()),
        (Format::LammpsData, WriteOptions::default()),
        (Format::LammpsData, shortest.clone()),
    ] {
        let b = write(fmt, &f, &opts);
        let text = String::from_utf8_lossy(&b);
        match fmt {
            Format::LammpsDump => assert!(text.contains("ITEM: BOX BOUNDS abc origin pp pp pp\n")),
            Format::LammpsData => {
                assert!(text.contains(" avec\n") && text.contains(" abc origin\n"));
                assert!(!text.contains("xlo xhi"));
            }
            _ => {}
        }
        let g = read(fmt, &b);
        let ctx = format!("{fmt:?} {:?}", opts.precision);
        assert!(g.cell.is_restricted(), "{ctx}");
        let tol = if opts.precision.is_some() {
            1e-12
        } else {
            1e-8
        };
        assert!(
            (g.cell.volume() - f.cell.volume()).abs() < tol * f.cell.volume(),
            "{ctx}"
        );
        let (l0, l1) = (f.cell.lengths(), g.cell.lengths());
        for d in 0..3 {
            assert!(
                (l0[d] - l1[d]).abs() < tol * 20.0,
                "{ctx}: lengths {l0:?} {l1:?}"
            );
        }
        for i in 0..f.natoms() {
            let a = f.cell.x2lamda(f.particles.positions[i]).unwrap();
            let b = g.cell.x2lamda(g.particles.positions[i]).unwrap();
            for d in 0..3 {
                assert!((a[d] - b[d]).abs() < tol, "{ctx}: atom {i} {a:?} {b:?}");
            }
        }
        if opts.props != PropSelect::None {
            let v = |fr: &Frame, i: usize| -> [f64; 3] {
                let c = |n: &str| fr.particles.props.get(n).unwrap().as_f64().unwrap()[i];
                [c("vx"), c("vy"), c("vz")]
            };
            for i in 0..f.natoms() {
                let want = rot.mul_v(v(&f, i));
                let got = v(&g, i);
                for d in 0..3 {
                    assert!((want[d] - got[d]).abs() < 1e-12 * 40.0, "{ctx}: v {i}");
                }
            }
        }
    }
    // A left-handed cell has no LAMMPS form.
    let mut l = f.clone();
    let h = *l.cell.h_matrix();
    l.cell = UnitCell::from_matrix(
        Mat3::from_rows(*h.row(1), *h.row(0), *h.row(2)),
        l.cell.pbc,
        l.cell.origin,
    );
    let mut b = Vec::new();
    assert!(Writer::write_with(
        &mut b,
        Format::LammpsDump,
        std::slice::from_ref(&l),
        &WriteOptions::default()
    )
    .is_err());
}

/// LAMMPS-style tilt reduction: the reduced cell is the same lattice
/// (`h' = M·h`, `det M = 1`), every tilt is within half a period, atoms are
/// inside the new cell, and unwrapped positions (`x + n·h`) are unchanged.
#[test]
fn tilt_reduction_keeps_the_lattice_and_the_unwrapped_atoms() {
    for &(xy, xz, yz, pbc) in &[
        (7.0, 0.0, 0.0, [true; 3]),
        (-13.0, 4.0, 9.5, [true; 3]),
        (2.0, 21.0, -17.0, [true; 3]),
        (6.0, 6.0, 8.0, [false, true, true]),
    ] {
        let mut cell = UnitCell::from_box_bounds(0.5, 10.5, xy, -1.0, 11.0, xz, 0.0, 9.0, yz);
        cell.pbc = pbc;
        let mut rng = Rng(5);
        let n = 60;
        let mut p = ParticleData::new(n);
        for i in 0..n {
            p.positions
                .push(cell.lamda2x([rng.unit(), rng.unit(), rng.unit()]));
            p.ids.push(i as i64 + 1);
            p.types.push(1);
        }
        p.images = Some(
            (0..n)
                .map(|i| pack_image(i as i64 % 3 - 1, 2 - (i as i64 % 5), (i as i64 % 2) * 3))
                .collect(),
        );
        let f0 = Frame::new(cell, p);
        let unwrapped = |f: &Frame| -> Vec<[f64; 3]> {
            (0..n)
                .map(|i| {
                    let im = f.particles.images.as_ref().unwrap()[i];
                    let s =
                        f.cell
                            .shift_vector([image_ipart(im), image_jpart(im), image_kpart(im)]);
                    let x = f.particles.positions[i];
                    [x[0] + s[0], x[1] + s[1], x[2] + s[2]]
                })
                .collect()
        };
        let mut f = f0.clone();
        let changed = f.reduce_tilts().unwrap();
        let m = f.cell.h_matrix().m;
        let ctx = format!("{xy} {xz} {yz} {pbc:?}");
        if pbc[0] {
            assert!(changed, "{ctx}");
            assert!(m[1][0].abs() <= 5.0 && m[2][0].abs() <= 5.0, "{ctx}: {m:?}");
        } else {
            // x not periodic: xy stays and xz is not flipped along a; yz = 8
            // > 12/2 is reduced, c - b, which moves xz by -xy.
            assert!(changed, "{ctx}");
            assert_eq!((m[1][0], m[2][0], m[2][1]), (6.0, 0.0, -4.0), "{ctx}");
        }
        if pbc[1] {
            assert!(m[2][1].abs() <= 6.0, "{ctx}: {m:?}");
        }
        assert!(f.cell.is_restricted());
        assert!((f.cell.volume() - f0.cell.volume()).abs() < 1e-9, "{ctx}");
        let (_, mm) = f0.cell.reduce_tilts().unwrap();
        let h0 = f0.cell.h_matrix().m;
        for r in 0..3 {
            for c in 0..3 {
                let want: f64 = (0..3).map(|k| mm[r][k] as f64 * h0[k][c]).sum();
                assert!((want - m[r][c]).abs() < 1e-9, "{ctx}: M·h row {r}");
            }
        }
        for (a, b) in unwrapped(&f0).iter().zip(unwrapped(&f)) {
            for d in 0..3 {
                assert!((a[d] - b[d]).abs() < 1e-9, "{ctx}: {a:?} {b:?}");
            }
        }
        for x in &f.particles.positions {
            let lam = f.cell.x2lamda(*x).unwrap();
            for d in 0..3 {
                if pbc[d] {
                    assert!((-1e-12..1.0).contains(&lam[d]), "{ctx}: {lam:?}");
                }
            }
        }
        // Reduced already: nothing more to do.
        assert!(!f.clone().reduce_tilts().unwrap());
    }
    // LAMMPS flips only beyond half a period (strict comparison).
    let edge = UnitCell::from_box_bounds(0.0, 10.0, 5.0, 0.0, 10.0, -5.0, 0.0, 10.0, 5.0);
    assert!(edge.reduce_tilts().is_none());
    // And a flipped LAMMPS cell: xy 0.6 lx -> -0.4 lx.
    let c = UnitCell::from_box_bounds(0.0, 10.0, 6.0, 0.0, 10.0, 0.0, 0.0, 10.0, 0.0);
    let (r, mm) = c.reduce_tilts().unwrap();
    assert_eq!(r.h_matrix().m[1][0], 6.0 - 10.0);
    assert_eq!(mm, [[1, 0, 0], [-1, 1, 0], [0, 0, 1]]);
}
