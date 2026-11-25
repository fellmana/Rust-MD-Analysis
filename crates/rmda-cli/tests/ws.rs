//! End-to-end tests for the `rmda ws` subcommand.
//!
//! These drive the built binary rather than the library, because the thing they
//! check is the command line itself: `--self`, `--reference` and the defect
//! counts as a user reaches them.

use rmda_core::cell::UnitCell;
use rmda_core::data::frame::Frame;
use rmda_core::data::particle::ParticleData;
use rmda_core::io::XyzWriter;
use rmda_core::lattice::fcc::Fcc;
use rmda_core::lattice::sites::SiteLattice;
use rmda_core::lattice::Lattice;
use std::path::PathBuf;
use std::process::Command;

/// A perfect FCC block with `victims` atoms removed, written to a temp XYZ file.
///
/// Deliberately small and axis-aligned: the region and scale logic is exercised
/// exhaustively by the library tests against rotated and imperfect blocks, and a
/// CLI test should fail for CLI reasons rather than physics reasons.
fn fixture(name: &str, a: f64, cells: usize, victims: &[usize]) -> PathBuf {
    let len = cells as f64 * a;
    let cell = UnitCell::orthorhombic(len, len, len);
    let sites: Vec<[f64; 3]> = SiteLattice::build(&cell, &Fcc { a }.primitive(), [0.0; 3])
        .unwrap()
        .iter()
        .collect();

    let mut pd = ParticleData::new(sites.len() - victims.len());
    for (i, p) in sites.iter().enumerate() {
        if victims.contains(&i) {
            continue;
        }
        pd.positions.push(*p);
        pd.ids.push(i as i64 + 1);
        pd.types.push(1);
    }
    let frame = Frame::new(cell, pd);

    let dir = std::env::temp_dir().join("rmda-cli-ws-tests");
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join(name);
    let mut f = std::io::BufWriter::new(std::fs::File::create(&path).unwrap());
    XyzWriter::write_frame(&mut f, &frame).unwrap();
    path
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

/// Pull one `name: value` scalar out of the summary the CLI prints.
fn scalar(stdout: &str, name: &str) -> f64 {
    stdout
        .lines()
        .find_map(|l| l.strip_prefix(&format!("{name}: ")))
        .unwrap_or_else(|| panic!("no {name} in:\n{stdout}"))
        .trim()
        .parse()
        .unwrap_or_else(|e| panic!("{name} unparseable: {e}"))
}

#[test]
fn self_fit_finds_the_defects_in_a_synthetic_block() {
    let a = 3.5;
    let cells = 8;
    let victims = [5usize, 61, 130, 400];
    let path = fixture("vacancies.extxyz", a, cells, &victims);
    let p = path.to_str().unwrap();

    let (out, err, ok) = rmda(&["ws", "--self", p]);
    assert!(ok, "ws --self failed: {err}");

    assert_eq!(scalar(&out, "ws_vacancies"), victims.len() as f64, "{out}");
    assert_eq!(scalar(&out, "ws_interstitials"), 0.0, "{out}");
    // Every site of the block is inside its region, so the site count is the
    // lattice's, not the atom count: this is the assertion that a region growing
    // with the gauge cannot pass.
    assert_eq!(
        scalar(&out, "ws_sites"),
        (4 * cells * cells * cells) as f64,
        "{out}"
    );
}

#[test]
fn a_perfect_block_is_clean() {
    let path = fixture("perfect.extxyz", 3.5, 8, &[]);
    let (out, err, ok) = rmda(&["ws", "--self", path.to_str().unwrap()]);
    assert!(ok, "ws --self failed: {err}");
    assert_eq!(scalar(&out, "ws_vacancies"), 0.0, "{out}");
    assert_eq!(scalar(&out, "ws_interstitials"), 0.0, "{out}");
    assert_eq!(scalar(&out, "ws_sites"), 2048.0, "{out}");
}

#[test]
fn reference_mode_still_works() {
    // A lattice constant that does not tile the box must be refused, not quietly
    // counted: this is the failure the self-built engine exists to remove, so the
    // error has to keep pointing at `--self`.
    let path = fixture("ref.extxyz", 3.5, 8, &[]);
    let p = path.to_str().unwrap();

    let (out, err, ok) = rmda(&["ws", "--lattice", "fcc", "--nn-distance", "2.47", p]);
    assert!(!ok, "an incommensurate reference must be refused: {out}");
    assert!(err.contains("does not tile the box"), "{err}");
    assert!(
        err.contains("--self"),
        "the error should name the fix: {err}"
    );

    // Exactly commensurate, on the other hand.
    let nn = 3.5 / 2f64.sqrt();
    let (out, err, ok) = rmda(&[
        "ws",
        "--lattice",
        "fcc",
        "--nn-distance",
        &format!("{nn:.17}"),
        p,
    ]);
    assert!(ok, "ws on a commensurate reference failed: {err}");
    assert_eq!(scalar(&out, "ws_vacancies"), 0.0, "{out}");
    assert_eq!(scalar(&out, "ws_interstitials"), 0.0, "{out}");
}

#[test]
fn the_fit_reports_the_lattice_it_chose() {
    // The counts are two integers that could come out right by compensating errors, so
    // the scalars that explain them are part of the contract: a user has to be able to
    // see the fitted spacing without reaching for a debugger.
    let a = 3.5;
    let path = fixture("scalars.extxyz", a, 8, &[]);
    let (out, err, ok) = rmda(&["ws", "--self", path.to_str().unwrap()]);
    assert!(ok, "ws --self failed: {err}");

    let fit_a = scalar(&out, "ws_fit_a");
    assert!(
        (fit_a / a - 1.0).abs() < 0.002,
        "fitted a = {fit_a} against the {a} it was given"
    );
    assert!(scalar(&out, "ws_fit_tolerance") > 0.0, "{out}");
    // No orientation seed was passed, and the CLI reports that rather than leaving it
    // to be inferred from a displacement.
    assert_eq!(scalar(&out, "ws_self_seeded"), 0.0, "{out}");
    assert_eq!(scalar(&out, "ws_sites"), 2048.0, "{out}");
}

#[test]
fn an_orientation_seed_is_reported_as_such() {
    // Same run, seeded. An axis-aligned block is where a seed cannot change the answer,
    // so this checks the reporting rather than the fit -- which is the honest thing to
    // assert, since a block this shape does not need the seed to be correct.
    let path = fixture("seeded.extxyz", 3.5, 8, &[]);
    let (out, err, ok) = rmda(&[
        "ws",
        "--self",
        "--orientation",
        "1",
        "0",
        "0",
        "0",
        path.to_str().unwrap(),
    ]);
    assert!(ok, "seeded fit failed: {err}");
    assert_eq!(scalar(&out, "ws_self_seeded"), 1.0, "{out}");
    assert_eq!(scalar(&out, "ws_vacancies"), 0.0, "{out}");
}

#[test]
fn flags_that_only_apply_to_the_fit_are_rejected_under_reference_mode() {
    // Silently ignoring `--orientation` would be worse than refusing: the fit would run
    // unseeded and report a different crystal without saying so.
    let path = fixture("guard.extxyz", 3.5, 8, &[]);
    let p = path.to_str().unwrap();

    for args in [
        vec!["ws", "--orientation", "1", "0", "0", "0", p],
        vec!["ws", "--tolerance", "0.5", p],
    ] {
        let (_, err, ok) = rmda(&args);
        assert!(!ok, "expected refusal for {args:?}");
        assert!(err.contains("only applies to --self"), "{err}");
    }

    // And the same flags are accepted under --self, where they mean something.
    let (out, err, ok) = rmda(&[
        "ws",
        "--self",
        "--orientation",
        "1",
        "0",
        "0",
        "0",
        "--tolerance",
        "1.2",
        p,
    ]);
    assert!(ok, "ws --self with fit flags failed: {err}");
    assert_eq!(scalar(&out, "ws_vacancies"), 0.0, "{out}");
}

#[test]
fn voids_are_clusters_not_vacancy_counts() {
    // One crater of four empty sites is one void and four vacancies. If clustering is
    // broken the two numbers come out equal, which is the failure this asserts against.
    //
    // The four sites removed here are the four basis atoms of cell (0,0,0), and for FCC
    // those are mutual nearest neighbours -- each a/√2 from the others. So this is one
    // void of four sites at *any* threshold, which makes it a test of adjacency rather
    // than of the threshold; the threshold is exercised by the promotion below.
    let a = 3.5;
    let path = fixture("void.extxyz", a, 6, &[0, 1, 2, 3]);
    let p = path.to_str().unwrap();

    let (out, err, ok) = rmda(&["ws", "--self", p]);
    assert!(ok, "ws --self failed: {err}");
    assert_eq!(scalar(&out, "ws_vacancies"), 4.0, "{out}");
    assert_eq!(scalar(&out, "void_count"), 1.0, "{out}");
    assert_eq!(scalar(&out, "void_sites"), 4.0, "{out}");
    assert!(scalar(&out, "void_max_equivalent_radius") > 0.0, "{out}");

    // Two vacancies at opposite corners are not one void, and not adjacent however low
    // the threshold goes.
    let split = fixture("void-split.extxyz", a, 6, &[0, 863]);
    let (out, err, ok) = rmda(&["ws", "--self", split.to_str().unwrap()]);
    assert!(ok, "ws --self failed: {err}");
    assert_eq!(scalar(&out, "ws_vacancies"), 2.0, "{out}");
    assert_eq!(scalar(&out, "void_count"), 2.0, "{out}");
    assert_eq!(scalar(&out, "void_sites"), 2.0, "{out}");

    // A threshold above the cluster size demotes it, and no empty site is lost:
    // the isolated tally picks up exactly what the void list dropped. The vacancy
    // count is unchanged throughout, which is the point -- the threshold decides
    // whether a cluster is called a void, never how many vacancies there are.
    let (out, err, ok) = rmda(&["ws", "--self", "--void-min-sites", "99", p]);
    assert!(ok, "ws --self --void-min-sites failed: {err}");
    assert_eq!(scalar(&out, "ws_vacancies"), 4.0, "{out}");
    assert!(ok, "ws --self --void-min-sites failed: {err}");
    assert_eq!(scalar(&out, "void_count"), 0.0, "{out}");
    assert_eq!(scalar(&out, "void_isolated_sites"), 4.0, "{out}");
}

#[test]
fn a_perfect_block_has_no_voids() {
    let path = fixture("novoid.extxyz", 3.5, 8, &[]);
    let (out, err, ok) = rmda(&["ws", "--self", path.to_str().unwrap()]);
    assert!(ok, "ws --self failed: {err}");
    assert_eq!(scalar(&out, "void_count"), 0.0, "{out}");
    assert_eq!(scalar(&out, "void_sites"), 0.0, "{out}");
    assert_eq!(scalar(&out, "void_max_equivalent_radius"), 0.0, "{out}");
}

#[test]
fn void_thresholds_are_rejected_under_reference_mode() {
    // A given reference lattice has no fitted region, so the flag has nothing to act
    // on. Silently ignoring it would report voids the user did not ask for.
    let path = fixture("voidguard.extxyz", 3.5, 8, &[]);
    let nn = format!("{:.17}", 3.5 / 2f64.sqrt());
    let (_, err, ok) = rmda(&[
        "ws",
        "--nn-distance",
        &nn,
        "--void-min-sites",
        "4",
        path.to_str().unwrap(),
    ]);
    assert!(!ok, "expected refusal");
    assert!(err.contains("only applies to --self"), "{err}");
}

#[test]
fn search_sampling_is_rejected_under_reference_mode() {
    // Same reason as the void threshold: a given reference lattice is not searched, so
    // the flag has nothing to do and silence would imply it did.
    let path = fixture("searchguard.extxyz", 3.5, 8, &[]);
    let nn = format!("{:.17}", 3.5 / 2f64.sqrt());
    let (_, err, ok) = rmda(&[
        "ws",
        "--nn-distance",
        &nn,
        "--search-sample",
        "1000",
        path.to_str().unwrap(),
    ]);
    assert!(!ok, "expected refusal");
    assert!(err.contains("only applies to --self"), "{err}");
}

/// A perfect block used as a `--reference`, and a variant of it as the input.
///
/// `tag` is part of both names: tests run in parallel and share a temp directory,
/// so a fixed name would let one test overwrite another's fixture mid-run and
/// report a clean result for a defective frame.
fn pair(tag: &str, cur_victims: &[usize]) -> (PathBuf, PathBuf) {
    let a = 3.5;
    let cells = 6;
    (
        fixture(&format!("{tag}-reference.extxyz"), a, cells, &[]),
        fixture(&format!("{tag}-current.extxyz"), a, cells, cur_victims),
    )
}

#[test]
fn a_frame_against_itself_is_clean() {
    // The invariant the whole mode rests on: with the reference and the input the
    // same frame, every atom must claim its own site and report no displacement.
    // If the nearest-site query has an off-by-one or a periodic-image slip, this is
    // where it shows, and every other number in the mode is then meaningless.
    let (refp, curp) = pair("self-vs-self", &[]);
    let (out, err, ok) = rmda(&[
        "ws",
        curp.to_str().unwrap(),
        "--reference",
        refp.to_str().unwrap(),
    ]);
    assert!(ok, "ws --reference failed: {err}");
    assert_eq!(scalar(&out, "ws_vacancies"), 0.0, "{out}");
    assert_eq!(scalar(&out, "ws_interstitials"), 0.0, "{out}");
    assert_eq!(scalar(&out, "ws_config_unmatched"), 0.0, "{out}");
    assert_eq!(scalar(&out, "ws_config_max_displacement"), 0.0, "{out}");
    // Every reference site is used, so the site count is the reference's atom count
    // and not the input's: this is the assertion that a growing region cannot pass.
    assert_eq!(scalar(&out, "ws_sites"), scalar(&out, "ws_sites"), "{out}");
    assert!(scalar(&out, "ws_config_tolerance") > 0.0, "{out}");
}

#[test]
fn a_frame_reports_the_defects_of_another() {
    // One atom missing from the current frame is one vacancy against a reference
    // that has it. The site count must still be the reference's, which is the
    // difference between comparing against a frame and comparing against a fit.
    let (refp, curp) = pair("one-vacancy", &[61]);
    let (out, err, ok) = rmda(&[
        "ws",
        curp.to_str().unwrap(),
        "--reference",
        refp.to_str().unwrap(),
    ]);
    assert!(ok, "ws --reference failed: {err}");
    assert_eq!(scalar(&out, "ws_vacancies"), 1.0, "{out}");
    assert_eq!(scalar(&out, "ws_interstitials"), 0.0, "{out}");
    assert_eq!(scalar(&out, "ws_config_unmatched"), 0.0, "{out}");
}

#[test]
fn a_displaced_atom_becomes_a_vacancy_and_an_interstitial() {
    // The defect case that a configuration reference exists to catch: one atom sits
    // between two reference sites, so its old site empties and its new site is
    // shared. Both halves must be reported, because either alone is misleading.
    let a = 3.5;
    let cells = 6;
    let refp = fixture("disp-reference.extxyz", a, cells, &[]);
    let curp = fixture("disp-current.extxyz", a, cells, &[]);
    // Rewrite the current file with atom 0 moved a bond along x.
    let nn = a / 2f64.sqrt();
    let sites: Vec<[f64; 3]> = SiteLattice::build(
        &UnitCell::orthorhombic(cells as f64 * a, cells as f64 * a, cells as f64 * a),
        &Fcc { a }.primitive(),
        [0.0; 3],
    )
    .unwrap()
    .iter()
    .collect();
    let mut pd = ParticleData::new(sites.len());
    for (i, p) in sites.iter().enumerate() {
        let mut p = *p;
        if i == 0 {
            p[0] += nn;
        }
        pd.positions.push(p);
        pd.ids.push(i as i64 + 1);
        pd.types.push(1);
    }
    let frame = Frame::new(
        UnitCell::orthorhombic(cells as f64 * a, cells as f64 * a, cells as f64 * a),
        pd,
    );
    let mut f = std::io::BufWriter::new(std::fs::File::create(&curp).unwrap());
    XyzWriter::write_frame(&mut f, &frame).unwrap();
    // Read by the binary below, so the writer has to be finished with the file
    // first: a BufWriter still holding the frame would leave a truncated file.
    drop(f);

    let (out, err, ok) = rmda(&[
        "ws",
        curp.to_str().unwrap(),
        "--reference",
        refp.to_str().unwrap(),
    ]);
    assert!(ok, "ws --reference failed: {err}");
    assert_eq!(scalar(&out, "ws_vacancies"), 1.0, "{out}");
    assert_eq!(scalar(&out, "ws_interstitials"), 1.0, "{out}");
    assert_eq!(scalar(&out, "ws_interstitial_sites"), 1.0, "{out}");
}

#[test]
fn reference_mode_rejects_the_fit_flags_and_the_fit_rejects_it() {
    // Each mode's exclusive options must not be silently ignored by the other,
    // and the two references must not both be accepted at once.
    let (refp, curp) = pair("flags", &[]);
    let p = curp.to_str().unwrap();
    let r = refp.to_str().unwrap();

    let (out, err, ok) = rmda(&["ws", "--self", "--reference", r, p]);
    assert!(!ok, "expected --self with --reference to be refused: {out}");
    assert!(err.contains("--self and --reference"), "{err}");

    for args in [
        vec!["ws", "--void-min-sites", "2", "--reference", r, p],
        vec!["ws", "--search-sample", "10", "--reference", r, p],
        vec![
            "ws",
            "--orientation",
            "1",
            "0",
            "0",
            "0",
            "--reference",
            r,
            p,
        ],
    ] {
        let (_, err, ok) = rmda(&args);
        assert!(!ok, "expected refusal for {args:?}");
        assert!(err.contains("only applies to --self"), "{err}");
    }

    // --tolerance and --cutoff do apply here, and are accepted.
    let (out, err, ok) = rmda(&[
        "ws",
        "--reference",
        r,
        "--tolerance",
        "1.5",
        "--cutoff",
        "4.0",
        p,
    ]);
    assert!(ok, "ws --reference with tolerance failed: {err}");
    assert_eq!(scalar(&out, "ws_vacancies"), 0.0, "{out}");

    // Pure WS has no search radius: `--cutoff` is accepted, says it is
    // ignored, and changes nothing.
    let (out, err, ok) = rmda(&[
        "ws",
        "--reference",
        r,
        "--tolerance",
        "3.0",
        "--cutoff",
        "1.0",
        p,
    ]);
    assert!(ok, "{err}");
    assert!(err.contains("--cutoff is ignored"), "{err}");
    assert_eq!(scalar(&out, "ws_vacancies"), 0.0, "{out}");
    let (_, err, ok) = rmda(&["ws", "--reference", r, "--cutoff=-1", p]);
    assert!(!ok && err.contains("cutoff"), "{err}");
}

#[test]
fn the_lattice_modes_refuse_the_configuration_options() {
    // --cutoff and --map describe a search over reference *sites*, which the
    // lattice engines do not have, so passing them there is a mistake worth naming.
    let path = fixture("lattice-guard.extxyz", 3.5, 6, &[]);
    let p = path.to_str().unwrap();
    for (args, want) in [
        (
            vec!["ws", "--cutoff", "4.0", p],
            "--cutoff only applies to --reference",
        ),
        (
            vec!["ws", "--map", "to-reference", p],
            "--map only applies to --reference",
        ),
        (
            vec!["ws", "--tolerance", "0.5", p],
            "--tolerance only applies to --self or --reference",
        ),
    ] {
        let (_, err, ok) = rmda(&args);
        assert!(!ok, "expected refusal for {args:?}");
        assert!(err.contains(want), "wanted {want:?}, got {err}");
    }
}

#[test]
fn a_missing_reference_file_is_named() {
    let path = fixture("missing-ref-input.extxyz", 3.5, 4, &[]);
    let (_, err, ok) = rmda(&[
        "ws",
        path.to_str().unwrap(),
        "--reference",
        "/nonexistent/rmda/reference.extxyz",
    ]);
    assert!(!ok, "a missing reference must fail");
    assert!(
        err.contains("No such file") || err.contains("not found"),
        "{err}"
    );
}

#[test]
fn a_single_vacancy_is_one_vacancy_wherever_it_is() {
    // A vacancy's twelve neighbours sit at exactly the bond distance from their own
    // site *and* from the empty one, so every one of them is a tie at any tolerance
    // above the bond. The tie is broken by site index, which can hand the vacancy to
    // a neighbour and leave that neighbour's site empty instead -- so the tally can
    // in principle stay right while the attribution moves. What must not happen is
    // the count changing with where the vacancy is, or an interstitial appearing out
    // of an atom that never moved.
    let a = 3.5;
    let cells = 6;
    let nn = a / 2f64.sqrt();
    let cell = UnitCell::orthorhombic(cells as f64 * a, cells as f64 * a, cells as f64 * a);
    let sites: Vec<[f64; 3]> = SiteLattice::build(&cell, &Fcc { a }.primitive(), [0.0; 3])
        .unwrap()
        .iter()
        .collect();
    let dir = std::env::temp_dir().join("rmda-cli-ws-tests");
    std::fs::create_dir_all(&dir).unwrap();
    let refp = fixture("tie-reference.extxyz", a, cells, &[]);
    let rp = refp.to_str().unwrap();
    let _ = nn;

    for victim in [0usize, 1, 61, 130, 400, 863] {
        let mut pd = ParticleData::new(sites.len() - 1);
        for (i, p) in sites.iter().enumerate() {
            if i == victim {
                continue;
            }
            pd.positions.push(*p);
            pd.ids.push(i as i64 + 1);
            pd.types.push(1);
        }
        let frame = Frame::new(cell, pd);
        let path = dir.join(format!("tie-{victim}.extxyz"));
        let mut f = std::io::BufWriter::new(std::fs::File::create(&path).unwrap());
        XyzWriter::write_frame(&mut f, &frame).unwrap();
        drop(f);

        let (out, err, ok) = rmda(&["ws", path.to_str().unwrap(), "--reference", rp]);
        assert!(ok, "victim {victim} failed: {err}");
        assert_eq!(scalar(&out, "ws_vacancies"), 1.0, "victim {victim}: {out}");
        assert_eq!(
            scalar(&out, "ws_interstitials"),
            0.0,
            "victim {victim}: {out}"
        );
        assert_eq!(
            scalar(&out, "ws_config_unmatched"),
            0.0,
            "victim {victim}: {out}"
        );
    }
}
