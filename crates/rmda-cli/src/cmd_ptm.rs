//! `rmda ptm`.

use crate::prelude::*;

pub(crate) fn command(ctx: &Ctx) -> Result<()> {
    let cli = ctx.cli;
    let par = ctx.par;
    let load_frame = || ctx.load_frame();
    let Cmd::Ptm {
        rmsd_cutoff,
        types,
        ptm_order: order,
        outputs,
        ..
    } = &cli.cmd
    else {
        unreachable!("dispatched by command");
    };
    let frame = load_frame()?;
    let f = &frame;
    let mut flags = 0u32;
    for t in types {
        flags |= t.to_flag();
    }
    if flags == 0 {
        eprintln!("error: --types selected no structures");
        std::process::exit(2);
    }
    let ptm = Ptm {
        rmsd_cutoff: *rmsd_cutoff,
        flags,
    };
    // PTM builds its own k-NN index; nothing else here needs a list.
    let nk = NeighborhoodContext { cell: None };
    let ctx = AnalysisContext {
        frame: f,
        nkeys: &nk,
        par: &par,
    };
    let mut out = PropertyStore::new();
    let mut summary = Summary::default();
    let outputs = outputs.iter().fold(0u32, |a, b| a | b);
    ptm.run_extended(
        &ctx,
        &mut out,
        &mut summary,
        ptm_order(*order, false),
        outputs,
    )?;

    eprintln!(
        "ptm: rmsd_cutoff={rmsd_cutoff} types={}",
        types.iter().map(|t| t.name()).collect::<Vec<_>>().join(",")
    );
    // Report every structure in a fixed order, including those that came
    // out empty, so the output is diffable between runs.
    for label in [
        "other", "fcc", "hcp", "bcc", "ico", "sc", "dcub", "dhex", "graphene",
    ] {
        let key = format!("ptm_{label}");
        if let Some((_, v)) = summary.scalars.iter().find(|(k, _)| *k == key) {
            println!("{key}: {v}");
        }
    }
    // Ordering counts, only when `--outputs ordering` asked for them.
    for (k, v) in &summary.scalars {
        if k.starts_with("ptm_ordering_") {
            println!("{k}: {v}");
        }
    }
    if let (Some(types), Some(rmsd)) = (out.get("StructureType"), out.get("StructureRmsd")) {
        let types = types.as_u8().expect("StructureType is u8");
        let rmsd = rmsd.as_f64().expect("StructureRmsd is f64");
        let n = types.len();
        let mut n_matched = 0usize;
        let mut sum = 0.0;
        let mut worst = 0.0f64;
        for (&t, &r) in types.iter().zip(rmsd.iter()) {
            if t != structure::OTHER as u8 && r.is_finite() {
                n_matched += 1;
                sum += r;
                worst = worst.max(r);
            }
        }
        println!(
            "ptm_matched: {n_matched}/{n} ({:.2}%) mean_rmsd={:.6} max_rmsd={:.6}",
            100.0 * n_matched as f64 / n as f64,
            if n_matched == 0 {
                0.0
            } else {
                sum / n_matched as f64
            },
            worst
        );

        // The recovered lengths are per-structure, so report the mean
        // over matched atoms only. A hot atom is a sanity check on the
        // whole normalise -> fit -> derive chain: a BCC lattice at
        // `a` must come back as `a` (and a*sqrt(3)/2 nearest-neighbour
        // distance), an FCC one as `a/sqrt(2)`, and so on.
        for (key, label) in [
            ("StructureLatticeConstant", "lattice_constant"),
            ("StructureInteratomicDistance", "interatomic_distance"),
        ] {
            let Some(vals) = out.get(key).and_then(|p| p.as_f64()) else {
                continue;
            };
            let mut sum = 0.0;
            let mut n_v = 0usize;
            let mut lo = f64::INFINITY;
            let mut hi = f64::NEG_INFINITY;
            for (&t, &v) in types.iter().zip(vals.iter()) {
                if t != structure::OTHER as u8 && v > 0.0 {
                    sum += v;
                    n_v += 1;
                    lo = lo.min(v);
                    hi = hi.max(v);
                }
            }
            if n_v > 0 {
                println!(
                    "ptm_{label}: mean={:.6} min={:.6} max={:.6} (n={n_v})",
                    sum / n_v as f64,
                    lo,
                    hi
                );
            }
        }
    }
    write_props(cli, f, &out)?;
    Ok(())
}
