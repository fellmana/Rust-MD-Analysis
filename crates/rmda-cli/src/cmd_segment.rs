//! `rmda segment`: grain segmentation (design §15).

use rmda_core::analysis::defects::frame::{self, FrameObserver, PtmSettings};
use rmda_core::analysis::grain_seg::{segment, SegConfig, Segmentation};
use rmda_core::analysis::ptm::{NeighbourOrder, PtmAtom};
use rmda_core::data::frame::Frame;
use rmda_core::data::property::{DType, PropertyStore};
use rmda_core::error::Result;
use rmda_core::exec::Par;
use std::io::Write;
use std::path::{Path, PathBuf};

/// PTM with OVITO's default structures, one compact record per atom, then
/// with `dhcp` double HCP derived from its FCC and HCP labels (for
/// segmentation and defects; `rmda ptm` writes PTM's own labels).
pub fn run_ptm(
    f: &Frame,
    rmsd_cutoff: f64,
    dhcp: bool,
    order: NeighbourOrder,
    par: &Par,
) -> Result<Vec<PtmAtom>> {
    struct Report;
    impl FrameObserver for Report {
        fn dhcp(&mut self, n: usize) {
            eprintln!("dhcp: {n} close-packed atoms in ABAC stacking");
        }
    }
    frame::run_ptm(
        f,
        &PtmSettings {
            rmsd_cutoff,
            dhcp,
            order,
        },
        par,
        &mut Report,
    )
}

pub fn segment_frame(
    f: &Frame,
    ptm: &[PtmAtom],
    cfg: &SegConfig,
    par: &Par,
) -> Result<Segmentation> {
    let _t = rmda_core::prof::StageTimer::start("segment");
    segment(&f.cell, &f.particles.positions, ptm, cfg, par)
}

pub fn print_summary(s: &Segmentation) {
    let assigned = s.grain.iter().filter(|&&g| g >= 0).count();
    println!("segment_grains: {}", s.grains.len());
    println!("segment_crystallites: {}", s.n_crystallites);
    println!("segment_atoms_assigned: {assigned}");
    println!("segment_atoms_unassigned: {}", s.grain.len() - assigned);
    println!("segment_atoms_adopted: {}", s.adopted);
    if let Some(t) = s.merge_threshold {
        println!("segment_merge_threshold: {t}");
    }
    for g in s.grains.iter().take(20) {
        eprintln!(
            "grain {}: size={} structure={} crystallites={} q={:?}",
            g.id,
            g.size,
            g.structure,
            g.crystallites,
            g.orientation.map(|v| (v * 1e4).round() / 1e4)
        );
    }
    if s.grains.len() > 20 {
        eprintln!("... {} more grains (see --table)", s.grains.len() - 20);
    }
}

pub fn write_table(path: &Path, s: &Segmentation, timestep: i64) -> Result<()> {
    let mut w = std::io::BufWriter::new(std::fs::File::create(path)?);
    writeln!(w, "timestep,grain,size,structure,crystallites,qw,qx,qy,qz")?;
    for g in &s.grains {
        let q = g.orientation;
        writeln!(
            w,
            "{timestep},{},{},{},{},{:.8},{:.8},{:.8},{:.8}",
            g.id, g.size, g.structure, g.crystallites, q[0], q[1], q[2], q[3]
        )?;
    }
    let pairs = path.with_extension("pairs.csv");
    let mut w = std::io::BufWriter::new(std::fs::File::create(&pairs)?);
    writeln!(w, "timestep,grain_a,grain_b,bonds,disorientation_deg")?;
    for p in &s.pairs {
        writeln!(
            w,
            "{timestep},{},{},{},{:.4}",
            p.a, p.b, p.bonds, p.disorientation
        )?;
    }
    eprintln!(
        "segment: grain table {} and pair table {}",
        path.display(),
        pairs.display()
    );
    Ok(())
}

pub struct SegmentArgs {
    pub cfg: SegConfig,
    pub rmsd_cutoff: f64,
    pub table: Option<PathBuf>,
    pub dhcp: bool,
    pub ptm_order: NeighbourOrder,
}

/// Returns the per-atom `Grain` column for the caller to write.
pub fn run(f: &Frame, a: &SegmentArgs, par: &Par) -> Result<PropertyStore> {
    let ptm = run_ptm(f, a.rmsd_cutoff, a.dhcp, a.ptm_order, par)?;
    let s = segment_frame(f, &ptm, &a.cfg, par)?;
    print_summary(&s);
    if let Some(t) = &a.table {
        write_table(t, &s, f.meta.timestep)?;
    }
    let mut out = PropertyStore::new();
    out.register_scalar(DType::I32, "Grain", s.grain);
    out.register_scalar(
        DType::U8,
        "StructureType",
        ptm.iter().map(|p| p.structure).collect(),
    );
    Ok(out)
}

use crate::prelude::*;

pub(crate) fn command(ctx: &Ctx) -> Result<()> {
    let cli = ctx.cli;
    let par = ctx.par;
    let load_frame = || ctx.load_frame();
    let Cmd::Segment {
        mode,
        ptm_order: ptm_order_arg,
        threshold,
        core_angle,
        min_size,
        min_bonds,
        stacking_faults,
        no_adopt,
        rmsd_cutoff,
        table,
        dhcp,
        dhcp_merge_faults,
        ..
    } = &cli.cmd
    else {
        unreachable!("dispatched by command");
    };
    let frame = load_frame()?;
    let out = cmd_segment::run(
        &frame,
        &cmd_segment::SegmentArgs {
            cfg: rmda_core::analysis::grain_seg::SegConfig {
                mode: mode.to_mode(),
                threshold: *threshold,
                core_angle: *core_angle,
                min_size: *min_size,
                min_bonds: *min_bonds,
                stacking_faults: *stacking_faults == StackingFaults::Merge,
                adopt: !*no_adopt,
                dhcp_faults: *dhcp_merge_faults,
            },
            rmsd_cutoff: *rmsd_cutoff,
            table: table.clone(),
            dhcp: *dhcp,
            ptm_order: ptm_order(*ptm_order_arg, *mode == SegModeArg::Auto),
        },
        &par,
    )?;
    write_props(cli, &frame, &out)?;
    Ok(())
}
