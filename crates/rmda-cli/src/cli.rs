//! The `rmda` command line: global options, subcommands and their argument
//! types (clap). Defaults and checks live here, once; `rmda run` configs
//! resolve through the same definitions.

use crate::prelude::*;

/// CLI spelling of [`rmda_core::io::XyzOrigin`].
#[derive(ValueEnum, Clone, Copy, Debug, PartialEq, Eq, Default)]
pub(crate) enum XyzOriginArg {
    /// Reproduce the input box's own origin.
    #[default]
    Native,
    /// Shift the box to start at `(0, 0, 0)`, translating atoms with it.
    Zero,
}

impl From<XyzOriginArg> for rmda_core::io::XyzOrigin {
    fn from(a: XyzOriginArg) -> Self {
        match a {
            XyzOriginArg::Native => rmda_core::io::XyzOrigin::Native,
            XyzOriginArg::Zero => rmda_core::io::XyzOrigin::Zero,
        }
    }
}

#[derive(Parser)]
#[command(
    name = "rmda",
    version,
    about = "RMDA (Rust MD Analysis): analysis of atomistic molecular-dynamics frames"
)]
pub(crate) struct Cli {
    /// Number of threads (0 = the job's CPUs under SLURM, `SLURM_CPUS_PER_TASK`,
    /// else all cores).
    #[arg(short, long, global = true, default_value_t = 0)]
    pub(crate) threads: usize,

    /// Input file (alternative to the positional FILE argument).
    #[arg(short, long, global = true)]
    pub(crate) input: Option<PathBuf>,

    /// Output file.
    #[arg(short, long, global = true)]
    pub(crate) output: Option<PathBuf>,

    /// Print per-stage timings and peak RSS to stderr on exit (design §23).
    #[arg(long, global = true)]
    pub(crate) profile: bool,

    /// Print the peak-RSS summary only, without stage timings.
    #[arg(long, global = true)]
    pub(crate) memreport: bool,

    /// Where to put the origin of cells written to `.xyz`.
    ///
    /// `native` (the default) reproduces the input box, which is what makes an
    /// overlay line up. `zero` shifts the box to start at the origin and moves
    /// the atom coordinates with it, for tools that assume a box begins at zero.
    #[arg(long, global = true, value_enum, default_value_t = XyzOriginArg::Native)]
    pub(crate) xyz_origin: XyzOriginArg,

    /// Input columns to skip, comma-separated (`vx,vy,vz,ix,iy,iz,c_pe`).
    ///
    /// A skipped column is stepped over and never stored. By default the
    /// analysis commands skip velocities and image flags, which no analysis
    /// reads (`vx,vy,vz,ix,iy,iz`), and `info`/`convert` keep everything.
    /// An empty list (`--skip-columns ''`) keeps every column everywhere.
    #[arg(long, global = true, value_name = "LIST")]
    pub(crate) skip_columns: Option<String>,

    /// LAMMPS atom style of an input data file whose `Atoms` line has no
    /// `# style` comment (atomic, charge, full, molecular, bond, angle,
    /// sphere). Needed for 6- and 9-column rows, which are ambiguous.
    #[arg(long, global = true, value_name = "STYLE")]
    pub(crate) atom_style: Option<String>,

    /// Keep a general-triclinic input box (data `avec bvec cvec`, dump
    /// `abc origin`, binary) as written, with positions and velocities in
    /// the file's frame, instead of rotating it into LAMMPS' restricted
    /// frame; `convert` then writes the general form again.
    #[arg(long, global = true)]
    pub(crate) keep_general: bool,

    /// Process only this frame index (0-based).
    #[arg(long, global = true)]
    pub(crate) frame: Option<usize>,

    /// Start frame index (0-based).
    #[arg(long, global = true)]
    pub(crate) start: Option<usize>,

    /// End frame index (0-based, exclusive).
    #[arg(long, global = true)]
    pub(crate) end: Option<usize>,

    /// Frame stride.
    #[arg(long, global = true, default_value_t = 1)]
    pub(crate) stride: usize,

    /// State the box explicitly: `--cell A B C`, in ångström.
    ///
    /// A file with a `Lattice=` line carries its own box, and a file without
    /// one is read as the isolated system it says it is — in the box its own
    /// atoms occupy. This is how a periodic box is stated for a file that has
    /// none: the edges are taken as given, the origin is the one the file
    /// already had, so stating a box does not move the atoms, and the box is
    /// periodic unless `--pbc false` says otherwise.
    #[arg(long, global = true, value_names = ["A", "B", "C"], num_args = 3)]
    pub(crate) cell: Option<Vec<f64>>,

    /// Whether the box is periodic: `--pbc` or `--pbc false`.
    ///
    /// `--cell` states a periodic box unless this says otherwise. On its own
    /// it overrides the periodicity a file carries, which is how a file with a
    /// box is read as the free system it may also be.
    #[arg(
        long,
        global = true,
        value_names = ["BOOL"],
        num_args = 0..=1,
        default_missing_value = "true"
    )]
    pub(crate) pbc: Option<bool>,

    /// Stop the run, with exit code 4, when the memory it allocated (anonymous
    /// RSS; not the mapped input, which the kernel can drop) exceeds SIZE:
    /// bytes, or with a K, M, G or T suffix (`--mem-limit 200G`). Checked every
    /// 100 ms; a series keeps the frames it finished and resumes. Set it a
    /// little under the job's memory, so the run reports which stage needed
    /// more instead of being killed without a word.
    #[arg(long, global = true, value_name = "SIZE", value_parser = parse_size)]
    pub(crate) mem_limit: Option<u64>,

    #[command(subcommand)]
    pub(crate) cmd: Cmd,
}

/// `--mem-limit`: bytes, or a number with a K, M, G or T suffix (powers of
/// 1024).
pub(crate) fn parse_size(s: &str) -> std::result::Result<u64, String> {
    let t = s.trim();
    let (num, mult) = match t.char_indices().last() {
        Some((i, c)) if c.is_ascii_alphabetic() => {
            let m: u64 = match c.to_ascii_uppercase() {
                'K' => 1 << 10,
                'M' => 1 << 20,
                'G' => 1 << 30,
                'T' => 1 << 40,
                _ => return Err(format!("`{s}`: the suffix must be K, M, G or T")),
            };
            (&t[..i], m)
        }
        _ => (t, 1),
    };
    let v: f64 = num
        .trim()
        .parse()
        .map_err(|_| format!("`{s}`: expected a size such as 64G or 500M"))?;
    if !(v.is_finite() && v > 0.0) {
        return Err(format!("`{s}`: the size must be positive"));
    }
    Ok((v * mult as f64) as u64)
}

/// `rmda defects --defects-layout`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum)]
pub(crate) enum DefectsLayoutArg {
    /// defects/<index>/vacancies.xyz: a directory per frame (the default).
    Dirs,
    /// defects/vacancies-<index>.xyz: every frame's files in one directory.
    Files,
    /// defects/vacancies.xyz: every frame appended in turn, its comment line
    /// carrying index=<index> (OVITO reads it as a trajectory).
    Append,
}

/// `rmda defects --reference`.
#[derive(Clone, Copy, Debug, ValueEnum)]
pub(crate) enum ReferenceArg {
    Auto,
    Global,
    Patch,
    Local,
    /// A periodic single crystal: Wigner-Seitz against its one lattice
    /// (commensurate with the box), ambiguous atoms resolved, past the
    /// displacement gates; counts defects stored as whole planes added or
    /// removed, which the local census cannot see. Other grains: as auto.
    Periodic,
}

impl ReferenceArg {
    pub(crate) fn to_mode(self) -> rmda_core::analysis::defects::ReferenceMode {
        use rmda_core::analysis::defects::ReferenceMode;
        match self {
            ReferenceArg::Auto => ReferenceMode::Auto,
            ReferenceArg::Global => ReferenceMode::Global,
            ReferenceArg::Patch => ReferenceMode::Patch,
            ReferenceArg::Local => ReferenceMode::Local,
            ReferenceArg::Periodic => ReferenceMode::Periodic,
        }
    }
}

/// `rmda defects --local-merge`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum)]
pub(crate) enum LocalMergeArg {
    /// Try the union with the regions a cluster overlaps, else count it as a
    /// partial region (most accurate; serial).
    Union,
    /// Count an overlapping cluster as a partial region at once (parallel,
    /// much faster; less exact where loops meet).
    Partial,
}

impl LocalMergeArg {
    pub(crate) fn to_mode(self) -> rmda_core::analysis::defects::LocalMerge {
        match self {
            LocalMergeArg::Union => rmda_core::analysis::defects::LocalMerge::Union,
            LocalMergeArg::Partial => rmda_core::analysis::defects::LocalMerge::Partial,
        }
    }
}

/// `rmda defects --ws-affine` (OVITO's `affine_mapping`).
#[derive(Clone, Copy, Debug, ValueEnum)]
pub(crate) enum WsAffineArg {
    /// Positions as they are (OVITO's default).
    Off,
    /// Map current positions into the reference cell (OVITO's `ToReference`).
    ToReference,
    /// Map the reference sites into the current cell (OVITO refuses this one).
    ToCurrent,
}

impl WsAffineArg {
    pub(crate) fn to_mapping(self) -> rmda_core::analysis::defects::explicit::AffineMapping {
        use rmda_core::analysis::defects::explicit::AffineMapping;
        match self {
            WsAffineArg::Off => AffineMapping::Off,
            WsAffineArg::ToReference => AffineMapping::ToReference,
            WsAffineArg::ToCurrent => AffineMapping::ToCurrent,
        }
    }
}

/// `--stacking-faults` of `rmda segment` and `rmda defects --segment`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum)]
pub(crate) enum StackingFaults {
    /// HCP fault and twin planes join an FCC grain (and FCC layers an HCP
    /// grain), as OVITO's grain segmentation does.
    Merge,
    /// FCC and HCP never share a grain: a fault that crosses a grain splits it,
    /// so each side can be counted against its own lattice.
    Separate,
}

/// `--mode` of `rmda segment`, `--seg-mode` of `rmda defects --segment`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum)]
pub(crate) enum SegModeArg {
    /// Crystallites of coherent orientation merged on their mean orientations
    /// below `--threshold`.
    Crystallite,
    /// OVITO 3.15's GrainSegmentation (GraphClusteringAuto): node pair
    /// sampling over PTM-neighbour bonds with the merge threshold chosen
    /// automatically, and OVITO's orphan adoption. Ignores `--threshold`,
    /// `--core-angle` and `--min-bonds`. Serial and about 50 B/bond more
    /// memory than `crystallite`.
    Auto,
}

/// One `rmda ptm --outputs` name ([`rmda_core::analysis::ptm::output`]).
pub(crate) fn parse_ptm_output(s: &str) -> std::result::Result<u32, String> {
    rmda_core::analysis::ptm::output::from_name(s)
        .ok_or_else(|| format!("unknown PTM output '{s}' (expected ordering, F, template, correspondences, residual, polar or all)"))
}

/// `--ptm-order` of `rmda segment` and `rmda defects`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum)]
pub(crate) enum PtmOrderArg {
    /// PTM matches the nearest neighbours.
    Distance,
    /// PTM matches neighbours in decreasing Voronoi solid angle, as OVITO
    /// 3.15 does (atom-for-atom OVITO structure types on the reference frames).
    Voronoi,
}

/// The PTM order: as given, else Voronoi for `auto` segmentation (OVITO
/// parity) and distance otherwise.
pub(crate) fn ptm_order(
    arg: Option<PtmOrderArg>,
    auto: bool,
) -> rmda_core::analysis::ptm::NeighbourOrder {
    use rmda_core::analysis::ptm::NeighbourOrder;
    match arg {
        Some(PtmOrderArg::Distance) => NeighbourOrder::Distance,
        Some(PtmOrderArg::Voronoi) => NeighbourOrder::Voronoi,
        None if auto => NeighbourOrder::Voronoi,
        None => NeighbourOrder::Distance,
    }
}

impl SegModeArg {
    pub(crate) fn to_mode(self) -> rmda_core::analysis::grain_seg::SegMode {
        match self {
            SegModeArg::Crystallite => rmda_core::analysis::grain_seg::SegMode::Crystallite,
            SegModeArg::Auto => rmda_core::analysis::grain_seg::SegMode::Auto,
        }
    }
}

#[derive(Subcommand)]
pub(crate) enum Cmd {
    /// Run a whole analysis from a TOML config: `[input] files`, `[run]` global
    /// options, and one command section (e.g. `[defects]`) whose keys are that
    /// command's long options (`rmsd_cutoff = 0.15` is `--rmsd-cutoff 0.15`).
    /// Writes report.json (version, resolved command, config, host, threads,
    /// timings, stage profile, per-frame summary) into the command's `out`
    /// directory, or beside the config. See documentation.md, `rmda run`.
    Run {
        /// The TOML config.
        #[arg(value_name = "CONFIG")]
        config: PathBuf,
        /// Override a config value: `--set defects.reference=local` (repeatable;
        /// the value is read as TOML, else as a string).
        #[arg(long = "set", value_name = "SECTION.KEY=VALUE")]
        sets: Vec<String>,
        /// Print the resolved `rmda ...` command line and exit without running.
        #[arg(long)]
        dry_run: bool,
        /// Run shard I of N of the config's series (as the command's --shard),
        /// e.g. `--shard $SLURM_ARRAY_TASK_ID/$SLURM_ARRAY_TASK_COUNT`; then
        /// `rmda merge <out>`.
        #[arg(long, value_name = "I/N", value_parser = parse_shard)]
        shard: Option<(usize, usize)>,
    },
    /// What `rmda run CONFIG` would do, without running it: the resolved
    /// command, the input files with their atom counts (from each file's
    /// head), the frames a --shard takes, and a memory estimate per frame from
    /// measured runs, for sizing a job.
    Explain {
        /// The TOML config, as for `rmda run`.
        #[arg(value_name = "CONFIG")]
        config: PathBuf,
        /// Override a config value, as for `rmda run`.
        #[arg(long = "set", value_name = "SECTION.KEY=VALUE")]
        sets: Vec<String>,
    },
    /// Per-stage throughput on a synthetic crystal built in memory (PTM, CNA,
    /// segmentation, the defect census with the global and local references,
    /// RDF): wall time, atoms per second and peak memory per stage, to compare
    /// machines, thread counts and builds. Fails when a census misses the
    /// planted vacancies.
    Bench {
        /// Atoms in the crystal (rounded to whole cubic cells).
        #[arg(long, default_value_t = 1_000_000)]
        atoms: usize,
        /// `bcc` (W, a = 3.165 Å) or `fcc` (Cu, a = 3.615 Å).
        #[arg(long, default_value = "bcc")]
        structure: String,
        /// Gaussian thermal displacement per component, Å.
        #[arg(long, default_value_t = 0.05)]
        sigma: f64,
        /// Also write the results as JSON here.
        #[arg(long, value_name = "FILE")]
        json: Option<PathBuf>,
    },
    /// Join the shards of a sharded series (`rmda defects --out OUT --shard I/N`)
    /// into OUT's tables, as an unsharded run writes them: rows in frame order,
    /// grain tracking (--segment) replayed over all frames, defect files
    /// gathered under OUT/defects/. Refuses unfinished or mismatched shards.
    Merge {
        /// The series' --out directory, holding shard-I-of-N/.
        #[arg(value_name = "OUT")]
        out: PathBuf,
        /// Replace tables already in OUT.
        #[arg(long)]
        overwrite: bool,
    },
    /// Print summary statistics of the input frames.
    Info {
        /// Input file (dump/xyz/data/extxyz).
        #[arg(value_name = "FILE")]
        file: Option<PathBuf>,
    },
    /// Write `FILE.rmdaidx`, a frame index that makes `--frame k` / `--start k`
    /// seek instead of scanning k frames. For a `.gz` file it is a gzip index
    /// instead (access points every 8 MB of text), from which the file is
    /// inflated in parallel. Readers use it automatically while it matches the
    /// file's size and modification time.
    Index {
        /// Input file (dump/xyz/binary dump, or any `.gz`).
        #[arg(value_name = "FILE")]
        file: Option<PathBuf>,
    },
    /// Compute the neighbour census (per-atom coordination + histogram).
    Census {
        /// Input file (dump/xyz/data/extxyz).
        #[arg(value_name = "FILE")]
        file: Option<PathBuf>,
        /// Neighbour cutoff in Å.
        #[arg(long, default_value_t = 3.5)]
        cutoff: f64,
    },
    /// Cluster analysis: connected components under a distance criterion.
    ///
    /// A cluster is "these atoms are transitively within `--cutoff` of each
    /// other", with no opinion about crystal structure — so this is the cheap
    /// one. Use `rmda segment` when the boundary has to be a physical claim.
    ///
    /// Writes three properties, because two questions are being asked:
    /// `Cluster` labels every atom with the connected set it belongs to,
    /// including sets below `--min-size`, and `ParticleCluster` is the filtered
    /// view (-1 for an atom in too small a cluster). Ids are OVITO's (3.17 and
    /// later): 0-based, -1 for an atom in no cluster. `ClusterSize` is per
    /// cluster, not per atom.
    Cluster {
        /// Input file (dump/xyz/data/extxyz).
        #[arg(value_name = "FILE")]
        file: Option<PathBuf>,
        /// Bond cutoff in Å. Ignored when `--knear` is given.
        #[arg(long)]
        cutoff: Option<f64>,
        /// Bond each atom to its k nearest instead of using a cutoff.
        ///
        /// For frames whose density varies enough that no single radius is right
        /// everywhere. The bond count per atom is then fixed at k, whatever the
        /// local spacing.
        #[arg(long, value_name = "K", conflicts_with = "cutoff")]
        knear: Option<usize>,
        /// Per type-pair cutoff `A-B=CUT` (repeatable; types by number or by
        /// name), OVITO's `neighbor_mode = ToPairCutoffs`: two atoms bond at or
        /// below their pair's cutoff, and a pair not listed never bonds.
        #[arg(
            long = "pair-cutoff",
            value_name = "A-B=CUT",
            conflicts_with_all = ["cutoff", "knear"]
        )]
        pair_cutoff: Vec<String>,
        /// Clusters smaller than this are excluded from `ParticleCluster`.
        #[arg(long, default_value_t = 1)]
        min_size: usize,
        /// Cluster only the atoms matching this expression, e.g. `'type == 2'`
        /// or `'z > 500 && type != 3'` (fields x y z sx sy sz type id).
        ///
        /// Unselected atoms take part in no bond and get `Cluster = -1`.
        #[arg(long, value_name = "EXPR")]
        select: Option<String>,
        /// Cluster only the atoms flagged by the input's `Selection` column
        /// (OVITO's `only_selected`); with `--select` an atom must pass both.
        #[arg(long)]
        only_selected: bool,
        /// Number clusters largest first (ties in first-atom order) instead of
        /// in first-atom order.
        #[arg(long)]
        sort_by_size: bool,
        /// Write unwrapped coordinates to `-o`: each atom moved across the
        /// periodic boundaries to sit next to its cluster's first atom.
        #[arg(long)]
        unwrap: bool,
        /// Write the per-cluster table here (CSV): size, mass, centre of mass,
        /// radius of gyration and gyration tensor, over unwrapped coordinates
        /// and mass-weighted when the input has masses, and whether the
        /// cluster percolates (is bonded to its own periodic image, so has no
        /// unwrapping and no meaningful centre or Rg).
        #[arg(long, value_name = "FILE")]
        table: Option<PathBuf>,
    },
    /// Radial distribution function (g(r), coordination number, partials).
    Rdf {
        /// Input file (dump/xyz/data/extxyz).
        #[arg(value_name = "FILE")]
        file: Option<PathBuf>,
        /// Largest pair distance in Å. Required.
        #[arg(long, value_name = "R")]
        rmax: f64,
        /// Bin width in Å.
        #[arg(long, default_value_t = 0.02)]
        dr: f64,
        /// Skip the per-type-pair partials `g_ab(r)`.
        #[arg(long)]
        no_partials: bool,
        /// Skip the coordination-number integral `n(r)`.
        #[arg(long)]
        no_coordination: bool,
        /// Normalisation volume: `auto` (effective when an axis is open or a
        /// periodic axis has a vacuum gap, else the box), `box`, `effective`
        /// (N over the material's interior number density) or a volume in Å³.
        #[arg(long, default_value = "auto", value_name = "MODE")]
        volume: RdfVolume,
        /// Count pairs only around atoms more than rmax from a free surface,
        /// so every counted shell is whole and g(r) → 1 in a film.
        #[arg(long)]
        edge_correct: bool,
    },
    /// Warren-Cowley short-range-order parameters (design §18.1).
    ///
    /// `alpha_ij = 1 - p_ij / c_j` per species pair and per shell, with shells
    /// given as cumulative k-nearest ranks the way OVITO's `nneigh=[0, 12, 18]`
    /// is. Negative means the two species like each other's company (short-range
    /// order), positive means they avoid it.
    Wc {
        /// Input file (dump/xyz/data/extxyz).
        #[arg(value_name = "FILE")]
        file: Option<PathBuf>,
        /// Cumulative neighbour counts ending each shell.
        ///
        /// `"12"` is one shell of the 12 nearest, `"12,18"` adds a second shell of
        /// the 18 nearest minus the 12 nearest. A leading zero is optional.
        #[arg(long, value_name = "RANKS", default_value = "0,12")]
        wc_shells: String,
        /// Measure only the atoms flagged by a `Selection` column.
        ///
        /// Concentrations are then taken over the selected atoms plus every atom
        /// their outermost shell reaches, which is what makes the parameters
        /// relative to the region rather than to the whole box.
        #[arg(long)]
        only_selected: bool,
        /// Report only the diagonal `alpha_ii` of each matrix.
        #[arg(long)]
        only_same_type: bool,
        /// Also write the per-atom matrix, for a spatial map of the order.
        #[arg(long)]
        per_particle: bool,
        /// Measure only atoms deeper below a free surface than their own k-th
        /// neighbour, so no shell is cut by a surface (thin films, clusters).
        ///
        /// In a film the k nearest of a surface atom reach into its second
        /// shell and bias alpha. Concentrations stay over every atom (or the
        /// selection); the excluded count is reported as wc_atoms_surface.
        #[arg(long)]
        edge_correct: bool,
    },
    /// Common-neighbour analysis (per-atom structure type + histogram).
    Structure {
        /// Input file (dump/xyz/data/extxyz).
        #[arg(value_name = "FILE")]
        file: Option<PathBuf>,
        /// CNA variant (OVITO's modes plus identify-diamond).
        #[arg(long, value_enum, default_value_t = CnaModeArg::Adaptive)]
        mode: CnaModeArg,
        /// Cutoff radius in Å (fixed, shell and bond modes).
        #[arg(long, default_value_t = 3.5)]
        cutoff: f64,
        /// Inner cutoff radius in Å (shell mode).
        #[arg(long, default_value_t = 0.0)]
        low: f64,
        /// Also emit the per-signature counters (n421/n422/n555/n444/n666).
        #[arg(long)]
        full: bool,
    },
    /// Polyhedral template matching (per-atom structure type + RMSD cutoff).
    ///
    /// Writes a histogram of the per-atom `StructureType` property, the same
    /// classification OVITO's PTM modifier performs, with OVITO's own defaults.
    Ptm {
        /// Input file (dump/xyz/data/extxyz).
        #[arg(value_name = "FILE")]
        file: Option<PathBuf>,
        /// RMSD cutoff, in units of the mean neighbour length. Atoms whose best
        /// fit exceeds it are reported as OTHER.
        #[arg(long, default_value_t = 0.1)]
        rmsd_cutoff: f64,
        /// Comma-separated structures to test for.
        ///
        /// The default is OVITO 3.15's set (fcc, hcp, bcc). OVITO ships ico,
        /// sc, cubic diamond (`dcub`), hexagonal diamond (`dhex`) and graphene
        /// disabled too: enabling a structure can change other atoms' labels,
        /// because the lowest RMSD wins. Upstream `lib_ptm`'s C++ default
        /// also tests `ico`.
        #[arg(long, value_delimiter = ',', default_value = "fcc,hcp,bcc")]
        types: Vec<PtmTypeArg>,
        /// Neighbour order PTM receives: `distance` (default) or `voronoi`,
        /// OVITO 3.15's (decreasing Voronoi face solid angle; atom-for-atom
        /// OVITO types on the reference frames, at the cost of one Voronoi
        /// cell per atom).
        #[arg(long, value_enum)]
        ptm_order: Option<PtmOrderArg>,
        /// Comma-separated opt-in per-atom outputs, as OVITO's PTM options:
        /// `ordering` (alloy ordering type from the particle types,
        /// `StructureOrdering`), `F` (elastic deformation gradient, 9
        /// row-major components, `StructureDeformationGradient`), `template`
        /// (conventional template index, `StructureTemplate`),
        /// `correspondences` (OVITO's encoded template-to-neighbour mapping,
        /// int64 `Correspondences`), and two upstream-PTM outputs OVITO lacks:
        /// `residual` (the F fit's per-axis squared residual,
        /// `StructureDeformationResidual`) and `polar` (F = P U,
        /// `StructurePolarU` rotation and `StructurePolarP` stretch), or `all`.
        /// None by default. Use `--ptm-order voronoi` for OVITO's values atom
        /// for atom.
        #[arg(long, value_delimiter = ',', value_parser = parse_ptm_output)]
        outputs: Vec<u32>,
    },
    /// Mark the atoms sitting on a grain boundary.
    ///
    /// A Wigner-Seitz vacancy count is only meaningful where the reference lattice
    /// is locally right, and across a grain boundary it is locally right for
    /// *neither* grain. Counting those atoms anyway makes the defect density a
    /// function of grain size: a finely grained sample then reads as more defective
    /// than one that is genuinely damaged. This identifies the seam so the
    /// per-grain counts can leave it out.
    ///
    /// An atom is on the boundary when the grain ids among *itself and its
    /// neighbours* within `--w-db` hold two or more distinct values.
    GrainBoundaries {
        /// Input file (dump/xyz/data/extxyz).
        #[arg(value_name = "FILE")]
        file: Option<PathBuf>,
        /// Exclusion half-width in Å.
        ///
        /// Roughly one extra interplanar spacing in a typical metal: wide enough to
        /// take in the atoms whose reference lattice the seam has already distorted,
        /// narrow enough not to eat the grain interior. `gb_slab_thickness` reports
        /// how wide the seam actually came out, which is how you tell whether this
        /// value was the binding constraint or whether the boundary is genuinely
        /// thinner than it.
        #[arg(long = "w-db", value_name = "ANGSTROM")]
        w_db: Option<f64>,
        /// Treat atoms with no grain id as boundary atoms.
        ///
        /// Off by default, because an unassigned atom is usually not on a seam but
        /// simply not crystalline enough to segment. Either way, a neighbour with
        /// no grain id counts as a second id, so the crystalline atoms next to an
        /// unassigned layer are boundary atoms; this flag only decides whether the
        /// unassigned atoms themselves are in the boundary class too.
        #[arg(long)]
        unassigned_is_boundary: bool,
        /// Dissolve crystallite clusters smaller than this many atoms into the
        /// neighbouring grains (OVITO's `min_grain_size`).
        ///
        /// A crystallite below this is a defect's disordered shell rather than a
        /// grain, and counting it as one fragments the very boundaries this pass
        /// is meant to measure. The OVITO default of 100 atoms is the default
        /// here too.
        #[arg(long, default_value_t = 100, value_name = "ATOMS")]
        min_grain_size: usize,
        /// Do not hand non-crystalline atoms (PTM `OTHER`) to the nearest adjacent
        /// grain. Atoms of clusters dissolved by `--min-grain-size` are adopted
        /// regardless.
        ///
        /// Orphan adoption is OVITO's default: the disordered layer along a
        /// boundary belongs to the grain it sits in, so the `Grain` column has
        /// one id per atom and the seam can be measured between them. Without it
        /// those atoms stay at -1, and `gb_atoms_unassigned` is their only
        /// report.
        #[arg(long)]
        no_adopt_orphans: bool,
        /// Write the per-grain table here, one CSV row per grain.
        ///
        /// Includes the seam counts, so a grain's row can say how much of it is
        /// boundary before any defect count exists.
        #[arg(long, value_name = "FILE")]
        grain_table: Option<PathBuf>,
    },
    /// Per-grain Wigner-Seitz defect census (design §16).
    ///
    /// Fits a full affine lattice (rotation, strain, shear) to each grain from
    /// its PTM orientation and neighbour vectors, makes it commensurate with the
    /// box on periodic axes the grain spans, and counts vacancies and
    /// interstitials by nearest-site occupancy. Every count conserves atoms:
    /// vacancies - interstitials = sites - atoms, separately for grain interiors
    /// and the grain-boundary zone. Grains come from `--region` expressions, an
    /// input column (`--grain-column`), or the whole frame as one grain.
    Defects {
        /// Input file(s) (dump/xyz/data/extxyz); several need --out.
        #[arg(value_name = "FILE")]
        files: Vec<PathBuf>,
        /// Time series: stream every FILE (ordered by the trailing integer of its
        /// name) and append long-format tables to this directory: summary.csv,
        /// grains.csv, clusters.csv and, with --segment, grain_events.csv. A rerun
        /// skips files already done.
        #[arg(long, value_name = "DIR")]
        out: Option<PathBuf>,
        /// With --out: also write each frame's vacancies.xyz and
        /// interstitials.xyz (and non_lattice.xyz) under <out>/defects/, laid out
        /// by --defects-layout.
        #[arg(long)]
        defects_xyz: bool,
        /// Element of each atom type, in type order (`--elements V W` makes
        /// type 1 V and type 2 W), for the species column of interstitials.xyz
        /// and non_lattice.xyz. Types not named take the file's own labels, then
        /// the element whose standard atomic weight matches the type's mass in a
        /// data file's Masses section (183.84 is W), else X.
        #[arg(long, value_name = "ELEMENT", num_args = 1..)]
        elements: Vec<String>,
        /// With --out: how each frame's defect files are laid out under
        /// <out>/defects/. Giving it implies --defects-xyz.
        #[arg(long, value_enum, value_name = "LAYOUT")]
        defects_layout: Option<DefectsLayoutArg>,
        /// Also write non_lattice.xyz beside the defect files: every atom whose
        /// PTM structure is not its grain's lattice (OTHER, or e.g. FCC/HCP atoms
        /// in a BCC grain), and every atom in no grain.
        ///
        /// Each vacancy should sit inside a shell of these atoms, which makes the
        /// defect files checkable by eye in OVITO (`StructureType` uses OVITO's
        /// codes). Needs `--defects DIR`, or `--out DIR` with `--defects-xyz`.
        #[arg(long)]
        non_lattice_atoms: bool,
        /// With --out: discard existing tables instead of resuming.
        #[arg(long)]
        overwrite: bool,
        /// With --out: run shard I of N (counted from 0), e.g. one SLURM array
        /// task each (`--shard $SLURM_ARRAY_TASK_ID/$SLURM_ARRAY_TASK_COUNT`).
        /// The ordered frames are cut into N contiguous blocks; shard I does
        /// block I into <out>/shard-I-of-N/. `rmda merge <out>` then joins the
        /// shards into the tables an unsharded run writes, grain tracking
        /// (--segment) included.
        #[arg(long, value_name = "I/N", value_parser = parse_shard, requires = "out")]
        shard: Option<(usize, usize)>,
        /// Defect cluster cutoff in Å (default: halfway between the first and
        /// second neighbour shells).
        #[arg(long, value_name = "ANGSTROM")]
        cluster_cutoff: Option<f64>,
        /// A region defining one grain, e.g. `--region 'z <= 500' --region 'z > 500'`.
        /// Atoms take the first region they match; atoms in none are in no grain.
        /// Fields: x y z (Å), sx sy sz (fractional), type, id; clauses join with &&.
        #[arg(long = "region", value_name = "EXPR")]
        regions: Vec<String>,
        /// Take grain labels from this input column instead (e.g. `Grain`).
        #[arg(long, value_name = "NAME", conflicts_with = "regions")]
        grain_column: Option<String>,
        /// Grain-boundary zone width in Å (default: 1.5 first-neighbour distances).
        #[arg(long, value_name = "ANGSTROM")]
        boundary_width: Option<f64>,
        /// Grains with fewer atoms are not measured.
        #[arg(long, default_value_t = 100)]
        min_atoms: usize,
        /// PTM RMSD cutoff for the structure and orientation seed.
        #[arg(long, default_value_t = 0.1)]
        rmsd_cutoff: f64,
        /// Write the per-grain table (CSV) here.
        #[arg(long, value_name = "FILE")]
        table: Option<PathBuf>,
        /// Write `vacancies.xyz` and `interstitials.xyz` into this directory.
        #[arg(long, value_name = "DIR")]
        defects: Option<PathBuf>,
        /// Reference per grain: `global` (one affine lattice), `patch` (patch-local
        /// lattices sharing one index space, for bent or graded grains), `auto`
        /// (global, falling back to patches when its displacement gates fail, and to
        /// `local` when those fail too), or
        /// `local` (no lattice for the whole grain: each atom off the lattice is
        /// placed on a lattice fitted around it, for grains with dislocation
        /// networks, curvature or cascade damage), or `periodic` (a periodic
        /// single crystal whose lattice is commensurate with the box:
        /// Wigner-Seitz against that lattice, ambiguous atoms resolved, past the
        /// displacement gates; counts defects stored as whole planes added or
        /// removed, which `local` cannot see; other grains as `auto`).
        #[arg(long, value_enum, default_value_t = ReferenceArg::Auto)]
        reference: ReferenceArg,
        /// Patch edge length in Å (default: 12 first-neighbour distances).
        #[arg(long, value_name = "ANGSTROM")]
        patch_size: Option<f64>,
        /// Local census (`--reference local`, or auto's fallback): how a cluster
        /// that overlaps a counted region is counted. `union` (default, most
        /// accurate) tries their union first; `partial` counts it at once on
        /// what is left, parallel and much faster (a 128M-atom polycrystal on
        /// 80 threads: 7 min against 49, totals within 0.2 %), but less exact
        /// where loops meet (on a 5.8M-atom irradiated film: I -2.3 % against
        /// -0.6 % of the global reference).
        #[arg(long, value_enum, default_value_t = LocalMergeArg::Union)]
        local_merge: LocalMergeArg,
        /// Global and patch references: an atom farther than 0.3 bonds from
        /// its nearest site takes the site its neighbours point to (as the
        /// local census does). Off by default: every atom goes to its
        /// nearest site, Wigner-Seitz as OVITO counts, which reports
        /// spurious Frenkel pairs inside large dislocation loops (13-41 per
        /// a/2<111> loop in W, where the crystal sits nearly b/2 off the
        /// lattice); on, they are gone and the net count is unchanged.
        #[arg(long)]
        resolve_ambiguous: bool,
        /// Explicit-reference Wigner-Seitz (OVITO's WignerSeitzAnalysisModifier):
        /// the sites are this frame's atoms instead of a fitted lattice. Every
        /// atom goes to its nearest site; empty sites are vacancies, extra
        /// occupants interstitials. `--region`, `--grain-column` and `--segment`
        /// are evaluated on this reference frame, once; an atom counts in the
        /// grain and class of its site. Without them all sites are one grain.
        #[arg(long, value_name = "FILE", conflicts_with_all = ["reference", "patch_size"])]
        ws_reference: Option<PathBuf>,
        /// With --ws-reference: `off` (positions as they are, OVITO's default),
        /// `to-reference` (map current positions into the reference cell, as
        /// OVITO's ToReference) or `to-current` (map the sites into the current
        /// cell). Use a mapping when the box was deformed between the frames.
        #[arg(long, value_enum, default_value_t = WsAffineArg::Off, requires = "ws_reference")]
        ws_affine: WsAffineArg,
        /// Also count antisites: singly occupied sites holding an atom of
        /// another type than the site's (OVITO's per-type occupancy). With
        /// --ws-reference a site's type is its reference atom's; with a fitted
        /// reference (global or patch) it is the majority type on each motif
        /// basis site of the grain (B2 on BCC, L1_2 or L1_0 on FCC); under the
        /// local reference, the majority on each basis site of every vote
        /// region. Adds antisite columns to the tables,
        /// `antisites.csv` (series) and `antisites.xyz` beside the defect
        /// files. Vacancies and interstitials do not change.
        #[arg(long)]
        antisites: bool,
        /// Recognise double HCP (ABAC stacking, e.g. La, Pr, Nd): close-packed
        /// atoms whose layers alternate FCC and HCP labels become one dHCP
        /// grain (structure 32) with its own lattice. Off by default: FCC with
        /// dense stacking faults holds such runs too, and would lose them to
        /// separate dHCP grains.
        #[arg(long)]
        dhcp: bool,

        /// Segment grains from PTM orientations (within each `--region`) instead
        /// of taking regions or a column as the grains.
        #[arg(long)]
        segment: bool,
        /// Segmentation: crystallites whose mean orientations differ by less than
        /// this merge, degrees.
        #[arg(long, default_value_t = 5.0, value_name = "DEG")]
        seg_threshold: f64,
        /// Segmentation: grains smaller than this dissolve into their neighbours.
        #[arg(long, default_value_t = 100, value_name = "ATOMS")]
        seg_min_size: usize,
        /// Segmentation: stacking-fault and twin planes. `merge` keeps an FCC
        /// grain with a fault whole, but one lattice cannot count both sides of
        /// a fault that crosses the grain (the grain is gated); `separate` splits
        /// it there.
        #[arg(long, value_enum, default_value_t = StackingFaults::Merge)]
        seg_stacking_faults: StackingFaults,
        /// Segmentation: merge algorithm (see `rmda segment --mode`).
        #[arg(long, value_enum, default_value_t = SegModeArg::Crystallite)]
        seg_mode: SegModeArg,
        /// PTM neighbour order: `distance`, or `voronoi` as OVITO. Default:
        /// `voronoi` with `--segment --seg-mode auto`, else `distance`.
        #[arg(long, value_enum)]
        ptm_order: Option<PtmOrderArg>,
    },
    /// Grain segmentation (design §15): crystallites of coherent PTM
    /// orientation, merged on their mean orientations, small grains dissolved
    /// and orphans adopted. Writes the `Grain` column with `-o`.
    Segment {
        /// Input file (dump/xyz/data/extxyz).
        #[arg(value_name = "FILE")]
        file: Option<PathBuf>,
        /// Merge algorithm: `crystallite` (default) or `auto` (OVITO's
        /// automatic graph clustering, for OVITO parity).
        #[arg(long, value_enum, default_value_t = SegModeArg::Crystallite)]
        mode: SegModeArg,
        /// PTM neighbour order: `distance`, or `voronoi` as OVITO. Default:
        /// `voronoi` with `--mode auto`, else `distance`.
        #[arg(long, value_enum)]
        ptm_order: Option<PtmOrderArg>,
        /// Crystallites whose mean orientations differ by less than this merge,
        /// degrees.
        #[arg(long, default_value_t = 5.0, value_name = "DEG")]
        threshold: f64,
        /// Bonds below this disorientation form crystallites, degrees.
        #[arg(long, default_value_t = 2.0, value_name = "DEG")]
        core_angle: f64,
        /// Grains smaller than this dissolve into their neighbours.
        #[arg(long, default_value_t = 100, value_name = "ATOMS")]
        min_size: usize,
        /// Crystallites merge only when they share at least this many bonds.
        #[arg(long, default_value_t = 4, value_name = "BONDS")]
        min_bonds: u32,
        /// HCP stacking-fault and twin planes: join the FCC grain around them
        /// (`merge`, as OVITO) or never share a grain with FCC (`separate`).
        #[arg(long, value_enum, default_value_t = StackingFaults::Merge)]
        stacking_faults: StackingFaults,
        /// Leave unassigned atoms unassigned instead of adopting them.
        #[arg(long)]
        no_adopt: bool,
        /// PTM RMSD cutoff.
        #[arg(long, default_value_t = 0.1)]
        rmsd_cutoff: f64,
        /// Write the per-grain table (CSV) here, and the grain-pair table next to it.
        #[arg(long, value_name = "FILE")]
        table: Option<PathBuf>,
        /// Recognise double HCP (ABAC stacking, e.g. La, Pr, Nd): close-packed
        /// atoms whose layers alternate FCC and HCP labels become one dHCP
        /// grain (structure 32) with its own lattice. Off by default: FCC with
        /// dense stacking faults holds such runs too, and would lose them to
        /// separate dHCP grains.
        #[arg(long)]
        dhcp: bool,
        /// With --dhcp and stacking-fault merging: a dHCP crystal's stacking
        /// faults (FCC/HCP runs of at most 6 layers between dHCP layers) join
        /// its grain, so a faulted crystal is one grain. Not offered by
        /// `rmda defects`: the defect census cannot count such a grain.
        #[arg(long, requires = "dhcp")]
        dhcp_merge_faults: bool,
    },
    /// Wigner-Seitz defect analysis.
    ///
    /// Three ways to get the reference, in increasing order of how much they
    /// assume. By default the lattice is given on the command line and taken as
    /// exact, which requires a box it tiles. `--self` fits the lattice to
    /// the frame (design §16.3), which is what an atomistic simulation needs when no
    /// reference exists and the crystal is not on its ideal sites. `--reference`
    /// takes the sites from another frame (design §16.3a), which needs no
    /// commensurability and no fit and is the mode for a reference configuration.
    Ws {
        /// Input file (dump/xyz/data/extxyz).
        #[arg(value_name = "FILE")]
        file: Option<PathBuf>,
        /// Reference lattice type. Also the type that `--self` fits.
        #[arg(long, value_enum, default_value_t = LatticeArg::Fcc)]
        lattice: LatticeArg,
        /// Nearest-neighbour distance of the reference lattice, in Å.
        ///
        /// Under `--self` this is only the search's starting guess for the spacing,
        /// which is then measured from the atoms: the fitted scale comes from the
        /// structure, not from here. It is not ignored -- the guess sets the neighbour
        /// search's cutoff -- but a wrong guess costs accuracy in the search rather
        /// than producing a wrong answer.
        #[arg(long, default_value_t = 2.86)]
        nn_distance: f64,
        /// Fit the reference lattice to the frame instead of taking it as given.
        #[arg(long = "self")]
        fit_self: bool,
        /// Reference *frame* instead of a lattice: analyse the input against the
        /// sites of this file's first frame (design §16.3a).
        ///
        /// This is the mode for a reference configuration rather than a reference
        /// lattice: it answers "which site of the reference does this atom occupy
        /// now", so it survives cell strain, relaxed geometry and a defect count
        /// without refitting anything. Mutually exclusive with `--self`. Runs the
        /// explicit-reference census (`rmda defects --ws-reference`): every atom
        /// on its nearest reference site, OVITO's Wigner-Seitz rule.
        #[arg(long, value_name = "FILE")]
        reference: Option<PathBuf>,
        /// How to carry input positions into the reference frame under
        /// `--reference`.
        ///
        /// `none` compares the positions as given, which is right when both frames
        /// share a box. `to-reference` maps them through fractional coordinates
        /// first, which is right when the cell has strained or changed shape
        /// between the frames — otherwise every atom reads as displaced.
        #[arg(long, value_enum, default_value_t = MapArg::None)]
        map: MapArg,
        /// Ignored (with a warning): pure WS has no search radius under
        /// `--reference`. Every atom goes to its nearest site, however far.
        #[arg(long)]
        cutoff: Option<f64>,
        /// Orientation seed `(w, x, y, z)` for the fit, normally a PTM quaternion.
        ///
        /// Without a seed the search starts from the identity, which only suits a
        /// region whose axes already line up with the box. A grain tilted out of
        /// that plane needs the seed.
        #[arg(long, value_names = &["W", "X", "Y", "Z"], num_args = 4)]
        orientation: Option<Vec<f64>>,
        /// How far an atom may sit from a lattice point and still count as on it,
        /// in Å. Defaults to half the nearest-neighbour distance. Under
        /// `--reference` only a diagnostic: atoms farther than this from their
        /// site are reported in `ws_config_unmatched` (and still counted);
        /// default 1.05 x the reference's first-neighbour distance.
        #[arg(long)]
        tolerance: Option<f64>,
        /// Smallest cluster of empty sites to report as a void, under `--self`.
        ///
        /// The default of 1 counts every empty site, which is the vacancy count under
        /// another name. A crater is several adjacent empty cells, so this is how a
        /// caller asks for voids rather than vacancies.
        #[arg(long, default_value_t = 1)]
        void_min_sites: usize,
        /// Directory to write the defect sites to as separate atom files.
        ///
        /// Writes `vacancies.xyz` (one row per empty lattice site, species `VAC`),
        /// `interstitials.xyz` (the input atoms seated on over-occupied sites, with
        /// their own species and the site's occupancy) and `interstitials_sites.xyz`
        /// (one row per over-occupied site, species `INT`, with its occupancy).
        ///
        /// These are separate files rather than extra columns on `-o` because a
        /// vacancy has no atom: `ws_occupancy` holds one entry per *site* and so
        /// cannot be a column in a per-atom table without misaligning it. Under
        /// `--void-min-sites` the vacancy rows also carry `void_id`, so a crater can
        /// be selected as a group.
        #[arg(long, value_name = "DIR")]
        defects: Option<PathBuf>,
        /// Atoms used for the `--self` scale and orientation search.
        ///
        /// The winning lattice is always re-counted over every atom, so this only
        /// bounds the search's cost, never the result.
        #[arg(long, default_value_t = 100_000)]
        search_sample: usize,
    },
    /// Void census (design §17).
    ///
    /// Method A (default): voids are clusters of vacancy sites of the `rmda
    /// defects` census (same references: `--site-reference`, as
    /// `rmda defects --reference`, or `--ws-reference FILE`). Volume = sites x Omega_site of each
    /// site's grain (det T / n_basis), equivalent radius (3V/4pi)^(1/3); a void
    /// open to empty space the census did not count (a free surface) is
    /// classed `surface`, else `boundary` or `bulk`.
    ///
    /// Method B (`--experimental`): the phi-field volume, still
    /// unvalidated on thermal or relaxed crystals (design §17.2): on a defect-free
    /// thermal crystal it measures a shell around every displaced atom. Its
    /// options (`--lattice`, `--nn-distance`, `--reference FILE`, `--map`,
    /// `--region`, `--samples`, ...) need `--experimental`.
    Void {
        /// Input file(s) (dump/xyz/data/extxyz); several need --out (method A).
        #[arg(value_name = "FILE")]
        files: Vec<PathBuf>,
        /// Method A time series: stream every FILE (ordered by the trailing
        /// integer of its name) and append long-format tables to this
        /// directory: summary.csv (one row per frame), voids.csv (one row per
        /// void per frame) and grains.csv (one row per grain per frame). With
        /// --segment the grains are tracked across frames (stable_grain
        /// columns, grain_events.csv), as `rmda defects --out` tracks them. A
        /// rerun skips files already done and refuses other settings, as `rmda
        /// defects --out`.
        #[arg(long, value_name = "DIR", conflicts_with_all = ["experimental", "table", "sites"])]
        out: Option<PathBuf>,
        /// With --out: write each frame's void vacancy sites (as --sites) to
        /// <out>/sites/<index>.xyz, grains under their stable ids.
        #[arg(long, requires = "out")]
        sites_xyz: bool,
        /// With --out: discard existing tables instead of resuming.
        #[arg(long, requires = "out")]
        overwrite: bool,
        /// With --out: run shard I of N of the series into <out>/shard-I-of-N/,
        /// as `rmda defects --shard`; then `rmda merge <out>`.
        #[arg(long, value_name = "I/N", value_parser = parse_shard, requires = "out")]
        shard: Option<(usize, usize)>,
        /// Run method B (the phi field) instead of method A. Acknowledges that
        /// method B is unvalidated on thermal and relaxed material.
        #[arg(long)]
        experimental: bool,
        /// Reference lattice type, for the ideal neighbour directions.
        #[arg(long, requires = "experimental", value_enum, default_value_t = LatticeArg::Fcc)]
        lattice: LatticeArg,
        /// Nearest-neighbour distance of the reference lattice, in Å.
        #[arg(long, requires = "experimental", default_value_t = 2.86)]
        nn_distance: f64,
        /// Reference *frame* instead of a lattice: take the ideal configuration's
        /// cell and directions from this file's first frame (design §17.2).
        #[arg(long, requires = "experimental", value_name = "FILE")]
        reference: Option<PathBuf>,
        /// How to carry input positions into the reference frame under
        /// `--reference`. `to-reference` is required when the cell has strained or
        /// changed shape, or every atom reads as displaced.
        #[arg(long, requires = "experimental", value_enum, default_value_t = MapArg::None)]
        map: MapArg,
        /// Region to integrate, as `xlo,xhi,ylo,yhi,zlo,zhi` in Å. Defaults to the
        /// box interior inset by one nearest-neighbour distance, so the integral
        /// never straddles a boundary where the periodic images are.
        ///
        /// Six comma-separated numbers. Parsed here rather than by clap: a variable
        /// value count makes clap greedily consume the positional FILE, and the axis
        /// order deserves an error message of its own anyway.
        #[arg(
            long,
            requires = "experimental",
            value_name = "XLO,XHI,YLO,YHI,ZLO,ZHI"
        )]
        region: Option<String>,
        /// Monte Carlo samples for the volume integral.
        #[arg(long, requires = "experimental", default_value_t = 40_000)]
        samples: usize,
        /// Seed for the sampler. The same seed makes two fields comparable.
        #[arg(long, requires = "experimental", default_value_t = 11)]
        seed: u64,
        /// Threshold: points with `phi > 1 + epsilon` count as void.
        #[arg(long, requires = "experimental", default_value_t = 0.01)]
        epsilon: f64,
        /// Measure at each of these thresholds and fit `V(eps)` to extrapolate to
        /// `phi = 1`. Needs at least 3 distinct values; the quadratic degenerates
        /// to a straight line below that, which overstates a defect-free crystal by
        /// orders of magnitude.
        #[arg(long, requires = "experimental", value_name = "E1,E2,E3,...")]
        epsilons: Option<String>,
        /// Fit design §17.2's roto-strain field `T(x)` and evaluate phi through it.
        ///
        /// Off by default. It reduces the reported volume on real relaxed data
        /// (4.9% clean, 6.4% with planted defects) and removes none of the thermal
        /// shell, so it is a knob to demonstrate that, not a correction that helps.
        #[arg(long, requires = "experimental")]
        strain: bool,
        /// Mesh spacing for `--strain`, in Å. Default: half the lattice constant.
        #[arg(long, requires = "experimental")]
        strain_spacing: Option<f64>,
        /// Gaussian width for `--strain`, in Å. Default: the lattice constant.
        #[arg(long, requires = "experimental")]
        strain_sigma: Option<f64>,
        /// Report the excess over a reference frame's field instead of the raw
        /// volume, using the same region and thresholds.
        ///
        /// This is only a void volume when the reference shares the sample's local
        /// relaxation; against an ideal lattice it is a displacement measurement.
        #[arg(long, requires_all = ["reference", "experimental"])]
        excess: bool,
        /// Measure only above the thermal shell and extrapolate (design §17.2):
        /// the window starts at the sample's largest face gap (how far apart
        /// displaced neighbours' cells have parted, in epsilon), spans 0.3 in
        /// 7 thresholds, and V^(1/3) is extrapolated linearly to phi = 1.
        #[arg(
            long,
            requires = "experimental",
            conflicts_with_all = ["epsilons", "excess", "strain"]
        )]
        thermal_window: bool,
        /// Method A: smallest vacancy cluster reported as a void (smaller ones
        /// are counted as isolated vacancies).
        #[arg(
            long,
            alias = "void-min-sites",
            default_value_t = 2,
            conflicts_with = "experimental"
        )]
        min_sites: usize,
        /// Method A: joining distance of two vacancies, Å (default: each
        /// grain's halfway between its first and second neighbour shells).
        #[arg(long, value_name = "ANGSTROM", conflicts_with = "experimental")]
        cutoff: Option<f64>,
        /// Method A: write the per-void table (CSV) here.
        #[arg(long, value_name = "FILE", conflicts_with = "experimental")]
        table: Option<PathBuf>,
        /// Method A: write every void's vacancy sites (extxyz, species VAC,
        /// with `void_id`, `grain`, `class`) here.
        #[arg(long, value_name = "FILE", conflicts_with = "experimental")]
        sites: Option<PathBuf>,
        /// Method A: reference per grain, as `rmda defects --reference`.
        #[arg(long, value_enum, default_value_t = ReferenceArg::Auto, conflicts_with = "experimental")]
        site_reference: ReferenceArg,
        /// Method A: patch edge length, Å, as `rmda defects --patch-size`.
        #[arg(long, value_name = "ANGSTROM", conflicts_with = "experimental")]
        patch_size: Option<f64>,
        /// Method A: explicit-reference sites, as `rmda defects --ws-reference`.
        #[arg(long, value_name = "FILE", conflicts_with_all = ["experimental", "patch_size"])]
        ws_reference: Option<PathBuf>,
        /// Method A: with --ws-reference, as `rmda defects --ws-affine`.
        #[arg(long, value_enum, default_value_t = WsAffineArg::Off, requires = "ws_reference")]
        ws_affine: WsAffineArg,
        /// Method A: a region defining one grain, as `rmda defects --region`.
        #[arg(
            long = "grain-region",
            value_name = "EXPR",
            conflicts_with = "experimental"
        )]
        grain_regions: Vec<String>,
        /// Method A: grain labels from this input column.
        #[arg(long, value_name = "NAME", conflicts_with_all = ["experimental", "grain_regions"])]
        grain_column: Option<String>,
        /// Method A: segment grains from PTM orientations (default settings of
        /// `rmda defects --segment`).
        #[arg(long, conflicts_with = "experimental")]
        segment: bool,
        /// Method A: grain-boundary zone width, Å.
        #[arg(long, value_name = "ANGSTROM", conflicts_with = "experimental")]
        boundary_width: Option<f64>,
        /// Method A: grains with fewer atoms are not measured.
        #[arg(long, default_value_t = 100, conflicts_with = "experimental")]
        min_atoms: usize,
        /// Method A: PTM RMSD cutoff.
        #[arg(long, default_value_t = 0.1, conflicts_with = "experimental")]
        rmsd_cutoff: f64,
        /// Method A: recognise double HCP, as `rmda defects --dhcp`.
        #[arg(long, conflicts_with = "experimental")]
        dhcp: bool,
    },
    /// Multi-analysis pipeline over the frame window (design §21).
    ///
    /// Runs the requested analyses together, one parse and one neighbour index
    /// per frame, instead of one `rmda` command per analysis. PTM then grain
    /// segmentation share a store -- `Grain` is built on the `StructureType`
    /// and `StructureOrientation` PTM writes -- and every analysis accepts the
    /// same per-frame cell list. `--rdf` and `--wc` deliberately build their own
    /// neighbourhoods (rmax and the k-th shell exceed what the shared cell list
    /// covers), so they are unaffected by the other analyses' cutoffs.
    ///
    /// CNA is not offered here, on purpose: `rmda structure` and `rmda ptm` both write a
    /// `StructureType` column but with different code encodings, and running
    /// both into one store would leave grain segmentation consuming whichever
    /// overwrote the other. `analyze` therefore has `--ptm` only.
    ///
    /// `--grains` implies `--ptm`: segmentation reads the orientation PTM
    /// measures, so grains alone is not a well-posed request.
    ///
    /// For `rmda analyze`, `-o/--output` is a *prefix*. The per-frame property
    /// files are `PREFIX.<frame>.xyz` and the g(r) tables are
    /// `PREFIX.rdf.<frame>.dat`, `<frame>` being the index within the window.
    Analyze {
        /// Input file (dump/xyz/data/extxyz).
        #[arg(value_name = "FILE")]
        file: Option<PathBuf>,
        /// Run PTM structure classification.
        #[arg(long)]
        ptm: bool,
        /// RMSD cutoff for --ptm, in units of the mean neighbour length.
        #[arg(long, default_value_t = 0.1)]
        ptm_rmsd: f64,
        /// Comma-separated structures --ptm tests for.
        #[arg(long, value_delimiter = ',', default_value = "fcc,hcp,bcc")]
        ptm_structures: Vec<PtmTypeArg>,
        /// Run the radial distribution function.
        #[arg(long)]
        rdf: bool,
        /// Largest pair distance for --rdf, in Å. Required with --rdf.
        #[arg(long, value_name = "R")]
        rdf_rmax: Option<f64>,
        /// Bin width for --rdf, in Å.
        #[arg(long, default_value_t = 0.02)]
        rdf_dr: f64,
        /// Skip the per-type-pair partials g_ab(r).
        #[arg(long)]
        rdf_no_partials: bool,
        /// Skip the coordination-number integral n(r).
        #[arg(long)]
        rdf_no_coordination: bool,
        /// Normalisation volume for --rdf (as `rmda rdf --volume`).
        #[arg(long, default_value = "auto", value_name = "MODE")]
        rdf_volume: RdfVolume,
        /// Edge-corrected centres for --rdf (as `rmda rdf --edge-correct`).
        #[arg(long)]
        rdf_edge_correct: bool,
        /// Run the Warren-Cowley short-range-order parameters.
        #[arg(long)]
        wc: bool,
        /// Cumulative neighbour counts ending each shell for --wc.
        #[arg(long, value_name = "RANKS", default_value = "0,12")]
        wc_shells: String,
        /// Run the cluster census.
        #[arg(long)]
        cluster: bool,
        /// Bond cutoff for --cluster, in Å (else --cluster-knear).
        #[arg(long)]
        cluster_cutoff: Option<f64>,
        /// Bond each atom to its k nearest instead (else --cluster-cutoff).
        #[arg(long, value_name = "K", conflicts_with = "cluster_cutoff")]
        cluster_knear: Option<usize>,
        /// Clusters smaller than this are excluded from a cluster id.
        #[arg(long, default_value_t = 1)]
        cluster_min_size: usize,
    },
    /// Convert the input to extended XYZ, a LAMMPS text or binary dump, or a
    /// LAMMPS data file (`-o FILE`; a `.gz` suffix compresses).
    Convert {
        /// Input file (dump/xyz/data/extxyz).
        #[arg(value_name = "FILE")]
        file: Option<PathBuf>,
        /// Output format; without it, taken from the `-o` suffix
        /// (`.xyz`, `.dump`/`.lammpstrj`, `.data`, `.bin`, each optionally
        /// `.gz`).
        #[arg(long, value_parser = ["xyz", "dump", "data", "bin"])]
        format: Option<String>,
        /// Per-atom properties to write: `all`, `none` or a comma-separated
        /// list (`vx,vy,vz,c_pe`). Given at all, it also writes type labels,
        /// extended-XYZ ids and the selection, and data-file image flags.
        /// Without it: positions, types, ids (not in XYZ) and image flags.
        #[arg(long, value_name = "LIST")]
        props: Option<String>,
        /// Decimals for positions and the box (`N` or `full` for the
        /// shortest exact form). Default: 10 in dump and XYZ, exact in data.
        #[arg(long, value_name = "N|full", value_parser = parse_float_fmt)]
        precision: Option<rmda_core::io::writer::FloatFmt>,
        /// Decimals for float property columns (`N` or `full`, the default).
        #[arg(long, value_name = "N|full", value_parser = parse_float_fmt, requires = "props")]
        prop_precision: Option<rmda_core::io::writer::FloatFmt>,
        /// Reduce the box tilts LAMMPS-style (flip until |xy|, |xz| <= lx/2
        /// and |yz| <= ly/2, along periodic axes) and remap the atoms, image
        /// flags kept consistent, before writing.
        #[arg(long)]
        reduce_tilt: bool,
        /// Atom style of a data-file output (`Atoms # STYLE`): atomic,
        /// charge, full, molecular, bond, angle or sphere. Its mol, q,
        /// diameter and density columns come from the frame's properties of
        /// those names, which must exist. Default: picked from the written
        /// properties (atomic without `--props`). `--atom-style` is the
        /// style of a data-file *input*.
        #[arg(long, value_name = "STYLE",
              value_parser = ["atomic", "charge", "full", "molecular", "bond", "angle", "sphere"])]
        out_atom_style: Option<String>,
    },
}

#[derive(Clone, Copy, clap::ValueEnum)]
pub(crate) enum CnaModeArg {
    /// Adaptive CNA (OVITO `AdaptiveCutoff`, the default).
    Adaptive,
    /// One global cutoff, `--cutoff` (OVITO `FixedCutoff`).
    Fixed,
    /// Larsen's parameter-free interval CNA (OVITO `IntervalCutoff`).
    Interval,
    /// Bond-based CNA over the bonds within `--cutoff` (OVITO `BondBased`
    /// after `CreateBondsModifier`).
    Bond,
    /// Neighbours in (`--low`, `--cutoff`], an exact 12 or 14 of them; not an
    /// OVITO mode.
    Shell,
    /// Identify diamond structure (OVITO `IdentifyDiamondModifier`): writes
    /// codes 0-6 (other, cubic diamond and its 1st/2nd neighbours, hexagonal
    /// diamond and its 1st/2nd neighbours).
    Diamond,
}

/// How `rmda ws --reference` maps positions between two frames.
#[derive(Clone, Copy, PartialEq, Eq, clap::ValueEnum)]
pub(crate) enum MapArg {
    /// Compare positions as given.
    None,
    /// Carry positions through fractional coordinates into the reference cell.
    ToReference,
}

impl MapArg {
    pub(crate) fn build(self) -> rmda_core::analysis::ws::AffineMapping {
        match self {
            MapArg::None => rmda_core::analysis::ws::AffineMapping::None,
            MapArg::ToReference => rmda_core::analysis::ws::AffineMapping::ToReferenceCell,
        }
    }
}

/// Reference lattices the Wigner-Seitz engine can build from.
#[derive(Clone, Copy, Debug, clap::ValueEnum)]
pub(crate) enum LatticeArg {
    Fcc,
    Bcc,
    Hcp,
    Sc,
    Dia,
    Dhcp,
}

impl LatticeArg {
    /// The name `rmda_core::lattice::from_nn` and the registry know this by.
    pub(crate) fn name(self) -> &'static str {
        match self {
            LatticeArg::Fcc => "fcc",
            LatticeArg::Bcc => "bcc",
            LatticeArg::Hcp => "hcp",
            LatticeArg::Sc => "sc",
            LatticeArg::Dia => "dia",
            LatticeArg::Dhcp => "dhcp",
        }
    }

    /// Build the lattice with nearest-neighbour distance `nn`.
    ///
    /// The nearest-neighbour distance -> conventional parameter table lives in
    /// [`rmda_core::lattice::from_nn`], shared with the Python bindings, so the
    /// two cannot disagree.
    pub(crate) fn build(self, nn: f64) -> Box<dyn rmda_core::lattice::Lattice> {
        rmda_core::lattice::from_nn(self.name(), nn)
            .expect("every LatticeArg variant has a from_nn entry")
    }
}

/// A structure `rmda ptm --types` can ask for: all eight of OVITO's.
#[derive(Clone, Copy, clap::ValueEnum)]
pub(crate) enum PtmTypeArg {
    Fcc,
    Hcp,
    Bcc,
    Ico,
    Sc,
    /// Cubic diamond (two-shell: 4 + 12 neighbours).
    Dcub,
    /// Hexagonal diamond (two-shell: 4 + 12 neighbours).
    Dhex,
    /// Graphene (two-shell: 3 + 6 neighbours).
    Graphene,
}

impl PtmTypeArg {
    pub(crate) fn to_flag(self) -> u32 {
        match self {
            PtmTypeArg::Fcc => check::FCC,
            PtmTypeArg::Hcp => check::HCP,
            PtmTypeArg::Bcc => check::BCC,
            PtmTypeArg::Ico => check::ICO,
            PtmTypeArg::Sc => check::SC,
            PtmTypeArg::Dcub => check::DCUB,
            PtmTypeArg::Dhex => check::DHEX,
            PtmTypeArg::Graphene => check::GRAPHENE,
        }
    }

    /// The five first-shell structures. Grain segmentation has
    /// no `--types` flag: the point group it uses comes from the structure type
    /// PTM assigned, so narrowing the set would change which grains can exist.
    pub(crate) fn all_flags() -> u32 {
        check::FCC | check::HCP | check::BCC | check::ICO | check::SC
    }

    pub(crate) fn name(self) -> &'static str {
        match self {
            PtmTypeArg::Fcc => "fcc",
            PtmTypeArg::Hcp => "hcp",
            PtmTypeArg::Bcc => "bcc",
            PtmTypeArg::Ico => "ico",
            PtmTypeArg::Sc => "sc",
            PtmTypeArg::Dcub => "dcub",
            PtmTypeArg::Dhex => "dhex",
            PtmTypeArg::Graphene => "graphene",
        }
    }
}

impl CnaModeArg {
    pub(crate) fn to_mode(self, cutoff: f64, low: f64) -> CnaMode {
        match self {
            CnaModeArg::Adaptive => CnaMode::Adaptive,
            CnaModeArg::Fixed => CnaMode::Fixed { cutoff },
            CnaModeArg::Interval => CnaMode::Interval,
            CnaModeArg::Bond => CnaMode::Bond { cutoff },
            CnaModeArg::Shell => CnaMode::Shell { low, high: cutoff },
            CnaModeArg::Diamond => CnaMode::Diamond,
        }
    }
}

/// `--shard I/N`: shard `I` of `N`, `I` counted from 0.
pub(crate) fn parse_shard(s: &str) -> std::result::Result<(usize, usize), String> {
    let (i, n) = s
        .split_once('/')
        .ok_or_else(|| format!("`{s}`: expected I/N, e.g. 0/4"))?;
    let i: usize = i
        .trim()
        .parse()
        .map_err(|_| format!("`{s}`: I is not a number"))?;
    let n: usize = n
        .trim()
        .parse()
        .map_err(|_| format!("`{s}`: N is not a number"))?;
    if n == 0 || i >= n {
        return Err(format!("`{s}`: need 0 <= I < N"));
    }
    Ok((i, n))
}
