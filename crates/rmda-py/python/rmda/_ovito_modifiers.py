"""OVITO modifiers running RMDA's analyses.

Each class is an OVITO Python modifier (``ovito.pipeline.ModifierInterface``)
whose parameters are Traits, so OVITO Pro draws their settings panel. The
package registers them under the ``OVITO.Modifier`` entry point, so after
``ovitos -m pip install rmda`` they appear in OVITO Pro's modifier list. With
the free ``ovito`` module they are used from a script::

    from ovito.io import import_file
    from rmda.ovito import DefectsModifier

    pipeline = import_file("frame.data")
    pipeline.modifiers.append(DefectsModifier(segment=True))
    data = pipeline.compute()
    data.attributes["defects_vacancies"], data.tables["rmda.defects.grains"]

Every modifier copies the frame out of the pipeline once (``from_ovito``),
runs the analysis with the GIL released, and writes the results back
(``to_ovito``): per-atom results as particle properties, tables as
``rmda.*`` data tables, scalars as global attributes. They need OVITO 3.11 or
newer. Imported by ``rmda.ovito`` on first use of a class name, never by
``import rmda``.
"""

import numpy as np
from traits.api import Bool, Enum, Float, Int, Str

from ovito.data import DataCollection, ParticleType
from ovito.pipeline import FileSource, ModifierInterface
from ovito.traits import OvitoObject, PropertyReference

import rmda

from .ovito import _create_property, _write_table, from_ovito, to_ovito

__all__ = ["SegmentModifier", "DefectsModifier", "VoidModifier"]


def _opt(value):
    """A trait's 0 as the analysis' "automatic" (``None``)."""
    return None if value == 0 else value


def _regions(text):
    """Region expressions separated by ``;`` (``"z <= 500; z > 500"``)."""
    return [r.strip() for r in text.split(";") if r.strip()]


class _RmdaModifier(ModifierInterface):
    """What every RMDA modifier shares: the type coding and the threads."""

    type_coding = Enum(
        "ids",
        "names",
        label="Type coding",
        desc="ids: OVITO's numeric type ids (LAMMPS files); names: the named types coded "
        "1.. in sorted order, as rmda.read codes extended XYZ",
    )
    threads = Int(0, label="Threads (0 = all cores)")

    def _frame(self, data):
        if self.threads < 0:
            raise ValueError("threads must be 0 (all cores) or positive")
        rmda.set_threads(self.threads)
        return from_ovito(data, types=self.type_coding)


class SegmentModifier(_RmdaModifier):
    """RMDA grain segmentation (``rmda segment``).

    Adds the particle properties ``Grain`` (-1 for atoms in no grain) and
    ``StructureType`` (PTM), the tables ``rmda.segment.grains`` and
    ``rmda.segment.pairs``, and the ``segment_*`` attributes.
    """

    threshold = Float(5.0, label="Merge threshold (deg)")
    core_angle = Float(2.0, label="Crystallite core angle (deg)")
    min_size = Int(100, label="Minimum grain size (atoms)")
    adopt = Bool(True, label="Adopt orphan atoms")
    rmsd_cutoff = Float(0.1, label="PTM RMSD cutoff")
    mode = Enum("crystallite", "auto", label="Algorithm", desc="auto: OVITO's GraphClusteringAuto")
    dhcp = Bool(False, label="Identify double HCP")

    def modify(self, data: DataCollection, *, frame: int, **kwargs):
        yield 0.0
        f = self._frame(data)
        yield 0.1
        res = rmda.segment(
            f,
            threshold=self.threshold,
            core_angle=self.core_angle,
            min_size=self.min_size,
            adopt=self.adopt,
            rmsd_cutoff=self.rmsd_cutoff,
            dhcp=self.dhcp,
            mode=self.mode,
        )
        yield 0.9
        to_ovito(f, data, result=res)


class DefectsModifier(_RmdaModifier):
    """RMDA per-grain Wigner-Seitz defect census (``rmda defects``).

    Grains come from ``grain_column`` (a particle property, e.g. ``Grain``
    from a :class:`SegmentModifier` upstream or a property of the file), else
    from ``regions`` (``;``-separated expressions), else from segmentation
    when ``segment`` is on, else the whole frame is one region; with
    ``segment`` on, regions or the column are split further by it, as on the
    command line. With a file in ``ws_reference`` the census is OVITO's
    explicit-reference Wigner-Seitz rule against that frame.

    ``output = "atoms"`` keeps the atoms and adds ``atom_grain``,
    ``displacement`` and ``Interstitial`` (1 on interstitial atoms) as
    particle properties. ``output = "defects"`` replaces the particles by the
    defects themselves, as OVITO's Wigner-Seitz modifier does in its sites
    mode: vacancy sites and interstitial atoms, with ``Defect Type``
    (Vacancy, Interstitial), ``Grain``, ``Defect Cluster`` and ``Boundary``. Either
    way the tables ``rmda.defects.*`` and the ``defects_*`` attributes are
    written.
    """

    reference = Enum("auto", "global", "patch", "local", "periodic", label="Reference")
    segment = Bool(True, label="Segment into grains")
    grain_column = PropertyReference(
        default_value="",
        mode=PropertyReference.Mode.Properties,
        label="Grain property (optional)",
    )
    regions = Str("", label="Regions (';'-separated expressions)")
    min_atoms = Int(100, label="Minimum grain size (atoms)")
    boundary_width = Float(0.0, label="Boundary width (0 = automatic)")
    cluster_cutoff = Float(0.0, label="Cluster cutoff (0 = automatic)")
    patch_size = Float(0.0, label="Patch size (0 = automatic)")
    rmsd_cutoff = Float(0.1, label="PTM RMSD cutoff")
    seg_threshold = Float(5.0, label="Segmentation merge threshold (deg)")
    seg_min_size = Int(100, label="Segmentation minimum grain size")
    seg_mode = Enum("crystallite", "auto", label="Segmentation algorithm")
    dhcp = Bool(False, label="Identify double HCP")
    local_merge = Enum("union", "partial", label="Local census merge")
    resolve_ambiguous = Bool(False, label="Resolve ambiguous atoms")
    antisites = Bool(False, label="Count antisites")
    ws_reference = OvitoObject(FileSource)
    ws_affine = Enum("off", "to-reference", "to-current", label="Reference mapping")
    output = Enum("atoms", "defects", label="Output")

    def _reference_frame(self, input_slots):
        if not self.ws_reference.source_path:
            return None
        return from_ovito(input_slots["ws_reference"].compute(0), types=self.type_coding)

    def modify(self, data: DataCollection, *, frame: int, input_slots, **kwargs):
        yield 0.0
        f = self._frame(data)
        ref = self._reference_frame(input_slots)
        if ref is None and self.ws_affine != "off":
            raise ValueError("ws_affine needs a reference file (ws_reference)")
        yield 0.1
        res = rmda.defects(
            f,
            regions=_regions(self.regions),
            grain_column=self.grain_column or None,
            reference=self.reference,
            segment=self.segment,
            seg_threshold=self.seg_threshold,
            seg_min_size=self.seg_min_size,
            boundary_width=_opt(self.boundary_width),
            min_atoms=self.min_atoms,
            rmsd_cutoff=self.rmsd_cutoff,
            patch_size=_opt(self.patch_size),
            cluster_cutoff=_opt(self.cluster_cutoff),
            dhcp=self.dhcp,
            seg_mode=self.seg_mode,
            ws_reference=ref,
            ws_affine=self.ws_affine,
            antisites=self.antisites,
            local_merge=self.local_merge,
            resolve_ambiguous=self.resolve_ambiguous,
        )
        yield 0.9
        if self.output == "atoms":
            interstitial = np.zeros(f.natoms, dtype=np.int32)
            interstitial[np.asarray(res["interstitials"]["atom"], dtype=np.int64)] = 1
            f["Interstitial"] = interstitial
            f["atom_grain"] = np.asarray(res["atom_grain"])
            f["displacement"] = np.asarray(res["displacement"])
            to_ovito(f, data, result=res)
        else:
            to_ovito(f, data, result=res)
            _replace_by_defects(data, res)


def _replace_by_defects(data, res):
    """Swap the particles for the defects: vacancy sites, then interstitial
    atoms, each with its grain, cluster and boundary flag."""
    vac, its = res["vacancies"], res["interstitials"]
    nv, ni = len(vac["position"]), len(its["position"])
    pos = np.concatenate([np.asarray(vac["position"]).reshape(nv, 3), np.asarray(its["position"]).reshape(ni, 3)])
    data.objects.remove(data.particles)
    particles = data.create_particles(count=nv + ni)
    particles.create_property("Position", data=pos)
    kind = particles.create_property(
        "Defect Type", dtype=np.int32, data=np.concatenate([np.ones(nv, np.int32), np.full(ni, 2, np.int32)])
    )
    kind.types.append(ParticleType(id=1, name="Vacancy"))
    kind.types.append(ParticleType(id=2, name="Interstitial"))
    # The census's defect-cluster ids (0-based, every defect in one) are not
    # OVITO's `Cluster` property, so they get a name of their own.
    for name, col in (("Grain", "grain"), ("Defect Cluster", "cluster"), ("Boundary", "boundary")):
        values = np.concatenate([np.asarray(vac[col]), np.asarray(its[col])]).astype(np.int64)
        _create_property(particles, name, values)
    ids = np.concatenate([np.zeros(nv, np.int64), np.asarray(its["id"], dtype=np.int64)])
    _create_property(particles, "Atom Id", ids)


class VoidModifier(_RmdaModifier):
    """RMDA voids, method A (``rmda void``): clusters of the defect
    census' vacancy sites. Writes the table ``rmda.voids`` and the ``void_*``
    attributes."""

    site_reference = Enum("auto", "global", "patch", "local", "periodic", label="Reference")
    segment = Bool(True, label="Segment into grains")
    grain_column = PropertyReference(
        default_value="",
        mode=PropertyReference.Mode.Properties,
        label="Grain property (optional)",
    )
    regions = Str("", label="Regions (';'-separated expressions)")
    min_sites = Int(2, label="Minimum vacancies per void")
    cutoff = Float(0.0, label="Cluster cutoff (0 = automatic)")

    def modify(self, data: DataCollection, *, frame: int, **kwargs):
        yield 0.0
        f = self._frame(data)
        yield 0.1
        res = rmda.void(
            f,
            site_reference=self.site_reference,
            min_sites=self.min_sites,
            cutoff=_opt(self.cutoff),
            regions=_regions(self.regions),
            grain_column=self.grain_column or None,
            segment=self.segment,
        )
        yield 0.9
        if res.get("voids_table") is not None:
            _write_table(data, "voids", res["voids_table"])
        to_ovito(f, data, result=res)
