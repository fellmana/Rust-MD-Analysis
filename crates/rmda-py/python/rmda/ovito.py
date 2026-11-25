"""Conversion between RMDA frames and OVITO data collections.

``import rmda.ovito`` needs the ``ovito`` Python module (``pip install ovito``,
or OVITO Pro's embedded interpreter); ``import rmda`` alone never imports it.

- :func:`from_ovito` builds an :class:`rmda.Frame` from a ``DataCollection``
  (``pipeline.compute()``, or the ``data`` a modifier receives).
- :func:`to_ovito` writes a frame's per-atom results and tables, and an
  analysis' scalars, into a ``DataCollection`` -- the one a modifier is
  modifying, or a new one.
- :class:`SegmentModifier`, :class:`DefectsModifier` and :class:`VoidModifier`
  are OVITO pipeline modifiers running ``rmda.segment``, ``rmda.defects`` and
  ``rmda.void`` (OVITO 3.11 or newer); OVITO Pro lists them in its GUI.

The conversions copy the arrays once; nothing loops over atoms in Python.
"""

import numpy as np

from ._rmda import Frame

__all__ = ["from_ovito", "to_ovito", "SegmentModifier", "DefectsModifier", "VoidModifier"]

_MODIFIERS = ("SegmentModifier", "DefectsModifier", "VoidModifier")


def __getattr__(name):
    # The modifier classes need ovito.pipeline and Traits at class definition,
    # so they load on first use (and when OVITO resolves the entry points).
    if name in _MODIFIERS:
        from . import _ovito_modifiers

        return getattr(_ovito_modifiers, name)
    raise AttributeError(f"module 'rmda.ovito' has no attribute {name!r}")

# Particle properties that are the frame itself, not extra columns. Image
# flags are OVITO's bookkeeping of the same positions; RMDA keeps positions
# as read.
_STRUCTURAL = ("Position", "Particle Type", "Particle Identifier", "Periodic Image")


def from_ovito(data, *, types="ids"):
    """An :class:`rmda.Frame` from an OVITO ``DataCollection``.

    Positions, the simulation cell (vectors, origin, periodicity), particle
    identifiers and every other numeric particle property (as input columns,
    e.g. a ``Grain`` property for ``grain_column=``) are carried over. A
    collection without a cell gets the box a cell-less file is read in: the
    atoms' bounding box, non-periodic.

    ``types`` says how particle types become RMDA's 1-based type codes:

    - ``"ids"`` (default): OVITO's numeric type ids, with their names as
      ``type_names``. This is what ``rmda.read`` gives for LAMMPS files.
    - ``"names"``: codes 1.. for the sorted set of type names present, as
      ``rmda.read`` codes the species of an extended-XYZ file (OVITO numbers
      those in order of appearance). Every type must have a name.

    A LAMMPS dump or data file with a general triclinic box (``BOX BOUNDS abc
    origin``, ``avec/bvec/cvec``) keeps its orientation in OVITO, while
    ``rmda.read`` rotates it into LAMMPS's restricted form (a along x, b in the
    xy plane), as LAMMPS does. The two frames are the same structure rotated:
    counts agree, orientations differ by that rotation.
    """
    if types not in ("ids", "names"):
        raise ValueError(f"types must be 'ids' or 'names', got {types!r}")
    particles = data.particles
    if particles is None:
        raise ValueError("the data collection holds no particles")
    n = particles.count
    positions = np.asarray(particles.positions, dtype=np.float64)

    kw = {}
    cell = data.cell
    if cell is not None:
        m = np.asarray(cell, dtype=np.float64)
        # OVITO stores the cell vectors as the first three columns and the
        # origin as the fourth; RMDA takes one vector per row.
        kw.update(cell=m[:, :3].T.copy(), pbc=tuple(bool(p) for p in cell.pbc), origin=m[:, 3].copy())

    tprop = particles.particle_types
    if tprop is not None:
        codes = np.asarray(tprop, dtype=np.int64)
        names = {t.id: t.name for t in tprop.types if t.name}
        if types == "names":
            missing = sorted(set(np.unique(codes).tolist()) - set(names))
            if missing:
                raise ValueError(f"types='names': particle type(s) {missing} have no name")
            lookup = np.empty(int(codes.max()) + 1 if n else 1, dtype=object)
            for tid, name in names.items():
                if 0 <= tid < len(lookup):
                    lookup[tid] = name
            kw["types"] = lookup[codes].tolist()
        else:
            if n and codes.min() < 1:
                raise ValueError(
                    "particle type id 0 or below; RMDA's types are 1-based "
                    "(use types='names' to code the types by name)"
                )
            kw["types"] = codes
            if names:
                kw["type_names"] = names

    if "Particle Identifier" in particles:
        kw["ids"] = np.asarray(particles["Particle Identifier"], dtype=np.int64)

    columns = {}
    for name in particles.keys():
        if name in _STRUCTURAL:
            continue
        arr = np.asarray(particles[name])
        if arr.dtype.kind in "biuf":
            columns[name] = arr.copy()
    if columns:
        kw["properties"] = columns

    return Frame.from_arrays(positions, **kw)


def _ovito_array(arr):
    """``arr`` in a dtype OVITO properties accept (no bool or unsigned)."""
    arr = np.asarray(arr)
    if arr.dtype.kind in "bu":
        return arr.astype(np.int64 if arr.dtype.itemsize >= 4 else np.int32)
    return arr


def _create_property(container, name, values):
    """A property of ``container`` named ``name`` holding ``values``, in
    their own dtype (OVITO narrows int64 data to int32 unless told)."""
    values = _ovito_array(values)
    if values.ndim > 2:
        # A 3x3 per row (a grain's fitted cell): OVITO properties are at most
        # (N, c), so row-major (N, 9).
        values = values.reshape(len(values), -1)
    if values.dtype.kind in "iuf":
        components = 1 if values.ndim == 1 else values.shape[1]
        try:
            return container.create_property(name, dtype=values.dtype, components=components, data=values)
        except ValueError:
            # A standard property ("Particle Identifier", "Velocity", ...)
            # has a fixed layout and refuses an explicit dtype or count.
            pass
    return container.create_property(name, data=values)


def _string_column(table, name, values):
    """A column of strings (a grain's ``status``) as an integer property
    with named types, the way OVITO stores particle types."""
    from ovito.data import ElementType

    labels = sorted(set(values))
    code = {s: i + 1 for i, s in enumerate(labels)}
    prop = table.create_property(name, dtype=np.int32, data=np.array([code[s] for s in values], dtype=np.int32))
    for s in labels:
        prop.types.append(ElementType(id=code[s], name=str(s)))


def _write_table(data, tname, columns):
    """``columns`` (``{name: array}``) as the data table ``rmda.<tname>``,
    replacing one of that identifier whole: OVITO hands back an existing
    table, whose columns and named types would otherwise mix with these."""
    from ovito.data import DataTable

    ident = f"rmda.{tname}"
    if ident in data.tables:
        del data.tables_[ident]
    table = data.tables.create(identifier=ident, title=tname, plot_mode=DataTable.PlotMode.NoPlot)
    for col, values in columns.items():
        arr = np.asarray(values)
        if arr.dtype.kind in "biuf":
            _create_property(table, col, arr)
        elif arr.dtype.kind in "OUS":
            _string_column(table, col, [str(v) for v in arr.tolist()])
    return table


def to_ovito(frame, data=None, *, result=None):
    """Write ``frame``'s results into an OVITO ``DataCollection``.

    With ``data`` (the collection a modifier is modifying, holding the same
    atoms in the same order), only results are added:

    - every attached per-atom property (``frame.properties()``: analysis
      outputs, and columns set with ``frame[name] = ...``) as a particle
      property of the same name;
    - every table in ``frame.tables`` (``defects.grains``, ``segment.pairs``,
      ...) as a ``DataTable`` with identifier ``rmda.<name>``; a string
      column (a grain's ``status``) becomes an integer column with named
      types;
    - with ``result`` (the dict an analysis returned), its numeric scalars as
      global attributes under their own names.

    Without ``data``, a new collection is made holding the frame itself --
    cell, positions, types (named), identifiers and input columns -- and then
    the results. Returns the collection.
    """
    if data is None:
        from ovito.data import DataCollection, ParticleType

        data = DataCollection()
        h = np.asarray(frame.cell, dtype=np.float64)
        m = np.zeros((3, 4))
        m[:, :3] = h.T
        m[:, 3] = frame.origin
        data.create_cell(m, pbc=tuple(frame.pbc))
        particles = data.create_particles(count=frame.natoms)
        particles.create_property("Position", data=np.asarray(frame.positions))
        tprop = particles.create_property("Particle Type", data=np.asarray(frame.types, dtype=np.int32))
        names = frame.type_names
        for t in sorted(set(np.unique(frame.types).tolist()) | set(range(1, len(names) + 1))):
            name = names[t - 1] if t - 1 < len(names) else ""
            tprop.types.append(ParticleType(id=t, name=name))
        _create_property(particles, "Particle Identifier", np.asarray(frame.ids, dtype=np.int64))
        for name in frame.columns:
            _create_property(particles, name, frame[name])
    else:
        if data.particles is None or data.particles.count != frame.natoms:
            have = 0 if data.particles is None else data.particles.count
            raise ValueError(
                f"the data collection holds {have} particles, the frame {frame.natoms}; "
                "results can only be written onto the same atoms"
            )

    particles = data.particles_
    for name in frame.properties():
        _create_property(particles, name, frame[name])

    for tname, columns in frame.tables.items():
        _write_table(data, tname, columns)

    if result is not None:
        for k, v in result.items():
            if isinstance(v, (bool, np.bool_)):
                data.attributes[k] = int(v)
            elif isinstance(v, (int, float, np.integer, np.floating)):
                data.attributes[k] = v.item() if hasattr(v, "item") else v
    return data
