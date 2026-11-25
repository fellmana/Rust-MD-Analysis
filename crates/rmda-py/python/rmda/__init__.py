"""RMDA (Rust MD Analysis): structural analysis of atomistic molecular-dynamics
frames — CNA, PTM, grain segmentation, per-grain Wigner–Seitz defects, voids,
clusters, RDF and Warren–Cowley order — in Rust, from Python.

The functions and classes live in the compiled extension ``rmda._rmda`` and are
re-exported here. ``rmda`` on the command line is the same tool as the Rust
binary: ``rmda --help``.
"""

from . import _rmda
from ._rmda import *  # noqa: F401,F403
from ._rmda import __version__  # noqa: F401

__all__ = [name for name in dir(_rmda) if not name.startswith("_")] + ["__version__"]


def _cli() -> None:
    """Entry point of the ``rmda`` console script: runs the command line in
    this process and exits with its status."""
    import sys

    _rmda._cli_main(sys.argv)
