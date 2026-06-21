"""Path resolution and small geometry helpers.

``resolve_path`` is ported verbatim (in spirit) from the source repo so the
package can reference its own bundled assets by relative path.
"""

from __future__ import annotations

import inspect
import os

import torch
from torch import Tensor


def resolve_path(fn: str) -> str:
    """Resolve a file relative to the root of the installed package.

    Mirrors the source repo's helper: paths are resolved relative to the package
    root (the parent of ``mlip_smoothness_eval/``), so bundled example assets can
    be referenced the same way regardless of the install location.
    """
    frame = inspect.currentframe()
    assert frame is not None
    this_file_path = inspect.getfile(frame)
    this_dir_path = os.path.abspath(os.path.dirname(this_file_path))
    return os.path.realpath(os.path.join(this_dir_path, "..", fn))


def frac_to_cart(pos: Tensor, lattice: Tensor) -> Tensor:
    """Fractional -> Cartesian for a row-vector lattice (cart = frac @ cell)."""
    return pos @ lattice


def cart_to_frac(pos: Tensor, lattice: Tensor) -> Tensor:
    """Cartesian -> fractional for a row-vector lattice."""
    typed_lattice = lattice.to(device=pos.device, dtype=pos.dtype)
    return pos @ torch.linalg.inv(typed_lattice)
