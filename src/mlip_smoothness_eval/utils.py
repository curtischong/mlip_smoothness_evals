"""Small shared helpers."""

from __future__ import annotations

import inspect
import os


def resolve_path(fn: str) -> str:
    """Resolve a file relative to the root of the codebase."""
    frame = inspect.currentframe()
    assert frame is not None
    this_file_path = inspect.getfile(frame)
    this_dir_path = os.path.abspath(os.path.dirname(this_file_path))
    return os.path.realpath(os.path.join(this_dir_path, "../..", fn))
