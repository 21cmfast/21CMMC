"""Workaround: make nautilus checkpoint writes atomic.

nautilus 1.0.6 rewrites its HDF5 checkpoint in place: ``Sampler.write`` unlinks the
old file and then writes a new one, and ``Sampler.write_shell_update`` edits the file
in ``'r+'`` mode. A kill during either leaves no usable checkpoint, and the run is
lost. This patch routes both writes through a temporary file that is fsynced and
then renamed over the checkpoint with ``os.replace``, which is atomic on POSIX, so a
kill at any moment leaves either the previous or the new checkpoint.

Call ``patch()`` once, before CosmoSIS (or anything else) builds a nautilus Sampler.
Remove once nautilus writes checkpoints atomically upstream.
"""

import os
import shutil
from pathlib import Path

from nautilus import sampler as _ns

_orig_write = _ns.Sampler.write
_orig_shell_update = _ns.Sampler.write_shell_update


def _tmp_path(path):
    # Keep the .hdf5/.h5 suffix: nautilus refuses any other extension.
    return path.with_name(f"{path.stem}.tmp{path.suffix}")


def _commit(tmp, path):
    with open(tmp, "rb") as f:
        os.fsync(f.fileno())
    os.replace(tmp, path)
    dir_fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(dir_fd)
    finally:
        os.close(dir_fd)


def _atomic_write(self, filepath, overwrite=False):
    path = Path(filepath)
    if path.exists() and not overwrite:
        raise RuntimeError(f"File {path} already exists.")
    tmp = _tmp_path(path)
    _orig_write(self, tmp, overwrite=True)
    _commit(tmp, path)


def _atomic_shell_update(self, filepath, shell):
    path = Path(filepath)
    tmp = _tmp_path(path)
    shutil.copyfile(path, tmp)
    _orig_shell_update(self, tmp, shell)
    _commit(tmp, path)


def patch():
    """Install the atomic writers on nautilus.Sampler (idempotent)."""
    _ns.Sampler.write = _atomic_write
    _ns.Sampler.write_shell_update = _atomic_shell_update
