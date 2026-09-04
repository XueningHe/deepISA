"""tf-modisco-lite orchestration: NPZ input preparation + CLI invocation.

This is a thin Python orchestrator around the external ``modisco`` binary. The
heavy motif discovery runs in the CLI; we only build its inputs and parse its
outputs. The binary is *optional*: if it is not on ``PATH``, every function
still imports and the input-preparation step still runs -- only
:func:`run_modisco` raises a clear :class:`RuntimeError` with install hints.

Unlike the mc000 original, attributions here are already in deepISA's native
``(N, 4, L)`` layout (produced by :mod:`deepISA.scoring.discover.attribution`),
so no transpose is needed -- we only cast and trim.

CLI reference
-------------
``modisco motifs -s <ohe.npz> -a <hyp.npz> -n <n_seqlets> -w <window> -o <out.h5>``
https://github.com/jmschrei/tfmodisco-lite
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import h5py
import numpy as np
from loguru import logger

__all__ = ["prepare_modisco_input", "run_modisco"]

_MODISCO_INSTALL_HINT = (
    "tf-modisco-lite CLI not found on PATH.\n"
    "  Install:  pip install modisco-lite\n"
    "  or:       conda install -c bioconda modisco-lite\n"
    "  GitHub:   https://github.com/jmschrei/tfmodisco-lite"
)


def resolve_cli(name: str) -> str | None:
    """Cross-platform CLI resolver.

    On Windows, ``shutil.which`` only matches files whose extension is in
    ``PATHEXT`` (``.EXE``, ``.BAT``, ...), so console scripts installed by pip
    without an extension (e.g. ``modisco``) are missed. This falls back to:

    1. The ``Scripts`` (Windows) / ``bin`` (POSIX) directory next to the
       running Python interpreter -- the canonical install location for the
       current environment.
    2. A manual scan of ``PATH`` accepting extensionless scripts.

    making the same install work on Windows, Linux, and macOS without forcing
    the user to fiddle with PATH.
    """
    import sys
    found = shutil.which(name)
    if found:
        return found

    from os.path import join, isfile, dirname
    # 1. Environment-relative install dir (most reliable on Windows).
    exe_dir = dirname(sys.executable)
    candidates = [
        join(exe_dir, "Scripts", name),        # Windows venv/conda
        join(exe_dir, name),                    # POSIX bin, or Windows Scripts inPATH
        join(dirname(exe_dir), "Scripts", name),
    ]
    for candidate in candidates:
        if isfile(candidate):
            return candidate

    # 2. Manual PATH scan for an extensionless executable named `name`.
    for search_dir in os.environ.get("PATH", "").split(os.pathsep):
        if not search_dir:
            continue
        candidate = join(search_dir, name)
        if isfile(candidate):
            return candidate
    return None


def python_wrap(cli_path: str) -> list:
    """Build a subprocess argv that runs *cli_path* portably.

    Windows cannot directly execute extensionless Python console scripts
    (``WinError 193``); they must be launched via the interpreter. On POSIX
    systems the shebang handles this and we return the path unchanged.
    """
    import sys
    if sys.platform == "win32":
        _, ext = os.path.splitext(cli_path)
        # No extension or non-executable extension -> run via python.
        if ext.lower() not in (".exe", ".com", ".bat", ".cmd"):
            return [sys.executable, cli_path]
    return [cli_path]


def _decode_attr(val) -> str:
    """Decode an H5 attribute that may be ``str``, ``bytes``, or ``None``."""
    if val is None:
        return ""
    if isinstance(val, bytes):
        return val.decode("utf-8")
    return str(val)


def read_attribution_h5(
    h5_path: str,
    track_index: int = 0,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Load sequences + hypothetical scores for one track from an attribution H5.

    Accepts either the channels-first layout ``(T, N, 4, L)`` written by
    :mod:`deepISA.scoring.discover.attribution` (``layout=channels_first`` attr)
    or the legacy mc000 layout ``(N, L, 4)``. Returns ``(ohe, hyp)`` both in
    ``(N, 4, L)`` layout, plus the number of tracks ``T``.

    Parameters
    ----------
    h5_path : str
        Path to an attribution HDF5 file.
    track_index : int
        For channels-first multi-track files, which output track to extract.
        Averaging tracks silently produces a meaningless mix of regression and
        classification attributions; we therefore select a single track by
        index rather than averaging.
    """
    with h5py.File(h5_path, "r") as f:
        layout = _decode_attr(f.attrs.get("layout")) if "layout" in f.attrs else ""
        seqs = f["sequences"][:]
        hyp = f["hyp_scores"][:]

    if layout == "channels_first":
        # (T, N, 4, L): select the requested track (NOT average across tracks).
        T = hyp.shape[0]
        if track_index < 0 or track_index >= T:
            raise IndexError(
                f"track_index={track_index} out of range for H5 with {T} track(s)."
            )
        hyp = hyp[track_index]                # (N, 4, L)
        if seqs.ndim == 3 and seqs.shape[1] != 4 and seqs.shape[2] == 4:
            seqs = np.transpose(seqs, (0, 2, 1))
        return seqs.astype(np.float32), hyp.astype(np.float32), T

    # Legacy mc000 layout: (N, L, 4) -> (N, 4, L)
    T = 1
    if seqs.ndim == 3 and seqs.shape[2] == 4:
        seqs = np.transpose(seqs, (0, 2, 1))
    if hyp.ndim == 3 and hyp.shape[2] == 4:
        hyp = np.transpose(hyp, (0, 2, 1))
    return seqs.astype(np.float32), hyp.astype(np.float32), T


def prepare_modisco_input(
    h5_path: str,
    out_dir: str,
    track_index: int = 0,
) -> tuple[str, str]:
    """Build tf-modisco-lite NPZ inputs from an attribution HDF5 file.

    The full sequence length is preserved -- tf-modisco-lite's ``-w`` flag
    (window size around the center of each region) is a *separate* concern and
    is passed at :func:`run_modisco` time. We only enforce the even-length
    requirement tf-modisco-lite imposes: an odd trailing position is trimmed by
    one (this is a length-parity fix, not a content trim).

    Parameters
    ----------
    h5_path : str
        Attribution H5 produced by
        :func:`deepISA.scoring.discover.attribution.compute_attribution`
        (or compatible mc000 schema).
    out_dir : str
        Directory in which to write ``ohe.npz`` and ``hyp.npz``.
    track_index : int
        For channels-first multi-track attribution files, which output track to
        extract for motif discovery. Defaults to ``0``.

    Returns
    -------
    (ohe_npz, hyp_npz) : tuple of str
        Paths to the one-hot and hypothetical-score NPZ files.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ohe_npz = out_dir / "ohe.npz"
    hyp_npz = out_dir / "hyp.npz"

    seqs, hyp, T = read_attribution_h5(h5_path, track_index=track_index)  # both (N, 4, L)
    # tf-modisco-lite requires even sequence length; trim a single trailing
    # position if needed. We do NOT trim to a fixed window here.
    if seqs.shape[2] % 2 != 0:
        seqs = seqs[:, :, :-1]
        hyp = hyp[:, :, :-1]

    seqs = seqs.astype(np.int8)
    hyp = hyp.astype(np.float32)

    np.savez_compressed(ohe_npz, seqs)
    np.savez_compressed(hyp_npz, hyp)
    logger.info(f"modisco inputs ready: ohe={seqs.shape} hyp={hyp.shape} (track={track_index}/{T})")
    return str(ohe_npz), str(hyp_npz)


def run_modisco(
    ohe_npz: str,
    hyp_npz: str,
    out_h5: str,
    n_seqlets: int = 50000,
    window: int | None = None,
) -> str:
    """Invoke the ``modisco motifs`` CLI to discover motifs.

    Parameters
    ----------
    ohe_npz, hyp_npz : str
        NPZ paths produced by :func:`prepare_modisco_input`.
    out_h5 : str
        Destination HDF5 path for ``*_modisco_results.h5``.
    n_seqlets : int
        Maximum number of seqlets to process (``-n``).
    window : int, optional
        tf-modisco-lite ``-w``: the window size around the center of each input
        region that is used for seqlet discovery. If ``None`` (default), it is
        set to the NPZ sequence length so the *entire* region is considered.
        Pass a smaller value only if you deliberately want to restrict analysis
        to the central portion of each region.

    Returns
    -------
    str
        The ``out_h5`` path on success.

    Raises
    ------
    RuntimeError
        If the ``modisco`` binary is not on ``PATH`` or exits non-zero.
    """
    modisco_bin = resolve_cli("modisco")
    if modisco_bin is None:
        raise RuntimeError(_MODISCO_INSTALL_HINT)

    # Infer sequence length from the NPZ so -w defaults to "use everything".
    if window is None:
        with np.load(ohe_npz) as dz:
            window = int(dz[dz.files[0]].shape[-1])
    if window % 2 != 0:
        raise ValueError(f"window must be even for tf-modisco-lite; got {window}")

    os.makedirs(os.path.dirname(os.path.abspath(out_h5)) or ".", exist_ok=True)
    cmd = python_wrap(modisco_bin) + [
        "motifs",
        "-s", ohe_npz,
        "-a", hyp_npz,
        "-n", str(n_seqlets),
        "-w", str(window),
        "-o", out_h5,
    ]
    logger.info(f"Running tf-modisco-lite: {' '.join(cmd)}")
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(
            f"tf-modisco-lite failed (exit {proc.returncode}):\n{proc.stderr}"
        )
    logger.info(f"tf-modisco-lite results: {out_h5}")
    return out_h5
