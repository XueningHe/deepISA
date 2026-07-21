"""Read discovered motifs back from tf-modisco-lite / Fi-NeMo H5 files.

The recursive group walk over HDF5 is an unavoidable *structural* loop (the
data is a tree), but every per-motif resize/center-crop is fully vectorized.

Adapted from mc000 ``h5_io.py`` with the layout convention fixed to deepISA's
native ``(L, 4)`` per-motif arrays (no transpose needed downstream).
"""

from __future__ import annotations

import hashlib
import os
import re
from typing import Optional

import h5py
import numpy as np

__all__ = ["parse_motif_name", "extract_motifs_from_group", "load_motifs"]

_BG_FILL = 0.25   # uniform background for padding 'seq' rows
# Anchored regex for the dedup suffix "_v{n}" appended by extract_motifs_from_group.
# Using a substring check like "_v" in name would wrongly drop motifs whose
# legitimate name happens to contain that substring.
_DEDUP_SUFFIX_RE = re.compile(r"_v\d+$")


def parse_motif_name(path_str: str, task_name: str) -> str:
    """Render an H5 node path as a human-readable motif id.

    ``pattern_N / subpattern_N / subcluster_N`` becomes
    ``{task}_pN_subN`` (or ``{task}_pN_main`` when no sub-level is present).
    """
    p_name, sub_name = "", ""
    for part in path_str.split("/"):
        if part.startswith("pattern_"):
            p_name = part.replace("pattern_", "p")
        elif part.startswith("subpattern_") or part.startswith("subcluster_"):
            sub_name = (
                part.replace("subpattern_", "sub").replace("subcluster_", "sub")
            )
    if sub_name:
        return f"{task_name}_{p_name}_{sub_name}"
    if p_name:
        return f"{task_name}_{p_name}_main"
    # Deterministic fallback hash (built-in hash() is randomized per process).
    digest = hashlib.md5(path_str.encode("utf-8")).hexdigest()[:6]
    return f"{task_name}_x{int(digest, 16) % 100000}"


def _read_cwm_and_seq(node: h5py.Group) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Extract ``(cwm, seq)`` from a pattern group, handling nested layouts.

    Supports both the direct dataset layout (``contrib_scores`` /
    ``sequence`` as datasets) and the older ``contrib_scores_contrib_scores``
    nested ``fwd`` layout.
    """
    keys = list(node.keys())
    cwm = seq = None
    if "contrib_scores" in keys and "sequence" in keys:
        if isinstance(node["contrib_scores"], h5py.Dataset):
            cwm = node["contrib_scores"][:]
            seq = node["sequence"][:]
    nested_cwm = [k for k in keys if k.endswith("contrib_scores_contrib_scores")]
    if nested_cwm and "sequence" in keys and cwm is None:
        cwm_grp = node[nested_cwm[0]]
        seq_grp = node["sequence"]
        # "fwd" membership test is only valid on Groups; a flat Dataset raises
        # TypeError, so guard both sides before descending.
        seq_is_group = isinstance(seq_grp, h5py.Group)
        if "fwd" in cwm_grp and seq_is_group and "fwd" in seq_grp:
            cwm = cwm_grp["fwd"][:]
            seq = seq_grp["fwd"][:]
    return cwm, seq


def _read_num_seqlets(node: h5py.Group) -> int:
    """Best-effort seqlet count extraction across modisco-lite H5 variants."""
    if "seqlets" in node:
        sn = node["seqlets"]
        if isinstance(sn, h5py.Group):
            if "n_seqlets" in sn:
                val = np.array(sn["n_seqlets"])
                return int(val[0]) if val.ndim > 0 else int(val)
            if "start" in sn:
                return len(sn["start"])
        elif isinstance(sn, h5py.Dataset):
            return len(sn)
    if "seqlets_and_alnmts" in node:
        saa = node["seqlets_and_alnmts"]
        if "seqlets" in saa:
            return len(saa["seqlets"])
    return 0


def extract_motifs_from_group(
    node, task_name: str, results: Optional[dict] = None
) -> dict:
    """Recursively collect subcluster-level motifs from an H5 group.

    Skips ``pattern_merge_hierarchy`` nodes and resolves duplicate names with
    ``_v2`` / ``_v3`` suffixes. Each result entry is::

        {"cwm": (L,4), "seq": (L,4), "num_seqlets": int, "raw_path": str}

    Parameters
    ----------
    node : h5py.Group
        Root group of a tf-modisco-lite results H5 (typically the file handle).
    task_name : str
        Prefix used to build readable motif ids.
    results : dict, optional
        Accumulator (used by the recursion).
    """
    if results is None:
        results = {}
    if not isinstance(node, h5py.Group):
        return results
    if "pattern_merge_hierarchy" in node.name:
        return results

    cwm, seq = _read_cwm_and_seq(node)
    if cwm is not None and seq is not None and cwm.ndim == 2 and cwm.shape[1] == 4:
        nice_name = parse_motif_name(node.name, task_name)
        base, counter = nice_name, 1
        while nice_name in results:
            nice_name = f"{base}_v{counter}"
            counter += 1
        results[nice_name] = {
            "cwm": cwm,
            "seq": seq,
            "num_seqlets": _read_num_seqlets(node),
            "raw_path": node.name,
        }

    for key in node.keys():
        if key in ("seqlets", "seqlets_and_alnmts"):
            continue
        extract_motifs_from_group(node[key], task_name, results)
    return results


def _center_crop_or_pad(arr: np.ndarray, target_len: int, fill: float) -> np.ndarray:
    """Center-crop a ``(L, 4)`` array to ``target_len`` or zero/uniform-pad.

    Crop center is the position of max total |contribution|, vectorized via
    :func:`numpy.sum` + :func:`numpy.argmax`. Padding uses symmetric edges and
    falls back to right-alignment when the right edge would overflow.
    """
    L = arr.shape[0]
    if L == target_len:
        return arr
    if L > target_len:
        center = np.argmax(np.abs(arr).sum(axis=1))
        start = int(np.clip(center - target_len // 2, 0, L - target_len))
        return arr[start:start + target_len]
    pad_before = (target_len - L) // 2
    pad_after = target_len - L - pad_before
    return np.pad(
        arr,
        ((pad_before, pad_after), (0, 0)),
        mode="constant",
        constant_values=fill,
    )


def load_motifs(
    h5_path: str,
    task_name: str,
    target_len: int = 40,
    skip_main: bool = True,
) -> dict:
    """Load and length-normalize motifs from a tf-modisco-lite results H5.

    Parameters
    ----------
    h5_path : str
        Path to a ``*_modisco_results.h5`` file.
    task_name : str
        Prefix for motif ids (see :func:`parse_motif_name`).
    target_len : int
        Every motif is center-cropped or padded to this length. Defaults to 40.
    skip_main : bool
        If True (default), drop ``_main`` / ``_v*`` entries and keep only
        subcluster-level motifs, matching the mc000 convention.

    Returns
    -------
    dict
        ``{motif_id -> {"cwm": (target_len,4), "seq": (target_len,4),
        "num_seqlets": int, "task": str}}``.
    """
    if not os.path.exists(h5_path):
        raise FileNotFoundError(h5_path)

    with h5py.File(h5_path, "r") as f:
        raw = extract_motifs_from_group(f, task_name)

    out = {}
    for name, data in raw.items():
        # skip_main drops "pattern-level" entries (the _main suffix from
        # parse_motif_name) and the dedup suffix _v{n} appended when two
        # subclusters collide on the same name. Use an ANCHORED regex so a
        # legitimate motif name containing "_v" (e.g. a TF label) is not dropped.
        if skip_main and (name.endswith("_main") or _DEDUP_SUFFIX_RE.search(name)):
            continue
        cwm = _center_crop_or_pad(data["cwm"].astype(np.float32), target_len, 0.0)
        seq = _center_crop_or_pad(data["seq"].astype(np.float32), target_len, _BG_FILL)
        out[name] = {
            "cwm": cwm,
            "seq": seq,
            "num_seqlets": data["num_seqlets"],
            "task": task_name,
        }
    return out
