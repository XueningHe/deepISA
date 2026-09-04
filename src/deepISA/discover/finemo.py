"""Fi-NeMo orchestration: NPZ input, CLI scanning, hits loading, motif DB build.

Like :mod:`deepISA.scoring.discover.modisco`, this is a thin orchestrator around
the external ``finemo`` binary. Inputs are built and outputs parsed in Python;
the actual hit-calling happens in the CLI. If ``finemo`` is not on ``PATH`` the
input-building and hits-parsing functions still work -- only
:func:`run_finemo_scan` raises a clear :class:`RuntimeError`.

Vectorization notes (vs. the mc000 original)
--------------------------------------------
* ``regions.bed`` is built with one pandas ``to_csv`` call instead of a row loop.
* Hit annotation mapping uses a single :meth:`~pandas.DataFrame.merge`
  instead of per-row ``apply(lambda)``.
* H5 motif-DB attrs are read with a dict comprehension over ``h5py`` groups.

CLI reference
-------------
``finemo call-hits -r <npz> -m <motif_db.h5> -o <out_dir> -l <lambda> --max-steps N``
https://github.com/kundajelab/Fi-NeMo
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Optional

import h5py
import numpy as np
import pandas as pd
from loguru import logger

__all__ = [
    "build_finemo_input",
    "run_finemo_scan",
    "load_hits_with_annotation",
    "build_finemo_db",
]

_FINEMO_INSTALL_HINT = (
    "Fi-NeMo CLI not found on PATH.\n"
    "  Install:  pip install finemo\n"
    "  GitHub:   https://github.com/kundajelab/Fi-NeMo"
)


# ---------------------------------------------------------------------------
# 1. Build finemo input NPZ + regions BED
# ---------------------------------------------------------------------------
def build_finemo_input(
    seqs_ohe: np.ndarray,
    hyp_scores: np.ndarray,
    out_dir: str,
    ids: Optional[np.ndarray] = None,
) -> str:
    """Write a Fi-NeMo input NPZ and ``regions.bed`` from attribution arrays.

    Parameters
    ----------
    seqs_ohe, hyp_scores : np.ndarray
        One-hot sequences and hypothetical attribution scores. Both accept
        either deepISA's native ``(N, 4, L)`` layout or ``(N, L, 4)``; the
        channel axis is auto-detected. They are normalized to ``(N, 4, L)``
        (no transpose) and cast to ``int8`` / ``float32``. An odd trailing
        position is trimmed to satisfy Fi-NeMo's even-length requirement.
    out_dir : str
        Output directory; ``finemo_input.npz`` and ``regions.bed`` are written
        inside it.
    ids : np.ndarray of str, optional
        Per-sequence identifiers written to the BED ``name`` column. If
        omitted, sequential ``seq_0, seq_1, ...`` ids are generated.

    Returns
    -------
    str
        Path to the written ``finemo_input.npz``.
    """
    seq_arr = _to_channels_first(seqs_ohe).astype(np.int8)
    attr_arr = _to_channels_first(hyp_scores).astype(np.float32)

    # Fi-NeMo requires even sequence length.
    if seq_arr.shape[2] % 2 != 0:
        seq_arr = seq_arr[:, :, :-1]
        attr_arr = attr_arr[:, :, :-1]
    N, _, L = seq_arr.shape

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Vectorized BED: build all rows at once, single to_csv.
    if ids is None:
        names = [f"seq_{i}" for i in range(N)]
    else:
        names = np.asarray(ids).astype(str).tolist()
    bed = pd.DataFrame({
        "chrom": ["chrFake"] * N,
        "start": np.zeros(N, dtype=np.int64),
        "end": np.full(N, L, dtype=np.int64),
        "name": names,
        "score": np.zeros(N, dtype=np.int64),
        "strand": ["+"] * N,
        "a": np.zeros(N, dtype=np.int64),
        "b": np.zeros(N, dtype=np.int64),
        "c": np.zeros(N, dtype=np.int64),
        "summit": np.full(N, L // 2, dtype=np.int64),
    })
    bed.to_csv(out_dir / "regions.bed", sep="\t", index=False, header=False)

    npz_path = out_dir / "finemo_input.npz"
    np.savez(npz_path, sequences=seq_arr, contributions=attr_arr)
    logger.info(f"Fi-NeMo input: seq={seq_arr.shape} attr={attr_arr.shape} -> {npz_path}")
    return str(npz_path)


def _to_channels_first(arr: np.ndarray) -> np.ndarray:
    """Coerce one-hot / attribution array to ``(N, 4, L)``."""
    if arr.ndim != 3 or 4 not in arr.shape:
        raise ValueError(f"expected (N,4,L) or (N,L,4); got {arr.shape}")
    if arr.shape[1] == 4 and arr.shape[2] != 4:
        return np.ascontiguousarray(arr)
    if arr.shape[2] == 4:
        return np.ascontiguousarray(np.transpose(arr, (0, 2, 1)))
    raise ValueError(f"ambiguous one-hot shape {arr.shape}; expected a 4-channel axis")


# ---------------------------------------------------------------------------
# 2. Run finemo call-hits CLI
# ---------------------------------------------------------------------------
def run_finemo_scan(
    npz_path: str,
    out_dir: str,
    motif_db_h5: str,
    lam: float = 0.7,
    max_steps: int = 10000,
) -> str:
    """Invoke ``finemo call-hits`` and return the path to ``hits.tsv``.

    Parameters
    ----------
    npz_path : str
        NPZ produced by :func:`build_finemo_input`.
    out_dir : str
        Output directory for Fi-NeMo results.
    motif_db_h5 : str
        Path to the motif database H5 (a tf-modisco-lite results file, or one
        built by :func:`build_finemo_db`).
    lam : float
        Lambda trade-off parameter (``-l``).
    max_steps : int
        Maximum optimization steps (``--max-steps``).

    Returns
    -------
    str
        Path to the resulting ``hits.tsv``.

    Raises
    ------
    RuntimeError
        If the ``finemo`` binary is not on ``PATH`` or exits non-zero.
    """
    from deepISA.scoring.discover.modisco import resolve_cli, python_wrap
    finemo_bin = resolve_cli("finemo")
    if finemo_bin is None:
        raise RuntimeError(_FINEMO_INSTALL_HINT)

    Path(out_dir).mkdir(parents=True, exist_ok=True)
    cmd = python_wrap(finemo_bin) + [
        "call-hits",
        "-r", npz_path,
        "-m", motif_db_h5,
        "-o", out_dir,
        "-l", str(lam),
        "--max-steps", str(max_steps),
    ]
    logger.info(f"Running Fi-NeMo: {' '.join(cmd)}")
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"Fi-NeMo failed (exit {proc.returncode}):\n{proc.stderr}")

    hits_tsv = str(Path(out_dir) / "hits.tsv")
    logger.info(f"Fi-NeMo hits: {hits_tsv}")
    return hits_tsv


# ---------------------------------------------------------------------------
# 3. Load hits with motif annotations (vectorized merge)
# ---------------------------------------------------------------------------
def _decode_attr(val) -> str:
    if val is None:
        return ""
    if isinstance(val, bytes):
        return val.decode("utf-8")
    return str(val)


def load_hits_with_annotation(hits_tsv_path: str, motif_db_h5: str) -> pd.DataFrame:
    """Load ``hits.tsv`` and join motif ``mc_id`` / ``tf_label`` annotations.

    The annotation table is read from the motif DB H5 in one vectorized pass
    (a dict comprehension over ``pos_patterns`` groups), then merged into the
    hits DataFrame with a single :meth:`~pandas.DataFrame.merge` -- no per-row
    ``apply``.

    Parameters
    ----------
    hits_tsv_path : str
        Path to the ``hits.tsv`` written by :func:`run_finemo_scan`.
    motif_db_h5 : str
        Path to the motif database H5 (same one passed to the CLI).

    Returns
    -------
    pd.DataFrame
        Hits with added ``MC_ID`` and ``TF_Name`` columns. Motifs missing from
        the DB fall back to the raw ``motif_name`` / ``"Unknown"``.
    """
    df = pd.read_csv(hits_tsv_path, sep="\t")

    with h5py.File(motif_db_h5, "r") as f:
        pos = f.get("pos_patterns")
        if pos is None:
            raise KeyError(f"'pos_patterns' group not found in {motif_db_h5}")
        records = [
            {
                "motif_name": f"pos_patterns.{name}",
                "MC_ID": _decode_attr(grp.attrs.get("mc_id")) or name,
                "TF_Name": _decode_attr(grp.attrs.get("tf_label")) or "Unknown",
            }
            for name, grp in pos.items()
        ]
    anno = pd.DataFrame.from_records(records)

    # Some finemo versions emit motif_name WITHOUT the "pos_patterns." prefix.
    # Build a second key column so the merge matches either form, then prefer
    # the prefixed match by joining on it first.
    df["_bare"] = df["motif_name"].str.removeprefix("pos_patterns.")
    anno["_bare"] = anno["motif_name"].str.removeprefix("pos_patterns.")
    merged = df.merge(anno.drop(columns="motif_name"), on="_bare", how="left")
    merged = merged.drop(columns="_bare")

    # Fill NaNs from unmatched motifs with safe per-column scalars. Avoid
    # fillna(Series) -- it aligns by index and silently misaligns on non-default
    # DataFrame indices.
    merged["MC_ID"] = merged["MC_ID"].fillna(merged["motif_name"])
    merged["TF_Name"] = merged["TF_Name"].fillna("Unknown")
    return merged


# ---------------------------------------------------------------------------
# 4. Build a Fi-NeMo-compatible motif DB H5
# ---------------------------------------------------------------------------
def build_finemo_db(
    motifs: dict,
    out_path: str,
    annotations: Optional[dict] = None,
    task_filter: Optional[str] = None,
) -> str:
    """Write a Fi-NeMo-compatible motif database H5 from CWM/sequence arrays.

    Parameters
    ----------
    motifs : dict
        Mapping ``motif_id -> {"cwm": (L,4) array, "seq": (L,4) array}``, e.g.
        produced by :func:`deepISA.scoring.discover.h5_io.load_motifs`.
    out_path : str
        Destination H5 path.
    annotations : dict, optional
        Mapping ``motif_id -> TF label``. Missing ids default to ``"Unknown"``.
    task_filter : str, optional
        If given, only ``motif_id`` starting with this prefix are written.

    Returns
    -------
    str
        The ``out_path`` on success.
    """
    annotations = annotations or {}
    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
    if task_filter:
        items = [(k, v) for k, v in motifs.items() if k.startswith(task_filter)]
    else:
        items = list(motifs.items())

    with h5py.File(out_path, "w") as f:
        pos_grp = f.create_group("pos_patterns")
        for idx, (motif_id, data) in enumerate(items):
            cwm = np.asarray(data["cwm"], dtype=np.float32)
            seq = np.asarray(data["seq"], dtype=np.float32)
            grp = pos_grp.create_group(f"pattern_{idx}")
            grp.create_dataset("contrib_scores", data=cwm)
            grp.create_dataset("sequence", data=seq)
            grp.attrs["original_id"] = motif_id
            grp.attrs["mc_id"] = motif_id
            grp.attrs["tf_label"] = annotations.get(motif_id, "Unknown")

    logger.info(f"Wrote {len(items)} motifs to Fi-NeMo DB: {out_path}")
    return out_path
