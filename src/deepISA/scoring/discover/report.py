"""tf-modisco-lite report generation (HTML + optional MEME/TOMTOM).

Thin orchestrator around the external ``modisco report`` CLI. Generates the
standard tf-modisco-lite HTML report (logos, seqlet tables, pattern hierarchy)
and, when a MEME motif database is supplied, runs TOMTOM to annotate discovered
motifs against known TF motifs.

CLI reference
-------------
```
modisco report -i <modisco_results.h5> -o <out_dir> -s <out_dir> [-m <meme_db.txt>]
```
https://github.com/jmschrei/tfmodisco-lite
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Optional

from loguru import logger

from deepISA.scoring.discover.modisco import resolve_cli, python_wrap

__all__ = ["run_motif_report", "cwm_to_meme"]

_MODISCO_INSTALL_HINT = (
    "tf-modisco-lite CLI not found on PATH.\n"
    "  Install:  pip install modisco-lite\n"
    "  or:       conda install -c bioconda modisco-lite\n"
    "  GitHub:   https://github.com/jmschrei/tfmodisco-lite"
)


def run_motif_report(
    modisco_h5: str,
    out_dir: str,
    meme_db: Optional[str] = None,
) -> str:
    """Invoke ``modisco report`` to build the HTML report (+ TOMTOM if MEME given).

    Parameters
    ----------
    modisco_h5 : str
        Path to a ``*_modisco_results.h5`` produced by
        :func:`deepISA.scoring.discover.modisco.run_modisco`.
    out_dir : str
        Destination directory for the report. Created if missing.
    meme_db : str, optional
        Path to a MEME-format motif database (e.g. JASPAR converted to MEME).
        When provided, tf-modisco-lite runs TOMTOM to annotate each discovered
        motif with its best known-TF match and includes the hits in the report.
        If ``None``, only the unannotated report is produced.

    Returns
    -------
    str
        The ``out_dir`` path on success.

    Raises
    ------
    RuntimeError
        If the ``modisco`` binary is not on ``PATH`` or exits non-zero.
    FileNotFoundError
        If ``modisco_h5`` (or ``meme_db`` when given) does not exist.
    """
    if not os.path.exists(modisco_h5):
        raise FileNotFoundError(f"modisco results H5 not found: {modisco_h5}")
    if meme_db is not None and not os.path.exists(meme_db):
        raise FileNotFoundError(f"MEME database not found: {meme_db}")

    modisco_bin = resolve_cli("modisco")
    if modisco_bin is None:
        raise RuntimeError(_MODISCO_INSTALL_HINT)

    Path(out_dir).mkdir(parents=True, exist_ok=True)
    # -o is the report output dir; -s is the image-path prefix used inside the
    # HTML. Following the README example we point both at out_dir.
    cmd = python_wrap(modisco_bin) + [
        "report",
        "-i", modisco_h5,
        "-o", out_dir,
        "-s", out_dir,
    ]
    if meme_db is not None:
        cmd.extend(["-m", meme_db])
        logger.info("TOMTOM annotation enabled via MEME database.")
    else:
        logger.info("No MEME database given; report will not include TOMTOM hits. "
                    "Pass meme_db=... to annotate discovered motifs against known TFs.")

    logger.info(f"Running tf-modisco-lite report: {' '.join(cmd)}")
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(
            f"tf-modisco-lite report failed (exit {proc.returncode}):\n{proc.stderr}"
        )
    logger.info(f"Report written to {out_dir}")
    return out_dir


# ---------------------------------------------------------------------------
# MEME export helper -- convert discovered CWMs to a MEME file (vectorized)
# ---------------------------------------------------------------------------
def cwm_to_meme(
    motifs: dict,
    out_path: str,
    alphabet: str = "ACGT",
    bg_freq: Optional[list] = None,
) -> str:
    """Write a MEME-format motif file from CWM-derived PPMs.

    Useful when you want to compare discovered motifs against an *external*
    MEME database with TOMTOM standalone (outside the tf-modisco-lite report),
    or to convert your own motif set into a database that ``run_motif_report``
    can then use via ``-m``.

    Parameters
    ----------
    motifs : dict
        ``{motif_id -> {"cwm": (L, 4) array, ...}}``, e.g. from
        :func:`deepISA.scoring.discover.h5_io.load_motifs`. The CWM is reduced
        to a PPM by clipping negatives to 0 and renormalizing per position.
    out_path : str
        Destination MEME file path.
    alphabet : str
        Alphabet string (default ``"ACGT"``).
    bg_freq : list of float, optional
        Background letter frequencies (length 4). Defaults to uniform 0.25.

    Returns
    -------
    str
        The ``out_path`` on success.
    """
    import numpy as np

    if bg_freq is None:
        bg_freq = [0.25] * len(alphabet)
    if len(bg_freq) != len(alphabet):
        raise ValueError(f"bg_freq length {len(bg_freq)} != alphabet length {len(alphabet)}")

    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
    lines = [
        "MEME version 5",
        f"ALPHABET= {alphabet}",
        f"Background letter frequencies\n{'  '.join(f'{a} {f:.4f}' for a, f in zip(alphabet, bg_freq))}",
        "",
    ]
    for motif_id, data in motifs.items():
        cwm = np.asarray(data["cwm"], dtype=np.float64)    # (L, 4)
        # Reduce CWM to PPM: clip negatives, renormalize, guard zero-rows.
        ppm = np.clip(cwm, 0, None)
        row_sums = ppm.sum(axis=1, keepdims=True)
        ppm = np.where(row_sums > 0, ppm / row_sums, 1.0 / len(alphabet))
        L = ppm.shape[0]
        lines.append(f"MOTIF {motif_id}")
        lines.append(f"letter-probability matrix: alength= {len(alphabet)} w= {L}")
        for row in ppm:
            lines.append("  " + "  ".join(f"{v:.4f}" for v in row))
        lines.append("")
    Path(out_path).write_text("\n".join(lines), encoding="utf-8")
    logger.info(f"Wrote {len(motifs)} motifs to MEME file: {out_path}")
    return out_path
