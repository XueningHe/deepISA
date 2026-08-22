"""Input-set curation for motif discovery.

Which regions are fed to tf-modisco-lite matters as much as the attribution
itself: seqlets sampled from low-activity regions reflect noise the model
never learned, and discovered "motifs" there are artifacts. The mc000
predecessor of this pipeline therefore only attributed sequences with
``y >= THRESH`` (high measured signal) before running tf-modisco-lite.

This module provides the two curation primitives that reproduce -- and
generalize -- that step:

* :func:`select_top_regions` -- rank regions by an activity score (measured
  signal or model prediction) and keep the top fraction (default 10%).
* :func:`drop_non_acgt_regions` -- drop regions whose sequence contains
  unknown bases (``N`` / IUPAC codes), i.e. assembly gaps.

Both are pure DataFrame filters; the QuickStart layer
(:meth:`deepISA.quickstart.QuickStart.run_modisco`) composes them and decides
where the activity score comes from. Any ``N`` that still slips through to
attribution is imputed by
:func:`deepISA.scoring.discover.attribution.compute_attribution` as a last
resort.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = ["select_top_regions", "drop_non_acgt_regions"]


def select_top_regions(
    df: pd.DataFrame,
    score,
    top_frac: float = 0.1,
) -> pd.DataFrame:
    """Keep the top ``top_frac`` regions ranked by ``score`` (descending).

    Parameters
    ----------
    df : pd.DataFrame
        Region table (one row per region). Column content is untouched.
    score : array-like or str
        Activity score per region. Either an array-like aligned with ``df``,
        or the name of a column in ``df`` (e.g. a measured-signal column).
    top_frac : float
        Fraction of regions to keep, in ``(0, 1]``. The count is rounded *up*
        so a requested 10% of 95 regions yields 10, never 9. At least one
        region is always kept.

    Returns
    -------
    pd.DataFrame
        The selected rows in descending-score order, ``reset_index``-ed, with
        the ranking score attached as a new ``activity_score`` column (so the
        choice is auditable in downstream logs/CSVs).
    """
    if not 0 < top_frac <= 1:
        raise ValueError(f"top_frac must be in (0, 1]; got {top_frac}")
    if isinstance(score, str):
        score = df[score].to_numpy(dtype=float)

    scores = np.asarray(score, dtype=float)
    if len(df) == 0:
        return df.copy()
    if scores.shape[0] != len(df):
        raise ValueError(
            f"score has {scores.shape[0]} entries but df has {len(df)} rows"
        )

    k = max(1, int(np.ceil(len(df) * top_frac)))
    # Stable sort: ties keep their original relative order.
    order = np.argsort(-scores, kind="stable")[:k]
    out = df.iloc[order].copy().reset_index(drop=True)
    out["activity_score"] = scores[order]
    return out


def drop_non_acgt_regions(
    df: pd.DataFrame,
    fasta_path: str,
) -> tuple[pd.DataFrame, int]:
    """Drop regions whose sequence contains any non-ACGT character.

    ``N`` blocks are almost always assembly gaps (centromeres/telomeres): they
    carry no regulatory signal, and the downstream stack (tangermeme,
    tf-modisco-lite, Fi-NeMo) requires strictly one-hot sequences. Dropping
    such regions up front keeps the discovery set fully real.

    Parameters
    ----------
    df : pd.DataFrame
        Region table with ``chrom`` / ``start`` / ``end`` columns.
    fasta_path : str
        Reference FASTA the regions are fetched from.

    Returns
    -------
    (kept_df, n_dropped) : tuple
        The ACGT-only rows (``reset_index``-ed) and how many were dropped.
        The function itself is silent; callers decide whether (and how) to
        apply the drop and do the logging -- see
        :meth:`deepISA.quickstart.QuickStart._select_modisco_regions`, which
        only drops while the fraction stays under a configurable cap and
        otherwise falls back to imputation.
    """
    from deepISA.utils import get_sequences_from_df, load_fasta

    if len(df) == 0:
        return df.copy(), 0

    seqs = pd.Series(get_sequences_from_df(df, load_fasta(fasta_path)))
    # get_sequences_from_df upper-cases; anything outside ACGT (N, R, Y, ...)
    # fails the fullmatch and the region is dropped.
    keep = seqs.str.fullmatch("[ACGT]+").fillna(False).to_numpy()
    n_dropped = int((~keep).sum())
    return df.loc[keep].copy().reset_index(drop=True), n_dropped
