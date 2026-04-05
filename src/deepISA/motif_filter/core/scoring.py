"""
Scoring functions for motif_filter.

Step 3 of the pipeline.
Computes s_motif and p_max for each window.
"""
import numpy as np
from typing import Tuple


def compute_window_scores(attr: np.ndarray, start: int, end: int) -> Tuple[float, float]:
    """
    Compute s_motif and p_max for a window from attribution scores.

    Parameters
    ----------
    attr : np.ndarray, shape (length, 4)
        Attribution scores for one sequence.
    start : int
        Window start (relative to sequence, 0-based).
    end : int
        Window end (relative to sequence, exclusive).

    Returns
    -------
    s_motif : float
        Total energy: sum of |attr| over window positions and channels.
    p_max : float
        Peak: max of |attr| across window positions and channels.
    """
    window_attr = attr[start:end, :]  # (window_len, 4)
    abs_attr = np.abs(window_attr)     # (window_len, 4)

    # Sum over channels (ACGT) first, then over positions
    per_pos = abs_attr.sum(axis=1)    # (window_len,)
    s_motif = float(per_pos.sum())
    p_max = float(per_pos.max())

    return s_motif, p_max


def score_windows(attr: np.ndarray, windows: list) -> list:
    """
    Score a list of windows.

    Parameters
    ----------
    attr : np.ndarray, shape (L, 4)
        Attribution scores for one sequence.
    windows : list of (start, end)
        List of windows in relative coordinates [start, end).

    Returns
    -------
    scores : list of (s_motif, p_max)
    """
    return [compute_window_scores(attr, s, e) for s, e in windows]
