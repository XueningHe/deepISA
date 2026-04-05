"""
Non-motif window generation for motif_filter.

Step 2 of the pipeline.
"""
import numpy as np
import pandas as pd
from typing import List, Tuple


def generate_nonmotif_windows(
    chrom: str,
    region_start: int,
    region_end: int,
    motif_df: pd.DataFrame,
    window_size: int = 20,
    stride: int = 20,
) -> List[Tuple[int, int]]:
    """
    Generate non-overlapping windows that do NOT overlap any motif.

    Parameters
    ----------
    chrom : str
        Chromosome name (e.g., "chr1").
    region_start : int
        Genomic start of the region.
    region_end : int
        Genomic end of the region.
    motif_df : pd.DataFrame
        Motifs in this region. Must have columns: start, end (genomic coords).
    window_size : int
        Window length in bp.
    stride : int
        Step size in bp (must equal window_size for no overlap).

    Returns
    -------
    windows : list of tuple (genomic_start, genomic_end)
        List of non-overlapping non-motif windows in genomic coordinates.
        Style: [start, end) — end is exclusive.
    """
    # Build list of (start, end) intervals for all motifs in this region
    motif_intervals = []
    for _, row in motif_df.iterrows():
        motif_intervals.append((int(row["start"]), int(row["end"])))

    windows = []
    pos = region_start
    while pos + window_size <= region_end:
        win_end = pos + window_size

        # Check if this window overlaps any motif
        overlaps = any(ms < win_end and me > pos for ms, me in motif_intervals)
        if not overlaps:
            windows.append((pos, win_end))

        pos += stride  # no overlap by construction if stride == window_size

    return windows


def windows_to_rel_coords(
    windows: List[Tuple[int, int]],
    region_start: int,
) -> List[Tuple[int, int]]:
    """
    Convert genomic [start, end) to relative coordinates within the region.

    Parameters
    ----------
    windows : list of (genomic_start, genomic_end)
    region_start : int
        Genomic start of the region.

    Returns
    -------
    rel_windows : list of (rel_start, rel_end) in [start, end) style
    """
    return [(s - region_start, e - region_start) for s, e in windows]
