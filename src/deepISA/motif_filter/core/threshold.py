"""
Threshold computation for motif_filter.

Step 4 of the pipeline.
"""
import numpy as np
import pandas as pd


def compute_thresholds(
    s_null: list,
    p_null: list,
    percentile_sum: int = 95,
    percentile_peak: int = 80,
) -> dict:
    """
    Compute energy and peak thresholds from null distributions.

    Parameters
    ----------
    s_null : list of float
        s_motif values from non-motif windows.
    p_null : list of float
        p_max values from non-motif windows.
    percentile_sum : int
        Percentile for energy gate (default 95).
    percentile_peak : int
        Percentile for peak gate (default 80).

    Returns
    -------
    dict
        {"T_sum": float, "T_peak": float}
    """
    s_arr = np.array(s_null, dtype=np.float32)
    p_arr = np.array(p_null, dtype=np.float32)

    return {
        "T_sum": float(np.percentile(s_arr, percentile_sum)),
        "T_peak": float(np.percentile(p_arr, percentile_peak)),
    }


def apply_filter(
    motifs_df: pd.DataFrame,
    T_sum: float,
    T_peak: float,
) -> pd.DataFrame:
    """
    Apply dual-gate OR filter to motifs.

    Parameters
    ----------
    motifs_df : pd.DataFrame
        Must have columns: s_motif, p_max.
    T_sum : float
        Energy gate threshold.
    T_peak : float
        Peak gate threshold.

    Returns
    -------
    pd.DataFrame
        DataFrame with added passed_sum and passed_peak columns.
    """
    df = motifs_df.copy()
    df["passed_sum"] = df["s_motif"] > T_sum
    df["passed_peak"] = df["p_max"] > T_peak
    return df
