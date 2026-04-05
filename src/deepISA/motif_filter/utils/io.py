"""I/O utilities for motif_filter."""
import pandas as pd
from pathlib import Path
from typing import Union


def load_motif_locs(path: Union[str, Path]) -> pd.DataFrame:
    """
    Load motif locations CSV.

    Expected columns: chrom, start, end, tf, score, strand, region
    """
    df = pd.read_csv(path)
    required = {"chrom", "start", "end", "tf", "region"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in motif_locs: {missing}")
    return df


def save_filtered_motifs(df: pd.DataFrame, path: Union[str, Path]) -> None:
    """
    Save filtered motif DataFrame.

    Columns: chrom, start, end, tf, s_motif, p_max, passed_sum, passed_peak, region
    """
    out = df[["chrom", "start", "end", "tf", "s_motif", "p_max",
              "passed_sum", "passed_peak", "region"]].copy()
    out.to_csv(path, index=False)


def load_regions(path: Union[str, Path]) -> pd.DataFrame:
    """Load regions CSV. Expected columns: chrom, start, end."""
    df = pd.read_csv(path)
    if "region" not in df.columns:
        df["region"] = (
            df["chrom"].astype(str) + ":" +
            df["start"].astype(str) + "-" +
            df["end"].astype(str)
        )
    return df


def validate_regions(regions_df: pd.DataFrame, seq_len: int = 600) -> None:
    """
    Validate that all regions have the expected length.

    Raises ValueError if any region length deviates from seq_len.
    """
    lengths = regions_df["end"] - regions_df["start"]
    bad = lengths[lengths != seq_len]
    if len(bad) > 0:
        raise ValueError(
            f"{len(bad)} regions have length != {seq_len}: "
            f"{bad.head(5).index.tolist()}"
        )
