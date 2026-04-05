"""High-level pipeline API for developers.

Provides a clean, reusable entry point that wraps the full motif_filter pipeline
without requiring CLI usage.

Example:
    from deepISA.motif_filter.pipeline.api import run_pipeline

    result = run_pipeline(
        fasta_dir="/path/to/hg38/chroms",
        output_dir="./results",
        regions_df=regions_df,
        motif_locs_df=motifs_df,
    )
"""
from __future__ import annotations
from pathlib import Path
from typing import Optional

import pandas as pd
import torch

from deepISA.motif_filter.pipeline.filter_pipeline import run_pipeline as _run_pipeline
from deepISA.motif_filter.utils.io import load_regions, load_motif_locs, save_filtered_motifs
from deepISA.modeling.cnn import Conv
from deepISA.utils.deepisa_guard import validate_deepisa_environment


def run_pipeline(
    fasta_dir: str,
    regions_df: pd.DataFrame | str,
    motif_locs_df: pd.DataFrame | str,
    model_path: str = "data/model_blympho.pt",
    output_dir: Optional[str] = None,
    seq_len: int = 600,
    window_size: int = 20,
    stride: int = 20,
    device: str = "cuda",
    n_regions: Optional[int] = None,
    save_filtered: bool = True,
    return_model: bool = False,
):
    """
    Run the motif_filter pipeline.

    Parameters
    ----------
    fasta_dir : str
        Path to per-chromosome FASTA directory (chr*.fa).
    regions_df : pd.DataFrame or str
        Regions DataFrame, or path to CSV. Must have chrom/start/end/region columns.
    motif_locs_df : pd.DataFrame or str
        Motif locations DataFrame, or path to CSV.
        Must have chrom/start/end/tf/score/strand/region columns.
    model_path : str
        Path to trained .pt model (default: data/model_blympho.pt).
    output_dir : str, optional
        Directory to write filtered motif CSV. Created if needed.
    seq_len : int
        Sequence length in bp (default: 600).
    window_size : int
        Non-motif window size in bp (default: 20).
    stride : int
        Non-motif window stride in bp (default: 20).
    device : str
        Compute device: 'cuda' or 'cpu' (default: cuda).
    n_regions : int, optional
        Limit to first N regions (default: all).
    save_filtered : bool
        Write filtered motifs to output_dir/motif_locs_filtered.csv (default: True).
    return_model : bool
        Return (result_df, model) instead of just result_df (default: False).

    Returns
    -------
    pd.DataFrame
        Filtered motifs DataFrame with attrs attached.
        If return_model=True, returns (result_df, model).
    """
    # ── Environment guard ──────────────────────────────────────
    validate_deepisa_environment()

    # ── Resolve paths ─────────────────────────────────────────
    if isinstance(regions_df, str):
        regions_df = load_regions(regions_df)
    if isinstance(motif_locs_df, str):
        motif_locs_df = load_motif_locs(motif_locs_df)

    model_path = Path(model_path)
    if not model_path.is_absolute():
        model_path = Path("data") / model_path

    # ── Load model ───────────────────────────────────────────
    state = torch.load(model_path, map_location=device, weights_only=False)
    model = Conv(seq_len=seq_len, ks=[15, 9, 9, 9, 9], cs=[64] * 5, ds=[1, 2, 4, 8, 16])
    model.load_state_dict(state)
    model.to(device).eval()

    # ── Run filter ────────────────────────────────────────────
    result = _run_pipeline(
        model=model,
        regions_df=regions_df,
        motif_locs_df=motif_locs_df,
        fasta_dir=fasta_dir,
        seq_len=seq_len,
        window_size=window_size,
        stride=stride,
        device=device,
        n_regions=n_regions,
    )

    # ── Save output ───────────────────────────────────────────
    if save_filtered and output_dir:
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "motif_locs_filtered.csv"
        save_filtered_motifs(result, out_path)

    if return_model:
        return result, model
    return result
