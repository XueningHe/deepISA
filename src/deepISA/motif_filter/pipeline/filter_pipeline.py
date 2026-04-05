"""Non-motif null attribution filtering pipeline."""
import warnings
import pandas as pd
import numpy as np
import torch
from tqdm.auto import tqdm
from typing import Optional

from ..utils.fasta import FastaReader
from ..utils.onehot import encode_sequences
from ..utils.io import validate_regions
from ..core.attribution import compute_attribution, init_explainer
from ..core.window import generate_nonmotif_windows, windows_to_rel_coords
from ..core.scoring import compute_window_scores
from ..core.threshold import compute_thresholds, apply_filter


def run_pipeline(
    model: torch.nn.Module,
    regions_df: pd.DataFrame,
    motif_locs_df: pd.DataFrame,
    fasta_dir: str,
    seq_len: int = 600,
    window_size: int = 20,
    stride: int = 20,
    device: str = "cuda",
    n_regions: Optional[int] = None,
) -> pd.DataFrame:
    """
    Run the full non-motif null attribution filtering pipeline.

    Parameters
    ----------
    model : torch.nn.Module
        Trained deepISA Conv model.
    regions_df : pd.DataFrame
        Must have columns: chrom, start, end, region.
    motif_locs_df : pd.DataFrame
        Must have columns: chrom, start, end, tf, score, strand, region.
    fasta_dir : str
        Path to chromosome FASTA files (chr*.fa).
    seq_len : int
        Sequence length (default 600).
    window_size : int
        Non-motif window size in bp (default 20).
    stride : int
        Step size for non-motif windows (default 20, no overlap).
    device : str
        Compute device.
    n_regions : int, optional
        Limit to first n regions. If None, process all.

    Returns
    -------
    pd.DataFrame
        Filtered motifs with columns:
        chrom, start, end, tf, s_motif, p_max,
        passed_sum, passed_peak, region

    Metadata attached to returned DataFrame:
        T_sum, T_peak, n_null_windows, n_input_motifs,
        n_passed, n_regions_low_windows
    """
    # ── Input validation ────────────────────────────────────────────
    assert stride <= window_size, (
        f"stride ({stride}) must be <= window_size ({window_size}) "
        "to ensure non-overlapping windows"
    )
    validate_regions(regions_df, seq_len)

    if n_regions is not None:
        regions_df = regions_df.head(n_regions).reset_index(drop=True)

    # ── Init explainer once (avoids repeated GPU allocation) ──────────
    explainer, _ = init_explainer(model, device=device)
    fasta = FastaReader(fasta_dir)

    # Global collectors for null distributions
    s_null_list = []
    p_null_list = []
    motif_results = []
    low_window_regions = []   # regions with < 10 non-motif windows

    region_ids = set(regions_df["region"].values)
    motifs_sub = motif_locs_df[motif_locs_df["region"].isin(region_ids)].copy()

    MIN_WINDOWS = 10  # threshold for "low window count" warning

    for _, region_row in tqdm(regions_df.iterrows(), total=len(regions_df), desc="Regions"):
        region_id = region_row["region"]
        chrom = region_row["chrom"]
        region_start = int(region_row["start"])
        region_end = int(region_row["end"])

        # Step 1: fetch sequence
        seq = fasta.fetch(chrom, region_start, region_end)
        if seq is None or len(seq) < 50:
            continue
        seq = seq[:seq_len]

        # Step 2a: get non-motif windows
        region_motifs = motifs_sub[motifs_sub["region"] == region_id]
        nonmotif_windows_genomic = generate_nonmotif_windows(
            chrom, region_start, region_end, region_motifs,
            window_size=window_size, stride=stride,
        )
        nonmotif_windows_rel = windows_to_rel_coords(
            nonmotif_windows_genomic, region_start,
        )

        if len(nonmotif_windows_rel) < MIN_WINDOWS:
            low_window_regions.append(region_id)

        # Step 2b: get motif windows (relative coords)
        motif_windows_rel = []
        for _, m_row in region_motifs.iterrows():
            ms = max(0, int(m_row["start"]) - region_start)
            me = min(seq_len, int(m_row["end"]) - region_start)
            if ms < me:
                motif_windows_rel.append((ms, me, m_row))

        if not motif_windows_rel and not nonmotif_windows_rel:
            continue

        # Step 1 (DeepLIFT): encode + compute attribution (reuses cached explainer)
        ohe = encode_sequences([seq], target_len=seq_len)
        attr = compute_attribution(
            model, ohe, device=device, explainer=explainer
        )[0]  # (L, 4)

        # Step 3: score non-motif windows → null distribution
        for s, e in nonmotif_windows_rel:
            s_val, p_val = compute_window_scores(attr, s, e)
            s_null_list.append(s_val)
            p_null_list.append(p_val)

        # Step 3: score motif windows
        for ms, me, m_row in motif_windows_rel:
            s_val, p_val = compute_window_scores(attr, ms, me)
            motif_results.append({
                "chrom": chrom,
                "start": int(m_row["start"]),   # genomic coords
                "end": int(m_row["end"]),       # genomic coords
                "tf": m_row["tf"],
                "score": m_row.get("score", 0),
                "strand": m_row.get("strand", "+"),
                "region": region_id,
                "s_motif": s_val,
                "p_max": p_val,
            })

    # Step 4: compute thresholds
    assert len(s_null_list) > 0, (
        "No non-motif windows collected. "
        "Check --window-size and --stride parameters."
    )
    thresholds = compute_thresholds(s_null_list, p_null_list)
    T_sum = thresholds["T_sum"]
    T_peak = thresholds["T_peak"]

    # Step 5: apply filter
    motifs_df = pd.DataFrame(motif_results)
    motifs_filtered = apply_filter(motifs_df, T_sum, T_peak)
    passed = motifs_filtered[
        motifs_filtered["passed_sum"] | motifs_filtered["passed_peak"]
    ]

    n_input = len(motifs_df)
    n_passed = len(passed)
    pass_rate = n_passed / max(1, n_input)

    # Pass rate sanity check
    if not (0.01 < pass_rate < 0.80):
        warnings.warn(
            f"Pass rate {pass_rate:.2%} is outside expected range (1%–80%). "
            f"Check thresholds or data quality. "
            f"T_sum={T_sum:.4f}, T_peak={T_peak:.4f}, "
            f"n_null_windows={len(s_null_list)}, n_input={n_input}."
        )

    if low_window_regions:
        warnings.warn(
            f"{len(low_window_regions)}/{len(regions_df)} regions have "
            f"< {MIN_WINDOWS} non-motif windows. "
            f"Null distribution may be unstable for those regions. "
            f"First few: {low_window_regions[:3]}"
        )

    # Attach metadata
    passed.attrs["T_sum"] = T_sum
    passed.attrs["T_peak"] = T_peak
    passed.attrs["n_null_windows"] = len(s_null_list)
    passed.attrs["n_input_motifs"] = n_input
    passed.attrs["n_passed"] = n_passed
    passed.attrs["pass_rate"] = pass_rate
    passed.attrs["n_regions_low_windows"] = len(low_window_regions)

    return passed
