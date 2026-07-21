"""DeepLIFT-based motif filtering.

This module was specified by ``tests/test_filter.py`` (555 lines) but its
implementation was never committed. It provides *per-position* DeepLIFT
attribution to **filter** JASPAR-mapped motifs: a motif is kept only if its
second-highest per-position attribution clears a percentile threshold derived
from the non-motif (background) attribution distribution.

This is conceptually orthogonal to :mod:`deepISA.scoring.discover`, which does
*de novo* motif discovery via tf-modisco-lite / Fi-NeMo. Both share the
``captum`` attribution backbone; here we use the single-reference
:class:`captum.attr.DeepLift` (zero baseline) per the test contract, whereas
discovery uses :class:`captum.attr.DeepLiftShap` with dinucleotide backgrounds.

Contract (see ``tests/test_filter.py``)
---------------------------------------
- ``scan_deeplift_scores`` -> ``dict[region_str, np.ndarray (num_tracks, seq_len)]``
- per-region score is ``abs(attr).sum(dim=1)`` summed over the 4 bases
- ``attr_filter`` annotates each motif with ``second_max_t{t}`` and
  ``pass_threshold_t{t}`` columns; a motif passes if it clears the threshold on
  *any* track.
"""

from __future__ import annotations

from typing import Dict, Iterator

import bioframe as bf
import numpy as np
import pandas as pd
import torch
from captum.attr import DeepLift
from loguru import logger

from deepISA.utils import one_hot_encode, load_fasta

__all__ = [
    "extract_regions",
    "scan_deeplift_scores",
    "get_slices",
    "_get_second_max",
    "get_attr_threshold",
    "attr_filter",
]


# ---------------------------------------------------------------------------
# Region helpers
# ---------------------------------------------------------------------------
def extract_regions(motif_df: pd.DataFrame) -> pd.DataFrame:
    """Unique ``(region, chrom, start, end)`` rows from a motif-locations frame.

    Output column order is exactly ``[region, chrom, start, end]`` per the test
    contract. ``start`` / ``end`` are taken from the region string so they
    reflect the *region* span, not any individual motif within it.
    """
    if motif_df.empty:
        return pd.DataFrame(columns=["region", "chrom", "start", "end"])

    regions = motif_df[["region"]].drop_duplicates().reset_index(drop=True)
    parsed = regions["region"].str.extract(r"^(?P<chrom>[^:]+):(?P<start>\d+)-(?P<end>\d+)$")
    parsed["start"] = parsed["start"].astype(int)
    parsed["end"] = parsed["end"].astype(int)
    parsed.insert(0, "region", regions["region"])
    return parsed


def _regions_to_seqs(regions_df: pd.DataFrame, fasta) -> list:
    """Fetch the DNA string for each region row (chrom/start/end)."""
    return [
        str(fasta[row.chrom][int(row.start):int(row.end)]).upper()
        for row in regions_df.itertuples()
    ]


# ---------------------------------------------------------------------------
# Core DeepLIFT scan
# ---------------------------------------------------------------------------
def scan_deeplift_scores(
    model,
    regions_df: pd.DataFrame,
    fasta_path: str,
    tracks,
    device,
    attr_batch_size: int = 1,
) -> Dict[str, np.ndarray]:
    """Per-region, per-position DeepLIFT importance -> ``(num_tracks, seq_len)``.

    Vectorized over the batch dimension: each batch of ``attr_batch_size``
    regions is one-hot encoded, attributed once per track via
    :class:`captum.attr.DeepLift` with a zero baseline, reduced with
    ``abs(attr).sum(dim=1)``, then scattered back into the per-region map.

    Parameters
    ----------
    model : torch.nn.Module
        Trained deepISA model.
    regions_df : pd.DataFrame
        Columns ``region, chrom, start, end`` (see :func:`extract_regions`).
    fasta_path : str
        Path to a FASTA file.
    tracks : sequence of int
        Output indices to attribute (one DeepLift pass per track).
    device : torch.device
        Compute device.
    attr_batch_size : int
        Regions per attribution batch.

    Returns
    -------
    dict
        ``{region_str: np.ndarray (len(tracks), seq_len)}``.
    """
    device = torch.device(device) if not isinstance(device, torch.device) else device
    model = model.to(device).eval()
    tracks = list(tracks)
    dl = DeepLift(model)

    fasta = load_fasta(fasta_path)
    seqs = _regions_to_seqs(regions_df, fasta)
    regions = regions_df["region"].tolist()
    seq_len = len(seqs[0]) if seqs else 0

    score_map: Dict[str, np.ndarray] = {}
    # Accumulate per-region scores; pre-allocate arrays once we know seq_len.
    pending = {r: np.empty((len(tracks), seq_len), dtype=np.float32) for r in regions}

    with torch.enable_grad():
        for start in range(0, len(seqs), attr_batch_size):
            end = min(start + attr_batch_size, len(seqs))
            batch_seqs = seqs[start:end]
            x_ohe = one_hot_encode(batch_seqs)                              # (B, 4, L)
            x = torch.tensor(x_ohe, dtype=torch.float32, device=device, requires_grad=True)
            baseline = torch.zeros_like(x)

            # One DeepLift call per track; stack along a new track axis.
            per_track = []
            for t in tracks:
                attr = dl.attribute(x, baseline, target=t)
                if attr.ndim == 4 and attr.shape[-1] == 1:
                    attr = attr.squeeze(-1)
                # (B, 4, L) -> (B, L): abs contribution summed over bases
                per_track.append(torch.abs(attr).sum(dim=1))
            scores = torch.stack(per_track, dim=1).detach().cpu().numpy()   # (B, T, L)

            for i, region in enumerate(regions[start:end]):
                pending[region] = scores[i]

    score_map = pending
    return score_map


# ---------------------------------------------------------------------------
# Slice / threshold helpers (fully vectorized)
# ---------------------------------------------------------------------------
def get_slices(df: pd.DataFrame, score_map: Dict[str, np.ndarray]) -> Iterator[np.ndarray]:
    """Yield ``score_map[region][:, start_rel:end_rel]`` for every motif row.

    Each yielded slice has shape ``(num_tracks, motif_len)`` and is a *view*
    into the region's score array -- no copy.
    """
    for row in df.itertuples():
        yield score_map[row.region][:, int(row.start_rel):int(row.end_rel)]


def _get_second_max(arr: np.ndarray, row_idx: int) -> float:
    """Second-largest value along ``arr[row_idx]``.

    ``arr`` is a ``(num_tracks, motif_len)`` per-motif score slice and
    ``row_idx`` selects the track. Returns the runner-up of the flattened row
    via an O(n) partial partition (no full sort). For rows shorter than two
    positions the single value (or 0) is returned, matching the test contract.
    """
    flat = np.asarray(arr[row_idx]).reshape(-1)
    if flat.size < 2:
        return float(flat.max()) if flat.size else 0.0
    return float(np.partition(flat, -2)[-2])


def get_attr_threshold(
    df: pd.DataFrame,
    score_map: Dict[str, np.ndarray],
    track_internal_idx: int,
    percentile: float,
) -> float:
    """Percentile threshold of DeepLIFT scores over all non-motif positions.

    Concatenates, across every non-motif interval in ``df``, the per-position
    scores of ``score_map[region][track_internal_idx]`` between
    ``start_rel:end_rel`` -- fully vectorized via a list-comprehension +
    :func:`numpy.percentile`.
    """
    if df.empty:
        return 0.0
    # Vectorized slice collection across all rows at once.
    starts = df["start_rel"].to_numpy()
    ends = df["end_rel"].to_numpy()
    regions = df["region"].to_numpy()
    # Build a flat array of all non-motif positions for this track via
    # boolean masking per region (regions may differ in span, but slices are
    # independent so we gather then concatenate).
    pieces = [
        score_map[regions[i]][track_internal_idx, int(starts[i]):int(ends[i])]
        for i in range(len(df))
    ]
    pooled = np.concatenate(pieces) if pieces else np.array([0.0])
    return float(np.percentile(pooled, percentile))


# ---------------------------------------------------------------------------
# End-to-end filter
# ---------------------------------------------------------------------------
def attr_filter(
    motif_locs_path: str,
    non_motif_locs_path: str,
    model,
    fasta_path: str,
    tracks,
    attr_percentile: float,
    device,
    attr_batch_size: int = 1,
) -> pd.DataFrame:
    """Filter JASPAR motifs by DeepLIFT importance against a non-motif null.

    For each track ``t`` a motif is annotated with:

    * ``second_max_t{t}`` -- second-largest per-position attribution inside the
      motif span (robust to a single noisy peak).
    * ``pass_threshold_t{t}`` -- 1 if ``second_max_t{t}`` >= the
      ``attr_percentile`` threshold of non-motif scores, else 0.

    A motif is retained iff it passes on *any* track (logical OR).
    """
    motif_df = pd.read_csv(motif_locs_path)
    if motif_df.empty:
        return motif_df

    non_motif_df = pd.read_csv(non_motif_locs_path)
    tracks = list(tracks)

    # The DeepLIFT scan must cover regions referenced by BOTH the motif table
    # and the non-motif (background) table; otherwise get_attr_threshold would
    # raise KeyError on background regions absent from the score_map. Scan the
    # union of region strings.
    motif_regions = extract_regions(motif_df)
    if not non_motif_df.empty:
        non_motif_regions = extract_regions(non_motif_df)
        regions_df = pd.concat(
            [motif_regions, non_motif_regions]
        ).drop_duplicates(subset=["region"]).reset_index(drop=True)
    else:
        regions_df = motif_regions

    score_map = scan_deeplift_scores(
        model=model,
        regions_df=regions_df,
        fasta_path=fasta_path,
        tracks=tracks,
        device=device,
        attr_batch_size=attr_batch_size,
    )

    # Per-track thresholds from non-motif null, vectorized.
    thresholds = {
        t: get_attr_threshold(non_motif_df, score_map, track_internal_idx=ti,
                              percentile=attr_percentile)
        for ti, t in enumerate(tracks)
    }
    logger.info(f"attr thresholds (p{attr_percentile}) by track: {thresholds}")

    # Per-motif second-max, vectorized via list comprehension over slices.
    slices = list(get_slices(motif_df, score_map))   # one (T, motif_len) per row
    for ti, t in enumerate(tracks):
        second_maxes = np.array([_get_second_max(s, ti) for s in slices], dtype=np.float32)
        motif_df[f"second_max_t{t}"] = second_maxes
        motif_df[f"pass_threshold_t{t}"] = (second_maxes >= thresholds[t]).astype(int)

    pass_cols = [f"pass_threshold_t{t}" for t in tracks]
    keep = motif_df[pass_cols].any(axis=1)
    kept = motif_df[keep].reset_index(drop=True)
    logger.info(f"attr_filter kept {len(kept)}/{len(motif_df)} motifs.")
    return kept
