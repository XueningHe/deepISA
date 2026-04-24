import bioframe as bf
import pandas as pd
import numpy as np
from loguru import logger
import torch

from captum.attr import DeepLift
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="captum")


from deepISA.utils import get_sequences_from_df, one_hot_encode



def extract_regions(motif_locs_df):
    """
    Extracts unique parent regions from motif_locs_df 'region' column.
    Returns DataFrame with [chrom, start, end, region].
    """
    # Get unique strings like 'chr1:1000-1600'
    unique_regions = motif_locs_df['region'].unique()
    regions_df = pd.DataFrame({'region': unique_regions})
    # regex extract: everything before ':', then digits, then digits after '-'
    coords = regions_df['region'].str.extract(r'([^:]+):(\d+)-(\d+)')
    regions_df['chrom'] = coords[0]
    regions_df['start'] = coords[1].astype(int)
    regions_df['end'] = coords[2].astype(int)
    return regions_df



def add_relative_coords(df):
    """One-time vectorized calculation of relative offsets."""
    # Extract reg_start once for the whole DF
    reg_starts = df['region'].str.extract(r':(\d+)-')[0].astype(int)
    df['start_rel'] = (df['start'] - reg_starts)
    df['end_rel'] = (df['end'] - reg_starts)
    return df


def map_non_motifs(regions_df, motif_locs_df):
    """Returns genomic intervals not covered by motifs."""
    # bioframe subtract finds the 'gaps' in regions_df not covered by motif_locs_df
    non_motif_df = bf.subtract(regions_df, motif_locs_df)
    non_motif_df = non_motif_df[['chrom', 'start', 'end', 'region']]
    non_motif_df = add_relative_coords(non_motif_df)
    return non_motif_df



def scan_deeplift_scores(model, 
                         regions_df, 
                         fasta_path, 
                         tracks, 
                         device, 
                         attr_batch_size):
    fasta = bf.load_fasta(fasta_path)
    model.to(device).eval()
    dl = DeepLift(model)
    score_map = {}
    for i in range(0, len(regions_df), attr_batch_size):
        batch_df = regions_df.iloc[i : i + attr_batch_size]
        seq_list = get_sequences_from_df(batch_df, fasta)
        x_ohe = one_hot_encode(seq_list)
        input_tensor = torch.tensor(x_ohe, device=device, requires_grad=True)
        baseline = torch.zeros_like(input_tensor)
        # Container for this batch's multi-track scores
        # Shape: (batch_size, num_tracks, seq_len)
        batch_track_scores = []
        for t in tracks:
            attr = dl.attribute(input_tensor, baseline, target=t)                
            scores = torch.abs(attr).sum(dim=1).cpu().detach().numpy()
            batch_track_scores.append(scores)
        final_batch_scores = np.stack(batch_track_scores, axis=1)
        for idx, row in enumerate(batch_df.itertuples()):
            # Store the multi-track array for this region
            score_map[row.region] = final_batch_scores[idx]
    return score_map



def get_slices(df, score_map):
    for row in df.itertuples():
        full_scores = score_map.get(row.region)
        if full_scores is not None:
            # Slice all tracks (:), but only the specific genomic window
            yield full_scores[:, row.start_rel:row.end_rel]


def _get_second_max(s, track_internal_idx):
    track_s = s[track_internal_idx]  # shape (motif_len,)
    return np.partition(track_s, -2)[-2]


def get_attr_threshold(non_motif_df, score_map, track_internal_idx, percentile):
    # Extract slices for a specific track index within the score_map array
    all_values = np.concatenate([
        s[track_internal_idx] for s in get_slices(non_motif_df, score_map)
    ])
    return np.percentile(all_values, percentile)



def attr_filter(
    motif_locs_path,
    model,
    fasta_path,
    tracks,
    attr_percentile,
    device,
    attr_batch_size,
):
    """
    Given a motif DataFrame, calculates DeepLift importance scores and filters
    out motifs that do not exceed the functional noise floor of the sequence.
 
    Returns the filtered DataFrame with an added 'second_max_score' column.
    """
    motif_locs_df = pd.read_csv(motif_locs_path)
    
    if motif_locs_df.empty:
        logger.warning("Input motif DataFrame is empty; skipping functional filter.")
        return motif_locs_df
 
    # 1. Identify parent regions and run DeepLift
    regions_df = extract_regions(motif_locs_df)
    logger.info(f"Generating DeepLift scores (track {tracks}) for {len(regions_df)} regions...")
    score_map = scan_deeplift_scores(
        model=model,
        regions_df=regions_df,
        fasta_path=fasta_path,
        tracks=tracks,
        device=device,
        attr_batch_size=attr_batch_size,
    )
 
    # 2. Derive noise floor from non-motif positions
    logger.info(f"Calculating functional threshold ({attr_percentile}th percentile of non-motifs)...")
    non_motif_df = map_non_motifs(regions_df, motif_locs_df)
    motif_locs_df = add_relative_coords(motif_locs_df)
    
    slices = list(get_slices(motif_locs_df, score_map))  # materialise once
    for i, t in enumerate(tracks):
        motif_locs_df[f"second_max_t{t}"] = [_get_second_max(s, i) for s in slices]
        
    pass_cols = []
    for i, t in enumerate(tracks):
        # Calculate threshold for this specific track
        thresh = get_attr_threshold(non_motif_df, score_map, i, attr_percentile)
        logger.info(f"Track {t}: Functional threshold set at {thresh:.4f} based on non-motif importance scores.")
        # Calculate second_max for this track
        col_name = f"pass_threshold_t{t}"
        pass_cols.append(col_name)
        motif_locs_df[col_name] = (motif_locs_df[f"second_max_t{t}"] > thresh).astype(int)

    # Final Filter: Keep row if it passes for ANY track
    filtered_df = motif_locs_df[motif_locs_df[pass_cols].any(axis=1)].reset_index(drop=True).copy()
    logger.info(f"Filtered motifs: {len(filtered_df)} out of {len(motif_locs_df)} passed the functional threshold.")
    return filtered_df
    