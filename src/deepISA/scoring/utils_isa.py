import numpy as np
import pandas as pd
from loguru import logger



def load_pred_orig(region_orig_pred_path, tracks):
    df = pd.read_csv(region_orig_pred_path)
    needed = ["region"] + [f"pred_t{t}" for t in tracks]
    return {
        r["region"]: np.array([r[f"pred_t{t}"] for t in tracks], dtype=float)
        for _, r in df[needed].iterrows()
    }
    



def ablate_motifs(seq, motif_starts, motif_ends):
    """
    Scramble the sequence between multiple motif start and end positions.
    Args:
        seq: A string of sequence.
        motif_starts: A list of integers for motif starts.
        motif_ends: A list of integers for motif ends.
    Returns:
        A string of scrambled sequence.
    """
    if isinstance(motif_starts, int):
        motif_starts = [motif_starts]
    if isinstance(motif_ends, int):
        motif_ends = [motif_ends]
    if len(motif_starts) != len(motif_ends):
        raise ValueError("motif_starts and motif_ends must have the same length")
    # Sort the motifs by start position
    motifs = sorted(zip(motif_starts, motif_ends), key=lambda x: x[0])
    # Initialize variables
    seq_ablated = ''
    previous_end = 0
    # Iterate and ablate each motif
    for start, end in motifs:
        if start < previous_end:
            logger.warning("Motif overlap detected: motif_starts={}, motif_ends={}", motif_starts, motif_ends)
            continue  # Skip overlapping motif
        motif = seq[start:end]
        motif_ablated = "N" * len(motif)  
        # Append non-motif and scrambled motif parts
        seq_ablated += seq[previous_end:start] + motif_ablated
        previous_end = end
    # Append the remaining part of the sequence if any
    seq_ablated += seq[previous_end:]
    return seq_ablated


def region_str_to_seq(fasta, region_str: str) -> str:
    chrom, coords = region_str.split(":")
    start_r, end_r = map(int, coords.split("-"))
    return str(fasta[chrom][start_r:end_r]).upper()


