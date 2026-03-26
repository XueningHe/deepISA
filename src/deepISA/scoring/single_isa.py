
import pandas as pd
import os
from loguru import logger
from scipy.stats import ks_2samp
import bioframe as bf

# Internal imports
from deepISA.modeling.predict import compute_predictions 
from deepISA.utils import ablate_motifs
from deepISA.utils import find_available_gpu, load_data





def run_single_isa(model, 
                   fasta_path, 
                   motif_locs, 
                   outpath,  
                   track_idx=0,
                   device=None, 
                   batch_size=100):
    """
    Computes ISA scores and writes them to disk incrementally to manage RAM.
    
    Args:
        model: Trained deepISA model.
        fasta: Fasta records object (e.g., from bf.load_fasta).
        motif_locs: DataFrame or path to CSV containing motifs and their genomic regions.
        track_idx: Index of the track to compute ISA for.
        device: torch device.
        outpath: Path to save the ISA results CSV.
        batch_size: Number of unique regions to process before writing to disk. One region can have multiple motifs.
    """
    if device is None:
        device = find_available_gpu()

    if os.path.exists(outpath):
        os.remove(outpath)

    if isinstance(track_idx, int):
        track_idx = [track_idx]
        
    logger.info(f"Single ISA for {len(motif_locs)} motifs.")
    motif_locs = load_data(motif_locs)
    # Group motifs by region to minimize redundant sequence extractions
    region_groups = list(motif_locs.groupby("region"))
    
    for i in range(0, len(region_groups), batch_size):
        batch = region_groups[i : i + batch_size]
        batch_results = []

        for region_str, group in batch:
            chrom, coords = region_str.split(":")
            start_gen, end_gen = map(int, coords.split("-"))
            
            # 1. Sequence Extraction (Original)
            fasta = bf.load_fasta(fasta_path)
            seq_orig = str(fasta[chrom][start_gen:end_gen]).upper()
            
            # 2. Sequence Ablation (Digital Knockout)
            group = group.copy()
            group["start_rel"] = group["start"] - start_gen
            group["end_rel"] = group["end"] - start_gen
            
            # We add the original sequence to every row for the prediction wrapper
            group["seq_orig"] = seq_orig
            group["seq_mut"] = group.apply(
                lambda row: ablate_motifs(row['seq_orig'], row['start_rel'], row['end_rel']), 
                axis=1
            )
            batch_results.append(group)

        # Combine current batch
        current_df = pd.concat(batch_results).reset_index(drop=True)

        # 3. Model Inference (Vectorized across the batch)
        # compute_predictions should return (N, tracks) for both orig and mut
        preds_orig = compute_predictions(model, current_df["seq_orig"].values, device)
        preds_mut = compute_predictions(model, current_df["seq_mut"].values, device)

        # 4. Calculate ISA Score (Difference in prediction)
        # Assuming preds are (Samples, 2) where [:, 0] is Regression and [:, 1] is Classification
        for t in track_idx:
            col_name = f"isa_t{t}"
            current_df[col_name] = preds_orig[:, t] - preds_mut[:, t]

        # 5. Incremental Write to Disk
        header = not os.path.exists(outpath)
        current_df = current_df.drop(columns=["seq_orig", "seq_mut"])
        current_df.to_csv(outpath, mode='a', index=False, header=header)
        
        logger.info(f"Processed batch {i//batch_size + 1}: {len(current_df)} motif importance saved.")
    
    logger.info(f"Single ISA complete. Results saved to {outpath}")





def calc_tf_importance(df):
    """
    Aggregates ISA scores across all proteins for all available tracks in the data.
    Input df only contains the tracks subsetted during run_single_isa.
    """
    df=load_data(df)

    # 2. Automatically identify all ISA tracks present in the file
    isa_cols = [c for c in df.columns if c.startswith("isa_t")]
    
    if not isa_cols:
        logger.warning("No columns starting with 'isa_' found in the input data.")
        return pd.DataFrame()

    logger.info(f"Calculating TF importance for tracks: {isa_cols}")
    
    results = []
    # Group by TF to calculate metrics per protein
    for tf, tf_data in df.groupby("tf"):
        # Base info: TF name and how many instances we measured
        res = {"tf": tf, "count": len(tf_data)}
        for col in isa_cols:
            res[f"mean_{col}"] = tf_data[col].mean()
            res[f"median_{col}"] = tf_data[col].median()
            res[f"ks_{col}"] = _signed_ks_test(df, tf, col)
        results.append(res)
    logger.info("TF importance calculation complete.")
    return pd.DataFrame(results)



def _signed_ks_test(df, tf, isa_col):
    """Calculates signed KS test statistic for a TF vs the background distribution."""
    df_this_protein = df[df.tf == tf].reset_index(drop=True)
    if df_this_protein.shape[0] < 5:
        logger.warning(f"Insufficient data for TF {tf} to perform KS test.")
        return None
    dstat, _ = ks_2samp(df_this_protein[isa_col], df[isa_col])
    if df_this_protein[isa_col].median() < df[isa_col].median():
        dstat = -dstat
    return dstat