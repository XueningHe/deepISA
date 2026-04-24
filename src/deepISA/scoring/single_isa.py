
import pandas as pd
import os
import json
from loguru import logger
from scipy.stats import ks_2samp
import bioframe as bf

# Internal imports
from deepISA.modeling.predict import compute_predictions 
from deepISA.utils import ablate_motifs






def run_single_isa(model, 
                   fasta_path, 
                   motif_locs_path, 
                   outpath,
                   device,
                   tracks=[0],
                   num_regions_per_batch=200,
                   pred_batch_size=1024):
    """
    Computes ISA scores and writes them to disk incrementally to manage RAM.
    
    Args:
        model: Trained deepISA model.
        fasta: Fasta records object (e.g., from bf.load_fasta).
        motif_locs_path: Path to CSV containing motif locations with columns ['region', 'start', 'end', 'tf'].
        track_idx: Index of the track to compute ISA for.
        device: torch device.
        outpath: Path to save the ISA results CSV.
        num_regions_per_batch: Number of unique regions to process before writing to disk. One region can have multiple motifs.
    """
    if os.path.exists(outpath):
        logger.warning(f"Removing existing single ISA results file: {outpath}")
        os.remove(outpath)
        
    df_motif_locs = pd.read_csv(motif_locs_path)    
    if df_motif_locs.empty:
        logger.warning("No motifs passed the functional filter. Try lowering the functional_filter_percentile.")
        return None
    
    logger.info(f"Single ISA started. Total motifs to process: {len(df_motif_locs)}")

    # Group motifs by region to minimize redundant sequence extractions
    region_groups = list(df_motif_locs.groupby("region"))
    fasta = bf.load_fasta(fasta_path)
    
    for i in range(0, len(region_groups), num_regions_per_batch):
        batch = region_groups[i : i + num_regions_per_batch]
        batch_results = []

        for region_str, group in batch:
            chrom, coords = region_str.split(":")
            start_region, end_region = map(int, coords.split("-"))
            group = group.copy()
            # We add the original sequence to every row for the prediction wrapper
            group["seq_orig"] = str(fasta[chrom][start_region:end_region]).upper()
            group["seq_mut"] = group.apply(
                lambda row: ablate_motifs(row['seq_orig'], row['start_rel'], row['end_rel']), 
                axis=1
            )
            batch_results.append(group)

        # Combine current batch
        current_df = pd.concat(batch_results).reset_index(drop=True)
        preds_orig = compute_predictions(model, current_df["seq_orig"].values, device, batch_size=pred_batch_size)
        preds_mut = compute_predictions(model, current_df["seq_mut"].values, device, batch_size=pred_batch_size)
        # 4. Calculate ISA Score (Difference in prediction)
        for t in tracks:
            col_name = f"isa_t{t}"
            current_df[col_name] = preds_orig[:, t] - preds_mut[:, t]

        # 5. Incremental Write to Disk
        header = not os.path.exists(outpath)
        current_df = current_df.drop(columns=["seq_orig", "seq_mut"])
        current_df.to_csv(outpath, mode='a', index=False, header=header)
    
    logger.info(f"Single ISA complete. Results saved to {outpath}")





def calc_tf_importance(single_isa_path, min_count):
    """
    Aggregates ISA scores across all proteins for all available tracks in the data.
    Input df only contains the tracks subsetted during run_single_isa.
    """
    df = pd.read_csv(single_isa_path)
    isa_cols = [c for c in df.columns if c.startswith("isa_t")]
    
    if not isa_cols:
        logger.warning("No columns starting with 'isa_' found in the input data.")
        return pd.DataFrame()

    logger.info(f"Calculating TF importance for tracks: {isa_cols}...")
    
    results = []
    # Group by TF to calculate metrics per protein
    for tf, tf_data in df.groupby("tf"):
        # Base info: TF name and how many instances we measured
        res = {"tf": tf, "count": len(tf_data)}
        for col in isa_cols:
            track_id = col.replace("isa_", "") # e.g., "t0"
            pass_col = f"pass_threshold_{track_id}"
            tf_data = tf_data[tf_data[pass_col] == 1]
            res[f"mean_{col}"] = tf_data[col].mean()
            res[f"median_{col}"] = tf_data[col].median()
            res[f"ks_{col}"] = _signed_ks_test(df, tf, col, min_count)
        results.append(res)
    return pd.DataFrame(results)



def _signed_ks_test(df, tf, isa_col, min_count):
    """Calculates signed KS test statistic for a TF vs the background distribution."""
    df_this_protein = df[df.tf == tf].reset_index(drop=True)
    if df_this_protein.shape[0] < min_count:
        return None
    dstat, _ = ks_2samp(df_this_protein[isa_col], df[isa_col])
    if df_this_protein[isa_col].median() < df[isa_col].median():
        dstat = -dstat
    return dstat