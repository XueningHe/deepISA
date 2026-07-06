
import pandas as pd
from loguru import logger
from scipy.stats import ks_2samp
import numpy as np
import bioframe as bf    

# Internal imports
from deepISA.modeling.predict import compute_predictions 
from deepISA.utils import remove_if_exists, write_stream_csv, apply_threshold_filter
from deepISA.scoring.null import derive_null_thresholds, sample_null_kmers
from deepISA.scoring.utils_isa import load_pred_orig,ablate_motifs, region_str_to_seq



def get_pred_orig(
    model,
    fasta,
    regions,
    tracks,
    outpath,
    device,
    pred_batch_size,
):
    remove_if_exists(outpath, label="prediction of original regions")
    uniq_regions = list(pd.unique(pd.Series(regions)))
    if len(uniq_regions) == 0:
        raise ValueError("No regions provided to compute original predictions.")

    seqs = [region_str_to_seq(fasta, r) for r in uniq_regions]
    preds = compute_predictions(model, seqs, device=device, batch_size=pred_batch_size, tracks=tracks)
    pred_cols = [f"pred_t{i}" for i in tracks]
    df_pred = pd.DataFrame(preds, columns=pred_cols)
    df_pred.insert(0, "region", uniq_regions)
    df_pred.to_csv(outpath, index=False)
    logger.info(f"Saved region original predictions: {outpath} ({len(df_pred)} regions)")





def single_isa_core(
    model,
    fasta,
    locs_df,
    pred_orig_path,
    outpath,     
    device,
    tracks,
    num_regions_per_batch,
    pred_batch_size,
):
    remove_if_exists(outpath, label="single ISA file")
    logger.info(f"Single ISA started. Total rows to process: {len(locs_df)}")
    orig_pred_map = load_pred_orig(pred_orig_path, tracks)
    region_groups = list(locs_df.groupby("region"))

    for i in range(0, len(region_groups), num_regions_per_batch):
        batch = region_groups[i : i + num_regions_per_batch]
        batch_results = []
        for region_str, group in batch:
            seq_orig = region_str_to_seq(fasta, region_str)
            group = group.copy()
            group["seq_mut"] = [
                ablate_motifs(seq_orig, [int(s)], [int(e)])
                for s, e in zip(group["start_rel"].to_numpy(), group["end_rel"].to_numpy())
            ]
            batch_results.append(group)

        current_df = pd.concat(batch_results).reset_index(drop=True)
        preds_orig_sel = np.vstack([orig_pred_map[reg] for reg in current_df["region"].values])
        preds_mut = compute_predictions(model, current_df["seq_mut"].values, device, pred_batch_size, tracks=tracks)
        current_df = current_df.drop(columns=["seq_mut"])
        for j,t in enumerate(tracks):
            current_df[f"isa_t{t}"] = preds_orig_sel[:, j] - preds_mut[:, j]

        write_stream_csv(current_df, outpath)






def isa_filter(
    single_isa_path,
    null_isa_path,
    outpath,
    null_percentile,
):
    df = pd.read_csv(single_isa_path)
    null_df = pd.read_csv(null_isa_path)
    isa_cols = [c for c in df.columns if c.startswith("isa_t")]
    thresholds = derive_null_thresholds(
        null_df=null_df,
        cols=isa_cols,
        percentile=null_percentile,
    )
    logger.info(f"Filtering motifs using null-derived thresholds (p{null_percentile}): {thresholds}")
    # Keep row if ANY track passes threshold.
    filtered_df, _ = apply_threshold_filter(
        df=df,
        cols=isa_cols,
        thresholds=thresholds,
        rule="any_tails",
    )
    filtered_df.to_csv(outpath, index=False)
    logger.info(f"Filtered single ISA saved to {outpath}. Kept {len(filtered_df)}/{len(df)} motifs.")
    return filtered_df



def run_single_isa(
    model,
    fasta,
    motif_locs_path,
    non_motif_locs_path,
    single_isa_outpath,
    null_isa_outpath,
    pred_orig_outpath,
    null_percentile,
    device,
    tracks=[0],
    num_regions_per_batch=200,
    pred_batch_size=1024,
    null_n_samples: int = 8192,
):
    if isinstance(fasta, str):
        fasta=bf.load_fasta(fasta)
        
    locs_df = pd.read_csv(motif_locs_path)
    non_motif_locs_df = pd.read_csv(non_motif_locs_path)
    regions_pos = locs_df["region"].unique().tolist()
    regions_neg = non_motif_locs_df["region"].unique().tolist()

    regions = list(set(regions_pos + regions_neg))
    logger.info("Computing original predictions for all regions")
    get_pred_orig(
        model=model,
        fasta=fasta,
        regions=regions,
        tracks=tracks,
        outpath=pred_orig_outpath,
        device=device,
        pred_batch_size=pred_batch_size,
    )
    
    # deduplicate based on [chrom,start,end,region], keep the row with largest score, 
    locs_df = locs_df.sort_values("score", ascending=False).drop_duplicates(subset=["chrom","start","end","region"], keep="first").reset_index(drop=True)
    logger.info("Running single ISA")
    single_isa_core(
        model=model,
        fasta=fasta,
        locs_df=locs_df,
        outpath=single_isa_outpath,
        device=device,
        tracks=tracks,
        num_regions_per_batch=num_regions_per_batch,
        pred_batch_size=pred_batch_size,
        pred_orig_path=pred_orig_outpath,
    )
    df_isa = pd.read_csv(single_isa_outpath)
    df_isa["len"] = df_isa["end_rel"] - df_isa["start_rel"]
    
    non_motif_locs_df = pd.read_csv(non_motif_locs_path)
    logger.info("Sampling non-motif kmers for null ISA")
    null_kmers_df = sample_null_kmers(
        non_motif_df=non_motif_locs_df,
        target_lengths=df_isa["len"].to_numpy(),
        n_samples=null_n_samples
    )

    logger.info("Running null ISA")
    single_isa_core(
        model=model,
        fasta=fasta,
        locs_df=null_kmers_df,
        outpath=null_isa_outpath,
        device=device,
        tracks=tracks,
        num_regions_per_batch=num_regions_per_batch,
        pred_batch_size=pred_batch_size,
        pred_orig_path=pred_orig_outpath,
    )
    
    logger.info("Filtering motif ISA by null-derived thresholds")
    isa_filter(
        single_isa_path=single_isa_outpath,
        null_isa_path=null_isa_outpath,
        outpath=single_isa_outpath,
        null_percentile=null_percentile,
    )
    
    logger.info(f"Single ISA complete. Results saved to {single_isa_outpath}.")





#---------------------------
# Aggregation functions
#---------------------------

def _signed_ks_test(fg_vals, bg_vals):
    """
    Signed KS:
      + if foreground tends to be larger than background
      - if foreground tends to be smaller than background
    """
    stat, pval= ks_2samp(fg_vals, bg_vals, alternative="two-sided", mode="auto")
    sign = np.sign(np.median(fg_vals) - np.median(bg_vals))
    return sign*stat, pval


def calc_tf_importance(
    single_isa_path,
    null_isa_path,
    outpath,
    null_percentile,
    min_count=10,
):
    """
    Compute TF importance from motif ISA and null ISA.

    - Threshold per track is derived from null ISA percentile.
    - Supports multiple tracks (isa_t0, isa_t2, ...).
    - KS compares TF ISA distribution vs null ISA distribution (per track).
    """
    remove_if_exists(outpath, label="TF importance file")

    df = pd.read_csv(single_isa_path)
    null_df = pd.read_csv(null_isa_path)

    isa_cols = sorted([c for c in df.columns if c.startswith("isa_t")])
    # Derive per-track threshold from null percentile
    thresholds = derive_null_thresholds(
        null_df=null_df,
        cols=isa_cols,
        percentile=null_percentile
    )
    logger.info(f"ISA thresholds from null (percentile {null_percentile}): {thresholds}")
    
    logger.info(f"Calculating TF importance for tracks: {isa_cols}")
    results = []
    for tf, tf_data_all in df.groupby("tf"):
        res = {"tf": tf, "n_total": tf_data_all.shape[0]}
        for col in isa_cols:
            # calculate only motifs with isa above threshold 
            null_vals = null_df[col].to_numpy(dtype=float)
            thresh_pos = thresholds[col]["pos"]
            thresh_neg = thresholds[col]["neg"]
            tf_vals_all = tf_data_all[col].to_numpy(dtype=float) # for ks
            tf_vals = tf_vals_all[(tf_vals_all > thresh_pos) | (tf_vals_all < thresh_neg)] # for mean/median
            if len(tf_vals) < min_count: continue
            res[f"n_effective_{col}"] = len(tf_vals)
            res[f"mean_{col}"] = float(np.mean(tf_vals))
            res[f"median_{col}"] = float(np.median(tf_vals))
            res[f"ks_d_{col}"], res[f"ks_pval_{col}"] = _signed_ks_test(tf_vals_all, null_vals)
            results.append(res)

    out_df = pd.DataFrame(results)
    out_df.to_csv(outpath, index=False, float_format="%.4f")
    logger.info(f"TF importance saved to {outpath}")
    return out_df