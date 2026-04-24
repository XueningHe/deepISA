import os
import json
import pandas as pd
import numpy as np
from loguru import logger
from itertools import combinations
import bioframe as bf
from scipy.stats import ks_2samp, mannwhitneyu
from statsmodels.stats.multitest import multipletests


# Internal imports
from deepISA.modeling.predict import compute_predictions 

from deepISA.utils import (
    ablate_motifs, 
    format_cooperativity_categorical,
)



def _process_region(seq_orig, 
                    region_motif_rows, 
                    inde_dist_max, 
                    pass_cols):
    """
    region_motif_rows: all motif rows belonging to one region (subset of df_motif_locs)
    Returns pair_df and the 3 ablated sequence lists, or None if no valid pairs.
    """
    if len(region_motif_rows) < 2:
        return None

    pairs = []
    for idx1, idx2 in combinations(region_motif_rows.index, 2):
        m1, m2 = region_motif_rows.loc[idx1], region_motif_rows.loc[idx2]
        if max(m1.start_rel, m2.start_rel) < min(m1.end_rel, m2.end_rel):
            continue
        dist = m2.start_rel - m1.start_rel
        if dist > inde_dist_max:
            continue
        pair_data = {
            "region": m1.region,
            "tf1": m1.tf, "tf2": m2.tf,
            "start1": m1.start_rel, "end1": m1.end_rel,
            "start2": m2.start_rel, "end2": m2.end_rel,
            "distance": dist,
        }
        for col in pass_cols:
            pair_data[f"tf1_{col}"] = m1[col]
            pair_data[f"tf2_{col}"] = m2[col]
        pairs.append(pair_data)

    if not pairs:
        return None

    pair_df = pd.DataFrame(pairs)
    seqs_m1   = [ablate_motifs(seq_orig, r.start1, r.end1) for r in pair_df.itertuples()]
    seqs_m2   = [ablate_motifs(seq_orig, r.start2, r.end2) for r in pair_df.itertuples()]
    seqs_both = [ablate_motifs(ablate_motifs(seq_orig, r.start1, r.end1), r.start2, r.end2)
                 for r in pair_df.itertuples()]

    return pair_df, seqs_m1, seqs_m2, seqs_both





def run_combi_isa(model, 
                  fasta_path, 
                  motif_locs_path, 
                  outpath,
                  device,
                  inde_dist_max,
                  tracks=[0],
                  num_regions_per_batch=200,
                  pred_batch_size=1024):
    
    if os.path.exists(outpath):
        logger.info(f"Removing existing combinatorial ISA results file: {outpath}")
        os.remove(outpath)

    df_motif_locs = pd.read_csv(motif_locs_path)
    if df_motif_locs.empty:
        logger.warning("No motifs. Try lowering attr_percentile or motif_score_thresh.")
        return None
    
    logger.info(f"Perform combinatorial ISA. Loaded motif locations from {motif_locs_path}...")    
    regions = df_motif_locs["region"].unique()
    fasta = bf.load_fasta(fasta_path)
    
    log_interval = 20  # Log every 10th batch
    batch_count = 0
    
    for batch_start in range(0, len(regions), num_regions_per_batch):
        # logging info about progress
        batch_count += 1
        if batch_count % log_interval == 0:
            end_idx = min(batch_start + num_regions_per_batch, len(regions))
            logger.info(f"Batch {batch_count}: Processing regions {batch_start} to {end_idx} of {len(regions)}")
                
        batch_regions = regions[batch_start : batch_start + num_regions_per_batch]

        pair_dfs = []
        all_seqs_m1, all_seqs_m2, all_seqs_both = [], [], []
        orig_seqs, orig_region_labels = [], []
        # track which pair_df rows belong to which flat index in m1/m2/both arrays
        pair_offsets = []  # (start, n) into the flat ablated arrays

        for region_str in batch_regions:
            region_motif_rows = df_motif_locs[df_motif_locs["region"] == region_str].copy()
            chrom, coords = region_str.split(":")
            start_r, end_r = map(int, coords.split("-"))
            seq_orig = str(fasta[chrom][start_r:end_r]).upper()
            pass_cols = [c for c in region_motif_rows.columns if c.startswith("pass_threshold_t")]

            result = _process_region(seq_orig, region_motif_rows, inde_dist_max, pass_cols)
            if result is None:
                continue

            pair_df, seqs_m1, seqs_m2, seqs_both = result
            pair_offsets.append((len(all_seqs_m1), len(pair_df)))

            all_seqs_m1.extend(seqs_m1)
            all_seqs_m2.extend(seqs_m2)
            all_seqs_both.extend(seqs_both)

            # One orig seq per region, tagged for merging
            orig_seqs.append(seq_orig)
            orig_region_labels.append(region_str)
            pair_dfs.append(pair_df)

        if not pair_dfs:
            continue

        # Predict ablated — separate calls, full batch across regions
        p_m1   = compute_predictions(model, all_seqs_m1,   device=device, batch_size=pred_batch_size)
        p_m2   = compute_predictions(model, all_seqs_m2,   device=device, batch_size=pred_batch_size)
        p_both = compute_predictions(model, all_seqs_both, device=device, batch_size=pred_batch_size)

        # Predict orig — one per region, merge by region label
        p_orig = compute_predictions(model, orig_seqs, device=device, batch_size=pred_batch_size)
        orig_pred_df = pd.DataFrame({
            "region": orig_region_labels,
            **{f"p_orig_t{t}": p_orig[:, t] for t in tracks}
        })

        # Assemble results per region
        for pair_df, (start, n) in zip(pair_dfs, pair_offsets):
            sl = slice(start, start + n)
            pair_df = pair_df.merge(orig_pred_df, on="region", how="left")

            for t in tracks:
                pair_df[f"isa1_t{t}"]        = pair_df[f"p_orig_t{t}"] - p_m1[sl, t]
                pair_df[f"isa2_t{t}"]        = pair_df[f"p_orig_t{t}"] - p_m2[sl, t]
                pair_df[f"isa_both_t{t}"]    = pair_df[f"p_orig_t{t}"] - p_both[sl, t]
                pair_df[f"interaction_t{t}"] = (pair_df[f"isa1_t{t}"] + pair_df[f"isa2_t{t}"]) - pair_df[f"isa_both_t{t}"]
                pair_df = pair_df.drop(columns=[f"p_orig_t{t}"])

            header = not os.path.exists(outpath)
            pair_df.to_csv(outpath, mode='a', index=False, header=header)

    logger.info(f"Combinatorial ISA complete. Results saved to {outpath}")




def _filter_pos_interaction(df, t):
    """
    Filters out rows where the TFs aren't acting as activators 
    or joint effect is smaller than individual effect.
    """
    isa1 = df[f"isa1_t{t}"]
    isa2 = df[f"isa2_t{t}"]
    isa_both = df[f"isa_both_t{t}"]
    valid_mask = (isa1 > 0) & (isa2 > 0) & (isa_both > isa1) & (isa_both > isa2) 
    return df[valid_mask].copy()



    # TODO: subset by first few columns, not all columns
def calc_coop_score(combi_isa_path, 
                    outpath, 
                    level, # can be 'tf_pair' or 'tf'
                    inde_dist_min,
                    inde_dist_max,
                    track_idx=0,
                    min_count=10,
                    q_val_thresh=0.1):
    """
    Unified logic for Pair-level or TF-level distributional analysis.
    """
    if os.path.exists(outpath):
        logger.info(f"Removing existing cooperativity results file: {outpath}")
        os.remove(outpath)
    
    df = pd.read_csv(combi_isa_path)
        
    t_pass_col = f"pass_threshold_t{track_idx}"
    tf1_check = f"tf1_{t_pass_col}"
    tf2_check = f"tf2_{t_pass_col}"
    
    initial_len = len(df)
    # Apply the AND filter
    mask = df[tf1_check].astype(bool) & df[tf2_check].astype(bool)
    df = df[mask].copy()
    logger.info(
        f"Attribution Filter (Track {track_idx}): Kept {len(df)}/{initial_len} pairs "
        f"where both motifs exceed the noise floor."
    )

    if df.empty:
        logger.warning(f"No pairs remaining after Attribution filtering for track {track_idx}.")
        return pd.DataFrame()
    
    inter_col = f"interaction_t{track_idx}"
    # sort alphabetically here
    df["tf1"], df["tf2"] = np.minimum(df["tf1"], df["tf2"]), np.maximum(df["tf1"], df["tf2"])
    df = df.drop_duplicates()
    inter_col = f"interaction_t{track_idx}"
    
    # 1. Filter and Define Null Distribution
    df = _filter_pos_interaction(df, track_idx)
    null_vals = df[(df["distance"] > inde_dist_min) & (df["distance"] <= inde_dist_max)][inter_col]
    
    if null_vals.empty:
        logger.warning(f"No null distribution for track {track_idx}. Try lowering attr_percentile.")
        return pd.DataFrame()

    # 2. Reshape data if level is 'tf'
    if level == 'tf':
        # Stack tf1 and tf2 to count every interaction a TF participates in
        df_melt = pd.concat([
            df[['tf1', inter_col, 'distance']].rename(columns={'tf1': 'name'}),
            df[['tf2', inter_col, 'distance']].rename(columns={'tf2': 'name'})
        ])
    else:
        # sort alphabetically then join
        if 'pair' not in df.columns:
            df['pair'] = df.apply(lambda r: f"{min(r.tf1, r.tf2)}|{max(r.tf1, r.tf2)}", axis=1)
        
        df_melt = df.copy()
        df_melt = df_melt.rename(columns={'pair': 'name'})

    # 3. Aggregate Distributions
    group_col = 'name'
    results = []
    for name, group in df_melt.groupby(group_col):
        # remove pairs with too few interactions to be meaningful
        if len(group) < min_count:
            continue
        vals = group[inter_col]
        ks_res = ks_2samp(vals, null_vals)
        # mann whitney u test TODO: choose better test later on.
        mw_res = mannwhitneyu(vals, null_vals)
        i_sum = vals.sum()
        abs_i_sum = vals.abs().sum()
        results.append({
            level: name,
            "abs_i_sum": round(abs_i_sum, 4),
            "coop_score": round(i_sum / abs_i_sum, 4) if abs_i_sum > 0 else 0,
            "ks_p": ks_res.pvalue,
            "ks_d": round(ks_res.statistic, 6),
            "mw_p": mw_res.pvalue,
            "mw_d": round(mw_res.statistic, 6),
            "count": len(vals),
            "mean_distance": group["distance"].mean()
        })

    res_df = pd.DataFrame(results)
    
    # 4. Statistical Correction and Assignment
    res_df["ks_q"] = multipletests(res_df["ks_p"], method='fdr_bh')[1]
    res_df["mw_q"] = multipletests(res_df["mw_p"], method='fdr_bh')[1]
    res_df = assign_cooperativity(res_df, q_val_thresh)
    
    res_df.to_csv(outpath, mode='w', index=False)
    logger.info(f"Cooperativity stats saved to {outpath}")



def assign_cooperativity(df, q_val_thresh):
    """
    Categorizes TF pairs based on FDR-corrected significance and intensity thresholds.
    """
    df["cooperativity"] = "Independent"
    is_significant = df["ks_q"] < q_val_thresh
    # Determine thresholds based on percentiles of significant coop_scores
    synergy_thresh = df.loc[is_significant, "coop_score"].quantile(0.7)
    redun_thresh = df.loc[is_significant, "coop_score"].quantile(0.3)
    # Apply Directional Thresholds to significant pairs
    df.loc[is_significant & (df["coop_score"] > synergy_thresh), "cooperativity"] = "Synergistic"
    df.loc[is_significant & (df["coop_score"] < redun_thresh), "cooperativity"] = "Redundant"
    df.loc[is_significant & (df["coop_score"].between(redun_thresh, synergy_thresh)), "cooperativity"] = "Intermediate"
    # Nullify coop_score for Independent cases
    # df.loc[df["cooperativity"] == "Independent", "coop_score"] = np.nan
    return format_cooperativity_categorical(df)






