import os
import pandas as pd
import numpy as np
from loguru import logger
from itertools import combinations
import bioframe as bf
from scipy.stats import ks_2samp, mannwhitneyu
from statsmodels.stats.multitest import multipletests


# Internal imports
from deepISA.utils import find_available_gpu
from deepISA.modeling.predict import compute_predictions 
from deepISA.utils import (
    ablate_motifs, 
    format_cooperativity_categorical,
    load_data
)



def run_combi_isa(model,
                  fasta_path,
                  motif_locs,
                  outpath,
                  track_idx=0,
                  receptive_field=255,
                  device=None):
    """
    Computes Pairwise ISA and cooperativity terms. 
    Now writes RAW values to CSV without NaN masking.
    """
    if os.path.exists(outpath):
        logger.info(f"Removing existing results file: {outpath}")
        os.remove(outpath)
                
    if device is None:
        device = find_available_gpu()
    
    df_motif_locs = load_data(motif_locs)
    regions = df_motif_locs["region"].unique()
    logger.info(f"Combinatorial ISA for {len(regions)} regions.")

    for i, region_str in enumerate(regions):
        df_reg = df_motif_locs[df_motif_locs["region"] == region_str].copy()
        if len(df_reg) < 2: continue
        
        chrom, coords = region_str.split(":")
        start_gen, end_gen = map(int, coords.split("-"))
        fasta = bf.load_fasta(fasta_path)
        seq_orig = str(fasta[chrom][start_gen:end_gen]).upper()
        
        df_reg["start_rel"] = df_reg["start"] - start_gen
        df_reg["end_rel"] = df_reg["end"] - start_gen
        
        pairs = []
        for idx1, idx2 in combinations(df_reg.index, 2):
            # combinations ensures that idx1 < idx2, so we won't have duplicate pairs in reverse order
            m1, m2 = df_reg.loc[idx1], df_reg.loc[idx2]
            # Skip physical overlaps
            if max(m1.start_rel, m2.start_rel) < min(m1.end_rel, m2.end_rel):
                continue
            
            dist = m2.start_rel - m1.start_rel
            if dist > receptive_field:
                continue
            
            pairs.append({
                "region": region_str,
                "tf1": m1.tf, "tf2": m2.tf,
                "start1": m1.start_rel, "end1": m1.end_rel,
                "start2": m2.start_rel, "end2": m2.end_rel,
                "distance": dist  
            })
        
        if not pairs: continue
        pair_df = pd.DataFrame(pairs)
        
        # Sequence generation for 4-state logic: Original, A, B, A+B
        seqs_m1 = [ablate_motifs(seq_orig, r.start1, r.end1) for r in pair_df.itertuples()]
        seqs_m2 = [ablate_motifs(seq_orig, r.start2, r.end2) for r in pair_df.itertuples()]
        seqs_both = [ablate_motifs(ablate_motifs(seq_orig, r.start1, r.end1), r.start2, r.end2) for r in pair_df.itertuples()]

        p_orig = compute_predictions(model, [seq_orig], device=device) 
        p_m1 = compute_predictions(model, seqs_m1, device=device)
        p_m2 = compute_predictions(model, seqs_m2, device=device)
        p_both = compute_predictions(model, seqs_both, device=device)
        
        # determine tracks
        track_idx = track_idx if isinstance(track_idx, list) else [track_idx]
        for t in track_idx:
            # Calculate raw ISA values
            pair_df[f"isa1_t{t}"] = p_orig[0, t] - p_m1[:, t]
            pair_df[f"isa2_t{t}"] = p_orig[0, t] - p_m2[:, t]
            pair_df[f"isa_both_t{t}"] = p_orig[0, t] - p_both[:, t]
            # Synergy calculation: ISA(A+B) - [ISA(A) + ISA(B)]
            pair_df[f"interaction_t{t}"] = (pair_df[f"isa1_t{t}"] + pair_df[f"isa2_t{t}"]) - pair_df[f"isa_both_t{t}"] 

        header = not os.path.exists(outpath)
        pair_df.to_csv(outpath, mode='a', index=False, header=header)
    
    logger.info(f"Combinatorial ISA complete. Results saved to {outpath}")





def _filter_valid_interaction(df, t):
    """
    Filters out rows where the TFs aren't acting as activators 
    or the interaction isn't functionally valid.
    """
    isa1 = df[f"isa1_t{t}"]
    isa2 = df[f"isa2_t{t}"]
    isa_both = df[f"isa_both_t{t}"]
    valid_mask = (isa1 > 0) & (isa2 > 0) & ((isa_both - isa1) > 0)
    return df[valid_mask].copy()




def calc_coop_score(df, outpath, 
                    receptive_field=255, 
                    track_idx=0, 
                    level='tf_pair', 
                    dist_min=100,
                    min_count=10):
    """
    Unified logic for Pair-level or TF-level distributional analysis.
    """
    # 1. Handle File Path Input
    df = load_data(df)
    # sort alphabetically here
    df["tf1"], df["tf2"] = np.minimum(df["tf1"], df["tf2"]), np.maximum(df["tf1"], df["tf2"])
    df = df.drop_duplicates()
    inter_col = f"interaction_t{track_idx}"
    
    # 1. Filter and Define Null Distribution
    df = _filter_valid_interaction(df, track_idx)
    null_vals = df[(df["distance"] > dist_min) & (df["distance"] <= receptive_field)][inter_col]
    
    if null_vals.empty:
        logger.warning(f"No null distribution for track {track_idx}")
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
        # mann whitney u test
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
            "mean_shift": round(vals.mean() - null_vals.mean(), 6),
            "count": len(vals),
            "mean_distance": group["distance"].mean()
        })

    res_df = pd.DataFrame(results)
    
    # 4. Statistical Correction and Assignment
    res_df["ks_q"] = multipletests(res_df["ks_p"], method='fdr_bh')[1]
    res_df["mw_q"] = multipletests(res_df["mw_p"], method='fdr_bh')[1]
    # res_df = assign_cooperativity(res_df)
    
    res_df.to_csv(outpath, mode='w', index=False)
    logger.info(f"Cooperativity stats saved to {outpath}")



# TODO: maybe top 25% of coop_score as synergistic, bottom 25% as redundant, rest as intermediate?
def assign_cooperativity(df, 
                         q_val_thresh=0.1, 
                         redun_thresh=-0.3, 
                         synergy_thresh=0.3):
    """
    Categorizes TF pairs based on FDR-corrected significance and intensity thresholds.
    """
    df["cooperativity"] = "Independent"
    is_significant = df["ks_q"] < q_val_thresh
    # Apply Directional Thresholds to significant pairs
    df.loc[is_significant & (df["coop_score"] > synergy_thresh), "cooperativity"] = "Synergistic"
    df.loc[is_significant & (df["coop_score"] < redun_thresh), "cooperativity"] = "Redundant"
    df.loc[is_significant & (df["coop_score"].between(redun_thresh, synergy_thresh)), "cooperativity"] = "Intermediate"
    # Nullify coop_score for Independent cases
    df.loc[df["cooperativity"] == "Independent", "coop_score"] = np.nan
    return format_cooperativity_categorical(df)






