import json
import pandas as pd
import numpy as np
from loguru import logger
from itertools import combinations
import bioframe as bf
from statsmodels.stats.multitest import multipletests
# mann-whitney U test
from scipy.stats import mannwhitneyu

# Internal imports
from deepISA.modeling.predict import compute_predictions
from deepISA.utils import (
    remove_if_exists,
    write_stream_csv,
)

from deepISA.scoring.utils_isa import (
    load_pred_orig, 
    ablate_motifs, 
    region_str_to_seq
)

from deepISA.scoring.null import generate_null_pairs, derive_null_thresholds



def make_pairs_for_region(
    region_motif_rows: pd.DataFrame,
    receptive_field: int,
    isa_cols: list[str],
) -> pd.DataFrame | None:
    """
    Build pair_df for ONE region.
    Required cols: region, tf, start_rel, end_rel, isa_t*
    Output includes isa1_t*, isa2_t* copied from motif_single_isa rows.
    """
    if len(region_motif_rows) < 2:
        return None

    region_motif_rows = region_motif_rows.sort_values("start_rel")
    pairs = []
    for idx1, idx2 in combinations(region_motif_rows.index, 2):
        m1, m2 = region_motif_rows.loc[idx1], region_motif_rows.loc[idx2]
        dist = m2.start_rel - m1.end_rel
        if dist < 1 or dist > receptive_field: continue
        pair_data = {
            "region": m1.region,
            "tf1": m1.tf,
            "tf2": m2.tf,
            "start1_rel": m1.start_rel,
            "end1_rel": m1.end_rel,
            "start2_rel": m2.start_rel,
            "end2_rel": m2.end_rel,
            "distance": dist,
        }
        for col in isa_cols:
            pair_data[f"isa1_{col.split('isa_')[-1]}"] = m1[col]  # isa1_t0, isa1_t1...
            pair_data[f"isa2_{col.split('isa_')[-1]}"] = m2[col]  # isa2_t0, isa2_t1...
        pairs.append(pair_data)
    if not pairs: return None
    
    return pd.DataFrame(pairs)





def build_combi_pairs_by_region(df_motif_single_isa: pd.DataFrame, receptive_field: int) -> dict:
    pairs_by_region = {}
    isa_cols = [c for c in df_motif_single_isa.columns if c.startswith("isa_t")]
    for region_str, region_motif_rows in df_motif_single_isa.groupby("region"):
        region_motif_rows = region_motif_rows.copy()
        pair_df = make_pairs_for_region(region_motif_rows, receptive_field, isa_cols)
        if pair_df is None or pair_df.empty:
            continue
        pairs_by_region[region_str] = pair_df
    return pairs_by_region





def score_pairs(
    model,
    device,
    tracks,
    fasta,
    regions,
    pairs_by_region,            
    outpath,
    pred_orig_path,    
    num_regions_per_batch,
    pred_batch_size,
):
    remove_if_exists(outpath)
    orig_pred_map = load_pred_orig(pred_orig_path, tracks) 
    regions = list(regions)
    
    # determine compute_single_isa
    probe_df = next(df for df in pairs_by_region.values())
    single_isa_cols = [f"isa1_t{t}" for t in tracks] + [f"isa2_t{t}" for t in tracks]
    compute_single_isa = not all(c in probe_df.columns for c in single_isa_cols)

    for batch_start in range(0, len(regions), num_regions_per_batch):
        batch_end = min(batch_start + num_regions_per_batch, len(regions))
        logger.info(f"Processing regions {batch_start}-{batch_end} / {len(regions)}")
        batch_regions = regions[batch_start:batch_end]
        pair_dfs = []
        pair_offsets = []
        all_seqs_both = []
        if compute_single_isa:
            all_seqs_m1 = []
            all_seqs_m2 = []
        for region_str in batch_regions:
            pair_df = pairs_by_region.get(region_str)
            if pair_df is None or pair_df.empty: continue
            seq_orig = region_str_to_seq(fasta, region_str)
            seqs_both = [ablate_motifs(seq_orig, [r.start1_rel, r.start2_rel], [r.end1_rel, r.end2_rel]) for r in pair_df.itertuples()]
            pair_offsets.append((len(all_seqs_both), len(pair_df)))
            all_seqs_both.extend(seqs_both)
            pair_dfs.append(pair_df)
            if compute_single_isa:
                seqs_m1 = [ablate_motifs(seq_orig, [r.start1_rel], [r.end1_rel]) for r in pair_df.itertuples()]
                seqs_m2 = [ablate_motifs(seq_orig, [r.start2_rel], [r.end2_rel]) for r in pair_df.itertuples()]
                all_seqs_m1.extend(seqs_m1)
                all_seqs_m2.extend(seqs_m2)

        if not pair_dfs: continue
        
        p_both = compute_predictions(model, all_seqs_both, device=device, batch_size=pred_batch_size, tracks=tracks)
        if compute_single_isa:
            p_m1 = compute_predictions(model, all_seqs_m1, device=device, batch_size=pred_batch_size, tracks=tracks)
            p_m2 = compute_predictions(model, all_seqs_m2, device=device, batch_size=pred_batch_size, tracks=tracks)

        for pair_df, (start, n) in zip(pair_dfs, pair_offsets):
            sl = slice(start, start + n)
            pair_df = pair_df.copy()
            region_val = pair_df["region"].iloc[0]
            p_orig = orig_pred_map[region_val]  
            for j, t in enumerate(tracks):
                pair_df[f"isa_both_t{t}"] = p_orig[j] - p_both[sl, j]
                if compute_single_isa:
                    pair_df[f"isa1_t{t}"] = p_orig[j] - p_m1[sl, j]
                    pair_df[f"isa2_t{t}"] = p_orig[j] - p_m2[sl, j]

            write_stream_csv(pair_df, outpath)



def run_combi_isa(
    model,
    fasta,
    single_isa_path,
    outpath,
    device,
    receptive_field,
    pred_orig_path, 
    tracks=[0],
    num_regions_per_batch=200,
    pred_batch_size=1024,
):
    remove_if_exists(outpath)
    
    if isinstance(fasta, str):
        fasta=bf.load_fasta(fasta)

    df_motif_single_isa = pd.read_csv(single_isa_path)
    if df_motif_single_isa.empty:
        logger.warning("No motifs in motif_single_isa file.")
        return None

    logger.info(f"Perform combinatorial ISA from motif_single_isa: {single_isa_path}")
    pairs_by_region = build_combi_pairs_by_region(df_motif_single_isa, receptive_field)
    regions = list(pairs_by_region.keys())
    score_pairs(
        model=model,
        device=device,
        tracks=tracks,
        fasta=fasta,
        regions=regions,
        pairs_by_region=pairs_by_region,
        outpath=outpath,
        pred_orig_path=pred_orig_path,
        num_regions_per_batch=num_regions_per_batch,
        pred_batch_size=pred_batch_size,
    )
    logger.info(f"Combinatorial ISA complete. Results saved to {outpath}")





def run_null_interaction(
    model,
    fasta,
    non_motif_locs_path,
    combi_isa_path,
    pred_orig_path, 
    outpath,
    device,
    tracks=[0],
    n_samples=8192,
    num_regions_per_batch=200,
    pred_batch_size=1024,
    receptive_field=255,
    n_bins=20,
):
    remove_if_exists(outpath, label="null ISA results file")    
    df_combi_isa = pd.read_csv(combi_isa_path)
    target_distances = df_combi_isa["distance"].dropna().to_numpy()
    # get median length of motifs in combi_isa
    motif1_lengths = df_combi_isa["end1_rel"] - df_combi_isa["start1_rel"]
    motif2_lengths = df_combi_isa["end2_rel"] - df_combi_isa["start2_rel"]
    median_motif_length = int(np.median(np.concatenate([motif1_lengths, motif2_lengths])))
    null_pairs_df = generate_null_pairs(
        non_motif_locs_path,
        np.asarray(target_distances),
        receptive_field=receptive_field,
        k=median_motif_length,
        n_samples=n_samples,
        n_bins=n_bins,
    )
    logger.info(f"Generating null pairs (k={median_motif_length}, n_samples={n_samples}) from {non_motif_locs_path} ...")

    if null_pairs_df.empty:
        logger.warning("generate_null_pairs_from_df produced no null pairs; nothing to score.")
        return None

    pairs_by_region = {r: g.copy() for r, g in null_pairs_df.groupby("region")}
    regions = list(pairs_by_region.keys())
    if isinstance(fasta, str):
        fasta = bf.load_fasta(fasta)

    score_pairs(
        model=model,
        device=device,
        tracks=tracks,
        fasta=fasta,
        regions=regions,
        pairs_by_region=pairs_by_region,
        outpath=outpath,
        pred_orig_path=pred_orig_path,  
        num_regions_per_batch=num_regions_per_batch,
        pred_batch_size=pred_batch_size,
    )

    logger.info(f"Null ISA complete. Results saved to {outpath}")
    return outpath



#-------------------
# Aggregation functions
#-------------------

def compute_interaction_per_track(df: pd.DataFrame, 
                                  track_idx: int, 
                                  tau: float,
                                  isa_thresh=0) -> pd.DataFrame:
    isa1_col = f"isa1_t{track_idx}"
    isa2_col = f"isa2_t{track_idx}"
    both_col = f"isa_both_t{track_idx}"
    inter_col = f"interaction_t{track_idx}"
    
    isa1 = df[isa1_col].to_numpy(dtype=float)
    isa2 = df[isa2_col].to_numpy(dtype=float)
    both = df[both_col].to_numpy(dtype=float)

    isa1_wo2 = both - isa2
    isa2_wo1 = both - isa1
    qualified = (isa1 >= isa_thresh) & (isa2 >= isa_thresh) & (isa1_wo2 >= 0) & (isa2_wo1 >= 0)
    vals = np.full(df.shape[0], np.nan, dtype=float)
    denom = isa1 + isa2 + tau
    num = isa1 + isa2 - both
    valid = qualified & np.isfinite(num) & np.isfinite(denom) & (denom != 0)
    vals[valid] = num[valid] / denom[valid]
    df[inter_col] = vals




def add_interaction(
    combi_isa_path: str,
    null_interaction_path: str,
    null_isa_path: str,
    null_percentile: float,
    tracks=[0],
    tau_quantile: float = 50.0,
):

    df_null = pd.read_csv(null_interaction_path)
    df_obs = pd.read_csv(combi_isa_path)
    df_null_isa= pd.read_csv(null_isa_path)
    isa_thresh_map= derive_null_thresholds(df_null_isa,[f"isa_t{t}" for t in tracks],null_percentile)
    tau_map: dict[int, float] = {}
    # ---------- per-track processing ----------
    for t in tracks:
        isa1_col = f"isa1_t{t}"
        isa2_col = f"isa2_t{t}"
        # calculate null-calibrated tau 
        tau_mask = (df_null[isa1_col] > 0) & (df_null[isa2_col] > 0)
        den_for_tau = (df_null.loc[tau_mask, isa1_col] + df_null.loc[tau_mask, isa2_col]).to_numpy(dtype=float)
        tau_t = float(np.nanpercentile(den_for_tau, tau_quantile))
        tau_map[t] = tau_t
        logger.info(f"[track {t}] tau (q{tau_quantile}) = {tau_t:.4f}")
        compute_interaction_per_track(df_null, track_idx=t, tau=tau_t)
        compute_interaction_per_track(df_obs, track_idx=t, tau=tau_t, 
                                      isa_thresh=isa_thresh_map[f"isa_t{t}"]["pos"])
    # ---------- write back (same row count/order) ----------
    df_null.to_csv(null_interaction_path, index=False, float_format="%.4f", na_rep="")
    logger.info(f"Wrote normalized interactions to null file: {null_interaction_path}")
    df_obs.to_csv(combi_isa_path, index=False, float_format="%.4f",na_rep="")
    logger.info(f"Wrote normalized interactions to combi file: {combi_isa_path}")
    return tau_map






def calc_coop_score(
    combi_isa_path,
    null_interaction_path,
    outpath,
    level,  # 'tf_pair' or 'tf'
    null_percentile,
    track_idx=0,
    min_count=10,
    q_val_thresh=0.1,
):
    remove_if_exists(outpath, label="cooperativity score file")
    
    df = pd.read_csv(combi_isa_path)
    inter_col = f"interaction_t{track_idx}"
    
    #---------------
    # Format df
    #---------------
    # sort TF names alphabetically within row
    df["tf1"], df["tf2"] = np.minimum(df["tf1"], df["tf2"]), np.maximum(df["tf1"], df["tf2"])
    if level == "tf":
        df_melt = pd.concat(
            [
                df[["tf1", inter_col, "distance"]].rename(columns={"tf1": "name"}),
                df[["tf2", inter_col, "distance"]].rename(columns={"tf2": "name"}),
            ],
            ignore_index=True,
        )
    else:
        df_melt = df.copy()
        df_melt["name"] = df_melt["tf1"] + "|" + df_melt["tf2"]
    
    #-----------------------------
    # get null-derived thresholds
    #-----------------------------
    df_null_inter = pd.read_csv(null_interaction_path)
    # inter_thresh = derive_null_thresholds(df_null_inter,[inter_col],percentile=null_percentile)
    # logger.info(f"Null-derived thresholds for {inter_col}: pos={inter_thresh[inter_col]['pos']:.4f}, neg={inter_thresh[inter_col]['neg']:.4f}")
    # pos_inter_thresh = inter_thresh[inter_col]["pos"]
    # neg_inter_thresh = inter_thresh[inter_col]["neg"]
    null_interactions = df_null_inter[inter_col].dropna().to_numpy()
    
    results = []
    for name, group in df_melt.groupby("name"):
        interactions = group[inter_col].dropna().to_numpy()
        if len(interactions)< min_count: continue
        _, p_val = mannwhitneyu(interactions, null_interactions, alternative="two-sided")
        # remove gray zone value
        # interactions = interactions[(interactions > pos_inter_thresh) | (interactions < neg_inter_thresh)]
        # if len(interactions) < min_count: continue
        coop_score = interactions.sum() / np.abs(interactions).sum()
        results.append(
            {
                level: name,
                "n_total": len(group),
                "n_effective": len(interactions),
                "abs_i_sum": np.abs(interactions).sum(),
                "coop_score": coop_score,
                "p_val": p_val,
                "count": len(interactions),
                "median_distance": group["distance"].median()
            }
        )

    res_df = pd.DataFrame(results)
    if res_df.empty:
        logger.warning("No groups passed min_count/effective filters.")
        res_df.to_csv(outpath, index=False)
        return res_df
    res_df["q_val"] = multipletests(res_df["p_val"], method="fdr_bh")[1]
    res_df = assign_cooperativity(res_df, q_val_thresh)
    res_df.to_csv(outpath, mode="w", index=False, float_format="%.4f")
    logger.info(f"Coop score saved to {outpath}")
    return res_df






def assign_cooperativity(df, q_val_thresh=0.1):
    df = df.copy()
    df["cooperativity"] = "Independent"
    is_significant = df["q_val"] < q_val_thresh
    synergy_thresh = df.loc[is_significant, "coop_score"].quantile(0.7)
    redun_thresh = df.loc[is_significant, "coop_score"].quantile(0.3)
    df.loc[is_significant & (df["coop_score"] > synergy_thresh), "cooperativity"] = "Synergistic"
    df.loc[is_significant & (df["coop_score"] < redun_thresh), "cooperativity"] = "Redundant"
    df.loc[is_significant & (df["coop_score"].between(redun_thresh, synergy_thresh)), "cooperativity"] = "Intermediate"
    df.loc[df["cooperativity"] == "Independent", "coop_score"] = np.nan
    return df




