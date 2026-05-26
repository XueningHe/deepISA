import json
import pandas as pd
import numpy as np
from loguru import logger
from itertools import combinations
import bioframe as bf
from scipy.stats import mannwhitneyu
from statsmodels.stats.multitest import multipletests

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

from deepISA.scoring.null import generate_null_pairs, derive_null_thresholds, apply_threshold_filter



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

    if not pairs:
        return None

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

        if not pair_dfs:
            continue
        
        p_both = compute_predictions(model, all_seqs_both, device=device, batch_size=pred_batch_size, tracks=tracks)
        if compute_single_isa:
            p_m1 = compute_predictions(model, all_seqs_m1, device=device, batch_size=pred_batch_size, tracks=tracks)
            p_m2 = compute_predictions(model, all_seqs_m2, device=device, batch_size=pred_batch_size, tracks=tracks)

        for pair_df, (start, n) in zip(pair_dfs, pair_offsets):
            sl = slice(start, start + n)
            pair_df = pair_df.copy()
            region_val = pair_df["region"].iloc[0]
            p0 = orig_pred_map[region_val]  # ordered by tracks list

            for j, t in enumerate(tracks):
                p_orig_t = p0[j]
                pair_df[f"isa_both_t{t}"] = p_orig_t - p_both[sl, t]
                if compute_single_isa:
                    pair_df[f"isa1_t{t}"] = p_orig_t - p_m1[sl, t]
                    pair_df[f"isa2_t{t}"] = p_orig_t - p_m2[sl, t]
                pair_df[f"interaction_t{t}"] = (
                    pair_df[f"isa1_t{t}"] + pair_df[f"isa2_t{t}"] - pair_df[f"isa_both_t{t}"]
                )

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
    k=9,
    n_samples=2000,
    num_regions_per_batch=200,
    pred_batch_size=1024,
    receptive_field=255,
    n_bins=20,
):
    remove_if_exists(outpath, label="null ISA results file")

    logger.info(f"Generating null pairs (k={k}, n_samples={n_samples}) from {non_motif_locs_path} ...")
    
    df_combi_isa = pd.read_csv(combi_isa_path)
    target_distances = df_combi_isa["distance"].dropna().to_numpy()
    null_pairs_df = generate_null_pairs(
        non_motif_locs_path,
        np.asarray(target_distances),
        receptive_field=receptive_field,
        k=k,
        n_samples=n_samples,
        n_bins=n_bins,
    )

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

def calc_coop_score(
    combi_isa_path,
    null_isa_path,
    null_interaction_path,
    outpath,
    level,  # 'tf_pair' or 'tf'
    null_percentile,
    track_idx=0,
    min_count=10,
    q_val_thresh=0.1,
):
    remove_if_exists(outpath, label="cooperativity score file")
    
    # read
    df = pd.read_csv(combi_isa_path)
    shape_before = df.shape[0]
    # derive ISA thresholds from null_isa
    df_null_isa= pd.read_csv(null_isa_path)
    null_isa_col = f"isa_t{track_idx}"
    noise_thresh_map = derive_null_thresholds(
        null_df=df_null_isa,
        cols=[null_isa_col],
        percentile=null_percentile,
    )
    noise_thresh = noise_thresh_map[null_isa_col]["pos"]
    # sort TF names alphabetically within row
    df["tf1"], df["tf2"] = np.minimum(df["tf1"], df["tf2"]), np.maximum(df["tf1"], df["tf2"])
    df = df.drop_duplicates()
    # filter pairs where either motif is below noise threshold or conditional ISA is negative
    isa1_col = f"isa1_t{track_idx}"
    isa2_col = f"isa2_t{track_idx}"
    isa_both_col = f"isa_both_t{track_idx}"
    df["isa1_wo2"] = df[isa_both_col] - df[isa1_col]
    df["isa2_wo1"] = df[isa_both_col] - df[isa2_col]
    all_filter_cols = [isa1_col, isa2_col, "isa1_wo2", "isa2_wo1"]
    all_thresholds = {
        isa1_col: noise_thresh,   
        isa2_col: noise_thresh,   
        "isa1_wo2": 0.0,          
        "isa2_wo1": 0.0,          
    }
    df, _ = apply_threshold_filter(
        df=df,
        cols=all_filter_cols,
        thresholds=all_thresholds,
        rule="all_above",
    )
    logger.info(
        f"Combined filtering (ISA >= {noise_thresh:.4f}, conditional ISA >= 0) "
        f"reduced pairs from {shape_before} to {df.shape[0]}"
    )
    if df.empty:
        logger.warning(f"No pairs remaining after motif ISA filtering for track {track_idx}.")
        return pd.DataFrame()
    
    df_null_inter = pd.read_csv(null_interaction_path)
    inter_col = f"interaction_t{track_idx}"
    null_interaction = df_null_inter[inter_col].dropna().to_numpy()
    thresholds = derive_null_thresholds(
        null_df=df_null_inter,
        cols=[inter_col],
        percentile=null_percentile,
    )
    pos_thresh = thresholds[inter_col]["pos"]
    neg_thresh = thresholds[inter_col]["neg"]
    
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

    results = []
    for name, group in df_melt.groupby("name"):
        if len(group) < min_count: continue
        vals = group[inter_col].to_numpy()
        mw_res = mannwhitneyu(vals, null_interaction)
        # remove gray zone value
        vals = vals[(vals > pos_thresh) | (vals < neg_thresh)]
        if len(vals) < min_count: continue
        coop_score = vals.sum() / np.abs(vals).sum()
        results.append(
            {
                level: name,
                "n_total": len(group),
                "n_effective": len(vals),
                "abs_i_sum": np.abs(vals).sum(),
                "coop_score": coop_score,
                "mw_p": mw_res.pvalue,
                "count": len(vals),
                "median_distance": group["distance"].median()
            }
        )

    res_df = pd.DataFrame(results)

    if res_df.empty:
        logger.warning("No groups met min_count; no cooperativity results written.")
        return res_df

    res_df["mw_q"] = multipletests(res_df["mw_p"], method="fdr_bh")[1]
    res_df = assign_cooperativity(res_df, q_val_thresh)

    res_df.to_csv(outpath, mode="w", index=False, float_format="%.4f")
    logger.info(f"Coop score saved to {outpath}")
    return res_df






def assign_cooperativity(df, q_val_thresh):
    df = df.copy()
    df["cooperativity"] = "Independent"
    is_significant = df["mw_q"] < q_val_thresh

    synergy_thresh = df.loc[is_significant, "coop_score"].quantile(0.7)
    redun_thresh = df.loc[is_significant, "coop_score"].quantile(0.3)
    df.loc[is_significant & (df["coop_score"] > synergy_thresh), "cooperativity"] = "Synergistic"
    df.loc[is_significant & (df["coop_score"] < redun_thresh), "cooperativity"] = "Redundant"
    df.loc[is_significant & (df["coop_score"].between(redun_thresh, synergy_thresh)), "cooperativity"] = "Intermediate"

    df.loc[df["cooperativity"] == "Independent", "coop_score"] = np.nan
    return df




