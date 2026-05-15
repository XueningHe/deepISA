import os
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
    get_seq_from_fasta,
    ablate_motifs,
    format_cooperativity_categorical,
    write_stream_csv,
)





def make_pairs_for_region(region_motif_rows: pd.DataFrame,
                          receptive_field: int,
                          pass_cols: list[str]) -> pd.DataFrame | None:
    """
    Build a pair_df for ONE region (no ablations here).
    Required input cols on region_motif_rows: region, tf, start_rel, end_rel (+ pass_cols optional)
    Output cols: region, tf1, tf2, start1_rel, end1_rel, start2_rel, end2_rel, distance, tf1_*, tf2_*
    """
    if len(region_motif_rows) < 2:
        return None

    region_motif_rows = region_motif_rows.sort_values("start_rel")
    pairs = []
    for idx1, idx2 in combinations(region_motif_rows.index, 2):
        # sort + combinations make sure idx1 < idx2, so m1 is always left of m2 in the sequence
        m1, m2 = region_motif_rows.loc[idx1], region_motif_rows.loc[idx2]
        dist = m2.start_rel - m1.end_rel
        # skip overlapping motifs
        if dist < 1 or dist > receptive_field: continue
        pair_data = {
            "region": m1.region,
            "tf1": m1.tf, "tf2": m2.tf,
            "start1_rel": m1.start_rel, "end1_rel": m1.end_rel,
            "start2_rel": m2.start_rel, "end2_rel": m2.end_rel,
            "distance": dist,
        }
        for col in pass_cols:
            pair_data[f"tf1_{col}"] = m1[col]
            pair_data[f"tf2_{col}"] = m2[col]

        pairs.append(pair_data)

    if not pairs:
        return None

    return pd.DataFrame(pairs)




def build_combi_pairs_by_region(df_motif_locs: pd.DataFrame, receptive_field: int) -> dict:
    pairs_by_region = {}
    for region_str, region_motif_rows in df_motif_locs.groupby("region"):
        region_motif_rows = region_motif_rows.copy()
        pass_cols = [c for c in region_motif_rows.columns if c.startswith("pass_threshold_t")]
        pair_df = make_pairs_for_region(region_motif_rows, receptive_field, pass_cols)
        if pair_df is None or pair_df.empty:
            continue
        pairs_by_region[region_str] = pair_df

    return pairs_by_region




def get_ablated_seqs(seq_orig: str, pair_df: pd.DataFrame):
    seqs_m1 = [ablate_motifs(seq_orig, r.start1_rel, r.end1_rel) for r in pair_df.itertuples()]
    seqs_m2 = [ablate_motifs(seq_orig, r.start2_rel, r.end2_rel) for r in pair_df.itertuples()]
    seqs_both = [ablate_motifs(seq_orig, [r.start1_rel, r.start2_rel], [r.end1_rel, r.end2_rel]) for r in pair_df.itertuples()]
    return seqs_m1, seqs_m2, seqs_both





def score_pairs_minibatched(
    model,
    device,
    tracks,
    fasta,
    regions,
    pairs_by_region,            # dict[str, pd.DataFrame]
    outpath,
    num_regions_per_batch=200,
    pred_batch_size=1024,
):
    """
    Core minibatched scoring engine.
    pairs_by_region[region_str] ->

    Steps:
      1) For each minibatch of regions, build flat lists of ablated sequences across regions
      2) Predict on those flat lists to maximize GPU utilization
      3) Compute ISA/interaction per track
      4) Stream results to disk to control RAM
    """
    # CHANGED: use shared cleanup helper
    remove_if_exists(outpath)

    regions = list(regions)

    for batch_start in range(0, len(regions), num_regions_per_batch):
        batch_end = min(batch_start + num_regions_per_batch, len(regions))
        logger.info(f"Processing regions {batch_start}-{batch_end} / {len(regions)}")
        batch_regions = regions[batch_start:batch_end]

        pair_dfs = []
        pair_offsets = []  # (start, n) into the flat ablated arrays
        all_seqs_m1, all_seqs_m2, all_seqs_both = [], [], []
        orig_seqs, orig_region_labels = [], []

        # Build minibatch payloads
        for region_str in batch_regions:
            pair_df = pairs_by_region.get(region_str)
            if pair_df is None or pair_df.empty: continue
            seq_orig = get_seq_from_fasta(fasta, region_str)
            # CHANGED: reuse shared ablation builder
            seqs_m1, seqs_m2, seqs_both = get_ablated_seqs(seq_orig, pair_df)
            pair_offsets.append((len(all_seqs_m1), len(pair_df)))
            all_seqs_m1.extend(seqs_m1)
            all_seqs_m2.extend(seqs_m2)
            all_seqs_both.extend(seqs_both)
            orig_seqs.append(seq_orig)
            orig_region_labels.append(region_str)
            pair_dfs.append(pair_df)
        if not pair_dfs:continue

        # Predict ablated (big flat lists)
        p_m1 = compute_predictions(model, all_seqs_m1, device=device, batch_size=pred_batch_size)
        p_m2 = compute_predictions(model, all_seqs_m2, device=device, batch_size=pred_batch_size)
        p_both = compute_predictions(model, all_seqs_both, device=device, batch_size=pred_batch_size)

        # Predict orig (one per region)
        p_orig = compute_predictions(model, orig_seqs, device=device, batch_size=pred_batch_size)

        # CHANGED: avoid merge; map region -> p_orig row
        orig_map = {reg: p_orig[i, :] for i, reg in enumerate(orig_region_labels)}

        # Assemble and write per-region
        for pair_df, (start, n) in zip(pair_dfs, pair_offsets):
            sl = slice(start, start + n)
            pair_df = pair_df.copy()

            region_val = pair_df["region"].iloc[0]
            p0 = orig_map[region_val]

            # compute per track
            for t in tracks:
                p_orig_t = p0[t]
                pair_df[f"isa1_t{t}"] = p_orig_t - p_m1[sl, t]
                pair_df[f"isa2_t{t}"] = p_orig_t - p_m2[sl, t]
                pair_df[f"isa_both_t{t}"] = p_orig_t - p_both[sl, t]
                pair_df[f"interaction_t{t}"] = (pair_df[f"isa1_t{t}"] + pair_df[f"isa2_t{t}"]) - pair_df[f"isa_both_t{t}"]

            write_stream_csv(pair_df, outpath)




def run_combi_isa(
    model,
    fasta_path,
    motif_locs_path,
    outpath,
    device,
    receptive_field,
    tracks=[0],
    num_regions_per_batch=200,
    pred_batch_size=1024,
):
    remove_if_exists(outpath)
    df_motif_locs = pd.read_csv(motif_locs_path)
    if df_motif_locs.empty:
        logger.warning("No motifs. Try lowering attr_percentile or motif_score_thresh.")
        return None

    logger.info(f"Perform combinatorial ISA. Loaded motif locations from {motif_locs_path}...")
    fasta = bf.load_fasta(fasta_path)

    # CHANGED: precompute all pairs once
    pairs_by_region = build_combi_pairs_by_region(df_motif_locs, receptive_field)
    regions = list(pairs_by_region.keys())

    score_pairs_minibatched(
        model=model,
        device=device,
        tracks=tracks,
        fasta=fasta,
        regions=regions,
        pairs_by_region=pairs_by_region,
        outpath=outpath,
        num_regions_per_batch=num_regions_per_batch,
        pred_batch_size=pred_batch_size,
    )

    logger.info(f"Combinatorial ISA complete. Results saved to {outpath}")


def _filter_pos_interaction(df, t):
    isa1 = df[f"isa1_t{t}"]
    isa2 = df[f"isa2_t{t}"]
    isa_both = df[f"isa_both_t{t}"]
    valid_mask = (isa1 > 0) & (isa2 > 0) & (isa_both > isa1) & (isa_both > isa2)
    return df[valid_mask].copy()




def generate_null_pairs(
    non_motif_df_path,
    target_distances,
    k=9,
    n_samples=2000,
    max_dist=255,
    n_bins=60,
):
    """
    Hybrid Null Generation:
    1. Fixed centers for short gaps.
    2. Combinatorial inter-gap pairing.
    3. Distribution matching for intra-gap sampling in long regions.
    """
    non_motif_df = pd.read_csv(non_motif_df_path)
    non_motif_df["length"] = non_motif_df["end_rel"] - non_motif_df["start_rel"]
    df = non_motif_df[non_motif_df["length"] >= k].copy()
    if df.empty:
        return pd.DataFrame()

    all_possible_inter_pairs = []
    logger.info("Generating Inter-gap combinatorial anchors...")
    for reg_id, group in df.groupby("region"):
        if len(group) < 2: 
            continue
        
        group = group.sort_values(["start_rel", "end_rel"]).reset_index(drop=True)
        for i, j in combinations(range(len(group)), 2):
            gap1 = group.iloc[i]
            gap2 = group.iloc[j]
            s1 = int(gap1["start_rel"])
            e1 = int(s1 + k)
            s2 = int(gap2["start_rel"])
            e2 = int(s2 + k)
            dist = s2 - e1
            if dist <= 0:
                continue
            all_possible_inter_pairs.append(
                {
                    "region": reg_id,
                    "start1_rel": s1,
                    "end1_rel": e1,
                    "start2_rel": s2,
                    "end2_rel": e2,
                    "distance": dist,
                }
            )

    pool_df = pd.DataFrame(all_possible_inter_pairs)
    if pool_df.empty:
        return pd.DataFrame()

    # bins as ints
    bins = np.linspace(0, max_dist, n_bins + 1).astype(int)

    target_distances = np.asarray(target_distances)
    target_counts, _ = np.histogram(target_distances, bins=bins)
    if target_counts.sum() == 0:
        return pd.DataFrame()

    target_counts = (target_counts / target_counts.sum() * n_samples).astype(int)

    final_nulls = []
    logger.info("Matching distribution and filling gaps with intra-gap sampling...")

    for i in range(len(bins) - 1):
        count_needed = int(target_counts[i])
        if count_needed <= 0:
            continue

        lo, hi = int(bins[i]), int(bins[i + 1])

        mask = (pool_df["distance"] >= lo) & (pool_df["distance"] < hi)
        available_inter = pool_df[mask]

        if len(available_inter) >= count_needed:
            final_nulls.append(available_inter.sample(count_needed))
            continue

        final_nulls.append(available_inter)
        remaining = count_needed - len(available_inter)

        # Long gaps must be able to host 2*k + distance
        long_gaps = df[df["length"] >= (2 * k + lo)]
        if long_gaps.empty:
            continue

        sampled_intra = []
        for _ in range(remaining):
            gap = long_gaps.sample(1).iloc[0]
            max_d = min(hi, int(gap["length"] - 2 * k))
            if max_d <= lo: continue

            d = np.random.randint(lo, max_d)
            s1_min = int(gap["start_rel"])
            s1_max = int(gap["end_rel"] - (2 * k + d))
            if s1_max <= s1_min:
                continue

            s1 = np.random.randint(s1_min, s1_max)
            e1 = int(s1 + k)
            s2 = int(e1 + d)
            e2 = int(s2 + k)

            sampled_intra.append(
                {
                    "region": gap["region"],
                    "start1_rel": int(s1),
                    "end1_rel": int(e1),
                    "start2_rel": int(s2),
                    "end2_rel": int(e2),
                    "distance": int(d),
                }
            )
        if sampled_intra:
            final_nulls.append(pd.DataFrame(sampled_intra))

    if not final_nulls:
        return pd.DataFrame()

    return pd.concat(final_nulls, ignore_index=True)





def run_null_combi_isa(
    model,
    fasta_path,
    non_motif_locs_path,
    motif_combi_isa_path,
    outpath,
    device,
    tracks=[0],
    k=9,
    n_samples=2000,
    num_regions_per_batch=200,
    pred_batch_size=1024,
    seed=1337,
):
    np.random.seed(seed)
    remove_if_exists(outpath, label="null ISA results file")

    logger.info(f"Generating null pairs (k={k}, n_samples={n_samples}) from {non_motif_locs_path} ...")
    
    df_combi_isa = pd.read_csv(motif_combi_isa_path)
    target_distances = df_combi_isa["distance"].dropna().to_numpy()
    null_pairs_df = generate_null_pairs(
        non_motif_locs_path,
        np.asarray(target_distances),
        k=k,
        n_samples=n_samples,
    )

    if null_pairs_df.empty:
        logger.warning("generate_null_pairs_from_df produced no null pairs; nothing to score.")
        return None

    pairs_by_region = {r: g.copy() for r, g in null_pairs_df.groupby("region")}
    regions = list(pairs_by_region.keys())
    fasta = bf.load_fasta(fasta_path)

    score_pairs_minibatched(
        model=model,
        device=device,
        tracks=tracks,
        fasta=fasta,
        regions=regions,
        pairs_by_region=pairs_by_region,
        outpath=outpath,
        num_regions_per_batch=num_regions_per_batch,
        pred_batch_size=pred_batch_size,
    )

    logger.info(f"Null ISA complete. Results saved to {outpath}")
    return outpath







def calc_coop_score(
    combi_isa_path,
    null_isa_path,
    outpath,
    level,  # 'tf_pair' or 'tf'
    percentile=90,
    track_idx=0,
    min_count=10,
    q_val_thresh=0.1,
):
    remove_if_exists(outpath, label="cooperativity score file")

    df = pd.read_csv(combi_isa_path)
    t_pass_col = f"pass_threshold_t{track_idx}"
    tf1_check = f"tf1_{t_pass_col}"
    tf2_check = f"tf2_{t_pass_col}"

    initial_len = len(df)
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

    # sort TF names alphabetically within row
    df["tf1"], df["tf2"] = np.minimum(df["tf1"], df["tf2"]), np.maximum(df["tf1"], df["tf2"])
    df = df.drop_duplicates()

    df = _filter_pos_interaction(df, track_idx)
    if df.empty:
        logger.warning("No pairs remaining after positive-interaction filtering.")
        return pd.DataFrame()

    df_null = pd.read_csv(null_isa_path, usecols=[inter_col])
    null_vals = df_null[inter_col].dropna().to_numpy()
    null_pos_vals = null_vals[null_vals > 0]
    pos_thresh = np.percentile(null_pos_vals, percentile) if len(null_pos_vals) > 0 else 0
    null_neg_vals = null_vals[null_vals < 0]
    neg_thresh = np.percentile(null_neg_vals, 100 - percentile) if len(null_neg_vals) > 0 else 0
    
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
        mw_res = mannwhitneyu(vals, null_vals)
        # remove gray zone value
        vals = vals[(vals > pos_thresh) | (vals < neg_thresh)]
        if len(vals) < min_count: continue
        coop_score = vals.sum() / np.abs(vals).sum()
        results.append(
            {
                level: name,
                "abs_i_sum": round(np.abs(vals).sum(), 4),
                "coop_score": round(coop_score, 4),
                "mw_p": round(mw_res.pvalue, 4),
                "count": len(vals),
                "mean_distance": group["distance"].mean(),
            }
        )

    res_df = pd.DataFrame(results)

    if res_df.empty:
        logger.warning("No groups met min_count; no cooperativity results written.")
        return res_df

    res_df["mw_q"] = multipletests(res_df["mw_p"], method="fdr_bh")[1]
    res_df = assign_cooperativity(res_df, q_val_thresh)

    res_df.to_csv(outpath, mode="w", index=False)
    logger.info(f"Coop score saved to {outpath}")
    return res_df






def assign_cooperativity(df, q_val_thresh):
    df = df.copy()
    df["cooperativity"] = "Independent"
    is_significant = df["mw_q"] < q_val_thresh

    # CHANGED: handle no-significant case
    if is_significant.sum() == 0:
        df.loc[df["cooperativity"] == "Independent", "coop_score"] = np.nan
        return format_cooperativity_categorical(df)

    synergy_thresh = df.loc[is_significant, "coop_score"].quantile(0.7)
    redun_thresh = df.loc[is_significant, "coop_score"].quantile(0.3)

    df.loc[is_significant & (df["coop_score"] > synergy_thresh), "cooperativity"] = "Synergistic"
    df.loc[is_significant & (df["coop_score"] < redun_thresh), "cooperativity"] = "Redundant"
    df.loc[is_significant & (df["coop_score"].between(redun_thresh, synergy_thresh)), "cooperativity"] = "Intermediate"

    df.loc[df["cooperativity"] == "Independent", "coop_score"] = np.nan
    return format_cooperativity_categorical(df)




