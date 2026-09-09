import json

import numpy as np
import pandas as pd
import bioframe as bf
from loguru import logger
from itertools import combinations

from deepISA.utils import remove_if_exists
from deepISA.score.combi_isa import combi_isa_core
from deepISA.score.single_isa import run_single_isa
from deepISA.score.aggregate_isa import calc_interaction


#---------------------------
# Thresholding utilities
#---------------------------

def derive_null_thresholds(null_df, cols, percentile):
    out = {}
    for c in cols:
        vals = null_df[c].dropna().to_numpy(dtype=float)
        pos  = vals[vals > 0]
        neg  = vals[vals < 0]
        out[c] = {
            "pos": np.percentile(pos, percentile),
            "neg": np.percentile(neg, 100 - percentile),
        }
    return out


#---------------------------
# Private sampling utilities
#---------------------------

def _allocate_counts_from_target_distribution(
    target_values: np.ndarray,
    n_samples: int,
    bins: np.ndarray,
) -> np.ndarray:
    target_values = np.asarray(target_values, dtype=float)
    hist, _  = np.histogram(target_values, bins=bins)
    frac     = hist / hist.sum()
    raw      = frac * int(n_samples)
    counts   = np.floor(raw).astype(int)
    remainder = int(n_samples) - counts.sum()
    if remainder > 0:
        order = np.argsort(raw - counts)[::-1]
        counts[order[:remainder]] += 1
    elif remainder < 0:
        order = np.argsort(raw - counts)
        for idx in order:
            if remainder == 0:
                break
            if counts[idx] > 0:
                counts[idx] -= 1
                remainder  += 1
    return counts


def _sample_null_kmers(non_motif_df, target_lengths, n_samples) -> pd.DataFrame:
    target_lengths = np.asarray(target_lengths)
    df = non_motif_df.copy()
    df["interval_len"] = df["end"] - df["start"]
    df = df[df["interval_len"] > 0].reset_index(drop=True)

    min_len = int(target_lengths.min())
    max_len = int(target_lengths.max())
    bins    = np.arange(min_len - 0.5, max_len + 1.5, 1.0)
    per_bin_counts = _allocate_counts_from_target_distribution(target_lengths, n_samples, bins)

    rng = np.random.default_rng(0)
    sampled_chunks = []
    for i, need in enumerate(per_bin_counts):
        if need <= 0:
            continue
        k = int(round((bins[i] + bins[i + 1]) / 2.0))
        if k <= 0:
            continue
        eligible    = df[df["interval_len"] >= k]
        chosen      = eligible.sample(n=need, replace=True, random_state=0).copy()
        high_bounds = chosen["interval_len"] - k + 1
        offsets     = rng.integers(0, high_bounds)
        chosen["start"]     = chosen["start"]     + offsets
        chosen["end"]       = chosen["start"]     + k
        chosen["start_rel"] = chosen["start_rel"] + offsets
        chosen["end_rel"]   = chosen["start_rel"] + k
        sampled_chunks.append(
            chosen[["chrom", "start", "end", "region", "start_rel", "end_rel"]]
        )

    out = pd.concat(sampled_chunks, ignore_index=True)
    if len(out) > n_samples:
        out = out.sample(n=n_samples, random_state=0).reset_index(drop=True)
    elif len(out) < n_samples and len(out) > 0:
        topup = out.sample(n=n_samples - len(out))
        out   = pd.concat([out, topup], ignore_index=True)
    return out.reset_index(drop=True)


def _generate_null_pairs(
    non_motif_df_path,
    target_distances,
    k,
    n_samples,
    n_bins,
) -> pd.DataFrame:
    """
    Hybrid null pair generation matched to target_distances distribution.
    receptive_field removed — upper bin edge derived from max(target_distances).
    """
    non_motif_df = pd.read_csv(non_motif_df_path)
    non_motif_df["length"] = non_motif_df["end_rel"] - non_motif_df["start_rel"]
    df = non_motif_df[non_motif_df["length"] >= k].copy()
    if df.empty:
        return pd.DataFrame()

    # CHANGED: upper bin edge = max(target_distances), not receptive_field
    bins = np.linspace(0, int(target_distances.max()), n_bins + 1).astype(int)
    target_distances = np.asarray(target_distances)
    target_counts = _allocate_counts_from_target_distribution(target_distances, n_samples, bins)

    # --- inter-gap combinatorial pool ---
    logger.info("Generating inter-gap combinatorial anchors...")
    all_inter = []
    for reg_id, group in df.groupby("region"):
        if len(group) < 2:
            continue
        group = group.sort_values(["start_rel", "end_rel"]).reset_index(drop=True)
        for i, j in combinations(range(len(group)), 2):
            g1, g2 = group.iloc[i], group.iloc[j]
            s1 = int(g1["start_rel"]); e1 = s1 + k
            s2 = int(g2["start_rel"]); e2 = s2 + k
            dist = s2 - e1
            if dist <= 0:
                continue
            all_inter.append({
                "region": reg_id,
                "start1_rel": s1, "end1_rel": e1,
                "start2_rel": s2, "end2_rel": e2,
                "distance": dist,
            })
    pool_df = pd.DataFrame(all_inter)

    # --- distribution-matched sampling ---
    logger.info("Matching distance distribution, filling gaps with intra-gap sampling...")
    final_nulls = []
    for i in range(len(bins) - 1):
        need = int(target_counts[i])
        if need <= 0:
            continue
        lo, hi = int(bins[i]), int(bins[i + 1])
        mask    = (pool_df["distance"] >= lo) & (pool_df["distance"] < hi)
        avail   = pool_df[mask]

        if len(avail) >= need:
            final_nulls.append(avail.sample(need))
            continue

        final_nulls.append(avail)
        remaining  = need - len(avail)
        long_gaps  = df[df["length"] >= (2 * k + lo)]
        if long_gaps.empty:
            continue
        sampled_intra = []
        for _ in range(remaining):
            gap    = long_gaps.sample(1).iloc[0]
            max_d  = min(hi, int(gap["length"] - 2 * k))
            if max_d <= lo:
                continue
            d      = np.random.randint(lo, max_d)
            s1_min = int(gap["start_rel"])
            s1_max = int(gap["end_rel"] - (2 * k + d))
            if s1_max <= s1_min:
                continue
            s1 = np.random.randint(s1_min, s1_max)
            e1 = s1 + k; s2 = e1 + d; e2 = s2 + k
            sampled_intra.append({
                "region": gap["region"],
                "start1_rel": s1, "end1_rel": e1,
                "start2_rel": s2, "end2_rel": e2,
                "distance": d,
            })
        if sampled_intra:
            final_nulls.append(pd.DataFrame(sampled_intra))

    return pd.concat(final_nulls, ignore_index=True) if final_nulls else pd.DataFrame()


#---------------------------------------
# Private orchestration helpers
#---------------------------------------

def _calc_non_motif_isa(
    model, fasta,
    non_motif_locs_path: str,
    single_isa_path: str,
    pred_orig_path: str,
    outpath: str,
    device, tracks: list[int],
    num_regions_per_batch: int = 200,
    pred_batch_size: int = 1024,
    n_samples: int = 8192,
) -> str:
    remove_if_exists(outpath, label="non-motif ISA file")

    non_motif_df = pd.read_csv(non_motif_locs_path)
    df_single = pd.read_csv(single_isa_path)
    target_lengths = (df_single["end_rel"] - df_single["start_rel"]).to_numpy()

    null_kmers_df = _sample_null_kmers(non_motif_df, target_lengths, n_samples)
    logger.info(
        f"Running non-motif ISA on {len(null_kmers_df)} sampled k-mers "
        f"(lengths matched to {single_isa_path})"
    )
    null_kmers_df.to_csv(outpath + "_null_kmers.csv", index=False)
    run_single_isa(
        model=model, fasta=fasta, 
        motif_locs_path=outpath + "_null_kmers.csv",
        pred_orig_path=pred_orig_path, 
        outpath=outpath,
        device=device, 
        tracks=tracks,
        num_regions_per_batch=num_regions_per_batch,
        pred_batch_size=pred_batch_size,
    )
    logger.info(f"Non-motif ISA written to {outpath}")
    return outpath




def _calc_non_motif_interaction(
    model, fasta,
    non_motif_locs_path: str,
    combi_isa_path: str,
    pred_orig_path: str,
    outpath: str,
    device, tracks: list[int],
    pred_batch_size: int = 1024,
    num_regions_per_batch: int = 200,
    n_samples: int = 2048,
    n_bins: int = 20,
) -> str:
    """Score non-motif pairs matched to combi ISA distance distribution."""
    remove_if_exists(outpath, label="non-motif interaction file")

    if isinstance(fasta, str):
        fasta = bf.load_fasta(fasta)

    df_combi = pd.read_csv(combi_isa_path)
    target_distances = df_combi["distance"].dropna().to_numpy()
    median_motif_len = int(np.median(np.concatenate([
        df_combi["end1_rel"] - df_combi["start1_rel"],
        df_combi["end2_rel"] - df_combi["start2_rel"],
    ])))

    logger.info(
        f"Generating non-motif pairs (k={median_motif_len}, n_samples={n_samples})"
    )

    null_pairs_df = _generate_null_pairs(
        non_motif_df_path=non_motif_locs_path,
        target_distances=target_distances,
        k=median_motif_len,
        n_samples=n_samples,
        n_bins=n_bins,
    )

    if null_pairs_df.empty:
        logger.warning("_generate_null_pairs produced no pairs; nothing to score.")
        return outpath


    # Build pairs by region (same as in run_combi_isa)
    batch_pairs = {}
    for region_str, group in null_pairs_df.groupby("region"):
        batch_pairs[region_str] = group.reset_index(drop=True)

    logger.info(f"Scoring {len(null_pairs_df)} non-motif pairs across {len(batch_pairs)} regions")

    combi_isa_core(
        model=model,
        device=device,
        tracks=tracks,
        fasta=fasta,
        pairs_by_region=batch_pairs,
        outpath=outpath,
        pred_orig_path=pred_orig_path,
        num_regions_per_batch=num_regions_per_batch,
        pred_batch_size=pred_batch_size,
    )

    logger.info(f"Non-motif interaction written to {outpath}")
    return outpath


def _derive_tau_map(
    non_motif_interaction_path: str,  # ← Now .csv
    tracks: list[int],
    tau_quantile: float = 50.0,
) -> dict[int, float]:
    tau_map = {}
    for t in tracks:
        cols = [f"isa1_t{t}", f"isa2_t{t}"]
        df_t = pd.read_csv(non_motif_interaction_path, usecols=cols)  # ← Use read_csv
        mask = (df_t[f"isa1_t{t}"] > 0) & (df_t[f"isa2_t{t}"] > 0)
        den  = (df_t.loc[mask, f"isa1_t{t}"] + df_t.loc[mask, f"isa2_t{t}"]).to_numpy(float)
        tau_map[t] = float(np.nanpercentile(den, tau_quantile))
        logger.info(f"[track {t}] tau (q{tau_quantile}) = {tau_map[t]:.4f}")
    return tau_map


#---------------------------------------
# Public entry point
#---------------------------------------

def calc_non_motif_stats(
    model, fasta,
    non_motif_locs_path: str,
    single_isa_path: str,
    combi_isa_path: str,
    pred_orig_path: str,
    non_motif_isa_outpath: str,
    non_motif_interaction_outpath: str,
    device, 
    tracks: list[int],
    pred_batch_size: int = 1024,
    num_regions_per_batch: int = 200,
    n_samples: int = 8192,
    n_bins: int = 20,
    tau_quantile: float = 50.0,
) -> tuple[str, str, dict[int, float]]:
    """
    Full non-motif background pipeline:
      1. Score non-motif k-mers  -> non_motif_isa.csv
      3. Derive tau from step 2

    Returns
    -------
    (non_motif_isa_outpath, non_motif_interaction_outpath, tau_map)
    """
    if isinstance(fasta, str):
        fasta = bf.load_fasta(fasta)

    isa_path = _calc_non_motif_isa(
        model=model, fasta=fasta,
        non_motif_locs_path=non_motif_locs_path,
        single_isa_path=single_isa_path,
        pred_orig_path=pred_orig_path,
        outpath=non_motif_isa_outpath,
        device=device, tracks=tracks,
        pred_batch_size=pred_batch_size,
        n_samples=n_samples,
    )

    inter_path = _calc_non_motif_interaction(
        model=model, fasta=fasta,
        non_motif_locs_path=non_motif_locs_path,
        combi_isa_path=combi_isa_path,
        pred_orig_path=pred_orig_path,
        outpath=non_motif_interaction_outpath,
        device=device, tracks=tracks,
        pred_batch_size=pred_batch_size,
        num_regions_per_batch=num_regions_per_batch,
        n_samples=n_samples,
        n_bins=n_bins,
    )

    # Derive tau before adding interaction columns
    tau_map = _derive_tau_map(inter_path, tracks, tau_quantile)
    tau_json_path = non_motif_interaction_outpath.replace(".csv", "_tau.json")
    with open(tau_json_path, "w") as f:
        json.dump({str(k): v for k, v in tau_map.items()}, f, indent=2)
    logger.info(f"tau_map written to {tau_json_path}")
    # TODO: is it really normalized?
    calc_interaction(
        combi_isa_path=inter_path, 
        tracks=tracks,
        tau_map=tau_map,
        normalize=True,
    )
    logger.info("Non-motif interaction columns added. calc_non_motif_stats complete.")

    return isa_path, inter_path, tau_map