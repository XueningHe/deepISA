import numpy as np
import pandas as pd
from itertools import combinations
from loguru import logger

#---------------------------
# Thresholding utilities
#---------------------------


def derive_null_thresholds(
    null_df,
    cols,
    percentile
):
    out = {}
    for c in cols:
        vals = null_df[c].dropna().to_numpy(dtype=float)
        pos = vals[vals > 0]
        neg = vals[vals < 0]
        out[c] = {
            "pos": np.percentile(pos, percentile),
            "neg": np.percentile(neg, 100 - percentile),
        }
    return out



def apply_threshold_filter(
    df,
    cols,
    thresholds,
    rule,   # "any_above" | "all_above"
):

    if rule == "all_above":
        mask = np.ones(len(df), dtype=bool)
        for c in cols:
            mask &= (df[c] > thresholds[c])
        return df[mask].copy(), mask
    
    if rule == "any_tails":
        mask = np.zeros(len(df), dtype=bool)
        for c in cols:
            t = thresholds[c]
            mask |= ((df[c] > t["pos"]) | (df[c] < t["neg"]))
        return df[mask].copy(), mask

    raise ValueError(f"Unknown rule: {rule}")




#---------------------------
# Null generation utilities
#---------------------------

def _allocate_counts_from_target_distribution(
    target_values: np.ndarray,
    n_samples: int,
    bins: np.ndarray,
) -> np.ndarray:
    """
    Convert target-value histogram into integer sample counts per bin that sum to n_samples.
    """
    target_values = np.asarray(target_values, dtype=float)
    hist, _ = np.histogram(target_values, bins=bins)
    frac = hist / hist.sum()
    raw = frac * int(n_samples)
    counts = np.floor(raw).astype(int)
    # Distribute remainder by largest fractional parts to guarantee exact sum.
    remainder = int(n_samples) - counts.sum()
    if remainder > 0:
        order = np.argsort(raw - counts)[::-1]
        counts[order[:remainder]] += 1
    elif remainder < 0:
        order = np.argsort(raw - counts)  # remove from smallest fractions first
        for idx in order:
            if remainder == 0:
                break
            if counts[idx] > 0:
                counts[idx] -= 1
                remainder += 1

    return counts


def sample_null_kmers(
    non_motif_df,
    target_lengths, 
    n_samples,
) -> pd.DataFrame:

    target_lengths = np.asarray(target_lengths)
    df = non_motif_df.copy()
    df["interval_len"] = df["end"] - df["start"]
    df = df[df["interval_len"] > 0].reset_index(drop=True)
    # e.g., length 7 goes into [6.5, 7.5)
    min_len = int(target_lengths.min())
    max_len = int(target_lengths.max())
    bins = np.arange(min_len - 0.5, max_len + 1.5, 1.0)
    per_bin_counts = _allocate_counts_from_target_distribution(
        target_values=target_lengths,
        n_samples=n_samples,
        bins=bins,
    )
    rng = np.random.default_rng(0)
    sampled_chunks = []

    # sample kmers bin-by-bin, where each bin maps to one integer length.
    for i, need in enumerate(per_bin_counts):
        if need <= 0: continue
        # Bin i corresponds to integer length:[L-0.5, L+0.5) -> L
        k = int(round((bins[i] + bins[i + 1]) / 2.0))
        if k <= 0: continue

        eligible = df[df["interval_len"] >= k]
        chosen = eligible.sample(n=need).copy()
        high_bounds = chosen["interval_len"] - k + 1
        offsets = rng.integers(0, high_bounds)
        chosen["start"] = chosen["start"]+ offsets
        chosen["end"] = chosen["start"] + k
        chosen["start_rel"] = chosen["start_rel"] + offsets
        chosen["end_rel"] = chosen["start_rel"] + k
        sampled_chunks.append(chosen[["chrom", "start", "end", "region", "start_rel", "end_rel"]])

    out = pd.concat(sampled_chunks, ignore_index=True)

    # ensure exact n_samples when possible.
    if len(out) > n_samples:
        out = out.sample(n=n_samples, random_state=0).reset_index(drop=True)
    elif len(out) < n_samples:
        # Top up from already sampled rows (keeps approximate matched distribution).
        if len(out) > 0:
            topup = out.sample(n=n_samples - len(out))
            out = pd.concat([out, topup], ignore_index=True)

    return out.reset_index(drop=True)




def generate_null_pairs(
    non_motif_df_path,
    target_distances,
    receptive_field,
    k,
    n_samples,
    n_bins,
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
        if len(group) < 2: continue
        
        group = group.sort_values(["start_rel", "end_rel"]).reset_index(drop=True)
        for i, j in combinations(range(len(group)), 2):
            gap1 = group.iloc[i]
            gap2 = group.iloc[j]
            s1 = int(gap1["start_rel"])
            e1 = int(s1 + k)
            s2 = int(gap2["start_rel"])
            e2 = int(s2 + k)
            dist = s2 - e1
            if dist <= 0: continue
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

    # bins as ints
    bins = np.linspace(0, receptive_field, n_bins + 1).astype(int)
    target_distances = np.asarray(target_distances)
    target_counts = _allocate_counts_from_target_distribution(
        target_values=target_distances,
        n_samples=n_samples,
        bins=bins,
    )

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
        if long_gaps.empty: continue
        sampled_intra = []
        for _ in range(remaining):
            gap = long_gaps.sample(1).iloc[0]
            max_d = min(hi, int(gap["length"] - 2 * k))
            if max_d <= lo: continue
            d = np.random.randint(lo, max_d)
            s1_min = int(gap["start_rel"])
            s1_max = int(gap["end_rel"] - (2 * k + d))
            if s1_max <= s1_min: continue
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



