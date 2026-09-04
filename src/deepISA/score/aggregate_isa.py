import os

import numpy as np
import pandas as pd
from loguru import logger
from scipy.stats import ks_2samp, mannwhitneyu
from statsmodels.stats.multitest import multipletests

from deepISA.utils import remove_if_exists



#---------------------------
# Private helpers
#---------------------------

def _signed_ks_test(fg_vals: np.ndarray, bg_vals: np.ndarray) -> tuple[float, float]:
    """Signed KS statistic: positive if fg tends to be larger than bg."""
    stat, pval = ks_2samp(fg_vals, bg_vals, alternative="two-sided", mode="auto")
    sign = np.sign(np.median(fg_vals) - np.median(bg_vals))
    return float(sign * stat), float(pval)


def _compute_interaction_per_track(
    df: pd.DataFrame,
    track_idx: int,
    isa_thresh: float = 0.0,
    normalize: bool = False,
    tau: float = 0.0,
) -> pd.DataFrame:

    isa1  = df[f"isa1_t{track_idx}"].to_numpy(dtype=float)
    isa2  = df[f"isa2_t{track_idx}"].to_numpy(dtype=float)
    both  = df[f"isa_both_t{track_idx}"].to_numpy(dtype=float)

    isa1_wo2 = both - isa2   # isa of motif1 when motif2 is also ablated
    isa2_wo1 = both - isa1   # isa of motif2 when motif1 is also ablated

    qualified = ((isa1>=isa_thresh) & (isa2>=isa_thresh) & (isa1_wo2>=0) & (isa2_wo1 >= 0))

    num   = isa1 + isa2 - both
    denom = isa1 + isa2 + tau

    vals = np.full(len(df), np.nan, dtype=float)
    if normalize:
        valid = qualified & np.isfinite(num) & np.isfinite(denom) & (denom > 0)
        vals[valid] = num[valid] / denom[valid]
    else:
        valid = qualified & np.isfinite(num)
        vals[valid] = num[valid]

    df[f"interaction_t{track_idx}"] = vals
    return df


#---------------------------
# calc_interaction
#---------------------------


def get_isa_thresh(df_single_isa, tracks, null_threshold):
    isa_thresh = {}
    for t in tracks:
        df_pos = df_single_isa[df_single_isa[f"isa_t{t}"] >= 0]
        isa_thresh[t] = df_pos[f"isa_t{t}"].quantile(null_threshold)
        logger.info(f"ISA threshold for track {t}: {isa_thresh[t]:.4f}")
    return isa_thresh


# TODO: when normalize=True, tau_map must be provided
def calc_interaction(
    combi_isa_path: str,
    tracks: list[int],
    tau_map: dict[int, float] | None = None,
    isa_thresh_quantile: float | None = None,
    single_isa_path: str | None = None,
    normalize: bool = False,
) -> None:
    if not os.path.exists(combi_isa_path):
        raise FileNotFoundError(f"File not found: {combi_isa_path}")
    
    # Determine isa_thresh: quantile-based or explicit
    if isa_thresh_quantile is not None:
        if single_isa_path is None:
            raise ValueError(
                "isa_thresh_quantile provided but single_isa_path is None. "
                "Provide single_isa_path to derive thresholds from."
            )
        if not os.path.exists(single_isa_path):
            raise FileNotFoundError(f"File not found: {single_isa_path}")
        
        isa_thresh = get_isa_thresh(
            df_single_isa=pd.read_csv(single_isa_path),
            tracks=tracks,
            null_threshold=isa_thresh_quantile,
        )
        logger.info(
            f"Derived isa_thresh from {os.path.basename(single_isa_path)} "
            f"(quantile={isa_thresh_quantile}): {isa_thresh}"
        )
    else:
        isa_thresh = {t: 0.0 for t in tracks}
    
    # Set tau defaults
    if tau_map is None:
        tau_map = {t: 0.0 for t in tracks}
    
    # Validate
    if normalize:
        missing_tracks = set(tracks) - set(tau_map.keys())
        if missing_tracks:
            raise ValueError(
                f"normalize=True but tau_map missing tracks: {missing_tracks}. "
                f"Provide tau_map or set normalize=False."
            )
    
    df = pd.read_csv(combi_isa_path)
    
    # Validate required columns exist
    for t in tracks:
        for col in [f"isa1_t{t}", f"isa2_t{t}", f"isa_both_t{t}"]:
            if col not in df.columns:
                raise ValueError(f"Required column '{col}' not found in {combi_isa_path}")
    
    # Compute interaction for each track using helper
    for t in tracks:
        tau = tau_map.get(t, 0.0)
        thresh_t = isa_thresh.get(t, 0.0)
        
        _compute_interaction_per_track(
            df=df,
            track_idx=t,
            isa_thresh=thresh_t,
            normalize=normalize,
            tau=tau,
        )
    
    df.to_csv(combi_isa_path, index=False, float_format="%.6f")
    logger.info(f"Updated {combi_isa_path} with interaction columns")





#---------------------------
# calc_tf_importance
#---------------------------

def calc_tf_importance(
    single_isa_path: str,
    outpath: str,
    non_motif_isa_path: str = None,
    min_count: int = 10,
) -> pd.DataFrame:
    
    remove_if_exists(outpath, label="TF importance file")

    df=pd.read_csv(single_isa_path)

    isa_cols=sorted(c for c in df.columns if c.startswith("isa_t"))
    
    if non_motif_isa_path is not None:
        df_null=pd.read_csv(non_motif_isa_path)

    results = []
    for tf, tf_data in df.groupby("tf"):
        res = {"tf": tf, "n": int(tf_data.shape[0])}
        for col in isa_cols:
            tf_isas  = tf_data[col].to_numpy(dtype=float)
            if len(tf_isas) < min_count: continue
            res[f"mean_{col}"] = float(np.mean(tf_isas))
            res[f"ks_d_vs_all{col}"], res[f"ks_pval_vs_all_{col}"] = _signed_ks_test(tf_isas, df[col].to_numpy(dtype=float))
            if non_motif_isa_path is not None:
                null_isas = df_null[col].to_numpy(dtype=float)
                res[f"ks_d_vs_non_motifs_{col}"], res[f"ks_pval_vs_non_motifs_{col}"] = _signed_ks_test(tf_isas, null_isas)

        results.append(res)

    out_df = pd.DataFrame(results)
    out_df.to_csv(outpath, index=False, float_format="%.4f")
    logger.info(f"TF importance saved to {outpath}")
    return out_df


# -------------------------
# Combinatorial ISA
# -------------------------

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



def calc_coop_score(
    combi_isa_path: str,
    outpath: str,
    level: str,                     # "tf_pair" or "tf"
    non_motif_interaction_path: str = None,
    non_interation_distance: int = 150,
    track_idx: int = 0,
    min_count: int = 10,
    q_val_thresh: float = 0.1,
) -> pd.DataFrame:

    remove_if_exists(outpath, label="cooperativity score file")
    inter_col = f"interaction_t{track_idx}"

    # -------------- 1. Clean up df_combi_isa-----------------------------
    df = pd.read_csv(combi_isa_path)
    # Canonical TF pair ordering (alphabetical)
    df["tf1"], df["tf2"] = (np.minimum(df["tf1"], df["tf2"]),np.maximum(df["tf1"], df["tf2"]))

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
        
    # -------------- 2. Determine Null ----------------------------------
    if non_motif_interaction_path is not None:
        df_null = pd.read_csv(non_motif_interaction_path)
        null_interactions = df_null[inter_col].dropna().to_numpy(dtype=float)
    else:
        null_interactions = df.loc[df["distance"] > non_interation_distance, inter_col].dropna().to_numpy(dtype=float)

    # ------------ 3. Calculate coop score for each TF / TF-pair ----------------
    results = []
    for name, group in df_melt.groupby("name"):
        interactions = group[inter_col].dropna().to_numpy(dtype=float)
        if len(interactions) < min_count:
            continue
        _, p_val   = mannwhitneyu(interactions, null_interactions, alternative="two-sided")
        coop_score = interactions.sum() / np.abs(interactions).sum()
        results.append({
            level:name,
            "n":len(interactions),
            "abs_i_sum": np.abs(interactions).sum(),
            "coop_score":  coop_score,
            "p_val":p_val,
            "median_distance":group["distance"].median(),
        })
    
    # ------------ 4. Multiple testing correction and assign cooperativity ----------------
    res_df = pd.DataFrame(results)
    if res_df.empty:
        logger.warning("No groups passed min_count filter.")
        res_df.to_csv(outpath, index=False)
        return res_df

    res_df["q_val"] = multipletests(res_df["p_val"], method="fdr_bh")[1]
    res_df = assign_cooperativity(res_df, q_val_thresh)
    res_df.to_csv(outpath, index=False, float_format="%.4f")
    logger.info(f"Coop score ({level}) saved to {outpath}")
    return res_df






