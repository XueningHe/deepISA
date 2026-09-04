import os
import pandas as pd
import numpy as np
import bioframe as bf
from loguru import logger
from itertools import combinations


# Internal imports
from deepISA.utils import remove_if_exists
from deepISA.score.pred_cache import PredCache 
from deepISA.score.isa_core import combi_isa_core


# TODO: since the file names are almost determined, all paths should have a default value.


def make_pairs_for_region(
    region_motif_rows: pd.DataFrame,
    receptive_field: int,
) -> pd.DataFrame | None:
    if len(region_motif_rows) < 2:
        return None

    region_motif_rows = region_motif_rows.sort_values("start_rel")
    pairs = []
    for idx1, idx2 in combinations(region_motif_rows.index, 2):
        m1, m2 = region_motif_rows.loc[idx1], region_motif_rows.loc[idx2]
        dist = m2.start_rel - m1.end_rel
        if dist < 1 or dist > receptive_field:
            continue
        pair_data = {
            "region": m1.region,
            "tf1": m1.tf,
            "tf2": m2.tf,
            "start1_rel": m1.start_rel,
            "end1_rel": m1.end_rel,
            "start2_rel": m2.start_rel,
            "end2_rel": m2.end_rel,
            "strand1": m1.strand,
            "strand2": m2.strand,
            "distance": dist,
        }
        
        isa_cols = [c for c in region_motif_rows.columns if c.startswith("isa_t")]
        for col in isa_cols:
            t = col.split("isa_t")[-1]
            pair_data[f"isa1_t{t}"] = m1[col]
            pair_data[f"isa2_t{t}"] = m2[col]

        pred_mut_cols = [c for c in region_motif_rows.columns if c.startswith("pred_mut_t")]
        for col in pred_mut_cols:
            t = col.split("pred_mut_t")[-1]
            pair_data[f"pred_mut1_t{t}"] = m1[col]
            pair_data[f"pred_mut2_t{t}"] = m2[col]
        pairs.append(pair_data)

    if not pairs:
        return None
    return pd.DataFrame(pairs)




def build_combi_pairs_by_region(df_single_isa, receptive_field):
    pairs_by_region = {}
    seq_ref_col = "seq_ref" in df_single_isa.columns
    for region_str, grp in df_single_isa.groupby("region"):
        grp = grp.copy()
        pair_df = make_pairs_for_region(grp, receptive_field)
        if pair_df is None or pair_df.empty:
            continue
        seq_ref = grp["seq_ref"].iloc[0] if seq_ref_col else None
        pairs_by_region[region_str] = (pair_df, seq_ref)
    return pairs_by_region


def _build_single_mut_map_from_df(df: pd.DataFrame) -> dict:
    """
    Build single_mut_map from pre-computed df_single_isa.
    Returns: {(region_str, start_rel, end_rel): [motif_mut_seq, ...]}
    """
    single_mut_map = {}
    for row in df.itertuples():
        key = (row.region, row.start_rel, row.end_rel)
        single_mut_map.setdefault(key, [])
        if row.motif_mut not in single_mut_map[key]:
            single_mut_map[key].append(row.motif_mut)
    return single_mut_map



def _check_isa_cols_present(df: pd.DataFrame, tracks: list, destroy_mode: str) -> bool:
    if destroy_mode == "dinuc_shuffle":
        return False
    if "motif_mut" not in df.columns:
        return False
    for t in tracks:
        if f"isa_t{t}" not in df.columns:
            return False
        if f"pred_mut_t{t}" not in df.columns:
            return False
    return True




def run_combi_isa(
    model,
    fasta,
    single_isa_path,
    outpath,
    device,
    receptive_field,
    pred_orig_path=None,
    tracks=[0],
    num_regions_per_batch=200,
    pred_batch_size=1024,
    destroy_mode="ablate",
    n_shuffles=4,
):
    remove_if_exists(outpath)

    if isinstance(fasta, str):
        fasta = bf.load_fasta(fasta)

    df_single_isa = pd.read_csv(single_isa_path)
    if df_single_isa.empty:
        logger.warning("No motifs in motif_single_isa file.")
        return None

    df_pred_orig = pd.read_csv(pred_orig_path) if pred_orig_path is not None else None
    isa_cols_present = _check_isa_cols_present(df_single_isa, tracks, destroy_mode)

    all_regions = df_single_isa["region"].unique().tolist()

    logger.info(f"Combinatorial ISA: {len(all_regions)} regions, batch size {num_regions_per_batch}")

    for batch_start in range(0, len(all_regions), num_regions_per_batch):
        batch_regions = all_regions[batch_start : batch_start + num_regions_per_batch]
        batch_region_set = set(batch_regions)
        logger.info(f"Batch {batch_start}–{batch_start + len(batch_regions)} / {len(all_regions)}")

        # ── build pairs for this batch only ──────────────────────────
        batch_df = df_single_isa[df_single_isa["region"].isin(batch_region_set)]
        batch_pairs = build_combi_pairs_by_region(batch_df, receptive_field)
        if not batch_pairs:
            continue

        # ── fresh cache per batch ─────────────────────────────────────
        cache = PredCache()

        if df_pred_orig is not None:
            batch_orig_df = df_pred_orig[df_pred_orig["region"].isin(batch_region_set)]
            cache.load_pred_orig(batch_orig_df, tracks)

        if isa_cols_present:
            cache.load_single_isa(batch_df, tracks)
            single_mut_map = _build_single_mut_map_from_df(batch_df)
        else:
            single_mut_map = None

        # ── four GPU passes ───────────────────────────────────────────
        combi_isa_core(
            model=model,
            device=device,
            tracks=tracks,
            fasta=fasta,
            batch_pairs=batch_pairs,
            pred_batch_size=pred_batch_size,
            outpath=outpath,
            cache=cache,
            single_mut_map=single_mut_map,
            destroy_mode=destroy_mode,
            n_shuffles=n_shuffles,
        )

    logger.info(f"Combinatorial ISA complete. Results saved to {outpath}")





