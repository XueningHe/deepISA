
import pandas as pd
import bioframe as bf
from loguru import logger
from itertools import combinations

from deepISA.model.predict import compute_predictions

from deepISA.utils import remove_if_exists, write_stream_csv
from deepISA.model.predict import compute_predictions
from deepISA.utils import (
    remove_if_exists,
    write_stream_csv,
)

from deepISA.score.utils_isa import (
    ablate_motifs, 
    region_str_to_seq,
    load_pred_orig
)





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
        
        pairs.append(pair_data)
    if not pairs:
        return None
    return pd.DataFrame(pairs)



def build_combi_pairs_by_region(df_single_isa, receptive_field):
    pairs_by_region = {}
    for region_str, grp in df_single_isa.groupby("region"):
        grp = grp.copy()
        pair_df = make_pairs_for_region(grp, receptive_field)
        if pair_df is None or pair_df.empty:
            continue
        pairs_by_region[region_str] = pair_df
    return pairs_by_region


def combi_isa_core(
    model,
    device,
    tracks,
    fasta,
    pairs_by_region,            
    outpath,
    pred_orig_path,    
    num_regions_per_batch,
    pred_batch_size,
):
    remove_if_exists(outpath)
    orig_pred_map = load_pred_orig(pred_orig_path, tracks) 
    regions = list(pairs_by_region.keys())
    
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
            try:
                seq_orig = region_str_to_seq(fasta, region_str)
            except Exception as e:
                logger.error(f"Failed to parse region: {region_str}. Error: {e}. Skipping this region.")
                continue
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
    # TODO: future: build_combi_pairs_by_region all at once use up memory
    pairs_by_region = build_combi_pairs_by_region(df_motif_single_isa, receptive_field)
    combi_isa_core(
        model=model,
        device=device,
        tracks=tracks,
        fasta=fasta,
        pairs_by_region=pairs_by_region,
        outpath=outpath,
        pred_orig_path=pred_orig_path,
        num_regions_per_batch=num_regions_per_batch,
        pred_batch_size=pred_batch_size,
    )
    logger.info(f"Combinatorial ISA complete. Results saved to {outpath}")

