import pandas as pd
from loguru import logger
import bioframe as bf    

# Internal imports
from deepISA.model.predict import compute_predictions 
from deepISA.utils import remove_if_exists
from deepISA.score.utils_isa import region_str_to_seq, load_pred_orig



import numpy as np
import pandas as pd
from loguru import logger
from deepISA.model.predict import compute_predictions

from deepISA.utils import remove_if_exists, write_stream_csv
from deepISA.model.predict import compute_predictions
from deepISA.utils import (
    remove_if_exists,
    write_stream_csv,
)

from deepISA.score.utils_isa import (
    ablate_motifs, 
    region_str_to_seq
)



def calc_pred_orig(
    model,
    fasta,
    motif_locs_path,
    tracks,
    outpath,
    device,
    pred_batch_size,
):
    remove_if_exists(outpath, label="prediction of original regions")
    if isinstance(fasta, str):
        fasta = bf.load_fasta(fasta)
    locs_df = pd.read_csv(motif_locs_path)
    uniq_regions = list(pd.unique(locs_df['region']))
    if len(uniq_regions) == 0:
        raise ValueError("No regions provided to compute original predictions.")

    seqs = [region_str_to_seq(fasta, r) for r in uniq_regions]
    preds = compute_predictions(model, seqs, device=device, batch_size=pred_batch_size, tracks=tracks)
    pred_cols = [f"pred_t{i}" for i in tracks]
    df_pred = pd.DataFrame(preds, columns=pred_cols)
    df_pred.insert(0, "region", uniq_regions)
    df_pred.to_csv(outpath, index=False)
    logger.info(f"Saved region original predictions: {outpath} ({len(df_pred)} regions)")





def run_single_isa(
    model,
    fasta,
    motif_locs_path,
    pred_orig_path,
    outpath,
    device,
    tracks=[0],
    num_regions_per_batch=200,
    pred_batch_size=1024,    
):
    remove_if_exists(outpath, label="single ISA file")

    if isinstance(fasta, str):
        fasta=bf.load_fasta(fasta)
    
    orig_pred_map = load_pred_orig(pred_orig_path, tracks)
    locs_df = pd.read_csv(motif_locs_path)
    region_groups = list(locs_df.groupby("region"))
    
    logger.info(f"Single ISA started. Total rows to process: {len(locs_df)}")

    locs_df = pd.read_csv(motif_locs_path)
    for i in range(0, len(region_groups), num_regions_per_batch):
        batch = region_groups[i : i + num_regions_per_batch]
        batch_results = []
        for region_str, group in batch:
            try:
                seq_orig = region_str_to_seq(fasta, region_str)
            except Exception as e:
                logger.error(f"Failed to parse region: {region_str}. Error: {e}. Skipping this region.")
                continue
            group = group.copy()
            group["seq_mut"] = [
                ablate_motifs(seq_orig, [int(s)], [int(e)])
                for s, e in zip(group["start_rel"].to_numpy(), group["end_rel"].to_numpy())
            ]
            batch_results.append(group)

        current_df = pd.concat(batch_results).reset_index(drop=True)
        preds_orig_sel = np.vstack([orig_pred_map[reg] for reg in current_df["region"].values])
        preds_mut = compute_predictions(model, current_df["seq_mut"].values, device, pred_batch_size, tracks=tracks)
        current_df = current_df.drop(columns=["seq_mut"])
        for j,t in enumerate(tracks):
            current_df[f"isa_t{t}"] = preds_orig_sel[:, j] - preds_mut[:, j]

        write_stream_csv(current_df, outpath)

    logger.info(f"Single ISA complete. Results saved to {outpath}.")


