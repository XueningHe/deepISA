import pandas as pd
from loguru import logger
from scipy.stats import ks_2samp
import numpy as np
import bioframe as bf    

# Internal imports
from deepISA.model.predict import compute_predictions 
from deepISA.score.isa_core import single_isa_core
from deepISA.utils import remove_if_exists
from deepISA.score.utils_isa import region_str_to_seq


# TODO: df_single_isa should add columns
# motif_orig           
# motif_mut,
# pred_mut,
# strand,



# TODO: take in all regions, not just the ones in motif_locs.csv
def get_pred_orig(
    model,
    fasta,
    regions,
    tracks,
    outpath,
    device,
    pred_batch_size,
):
    remove_if_exists(outpath, label="prediction of original regions")
    uniq_regions = list(pd.unique(pd.Series(regions)))
    if len(uniq_regions) == 0:
        raise ValueError("No regions provided to compute original predictions.")

    seqs = [region_str_to_seq(fasta, r) for r in uniq_regions]
    preds = compute_predictions(model, seqs, device=device, batch_size=pred_batch_size, tracks=tracks)
    pred_cols = [f"pred_t{i}" for i in tracks]
    df_pred = pd.DataFrame(preds, columns=pred_cols)
    df_pred.insert(0, "region", uniq_regions)
    df_pred.to_csv(outpath, index=False)
    logger.info(f"Saved region original predictions: {outpath} ({len(df_pred)} regions)")



# TODO: since the file names are almost determined, all paths should have a default value.


def run_single_isa(
    model,
    fasta,
    motif_locs_path,
    single_isa_outpath,
    pred_orig_outpath,
    device,
    tracks=[0],
    num_regions_per_batch=200,
    pred_batch_size=1024,
    destroy_mode="ablate",   
    n_shuffles=10,           
    single_isa_cache_outpath=None,
):
    if isinstance(fasta, str):
        fasta=bf.load_fasta(fasta)
        
    locs_df = pd.read_csv(motif_locs_path)
    regions = locs_df["region"].unique().tolist()
    logger.info("Computing original predictions for all regions")
    get_pred_orig(
        model=model,
        fasta=fasta,
        regions=regions,
        tracks=tracks,
        outpath=pred_orig_outpath,
        device=device,
        pred_batch_size=pred_batch_size,
    )
    
    logger.info("Running single ISA")
    single_isa_core(
        model=model,
        fasta=fasta,
        locs_df=locs_df,
        outpath=single_isa_outpath,
        device=device,
        tracks=tracks,
        num_regions_per_batch=num_regions_per_batch,
        pred_batch_size=pred_batch_size,
        pred_orig_path=pred_orig_outpath,
        cache_outpath=single_isa_cache_outpath,
        destroy_mode=destroy_mode,
        n_shuffles=n_shuffles,
    )
    logger.info(f"Single ISA complete. Results saved to {single_isa_outpath}.")


