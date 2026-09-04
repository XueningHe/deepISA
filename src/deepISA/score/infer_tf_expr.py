import numpy as np
import pandas as pd
import bioframe as bf
from loguru import logger
from deepISA.utils import quantify_bw, estimate_noise_threshold, get_data_resource

# TODO: ask robin comment.

def get_expressed_tfs(bw_paths, seq_len=1000, percentile=99):
    """
    Infers expressed TFs by comparing promoter signals against a 
    standardized noise threshold.
    """
    # 1. Load Pre-calculated Promoters
    tf_promoter_bed = get_data_resource("hg38_TF_promoters_500bp.bed")
    logger.info(f"Loading TF promoters from {tf_promoter_bed}...")
    df_promoters = bf.read_table(tf_promoter_bed, schema='bed')
    
    # 2. Quantify expression at TF Promoters
    logger.info(f"Quantifying signal for {len(df_promoters)} TF promoters...")
    signals, _ = quantify_bw(df_promoters, bw_paths, seq_len)
    df_promoters['expression'] = signals

    # 3. Estimate Noise Threshold (Reuse the training logic)
    threshold = estimate_noise_threshold(bw_paths, seq_len, percentile=percentile)

    # 4. Filter for TFs above the noise floor
    expressed_list = sorted(df_promoters.loc[df_promoters['expression'] > threshold, 'name'].unique())
    
    logger.info(f"Inferred {len(expressed_list)} expressed TFs.")
    return expressed_list


