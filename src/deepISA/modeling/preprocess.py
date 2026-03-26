import pandas as pd
import bioframe as bf
from loguru import logger
from deepISA.utils import get_data_resource, resize_regions, quantify_bw, estimate_noise_threshold


def _balance_and_label(df, neg_pool_df, seq_len):
    """Labels data based on target_class and performs 1:1 balancing."""
    positives = df[df['target_class'] == 1.0].copy()
    negatives = df[df['target_class'] == 0.0].copy()
    
    n_pos = len(positives)
    n_neg = len(negatives)
    logger.info(f"Initial counts: {n_pos} positives, {n_neg} negatives.")
    
    if n_pos > n_neg:
        needed = n_pos - n_neg
        logger.info(f"Sampling {needed} extra negatives from background pool.")
        extra_negs = neg_pool_df.sample(n=needed, random_state=42).copy()
        extra_negs = resize_regions(extra_negs, seq_len)
        extra_negs['target_reg'], extra_negs['target_class'] = 0.0, 0.0
        final_df = pd.concat([positives, negatives, extra_negs])
    else:
        logger.info("Downsampling negatives to match positive count.")
        final_df = pd.concat([positives, negatives.sample(n=n_pos, random_state=42)])
        
    return final_df.sample(frac=1, random_state=42).reset_index(drop=True)

# TODO:
# Have to do train-val-test split here.
# Test set need to use chr2,3
# Have to save train, val, test for further use
def compile_training_data(df, 
                          seq_len=600, 
                          target_reg_col=None, 
                          target_class_col=None, 
                          bw_paths=None,
                          outpath="training_data.csv"):
    """
    Unified entry point for data. Handles three scenarios and returns a 
    standardized DataFrame with 'target_reg' and 'target_class'.
    """
    df = df.copy()
    df = resize_regions(df, seq_len)

    # --- Scenario 2: Bed file + BigWigs ---
    if bw_paths:
        logger.info("Scenario 2: Quantifying from BigWigs.")
        if isinstance(bw_paths, str): bw_paths = [bw_paths]
        signals, df = quantify_bw(df, bw_paths, seq_len)
        df['target_reg'] = signals
        threshold = estimate_noise_threshold(bw_paths, seq_len)
        logger.info(f"Setting threshold to {threshold:.4f}") 
        df['target_class'] = (df['target_reg'] > threshold).astype(float)

    # --- Scenario 1: Pre-quantified signal ---
    elif target_reg_col in df.columns:
        logger.info(f"Scenario 1: Using provided signal column '{target_reg_col}'.")
        df['target_reg'] = df[target_reg_col]
        
        # Check if user provided their own classification labels
        if target_class_col and target_class_col in df.columns:
            logger.info(f"Using provided class column '{target_class_col}'.")
            df['target_class'] = df[target_class_col].astype(float)
        else:
            logger.info("No class column provided. Inferring target_class (signal > 0).")
            df['target_class'] = (df['target_reg'] > 0.0).astype(float)
            
    # --- Scenario 3: Pure BED file (Positives only) ---
    else:
        logger.info("Scenario 3: Pure BED file provided. All regions treated as positives.")
        df['target_class'] = 1.0

    # --- Background Sampling & Balancing ---
    bg_regions_path = get_data_resource("non_cCRE_non_blacklist_non_exon.bed")
    bg_regions = bf.read_table(bg_regions_path, schema='bed',names=["chrom", "start", "end"])
    
    df_final = _balance_and_label(df, bg_regions, seq_len)
    df_final.to_csv(outpath, index=False)
    logger.info(f"Saved training data to {outpath}")
    return df_final
