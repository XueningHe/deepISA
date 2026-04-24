import pandas as pd
import bioframe as bf
from loguru import logger

import os
import numpy as np
import json


from deepISA.utils import (
    get_data_resource, 
    resize_regions, 
    quantify_bw, 
    estimate_noise_threshold,
    get_sequences_from_df,
    one_hot_encode
)




def _balance_and_label(df, 
                       neg_pool_df, 
                       seq_len):
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



def _write_memmap(df, 
                  fasta, 
                  out_dir, 
                  seq_len, 
                  rc_aug, 
                  chunk_size):
    """Extracted from your original prepare_features to handle the actual disk writing."""
    os.makedirs(out_dir, exist_ok=True)
    n_samples = len(df)
    total_samples = n_samples * 2 if rc_aug else n_samples
    
    # Define shapes
    x_shape = (total_samples, 4, seq_len)
    y_shape = (total_samples,)

    # Initialize memmaps
    f_X = np.memmap(os.path.join(out_dir, "X.npy"), dtype='float32', mode='w+', shape=x_shape)
    f_Yr = np.memmap(os.path.join(out_dir, "Yr.npy"), dtype='float32', mode='w+', shape=y_shape)
    f_Yc = np.memmap(os.path.join(out_dir, "Yc.npy"), dtype='float32', mode='w+', shape=y_shape)
    
    for i in range(0, n_samples, chunk_size):
        chunk_df = df.iloc[i : i + chunk_size]
        curr_len = len(chunk_df)
        chunk_df = chunk_df.copy()
        chunk_df['seq'] = get_sequences_from_df(chunk_df, fasta)
        
        X_chunk = one_hot_encode(chunk_df['seq'].tolist()).astype('float32')
        if X_chunk.shape == (curr_len, seq_len, 4):
            X_chunk = X_chunk.transpose(0, 2, 1)

        # Write data
        f_X[i : i + curr_len, :, :] = X_chunk
        # Use log1p for regression targets, 0.0 if not available
        f_Yr[i : i + curr_len] = np.log1p(chunk_df.get('target_reg', 0.0)).astype('float32')
        # regression label is optional, but classification label is always given for free.
        f_Yc[i : i + curr_len] = chunk_df['target_class'].values.astype('float32')

        if rc_aug:
            f_X[n_samples + i : n_samples + i + curr_len, :, :] = X_chunk[:, ::-1, ::-1]
            f_Yr[n_samples + i : n_samples + i + curr_len] = f_Yr[i : i + curr_len]
            f_Yc[n_samples + i : n_samples + i + curr_len] = f_Yc[i : i + curr_len]

    # Save metadata for DualDataset to read later
    meta = {"X": x_shape, "Yr": (total_samples,), "Yc": (total_samples,), "rc_aug": rc_aug}
    with open(os.path.join(out_dir, "metadata.json"), "w") as f:
        json.dump(meta, f)
    
    f_X.flush(); del f_X; del f_Yr; del f_Yc






def compile_training_data(df, 
                          fasta_path,
                          out_dir,
                          seq_len=600, 
                          target_reg_col=None, 
                          target_class_col=None, 
                          bw_paths=None,
                          random_state=42,
                          rc_aug=True,
                          chunk_size=8192):
    """
    Unified entry point for data. Handles three scenarios and returns a 
    standardized DataFrame with 'target_reg' and 'target_class'.
    """
    df = df.copy()
    df = resize_regions(df, seq_len)
    fasta = bf.load_fasta(fasta_path)
    
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
        logger.info(f"Scenario 1: Using provided signal column '{target_reg_col}'. Assume signal values are not log-transformed.")
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
        raise ValueError("Data must have either bw_paths or a pre-existing signal column.")

    # --- Background Sampling & Balancing ---
    bg_regions_path = get_data_resource("non_cCRE_non_blacklist_non_exon.bed")
    bg_regions = bf.read_table(bg_regions_path, schema='bed',names=["chrom", "start", "end"])
    df = _balance_and_label(df, bg_regions, seq_len)
    
    # --- Chromosome Holdout Split (chr2) ---
    test_df = df[df['chrom'] == 'chr2'].copy()
    train_val_pool = df[df['chrom'] != 'chr2'].copy()
    # 85/15 random split for the remaining chromosomes
    train_df = train_val_pool.sample(frac=0.85, random_state=random_state).copy()
    val_df = train_val_pool.drop(train_df.index)

    # 4. Write three separate memmap folders
    for name, split_df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        if len(split_df) == 0:
            logger.warning(f"Split '{name}' is empty. Skipping memmap creation.")
            continue
        _write_memmap(split_df,
                      fasta, 
                      os.path.join(out_dir, name), 
                      seq_len, 
                      rc_aug,
                      chunk_size=chunk_size)
    
    return df