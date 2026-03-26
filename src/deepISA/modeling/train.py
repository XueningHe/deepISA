import os
import torch
from loguru import logger
from .data_loader import DualDataset
from deepISA.modeling.cnn import Conv
from deepISA.modeling.trainer import Trainer
from deepISA.utils import find_available_gpu
from deepISA.utils import one_hot_encode
import bioframe as bf
import numpy as np
import json



def prepare_features(df, fasta_path, out_dir, seq_len, 
                     reg_col='target_reg', class_col='target_class', 
                     rc_aug=True, chunk_size=8096):
    """
    Extracts sequences and saves to memmapped arrays in chunks.
    Maintains low RAM overhead while utilizing fast vectorized operations.
    """
    os.makedirs(out_dir, exist_ok=True)
    logger.info(f"Loading FASTA from {fasta_path}")
    fasta = bf.load_fasta(fasta_path)
    
    # 1. Boundary-safe sequence extraction (Pre-filtering)
    def get_seq(r):
        if r.chrom not in fasta: return ""
        return str(fasta[r.chrom][int(r.start):int(r.end)]).upper()

    logger.info(f"Extracting sequences from {len(df)} regions.")
    df['seq'] = df.apply(get_seq, axis=1)
    df = df[df['seq'].str.len() == seq_len].copy()
    logger.info(f"Filtered to {len(df)} valid sequences.")

    
    n_samples = len(df)
    if n_samples == 0:
        raise ValueError("No valid sequences found.")

    # 2. Define shapes and pre-allocate memmaps
    total_samples = n_samples * 2 if rc_aug else n_samples
    # Shape: (Samples, Channels, Length) for PyTorch Conv1d
    x_shape = (total_samples, 4, seq_len) 
    yr_shape = (total_samples,)
    yc_shape = (total_samples,)

    f_X = np.memmap(os.path.join(out_dir, "X.npy"), dtype='float32', mode='w+', shape=x_shape)
    f_Yr = np.memmap(os.path.join(out_dir, "Yr.npy"), dtype='float32', mode='w+', shape=yr_shape)
    f_Yc = np.memmap(os.path.join(out_dir, "Yc.npy"), dtype='float32', mode='w+', shape=yc_shape)

    logger.info(f"Writing {total_samples} samples to disk in chunks of size {chunk_size}.")  
    for i in range(0, n_samples, chunk_size):
        if i % 10 == 0:
            logger.info(f"Writing chunk {i//chunk_size}/{n_samples//chunk_size} to disk") 
        
        chunk_df = df.iloc[i : i + chunk_size]
        curr_len = len(chunk_df)
        
        # 1. Get sequences and Encode
        seq_list = chunk_df['seq'].tolist()
        X_chunk = one_hot_encode(seq_list).astype('float32')
        
        # --- SHAPE CORRECTION ---
        if X_chunk.shape == (curr_len, seq_len, 4):
            X_chunk = X_chunk.transpose(0, 2, 1)
        
        # 2. Assignment with explicit shapes
        f_X[i : i + curr_len, :, :] = X_chunk
        f_Yr[i : i + curr_len] = np.log1p(chunk_df[reg_col].values).astype('float32')
        f_Yc[i : i + curr_len] = chunk_df[class_col].values.astype('float32')

        if rc_aug:
            f_X[n_samples + i : n_samples + i + curr_len, :, :] = X_chunk[:, ::-1, ::-1]
            f_Yr[n_samples + i : n_samples + i + curr_len] = f_Yr[i : i + curr_len]
            f_Yc[n_samples + i : n_samples + i + curr_len] = f_Yc[i : i + curr_len]

        f_X.flush(); f_Yr.flush(); f_Yc.flush()

    # 3. Final Persistence
    metadata = {
        "X": x_shape, "Yr": yr_shape, "Yc": yc_shape,
        "n_original": n_samples, "rc_aug": rc_aug
    }
    with open(os.path.join(out_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f)
    
    logger.info("One hot encoded sequences saved to disk.")
    # Release handles
    del f_X, f_Yr, f_Yc
    return out_dir





def train_model(df, fasta_path, seq_len=600, 
                rc_aug=True, use_cached_data=False, **kwargs):
    """
    Train model with a 70/15/15 Train/Val/Test split using memmapped data.
    """
    device = kwargs.get('device') or find_available_gpu()
    processed_dir = os.path.join(kwargs.get('model_dir', 'Models'), "processed_data")
    
    # 1. Encoding and Persistence (Memmapped)
    if not use_cached_data:
        prepare_features(
            df=df, 
            fasta_path=fasta_path, 
            out_dir=processed_dir, 
            seq_len=seq_len,
            rc_aug=rc_aug,
            chunk_size=kwargs.get('chunk_size', 8096)
        )
    
    # 2. Metadata and Index Calculation
    with open(os.path.join(processed_dir, "metadata.json"), "r") as f:
        meta = json.load(f)
    total_samples = meta['X'][0]
    # Create shuffled indices
    indices = np.random.permutation(total_samples)
    # Calculate Split Points
    train_cut = int(0.70 * total_samples)
    val_cut = int(0.85 * total_samples) # 0.70 + 0.15

    # 3. Create Datasets
    train_ds = DualDataset(processed_dir, indices=indices[:train_cut])
    val_ds   = DualDataset(processed_dir, indices=indices[train_cut:val_cut])
    test_ds  = DualDataset(processed_dir, indices=indices[val_cut:])
    logger.info(f"Splits created: Train={len(train_ds)}, Val={len(val_ds)}, Test={len(test_ds)}")

    # 4. Model Initialization
    # TODO: make model architecture configurable via kwargs
    model = Conv(seq_len=seq_len)
    model.to(device)
    
    # 5. Training Execution
    trainer = Trainer(
        model=model,
        train_dat=train_ds,
        val_dat=val_ds,
        device=device,
        optimizer=torch.optim.Adam(model.parameters(), lr=kwargs.get('lr', 1e-3)),
        model_dir=kwargs.get('model_dir'),
        model_name=kwargs.get('model_name') or "model",
        batch_size=kwargs.get('batch_size', 128),
        save_half=kwargs.get('save_half', False), 
        save_one=kwargs.get('save_one', False)   
    )
    
    history = trainer.train(epochs=kwargs.get('epochs', 10))
    
    # Return the test_ds so you can run final evaluation metrics externally
    return model, history, test_ds