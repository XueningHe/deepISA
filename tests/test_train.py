import os
import torch
import pandas as pd
import pytest
import shutil
import json
import numpy as np
from deepISA.modeling.preprocess import compile_training_data
from deepISA.modeling.train import train_model, prepare_features

# Constants
DATA_DIR = "data/mini_data_chr1_1MB"
MODEL_DIR = "tests/models"

@pytest.fixture
def clean_model_dir():
    if os.path.exists(MODEL_DIR):
        # We use a try-except here because memmaps can be stubborn with file locks
        try:
            shutil.rmtree(MODEL_DIR)
        except OSError:
            pass 
    os.makedirs(MODEL_DIR, exist_ok=True)
    yield MODEL_DIR

@pytest.fixture
def temp_out_dir():
    path = "tests/temp_data"
    if os.path.exists(path):
        try:
            shutil.rmtree(path)
        except OSError:
            pass
    os.makedirs(path, exist_ok=True)
    yield path

# --- Integration Tests ---

def test_full_pipeline_workflow(clean_model_dir):
    df_path = os.path.join(DATA_DIR, "regions_1mb.csv")
    fasta_path = os.path.join(DATA_DIR, "hg38_1mb.fa")
    
    raw_df = pd.read_csv(df_path)
    raw_df = raw_df[raw_df['chrom'] == 'chr1'].copy()
    raw_df['target_reg'] = raw_df['log1pTPM']
    raw_df['target_class'] = (raw_df['log1pTPM'] > 0).astype(float)
    
    compiled_df = compile_training_data(
        df=raw_df,
        seq_len=600,
        target_reg_col="target_reg",
        target_class_col="target_class"
    )
    
    model, best_pearson, test_ds = train_model(
        df=compiled_df,
        fasta_path=fasta_path,
        target_reg_col="target_reg",
        target_class_col="target_class",
        epochs=2,
        model_dir=clean_model_dir,
        batch_size=16,
        patience=1
    )

    processed_dir = os.path.join(clean_model_dir, "processed_data")
    assert os.path.exists(os.path.join(processed_dir, "X.npy"))
    
    # FIX: best_pearson might be a numpy.float32, so we check for real numbers
    assert isinstance(best_pearson, (float, np.floating)), f"Expected float, got {type(best_pearson)}"
    
    # Cleanup memmap handles to allow fixture teardown
    if hasattr(test_ds, 'X'):
        del test_ds.X, test_ds.Yr, test_ds.Yc

# --- Unit Tests ---

def test_prepare_features_shapes_and_metadata(temp_out_dir):
    data = {
        'chrom': ['chr1', 'chr1'],
        'start': [100, 200],
        'end': [700, 800],
        'target_reg': [1.5, 2.5],
        'target_class': [1, 0]
    }
    df = pd.DataFrame(data)
    fasta_path = os.path.join(DATA_DIR, "hg38_1mb.fa")
    
    prepare_features(df, fasta_path, temp_out_dir, seq_len=600, rc_aug=True)
    
    with open(os.path.join(temp_out_dir, "metadata.json"), "r") as f:
        meta = json.load(f)
    
    assert meta["X"][0] == 4 
    assert meta["n_original"] == 2

def test_reverse_complement_consistency(temp_out_dir):
    data = {
        'chrom': ['chr1'], 'start': [100], 'end': [700],
        'target_reg': [1.0], 'target_class': [1]
    }
    df = pd.DataFrame(data)
    fasta_path = os.path.join(DATA_DIR, "hg38_1mb.fa")
    
    prepare_features(df, fasta_path, temp_out_dir, seq_len=600, rc_aug=True)
    
    with open(os.path.join(temp_out_dir, "metadata.json"), "r") as f:
        meta = json.load(f)
        
    X = np.memmap(os.path.join(temp_out_dir, "X.npy"), dtype='float32', mode='r', shape=tuple(meta['X']))
    original, augmented = X[0], X[1]
    
    # Verification logic
    np.testing.assert_array_almost_equal(original[::-1, ::-1], augmented)
    del X



def test_chunking_logic(temp_out_dir):
    """Verifies that the remainder of the chunk (index 9) is written correctly."""
    n_total = 10
    # Use a start position later in the 1MB file to avoid leading 'N's
    offset = 500000 
    data = {
        'chrom': ['chr1'] * n_total,
        'start': [offset + i*10 for i in range(n_total)],
        'end': [offset + 600 + i*10 for i in range(n_total)],
        'target_reg': np.random.rand(n_total),
        'target_class': [1] * n_total
    }
    df = pd.DataFrame(data)
    fasta_path = os.path.join(DATA_DIR, "hg38_1mb.fa")
    
    # Process
    prepare_features(df, fasta_path, temp_out_dir, seq_len=600, rc_aug=False, chunk_size=3)

    with open(os.path.join(temp_out_dir, "metadata.json"), "r") as f:
        meta = json.load(f)
    
    X = np.memmap(os.path.join(temp_out_dir, "X.npy"), dtype='float32', mode='r', shape=tuple(meta['X']))
    
    last_sum = np.sum(X[9])
    # Debug print to see what's actually there if it fails
    if last_sum == 0:
        print(f"\nDEBUG: X[9] content: {X[9]}")
        print(f"DEBUG: X[0] sum: {np.sum(X[0])}")
        
    del X
    assert last_sum > 0, f"Last chunk (index 9) is empty. Total samples in meta: {meta['X'][0]}"
    
    
    
def test_checkpoint_creation_and_cleanup(clean_model_dir):
    """
    Verifies that save_half, save_one, and 'best' checkpoints are created 
    correctly and that all temporary files are cleaned up.
    """
    df_path = os.path.join(DATA_DIR, "regions_1mb.csv")
    fasta_path = os.path.join(DATA_DIR, "hg38_1mb.fa")
    
    raw_df = pd.read_csv(df_path)
    raw_df['target_reg'] = raw_df['log1pTPM']
    raw_df['target_class'] = (raw_df['log1pTPM'] > 0).astype(float)
    
    # We use a small batch size to ensure the 'half_epoch' logic is triggered
    model_name = "test_checkpoint_model"
    model, best_r, test_ds = train_model(
        df=raw_df,
        fasta_path=fasta_path,
        target_reg_col="target_reg",
        target_class_col="target_class",
        epochs=1,
        model_dir=clean_model_dir,
        batch_size=4, 
        save_half=True, 
        save_one=True,
        model_name=model_name
    )

    # 1. Verify checkpoint file existence
    expected_files = [
        f"{model_name}_half_epoch.pt",
        f"{model_name}_one_epoch.pt",
        f"{model_name}_best.pt"
    ]
    
    for filename in expected_files:
        file_path = os.path.join(clean_model_dir, filename)
        assert os.path.exists(file_path), f"Checkpoint {filename} was not found in {clean_model_dir}"

    # 2. Cleanup: Explicitly delete memmap handles to release file locks
    # This is critical for Windows/OSX users where open files cannot be deleted
    if hasattr(test_ds, 'X'):
        del test_ds.X, test_ds.Yr, test_ds.Yc
    
    # Optional: Manually remove the model directory to test full cleanup capability
    try:
        shutil.rmtree(clean_model_dir)
        assert not os.path.exists(clean_model_dir)
    except OSError as e:
        pytest.fail(f"Cleanup failed: {e}")