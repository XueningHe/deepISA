import os
import torch
import pandas as pd
import pytest
import numpy as np
import bioframe as bf
from deepISA.modeling.train import train_model

@pytest.fixture
def fake_genomic_data(tmp_path):
    """Generates a fake FASTA and regions dataframe for testing."""
    # 1. Create Fake FASTA (chr1 and chr2)
    fasta_path = tmp_path / "fake.fa"
    with open(fasta_path, "w") as f:
        f.write(">chr1\n" + "ATGC" * 2500 + "\n") # 10kb
        f.write(">chr2\n" + "GCTA" * 2500 + "\n") # 10kb
    
    # 2. Create Fake Regions
    data = {
        'chrom': ['chr1'] * 8 + ['chr2'] * 2,
        'start': [100, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 100, 500],
        'end':   [700, 1600, 2600, 3600, 4600, 5600, 6600, 7600, 700, 1100],
        'target_reg': np.random.rand(10),
        'target_class': [1, 0, 1, 0, 1, 0, 1, 0, 1, 1]
    }
    df = pd.DataFrame(data)
    
    return df, str(fasta_path)

def test_full_pipeline_workflow(tmp_path, fake_genomic_data, monkeypatch):
    df, fasta_path = fake_genomic_data
    model_dir = tmp_path / "models"
    
    # Define a background pool that ONLY uses chr1 to avoid KeyError: 'chr3'
    fake_bg = pd.DataFrame({
        'chrom': ['chr1'] * 10,
        'start': [2000, 2100, 2200, 2300, 2400, 2500, 2600, 2700, 2800, 2900],
        'end':   [2600, 2700, 2800, 2900, 3000, 3100, 3200, 3300, 3400, 3500]
    })
    
    # CRITICAL: Patch where bioframe is actually used in the preprocess module
    # If preprocess.py does 'import bioframe as bf', patch 'deepISA.modeling.preprocess.bf.read_table'
    monkeypatch.setattr("deepISA.modeling.preprocess.bf.read_table", lambda *args, **kwargs: fake_bg)
    
    # Also patch get_data_resource just in case it's checking for file existence
    monkeypatch.setattr("deepISA.modeling.preprocess.get_data_resource", lambda x: "fake_path.bed")

    # Run the full training logic
    model, history, test_metrics = train_model(
        df=df,
        fasta_path=fasta_path,
        epochs=1,
        model_dir=str(model_dir),
        batch_size=2,
        target_reg_col="target_reg",
        target_class_col="target_class",
        model_name="test_model"
    )

    # Verify artifacts
    assert (model_dir / "test_model_best.pt").exists()

    # 2. Verify Memmap data exists in the processed_data subfolder
    processed_dir = model_dir / "processed_data"
    # Search specifically in the 'train' split folder
    train_x = processed_dir / "train" / "X.npy"
    assert train_x.exists(), f"Memmap data not found at {train_x}"
    
    # 3. CRITICAL: Cleanup memmap handles to avoid leaks/locks
    # This triggers the __del__ of the np.memmap objects
    import gc
    del model, history, test_metrics
    gc.collect()