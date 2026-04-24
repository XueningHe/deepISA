import os
import json
import pytest
import numpy as np
import pandas as pd
import bioframe as bf
from deepISA.modeling.preprocess import compile_training_data, _balance_and_label

@pytest.fixture
def mock_genomic_data(tmp_path):
    """Creates physical mock FASTA and BED files with enough padding for resizing."""
    # chr1: 2000bp of 'A', chr2: 2000bp of 'G'
    fa_path = tmp_path / "mock.fa"
    with open(fa_path, "w") as f:
        f.write(">chr1\n" + "A" * 2000 + "\n")
        f.write(">chr2\n" + "G" * 2000 + "\n")
    
    bg_path = tmp_path / "background.bed"
    bg_df = pd.DataFrame({
        'chrom': ['chr1', 'chr1'],
        'start': [1500, 1600],
        'end': [1600, 1700]
    })
    bg_df.to_csv(bg_path, sep='\t', header=False, index=False)
    return fa_path, bg_path



def test_balance_and_label_logic():
    # Tests that we get 1:1 ratio even if we start with only positives
    df = pd.DataFrame({
        'chrom': ['chr1'], 'start': [500], 'end': [600], 'target_class': [1.0]
    })
    neg_pool = pd.DataFrame({
        'chrom': ['chr1', 'chr1'], 'start': [1000, 1200], 'end': [1100, 1300]
    })
    
    balanced = _balance_and_label(df, neg_pool, seq_len=600)
    assert len(balanced) == 2
    assert (balanced['target_class'] == 1.0).sum() == 1
    assert (balanced['target_class'] == 0.0).sum() == 1



def test_compile_training_data_end_to_end(mock_genomic_data, tmp_path, monkeypatch):
    fa_path, bg_path = mock_genomic_data
    out_dir = tmp_path / "processed"
    
    monkeypatch.setattr("deepISA.modeling.preprocess.get_data_resource", lambda x: str(bg_path))
    
    # Input regions (centered at 500, resize will make them 200-800)
    df = pd.DataFrame({
        'chrom': ['chr1', 'chr2'],
        'start': [450, 450],
        'end': [550, 550],
        'signal': [10.0, 5.0]
    })

    compile_training_data(
        df=df,
        fasta_path=str(fa_path),
        out_dir=str(out_dir),
        seq_len=600,
        rc_aug=False,
        target_reg_col='signal',
        chunk_size=10 # Small chunk size to test looping logic
    )

    # 1. Test Metadata and Shape
    test_dir = out_dir / "test"
    with open(test_dir / "metadata.json", "r") as f:
        meta = json.load(f)
    
    assert meta['X'][1] == 4   # 4 channels
    assert meta['X'][2] == 600 # seq_len
    
    # 2. Test Chromosome Holdout & Content
    # Open the memmap to check the actual sequence bits
    test_X = np.memmap(test_dir / "X.npy", dtype='float32', mode='r', shape=tuple(meta['X']))
    
    # chr2 was all 'G' (index 2 in A,C,G,T). 
    # Check that the encoded sequence is all Gs
    assert np.all(test_X[0, 2, :] == 1.0) 
    assert np.all(test_X[0, 0, :] == 0.0) # No As
    
    # 3. Test Regression Target Transformation (log1p)
    # Original signal for chr2 was 5.0 -> expected log1p(5.0)
    test_Yr = np.memmap(test_dir / "Yr.npy", dtype='float32', mode='r', shape=tuple(meta['Yr']))
    expected_log = np.log1p(5.0).astype('float32')
    assert np.isclose(test_Yr[0], expected_log, atol=1e-5)