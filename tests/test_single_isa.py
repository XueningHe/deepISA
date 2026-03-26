import os
import pytest
import pandas as pd
import torch
import pyfaidx
from deepISA.modeling.cnn import Conv
from deepISA.scoring.single_isa import run_single_isa, calc_tf_importance


# TODO: make sure it runs for both single track and multi track models


@pytest.fixture
def mock_setup(tmp_path):
    """Provides common resources for ISA testing."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Mock Model
    model = Conv(600).to(device)
    # Note: In a real test, you'd use a small weight file or random weights
    model.eval()
    
    # 2. Mock Fasta (Create a tiny dummy fasta)
    fasta_path = tmp_path / "test.fa"
    with open(fasta_path, "w") as f:
        f.write(">chr1\n" + "A" * 2000)
    fasta = pyfaidx.Fasta(str(fasta_path))
    
    return model, fasta, device, tmp_path



def test_run_single_isa_incremental_io(mock_setup):
    """Verify that run_single_isa writes to disk in batches and doesn't crash."""
    model, fasta, device, tmp_path = mock_setup
    outpath = tmp_path / "isa_results.csv"
    
    # Create motifs spanning two different regions to test grouping
    regions = ["chr1:100-700", "chr1:800-1400"]
    data = {
        "chrom": ["chr1"] * 10,
        "start": [110, 120, 130, 140, 150, 810, 820, 830, 840, 850],
        "end":   [115, 125, 135, 145, 155, 815, 825, 835, 845, 855],
        "tf": ["GATA1"] * 5 + ["CTCF"] * 5,
        "region": [regions[0]] * 5 + [regions[1]] * 5
    }
    motif_df = pd.DataFrame(data)

    # Run ISA with a tiny batch_size to force multiple disk writes
    result_file = run_single_isa(
        model, fasta, motif_df, 
        outpath=str(outpath), 
        device=device,
        batch_size=1 # Force write after every region
    )
    assert os.path.exists(result_file)
    results_df = pd.read_csv(result_file)
    # Verify content
    assert len(results_df) == 10
    assert "isa_t0" in results_df.columns




def test_calc_tf_importance(tmp_path):
    isa_path = tmp_path / "mock_isa.csv"
    mock_data = pd.DataFrame({
        "tf": ["GATA1"] * 15 + ["CTCF"] * 15,
        "isa_t0": [0.5] * 15 + [0.1] * 15,  # Changed to isa_t0
        "isa_t1": [0.8] * 15 + [0.2] * 15   # Changed to isa_t1
    })
    mock_data.to_csv(isa_path, index=False)
    
    # This call now works because the function accepts the string path
    agg_df = calc_tf_importance(str(isa_path))
    
    assert "mean_isa_t0" in agg_df.columns
    assert agg_df.shape[0] == 2




def test_empty_motif_df_handling(mock_setup):
    """Ensure the system handles empty inputs gracefully."""
    model, fasta, device, tmp_path = mock_setup
    outpath = tmp_path / "empty.csv"
    res = run_single_isa(model, fasta, pd.DataFrame(), outpath=str(outpath),device=device)
    assert res is None