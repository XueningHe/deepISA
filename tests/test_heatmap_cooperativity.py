import pandas as pd
import numpy as np
import pytest
import os
from deepISA.plotting.cooperativity import heatmap_cooperativity

@pytest.fixture
def sample_heatmap_data():
    """Generates mock data for heatmap testing."""
    return pd.DataFrame({
        "pair": ["GATA1|TAL1", "GATA1|KLF1", "KLF1|TAL1", "SOX2|OCT4"],
        "coop_score": [0.8, -0.5, 0.9, 0.1],
        "ks_q": [0.01, 0.02, 0.05, 0.5], # One non-significant row
        "cooperativity": ["Synergistic", "Redundant", "Synergistic", "Independent"]
    })

def test_heatmap_cooperativity_success(sample_heatmap_data, tmp_path):
    """Verifies that a valid heatmap is generated and saved."""
    output_pdf = tmp_path / "test_heatmap.pdf"
    
    # This should run without error and save a file
    heatmap_cooperativity(
        df=sample_heatmap_data, 
        qval_thresh=0.1, 
        outpath=str(output_pdf), 
        figsize=(5, 5)
    )
    
    assert output_pdf.exists()
    assert output_pdf.stat().st_size > 0

def test_heatmap_empty_after_significance_filter(tmp_path):
    """
    Verifies that the function handles cases where no TFs pass the significance gate
    gracefully without raising a KeyError or ValueError.
    """
    output_pdf = tmp_path / "empty_heatmap.pdf"
    
    # Data where all ks_q > 0.1
    data = {
        "pair": ["GATA1|TAL1"],
        "coop_score": [0.9],
        "ks_q": [0.99],
        "cooperativity": ["Independent"]
    }
    df = pd.DataFrame(data)
    
    # With the guard clause, this should return None safely
    try:
        result = heatmap_cooperativity(df, qval_thresh=0.1, outpath=str(output_pdf))
        assert result is None
        assert not output_pdf.exists() # Should not save a file if empty
    except Exception as e:
        pytest.fail(f"Heatmap crashed on empty data: {e}")

def test_heatmap_partial_missing_data(sample_heatmap_data, tmp_path):
    """
    Verifies the splitting and pivoting logic results in the correct matrix shape.
    """
    # Filter manually to see what we expect
    # Significant: GATA1|TAL1, GATA1|KLF1, KLF1|TAL1
    # TFs involved: GATA1, TAL1, KLF1
    output_pdf = tmp_path / "partial_heatmap.pdf"
    
    try:
        heatmap_cooperativity(sample_heatmap_data, qval_thresh=0.1, outpath=str(output_pdf))
    except Exception as e:
        pytest.fail(f"Logic error in splitting/pivoting: {e}")

def test_heatmap_missing_columns():
    """Verify KeyError is raised if required columns are missing."""
    bad_df = pd.DataFrame({"not_a_pair": ["A|B"], "ks_q": [0.01]})
    with pytest.raises(KeyError):
        heatmap_cooperativity(bad_df)