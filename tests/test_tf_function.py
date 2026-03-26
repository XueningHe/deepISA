import pytest
import pandas as pd
import os
from deepISA.validating.tf_function import plot_usf_pfs, plot_cell_specificity

@pytest.fixture
def mock_tf_data():
    """Creates a mock dataset simulating interaction_profile_tf.csv."""
    data = {
        "TF": ["SP1", "KLF4", "FOXA1", "GATA1", "CTCF", "MYC", "SOX2", "NANOG", "ATF6", "EGR1"],
        "synergy_score": [0.1, 0.85, 0.9, 0.75, 0.05, 0.4, 0.6, 0.5, 0.2, 0.15],
        "independence_score": [0.9, 0.15, 0.1, 0.25, 0.95, 0.6, 0.4, 0.5, 0.8, 0.85]
    }
    return pd.DataFrame(data)

def test_plot_usf_pfs_mock(mock_tf_data):
    """Tests the Pioneer/Stripe Factor plot and ensures cleanup."""
    out_synergy = "tests/mock_usf_pf_synergy.pdf"
    out_indep = "tests/mock_usf_pf_independence.pdf"
    
    try:
        plot_usf_pfs(mock_tf_data, score_col="synergy_score", outpath=out_synergy)
        assert os.path.exists(out_synergy)
        
        plot_usf_pfs(mock_tf_data, score_col="independence_score", outpath=out_indep)
        assert os.path.exists(out_indep)
        
    finally:
        for path in [out_synergy, out_indep]:
            if os.path.exists(path):
                os.remove(path)

def test_plot_cell_specificity_mock(mock_tf_data):
    """Tests the Gini scatter plot and ensures cleanup."""
    out = "tests/mock_gini_vs_synergy.pdf"
    
    try:
        plot_cell_specificity(mock_tf_data, score_col="synergy_score", outpath=out)
        assert os.path.exists(out)
        
    finally:
        if os.path.exists(out):
            os.remove(out)