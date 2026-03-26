import pytest
import pandas as pd
import numpy as np
import os
from unittest.mock import patch


from deepISA.plotting.tf import plot_cooperativity_vs_importance

def test_plot_cooperativity_vs_importance_logic(tmp_path):
    # Create a synthetic dataset with a perfect correlation
    df_synergy = pd.DataFrame({
        'tf': ['P1', 'P2', 'P3', 'P4'],
        'cooperativity': ['Redundant', 'Intermediate', 'Synergistic', 'Independent'],
        'synergy_score': [0.2, 0.5, 0.8, 0.1]
    })
    
    df_importance = pd.DataFrame({
        'tf': ['P1', 'P2', 'P3', 'P4'],
        'mean_isa_t0': [0.2, 0.5, 0.8, 0.1]
    })
    
    outpath = tmp_path / "cooperativity_test.pdf"

    # We patch plt.show/plt.savefig to avoid creating actual files during logic testing
    with patch('matplotlib.pyplot.savefig') as mock_save:
        plot_cooperativity_vs_importance(
            df_synergy, 
            df_importance, 
            outpath=str(outpath)
        )
        
        # Verify savefig was called with the correct path
        mock_save.assert_called_once()
        assert str(mock_save.call_args[0][0]) == str(outpath)

def test_cooperativity_merge_integrity():
    """Ensure that the merge handles missing proteins gracefully."""
    df_synergy = pd.DataFrame({'tf': ['A', 'B'], 'synergy_score': [1, 2], 'cooperativity': ['S', 'I']})
    df_importance = pd.DataFrame({'tf': ['A'], 'mean_isa_t0': [10]})
    
    # Inner merge should result in only 1 row
    df_merged = df_synergy.merge(df_importance, on="tf")
    assert len(df_merged) == 1
    assert df_merged.iloc[0]['tf'] == 'A'