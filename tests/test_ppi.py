import pytest
import pandas as pd
import numpy as np
import sys
import matplotlib.pyplot as plt
from loguru import logger
from unittest.mock import patch

from deepISA.exploring.tf_pair_ppi import (
    plot_ppi_enrichment, 
    annotate_cofactor_recruitment, 
    plot_dna_mediated_ppi,
    plot_cofactor_recruitment
)

@pytest.fixture
def sample_df():
    """Provides a minimal version of data with all required columns."""
    return pd.DataFrame({
        'tf_pair': ['TF1|TF2', 'TF3|TF4', 'TF1|TF3', 'TF2|TF4'],
        'coop_score': [0.8, -0.1, 0.5, 0.2],
        'ks_p': [0.001, 0.5, 0.02, 0.1],
        'ks_q': [0.001, 0.5, 0.02, 0.1],
        'cooperativity': ['Synergistic', 'Independent', 'Synergistic', 'Independent']
    })

@pytest.fixture
def setup_mocks(monkeypatch):
    """Mocks external files and styling."""
    # Mock data for DNA-mediated PPI: must be a matrix format (prey vs baits) 
    # to avoid the .melt() ValueError
    ppi_data = pd.DataFrame({'TF1': ['TF1'], 'TF2': ['TF2']})
    cof_data = pd.DataFrame({'TF': ['TF1', 'TF2'], 'p300': [1, 0]})
    
    # Correcting the DNA matrix mock: column names are baits, index/col is prey
    binding_matrix = pd.DataFrame({
        'prey': ['TF1', 'TF3'],
        'TF2': [1, 0],
        'TF4': [0, 1]
    })

    def mock_read_csv(path, **kwargs):
        path_str = str(path)
        if "TF_TF_I" in path_str: return ppi_data
        if "TF_Cof_I" in path_str: return cof_data
        if "TF_binding_coop_cleaned" in path_str: return binding_matrix
        return pd.DataFrame()

    module_path = "deepISA.exploring.tf_pair_ppi"
    monkeypatch.setattr("pandas.read_csv", mock_read_csv)
    monkeypatch.setattr(f"{module_path}.apply_plot_style", lambda ax, fs: {'scale': 1.0, 'main': 10, 'small': 8})
    monkeypatch.setattr(f"{module_path}.get_data_resource", lambda x: x)

### --- TESTS ---

def test_annotate_cofactor_recruitment(sample_df, setup_mocks):
    result = annotate_cofactor_recruitment(sample_df.copy(), cofactors=['p300'])
    assert 'count_p300' in result.columns
    val = result.loc[result['tf_pair'] == 'TF1|TF2', 'count_p300'].values[0]
    assert int(val) == 1

def test_plot_ppi_enrichment_logic(sample_df, setup_mocks):
    """Verify internal data processing by intercepting the plotting call."""
    with patch("seaborn.lineplot") as mock_lp:
        plot_ppi_enrichment(sample_df.copy(), outpath=None)
        # Ensure it actually tried to plot something
        assert mock_lp.called
        # Check columns of the dataframe passed to lineplot
        processed_df = mock_lp.call_args[1]['data']
        assert 'is_ppi' in processed_df.columns

def test_plot_dna_mediated_ppi_execution(sample_df, setup_mocks):
    """
    Verify that the distribution plot returns a Figure and correctly 
    categorizes the binary interaction groups.
    """
    # 1. Execute the function
    # We pass outpath=None so your real save_or_show returns the Figure object
    fig = plot_dna_mediated_ppi(sample_df.copy(), outpath=None)
    # 2. Basic assertions
    assert isinstance(fig, plt.Figure), "Function should return a matplotlib Figure"
    assert len(fig.axes) == 2, "Figure should have two subplots (Coop Score and Significance)"
    # 3. Verify the data processing inside the axes
    labels = [tick.get_text() for tick in fig.axes[0].get_xticklabels()]
    assert 'Zero' in labels or 'Non-zero' in labels
    plt.close(fig)



def test_plot_cofactor_recruitment_calls(sample_df, setup_mocks):
    with patch("deepISA.exploring.tf_pair_ppi.plot_box_strip_statistics") as mock_box:
        plot_cofactor_recruitment(sample_df.copy(), cofactor_name="p300")
        assert mock_box.called

