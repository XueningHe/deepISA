import pytest
import pandas as pd
from unittest.mock import patch
from deepISA.plotting.cooperativity import (
    get_prefix, 
    get_compressed_labels, 
    hist_coop_score, 
    heatmap_coop_score,
    plot_motif_distance_by_category
)

# --- Fixtures ---

@pytest.fixture
def mock_tf_df():
    """
    Generates a dataframe that mimics post-processed TF data.
    Added 'ks_q' because the internal assign_cooperativity logic requires it.
    """
    return pd.DataFrame({
        "tf_pair": ["SOX2|OCT4", "SOX17|OCT4", "GATA1|GATA2", "KLF4|KLF4"],
        "tf": ["SOX2", "SOX17", "GATA1", "KLF4"],
        "coop_score": [0.8, 0.7, -0.2, 0.5],
        "ks_q": [0.001, 0.001, 0.01, 0.001], # Essential for assign_cooperativity
        "cooperativity": ["Synergistic", "Synergistic", "Redundant", "Intermediate"],
        "mean_distance": [20, 25, 50, 15]
    })
# --- Unit Tests for Label Logic ---

@pytest.mark.parametrize("input_name, expected", [
    ("SOX2", "SOX"),
    ("ESRRA", "ESRRA"), # Should grab all letters
    ("GATA1", "GATA"),
    ("123", "123"),     # Fallback for no letters
])
def test_get_prefix(input_name, expected):
    assert get_prefix(input_name) == expected

def test_get_compressed_labels():
    names = ["SOX2", "SOX17", "OCT4", "GATA1", "GATA2", "GATA3"]
    # Expect: SOX2/17 -> [SOXs, ""], OCT4 -> [OCT4], GATAs -> [GATAs, "", ""]
    expected = ["SOXs", "", "OCT4", "GATAs", "", ""]
    assert get_compressed_labels(names) == expected

# --- Integration Tests for Plotting ---

@patch("deepISA.plotting.cooperativity.assign_cooperativity")
def test_hist_coop_score(mock_assign, mock_tf_df, tmp_path):
    mock_assign.return_value = mock_tf_df
    out = tmp_path / "hist.png"
    
    # Test with annotations and vlines
    hist_coop_score(
        mock_tf_df, 
        outpath=str(out), 
        vlines=[0, 0.5], 
        annotations=[(0.7, 0.5, "High")]
    )
    
    assert out.exists()

@patch("deepISA.plotting.cooperativity.assign_cooperativity")
def test_heatmap_coop_score(mock_assign, tmp_path):
    # Prepare data specifically for a pivot-able heatmap
    heatmap_data = pd.DataFrame({
        "tf_pair": ["A|B", "A|C", "B|C"],
        "coop_score": [0.5, 0.1, -0.2],
        "ks_q": [0, 0, 0], # Add this
        "cooperativity": ["Synergistic", "Intermediate", "Redundant"]
    })
    mock_assign.return_value = heatmap_data
    out = tmp_path / "heatmap.pdf"
    
    heatmap_coop_score(heatmap_data, outpath=str(out), fig_size=(5, 5))
    
    assert out.exists()

@patch("deepISA.utils.format_cooperativity_categorical")
@patch("deepISA.plotting.cooperativity.plot_violin_with_statistics")
def test_plot_motif_distance_by_category(mock_violin, mock_format, mock_tf_df, tmp_path):
    mock_format.return_value = mock_tf_df
    out = tmp_path / "violin.png"
    
    plot_motif_distance_by_category(mock_tf_df, outpath=str(out))
    
    # Verify that the internal utility function was actually called
    assert mock_violin.called
    # Check that it passed the correct column names
    args, kwargs = mock_violin.call_args
    assert kwargs['x_col'] == "cooperativity"
    assert kwargs['y_col'] == "mean_distance"



