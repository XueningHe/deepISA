import os
import pytest
import pandas as pd
import numpy as np
from deepISA.plotting.interaction import plot_abs_interaction_decay, plot_signed_interaction_decay

@pytest.fixture
def mock_decay_df():
    """
    Creates mock data simulating the output of run_combi_isa.
    Includes interaction tracks for regression (0) and classification (1).
    """
    # Create distances ranging from 0 to 600
    distances = np.tile(np.arange(10, 610, 50), 5)
    data = {
        "distance": distances,
        # Mixed positive and negative interactions
        "interaction_t0": np.random.normal(0.1, 0.2, len(distances)), 
        "interaction_t1": np.random.normal(-0.05, 0.1, len(distances)) 
    }
    return pd.DataFrame(data)

def test_plot_abs_decay_multi_track(mock_decay_df, tmp_path):
    """Tests if multi-track absolute decay plotting works and saves a file."""
    out_file = tmp_path / "test_abs_decay_multi.pdf"
    
    # Updated function name and argument 'outpath'
    plot_abs_interaction_decay(
        df=mock_decay_df,
        outpath=str(out_file),
        track_idx=[0, 1],
        track_names={0: "reg_test", 1: "clf_test"}
    )
    
    assert out_file.exists()
    assert out_file.stat().st_size > 0

def test_plot_abs_decay_single_track_int(mock_decay_df, tmp_path):
    """Tests if specifying a single track as an integer works (isinstance check)."""
    out_file = tmp_path / "test_abs_decay_single.pdf"
    
    plot_abs_interaction_decay(
        df=mock_decay_df,
        outpath=str(out_file),
        track_idx=0
    )
    
    assert out_file.exists()

def test_plot_abs_decay_missing_track_handling(mock_decay_df, tmp_path):
    """Ensures the function handles missing track columns without crashing."""
    out_file = tmp_path / "test_missing_track.pdf"
    
    # Track 99 does not exist; function should return None/logger.error
    plot_abs_interaction_decay(
        df=mock_decay_df,
        outpath=str(out_file),
        track_idx=[99]
    )
    
    assert not out_file.exists()

def test_plot_signed_decay_exists(mock_decay_df, tmp_path):
    """Verifies the signed interaction decay function signature and output."""
    out_file = tmp_path / "test_signed_decay.pdf"
    
    # Testing the second function in your new script
    plot_signed_interaction_decay(
        df=mock_decay_df,
        outpath=str(out_file),
        track_idx=[0, 1],
        track_names={0: "Reg", 1: "Clf"}
    )
    
    assert out_file.exists()

def test_plot_decay_formatting_parameters(mock_decay_df, tmp_path):
    """Verifies that custom font and figure sizes are passed correctly."""
    out_file = tmp_path / "test_decay_styled.pdf"
    
    plot_abs_interaction_decay(
        df=mock_decay_df,
        outpath=str(out_file),
        figsize=(4, 3),
        label_size=12,
        tick_size=10,
        legend_size=10
    )
    
    assert out_file.exists()

def test_plot_decay_distance_case_sensitivity(tmp_path):
    """Confirms the fix for case-sensitivity regarding the 'distance' column."""
    out_file = tmp_path / "test_case.pdf"
    # Create df with strictly lowercase 'distance'
    df = pd.DataFrame({
        "distance": [10, 20, 30],
        "interaction_t0": [0.1, 0.2, 0.3]
    })
    
    try:
        plot_abs_interaction_decay(df, outpath=str(out_file), track_idx=[0])
    except KeyError:
        pytest.fail("Case sensitivity issue: 'distance' column not found.")
    
    assert out_file.exists()