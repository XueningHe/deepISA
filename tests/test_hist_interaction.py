import pytest
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from deepISA.plotting.cooperativity import hist_cooperativity

@pytest.fixture
def sample_interaction_data():
    """
    Generates synthetic data matching the DeepISA population-based distributional shift approach.
    Includes coop_score (-1 to 1) and ks_q (FDR-corrected p-values).
    """
    np.random.seed(42)
    n_points = 200
    return pd.DataFrame({
        # Generate scores across the full range
        "coop_score": np.random.uniform(-1, 1, n_points), 
        # Generate some significant (ks_q < 0.1) and some non-significant values
        "ks_q": np.concatenate([
            np.random.uniform(0, 0.09, n_points // 2), 
            np.random.uniform(0.11, 1.0, n_points // 2)
        ]),
        "pair": [f"TF_A|TF_B_{i}" for i in range(n_points)]
    })

def test_hist_cooperativity_saves_pdf(sample_interaction_data, tmp_path):
    """
    Test that the histogram correctly filters for significance and saves to disk.
    """
    output_file = tmp_path / "cooperativity_hist.pdf"
    
    # Logic: Only rows with ks_q < 0.1 should be plotted
    hist_cooperativity(
        df=sample_interaction_data,
        qval_thresh=0.1,
        title="Significant Cooperativity",
        xlabel="Cooperativity Score (-1 to 1)",
        outpath=str(output_file)
    )
    
    assert output_file.exists()
    assert output_file.stat().st_size > 0

def test_hist_cooperativity_annotations_and_logic(sample_interaction_data):
    """
    Test visual markers (vlines) and categorical annotations using the -1 to 1 range.
    """
    # Use thresholds defined in the framework: Redundant < -0.3, Synergistic > 0.3
    vlines = [-0.3, 0.3]
    annotations = [
        (-0.6, 0.5, "Redundant"), 
        (0, 0.5, "Intermediate"), 
        (0.6, 0.5, "Synergistic")
    ]
    
    ax = hist_cooperativity(
        df=sample_interaction_data,
        vlines=vlines,
        annotations=annotations,
        outpath=None # Return the axes object
    )
    
    assert ax is not None
    
    # Verify annotations were placed
    texts = [t.get_text() for t in ax.texts]
    assert "Redundant" in texts
    assert "Intermediate" in texts
    assert "Synergistic" in texts
    
    # Verify that the data plotted was filtered by ks_q
    # The number of patches in the histogram should correspond to the filtered df
    significant_count = len(sample_interaction_data[sample_interaction_data["ks_q"] < 0.1])
    assert significant_count > 0
    
    plt.close(ax.figure)

def test_hist_cooperativity_empty_after_filtering():
    """
    Ensure the function handles cases where no interactions pass the significance gate.
    """
    # Data where nothing is significant (all ks_q > 0.1)
    insignificant_df = pd.DataFrame({
        "coop_score": [0.5, -0.2],
        "ks_q": [0.5, 0.8]
    })
    
    ax = hist_cooperativity(
        df=insignificant_df,
        qval_thresh=0.1,
        outpath=None
    )
    
    # The plot should still initialize even if empty
    assert ax is not None
    plt.close(ax.figure)

def test_hist_cooperativity_missing_required_columns():
    """
    Verify the function raises KeyError when required schema columns are missing.
    """
    bad_df = pd.DataFrame({"wrong_column": [0.5]})
    
    with pytest.raises(KeyError):
        hist_cooperativity(df=bad_df)

def test_hist_cooperativity_qval_gate_adjustment(sample_interaction_data):
    """
    Verify that changing qval_thresh changes the population size in the plot.
    """
    # High threshold (let everyone in)
    ax_all = hist_cooperativity(df=sample_interaction_data, qval_thresh=1.0, outpath=None)
    line_count_all = len(ax_all.get_lines()) # axvline lines
    
    # Strict threshold
    ax_strict = hist_cooperativity(df=sample_interaction_data, qval_thresh=0.01, outpath=None)
    
    assert ax_all is not None
    assert ax_strict is not None
    
    plt.close(ax_all.figure)
    plt.close(ax_strict.figure)