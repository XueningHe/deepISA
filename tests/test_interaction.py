import pytest
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from unittest.mock import patch

from deepISA.plotting.interaction import (
    plot_null,
    plot_tf_pair_against_null, 
    plot_interaction_decay,
    _get_cbrt_scale
)



# Assuming the functions are in a file named visualization.py
# from visualization import plot_null, plot_tf_pair_against_null, plot_interaction_decay

@pytest.fixture
def sample_df():
    """Creates a dummy dataframe for testing."""
    np.random.seed(42)
    data = {
        "distance": np.random.randint(50, 300, 100),
        "interaction_t0": np.random.normal(0, 1, 100),
        "interaction_t1": np.random.normal(0, 1, 100),
        "tf1": np.random.choice(["A", "B", "C"], 100),
        "tf2": np.random.choice(["X", "Y", "Z"], 100),
    }
    return pd.DataFrame(data)

## --- Tests for plot_null ---

def test_plot_null_empty_range(sample_df):
    # Test with a distance range that doesn't exist
    result = plot_null(sample_df, min_dist=1000, max_dist=2000)
    assert result is None

def test_plot_null_returns_axes(sample_df):
    ax = plot_null(sample_df, track_idx=0)
    assert isinstance(ax, plt.Axes)
    plt.close()

@patch("matplotlib.pyplot.savefig")
def test_plot_null_saves_file(mock_save, sample_df):
    plot_null(sample_df, outpath="test_plot.png")
    mock_save.assert_called_once()

## --- Tests for plot_tf_pair_against_null ---

def test_plot_tf_pair_no_data(sample_df):
    # Search for a TF pair that doesn't exist in the random sample
    result = plot_tf_pair_against_null(sample_df, tf_pair=("Non", "Existent"))
    assert result is None

def test_plot_tf_pair_kde_vs_cdf(sample_df):
    # Get a valid pair from the dataframe
    pair = (sample_df.iloc[0]['tf1'], sample_df.iloc[0]['tf2'])
    
    ax_kde = plot_tf_pair_against_null(sample_df, tf_pair=pair, plot_type='kde')
    assert ax_kde.get_ylabel() == 'Density (cbrt)'
    
    ax_cdf = plot_tf_pair_against_null(sample_df, tf_pair=pair, plot_type='cdf')
    assert ax_cdf.get_ylabel() == 'Cumulative Prob'
    plt.close('all')

## --- Tests for plot_interaction_decay ---

def test_plot_interaction_decay_multi_track(sample_df):
    # Test passing a list of tracks
    ax = plot_interaction_decay(sample_df, track_idx=[0, 1], mode='absolute')
    # Check if legend contains both tracks
    legend_texts = [t.get_text() for t in ax.get_legend().get_texts()]
    assert "T0" in legend_texts
    assert "T1" in legend_texts
    plt.close()

def test_plot_interaction_decay_signed_logic(sample_df):
    ax = plot_interaction_decay(sample_df, track_idx=0, mode='signed')
    # In signed mode, there should be an axhline at 0
    # Matplotlib stores lines; index 0 is usually the data, look for the hline
    has_hline = any(line.get_ydata()[0] == 0 for line in ax.get_lines() if len(line.get_ydata()) > 0)
    assert has_hline
    plt.close()

## --- Tests for Helper Logic ---


def test_cbrt_scale_math():
    f_wd, f_inv = _get_cbrt_scale()
    assert f_wd(8) == 2.0
    assert f_wd(-8) == -2.0
    assert f_inv(2) == 8.0