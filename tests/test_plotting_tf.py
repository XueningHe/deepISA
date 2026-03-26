import pytest
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from deepISA.plotting.tf import compare_tf_importance


# TODO: this might be more used for cross model comparison

@pytest.fixture
def sample_tf_data():
    """Provides a synthetic DataFrame mimicking ISA importance results."""
    return pd.DataFrame({
        "tf": ["GATA1", "TAL1", "CTCF", "MYC", "HNF4A", "CEBPA", "SPI1", "FOXA1"],
        "dstat_1": [0.8, 0.75, 0.9, 0.4, 0.1, 0.05, 0.6, 0.0],
        "dstat_2": [0.2, 0.15, 0.85, 0.5, 0.7, 0.8, 0.1, 0.75]
    })

def test_compare_tf_importance_saves_file(sample_tf_data, tmp_path):
    """Verify that the function correctly writes a PDF to disk."""
    output_file = tmp_path / "test_plot.pdf"
    
    # Run function with file path
    compare_tf_importance(
        data=sample_tf_data,
        x_col="dstat_1",
        y_col="dstat_2",
        output_path=str(output_file)
    )
    
    assert output_file.exists()
    assert output_file.stat().st_size > 0

def test_compare_tf_importance_returns_figure(sample_tf_data):
    """Verify that the function returns a Figure object when output_path is None."""
    fig = compare_tf_importance(
        data=sample_tf_data,
        x_col="dstat_1",
        y_col="dstat_2",
        output_path=None
    )
    
    assert isinstance(fig, plt.Figure)
    # Check if we have the expected number of axes (Main plot + Colorbar)
    assert len(fig.axes) == 2 
    plt.close(fig)


def test_compare_tf_importance_with_custom_args(sample_tf_data, tmp_path):
    """Ensure flat arguments for aesthetics are respected without crashing."""
    output_file = tmp_path / "config_test.png"
    try:
        compare_tf_importance(
            data=sample_tf_data,
            x_col="dstat_1",
            y_col="dstat_2",
            output_path=str(output_file),
            fig_size=(5, 5),
            font_size=12,
            dpi=100,
            marker_size=25,
            text_alpha=1.0
        )
    except Exception as e:
        pytest.fail(f"Plotting with custom arguments raised {e}")



def test_compare_tf_importance_empty_data_handling():
    """Verify behavior with empty dataframe (it should not crash)."""
    empty_df = pd.DataFrame({"tf": [], "x": [], "y": []})
    # Run without expecting a crash
    fig = compare_tf_importance(
        data=empty_df, 
        x_col="x", 
        y_col="y", 
        output_path=None
    )
    assert isinstance(fig, plt.Figure)
    # Check that axes exist even if empty
    assert len(fig.axes) >= 1
    plt.close(fig)



def test_compare_tf_importance_label_fallback(sample_tf_data):
    """Verify that column names are used as labels if none are provided."""
    fig = compare_tf_importance(
        data=sample_tf_data,
        x_col="dstat_1",
        y_col="dstat_2",
        x_label=None,
        y_label=None,
        output_path=None
    )
    ax = fig.gca()
    assert ax.get_xlabel() == "dstat_1"
    assert ax.get_ylabel() == "dstat_2"
    plt.close(fig)


