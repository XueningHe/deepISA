import pytest
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from deepISA.plotting.tf import (
    parse_jaspar_pfms, 
    plot_coop_vs_importance, 
    plot_partner_specificity
)

# --- FIXTURES ---

@pytest.fixture
def mock_jaspar_file(tmp_path):
    """Creates a dummy JASPAR PFM file for parsing tests."""
    d = tmp_path / "dummy_jaspar.txt"
    content = (
        ">MA0001.1\tTF1\n"
        "A [ 10.0 10.0 ]\nC [ 10.0 10.0 ]\nG [ 10.0 10.0 ]\nT [ 10.0 10.0 ]\n"
        ">MA0002.1\tTF2\n"
        "A [ 15.0 15.0 ]\nC [ 5.0 5.0 ]\nG [ 5.0 5.0 ]\nT [ 15.0 15.0 ]\n"
    )
    d.write_text(content)
    return str(d)

@pytest.fixture
def dummy_tf_data():
    """Provides columns to satisfy assign_cooperativity and plotting slices."""
    return pd.DataFrame({
        'tf': ['TF1', 'TF2', 'TF3', 'TF4', 'TF5'],
        'coop_score': [1.5, -1.2, 0.1, 0.8, -0.5],
        'ks_q': [0.001, 0.001, 0.5, 0.01, 0.02],
        'ks_stat': [0.5, 0.5, 0.1, 0.3, 0.2],
        # Added this so the merge doesn't fail if assign_cooperativity isn't called yet
        'cooperativity': ['Synergistic', 'Redundant', 'Independent', 'Synergistic', 'Redundant']
    })

# --- TESTS ---

def test_parse_jaspar_pfms(mock_jaspar_file):
    df = parse_jaspar_pfms(mock_jaspar_file)
    assert len(df) == 2
    assert df.loc[df['tf'] == 'TF1', 'GC'].iloc[0] == 0.5

def test_plot_coop_vs_importance_execution(dummy_tf_data):
    df_imp = pd.DataFrame({
        'tf': ['TF1', 'TF2', 'TF3', 'TF4', 'TF5'],
        'mean_isa_t0': [0.5, 0.4, 0.1, 0.3, 0.2]
    })
    fig = plot_coop_vs_importance(dummy_tf_data, df_imp)
    assert isinstance(fig, plt.Figure)
    plt.close(fig)

def test_plot_partner_specificity_logic(dummy_tf_data):
    df_pairs = pd.DataFrame({
        'tf_pair': ['TF1|TF2', 'TF1|TF3', 'TF1|TF4', 'TF2|TF3', 'TF2|TF4', 'TF2|TF5'],
        'abs_i_sum': [100, 20, 10, 80, 50, 30]
    })
    # Fixed: KDE Guard handled in the tf.py function provided in previous step
    fig = plot_partner_specificity(df_pairs, dummy_tf_data, top_n=1, min_partners=2)
    assert isinstance(fig, plt.Figure)
    plt.close(fig)

def test_partner_specificity_empty_returns_none(dummy_tf_data):
    df_pairs = pd.DataFrame({'tf_pair': ['TF1|TF2'], 'abs_i_sum': [10]})
    result = plot_partner_specificity(df_pairs, dummy_tf_data, min_partners=10)
    assert result is None

def test_empty_merge_handling():
    """Ensure code handles cases where TF names do not overlap without crashing on ylim."""
    df1 = pd.DataFrame({
        'tf': ['A'], 'coop_score': [1], 'ks_q': [0.001], 'ks_stat': [0.5], 'cooperativity': ['Synergistic']
    })
    df2 = pd.DataFrame({
        'tf': ['B'], 'mean_isa_t0': [1]
    })
    
    # We expect the function to either return None or a valid Figure
    # If your function crashes on ylim, you should add 'if df.empty: return None' to tf.py
    try:
        fig = plot_coop_vs_importance(df1, df2)
        if fig:
            assert isinstance(fig, plt.Figure)
            plt.close(fig)
    except ValueError as e:
        pytest.fail(f"Plotting crashed on empty merge: {e}")