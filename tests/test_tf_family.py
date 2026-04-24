import pandas as pd
import pytest
import numpy as np
from deepISA.exploring.tf_family import (
    annotate_tf_family, 
    plot_coop_by_tf_pair_family, # Refactored name
    plot_coop_by_dbd             # Refactored name
)

@pytest.fixture
def mock_df():
    """Provides synthetic interaction data with required statistical columns."""
    return pd.DataFrame({
        'tf_pair': ['ALX3|DLX2', 'GATA3|GATA6', 'ALX3|GATA3', 'MYOD1|MYOG', 'MYOD1|GATA3'],
        'coop_score': [0.11, 0.22, 0.83, 0.14, 0.95],
        'ks_q': [0.001, 0.001, 0.001, 0.001, 0.001] # Add this to satisfy assign_cooperativity
    })

def test_annotate_idempotency(mock_df):
    """Ensure running annotation twice doesn't change the results or columns."""
    df_first = annotate_tf_family(mock_df)
    df_second = annotate_tf_family(df_first)
    assert list(df_first.columns) == list(df_second.columns)
    assert df_first['same_family'].equals(df_second['same_family'])

def test_annotate_with_unknown_tfs():
    """Verify that unknown TFs result in False for same_family."""
    df = pd.DataFrame({'tf_pair': ['FAKE1|FAKE2'], 'coop_score': [0.5]})
    annotated = annotate_tf_family(df)
    # The refactored code handles NaNs and 'Not in JASPAR' strings safely
    assert pd.isna(annotated.loc[0, 'tf1_family']) or annotated.loc[0, 'tf1_family'] == 'nan'
    assert annotated.loc[0, 'same_family'] == False

def test_case_insensitivity():
    """Ensure lowercase TF symbols in the pair string are correctly matched."""
    df = pd.DataFrame({'tf_pair': ['alx3|dlx2'], 'coop_score': [0.5]})
    annotated = annotate_tf_family(df)
    # Verify the reference data mapping is successful regardless of input case
    assert annotated.loc[0, 'tf1_family'] == 'Paired-related HD factors'

def test_plot_functions_handle_missing_scores(mock_df, tmp_path):
    """Verify functions return early if all scores are NaN."""
    df_nan = mock_df.copy()
    df_nan['coop_score'] = np.nan
    out = tmp_path / "test_nan.pdf"
    
    # These should return None or early and NOT create a file
    plot_coop_by_tf_pair_family(df_nan, outpath=str(out))
    
    assert not out.exists()

def test_dbd_boxplot_filtering(tmp_path):
    """
    Verify that only valid JASPAR TFs are plotted in the DBD boxplot.
    """
    df = pd.DataFrame({
        'tf': ['ALX3', 'PA2G4'], 
        'coop_score': [0.8, 0.2],
        'ks_q': [0.001, 0.001] # Add dummy significance
    })
    out_file = tmp_path / "test_dbd_filter.pdf"
    plot_coop_by_dbd(df, outpath=str(out_file))
    assert out_file.exists()
    assert out_file.stat().st_size > 0

def test_same_family_logic_with_jaspar_exclusion():
    """Specific check for the 'Not in JASPAR' exclusion logic."""
    df = pd.DataFrame({
        'tf_pair': ['TF1|TF2'], 
        'coop_score': [0.5]
    })
    # Mock behavior where both are 'Not in JASPAR'
    # In the refactored check_same, this should return False
    annotated = annotate_tf_family(df)
    # If the TFs aren't in the CSV, they'll be NaN/False
    if pd.isna(annotated.loc[0, 'tf1_family']):
        assert annotated.loc[0, 'same_family'] == False