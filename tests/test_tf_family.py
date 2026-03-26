import os
import pandas as pd
import pytest
import numpy as np
from deepISA.validating.tf_family import (
    annotate_tf_family, 
    plot_synergy_by_tf_pair_class, 
    plot_synergy_by_dbd
)

@pytest.fixture
def mock_df():
    """Provides synthetic interaction data with the new pipe delimiter."""
    return pd.DataFrame({
        'pair': ['ALX3|DLX2', 'GATA3|GATA6', 'ALX3|GATA3', 'MYOD1|MYOG', 'MYOD1|GATA3'],
        'synergy_score': [0.11, 0.22, 0.83, 0.14, 0.95]
    })

def test_annotate_idempotency(mock_df):
    """Ensure running annotation twice doesn't change the results or columns."""
    df_first = annotate_tf_family(mock_df)
    df_second = annotate_tf_family(df_first)
    assert list(df_first.columns) == list(df_second.columns)
    assert df_first['same_family'].equals(df_second['same_family'])

def test_annotate_with_unknown_tfs():
    """Verify that unknown TF symbols don't crash the merger and result in False for same_family."""
    df = pd.DataFrame({'pair': ['FAKE1|FAKE2'], 'synergy_score': [0.5]})
    annotated = annotate_tf_family(df)
    assert pd.isna(annotated.loc[0, 'tf1_family'])
    assert annotated.loc[0, 'same_family'] == False

def test_case_insensitivity():
    """Ensure lowercase TF symbols in the pair string are correctly matched to the reference."""
    df = pd.DataFrame({'pair': ['alx3|dlx2'], 'synergy_score': [0.5]})
    annotated = annotate_tf_family(df)
    # Check against the actual JASPAR family name in your reference CSV
    assert annotated.loc[0, 'tf1_family'] == 'Paired-related HD factors'

def test_plot_functions_handle_missing_scores(mock_df, tmp_path):
    """Verify that if all scores are NaN, the function returns without attempting to save a file."""
    df_nan = mock_df.copy()
    df_nan['synergy_score'] = np.nan
    out = tmp_path / "test_nan.pdf"
    
    plot_synergy_by_tf_pair_class(df_nan, outpath=str(out))
    
    # The file should NOT exist because the function returns early on empty/NaN data
    assert not out.exists()

def test_dbd_boxplot_filtering(tmp_path):
    """
    Verify that only valid JASPAR TFs are plotted in the DBD boxplot.
    PA2G4 should be filtered out.
    """
    # 1. Prepare data
    df = pd.DataFrame({'pair': ['ALX3|PA2G4'], 'synergy_score': [0.8]})
    annotated = annotate_tf_family(df)
    
    # 2. Use tmp_path joined with a filename
    # This creates an absolute path in a temp directory, making it directory-independent
    out_file = tmp_path / "test_dbd_filter.pdf"
    
    # 3. Pass the path as a string
    plot_synergy_by_dbd(annotated, outpath=str(out_file))
    
    # 4. Verify using the Path object's exists method
    assert out_file.exists()
    assert out_file.stat().st_size > 0