import pytest
import pandas as pd
import numpy as np
from unittest.mock import MagicMock, patch
from deepISA.modeling.preprocess import compile_training_data

@pytest.fixture
def mock_df():
    """Returns a basic 3-region dataframe."""
    return pd.DataFrame({
        'chrom': ['chr1', 'chr1', 'chr1'],
        'start': [100, 200, 300],
        'end': [700, 800, 900]
    })

def test_compile_scenario_3_direct(mock_df):
    """Scenario 3: Verifies that pre-labeled data skips sampling and quantification."""
    df = mock_df.copy()
    df['reg_val'] = [1.5, 2.5, 3.5]
    df['class_val'] = [1.0, 1.0, 1.0]
    
    # We do NOT patch background pool here to ensure it isn't even called
    result = compile_training_data(
        df, seq_len=600, 
        target_reg_col='reg_val', 
        target_class_col='class_val'
    )
    
    assert 'target_reg' in result.columns
    assert 'target_class' in result.columns
    assert len(result) == 3  # Should not have sampled extra negatives
    assert (result['target_class'] == 1.0).all()

@patch('deepISA.modeling.preprocess.bf.read_table')
@patch('deepISA.modeling.preprocess.get_data_resource')
def test_compile_scenario_1_balancing(mock_get_res, mock_read_table, mock_df):
    """Scenario 1: Verifies that signal-only data triggers negative sampling."""
    df = mock_df.copy()
    df['my_signal'] = [10.0, 20.0, 30.0]
    
    # Mock background pool with 10 regions
    bg_df = pd.DataFrame({
        'chrom': ['chr1']*10, 'start': range(0, 10000, 1000), 'end': range(600, 10600, 1000)
    })
    mock_read_table.return_value = bg_df
    
    result = compile_training_data(
        df, seq_len=600, target_reg_col='my_signal'
    )
    
    # 3 positives + 3 sampled negatives = 6 total
    assert len(result) == 6
    assert (result['target_class'] == 0.0).sum() == 3
    assert (result['target_class'] == 1.0).sum() == 3

@patch('deepISA.modeling.preprocess.quantify_bw')
@patch('deepISA.modeling.preprocess.estimate_noise_threshold')
@patch('deepISA.modeling.preprocess.bf.read_table')
def test_compile_scenario_2_bigwig(mock_read_table, mock_noise, mock_quant, mock_df):
    """Scenario 2: Verifies that BigWig paths trigger quantification and balancing."""
    # Mock 2 positives (>1.0) and 1 negative (<1.0)
    mock_quant.return_value = (np.array([10.0, 0.1, 15.0]), mock_df)
    mock_noise.return_value = 1.0
    mock_read_table.return_value = pd.DataFrame({
        'chrom': ['chr1']*5, 'start': [0]*5, 'end': [600]*5
    })
    
    result = compile_training_data(mock_df, seq_len=600, bw_paths=['test.bw'])
    
    assert mock_quant.called
    assert mock_noise.called
    # 2 positives, 1 internal negative -> needs 1 more from pool = 4 total
    assert len(result) == 4


