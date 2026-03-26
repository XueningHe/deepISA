import pytest
import pandas as pd
import numpy as np
import os
from unittest.mock import patch, MagicMock
from deepISA.scoring.infer_tf_expr import get_expressed_tfs

@pytest.fixture
def mock_internal_bed(tmp_path):
    """Creates a dummy promoter BED file and returns its path."""
    bed_path = tmp_path / "hg38_TF_promoters_500bp.bed"
    data = [
        ["chr1", 1000, 2000, "GATA1", 0, "+"],   # High Signal
        ["chr1", 3000, 4000, "SOX2", 0, "-"],    # Low Signal
        ["chr2", 5000, 6000, "TAL1", 0, "+"],    # On Threshold
        ["chr3", 100, 1100,  "NANOG", 0, "+"]    # High Signal
    ]
    df = pd.DataFrame(data)
    df.to_csv(bed_path, sep='\t', index=False, header=False)
    return str(bed_path)

# IMPORTANT: Patch the namespace where get_expressed_tfs consumes these utilities
@patch("deepISA.scoring.infer_tf_expr.get_data_resource")
@patch("deepISA.scoring.infer_tf_expr.quantify_bw")
@patch("deepISA.scoring.infer_tf_expr.estimate_noise_threshold")
def test_get_expressed_tfs_full_logic(mock_noise, mock_quantify, mock_get_res, mock_internal_bed):
    """Tests resource loading, signal quantification, and noise filtering."""
    # 1. Setup Mocks
    mock_get_res.return_value = mock_internal_bed
    mock_noise.return_value = 5.0
    
    # Mock signals corresponding to the 4 rows in mock_internal_bed
    mock_signals = np.array([12.0, 1.0, 5.0, 50.0])
    mock_quantify.return_value = (mock_signals, MagicMock())

    # 2. Execute
    bw_paths = ["mock_plus.bw", "mock_minus.bw"]
    expressed_tfs = get_expressed_tfs(bw_paths=bw_paths, seq_len=600)

    # 3. Assertions
    mock_get_res.assert_called_once_with("hg38_TF_promoters_500bp.bed")
    
    # Expect GATA1 (>5) and NANOG (>5). TAL1 (==5) is excluded by strict inequality.
    expected_tfs = ["GATA1", "NANOG"]
    assert expressed_tfs == expected_tfs
    assert len(expressed_tfs) == 2
    assert expressed_tfs == sorted(expressed_tfs)

@patch("deepISA.scoring.infer_tf_expr.get_data_resource")
@patch("deepISA.scoring.infer_tf_expr.estimate_noise_threshold")
@patch("deepISA.scoring.infer_tf_expr.quantify_bw")
def test_get_expressed_tfs_all_silent(mock_quantify, mock_noise, mock_get_res, mock_internal_bed):
    """Verifies empty list when no TFs exceed noise."""
    mock_get_res.return_value = mock_internal_bed
    mock_noise.return_value = 100.0
    mock_quantify.return_value = (np.array([1.0, 2.0, 3.0, 4.0]), MagicMock())

    result = get_expressed_tfs(["fake.bw"])
    assert result == []

@patch("deepISA.scoring.infer_tf_expr.get_data_resource")
def test_missing_internal_resource(mock_get_res):
    """Tests handling of missing internal resource file."""
    mock_get_res.side_effect = FileNotFoundError("Internal resource not found")
    
    with pytest.raises(FileNotFoundError):
        get_expressed_tfs(["fake.bw"])