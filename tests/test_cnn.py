import torch
import pytest
from deepISA.modeling.cnn import Conv

def test_conv_output_shape():
    seq_len = 600
    ks = [15, 9]
    cs = [16, 32]
    ds = [1, 2]
    # Test Dual Mode
    model = Conv(seq_len, ks, cs, ds, mode='dual')
    x = torch.randn(8, 4, seq_len)
    out = model(x)
    assert out.shape == (8, 2)  # [batch, reg + clf]


def test_regression_mode_shape():
    """
    Test that 'regression' mode returns (Batch, 1) and 
    only initializes the regression head.
    """
    seq_len = 600
    ks = [15, 9, 9]
    cs = [16, 32, 64]
    ds = [1, 2, 4]
    batch_size = 4
    model = Conv(seq_len, ks, cs, ds, mode='regression')
    x = torch.randn(batch_size, 4, seq_len)
    out = model(x)
    # Check output dimensions
    assert out.shape == (batch_size, 1)
    # Check internal structure: regression head exists, classification doesn't
    assert model.regression_head is not None
    assert model.classification_head is None


def test_classification_mode_shape():
    """
    Test that 'classification' mode returns (Batch, 1) and 
    only initializes the classification head.
    """
    seq_len = 600
    ks = [15, 9, 9]
    cs = [16, 32, 64]
    ds = [1, 2, 4]
    batch_size = 4
    model = Conv(seq_len, ks, cs, ds, mode='classification')
    x = torch.randn(batch_size, 4, seq_len)
    out = model(x)
    # Check output dimensions (logits)
    assert out.shape == (batch_size, 1)
    # Check internal structure: classification head exists, regression doesn't
    assert model.classification_head is not None
    assert model.regression_head is None


def test_receptive_field_calc():
    # RF = 1 + sum((k-1)*d)
    # 1 + (15-1)*1 + (9-1)*2 = 1 + 14 + 16 = 31
    model = Conv(600, [15, 9], [16, 32], [1, 2], 'regression')
    assert model.rf == 31


def test_invalid_params():
    with pytest.raises(ValueError):
        # Mismatched lengths
        Conv(600, [15], [16, 32], [1], 'regression')