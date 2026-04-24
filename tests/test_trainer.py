import torch
import torch.nn as nn
import pytest
import os
from deepISA.modeling.trainer import Trainer

# Define a minimal model for testing
class SimpleModel(nn.Module):
    def __init__(self, mode='dual'):
        super().__init__()
        self.mode = mode
        self.fc = nn.Linear(1, 1)

    def forward(self, x):
        # Return dummy values matching dual mode (2 outputs: regression, classification)
        if self.mode == 'dual':
            return torch.cat([x, x], dim=1)
        return x

def test_compute_loss_logic(tmp_path):
    """Verify loss calculation switches correctly between modes."""
    model = SimpleModel(mode='dual')
    
    # Matching the POSITIONAL order in your trainer.py:
    # 1. model, 2. train_dat, 3. val_dat, 4. test_dat, 5. device, 6. model_dir
    trainer = Trainer(
        model,              # model
        None,               # train_dat
        None,               # val_dat
        None,               # test_dat
        torch.device('cpu'),# device
        str(tmp_path)       # model_dir
    )
    
    # Create fake preds [batch, 2] and targets
    # Column 0: Regression, Column 1: Classification Logit
    preds = torch.tensor([[1.0, 5.0]], requires_grad=True) 
    byr = torch.tensor([1.0])
    byc = torch.tensor([1.0])
    
    # Test Dual Loss
    loss_dual = trainer._compute_loss(preds, byr, byc)
    assert loss_dual > 0
    
    # Test Regression-only Loss
    trainer.mode = 'regression'
    loss_reg = trainer._compute_loss(preds, byr, byc)
    # Regression only (MSE) should be smaller than Dual (MSE + BCE)
    assert loss_reg < loss_dual

def test_trainer_initialization(tmp_path):
    """Ensure trainer correctly identifies mode and sets up directory."""
    model = SimpleModel(mode='classification')
    
    trainer = Trainer(
        model, 
        None, 
        None, 
        None, 
        torch.device('cpu'), 
        str(tmp_path)
    )
    
    assert trainer.mode == 'classification'
    # Verify that the trainer actually created the directory
    assert os.path.exists(str(tmp_path))
    assert isinstance(trainer.reg_criterion, torch.nn.MSELoss)
    assert isinstance(trainer.clf_criterion, torch.nn.BCEWithLogitsLoss)