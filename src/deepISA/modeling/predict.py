import torch
import numpy as np
from loguru import logger
from scipy.stats import pearsonr

from deepISA.utils import one_hot_encode

from deepISA.utils import find_available_gpu




def compute_predictions(model, seqs, device=None, batch_size=1024):
    """
    Computes predictions for a list of DNA sequences.
    
    Args:
        model: Trained deepISA.modeling.cnn.Conv model.
        seqs: List of strings (DNA sequences, length 600).
        batch_size: Number of sequences to process at once.
        device: torch.device (defaults to GPU if available).
        
    Returns:
        numpy.ndarray: Predicted values of shape [num_seqs, num_tracks].
    """
    if device is None:
        device = find_available_gpu()
    model.to(device)
    model.eval()
    
    all_preds = []
    
    # Process sequences in batches to save memory
    # TODO: maybe write a logger to indicate progress?
    for i in range(0, len(seqs), batch_size):
        batch_seqs = seqs[i : i + batch_size]
        # 1. One-hot encode strings to (N, 4, 600)
        x_np = one_hot_encode(batch_seqs)
        x_tensor = torch.from_numpy(x_np).to(device)
        # 2. Forward pass
        with torch.no_grad():
            # Based on your Conv forward pass: returns torch.cat([reg, clf], dim=1)
            batch_preds = model(x_tensor)
            all_preds.append(batch_preds.cpu().numpy())
    # Concatenate all batch results into a single matrix
    return np.concatenate(all_preds, axis=0)










def evaluate_model(model, dataset, device=None, batch_size=1024):
    """
    Evaluates the model on a given dataset (typically the test_ds).
    Returns the Pearson correlation coefficient for the regression task.
    """
    if device is None:
        device = find_available_gpu()
        
    model.to(device)
    model.eval()
    all_preds = []
    all_gts = []
    
    
    logger.info(f"Evaluating model on {len(dataset)} samples...")
    
    with torch.no_grad():
        # Iterate in batches to respect GPU memory
        for i in range(0, len(dataset), batch_size):
            X, Yr, Yc = dataset[i : i + batch_size]
            X = X.to(device)
            output = model(X)
            preds = output[:, 0].cpu().numpy()
            all_preds.extend(preds)
            all_gts.extend(Yr.numpy())
            
    # Calculate Pearson correlation
    r_val, p_val = pearsonr(all_preds, all_gts)
    
    logger.info(f"Evaluation Complete | Pearson r: {r_val:.4f} (p={p_val:.2e})")
    return r_val