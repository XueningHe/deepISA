import torch
import numpy as np
from loguru import logger

from deepISA.utils import one_hot_encode
from deepISA.utils import find_available_gpu




def compute_predictions(model, seqs, device, batch_size):
    """
    Computes predictions for a list of DNA sequences.
    
    Args:
        model: Trained deepISA.modeling.cnn.Conv model.
        seqs: List of strings (DNA sequences, length 600).
        device: torch.device (defaults to GPU if available).
        batch_size: Number of sequences to process at once.
        
    Returns:
        numpy.ndarray: Predicted values of shape [num_seqs, num_tracks].
    """
    model.to(device)
    model.eval()
    
    all_preds = []
    
    # Process sequences in batches
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





