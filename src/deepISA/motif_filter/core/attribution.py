"""
DeepLIFT attribution for motif_filter.

Step 1 of the pipeline.
"""
import torch
import numpy as np
from typing import Optional, Tuple
from captum.attr import DeepLift


class _RegressModel(torch.nn.Module):
    """Wrap deepISA Conv model to expose only the regression output."""

    def __init__(self, base_model: torch.nn.Module):
        super().__init__()
        self.base = base_model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Conv1d expects (N, C, L); output shape (N, 2) → return (N, 1)
        return self.base(x)[:, 0:1]


def init_explainer(model: torch.nn.Module, device: str = "cuda") -> Tuple[DeepLift, _RegressModel]:
    """
    Initialize the DeepLIFT explainer and wrapped model once.

    Call this at pipeline entry, then pass the returned explainer
    to compute_attribution to avoid repeated GPU allocation.

    Parameters
    ----------
    model : torch.nn.Module
        Trained deepISA Conv model.
    device : str
        Compute device.

    Returns
    -------
    tuple: (explainer, wrapped_model)
    """
    wrapped = _RegressModel(model).to(device).eval()
    explainer = DeepLift(wrapped)
    return explainer, wrapped


def compute_attribution(
    model: torch.nn.Module,
    seqs_ohe: np.ndarray,
    device: str = "cuda",
    explainer: Optional[DeepLift] = None,
) -> np.ndarray:
    """
    Compute DeepLIFT attribution scores for one-hot encoded sequences.

    Parameters
    ----------
    model : torch.nn.Module
        Trained deepISA Conv model (must expose .seq_len attribute or seq_len=600).
    seqs_ohe : np.ndarray, shape (batch, length, 4)
        One-hot encoded DNA sequences.
    device : str
        Compute device.
    explainer : DeepLift, optional
        Pre-initialized explainer from init_explainer().
        If None, creates one internally (slower for repeated calls).

    Returns
    -------
    attr : np.ndarray, shape (batch, length, 4)
        Attribution scores per position per channel.
        baseline = zeros (uniform neutral).
    """
    # (batch, length, 4) → (batch, 4, length) for Conv1d
    inp = (
        torch.tensor(seqs_ohe.transpose(0, 2, 1), dtype=torch.float32, device=device)
        .requires_grad_(True)
    )
    baseline = torch.zeros_like(inp)

    if explainer is None:
        wrapped = _RegressModel(model).to(device).eval()
        _explainer = DeepLift(wrapped)
    else:
        _explainer = explainer

    attr = (
        _explainer.attribute(inp, baselines=baseline, target=0)
        .cpu()
        .detach()
        .numpy()
    )  # (batch, 4, length)

    # Transpose back to (batch, length, 4)
    return attr.transpose(0, 2, 1)
