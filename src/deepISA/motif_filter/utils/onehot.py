"""One-hot encoding for DNA sequences."""
import numpy as np
from typing import List


def encode_sequences(sequences: List[str], target_len: int = 600) -> np.ndarray:
    """
    One-hot encode a list of DNA sequences.

    Parameters
    ----------
    sequences : list of str
        DNA sequences (A/C/G/T only; N/n handled as zeros).
    target_len : int
        Fixed-length output; sequences shorter are padded with zeros,
        longer are truncated.

    Returns
    -------
    arr : np.ndarray, shape (len(sequences), target_len, 4)
        One-hot encoding in (batch, length, channels) format.
        Channels are [A, C, G, T] in alphabetical order.
    """
    mapping = {"A": 0, "C": 1, "G": 2, "T": 3}
    arr = np.zeros((len(sequences), target_len, 4), dtype=np.float32)
    for i, seq in enumerate(sequences):
        if not isinstance(seq, str):
            continue
        valid = min(len(seq), target_len)
        for j in range(valid):
            if seq[j] in mapping:
                arr[i, j, mapping[seq[j]]] = 1.0
    return arr
