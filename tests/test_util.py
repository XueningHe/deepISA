from deepISA.utils import one_hot_encode, ablate_motifs
import numpy as np


def test_one_hot_encode():
    seqs = ["ACGT", "NNNN"]
    out = one_hot_encode(seqs)
    assert out.shape == (2, 4, 4)
    # Check A (first channel, first pos)
    assert out[0, 0, 0] == 1.0
    # Check N (all zeros)
    assert np.all(out[1] == 0)

def test_ablate_motifs():
    seq = "ATGCATGC"
    # Ablate "GC" at index 2-3
    ablated = ablate_motifs(seq, [2], [3])
    assert ablated == "ATNNATGC"