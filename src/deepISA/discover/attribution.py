"""Per-position DeepLIFT-SHAP attribution via :mod:`tangermeme`.

This module replaces the mc000 ``tf.keras + shap + deeplift`` attribution stack
with a torch-native implementation. Rather than hand-rolling the DeepLIFT
multiplier, the hypothetical projection, and the dinucleotide-shuffle
background (three places where subtle bugs crept in across multiple review
rounds), we delegate to :func:`tangermeme.deep_lift_shap.deep_lift_shap` --
Jacob Schreiber's PyTorch-native, Kundaje-lab-ecosystem implementation of
DeepLIFT/SHAP for genomics models.

Why tangermeme
--------------
* It is the canonical PyTorch counterpart to the original TF-based
  ``deeplift`` / ``shap.DeepExplainer`` stack that mc000 used.
* It computes DeepLIFT-SHAP against **per-sequence** dinucleotide-shuffled
  references (each sequence against its own ``n_shuffles`` shuffles), which
  captum's ``DeepLiftShap`` alone cannot do (captum requires baselines that
  broadcast across the batch -- see meta-pytorch/captum#933).
* It implements the hypothetical-contribution projection correctly
  (``hypothetical=True`` returns the per-base multipliers; ``hypothetical=False``
  multiplies them by the one-hot input), so we get both ``hyp_scores`` and
  ``act_scores`` in the exact format tf-modisco-lite and Fi-NeMo expect.
* It is the same author/ecosystem as tf-modisco-lite, guaranteeing that the
  attribution convention matches what the downstream CLI consumes.

Two scores are produced for every input, both shaped ``(T, N, 4, L)``:

* ``hyp_scores`` -- *hypothetical* DeepLIFT-SHAP contributions (the multiplier
  at every base, regardless of which is observed). This is the standard ``-a``
  input to tf-modisco-lite and Fi-NeMo.
* ``act_scores`` -- *actual* contributions, i.e. ``hyp_scores`` multiplied by
  the one-hot input (only the observed base is non-zero). Useful for logos.
"""

from __future__ import annotations

import os
from typing import Optional, Sequence

import h5py
import numpy as np
import torch
from loguru import logger
from tqdm.auto import tqdm

__all__ = ["compute_attribution"]

_H5_LAYOUT = "channels_first"   # deepISA native (N, 4, L)


def compute_attribution(
    model: torch.nn.Module,
    seqs_ohe: np.ndarray,
    tracks: Sequence[int] = (0,),
    device: Optional[torch.device | str] = None,
    n_refs: int = 100,
    batch_size: int = 64,
    seed: Optional[int] = 0,
    save_h5_path: Optional[str] = None,
    ids: Optional[Sequence[str]] = None,
    show_progress: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute hypothetical + actual DeepLIFT-SHAP attributions via tangermeme.

    Parameters
    ----------
    model : torch.nn.Module
        Trained deepISA model (e.g. :class:`deepISA.modeling.cnn.Conv`).
    seqs_ohe : np.ndarray
        One-hot input sequences, shape ``(N, 4, L)`` (deepISA native layout).
    tracks : sequence of int
        Output indices to attribute. One attribution pass is run per track and
        results are stacked along a new leading track axis.
    device : torch.device or str, optional
        Compute device. Defaults to CUDA if available else CPU.
    n_refs : int
        Number of dinucleotide-shuffled references per sequence. Each sequence
        is attributed against its own ``n_refs`` shuffles (DeepLIFT-SHAP
        averages over them internally).
    batch_size : int
        Number of (sequence, reference) pairs processed simultaneously by
        tangermeme. Memory/throughput trade-off.
    seed : int, optional
        Reproducibility seed forwarded to tangermeme's shuffler.
    save_h5_path : str, optional
        If given, streams results to an HDF5 file with datasets ``sequences``,
        ``hyp_scores``, ``act_scores`` (each ``(T, N, 4, L)``), plus optional
        ``id`` and a ``layout`` attribute recording the channels-first convention.
    ids : sequence of str, optional
        Per-sequence identifiers written to the H5 ``id`` dataset if saving.
    show_progress : bool
        Whether to display a tqdm progress bar over tracks.

    Returns
    -------
    (hyp_scores, act_scores) : tuple of np.ndarray
        Both arrays have shape ``(len(tracks), N, 4, L)`` -- track-leading so
        downstream consumers can index ``hyp_scores[t]`` per task.

    Notes
    -----
    Unknown bases (``N``) encode to all-zero columns which tangermeme rejects
    (``X must be one-hot encoded ... cannot have unknown characters``). They
    are imputed with seeded-random ACGT before attribution; the H5 ``sequences``
    dataset holds the imputed, strictly one-hot sequences. The caller's input
    array is never modified.
    """
    # Local import so importing deepISA does not hard-require tangermeme at
    # module load time (only when attribution is actually computed).
    from tangermeme.deep_lift_shap import deep_lift_shap

    if seqs_ohe.ndim != 3 or seqs_ohe.shape[1] != 4:
        raise ValueError(f"seqs_ohe must be (N, 4, L); got {seqs_ohe.shape}")
    if not tracks:
        raise ValueError("tracks must contain at least one output index")

    device = torch.device(device) if device is not None else torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    model = model.to(device).eval()

    N, _, L = seqs_ohe.shape
    seqs_np = np.ascontiguousarray(seqs_ohe, dtype=np.float32)
    # Genomic sequences routinely contain N; one_hot_encode maps them to
    # all-zero columns which tangermeme (and tf-modisco-lite) reject. Impute
    # them *before* anything downstream sees the data -- the saved H5 must
    # hold the imputed (strictly one-hot) sequences.
    seqs_np = _impute_unknown_bases(seqs_np, seed)
    not_onehot = seqs_np.sum(axis=1) != 1
    if not_onehot.any():
        raise ValueError(
            "seqs_ohe must be strictly one-hot (every position sums to 1); "
            f"{int(not_onehot.sum())} position(s) sum to something other than "
            "0 (unknown base) or 1. Encode raw ACGT strings with "
            "deepISA.utils.one_hot_encode."
        )
    tracks = list(tracks)
    T = len(tracks)

    # tangermeme wants a torch tensor on the model's device.
    X = torch.from_numpy(seqs_np).to(device)

    # Pre-allocate result buffers (track-leading).
    hyp_all = np.empty((T, N, 4, L), dtype=np.float32)
    act_all = np.empty((T, N, 4, L), dtype=np.float32)

    h5_sink = _H5Sink(save_h5_path, seqs_np, tracks, ids) if save_h5_path else None

    track_iter = tracks
    if show_progress:
        track_iter = tqdm(tracks, desc="attribution", unit="track")
    try:
        for ti, t in enumerate(track_iter):
            # hypothetical=True returns per-base multipliers (the -a input for
            # tf-modisco-lite / Fi-NeMo); hypothetical=False multiplies them by
            # the one-hot input to give actual contributions (logos).
            hyp_t = deep_lift_shap(
                model, X, target=t, n_shuffles=n_refs,
                hypothetical=True, batch_size=batch_size,
                device=device, random_state=seed,
            )
            act_t = deep_lift_shap(
                model, X, target=t, n_shuffles=n_refs,
                hypothetical=False, batch_size=batch_size,
                device=device, random_state=seed,
            )
            # tangermeme returns torch tensors shaped (N, 4, L).
            hyp_arr = hyp_t.detach().cpu().numpy().astype(np.float32)
            act_arr = act_t.detach().cpu().numpy().astype(np.float32)
            hyp_all[ti] = hyp_arr
            act_all[ti] = act_arr
            if h5_sink is not None:
                h5_sink.write_track(ti, hyp_arr, act_arr)
            del hyp_t, act_t
            if device.type == "cuda":
                torch.cuda.empty_cache()
    finally:
        if h5_sink is not None:
            h5_sink.close()

    logger.info(f"Attribution done (tangermeme): {T} track(s) x {N} seqs x {L} positions.")
    return hyp_all, act_all


def _impute_unknown_bases(seqs_np: np.ndarray, seed: Optional[int]) -> np.ndarray:
    """Replace all-zero one-hot columns (``N`` / unknown bases) with random ACGT.

    Real reference genomes contain ``N`` bases and :func:`deepISA.utils.one_hot_encode`
    encodes them as all-zero columns. tangermeme validates that every position sums
    to exactly 1 and otherwise raises ``ValueError: X must be one-hot encoded ...
    cannot have unknown characters``; tf-modisco-lite has the same requirement.
    Imputing a uniformly random base is the standard remedy: imputed positions
    carry no signal and receive ~0 attribution, so motif discovery is unaffected.

    Returns a new array when anything was imputed (the caller's input is never
    modified); returns the input unchanged when there is nothing to do.
    """
    unknown = seqs_np.sum(axis=1) == 0                       # (N, L)
    n_bad = int(unknown.sum())
    if n_bad == 0:
        return seqs_np
    rows, cols = np.nonzero(unknown)
    rng = np.random.default_rng(seed)
    imputed = np.eye(4, dtype=seqs_np.dtype)[rng.integers(0, 4, size=n_bad)]
    out = seqs_np.copy()
    out[rows, :, cols] = imputed
    logger.warning(
        f"Imputed {n_bad} unknown base(s) (N) in {np.unique(rows).size}/{seqs_np.shape[0]} "
        f"sequences with random ACGT (seed={seed}); their attribution is ~0 by construction."
    )
    return out


# ---------------------------------------------------------------------------
# Streaming HDF5 sink -- keeps the on-disk schema compatible with mc000
# ---------------------------------------------------------------------------
class _H5Sink:
    """Writer for the attribution H5 (channels-first, track-leading layout)."""

    def __init__(
        self,
        path: str,
        seqs_np: np.ndarray,
        tracks: Sequence[int],
        ids: Optional[Sequence[str]],
    ) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
        N, _, L = seqs_np.shape
        self.f = h5py.File(path, "w")
        self.f.attrs["layout"] = _H5_LAYOUT
        self.f.attrs["tracks"] = np.asarray(list(tracks), dtype=np.int32)
        self.f.create_dataset("sequences", data=seqs_np, compression="gzip")
        self.f.create_dataset(
            "hyp_scores", shape=(len(tracks), N, 4, L), dtype="float32", compression="gzip"
        )
        self.f.create_dataset(
            "act_scores", shape=(len(tracks), N, 4, L), dtype="float32", compression="gzip"
        )
        if ids is not None:
            self.f.create_dataset("id", data=np.asarray(ids, dtype=object).astype("S"))

    def write_track(self, track_idx: int, hyp: np.ndarray, act: np.ndarray) -> None:
        self.f["hyp_scores"][track_idx] = hyp
        self.f["act_scores"][track_idx] = act

    def close(self) -> None:
        self.f.close()
