"""AlphaGenome adapter — drop-in nn.Module backend for deepISA.

Config formats
--------------
Single track (backward-compatible):
    api_key: YOUR_KEY
    output_type: DNASE
    biosample_name: GM12878
    context_len: 16384   # optional, default 16384
    seq_len: 600          # optional, default 600
    aggregation: sum      # optional, default sum

Multi-track (new):
    api_key: YOUR_KEY
    tracks:
      - output_type: DNASE
        biosample_name: GM12878
      - output_type: CAGE
        biosample_name: GM12878
      - output_type: ATAC
        biosample_name: K562
    context_len: 16384
    seq_len: 600
    aggregation: sum

Every sequence makes exactly ONE API call regardless of how many tracks are
configured.  Columns in the output tensor are ordered by the `tracks` list.
"""
from __future__ import annotations
from typing import Any
import numpy as np
import torch
import yaml

_DEFAULTS = {"context_len": 16384, "seq_len": 600, "aggregation": "sum"}

_BASES = np.array(['A', 'C', 'G', 'T'], dtype='U1')


def load_config(path: str) -> dict[str, Any]:
    with open(path) as f:
        cfg = yaml.safe_load(f)
    if "api_key" not in cfg:
        raise KeyError("alpha_genome config missing required key: 'api_key'")
    # Normalise old single-track format → new tracks list
    if "tracks" not in cfg:
        for key in ("output_type", "biosample_name"):
            if key not in cfg:
                raise KeyError(f"alpha_genome config missing required key: '{key}'")
        cfg["tracks"] = [{"output_type": cfg["output_type"],
                           "biosample_name": cfg["biosample_name"]}]
    return {**_DEFAULTS, **cfg}


def _tensor_to_seqs(x: torch.Tensor) -> list[str]:
    """(N, 4, L) one-hot tensor → list[str]. Vectorized via argmax."""
    x_np     = x.cpu().numpy()
    idx      = x_np.argmax(axis=1)
    has_base = x_np.max(axis=1) > 0
    chars    = np.where(has_base, _BASES[idx], 'N')
    return [''.join(row) for row in chars]


def _pad_seqs(seqs: list[str], context_len: int, seq_len: int) -> list[str]:
    """Centre each seq in context_len of N padding."""
    pad_left  = (context_len - seq_len) // 2
    pad_right = context_len - seq_len - pad_left
    pre, suf  = 'N' * pad_left, 'N' * pad_right
    return [pre + s + suf for s in seqs]


import torch.nn as nn
from alphagenome.models import dna_client
from alphagenome.models.dna_output import OutputType


class AlphaGenomeAdapter(nn.Module):
    """
    Drop-in nn.Module replacement for deepISA's Conv model.

    Supports one or more (output_type, biosample_name) track combinations via
    the config file.  Every sequence prediction is a single API call; columns
    are concatenated in the order the tracks appear in the config.

    Returns (N, n_tracks) float32 tensor compatible with run_single_isa /
    run_combi_isa.  Use adapter.n_tracks to know the output width.
    """

    def __init__(self, config_path: str) -> None:
        super().__init__()
        cfg = load_config(config_path)
        self._cfg = cfg

        self._dna_model = dna_client.create(cfg["api_key"])

        meta = self._dna_model.output_metadata(
            dna_client.Organism.HOMO_SAPIENS
        ).concatenate()

        ctx = cfg["context_len"]
        sl  = cfg["seq_len"]
        self._context_len = ctx
        self._seq_len     = sl
        self._start_idx   = (ctx - sl) // 2
        self._end_idx     = self._start_idx + sl

        # ── Resolve each (output_type, biosample) track ───────────────────────
        tracks_cfg = cfg["tracks"]
        all_terms: list[str] = []
        all_output_type_enums: list[OutputType] = []

        for track in tracks_cfg:
            ot_str = track["output_type"]
            bio    = track["biosample_name"]
            ot_enum = OutputType[ot_str]
            matched = meta[
                (meta["output_type"] == ot_enum) &
                (meta["biosample_name"] == bio)
            ]
            if matched.empty:
                available = sorted(
                    meta[meta["output_type"] == ot_enum]
                    ["biosample_name"].dropna().unique()
                )[:15]
                raise ValueError(
                    f"biosample_name='{bio}' not found for output_type='{ot_str}'.\n"
                    f"Available (first 15): {available}\n"
                    f"Browse notebooks/ag_biosample_reference.csv to find valid names."
                )
            terms = matched["ontology_curie"].dropna().unique().tolist()
            all_terms.extend(terms)
            if ot_enum not in all_output_type_enums:
                all_output_type_enums.append(ot_enum)

        self._all_output_type_enums: list[OutputType] = all_output_type_enums
        self._all_terms: list[str] = list(dict.fromkeys(all_terms))  # dedup, keep order

        # Keep _ontology_terms as alias for backward compatibility
        self._ontology_terms = self._all_terms

        # ── Probe call: learn exact column indices for each desired track ─────
        # One call with all output types + all terms reveals which columns
        # belong to which biosample via TrackData.metadata.biosample_name.
        probe_out = self._dna_model.predict_sequence(
            sequence="N" * ctx,
            requested_outputs=self._all_output_type_enums,
            ontology_terms=self._all_terms,
        )

        # _extraction_plan: ordered list of (attr_name, col_indices_array)
        # Each entry corresponds to one desired (output_type, biosample) pair.
        self._extraction_plan: list[tuple[str, np.ndarray]] = []
        for track in tracks_cfg:
            ot_str = track["output_type"]
            bio    = track["biosample_name"]
            attr   = ot_str.lower()          # "DNASE" → "dnase", "RNA_SEQ" → "rna_seq"
            track_data = getattr(probe_out, attr)
            tmeta = track_data.metadata.reset_index(drop=True)
            col_idx = np.where(tmeta["biosample_name"] == bio)[0]
            if len(col_idx) == 0:
                raise ValueError(
                    f"Probe returned no columns for biosample='{bio}' in {ot_str}. "
                    f"Available in probe: {tmeta['biosample_name'].tolist()}"
                )
            self._extraction_plan.append((attr, col_idx))

        self._n_tracks: int = sum(len(idx) for _, idx in self._extraction_plan)

        # Sequence-level cache: raw 600bp string → list[float] (n_tracks,)
        self._cache: dict[str, list[float]] = {}

    # ── Public properties ─────────────────────────────────────────────────────

    @property
    def n_tracks(self) -> int:
        """Total number of output tracks across all configured (output_type, biosample) pairs."""
        return self._n_tracks

    @property
    def cache_size(self) -> int:
        return len(self._cache)

    def clear_cache(self) -> None:
        self._cache.clear()

    # ── nn.Module interface ───────────────────────────────────────────────────

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x      : (N, 4, seq_len) one-hot tensor from compute_predictions
        returns: (N, n_tracks) float32 tensor
        """
        seqs        = _tensor_to_seqs(x)
        seqs_padded = _pad_seqs(seqs, self._context_len, self._seq_len)
        scalars     = self._predict_sequential(seqs, seqs_padded)
        return torch.tensor(scalars, dtype=torch.float32)

    def _predict_sequential(
        self, seqs: list[str], seqs_padded: list[str]
    ) -> list[list[float]]:
        """One API call per unique sequence; cache hits skip the API entirely."""
        result = []
        for raw, padded in zip(seqs, seqs_padded):
            if raw not in self._cache:
                output = self._dna_model.predict_sequence(
                    sequence=padded,
                    requested_outputs=self._all_output_type_enums,
                    ontology_terms=self._all_terms,
                )
                parts = []
                for attr, col_idx in self._extraction_plan:
                    track_data = getattr(output, attr)
                    window = track_data.values[self._start_idx:self._end_idx, :]
                    parts.append(window[:, col_idx].sum(axis=0))  # (n_cols_for_this_bio,)
                self._cache[raw] = np.concatenate(parts).tolist()
            result.append(self._cache[raw])
        return result
