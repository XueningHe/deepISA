"""Motif discovery subpackage: per-position attribution + tf-modisco-lite + FiNeMo.

This subpackage adds *de novo* motif discovery to deepISA without touching the
existing in-silico ablation (ISA) pipeline. It is intentionally framework-pure:
all attribution is computed with ``tangermeme`` (PyTorch-native DeepLIFT-SHAP)
on the torch ``Conv`` model, and the heavy motif discovery is delegated to the
external ``modisco`` / ``finemo`` CLI binaries via thin subprocess orchestrators.

Why tangermeme for attribution
------------------------------
Earlier iterations hand-rolled the DeepLIFT multiplier, the hypothetical
projection, and the dinucleotide-shuffle background. Three review rounds found
bugs in each (a missing projection, a wrong numpy axis, an incorrect act
definition). ``tangermeme.deep_lift_shap`` is the canonical PyTorch counterpart
to the TF-based ``deeplift``/``shap.DeepExplainer`` stack that mc000 used; it is
authored by the same lab as tf-modisco-lite and correctly handles per-sequence
dinucleotide-shuffled references (which captum's ``DeepLiftShap`` alone cannot).
Delegating to it removes an entire class of subtle bugs.

Public API
----------
- :func:`compute_attribution` -- per-position DeepLIFT-SHAP attributions.
- :func:`select_top_regions`, :func:`drop_non_acgt_regions` -- input-set
  curation (rank by activity, drop N-containing regions).
- :func:`prepare_modisco_input`, :func:`run_modisco` -- tf-modisco-lite orchestration.
- :func:`read_attribution_h5` -- read saved attributions back (single-track selection).
- :func:`build_finemo_input`, :func:`run_finemo_scan`, :func:`load_hits_with_annotation`,
  :func:`build_finemo_db` -- Fi-NeMo orchestration.
- :func:`load_motifs`, :func:`extract_motifs_from_group`, :func:`parse_motif_name` --
  read discovered motifs back.
- :func:`run_motif_report`, :func:`cwm_to_meme` -- HTML report + MEME export.

Convention
----------
All arrays follow deepISA's native layout ``(N, 4, L)`` (channels-first), matching
:func:`deepISA.utils.one_hot_encode`. Transposition to the ``(N, L, 4)`` /
``(N, 4, L)`` layout expected by the external CLIs happens at the NPZ boundary.
"""

from deepISA.scoring.discover.attribution import compute_attribution
from deepISA.scoring.discover.modisco import (
    prepare_modisco_input,
    run_modisco,
    read_attribution_h5,
)
from deepISA.scoring.discover.finemo import (
    build_finemo_input,
    run_finemo_scan,
    load_hits_with_annotation,
    build_finemo_db,
)
from deepISA.scoring.discover.h5_io import (
    extract_motifs_from_group,
    load_motifs,
    parse_motif_name,
)
from deepISA.scoring.discover.report import run_motif_report, cwm_to_meme
from deepISA.scoring.discover.select import select_top_regions, drop_non_acgt_regions

__all__ = [
    "compute_attribution",
    "prepare_modisco_input",
    "run_modisco",
    "read_attribution_h5",
    "build_finemo_input",
    "run_finemo_scan",
    "load_hits_with_annotation",
    "build_finemo_db",
    "extract_motifs_from_group",
    "load_motifs",
    "parse_motif_name",
    "run_motif_report",
    "cwm_to_meme",
    "select_top_regions",
    "drop_non_acgt_regions",
]
