"""Tests for the optional AlphaGenome adapter (modeling/alpha_genome_adapter.py).

Ported from the ``alpha-genome`` branch of JoneSu1/deepISA_filter with the
module paths fixed (``deepisa_ag.adapter`` -> ``deepISA.modeling.alpha_genome_adapter``)
and the ``alphagenome`` import made optional: every test below runs against a
mocked ``dna_client``, so the suite is green whether or not the optional extra
is installed. No test touches the network.
"""

from __future__ import annotations

from contextlib import contextmanager
from enum import Enum
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
import torch
import yaml

from deepISA.modeling import alpha_genome_adapter as aga
from deepISA.modeling.alpha_genome_adapter import (
    AlphaGenomeAdapter,
    _pad_seqs,
    _tensor_to_seqs,
    load_config,
)
from deepISA.utils import one_hot_encode

try:
    from alphagenome.models.dna_output import OutputType
except ImportError:
    class OutputType(Enum):
        """Stub enum so the mocked tests run without the optional extra."""

        DNASE = "DNASE"
        CAGE = "CAGE"
        ATAC = "ATAC"
        RNA_SEQ = "RNA_SEQ"


@contextmanager
def _mocked_alphagenome():
    """Patch the adapter module's optional imports (they are None without the extra)."""
    with patch.object(aga, "dna_client") as mock_dc, patch.object(
        aga, "OutputType", OutputType
    ):
        yield mock_dc


# ── load_config ───────────────────────────────────────────────────────────────


def test_load_config_reads_fields(tmp_path):
    cfg = {
        "api_key": "testkey",
        "output_type": "DNASE",
        "biosample_name": "GM12878",
        "context_len": 16384,
        "seq_len": 600,
        "aggregation": "sum",
    }
    p = tmp_path / "config.yaml"
    p.write_text(yaml.dump(cfg))

    loaded = load_config(str(p))
    assert loaded["api_key"] == "testkey"
    assert loaded["context_len"] == 16384
    assert loaded["aggregation"] == "sum"


def test_load_config_missing_required_key(tmp_path):
    p = tmp_path / "bad.yaml"
    p.write_text(yaml.dump({"output_type": "DNASE"}))

    with pytest.raises(KeyError):
        load_config(str(p))


def test_load_config_rejects_unknown_aggregation(tmp_path):
    """Regression: the documented aggregation knob is validated, not silently ignored."""
    cfg = {
        "api_key": "k",
        "output_type": "DNASE",
        "biosample_name": "GM12878",
        "aggregation": "median",
    }
    p = tmp_path / "cfg.yaml"
    p.write_text(yaml.dump(cfg))

    with pytest.raises(ValueError, match="aggregation"):
        load_config(str(p))


# ── vectorized sequence utilities ────────────────────────────────────────────


def test_tensor_to_seqs_roundtrip():
    """one_hot_encode → tensor → _tensor_to_seqs should recover original strings."""
    seqs = ["ACGT" * 150]                  # 600 bp
    x = torch.from_numpy(one_hot_encode(seqs))  # (1, 4, 600)
    assert _tensor_to_seqs(x) == seqs


def test_tensor_to_seqs_n_positions():
    x = torch.zeros(1, 4, 4)              # all-zero → 'N'
    assert _tensor_to_seqs(x)[0] == "NNNN"


def test_pad_seqs_total_length():
    padded = _pad_seqs(["ACGT" * 150], context_len=16384, seq_len=600)
    assert len(padded[0]) == 16384


def test_pad_seqs_centre_preserved():
    seq = "ACGT" * 150
    padded = _pad_seqs([seq], context_len=16384, seq_len=600)[0]
    pad_left = (16384 - 600) // 2
    assert padded[pad_left: pad_left + 600] == seq


def test_pad_seqs_flanks_are_n():
    padded = _pad_seqs(["A" * 600], context_len=16384, seq_len=600)[0]
    pad_left = (16384 - 600) // 2
    assert set(padded[:pad_left]) == {"N"}
    assert set(padded[pad_left + 600:]) == {"N"}


# ── AlphaGenomeAdapter (all API calls mocked) ────────────────────────────────


def _fake_metadata(biosample: str, output_type: str) -> pd.DataFrame:
    """Return metadata with real OutputType enum objects, matching the live API."""
    return pd.DataFrame({
        "biosample_name": [biosample],
        "output_type":    [OutputType[output_type]],
        "ontology_curie": ["CL:0000000"],
    })


def _fake_track_output(n_positions: int, n_tracks: int, value: float,
                       biosample: str = "GM12878"):
    td = MagicMock()
    td.values = np.full((n_positions, n_tracks), value, dtype=np.float32)
    # metadata must be a real DataFrame so probe-call col-index logic works
    td.metadata = pd.DataFrame({"biosample_name": [biosample] * n_tracks})
    return td


def _fake_predict_output(value: float, output_attr: str = "dnase",
                         biosample: str = "GM12878"):
    out = MagicMock()
    setattr(out, output_attr, _fake_track_output(16384, 1, value, biosample))
    return out


def _make_adapter(tmp_path, biosample="GM12878", output_type="DNASE", mock_dc=None,
                  aggregation="sum"):
    cfg = {"api_key": "k", "output_type": output_type, "biosample_name": biosample,
           "context_len": 16384, "seq_len": 600, "aggregation": aggregation}
    (tmp_path / "cfg.yaml").write_text(yaml.dump(cfg))
    mock_dc.create.return_value.output_metadata.return_value.concatenate.return_value = (
        _fake_metadata(biosample, output_type))
    return AlphaGenomeAdapter(str(tmp_path / "cfg.yaml"))


def test_adapter_requires_extra_when_not_installed(tmp_path):
    """Without the alphagenome extra, construction raises a helpful ImportError."""
    cfg = {"api_key": "k", "output_type": "DNASE", "biosample_name": "GM12878"}
    (tmp_path / "cfg.yaml").write_text(yaml.dump(cfg))
    with patch.object(aga, "dna_client", None):
        with pytest.raises(ImportError, match=r"alphagenome is not installed"):
            AlphaGenomeAdapter(str(tmp_path / "cfg.yaml"))


def test_adapter_forward_returns_n_by_n_tracks(tmp_path):
    with _mocked_alphagenome() as mock_dc:
        mock_dc.create.return_value.predict_sequence.return_value = (
            _fake_predict_output(1.0))
        adapter = _make_adapter(tmp_path, mock_dc=mock_dc)

        x = torch.from_numpy(one_hot_encode(["ACGT" * 150]))  # (1, 4, 600)
        out = adapter(x)

        assert out.shape == (1, 1)   # 1 seq × 1 track (mock has 1 track)
        assert out.dtype == torch.float32


def test_adapter_col0_equals_signal_sum(tmp_path):
    """col 0 = sum of central 600 bp × 1 track × signal_value."""
    signal_value = 0.5
    with _mocked_alphagenome() as mock_dc:
        mock_dc.create.return_value.predict_sequence.return_value = (
            _fake_predict_output(signal_value))
        adapter = _make_adapter(tmp_path, mock_dc=mock_dc)

        x = torch.from_numpy(one_hot_encode(["ACGT" * 150]))
        out = adapter(x)

        expected = signal_value * 600 * 1   # sum over 600 positions × 1 track
        assert float(out[0, 0]) == pytest.approx(expected)


def test_adapter_aggregation_mean(tmp_path):
    """Regression: aggregation='mean' averages the window instead of summing it."""
    with _mocked_alphagenome() as mock_dc:
        mock_dc.create.return_value.predict_sequence.return_value = (
            _fake_predict_output(2.0))
        adapter = _make_adapter(tmp_path, mock_dc=mock_dc, aggregation="mean")

        x = torch.from_numpy(one_hot_encode(["ACGT" * 150]))
        out = adapter(x)

        assert float(out[0, 0]) == pytest.approx(2.0)   # mean, not 2.0 * 600


def test_adapter_cache_deduplicates_api_calls(tmp_path):
    """Identical sequences must produce only one API call, not two."""
    with _mocked_alphagenome() as mock_dc:
        mock_dc.create.return_value.predict_sequence.return_value = (
            _fake_predict_output(1.0))
        adapter = _make_adapter(tmp_path, mock_dc=mock_dc)

        calls_after_init = mock_dc.create.return_value.predict_sequence.call_count

        x = torch.from_numpy(one_hot_encode(["ACGT" * 150]))

        adapter(x)   # first call  → API hit, stored in cache
        adapter(x)   # second call → cache hit, no API call

        assert mock_dc.create.return_value.predict_sequence.call_count == calls_after_init + 1
        assert adapter.cache_size == 1


def test_adapter_clear_cache(tmp_path):
    """clear_cache() resets the cache so the next call hits the API again."""
    with _mocked_alphagenome() as mock_dc:
        mock_dc.create.return_value.predict_sequence.return_value = (
            _fake_predict_output(1.0))
        adapter = _make_adapter(tmp_path, mock_dc=mock_dc)

        calls_after_init = mock_dc.create.return_value.predict_sequence.call_count

        x = torch.from_numpy(one_hot_encode(["ACGT" * 150]))

        adapter(x)
        assert adapter.cache_size == 1
        adapter.clear_cache()
        assert adapter.cache_size == 0
        adapter(x)   # cache was cleared → one more API call
        assert mock_dc.create.return_value.predict_sequence.call_count == calls_after_init + 2


def test_adapter_bad_biosample_raises(tmp_path):
    with _mocked_alphagenome() as mock_dc:
        mock_dc.create.return_value.output_metadata.return_value.concatenate.return_value = (
            _fake_metadata("GM12878", "DNASE"))
        cfg = {"api_key": "k", "output_type": "DNASE", "biosample_name": "NonExistent",
               "context_len": 16384, "seq_len": 600, "aggregation": "sum"}
        (tmp_path / "cfg.yaml").write_text(yaml.dump(cfg))

        with pytest.raises(ValueError, match="not found"):
            AlphaGenomeAdapter(str(tmp_path / "cfg.yaml"))


# ── full-chain integration (compute_predictions, all mocked) ─────────────────


def test_full_chain_compute_predictions(tmp_path):
    """adapter works as model arg in deepISA's compute_predictions — zero ISA code changes."""
    from deepISA.modeling.predict import compute_predictions

    with _mocked_alphagenome() as mock_dc:
        mock_dc.create.return_value.output_metadata.return_value.concatenate.return_value = (
            _fake_metadata("GM12878", "DNASE"))
        mock_dc.create.return_value.predict_sequence.side_effect = [
            _fake_predict_output(1.0),   # probe in __init__
            _fake_predict_output(2.0),   # seq 1 original
            _fake_predict_output(1.0),   # seq 1 ablated
        ]
        adapter = _make_adapter(tmp_path, mock_dc=mock_dc)

        device = torch.device("cpu")
        seqs_orig  = ["ACGT" * 150]
        seqs_ablat = ["NNNN" * 150]

        preds_orig  = compute_predictions(adapter, seqs_orig,  device, batch_size=1)
        preds_ablat = compute_predictions(adapter, seqs_ablat, device, batch_size=1)

        isa = preds_orig[:, 0] - preds_ablat[:, 0]
        assert preds_orig.shape  == (1, 1)
        assert preds_ablat.shape == (1, 1)
        assert float(isa[0]) == pytest.approx(2.0 * 600 - 1.0 * 600)  # 600.0


def test_run_single_isa_with_adapter_end_to_end(tmp_path):
    """Regression for the tutorial's core claim: swapping Conv -> adapter
    leaves the ISA pipeline unchanged. Runs the *real* run_single_isa
    (pred-orig → single ISA → null ISA → null-threshold filtering) against a
    fully mocked AlphaGenome API — no network, works without the extra.

    The fake API scores each base (A=4, C=3, G=2, T=-1, N=0), so every value
    is exactly computable: the genome sums to 1600, ablating the 10 bp all-A
    motif at [0, 10) costs 40.0, and the non-motif interval [280, 330) spans
    the G/T boundary so null kmers come out with BOTH signs (upstream's
    derive_null_thresholds percentiles the positive and negative sides
    separately and crashes on an empty side).
    """
    from deepISA.scoring.single_isa import run_single_isa
    from deepISA.utils import load_fasta

    genome = "A" * 100 + "C" * 100 + "G" * 100 + "T" * 100 + "A" * 200  # 600 bp
    with open(tmp_path / "genome.fa", "w") as f:
        f.write(">chr1\n")
        for i in range(0, 600, 60):
            f.write(genome[i:i + 60] + "\n")
    fasta = load_fasta(str(tmp_path / "genome.fa"))  # pre-loaded: skips the pysam path

    motif_locs = pd.DataFrame({
        "chrom": ["chr1"], "start": [0], "end": [10],
        "region": ["chr1:0-600"], "score": [7.5],
        "start_rel": [0], "end_rel": [10], "tf": ["MA0001.1"],
    })
    motif_locs.to_csv(tmp_path / "motif_locs.csv", index=False)
    non_motif = pd.DataFrame({
        "chrom": ["chr1"], "start": [280], "end": [330],
        "region": ["chr1:0-600"], "start_rel": [280], "end_rel": [330],
    })
    non_motif.to_csv(tmp_path / "non_motif_locs.csv", index=False)

    with _mocked_alphagenome() as mock_dc:
        mock_dc.create.return_value.output_metadata.return_value.concatenate.return_value = (
            _fake_metadata("GM12878", "DNASE"))

        api_calls = []
        base_w = {"A": 4.0, "C": 3.0, "G": 2.0, "T": -1.0}

        def fake_predict(sequence=None, **kwargs):
            api_calls.append(sequence)
            td = MagicMock()
            td.values = np.array(
                [base_w.get(ch, 0.0) for ch in sequence], dtype=np.float32
            )[:, None]  # (context_len, 1)
            td.metadata = pd.DataFrame({"biosample_name": ["GM12878"]})
            out = MagicMock()
            out.dnase = td
            return out

        mock_dc.create.return_value.predict_sequence.side_effect = fake_predict
        adapter = _make_adapter(tmp_path, mock_dc=mock_dc)  # probe call consumed

        out_single = str(tmp_path / "motif_single_isa_ag.csv")
        run_single_isa(
            model=adapter,
            fasta=fasta,
            motif_locs_path=str(tmp_path / "motif_locs.csv"),
            non_motif_locs_path=str(tmp_path / "non_motif_locs.csv"),
            single_isa_outpath=out_single,
            null_isa_outpath=str(tmp_path / "null_isa_ag.csv"),
            pred_orig_outpath=str(tmp_path / "pred_orig_ag.csv"),
            null_percentile=80,
            device=torch.device("cpu"),
            num_regions_per_batch=4,
            pred_batch_size=1,
            null_n_samples=6,
        )

        # pred-orig: whole-region sum = 100*(4+3+2-1) + 200*4 = 1600
        df_pred = pd.read_csv(tmp_path / "pred_orig_ag.csv")
        assert list(df_pred.columns) == ["region", "pred_t0"]
        assert df_pred.loc[0, "pred_t0"] == pytest.approx(1600.0)

        # motif ISA = 1600 − (1600 − 10*4) = 40.0; row survives the null filter
        df_isa = pd.read_csv(out_single)
        assert "isa_t0" in df_isa.columns
        assert len(df_isa) == 1
        assert df_isa.loc[0, "isa_t0"] == pytest.approx(40.0)

        # probe + pred-orig + motif-ablation + ≤ null_n_samples unique null seqs
        assert 4 <= len(api_calls) <= 3 + 6


# ── multi-track config ────────────────────────────────────────────────────────


def _fake_metadata_multi(pairs: list) -> pd.DataFrame:
    """pairs = [(biosample, output_type_str), ...]"""
    return pd.DataFrame({
        "biosample_name": [b for b, _ in pairs],
        "output_type":    [OutputType[ot] for _, ot in pairs],
        "ontology_curie": [f"CL:{i:07d}" for i in range(len(pairs))],
    })


def test_multi_track_config_new_format(tmp_path):
    """tracks: list config → correct n_tracks and output shape."""
    biosample_a, biosample_b = "GM12878", "K562"
    with _mocked_alphagenome() as mock_dc:
        mock_dc.create.return_value.output_metadata.return_value.concatenate.return_value = (
            _fake_metadata_multi([
                (biosample_a, "DNASE"),
                (biosample_b, "ATAC"),
            ])
        )
        # probe + forward calls: each returns dnase(1 col for A) + atac(1 col for B)
        def make_output():
            out = MagicMock()
            out.dnase = _fake_track_output(16384, 1, 1.0, biosample_a)
            out.atac  = _fake_track_output(16384, 1, 2.0, biosample_b)
            return out
        mock_dc.create.return_value.predict_sequence.return_value = make_output()

        cfg = {"api_key": "k",
               "tracks": [{"output_type": "DNASE", "biosample_name": biosample_a},
                           {"output_type": "ATAC",  "biosample_name": biosample_b}],
               "context_len": 16384, "seq_len": 600}
        (tmp_path / "cfg.yaml").write_text(yaml.dump(cfg))
        adapter = AlphaGenomeAdapter(str(tmp_path / "cfg.yaml"))

        assert adapter.n_tracks == 2

        x = torch.from_numpy(one_hot_encode(["ACGT" * 150]))
        out = adapter(x)
        assert out.shape == (1, 2)
        # col 0 = DNASE signal (1.0 × 600), col 1 = ATAC signal (2.0 × 600)
        assert float(out[0, 0]) == pytest.approx(600.0)
        assert float(out[0, 1]) == pytest.approx(1200.0)


def test_single_track_old_format_still_works(tmp_path):
    """Old output_type / biosample_name keys still accepted (backward compat)."""
    with _mocked_alphagenome() as mock_dc:
        mock_dc.create.return_value.predict_sequence.return_value = (
            _fake_predict_output(1.0))
        adapter = _make_adapter(tmp_path, mock_dc=mock_dc)
        assert adapter.n_tracks == 1
