import sys
import pytest
import yaml
import numpy as np
import torch

sys.path.insert(0, "deepISA/src")   # make deepISA importable without modifying it


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

    from deepisa_ag.adapter import load_config
    loaded = load_config(str(p))
    assert loaded["api_key"] == "testkey"
    assert loaded["context_len"] == 16384
    assert loaded["aggregation"] == "sum"


def test_load_config_missing_required_key(tmp_path):
    p = tmp_path / "bad.yaml"
    p.write_text(yaml.dump({"output_type": "DNASE"}))

    from deepisa_ag.adapter import load_config
    with pytest.raises(KeyError):
        load_config(str(p))


# ── Task 2: vectorized sequence utilities ─────────────────────────────────────

def test_tensor_to_seqs_roundtrip():
    """one_hot_encode → tensor → _tensor_to_seqs should recover original strings."""
    from deepISA.utils import one_hot_encode
    from deepisa_ag.adapter import _tensor_to_seqs
    seqs = ["ACGT" * 150]                  # 600 bp
    x = torch.from_numpy(one_hot_encode(seqs))  # (1, 4, 600)
    assert _tensor_to_seqs(x) == seqs


def test_tensor_to_seqs_n_positions():
    from deepisa_ag.adapter import _tensor_to_seqs
    x = torch.zeros(1, 4, 4)              # all-zero → 'N'
    assert _tensor_to_seqs(x)[0] == "NNNN"


def test_pad_seqs_total_length():
    from deepisa_ag.adapter import _pad_seqs
    padded = _pad_seqs(["ACGT" * 150], context_len=16384, seq_len=600)
    assert len(padded[0]) == 16384


def test_pad_seqs_centre_preserved():
    from deepisa_ag.adapter import _pad_seqs
    seq = "ACGT" * 150
    padded = _pad_seqs([seq], context_len=16384, seq_len=600)[0]
    pad_left = (16384 - 600) // 2
    assert padded[pad_left: pad_left + 600] == seq


def test_pad_seqs_flanks_are_n():
    from deepisa_ag.adapter import _pad_seqs
    padded = _pad_seqs(["A" * 600], context_len=16384, seq_len=600)[0]
    pad_left = (16384 - 600) // 2
    assert set(padded[:pad_left]) == {"N"}
    assert set(padded[pad_left + 600:]) == {"N"}


# ── Task 3: AlphaGenomeAdapter class ─────────────────────────────────────────

import pandas as pd
from unittest.mock import MagicMock, patch
from alphagenome.models.dna_output import OutputType


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


def _make_adapter(tmp_path, biosample="GM12878", output_type="DNASE", mock_dc=None):
    cfg = {"api_key": "k", "output_type": output_type, "biosample_name": biosample,
           "context_len": 16384, "seq_len": 600, "aggregation": "sum"}
    (tmp_path / "cfg.yaml").write_text(yaml.dump(cfg))
    mock_dc.create.return_value.output_metadata.return_value.concatenate.return_value = (
        _fake_metadata(biosample, output_type))
    from deepisa_ag.adapter import AlphaGenomeAdapter
    return AlphaGenomeAdapter(str(tmp_path / "cfg.yaml"))


def test_adapter_forward_returns_n_by_n_tracks(tmp_path):
    with patch("deepisa_ag.adapter.dna_client") as mock_dc:
        mock_dc.create.return_value.predict_sequence.return_value = (
            _fake_predict_output(1.0))
        adapter = _make_adapter(tmp_path, mock_dc=mock_dc)

        from deepISA.utils import one_hot_encode
        x = torch.from_numpy(one_hot_encode(["ACGT" * 150]))  # (1, 4, 600)
        out = adapter(x)

        assert out.shape == (1, 1)   # 1 seq × 1 track (mock has 1 track)
        assert out.dtype == torch.float32


def test_adapter_col0_equals_signal_sum(tmp_path):
    """col 0 = sum of central 600 bp × 1 track × signal_value."""
    signal_value = 0.5
    with patch("deepisa_ag.adapter.dna_client") as mock_dc:
        mock_dc.create.return_value.predict_sequence.return_value = (
            _fake_predict_output(signal_value))
        adapter = _make_adapter(tmp_path, mock_dc=mock_dc)

        from deepISA.utils import one_hot_encode
        x = torch.from_numpy(one_hot_encode(["ACGT" * 150]))
        out = adapter(x)

        expected = signal_value * 600 * 1   # sum over 600 positions × 1 track
        assert float(out[0, 0]) == pytest.approx(expected)


def test_adapter_cache_deduplicates_api_calls(tmp_path):
    """Identical sequences must produce only one API call, not two."""
    with patch("deepisa_ag.adapter.dna_client") as mock_dc:
        mock_dc.create.return_value.predict_sequence.return_value = (
            _fake_predict_output(1.0))
        adapter = _make_adapter(tmp_path, mock_dc=mock_dc)

        calls_after_init = mock_dc.create.return_value.predict_sequence.call_count

        from deepISA.utils import one_hot_encode
        x = torch.from_numpy(one_hot_encode(["ACGT" * 150]))

        adapter(x)   # first call  → API hit, stored in cache
        adapter(x)   # second call → cache hit, no API call

        assert mock_dc.create.return_value.predict_sequence.call_count == calls_after_init + 1
        assert adapter.cache_size == 1


def test_adapter_clear_cache(tmp_path):
    """clear_cache() resets the cache so the next call hits the API again."""
    with patch("deepisa_ag.adapter.dna_client") as mock_dc:
        mock_dc.create.return_value.predict_sequence.return_value = (
            _fake_predict_output(1.0))
        adapter = _make_adapter(tmp_path, mock_dc=mock_dc)

        calls_after_init = mock_dc.create.return_value.predict_sequence.call_count

        from deepISA.utils import one_hot_encode
        x = torch.from_numpy(one_hot_encode(["ACGT" * 150]))

        adapter(x)
        assert adapter.cache_size == 1
        adapter.clear_cache()
        assert adapter.cache_size == 0
        adapter(x)   # cache was cleared → one more API call
        assert mock_dc.create.return_value.predict_sequence.call_count == calls_after_init + 2


def test_adapter_bad_biosample_raises(tmp_path):
    with patch("deepisa_ag.adapter.dna_client") as mock_dc:
        mock_dc.create.return_value.output_metadata.return_value.concatenate.return_value = (
            _fake_metadata("GM12878", "DNASE"))
        cfg = {"api_key": "k", "output_type": "DNASE", "biosample_name": "NonExistent",
               "context_len": 16384, "seq_len": 600, "aggregation": "sum"}
        (tmp_path / "cfg.yaml").write_text(yaml.dump(cfg))

        from deepisa_ag import AlphaGenomeAdapter
        with pytest.raises(ValueError, match="not found"):
            AlphaGenomeAdapter(str(tmp_path / "cfg.yaml"))


# ── Task 4: Full-chain integration test ──────────────────────────────────────

def test_full_chain_compute_predictions(tmp_path):
    """adapter works as model arg in deepISA's compute_predictions — zero ISA code changes."""
    from deepISA.modeling.predict import compute_predictions

    with patch("deepisa_ag.adapter.dna_client") as mock_dc:
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


# ── Task 5: multi-track config ────────────────────────────────────────────────

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
    with patch("deepisa_ag.adapter.dna_client") as mock_dc:
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
        from deepisa_ag.adapter import AlphaGenomeAdapter
        adapter = AlphaGenomeAdapter(str(tmp_path / "cfg.yaml"))

        assert adapter.n_tracks == 2

        from deepISA.utils import one_hot_encode
        x = torch.from_numpy(one_hot_encode(["ACGT" * 150]))
        out = adapter(x)
        assert out.shape == (1, 2)
        # col 0 = DNASE signal (1.0 × 600), col 1 = ATAC signal (2.0 × 600)
        assert float(out[0, 0]) == pytest.approx(600.0)
        assert float(out[0, 1]) == pytest.approx(1200.0)


def test_single_track_old_format_still_works(tmp_path):
    """Old output_type / biosample_name keys still accepted (backward compat)."""
    with patch("deepisa_ag.adapter.dna_client") as mock_dc:
        mock_dc.create.return_value.predict_sequence.return_value = (
            _fake_predict_output(1.0))
        adapter = _make_adapter(tmp_path, mock_dc=mock_dc)
        assert adapter.n_tracks == 1
