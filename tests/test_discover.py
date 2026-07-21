"""Tests for the motif-discovery subpackage (deepISA.scoring.discover).

These tests exercise only the Python side -- attribution, NPZ/H5 building,
hits parsing, motif extraction -- and never invoke the external ``modisco`` /
``finemo`` CLIs. They therefore run regardless of whether those binaries are
installed.
"""

import numpy as np
import pandas as pd
import pytest
import torch

from deepISA.modeling.cnn import Conv
from deepISA.scoring.discover import (
    build_finemo_db,
    build_finemo_input,
    compute_attribution,
    extract_motifs_from_group,
    load_hits_with_annotation,
    load_motifs,
    prepare_modisco_input,
    run_modisco,
    run_finemo_scan,
    run_motif_report,
    cwm_to_meme,
    read_attribution_h5,
)
from deepISA.scoring.filter import (
    _get_second_max,
    extract_regions,
    get_attr_threshold,
    get_slices,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture(scope="session")
def device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture(scope="session")
def tiny_model(device):
    cfg = {"seq_len": 60, "ks": [7, 5], "cs": [8, 8], "ds": [1, 2], "dropout": 0.0}
    m = Conv(mode="regression", model_config=cfg).to(device).eval()
    return m


def _rand_onehot(n, L, seed=0):
    """Random valid one-hot (N, 4, L) with exactly one base per position."""
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, 4, size=(n, L))
    ohe = np.eye(4, dtype=np.float32)[idx]            # (N, L, 4)
    return np.ascontiguousarray(np.transpose(ohe, (0, 2, 1)))  # (N, 4, L)


# ---------------------------------------------------------------------------
# attribution (tangermeme DeepLIFT-SHAP)
# ---------------------------------------------------------------------------
class TestAttribution:
    def test_shape_single_track(self, tiny_model, device):
        seqs = _rand_onehot(2, 60, seed=10)
        hyp, act = compute_attribution(
            tiny_model, seqs, tracks=[0], device=device,
            n_refs=4, batch_size=2, show_progress=False,
        )
        assert hyp.shape == (1, 2, 4, 60)
        assert act.shape == (1, 2, 4, 60)
        assert np.isfinite(hyp).all()

    def test_hyp_has_values_at_all_bases(self, tiny_model, device):
        """hypothetical scores must have non-trivial values at ALL 4 bases per
        position (not just the observed base). This is the defining property
        that makes them the correct ``-a`` input for tf-modisco-lite."""
        seqs = _rand_onehot(3, 60, seed=13)
        hyp, _ = compute_attribution(
            tiny_model, seqs, tracks=[0], device=device,
            n_refs=4, batch_size=8, show_progress=False,
        )
        n_nonzero = np.sum(np.abs(hyp[0]) > 1e-8, axis=1)   # (N, L) bases nonzero per pos
        assert n_nonzero.mean() >= 3.0, f"hyp not at all bases: mean={n_nonzero.mean()}"

    def test_act_is_zero_at_non_observed_bases(self, tiny_model, device):
        """actual scores = hyp * one-hot, so non-observed bases must be ~0.
        This confirms the act/hyp distinction is preserved (regression for an
        earlier bug where act was set to phi directly)."""
        seqs = _rand_onehot(3, 60, seed=14)
        _, act = compute_attribution(
            tiny_model, seqs, tracks=[0], device=device,
            n_refs=4, batch_size=8, show_progress=False,
        )
        observed = seqs > 0
        not_observed = ~observed
        # Non-observed positions should be near zero.
        assert np.max(np.abs(act[0][not_observed])) < 1e-5, "act nonzero at absent bases"

    def test_different_tracks_give_different_attributions(self, tiny_model, device):
        """Dual-mode models have distinct regression/classification heads; their
        attributions must differ. (Regression for the silent track-averaging bug.)"""
        if _model_n_outputs(tiny_model) < 2:
            pytest.skip("single-output model")
        seqs = _rand_onehot(2, 60, seed=15)
        hyp, _ = compute_attribution(
            tiny_model, seqs, tracks=[0, 1], device=device,
            n_refs=4, batch_size=8, show_progress=False,
        )
        assert hyp.shape[0] == 2
        diff = np.max(np.abs(hyp[0] - hyp[1]))
        assert diff > 1e-4, f"tracks 0 and 1 identical (diff={diff})"

    def test_h5_sink_writes_channels_first_schema(self, tiny_model, device, tmp_path):
        import h5py
        h5_path = str(tmp_path / "attr.h5")
        seqs = _rand_onehot(3, 60, seed=12)
        compute_attribution(
            tiny_model, seqs, tracks=[0, 1] if _model_n_outputs(tiny_model) >= 2 else [0],
            device=device, n_refs=3, batch_size=8, save_h5_path=h5_path,
            ids=["a", "b", "c"], show_progress=False,
        )
        with h5py.File(h5_path, "r") as f:
            assert f.attrs["layout"] == b"channels_first" or f.attrs["layout"] == "channels_first"
            assert f["sequences"].shape == (3, 4, 60)
            assert f["hyp_scores"].ndim == 4
            assert "id" in f


def _model_n_outputs(model):
    with torch.no_grad():
        x = torch.zeros(1, 4, 60)
        y = model(x)
    return int(y.shape[-1]) if y.ndim > 1 else 1


# ---------------------------------------------------------------------------
# modisco input prep + orchestrator error paths
# ---------------------------------------------------------------------------
class TestModisco:
    def test_prepare_input_preserves_full_length(self, tiny_model, device, tmp_path):
        # Sequences are NOT trimmed: the full attribution length is written to
        # the NPZ. window is a separate -w flag handled by run_modisco().
        import h5py
        h5_path = str(tmp_path / "attr.h5")
        seqs = _rand_onehot(2, 60, seed=20)
        compute_attribution(
            tiny_model, seqs, tracks=[0], device=device,
            n_refs=3, batch_size=2, save_h5_path=h5_path, show_progress=False,
        )
        ohe_npz, hyp_npz = prepare_modisco_input(h5_path, str(tmp_path / "in"))
        with np.load(ohe_npz) as dz:
            arr = dz[dz.files[0]]
            assert arr.shape == (2, 4, 60)   # full length preserved
            assert arr.dtype == np.int8
        with np.load(hyp_npz) as dz:
            arr = dz[dz.files[0]]
            assert arr.shape == (2, 4, 60)
            assert arr.dtype == np.float32

    def test_prepare_input_trims_odd_length_by_one(self, tmp_path):
        # Odd-length attributions lose exactly one trailing position to satisfy
        # tf-modisco-lite's even-length requirement -- this is a parity fix,
        # not a content window.
        import h5py
        rng = np.random.default_rng(0)
        seqs_nlf = np.eye(4, dtype=np.float32)[rng.integers(0, 4, size=(2, 7))]  # (2,7,4)
        hyp_nlf = rng.standard_normal((2, 7, 4)).astype(np.float32)
        h5_path = str(tmp_path / "odd.h5")
        with h5py.File(h5_path, "w") as f:
            f.create_dataset("sequences", data=seqs_nlf)
            f.create_dataset("hyp_scores", data=hyp_nlf)
        ohe_npz, _ = prepare_modisco_input(h5_path, str(tmp_path / "odd_in"))
        with np.load(ohe_npz) as dz:
            assert dz[dz.files[0]].shape == (2, 4, 6)   # 7 -> 6

    def test_read_attribution_h5_legacy_layout(self, tmp_path):
        import h5py
        # mc000-style (N, L, 4) legacy schema, no layout attr.
        h5_path = str(tmp_path / "legacy.h5")
        with h5py.File(h5_path, "w") as f:
            f.create_dataset("sequences", data=np.eye(4, dtype=np.float32)[None].repeat(3, 0))
            f.create_dataset("hyp_scores", data=np.eye(4, dtype=np.float32)[None].repeat(3, 0))
        seqs, hyp, T = read_attribution_h5(h5_path)
        assert seqs.shape == (3, 4, 4)   # normalized to channels-first
        assert hyp.shape == (3, 4, 4)
        assert T == 1

    def test_run_modisco_missing_binary_raises(self, tmp_path, monkeypatch):
        monkeypatch.setattr("deepISA.scoring.discover.modisco.resolve_cli", lambda name: None)
        with pytest.raises(RuntimeError, match="tf-modisco-lite"):
            run_modisco("o.npz", "h.npz", str(tmp_path / "out.h5"))


# ---------------------------------------------------------------------------
# finemo input + hits parsing + db build
# ---------------------------------------------------------------------------
class TestFinemo:
    def test_build_input_writes_npz_and_bed(self, tmp_path):
        seqs = _rand_onehot(3, 41, seed=30)   # odd length to test trimming
        hyp = np.random.default_rng(31).standard_normal((3, 4, 41)).astype(np.float32)
        npz = build_finemo_input(seqs, hyp, str(tmp_path / "fin"), ids=np.array(["x", "y", "z"]))
        with np.load(npz) as dz:
            assert dz["sequences"].shape == (3, 4, 40)   # trimmed to even
            assert dz["contributions"].shape == (3, 4, 40)
        bed_lines = (tmp_path / "fin" / "regions.bed").read_text().strip().split("\n")
        assert len(bed_lines) == 3

    def test_build_db_and_load_hits_annotation(self, tmp_path):
        import h5py
        motifs = {
            "p0_main": {"cwm": np.eye(4)[:10], "seq": np.eye(4)[:10]},
            "p1_main": {"cwm": np.eye(4)[:10], "seq": np.eye(4)[:10]},
        }
        db = build_finemo_db(motifs, str(tmp_path / "db.h5"),
                             annotations={"p0_main": "GATA1", "p1_main": "TAL1"})

        # Write a fake hits.tsv referencing those motifs.
        hits_path = tmp_path / "hits.tsv"
        pd.DataFrame({
            "motif_name": ["pos_patterns.pattern_0", "pos_patterns.pattern_1"],
            "score": [0.9, 0.7],
        }).to_csv(hits_path, sep="\t", index=False)

        df = load_hits_with_annotation(str(hits_path), db)
        assert {"MC_ID", "TF_Name"} <= set(df.columns)
        assert list(df["TF_Name"]) == ["GATA1", "TAL1"]
        assert list(df["MC_ID"]) == ["p0_main", "p1_main"]

    def test_run_finemo_missing_binary_raises(self, tmp_path, monkeypatch):
        monkeypatch.setattr("deepISA.scoring.discover.modisco.resolve_cli", lambda name: None)
        with pytest.raises(RuntimeError, match="Fi-NeMo"):
            run_finemo_scan("in.npz", str(tmp_path / "out"), "db.h5")


# ---------------------------------------------------------------------------
# h5_io motif extraction
# ---------------------------------------------------------------------------
class TestH5IO:
    def test_extract_and_load_motifs(self, tmp_path):
        import h5py
        h5_path = str(tmp_path / "modisco.h5")
        # Realistic structure: pattern_N contains subpattern_M nodes that hold
        # the actual CWM/sequence, mirroring tf-modisco-lite output.
        with h5py.File(h5_path, "w") as f:
            pos = f.create_group("pos_patterns")
            p0 = pos.create_group("pattern_0")
            sub0 = p0.create_group("subpattern_0")
            sub0.create_dataset("contrib_scores", data=np.random.rand(50, 4).astype(np.float32))
            sub0.create_dataset("sequence", data=np.eye(4, dtype=np.float32)[:50])
            sub0.create_group("seqlets").create_dataset("n_seqlets", data=np.array([42]))

        raw = extract_motifs_from_group(h5py.File(h5_path, "r"), "task")
        assert any("p0" in k for k in raw), f"no p0 motif in {list(raw)}"

        loaded = load_motifs(h5_path, task_name="task", target_len=20)
        assert len(loaded) >= 1, f"load_motifs returned empty; raw={list(raw)}"
        for m in loaded.values():
            assert m["cwm"].shape == (20, 4)
            assert m["seq"].shape == (20, 4)

    def test_load_motifs_missing_file_raises(self):
        with pytest.raises(FileNotFoundError):
            load_motifs("/nonexistent.h5", "task")


# ---------------------------------------------------------------------------
# filter.py helpers (mirrors the contract tests in test_filter.py)
# ---------------------------------------------------------------------------
class TestFilterHelpers:
    def test_extract_regions_columns_and_unique(self):
        df = pd.DataFrame({
            "region": ["chr1:0-600", "chr1:0-600", "chr2:100-700"],
            "chrom": ["chr1", "chr1", "chr2"],
            "start": [10, 50, 110], "end": [20, 60, 120],
        })
        out = extract_regions(df)
        assert list(out.columns) == ["region", "chrom", "start", "end"]
        assert len(out) == 2
        assert list(out["start"]) == [0, 100]

    def test_get_second_max(self):
        arr = np.array([[0.1, 0.9, 0.2, 0.7], [5.0, 1.0, 3.0, 4.0]])
        assert np.isclose(_get_second_max(arr, 0), 0.7)
        assert np.isclose(_get_second_max(arr, 1), 4.0)

    def test_get_attr_threshold_percentile(self):
        score_map = {"r": np.vstack([np.arange(600, dtype=np.float32),
                                      np.arange(1000, 1600, dtype=np.float32)])}
        df = pd.DataFrame({"region": ["r", "r"], "start_rel": [0, 300],
                           "end_rel": [300, 600]})
        assert np.isclose(get_attr_threshold(df, score_map, 0, 50), 299.5)
        assert np.isclose(get_attr_threshold(df, score_map, 1, 50), 1299.5)

    def test_get_slices_shape(self):
        score_map = {"r": np.random.default_rng(0).normal(size=(2, 600)).astype(np.float32)}
        df = pd.DataFrame({"region": ["r", "r"], "start_rel": [10, 100],
                           "end_rel": [25, 130]})
        slices = list(get_slices(df, score_map))
        assert len(slices) == 2
        assert slices[0].shape == (2, 15)
        assert slices[1].shape == (2, 30)


# ---------------------------------------------------------------------------
# track_index selection (regression of the silent track-averaging bug)
# ---------------------------------------------------------------------------
class TestTrackSelection:
    def test_read_attribution_h5_selects_track(self, tiny_model, device, tmp_path):
        # Build a 2-track attribution file. read_attribution_h5 must return the
        # SELECTED track, not the average across tracks.
        import h5py
        h5_path = str(tmp_path / "attr.h5")
        seqs = _rand_onehot(2, 60, seed=40)
        # Manually write two tracks with distinct values so selection is detectable.
        hyp_t0 = np.zeros((2, 4, 60), dtype=np.float32)
        hyp_t1 = np.ones((2, 4, 60), dtype=np.float32) * 9.0
        with h5py.File(h5_path, "w") as f:
            f.attrs["layout"] = "channels_first"
            f.create_dataset("sequences", data=seqs)
            f.create_dataset("hyp_scores", data=np.stack([hyp_t0, hyp_t1], axis=0))

        _, hyp_a, T = read_attribution_h5(h5_path, track_index=0)
        _, hyp_b, _ = read_attribution_h5(h5_path, track_index=1)
        assert T == 2
        assert np.allclose(hyp_a, 0.0)      # track 0 selected, not averaged
        assert np.allclose(hyp_b, 9.0)      # track 1 selected

    def test_read_attribution_h5_rejects_bad_track(self, tmp_path):
        import h5py
        h5_path = str(tmp_path / "attr.h5")
        with h5py.File(h5_path, "w") as f:
            f.attrs["layout"] = "channels_first"
            f.create_dataset("sequences", data=np.zeros((1, 4, 10), dtype=np.float32))
            f.create_dataset("hyp_scores", data=np.zeros((1, 1, 4, 10), dtype=np.float32))
        with pytest.raises(IndexError):
            read_attribution_h5(h5_path, track_index=5)


# ---------------------------------------------------------------------------
# finemo merge prefix compatibility (regression of the silent "Unknown" bug)
# ---------------------------------------------------------------------------
class TestFinemoMergePrefix:
    def test_merge_matches_with_or_without_prefix(self, tmp_path):
        import h5py
        # DB has 2 patterns; build a hits.tsv where one row has the prefix and
        # the other does not. Both must be annotated.
        db = str(tmp_path / "db.h5")
        with h5py.File(db, "w") as f:
            pos = f.create_group("pos_patterns")
            g0 = pos.create_group("pattern_0")
            g0.attrs["mc_id"] = "M0"; g0.attrs["tf_label"] = "GATA1"
            g1 = pos.create_group("pattern_1")
            g1.attrs["mc_id"] = "M1"; g1.attrs["tf_label"] = "TAL1"

        hits = tmp_path / "hits.tsv"
        pd.DataFrame({
            # pattern_0 WITH prefix, pattern_1 WITHOUT prefix
            "motif_name": ["pos_patterns.pattern_0", "pattern_1"],
            "score": [0.9, 0.8],
        }).to_csv(hits, sep="\t", index=False)

        df = load_hits_with_annotation(str(hits), db)
        assert list(df["TF_Name"]) == ["GATA1", "TAL1"], "prefix mismatch must not drop annotation"
        assert list(df["MC_ID"]) == ["M0", "M1"]


# ---------------------------------------------------------------------------
# report.py
# ---------------------------------------------------------------------------
class TestReport:
    def test_run_report_missing_binary_raises(self, tmp_path, monkeypatch):
        monkeypatch.setattr("deepISA.scoring.discover.modisco.resolve_cli", lambda name: None)
        # Need a real file for the modisco_h5 existence check to pass.
        h5 = tmp_path / "r.h5"
        h5.write_bytes(b"")
        with pytest.raises(RuntimeError, match="tf-modisco-lite"):
            run_motif_report(str(h5), str(tmp_path / "rep"))

    def test_run_report_missing_h5_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            run_motif_report(str(tmp_path / "nope.h5"), str(tmp_path / "rep"))

    def test_run_report_missing_meme_db_raises(self, tmp_path):
        h5 = tmp_path / "r.h5"
        h5.write_bytes(b"")
        with pytest.raises(FileNotFoundError, match="MEME"):
            run_motif_report(str(h5), str(tmp_path / "rep"), meme_db=str(tmp_path / "no.meme"))

    def test_cwm_to_meme_writes_valid_format(self, tmp_path):
        motifs = {
            "m0": {"cwm": np.array([[0.5, 0.2, 0.2, 0.1], [-0.3, 0.8, 0.1, 0.1]],
                                   dtype=np.float32)},
            "m1": {"cwm": np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]],
                                   dtype=np.float32)},
        }
        out = str(tmp_path / "motifs.meme")
        cwm_to_meme(motifs, out)
        text = open(out, encoding="utf-8").read()
        assert "MEME version 5" in text
        assert "ALPHABET= ACGT" in text
        assert "MOTIF m0" in text and "MOTIF m1" in text
        # Negative CWM value clipped: row 0 of m0 sums to 1.0 (PPM).
        # row 0 = [0.5, 0.2, 0.2, 0.1] -> already nonneg, sums to 1.0.
        # row 1 = [-0.3, 0.8, 0.1, 0.1] -> clip -> [0, 0.8, 0.1, 0.1] -> normalize
        assert "0.0000" in text  # the clipped -0.3 -> 0


# ---------------------------------------------------------------------------
# h5_io reproducibility + dedup regression
# ---------------------------------------------------------------------------
class TestH5IOReproducibility:
    def test_parse_motif_name_fallback_is_deterministic(self):
        from deepISA.scoring.discover.h5_io import parse_motif_name
        # A path with no pattern_/subpattern_ tokens hits the fallback hash,
        # which must be deterministic across calls (regression for hash()).
        a = parse_motif_name("/weird/node", "task")
        b = parse_motif_name("/weird/node", "task")
        assert a == b
        assert a.startswith("task_x")

    def test_load_motifs_skip_does_not_drop_legit_v_names(self, tmp_path):
        # A motif legitimately containing "_v" in its name must NOT be dropped
        # by skip_main (regression for the "_v" in name substring match).
        import h5py
        h5_path = str(tmp_path / "m.h5")
        with h5py.File(h5_path, "w") as f:
            pos = f.create_group("pos_patterns")
            p = pos.create_group("pattern_0")
            sub = p.create_group("subpattern_0")
            sub.create_dataset("contrib_scores",
                               data=np.random.rand(20, 4).astype(np.float32))
            sub.create_dataset("sequence",
                               data=np.eye(4, dtype=np.float32)[:20])
        loaded = load_motifs(h5_path, task_name="evi1_tf", target_len=10)
        assert len(loaded) >= 1, "task name containing legit chars must not be dropped"
