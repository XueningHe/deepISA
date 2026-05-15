import numpy as np
import pandas as pd
import pytest
import torch
import bioframe as bf

from captum.attr import DeepLift

from deepISA.modeling.cnn import Conv
from deepISA.scoring.filter import (
    extract_regions,
    scan_deeplift_scores,
    get_slices,
    _get_second_max,
    get_attr_threshold,
    attr_filter,
)
from deepISA.utils import get_data_resource, get_sequences_from_df, one_hot_encode


MODEL_PATH = get_data_resource("model_blympho.pt")


# -----------------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------------

@pytest.fixture(scope="session")
def device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture(scope="session")
def real_model(device):
    model = Conv()
    state_dict = torch.load(MODEL_PATH, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()
    model.to(device)
    return model


@pytest.fixture(scope="session")
def inferred_num_outputs(real_model, device):
    """
    Infer output dimensionality using a 600bp input in the real one-hot layout.
    """
    x_np = np.zeros((1, 4, 600), dtype=np.float32)
    x = torch.tensor(x_np, dtype=torch.float32, device=device)
    with torch.no_grad():
        y = real_model(x)

    print(f"[DEBUG] probe input shape to real model: {tuple(x.shape)}")
    print(f"[DEBUG] probe output shape from real model: {tuple(y.shape)}")

    if y.ndim == 1:
        return int(y.shape[0])
    return int(y.shape[-1])


@pytest.fixture
def tiny_600bp_fasta(tmp_path):
    """
    Create a FASTA with enough sequence to support fetching chr12:0-600.
    """
    fasta_path = tmp_path / "tiny_600bp.fa"
    seq = "ACGT" * 150  # 600 bp
    assert len(seq) == 600
    fasta_path.write_text(f">chr12\n{seq}\n")
    return fasta_path


@pytest.fixture
def tiny_1200bp_fasta(tmp_path):
    """
    FASTA long enough to support two distinct 600bp regions on chr12:
      - chr12:0-600
      - chr12:600-1200
    """
    fasta_path = tmp_path / "tiny_1200bp.fa"
    seq = "ACGT" * 300  # 1200 bp
    assert len(seq) == 1200
    fasta_path.write_text(f">chr12\n{seq}\n")
    return fasta_path


@pytest.fixture
def real_like_motif_df():
    """
    Same schema as motif_locs.csv, but coordinates are made compatible with the
    tiny test FASTA. Region length remains 600 bp.
    """
    return pd.DataFrame(
        {
            "chrom": ["chr12", "chr12"],
            "start": [30, 57],
            "end": [45, 76],
            "start_rel": [30, 57],
            "end_rel": [45, 76],
            "tf": ["ZNF418", "RREB1"],
            "score": [500, 571],
            "strand": ["+", "+"],
            "region": ["chr12:0-600", "chr12:0-600"],
            "second_max_t0": [0.04717009, 0.07814141],
            "pass_threshold_t0": [1, 1],
        }
    )


@pytest.fixture
def real_like_non_motif_df():
    """
    Same schema as non_motif_locs.csv, but coordinates are compatible with the
    tiny FASTA.
    """
    return pd.DataFrame(
        {
            "chrom": ["chr12", "chr12", "chr12"],
            "start": [0, 45, 76],
            "end": [30, 57, 134],
            "region": ["chr12:0-600", "chr12:0-600", "chr12:0-600"],
            "start_rel": [0, 45, 76],
            "end_rel": [30, 57, 134],
        }
    )


@pytest.fixture
def motif_csv(tmp_path, real_like_motif_df):
    p = tmp_path / "motif_locs.csv"
    real_like_motif_df.to_csv(p, index=False)
    return p


@pytest.fixture
def non_motif_csv(tmp_path, real_like_non_motif_df):
    p = tmp_path / "non_motif_locs.csv"
    real_like_non_motif_df.to_csv(p, index=False)
    return p


# -----------------------------------------------------------------------------
# Pure helper tests
# -----------------------------------------------------------------------------

def test_extract_regions_parses_unique_regions_from_realistic_motif_df(real_like_motif_df):
    regions_df = extract_regions(real_like_motif_df)

    assert list(regions_df.columns) == ["region", "chrom", "start", "end"]
    assert len(regions_df) == 1

    row = regions_df.iloc[0]
    assert row["region"] == "chr12:0-600"
    assert row["chrom"] == "chr12"
    assert row["start"] == 0
    assert row["end"] == 600


def test_get_second_max_basic():
    s = np.array(
        [
            [0.1, 0.9, 0.2, 0.7],
            [5.0, 1.0, 3.0, 4.0],
        ]
    )
    assert np.isclose(_get_second_max(s, 0), 0.7)
    assert np.isclose(_get_second_max(s, 1), 4.0)


def test_get_attr_threshold_uses_correct_track_and_percentile_with_real_scan_shape():
    score_map = {
        "chr12:0-600": np.array(
            [
                np.arange(600, dtype=np.float32),
                np.arange(1000, 1600, dtype=np.float32),
            ]
        )
    }

    df = pd.DataFrame(
        {
            "chrom": ["chr12", "chr12"],
            "start": [0, 300],
            "end": [300, 600],
            "region": ["chr12:0-600", "chr12:0-600"],
            "start_rel": [0, 300],
            "end_rel": [300, 600],
        }
    )

    thresh0 = get_attr_threshold(df, score_map, track_internal_idx=0, percentile=50)
    thresh1 = get_attr_threshold(df, score_map, track_internal_idx=1, percentile=50)

    assert np.isclose(thresh0, 299.5)
    assert np.isclose(thresh1, 1299.5)


# -----------------------------------------------------------------------------
# Real model + real one_hot_encode + real Captum checks
# -----------------------------------------------------------------------------

def test_real_one_hot_encode_shape_is_model_compatible(tiny_600bp_fasta, real_model, device):
    regions_df = pd.DataFrame(
        {
            "chrom": ["chr12"],
            "start": [0],
            "end": [600],
            "region": ["chr12:0-600"],
        }
    )

    fasta = bf.load_fasta(str(tiny_600bp_fasta))
    seqs = get_sequences_from_df(regions_df, fasta)

    assert len(seqs) == 1
    assert len(seqs[0]) == 600

    x_ohe = one_hot_encode(seqs)
    print(f"[DEBUG] fetched sequence length: {len(seqs[0])}")
    print(f"[DEBUG] one_hot_encode output shape: {x_ohe.shape}")

    assert x_ohe.shape == (1, 4, 600)

    x = torch.tensor(x_ohe, dtype=torch.float32, device=device, requires_grad=True)
    with torch.enable_grad():
        y = real_model(x)

    print(f"[DEBUG] model input tensor shape: {tuple(x.shape)}")
    print(f"[DEBUG] model output tensor shape: {tuple(y.shape)}")

    assert x.shape == (1, 4, 600)
    assert y.ndim >= 1


def test_real_deeplift_attr_shape_single_track(real_model, device, tiny_600bp_fasta):
    regions_df = pd.DataFrame(
        {
            "chrom": ["chr12"],
            "start": [0],
            "end": [600],
            "region": ["chr12:0-600"],
        }
    )

    fasta = bf.load_fasta(str(tiny_600bp_fasta))
    seqs = get_sequences_from_df(regions_df, fasta)
    assert len(seqs[0]) == 600

    x_ohe = one_hot_encode(seqs)
    x = torch.tensor(x_ohe, dtype=torch.float32, device=device, requires_grad=True)
    baseline = torch.zeros_like(x)

    dl = DeepLift(real_model)

    with torch.enable_grad():
        y = real_model(x)

    print(f"[DEBUG] single-track raw model output shape: {tuple(y.shape)}")

    attr = dl.attribute(x, baseline, target=0)
    print(f"[DEBUG] single-track raw captum attr shape: {tuple(attr.shape)}")

    scores = torch.abs(attr).sum(dim=1).cpu().detach().numpy()
    print(f"[DEBUG] single-track reduced scores shape after sum(dim=1): {scores.shape}")

    assert tuple(x.shape) == (1, 4, 600)
    assert tuple(attr.shape) == (1, 4, 600)
    assert scores.shape == (1, 600)


def test_scan_deeplift_scores_single_track_shape_with_real_model(
    real_model, device, tiny_600bp_fasta, real_like_motif_df
):
    regions_df = extract_regions(real_like_motif_df)
    tracks = [0]

    score_map = scan_deeplift_scores(
        model=real_model,
        regions_df=regions_df,
        fasta_path=str(tiny_600bp_fasta),
        tracks=tracks,
        device=device,
        attr_batch_size=1,
    )

    assert set(score_map.keys()) == {"chr12:0-600"}

    arr = score_map["chr12:0-600"]
    print(f"[DEBUG] scan_deeplift_scores single-track output shape: {arr.shape}")

    assert isinstance(arr, np.ndarray)
    assert arr.shape == (1, 600)
    assert np.isfinite(arr).all()


def test_scan_deeplift_scores_multi_track_shape_with_real_model(
    real_model, device, inferred_num_outputs, tiny_600bp_fasta, real_like_motif_df
):
    if inferred_num_outputs < 2:
        pytest.skip("Real model has fewer than 2 outputs; cannot test multi-track behavior.")

    regions_df = extract_regions(real_like_motif_df)
    tracks = [0, 1]

    score_map = scan_deeplift_scores(
        model=real_model,
        regions_df=regions_df,
        fasta_path=str(tiny_600bp_fasta),
        tracks=tracks,
        device=device,
        attr_batch_size=1,
    )

    arr = score_map["chr12:0-600"]
    print(f"[DEBUG] scan_deeplift_scores multi-track output shape: {arr.shape}")

    assert isinstance(arr, np.ndarray)
    assert arr.shape == (2, 600)
    assert np.isfinite(arr).all()


def test_scan_deeplift_scores_multiple_sequences_single_track_real_batch(
    real_model, device, tiny_1200bp_fasta
):
    """
    Real batch test: 2 distinct 600bp sequences, single track, one call.
    """
    regions_df = pd.DataFrame(
        {
            "region": ["chr12:0-600", "chr12:600-1200"],
            "chrom": ["chr12", "chr12"],
            "start": [0, 600],
            "end": [600, 1200],
        }
    )

    score_map = scan_deeplift_scores(
        model=real_model,
        regions_df=regions_df,
        fasta_path=str(tiny_1200bp_fasta),
        tracks=[0],
        device=device,
        attr_batch_size=2,
    )

    assert set(score_map.keys()) == {"chr12:0-600", "chr12:600-1200"}

    for region in ["chr12:0-600", "chr12:600-1200"]:
        arr = score_map[region]
        print(f"[DEBUG] {region} single-track batched output shape: {arr.shape}")
        assert arr.shape == (1, 600)
        assert np.isfinite(arr).all()


def test_scan_deeplift_scores_multiple_sequences_multi_track_real_batch(
    real_model, device, inferred_num_outputs, tiny_1200bp_fasta
):
    """
    Real batch test: 2 distinct 600bp sequences, multiple tracks, one call.
    """
    if inferred_num_outputs < 2:
        pytest.skip("Real model has fewer than 2 outputs; cannot test multi-track behavior.")

    regions_df = pd.DataFrame(
        {
            "region": ["chr12:0-600", "chr12:600-1200"],
            "chrom": ["chr12", "chr12"],
            "start": [0, 600],
            "end": [600, 1200],
        }
    )

    tracks = [0, 1]
    score_map = scan_deeplift_scores(
        model=real_model,
        regions_df=regions_df,
        fasta_path=str(tiny_1200bp_fasta),
        tracks=tracks,
        device=device,
        attr_batch_size=2,
    )

    assert set(score_map.keys()) == {"chr12:0-600", "chr12:600-1200"}

    for region in ["chr12:0-600", "chr12:600-1200"]:
        arr = score_map[region]
        print(f"[DEBUG] {region} multi-track batched output shape: {arr.shape}")
        assert arr.shape == (2, 600)
        assert np.isfinite(arr).all()


# -----------------------------------------------------------------------------
# Mock tests preserving real scan_deeplift_scores output convention
# -----------------------------------------------------------------------------

def test_get_slices_preserves_real_scan_output_convention_single_track():
    seq_len = 600
    num_tracks = 1
    rng = np.random.default_rng(123)

    region_scores = rng.normal(size=(num_tracks, seq_len)).astype(np.float32)
    score_map = {"chr12:0-600": region_scores}

    df = pd.DataFrame(
        {
            "chrom": ["chr12", "chr12"],
            "start": [30, 57],
            "end": [45, 76],
            "start_rel": [30, 57],
            "end_rel": [45, 76],
            "region": ["chr12:0-600", "chr12:0-600"],
        }
    )

    slices = list(get_slices(df, score_map))
    assert len(slices) == 2

    for row, s in zip(df.itertuples(), slices):
        manual = region_scores[:, row.start_rel:row.end_rel]
        print(
            f"[DEBUG] single-track get_slices row={row.Index} "
            f"start_rel={row.start_rel} end_rel={row.end_rel} "
            f"slice_shape={s.shape}"
        )
        assert s.shape == (1, row.end_rel - row.start_rel)
        np.testing.assert_allclose(s, manual, rtol=1e-7, atol=1e-7)


def test_get_slices_preserves_real_scan_output_convention_multi_track():
    seq_len = 600
    num_tracks = 2
    rng = np.random.default_rng(456)

    region_scores = rng.normal(size=(num_tracks, seq_len)).astype(np.float32)
    score_map = {"chr12:0-600": region_scores}

    df = pd.DataFrame(
        {
            "chrom": ["chr12", "chr12"],
            "start": [30, 57],
            "end": [45, 76],
            "start_rel": [30, 57],
            "end_rel": [45, 76],
            "region": ["chr12:0-600", "chr12:0-600"],
        }
    )

    slices = list(get_slices(df, score_map))
    assert len(slices) == 2

    for row, s in zip(df.itertuples(), slices):
        manual = region_scores[:, row.start_rel:row.end_rel]
        print(
            f"[DEBUG] multi-track get_slices row={row.Index} "
            f"start_rel={row.start_rel} end_rel={row.end_rel} "
            f"slice_shape={s.shape}"
        )
        assert s.shape == (2, row.end_rel - row.start_rel)
        np.testing.assert_allclose(s, manual, rtol=1e-7, atol=1e-7)


# -----------------------------------------------------------------------------
# End-to-end integration
# -----------------------------------------------------------------------------

def test_get_slices_matches_real_scan_output_exactly(
    real_model, device, inferred_num_outputs, tiny_600bp_fasta, real_like_motif_df
):
    tracks = [0] if inferred_num_outputs < 2 else [0, 1]
    regions_df = extract_regions(real_like_motif_df)

    score_map = scan_deeplift_scores(
        model=real_model,
        regions_df=regions_df,
        fasta_path=str(tiny_600bp_fasta),
        tracks=tracks,
        device=device,
        attr_batch_size=1,
    )

    slices = list(get_slices(real_like_motif_df, score_map))
    assert len(slices) == len(real_like_motif_df)

    for row, s in zip(real_like_motif_df.itertuples(), slices):
        manual = score_map[row.region][:, row.start_rel:row.end_rel]
        print(
            f"[DEBUG] exact-match row={row.Index} tracks={len(tracks)} "
            f"manual_shape={manual.shape} get_slices_shape={s.shape}"
        )
        np.testing.assert_allclose(s, manual, rtol=1e-6, atol=1e-6)


def test_attr_filter_end_to_end_with_real_model_and_realistic_csvs(
    real_model,
    device,
    inferred_num_outputs,
    tiny_600bp_fasta,
    motif_csv,
    non_motif_csv,
):
    tracks = [0] if inferred_num_outputs < 2 else [0, 1]

    filtered_df = attr_filter(
        motif_locs_path=str(motif_csv),
        non_motif_locs_path=str(non_motif_csv),
        model=real_model,
        fasta_path=str(tiny_600bp_fasta),
        tracks=tracks,
        attr_percentile=90,
        device=device,
        attr_batch_size=1,
    )

    original_df = pd.read_csv(motif_csv)
    assert len(filtered_df) <= len(original_df)

    for t in tracks:
        second_col = f"second_max_t{t}"
        pass_col = f"pass_threshold_t{t}"

        assert second_col in filtered_df.columns
        assert pass_col in filtered_df.columns
        assert filtered_df[pass_col].isin([0, 1]).all()

    pass_cols = [f"pass_threshold_t{t}" for t in tracks]
    if len(filtered_df) > 0:
        assert filtered_df[pass_cols].any(axis=1).all()


def test_attr_filter_empty_input_returns_empty_df(
    tmp_path, real_model, device, inferred_num_outputs, tiny_600bp_fasta, non_motif_csv
):
    empty_motif_csv = tmp_path / "empty_motifs.csv"
    pd.DataFrame(
        columns=[
            "chrom", "start", "end", "start_rel", "end_rel",
            "tf", "score", "strand", "region"
        ]
    ).to_csv(empty_motif_csv, index=False)

    tracks = [0] if inferred_num_outputs < 2 else [0, 1]

    out = attr_filter(
        motif_locs_path=str(empty_motif_csv),
        non_motif_locs_path=str(non_motif_csv),
        model=real_model,
        fasta_path=str(tiny_600bp_fasta),
        tracks=tracks,
        attr_percentile=90,
        device=device,
        attr_batch_size=1,
    )

    assert isinstance(out, pd.DataFrame)
    assert out.empty