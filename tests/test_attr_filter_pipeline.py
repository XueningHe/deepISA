import sys, os, pytest, yaml, torch
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock

sys.path.insert(0, "deepISA/src")


def _make_filtered_motif_csv(tmp_path):
    """Minimal motif CSV matching attr_filter output: 3 motifs, 1 region."""
    df = pd.DataFrame({
        "chrom":              ["chr1", "chr1", "chr1"],
        "start":              [1010,   1030,   1050],
        "end":                [1025,   1045,   1065],
        "tf":                 ["NFKB1","SP1","IRF1"],
        "score":              [900,    850,    800],
        "strand":             ["+",    "+",    "-"],
        "region":             ["chr1:1000-1600"] * 3,
        "start_rel":          [10,     30,     50],
        "end_rel":            [25,     45,     65],
        "second_max_t0":      [0.9,    0.7,    0.85],
        "pass_threshold_t0":  [1,      1,      1],
    })
    p = tmp_path / "motif_filtered.csv"
    df.to_csv(p, index=False)
    return str(p)


def _make_fasta(tmp_path):
    """Write a minimal FASTA for chr1 (2000 bp of A) with index."""
    fa = tmp_path / "mini.fa"
    seq = "A" * 2000
    fa.write_text(f">chr1\n{seq}\n")
    fai = tmp_path / "mini.fa.fai"
    fai.write_text(f"chr1\t2000\t6\t2000\t2001\n")
    return str(fa)


def test_two_stage_pipeline_isa_cols(tmp_path):
    """
    Verifies that run_single_isa preserves the pass_threshold_t0 column from
    a pre-filtered motif CSV (as attr_filter would produce) and writes correct
    isa_t0 values.
    """
    filtered_path = _make_filtered_motif_csv(tmp_path)
    fasta_path    = _make_fasta(tmp_path)
    out_path      = str(tmp_path / "isa_out.csv")

    from alphagenome.models.dna_output import OutputType

    cfg = {"api_key": "k", "output_type": "DNASE",
           "biosample_name": "GM12878", "context_len": 16384,
           "seq_len": 600, "aggregation": "sum"}
    (tmp_path / "cfg.yaml").write_text(yaml.dump(cfg))

    fake_meta = pd.DataFrame({
        "biosample_name": ["GM12878"],
        "output_type":    [OutputType["DNASE"]],
        "ontology_curie": ["EFO:0002784"],
    })

    def _fake_output(val):
        out = MagicMock()
        track = MagicMock()
        track.values = np.full((16384, 1), val, dtype=np.float32)
        # metadata must be real DataFrame so probe col-index extraction works
        track.metadata = pd.DataFrame({"biosample_name": ["GM12878"]})
        out.dnase = track
        return out

    with patch("deepisa_ag.adapter.dna_client") as mock_dc:
        mock_dc.create.return_value.output_metadata.return_value.concatenate.return_value = fake_meta
        # 1 probe (__init__) + 1 orig + 3 ablated = 5 calls; extra entries are unused
        mock_dc.create.return_value.predict_sequence.side_effect = [
            _fake_output(1.0),                                    # probe in __init__
            _fake_output(1.0), _fake_output(0.5),
            _fake_output(1.0), _fake_output(0.5),
            _fake_output(1.0), _fake_output(0.5),
        ]
        from deepisa_ag import AlphaGenomeAdapter
        adapter = AlphaGenomeAdapter(str(tmp_path / "cfg.yaml"))

    from deepISA.scoring.single_isa import run_single_isa
    run_single_isa(
        model                 = adapter,
        fasta_path            = fasta_path,
        motif_locs_path       = filtered_path,
        outpath               = out_path,
        device                = torch.device("cpu"),
        tracks                = [0],
        num_regions_per_batch = 10,
        pred_batch_size       = 1,
    )

    result = pd.read_csv(out_path)
    assert "isa_t0" in result.columns
    assert "pass_threshold_t0" in result.columns      # pass-through from filter
    assert len(result) == 3
    # orig sum = 1.0 * 600 = 600, mut sum = 0.5 * 600 = 300  →  isa = 300
    assert float(result["isa_t0"].iloc[0]) == pytest.approx(300.0, rel=1e-3)
