"""
Standalone CLI for motif_filter.

Usage:
    python -m deepISA.motif_filter.cli.run_filter \\
        --model /path/to/model.pt \\
        --fasta-dir /path/to/hg38/chroms/ \\
        --regions /path/to/regions.csv \\
        --motifs /path/to/motif_locs.csv \\
        --out /path/to/output.csv \\
        --device cpu \\
        --n-regions 200
"""
import argparse
import json
import sys
from pathlib import Path

import torch

# Allow both `python -m` and `python src/deepISA/.../run_filter.py`
_src_root = Path(__file__).resolve().parents[3]  # cli → motif_filter → deepISA → src
if str(_src_root) not in sys.path:
    sys.path.insert(0, str(_src_root))

from deepISA.motif_filter.cli.common_args import add_common_args
from deepISA.motif_filter.pipeline.filter_pipeline import run_pipeline
from deepISA.motif_filter.utils.io import load_regions, load_motif_locs, save_filtered_motifs


def parse_args():
    p = argparse.ArgumentParser(description="Non-motif null attribution motif filter")
    add_common_args(p)
    p.add_argument("--motifs", required=True, help="Path to motif_locs CSV")
    p.add_argument("--out", required=True, help="Output CSV path")
    p.add_argument(
        "--report", default=None, help="Optional JSON report output path"
    )
    return p.parse_args()


def main():
    args = parse_args()

    # ── Environment guard (fail fast on version mismatch) ────────
    from deepISA.utils.deepisa_guard import validate_deepisa_environment
    validate_deepisa_environment()

    from deepISA.modeling.cnn import Conv

    print(f"Loading model from {args.model} ...")
    state = torch.load(args.model, map_location=args.device, weights_only=False)
    model = Conv(seq_len=600, ks=[15, 9, 9, 9, 9], cs=[64] * 5, ds=[1, 2, 4, 8, 16])
    model.load_state_dict(state)
    model.to(args.device).eval()

    print(f"Loading regions from {args.regions} ...")
    regions_df = load_regions(args.regions)

    print(f"Loading motifs from {args.motifs} ...")
    motif_df = load_motif_locs(args.motifs)

    print(f"Running pipeline on {args.n_regions or len(regions_df)} regions ...")
    result = run_pipeline(
        model=model,
        regions_df=regions_df,
        motif_locs_df=motif_df,
        fasta_dir=args.fasta_dir,
        seq_len=args.seq_len,
        window_size=args.window_size,
        stride=args.stride,
        device=args.device,
        n_regions=args.n_regions,
    )

    save_filtered_motifs(result, args.out)
    print(f"Saved {len(result)} filtered motifs to {args.out}")

    report = {
        "T_sum": float(result.attrs.get("T_sum", 0)),
        "T_peak": float(result.attrs.get("T_peak", 0)),
        "n_input_motifs": int(result.attrs.get("n_input_motifs", 0)),
        "n_passed": int(result.attrs.get("n_passed", len(result))),
        "pass_rate": float(result.attrs.get("pass_rate", 0)),
        "n_null_windows": int(result.attrs.get("n_null_windows", 0)),
        "n_regions_low_windows": int(result.attrs.get("n_regions_low_windows", 0)),
        "window_size": args.window_size,
        "stride": args.stride,
        "n_regions": args.n_regions or len(regions_df),
    }
    print(json.dumps(report, indent=2))

    if args.report:
        with open(args.report, "w") as f:
            json.dump(report, f, indent=2)
        print(f"Report saved to {args.report}")


if __name__ == "__main__":
    main()
