#!/usr/bin/env python3
"""
run_single_isa.py — Run ISA with optional motif prefiltering.

Usage:
    python scripts/run_single_isa.py \\
        --prefilter motif_filter \\
        --jaspar data/JASPAR2026_CORE_non-redundant_pfms_jaspar.txt \\
        --fasta_dir /path/to/hg38/chroms \\
        --regions data/regions_pos_with_count.csv \\
        --model data/model_blympho.pt \\
        --n_regions 20 \\
        --device cuda

With --prefilter motif_filter:
    1. Load regions + motif hits from JASPAR
    2. Run motif_filter pipeline (DeepLIFT non-motif null)
    3. Feed filtered motifs into ISA

Without --prefilter:
    Works as original ISA (backward compatible).
"""
import argparse
import os
import sys

# Add src/ to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import torch
import pandas as pd
from loguru import logger

from deepISA.modeling.cnn import Conv
from deepISA.scoring.single_isa import run_single_isa as _run_isa
from deepISA.scoring.annotation import map_motifs


def parse_args():
    parser = argparse.ArgumentParser(description="Run ISA with optional motif prefiltering")
    parser.add_argument("--prefilter", type=str, default=None,
                        choices=["motif_filter", None],
                        help="Prefilter motif hits before ISA. 'motif_filter' uses "
                             "DeepLIFT non-motif null attribution filter.")
    parser.add_argument("--jaspar", type=str, required=True,
                        help="Path to JASPAR PFM file")
    parser.add_argument("--fasta_dir", type=str, required=True,
                        help="Path to chromosome FASTA directory (chr*.fa)")
    parser.add_argument("--regions", type=str, required=True,
                        help="Path to regions CSV")
    parser.add_argument("--model", type=str, required=True,
                        help="Path to trained model .pt file")
    parser.add_argument("--outdir", type=str, default="results",
                        help="Output directory")
    parser.add_argument("--n_regions", type=int, default=None,
                        help="Limit to first N regions (for testing)")
    parser.add_argument("--device", type=str, default="cuda",
                        help="torch device")
    parser.add_argument("--seq_len", type=int, default=600,
                        help="Sequence length")
    parser.add_argument("--window_size", type=int, default=20,
                        help="Non-motif window size (bp)")
    parser.add_argument("--stride", type=int, default=20,
                        help="Non-motif window stride (bp)")
    parser.add_argument("--track_idx", type=int, default=0,
                        help="ISA track index")
    return parser.parse_args()


def load_model(model_path, device):
    state = torch.load(model_path, map_location=device, weights_only=False)
    model = Conv(seq_len=600, ks=[15, 9, 9, 9, 9], cs=[64]*5, ds=[1, 2, 4, 8, 16])
    model.load_state_dict(state)
    model.to(device).eval()
    return model


def load_regions(regions_path, n_regions=None):
    df = pd.read_csv(regions_path)
    df["region"] = df["chrom"].astype(str) + ":" + df["start"].astype(str) + "-" + df["end"].astype(str)
    if n_regions is not None:
        df = df.head(n_regions).reset_index(drop=True)
    return df


def load_motif_locs(motif_locs_path):
    df = pd.read_csv(motif_locs_path)
    return df


def main():
    args = parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    logger.info(f"Device: {args.device}")
    logger.info(f"Prefilter: {args.prefilter}")

    # ── Load model ────────────────────────────────────────────
    logger.info(f"Loading model from {args.model}")
    model = load_model(args.model, args.device)

    # ── Load regions ──────────────────────────────────────────
    logger.info(f"Loading regions from {args.regions}")
    regions_df = load_regions(args.regions, args.n_regions)
    logger.info(f"Regions: {len(regions_df)}")

    # ── Map JASPAR motifs ────────────────────────────────────
    motif_locs_path = os.path.join(args.outdir, "motif_locs_raw.csv")
    logger.info(f"Mapping JASPAR motifs from {args.jaspar}")
    motif_locs_df = map_motifs(
        regions_df=regions_df,
        jaspar_path=args.jaspar,
        outpath=motif_locs_path,
        score_thresh=0,      # keep all hits for filtering
        expressed_tfs=None,
    )
    logger.info(f"Raw motif hits: {len(motif_locs_df)}")

    # ── Optional prefilter ────────────────────────────────────
    if args.prefilter == "motif_filter":
        from deepISA.motif_filter.pipeline.filter_pipeline import run_pipeline
        from deepISA.motif_filter.utils.io import load_motif_locs as _load_ml

        logger.info("Running motif_filter prefilter...")

        filtered_df = run_pipeline(
            model=model,
            regions_df=regions_df,
            motif_locs_df=motif_locs_df,
            fasta_dir=args.fasta_dir,
            seq_len=args.seq_len,
            window_size=args.window_size,
            stride=args.stride,
            device=args.device,
        )

        motif_locs_filtered_path = os.path.join(args.outdir, "motif_locs_filtered.csv")
        filtered_df.to_csv(motif_locs_filtered_path, index=False)
        logger.info(f"Filtered motifs: {len(filtered_df)}  "
                    f"(pass_rate={filtered_df.attrs.get('pass_rate', 'N/A'):.1%})")
        logger.info(f"T_sum={filtered_df.attrs.get('T_sum', 'N/A'):.4f}, "
                    f"T_peak={filtered_df.attrs.get('T_peak', 'N/A'):.4f}")

        motif_locs_for_isa = motif_locs_filtered_path
    else:
        motif_locs_for_isa = motif_locs_path

    # ── Run ISA ───────────────────────────────────────────────
    isa_out_path = os.path.join(args.outdir, "isa_single.csv")
    logger.info(f"Running ISA...")

    _run_isa(
        model=model,
        fasta_path=args.fasta_dir,
        motif_locs=motif_locs_for_isa,
        outpath=isa_out_path,
        track_idx=args.track_idx,
        device=args.device,
        batch_size=200,
    )

    logger.info(f"ISA complete. Results → {isa_out_path}")


if __name__ == "__main__":
    main()
