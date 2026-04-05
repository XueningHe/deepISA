"""Shared CLI argument definitions for motif_filter and ISA scripts."""
import argparse


def add_common_args(p: argparse.ArgumentParser) -> None:
    """Add common arguments to any ArgumentParser instance.

    Parameters
    ----------
    p : argparse.ArgumentParser
        Parser to extend.
    """
    p.add_argument(
        "--fasta-dir",
        required=True,
        dest="fasta_dir",
        help="Path to chromosome FASTA directory (chr*.fa)",
    )
    p.add_argument(
        "--regions",
        required=True,
        help="Path to regions CSV (chrom, start, end columns)",
    )
    p.add_argument(
        "--model",
        required=True,
        help="Path to trained .pt model file",
    )
    p.add_argument(
        "--device",
        default="cuda",
        help="Compute device: cuda or cpu (default: cuda)",
    )
    p.add_argument(
        "--n-regions",
        type=int,
        default=None,
        help="Limit to first N regions (default: all)",
    )
    p.add_argument(
        "--seq-len",
        type=int,
        default=600,
        help="Sequence length in bp (default: 600)",
    )
    p.add_argument(
        "--window-size",
        type=int,
        default=20,
        help="Non-motif window size in bp (default: 20)",
    )
    p.add_argument(
        "--stride",
        type=int,
        default=20,
        help="Non-motif window stride in bp (default: 20, no overlap)",
    )
