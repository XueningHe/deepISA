#!/bin/bash
# ============================================================
# Example bash script to run motif_filter
# ============================================================
#
# Requirements:
#   - trained deepISA model (.pt file)
#   - chromosome FASTA files (chr*.fa)
#   - regions CSV (chrom, start, end, region)
#   - motif_locs CSV (chrom, start, end, tf, score, strand, region)
#
# Output:
#   - filtered motif CSV
#   - JSON report with T_sum, T_peak, pass rates
# ============================================================

MODEL="/path/to/your/model.pt"
FASTA_DIR="/path/to/hg38/"
REGIONS="/path/to/regions_pos_with_count.csv"
MOTIFS="/path/to/motif_locs_raw.csv"
OUT="/path/to/motif_locs_filtered.csv"
REPORT="/path/to/filter_report.json"
DEVICE="cuda"

# Run filter
python -m motif_filter.cli.run_filter \
    --model "$MODEL" \
    --fasta-dir "$FASTA_DIR" \
    --regions "$REGIONS" \
    --motifs "$MOTIFS" \
    --out "$OUT" \
    --device "$DEVICE" \
    --n-regions 200 \
    --window-size 20 \
    --stride 20 \
    --report "$REPORT"
