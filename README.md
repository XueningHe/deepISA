# deepISA_filter

Attribution-based motif filtering + ISA for deepISA.

## Project Structure

```
deepisa_filter/
├── data/                          ← shipped with repo
│   ├── model_blympho.pt           ← trained model
│   ├── regions_pos_with_count.csv  ← genomic regions
│   ├── motif_locs.csv             ← JASPAR motif hits
│   └── JASPAR2026_CORE_...pfms   ← JASPAR PFM database
│
├── src/deepISA/
│   ├── motif_filter/              ← prefilter plugin
│   │   ├── core/                 ← attribution, window, scoring, threshold
│   │   ├── pipeline/             ← filter_pipeline.py
│   │   ├── utils/                ← onehot, fasta, io
│   │   ├── notebooks/            ← motif_filter_tutorial.ipynb
│   │   └── METHOD.md             ← method documentation
│   └── scoring/                  ← ISA scoring
│
├── scripts/
│   └── run_single_isa.py         ← CLI: prefilter → ISA
│
└── tutorials/
    └── motif_filter_tutorial.ipynb
```

## Data Setup (Required)

The pipeline requires the hg38 genome (FASTA, chromosome-level).

### Download hg38

```bash
# Download hg38 (one-time, ~1 GB)
mkdir -p /path/to/hg38
cd /path/to/hg38

wget https://hgdownload.soe.ucsc.edu/goldenPath/hg38/bigZips/hg38.fa.gz
gunzip hg38.fa.gz

# Index and split into per-chromosome FASTA
mkdir -p chroms
samtools faidx hg38.fa
awk '{print $1}' hg38.fa.fai | while read chr; do
    samtools faidx hg38.fa "$chr" > chroms/${chr}.fa
done

# Verify
ls chroms/chr1.fa   # should exist
```

Expected path: `/path/to/hg38/chroms/`

### Pre-existing data (shipped with repo)

```bash
# Verify shipped data files
ls data/model_blympho.pt
ls data/regions_pos_with_count.csv
ls data/motif_locs.csv
```

## Quick Start

### Option 1: Python API

```python
import sys
sys.path.insert(0, "src")

from deepISA.motif_filter.pipeline.filter_pipeline import run_pipeline
from deepISA.modeling.cnn import Conv
import torch

# Load model
model = Conv(seq_len=600, ks=[15,9,9,9,9], cs=[64]*5, ds=[1,2,4,8,16])
model.load_state_dict(torch.load("data/model_blympho.pt", weights_only=False))
model.eval()

# Run filter
result = run_pipeline(
    model=model,
    regions_df=regions_df,
    motif_locs_df=motifs_df,
    fasta_dir="/path/to/hg38/chroms",
    seq_len=600,
    window_size=20,
    stride=20,
    device="cuda",
    n_regions=200,
)
```

### Option 2: CLI Script

```bash
python scripts/run_single_isa.py \
    --prefilter motif_filter \
    --jaspar data/JASPAR2026_CORE_non-redundant_pfms_jaspar.txt \
    --fasta_dir /path/to/hg38/chroms \
    --regions data/regions_pos_with_count.csv \
    --model data/model_blympho.pt \
    --n_regions 20 \
    --device cuda \
    --outdir results
```

**Expected output:**
```
T_sum=<float>
T_peak=<float>
pass_rate=<float>   (should be 1%–80%)
```

### Option 3: Notebook Tutorial

```bash
cd tutorials/
jupyter notebook motif_filter_tutorial.ipynb
```

## Method Overview

```
Input: DNA sequence (600bp) + JASPAR motif hits
DeepLIFT attribution (baseline=zeros) → per-position |attr|
Non-motif 20bp windows → empirical null distribution
s = Σ|attr| (energy), p = max|attr| (peak)
T_sum = P95(s_null), T_peak = P80(p_null)
Keep if: s > T_sum OR p > T_peak
Output: Filtered motif hits → ISA
```

## Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--window_size` | 20 | Non-motif window size (bp) |
| `--stride` | 20 | Non-motif window stride (bp) |
| `--seq_len` | 600 | Region length (bp) |
| `--n_regions` | all | Limit number of regions processed |

## Output Files

After running:

```
results/
├── motif_locs_raw.csv        ← all JASPAR hits
├── motif_locs_filtered.csv  ← motif_filter output (passed motifs)
└── isa_single.csv           ← ISA scores
```
