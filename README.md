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

## Installation

```bash
pip install -r requirements.txt
```

**External dependency (samtools):**

The genome setup script requires `samtools` to index and split hg38.

```bash
# Linux
sudo apt install samtools

# Mac
brew install samtools
```

Or set the `SAMTOOLS` environment variable to the binary path:

```bash
export SAMTOOLS=/path/to/samtools
```

## DeepISA Environment Binding (STRICT)

**This project is STRICTLY bound to the validated DeepISA environment.**
Any version deviation may change scientific results (attribution scores, thresholds, ISA output).

### Installation (Strict — Required)

```bash
# 1. Install samtools (Linux)
sudo apt install samtools

# 2. Install STRICT pinned dependencies (REQUIRED)
make lock-install

# 3. Verify environment matches DeepISA
make verify-env

# 4. Setup genome + run test
make setup
make test
```

### Why strict pinning?

- `numpy 2.x` is **binary-incompatible** with pandas/pyarrow → silent crash
- `torch` version changes model weight behavior
- `captum` version changes attribution algorithm
- Scientific reproducibility requires exact package versions

### Environment validator

The guard runs automatically before any computation:

- `run_pipeline()` API → validates before processing
- CLI `run_filter.py` → validates before loading model
- Notebook → validates in setup cell

To manually validate:

```bash
PYTHONPATH=src python -c "from deepISA.utils.deepisa_guard import validate_deepisa_environment; validate_deepisa_environment()"
```

### Docker

```bash
docker build -t deepisa_filter .
docker run --rm deepisa_filter
```

## Genome Data

hg38 must be downloaded separately (required by the pipeline).

Download:
```bash
wget https://hgdownload.soe.ucsc.edu/goldenPath/hg38/bigZips/hg38.fa.gz
gunzip hg38.fa.gz
```

Split into per-chromosome FASTA files (required format):
```bash
mkdir -p chroms
samtools faidx hg38.fa
awk '{print $1}' hg38.fa.fai | while read chr; do
    samtools faidx hg38.fa "$chr" > chroms/${chr}.fa
done
ls chroms/chr1.fa   # verify
```

Set the genome path via environment variable or update your code:
```python
GENOME_DIR = "/path/to/hg38/chroms"   # or set HG38_DIR env var
```

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
    --fasta-dir /path/to/hg38/chroms \
    --regions data/regions_pos_with_count.csv \
    --model data/model_blympho.pt \
    --n-regions 20 \
    --device cuda \
    --outdir results
```

**Expected output:**
```
T_sum=<float>
T_peak=<float>
pass_rate=<float>   (should be 1%–80%)
```

### Option 3: Standalone Filter (no ISA)

Run only the motif filter, without ISA:

```bash
python -m deepISA.motif_filter.cli.run_filter \
    --model data/model_blympho.pt \
    --fasta-dir /path/to/hg38/chroms \
    --regions data/regions_pos_with_count.csv \
    --motifs data/motif_locs.csv \
    --out filtered.csv
```

### Option 4: Notebook Tutorial

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

## Known Issues

- `run_single_isa.py` depends on upstream deepISA annotation code which
  incorrectly calls `pyBigWig.open()` on a JASPAR text file.
  This is unrelated to the motif_filter module.
- The motif_filter pipeline and tutorial notebook run correctly as standalone.

## CPU Usage

On machines without a GPU, add `--device cpu` to any command:

```bash
python scripts/run_single_isa.py ... --device cpu
python -m deepISA.motif_filter.cli.run_filter ... --device cpu
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

---

## Developer Guide

### Project Structure

```
src/deepISA/
├── motif_filter/
│   ├── cli/
│   │   ├── common_args.py    ← shared CLI argument definitions
│   │   └── run_filter.py     ← standalone filter CLI
│   ├── core/
│   │   ├── attribution.py   ← DeepLIFT computation
│   │   ├── scoring.py       ← s (energy) and p (peak) scoring
│   │   ├── threshold.py    ← T_sum / T_peak threshold logic
│   │   └── window.py       ← non-motif window generation
│   ├── pipeline/
│   │   ├── filter_pipeline.py  ← main pipeline (internal)
│   │   └── api.py              ← high-level API for developers
│   ├── utils/
│   │   ├── genome_setup.py    ← hg38 auto-download + split
│   │   └── config_loader.py   ← YAML config loader
│   └── utils/                 ← io, fasta, onehot utilities
│
├── scoring/                    ← ISA scoring (upstream)
├── utils/
│   ├── genome_setup.py
│   └── config_loader.py
│
├── config/
│   └── default_config.yaml    ← default parameters
│
└── scripts/
    └── run_single_isa.py      ← CLI: prefilter → ISA
```

### Python API

Minimal usage:

```python
from deepISA.motif_filter.pipeline.api import run_pipeline

result = run_pipeline(
    fasta_dir="/path/to/hg38/chroms",
    regions_df="data/regions_pos_with_count.csv",
    motif_locs_df="data/motif_locs.csv",
    model_path="data/model_blympho.pt",
    device="cuda",
    n_regions=200,
)
```

Using regions directly:

```python
import pandas as pd
from deepISA.motif_filter.pipeline.api import run_pipeline

regions_df = pd.read_csv("data/regions_pos_with_count.csv")
motifs_df  = pd.read_csv("data/motif_locs.csv")

result, model = run_pipeline(
    fasta_dir="/path/to/hg38/chroms",
    regions_df=regions_df,
    motif_locs_df=motifs_df,
    device="cpu",
    return_model=True,
)
```

### Configuration File

Parameters can be overridden via YAML:

```python
from deepISA.utils.config_loader import load_config

cfg = load_config("config/default_config.yaml")
device = cfg["runtime"]["device"]
```

### Auto-download Genome

```python
from deepISA.utils.genome_setup import ensure_hg38

genome_dir = ensure_hg38("./data/genome")  # downloads hg38 if missing
# Returns: "./data/genome/hg38/chroms"
```

### CLI (unified interface)

All scripts now use consistent kebab-case arguments:

```bash
# Standalone filter
python -m deepISA.motif_filter.cli.run_filter \
    --fasta-dir /path/to/hg38/chroms \
    --regions data/regions_pos_with_count.csv \
    --motifs data/motif_locs.csv \
    --model data/model_blympho.pt \
    --out filtered.csv

# Full pipeline (prefilter + ISA)
python scripts/run_single_isa.py \
    --prefilter motif_filter \
    --jaspar data/JASPAR2026_CORE_non-redundant_pfms_jaspar.txt \
    --fasta-dir /path/to/hg38/chroms \
    --regions data/regions_pos_with_count.csv \
    --model data/model_blympho.pt \
    --device cpu \
    --outdir results
```
