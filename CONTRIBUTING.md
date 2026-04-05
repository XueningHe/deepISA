# Contributing to deepISA_filter

## Getting Started

```bash
git clone https://github.com/JoneSu1/deepISA_filter.git
cd deepISA_filter
make setup    # install deps + download hg38
make test    # verify with 10 regions on CPU
```

## Development Setup

```bash
pip install -r requirements.txt
export HG38_DIR=./data/genome/hg38/chroms   # or use ensure_hg38()
```

## Making Changes

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/my-feature`
3. Make changes and test: `make test`
4. Commit with clear messages (no AI-style templates)
5. Push and open a pull request

## Project Structure

```
src/deepISA/motif_filter/
├── cli/          # CLI entry points
├── core/         # attribution, scoring, threshold, window
├── pipeline/     # filter_pipeline.py + api.py
└── utils/        # io, fasta, onehot utilities
```

## Code Style

- Use kebab-case for CLI arguments
- docstrings: concise, no "Step N" patterns
- Type hints preferred for public functions

## Testing

```bash
# Unit test CLI
PYTHONPATH=src python -m deepISA.motif_filter.cli.run_filter \
    --fasta-dir ./data/genome/hg38/chroms \
    --regions data/regions_pos_with_count.csv \
    --motifs data/motif_locs.csv \
    --model data/model_blympho.pt \
    --out test_out.csv --n-regions 5 --device cpu

# Full pipeline
make run-isa
```
