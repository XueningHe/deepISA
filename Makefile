.PHONY: help setup install test clean run-filter run-isa

PYTHON := python3
PIP := pip
SAMTOOLS := $(shell command -v samtools 2>/dev/null || echo "")

# ── Helpers ────────────────────────────────────────────────────────────
help:
	@echo "deepISA_filter Makefile"
	@echo ""
	@echo "  make setup          Install Python deps, check samtools, download hg38"
	@echo "  make install        Install Python dependencies only"
	@echo "  make test          Run motif_filter CLI with 10 regions (cpu)"
	@echo "  make test-notebook Run the tutorial notebook (requires jupyter)"
	@echo "  make clean         Remove results/ and filtered outputs"
	@echo "  make check-samtools Check if samtools is installed"

# ── Install ─────────────────────────────────────────────────────────
install:
	@echo "[install] Installing Python dependencies ..."
	$(PIP) install -r requirements.txt
	@echo "[install] Done."

# ── Check samtools ───────────────────────────────────────────────────
check-samtools:
ifneq ($(SAMTOOLS),)
	@echo "[samtools] Found: $(SAMTOOLS)"
else
	@echo "[samtools] NOT found. Install with: sudo apt install samtools"
	@exit 1
endif

# ── Setup (all) ─────────────────────────────────────────────────────
setup: check-samtools install
	@echo "[setup] Downloading and setting up hg38 ..."
	$(PYTHON) -m deepISA.utils.genome_setup \
		--genome-dir ./data/genome
	@echo "[setup] Done. hg38 ready at: ./data/genome/hg38/chroms"

# ── Test (CLI) ─────────────────────────────────────────────────────
test: check-samtools
	@echo "[test] Running motif_filter on 10 regions (cpu) ..."
	PYTHONPATH=src $(PYTHON) -m deepISA.motif_filter.cli.run_filter \
		--fasta-dir ./data/genome/hg38/chroms \
		--regions data/regions_pos_with_count.csv \
		--motifs data/motif_locs.csv \
		--model data/model_blympho.pt \
		--out results_test/motif_locs_filtered.csv \
		--n-regions 10 \
		--device cpu \
		--report results_test/report.json
	@echo "[test] Results → results_test/"

# ── Test notebook ───────────────────────────────────────────────────
test-notebook:
	@echo "[test-notebook] Ensure jupyter and the deepisa_test kernel are available."
	@echo "Then run: HG38_DIR=./data/genome/hg38/chroms jupyter notebook tutorials/motif_filter_tutorial.ipynb"
	$(PYTHON) -m jupyter --version || echo "[test-notebook] jupyter not installed. Run: pip install jupyter nbconvert"

# ── Run standalone filter ───────────────────────────────────────────
run-filter:
	@echo "[run-filter] Starting motif_filter CLI ..."
	PYTHONPATH=src $(PYTHON) -m deepISA.motif_filter.cli.run_filter \
		--fasta-dir ./data/genome/hg38/chroms \
		--regions data/regions_pos_with_count.csv \
		--motifs data/motif_locs.csv \
		--model data/model_blympho.pt \
		--out results/motif_locs_filtered.csv \
		--device cpu

# ── Run full ISA pipeline ──────────────────────────────────────────
run-isa: check-samtools
	@echo "[run-isa] Starting full pipeline (prefilter + ISA) ..."
	PYTHONPATH=src $(PYTHON) scripts/run_single_isa.py \
		--prefilter motif_filter \
		--jaspar data/JASPAR2026_CORE_non-redundant_pfms_jaspar.txt \
		--fasta-dir ./data/genome/hg38/chroms \
		--regions data/regions_pos_with_count.csv \
		--model data/model_blympho.pt \
		--device cpu \
		--outdir results

# ── Clean ────────────────────────────────────────────────────────────
clean:
	@echo "[clean] Removing results/ ..."
	rm -rf results/ results_test/ motif_locs_filtered.csv
	@echo "[clean] Done."
