# Optional AlphaGenome API backend (pip install -e ".[alphagenome]").
# Kept in a try/except so the default Conv-based path works without the extra.
try:
    from deepISA.model.alpha_genome_adapter import AlphaGenomeAdapter
except ImportError:
    pass
