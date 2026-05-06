# Alpha-Genome backend (optional — requires: pip install alphagenome pyyaml)
try:
    from deepISA.modeling.alpha_genome_adapter import AlphaGenomeAdapter
except ImportError:
    pass   # alphagenome not installed; ConvModel path unaffected
