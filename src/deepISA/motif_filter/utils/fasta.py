"""FASTA utilities."""
import bioframe as bf
from pathlib import Path
from typing import Optional, Union


class FastaReader:
    """Lightweight FASTA reader using bioframe."""

    def __init__(self, fasta_dir: Union[str, Path]):
        self.fasta_dir = Path(fasta_dir)
        fasta_files = list(self.fasta_dir.glob("chr*.fa"))
        if not fasta_files:
            raise FileNotFoundError(
                f"No FASTA files matching 'chr*.fa' found in {self.fasta_dir}"
            )
        self._fasta = bf.load_fasta(fasta_files)

    def fetch(self, chrom: str, start: int, end: int) -> Optional[str]:
        """Fetch uppercase sequence from FASTA, or None on failure."""
        try:
            return str(self._fasta[chrom][start:end]).upper()
        except Exception:
            return None
