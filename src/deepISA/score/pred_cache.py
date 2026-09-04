# PredCache — unified prediction cache for combi ISA


import pandas as pd 
from dataclasses import dataclass, field
import numpy as np
from loguru import logger

from deepISA.model.predict import compute_predictions

from deepISA.score.utils_isa import destroy_motifs, region_str_to_seq



@dataclass
class PredCache:
    """
    Unified cache for all model predictions needed in combi ISA.

    Key schemas
    -----------
    pred_orig  : region_str
    isa1/isa2  : (region_str, frozenset([(start, end)]), mut_motif_seq)
    isa_both   : (region_str, frozenset([(s1,e1),(s2,e2)]), mut_motif1_seq, mut_motif2_seq)
    """
    _store: dict = field(default_factory=dict)

    # ------------------------------------------------------------------
    # key builders
    # ------------------------------------------------------------------
    @staticmethod
    def orig_key(region_str: str) -> str:
        return region_str

    @staticmethod
    def single_key(region_str: str, start: int, end: int, mut_motif: str) -> tuple:
        return (region_str, frozenset([(start, end)]), mut_motif)

    @staticmethod
    def both_key(
        region_str: str,
        start1: int, end1: int, mut_motif1: str,
        start2: int, end2: int, mut_motif2: str,
    ) -> tuple:
        # order the two motifs by position so key is canonical
        if start1 <= start2:
            return (region_str, frozenset([(start1,end1),(start2,end2)]), mut_motif1, mut_motif2)
        else:
            return (region_str, frozenset([(start1,end1),(start2,end2)]), mut_motif2, mut_motif1)

    # ------------------------------------------------------------------
    # read / write
    # ------------------------------------------------------------------
    def has(self, key) -> bool:
        return key in self._store

    def get(self, key) -> np.ndarray:
        return self._store[key]

    def set(self, key, value: np.ndarray):
        self._store[key] = value
        
    # ---------------------------------------
    # Load cache
    # ---------------------------------------
    def load_pred_orig(self, df: pd.DataFrame, tracks: list):
        for _, r in df.iterrows():
            key = self.orig_key(r["region"])
            if not self.has(key):
                self.set(key, np.array([r[f"pred_t{t}"] for t in tracks], dtype=float))
        logger.info(f"Loaded pred_orig for {len(df)} regions into cache")
            
        
    def load_single_isa(self, df_single_isa: pd.DataFrame, tracks: list):
        """
        Load single-motif predictions directly from df_single_isa rows into cache.
        Required cols: region, start_rel, end_rel, motif_mut, pred_mut_t{t}
        """
        loaded = 0
        for row in df_single_isa.itertuples():
            key = self.single_key(row.region, row.start_rel, row.end_rel, row.motif_mut)
            if not self.has(key):
                pred_mut = np.array(
                    [getattr(row, f"pred_mut_t{t}") for t in tracks],
                    dtype=float,
                )
                self.set(key, pred_mut)
                loaded += 1
        logger.info(f"Loaded {loaded} single-motif pred entries from df_single_isa")

    # ------------------------------------------------------------------
    # derived ISA values  (all return np.ndarray shape (n_tracks,))
    # ------------------------------------------------------------------
    def isa1(self, region_str, start1, end1, mut_motif1) -> np.ndarray:
        return (
            self.get(self.orig_key(region_str))
            - self.get(self.single_key(region_str, start1, end1, mut_motif1))
        )

    def isa2(self, region_str, start2, end2, mut_motif2) -> np.ndarray:
        return (
            self.get(self.orig_key(region_str))
            - self.get(self.single_key(region_str, start2, end2, mut_motif2))
        )

    def isa_both(
        self, region_str,
        start1, end1, mut_motif1,
        start2, end2, mut_motif2,
    ) -> np.ndarray:
        return (
            self.get(self.orig_key(region_str))
            - self.get(self.both_key(region_str, start1, end1, mut_motif1, start2, end2, mut_motif2))
        )

    def interaction(
        self, region_str,
        start1, end1, mut_motif1,
        start2, end2, mut_motif2,
    ) -> np.ndarray:
        return (
            self.isa1(region_str, start1, end1, mut_motif1)
            + self.isa2(region_str, start2, end2, mut_motif2)
            - self.isa_both(region_str, start1, end1, mut_motif1, start2, end2, mut_motif2)
        )
        
        
        
        
        
        
# -------------------------------------------------------------------------
# Helpers to build the four sequence lists (one per GPU pass)
# -------------------------------------------------------------------------

def _get_seq(region_str, seq_ref, fasta):
    """Return reference sequence for a region."""
    if seq_ref is not None:
        return seq_ref
    return region_str_to_seq(fasta, region_str)


def _mut_motif(seq: str, start: int, end: int, mut_motif: str) -> str:
    """Reconstruct the full mutated sequence from its motif replacement."""
    return seq[:start] + mut_motif + seq[end:]


def _collect_pass_orig(pairs_by_region, cache, fasta):
    """
    Pass 0 — pred_orig.
    One sequence per region not already in cache.
    Returns: keys[], seqs[]
    """
    keys, seqs = [], []
    for region_str, (pair_df, seq_ref) in pairs_by_region.items():
        key = PredCache.orig_key(region_str)
        if cache.has(key):
            continue
        seq = _get_seq(region_str, seq_ref, fasta)
        keys.append(key)
        seqs.append(seq)
    return keys, seqs


def _collect_pass_single(
    pairs_by_region, cache, fasta, motif_num,
    destroy_mode, n_shuffles,
    sa_cols_present=False,
    cached_single_mut_map=None,
):
    start_col = f"start{motif_num}_rel"
    end_col   = f"end{motif_num}_rel"
    seen_intervals = set()
    keys, seqs = [], []
    # (region_str, start, end) → list[mut_motif_seq]  (one per shuffle replicate)
    single_mut_map: dict[tuple, list[str]] = {}

    for region_str, (pair_df, seq_ref) in pairs_by_region.items():
        seq = _get_seq(region_str, seq_ref, fasta)
        for row in pair_df.itertuples():
            start = getattr(row, start_col)
            end   = getattr(row, end_col)
            interval_id = (region_str, start, end)
            if interval_id in seen_intervals:
                continue
            seen_intervals.add(interval_id)
            mut_motifs_for_interval = (
                cached_single_mut_map.get(interval_id)
                if cached_single_mut_map is not None
                else None
            )
            if mut_motifs_for_interval is None:
                mut_motifs_for_interval = [
                    ms[start:end]
                    for ms in destroy_motifs(
                        seq, [start], [end], mode=destroy_mode, n=n_shuffles
                    )
                ]
            for mut_motif in list(mut_motifs_for_interval):
                ms = _mut_motif(seq, start, end, mut_motif)
                key = PredCache.single_key(region_str, start, end, mut_motif)
                if not cache.has(key):
                    keys.append(key)
                    seqs.append(ms)
            single_mut_map[interval_id] = mut_motifs_for_interval

    return keys, seqs, single_mut_map



def _collect_pass_both(
    pairs_by_region, cache, fasta,
    destroy_mode, n_shuffles,
):
    """
    Pass 3 — double-motif destruction.
    Returns: keys[], seqs[], combi_mut_map
    combi_mut_map: (region_str, start1, end1, start2, end2) -> list[both_key]
                   one entry per unique pair position, values are per-shuffle cache keys.
    """
    keys, seqs = [], []
    combi_mut_map: dict[tuple, list[tuple[str, str]]] = {}
    # key:   (region_str, s1, e1, s2, e2)
    # value: [(mut_motif1_shuf1, mut_motif2_shuf1), (mut_motif1_shuf2, mut_motif2_shuf2), ...]

    for region_str, (pair_df, seq_ref) in pairs_by_region.items():
        seq = _get_seq(region_str, seq_ref, fasta)
        for row in pair_df.itertuples():
            mut_seqs = destroy_motifs(
                seq,
                [row.start1_rel, row.start2_rel],
                [row.end1_rel,   row.end2_rel],
                mode=destroy_mode, n=n_shuffles,
            )
            pair_coord_key = (region_str, row.start1_rel, row.end1_rel, row.start2_rel, row.end2_rel)
            row_keys = []
            for ms in mut_seqs:
                mut_motif1 = ms[row.start1_rel:row.end1_rel]
                mut_motif2 = ms[row.start2_rel:row.end2_rel]
                key = PredCache.both_key(
                    region_str,
                    row.start1_rel, row.end1_rel, mut_motif1,
                    row.start2_rel, row.end2_rel, mut_motif2,
                )
                if not cache.has(key):
                    keys.append(key)
                    seqs.append(ms)
                row_keys.append((mut_motif1, mut_motif2))
            combi_mut_map[pair_coord_key] = row_keys

    return keys, seqs, combi_mut_map



# -------------------------------------------------------------------------
# Single GPU pass: run model, average shuffles, write into cache
# -------------------------------------------------------------------------

def _run_gpu_pass(model, device, tracks, seqs, keys, cache, pred_batch_size, pass_name):
    if not seqs:
        logger.info(f"Compute prediction for {pass_name}: nothing to compute, skipping.")
        return
    logger.info(f"Compute prediction for {pass_name}: {len(seqs)} sequences")
    preds = compute_predictions(
        model, seqs, device=device, batch_size=pred_batch_size, tracks=tracks
    )  # shape (len(seqs), n_tracks)
    for i, key in enumerate(keys):
        cache.set(key, preds[i])