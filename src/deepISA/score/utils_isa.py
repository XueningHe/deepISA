import numpy as np
import pandas as pd
from loguru import logger
import random
from collections import defaultdict, Counter



#---------------------------
# Dinucleotide shuffle utilities
#---------------------------


def _validate_acgt(seq: str, alphabet="ACGT"):
    if not isinstance(seq, str):
        raise TypeError("Sequence must be a string.")
    if len(seq) < 2:
        raise ValueError("Sequence must have length >= 2.")
    bad = [c for c in seq if c not in alphabet]
    if bad:
        raise ValueError(
            f"Sequence contains invalid characters: {sorted(set(bad))}. "
            f"Allowed alphabet: {alphabet}"
        )


def _normalize_start_end(length: int, start: int, end: int):
    """
    Match original semantics:
      - if end < 0: end = length + 1 + end   (so end=-1 => full length)
      - shuffled region is [start:end)
    """
    if end < 0:
        end = length + 1 + end

    if end <= start:
        raise ValueError("End must come after start.")

    if start < 0 or end > length:
        raise ValueError("Start or end are falling off the edge of sequence.")

    return start, end


def _build_successor_lists(seq: str):
    # outgoing edges grouped by source character
    next_idxs = defaultdict(list)
    for i in range(len(seq) - 1):
        next_idxs[seq[i]].append(i + 1)
    return next_idxs


def _single_dinuc_shuffle_region(region: str, rng: random.Random):
    """
    One dinucleotide-preserving shuffle of a region.
    Keeps first and last character of region fixed.
    """
    L = len(region)
    if L < 2:
        return region

    next_idxs = _build_successor_lists(region)

    # Shuffle all but last outgoing edge for each source base
    for ch, lst in next_idxs.items():
        n = len(lst)
        if n > 1:
            prefix = lst[:-1]
            rng.shuffle(prefix)
            next_idxs[ch] = prefix + [lst[-1]]

    used = Counter()
    out = [region[0]]
    idx = 0

    for _ in range(1, L):
        ch = region[idx]
        k = used[ch]
        idx = next_idxs[ch][k]
        used[ch] += 1
        out.append(region[idx])

    return "".join(out)


def dinucleotide_shuffle(
    seq,
    start=0,
    end=-1,
    n=20,
    random_state=None,
    verify=False,
) -> list[str]:
    """
    Dinucleotide shuffle for ACGT strings, optionally on a substring.

    Parameters
    ----------
    seq : str or list[str]
        Input sequence(s), characters in A/C/G/T.
    start : int
        Inclusive start of region to shuffle.
    end : int
        End of region. If end >= 0: non-inclusive [start:end).
        If end < 0: converted as end = len(seq)+1+end (so -1 means full end).
    n : int
        Number of shuffles per sequence.
    random_state : int or None
        Seed for reproducibility.
    verify : bool
        If True, verify dinucleotide counts in shuffled region are preserved.

    Returns
    -------
    list[str] or list[list[str]]
        - input str -> list[str] (length n)
        - input list[str] -> list[list[str]] shape [num_seqs][n]
    """
    if n < 1:
        raise ValueError("n must be >= 1")

    single = isinstance(seq, str)
    seqs = [seq] if single else seq
    if not isinstance(seqs, list) or len(seqs) == 0:
        raise ValueError("seq must be a non-empty string or list of strings.")
    rng = random.Random(random_state)
    def dinuc_counts(s):
        return Counter(s[i:i+2] for i in range(len(s) - 1))
    all_out = []
    for s in seqs:
        _validate_acgt(s)
        st, en = _normalize_start_end(len(s), start, end)
        left = s[:st]
        region = s[st:en]
        right = s[en:]
        if len(region) < 2:
            # nothing meaningful to shuffle; return identical n times
            all_out.append([s for _ in range(n)])
            continue
        target_counts = dinuc_counts(region) if verify else None
        shufs = []
        for _ in range(n):
            reg_shuf = _single_dinuc_shuffle_region(region, rng)
            if verify and dinuc_counts(reg_shuf) != target_counts:
                raise RuntimeError("Dinucleotide counts mismatch in region.")
            shufs.append(left + reg_shuf + right)
        all_out.append(shufs)
    return all_out[0] if single else all_out













def destroy_motifs(seq, motif_starts, motif_ends, mode="ablate", n=4):
    if isinstance(motif_starts, int):
        motif_starts = [motif_starts]
    if isinstance(motif_ends, int):
        motif_ends = [motif_ends]
    if len(motif_starts) != len(motif_ends):
        raise ValueError("motif_starts and motif_ends must have the same length")

    motifs = sorted(zip(motif_starts, motif_ends), key=lambda x: x[0])
    previous_end = 0

    if mode == "ablate":
        for start, end in motifs:
            if start < previous_end:
                logger.warning("Motif overlap detected: motif_starts={}, motif_ends={}", motif_starts, motif_ends)
                continue
            seq = _ablate_motif(seq, start, end)
            previous_end = end
        return [seq]

    elif mode == "dinuc_shuffle":
        seqs = [seq]
        for start, end in motifs:
            if start < previous_end:
                logger.warning("Motif overlap detected: motif_starts={}, motif_ends={}", motif_starts, motif_ends)
                continue
            # TODO: make each shuffled motif unique
            seqs = [s for seq in seqs for s in dinucleotide_shuffle(seq, start=start, end=end, n=n)]
            previous_end = end
        return seqs

    else:
        raise ValueError(f"Unknown mode '{mode}'. Choose 'ablate' or 'dinuc_shuffle'.")











def region_str_to_seq(fasta, region_str: str) -> str:
    chrom, coords = region_str.split(":")
    start_r, end_r = map(int, coords.split("-"))
    return str(fasta[chrom][start_r:end_r]).upper()



def scramble_motif(seq: str, 
                   n: int = 4,
                   random_state=None) -> list[str]:
    bases = list(seq)
    rng = random.Random(random_state)
    pool = []
    for _ in range(n):
        rng.shuffle(bases)
        pool.append("".join(bases))
    return pool


def destroy_motifs(seq, motif_starts, motif_ends, mode="ablate", n=4, random_state=None):
    if isinstance(motif_starts, int):
        motif_starts = [motif_starts]
    if isinstance(motif_ends, int):
        motif_ends = [motif_ends]
    if len(motif_starts) != len(motif_ends):
        raise ValueError("motif_starts and motif_ends must have the same length")

    motifs = sorted(zip(motif_starts, motif_ends), key=lambda x: x[0])
    rng = random.Random(random_state)   

    # Step 1 & 2: carve out each motif and transform it into a pool of variants
    valid_motifs = []
    per_motif_pools = []  # per_motif_pools[i] is a list of transformed motif strings
    previous_end = 0

    for start, end in motifs:
        if start < previous_end:
            logger.warning("Motif overlap detected: motif_starts={}, motif_ends={}", motif_starts, motif_ends)
            continue
        motif_seq = seq[start:end]
        
        # shuffle motifs
        if mode == "ablate":
            pool = ["N" * len(motif_seq)]
        elif mode == "dinuc_shuffle":
            pool = dinucleotide_shuffle(motif_seq, n=n)
        elif mode == "scramble":
            pool = scramble_motif(motif_seq, n=n)
        else:
            raise ValueError(f"Unknown mode '{mode}'. Choose 'ablate', 'dinuc_shuffle', or 'scramble'.")

        valid_motifs.append((start, end))
        per_motif_pools.append(pool)
        previous_end = end

    if not valid_motifs:
        return [seq]

    # Step 3: stitch — randomly draw one variant per motif and splice into original sequence
    n_out = 1 if mode == "ablate" else n
    result = []
    for _ in range(n_out):
        rebuilt = list(seq)
        for (start, end), pool in zip(valid_motifs, per_motif_pools):
            rebuilt[start:end] = list(rng.choice(pool))
        result.append("".join(rebuilt))

    return result


