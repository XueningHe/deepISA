# Non-Motif Null Attribution Filter

Removes motif hits the deepISA model does not attend to, using DeepLIFT
attribution against an empirical null from non-motif windows within the same regions.

---

## Pipeline

### 1. DeepLIFT Attribution (baseline=zeros)

For each 600bp region:

```
one-hot DNA (1, 600, 4), baseline = zeros
attr = DeepLIFT(model, input_onehot, baseline=zeros)
```

`baseline=zeros` treats all nucleotides equally.

### 2. Non-Motif Window Generation

Non-overlapping 20bp windows covering the region, excluding any JASPAR-annotated
motif interval:

```
non_motif_windows = [(region_start + i*20, region_start + (i+1)*20) for i in ...]
```

Only windows fully free of motif overlap are collected.

### 3. Score Computation

For each window:

```
abs_attr = |attr|
per_pos  = abs_attr.sum(axis=1)   # sum over ACGT
s_motif  = per_pos.sum()          # energy
p_max    = per_pos.max()           # peak
```

`s_motif` (energy) captures distributed attribution; `p_max` (peak) captures
localized spikes.

### 4. Null Distribution & Thresholds

Collect scores from all non-motif windows across all regions:

```
T_sum  = P95(s_motif over non-motif windows)  # energy gate
T_peak = P80(p_max   over non-motif windows)  # peak gate
```

### 5. Dual-Gate OR Filter

Keep motif if **either** gate passes:

```
keep if: s_motif > T_sum  OR  p_max > T_peak
```

The peak gate rescues motifs with a single strong base contact but low total energy
(e.g., TATA-box-like binding).

---

## Limitations

1. **Non-motif ≠ truly neutral**: windows in functional regulatory regions may
   still carry signal even without known motifs.
2. **Engineering filter, not a statistical test**: do not over-interpret pass/fail.
3. **ISA is the gold standard**: downstream ISA validates causal effect on expression.

---

## Implementation Notes

### Region length
All regions must be exactly 600bp. The pipeline validates this on load and raises
`ValueError` on deviation. For a different region length, pass `seq_len=<N>` to
`run_pipeline()` and adjust `window_size` accordingly.

### Strand handling
Uses forward strand only. Motif strand is recorded in output for downstream
compatibility but is not used to flip the reference sequence. If the model
is strand-sensitive, add reverse-complement augmentation before production use.

### Non-motif window count
Regions with < 10 non-motif windows produce unstable null estimates.
The pipeline logs a warning listing affected region IDs. Prefer regions
with substantial inter-motif spacing (e.g., distal enhancers over dense promoters).

---

## Output Format

```
chrom, start, end, tf, score, strand, region, s_motif, p_max, passed_sum, passed_peak
```

- `start`, `end`: genomic coordinates in [start, end) style
- `passed_sum` / `passed_peak`: boolean filter decisions
