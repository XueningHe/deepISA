
import numpy as np
import pandas as pd
from loguru import logger
from deepISA.model.predict import compute_predictions

from deepISA.utils import remove_if_exists, write_stream_csv
from deepISA.model.predict import compute_predictions
from deepISA.utils import (
    remove_if_exists,
    write_stream_csv,
)

from deepISA.score.utils_isa import (
    destroy_motifs, 
    region_str_to_seq
)

from deepISA.score.pred_cache import (
    PredCache,
    _collect_pass_orig,
    _collect_pass_single,
    _collect_pass_both,
    _run_gpu_pass,
)


def load_pred_orig(region_orig_pred_path, tracks):
    df = pd.read_csv(region_orig_pred_path)
    needed = ["region"] + [f"pred_t{t}" for t in tracks]
    return {
        r["region"]: np.array([r[f"pred_t{t}"] for t in tracks], dtype=float)
        for _, r in df[needed].iterrows()
    }



def single_isa_core(
    model,
    fasta,
    locs_df,
    pred_orig_path,
    outpath,
    device,
    tracks,
    num_regions_per_batch,
    pred_batch_size,
    destroy_mode="ablate",   
    n_shuffles=8,
):
    remove_if_exists(outpath, label="single ISA file")
    logger.info(f"Single ISA started. Total rows: {len(locs_df)}, destroy_mode={destroy_mode}")
    orig_pred_map = load_pred_orig(pred_orig_path, tracks)
    region_groups = list(locs_df.groupby("region"))

    for i in range(0, len(region_groups), num_regions_per_batch):
        batch = region_groups[i : i + num_regions_per_batch]
        batch_rows = []
        for region_str, group in batch:
            seq_orig = region_str_to_seq(fasta, region_str)
            for _, row in group.iterrows():
                mut_seqs = destroy_motifs(
                    seq_orig,
                    [int(row["start_rel"])],
                    [int(row["end_rel"])],
                    mode=destroy_mode,
                    n=n_shuffles,
                )
                batch_rows.append((row, mut_seqs))

        # Flatten: one entry per mutant sequence
        row_indices = []
        flat_seqs = []
        for idx, (row, mut_seqs) in enumerate(batch_rows):
            row_indices.extend([idx] * len(mut_seqs))
            flat_seqs.extend(mut_seqs)

        row_indices = np.array(row_indices)

        # Single batched GPU call
        preds_mut_flat = compute_predictions(
            model, flat_seqs, device, pred_batch_size, tracks=tracks
        )  # shape: (total_mutants, len(tracks))

        # Average mutant predictions back to one value per row
        n_rows = len(batch_rows)
        preds_mut_mean = np.zeros((n_rows, len(tracks)), dtype=np.float32)
        counts = np.zeros(n_rows, dtype=np.int32)
        for flat_i, row_i in enumerate(row_indices):
            preds_mut_mean[row_i] += preds_mut_flat[flat_i]
            counts[row_i] += 1
        preds_mut_mean /= counts[:, None]

        # ── Reconstruct DataFrame ─────────────────────────────────────
        if destroy_mode == "ablate":
            # One mutant per row — save motif_mut and pred_mut_t{t} for combi ISA cache reuse
            current_df_rows = []
            for flat_i, row_i in enumerate(row_indices):
                row, _ = batch_rows[row_i]
                start, end = int(row["start_rel"]), int(row["end_rel"])
                new_row = row.to_dict()
                new_row["motif_mut"] = flat_seqs[flat_i][start:end]
                pred_orig = orig_pred_map[row["region"]]
                for j, t in enumerate(tracks):
                    new_row[f"pred_mut_t{t}"] = preds_mut_flat[flat_i, j]
                    new_row[f"isa_t{t}"] = pred_orig[j] - preds_mut_flat[flat_i, j]
                current_df_rows.append(new_row)
            current_df = pd.DataFrame(current_df_rows)

        else:  # dinuc_shuffle
            # Multiple shuffles per row — average them, don't save motif_mut or pred_mut
            current_df = pd.DataFrame([row for row, _ in batch_rows]).reset_index(drop=True)
            preds_orig_sel = np.vstack([orig_pred_map[row["region"]] for row, _ in batch_rows])
            for j, t in enumerate(tracks):
                current_df[f"isa_t{t}"] = preds_orig_sel[:, j] - preds_mut_mean[:, j]

        write_stream_csv(current_df, outpath)





def combi_isa_core(
    model,
    device,
    tracks,
    fasta,
    batch_pairs,
    pred_batch_size,
    outpath,
    cache,
    single_mut_map,
    destroy_mode,
    n_shuffles,
):
    # ── Pass 0: pred_orig ─────────────────────────────────────────────
    keys0, seqs0 = _collect_pass_orig(batch_pairs, cache, fasta)
    _run_gpu_pass(model, device, tracks, seqs0, keys0, cache, pred_batch_size, "0-orig")

    # ── Pass 1 & 2: single-motif ablation ────────────────────────────
    keys1, seqs1, single_mut_map1 = _collect_pass_single(
        batch_pairs, cache, fasta, motif_num=1,
        destroy_mode=destroy_mode, n_shuffles=n_shuffles
    )
    _run_gpu_pass(model, device, tracks, seqs1, keys1, cache, pred_batch_size, "1-single-motif1")
    keys2, seqs2, single_mut_map2 = _collect_pass_single(
        batch_pairs, cache, fasta, motif_num=2,
        destroy_mode=destroy_mode, n_shuffles=n_shuffles,
    )
    _run_gpu_pass(model, device, tracks, seqs2, keys2, cache, pred_batch_size, "2-single-motif2")
    
    if single_mut_map is None:
        single_mut_map = {**single_mut_map1, **single_mut_map2}
    else:
        single_mut_map = {**single_mut_map, **single_mut_map1, **single_mut_map2}

    # ── Pass 3: isa_both ──────────────────────────────────────────────
    keys3, seqs3, combi_mut_map = _collect_pass_both(
        batch_pairs, cache, fasta,
        destroy_mode=destroy_mode, n_shuffles=n_shuffles,
    )
    _run_gpu_pass(model, device, tracks, seqs3, keys3, cache, pred_batch_size, "3-both")

    # ── Assemble results ──────────────────────────────────────────────
    for region_str, (pair_df, _) in batch_pairs.items():
        if pair_df is None or pair_df.empty:
            continue
        pair_df = pair_df.copy()

        for t in tracks:
            isa1_vals     = []
            isa2_vals     = []
            isa_both_vals = []

            for row in pair_df.itertuples():
                # isa1
                mut_motifs1 = single_mut_map[(region_str, row.start1_rel, row.end1_rel)]
                i1 = np.mean([
                    cache.isa1(region_str, row.start1_rel, row.end1_rel, mm)[tracks.index(t)]
                    for mm in mut_motifs1
                ])
                # isa2
                mut_motifs2 = single_mut_map[(region_str, row.start2_rel, row.end2_rel)]
                i2 = np.mean([
                    cache.isa2(region_str, row.start2_rel, row.end2_rel, mm)[tracks.index(t)]
                    for mm in mut_motifs2
                ])
                # isa_both
                bkeys = combi_mut_map[(region_str, row.start1_rel, row.end1_rel, row.start2_rel, row.end2_rel)]
                ib = np.mean([cache.isa_both(
                                region_str,
                                row.start1_rel, row.end1_rel, mut_seq1,
                                row.start2_rel, row.end2_rel, mut_seq2,
                                )[tracks.index(t)]
                                for mut_seq1, mut_seq2 in bkeys
                            ])

                isa1_vals.append(i1)
                isa2_vals.append(i2)
                isa_both_vals.append(ib)

            pair_df[f"isa1_t{t}"]     = isa1_vals
            pair_df[f"isa2_t{t}"]     = isa2_vals
            pair_df[f"isa_both_t{t}"] = isa_both_vals

        write_stream_csv(pair_df, outpath)

