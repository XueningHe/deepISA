#!/usr/bin/env python3
from __future__ import annotations

import itertools
import multiprocessing as mp
from pathlib import Path

from deepISA.scoring.combi_isa import calc_coop_score


BASE_DIR = Path("/maps/projects/ralab/people/pcr980/DeepCompare/D8_main_reproduce2")

CELLTYPE_TRACKS = {
    "hepg2": [0, 2, 4, 6],
    "k562": [1, 3, 5, 7],
}

LEVEL_TO_OUTNAME = {
    "tf_pair": "coop_tf_pair_t{track}.csv",
    "tf": "coop_tf_t{track}.csv",
}


def build_jobs():
    jobs = []

    for celltype, tracks in CELLTYPE_TRACKS.items():
        results_dir = BASE_DIR / f"results_cres_{celltype}"
        if not results_dir.exists():
            print(f"[WARN] Missing results dir: {results_dir}")
            continue

        for subdir in sorted(results_dir.iterdir()):
            data_dir = subdir / "Data"
            if not data_dir.is_dir():
                continue

            combi_isa_path = data_dir / "motif_combi_isa.csv"
            null_isa_path = data_dir / "non_motif_combi_isa.csv"

            if not combi_isa_path.exists():
                print(f"[WARN] Missing combi file: {combi_isa_path}")
                continue
            if not null_isa_path.exists():
                print(f"[WARN] Missing null file: {null_isa_path}")
                continue

            for track, level in itertools.product(tracks, ["tf_pair", "tf"]):
                outpath = data_dir / LEVEL_TO_OUTNAME[level].format(track=track)
                jobs.append(
                    {
                        "combi_isa_path": str(combi_isa_path),
                        "null_isa_path": str(null_isa_path),
                        "outpath": str(outpath),
                        "level": level,
                        "track_idx": track,
                    }
                )

    return jobs


def run_one(job):
    print(
        f"[RUN] level={job['level']:<7} "
        f"track={job['track_idx']} "
        f"out={job['outpath']}"
    )
    calc_coop_score(
        combi_isa_path=job["combi_isa_path"],
        null_isa_path=job["null_isa_path"],
        outpath=job["outpath"],
        level=job["level"],
        track_idx=job["track_idx"],
    )
    return job["outpath"]


def main():
    jobs = build_jobs()
    print(f"[INFO] Total jobs: {len(jobs)}")

    if not jobs:
        print("[INFO] No jobs found.")
        return

    nproc = min(len(jobs), mp.cpu_count())
    print(f"[INFO] Using {nproc} processes")

    with mp.Pool(processes=nproc) as pool:
        results = pool.map(run_one, jobs)

    print(f"[INFO] Finished {len(results)} jobs")


if __name__ == "__main__":
    main()