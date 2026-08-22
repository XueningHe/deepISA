import os
import torch
import numpy as np
import pandas as pd
import json
from loguru import logger

# DeepISA Imports
from deepISA.modeling.cnn import Conv
from deepISA.modeling.preprocess import compile_training_data
from deepISA.modeling.train import train_model
from deepISA.scoring.mapper import map_motifs
from deepISA.scoring.single_isa import (
    run_single_isa, 
    calc_tf_importance
)

from deepISA.scoring.combi_isa import (
    run_combi_isa,
    run_null_interaction,
    add_interaction,
    calc_coop_score
)
from deepISA.utils import setup_logger, find_available_gpu

# Motif discovery (tf-modisco-lite + Fi-NeMo) -- optional, additive API.
from deepISA.scoring.discover import (
    compute_attribution,
    prepare_modisco_input,
    run_modisco,
    build_finemo_input,
    run_finemo_scan,
    load_hits_with_annotation,
    load_motifs,
    run_motif_report,
    cwm_to_meme,
)


# Plotting functions
from deepISA.plotting.interaction import (
    plot_interaction_decay, 
)

from deepISA.plotting.null import (
    plot_null_isa,
    plot_motif_length,
    plot_null_interaction,
    plot_motif_distance
)

from deepISA.plotting.cooperativity import (
    hist_coop_score,
    heatmap_coop_score,
    plot_motif_distance_by_category
)

from deepISA.plotting.tf import (
    plot_motif_gc_by_coop,
    plot_coop_vs_importance,
    plot_partner_specificity
)


# exploring functions
from deepISA.exploring.tf_family import (
    plot_coop_by_tf_pair_family,
    plot_coop_by_dbd,
    plot_intra_family_coop_score
)

from deepISA.exploring.tf_pair_ppi import (
    plot_ppi_enrichment,
    plot_cofactor_recruitment,
    plot_dna_mediated_ppi
)
from deepISA.exploring.tf_function import (
    plot_usf_pfs,
    plot_cell_specificity
)

ISA_STAGES = [
    "map_motifs",
    "single_isa",
    "combi_isa",
    "null_interaction",
    "aggregate_isa",
]

class QuickStart:
    def __init__(self, 
                 results_dir, 
                 fasta_path, 
                 df_input, 
                 device=None):
        # create results directory and subdirectories for data, plots, and models
        self.results_dir = results_dir
        self.data_dir = os.path.join(self.results_dir, "Data")
        self.plots_dir = os.path.join(self.results_dir, "Plots")
        self.model_dir = os.path.join(self.results_dir, "Models")
        setup_logger(self.results_dir)
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(self.plots_dir, exist_ok=True)
        os.makedirs(self.model_dir, exist_ok=True)
        # register paths for key files that will be generated and used throughout the pipeline
        self.files = {
            "motif_locs":    os.path.join(self.data_dir, "motif_locs.csv"),
            "non_motif_locs":os.path.join(self.data_dir, "non_motif_locs.csv"),
            "null_isa":      os.path.join(self.data_dir, "null_isa.csv"),
            "pred_orig": os.path.join(self.data_dir, "pred_orig.csv"),
            "isa_single":    os.path.join(self.data_dir, "motif_single_isa.csv"),
            "isa_combi":     os.path.join(self.data_dir, "motif_combi_isa.csv"),
            "null_interaction":os.path.join(self.data_dir, "null_interaction.csv"),
            "imp_tf":        os.path.join(self.data_dir, "tf_importance.csv"),
            "coop_tf_pair":  os.path.join(self.data_dir, "coop_tf_pair.csv"),
            "coop_tf":       os.path.join(self.data_dir, "coop_tf.csv"),
            # Motif discovery (tf-modisco-lite + Fi-NeMo) -- registered lazily;
            # these keys are populated by run_modisco()/run_finemo() and stay
            # inert unless those methods are called.
            "attr_h5":       os.path.join(self.data_dir, "attribution.h5"),
            "modisco_h5":    os.path.join(self.data_dir, "modisco_results.h5"),
            "discovered_motifs": os.path.join(self.data_dir, "discovered_motifs.csv"),
            "finemo_npz":    os.path.join(self.data_dir, "finemo_input.npz"),
            "finemo_hits":   os.path.join(self.data_dir, "finemo_hits.tsv"),
            "motif_report":  os.path.join(self.results_dir, "MotifReport"),
            "discovered_meme": os.path.join(self.data_dir, "discovered_motifs.meme"),
        }
        self.fasta_path = fasta_path
        self.df_input = df_input # can be either positive or negative regions
        self.df_labeled = None
        self.device = device if device is not None else find_available_gpu()
        

    def define_model(self, 
                     model_config=None,
                     model_obj=None, 
                     mode='dual'):
        """
        Internalizes a model. 
        Pass a pre-instantiated object (e.g. AlphaGenome) OR 
        pass params (ks, cs, ds, seq_len) to build the internal Conv model.
        """
        self.mode = mode
        if model_obj is not None:
            self.model = model_obj.to(self.device)
            logger.info("External model internalized successfully.")
        elif model_config is not None:
            self.model_config = model_config 
            # Build the internal Conv class from provided cnn.py
            self.model = Conv(self.mode, self.model_config).to(self.device) 
            logger.info(f"Internal Conv model initialized. Receptive field: {self.model.rf}")
            with open(os.path.join(self.model_dir, "model_config.json"), 'w') as f:
                json.dump(model_config, f, indent=4)
                

    def train(self, 
              trainer_config=None,
              bw_paths=None,
              target_reg_col="target_reg",
              target_class_col=None):
        """
        Compiles training data and executes the Trainer. 
        The best model is automatically saved by the Trainer class.
        """
        if self.model is None:
            raise ValueError("Model not defined. Call define_model() first.")
        
        if trainer_config is None:
            trainer_config={
                "epochs": 10,
                "batch_size": 128,
                "patience": 3,
                "min_delta": 0.001, # minimum change in the monitored metric to qualify as an improvement
                "learning_rate": 1e-3,
                "save_one_fourth": False,
                "save_one": False,
                "log_transform":True,
                "rc_aug":True
            }

        
        train_data_path = os.path.join(self.data_dir, "Training_data")
        # 1. Compile Data
        logger.info("Compiling training data...")
        self.df_labeled = compile_training_data(
            df=self.df_input,
            fasta_path=self.fasta_path,
            out_dir=train_data_path,
            seq_len=self.model_config['seq_len'],
            bw_paths=bw_paths,
            target_reg_col=target_reg_col,
            target_class_col=target_class_col,
            log_transform=trainer_config.get('log_transform', True),
            rc_aug=trainer_config.get('rc_aug', True)
        )
        
        train_model(
            model=self.model,
            device=self.device,
            train_dat_dir=train_data_path,
            trainer_config=trainer_config,
            mode=self.mode,
            model_dir=self.model_dir, 
        )
        logger.info(f"Training complete. Check {self.model_dir} for model_best.pt")


    def load_checkpoint(self, suffix="best"):
        """Explicitly loads a specific checkpoint into self.model."""
        if self.model is None:
            raise ValueError("Model structure not defined. Call define_model first.")
        filename = f"model{suffix}.pt"
        path = os.path.join(self.model_dir, filename)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Checkpoint {filename} not found at {path}")
        self.model.load_state_dict(torch.load(path, map_location=self.device, weights_only=True))
        logger.info(f"Successfully loaded checkpoint: {filename}")
        return self.model
    
    
    
    def _validate_start_from(self, start_from):
        if start_from not in ISA_STAGES:
            raise ValueError(f"Invalid start_from='{start_from}'. Expected one of: {ISA_STAGES}")
    
    
    def _check_isa_dependencies(self, start_from):
        if start_from == "map_motifs":
            return
        if start_from == "single_isa":
            required = [self.files["motif_locs"], self.files["non_motif_locs"]]  
        elif start_from == "combi_isa":
            required = [self.files["isa_single"], self.files["pred_orig"]]        
        elif start_from == "null_interaction":
            required = [self.files["non_motif_locs"], self.files["isa_combi"], self.files["pred_orig"]]  
        elif start_from == "aggregate_isa":
            required = [self.files["isa_single"], self.files["null_isa"], self.files["isa_combi"], self.files["null_interaction"]]  
            required = []
        missing = [path for path in required if not os.path.exists(path)]
        if missing:
            missing_str = "\n".join([f" - {path}" for path in missing])
            raise FileNotFoundError(
                f"Cannot start from stage '{start_from}' because required files are missing:\n{missing_str}"
            )
    
    
    def run_isa(self, 
            jaspar_path,
            isa_config,
            df_pos=None,
            expressed_tfs=None,
            start_from="map_motifs"):
            
        if self.model is None:
            raise ValueError("Model not defined. Call define_model() first.")

        self._validate_start_from(start_from)
        self._check_isa_dependencies(start_from)

        if df_pos is None:
            if self.df_labeled is not None and 'target_class' in self.df_labeled.columns:
                df_pos = self.df_labeled[self.df_labeled['target_class'] == 1].copy()
                logger.info(f"{len(df_pos)} positive regions identified from input data. Use as input for ISA.")
            else:
                df_pos = self.df_input.copy()
                logger.warning("Using df_input as positives for ISA. Ensure this is intended.")
        else:
            logger.info(f"{len(df_pos)} positive regions provided for ISA.")
                        
        self.tracks = isa_config.get('tracks', [0])
        start_idx = ISA_STAGES.index(start_from)

        # 1. Motif Mapping
        if start_idx <= ISA_STAGES.index("map_motifs"):
            logger.info("Running stage: map_motifs")
            map_motifs(
                regions_df=df_pos,
                jaspar_path=jaspar_path,
                motif_outpath=self.files["motif_locs"],
                non_motif_outpath=self.files["non_motif_locs"],
                motif_score_thresh=isa_config.get('motif_score_threshold', 500),
                remap_path=isa_config.get('remap_path', None),
                expressed_tfs=expressed_tfs
            )

            if isa_config.get('subset_by_remap', False):
                logger.info("Subsetting motif locations by ReMap evidence...")
                df_temp = pd.read_csv(self.files["motif_locs"])
                df_temp = df_temp[df_temp['remap_evidence'] == True].reset_index(drop=True)
                df_temp.to_csv(self.files["motif_locs"], index=False)
        else:
            logger.info(f"Skipping stage: map_motifs (start_from='{start_from}')")

        # 2. Single ISA now writes raw then filters to final motif_single_isa.csv
        if start_idx <= ISA_STAGES.index("single_isa"):
            logger.info("Running stage: single_isa")
            run_single_isa(
                model=self.model,
                fasta=self.fasta_path,  
                motif_locs_path=self.files["motif_locs"],
                non_motif_locs_path=self.files["non_motif_locs"],   
                single_isa_outpath=self.files["isa_single"],         
                null_isa_outpath=self.files["null_isa"], 
                null_percentile=isa_config["null_percentile"],        
                pred_orig_outpath=self.files["pred_orig"],           
                device=self.device,
                tracks=self.tracks,
                num_regions_per_batch=isa_config.get("num_regions_per_batch", 200),
                pred_batch_size=isa_config.get("pred_batch_size", 1024),
            )

        else:
            logger.info(f"Skipping stage: single_isa (start_from='{start_from}')")

        # 4. Combination ISA
        if start_idx <= ISA_STAGES.index("combi_isa"):
            logger.info("Running stage: combi_isa")
            run_combi_isa(
                model=self.model,
                fasta=self.fasta_path,  
                single_isa_path=self.files["isa_single"],  
                outpath=self.files["isa_combi"],
                device=self.device,
                tracks=self.tracks,
                receptive_field=isa_config.get("receptive_field", getattr(self.model, "rf", 255)),
                pred_orig_path=self.files["pred_orig"],     
                num_regions_per_batch=isa_config.get("num_regions_per_batch", 200),
                pred_batch_size=isa_config.get("pred_batch_size", 1024),
            )
        else:
            logger.info(f"Skipping stage: combi_isa (start_from='{start_from}')")

        # 5. Null combination ISA
        if start_idx <= ISA_STAGES.index("null_interaction"):
            logger.info("Running stage: null_interaction")
            run_null_interaction(
                model=self.model,
                fasta=self.fasta_path,  
                non_motif_locs_path=self.files["non_motif_locs"],
                combi_isa_path=self.files["isa_combi"],    
                pred_orig_path=self.files["pred_orig"],
                tracks=self.tracks,    
                outpath=self.files["null_interaction"],
                device=self.device
            )
        else:
            logger.info(f"Skipping stage: null_interaction (start_from='{start_from}')")

        # 6. Aggregate / calculate cooperativity scores
        if start_idx <= ISA_STAGES.index("aggregate_isa"):
            logger.info(f"Running stage: calc_coop_score")
            logger.info(f"Aggregating results in {self.data_dir}")
            add_interaction(
                combi_isa_path=self.files["isa_combi"],
                null_interaction_path=self.files["null_interaction"],
                null_isa_path=self.files["null_isa"],
                tracks=self.tracks,
                null_percentile=isa_config["null_percentile"])
            
            calc_tf_importance(
                self.files["isa_single"],
                self.files["null_isa"],                    
                self.files["imp_tf"],    
                null_percentile=isa_config["null_percentile"],   
                min_count=isa_config.get("min_count", 10),
            )

            for t in self.tracks:
                calc_coop_score(
                    self.files["isa_combi"],               
                    self.files["null_interaction"],   
                    # add track suffix to output paths        
                    outpath=self.files["coop_tf_pair"].replace(".csv", f"_t{t}.csv"),
                    level="tf_pair",
                    track_idx=t,
                    null_percentile=isa_config["null_percentile"],                
                    min_count=isa_config.get("min_count", 10),
                    q_val_thresh=isa_config.get("q_val_thresh", 0.1),
                )
                                
                calc_coop_score(
                    self.files["isa_combi"],             
                    self.files["null_interaction"],            
                    outpath=self.files["coop_tf"].replace(".csv", f"_t{t}.csv"),
                    level="tf",
                    track_idx=t,
                    null_percentile=isa_config["null_percentile"],
                    min_count=isa_config.get("min_count", 10),
                    q_val_thresh=isa_config.get("q_val_thresh", 0.1),
                )
        else:
            logger.info(f"Skipping stage: calc_coop_score (start_from='{start_from}')")

        logger.info("ISA execution and aggregation complete.")
        
    
    def report(self, tracks=None):
        """
        Executes the full suite of visualization and exploration functions.
        """
        if tracks is not None:
            self.tracks = tracks
        if not hasattr(self, "tracks") or self.tracks is None:
            raise ValueError(
                "Tracks are not set. Pass tracks to report(tracks=[...]) "
                "or run run_isa(...) first."
            )
        logger.info("Generating comprehensive reports and plots...")
        
        # --- A. Null Distributions (null.py) ---
        
        plot_null_isa(self.files["null_isa"],
                    tracks=self.tracks, 
                    outpath=os.path.join(self.plots_dir, f"null_isa.png"))
        plot_motif_length(self.files["null_isa"],
                          self.files["motif_locs"],
                          outpath=os.path.join(self.plots_dir, f"motif_length.png"))
        plot_null_interaction(self.files["null_interaction"],
                              tracks=self.tracks, 
                              outpath=os.path.join(self.plots_dir, f"null_interaction.png"))
        plot_motif_distance(self.files["null_interaction"],
                            self.files["isa_combi"],
                            outpath=os.path.join(self.plots_dir, f"motif_distance.png"))
        # --- B. Interaction Plots (interaction.py) ---
        
        df_isa_combi = pd.read_csv(self.files["isa_combi"])
        plot_interaction_decay(df_isa_combi, 
                               self.tracks, 
                               mode='signed', 
                               outpath=os.path.join(self.plots_dir, f"interaction_decay_signed.png"))
        
        for t in self.tracks:
            t_suffix = f"_t{t}"
            # Helper to generate output paths
            def ppath(name): return os.path.join(self.plots_dir, f"{name}{t_suffix}.png")
            # 1. Load the specific results for this track
            coop_pair_path = self.files["coop_tf_pair"].replace(".csv", f"_t{t}.csv")
            coop_tf_path = self.files["coop_tf"].replace(".csv", f"_t{t}.csv")
            imp_path = self.files["imp_tf"]
            
            if not os.path.exists(coop_pair_path) or not os.path.exists(coop_tf_path):
                logger.warning(f"Results for track {t} not found. Skipping.")
                continue
                
            df_coop_pair = pd.read_csv(coop_pair_path)
            df_coop_tf = pd.read_csv(coop_tf_path)
            df_imp = pd.read_csv(imp_path)

            # --- C. Cooperativity Distribution (cooperativity.py) ---
            hist_coop_score(df_coop_pair, outpath=ppath("coop_score_hist"))
            heatmap_coop_score(df_coop_pair, outpath=ppath("coop_score_heatmap"))
            plot_motif_distance_by_category(df_coop_pair, outpath=ppath("distance_by_category"))

            # --- D. TF Importance & GC (tf.py) ---
            plot_motif_gc_by_coop(df_coop_tf, outpath=ppath("motif_gc_by_coop"))
            plot_coop_vs_importance(df_coop_tf, df_imp, 
                                     x_col="coop_score", 
                                     y_col=f"mean_isa_t{t}", 
                                     outpath=ppath("coop_vs_importance"))
            plot_partner_specificity(df_coop_pair, df_coop_tf, outpath=ppath("partner_specificity_ratio"))

            # --- E. TF Family Exploration (tf_family.py) ---
            plot_coop_by_tf_pair_family(df_coop_pair, outpath=ppath("family_coop_summary"))
            plot_coop_by_dbd(df_coop_tf, outpath=ppath("dbd_coop_summary"))
            plot_intra_family_coop_score(df_coop_pair, outpath=ppath("intra_family_distribution"))

            # --- F. TF Functional Evolution (tf_function.py) ---
            plot_usf_pfs(df_coop_tf, outpath=ppath("usf_pioneer_ecdf"))
            plot_cell_specificity(df_coop_tf, outpath=ppath("rolling_gini_specificity"))

            # --- G. PPI Validation (tf_pair_ppi.py) ---
            plot_ppi_enrichment(df_coop_pair, rank_by="coop_score", outpath=ppath("ppi_enrichment_by_coop_score"))
            plot_ppi_enrichment(df_coop_pair, rank_by="p_val", outpath=ppath("ppi_enrichment_by_p_val"))
            plot_cofactor_recruitment(df_coop_pair, outpath=ppath("ppi_violin_validation"))
            plot_dna_mediated_ppi(df_coop_pair, rank_by="coop_score", outpath=ppath("dna_ppi_enrichment_by_score"))
            plot_dna_mediated_ppi(df_coop_pair, rank_by="p_val", outpath=ppath("dna_ppi_enrichment_by_pval"))
        logger.info(f"Report complete. All plots saved to {self.plots_dir}")

    # ------------------------------------------------------------------
    # Motif discovery: tf-modisco-lite + Fi-NeMo
    # ------------------------------------------------------------------
    # These methods are *additive*: they do not touch run_isa() and can be
    # called independently or via the run_motif_discovery() orchestrator.
    def _resolve_positives(self, df_pos):
        """Pick the positive-region DataFrame, mirroring run_isa() semantics."""
        if df_pos is not None:
            logger.info(f"{len(df_pos)} positive regions provided for motif discovery.")
            return df_pos
        if self.df_labeled is not None and 'target_class' in self.df_labeled.columns:
            df_pos = self.df_labeled[self.df_labeled['target_class'] == 1].copy()
            logger.info(f"{len(df_pos)} positive regions identified from labeled data.")
            return df_pos
        logger.warning("Using df_input as positives for motif discovery. Ensure this is intended.")
        return self.df_input.copy()

    def _positives_to_onehot(self, df_pos):
        """Load FASTA once and one-hot encode all positive regions -> (N, 4, L)."""
        from deepISA.utils import get_sequences_from_df, one_hot_encode, load_fasta
        fasta = load_fasta(self.fasta_path)
        seqs = get_sequences_from_df(df_pos, fasta)
        return one_hot_encode(seqs)

    def _predict_activity(self, df_pos, track):
        """Model prediction per region on one output track -> 1D np.ndarray."""
        X = self._positives_to_onehot(df_pos)
        self.model.eval()  # dropout off -> ranking is deterministic
        preds = []
        with torch.no_grad():
            for s in range(0, len(X), 256):
                xb = torch.from_numpy(X[s:s + 256]).to(self.device)
                y = self.model(xb).detach().cpu().numpy()
                preds.append(y)
        pred = np.concatenate(preds, axis=0)
        return pred[:, track] if pred.ndim == 2 and pred.shape[1] > 1 else pred.ravel()

    def _select_modisco_regions(self, df_pos, tracks, top_frac=0.1, rank_by=None,
                                drop_N=True, max_drop_frac=0.2):
        """Curate the motif-discovery input set (mc000's ``y >= THRESH`` analog).

        N policy with a safety valve: N-containing regions are dropped only
        while they stay within ``max_drop_frac`` of the input. Beyond that the
        drop would remove too much data, so all regions are kept and their
        unknown bases are imputed with random ACGT during attribution instead.
        """
        from deepISA.scoring.discover import select_top_regions, drop_non_acgt_regions
        if drop_N and len(df_pos) > 0:
            kept, n_dropped = drop_non_acgt_regions(df_pos, self.fasta_path)
            frac = n_dropped / len(df_pos)
            if frac <= max_drop_frac:
                df_pos = kept
                if n_dropped:
                    logger.info(
                        f"Dropped {n_dropped}/{len(df_pos) + n_dropped} region(s) containing "
                        f"unknown bases (N): {frac:.0%} <= max_drop_frac {max_drop_frac:.0%}."
                    )
            else:
                logger.warning(
                    f"{n_dropped}/{len(df_pos)} ({frac:.0%}) regions contain N -- more than "
                    f"max_drop_frac {max_drop_frac:.0%}. Keeping them; unknown bases will be "
                    f"imputed with random ACGT during attribution instead of dropping."
                )
        if top_frac is not None:
            if len(df_pos) == 0:
                raise ValueError("No regions left after dropping N-containing ones.")
            if rank_by is not None:
                score = df_pos[rank_by].to_numpy(dtype=float)
                logger.info(f"Ranking {len(df_pos)} regions by column '{rank_by}'.")
            else:
                score = self._predict_activity(df_pos, tracks[0])
                logger.info(
                    f"Ranking {len(df_pos)} regions by model prediction on track {tracks[0]}."
                )
            df_pos = select_top_regions(df_pos, score, top_frac=top_frac)
            logger.info(f"Motif discovery input: top {top_frac:.0%} -> {len(df_pos)} regions.")
        return df_pos

    def run_modisco(self,
                    tracks=None,
                    df_pos=None,
                    n_refs=100,
                    attr_batch_size=64,
                    window=None,
                    n_seqlets=50000,
                    task_name="discovery",
                    target_motif_len=40,
                    save_motifs_csv=True,
                    top_frac=0.1,
                    rank_by=None,
                    drop_N=True,
                    max_drop_frac=0.2):
        """Discover motifs with tf-modisco-lite from model attributions.

        Runs: input-set curation -> attribution (tangermeme DeepLIFT-SHAP) ->
        NPZ prep -> ``modisco motifs``. Requires the external ``modisco``
        binary on PATH (see
        :func:`deepISA.scoring.discover.modisco.run_modisco`).

        Parameters
        ----------
        tracks : list of int, optional
            Output tracks to attribute. Defaults to ``self.tracks`` if set, else
            ``[0]``.
        df_pos : pd.DataFrame, optional
            Positive regions. Defaults to the same resolution logic as run_isa().
        n_refs : int
            Number of dinucleotide-shuffled references per attribution background.
        attr_batch_size : int
            Sequences per attribution batch (GPU memory trade-off).
        window : int, optional
            tf-modisco-lite ``-w``: window size around the center of each region
            used for seqlet discovery. ``None`` (default) uses the full region
            length (e.g. 600 for a 600bp model) -- the sequences themselves are
            *not* trimmed, only this analysis window is set. Pass a smaller
            value only to restrict discovery to the central portion.
        n_seqlets : int
            Maximum seqlets for tf-modisco-lite (``-n``).
        task_name : str
            Prefix for discovered motif ids.
        target_motif_len : int
            Length to center-crop discovered motifs to when saving the CSV.
        save_motifs_csv : bool
            If True, also materialize ``discovered_motifs.csv`` (one row per
            motif with metadata) for downstream inspection.
        top_frac : float or None
            Keep only the top fraction of regions by activity (default 0.1 =
            top 10%) so discovery focuses on sequences the model is confident
            about. Ranked by model prediction on ``tracks[0]``, or by the
            ``rank_by`` column when given. ``None`` disables ranking (all
            regions are used).
        rank_by : str, optional
            Column in ``df_pos`` to rank by (e.g. a measured-signal column).
            Defaults to model predictions.
        drop_N : bool
            Drop regions whose sequence contains unknown bases (N) instead of
            imputing them (default True). Any N that still reaches attribution
            is imputed with random ACGT there.
        max_drop_frac : float
            Safety valve for ``drop_N``: N-containing regions are dropped only
            while they account for at most this fraction of the input
            (default 0.2 = 20%). Beyond that, all regions are kept and their
            N bases are imputed with random ACGT during attribution instead.
        """
        if self.model is None:
            raise ValueError("Model not defined. Call define_model() first.")

        tracks = list(tracks) if tracks is not None else list(getattr(self, "tracks", [0]))
        df_pos = self._resolve_positives(df_pos)
        df_pos = self._select_modisco_regions(
            df_pos, tracks, top_frac=top_frac, rank_by=rank_by, drop_N=drop_N,
            max_drop_frac=max_drop_frac,
        )
        if len(df_pos) == 0:
            raise ValueError("Empty motif-discovery input set after curation.")
        seqs_ohe = self._positives_to_onehot(df_pos)
        ids = df_pos["region"].astype(str).tolist() if "region" in df_pos.columns else None

        logger.info(f"Computing attribution for {len(seqs_ohe)} regions, tracks={tracks}.")
        compute_attribution(
            model=self.model,
            seqs_ohe=seqs_ohe,
            tracks=tracks,
            device=self.device,
            n_refs=n_refs,
            batch_size=attr_batch_size,
            save_h5_path=self.files["attr_h5"],
            ids=ids,
        )

        modisco_dir = os.path.join(self.data_dir, "modisco_input")
        # prepare_modisco_input preserves the full sequence length (no trim).
        # Motif discovery runs on a single track (the first requested one);
        # tracks are kept separate because averaging regression+classification
        # attributions produces a meaningless mix.
        ohe_npz, hyp_npz = prepare_modisco_input(
            h5_path=self.files["attr_h5"],
            out_dir=modisco_dir,
            track_index=0,
        )
        run_modisco(
            ohe_npz=ohe_npz,
            hyp_npz=hyp_npz,
            out_h5=self.files["modisco_h5"],
            n_seqlets=n_seqlets,
            window=window,
        )

        if save_motifs_csv:
            motifs = load_motifs(
                self.files["modisco_h5"],
                task_name=task_name,
                target_len=target_motif_len,
            )
            rows = [
                {
                    "motif_id": mid,
                    "num_seqlets": m["num_seqlets"],
                    "task": m["task"],
                    "length": m["cwm"].shape[0],
                }
                for mid, m in motifs.items()
            ]
            pd.DataFrame(rows).to_csv(self.files["discovered_motifs"], index=False)
            logger.info(f"Discovered {len(rows)} motifs -> {self.files['discovered_motifs']}")

        logger.info(f"Motif discovery complete. Results: {self.files['modisco_h5']}")
        return self.files["modisco_h5"]

    def run_finemo(self,
                   motif_db_h5=None,
                   tracks=None,
                   df_pos=None,
                   n_refs=100,
                   attr_batch_size=64,
                   lam=0.7,
                   max_steps=10000,
                   top_frac=0.1,
                   rank_by=None,
                   drop_N=True,
                   max_drop_frac=0.2):
        """Scan sequences for motif hits with Fi-NeMo.

        Runs: attribution (reused if available) -> NPZ prep ->
        ``finemo call-hits`` -> annotated hits DataFrame. Requires the external
        ``finemo`` binary on PATH.

        Parameters
        ----------
        motif_db_h5 : str, optional
            Motif database H5. Defaults to ``self.files["modisco_h5"]`` (i.e.
            run :meth:`run_modisco` first). May also be a Fi-NeMo DB built by
            :func:`deepISA.scoring.discover.finemo.build_finemo_db`.
        tracks, df_pos, n_refs, attr_batch_size :
            Forwarded to attribution. Ignored if ``self.files["attr_h5"]``
            already exists (attributions are reused).
        lam : float
            Fi-NeMo lambda trade-off.
        max_steps : int
            Fi-NeMo optimization step budget.
        top_frac, rank_by, drop_N, max_drop_frac :
            Input-set curation, applied only when attribution is computed
            fresh (same semantics as :meth:`run_modisco`; an existing
            attribution H5 is reused unchanged).
        """
        if self.model is None:
            raise ValueError("Model not defined. Call define_model() first.")
        motif_db_h5 = motif_db_h5 or self.files["modisco_h5"]
        if not os.path.exists(motif_db_h5):
            raise FileNotFoundError(
                f"Motif DB not found: {motif_db_h5}. "
                "Run run_modisco() first or pass an explicit motif_db_h5."
            )

        tracks = list(tracks) if tracks is not None else list(getattr(self, "tracks", [0]))

        # Reuse existing attribution when available; otherwise compute fresh
        # (with the same input-set curation as run_modisco, so a fresh H5
        # matches what run_modisco would have produced).
        if not os.path.exists(self.files["attr_h5"]):
            df_pos = self._resolve_positives(df_pos)
            df_pos = self._select_modisco_regions(
                df_pos, tracks, top_frac=top_frac, rank_by=rank_by, drop_N=drop_N,
                max_drop_frac=max_drop_frac,
            )
            seqs_ohe = self._positives_to_onehot(df_pos)
            ids = df_pos["region"].astype(str).tolist() if "region" in df_pos.columns else None
            logger.info(f"Computing attribution for {len(seqs_ohe)} regions, tracks={tracks}.")
            compute_attribution(
                model=self.model,
                seqs_ohe=seqs_ohe,
                tracks=tracks,
                device=self.device,
                n_refs=n_refs,
                batch_size=attr_batch_size,
                save_h5_path=self.files["attr_h5"],
                ids=ids,
            )

        # Build finemo NPZ from the (channel-first) attribution H5.
        # Reuse the first track (consistent with run_modisco).
        from deepISA.scoring.discover.modisco import read_attribution_h5
        seqs_4lc, hyp_4lc, _ = read_attribution_h5(self.files["attr_h5"], track_index=0)
        finemo_npz = build_finemo_input(
            seqs_ohe=seqs_4lc,
            hyp_scores=hyp_4lc,
            out_dir=os.path.dirname(self.files["finemo_npz"]),
            ids=None,
        )

        hits_tsv = run_finemo_scan(
            npz_path=finemo_npz,
            out_dir=os.path.dirname(self.files["finemo_hits"]),
            motif_db_h5=motif_db_h5,
            lam=lam,
            max_steps=max_steps,
        )
        df_hits = load_hits_with_annotation(hits_tsv, motif_db_h5)
        df_hits.to_csv(self.files["finemo_hits"], sep="\t", index=False)
        logger.info(f"Fi-NeMo hits ({len(df_hits)} rows) -> {self.files['finemo_hits']}")
        return df_hits

    def run_motif_discovery(self,
                            tracks=None,
                            df_pos=None,
                            do_modisco=True,
                            do_finemo=True,
                            motif_db_h5=None,
                            **kwargs):
        """One-call orchestrator: tf-modisco-lite -> Fi-NeMo.

        Convenience wrapper that runs :meth:`run_modisco` (if ``do_modisco``)
        then :meth:`run_finemo` (if ``do_finemo``), sharing the attribution H5.
        Any extra ``kwargs`` are split between the two methods by keyword.
        """
        modisco_keys = {"n_refs", "attr_batch_size", "window", "n_seqlets",
                        "task_name", "target_motif_len", "save_motifs_csv",
                        "top_frac", "rank_by", "drop_N", "max_drop_frac"}
        finemo_keys = {"lam", "max_steps",
                       "top_frac", "rank_by", "drop_N", "max_drop_frac"}
        unknown = set(kwargs) - modisco_keys - finemo_keys
        if unknown:
            logger.warning(f"run_motif_discovery ignoring unknown kwargs: {sorted(unknown)}")

        if do_modisco:
            self.run_modisco(
                tracks=tracks,
                df_pos=df_pos,
                **{k: v for k, v in kwargs.items() if k in modisco_keys},
            )
        if do_finemo:
            self.run_finemo(
                motif_db_h5=motif_db_h5,
                tracks=tracks,
                df_pos=df_pos,
                **{k: v for k, v in kwargs.items() if k in finemo_keys},
            )

    # ------------------------------------------------------------------
    # Motif report: tf-modisco-lite HTML report + MEME/TOMTOM annotation
    # ------------------------------------------------------------------
    def run_motif_report(self,
                        meme_db=None,
                        task_name="discovery",
                        target_motif_len=40,
                        export_meme=True):
        """Generate the tf-modisco-lite HTML report, optionally with TOMTOM.

        Runs ``modisco report`` on :attr:`self.files["modisco_h5"]`, writing the
        HTML report (logos + seqlet tables) to :attr:`self.files["motif_report"]`.

        Parameters
        ----------
        meme_db : str, optional
            Path to a MEME-format motif database (e.g. JASPAR). When provided,
            tf-modisco-lite runs TOMTOM and annotates each discovered motif with
            its best known-TF match. If ``None``, only the unannotated report is
            produced.
        task_name : str
            Prefix for motif ids when exporting the MEME file (forwarded to
            :func:`deepISA.scoring.discover.h5_io.load_motifs`).
        target_motif_len : int
            Length to normalize motifs to when exporting the MEME file.
        export_meme : bool
            If True, also export discovered motifs to a MEME file at
            :attr:`self.files["discovered_meme"]` (useful for standalone TOMTOM
            runs against external databases).

        Returns
        -------
        str
            The report directory path.

        Raises
        ------
        FileNotFoundError
            If :attr:`self.files["modisco_h5"]` does not exist (run
            :meth:`run_modisco` first).
        """
        if not os.path.exists(self.files["modisco_h5"]):
            raise FileNotFoundError(
                f"modisco results not found: {self.files['modisco_h5']}. "
                "Run run_modisco() first."
            )

        if export_meme:
            motifs = load_motifs(
                self.files["modisco_h5"],
                task_name=task_name,
                target_len=target_motif_len,
            )
            cwm_to_meme(motifs, self.files["discovered_meme"])
            logger.info(f"Discovered motifs exported to MEME: {self.files['discovered_meme']}")

        return run_motif_report(
            modisco_h5=self.files["modisco_h5"],
            out_dir=self.files["motif_report"],
            meme_db=meme_db,
        )
