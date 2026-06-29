import os
import torch
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
            "coop_tf":       os.path.join(self.data_dir, "coop_tf.csv")
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
              rc_aug=True):
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
                "save_one": False
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
            rc_aug=rc_aug
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
        filename = f"model_{suffix}.pt"
        path = os.path.join(self.model_dir, filename)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Checkpoint {filename} not found at {path}")
        self.model.load_state_dict(torch.load(path, map_location=self.device, weights_only=True))
        logger.info(f"Successfully loaded checkpoint: {filename}")
    
    
    
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