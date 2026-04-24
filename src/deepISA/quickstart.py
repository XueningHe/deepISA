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
from deepISA.scoring.single_isa import run_single_isa, calc_tf_importance
from deepISA.scoring.combi_isa import (
    run_combi_isa, 
    calc_coop_score
)
from deepISA.utils import setup_logger, find_available_gpu


# Plotting functions

from deepISA.plotting.interaction import (
    plot_null,
    plot_interaction_decay, 
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
            "motif_locs":   os.path.join(self.data_dir, "motif_locs.csv"),
            "isa_single":   os.path.join(self.data_dir, "motif_single_isa.csv"),
            "isa_combi":    os.path.join(self.data_dir, "motif_combi_isa.csv"),
            "imp_tf":       os.path.join(self.data_dir, "tf_importance.csv"),
            "coop_tf_pair": os.path.join(self.data_dir, "coop_tf_pair.csv"),
            "coop_tf":      os.path.join(self.data_dir, "coop_tf.csv")
        }
        self.fasta_path = fasta_path
        self.df_input = df_input # can be either positive or negative regions
        self.df_train = None
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
        self.df_train = compile_training_data(
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
        
        
    def run_isa(self, 
                jaspar_path,
                isa_config,
                df_pos=None,
                expressed_tfs=None
                ):
                
        if df_pos is None:
            if self.df_train is not None and 'target_class' in self.df_train.columns:
                df_pos = self.df_train[self.df_train['target_class'] == 1].copy()
                logger.info(f"Automatically selected {len(df_pos)} positive regions from training set.")
            else:
                df_pos = self.df_input.copy()
                logger.warning("Using df_input as positives for ISA. Ensure this is intended.")
                        

        self.tracks = isa_config.get('tracks', [0])
        
        # 3. Motif Mapping
        map_motifs(
            regions_df=df_pos,
            jaspar_path=jaspar_path, # Explicitly required here
            outpath=self.files["motif_locs"],
            model=self.model,
            fasta_path=self.fasta_path,
            tracks=self.tracks,
            device=self.device,
            motif_score_thresh=isa_config.get('motif_score_threshold', 500),
            remap_path=isa_config.get('remap_path', None),
            attr_percentile=isa_config.get('attr_percentile', 70),
            attr_batch_size=isa_config.get('attr_batch_size', 1024),
            expressed_tfs=expressed_tfs
        )

        # Handle remap subsetting logic if requested
        if isa_config.get('subset_by_remap', False):
            logger.info("Subsetting motif locations by ReMap evidence...")
            df_temp = pd.read_csv(self.files["motif_locs"])
            df_temp = df_temp[df_temp['remap_evidence'] == True].reset_index(drop=True)
            df_temp.to_csv(self.files["motif_locs"], index=False)
        
        # 4. ISA Execution
        run_single_isa(
            model=self.model, 
            fasta_path=self.fasta_path, 
            motif_locs_path=self.files["motif_locs"],
            outpath=self.files["isa_single"],
            device=self.device, 
            tracks=self.tracks,
            num_regions_per_batch=isa_config['num_regions_per_batch'],
            pred_batch_size=isa_config['pred_batch_size'])
        
        run_combi_isa(
            model=self.model, 
            fasta_path=self.fasta_path,
            motif_locs_path=self.files["motif_locs"],
            outpath=self.files["isa_combi"], 
            tracks=self.tracks,
            inde_dist_max=isa_config["inde_dist_max"],
            device=self.device,
            num_regions_per_batch=isa_config['num_regions_per_batch'],
            pred_batch_size=isa_config['pred_batch_size'])
        
        # aggregate
        logger.info(f"Aggregating results in {self.data_dir}")
        df_imp = calc_tf_importance(self.files["isa_single"], 
                                    min_count=isa_config["min_count"])
        df_imp.to_csv(self.files["imp_tf"], index=False)

        for t in self.tracks:
            calc_coop_score(self.files["isa_combi"], 
                            outpath=self.files["coop_tf_pair"].replace(".csv", f"_t{t}.csv"),
                            level="tf_pair",
                            inde_dist_min=isa_config["inde_dist_min"],
                            inde_dist_max=isa_config["inde_dist_max"],
                            track_idx=t, 
                            min_count=isa_config['min_count'],
                            q_val_thresh=isa_config['q_val_thresh'])
                            
            calc_coop_score(self.files["isa_combi"], 
                            outpath=self.files["coop_tf"].replace(".csv", f"_t{t}.csv"),
                            level="tf",
                            inde_dist_min=isa_config["inde_dist_min"],
                            inde_dist_max=isa_config["inde_dist_max"],
                            track_idx=t, 
                            min_count=isa_config['min_count'],
                            q_val_thresh=isa_config['q_val_thresh'])

        logger.info("ISA execution and aggregation complete.")



    def report(self):
        """
        Executes the full suite of visualization and exploration functions.
        """
        logger.info("Generating comprehensive reports and plots...")
        # --- A. Interaction Plots (interaction.py) ---
        
        df_isa_combi = pd.read_csv(self.files["isa_combi"])
        plot_null(df_isa_combi, 
                  tracks=self.tracks, 
                  outpath=os.path.join(self.plots_dir, f"null_distribution.png"))
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

            # --- B. Cooperativity Distribution (cooperativity.py) ---
            hist_coop_score(df_coop_pair, outpath=ppath("coop_score_hist"))
            heatmap_coop_score(df_coop_pair, outpath=ppath("coop_score_heatmap"))
            plot_motif_distance_by_category(df_coop_pair, outpath=ppath("distance_by_category"))

            # --- C. TF Importance & GC (tf.py) ---
            plot_motif_gc_by_coop(df_coop_tf, outpath=ppath("motif_gc_by_coop"))
            plot_coop_vs_importance(df_coop_tf, df_imp, 
                                     x_col="coop_score", 
                                     y_col=f"mean_isa_t{t}", 
                                     outpath=ppath("coop_vs_importance"))
            plot_partner_specificity(df_coop_pair, df_coop_tf, outpath=ppath("cell_specificity_ratio"))

            # --- D. TF Family Exploration (tf_family.py) ---
            plot_coop_by_tf_pair_family(df_coop_pair, outpath=ppath("family_coop_summary"))
            plot_coop_by_dbd(df_coop_tf, outpath=ppath("dbd_coop_summary"))
            plot_intra_family_coop_score(df_coop_pair, outpath=ppath("intra_family_distribution"))

            # --- E. TF Functional Evolution (tf_function.py) ---
            plot_usf_pfs(df_coop_tf, outpath=ppath("usf_pioneer_ecdf"))
            plot_cell_specificity(df_coop_tf, outpath=ppath("rolling_gini_specificity"))

            # --- F. PPI Validation (tf_pair_ppi.py) ---
            plot_ppi_enrichment(df_coop_pair, outpath=ppath("ppi_enrichment_curve"))
            plot_cofactor_recruitment(df_coop_pair, outpath=ppath("ppi_violin_validation"))
            plot_dna_mediated_ppi(df_coop_pair, outpath=ppath("dna_mediated_ppi"))
        logger.info(f"Report complete. All plots saved to {self.plots_dir}")