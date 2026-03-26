import os
import torch
import pandas as pd
import bioframe as bf
from loguru import logger

# DeepISA Imports
from deepISA.modeling.preprocess import compile_training_data
from deepISA.modeling.train import train_model
from deepISA.modeling.cnn import Conv
from deepISA.modeling.predict import evaluate_model
from deepISA.scoring.infer_tf_expr import get_expressed_tfs # is not using get_expressed_tfs a problem?
from deepISA.scoring.annotation import map_motifs
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
    plot_coop_vs_importance
)


# Validating functions

from deepISA.validating.tf_family import (
    plot_coop_by_tf_pair_family,
    plot_coop_by_dbd,
    plot_intra_family_coop_score
)

from deepISA.validating.tf_pair_ppi import (
    plot_ppi_enrichment,
    validate_cofactor_recruitment
)
from deepISA.validating.tf_function import (
    plot_usf_pfs,
    plot_cell_specificity
)


class QuickStart:
    def __init__(self, 
                 results_dir, 
                 fasta_path, 
                 df_regions, 
                 device=None,
                 model_path=None):
        self.results_dir = results_dir
        setup_logger(self.results_dir)
        self.data_dir = os.path.join(self.results_dir, "Data")
        self.plots_dir = os.path.join(self.results_dir, "Plots")
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(self.plots_dir, exist_ok=True)
        self.files = {
            "training_data":os.path.join(self.data_dir, "training_data.csv"),
            "motif_locs":   os.path.join(self.data_dir, "motif_locs.csv"),
            "isa_single":   os.path.join(self.data_dir, "motif_single_isa.csv"),
            "isa_combi":    os.path.join(self.data_dir, "motif_combi_isa.csv"),
            "imp_tf":       os.path.join(self.data_dir, "tf_importance.csv"),
            "coop_tf_pair": os.path.join(self.data_dir, "coop_tf_pair.csv"),
            "coop_tf":      os.path.join(self.data_dir, "coop_tf.csv")
        }
        self.fasta_path = fasta_path
        self.df_regions = df_regions # can be either positive or negative regions
        self.model_path = model_path
        self.model = None
        
        if "target_class" in df_regions.columns:
            self.df_pos = df_regions[df_regions["target_class"] == 1].reset_index(drop=True)
        else:
            # If no target_class (e.g. you passed a pre-filtered df_pos), use all as positive
            # after .train(), df_pos will be updated again using df_train.
            self.df_pos = df_regions.reset_index(drop=True)
        
        # Hardware Setup
        self.device = device if device is not None else find_available_gpu()
        logger.info(f"Using device: {self.device}")



    def train(self, 
              bw_paths=None,
              seq_len=600, 
              epochs=10, 
              target_reg_col="target_reg",
              save_half=False,
              save_one=False):
        """
        Compiles training data and executes the Trainer. 
        The best model is automatically saved by the Trainer class.
        """
        logger.info("Starting Training...")

        # 1. Compile Data (Always fresh)
        logger.info(f"Compiling training data into {self.files['training_data']}...")
        df_train = compile_training_data(
            df=self.df_regions,
            seq_len=seq_len,
            bw_paths=bw_paths,
            target_reg_col=target_reg_col,
            outpath=self.files["training_data"]
        )

        # 2. Run Training
        # Note: train_model internally instantiates the Trainer class you provided
        logger.info("Initializing Trainer...")
        model, history, test_ds = train_model(
            df=df_train,
            fasta_path=self.fasta_path,
            model_dir=self.results_dir, 
            epochs=epochs,
            save_half=save_half,
            save_one=save_one,
        )

        # 3. Update Internal State
        self.df_pos = df_train[df_train["target_class"] == 1].reset_index(drop=True)
        self.model = model
        # The Trainer saves the best model as '{model_name}_best.pt'
        self.model_path = os.path.join(self.results_dir, "model_best.pt")
        _ = evaluate_model(self.model, test_ds)


    def run_isa(self, 
                jaspar_path, 
                expressed_tfs=None, 
                track_idx=0,
                motif_score_thresh=500,
                subset_by_remap=False,
                remap_path=None, 
                batch_size=200):
        
        
        self.track_idx = track_idx if isinstance(track_idx, list) else [track_idx]
        logger.info(f"Running ISA scenario: Threshold={motif_score_thresh}, RemapSubset={subset_by_remap}")
        
        # 1. Model management
        if self.model is None:
            checkpoint = torch.load(self.model_path, map_location=self.device, weights_only=False)
            if isinstance(checkpoint, torch.nn.Module):
                self.model = checkpoint
            else:
                self.model = Conv()
                self.model.load_state_dict(checkpoint)
            self.model.to(self.device).eval()


        # 3. Motif Mapping
        motif_loc_csv = map_motifs(
            regions_df=self.df_pos,
            jaspar_path=jaspar_path,
            expressed_tfs=expressed_tfs, 
            outpath=self.files["motif_locs"],
            score_thresh=motif_score_thresh,
            remap_path=remap_path
        )

        # Handle remap subsetting logic if requested
        if subset_by_remap:
            logger.info("Subsetting motif locations by ReMap evidence...")
            df_temp = pd.read_csv(motif_loc_csv)
            # Assuming 'remap_evidence' is a column added by map_motifs when remap_path is provided
            df_temp = df_temp[df_temp['remap_evidence'] == True].reset_index(drop=True)
            df_temp.to_csv(self.files["motif_locs"], index=False)
            motif_loc_csv = self.files["motif_locs"]
        
        # 4. ISA Execution
        run_single_isa(
            model=self.model, 
            fasta_path=self.fasta_path, 
            motif_locs=motif_loc_csv,
            outpath=self.files["isa_single"],
            track_idx=self.track_idx,
            device=self.device, 
            batch_size=batch_size
        )
        
        run_combi_isa(
            model=self.model, 
            fasta_path=self.fasta_path,
            motif_locs=motif_loc_csv,
            outpath=self.files["isa_combi"], 
            track_idx=self.track_idx,
            device=self.device
        )

    def aggregate_isa(self):
        """Process results using the paths set in run_isa."""
        logger.info(f"Aggregating results in {self.data_dir}")

        df_imp = calc_tf_importance(self.files["isa_single"])
        df_imp.to_csv(self.files["imp_tf"], index=False)

        for t in self.track_idx:
            calc_coop_score(self.files["isa_combi"], track_idx=t, level="tf_pair",
                            outpath=self.files["coop_tf_pair"].replace(".csv", f"_t{t}.csv"))
            calc_coop_score(self.files["isa_combi"], track_idx=t, level="tf",
                            outpath=self.files["coop_tf"].replace(".csv", f"_t{t}.csv"))

    def report(self):
        """Generates visualizations for the current scenario/tracks."""
        logger.info(f"Generating Report in {self.plots_dir}")
        
        df_imp = pd.read_csv(self.files["imp_tf"])
        df_raw_combi = pd.read_csv(self.files["isa_combi"])
        
        # joint plots: interaction
        plot_interaction_decay(df_raw_combi, track_idx=self.track_idx, outpath=os.path.join(self.plots_dir, "interaction_decay.png"))
        plot_null(df_raw_combi, track_idx=self.track_idx, outpath=os.path.join(self.plots_dir, "null.png"))

        # plots separated by track
        for t in self.track_idx:
            t_suffix = f"_t{t}"
            coop_pair_path = self.files["coop_tf_pair"].replace(".csv", f"{t_suffix}.csv")
            coop_tf_path = self.files["coop_tf"].replace(".csv", f"{t_suffix}.csv")
            
            if not os.path.exists(coop_pair_path): continue
                
            df_coop_pair = pd.read_csv(coop_pair_path)
            df_coop_tf = pd.read_csv(coop_tf_path)
            
            def ppath(name): return os.path.join(self.plots_dir, f"{name}{t_suffix}.png")
            
            validate_cofactor_recruitment(df_coop_pair, outpath=ppath("cofactor_recruitment"))
            plot_ppi_enrichment(df_coop_pair, outpath=ppath("ppi_enrichment")) 

            hist_coop_score(df_coop_pair, outpath=ppath("coop_score_hist"))
            heatmap_coop_score(df_coop_pair, outpath=ppath("coop_score_heatmap"))
            plot_motif_gc_by_coop(df_coop_tf, outpath=ppath("motif_gc_by_cooperativity"))
            plot_coop_vs_importance(df_coop_tf,
                                     df_imp,
                                     x_col="coop_score",
                                     y_col=f"mean_isa_t{t}",
                                     outpath=ppath("cooperativity_vs_importance"))
            plot_coop_by_tf_pair_family(df_coop_pair, outpath=ppath("coop_by_tf_pair_family"))
            plot_cell_specificity(df_coop_tf,outpath=ppath("cell_specificity")) 
            plot_usf_pfs(df_coop_tf, outpath=ppath("usf_pfs"))
            plot_coop_by_dbd(df_coop_tf, outpath=ppath("coop_by_dbd"))
            plot_intra_family_coop_score(df_coop_pair, outpath=ppath("coop_score_intra_family"))
            plot_motif_distance_by_category(df_coop_pair, outpath=ppath("motif_distance_by_category")) # 14/24