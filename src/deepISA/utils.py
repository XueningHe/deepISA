import torch
import numpy as np
import pandas as pd
from loguru import logger
import os
import sys
import pyBigWig
from pathlib import Path


import bioframe as bf
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu



def one_hot_encode(seqs):
    """Converts list of DNA strings to (N, 4, L) float32 numpy array."""
    mapping = {"A": [1,0,0,0], "C": [0,1,0,0], "G": [0,0,1,0], "T": [0,0,0,1]}
    # Default to [0,0,0,0] for 'N' or unknown bases
    X = np.array([[mapping.get(base, [0,0,0,0]) for base in seq] for seq in seqs], dtype='float32')
    return np.transpose(X, (0, 2, 1))



def ablate_motifs(seq, motif_starts, motif_ends):
    """
    Scramble the sequence between multiple motif start and end positions.
    Args:
        seq: A string of sequence.
        motif_starts: A list of integers for motif starts.
        motif_ends: A list of integers for motif ends.
    Returns:
        A string of scrambled sequence.
    """
    if isinstance(motif_starts, int):
        motif_starts = [motif_starts]
    if isinstance(motif_ends, int):
        motif_ends = [motif_ends]
    if len(motif_starts) != len(motif_ends):
        raise ValueError("motif_starts and motif_ends must have the same length")
    # Sort the motifs by start position
    motifs = sorted(zip(motif_starts, motif_ends), key=lambda x: x[0])
    # Initialize variables
    seq_ablated = ''
    previous_end = 0
    # Iterate and ablate each motif
    for start, end in motifs:
        if start < previous_end:
            raise ValueError("Overlapping motifs detected")
        end = end + 1  
        motif = seq[start:end]
        motif_scrambled = "N" * len(motif)  
        # Append non-motif and scrambled motif parts
        seq_ablated += seq[previous_end:start] + motif_scrambled
        previous_end = end
    # Append the remaining part of the sequence if any
    seq_ablated += seq[previous_end:]
    return seq_ablated




def find_available_gpu(min_memory_gb=2):
    """
    Finds the first GPU with at least min_memory_gb available.
    Returns: torch.device object (cuda:x or cpu)
    """
    if not torch.cuda.is_available():
        logger.warning("No CUDA GPUs detected. Falling back to CPU.")
        return torch.device("cpu")
    min_memory_bytes = min_memory_gb * 1024**3
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        if props.total_memory >= min_memory_bytes:
            logger.info(f"Using GPU {i}: {props.name} ({props.total_memory / 1024**3:.2f} GB)")
            return torch.device(f"cuda:{i}")
    logger.warning(f"No GPUs with >{min_memory_gb}GB found. Falling back to CPU.")
    return torch.device("cpu")





def get_data_resource(filename):
    """
    Finds data files relative to the package installation.
    """
    # 1. Get the absolute path to utils.py
    # src/deepISA/utils.py
    current_file = Path(__file__).resolve()
    
    # 2. Go up 3 levels to reach the root (utils.py -> deepISA -> src -> root)
    # This works during development (pip install -e .)
    project_root = current_file.parents[2]
    path = project_root / "data" / filename
    
    # 3. Fallback: Check if we are in a site-packages/ installed environment
    # Sometimes 'data' is moved inside 'src/deepISA/data' during packaging
    if not path.exists():
        # Look for deepISA/data/ relative to utils.py
        path = current_file.parent / "data" / filename

    if not path.exists():
        logger.warning(f"Resource {filename} not found at {path}")
        # Final fallback: just try the CWD (what you had before)
        path = Path("data") / filename
            
    return str(path)




def setup_logger(model_dir):
    os.makedirs(model_dir, exist_ok=True)
    log_file = os.path.join(model_dir, "workflow.log")
    logger.remove() 
    logger.add(log_file, 
               format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {message}", 
               level="INFO",
               backtrace=True, 
               diagnose=True) 
    logger.add(sys.stdout, 
               format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | {message}", 
               level="INFO")
    logger.info(f"Logger initialized. All logs and errors redirected to: {log_file}")
    return log_file





def resize_regions(df, seq_len):
    """
    Checks if regions are already at the target width. 
    Standardizes regions to a fixed width by centering if necessary.
    """
    # Calculate current lengths
    current_lengths = df['end'] - df['start']
    
    # Check if all rows match the target seq_len
    if (current_lengths == seq_len).all():
        logger.info(f"All {len(df)} regions already have the identical target length of {seq_len} bp. Skipping resize.")
        return df
    
    # If not, perform the centering and resizing
    logger.info(f"Regions have variable lengths. Centering and resizing {len(df)} regions to {seq_len} bp.")
    df = df.copy()
    centers = (df['start'] + df['end']) // 2
    df['start'] = centers - (seq_len // 2)
    df['end'] = df['start'] + seq_len
    return df






def quantify_bw(regions_df, bw_paths, seq_len):
    """Quantifies sum of signals from BigWig files with progress logging."""
    regions_df = resize_regions(regions_df, seq_len)
    total_signals = np.zeros(len(regions_df))
    n_regions = len(regions_df)
    for bw_path in bw_paths:
        logger.info(f"Processing BigWig: {os.path.basename(bw_path)}")
        with pyBigWig.open(bw_path) as bw:
            chrom_sizes = bw.chroms()
            signals = []
            for i, row in enumerate(regions_df.itertuples()):
                # Log progress every 100000 regions
                if (i + 1) % 100000 == 0:
                    logger.info(f"Quantified {i + 1}/{n_regions} regions...")
                if row.chrom not in chrom_sizes or row.start < 0 or row.end > chrom_sizes[row.chrom]:
                    signals.append(0.0)
                else:
                    # bw.stats with type="sum" returns a list; we take the first element
                    val = bw.stats(row.chrom, int(row.start), int(row.end), type="sum")[0] or 0.0
                    signals.append(abs(val))
            total_signals += np.array(signals)
    return total_signals, regions_df






def estimate_noise_threshold(bw_paths, seq_len, percentile=99):
    """Estimates a noise threshold using a non-functional background BED."""
    bg_bed_path = get_data_resource("non_cCRE_non_blacklist_non_exon.bed")
    bg_df = bf.read_table(bg_bed_path, schema='bed')
    bg_df = bg_df.sample(n=50000, random_state=42)
    logger.info(f"Estimating noise threshold from {len(bg_df)} background regions...")
    bg_df = resize_regions(bg_df, seq_len)
    signals, _ = quantify_bw(bg_df, bw_paths, seq_len)
    
    threshold = np.percentile(signals, percentile)
    logger.info(f"Calculated noise threshold: {threshold:.4f} ({percentile}th percentile)")
    return threshold










def plot_violin_with_statistics(
    figsize,
    df, 
    x_col, 
    y_col, 
    x_label, 
    y_label, 
    title, 
    rotation,
    outpath=None
):
    """
    Plots a violin plot, annotates with Mann-Whitney U test p-values using 
    non-overlapping stepped brackets, and handles small sample warnings.
    """
    # 1. Setup Colors and Categories
    white = "white"
    cool = "#1f77b4"
    gray = "#7f7f7f"
    warm = "#d62728"

    bins = df[x_col].cat.categories.tolist()
    bin_counts = df[x_col].value_counts()

    if bins == ["Independent", "Redundant", "Intermediate", "Synergistic"]: 
        custom_palette = {bins[0]: white, bins[1]: cool, bins[2]: gray, bins[3]: warm}
    elif bins == ["No", "Yes"]:
        custom_palette = {"No": white, "Yes": white}
    elif bins == [0, 1, 2]:
        custom_palette = {0: white, 1: white, 2: white}
    else:
        custom_palette = sns.color_palette("deep", len(bins))

    # 2. Create the Figure
    plt.figure(figsize=figsize)
    ax = plt.gca()
    
    # Clean up axes
    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)
    for spine in ['left', 'bottom']:
        ax.spines[spine].set_linewidth(0.5)

    # Use hue=x_col and legend=False to fix the FutureWarning
    # make sure inner only has median
    sns.violinplot(
        data=df, 
        x=x_col, 
        y=y_col, 
        order=bins, 
        cut=0, 
        linewidth=0.5, 
        palette=custom_palette,
        hue=x_col,   
        legend=False
    )

    # 3. Add Statistical Brackets (Fixing Overlaps and Warnings)
    y_min, y_max = df[y_col].min(), df[y_col].max()
    y_range = y_max - y_min
    
    for i in range(len(bins) - 1):
        bin1, bin2 = bins[i], bins[i+1]
        group1 = df[df[x_col] == bin1][y_col].dropna()
        group2 = df[df[x_col] == bin2][y_col].dropna()
        
        # Check sample size to prevent SmallSampleWarning
        if len(group1) > 1 and len(group2) > 1:
            stat, p_value = mannwhitneyu(group1, group2, alternative="two-sided")
            
            x1, x2 = i, i + 1
            # "h" increases with i to create stepped brackets that don't overlap
            h = y_max + (y_range * 0.08) + (i * y_range * 0.12)
            tick_len = y_range * 0.03
            
            # Draw the bracket line with small vertical "ticks" at the ends
            plt.plot([x1, x1, x2, x2], [h - tick_len, h, h, h - tick_len], lw=0.6, color="black")
            
            # Place text with va="bottom" and a dedicated offset to stay above the line
            plt.text(
                (x1 + x2) / 2, 
                h + (y_range * 0.02), 
                f"p={p_value:.1e}", 
                ha="center", 
                va="bottom", 
                fontsize=5
            )

    # 4. Final Formatting
    bin_labels = [f"{b}\n(n={bin_counts[b]})" for b in bins]
    plt.xticks(ticks=range(len(bins)), labels=bin_labels, fontsize=5, rotation=rotation)
    plt.yticks(fontsize=5)
    plt.xlabel(x_label, fontsize=7)
    plt.ylabel(y_label, fontsize=7)
    
    if title:
        plt.title(title, fontsize=6)
        
    if outpath:
        plt.tight_layout()
        plt.savefig(outpath, bbox_inches='tight')
        plt.close()
        logger.info(f"Saved: {outpath}")
    else:
        plt.show()








def format_cooperativity_categorical(df, categories = ["Independent", "Redundant", "Intermediate", "Synergistic"]):
    """
    Standardizes the 'cooperativity' column as a categorical type with fixed ordering.
    """
    df["cooperativity"] = pd.Categorical(df["cooperativity"], categories=categories, ordered=True)
    return df





def load_data(df):
    if isinstance(df, pd.DataFrame):
        return df
    if isinstance(df, str):
        if df.endswith(".csv"):
            df = pd.read_csv(df)
        elif df.endswith(".bed"):
            df = bf.read_table(df, schema='bed')
    return df