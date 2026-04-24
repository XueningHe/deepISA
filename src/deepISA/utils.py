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


def get_sequences_from_df(df, fasta):
    """Vectorized sequence fetching using bioframe-loaded fasta."""
    return [
        str(fasta[row.chrom][int(row.start):int(row.end)]).upper() 
        for row in df.itertuples()
    ]


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
    current_file = Path(__file__).resolve()
    project_root = current_file.parents[2]
    path = project_root / "data" / filename
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
    file_format = "{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {function}:{line} - {message}"
    colored_format = (
        "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
        "<level>{message}</level>"
    )
    logger.add(log_file,
               format=file_format,
               level="INFO",
               backtrace=True,
               diagnose=True)
    logger.add(sys.stdout,
               format=colored_format,
               level="INFO",
               colorize=True)
    
    logger.info(f"Logger initialized. Logs redirected to: {log_file}")
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
    # 1. Setup colors
    white, cool, gray, warm = "white", "#1f77b4", "#7f7f7f", "#d62728"
    bins = df[x_col].cat.categories.tolist()
    bin_counts = df[x_col].value_counts()

    if bins == ["Independent", "Redundant", "Intermediate", "Synergistic"]: 
        custom_palette = {bins[0]: white, bins[1]: cool, bins[2]: gray, bins[3]: warm}
    else:
        custom_palette = sns.color_palette("deep", len(bins))

    # 2. Apply Dynamic Scaling
    plt.figure(figsize=figsize)
    ax = plt.gca()
    styles = apply_plot_style(ax, figsize) # Use the utility!

    sns.violinplot(
        data=df, x=x_col, y=y_col, order=bins, cut=0, 
        linewidth=styles['scale'] * 0.5, # Scaled line width
        palette=custom_palette, hue=x_col, legend=False
    )

    # 3. Add Statistical Brackets with Scaled Offsets
    y_min, y_max = df[y_col].min(), df[y_col].max()
    y_range = y_max - y_min
    
    for i in range(len(bins) - 1):
        group1 = df[df[x_col] == bins[i]][y_col].dropna()
        group2 = df[df[x_col] == bins[i+1]][y_col].dropna()
        
        if len(group1) > 1 and len(group2) > 1:
            stat, p_value = mannwhitneyu(group1, group2, alternative="two-sided")
            
            x1, x2 = i, i + 1
            # Scale the bracket heights and tick lengths
            h = y_max + (y_range * 0.08) + (i * y_range * 0.12)
            tick_len = y_range * 0.03
            
            plt.plot([x1, x1, x2, x2], [h - tick_len, h, h, h - tick_len], 
                     lw=styles['scale'] * 0.6, color="black")
            
            plt.text((x1 + x2) / 2, h + (y_range * 0.02), f"p={p_value:.1e}", 
                     ha="center", va="bottom", fontsize=styles['small'])

    # 4. Scaled Formatting
    bin_labels = [f"{b}\n(n={bin_counts[b]})" for b in bins]
    plt.xticks(ticks=range(len(bins)), labels=bin_labels, 
               fontsize=styles['small'], rotation=rotation)
    plt.yticks(fontsize=styles['small'])
    plt.xlabel(x_label, fontsize=styles['main'])
    plt.ylabel(y_label, fontsize=styles['main'])
    
    if title:
        plt.title(title, fontsize=styles['main'])
        
    return save_or_show(outpath) 






def format_cooperativity_categorical(df, categories = ["Independent", "Redundant", "Intermediate", "Synergistic"]):
    """
    Standardizes the 'cooperativity' column as a categorical type with fixed ordering.
    """
    df["cooperativity"] = pd.Categorical(df["cooperativity"], categories=categories, ordered=True)
    return df









def apply_plot_style(ax, fig_size):
    """
    Dynamically scales fonts and line widths based on figure size.
    Base scale is calculated relative to a 2.5-inch width.
    """
    base_width = 2.5
    scale = fig_size[0] / base_width
    font_main = 7 * scale
    font_small = 5 * scale
    lw = 0.5 * scale
    for spine in ax.spines.values():
        spine.set_linewidth(lw)
    ax.tick_params(axis='both', which='major', labelsize=font_small, 
                   width=lw, length=2 * scale)
    return {'main': font_main, 'small': font_small, 'scale': scale}



def save_or_show(outpath):
    if outpath:
        plt.savefig(outpath, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()
        plt.close()