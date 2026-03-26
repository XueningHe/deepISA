import matplotlib.pyplot as plt
import seaborn as sns
from loguru import logger
import pandas as pd
import matplotlib.gridspec as gridspec

from deepISA.utils import plot_violin_with_statistics, format_cooperativity_categorical
from deepISA.scoring.combi_isa import assign_cooperativity





def hist_coop_score(
    df, 
    title=None, 
    xlabel="Cooperativity score", 
    outpath=None, 
    vlines=None, 
    annotations=None, # (x, relative_y, text)
    fig_size=(2.3, 2.0)
):
    # filter for non nan coop score
    df=assign_cooperativity(df)
    # remove cooperativity=Independent
    df=df[df["cooperativity"] != "Independent"].reset_index(drop=True)   
     
    plt.figure(figsize=fig_size)
    ax = plt.gca()
    # Consistent frame styling
    for spine in ax.spines.values():
        spine.set_linewidth(0.5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    # Histogram
    sns.histplot(df["coop_score"], bins=50, color='steelblue', edgecolor='black', linewidth=0.2)
    # Vertical dividers
    if vlines:
        for x in vlines:
            plt.axvline(x=x, color='grey', linestyle='--', linewidth=0.7, alpha=0.8)
    # Category labels using Relative Y-Coordinates
    if annotations:
        transform = ax.get_xaxis_transform()
        for x, rel_y, label in annotations:
            plt.text(x, rel_y, label, transform=transform, 
                     fontsize=6, ha='center', va='bottom')
    # Formatting and Margin Fixes
    plt.xlabel(xlabel, fontsize=7)
    plt.ylabel('Frequency', fontsize=7)
    if title:
        plt.title(title, fontsize=7)
    plt.xticks(fontsize=5)
    plt.yticks(fontsize=5)
    plt.tight_layout()
    if outpath:
        plt.savefig(outpath)
        plt.close()
        
    else:
        return ax
    
    



# TODO: make TF names not squeezed together
def heatmap_coop_score(df, 
                       outpath=None, 
                       figsize=(10, 10)
                       ):
    """
    Reproduces the TF|TF synergy score heatmap using deepISA profile data.
    """
    df=assign_cooperativity(df)
    # remove cooperativity=Independent
    df=df[df["cooperativity"] != "Independent"].reset_index(drop=True)   
    # 1. Expand the 'pair' column back into individual proteins for pivoting
    # 'pair' is sorted 'TF1|TF2', so we split them
    if df.empty:
        logger.warning(f"No interactions passed the significance gate (q < {qval_thresh}).")
        return None
    # 1. Safer splitting of the 'pair' column
    # Using n=1 ensures we only split on the FIRST hyphen found
    split_data = df['tf_pair'].str.split('|', n=1, expand=True)

    df['tf1'] = split_data[0]
    df['tf2'] = split_data[1]
    
    # 2. Pivot to matrix form
    # We use coop_score as the value (mapped to [0, 1] range)
    matrix = df.pivot(index="tf1", columns="tf2", values="coop_score")
    
    # 3. Clean and densify the matrix
    # Remove rows/cols with only NA and ensure it's symmetric for the heatmap
    matrix = matrix.dropna(how='all').dropna(axis=1, how='all')
    
    # Setup Figure and GridSpec
    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(1, 2, width_ratios=[0.05, 1])
    
    cbar_ax = fig.add_subplot(gs[0, 0])  # Color bar on the left
    heatmap_ax = fig.add_subplot(gs[0, 1])  # Heatmap on the right
    
    # 4. Generate Heatmap
    sns.heatmap(
        matrix, 
        cmap="coolwarm", 
        vmin=-1, 
        vmax=1, 
        ax=heatmap_ax, 
        cbar_ax=cbar_ax,
        xticklabels=True, 
        yticklabels=True
    )
    
    # Aesthetic adjustments
    heatmap_ax.set_xlabel("Transcription Factor")
    heatmap_ax.set_ylabel("Transcription Factor")
    plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)
    
    if outpath:
        plt.savefig(outpath, bbox_inches='tight')
        plt.close()
    else:
        plt.show()
        
        
        
        
# TODO: change "plot" to "analyze", then give option to return df or plot.
def plot_motif_distance_by_category(df, outpath=None, figsize=(2.3, 2.3), rotation=30):
    """
    Standardized wrapper to reproduce the TFBS distance vs Cooperativity plot.
    """
    # remove intermediate
    df = df[df["cooperativity"] != "Intermediate"].reset_index(drop=True)
    df = format_cooperativity_categorical(df, categories = ["Independent", "Redundant","Synergistic"])
    plot_violin_with_statistics(
        figsize=figsize,
        df=df,
        x_col="cooperativity",
        y_col="mean_distance",
        x_label="TF pair type",
        y_label="Mean distance\nbetween TFBS pair (bp)",
        title=None,
        rotation=rotation,
        outpath=outpath
    )
    
    
    
