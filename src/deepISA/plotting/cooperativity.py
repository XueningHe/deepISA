import matplotlib.pyplot as plt
import seaborn as sns
from loguru import logger
import matplotlib.gridspec as gridspec
import re

from deepISA.utils import (
    plot_violin_with_statistics,
    format_cooperativity_categorical,
    apply_plot_style,
    save_or_show
)


import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42


# --- Reuse Helpers from previous refactor ---
# (Assumes apply_plot_style and save_or_show are available in scope)

def prepare_filtered_df(df):
    """Assigns cooperativity and removes 'Independent' entries."""
    return df[df["cooperativity"] != "Independent"].reset_index(drop=True)



# --- Refactored Functions ---


def hist_coop_score(
    df, 
    title=None, 
    xlabel="Cooperativity score", 
    outpath=None, 
    vlines=None, 
    annotations=None, # list of (x, relative_y, text)
    fig_size=(2.3, 2.0)
):
    """Plots a distribution of cooperativity scores with vertical dividers."""
    df = prepare_filtered_df(df)
    
    fig, ax = plt.subplots(figsize=fig_size)
    styles = apply_plot_style(ax, fig_size)

    # Histogram
    sns.histplot(
        df["coop_score"], 
        bins=50, 
        color='steelblue', 
        edgecolor='black', 
        linewidth=0.2 * styles['scale'],
        ax=ax
    )

    # Vertical dividers
    if vlines:
        for x in vlines:
            ax.axvline(x=x, color='grey', linestyle='--', 
                       linewidth=0.7 * styles['scale'], alpha=0.8)

    # Category labels using Relative Y-Coordinates
    if annotations:
        transform = ax.get_xaxis_transform()
        for x, rel_y, label in annotations:
            ax.text(x, rel_y, label, transform=transform, 
                    fontsize=styles['small'], ha='center', va='bottom')

    ax.set_xlabel(xlabel, fontsize=styles['main'])
    ax.set_ylabel('Frequency', fontsize=styles['main'])
    if title:
        ax.set_title(title, fontsize=styles['main'])
    
    return save_or_show(outpath)



def get_prefix(name):
    """
    Extracts the alphabetical prefix from a TF name.
    Example: 'SOX2' -> 'SOX', 'ESRRA' -> 'ESRRA', 'GATA1' -> 'GATA'
    """
    # Find the transition from letters to numbers OR just return the name if no numbers
    match = re.match(r"([a-zA-Z]+)", str(name))
    return match.group(1) if match else str(name)

def get_compressed_labels(label_list):
    """
    Reduces consecutive TFs with the same letter prefix to 'Prefix-s'.
    Example: [SOX2, SOX17, GATA1] -> [SOXs, "", GATA1]
    """
    new_labels = []
    i = 0
    while i < len(label_list):
        current_name = label_list[i]
        prefix = get_prefix(current_name)
        
        # Look ahead to see how many consecutive TFs share this prefix
        j = i + 1
        count = 1
        while j < len(label_list) and get_prefix(label_list[j]) == prefix:
            count += 1
            j += 1
        
        if count > 1:
            # Add the pluralized prefix at the start of the block
            new_labels.append(f"{prefix}s")
            # Fill the rest of the block with empty strings to maintain axis alignment
            new_labels.extend([""] * (count - 1))
        else:
            new_labels.append(current_name)
        
        i = j # Move pointer to the next new prefix block
    return new_labels



def heatmap_coop_score(df, outpath=None, fig_size=(8,8)):
    """
    Generates a compressed TF-TF interaction heatmap by averaging 
    scores for TFs sharing the same prefix.
    """
    df = prepare_filtered_df(df) # Assuming this exists in your environment
    if df.empty:
        print("No interactions passed the significance gate.")
        return None

    # 1. Expand pairs and assign prefixes
    split_data = df['tf_pair'].str.split('|', n=1, expand=True)
    df['tf1_prefix'] = split_data[0].apply(get_prefix)
    df['tf2_prefix'] = split_data[1].apply(get_prefix)
    
    # 2. AGGREGATE: Group by prefixes and calculate the mean score
    # This reduces the number of rows and columns in the final matrix
    compressed_df = df.groupby(['tf1_prefix', 'tf2_prefix'])['coop_score'].mean().reset_index()
    # 3. Pivot the aggregated data
    matrix = compressed_df.pivot(index="tf1_prefix", columns="tf2_prefix", values="coop_score")
    # Optional: Pluralize labels for clarity (SOX -> SOXs)
    matrix.index = [f"{i}s" if len(i) > 0 else i for i in matrix.index]
    matrix.columns = [f"{c}s" if len(c) > 0 else c for c in matrix.columns]
    # 4. Setup Figure
    fig = plt.figure(figsize=fig_size)
    gs = gridspec.GridSpec(1, 2, width_ratios=[0.05, 1], wspace=0.1)
    cbar_ax = fig.add_subplot(gs[0, 0])
    heatmap_ax = fig.add_subplot(gs[0, 1])
    
    styles = apply_plot_style(heatmap_ax, fig_size)
    _ = apply_plot_style(cbar_ax, fig_size) # Apply to colorbar axis too
    
    # 5. Plot Heatmap
    sns.heatmap(
        matrix, cmap="coolwarm", vmin=-1, vmax=1, 
        ax=heatmap_ax, cbar_ax=cbar_ax,
        xticklabels=True, yticklabels=True,
        # CHANGE: Use scaled linewidth instead of hardcoded 0.5
        linewidths=0.1 * styles['scale'] 
    )
    
    # 6. Styling
    # CHANGE: Use styles['main'] instead of hardcoded 12
    heatmap_ax.set_xlabel("TF Family (Average Score)", fontsize=styles['main'])
    heatmap_ax.set_ylabel("TF Family (Average Score)", fontsize=styles['main'])
    
    # CHANGE: Ensure tick labels also use the scaled font size
    heatmap_ax.tick_params(axis='both', which='major', labelsize=styles['small'])
    
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    
    return save_or_show(outpath)



def plot_motif_distance_by_category(df, outpath=None, fig_size=(2.3, 2.3), rotation=30):
    """
    Wrapper for TFBS distance violin plots. 
    Note: plot_violin_with_statistics handles its own internal styling.
    """
    df = format_cooperativity_categorical(
        df, 
        categories=["Independent", "Redundant", "Intermediate", "Synergistic"]
    )
    
    # This utility function appears to handle figure creation internally
    plot_violin_with_statistics(
        figsize=fig_size,
        df=df,
        x_col="cooperativity",
        y_col="mean_distance",
        x_label="TF pair type",
        y_label="Mean distance\nbetween TFBS pair (bp)",
        title=None,
        rotation=rotation,
        outpath=outpath
    )