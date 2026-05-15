import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import matplotlib.gridspec as gridspec
import re

from deepISA.utils import (
    plot_violin_with_statistics,
    format_cooperativity_categorical,
    apply_plot_style,
    save_or_show,
    remove_if_exists
)


import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42


# --- Reuse Helpers from previous refactor ---


def prepare_filtered_df(df):
    """Assigns cooperativity and removes 'Independent' entries."""
    return df[df["cooperativity"] != "Independent"].reset_index(drop=True)



# --- Refactored Functions ---


def hist_coop_score(
    df, 
    title=None, 
    xlabel="Coop score", 
    outpath=None, 
    vlines=None, 
    annotations=None, # list of (x, relative_y, text)
    figsize=(2.3, 2.0)
):
    """Plots a distribution of cooperativity scores with vertical dividers."""
    remove_if_exists(outpath, label="Coop Score Histogram")
    df = df[df["cooperativity"] != "Independent"].reset_index(drop=True)
    fig, ax = plt.subplots(figsize=figsize)
    styles = apply_plot_style(ax, figsize)
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
    Extracts the family prefix. 
    Handles: E2F1 -> E2F, HOXA1 -> HOX, SOX2 -> SOX.
    """
    name = str(name).strip()
    # 1. Handle E2F specifically as it's a common outlier
    if name.startswith("E2F"):
        return "E2F"
    # 2. General Robust Regex:
    # Captures the leading alphabetical string, but stops if it 
    # encounters a trailing number or a single letter suffix (like A, B, 1).
    # [A-Z]{2,} matches at least 2 uppercase letters to avoid single-letter 'E'
    match = re.match(r"^([A-Z]{2,})", name)
    if match:
        prefix = match.group(1)
        # Special case: If prefix is 'HOXA' or 'HOXB', reduce to 'HOX'
        if prefix.startswith("HOX") and len(prefix) > 3:
            return "HOX"
        return prefix
    return name



def heatmap_coop_score(df, outpath=None, figsize=(16,16)):
    """
    Generates a compressed TF-TF interaction heatmap by averaging 
    scores for TFs sharing the same prefix.
    """
    remove_if_exists(outpath, label="Coop Score Heatmap")
    df = df[df["cooperativity"] != "Independent"].reset_index(drop=True)
    if df.empty:
        print("No interactions passed the significance gate.")
        return None
    
    
    split_tfs = df['tf_pair'].str.split('|', expand=True)
    df['tf1'] = split_tfs[0]
    df['tf2'] = split_tfs[1]
    
    all_tfs = set(df['tf1']).union(set(df['tf2']))
    
    # 2. Map TFs to prefixes and count family occurrences in this specific dataset
    tf_to_prefix = {tf: get_prefix(tf) for tf in all_tfs}
    prefix_counts = pd.Series(list(tf_to_prefix.values())).value_counts()
    
    # 3. Create a labeling map: Pluralize IF count > 1, else keep original name
    def determine_label(tf):
        prefix = tf_to_prefix[tf]
        if prefix_counts.get(prefix, 0) > 1:
            return f"{prefix}s"
        return tf

    df['tf1_label'] = df['tf1'].apply(determine_label)
    df['tf2_label'] = df['tf2'].apply(determine_label)
    
    mirrored_df = df.copy()
    mirrored_df['tf1_label'], mirrored_df['tf2_label'] = df['tf2_label'], df['tf1_label']
    full_df = pd.concat([df, mirrored_df]).drop_duplicates()
    # 4. AGGREGATE: Group by the new labels and calculate the mean score
    compressed_df = full_df.groupby(['tf1_label', 'tf2_label'])['coop_score'].mean().reset_index()
    matrix = compressed_df.pivot(index="tf1_label", columns="tf2_label", values="coop_score")
    
    # 5. Setup Figure with colorbar on the right and extra spacing for labels
    fig = plt.figure(figsize=figsize)
    # Heatmap is first (index 0), Colorbar is second (index 1)
    # wspace=0.2 provides breathing room between the heatmap and colorbar labels
    gs = gridspec.GridSpec(1, 2, width_ratios=[1, 0.02], wspace=0.2)
    heatmap_ax = fig.add_subplot(gs[0, 0])
    cbar_ax = fig.add_subplot(gs[0, 1])
    styles = apply_plot_style(heatmap_ax, figsize)
    
    # 6. Plot Heatmap
    sns.heatmap(
        matrix, cmap="coolwarm", vmin=-1, vmax=1, 
        ax=heatmap_ax, cbar_ax=cbar_ax,
        xticklabels=True, yticklabels=True,
        linewidths=0.1 * styles['scale'] 
    )
    
    # 7. Final Styling
    # Remove top and right frame lines as requested
    for spine in heatmap_ax.spines.values():
        spine.set_visible(False)
    
    heatmap_ax.tick_params(axis='both', which='both', length=0)
    cbar_ax.tick_params(axis='both', which='both', length=0)
    heatmap_ax.set_xlabel("TF Family (Average Score)", fontsize=styles['main'] * 0.5)
    heatmap_ax.set_ylabel("TF Family (Average Score)", fontsize=styles['main'] * 0.5)
    
    # Scale down tick labels to prevent overlap seen in Screenshot 2026-05-05 at 16.24.32.jpg
    heatmap_ax.tick_params(axis='both', which='major', labelsize=styles['small'] * 0.3)
    cbar_ax.tick_params(labelsize=styles['small'] * 0.3)
    
    plt.setp(heatmap_ax.get_xticklabels(), rotation=90)
    plt.setp(heatmap_ax.get_yticklabels(), rotation=0)
    
    return save_or_show(outpath)



def plot_motif_distance_by_category(df, outpath=None, figsize=(2.3, 2.3), rotation=30):
    """
    Wrapper for TFBS distance violin plots. 
    Note: plot_violin_with_statistics handles its own internal styling.
    """
    remove_if_exists(outpath, label="Motif Distance by Category")
    
    # remove 'Independent' category and reset index
    df = prepare_filtered_df(df)
    # plot scatter plot x = coop_score, y = mean_distance, hue = cooperativity
    sns.scatterplot(data=df, x="coop_score", y="mean_distance", hue="cooperativity")
    # save
    save_or_show(outpath)
    # This utility function appears to handle figure creation internally
    # plot_violin_with_statistics(
    #     figsize=figsize,
    #     df=df,
    #     x_col="cooperativity",
    #     y_col="mean_distance",
    #     x_label="TF pair type",
    #     y_label="Mean distance\nbetween TFBS pair (bp)",
    #     title=None,
    #     rotation=rotation,
    #     outpath=outpath
    # )