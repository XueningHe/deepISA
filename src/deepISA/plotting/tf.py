import matplotlib.pyplot as plt
import pandas as pd
from adjustText import adjust_text
from scipy.stats import pearsonr
from matplotlib.lines import Line2D
import seaborn as sns
import numpy as np

from deepISA.utils import (
    plot_violin_with_statistics, 
    get_data_resource,
    format_cooperativity_categorical
)

from deepISA.scoring.combi_isa import assign_cooperativity


def compare_tf_importance(
    data, 
    x_col, 
    y_col, 
    label_col="tf",
    x_label=None,
    y_label=None,
    title="TF Importance Comparison",
    outpath=None, 
    x_threshold=0.3, 
    y_threshold=0.3,
    # Aesthetic arguments directly in the signature
    fig_size=(3.5, 3.5),
    label_size=5,
    marker_size=10,
    text_alpha=0.7,
    font_size = 7,
    dpi=300
):
    """
    Generates a scatter plot comparing TF importance between two tracks.
    Highlights context-specific TFs based on distance from the diagonal.
    """
    df = data.copy()
    # Calculate bias (Distance from diagonal)
    df["bias_value"] = df[x_col] - df[y_col]

    plt.figure(figsize=fig_size, dpi=dpi)
    ax = plt.gca()
    
    # Clean spines
    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)
    for spine in ['bottom', 'left']:
        ax.spines[spine].set_linewidth(0.5)

    # Main Scatter
    scatter = plt.scatter(
        x=df[x_col],
        y=df[y_col],
        c=df["bias_value"],
        cmap="coolwarm",
        s=marker_size,
        edgecolor="none"
    )

    # Annotation Logic
    texts = []
    for _, row in df.iterrows():
        x, y = row[x_col], row[y_col]
        # Label if it exceeds thresholds or is extremely high impact
        if (x - y > x_threshold) or (y - x > y_threshold) or (abs(x + y) > 0.95):
            texts.append(plt.text(x, y, row[label_col], fontsize=label_size, alpha=text_alpha))

    if texts:
        adjust_text(texts, 
                    arrowprops=dict(arrowstyle='-', color='gray', lw=0.5))

    # Formatting
    x_label = x_label or str(x_col)
    y_label = y_label or str(y_col)
    plt.xlabel(x_label, fontsize=font_size)
    plt.ylabel(y_label, fontsize=font_size)
    plt.title(title, fontsize=font_size)
    plt.xticks(fontsize=font_size)
    plt.yticks(fontsize=font_size)
    # Reference lines
    plt.axvline(x=0, color="black", linewidth=0.5, linestyle="--", alpha=0.3)
    plt.axhline(y=0, color="black", linewidth=0.5, linestyle="--", alpha=0.3)
    # Dynamic Diagonal
    if not df.empty:
        min_val = min(df[x_col].min(), df[y_col].min())
        max_val = max(df[x_col].max(), df[y_col].max())
    else:
        min_val, max_val = -1, 1
        
    plt.plot([min_val, max_val], [min_val, max_val], color="gray", linewidth=0.5, linestyle="--", alpha=0.3)

    # Colorbar: Always draw it so the plot area doesn't jump around in size
    cbar = plt.colorbar(scatter)
    cbar.set_label(r"$\Delta$ Importance", fontsize=label_size)
    cbar.ax.tick_params(labelsize=label_size)

    plt.tight_layout()
    if outpath:
        plt.savefig(outpath, bbox_inches='tight')
        plt.close()
    else:
        plt.show()
        return plt.gcf()


def parse_jaspar_pfms(jaspar_path):
    """
    Parses a JASPAR PFM file and calculates GC content for each TF.
    """
    tf_gc_data = []
    with open(jaspar_path, 'r') as f:
        lines = f.readlines()
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if line.startswith(">"):
            # Header line: >ID Name
            header_parts = line[1:].split('\t')
            tf_id = header_parts[0]
            tf_name = header_parts[1].upper()
            # Read A, C, G, T rows
            # JASPAR format usually has counts inside [ ]
            counts = {}
            for base in ['A', 'C', 'G', 'T']:
                i += 1
                row = lines[i].strip().split('[')[1].split(']')[0].split()
                counts[base] = np.array([float(x) for x in row])
            
            # Calculate GC Content: (C + G) / (A + C + G + T)
            total_gc = counts['C'].sum() + counts['G'].sum()
            total_all = sum(counts[b].sum() for b in ['A', 'C', 'G', 'T'])
            gc_fraction = total_gc / total_all
            
            tf_gc_data.append({'tf': tf_name, 'GC': gc_fraction})
        i += 1
            
    return pd.DataFrame(tf_gc_data)


# TODO: why so poor reprodicibility....
def plot_motif_gc_by_coop(
    df_tf, 
    title="Motif GC%", 
    xlabel="TFBS Type", 
    ylabel="Motif GC%", 
    outpath=None,
    fig_size=(2.3, 2.1)
):
    # 1. Parse JASPAR file to get GC content per TF
    jaspar_path = get_data_resource("JASPAR2026_CORE_non-redundant_pfms_jaspar.txt")
    df_gc = parse_jaspar_pfms(jaspar_path)
    
    # 2. Merge with cooperativity profile
    # Note: Ensure TF names in df_tf match those in JASPAR (case-sensitive)
    df_tf=assign_cooperativity(df_tf)
    df = df_gc.merge(df_tf[['tf', 'cooperativity']], on='tf', how='inner')
    
    # 3. Categorize and Plot
    df = format_cooperativity_categorical(df)
    # Use your existing plotting function
    plot_violin_with_statistics(
        fig_size, df, "cooperativity", "GC", 
        xlabel, ylabel, title, 30, outpath
    )


# TODO: change "plot" to "analyze", then give option to return df or plot.

# TODO: remove independent TFs
def plot_coop_vs_importance(
    df_tf_coop,
    df_importance,
    x_col="coop_score",
    y_col="mean_isa_t0",
    title="Cooperativity vs Importance",
    xlabel=None,
    ylabel=None,
    outpath=None,
    fig_size=(2.3, 1.6)
):
    """
    Creates a combined scatter plot (for Synergistic/Redundant/Intermediate) 
    and a strip plot (for Independent) TFs.
    """
    if xlabel is None:
        xlabel = x_col
    if ylabel is None:
        ylabel = y_col
    
    df_tf_coop = assign_cooperativity(df_tf_coop)
    # Merge dataframes on 'tf'
    df = df_tf_coop.merge(df_importance, on="tf")

    # Split data
    df_coop = df[df["cooperativity"] != "Independent"].copy()
    df_independent = df[df["cooperativity"] == "Independent"].copy()

    #TODO: already done by the assignation of cooperativity

    # Use predefined categories
    df_coop=format_cooperativity_categorical(df_coop, ["Redundant", "Intermediate", "Synergistic"])

    # Create subplots
    fig, axes = plt.subplots(1, 2, gridspec_kw={'width_ratios': [5, 1]}, 
                             sharey=True, figsize=fig_size, constrained_layout=True)
    
    # Apply global spine and tick aesthetics
    # TODO: move to utils
    for ax in axes:
        for spine in ax.spines.values():
            spine.set_linewidth(0.5)
        ax.tick_params(axis='both', which='major', labelsize=5, width=0.5, length=2)

    palette = {"Intermediate": "gray", "Synergistic": "#d62728", "Redundant": "#1f77b4"}

    # Main Scatter Plot
    sns.scatterplot(
        x=x_col, 
        y=y_col, 
        data=df_coop, 
        hue="cooperativity", 
        ax=axes[0], 
        palette=palette, 
        s=5,
        legend=False
    )

    # Correlation Stats if there are enough data
    if len(df_coop) > 2:
        r, p = pearsonr(df_coop[x_col], df_coop[y_col])
        axes[0].text(0.3, 0.6, f"Pearson R={r:.2f}\nP={p:.2e}", 
                     ha='center', 
                     va='center', 
                     transform=axes[0].transAxes, 
                 fontsize=5)
    
    # # Legend Handling - Fix: uniform dot size
    # handles, labels = axes[0].get_legend_handles_labels()
    # independent_dot = Line2D([0], [0], marker='o', color='black', linestyle='None', markersize=3, label='Independent')
    
    # axes[0].legend(handles + [independent_dot], labels + ['Independent'], title="TF type", fontsize=5, title_fontsize=5)
    
    # Strip Plot for Independent
    sns.stripplot(
        x=x_col, 
        y=y_col, 
        data=df_independent, 
        ax=axes[1], 
        color="black", 
        size=2
    )

    # Formatting
    axes[1].set_xticks([])
    axes[1].set_xlabel("")
    axes[0].set_xlabel(xlabel, fontsize=7)
    axes[0].set_ylabel(ylabel, fontsize=7)
    axes[0].set_title(title, fontsize=7)

    # Set consistent y-limits
    ymax = df[y_col].max() * 1.3
    ymin = df[y_col].min() * 1.2
    axes[0].set_ylim(ymin, ymax)

    if outpath:
        plt.savefig(outpath, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()
        return fig
    
    
