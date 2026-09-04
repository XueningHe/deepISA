import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
from scipy.stats import pearsonr
from loguru import logger
from deepISA.utils import (
    plot_violin_with_statistics, 
    get_data_resource,
    format_cooperativity_categorical,
    apply_plot_style, 
    save_or_show,
    remove_if_exists
)

import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42

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
            gc_percentage = total_gc / total_all * 100
            
            tf_gc_data.append({'tf': tf_name, 'GC': gc_percentage})
        i += 1
    return pd.DataFrame(tf_gc_data).groupby('tf').mean().reset_index()


def plot_motif_gc_by_coop(df_tf, title="Motif GC%", outpath=None, figsize=(2.3, 2.1)):
    remove_if_exists(outpath, label="Motif GC plot")
    jaspar_path = get_data_resource("JASPAR2026_CORE_non-redundant_pfms_jaspar.txt")
    df_gc = parse_jaspar_pfms(jaspar_path)
    

    df = df_gc.merge(df_tf[['tf', 'cooperativity']], on='tf', how='inner')
    df = format_cooperativity_categorical(df)

    # Note: plot_violin_with_statistics should be updated internally to use apply_plot_style
    return plot_violin_with_statistics(figsize, df, "cooperativity", "GC", 
                                     "TF Cooperativity", "Motif GC%", title, 30, outpath)



def plot_coop_vs_importance(df_tf_coop, df_importance, x_col="coop_score", y_col="ks_d_isa_t0", 
                            title="Cooperativity vs Importance", outpath=None, figsize=(2.3, 1.6)):
    remove_if_exists(outpath, label="Cooperativity vs Importance plot")
    # drop nan
    df_tf_coop = df_tf_coop.dropna(subset=["coop_score"]).reset_index(drop=True)
    df = df_tf_coop.merge(df_importance, on="tf")
    if df.empty:
        logger.info("Warning: No matching TFs found for importance plot.")
        return None
    df_coop = df[df["cooperativity"] != "Independent"].copy()
    df_independent = df[df["cooperativity"] == "Independent"].copy()
    df_coop = format_cooperativity_categorical(df_coop, ["Redundant", "Intermediate", "Synergistic"])
    fig, axes = plt.subplots(1, 2, gridspec_kw={'width_ratios': [5, 1]}, 
                             sharey=True, figsize=figsize, constrained_layout=True)
    styles = apply_plot_style(axes[0], figsize)
    _ = apply_plot_style(axes[1], figsize)
    palette = {"Intermediate": "gray", "Synergistic": "#d62728", "Redundant": "#1f77b4"}
    sns.scatterplot(x=x_col, y=y_col, data=df_coop, hue="cooperativity", 
                    ax=axes[0], palette=palette, s=5*styles['scale'], legend=False)

    if len(df_coop) > 2:
        r, p = pearsonr(df_coop[x_col], df_coop[y_col])
        axes[0].text(0.3, 0.6, f"R={r:.2f}\nP={p:.2e}", transform=axes[0].transAxes, fontsize=styles['small'])
    
    sns.stripplot(x=x_col, y=y_col, data=df_independent, ax=axes[1], color="black", size=2*styles['scale'])

    axes[0].set_xlabel("Coop Score", fontsize=styles['main'])
    axes[0].set_ylabel(y_col, fontsize=styles['main'])
    axes[0].set_title(title, fontsize=styles['main'])
    axes[1].set_xticks([]); axes[1].set_xlabel("")
    
    axes[0].set_ylim(df[y_col].min() * 1.2, df[y_col].max() * 1.3)
    return save_or_show(outpath)





def plot_partner_specificity(
    df_tf_pair,
    df_tf,
    top_n=5,
    min_partners=10,
    title="Partner Specificity Comparison",
    xlabel="Top 5 Interactors Contribution Ratio",
    ylabel="Density",
    outpath=None,
    figsize=(3.5, 2.5)
):
    remove_if_exists(outpath, label="Partner Specificity plot")
    # 1. Prepare mirrored pair data
    df_mirrored = df_tf_pair.copy()
    pairs = df_mirrored['tf_pair'].str.split('|', expand=True)
    
    df_long = pd.concat([
        df_mirrored[['abs_i_sum']].assign(tf=pairs[0], partner=pairs[1]),
        df_mirrored[['abs_i_sum']].assign(tf=pairs[1], partner=pairs[0])
    ], ignore_index=True)
    
    # 2. Calculate specificity ratios
    def get_ratio(group):
        if len(group) < min_partners: return None
        return group.sort_values('abs_i_sum', ascending=False).head(top_n)['abs_i_sum'].sum() / group['abs_i_sum'].sum()

    # Fixed: Safe reset_index for Series output
    res_series = df_long.groupby('tf')['abs_i_sum'].apply(lambda x: get_ratio(df_long.loc[x.index]))
    df_res = res_series.dropna().reset_index()
    df_res.columns = ['tf', 'specificity_ratio']
    
    df_res = df_res.merge(df_tf[['tf', 'cooperativity']], on='tf')
    df_plot = df_res[df_res['cooperativity'].isin(['Synergistic', 'Redundant', 'Intermediate'])].copy()
    
    if df_plot.empty:
        return None

    # 3. Plotting
    fig, ax = plt.subplots(figsize=figsize)
    styles = apply_plot_style(ax, figsize)
    
    palette = {
        "Synergistic": "#d62728", 
        "Redundant": "#1f77b4", 
        "Intermediate": "gray", 
    }
    sns.kdeplot(
        data=df_plot, 
        x="specificity_ratio", 
        hue="cooperativity",
        fill=True,            # Set to True for a shaded area under the curve
        common_norm=False,    # Keeps categories comparable regardless of group size
        palette=palette,
        alpha=0.3,            # Adjust transparency of the fill
        linewidth=styles['scale'],
        ax=ax
    )
    
    ax.set_title(title, fontsize=styles['main'])
    ax.set_xlabel(xlabel, fontsize=styles['main'])
    ax.set_ylabel(ylabel, fontsize=styles['main'])
    
    # set x limits to [0, 1] since it's a ratio
    ax.set_xlim(0, 1)
    
    if ax.get_legend():
        plt.setp(ax.get_legend().get_texts(), fontsize=styles['small'])
        plt.setp(ax.get_legend().get_title(), fontsize=styles['small'])

    return save_or_show(outpath)