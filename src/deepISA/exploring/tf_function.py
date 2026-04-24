import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from deepISA.utils import get_data_resource, apply_plot_style, save_or_show
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42



def prepare_coop_df(df_tf):
    """Standard preprocessing for Cooperativity data."""
    return df_tf.dropna(subset=["coop_score"]).copy()


def load_and_expand_tfs(filename):
    """Loads TF list and generates homodimer strings."""
    path = get_data_resource(filename)
    tfs = pd.read_csv(path, comment='#', header=None)[0].dropna().tolist()
    # Using a set for O(1) lookups
    return set(tfs + [f"{a}::{b}" for a in tfs for b in tfs])

# --- Main Plotting Functions ---

def plot_usf_pfs(df_tf, fig_size=(3.5, 2.8), outpath=None):
    """Plots ECDF for USFs, PFs, and Context TFs with dynamic styling."""
    df = prepare_coop_df(df_tf)
    
    groups = {
        'USFs': {'set': load_and_expand_tfs("universal_stripe_factors.txt"), 'color': '#4169E1'},
        'Pioneers': {'set': load_and_expand_tfs("pioneer_factors.txt"), 'color': 'darkorange'},
        'Context-only': {'set': load_and_expand_tfs("context_only_tfs.txt"), 'color': '#2ca02c'}
    }

    fig, ax = plt.subplots(figsize=fig_size)
    styles = apply_plot_style(ax, fig_size)

    for name, info in groups.items():
        subset = df[df['tf'].isin(info['set'])]['coop_score']
        if subset.empty:
            continue
            
        sns.ecdfplot(subset, color=info['color'], label=name, lw=1.5 * styles['scale'], ax=ax)
        
        # Median vertical line
        median_val = subset.median()
        ax.axvline(median_val, color=info['color'], linestyle='--', 
                   lw=0.8 * styles['scale'], alpha=0.6)

    ax.axhline(0.5, color='gray', lw=0.5 * styles['scale'], ls=':', alpha=0.5)
    
    ax.set_xlabel("TF Coop Score", fontsize=styles['main'])
    ax.set_ylabel("Cumulative Proportion", fontsize=styles['main'])
    ax.set_title("Score Distribution by TF Category", fontsize=styles['main'])
    
    # Legend moved to lower right per TODO
    ax.legend(frameon=False, fontsize=styles['small'], loc='lower right')
    
    return save_or_show(outpath)


def plot_cell_specificity(df_tf, window_size=50, fig_size=(3, 2.5), outpath=None):
    """Plots continuous trend of enrichment for cell-type specificity."""
    df = prepare_coop_df(df_tf)
    
    # Load and merge dispersion data
    disp_path = get_data_resource("gtex.dispersionEstimates.tab")
    df_dispersion = pd.read_csv(disp_path, sep="\t")
    plot_df = df.merge(df_dispersion, left_on="tf", right_on="symbol", how="inner")
    plot_df = plot_df.dropna(subset=["gini"]).copy()

    # Define quantiles
    high_cutoff = plot_df["gini"].quantile(0.75)
    low_cutoff = plot_df["gini"].quantile(0.25)
    
    plot_df['is_specific'] = (plot_df['gini'] >= high_cutoff).astype(int)
    plot_df['is_ubiquitous'] = (plot_df['gini'] <= low_cutoff).astype(int)
    plot_df = plot_df.sort_values("coop_score")

    # Rolling enrichment
    plot_df['rolling_spec'] = plot_df['is_specific'].rolling(window=window_size, center=True).mean()
    plot_df['rolling_ubiq'] = plot_df['is_ubiquitous'].rolling(window=window_size, center=True).mean()

    fig, ax = plt.subplots(figsize=fig_size)
    styles = apply_plot_style(ax, fig_size)
    
    # Plotting lines
    lw = 2 * styles['scale']
    ax.plot(plot_df["coop_score"], plot_df["rolling_spec"], 
             color="#d62728", label="Cell-Specific (Top 25%)", linewidth=lw)
    ax.plot(plot_df["coop_score"], plot_df["rolling_ubiq"], 
             color="#1f77b4", label="Ubiquitous (Bottom 25%)", linewidth=lw)

    ax.axhline(0.25, color='gray', linestyle='--', linewidth=0.8 * styles['scale'], alpha=0.5)

    ax.set_xlabel("Cooperativity score", fontsize=styles['main'])
    ax.set_ylabel("Enrichment Proportion", fontsize=styles['main'])
    ax.set_title("Continuous Enrichment Trend", fontsize=styles['main'])
    
    ax.set_xticks([-1, -0.5, 0, 0.5, 1])
    ax.legend(frameon=False, fontsize=styles['small'])
    
    return save_or_show(outpath)