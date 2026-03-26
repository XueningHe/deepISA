import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu, pearsonr
from loguru import logger
from adjustText import adjust_text
from deepISA.utils import get_data_resource,format_cooperativity_categorical
from deepISA.scoring.combi_isa import assign_cooperativity
# TODO: make sure every plotting file has this
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42


# TODO: change "plot" to "analyze", then give option to return df or plot.

# TODO: put legend to lower right
def plot_usf_pfs(df_tf, outpath=None):
    """
    Plots an ECDF (Cumulative Distribution) for USFs, PFs, and Context TFs.
    This clearly shows the shift in populations and the medians.
    """
    df_tf = assign_cooperativity(df_tf)
    # remove nan coop scores
    df_tf = df_tf.dropna(subset=["coop_score"]).copy()
    # 1. Load and expand TF lists
    def load_and_expand(filename):
        path = get_data_resource(filename)
        tfs = pd.read_csv(path, comment='#', header=None)[0].dropna().tolist()
        return set(tfs + [f"{a}::{b}" for a in tfs for b in tfs])

    groups = {
        'USFs': {'set': load_and_expand("universal_stripe_factors.txt"), 'color': '#4169E1'},
        'Pioneers': {'set': load_and_expand("pioneer_factors.txt"), 'color': 'darkorange'},
        'Context-only': {'set': load_and_expand("context_only_tfs.txt"), 'color': '#2ca02c'}
    }

    df = df_tf.dropna(subset=["coop_score"]).copy()
    
    plt.figure(figsize=(3.5, 2.8))
    ax = plt.gca()

    # 2. Plot ECDFs and Calculate Medians
    for name, info in groups.items():
        subset = df[df['tf'].isin(info['set'])]['coop_score']
        if subset.empty: continue
            
        # Plot the ECDF
        sns.ecdfplot(subset, color=info['color'], label=name, lw=1.5, ax=ax)
        
        # Calculate and plot median line
        median_val = subset.median()
        ax.axvline(median_val, color=info['color'], linestyle='--', lw=0.8, alpha=0.6)
        
        # Print stats to console
        print(f"{name} Median: {median_val:.3f} (n={len(subset)})")

    # 3. Aesthetics
    ax.axhline(0.5, color='gray', lw=0.5, ls=':', alpha=0.5) # Median reference line
    
    plt.xlabel("TF Coop Score", fontsize=8)
    plt.ylabel("Cumulative Proportion", fontsize=8)
    plt.title("Score Distribution by TF Category", fontsize=9)
    
    plt.legend(frameon=False, fontsize=7, loc='upper left')
    plt.xticks(fontsize=7); plt.yticks(fontsize=7)
    for s in ['top', 'right']: ax.spines[s].set_visible(False)
    
    plt.tight_layout()
    if outpath:
        plt.savefig(outpath, dpi=300)
        plt.close()
    else:
        plt.show()


# TODO: change "plot" to "analyze", then give option to return df or plot.

def plot_cell_specificity(df_tf, window_size=50, outpath=None):
    """
    Plots a continuous trend of enrichment for cell-type specificity 
    across the range of cooperativity scores.
    """
    df_tf = assign_cooperativity(df_tf)
    # remove nan coop scores
    df_tf = df_tf.dropna(subset=["coop_score"]).copy()
    
    # 1. Prepare Data (Reuse your merging logic)
    disp_path = get_data_resource("gtex.dispersionEstimates.tab")
    df_dispersion = pd.read_csv(disp_path, sep="\t")
    merged = df_tf.merge(df_dispersion, left_on="tf", right_on="symbol", how="inner")
    plot_df = merged.dropna(subset=["coop_score", "gini"]).copy()

    # 2. Define "Specific" and "Ubiquitous" targets
    # We use quantiles so we aren't picking an arbitrary hard number
    high_cutoff = plot_df["gini"].quantile(0.75)
    low_cutoff = plot_df["gini"].quantile(0.25)
    
    plot_df['is_specific'] = (plot_df['gini'] >= high_cutoff).astype(int)
    plot_df['is_ubiquitous'] = (plot_df['gini'] <= low_cutoff).astype(int)

    # 3. Sort by coop_score for rolling calculation
    plot_df = plot_df.sort_values("coop_score")

    # Calculate rolling proportion (The "Enrichment Signal")
    plot_df['rolling_spec'] = plot_df['is_specific'].rolling(window=window_size, center=True).mean()
    plot_df['rolling_ubiq'] = plot_df['is_ubiquitous'].rolling(window=window_size, center=True).mean()

    # 4. Plotting
    plt.figure(figsize=(3, 2.5))
    ax = plt.gca()
    
    # Plot Specificity Enrichment (Upward Trend)
    plt.plot(plot_df["coop_score"], plot_df["rolling_spec"], 
             color="#d62728", label="Cell-Specific (Top 25%)", linewidth=2)
    
    # Plot Ubiquitous Enrichment (Downward Trend)
    plt.plot(plot_df["coop_score"], plot_df["rolling_ubiq"], 
             color="#1f77b4", label="Ubiquitous (Bottom 25%)", linewidth=2)

    # Add a baseline (chance level is 0.25 since we used quartiles)
    plt.axhline(0.25, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)

    # 5. Aesthetics
    plt.xlabel("Cooperativity score", fontsize=7)
    plt.ylabel("Enrichment Proportion", fontsize=7)
    plt.title("Continuous Enrichment Trend", fontsize=8)
    
    plt.xticks([-1, -0.5, 0, 0.5, 1], fontsize=6)
    plt.yticks(fontsize=6)
    plt.legend(frameon=False, fontsize=6)
    
    # Clean spines
    for s in ['top', 'right']: ax.spines[s].set_visible(False)
    
    plt.tight_layout()
    if outpath:
        plt.savefig(outpath, dpi=300)
        plt.close()
    else:
        plt.show()