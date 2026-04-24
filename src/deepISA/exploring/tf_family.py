import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu
from loguru import logger
import warnings
import matplotlib.pyplot as plt

from deepISA.utils import get_data_resource, apply_plot_style, save_or_show

import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42


def get_family_reference():
    """Single source of truth for the JASPAR family reference."""
    ref_path = get_data_resource("HTFs_with_JASPAR_Families.csv")
    df = pd.read_csv(ref_path)
    df["HGNC symbol"] = df["HGNC symbol"].str.upper()
    return df



def annotate_tf_family(df):
    """
    Annotates DF with TF family info. Handles both 'tf_pair' (tf1|tf2) 
    and single 'tf' columns.
    """
    if df.empty:
        return df
    
    if "same_family" in df.columns and "tf1_dbd" in df.columns:
        return df

    ref = get_family_reference()
    temp_df = df.copy()

    # Case 1: Processing TF Pairs
    if 'tf_pair' in temp_df.columns:
        split = temp_df['tf_pair'].str.split('|', expand=True)
        temp_df['tf1'], temp_df['tf2'] = split[0].str.upper(), split[1].str.upper()

        for suffix in ['1', '2']:
            temp_df = pd.merge(temp_df, ref, left_on=f"tf{suffix}", right_on="HGNC symbol", how="left")
            temp_df = temp_df.rename(columns={
                "JASPAR_Class": f"tf{suffix}_class",
                "JASPAR_Family": f"tf{suffix}_family",
                "DBD": f"tf{suffix}_dbd"
            }).drop(columns=["HGNC symbol"], errors='ignore')

        def is_same_fam(row):
            f1, f2 = str(row['tf1_family']), str(row['tf2_family'])
            invalid = ["Not in JASPAR", "nan", "None"]
            if any(x in f1 for x in invalid) or any(x in f2 for x in invalid):
                return False
            return f1 == f2

        temp_df["same_family"] = temp_df.apply(is_same_fam, axis=1)

    # Case 2: Processing Single TFs (for DBD ranking)
    elif 'tf' in temp_df.columns:
        temp_df['tf'] = temp_df['tf'].str.upper()
        temp_df = pd.merge(temp_df, ref, left_on="tf", right_on="HGNC symbol", how="left")
    
    return temp_df




def prepare_plot_df(df):
    """Common cleanup for all plotting functions."""
    return df.dropna(subset=["coop_score"]).copy()



# --- PLOTTING FUNCTIONS ---
def plot_coop_by_tf_pair_family(df, outpath=None, figsize=(2.3, 2.2)):
    """Plots KDE density of coop scores, separated by intra vs inter-family pairs."""
    df = annotate_tf_family(prepare_plot_df(df))
    if df.empty: 
        return

    fig, ax = plt.subplots(figsize=figsize)
    # Automatically get the right font sizes for this specific figsize
    fonts = apply_plot_style(ax, figsize)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sns.kdeplot(
            data=df, x="coop_score", hue='same_family',
            cut=0, common_norm=False, fill=True, linewidth=1.0 * fonts['scale'],
            palette="coolwarm", hue_order=[False, True], ax=ax
        )

    # Stats calculation
    g_true = df[df["same_family"] == True]["coop_score"]
    g_false = df[df["same_family"] == False]["coop_score"]
    
    if len(g_true) > 1 and len(g_false) > 1:
        _, p = mannwhitneyu(g_true, g_false)
        ax.text(0.05, 0.9, f"p={p:.1e}", fontsize=fonts['small'], transform=ax.transAxes)

    ax.set_xlabel("Coop score", fontsize=fonts['main'])
    ax.set_ylabel("Density", fontsize=fonts['main'])
    
    legend = ax.get_legend()
    if legend:
        legend.set_title("Same Family?", prop={'size': fonts['small']})
        # Scale the legend text as well
        for text in legend.get_texts():
            text.set_fontsize(fonts['small'])
    
    return save_or_show(outpath)


def plot_coop_by_dbd(df, outpath=None, top_n=15, figsize=(3.5, 4)):
    """Ranks DBDs by median cooperativity score."""
    df = annotate_tf_family(prepare_plot_df(df))
    if df.empty:
        return
        
    # Filter for top N most frequent DBDs
    top_dbds = df['DBD'].value_counts().nlargest(top_n).index
    plot_df = df[df['DBD'].isin(top_dbds)]
    order = plot_df.groupby("DBD")["coop_score"].median().sort_values(ascending=False).index

    fig, ax = plt.subplots(figsize=figsize)
    fonts = apply_plot_style(ax, figsize)

    sns.boxplot(
        data=plot_df, x="coop_score", y='DBD', order=order,
        hue='DBD', hue_order=order, palette="flare", 
        linewidth=0.6 * fonts['scale'],
        fliersize=0.5 * fonts['scale'],
        width=0.7, 
        legend=False, ax=ax, orientation='horizontal'
    )
    
    ax.axvline(0, color='black', linestyle='--', linewidth=0.8 * fonts['scale'], alpha=0.5)
    ax.set_title("TF Family Ranking", fontsize=fonts['main'])
    ax.set_xlabel("Cooperativity Score", fontsize=fonts['main'])
    ax.set_ylabel("", fontsize=fonts['main']) # DBD labels are clear enough
    
    sns.despine()
    return save_or_show(outpath)


def plot_intra_family_coop_score(df, min_pairs=10, outpath=None, figsize=None):
    """Plots distributions for families with multiple internal pairs."""
    df = annotate_tf_family(prepare_plot_df(df))
    plot_df = df[df["same_family"] == True].dropna(subset=["tf1_family"])
    
    counts = plot_df["tf1_family"].value_counts()
    keep = counts[counts >= min_pairs].index
    plot_df = plot_df[plot_df["tf1_family"].isin(keep)]
    
    if plot_df.empty:
        logger.info(f"No families meet the min_pairs={min_pairs} threshold.")
        return

    order = plot_df.groupby("tf1_family")["coop_score"].median().sort_values(ascending=False).index
    
    # If no figsize is provided, we auto-calculate height based on the number of families
    if figsize is None:
        figsize = (3.5, 0.4 * len(keep))
        
    fig, ax = plt.subplots(figsize=figsize)
    fonts = apply_plot_style(ax, figsize)

    sns.boxplot(
        data=plot_df, 
        y="tf1_family", 
        x="coop_score", 
        order=order, 
        palette="vlag", 
        linewidth=0.8 * fonts['scale'],
        fliersize=0, 
        ax=ax, 
        hue="tf1_family",  # Explicitly assign hue to the y-axis variable
        legend=False,      # Disable the redundant legend
        orientation='horizontal'
    )
    
    sns.stripplot(
        data=plot_df, y="tf1_family", x="coop_score", order=order, 
        color="black", 
        size=2 * fonts['scale'], alpha=0.3, ax=ax, orient='h'
    )

    ax.axvline(0, color='grey', linestyle='--', linewidth=0.5 * fonts['scale'])
    ax.set_title(f"Intra-Family Synergy (n >= {min_pairs})", fontsize=fonts['main'])
    ax.set_xlabel("Coop Score", fontsize=fonts['main'])
    ax.set_ylabel("TF Family", fontsize=fonts['main'])
    
    sns.despine()
    return save_or_show(outpath)