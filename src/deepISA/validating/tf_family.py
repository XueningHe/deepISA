import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu
from loguru import logger
import warnings
import matplotlib.pyplot as plt

from deepISA.utils import get_data_resource
from deepISA.scoring.combi_isa import assign_cooperativity


def annotate_tf_family(df):
    """
    Annotates TF pairs with Class, Family, and DBD info.
    Adds 'same_family' column based on DBD.
    """
    # Guard clause: if already annotated, skip
    if "same_family" in df.columns and "tf1_dbd" in df.columns:
        return df

    ref_path = get_data_resource("HTFs_with_JASPAR_Families.csv")
    df_family = pd.read_csv(ref_path)
    df_family["HGNC symbol"] = df_family["HGNC symbol"].str.upper()

    temp_df = df.copy()
    split_cols = temp_df['tf_pair'].str.split('|', expand=True)
    temp_df['tf1'] = split_cols[0].str.upper()
    temp_df['tf2'] = split_cols[1].str.upper()

    # Merge for TF 1
    temp_df = pd.merge(temp_df, df_family, left_on="tf1", right_on="HGNC symbol", how="left")
    temp_df.rename(columns={
        "JASPAR_Class": "tf1_class", 
        "JASPAR_Family": "tf1_family", 
        "DBD": "tf1_dbd"
    }, inplace=True)
    temp_df.drop(columns=["HGNC symbol"], inplace=True, errors='ignore')
    
    # Merge for TF 2
    temp_df = pd.merge(temp_df, df_family, left_on="tf2", right_on="HGNC symbol", how="left")
    temp_df.rename(columns={
        "JASPAR_Class": "tf2_class", 
        "JASPAR_Family": "tf2_family", 
        "DBD": "tf2_dbd"
    }, inplace=True)
    temp_df.drop(columns=["HGNC symbol"], inplace=True, errors='ignore')

    def check_same(row):
        f1, f2 = str(row['tf1_family']), str(row['tf2_family'])
        # Consider them same ONLY if they both exist and are not 'Not in JASPAR'
        if "Not in JASPAR" in f1 or "Not in JASPAR" in f2 or f1 == 'nan' or f2 == 'nan':
            return False
        return f1 == f2

    temp_df["same_family"] = temp_df.apply(check_same, axis=1).astype(bool)
    return temp_df


# TODO: change "plot" to "analyze", then give option to return df or plot.

def plot_coop_by_tf_pair_family(df, outpath=None):
    """Plots KDE density. Automatically runs annotation if columns are missing."""
    df = assign_cooperativity(df)
    # remove nan coop scores
    df = df.dropna(subset=["coop_score"]).copy()
    df = annotate_tf_family(df)
    plot_df = df.dropna(subset=["coop_score"]).copy()
    if len(plot_df) == 0: return

    plt.figure(figsize=(2.3, 2.2))
    ax = plt.gca()
    for spine in ax.spines.values():
        spine.set_linewidth(0.5)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        sns.kdeplot(
            data=plot_df, x="coop_score", hue='same_family',
            cut=0, common_norm=False, fill=True, linewidth=0.5,
            palette="coolwarm", hue_order=[False, True]
        )

    g_true = plot_df[plot_df["same_family"] == True]["coop_score"]
    g_false = plot_df[plot_df["same_family"] == False]["coop_score"]
    
    if len(g_true) > 1 and len(g_false) > 1 and g_true.var() > 0 and g_false.var() > 0:
        _, p = mannwhitneyu(g_true, g_false)
        plt.text(0.1, 0.8, f"p={p:.1e}", fontsize=5, transform=ax.transAxes)

    plt.xticks(fontsize=5); plt.yticks(fontsize=5)
    plt.xlabel("Coop score", fontsize=7); plt.ylabel("Density", fontsize=7)
    
    legend = ax.get_legend()
    if legend:
        legend.set_title("Same TF family?", prop={'size': 5})
        for t in legend.get_texts(): t.set_fontsize(5)
            
    plt.tight_layout()
    if outpath:
        plt.savefig(outpath, dpi=300); plt.close()
    else:
        plt.show()
    return df




# TODO: change "plot" to "analyze", then give option to return df or plot.

def plot_coop_by_dbd(df, outpath=None):
    """
    Plots Boxplot by DBD. Fixed color order by syncing 'order' and 'hue_order'.
    """
    ref_path = get_data_resource("HTFs_with_JASPAR_Families.csv")
    df_family = pd.read_csv(ref_path)
    df_family["HGNC symbol"] = df_family["HGNC symbol"].str.upper()
    
    # merge
    df=assign_cooperativity(df)
    # remove nan coop scores
    df = df.dropna(subset=["coop_score"]).copy()
    df = pd.merge(df, df_family, left_on="tf", right_on="HGNC symbol", how="left")
    
    # 1. Prepare and Clean Data
    plot_df = df.dropna(subset=["coop_score"]).copy()

    # 2. Determine Order by Median
    top_dbds_idx = plot_df['DBD'].value_counts().nlargest(15).index
    filtered_df = plot_df[plot_df['DBD'].isin(top_dbds_idx)]
    
    sorted_order = (
        filtered_df.groupby("DBD")["coop_score"]
        .median()
        .sort_values(ascending=False)
        .index
    )

    # 3. Plotting
    plt.figure(figsize=(3.5, 4))
    ax = plt.gca()

    # CRITICAL FIX: Add hue_order to match the sorted_order
    sns.boxplot(
        data=filtered_df, 
        x="coop_score", 
        y='DBD', 
        order=sorted_order,
        hue='DBD',
        hue_order=sorted_order, # This ensures color follows the rank
        palette="flare", 
        linewidth=0.6, 
        fliersize=0.5,
        width=0.7,
        legend=False
    )

    # 4. Aesthetics
    plt.axvline(0, color='black', linestyle='--', linewidth=0.8, alpha=0.5)
    plt.title("TF Family Cooperativity Ranking", fontsize=8)
    plt.xlabel("Cooperativity Score", fontsize=7)
    plt.ylabel("", fontsize=7)
    plt.xticks(fontsize=6)
    plt.yticks(fontsize=6)
    
    for s in ['top', 'right']: ax.spines[s].set_visible(False)
    plt.tight_layout()
    
    if outpath:
        plt.savefig(outpath, dpi=300)
        plt.close()
    else:
        plt.show()



# TODO: change "plot" to "analyze", then give option to return df or plot.

def plot_intra_family_coop_score(df, min_pairs=10, outpath=None):
    """
    Subsets for pairs in the same family and plots the distribution of 
    cooperative scores across different TF families.
    """
    # 1. Ensure annotation exists
    df = assign_cooperativity(df)
    # remove nan coop scores
    df = df.dropna(subset=["coop_score"]).copy()
    df = annotate_tf_family(df)
    
    # 2. Subset for 'same_family' and valid scores
    family_df = df[df["same_family"] == True].copy()
    family_df = family_df.dropna(subset=["coop_score", "tf1_family"])
    
    if family_df.empty:
        logger.info("No 'same_family' pairs found to plot.")
        return
    
    # 3. Filter for families with at least N pairs for statistical relevance
    counts = family_df["tf1_family"].value_counts()
    keep_families = counts[counts >= min_pairs].index
    plot_df = family_df[family_df["tf1_family"].isin(keep_families)].copy()
    
    if plot_df.empty:
        logger.info(f"No families have at least {min_pairs} pairs.")
        return

    # 4. Sort families by median coop_score for a cleaner visualization
    order = plot_df.groupby("tf1_family")["coop_score"].median().sort_values(ascending=False).index

    # 5. Plotting
    plt.figure(figsize=(3.5, 0.5 * len(keep_families))) # Dynamic height based on family count
    ax = plt.gca()
    
    sns.boxplot(
        data=plot_df, y="tf1_family", x="coop_score", 
        hue="tf1_family",legend=False,
        order=order, palette="vlag", linewidth=0.8, fliersize=1
    )
    
    # Add point clouds (stripplot) on top to show individual pair distribution
    sns.stripplot(
        data=plot_df, y="tf1_family", x="coop_score", 
        order=order, color="black", size=2, alpha=0.3
    )

    # Styling
    plt.axvline(0, color='grey', linestyle='--', linewidth=0.5) # Reference line at 0
    plt.xlabel("Synergy Score", fontsize=7)
    plt.ylabel("TF Family", fontsize=7)
    plt.xticks(fontsize=6)
    plt.yticks(fontsize=6)
    plt.title(f"Cooperativity within Families (n >= {min_pairs})", fontsize=8)
    
    sns.despine()
    plt.tight_layout()

    if outpath:
        plt.savefig(outpath, dpi=300)
        plt.close()
    else:
        plt.show()

    return plot_df



