import os
import pandas as pd
from loguru import logger
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# Internal imports
from deepISA.utils import get_data_resource, plot_violin_with_statistics
from deepISA.scoring.combi_isa import assign_cooperativity


# TODO: change "plot" to "analyze", then give option to return df or plot.



def plot_ppi_enrichment(
    df, 
    p_val_col="ks_p",
    outpath=None,
    title="PPI Enrichment",
    fig_size=(4, 4)
):
    """
    Generates a cumulative enrichment plot showing how many reported PPIs
    are recovered as we look at the most significant TF pairs.
    """
    df = assign_cooperativity(df)
    # remove nan coop scores
    df = df.dropna(subset=["coop_score"]).copy()    
    # 1. Preprocessing and Filtering
    # Exclude dimers and ensure we have p-values to rank by
    df = df[~df['tf_pair'].str.contains("::", na=False)].copy()
    
    # 2. Map Reported PPIs
    # (Assuming get_data_resource is available in your environment)
    ppi_ref_path = get_data_resource("TF_TF_I.txt")
    
    ref = pd.read_csv(ppi_ref_path, sep='\t')
    
    reported_pairs = set(
        ref.apply(lambda x: "|".join(sorted([str(x['TF1']), str(x['TF2'])])), axis=1)
    )
    
    def normalize_pair(p):
        return "|".join(sorted(str(p).split("|")))
    
    # Mark binary PPI status
    df["is_ppi"] = df["tf_pair"].apply(
        lambda x: 1 if normalize_pair(x) in reported_pairs else 0
    )
    
    # 3. Calculate Enrichment Metrics
    # Rank by p-value (smallest p-value at the top)
    df = df.sort_values(p_val_col).reset_index(drop=True)
    
    df['cum_ppi_count'] = df['is_ppi'].cumsum()
    df['rank'] = df.index + 1
    
    total_ppi_in_data = df['is_ppi'].sum()
    if total_ppi_in_data == 0:
        print("Warning: No reported PPIs found in the provided dataset.")
        return df

    # Calculate percentages for the axes
    df['pct_total_pairs'] = (df['rank'] / len(df)) * 100
    df['pct_ppi_found'] = (df['cum_ppi_count'] / total_ppi_in_data) * 100

    # 4. Plotting
    plt.figure(figsize=fig_size)
    
    # The Enrichment Curve
    sns.lineplot(
        data=df, 
        x='pct_total_pairs', 
        y='pct_ppi_found', 
        color='tomato', 
        lw=2.5, 
        label='Experimental Hits'
    )
    
    # The Random Baseline (y=x)
    plt.plot([0, 100], [0, 100], color='grey', linestyle='--', label='Random Expectation')
    
    # 5. Styling
    # Usually, we only care about the top 10-20% of the distribution
    plt.xlim(0, 20) 
    # Dynamically set ylim based on what was found in the top 20%
    y_max_focus = df[df['pct_total_pairs'] <= 20]['pct_ppi_found'].max()
    plt.ylim(0, min(100, y_max_focus * 1.2))

    plt.title(title, fontweight='bold')
    plt.xlabel("Top % of Pairs (Ranked by P-value)")
    plt.ylabel("% of Known PPIs Recovered")
    plt.legend(loc='lower right', frameon=False)
    sns.despine()
    plt.grid(axis='both', linestyle=':', alpha=0.5)

    if outpath:
        plt.savefig(outpath, bbox_inches='tight', dpi=300)
        plt.close()
    else:
        plt.show()

    return df











def annotate_cofactor_recruitment(df, cofactors=None):
    """
    Function 1: Adds columns (count_Cofactor) to the df.
    Excludes dimers (::) and maps 0, 1, or 2 based on TF_Cof_I.txt.
    """
    # 1. Strict Dimer Exclusion
    df = df[~df['tf_pair'].str.contains("::", na=False)].copy()
    ref_path = get_data_resource("TF_Cof_I.txt")
    try:    
        ref = pd.read_csv(ref_path, sep='\t')
        # Handle 'SWI/SNF' naming
        ref.columns = [c.replace("/", "_") for c in ref.columns]
        ref.set_index('TF', inplace=True)
    except Exception as e:
        logger.error(f"Failed to load cofactor reference: {e}")
        return df
    # Determine which cofactors to count
    target_cofactors = cofactors if cofactors else [c for c in ref.columns]
    for cof in target_cofactors:
        col_name = f"count_{cof}"
        
        def count_logic(pair_str):
            tfs = str(pair_str).split("|")
            # sum 1 if TF is in ref index AND marked as 1
            return sum(1 for tf in tfs if tf in ref.index and ref.loc[tf, cof] == 1)

        df[col_name] = df['tf_pair'].apply(count_logic)
        df[col_name] = pd.Categorical(df[col_name], categories=[0, 1, 2], ordered=True)
    
    return df






def plot_box_strip_statistics(
    df, x_col, y_col, x_label, y_label, title, figsize, rotation, outpath
):
    """
    New plotting function that uses a Box Plot with an overlaid Stripplot 
    to make the median highly visible.
    """
    plt.figure(figsize=figsize)
    
    # 1. Add the Stripplot (the 'dots') first so they are in the background
    # Low alpha (0.2) and small size (2) prevent 'ink blobbing' at high n
    sns.stripplot(
        data=df, x=x_col, y=y_col, 
        color="gray", alpha=0.2, size=2, jitter=True, zorder=1
    )
    
    # 2. Add the Boxplot on top
    # We set the median color to red and thicken the line for visibility
    sns.boxplot(
        data=df, x=x_col, y=y_col, 
        showfliers=False, # Hide outliers to avoid double-plotting with strip
        width=0.4,
        boxprops={'facecolor': 'none', 'edgecolor': 'black', 'zorder': 2},
        medianprops={'color': 'red', 'linewidth': 2.5, 'zorder': 3},
        whiskerprops={'zorder': 2},
        capprops={'zorder': 2}
    )

    # 3. Annotate N-counts on the X-axis
    counts = df.groupby(x_col, observed=True)[y_col].count()
    labels = [f"{int(val)}\n(n={counts[val]})" for val in sorted(df[x_col].unique())]
    plt.gca().set_xticklabels(labels)

    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.xticks(rotation=rotation)
    sns.despine()
    
    if outpath:
        plt.savefig(outpath, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

# TODO: change "plot" to "analyze", then give option to return df or plot.

def validate_cofactor_recruitment(
    df, 
    outpath=None,
    cofactor_name=None, 
    title=None,
    x_label=None,
    y_label=None,
    fig_size=(2.5, 3),
    rotation=0
):
    df = assign_cooperativity(df)
    df = df.dropna(subset=["coop_score"]).copy()   
     
    df = annotate_cofactor_recruitment(df, cofactors=[cofactor_name] if cofactor_name else None)

    if cofactor_name:
        target_cols = [f"count_{cofactor_name}"]
    else:
        target_cols = [c for c in df.columns if c.startswith("count_")]

    if not target_cols:
        return df

    plot_df = df.dropna(subset=["coop_score"]).copy()
    if outpath is not None:
        base, ext = os.path.splitext(outpath)
    else:
        base, ext = None, None

    for col in target_cols:
        if plot_df[col].nunique() < 2:
            continue
        cof_display_name = col.replace("count_", "")
        
        current_outpath = f"{base}_{cof_display_name}{ext}" if outpath else None
        
        # Use the new Box + Strip function here
        plot_box_strip_statistics(
            figsize=fig_size,
            df=plot_df,
            x_col=col,
            y_col="coop_score",
            x_label=x_label or f"TFs interacting with {cof_display_name}",
            y_label=y_label or "coop_score",
            title=title or f"Validation: {cof_display_name}",
            rotation=rotation,
            outpath=current_outpath
        )

    return df