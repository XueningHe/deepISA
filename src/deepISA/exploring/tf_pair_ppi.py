import os
import pandas as pd
from loguru import logger
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from deepISA.utils import get_data_resource, apply_plot_style, save_or_show


import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42

def plot_ppi_enrichment(
    df, p_val_col="ks_p", outpath=None, title="PPI Enrichment", fig_size=(4, 4)
):
    df = df.dropna(subset=["coop_score"]).copy()
    # Exclude dimers
    df = df[~df['tf_pair'].str.contains("::", na=False)].copy()
    
    # 1. Map Reported PPIs
    ppi_ref_path = get_data_resource("TF_TF_I.txt")
    ref = pd.read_csv(ppi_ref_path, sep='\t')
    reported_pairs = set(ref.apply(lambda x: "|".join(sorted([str(x['TF1']), str(x['TF2'])])), axis=1))
    
    df["is_ppi"] = df["tf_pair"].apply(
        lambda x: 1 if "|".join(sorted(str(x).split("|"))) in reported_pairs else 0
    )
    
    # 2. Metrics Calculation
    df = df.sort_values(p_val_col).reset_index(drop=True)
    df['cum_ppi_count'] = df['is_ppi'].cumsum()
    total_ppi = df['is_ppi'].sum()
    
    if total_ppi == 0:
        logger.warning("No reported PPIs found in dataset.")
        return df

    df['pct_total_pairs'] = ((df.index + 1) / len(df)) * 100
    df['pct_ppi_found'] = (df['cum_ppi_count'] / total_ppi) * 100

    # 3. Plotting with Utils
    fig, ax = plt.subplots(figsize=fig_size)
    styles = apply_plot_style(ax, fig_size)
    
    sns.lineplot(data=df, x='pct_total_pairs', y='pct_ppi_found', 
                 color='tomato', lw=styles['scale'], label='Experimental Hits', ax=ax)
    
    ax.plot([0, 100], [0, 100], color='grey', linestyle='--', 
            lw=styles['scale'], label='Random Expectation')
    
    # Focus on top 20%
    ax.set_xlim(0, 20)
    y_max_focus = df[df['pct_total_pairs'] <= 20]['pct_ppi_found'].max()
    ax.set_ylim(0, min(100, y_max_focus * 1.2))

    ax.set_title(title, fontsize=styles['main'], fontweight='bold')
    ax.set_xlabel("Top % of Pairs (Ranked by P-value)", fontsize=styles['main'])
    ax.set_ylabel("% of Known PPIs Recovered", fontsize=styles['main'])
    ax.legend(loc='lower right', frameon=False, fontsize=styles['small'])
    
    return save_or_show(outpath)




def annotate_cofactor_recruitment(df, cofactors=None):
    df = df[~df['tf_pair'].str.contains("::", na=False)].copy()
    ref = pd.read_csv(get_data_resource("TF_Cof_I.txt"), sep='\t')
    ref.columns = [c.replace("/", "_") for c in ref.columns]
    target_cofactors = cofactors if cofactors else [c for c in ref.columns if c != 'TF']
    # Create a mapping for faster lookup: {cofactor: set(TFs_that_recruit_it)}
    cof_map = {cof: set(ref.loc[ref[cof] == 1, 'TF']) for cof in target_cofactors}
    for cof in target_cofactors:
        col_name = f"count_{cof}"
        def count_tfs(pair):
            tfs = str(pair).split("|")
            return sum(1 for tf in tfs if tf in cof_map[cof])
        df[col_name] = df['tf_pair'].apply(count_tfs)
        df[col_name] = pd.Categorical(df[col_name], categories=[0, 1, 2], ordered=True)
    return df




def plot_box_strip_statistics(
    df, x_col, y_col, x_label, y_label, title, figsize, rotation, outpath
):
    fig, ax = plt.subplots(figsize=figsize)
    styles = apply_plot_style(ax, figsize)
    # Scaled dots
    sns.stripplot(
        data=df, x=x_col, y=y_col, ax=ax,
        color="gray", alpha=0.2, size=styles['scale'] * 2, jitter=True, zorder=1
    )
    # Scaled box with visible red median
    sns.boxplot(
        data=df, x=x_col, y=y_col, ax=ax,
        showfliers=False, width=0.4,
        boxprops={'facecolor': 'none', 'edgecolor': 'black', 'zorder': 2, 'linewidth': styles['scale'] * 0.5},
        medianprops={'color': 'red', 'linewidth': styles['scale'], 'zorder': 3},
        whiskerprops={'linewidth': styles['scale'] * 0.5},
        capprops={'linewidth': styles['scale'] * 0.5}
    )
    # N-counts and labels
    counts = df.groupby(x_col, observed=True)[y_col].count()
    labels = [f"{val}\n(n={counts.get(val, 0)})" for val in df[x_col].cat.categories]
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=styles['small'], rotation=rotation)
    ax.set_title(title, fontsize=styles['main'])
    ax.set_xlabel(x_label, fontsize=styles['main'])
    ax.set_ylabel(y_label, fontsize=styles['main'])
    sns.despine()
    return save_or_show(outpath)






def plot_cofactor_recruitment(
    df, outpath=None, cofactor_name=None, title=None, 
    x_label=None, y_label=None, fig_size=(2.5, 3), rotation=0
):
    df = df.dropna(subset=["coop_score"]).copy()   
    df = annotate_cofactor_recruitment(df, cofactors=[cofactor_name] if cofactor_name else None)

    target_cols = [f"count_{cofactor_name}"] if cofactor_name else [c for c in df.columns if c.startswith("count_")]
    
    base, ext = os.path.splitext(outpath) if outpath else (None, None)

    for col in target_cols:
        if df[col].nunique() < 2: continue
        
        cof_name = col.replace("count_", "")
        curr_out = f"{base}_{cof_name}{ext}" if outpath else None
        
        plot_box_strip_statistics(
            df=df, x_col=col, y_col="coop_score", figsize=fig_size, rotation=rotation, outpath=curr_out,
            x_label=x_label or f"TFs interacting with {cof_name}",
            y_label=y_label or "coop_score",
            title=title or f"Validation: {cof_name}"
        )
    return df





# TODO: split the two plots! after determining which is more consistent
def plot_dna_mediated_ppi(
    df_tf_pair,
    title="DNA-mediated PPI Distributions",
    fig_size=(3,2),
    outpath=None
):
    """
    Groups TF pairs by experimental interaction signal (Zero vs Non-zero) 
    and plots distributions of coop_score and -log10(ks_p).
    """
    binding_matrix_path = get_data_resource("TF_binding_coop_cleaned.csv")
    
    # 1. Load and process binding data
    df_bind = pd.read_csv(binding_matrix_path)
    # Pivot from matrix to long format (prey vs baits)
    melted = df_bind.melt(id_vars='prey', var_name='bait', value_name='interaction')
    
    # Vectorized standardized pair generation (faster than .apply)
    p = melted['prey'].astype(str)
    b = melted['bait'].astype(str)
    melted['tf_pair'] = np.where(p < b, p + "|" + b, b + "|" + p)
    
    # Classify interaction as Zero vs Non-zero
    melted['is_interaction'] = melted['interaction'].apply(
        lambda x: 0 if str(x).strip() == '0' else 1
    )
    binding_pairs = melted.groupby('tf_pair')['is_interaction'].max().reset_index()

    # 2. Merge with Cooperativity results
    merged = pd.merge(df_tf_pair, binding_pairs, on='tf_pair', how='inner')

    if merged.empty:
        logger.warning("No overlapping pairs found between results and binding matrix.")
        return None

    # 3. Data Preparation
    merged['Group'] = merged['is_interaction'].map({0: 'Zero', 1: 'Non-zero'})
    merged['-log10_ks_p'] = -np.log10(merged['ks_p'].replace(0, 1e-300))

    # 4. Plotting
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=fig_size)
    styles = apply_plot_style(ax1, fig_size)
    _ = apply_plot_style(ax2, fig_size) # Apply to second axis too

    # Violin Plot 1: Coop Score
    sns.violinplot(
        data=merged, x='Group', y='coop_score', ax=ax1, 
        palette={'Zero': 'lightgray', 'Non-zero': '#d62728'}, 
        hue='Group', legend=False, inner='quartile', cut=0
    )
    ax1.set_title("Cooperativity Score", fontsize=styles['main'])
    ax1.set_ylabel("Score", fontsize=styles['main'])

    # Violin Plot 2: -log10 Significance
    sns.violinplot(
        data=merged, x='Group', y='-log10_ks_p', ax=ax2, 
        palette={'Zero': 'lightgray', 'Non-zero': '#d62728'}, 
        hue='Group', legend=False, inner='quartile', cut=0
    )
    ax2.set_title("-log10(KS p-value)", fontsize=styles['main'])
    ax2.set_ylabel("-log10(p)", fontsize=styles['main'])

    for ax in [ax1, ax2]:
        ax.set_xlabel("Binding Signal", fontsize=styles['main'])
        sns.despine(ax=ax)

    plt.suptitle(title, fontsize=styles['main'] + 2, y=1.02)
    plt.tight_layout()
    
    return save_or_show(outpath)