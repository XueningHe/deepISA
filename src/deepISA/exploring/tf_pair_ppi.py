import os
import pandas as pd
from loguru import logger
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from deepISA.utils import get_data_resource, apply_plot_style, save_or_show, remove_if_exists


import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42





def plot_ppi_enrichment(
    df, 
    rank_by, 
    outpath=None, 
    title=None, 
    fig_size=(4, 4)
):
    """
    Plots PPI enrichment ranking by either cooperativity score or p-value.
    
    Logic:
    - Always excludes dimers ("::").
    - If rank_by="coop_score": Excludes "Independent" pairs (untrustworthy scores).
    - If rank_by="p_val": Retains "Independent" pairs to show the full distribution.
    """
    remove_if_exists(outpath)
    # Always exclude dimers
    df = df[~df['tf_pair'].str.contains("::", na=False)].copy()
    
    # 1. Ranking-specific Logic (Sorting & Filtering)
    if rank_by == "coop_score":
        # Conditional Filter: Remove Independent pairs for coop_score ranking
        if 'cooperativity' in df.columns:
            df = df[df['cooperativity'] != "Independent"].copy()
            
        df = df.sort_values("coop_score", ascending=False).reset_index(drop=True)
        xlabel = "Top % of Pairs (Ranked by Coop Score)"
        plot_color = "teal"
        if title is None: title = "PPI Enrichment (by Coop Score)"
        
    elif rank_by == "p_val":
        # Do NOT filter Independent pairs here; they represent the high p-value background
        df = df.sort_values("mw_p", ascending=True).reset_index(drop=True)
        xlabel = "Top % of Pairs (Ranked by P-value)"
        plot_color = "tomato"
        if title is None: title = "PPI Enrichment (by P-value)"
    else:
        raise ValueError("rank_by must be either 'coop_score' or 'p_val'")
    
    # 2. Map Reported PPIs
    # Note: Assumes get_data_resource is defined in your environment
    ppi_ref_path = get_data_resource("TF_TF_I.txt")
    ref = pd.read_csv(ppi_ref_path, sep='\t')
    reported_pairs = set(ref.apply(lambda x: "|".join(sorted([str(x['TF1']), str(x['TF2'])])), axis=1))
    
    df["is_ppi"] = df["tf_pair"].apply(
        lambda x: 1 if "|".join(sorted(str(x).split("|"))) in reported_pairs else 0
    )
    
    # 3. Metrics Calculation
    df['cum_ppi_count'] = df['is_ppi'].cumsum()
    total_ppi = df['is_ppi'].sum()
    
    if total_ppi == 0:
        print(f"Warning: No reported PPIs found in dataset for ranking by {rank_by}.")
        return df

    df['pct_total_pairs'] = ((df.index + 1) / len(df)) * 100
    df['pct_ppi_found'] = (df['cum_ppi_count'] / total_ppi) * 100

    # 4. Plotting
    fig, ax = plt.subplots(figsize=fig_size)
    styles = apply_plot_style(ax, fig_size) # Assumes custom util exists
    
    sns.lineplot(data=df, x='pct_total_pairs', y='pct_ppi_found', 
                 color=plot_color, lw=styles['scale'], label='Experimental Hits', ax=ax)
    
    ax.plot([0, 100], [0, 100], color='grey', linestyle='--', 
            lw=styles['scale'], label='Random Expectation')
    
    # Focus on top 20%
    ax.set_xlim(0, 20)
    y_max_focus = df[df['pct_total_pairs'] <= 20]['pct_ppi_found'].max()
    ax.set_ylim(0, min(100, (y_max_focus * 1.2) if y_max_focus > 0 else 10))

    ax.set_title(title, fontsize=styles['main'])
    ax.set_xlabel(xlabel, fontsize=styles['main'])
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
    remove_if_exists(outpath)
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
            x_label=x_label or f"# TFs interacting with {cof_name}",
            y_label= "Coop score",
            title=title or f"Validation: {cof_name}"
        )
    return df



def plot_dna_mediated_ppi(
    df_tf_pair,
    rank_by,
    title=None,
    fig_size=(4, 4),
    outpath=None
):
    remove_if_exists(outpath)
    binding_matrix_path = get_data_resource("TF_binding_coop_cleaned.csv")

    # 1. Start from analysis dataframe and remove dimers
    df = df_tf_pair.copy()
    df = df[~df["tf_pair"].str.contains("::", na=False)].copy()

    # 2. Ranking-specific filtering/sorting
    if rank_by == "coop_score":
        if "cooperativity" in df.columns:
            df = df[df["cooperativity"] != "Independent"].copy()

        df = df.sort_values("coop_score", ascending=False).reset_index(drop=True)
        xlabel = "Top % of Pairs (Ranked by Coop Score)"
        plot_color = "teal"
        if title is None:
            title = "DNA-mediated PPI Enrichment (by Coop Score)"

    elif rank_by == "p_val":
        df = df.sort_values("mw_p", ascending=True).reset_index(drop=True)
        xlabel = "Top % of Pairs (Ranked by P-value)"
        plot_color = "tomato"
        if title is None:
            title = "DNA-mediated PPI Enrichment (by P-value)"

    else:
        raise ValueError("rank_by must be either 'coop_score' or 'p_val'")

    # 3. Load binding matrix and define DNA-mediated positives
    df_bind = pd.read_csv(binding_matrix_path)

    melted = df_bind.melt(id_vars="prey", var_name="bait", value_name="interaction")

    p = melted["prey"].astype(str)
    b = melted["bait"].astype(str)
    melted["tf_pair"] = np.where(p < b, p + "|" + b, b + "|" + p)

    melted["is_dna_mediated_ppi"] = melted["interaction"].apply(
        lambda x: 0 if str(x).strip() == "0" else 1
    )

    binding_pairs = (
        melted.groupby("tf_pair")["is_dna_mediated_ppi"]
        .max()
        .reset_index()
    )

    # 4. Merge labels into ranked dataframe
    df = pd.merge(
        df,
        binding_pairs,
        on="tf_pair",
        how="inner"
    )

    if df.empty:
        logger.warning("No overlapping pairs found between results and binding matrix.")
        return None

    # 5. Cumulative enrichment metrics
    df["cum_dna_ppi_count"] = df["is_dna_mediated_ppi"].cumsum()
    total_dna_ppi = df["is_dna_mediated_ppi"].sum()

    if total_dna_ppi == 0:
        print(f"Warning: No DNA-mediated PPIs found in dataset for ranking by {rank_by}.")
        return df

    df["pct_total_pairs"] = ((df.index + 1) / len(df)) * 100
    df["pct_dna_ppi_found"] = (df["cum_dna_ppi_count"] / total_dna_ppi) * 100

    # 6. Plotting
    fig, ax = plt.subplots(figsize=fig_size)
    styles = apply_plot_style(ax, fig_size)

    sns.lineplot(
        data=df,
        x="pct_total_pairs",
        y="pct_dna_ppi_found",
        color=plot_color,
        lw=styles["scale"],
        label="DNA-mediated PPIs",
        ax=ax
    )

    ax.plot(
        [0, 100], [0, 100],
        color="grey",
        linestyle="--",
        lw=styles["scale"],
        label="Random Expectation"
    )

    # Focus on top 20%
    ax.set_xlim(0, 20)
    y_max_focus = df.loc[df["pct_total_pairs"] <= 20, "pct_dna_ppi_found"].max()
    ax.set_ylim(0, min(100, (y_max_focus * 1.2) if y_max_focus > 0 else 10))

    ax.set_title(title, fontsize=styles["main"])
    ax.set_xlabel(xlabel, fontsize=styles["main"])
    ax.set_ylabel("% of DNA-mediated PPIs Recovered", fontsize=styles["main"])
    ax.legend(loc="lower right", frameon=False, fontsize=styles["small"])

    return save_or_show(outpath)