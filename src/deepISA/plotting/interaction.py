import matplotlib.pyplot as plt
import seaborn as sns
from loguru import logger
import pandas as pd
import numpy as np


from scipy.stats import ks_2samp


def plot_null(
    df, 
    track_idx=0, 
    min_dist=100, 
    max_dist=255, 
    outpath=None, 
    figsize=(2.3, 2.0),
    label_size=7,
    tick_size=5,
    legend_size=5
):
    """
    Plots the KDE of the null interaction distribution for all tracks.
    Labels are dynamically generated as 'Track {i}'.
    """
    # 1. Filter for the null distance range
    null_df = df[(df["distance"] > min_dist) & (df["distance"] <= max_dist)]
    if null_df.empty:
        print(f"No pairs found in range {min_dist}-{max_dist}bp.")
        return None

    # Handle single integer or list of indices
    if isinstance(track_idx, int):
        track_idx = [track_idx]

    # 2. Prepare data for plotting
    plot_data_list = []
    for t in track_idx:
        col = f"interaction_t{t}"
        if col in null_df.columns:
            temp = pd.DataFrame({
                "interaction": null_df[col],
                "Track": f"Track {t}"  # Dynamic naming
            })
            plot_data_list.append(temp)
    
    if not plot_data_list:
        print("No valid track columns found in DataFrame.")
        return None
        
    plot_df = pd.concat(plot_data_list)

    # 3. Setup Plot
    plt.figure(figsize=figsize)
    ax = plt.gca()

    # Consistent frame styling
    for spine in ax.spines.values():
        spine.set_linewidth(0.5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # 4. Plot KDE
    sns.kdeplot(
        data=plot_df, 
        x="interaction", 
        hue="Track", 
        fill=True, 
        alpha=0.3, 
        linewidth=0.8,
        ax=ax
    )
    
    # Red dashed line at zero (Expected value for null)
    ax.axvline(x=0, color='red', linestyle='--', linewidth=0.7, alpha=0.6)
    
    # Formatting
    ax.set_xlabel("Interaction", fontsize=label_size)
    ax.set_ylabel('Density\n(cbrt scale)', fontsize=label_size)
    ax.set_title(f"Null Distribution\n({min_dist}-{max_dist}bp)", fontsize=label_size)
    
    # Symmetrical x-limits based on data spread
    limit = max(abs(plot_df["interaction"].min()), abs(plot_df["interaction"].max()), 0.1)
    ax.set_xlim(-limit, limit)
    # cube root scale
    ax.set_yscale('function', functions=(lambda x: np.power(x, 1/3), lambda x: np.power(x, 3))) 
    
    ax.tick_params(axis='both', which='major', labelsize=tick_size)

    # Manage legend (Only show if multiple tracks exist)
    if len(track_idx) > 1:
        sns.move_legend(ax, "upper right", fontsize=legend_size, title=None, frameon=False)
    else:
        # Remove legend if only one track to keep plot clean
        if ax.get_legend():
            ax.get_legend().remove()
        
    plt.tight_layout()
    
    if outpath:
        plt.savefig(outpath, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        return ax





def plot_tf_pair_against_null(
    df, 
    tf_pair,
    track_idx=0, 
    plot_type='kde', # 'kde' or 'cdf'
    min_dist=100, 
    max_dist=255, 
    outpath=None, 
    fig_size=(2.5, 2.2)
):
    """
    Visualizes TF pair interaction logic against the null distribution.
    Supports KDE (Density) or CDF (Cumulative) views with full statistical annotation.
    """
    t = track_idx
    col = f"interaction_t{t}"
    
    # 1. Extract Distributions
    null_dist = df[(df["distance"] > min_dist) & (df["distance"] <= max_dist)][col].dropna()
    
    pair_name = "|".join(sorted(tf_pair)) if isinstance(tf_pair, (list, tuple)) else tf_pair
    # Extract based on column identity (Standardize search)
    p1, p2 = (tf_pair[0], tf_pair[1]) if isinstance(tf_pair, (list, tuple)) else tf_pair.split('|')
    pair_mask = ((df['tf1'] == p1) & (df['tf2'] == p2)) | ((df['tf1'] == p2) & (df['tf2'] == p1))
    
    pair_data = df[pair_mask]
    total_count = len(pair_data)
    # The valid count is based on non-NaN values (already filtered by _filter_valid_interaction)
    pair_dist = pair_data[col].dropna()
    valid_count = len(pair_dist)

    if pair_dist.empty:
        print(f"No valid interaction data for: {pair_name}")
        return None

    # 2. Calculate Stats for Annotation
    ks_stat, p_val = ks_2samp(pair_dist, null_dist)
    med_shift = pair_dist.median() - null_dist.median()

    # 3. Plotting Setup
    plt.figure(figsize=fig_size)
    ax = plt.gca()
    for spine in ax.spines.values():
        spine.set_linewidth(0.5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    if plot_type == 'cdf':
        # Plot CDF (Cumulative)
        sns.ecdfplot(null_dist, color='grey', label='Null', linewidth=1, ax=ax)
        sns.ecdfplot(pair_dist, color='steelblue', ls='--', label=pair_name, linewidth=1.2, ax=ax)
        plt.ylabel('Cumulative Probability', fontsize=7)
    else:
        # Plot KDE (Density)
        sns.kdeplot(null_dist, color='grey', fill=True, alpha=0.2, label='Null', linewidth=1, ax=ax)
        sns.kdeplot(pair_dist, color='steelblue', ls='--', label=pair_name, linewidth=1.2, ax=ax)
        plt.ylabel('Density (cbrt scale)', fontsize=7)
        # Apply your sqrt scale for density visualization if needed
        # ax.set_yscale('function', functions=(lambda x: np.sqrt(x), lambda x: x**2))

    # Center Line (Additivity Benchmark)
    plt.axvline(x=0, color='red', linestyle=':', linewidth=0.8, alpha=0.5)
    ax.set_yscale('function', functions=(lambda x: np.power(x, 1/3), lambda x: np.power(x, 3)))
    # 4. Annotations
    stats_text = (
        f"Total N: {total_count}\n"
        f"Valid N: {valid_count}\n"
        f"KS D: {ks_stat:.3f} (p:{p_val:.1e})\n"
        f"Med Shift: {med_shift:.3f}\n"
    )
    # Place text in the upper left or right based on plot type
    plt.text(0.05, 0.95, stats_text, transform=ax.transAxes, fontsize=5,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.5, lw=0.3))

    # 5. Formatting
    plt.xlabel(f"Interaction (track {t})", fontsize=7)
    plt.title(f"{pair_name} ({plot_type.upper()})", fontsize=7)
    
    limit = max(pair_dist.abs().max(), null_dist.abs().max(), 0.1) * 1.1
    plt.xlim(-limit, limit)
    
    plt.legend(fontsize=5, frameon=False, loc='upper right')
    plt.xticks(fontsize=5)
    plt.yticks(fontsize=5)
    plt.tight_layout()

    if outpath:
        plt.savefig(outpath, dpi=300)
        plt.close()
    else:
        return ax




def plot_interaction_decay(
    df, 
    track_idx=0, 
    outpath=None,   
    mode='signed', # 'signed' or 'absolute'
    figsize=(2.3, 2),
    label_size=7,
    tick_size=6,
    legend_size=5
):
    """
    Unified function to plot either Signed or Absolute interaction decay across distances.
    Handles single or multiple track indices.
    """
    # 1. Standardize track_idx to a list
    if isinstance(track_idx, (int, float)):
        track_idx = [int(track_idx)]
    else:
        track_idx = list(track_idx)
        
    plt.figure(figsize=figsize)
    ax = plt.gca()
    
    # TODO: move frame styling to utils
    # Clean frame styling
    for spine in ax.spines.values():
        spine.set_linewidth(0.5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Define a color palette for multiple tracks
    palette = sns.color_palette("tab10", n_colors=len(track_idx))
    colors = {t: palette[i] for i, t in enumerate(track_idx)}

    for t in track_idx:
        col = f"interaction_t{t}"
        if col not in df.columns:
            logger.warning(f"Column {col} not found in dataframe.")
            continue
        
        label = f"Track {t}"
        color = colors[t]

        if mode == 'absolute':
            # Calculate mean of absolute values per distance
            decay = df.assign(abs_val=df[col].abs()).groupby("distance")["abs_val"].mean().reset_index()
            sns.lineplot(data=decay, x="distance", y="abs_val", 
                         color=color, linewidth=1, label=label)
            plt.ylabel("Mean Abs Interaction", fontsize=label_size)
            plt.title("Abs Interaction Decay", fontsize=label_size)
            
        else: # 'signed'
            # Calculate means for positive and negative populations separately
            pos_decay = df[df[col] > 0].groupby("distance")[col].mean().reset_index()
            neg_decay = df[df[col] < 0].groupby("distance")[col].mean().reset_index()
            
            # Label only the positive line to avoid duplicate legend entries
            sns.lineplot(data=pos_decay, x="distance", y=col, 
                         color=color, linewidth=1, label=label)
            sns.lineplot(data=neg_decay, x="distance", y=col, 
                         color=color, linewidth=1)
            
            plt.axhline(0, color='black', linewidth=0.5, alpha=0.3)
            plt.ylabel("Mean Interaction", fontsize=label_size)
            plt.title("Interaction Decay", fontsize=label_size)

    # Global formatting
    plt.xlabel("Distance (bp)", fontsize=label_size)
    plt.xticks(fontsize=tick_size)
    plt.yticks(fontsize=tick_size)
    
    # Handle Legend
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        # Dictionary zip removes duplicates if they occur
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(), 
                  fontsize=legend_size, frameon=False, loc='upper right')
    
    plt.tight_layout()

    if outpath:
        plt.savefig(outpath, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()