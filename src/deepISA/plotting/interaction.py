import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ks_2samp

from deepISA.utils import apply_plot_style, save_or_show, remove_if_exists
import matplotlib.ticker as ticker

import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42



def _get_cbrt_scale():
    """Returns functions for cube root scaling."""
    return (lambda x: np.sign(x) * np.power(np.abs(x), 1/3), 
            lambda x: np.sign(x) * np.power(np.abs(x), 3))



def plot_null(null_combi_isa_path, 
              tracks=[0], 
              outpath=None, 
              figsize=(2.3, 2.0)):
    
    remove_if_exists(outpath, label="Null interaction plot")
    # 1.reading data
    df_null = pd.read_csv(null_combi_isa_path)


    plot_data = []
    for t in tracks:
        col = f"interaction_t{t}"
        plot_data.append(pd.DataFrame({"interaction": df_null[col], "Track": f"Track {t}"}))
    
    if not plot_data: return None
    plot_df = pd.concat(plot_data)

    # 2. Plotting
    fig, ax = plt.subplots(figsize=figsize)
    style = apply_plot_style(ax, figsize)
    
    sns.kdeplot(data=plot_df, x="interaction", hue="Track", fill=True, 
                alpha=0.3, linewidth=style['scale'], ax=ax)
    
    ax.axvline(x=0, color='black', linestyle='--', linewidth=0.5*style['scale'], alpha=0.6)
    
    # 3. Formatting
    ax.set_xlabel("Interaction", fontsize=style['main'])
    ax.set_ylabel('Density\n(cbrt scale)', fontsize=style['main'])
    ax.set_title(f"Interaction between non-motif pairs", fontsize=style['main'])
    
    # X-Axis Rounding: Format to 2 decimal places
    ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))
    # Optional: If the labels overlap, you can reduce the number of ticks
    ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=5))
    
    limit = max(plot_df["interaction"].abs().max(), 0.1)
    ax.set_xlim(-limit, limit)
    ax.set_yscale('function', functions=_get_cbrt_scale())
    
    if len(tracks) > 1:
        sns.move_legend(ax, "upper right", fontsize=style['small'], title=None, frameon=False)
    elif ax.get_legend():
        ax.get_legend().remove()
        
    return save_or_show(outpath)




def plot_tf_pair_against_null(df, tf_pair, track_idx=0, plot_type='kde', 
                               min_dist=100, max_dist=255, outpath=None, figsize=(2.5, 2.2)):
    # 1. Data Selection
    remove_if_exists(outpath, label=f"TF pair vs Null plot for {tf_pair}")
    col = f"interaction_t{track_idx}"
    null_dist = df[(df["distance"] > min_dist) & (df["distance"] <= max_dist)][col].dropna()
    
    p1, p2 = tf_pair if isinstance(tf_pair, (list, tuple)) else tf_pair.split('|')
    pair_name = f"{p1}|{p2}"
    mask = ((df['tf1'] == p1) & (df['tf2'] == p2)) | ((df['tf1'] == p2) & (df['tf2'] == p1))
    pair_dist = df[mask][col].dropna()

    if pair_dist.empty: return None

    # 2. Stats
    ks_stat, p_val = ks_2samp(pair_dist, null_dist)
    med_shift = pair_dist.median() - null_dist.median()

    # 3. Plotting
    fig, ax = plt.subplots(figsize=figsize)
    style = apply_plot_style(ax, figsize)
    
    plot_args = {'linewidth': 1.2 * style['scale'], 'ax': ax}
    if plot_type == 'cdf':
        sns.ecdfplot(null_dist, color='grey', label='Null', **plot_args)
        sns.ecdfplot(pair_dist, color='steelblue', ls='--', label=pair_name, **plot_args)
        ax.set_ylabel('Cumulative Prob', fontsize=style['main'])
    else:
        sns.kdeplot(null_dist, color='grey', fill=True, alpha=0.2, label='Null', **plot_args)
        sns.kdeplot(pair_dist, color='steelblue', ls='--', label=pair_name, **plot_args)
        ax.set_ylabel('Density (cbrt)', fontsize=style['main'])
        ax.set_yscale('function', functions=_get_cbrt_scale())

    ax.axvline(x=0, color='red', linestyle=':', linewidth=0.8 * style['scale'], alpha=0.5)
    
    # 4. Annotations & Formatting
    stats_text = f"N: {len(pair_dist)}\nKS D: {ks_stat:.3f}\np:{p_val:.1e}\nMedΔ: {med_shift:.3f}"
    ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, fontsize=style['small'],
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.5, lw=0.3))

    ax.set_xlabel(f"Interaction (Track {track_idx})", fontsize=style['main'])
    ax.set_title(pair_name, fontsize=style['main'])
    limit = max(pair_dist.abs().max(), null_dist.abs().max(), 0.1) * 1.1
    ax.set_xlim(-limit, limit)
    ax.legend(fontsize=style['small'], frameon=False, loc='upper right')
    
    return save_or_show(outpath)


def plot_interaction_decay(df, track_idx=0, mode='signed', outpath=None, figsize=(2.3, 2)):
    remove_if_exists(outpath, label=f"Interaction decay plot for track {track_idx} ({mode})")
    tracks = [track_idx] if isinstance(track_idx, (int, float)) else list(track_idx)
    fig, ax = plt.subplots(figsize=figsize)
    style = apply_plot_style(ax, figsize)
    
    palette = sns.color_palette("tab10", n_colors=len(tracks))

    for i, t in enumerate(tracks):
        col = f"interaction_t{t}"
        if col not in df.columns: continue
        
        color = palette[i]
        if mode == 'absolute':
            decay = df.assign(abs_v=df[col].abs()).groupby("distance")["abs_v"].mean().reset_index()
            sns.lineplot(data=decay, x="distance", y="abs_v", color=color, linewidth=0.5 * style['scale'], ax=ax, label=f"T{t}")
        else:
            for sign, m in [(1, 'pos'), (-1, 'neg')]:
                sub = df[df[col] * sign > 0].groupby("distance")[col].mean().reset_index()
                sns.lineplot(data=sub, x="distance", y=col, color=color, ax=ax, linewidth=style['scale'],
                             label=f"T{t}" if m == 'pos' else None)
            ax.axhline(0, color='black', linewidth=0.5 * style['scale'], alpha=0.3)

    ax.set_xlabel("Distance (bp)", fontsize=style['main'])
    ax.set_ylabel("Mean Interaction", fontsize=style['main'])
    ax.set_title(f"Interaction decay ({mode.capitalize()})", fontsize=style['main'])
    
    if len(tracks) > 0:
        ax.legend(fontsize=style['small'], frameon=False, loc='upper right')
        
    return save_or_show(outpath)