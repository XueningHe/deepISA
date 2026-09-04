import matplotlib.pyplot as plt
import seaborn as sns

from deepISA.utils import apply_plot_style, save_or_show, remove_if_exists

import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42


def plot_interaction_decay(df, track_idx=0, mode='signed', outpath=None, figsize=(2.3, 2)):
    remove_if_exists(outpath, label=f"Interaction decay plot for track {track_idx} ({mode})")
    tracks = [track_idx] if isinstance(track_idx, (int, float)) else list(track_idx)
    fig, ax = plt.subplots(figsize=figsize)
    style = apply_plot_style(ax, figsize)
    palette = sns.color_palette("tab10", n_colors=len(tracks))

    for i, t in enumerate(tracks):
        col = f"interaction_t{t}"
        color = palette[i]
        valid = df[df[col].notna()]
        if mode == 'absolute':
            decay = (valid.assign(abs_v=valid[col].abs()).groupby("distance", as_index=False)["abs_v"].mean())
            sns.lineplot(data=decay, x="distance", y="abs_v",
                        color=color, linewidth=0.5 * style['scale'], ax=ax, label=f"T{t}")
        else:
            for sign, m in [(1, 'pos'), (-1, 'neg')]:
                sub = (valid[valid[col] * sign > 0].groupby("distance", as_index=False)[col].mean())
                sns.lineplot(data=sub, x="distance", y=col, color=color, ax=ax,
                            linewidth=style['scale'], label=f"T{t}" if m == 'pos' else None)
            ax.axhline(0, color='black', linewidth=0.5 * style['scale'], alpha=0.3)

    ax.set_xlabel("Distance (bp)", fontsize=style['main'])
    ax.set_ylabel("Mean Interaction", fontsize=style['main'])
    ax.set_title(f"Interaction decay ({mode.capitalize()})", fontsize=style['main'])
    
    if len(tracks) > 0:
        ax.legend(fontsize=style['small'], frameon=False, loc='upper right')
        
    return save_or_show(outpath)