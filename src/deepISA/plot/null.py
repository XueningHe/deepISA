
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ks_2samp

from deepISA.utils import apply_plot_style, save_or_show, remove_if_exists, get_cbrt_scale
import matplotlib.ticker as ticker

import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42





def plot_null_isa(null_isa_path, 
                  tracks=[0],
                  outpath=None, 
                  figsize=(2.3, 2.0)):
    
    remove_if_exists(outpath, label="Null ISA plot")
    # 1. Reading data
    df_null = pd.read_csv(null_isa_path)
    plot_data = []
    for t in tracks:
        col = f"isa_t{t}"
        plot_data.append(pd.DataFrame({"ISA": df_null[col], "Track": f"Track {t}"}))
    if not plot_data: return None
    plot_df = pd.concat(plot_data)
    # 2. Plotting
    fig, ax = plt.subplots(figsize=figsize)
    style = apply_plot_style(ax, figsize)
    sns.kdeplot(data=plot_df, x="ISA", hue="Track", fill=True,
                alpha=0.3, linewidth=style['scale'], ax=ax)
    ax.axvline(x=0, color='black', linestyle='--', linewidth=0.5*style['scale'], alpha=0.6)
    # 3. Formatting
    ax.set_xlabel("ISA", fontsize=style['main'])
    ax.set_ylabel('Density\n(cbrt scale)', fontsize=style['main'])
    ax.set_title(f"ISA distribution for non-motifs", fontsize=style['main'])
    ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))
    ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=5))
    
    limit = plot_df["ISA"].abs().max() * 1.1
    ax.set_xlim(-limit, limit)
    ax.set_yscale('function', functions=get_cbrt_scale())
    
    if len(tracks) > 1:
        sns.move_legend(ax, "upper right", fontsize=style['small'], title=None, frameon=False)
    elif ax.get_legend():
        ax.get_legend().remove()
        
    return save_or_show(outpath)





def plot_motif_length(null_isa_path,
                      motif_locs_path, 
                      outpath=None, 
                      figsize=(2.3, 2.0)):
    df_null = pd.read_csv(null_isa_path)
    df_null["len"]= df_null["end"] - df_null["start"]
    df_motif = pd.read_csv(motif_locs_path)
    df_motif["len"] = df_motif["end"] - df_motif["start"]
    # plot two length distributions on same kde plot
    df_plot= pd.concat([df_null["len"].rename("Length").to_frame().assign(Type="Null"),
                        df_motif["len"].rename("Length").to_frame().assign(Type="Motif")])
    fig, ax = plt.subplots(figsize=figsize)
    style = apply_plot_style(ax, figsize)
    sns.kdeplot(data=df_plot, x="Length", hue="Type", fill=True, alpha=0.3, linewidth=style['scale'], ax=ax, common_norm=False)
    ax.set_xlabel("Motif Length (bp)", fontsize=style['main'])
    ax.set_ylabel('Density', fontsize=style['main'])
    ax.set_title("Length distribution of motifs vs null regions", fontsize=style['main'])
    ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=5))
    ax.set_xlim(0, df_plot["Length"].max() * 1.1)
    if ax.get_legend():
        sns.move_legend(ax, "upper right", fontsize=style['small'], title=None, frameon=False)
    return save_or_show(outpath) 







def plot_null_interaction(
              null_interaction_path, 
              tracks=[0], 
              outpath=None, 
              figsize=(2.3, 2.0)):
    
    remove_if_exists(outpath, label="Null interaction plot")
    # 1.reading data
    df_null = pd.read_csv(null_interaction_path)
    plot_data = []
    for t in tracks:
        col = f"interaction_t{t}"
        # drop rows with nan
        df_null_sub = df_null[df_null[col].notna()].reset_index(drop=True)
        plot_data.append(pd.DataFrame({"interaction": df_null_sub[col], "Track": f"Track {t}"}))
    
    if not plot_data: return None
    plot_df = pd.concat(plot_data)

    # 2. Plotting
    fig, ax = plt.subplots(figsize=figsize)
    style = apply_plot_style(ax, figsize)
    
    sns.kdeplot(data=plot_df, x="interaction", hue="Track", fill=True, 
                alpha=0.3, linewidth=style['scale'], ax=ax,
                common_norm=False)
    ax.axvline(x=0, color='black', linestyle='--', linewidth=0.5*style['scale'], alpha=0.6)
    # 3. Formatting
    ax.set_xlabel("Interaction", fontsize=style['main'])
    ax.set_ylabel('Density\n(cbrt scale)', fontsize=style['main'])
    ax.set_title(f"Interaction between non-motif pairs", fontsize=style['main'])
    ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))
    ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=5))
    
    limit = plot_df["interaction"].abs().max() * 1.1
    ax.set_xlim(-limit, limit)
    ax.set_yscale('function', functions=get_cbrt_scale())
    
    if len(tracks) > 1:
        sns.move_legend(ax, "upper right", fontsize=style['small'], title=None, frameon=False)
    elif ax.get_legend():
        ax.get_legend().remove()
        
    return save_or_show(outpath)



def plot_motif_distance(null_interacton_path,  
                        combi_isa_path,
                        outpath=None,
                        figsize=(2.3, 2.0)):
    df_null= pd.read_csv(null_interacton_path)
    df_combi_isa = pd.read_csv(combi_isa_path)
    # extract column "distance" from df_null and plot its distribution as a kde plot
    df_plot = pd.concat([df_null["distance"].rename("Distance").to_frame().assign(Type="Null"),
                        df_combi_isa["distance"].rename("Distance").to_frame().assign(Type="Combi ISA")])
    fig, ax = plt.subplots(figsize=figsize)
    style = apply_plot_style(ax, figsize)
    sns.kdeplot(data=df_plot, x="Distance", hue="Type", fill=True, alpha=0.3, linewidth=style['scale'], ax=ax, common_norm=False)
    ax.set_xlabel("Distance between motifs (bp)", fontsize=style['main'])
    ax.set_ylabel('Density', fontsize=style['main'])
    ax.set_title("Distance distribution of motif pairs vs null pairs", fontsize=style['main'])
    ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=5))
    ax.set_xlim(0, df_plot["Distance"].max() * 1.1)
    if ax.get_legend():
        sns.move_legend(ax, "upper right", fontsize=style['small'], title=None, frameon=False)
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
        ax.set_yscale('function', functions=get_cbrt_scale())

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

