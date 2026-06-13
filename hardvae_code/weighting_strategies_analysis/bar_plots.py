"""
sensitivity_barplot.py
-----------------------
Bar plots for sensitivity analysis:
  - 2 rows: MFI (top), F1 Score (bottom)
  - 3 columns: Static, Self-Paced, Curriculum
  - Static: 3 bars (one per annealing weight)
  - Self-Paced: 3 groups (pacing function) x 3 bars (annealing weight)
  - Curriculum: 3 groups (split p) x 3 bars (annealing weight)
  - TVAE and CTGAN shown as horizontal reference lines

Two versions: mean only, mean+-std error bars
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

OUT_DIR = Path(r"C:\Users\MABROUKAs\Downloads\IEEE-Access-2026\HardVAE\hardvae_code\weighting_strategies_analysis\sensitivity")
OUT_DIR.mkdir(parents=True, exist_ok=True)

SENS_SEEDS    = [26226, 116740, 670488]
SENS_DATASETS = ['Hypothyroid', 'NewThyroid1', 'Vertebral']
ANNEAL_VALS   = [0.05, 0.1, 0.2]
PACING_FUNCS  = ['linear', 'root', 'logarithmic']
PACING_LABELS = {'linear': 'Linear', 'root': 'Root', 'logarithmic': 'Log'}
P_VALS        = [0.2, 0.3, 0.4]
P_SPLITS      = {'0.2': '(0.2, 0.2, 0.6)',
                 '0.3': '(0.3, 0.3, 0.4)',
                 '0.4': '(0.4, 0.4, 0.2)'}

# Colors per annealing weight
ANNEAL_COLORS = {0.05: '#1A5276', 0.1: '#117A65', 0.2: '#6C3483'}
BL_COLORS     = {'TVAE': '#C0392B', 'CTGAN': '#E67E22'}

plt.rcParams.update({'font.family': 'serif', 'font.size': 9,
                     'axes.linewidth': 0.8})

# ── Load & prepare ─────────────────────────────────────────────────────────────
fid_sens = pd.read_csv(r'C:\Users\MABROUKAs\Downloads\IEEE-Access-2026\HardVAE\hardvae_code\weighting_strategies_analysis\RESULTS_WEIGHTING_STRATEGIES_SENSITIVITY\fidelity\fidelity_summary.csv')
fid_sens = fid_sens[fid_sens['model'].str.contains('F4')].copy()
fid_sens['MFI'] = 4/(1/fid_sens['distributional_DistFidScore'] +
                     1/fid_sens['complexity_complexity_mean_sim'] +
                     1/fid_sens['hardness_hardness_mean_ks_statistic'] +
                     1/fid_sens['topological_d3_bottleneck_H0_similarity'])

util_sens = pd.read_csv(r'C:\Users\MABROUKAs\Downloads\IEEE-Access-2026\HardVAE\hardvae_code\weighting_strategies_analysis\RESULTS_WEIGHTING_STRATEGIES_SENSITIVITY\utility\utility_summary.csv')
util_sens = util_sens[util_sens['model'].str.contains('F4')].copy()

meta     = fid_sens[['model','strategy','annealing_weight',
                      'pacing_function','curriculum_split']].drop_duplicates()
agg_fid  = fid_sens.groupby(['model','dataset'])['MFI'].mean().reset_index(
           ).merge(meta, on='model', how='left')
agg_util = util_sens.groupby(['model','Dataset'])['F1 Score'].mean().reset_index(
           ).rename(columns={'Dataset':'dataset'}).merge(meta, on='model', how='left')

# ── Baseline stats ─────────────────────────────────────────────────────────────
fid_main   = pd.read_csv(r'C:\Users\MABROUKAs\Downloads\IEEE-Access-2026\HardVAE\hardvae_code\RESULTS\fidelity\mfi_results.csv')
util_gains = pd.read_csv(r'C:\Users\MABROUKAs\Downloads\IEEE-Access-2026\HardVAE\hardvae_code\RESULTS\utility\output\utility_gains.csv')

bl_fid_raw  = fid_main[fid_main['model'].isin(['TVAE','CTGAN']) &
                        fid_main['dataset'].isin(SENS_DATASETS) &
                        fid_main['seed'].isin(SENS_SEEDS)]
bl_util_raw = util_gains[util_gains['Dataset'].isin(SENS_DATASETS) &
                          util_gains['seed'].isin(SENS_SEEDS) &
                          (util_gains['metric']=='F1 Score')
                         ][['baseline','Dataset','seed','baseline_score']
                          ].drop_duplicates()

bl = {
    'MFI': {
        'TVAE' : (bl_fid_raw[bl_fid_raw['model']=='TVAE']['MFI'].mean(),
                  bl_fid_raw[bl_fid_raw['model']=='TVAE']['MFI'].std()),
        'CTGAN': (bl_fid_raw[bl_fid_raw['model']=='CTGAN']['MFI'].mean(),
                  bl_fid_raw[bl_fid_raw['model']=='CTGAN']['MFI'].std()),
    },
    'F1 Score': {
        'TVAE' : (bl_util_raw[bl_util_raw['baseline']=='TVAE']['baseline_score'].mean(),
                  bl_util_raw[bl_util_raw['baseline']=='TVAE']['baseline_score'].std()),
        'CTGAN': (bl_util_raw[bl_util_raw['baseline']=='CTGAN']['baseline_score'].mean(),
                  bl_util_raw[bl_util_raw['baseline']=='CTGAN']['baseline_score'].std()),
    },
}

# ── Cell value getter ──────────────────────────────────────────────────────────
def get_val(agg, metric, strategy, group_col, group_val, aw):
    if strategy == 'static':
        sub = agg[(agg['strategy']=='static') &
                  (agg['annealing_weight']==aw)]
    elif strategy == 'self_paced':
        sub = agg[(agg['strategy']=='self_paced') &
                  (agg['pacing_function']==group_val) &
                  (agg['annealing_weight']==aw)]
    else:  # curriculum
        sub = agg[(agg['strategy']=='curriculum') &
                  (agg['curriculum_split']==P_SPLITS[str(group_val)]) &
                  (agg['annealing_weight']==aw)]
    vals = sub[metric].values
    return vals.mean() if len(vals) else np.nan, vals.std() if len(vals)>1 else 0.0

# ── Draw ───────────────────────────────────────────────────────────────────────
def draw(show_std, out_path):
    fig, axes = plt.subplots(2, 3, figsize=(18, 9), facecolor='white',
                             gridspec_kw={'hspace': 0.12, 'wspace': 0.12})

    METRICS      = ['MFI', 'F1 Score']
    METRIC_YLABELS = {'MFI': 'MFI Score', 'F1 Score': 'F1 Score'}
    STRAT_TITLES = ['Static', 'Self-Paced', 'Curriculum']

    for row_idx, metric in enumerate(METRICS):
        agg = agg_fid if metric == 'MFI' else agg_util

        # Compute y-axis range from all values + baselines
        all_means = []
        for aw in ANNEAL_VALS:
            all_means.append(get_val(agg, metric, 'static', None, None, aw)[0])
        for pf in PACING_FUNCS:
            for aw in ANNEAL_VALS:
                all_means.append(get_val(agg, metric, 'self_paced', 'pacing_function', pf, aw)[0])
        for p in P_VALS:
            for aw in ANNEAL_VALS:
                all_means.append(get_val(agg, metric, 'curriculum', 'curriculum_split', p, aw)[0])
        all_means = [v for v in all_means if not np.isnan(v)]
        all_means += [bl[metric]['TVAE'][0], bl[metric]['CTGAN'][0]]
        ymin = min(all_means) - 0.025
        ymax = max(all_means) + 0.04

        # ── Static panel ──────────────────────────────────────────────────────
        ax = axes[row_idx, 0]
        ax.set_facecolor('white')
        x = np.arange(len(ANNEAL_VALS))
        for i, aw in enumerate(ANNEAL_VALS):
            mean, std = get_val(agg, metric, 'static', None, None, aw)
            bar = ax.bar(i, mean, color=ANNEAL_COLORS[aw],
                         width=0.55, alpha=0.85, zorder=3,
                         linewidth=0.5, edgecolor='#AAAAAA')
            if show_std:
                ax.errorbar(i, mean, yerr=std, fmt='none',
                            ecolor='#333333', elinewidth=1.0,
                            capsize=3, zorder=4)

        ax.set_xticks(x)
        ax.set_xticklabels([f'w={aw}' for aw in ANNEAL_VALS],
                           fontsize=8.5, color='#1A1A1A')

        # ── Self-paced panel ──────────────────────────────────────────────────
        ax = axes[row_idx, 1]
        ax.set_facecolor('white')
        n_groups  = len(PACING_FUNCS)
        n_bars    = len(ANNEAL_VALS)
        bar_w     = 0.22
        group_gap = 0.85

        for gi, pf in enumerate(PACING_FUNCS):
            for bi, aw in enumerate(ANNEAL_VALS):
                x_pos = gi * group_gap + bi * bar_w
                mean, std = get_val(agg, metric, 'self_paced', 'pacing_function', pf, aw)
                ax.bar(x_pos, mean, color=ANNEAL_COLORS[aw],
                       width=bar_w, alpha=0.85, zorder=3,
                       linewidth=0.5, edgecolor='#AAAAAA')
                if show_std:
                    ax.errorbar(x_pos, mean, yerr=std, fmt='none',
                                ecolor='#333333', elinewidth=1.0,
                                capsize=2.5, zorder=4)

        # Group tick positions
        group_centers = [gi*group_gap + (n_bars-1)*bar_w/2
                         for gi in range(n_groups)]
        ax.set_xticks(group_centers)
        ax.set_xticklabels([PACING_LABELS[pf] for pf in PACING_FUNCS],
                           fontsize=8.5, color='#1A1A1A')

        # ── Curriculum panel ──────────────────────────────────────────────────
        ax = axes[row_idx, 2]
        ax.set_facecolor('white')

        for gi, p in enumerate(P_VALS):
            for bi, aw in enumerate(ANNEAL_VALS):
                x_pos = gi * group_gap + bi * bar_w
                mean, std = get_val(agg, metric, 'curriculum', 'curriculum_split', p, aw)
                ax.bar(x_pos, mean, color=ANNEAL_COLORS[aw],
                       width=bar_w, alpha=0.85, zorder=3,
                       linewidth=0.5, edgecolor='#AAAAAA')
                if show_std:
                    ax.errorbar(x_pos, mean, yerr=std, fmt='none',
                                ecolor='#333333', elinewidth=1.0,
                                capsize=2.5, zorder=4)

        group_centers = [gi*group_gap + (n_bars-1)*bar_w/2
                         for gi in range(len(P_VALS))]
        ax.set_xticks(group_centers)
        ax.set_xticklabels([f'p={p}' for p in P_VALS],
                           fontsize=8.5, color='#1A1A1A')

        # ── Common styling for all 3 panels ───────────────────────────────────
        for col_idx in range(3):
            ax = axes[row_idx, col_idx]

            # Baseline reference lines
            for name, (mean, std) in bl[metric].items():
                ax.axhline(mean, color=BL_COLORS[name], lw=1.5,
                           ls='--', alpha=0.9, zorder=5,
                           label=f'{name} ({mean:.3f})')

            ax.set_ylim(ymin, ymax)
            ax.yaxis.grid(True, color='#E5E7E9', lw=0.6, zorder=0)
            ax.set_axisbelow(True)
            ax.spines[['top','right']].set_visible(False)
            ax.spines[['left','bottom']].set_color('#BDC3C7')
            ax.tick_params(axis='y', labelsize=7.5, colors='#1A1A1A')
            ax.tick_params(axis='x', colors='#1A1A1A', bottom=False)

            if row_idx == 0:
                ax.set_title(STRAT_TITLES[col_idx], fontsize=11,
                             fontweight='bold', color='#1A252F', pad=8)
            if col_idx == 0:
                ax.set_ylabel(METRIC_YLABELS[metric], fontsize=9.5,
                              fontweight='bold', color='#1A252F', labelpad=6)

            # Legend only on first panel of each row
            if col_idx == 2:
                ax.legend(loc='lower right', fontsize=7.5,
                          facecolor='white', edgecolor='#BDC3C7',
                          framealpha=0.9)

    # Annealing weight legend
    aw_handles = [mpatches.Patch(color=ANNEAL_COLORS[aw], alpha=0.85,
                                 label=f'w={aw}')
                  for aw in ANNEAL_VALS]
    fig.legend(handles=aw_handles, loc='lower center', ncol=3,
               fontsize=8.5, facecolor='white', edgecolor='#BDC3C7',
               bbox_to_anchor=(0.5, 0.01),
               title='Annealing weight', title_fontsize=8.5)

    std_note = ' (with error bars)' if show_std else ' (mean only)'
    # fig.suptitle(
    #     f'Sensitivity Analysis \u2014 HardTVAE (F4) Weighting Strategy Hyperparameters{std_note}\n'
    #     'MFI (top) | F1 Score (bottom) | Bars grouped by strategy-specific parameter '
    #     '| Dashed lines: baselines',
    #     color='#1A252F', fontsize=11, fontweight='bold', y=0.995)

    plt.tight_layout(rect=[0, 0.07, 1, 0.99])
    plt.savefig(out_path, dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print(f"  Saved -> {out_path.name}")

draw(show_std=False, out_path=OUT_DIR / 'sensitivity_barplot.png')
draw(show_std=True,  out_path=OUT_DIR / 'sensitivity_barplot_std.png')
print("Done.")