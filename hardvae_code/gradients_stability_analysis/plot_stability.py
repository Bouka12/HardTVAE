"""
Aggregate saved per-run CSVs and plot gradient/loss stability
across seeds (mean ± std) for each model configuration.
"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

RESULTS_DIR  = "hardvae_code\\gradients_stability_analysis\\RESULTS-GRADIENT-ANALYSIS"
RUN_METRICS  = os.path.join(RESULTS_DIR, "run_metrics")
PLOTS_OUT    = os.path.join(RESULTS_DIR, "stability_plots")
os.makedirs(PLOTS_OUT, exist_ok=True)


# ── 1. Load all saved CSVs ────────────────────────────────────────────────────

# Known values — order matters: longer/ambiguous ones first
# KNOWN_METRICS    = ['TD_P', 'Harmfulness', 'DCP', 'CLD', 'MV', 'CB', 
#                     'N2', 'LSC', 'LSR', 'F1', 'F4']
KNOWN_METRICS    = [ 'CB', 'LSC', 'F4'] # excluded: 'TD_P',  # TD_P first, then the rest alphabetically
KNOWN_STRATEGIES = ['self_paced', 'curriculum', 'static']  # self_paced first


def parse_hardtvae_filename(fname: str):
    """
    Robustly extract (metric, strategy) from a HardTVAE filename.
    Format: {dataset}_HardTVAE_{metric}_{strategy}_{seed}.csv
    Returns (None, None) for TVAE baseline files.
    """
    name = fname.replace('.csv', '')
    if 'HardTVAE' not in name:
        return None, None
    for metric in KNOWN_METRICS:
        for strategy in KNOWN_STRATEGIES:
            if f'HardTVAE_{metric}_{strategy}_' in name:
                return metric, strategy
    return None, None  # HardTVAE file but unrecognised combo — won't silently corrupt


def load_all_runs(run_metrics_dir: str) -> pd.DataFrame:
    frames = []
    for fname in os.listdir(run_metrics_dir):
        if not fname.endswith('.csv'):
            continue
        df_file = pd.read_csv(os.path.join(run_metrics_dir, fname))

        # Override metric/strategy from filename — immune to split('_') bugs
        metric, strategy = parse_hardtvae_filename(fname)
        if metric is not None:
            df_file['metric']   = metric
            df_file['strategy'] = strategy

        frames.append(df_file)

    if not frames:
        raise FileNotFoundError(f'No CSV files found in {run_metrics_dir}')

    df = pd.concat(frames, ignore_index=True)
    df['metric']   = df['metric'].fillna('None').astype(str)
    df['strategy'] = df['strategy'].fillna('None').astype(str)
    return df




# ── 2. Aggregate: mean ± std across seeds per (dataset, metric, strategy) ────

def aggregate(df: pd.DataFrame) -> pd.DataFrame:
    group_cols = ['dataset', 'metric', 'strategy', 'epoch']
    metric_cols = ['total_loss', 'recon_loss', 'kl_loss', 'grad_norm']

    agg = (
        df.groupby(group_cols)[metric_cols]
        .agg(['mean', 'std'])
        .reset_index()
    )
    # Flatten multi-level columns: (total_loss, mean) → total_loss_mean
    agg.columns = [
        '_'.join(c).strip('_') if isinstance(c, tuple) else c
        for c in agg.columns
    ]
    agg = agg.fillna(0)  # std is NaN when only 1 seed — treat as 0
    return agg


# ── 3. Plot ───────────────────────────────────────────────────────────────────

METRICS_TO_PLOT = {
    'grad_norm':   ('Gradient L2 Norm',   'L2 Norm'),
    'total_loss':  ('Total Loss',          'Loss'),
    'recon_loss':  ('Reconstruction Loss', 'Loss'),
    'kl_loss':     ('KL Divergence',       'Loss'),
}

STRATEGY_STYLES = {
    'curriculum':  {'linestyle': '-',  'linewidth': 1.8},
    'static':      {'linestyle': '--', 'linewidth': 1.8},
    'self_paced':  {'linestyle': ':',  'linewidth': 2.0},
}

BASELINE_STYLE = {'color': 'black', 'linestyle': '-.', 'linewidth': 1.5, 'label': 'TVAE (baseline)'}


def plot_one_metric_config(agg: pd.DataFrame, dataset: str, hardness_metric: str):
    """
    One figure per (dataset × hardness_metric).
    Rows  = tracked quantities (grad_norm, total_loss, recon_loss, kl_loss).
    Lines = the 3 strategies for that metric  +  TVAE baseline (black dashed).
    Shaded band = ± 1 std across seeds.
    """
    subset   = agg[agg['dataset'] == dataset]
    strategies = subset[subset['metric'] == hardness_metric]['strategy'].unique()

    colors = cm.tab10(np.linspace(0, 1, len(strategies)))

    n_rows = len(METRICS_TO_PLOT)
    fig, axes = plt.subplots(n_rows, 1, figsize=(15, 4 * 4), sharex=True)
    # fig.suptitle(
    #     f"Gradient & Loss Stability — {dataset}  |  metric: {hardness_metric}",
    #     fontsize=13, fontweight='bold'
    # )

    for ax, (col, (title, ylabel)) in zip(axes, METRICS_TO_PLOT.items()):

        # ── TVAE baseline ────────────────────────────────────────────
        baseline = subset[
          (subset['metric'] == 'None') & (subset['strategy'] == 'None') # 'static')
        ].sort_values('epoch')

        if not baseline.empty:
            mean = baseline[f'{col}_mean'].values
            std  = baseline[f'{col}_std'].values
            epochs = baseline['epoch'].values
            ax.plot(epochs, mean, **BASELINE_STYLE)
            ax.fill_between(epochs, mean - std, mean + std, alpha=0.12, color='black')

        # ── HardTVAE strategies ──────────────────────────────────────
        for strategy, color in zip(strategies, colors):
            mask = (subset['metric'] == hardness_metric) & (subset['strategy'] == strategy)
            data = subset[mask].sort_values('epoch')
            if data.empty:
                continue

            mean   = data[f'{col}_mean'].values
            std    = data[f'{col}_std'].values
            epochs = data['epoch'].values
            style  = STRATEGY_STYLES.get(strategy, {'linestyle': '-', 'linewidth': 1.8})

            ax.plot(epochs, mean, label=f"{strategy}", color=color, **style)
            ax.fill_between(epochs, mean - std, mean + std, alpha=0.15, color=color)

        ax.set_title(title, fontweight='bold')
        ax.set_ylabel(ylabel)
        ax.grid(True, linestyle='--', alpha=0.4)
        ax.legend(fontsize=8, ncol=2)

    axes[-1].set_xlabel('Epoch')
    plt.tight_layout()

    safe_metric = hardness_metric.replace('/', '_')   # sanitise for filename
    out_path = os.path.join(PLOTS_OUT, f"{dataset}_{safe_metric}_stability.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  Saved → {out_path}")

# --- plot TVAE-----------------------------------------------------------------
def plot_tvae_baseline(agg: pd.DataFrame, dataset: str):
    """Standalone plot for the TVAE baseline."""
    subset = agg[
        (agg['dataset'] == dataset) &
        (agg['metric'] == 'None') &
        (agg['strategy'] == 'None')
    ].sort_values('epoch')

    if subset.empty:
        print(f"  No TVAE data for {dataset}, skipping.")
        return

    n_rows = len(METRICS_TO_PLOT)
    fig, axes = plt.subplots(n_rows, 1, figsize=(10, 4 * n_rows), sharex=True)
    # fig.suptitle(f"TVAE Baseline — {dataset}", fontsize=13, fontweight='bold')

    for ax, (col, (title, ylabel)) in zip(axes, METRICS_TO_PLOT.items()):
        mean   = subset[f'{col}_mean'].values
        std    = subset[f'{col}_std'].values
        epochs = subset['epoch'].values

        ax.plot(epochs, mean, color='black', linewidth=1.8, label='TVAE')
        ax.fill_between(epochs, mean - std, mean + std, alpha=0.15, color='black')
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.grid(True, linestyle='--', alpha=0.4)
        ax.legend(fontsize=8)

    axes[-1].set_xlabel('Epoch')
    plt.tight_layout()

    out_path = os.path.join(PLOTS_OUT, f"{dataset}_TVAE_baseline_stability.png")
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved → {out_path}")

# ------------- plot per dataset ----------------------------------------------
def plot_dataset_overview(agg: pd.DataFrame, dataset: str):
    """
    One figure per (dataset × tracked quantity).
    Grid: each subplot = one hardness metric,
          lines = 3 strategies + TVAE baseline (black dash-dot).
    Produces 4 figures per dataset (grad_norm, total_loss, recon_loss, kl_loss).
    """
    subset          = agg[agg['dataset'] == dataset]
    hardness_metrics = KNOWN_METRICS #sorted([m for m in subset['metric'].unique() if m != 'None'])
    n_metrics        = len(hardness_metrics)

    # Grid dimensions — try to keep roughly square
    n_cols = 4
    n_rows = int(np.ceil(n_metrics / n_cols))

    # TVAE baseline — compute once, reused in every subplot
    baseline = subset[
        (subset['metric'] == 'None') & (subset['strategy'] == 'None')
    ].sort_values('epoch')

    colors = {
        strategy: cm.tab10(i)
        for i, strategy in enumerate(['curriculum', 'static', 'self_paced'])
    }

    for col, (title, ylabel) in METRICS_TO_PLOT.items():

        fig, axes = plt.subplots(
            n_rows, n_cols,
            figsize=(15, 4 * n_rows),
            sharex=True, sharey=False
        )
        axes_flat = axes.flatten()
        # fig.suptitle(
        #     f"{title} — {dataset}  (all metrics vs TVAE baseline)",
        #     fontsize=14, fontweight='bold'
        # )

        for idx, metric in enumerate(hardness_metrics):
            ax = axes_flat[idx]

            # ── TVAE baseline ────────────────────────────────────────
            if not baseline.empty:
                mean   = baseline[f'{col}_mean'].values
                std    = baseline[f'{col}_std'].values
                epochs = baseline['epoch'].values
                ax.plot(epochs, mean, **BASELINE_STYLE)
                ax.fill_between(epochs, mean - std, mean + std,
                                alpha=0.12, color='black')

            # ── 3 strategies for this metric ─────────────────────────
            for strategy in ['curriculum', 'static', 'self_paced']:
                mask = (
                    (subset['metric']   == metric) &
                    (subset['strategy'] == strategy)
                )
                data = subset[mask].sort_values('epoch')
                if data.empty:
                    continue

                mean   = data[f'{col}_mean'].values
                std    = data[f'{col}_std'].values
                epochs = data['epoch'].values
                style  = STRATEGY_STYLES[strategy]
                color  = colors[strategy]

                ax.plot(epochs, mean, label=strategy, color=color, **style)
                ax.fill_between(epochs, mean - std, mean + std,
                                alpha=0.15, color=color)

            ax.set_title(metric, fontweight='bold')
            ax.set_ylabel(ylabel)
            ax.grid(True, linestyle='--', alpha=0.4)
            ax.legend(fontsize=8, ncol=1)

        # Hide any unused subplot cells
        for idx in range(n_metrics, len(axes_flat)):
            axes_flat[idx].set_visible(False)

        # Shared x-label on the bottom row
        for ax in axes_flat[(n_rows - 1) * n_cols : n_metrics]:
            ax.set_xlabel('Epoch', fontsize=8)

        plt.tight_layout()
        out_path = os.path.join(PLOTS_OUT, f"{dataset}_overview_{col}.png")
        plt.savefig(out_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  Saved → {out_path}")


# ── 4. Main ───────────────────────────────────────────────────────────────────

def main():
    # METRICS = ["F4", "LSC", "Harmfulness", "CB", "TD_P"]
    print("Loading runs...")
    df = load_all_runs(RUN_METRICS)
    print(f"  {len(df)} epoch-rows loaded from {df['seed'].nunique()} seeds, "
          f"{df['dataset'].nunique()} datasets.")

    print("Aggregating across seeds...")
    agg = aggregate(df)

    print("Plotting...")
    for dataset in agg['dataset'].unique():
        print(f"  → {dataset} | TVAE baseline")
        # plot_tvae_baseline(agg, dataset)
        
        print(f"  → {dataset} | overview (all metrics vs TVAE baseline)")
        plot_dataset_overview(agg, dataset)

        # # All HardTVAE metrics (exclude the TVAE baseline 'None')
        # hardness_metrics = [m for m in METRICS if m != 'None'] #[m for m in agg['metric'].unique() if m != 'None']
        # for metric in hardness_metrics:
        #     print(f"  → {dataset} | {metric}")
        #     plot_one_metric_config(agg, dataset, metric)

    print("Done.")

if __name__ == "__main__":
    main()
# -------------------------------------------------------------------------------
