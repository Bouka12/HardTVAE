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

RESULTS_DIR  = "RESULTS-GRADIENT-ANALYSIS"
RUN_METRICS  = os.path.join(RESULTS_DIR, "run_metrics")
PLOTS_OUT    = os.path.join(RESULTS_DIR, "stability_plots")
os.makedirs(PLOTS_OUT, exist_ok=True)


# ── 1. Load all saved CSVs ────────────────────────────────────────────────────

def load_all_runs(run_metrics_dir: str) -> pd.DataFrame:
    frames = []
    for fname in os.listdir(run_metrics_dir):
        if fname.endswith(".csv"):
            frames.append(pd.read_csv(os.path.join(run_metrics_dir, fname)))
    if not frames:
        raise FileNotFoundError(f"No CSV files found in {run_metrics_dir}")
    df = pd.concat(frames, ignore_index=True)
    # Normalise None strings
    df['metric'] = df['metric'].fillna('None').astype(str)
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
    'grad_norm':   ('Gradient L2 Norm',  'L2 Norm'),
    'total_loss':  ('Total Loss',         'Loss'),
    'recon_loss':  ('Reconstruction Loss','Loss'),
    'kl_loss':     ('KL Divergence',      'Loss'),
}

def plot_stability(agg: pd.DataFrame, dataset: str):
    """
    One figure per dataset, one row per tracked metric.
    Each line = one configuration (metric × strategy); shaded band = ± 1 std.
    """
    configs = agg[
        (agg['dataset'] == dataset)
    ][['metric', 'strategy']].drop_duplicates()

    n_configs = len(configs)
    colors = cm.tab10(np.linspace(0, 1, n_configs))

    n_rows = len(METRICS_TO_PLOT)
    fig, axes = plt.subplots(n_rows, 1, figsize=(10, 4 * n_rows), sharex=True)
    fig.suptitle(f"Gradient & Loss Stability — {dataset}", fontsize=14, fontweight='bold')

    subset = agg[agg['dataset'] == dataset]

    for ax, (col, (title, ylabel)) in zip(axes, METRICS_TO_PLOT.items()):
        for (_, row), color in zip(configs.iterrows(), colors):
            mask  = (subset['metric'] == row['metric']) & (subset['strategy'] == row['strategy'])
            data  = subset[mask].sort_values('epoch')
            label = f"{row['metric']} | {row['strategy']}"

            mean = data[f'{col}_mean'].values
            std  = data[f'{col}_std'].values
            epochs = data['epoch'].values

            ax.plot(epochs, mean, label=label, color=color, linewidth=1.8)
            ax.fill_between(epochs, mean - std, mean + std,
                            alpha=0.15, color=color)

        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.grid(True, linestyle='--', alpha=0.4)
        ax.legend(fontsize=7, ncol=2)

    axes[-1].set_xlabel('Epoch')
    plt.tight_layout()

    out_path = os.path.join(PLOTS_OUT, f"{dataset}_stability.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  Saved → {out_path}")


# ── 4. Main ───────────────────────────────────────────────────────────────────

def main():
    print("Loading runs...")
    df = load_all_runs(RUN_METRICS)
    print(f"  {len(df)} epoch-rows loaded from {df['seed'].nunique()} seeds, "
          f"{df['dataset'].nunique()} datasets.")

    print("Aggregating across seeds...")
    agg = aggregate(df)

    print("Plotting...")
    for dataset in agg['dataset'].unique():
        print(f"  → {dataset}")
        plot_stability(agg, dataset)

    print("Done.")


if __name__ == "__main__":
    main()