import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import re
import os

METRICS = ["TD_P",  "CB", "LSC", "F4" ]


def load_and_aggregate_hardness(directory='.'):
    """
    Loads all hardness CSV files and aggregates mean/std across seeds.
    """
    all_data = []
    # Pattern: {dataset}_{metric}_seed{seed}_hardness_scores.csv
    files = glob.glob(os.path.join(directory, "*_hardness_scores.csv"))
    
    for file in files:
        basename = os.path.basename(file)
        match = re.match(r"(.+?)_(.+?)_seed(\d+)_hardness_scores\.csv", basename)
        if match:
            dataset, metric, seed = match.groups()
            df = pd.read_csv(file)
            
            # Assume rows represent instances; create index if not present
            if 'instance_id' not in df.columns:
                df['instance_id'] = df.index
            
            # Identify the score column (usually the only other numeric column)
            score_col = 'hardness_score' if 'hardness_score' in df.columns else df.select_dtypes(include=[np.number]).columns[0]
            
            temp_df = df[['instance_id', score_col]].copy()
            temp_df.rename(columns={score_col: 'score'}, inplace=True)
            temp_df['dataset'] = dataset
            temp_df['metric'] = metric
            temp_df['seed'] = int(seed)
            all_data.append(temp_df)
    
    if not all_data:
        return None
        
    full_df = pd.concat(all_data)
    # Aggregate mean and std across the 10 seeds for each specific instance
    agg_df = full_df.groupby(['dataset', 'metric', 'instance_id'])['score'].agg(['mean', 'std']).reset_index()
    return agg_df

def plot_individual_distributions(df_stats, metrics=METRICS):
    """
    Function 1: Plots the distribution for each hardness metric and dataset combination.
    """
    datasets = df_stats['dataset'].unique()
    # metrics = df_stats['metric'].unique()
    
    for dataset in datasets:
        for metric in metrics:
            data = df_stats[(df_stats['dataset'] == dataset) & (df_stats['metric'] == metric)]
            if data.empty: continue
            
            plt.figure(figsize=(7, 4))
            sns.histplot(data['mean'], kde=True, color='skyblue', bins=25)
            plt.title(f"Hardness Dist: {dataset} | Metric: {metric}\n(Aggregated over 10 seeds)")
            plt.xlabel(r"Mean Hardness Score ($\mu$)")
            plt.ylabel("Frequency")
            plt.tight_layout()
            PATH = os.path.join(PLOTS_OUT, f"dist_{dataset}_{metric}.png")
            plt.savefig(PATH)
            plt.close()

def plot_dataset_overview(df_stats, dataset_name, metrics=METRICS):
    """
    Function 2: Plots all metric distributions for a specific dataset in one overview plot.
    """
    data = df_stats[df_stats['dataset'] == dataset_name]
    # metrics = sorted(data['metric'].unique())
    
    n_metrics = len(metrics)
    cols = 3
    rows = (n_metrics + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(16, 4 * rows))
    axes = axes.flatten()
    
    for i, metric in enumerate(metrics):
        subset = data[data['metric'] == metric]
        sns.kdeplot(subset['mean'], ax=axes[i], fill=True, color='teal')
        axes[i].set_title(f"Metric: {metric}")
        axes[i].set_xlabel(r"Mean Hardness ($\mu$)")
        axes[i].set_ylabel("Density")
        
    # Clean up empty subplots
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])
        
    plt.suptitle(f"Hardness Overview: {dataset_name} (Averaged across 10 Seeds)", fontsize=18)
    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    out_path = os.path.join(PLOTS_OUT, f"{dataset_name}_hardness_overview.png")
    plt.savefig(out_path)
    plt.close()


def plot_kde_grid(df_agg, dataset_name, metrics = METRICS):
    """Plots a grid of KDEs for all metrics in one dataset."""
    subset = df_agg[df_agg['dataset'] == dataset_name]
    # metrics = sorted(subset['metric'].unique())

    
    # Set up the grid
    cols = 4
    n_rows = int(np.ceil(len(metrics) / cols)) #
    rows = (len(metrics) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(15, 4 * n_rows), sharex=True)
    axes = axes.flatten()
    
    for i, m in enumerate(metrics):
        data = subset[subset['metric'] == m]['mean']
        sns.kdeplot(data, ax=axes[i], fill=True, color='black', alpha=0.5)
        axes[i].set_title(f"{m}", fontweight='bold')
        axes[i].set_xlim(-0.1, 1.1) # Assuming hardness is 0-1
        axes[i].set_ylabel("Density")
        
    for j in range(i + 1, len(axes)): fig.delaxes(axes[j])
    
    # plt.suptitle(f"Detailed Density Shapes: {dataset_name}", fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    path = os.path.join(PLOTS_OUT, f"kde_grid_{dataset_name}.png")
    plt.savefig(path, dpi=300, bbox_inches='tight')


# let's plot
RESULTS_DIR = "hardvae_code\\gradients_stability_analysis\\RESULTS-GRADIENT-ANALYSIS"
HARDNESS_SCORES_DIR = os.path.join(RESULTS_DIR, "hardness_scores")
PLOTS_OUT    = os.path.join(RESULTS_DIR, "hardness_plots")
os.makedirs(PLOTS_OUT, exist_ok=True)
df_stats = load_and_aggregate_hardness(HARDNESS_SCORES_DIR)
plot_individual_distributions(df_stats)
plot_dataset_overview(df_stats, "Hypothyroid")
plot_kde_grid(df_stats, "Hypothyroid")
plot_kde_grid(df_stats, "NewThyroid1")
plot_kde_grid(df_stats, "Vertebral")
plot_kde_grid(df_stats, "BCWDD")
plot_kde_grid(df_stats, "ILPD")