"""
    This script processes topological fidelity metrics from a CSV file:
        -> boxplot plots for topo-fidelity per hardness metric or per weighting strategy
        -> Generate heatmaps and tables for topological fidelity metrics from CSV data.
        -> Saves individual dataset heatmaps and a summary heatmap across all datasets.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'serif' #o la que te dé la gana

# Load the data
data_path = r"path-to\fidelity_results_medical\fidelity_results_topological.csv"
df = pd.read_csv(data_path )

# Set column names based on the description
df.columns = ['dataset', 'hardness_metric', 'random_seed', 'weighting_strategy', 'Bottleneck_H0_mean', 'Bottleneck_H0_std', 'Bottleneck_H1_mean', 'Bottleneck_H1_std', 'Wasserstein_H0_mean', 'Wasserstein_H0_std', 'Wasserstein_H1_mean', 'Wasserstein_H1_std']

# Replace NaN in hardness_metric with 'Baseline'
df['hardness_metric'] = df['hardness_metric'].fillna('Baseline')
# Remove the IPCA and the relative entropy from the Similarity results:
df = df[~df['hardness_metric'].isin(['relative_entropy', 'pca_contribution'])]
df = df[~df['dataset'].isin(['Haberman','KidneyDisease'])]
# Prepare the column 'hardness_metric' --> "feature_{real_hardness_metric}"
df['hardness_metric'] = df['hardness_metric'].str.replace('feature_', '', regex=False)
print(df['hardness_metric'].unique())
# Create output directory
output_dir = Path(r'path-to\Fidelity_Viz\Topology_plots')
output_dir.mkdir(exist_ok=True)

# Save to CSV
save_path = output_dir / "filtered_Topological_results.csv"  # change path/filename as needed
df.to_csv(save_path, index=False)

print(f"Filtered DataFrame saved to {save_path}")
VIZ_METRIC = 'Bottleneck_H0_mean'

# Set up the plotting style
plt.style.use('default')
sns.set_palette("viridis")

baseline_df = df[df["hardness_metric"] == "Baseline"]
hardness_df = df[df["hardness_metric"] != "Baseline"]

########## Plot A
print("Plot 1....")
baseline_mean = baseline_df[VIZ_METRIC].mean()
# plt.rcParams['text.usetex'] = True
# plt.rcParams['font.family'] = 'serif' #o la que te dé la gana


plt.figure(figsize=(8, 6))

sns.boxplot(
    data=hardness_df,
    x='weighting_strategy',
    y=VIZ_METRIC,
    palette="Set2"
)
plt.xticks(fontfamily="Times New Roman", fontsize=8)
plt.yticks(fontfamily="Times New Roman", fontsize=8)
plt.axhline(
    y=baseline_mean,
    color = 'red',
    linestyle = '--',
    linewidth=2,
    label=f"Baseline Mean = {baseline_mean:.3f}"
)

plt.legend(prop={'family': 'Times New Roman', 'size': 8})
plt.xlabel("", fontsize=8, fontname="Times New Roman")
plt.ylabel("Mean of Bottleneck distance (H0)", fontsize=8, fontname="Times New Roman")
# plt.title("Weighting Strategies vs Baseline", fontsize=16)
# plt.legend()
plt.tight_layout()
pathC = output_dir / f"topology_boxplot_strategies_vs_baseline_mean_{VIZ_METRIC}.png"
plt.savefig(pathC, dpi=600, bbox_inches='tight')
plt.close()

print("Saved:", pathC)


### Plot for RQ 3 and Appendix: topology fidelity distribution per hardness measure
# STEP 1: prepare the data
# Exclude baseline for averaging
print("Plot 2....")

hardness_df = df[df["hardness_metric"]!= "Baseline"]

# Average over weighting strategies for each hardness metric
avg_per_metric = hardness_df.groupby("hardness_metric")[VIZ_METRIC].mean().reset_index()

# Get baseline mean (to show as threshold)
baseline_mean = df[df["hardness_metric"] == "Baseline"][VIZ_METRIC].mean()

# STEP 2: Plot boxplot per Hardness Metric (weighting startegy-averaged) with Baseline bound

plt.figure(figsize=(12, 6))

sns.boxplot(
    data=hardness_df,
    x="hardness_metric",
    y=VIZ_METRIC,
    color="lightblue"
)
plt.xticks(fontfamily="Times New Roman", fontsize=8)
plt.yticks(fontfamily="Times New Roman", fontsize=8)
# Overlay the baseline mean as a red dashed line
plt.axhline(
    y=baseline_mean,
    color='red',
    linestyle='--',
    linewidth=2,
    label=f"Baseline Mean = {baseline_mean:.3f}"
)

plt.legend(prop={'family': 'Times New Roman', 'size': 8})
plt.xlabel("", fontsize=8, fontname="Times New Roman")
plt.ylabel(f"Mean of Bottleneck Distance (H0)", fontsize=8, fontname="Times New Roman")
# plt.title("Topological Fidelity per Hardness Metric (Averaged over Strategies)", fontsize=8, fontname="Times New Roman")
# plt.xticks(rotation=45)
# plt.legend()
plt.tight_layout()

box_path = output_dir / f"topology_boxplot_per_metric_avg_{VIZ_METRIC}.png"
plt.savefig(box_path, dpi=600, bbox_inches='tight')
plt.close()

print(f"Saved: {box_path}")


###############
# heatmap generation
###############


# def create_dataset_heatmap(dataset_name):
#     """Create heatmap for a specific dataset"""
#     print(f"\nProcessing dataset: {dataset_name}")
    
#     # Filter data for the specific dataset
#     dataset_df = df[df['dataset'] == dataset_name].copy()
    
#     if dataset_df.empty:
#         print(f"No data found for dataset: {dataset_name}")
#         return None
    
#     print(f"  Records: {len(dataset_df)}")
#     print(f"  Hardness metrics: {dataset_df['hardness_metric'].nunique()}")
#     print(f"  Weighting strategies: {dataset_df['weighting_strategy'].nunique()}")
    
#     # Group by hardness_metric and weighting_strategy, then average across random seeds
#     grouped_mean = dataset_df.groupby(['hardness_metric', 'weighting_strategy'])[VIZ_METRIC].mean().reset_index()

    
#     print(f"  Grouped records (mean): {len(grouped_mean)}")

#     # Handle baseline case - baseline only exists for static weighting strategy
#     # We need to replicate it for other strategies as mentioned in requirements
#     baseline_mean = grouped_mean[grouped_mean['hardness_metric'] == 'Baseline']

    
#     if not baseline_mean.empty:
#         baseline_ks_mean = baseline_mean[baseline_mean['weighting_strategy'] == 'static'][VIZ_METRIC].iloc[0]

        
#         print(f"  Baseline KS mean: {baseline_ks_mean:.3f}")

        
#         # Add baseline for other weighting strategies
#         for strategy in ['curriculum', 'self_paced']:
#             if strategy not in baseline_mean['weighting_strategy'].values:
#                 new_row_mean = pd.DataFrame({
#                     'hardness_metric': ['Baseline'],
#                     'weighting_strategy': [strategy],
#                     VIZ_METRIC: [baseline_ks_mean]
#                 })
#                 grouped_mean = pd.concat([grouped_mean, new_row_mean], ignore_index=True)
                

#     # Create pivot tables for heatmaps
#     pivot_mean = grouped_mean.pivot(index='hardness_metric', columns='weighting_strategy', values=VIZ_METRIC)

    
#     # Ensure all weighting strategies are present
#     for strategy in ['static', 'curriculum', 'self_paced']:
#         if strategy not in pivot_mean.columns:
#             pivot_mean[strategy] = np.nan

    
#     # Reorder columns
#     pivot_mean = pivot_mean[['static', 'curriculum', 'self_paced']]

    
#     # Sort rows - put Baseline first, then alphabetically
#     if 'Baseline' in pivot_mean.index:
#         other_indices = sorted([idx for idx in pivot_mean.index if idx != 'Baseline'])
#         pivot_mean = pivot_mean.reindex(['Baseline'] + other_indices)

#     print(f"  Final pivot shape: {pivot_mean.shape}")
    
#     # Group by mean and std together
#     grouped = dataset_df.groupby(['hardness_metric', 'weighting_strategy']).agg(
#         mean_val=(VIZ_METRIC, 'mean'),
#         std_val=(VIZ_METRIC, 'std')  # or std_val=('Wasserstein_H0_mean', 'std') depending on goal
#     ).reset_index()

#     # Format as "mean ± std"
#     grouped['mean_std'] = grouped.apply(
#         lambda row: f"{row['mean_val']:.3f} ± {row['std_val']:.3f}", axis=1
#     )

#     # Fill missing strategies for Baseline
#     baseline = grouped[grouped['hardness_metric'] == 'Baseline']
#     if not baseline.empty:
#         baseline_val = baseline[baseline['weighting_strategy'] == 'static']['mean_std'].iloc[0]
#         for strategy in ['curriculum', 'self_paced']:
#             if not ((grouped['hardness_metric'] == 'Baseline') & (grouped['weighting_strategy'] == strategy)).any():
#                 new_row = {
#                     'hardness_metric': 'Baseline',
#                     'weighting_strategy': strategy,
#                     'mean_val': np.nan,
#                     'std_val': np.nan,
#                     'mean_std': baseline_val
#                 }
#                 grouped = pd.concat([grouped, pd.DataFrame([new_row])], ignore_index=True)

#     # Pivot to table format
#     pivot_table = grouped.pivot(index='hardness_metric', columns='weighting_strategy', values='mean_std')

#     # Ensure all strategies exist
#     for strategy in ['static', 'curriculum', 'self_paced']:
#         if strategy not in pivot_table.columns:
#             pivot_table[strategy] = np.nan

#     # Reorder and sort
#     pivot_table = pivot_table[['static', 'curriculum', 'self_paced']]
#     if 'Baseline' in pivot_table.index:
#         other_rows = sorted([r for r in pivot_table.index if r != 'Baseline'])
#         pivot_table = pivot_table.reindex(['Baseline'] + other_rows)

#     # Save the table to CSV and LaTeX
#     pivot_table_path_csv = output_dir / f'{dataset_name}_Topology{VIZ_METRIC}_table.csv'
#     pivot_table_path_tex = output_dir / f'{dataset_name}_Topology{VIZ_METRIC}_table.tex'
#     pivot_table.to_csv(pivot_table_path_csv)
#     pivot_table.to_latex(pivot_table_path_tex, escape=False)

#     print(f"  Saved table to {pivot_table_path_csv}")

#     return pivot_mean #, pivot_std

# # Create heatmaps for each dataset
# datasets = sorted(df['dataset'].unique())
# print(f"Found {len(datasets)} datasets: {datasets}")

##############################################################
####    Uncomment to generate topology heatmap per dataset
##############################################################

# for dataset in datasets:
#     result = create_dataset_heatmap(dataset)
    
#     if result is None:
#         continue
        
#     pivot_mean = result
#     plt.rcParams['text.usetex'] = True
#     plt.rcParams['font.family'] = 'serif' #o la que te dé la gana
#     # Create figure with subplots for both KS mean and KS std
#     fig, ax = plt.subplots( figsize=(16, max(8, len(pivot_mean) * 0.4)))
    
#     # KS Mean heatmap
#     sns.heatmap(pivot_mean, annot=True, fmt='.3f', cmap='viridis', 
#                ax=ax, cbar_kws={'label': VIZ_METRIC})
#     ax.set_title(f'{dataset} - {VIZ_METRIC} Mean\n(Mean of {VIZ_METRIC} )')
#     ax.set_xlabel(r'\textit{Weighting Strategy}')
#     ax.set_ylabel(r'\textit{Hardness Metric}')
    
#     plt.tight_layout()
    
#     # Save the plot
#     output_path = output_dir / f'{dataset}_Topology{VIZ_METRIC}_heatmap.png'
#     plt.savefig(output_path, dpi=300, bbox_inches='tight')
#     plt.close()
    
#     print(f"  Saved heatmap to {output_path}")
##############################################################
# print(f"\nAll corrected heatmaps saved to {output_dir}")


# # Create summary heatmap across all datasets 
# print("\nCreating summary heatmap across all datasets...")

# # Group by hardness_metric and weighting_strategy across all datasets
# all_grouped = df.groupby(['hardness_metric', 'weighting_strategy'])[VIZ_METRIC].mean().reset_index()

# # Handle baseline case for summary
# baseline_data = all_grouped[all_grouped['hardness_metric'] == 'Baseline']
# if not baseline_data.empty:
#     baseline_diff = baseline_data[baseline_data['weighting_strategy'] == 'static'][VIZ_METRIC].iloc[0]
    
#     # Add baseline for other weighting strategies
#     for strategy in ['curriculum', 'self_paced']:
#         if strategy not in baseline_data['weighting_strategy'].values:
#             new_row = pd.DataFrame({
#                 'hardness_metric': ['Baseline'],
#                 'weighting_strategy': [strategy],
#                 VIZ_METRIC : [baseline_diff]
#             })
#             all_grouped = pd.concat([all_grouped, new_row], ignore_index=True)

# # Create pivot table
# summary_pivot = all_grouped.pivot(index='hardness_metric', columns='weighting_strategy', values=VIZ_METRIC)

# # Ensure all weighting strategies are present and reorder
# for strategy in ['static', 'curriculum', 'self_paced']:
#     if strategy not in summary_pivot.columns:
#         summary_pivot[strategy] = np.nan

# summary_pivot = summary_pivot[['static', 'curriculum', 'self_paced']]

# # Sort rows - put Baseline first, then alphabetically
# if 'Baseline' in summary_pivot.index:
#     other_indices = sorted([idx for idx in summary_pivot.index if idx != 'Baseline'])
#     summary_pivot = summary_pivot.reindex(['Baseline'] + other_indices)


# plt.rcParams['text.usetex'] = True
# plt.rcParams['font.family'] = 'serif' #o la que te dé la gana
# # Create summary plot
# fig, ax = plt.subplots(figsize=(16, 12))
# # Determine min and max values for the colormap scale
# vmin = summary_pivot.min().min()
# vmax = summary_pivot.max().max()
# sns.heatmap(summary_pivot, annot=True, fmt='.3f', cmap='YlGnBu', vmin=vmin,
#             vmax=vmax, ax=ax, 
#             annot_kws={'size': 14},  # Increase font size for annotations
#             cbar_kws={'label': r'\textit{Mean of Bottleneck Distance (H0)}'})
# # ax.set_title(f'Summary Across All Datasets - Topology-Mean of \n{VIZ_METRIC}')
# ax.set_xlabel(r'\textit{Weighting Strategy}', fontsize=14)  # Increase axis label font size
# # ax.set_ylabel(r'\textit{Hardness Metric}', fontsize=14)     # Increase axis label font size
# ax.set_ylabel('')
# # Increase tick label font sizes
# ax.tick_params(axis='x', labelsize=12)
# ax.tick_params(axis='y', labelsize=12)

# plt.tight_layout()

# # Save summary plot
# summary_path = output_dir / f'summary_topology_{VIZ_METRIC}_heatmap.png'
# plt.savefig(summary_path, dpi=300, bbox_inches='tight')
# plt.close()

# print(f"Summary heatmap saved to {summary_path}")

