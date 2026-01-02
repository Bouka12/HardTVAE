"""
    This script analyses the hardness fidelity results to generate heatmaps and boxplots.
    -> boxplot plots for the given-fidelity-view per hardness metric (B) or per weighting strategy (A)

    -> It calculates the KS mean and KS std for different hardness metrics and weighting strategies.
    -> The results are saved as images and tables.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Load the data with correct interpretation
# data_path =  r"C:\Users\BOUKA\Downloads\PART2-Hardness-CSVAE\Hard-CSVAE\results_medical\all_ks_results_medical.csv"
data_path =  r"path-to\fidelity_results_medical\fidelity_results_hardness.csv"
df = pd.read_csv(data_path)
df.columns = ['dataset', 'hardness_metric', 'random_seed', 'weighting_strategy', 'Overall_similarity','ks_mean', 'ks_std', 'ks_pvalue']

print(df.head())
# Replace NaN in hardness_metric with 'Baseline'
df['hardness_metric'] = df['hardness_metric'].fillna('Baseline')
# Remove the IPCA and the relative entropy from the Similarity results:
df = df[~df['hardness_metric'].isin(['relative_entropy', 'pca_contribution'])]
df = df[~df['dataset'].isin(['Haberman','KidneyDisease'])]
df['hardness_metric'] = df['hardness_metric'].str.replace('feature_','', regex=False)

# print("Creating corrected individual dataset heatmaps...")
# print("KS Mean: Mean of KS statistic over features")
# print("KS Std: Standard deviation of KS statistic over features")

# Create output directory
output_dir = Path(r'path-t\Fidelity_Viz\Hardness_plots')
output_dir.mkdir(exist_ok=True)


# Save filtered Distributional fidelity results to CSV
save_path = output_dir / "filtered_Hardness_results.csv"
df.to_csv(save_path, index=False)
print(f"Filtered DataFrame saved to {save_path}")


VIZ_METRIC = 'ks_mean'

# Set up the plotting style
plt.style.use('default')
sns.set_palette("viridis")



baseline_df = df[df["hardness_metric"]== "Baseline"]
hardness_df = df[df["hardness_metric"] != "Baseline"]


####### Plot A
print("Plot 1 ....")
baseline_mean = baseline_df[VIZ_METRIC].mean()

# # Font~
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
    label=fr"Baseline Mean = {baseline_mean:.3f}"
)

plt.legend(prop={'family': 'Times New Roman', 'size': 8})
plt.xlabel("", fontsize=8)
plt.ylabel(r"Mean of KS statistic", fontsize=8, fontname="Times New Roman")
# plt.title("Weighting Strategies vs Baseline", fontsize=16)
# plt.legend()
plt.tight_layout()
pathC = output_dir / f"Hardness_boxplot_strategies_vs_baseline_mean_{VIZ_METRIC}.png"
plt.savefig(pathC, dpi=600, bbox_inches='tight')
plt.close()

print("Saved:", pathC)

### Plot for RQ 3 and Appendix: distributional fidelity distribution per hardness measure
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
    label=fr"Baseline Mean = {baseline_mean:.3f}$"
)

plt.legend(prop={'family': 'Times New Roman', 'size': 8})
plt.xlabel(r"", fontsize=8)
plt.ylabel(r"Mean of KS statistic", fontsize=8, fontname="Times New Roman")
# plt.title(r"Hardness-based Fidelity per Hardness Metric (Averaged over Strategies)}", fontsize=16)
# plt.xticks(rotation=45)
# plt.legend()
plt.tight_layout()

box_path = output_dir / f"hardness_boxplot_per_metric_avg_{VIZ_METRIC}.png"
plt.savefig(box_path, dpi=600, bbox_inches='tight')
plt.close()

print(f"Saved: {box_path}")
##################################
### Heatmap/tables generation with the code below (already executed):
##################################

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
#     grouped_mean = dataset_df.groupby(['hardness_metric', 'weighting_strategy'])['ks_mean'].mean().reset_index()
    
#     print(f"  Grouped records (mean): {len(grouped_mean)}")
#     # print(f"  Grouped records (std): {len(grouped_std)}")
    
#     # Handle baseline case - baseline only exists for static weighting strategy
#     # We need to replicate it for other strategies as mentioned in requirements
#     baseline_mean = grouped_mean[grouped_mean['hardness_metric'] == 'Baseline']
    
#     if not baseline_mean.empty:
#         baseline_ks_mean = baseline_mean[baseline_mean['weighting_strategy'] == 'static']['ks_mean'].iloc[0]
        
#         print(f"  Baseline KS mean: {baseline_ks_mean:.3f}")
        
#         # Add baseline for other weighting strategies
#         for strategy in ['curriculum', 'self_paced']:
#             if strategy not in baseline_mean['weighting_strategy'].values:
#                 new_row_mean = pd.DataFrame({
#                     'hardness_metric': ['Baseline'],
#                     'weighting_strategy': [strategy],
#                     'ks_mean': [baseline_ks_mean]
#                 })
#                 grouped_mean = pd.concat([grouped_mean, new_row_mean], ignore_index=True)

    
#     # Create pivot tables for heatmaps
#     pivot_mean = grouped_mean.pivot(index='hardness_metric', columns='weighting_strategy', values='ks_mean')
    
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


#     # CREATE TABLE WITH MEAN +- STD FOR EACH DATASET:

    
#     # Group by mean and std together
#     grouped = dataset_df.groupby(['hardness_metric', 'weighting_strategy']).agg(
#         mean_val=('ks_mean', 'mean'),
#         std_val=('ks_mean', 'std')  # or std_val=('Wasserstein_H0_mean', 'std') depending on goal
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
#     pivot_table_path_csv = output_dir / f'{dataset_name}_Hardness_ks_mean_table.csv'
#     pivot_table_path_tex = output_dir / f'{dataset_name}_Hardness_ks_mean_table.tex'
#     pivot_table.to_csv(pivot_table_path_csv)
#     pivot_table.to_latex(pivot_table_path_tex, escape=False)

#     print(f"  Saved table to {pivot_table_path_csv}")
    
#     return pivot_mean #, pivot_std

# # Create heatmaps for each dataset
# datasets = sorted(df['dataset'].unique())
# print(f"Found {len(datasets)} datasets: {datasets}")

# for dataset in datasets:
#     result = create_dataset_heatmap(dataset)
    
#     if result is None:
#         continue
        
#     pivot_mean = result
    
#     # Create figure with subplots for both KS mean and KS std
#     fig, ax = plt.subplots( figsize=(16, max(8, len(pivot_mean) * 0.4)))
    
#     # KS Mean heatmap
#     sns.heatmap(pivot_mean, annot=True, fmt='.3f', cmap='YlGnBu', 
#                ax=ax, cbar_kws={'label': 'KS Mean'})
#     ax.set_title(f'{dataset} - KS Mean\n(Mean of KS statistic over features)')
#     ax.set_xlabel('Weighting Strategy')
#     ax.set_ylabel('Hardness Metric')
    
    
#     plt.tight_layout()
    
#     # Save the plot
#     output_path = output_dir / f'{dataset}_HardnessKS_heatmap.png'
#     plt.savefig(output_path, dpi=300, bbox_inches='tight')
#     plt.close()
    
#     print(f"  Saved heatmap to {output_path}")

# print(f"\nAll corrected heatmaps saved to {output_dir}")

# # Create a summary heatmap across all datasets
# print("\nCreating summary heatmap across all datasets...")

# # Group by hardness_metric and weighting_strategy across all datasets
# all_grouped_mean = df.groupby(['hardness_metric', 'weighting_strategy'])['ks_mean'].mean().reset_index()

# # Handle baseline case for summary
# baseline_mean = all_grouped_mean[all_grouped_mean['hardness_metric'] == 'Baseline']

# if not baseline_mean.empty:
#     baseline_ks_mean = baseline_mean[baseline_mean['weighting_strategy'] == 'static']['ks_mean'].iloc[0]
    
#     # Add baseline for other weighting strategies
#     for strategy in ['curriculum', 'self_paced']:
#         if strategy not in baseline_mean['weighting_strategy'].values:
#             new_row_mean = pd.DataFrame({
#                 'hardness_metric': ['Baseline'],
#                 'weighting_strategy': [strategy],
#                 'ks_mean': [baseline_ks_mean]
#             })
#             all_grouped_mean = pd.concat([all_grouped_mean, new_row_mean], ignore_index=True)
            


# # Create pivot tables
# summary_pivot_mean = all_grouped_mean.pivot(index='hardness_metric', columns='weighting_strategy', values='ks_mean')

# # Ensure all weighting strategies are present and reorder
# for pivot in [summary_pivot_mean]:#, summary_pivot_std]:
#     for strategy in ['static', 'curriculum', 'self_paced']:
#         if strategy not in pivot.columns:
#             pivot[strategy] = np.nan

# # Reorder columns
# summary_pivot_mean = summary_pivot_mean[['static', 'curriculum', 'self_paced']]

# # Sort rows - put Baseline first, then alphabetically
# if 'Baseline' in summary_pivot_mean.index:
#     other_indices = sorted([idx for idx in summary_pivot_mean.index if idx != 'Baseline'])
#     summary_pivot_mean = summary_pivot_mean.reindex(['Baseline'] + other_indices)

# # Create summary plot
# fig, ax = plt.subplots(figsize=(16, 12))
# vmin = summary_pivot_mean.min().min()
# vmax = summary_pivot_mean.max().max()
# sns.heatmap(summary_pivot_mean, annot=True, fmt='.3f', cmap='YlGnBu', vmax=vmax,
#             vmin=vmin,  # Set min and max for consistent color scaling
#             annot_kws={'size': 14},  # Increase font size for annotations
#            ax=ax, cbar_kws={'label': 'KS Mean'})
# # ax.set_title('Summary Across All Datasets - KS Mean\n(Mean of KS statistic over features)')
# ax.set_xlabel('Weighting Strategy', fontsize=14)  # Increase axis label font size
# ax.set_ylabel('Hardness Metric', fontsize=14)     # Increase axis label font size

# # Increase tick label font sizes
# ax.tick_params(axis='x', labelsize=12)
# ax.tick_params(axis='y', labelsize=12)



# plt.tight_layout()

# # Save summary plot
# summary_path = output_dir / 'summary_all_datasets_HardnessKS_heatmap.png'
# plt.savefig(summary_path, dpi=300, bbox_inches='tight')
# plt.close()

# print(f"Summary heatmap saved to {summary_path}")

# # Print some statistics
# print(f"\nCorrected Summary Statistics:")
# print(f"Total number of hardness metrics (including baseline): {len(summary_pivot_mean.index)}")
# print(f"Hardness metrics: {list(summary_pivot_mean.index)}")
# print(f"Weighting strategies: {list(summary_pivot_mean.columns)}")
# print(f"Average KS mean across all combinations: {summary_pivot_mean.values.mean():.3f}")

# # Show some key insights
# print(f"\n=== KEY INSIGHTS (CORRECTED) ===")
# print("Highest KS means (indicating largest differences between distributions):")
# max_values = summary_pivot_mean.values.flatten()
# max_indices = np.unravel_index(np.argsort(max_values)[-5:], summary_pivot_mean.shape)
# for i in range(5):
#     row_idx, col_idx = max_indices[0][-(i+1)], max_indices[1][-(i+1)]
#     metric = summary_pivot_mean.index[row_idx]
#     strategy = summary_pivot_mean.columns[col_idx]
#     value = summary_pivot_mean.iloc[row_idx, col_idx]
#     print(f"  {metric} + {strategy}: {value:.3f}")



# print("Creating additional visualizations...")


# # Set up the plotting style
# plt.style.use('default')
# sns.set_palette("Set2")

# # 1. Box plots comparing weighting strategies
# print("1. Creating box plots for weighting strategies...")

# fig, ax1= plt.subplots(figsize=(15, 6))

# # KS Statistic box plot
# sns.boxplot(data=df, x='weighting_strategy', y='ks_mean', ax=ax1)
# ax1.set_title('Distribution of Mean KS Statistics by Weighting Strategy')
# ax1.set_xlabel('Weighting Strategy')
# ax1.set_ylabel('KS Statistic')
# ax1.tick_params(axis='x', rotation=45)


# plt.tight_layout()
# plt.savefig(output_dir / 'Hardness_weighting_strategy_boxplots.png', dpi=300, bbox_inches='tight')
# plt.close()
