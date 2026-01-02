"""
    This script analyses the complexity fidelity results to generate heatmaps and boxplots.
    -> A and B boxplots for effect analysis
    -> It calculates the difference score as the absolute difference between mean_real_score and mean_synth_score.
    -> The results are saved as images and tables.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Load the complexity fidelity data
data_path = r"path-to\fidelity_results_medical\fidelity_results_complexity.csv"
df = pd.read_csv(data_path)

# print("=== COMPLEXITY FIDELITY ANALYSIS ===")
# print("Creating heatmaps and boxplots for complexity analysis")

# Replace NaN in hardness_metric with 'Baseline'
df['hardness_metric'] = df['hardness_metric'].fillna('Baseline')

# Remove the IPCA and the relative entropy from the Similarity results:
df = df[~df['hardness_metric'].isin(['relative_entropy', 'pca_contribution'])]
df = df[~df['dataset'].isin(['Haberman','KidneyDisease'])]
df['hardness_metric'] = df['hardness_metric'].str.replace('feature_','',regex=False)

# Calculate difference score (mean_real_score - mean_synth_score)
# First, let's verify if the existing score_difference is correct
df['calculated_difference'] = np.abs(df['mean_real_score'] - df['mean_synth_score'])

# print("Verifying score_difference calculation:")
# print("Existing score_difference vs calculated (mean_real - mean_synth):")
# print("Are they equal?", np.allclose(df['score_difference'], df['calculated_difference'], rtol=1e-10))
# print("Max difference:", np.max(np.abs(df['score_difference'] - df['calculated_difference'])))

# Use the calculated difference to be sure
df['difference_score'] = df['calculated_difference']

# print(f"\nDifference score statistics:")
# print(df['difference_score'].describe())

# Create output directory
output_dir = Path(r'path-to\Fidelity_Viz\Complexity_plots')
output_dir.mkdir(exist_ok=True)

# Save filtered Distributional fidelity results to CSV
save_path = output_dir / "filtered_Complexity_results.csv"
df.to_csv(save_path, index=False)
print(f"Filtered DataFrame saved to {save_path}")

VIZ_METRIC = 'difference_score'

# Set up plotting style
plt.style.use('default')
sns.set_palette("RdYlBu_r")

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
plt.ylabel(r"Mean of Absolute Difference of Complexity Scores", fontsize=8, fontname="Times New Roman")
# plt.title("Weighting Strategies vs Baseline", fontsize=16)
# plt.legend()
plt.tight_layout()
pathC = output_dir / f"complexity_boxplot_strategies_vs_baseline_mean_{VIZ_METRIC}.png"
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
    label=fr"Baseline Mean = {baseline_mean:.3f}"
)

plt.legend(prop={'family': 'Times New Roman', 'size': 8})
plt.xlabel(r"", fontsize=8)
plt.ylabel(r"Mean of Absolute Difference of Complexity Scores", fontsize=8, fontname="Times New Roman")
# plt.title(r"\textit{Complexity-based Fidelity per Hardness Metric (Averaged over Strategies)}", fontsize=16)
# plt.xticks(rotation=45)
# plt.legend()
plt.tight_layout()

box_path = output_dir / f"complexity_boxplot_per_metric_avg_{VIZ_METRIC}.png"
plt.savefig(box_path, dpi=600, bbox_inches='tight')
plt.close()

print(f"Saved: {box_path}")


## Optional:
# print(f"\nData overview:")
# print(f"Total records: {len(df)}")
# print(f"Datasets: {df['dataset'].nunique()}")
# print(f"Hardness metrics: {df['hardness_metric'].nunique()}")
# print(f"Weighting strategies: {df['weighting_strategy'].nunique()}")
# print(f"Seeds: {df['seed'].nunique()}")

# # Check baseline distribution
# baseline_data = df[df['hardness_metric'] == 'Baseline']
# print(f"\nBaseline data:")
# print(f"Total baseline records: {len(baseline_data)}")
# print("Baseline by weighting strategy:")
# print(baseline_data['weighting_strategy'].value_counts())

# print("\n=== CREATING HEATMAPS ===")

# def create_complexity_heatmap_for_dataset(dataset_name):
#     """Create heatmap for a specific dataset"""
#     print(f"\nProcessing dataset: {dataset_name}")
    
#     # Filter data for the specific dataset
#     dataset_df = df[df['dataset'] == dataset_name].copy()
    
#     if dataset_df.empty:
#         print(f"No data found for dataset: {dataset_name}")
#         return None
    
#     print(f"  Records: {len(dataset_df)}")
    
#     # Group by hardness_metric and weighting_strategy, then average across seeds
#     grouped = dataset_df.groupby(['hardness_metric', 'weighting_strategy'])['difference_score'].mean().reset_index()
    
#     print(f"  Grouped records: {len(grouped)}")
    
#     # Handle baseline case - baseline only exists for static weighting strategy
#     # We need to replicate it for other strategies as mentioned in requirements
#     baseline_data = grouped[grouped['hardness_metric'] == 'Baseline']
    
#     if not baseline_data.empty:
#         baseline_diff = baseline_data[baseline_data['weighting_strategy'] == 'static']['difference_score'].iloc[0]
#         print(f"  Baseline difference score: {baseline_diff:.6f}")
        
#         # Add baseline for other weighting strategies
#         for strategy in ['curriculum', 'self_paced']:
#             if strategy not in baseline_data['weighting_strategy'].values:
#                 new_row = pd.DataFrame({
#                     'hardness_metric': ['Baseline'],
#                     'weighting_strategy': [strategy],
#                     'difference_score': [baseline_diff]
#                 })
#                 grouped = pd.concat([grouped, new_row], ignore_index=True)
    
#     # Create pivot table for heatmap
#     pivot_table = grouped.pivot(index='hardness_metric', columns='weighting_strategy', values='difference_score')
    
#     # Ensure all weighting strategies are present
#     for strategy in ['static', 'curriculum', 'self_paced']:
#         if strategy not in pivot_table.columns:
#             pivot_table[strategy] = np.nan
    
#     # Reorder columns
#     pivot_table = pivot_table[['static', 'curriculum', 'self_paced']]
    
#     # Sort rows - put Baseline first, then alphabetically
#     if 'Baseline' in pivot_table.index:
#         other_indices = sorted([idx for idx in pivot_table.index if idx != 'Baseline'])
#         pivot_table = pivot_table.reindex(['Baseline'] + other_indices)
    
#     print(f"  Final pivot shape: {pivot_table.shape}")

#     # CREATE TABLE TO CHECK THE COMPLEXITY MEAN +- STD FOR EACH DATASET
    
#     # Group by mean and std together
#     grouped = dataset_df.groupby(['hardness_metric', 'weighting_strategy']).agg(
#         mean_val=('calculated_difference', 'mean'),
#         std_val=('calculated_difference', 'std')  # or std_val=('Wasserstein_H0_mean', 'std') depending on goal
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
#     pivot_table_ = grouped.pivot(index='hardness_metric', columns='weighting_strategy', values='mean_std')

#     # Ensure all strategies exist
#     for strategy in ['static', 'curriculum', 'self_paced']:
#         if strategy not in pivot_table_.columns:
#             pivot_table_[strategy] = np.nan

#     # Reorder and sort
#     pivot_table_ = pivot_table_[['static', 'curriculum', 'self_paced']]
#     if 'Baseline' in pivot_table_.index:
#         other_rows = sorted([r for r in pivot_table_.index if r != 'Baseline'])
#         pivot_table_ = pivot_table_.reindex(['Baseline'] + other_rows)

#     # Save the table to CSV and LaTeX
#     pivot_table_path_csv = output_dir / f'{dataset_name}_Hardness_calculated_difference_table.csv'
#     pivot_table_path_tex = output_dir / f'{dataset_name}_Hardness_calculated_difference_table.tex'
#     pivot_table_.to_csv(pivot_table_path_csv)
#     pivot_table_.to_latex(pivot_table_path_tex, escape=False)

#     print(f"  Saved table to {pivot_table_path_csv}")
    
#     return pivot_table

# # Create heatmaps for each dataset
# datasets = sorted(df['dataset'].unique())
# print(f"Found {len(datasets)} datasets: {datasets}")

# for dataset in datasets:
#     pivot_table = create_complexity_heatmap_for_dataset(dataset)
    
#     if pivot_table is None:
#         continue
    
#     # Create figure
#     fig, ax = plt.subplots(figsize=(16, max(8, len(pivot_table) * 0.4)))
    
#     # Create heatmap
#     sns.heatmap(pivot_table, annot=True, fmt='.3f', cmap='YlGnBu', center=0,
#                ax=ax, cbar_kws={'label': 'Difference Score (Real - Synth)'})
#     ax.set_title(f'{dataset} - Complexity Difference Score\n(mean_real_score - mean_synth_score)')
#     ax.set_xlabel('Weighting Strategy')
#     ax.set_ylabel('Hardness Metric')
    
#     plt.tight_layout()
    
#     # Save the plot
#     output_path = output_dir / f'{dataset}_complexity_heatmap.png'
#     plt.savefig(output_path, dpi=300, bbox_inches='tight')
#     plt.close()
    
#     print(f"  Saved heatmap to {output_path}")

# print(f"\nAll individual heatmaps saved to {output_dir}")

# # Create summary heatmap across all datasets
# print("\nCreating summary heatmap across all datasets...")

# # Group by hardness_metric and weighting_strategy across all datasets
# all_grouped = df.groupby(['hardness_metric', 'weighting_strategy'])['difference_score'].mean().reset_index()

# # Handle baseline case for summary
# baseline_data = all_grouped[all_grouped['hardness_metric'] == 'Baseline']
# if not baseline_data.empty:
#     baseline_diff = baseline_data[baseline_data['weighting_strategy'] == 'static']['difference_score'].iloc[0]
    
#     # Add baseline for other weighting strategies
#     for strategy in ['curriculum', 'self_paced']:
#         if strategy not in baseline_data['weighting_strategy'].values:
#             new_row = pd.DataFrame({
#                 'hardness_metric': ['Baseline'],
#                 'weighting_strategy': [strategy],
#                 'difference_score': [baseline_diff]
#             })
#             all_grouped = pd.concat([all_grouped, new_row], ignore_index=True)

# # Create pivot table
# summary_pivot = all_grouped.pivot(index='hardness_metric', columns='weighting_strategy', values='difference_score')

# # Ensure all weighting strategies are present and reorder
# for strategy in ['static', 'curriculum', 'self_paced']:
#     if strategy not in summary_pivot.columns:
#         summary_pivot[strategy] = np.nan

# summary_pivot = summary_pivot[['static', 'curriculum', 'self_paced']]

# # Sort rows - put Baseline first, then alphabetically
# if 'Baseline' in summary_pivot.index:
#     other_indices = sorted([idx for idx in summary_pivot.index if idx != 'Baseline'])
#     summary_pivot = summary_pivot.reindex(['Baseline'] + other_indices)

# # Create summary plot
# fig, ax = plt.subplots(figsize=(16, 12))
# vmin = summary_pivot.min().min()
# vmax = summary_pivot.max().max()
# sns.heatmap(summary_pivot, annot=True, fmt='.3f', cmap='YlGnBu',vmax=vmax,
#             vmin=vmin,  # Set min and max for consistent color scalin
#             annot_kws={'size': 14}, # Increase font size for annotations
#            ax=ax, cbar_kws={'label': 'Difference Score (Real - Synth)'})
# # ax.set_title('Summary Across All Datasets - Complexity Difference Score\n(mean_real_score - mean_synth_score)')
# ax.set_xlabel('Weighting Strategy', fontsize=14)  # Increase axis label font size
# ax.set_ylabel('Hardness Metric', fontsize=14)     # Increase axis label font size
# # ax.set_ylabel('Hardness Metric')

# # Increase tick label font sizes
# ax.tick_params(axis='x', labelsize=12)
# ax.tick_params(axis='y', labelsize=12)
# plt.tight_layout()






# # Save summary plot
# summary_path = output_dir / 'summary_complexity_heatmap.png'
# plt.savefig(summary_path, dpi=300, bbox_inches='tight')
# plt.close()

# print(f"Summary heatmap saved to {summary_path}")
