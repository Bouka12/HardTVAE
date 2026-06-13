"""
Topological Fidelity Sensitivity Analysis
==========================================
Examines the effect of three TDA parameters on similarity scores:
  - Embedding dimension d  : [d3, d4, d5]
  - Distance metric        : [bottleneck, wasserstein]
  - Homology dimension H   : [H0, H1]

Fixed model configuration:
  - Hardness metric : F4
  - Strategy        : self_paced
"""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from pathlib import Path

# Check current working directory
print("Current working directory:", os.getcwd())

# Get the directory where the specific python directory is saved
script_dir = Path(__file__).parent.resolve()

# Joins the script directory with the target folder and file
INPUT_CSV = script_dir / 'Results-Topological-Fidelity-Sensitivity' / 'fidelity' / 'fidelity_summary.csv'
# ── Paths ──────────────────────────────────────────────────────────────────────
# INPUT_CSV   = r"./Results-Topological-Fidelity-Sensitivity/fidelity/fidelity_summary.csv"

# Check if it actually exists before trying to open it
if not INPUT_CSV.exists():
    print(f"Error: Could not find {INPUT_CSV}")
else:
    print("File found!")

OUTPUT_DIR  = Path(script_dir / "Results-Topological-Fidelity-Sensitivity" / "figures")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Style ──────────────────────────────────────────────────────────────────────
PALETTE_METRIC = {"bottleneck": "#2E86AB", "wasserstein": "#E84855"}
PALETTE_D      = {"d3": "#4C9BE8", "d4": "#F4A261", "d5": "#6BCB77"}
METRIC_LABELS  = {"bottleneck": "Bottleneck", "wasserstein": "Wasserstein"}
D_LABELS       = {"d3": "d = 3", "d4": "d = 4", "d5": "d = 5"}
H_LABELS       = {"H0": "H₀  (connected components)", "H1": "H₁  (loops)"}

sns.set_theme(style="whitegrid", font_scale=1.15)
plt.rcParams.update({
    "font.family"     : "serif",
    "axes.spines.top" : False,
    "axes.spines.right": False,
})

# ══════════════════════════════════════════════════════════════════════════════
# 1. Load & filter
# ══════════════════════════════════════════════════════════════════════════════
topo_raw_results = pd.read_csv(INPUT_CSV)

# Fix model configuration
df_filtered = topo_raw_results[
    (topo_raw_results["hardness_metric"] == "F4") &
    (topo_raw_results["strategy"]        == "self_paced")
].copy()

print(f"Rows after filtering (F4 / self_paced): {len(df_filtered)}")
print(f"  Datasets : {df_filtered['dataset'].unique().tolist()}")
print(f"  Seeds    : {df_filtered['seed'].unique().tolist()}")
print(f"  Model(s) : {df_filtered['model'].unique().tolist()}")

# ══════════════════════════════════════════════════════════════════════════════
# 2. Reshape to long format
#    One row per (dataset, seed, d, metric, H)
# ══════════════════════════════════════════════════════════════════════════════
id_cols = ["dataset", "seed"]
dims    = ["d3", "d4", "d5"]
metrics = ["bottleneck", "wasserstein"]
Hdims   = ["H0", "H1"]

sim_cols = [
    f"topo_{d}_{m}_{H}_similarity"
    for d in dims for m in metrics for H in Hdims
]

long_rows = []
for _, row in df_filtered.iterrows():
    for d in dims:
        for m in metrics:
            for H in Hdims:
                col = f"topo_{d}_{m}_{H}_similarity"
                long_rows.append({
                    "dataset"  : row["dataset"],
                    "seed"     : row["seed"],
                    "d"        : d,
                    "metric"   : m,
                    "H"        : H,
                    "similarity": row[col],
                })

df_long = pd.DataFrame(long_rows)

# Human-readable labels (kept as ordered categoricals for consistent axis order)
df_long["d_label"]      = df_long["d"].map(D_LABELS)
df_long["metric_label"] = df_long["metric"].map(METRIC_LABELS)
df_long["H_label"]      = df_long["H"].map(H_LABELS)

df_long["d_label"]      = pd.Categorical(df_long["d_label"],
                              categories=[D_LABELS[k] for k in dims], ordered=True)
df_long["metric_label"] = pd.Categorical(df_long["metric_label"],
                              categories=[METRIC_LABELS[k] for k in metrics], ordered=True)

print(f"\nLong-format rows: {len(df_long)}")
print(df_long.groupby(["d", "metric", "H"])["similarity"].describe().round(4))

# ══════════════════════════════════════════════════════════════════════════════
# 3. Plot 1 — Effect of Embedding Dimension  (faceted H0 | H1)
# ══════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), sharey=True)

for ax, H in zip(axes, Hdims):
    sub = df_long[df_long["H"] == H]
    sns.boxplot(
        data=sub, x="d_label", y="similarity", hue="d_label",
        palette=[PALETTE_D[k] for k in dims],
        width=0.5, linewidth=1.2, fliersize=4,
        legend=False, ax=ax
    )
    # Overlay individual points (jittered by dataset)
    sns.stripplot(
        data=sub, x="d_label", y="similarity",
        color="black", alpha=0.45, size=4, jitter=True, ax=ax
    )
    ax.set_title(H_LABELS[H], fontsize=12, pad=8)
    ax.set_xlabel("Embedding dimension", fontsize=11)
    ax.set_ylabel("Topological similarity" if H == "H0" else "", fontsize=11)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
    ax.set_ylim(0, 1.05)

fig.suptitle("Effect of Embedding Dimension on Topological Similarity\n"
             "(F4 hardness · self-paced strategy)", fontsize=13, y=1.02)
fig.tight_layout()
out1 = OUTPUT_DIR / "plot1_effect_embedding_dim.pdf"
fig.savefig(out1, bbox_inches="tight", dpi=150)
fig.savefig(str(out1).replace(".pdf", ".png"), bbox_inches="tight", dpi=150)
print(f"\nSaved → {out1}")
plt.close(fig)

# ══════════════════════════════════════════════════════════════════════════════
# 4. Plot 2 — Effect of Distance Metric  (faceted H0 | H1)
# ══════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(9, 4.5), sharey=True)

for ax, H in zip(axes, Hdims):
    sub = df_long[df_long["H"] == H]
    sns.boxplot(
        data=sub, x="metric_label", y="similarity", hue="metric_label",
        palette=[PALETTE_METRIC[k] for k in metrics],
        width=0.45, linewidth=1.2, fliersize=4,
        legend=False, ax=ax
    )
    sns.stripplot(
        data=sub, x="metric_label", y="similarity",
        color="black", alpha=0.45, size=4, jitter=True, ax=ax
    )
    ax.set_title(H_LABELS[H], fontsize=12, pad=8)
    ax.set_xlabel("Distance metric", fontsize=11)
    ax.set_ylabel("Topological similarity" if H == "H0" else "", fontsize=11)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
    ax.set_ylim(0, 1.05)

fig.suptitle("Effect of Distance Metric on Topological Similarity\n"
             "(F4 hardness · self-paced strategy)", fontsize=13, y=1.02)
fig.tight_layout()
out2 = OUTPUT_DIR / "plot2_effect_distance_metric.pdf"
fig.savefig(out2, bbox_inches="tight", dpi=150)
fig.savefig(str(out2).replace(".pdf", ".png"), bbox_inches="tight", dpi=150)
print(f"Saved → {out2}")
plt.close(fig)

# ══════════════════════════════════════════════════════════════════════════════
# 5. Plot 3 — Joint view: Embedding × Metric interaction  (faceted H0 | H1)
#    Lines = model (colour) × metric (linestyle solid/dashed)
# ══════════════════════════════════════════════════════════════════════════════

TARGET_MODELS = {
    "CTGAN"                  : "CTGAN",
    "TVAE"                   : "TVAE",
    "HardTVAE_F4_static"     : "HardTVAE (F4, static)",
    "HardTVAE_F4_curriculum" : "HardTVAE (F4, curriculum)",
    "HardTVAE_F4_self_paced" : "HardTVAE (F4, self-paced)",
}
PALETTE_MODEL = {
    "CTGAN"                  : "#4C72B0",
    "TVAE"                   : "#DD8452",
    "HardTVAE_F4_static"     : "#55A868",
    "HardTVAE_F4_curriculum" : "#9467BD",
    "HardTVAE_F4_self_paced" : "#C44E52",
}
LINESTYLE_METRIC = {"bottleneck": "-",  "wasserstein": "--"}
MARKER_METRIC    = {"bottleneck": "o",  "wasserstein": "s"}

# Rebuild long format for the four target models
df_multi = topo_raw_results[topo_raw_results["model"].isin(TARGET_MODELS)].copy()

long_rows_multi = []
for _, row in df_multi.iterrows():
    for d in dims:
        for m in metrics:
            for H in Hdims:
                col = f"topo_{d}_{m}_{H}_similarity"
                long_rows_multi.append({
                    "model"     : row["model"],
                    "dataset"   : row["dataset"],
                    "seed"      : row["seed"],
                    "d"         : d,
                    "metric"    : m,
                    "H"         : H,
                    "similarity": row[col],
                })

df_long_multi = pd.DataFrame(long_rows_multi)
df_long_multi["d_label"] = pd.Categorical(
    df_long_multi["d"].map(D_LABELS),
    categories=[D_LABELS[k] for k in dims], ordered=True
)

# Aggregate mean ± std per (model, d, metric, H)
agg = (
    df_long_multi
    .groupby(["model", "d_label", "metric", "H"])["similarity"]
    .agg(mean="mean", std="std")
    .reset_index()
)

fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)

for ax, H in zip(axes, Hdims):
    sub = agg[agg["H"] == H]
    for mdl_key, mdl_label in TARGET_MODELS.items():
        for m_key in metrics:
            s = sub[(sub["model"] == mdl_key) & (sub["metric"] == m_key)]
            ax.plot(
                s["d_label"], s["mean"],
                marker=MARKER_METRIC[m_key],
                linestyle=LINESTYLE_METRIC[m_key],
                linewidth=1.8, markersize=6,
                color=PALETTE_MODEL[mdl_key],
                label=f"{mdl_label} · {METRIC_LABELS[m_key]}"
            )
            ax.fill_between(
                s["d_label"],
                s["mean"] - s["std"],
                s["mean"] + s["std"],
                alpha=0.10, color=PALETTE_MODEL[mdl_key]
            )
    ax.set_title(H_LABELS[H], fontsize=12, pad=8)
    ax.set_xlabel("Embedding dimension", fontsize=11)
    ax.set_ylabel("Mean topological similarity" if H == "H0" else "", fontsize=11)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
    ax.set_ylim(0, 1.05)

# Shared legend outside at the bottom — 4 columns (one per model), linestyle encodes metric
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(
    handles, labels,
    fontsize=10, title_fontsize=10,
    loc="lower center", ncol=5,
    bbox_to_anchor=(0.5, -0.14),
    frameon=True, framealpha=0.9, edgecolor="lightgrey"
)

# fig.suptitle("Embedding Dimension × Distance Metric Interaction\n"
#              "(mean ± std  ·  solid = Bottleneck  ·  dashed = Wasserstein)",
#              fontsize=13, y=1.02)
fig.tight_layout()
out3 = OUTPUT_DIR / "plot3_interaction_embedding_metric_complete.pdf"
fig.savefig(out3, bbox_inches="tight", dpi=300)
fig.savefig(str(out3).replace(".pdf", ".png"), bbox_inches="tight", dpi=300)
print(f"Saved → {out3}")
plt.close(fig)



# ══════════════════════════════════════════════════════════════════════════════
# 5. Plot 3 — Joint view: Embedding × Metric interaction  (faceted H0 | H1)
#    x = embedding dimension, y = mean similarity, lines = metric
# ══════════════════════════════════════════════════════════════════════════════
# Compute mean ± std per (d, metric, H) aggregated across datasets & seeds
agg = (
    df_long
    .groupby(["d_label", "metric_label", "H"])["similarity"]
    .agg(mean="mean", std="std")
    .reset_index()
)
 
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
 
for ax, H in zip(axes, Hdims):
    sub = agg[agg["H"] == H]
    for m_key, m_label in METRIC_LABELS.items():
        s = sub[sub["metric_label"] == m_label]
        ax.plot(
            s["d_label"], s["mean"],
            marker="o", linewidth=2, markersize=7,
            color=PALETTE_METRIC[m_key], label=m_label
        )
        ax.fill_between(
            s["d_label"],
            s["mean"] - s["std"],
            s["mean"] + s["std"],
            alpha=0.15, color=PALETTE_METRIC[m_key]
        )
    ax.set_title(H_LABELS[H], fontsize=12, pad=8)
    ax.set_xlabel("Embedding dimension", fontsize=11)
    ax.set_ylabel("Mean topological similarity" if H == "H0" else "", fontsize=11) # FONTSIZE = 11
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
    ax.set_ylim(0, 1.05)
 
# Single shared legend below both subplots
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(
    handles, labels,
    title="Metric", fontsize=10, title_fontsize=10, # FONTSIZE=10
    loc="lower center", ncol=2,
    bbox_to_anchor=(0.5, -0.08),
    frameon=True, framealpha=0.9, edgecolor="lightgrey"
)
 
# fig.suptitle("Embedding Dimension × Distance Metric Interaction\n"
#              "(mean ± std  ·  F4 hardness · self-paced strategy)", fontsize=13, y=1.02)
fig.tight_layout()
out3 = OUTPUT_DIR / "plot3_interaction_embedding_metric.pdf"
fig.savefig(out3, bbox_inches="tight", dpi=300)
fig.savefig(str(out3).replace(".pdf", ".png"), bbox_inches="tight", dpi=300)
print(f"Saved → {out3}")
plt.close(fig)
 
print("\n✓ All plots generated.")