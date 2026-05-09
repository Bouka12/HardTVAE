"""
experiments.py
==============
Sensitivity analysis of topological fidelity to the IsUMap embedding dimension.
Compares HardTVAE configurations, TVAE baseline, and CTGAN.

Each experiment run produces one row in fidelity_summary.csv, with columns:
  dataset, seed, model, hardness_metric, strategy,
  topo_d3_bottleneck_H0, topo_d3_bottleneck_H0_similarity, ...,
  topo_d4_*, topo_d5_*, topo_d*_status
"""

import os
import itertools
import json
import random
from typing import List

import numpy as np
import pandas as pd
import torch

import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.hardness import HardnessCalculator, CVAEHardnessIntegrator
from load_data import load_data
from utils import preprocess_data
from models.hardtvae import (TabularCVAE, HardnessCalculator,
                              prepare_dataloader, HardnessAwareCVAETrainer)
from models.ctgan_wrapper import set_seed, ctgan
from evaluation.topological_fidelity import topological_fidelity_calculate


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def append_csv_row(csv_path: str, row: dict) -> None:
    """Append one dict as a row, creating the file with a header if needed."""
    pd.DataFrame([row]).to_csv(
        csv_path, mode='a',
        header=not os.path.exists(csv_path),
        index=False,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────

DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

N_EPOCHS     = 150
MASTER_SEED  = 42
random.seed(MASTER_SEED)
random_seeds: List[int] = random.sample(range(1, 10**6), 10)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    datasets = ['Hypothyroid','NewThyroid1',  'Vertebral']
    hardness_metrics = [None, 'F1', 'F4', 'N2', 'LSC', 'LSR', 'Harmfulness', 'MV', 'CB', 'DCP', 'TD_P', 'CLD']
    seeds            = list(random_seeds)
    print(f"seeds = {seeds}")

    # ── Output paths ──────────────────────────────────────────────────────────
    DIR             = "Results-Topological-Fidelity-Sensitivity"
    RESULTS_FID     = os.path.join(DIR, "fidelity")
    PLOTS_DIR       = os.path.join(RESULTS_FID, "plots")
    FID_SUMMARY_CSV = os.path.join(RESULTS_FID, "fidelity_summary.csv")

    for d in [DIR, RESULTS_FID, PLOTS_DIR]:
        os.makedirs(d, exist_ok=True)

    # ── Experiment loop ───────────────────────────────────────────────────────
    for dataset_name, seed in itertools.product(datasets, seeds):
        set_seed(seed)
        DATA_PATH = f'data/processed/{dataset_name}.csv'
        (X_train, y_train,
         X_val,   y_val,
         X_test,  y_test) = load_data(DATA_PATH, random_state=seed)

        # Preprocessing
        print("Preprocessing data ...")
        preprocessor = preprocess_data(random_state=seed)
        X_train_proc, df_train_proc = preprocessor.fit_transform(X_train)
        X_val_proc,   _             = preprocessor.transform(X_val)
        X_test_proc,  _             = preprocessor.transform(X_test)
        data_info = preprocessor.data_info
        print(f"data_info of {dataset_name}:\n{data_info}")

        label_counts     = pd.Series(y_train).value_counts()
        minority_label   = label_counts.idxmin()
        majority_label   = label_counts.idxmax()
        n_samples_needed = label_counts[majority_label] - label_counts[minority_label]

        # Shared identity columns for every CSV row in this (dataset, seed) run
        base_info = {'dataset': dataset_name, 'seed': seed}

        # ══════════════════════════════════════════════════════════════════════
        # 1.  CTGAN baseline
        # ══════════════════════════════════════════════════════════════════════
        model_name = "CTGAN"
        print(f"\n─── Training {model_name} for {dataset_name} ───")

        X_ctgan, y_ctgan = ctgan(
            df_train=df_train_proc, y_train=y_train,
            epochs=N_EPOCHS, batch_size=32, seed=seed,
        )
        print(f"X_ctgan shape: {X_ctgan.shape}")

        plots_ctgan = os.path.join(PLOTS_DIR, model_name, dataset_name)
        os.makedirs(plots_ctgan, exist_ok=True)

        # topological_fidelity_calculate returns a single flat dict
        topo_ctgan = topological_fidelity_calculate(
            X_real=X_train_proc, y_real=y_train,
            X_synth=X_ctgan,     y_synth=y_ctgan,
            random_state=seed,
            dataset_name=dataset_name,
            save_path=plots_ctgan,
        )

        # Prefix topo keys and write one CSV row
        row_ctgan = {
            **base_info,
            'model':           model_name,
            'hardness_metric': 'None',
            'strategy':        'None',
            **{f'topo_{k}': v for k, v in topo_ctgan.items()},
        }
        append_csv_row(FID_SUMMARY_CSV, row_ctgan)
        print(f"Saved row for {model_name} → {FID_SUMMARY_CSV}")

        # ══════════════════════════════════════════════════════════════════════
        # 2.  TVAE / HardTVAE
        # ══════════════════════════════════════════════════════════════════════
        for hardness_metric in hardness_metrics:
            strategies = (['static'] if hardness_metric is None
                          else ['curriculum', 'static', 'self_paced'])

            for strategy in strategies:
                model_name = ("TVAE" if hardness_metric is None
                              else f"HardTVAE_{hardness_metric}_{strategy}")
                print(f"\n─── Training {model_name} for {dataset_name} ───")

                model = TabularCVAE(
                    input_dim=X_train_proc.shape[1],
                    latent_dim=5,
                    condition_dim=1,
                    hidden_dims=[128, 64, 32],
                    data_info=data_info,
                )
                trainer = HardnessAwareCVAETrainer(
                    model=model,
                    hardness_calculator=HardnessCalculator(random_state=seed),
                    hardness_integrator=CVAEHardnessIntegrator(
                        hardness_strategy=strategy),
                    device=DEVICE,
                )

                trainer.calculate_hardness_scores(
                    X_train_proc, y_train, [hardness_metric])
                dataloader = prepare_dataloader(X_train_proc, y_train)

                if trainer.hardness_scores is None and hardness_metric is not None:
                    print(f"Skipping: invalid hardness scores for {dataset_name}")
                    continue

                for epoch in range(N_EPOCHS):
                    trainer.train_epoch(dataloader, epoch, N_EPOCHS)

                minority_condition = torch.tensor([minority_label])
                synthetic_samples  = trainer.generate_samples(
                    minority_condition, n_samples=n_samples_needed)
                X_synth_proc = synthetic_samples.cpu().numpy()
                y_synth      = np.full(len(X_synth_proc), minority_label)
                print(f"Samples needed: {n_samples_needed}  "
                      f"Generated: {X_synth_proc.shape}")

                plots_h = os.path.join(PLOTS_DIR, model_name, dataset_name)
                os.makedirs(plots_h, exist_ok=True)

                topo_h = topological_fidelity_calculate(
                    X_real=X_train_proc, y_real=y_train,
                    X_synth=X_synth_proc, y_synth=y_synth,
                    random_state=seed,
                    dataset_name=dataset_name,
                    save_path=plots_h,
                )

                row_h = {
                    **base_info,
                    'model':           model_name,
                    'hardness_metric': str(hardness_metric),
                    'strategy':        strategy,
                    **{f'topo_{k}': v for k, v in topo_h.items()},
                }
                append_csv_row(FID_SUMMARY_CSV, row_h)
                print(f"Saved row for {model_name} → {FID_SUMMARY_CSV}")

    print(f"\nAll done. Summary CSV: {FID_SUMMARY_CSV}")


if __name__ == "__main__":
    main()