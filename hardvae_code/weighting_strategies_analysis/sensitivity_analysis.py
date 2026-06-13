"""
    Sensitivity analysis to the weighting strategies' hyperparameters.

To demonstrate the resilience of the results to the design choices in the weighting strategies
    *   We analyse the results in function of a set of values for the hyperparameters of the used weighting strategies
    *   Curriculum learning: schedules (p, p, 1-2*p), change p values {0.2, 0.3, 0.4}
    *   Self-paced learning: 
        -   Pacing function: linear (current), Root pacing (Faster Early Progression), Logarithmic (Aggressive Early Progression) 
        -   annealing weight: [0.05, 0.1, 0.2]
        -   threshold ?!

**Note:** for this sensitivity analysis we fix the random seed, due the large set of combinations. 
We use all the datasets of the experiments in the sensitivity results.
"""

"""
experiments.py to run the experiments on the datasets 
comparing HardTVAE configurations, including the baseline TVAE, and CTGAN
"""

import os
import itertools
import pandas as pd
import numpy as np
import warnings
import random
from typing import Tuple, Optional, List, Dict
import torch
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# OURS
from hardness_ import (HardnessCalculator, 
                             CVAEHardnessIntegrator )
from load_data import load_data         # < THIS GOES IN `utils`
from utils import preprocess_data 
from hardtvae_ import TabularCVAE, HardnessCalculator, prepare_dataloader, HardnessAwareCVAETrainer
# set_seed also goes in utils
from models.ctgan_wrapper import set_seed, ctgan
# EVALUATION
from evaluation.distributional_fidelity import distributional_fidelity_calculate
from evaluation.complexity_fidelity import complexity_fidelity_calculate
from evaluation.hardness_fidelity import hardness_fidelity_calculate
from evaluation.topological_fidelity_exp import topological_fidelity_calculate
from evaluation.utility_evaluation import ClassificationEvaluator

# SAVING RESULTS
import json

def flatten_results(results_dict):
    """Converts nested dicts {view: {metric: val}} to {view_metric: val}."""
    flattened = {}
    for view, metrics in results_dict.items():
        for k, v in metrics.items():
            flattened[f"{view}_{k}"] = v
    return flattened

def save_detailed_results(path, filename, data):
    """Saves detailed results (dicts/lists) as JSON."""
    os.makedirs(path, exist_ok=True)
    full_path = os.path.join(path, f"{filename}.json")
    with open(full_path, 'w') as f:
        json.dump(data, f, indent=4)

#==================================================================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# ══════════════════════════════════════════════════════════════════════════════
#  Hyperparameter grid
# ══════════════════════════════════════════════════════════════════════════════

N_EPOCHS = 150

# Curriculum: epoch split is (p, p, 1-2p) — only p drives the split
curriculum_p_values = [0.2, 0.3, 0.4]
curriculum_splits   = [(p, p, round(1 - 2 * p, 10)) for p in curriculum_p_values]

annealing_weights = [0.05, 0.1, 0.2]

# Pacing functions mapped to their gamma values
PACING_FUNCTIONS = [
    ('linear',      1.0),
    ('root',        0.5),
    ('logarithmic', 2.0),
]

# ── Strategy-specific config lists ────────────────────────────────────────────
# Each strategy only sweeps over hyperparameters that are actually relevant to it.
# This avoids (a) redundant runs and (b) model-name collisions that arise when
# an irrelevant dimension is silently ignored in the name-building logic.

# Curriculum: relevant dims → curriculum_split × annealing_weight  (9 configs)
curriculum_configs = [
    {
        "curriculum_split":      split,
        "min_weight":            w,
        "pacing_gamma":          1.0,   # not used by curriculum; kept for uniform API
        "pacing_function_type":  None,
    }
    for split, w in itertools.product(curriculum_splits, annealing_weights)
]

# Self-paced: relevant dims → pacing_function × annealing_weight  (9 configs)
self_paced_configs = [
    {
        "curriculum_split":      None,  # not used by self-paced
        "min_weight":            w,
        "pacing_gamma":          gamma,
        "pacing_function_type":  pf,
    }
    for w, (pf, gamma) in itertools.product(annealing_weights, PACING_FUNCTIONS)
]

# Static: relevant dim → annealing_weight only  (3 configs)
static_configs = [
    {
        "curriculum_split":      None,
        "min_weight":            w,
        "pacing_gamma":          1.0,
        "pacing_function_type":  None,
    }
    for w in annealing_weights
]

# Map each strategy to its own config list
STRATEGY_CONFIGS: Dict[str, list] = {
    'curriculum': curriculum_configs,   # 9 runs
    'self_paced': self_paced_configs,   # 9 runs
    'static':     static_configs,       #  3 runs
}
# Total: 21 runs per (dataset × seed × hardness_metric)  — down from 81

MASTER_SEED  = 42
random.seed(MASTER_SEED)
random_seeds = random.sample(range(1, 10**6), 3) #[116740]


# ══════════════════════════════════════════════════════════════════════════════
#  Helpers
# ══════════════════════════════════════════════════════════════════════════════

def build_model_name(hardness_metric: str, strategy: str, config: dict) -> str:
    """
    Constructs a unique, human-readable model identifier.
    Each name encodes exactly the hyperparameters that are relevant for the
    given strategy, so no two distinct configs share a name.
    """
    base = f"HardTVAE_{hardness_metric}_{strategy}"
    w    = config['min_weight']

    if strategy == 'curriculum':
        p = config['curriculum_split'][0]
        return f"{base}_p{p}_anneal{w}"

    if strategy == 'self_paced':
        pf = config['pacing_function_type']
        return f"{base}_pacing{pf}_anneal{w}"

    # static
    return f"{base}_anneal{w}"


# ══════════════════════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    datasets         = ['NewThyroid1', 'Hypothyroid', 'Vertebral'] # 'NewThyroid1']  #,  'Vertebral
    hardness_metrics = ['F4', 'TD_P']
    seeds            = list(random_seeds)
    print(f"seeds = {seeds}")

    # ── Saving paths ──────────────────────────────────────────────────────────
    DIR              = "RESULTS_WEIGHTING_STRATEGIES_SENSITIVITY"
    os.makedirs(DIR, exist_ok=True)

    RESULTS_FID      = os.path.join(DIR, "fidelity")
    FID_SUMMARY_CSV  = os.path.join(RESULTS_FID, "fidelity_summary.csv")
    ARTIFACTS_DIR    = os.path.join(RESULTS_FID, "artifacts")
    plots_dir        = os.path.join(RESULTS_FID, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    UTILITY_RESULTS_DIR = os.path.join(DIR, "utility")
    os.makedirs(UTILITY_RESULTS_DIR, exist_ok=True)
    UTILITY_SUMMARY_CSV = os.path.join(UTILITY_RESULTS_DIR, "utility_summary.csv")
    BEST_PARAMS_CSV     = os.path.join(UTILITY_RESULTS_DIR, "best_params.csv")

    # ── Outer loops: dataset × seed ───────────────────────────────────────────
    for dataset_name, seed in itertools.product(datasets, seeds):
        set_seed(seed)
        DATA_PATH = f'data/processed/{dataset_name}.csv'
        (X_train, y_train, X_val, y_val, X_test, y_test) = load_data(
            DATA_PATH, random_state=seed
        )

        # ── Preprocessing ─────────────────────────────────────────────────────
        print("Preprocessing data...")
        preprocessor                      = preprocess_data(random_state=seed)
        X_train_proc, df_train_proc       = preprocessor.fit_transform(X_train)
        X_val_proc,   df_val_proc         = preprocessor.transform(X_val)
        X_test_proc,  df_test_proc        = preprocessor.transform(X_test)
        data_info = preprocessor.data_info
        print(f"data_info of {dataset_name}:\n {data_info}")

        # ── Class balance info ────────────────────────────────────────────────
        label_counts      = pd.Series(y_train).value_counts()
        minority_label    = label_counts.idxmin()
        majority_label    = label_counts.idxmax()
        n_samples_needed  = label_counts[majority_label] - label_counts[minority_label]

        df_train_full        = df_train_proc.copy()
        df_train_full['Outcome'] = y_train
        X_real_minority = X_train_proc[y_train == minority_label]

        # ── Inner loops: hardness metric × strategy × config ─────────────────
        for hardness_metric in hardness_metrics:
            for strategy, configs in STRATEGY_CONFIGS.items():
                for config in configs:

                    model_name = build_model_name(hardness_metric, strategy, config)
                    print(
                        f"\n--- {model_name} | dataset={dataset_name} | seed={seed} ---\n"
                        f"    curriculum_split={config['curriculum_split']}  "
                        f"min_weight={config['min_weight']}  "
                        f"pacing={config['pacing_function_type']}"
                    )

                    # ── Build & train model ───────────────────────────────────
                    model = TabularCVAE(
                        input_dim=X_train_proc.shape[1],
                        latent_dim=5,
                        condition_dim=1,
                        hidden_dims=[128, 64, 32],
                        data_info=data_info,
                    )

                    # Curriculum epochs are expressed as fractions of N_EPOCHS.
                    # For non-curriculum strategies the value is unused, but we
                    # pass a safe default to keep the API uniform.
                    curriculum_epoch_fractions = (
                        config['curriculum_split']
                        if strategy == 'curriculum'
                        else (0.33, 0.33, 0.34)
                    )

                    trainer = HardnessAwareCVAETrainer(
                        model=model,
                        hardness_calculator=HardnessCalculator(random_state=seed),
                        hardness_integrator=CVAEHardnessIntegrator(
                            hardness_strategy=strategy,
                            curriculum_epochs=curriculum_epoch_fractions,
                            min_weight=config['min_weight'],
                            gamma=config['pacing_gamma'],
                        ),
                        device=DEVICE,
                    )

                    trainer.calculate_hardness_scores(X_train_proc, y_train, [hardness_metric])
                    dataloader = prepare_dataloader(X_train_proc, y_train)

                    if trainer.hardness_scores is None and hardness_metric is not None:
                        print(f"Skipping: invalid hardness scores for {dataset_name}")
                        continue

                    for epoch in range(N_EPOCHS):
                        trainer.train_epoch(dataloader, epoch, N_EPOCHS)

                    # ── Generate synthetic minority samples ───────────────────
                    minority_condition = torch.tensor([minority_label])
                    synthetic_samples  = trainer.generate_samples(
                        minority_condition, n_samples=n_samples_needed
                    )
                    X_synth_proc = synthetic_samples.cpu().numpy()
                    y_synth      = np.full(len(X_synth_proc), minority_label)

                    print(f"samples needed: {n_samples_needed}")
                    print(f"type(X_synthetic): {type(X_synth_proc)}")

                    # ── Base metadata row ─────────────────────────────────────
                    base_info = {
                        "dataset":           dataset_name,
                        "seed":              seed,
                        "model":             model_name,
                        "hardness_metric":   str(hardness_metric),
                        "strategy":          strategy,
                        "annealing_weight":  config['min_weight'],
                        "pacing_function":   config['pacing_function_type'],
                        "curriculum_split":  config['curriculum_split'],
                    }

                    # ── Fidelity evaluation ───────────────────────────────────
                    fidelity_views = {}

                    # Distributional fidelity
                    keys, values = distributional_fidelity_calculate(
                        real_data=X_real_minority,
                        synthetic_data=X_synth_proc,
                        data_info=preprocessor.data_info,
                        feature_names=preprocessor.feature_names_out,
                    )
                    fidelity_views['distributional'] = dict(zip(keys, values))

                    # Complexity fidelity
                    keys, values, complexity_detailed = complexity_fidelity_calculate(
                        X_real=X_train_proc, y_real=y_train,
                        X_synth=X_synth_proc, y_synth=y_synth,
                        k=3, random_state=seed, return_detailed=True,
                    )
                    fidelity_views['complexity'] = dict(zip(keys, values))

                    # Hardness fidelity
                    path_plots = os.path.join(plots_dir, model_name, dataset_name)
                    os.makedirs(path_plots, exist_ok=True)
                    keys, values, hardness_detailed = hardness_fidelity_calculate(
                        X_real=X_train_proc, y_real=y_train,
                        X_synth=X_synth_proc, y_synth=y_synth,
                        k=3, random_state=seed,
                        save_path=path_plots,
                        dataset_name=dataset_name,
                        return_detailed=True,
                    )
                    fidelity_views['hardness'] = dict(zip(keys, values))

                    # Topological fidelity
                    topo_detailed = topological_fidelity_calculate(
                        X_real=X_train_proc, y_real=y_train,
                        X_synth=X_synth_proc, y_synth=y_synth,
                        random_state=seed,
                        dataset_name=dataset_name,
                        save_path=path_plots,
                        dimensions_to_test=[3],
                    )
                    fidelity_views['topological'] = topo_detailed

                    # ── Save fidelity summary row ─────────────────────────────
                    summary_row = {**base_info, **flatten_results(fidelity_views)}
                    pd.DataFrame([summary_row]).to_csv(
                        FID_SUMMARY_CSV, mode='a',
                        header=not os.path.exists(FID_SUMMARY_CSV),
                        index=False,
                    )

                    # ── Save detailed artefacts ───────────────────────────────
                    detail_path = os.path.join(
                        ARTIFACTS_DIR, dataset_name, f"seed_{seed}", model_name
                    )
                    save_detailed_results(detail_path, "complexity_details", complexity_detailed)
                    save_detailed_results(detail_path, "hardness_details",   hardness_detailed)

                    # ── Utility evaluation ────────────────────────────────────
                    print(f"Evaluating Utility for {model_name}...")
                    X_train_aug = np.vstack([X_train_proc, X_synth_proc])
                    y_train_aug = np.concatenate([y_train, y_synth])

                    evaluator = ClassificationEvaluator(
                        dataset_name=dataset_name,
                        combination_name=model_name,
                        random_state=seed,
                        results_path=UTILITY_RESULTS_DIR,
                    )
                    utility_res, best_params = evaluator.evaluate(
                        X_train_aug, y_train_aug,
                        X_val_proc, y_val,
                        X_test_proc, y_test,
                    )

                    # Save utility summary row
                    utility_res['model'] = model_name
                    utility_res['seed']  = seed
                    utility_res.to_csv(
                        UTILITY_SUMMARY_CSV, mode='a',
                        header=not os.path.exists(UTILITY_SUMMARY_CSV),
                        index=False,
                    )

                    # Save grid-search best params
                    best_params['model'] = model_name
                    best_params.to_csv(
                        BEST_PARAMS_CSV, mode='a',
                        header=not os.path.exists(BEST_PARAMS_CSV),
                        index=False,
                    )


if __name__ == "__main__":
    main()