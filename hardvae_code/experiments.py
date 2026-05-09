
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

# OURS
from models.hardness import (HardnessCalculator, 
                             CVAEHardnessIntegrator )
from load_data import load_data         # < THIS GOES IN `utils`
# from hardvae_code.gradients_stability_analysis.loss_viz import loss_plots         # < Goes in `utils` maybe
# from classifier_eval import evaluate_classification_model       # < Change names
from utils import preprocess_data 
from models.hardtvae import TabularCVAE, HardnessCalculator, prepare_dataloader, HardnessAwareCVAETrainer
# set_seed also goes in utils
from models.ctgan_wrapper import set_seed, ctgan
# EVALUATION
from evaluation.distributional_fidelity import distributional_fidelity_calculate
from evaluation.complexity_fidelity import complexity_fidelity_calculate
from evaluation.hardness_fidelity import hardness_fidelity_calculate
from evaluation.topological_fidelity import topological_fidelity_calculate
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
#  Main  — 
# ══════════════════════════════════════════════════════════════════════════════


N_EPOCHS          = 150
CURRICULUM_EPOCHS = (N_EPOCHS * 0.3, N_EPOCHS * 0.3, N_EPOCHS * 0.4)
# the curriculum hyperparameters (the epoch thresholds for switching hardness integration strategies) are set as a percentage of the total training epochs (N_EPOCHS).
# in this case, if p = 0.3 so n*p, n*p, n*(1-2*p), meaning that for the first 30% of the epochs, we use one strategy, then switch to another for the next 30%, and finally use a third strategy for the remaining 40% of the training.
# in our sensitivity analysis, we change the p value as it is the only curriculum hyperparameter, and we keep the same percentage distribution for the epochs (e.g., if p=0.2, then 20% of epochs for strategy 1, 20% for strategy 2, and 60% for strategy 3).

# The self paced weighting strategy has hyperparameters:
# the threshold
MASTER_SEED       = 42
random.seed(MASTER_SEED)
random_seeds      = random.sample(range(1, 10**6), 10)


def main():
    # datasets = ['BCWDD']
    datasets = [
        'BCWDD', 'HeartCleveland', 'Hepatitis', 'Hypothyroid',
        'ILPD', 'NewThyroid1', 'NewThyroid2', 'Pima', 'Thoracic', 'Vertebral'
    ]
    # hardness_metrics = [None, 'F1']
    hardness_metrics = [None, 'kDN', 'DS', 'DCP', 'TD_P',
                   'TD_U', 'CL', 'CLD', 'MV', 'CB', 'N1', 'N2', 'LSC', 
                   'LSR', 'Harmfulness', 'F1', 'F2', 'F3', 'F4']
    seeds            = list(random_seeds)
    print(f"seeds = {seeds}")

    # ---------------------------------------------------------------------------
    # SAVING PATHS
    # ---------------------------------------------------------------------------
    DIR = "RESULTS"
    os.makedirs(DIR, exist_ok=True)
    #
    RESULTS_FID = os.path.join(DIR, "fidelity")
    FID_SUMMARY_CSV = os.path.join(RESULTS_FID, "fidelity_summary.csv")
    ARTIFACTS_DIR = os.path.join(RESULTS_FID, "artifacts")
    #
    plots_dir = f"{RESULTS_FID}/plots"
    os.makedirs(plots_dir, exist_ok=True)
    #
    UTILITY_RESULTS_DIR = os.path.join(DIR, "utility")
    os.makedirs(UTILITY_RESULTS_DIR, exist_ok=True)
    UTILITY_SUMMARY_CSV = os.path.join(UTILITY_RESULTS_DIR, "utility_summary.csv")
    BEST_PARAMS_CSV = os.path.join(UTILITY_RESULTS_DIR, "best_params.csv")



    for dataset_name, seed in itertools.product(datasets, seeds):
        set_seed(seed)
        DATA_PATH = f'data/processed/{dataset_name}.csv'
        (X_train, y_train, X_val, y_val, X_test, y_test) = load_data(
            DATA_PATH, random_state=seed
        )

        # ── Preprocessing  ──────────────────────────────────
        print("Preprocessing data...")
        preprocessor = preprocess_data(random_state=seed)
        X_train_proc, df_train_proc = preprocessor.fit_transform(X_train)
        X_val_proc, df_val_proc   = preprocessor.transform(X_val)
        X_test_proc, df_test_proc  = preprocessor.transform(X_test)
        data_info = preprocessor.data_info          # ← {'n_numerical', 'cat_dims'}
        input_dim = X_train_proc.shape[1]           # numerical + sum(cat_dims)
        print(f"data_info of {dataset_name}:\n {data_info}")
        # print(f"data_info of the df_train_proc:\n {df_train_proc.info()}")
        # Identify labels
        label_counts = pd.Series(y_train).value_counts()
        minority_label = label_counts.idxmin()
        majority_label = label_counts.idxmax()
        n_samples_needed = label_counts[majority_label] - label_counts[minority_label]


        # --------------------------------------------------------------
        # 1. Baseline: CTGAN (via SDV)
        # --------------------------------------------------------------
        model_name = "CTGAN"
        print(f"\n--- Training CTGAN Baseline for {dataset_name} ---")
        df_train_full = df_train_proc.copy()
        df_train_full['Outcome'] = y_train
        X_real_minority = X_train_proc[y_train==minority_label]
        df_minority = df_train_full[df_train_full['Outcome'] == minority_label]

        X_ctgan, y_ctgan = ctgan(df_train=df_train_proc, y_train=y_train, epochs=N_EPOCHS, batch_size=32, seed=seed)

        #### TESTING ####
        # print(f"X_ctgan shape : {X_ctgan.shape}\n")
        # print(f"type(X_ctgan): {type(X_ctgan)}")


        # # Evaluation for CTGAN < X_train + X_ctgan, y_train + y_ctgan
        # # fidelity < to do it
        fidelity_views_ctgan = {}
        # DISTRIBUTIONAL FIDELITY:
        keys, values = distributional_fidelity_calculate(real_data=X_real_minority,synthetic_data=X_ctgan.values, 
                                                         data_info=preprocessor.data_info, feature_names=preprocessor.feature_names_out )
        # print(f"keys dist {keys}\n")
        # print(f"values dist {values}\n")
        fidelity_views_ctgan['distributional'] = dict(zip(keys, values))

        # COMPLEXITY FIDELITY
        keys, values, complexity_detailed_results = complexity_fidelity_calculate(X_real=X_train_proc,y_real = y_train,X_synth=X_ctgan,y_synth=y_ctgan,
                                                     k=3, random_state=seed, return_detailed=True)
        # print(f"keys complexity {keys}\n")
        # print(f"values complexity {values}\n")
        fidelity_views_ctgan['complexity'] = dict(zip(keys, values))

        # HARDNESS FIDELITY
        path_plots_ctgan = f"{plots_dir}/{model_name}/{dataset_name}/"
        os.makedirs(path_plots_ctgan, exist_ok=True)
        keys, values, hardness_detailed_results = hardness_fidelity_calculate(X_real=X_train_proc, y_real=y_train,X_synth=X_ctgan, y_synth=y_ctgan,
                                                   k=3, random_state=seed, save_path=path_plots_ctgan, dataset_name=dataset_name, return_detailed=True)
        fidelity_views_ctgan['hardness'] = dict(zip(keys, values)) 

        # TOPOLOGICAL FIDELITY
        topo_results =  topological_fidelity_calculate(X_real=X_train_proc, y_real=y_train, X_synth=X_ctgan, y_synth=y_ctgan,
                                                       random_state=seed, dataset_name=dataset_name, save_path=path_plots_ctgan,dimensions_to_test=[3])
                
        # print(f"keys topo {keys}\n")
        # print(f"values topo {values}\n")

        fidelity_views_ctgan['topological'] = topo_results
        
        # SAVING FIDELITY RESULTS --- CTGAN -----
        base_info = {
            'dataset': dataset_name,
            "seed": seed,
            "model": model_name,
            "hardness_metric": "None",
            "strategy": "None"
        }
        
        # Save Summary (CSV)
        summary_row = {**base_info, **flatten_results(fidelity_views_ctgan)}
        pd.DataFrame([summary_row]).to_csv(FID_SUMMARY_CSV, mode='a',
                                           header= not os.path.exists(FID_SUMMARY_CSV), index=False)
        
        # Save Details (JSON) : {fidelity_view}_detailed_results
        detail_path = f"{ARTIFACTS_DIR}/{dataset_name}/seed_{seed}/{model_name}"
        save_detailed_results(detail_path, "complexity_details", complexity_detailed_results)
        save_detailed_results(detail_path, "hardness_details", hardness_detailed_results)
        # save_detailed_results(detail_path, "topo_details", topo_detailed_results)


        ##### TESTING ######
        # print(f"Distributional Fidelity --{model_name}: {fidelity_views_ctgan['distributional']}")
        # print( f"complexity fidedlity --{model_name}: {fidelity_views_ctgan['complexity']}")
        # print(f"hardness fidelity --{model_name}: {fidelity_views_ctgan['hardness']}")
        print(f"Topological fidelity --{model_name} : {fidelity_views_ctgan['topological']}")

        # # utility < to fix saving or return logic
        print(f"Evaluating Utility for {model_name} ...")
        # Combine Real Train + CTGAN Synthetic Minority
        X_train_augmented = np.vstack([X_train_proc, X_ctgan.values])
        y_train_augmented = np.concatenate([y_train, y_ctgan])

        evaluator_ctgan = ClassificationEvaluator(
            dataset_name=dataset_name,
            combination_name=model_name,
            random_state=seed,
            results_path=UTILITY_RESULTS_DIR
        )
        
        utility_res_ctgan, best_params_ctgan = evaluator_ctgan.evaluate(
            X_train=X_train_augmented, y_train=y_train_augmented,
            X_val=X_val_proc, y_val=y_val,
            X_test=X_test_proc, y_test=y_test
        )

        # Save Utility Summary
        utility_res_ctgan['model'] = model_name
        utility_res_ctgan['seed'] = seed
        utility_res_ctgan.to_csv(UTILITY_SUMMARY_CSV, mode='a',
                                 header= not os.path.exists(UTILITY_SUMMARY_CSV), index=False)
        # Save hyperparameters of gridsearch
        best_params_ctgan['model'] = model_name
        best_params_ctgan.to_csv(BEST_PARAMS_CSV, mode='a',
                                     header = not os.path.exists(BEST_PARAMS_CSV), index=False)
        # synth_ctgan = np.concatenate([X_ctgan, y_ctgan], axis=1)
        # df_ctgan = pd.DataFrame(synth_ctgan, columns=)
        # # -------------------------------------------------------------
        # # 2. HardTVAE (incuding TVAE baseline)
        # # -------------------------------------------------------------
        for hardness_metric in hardness_metrics:
            strategies = ['static'] if hardness_metric is None else ['curriculum', 'static', 'self_paced']
            for strategy in strategies:
                model_name = "TVAE" if hardness_metric is None else f"HardTVAE_{hardness_metric}_{strategy}"
                print(f"\n--- Training {model_name} for {dataset_name} ---")

                model = TabularCVAE(
                    input_dim=X_train_proc.shape[1], 
                    latent_dim=5, condition_dim=1,
                    hidden_dims=[128, 64, 32],
                    data_info=data_info)
                trainer = HardnessAwareCVAETrainer(
                    model= model,
                    hardness_calculator=HardnessCalculator(random_state=seed),
                    hardness_integrator=CVAEHardnessIntegrator(hardness_strategy=strategy),
                    device=DEVICE
                )

                trainer.calculate_hardness_scores(X_train_proc, y_train, [hardness_metric])
                dataloader = prepare_dataloader(X_train_proc, y_train)
                if trainer.hardness_scores is None and hardness_metric is not None:
                    print(f"Skipping: invalid hardness scores for {dataset_name}")
                    continue

                for epoch in range(N_EPOCHS):
                    trainer.train_epoch(dataloader, epoch, N_EPOCHS)
                
                minority_condition = torch.tensor([minority_label])
                synthetic_samples = trainer.generate_samples(minority_condition, n_samples=n_samples_needed)
                X_synth_proc = synthetic_samples.cpu().numpy()
                y_synth = np.full(len(X_synth_proc), minority_label)

                ### TESTING ###
                print(f"samples needed {n_samples_needed}\n")
                print(f"type(X_synthetic) {type(X_synth_proc)}\n")
                
                base_info_h = {
                    "dataset": dataset_name,
                    "seed": seed,
                    "model": model_name,
                    "hardness_metric": str(hardness_metric),
                    "strategy": strategy
                }

                # Evaluation for HardTVAE/TVAE
                # evaluate fidelity
                fidelity_views_h = {}

                # DISTRIBUTIONAL FIDELITY:
                keys, values = distributional_fidelity_calculate(real_data=X_real_minority,synthetic_data=X_synth_proc, 
                                                                data_info=preprocessor.data_info, feature_names=preprocessor.feature_names_out )
                fidelity_views_h['distributional'] = dict(zip(keys, values))
                
                # COMPLEXITY FIDELITY
                keys, values, complexity_detailed_results_h = complexity_fidelity_calculate(X_real=X_train_proc,y_real = y_train,X_synth=X_synth_proc,y_synth=y_synth,
                                                     k=3, random_state=seed, return_detailed=True)
        
                fidelity_views_h['complexity'] = dict(zip(keys, values))
                
                # HARDNESS FIDELITY
                path_plots_h = f"{plots_dir}/{model_name}/{dataset_name}/"
                os.makedirs(path_plots_h, exist_ok=True)
                keys, values, hardness_detailed_results_h = hardness_fidelity_calculate(X_real=X_train_proc, y_real=y_train,X_synth=X_synth_proc, y_synth=y_synth,
                                                        k=3, random_state=seed, save_path=path_plots_h, dataset_name=dataset_name, return_detailed=True)
                fidelity_views_h['hardness'] = dict(zip(keys, values)) 

                # TOPOLOGICAL FIDELITY
                topo_detailed_results_h=  topological_fidelity_calculate(X_real=X_train_proc, y_real=y_train, X_synth=X_synth_proc, y_synth=y_synth,
                                                               random_state=seed,dataset_name=dataset_name, save_path=path_plots_h,dimensions_to_test=[3])
                
                fidelity_views_h['topological'] = topo_detailed_results_h

                # Save summary --- TVAE/HardTVAE ----
                summary_row_h = {**base_info_h, **flatten_results(fidelity_views_h)}
                pd.DataFrame([summary_row_h]).to_csv(FID_SUMMARY_CSV, mode='a',
                                                     header= not os.path.exists(FID_SUMMARY_CSV), index=False)
                
                # Save Details
                h_detail_path = f"{ARTIFACTS_DIR}/{dataset_name}/seed_{seed}/{model_name}"
                # Add your detailed return variables here
                save_detailed_results(h_detail_path, "complexity_details", complexity_detailed_results_h)
                save_detailed_results(h_detail_path, "hardness_details", hardness_detailed_results_h)
                # save_detailed_results(h_detail_path, "topo_details", topo_detailed_results_h)


                ##### TESTING ######
                # print(f"Distributional Fidelity {model_name}: {fidelity_views['distributional']}")
                # print(f"Complexity Fidelity {model_name}: {fidelity_views['complexity']}")
                # print(f"Hardness Fidelity {model_name}: {fidelity_views['hardness']}")
                # print(f"Topological fidelity {model_name}: {fidelity_views_h['topological']}")
                
                # UTILITY EVALUATION 
                # --- UTILITY EVALUATION (HardTVAE) ---
                print(f"Evaluating Utility for {model_name}...")
                # Combine Real Train + HardTVAE Synthetic Minority
                X_train_h_aug = np.vstack([X_train_proc, X_synth_proc])
                y_train_h_aug = np.concatenate([y_train, y_synth])

                evaluator_h = ClassificationEvaluator(
                    dataset_name=dataset_name, 
                    combination_name=model_name, 
                    random_state=seed, 
                    results_path=UTILITY_RESULTS_DIR
                )

                utility_res_h, best_params_h = evaluator_h.evaluate(
                    X_train_h_aug, y_train_h_aug, 
                    X_val_proc, y_val, 
                    X_test_proc, y_test
                )

                # Save Utility Summary
                utility_res_h['model'] = model_name
                utility_res_h['seed'] = seed
                utility_res_h.to_csv(UTILITY_SUMMARY_CSV, mode='a', 
                                     header=not os.path.exists(UTILITY_SUMMARY_CSV), index=False)
                # Save hyperparameters of gridsearch
                best_params_h['model'] = model_name
                best_params_h.to_csv(BEST_PARAMS_CSV, mode='a',
                                     header = not os.path.exists(BEST_PARAMS_CSV), index=False)


if __name__ == "__main__":
    main()