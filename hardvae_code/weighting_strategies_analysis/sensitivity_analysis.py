"""
    Sensitivity analysis to the weighting strategies' hyperparameters.

To demonstrate the resilience of the results to the design choices in the weighting strategies
    *   We analyse the results in function of a set of values for the hyperparameters of the used weighting strategies
    *   Curriculum learning: schedules (p, p, 1-2*p), change p values {0.1, 0.2, 0.3, 0.4}
    *   Self-paced learning: 
        -   Pacing function: linear (current), Root pacing (Faster Early Progression), Logarithmic (Aggressinve Early Progression) 
        -   annealing weight: [0.05, 0.1, 0.2, 0.3]
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
# from hardvae_code.gradients_stability_analysis.loss_viz import loss_plots         # < Goes in `utils` maybe
# from classifier_eval import evaluate_classification_model       # < Change names
from utils import preprocess_data 
from hardtvae_ import TabularCVAE, HardnessCalculator, prepare_dataloader, HardnessAwareCVAETrainer
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
# the threshold for including harder samples, which is determined by the pacing function and the annealing weight that controls how much the hardness scores influence the weights.
# Définition des plages d'hyperparamètres pour l'analyse de sensibilité
curriculum_p_values = [0.1, 0.2, 0.3, 0.4]
annealing_weights = [0.05, 0.1, 0.2, 0.3]
pacing_functions = ['linear', 'root', 'logarithmic'] # Correspond à pacing_gamma dans le code existant

# Générer les combinaisons pour curriculum_split
curriculum_splits = []
for p in curriculum_p_values:
    curriculum_splits.append((p, p, 1 - 2 * p))

# Générer toutes les combinaisons d'hyperparamètres
sensitivity_configs = []
for p_split, annealing_w, pacing_func in itertools.product(curriculum_splits, annealing_weights, pacing_functions):
    # Ici, nous devons mapper pacing_func à pacing_gamma. 
    # Pour simplifier, nous allons utiliser une valeur fixe pour pacing_gamma pour chaque type de fonction
    # ou introduire une logique plus complexe si les fonctions de pacing sont implémentées différemment.
    # Pour l'instant, nous allons utiliser une valeur par défaut et noter que cela pourrait nécessiter une adaptation.
    pacing_gamma_val = 1.0 # Valeur par défaut, à ajuster en fonction de l'implémentation réelle des fonctions de pacing
    if pacing_func == 'root':
        pacing_gamma_val = 0.5 # Exemple de valeur pour 'root'
    elif pacing_func == 'logarithmic':
        pacing_gamma_val = 2.0 # Exemple de valeur pour 'logarithmic'

    sensitivity_configs.append({
        "curriculum_split": p_split,
        "min_weight": annealing_w, # min_weight est utilisé comme annealing_weight ici
        "pacing_gamma": pacing_gamma_val, # Représente le type de fonction de pacing
        "pacing_function_type": pacing_func # Ajout pour garder une trace du type de fonction de pacing
    })

MASTER_SEED       = 42
random.seed(MASTER_SEED)
random_seeds      = random.sample(range(1, 10**6), 10)
"""
    Sensitivity analysis to the weighting strategies' hyperparameters.

To demonstrate the resilience of the results to the design choices in the weighting strategies
    *   We analyse the results in function of a set of values for the hyperparameters of the used weighting strategies
    *   Curriculum learning: schedules (p, p, 1-2*p), change p values {0.1, 0.2, 0.3, 0.4}
    *   Self-paced learning: 
        -   Pacing function: linear (current), Root pacing (Faster Early Progression), Logarithmic (Aggressinve Early Progression) 
        -   annealing weight: [0.05, 0.1, 0.2, 0.3]
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
# from hardvae_code.gradients_stability_analysis.loss_viz import loss_plots         # < Goes in `utils` maybe
# from classifier_eval import evaluate_classification_model       # < Change names
from utils import preprocess_data 
from hardtvae_ import TabularCVAE, HardnessCalculator, prepare_dataloader, HardnessAwareCVAETrainer
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
# the threshold for including harder samples, which is determined by the pacing function and the annealing weight that controls how much the hardness scores influence the weights.
# Définition des plages d'hyperparamètres pour l'analyse de sensibilité
curriculum_p_values = [0.1, 0.2, 0.3, 0.4]
annealing_weights = [0.05, 0.1, 0.2, 0.3]
pacing_functions = ['linear', 'root', 'logarithmic'] # Correspond à pacing_gamma dans le code existant

# Générer les combinaisons pour curriculum_split
curriculum_splits = []
for p in curriculum_p_values:
    curriculum_splits.append((p, p, 1 - 2 * p))

# Générer toutes les combinaisons d'hyperparamètres
sensitivity_configs = []
for p_split, annealing_w, pacing_func in itertools.product(curriculum_splits, annealing_weights, pacing_functions):
    # Ici, nous devons mapper pacing_func à pacing_gamma. 
    # Pour simplifier, nous allons utiliser une valeur fixe pour pacing_gamma pour chaque type de fonction
    # ou introduire une logique plus complexe si les fonctions de pacing sont implémentées différemment.
    # Pour l'instant, nous allons utiliser une valeur par défaut et noter que cela pourrait nécessiter une adaptation.
    pacing_gamma_val = 1.0 # Valeur par défaut, à ajuster en fonction de l'implémentation réelle des fonctions de pacing
    if pacing_func == 'root':
        pacing_gamma_val = 0.5 # Exemple de valeur pour 'root'
    elif pacing_func == 'logarithmic':
        pacing_gamma_val = 2.0 # Exemple de valeur pour 'logarithmic'

    sensitivity_configs.append({
        "curriculum_split": p_split,
        "min_weight": annealing_w, # min_weight est utilisé comme annealing_weight ici
        "pacing_gamma": pacing_gamma_val, # Représente le type de fonction de pacing
        "pacing_function_type": pacing_func # Ajout pour garder une trace du type de fonction de pacing
    })

MASTER_SEED       = 42
random.seed(MASTER_SEED)
random_seeds      = random.sample(range(1, 10**6), 10)


def main():
    # datasets = ['BCWDD']
    datasets = ['Hypothyroid','NewThyroid1',  'Vertebral']
    
    hardness_metrics = ['F1', 'F4', 'N2', 'LSC', 'LSR', 'Harmfulness', 'MV', 'CB', 'DCP', 'TD_P', 'CLD']
    # hardness_metrics = [None, 'kDN', 'DS', 'DCP', 'TD_P',
    #                'TD_U', 'CL', 'CLD', 'MV', 'CB', 'N1', 'N2', 'LSC', 
    #                'LSR', 'Harmfulness', 'F1', 'F2', 'F3', 'F4']
    seeds            = list(random_seeds)
    print(f"seeds = {seeds}")

    # ---------------------------------------------------------------------------
    # SAVING PATHS
    # ---------------------------------------------------------------------------
    DIR = "RESULTS_WEIGHTING_STRATEGIES_SENSITIVITY"
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



    # < CHNAGED
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



        df_train_full = df_train_proc.copy()
        df_train_full['Outcome'] = y_train
        X_real_minority = X_train_proc[y_train==minority_label]
        # df_minority = df_train_full[df_train_full['Outcome'] == minority_label]

        # # -------------------------------------------------------------
        # # 2. HardTVAE (incuding TVAE baseline)
        # # -------------------------------------------------------------
        for hardness_metric in hardness_metrics:
            strategies =  ['curriculum', 'static', 'self_paced']
            for strategy in strategies:
                # model_name = "TVAE" if hardness_metric is None else f"HardTVAE_{hardness_metric}_{strategy}"
                # print(f"\n--- Training {model_name} for {dataset_name} ---")
                for config in sensitivity_configs:
                    print(f"Testing config: Curriculum Split {config['curriculum_split']}, Annealing Weight {config['min_weight']}, Pacing Function {config['pacing_function_type']}")
                        
                    model = TabularCVAE(
                        input_dim=X_train_proc.shape[1], 
                        latent_dim=5, condition_dim=1,
                        hidden_dims=[128, 64, 32],
                        data_info=data_info)
                    trainer = HardnessAwareCVAETrainer(
                        model= model,
                        hardness_calculator=HardnessCalculator(random_state=seed),
                        hardness_integrator=CVAEHardnessIntegrator(hardness_strategy=strategy, curriculum_epochs=config['curriculum_split'], min_weight=config['min_weight'], gamma=config['pacing_gamma']),
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
                    if strategy == 'curriculum':
                        model_name = f"HardTVAE_{hardness_metric}_{strategy}_anneal{config['min_weight']}_p{config['curriculum_split'][0]}"
                    elif strategy == 'self_paced':
                        model_name = f"HardTVAE_{hardness_metric}_{strategy}_anneal{config['min_weight']}_pacing{config['pacing_function_type']}"
                    else:
                        model_name = f"HardTVAE_{hardness_metric}_{strategy}_anneal{config['min_weight']}"
                    # model_name = f"HardTVAE_{hardness_metric}_{strategy}_{}"

                    base_info_h = {
                        "dataset": dataset_name,
                        "seed": seed,
                        "model": model_name,
                        "hardness_metric": str(hardness_metric),
                        "strategy": strategy,
                        "annealing_weight": config['min_weight'],
                        "pacing_function": config['pacing_function_type'] if strategy == 'self_paced' else None,
                        "curriculum_split": config['curriculum_split'] if strategy == 'curriculum' else None
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
                                                                random_state=seed,dataset_name=dataset_name, save_path=path_plots_h, dimensions_to_test=[3])
                    
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

def main():
    # datasets = ['BCWDD']
    datasets = ['Hypothyroid','NewThyroid1',  'Vertebral']
    hardness_metrics = ['F1', 'F4', 'N2', 'LSC', 'LSR', 'Harmfulness', 'MV', 'CB', 'DCP', 'TD_P', 'CLD']
    # hardness_metrics = [None, 'kDN', 'DS', 'DCP', 'TD_P',
    #                'TD_U', 'CL', 'CLD', 'MV', 'CB', 'N1', 'N2', 'LSC', 
    #                'LSR', 'Harmfulness', 'F1', 'F2', 'F3', 'F4']
    seeds            = list(random_seeds)
    print(f"seeds = {seeds}")

    # ---------------------------------------------------------------------------
    # SAVING PATHS
    # ---------------------------------------------------------------------------
    DIR = "RESULTS_WEIGHTING_STRATEGIES_SENSITIVITY"
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



    # < CHNAGED
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
        # # --------------------------------------------------------------
        # model_name = "CTGAN"
        # print(f"\n--- Training CTGAN Baseline for {dataset_name} ---")
        df_train_full = df_train_proc.copy()
        df_train_full['Outcome'] = y_train
        X_real_minority = X_train_proc[y_train==minority_label]
        # df_minority = df_train_full[df_train_full['Outcome'] == minority_label]

        # # -------------------------------------------------------------
        # # 2. HardTVAE (incuding TVAE baseline)
        # # -------------------------------------------------------------
        for hardness_metric in hardness_metrics:
            strategies =  ['curriculum', 'static', 'self_paced']
            for strategy in strategies:
                # model_name = "TVAE" if hardness_metric is None else f"HardTVAE_{hardness_metric}_{strategy}"
                # print(f"\n--- Training {model_name} for {dataset_name} ---")
                for config in sensitivity_configs:
                    print(f"Testing config: Curriculum Split {config['curriculum_split']}, Annealing Weight {config['min_weight']}, Pacing Function {config['pacing_function_type']}")
                        
                    model = TabularCVAE(
                        input_dim=X_train_proc.shape[1], 
                        latent_dim=5, condition_dim=1,
                        hidden_dims=[128, 64, 32],
                        data_info=data_info)
                    trainer = HardnessAwareCVAETrainer(
                        model= model,
                        hardness_calculator=HardnessCalculator(random_state=seed),
                        hardness_integrator=CVAEHardnessIntegrator(hardness_strategy=strategy, curriculum_epochs=config['curriculum_split'], min_weight=config['min_weight'], gamma=config['pacing_gamma']),
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
                    if strategy == 'curriculum':
                        model_name = f"HardTVAE_{hardness_metric}_{strategy}_anneal{config['min_weight']}_p{config['curriculum_split'][0]}"
                    elif strategy == 'self_paced':
                        model_name = f"HardTVAE_{hardness_metric}_{strategy}_anneal{config['min_weight']}_pacing{config['pacing_function_type']}"
                    else:
                        model_name = f"HardTVAE_{hardness_metric}_{strategy}_anneal{config['min_weight']}"
                    # model_name = f"HardTVAE_{hardness_metric}_{strategy}_{}"

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
                    keys, values, topo_detailed_results_h=  topological_fidelity_calculate(X_real=X_train_proc, y_real=y_train, X_synth=X_synth_proc, y_synth=y_synth,
                                                                random_state=seed,dataset_name=dataset_name, save_path=path_plots_h, return_detailed=True)
                    
                    fidelity_views_h['topological'] = dict(zip(keys, values))

                    # Save summary --- TVAE/HardTVAE ----
                    summary_row_h = {**base_info_h, **flatten_results(fidelity_views_h)}
                    pd.DataFrame([summary_row_h]).to_csv(FID_SUMMARY_CSV, mode='a',
                                                        header= not os.path.exists(FID_SUMMARY_CSV), index=False)
                    
                    # Save Details
                    h_detail_path = f"{ARTIFACTS_DIR}/{dataset_name}/seed_{seed}/{model_name}"
                    # Add your detailed return variables here
                    save_detailed_results(h_detail_path, "complexity_details", complexity_detailed_results_h)
                    save_detailed_results(h_detail_path, "hardness_details", hardness_detailed_results_h)
                    save_detailed_results(h_detail_path, "topo_details", topo_detailed_results_h)


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