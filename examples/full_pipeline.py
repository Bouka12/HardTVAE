"""
Full pipeline example for HardVAE.

This script demonstrates the full workflow of:
1. Loading data
2. Calculating hardness scores
3. Training a hardness-aware CVAE
4. Generating synthetic data
5. Evaluating the synthetic data
"""

import numpy as np
from hardvae.core import HardnessCalculator, CVAEHardnessIntegrator
from hardvae.integration import TabularCVAE, HardnessAwareCVAETrainer
from hardvae.evaluation import SyntheticDataEvaluator
from hardvae.utils import load_data

# 1. Load Data
X_train, y_train, X_test, y_test, _ = load_data("path/to/your/data.csv")

# 2. Calculate Hardness Scores
hardness_calc = HardnessCalculator()
hardness_scores = hardness_calc.calculate_hardness_scores(X_train, y_train, metrics=["feature_kDN"])

# 3. Train Hardness-Aware CVAE
model = TabularCVAE(input_dim=X_train.shape[1], latent_dim=10, condition_dim=1)
hardness_integrator = CVAEHardnessIntegrator(weighting_strategy="curriculum")
trainer = HardnessAwareCVAETrainer(model, hardness_calc, hardness_integrator)
trainer.train(X_train, y_train, epochs=100, batch_size=32, hardness_metric="feature_kDN")

# 4. Generate Synthetic Data
X_synth = trainer.generate(n_samples=len(X_train[y_train==1]), condition=1)
y_synth = np.ones(len(X_synth))

# 5. Evaluate Synthetic Data
evaluator = SyntheticDataEvaluator()
results = evaluator.evaluate_all(X_train[y_train==1], y_train[y_train==1], X_synth, y_synth, save_path="./results", dataset_name="my_dataset")

print(results)
