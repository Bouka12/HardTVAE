# Experimental Reproducibility Guide

This guide describes how to reproduce the experiments reported in the paper
*"Hardness-Aware Tabular Variational Autoencoders for Synthetic Data
Generation in Imbalanced Learning"*, including environment setup, datasets,
model configurations, the experimental pipeline, and the evaluation metrics.

---

## 1. Environment Setup

The codebase relies on Python with PyTorch (GPU build assumed to be installed
separately) plus a set of scientific, generative-modeling, and
hardness/topology analysis libraries.

### Requirements

Install dependencies from `requirements.txt`:

```bash
pip install -r requirements.txt
```

Notes:
- PyTorch (`torch==2.6.0+cu124`) is assumed to be pre-installed (with CUDA
  support if a GPU is available) and is **not** included in
  `requirements.txt`.
- Key library groups:
  - **Core scientific**: `numpy`, `pandas`, `scipy`, `scikit-learn`,
    `statsmodels`
  - **Imbalanced learning**: `imbalanced-learn`, `imblearn`
  - **Generative models (SDV stack)**: `ctgan`, `sdv`, `sdmetrics`, `rdt`,
    `copulas`, `deepecho`
  - **Classifiers**: `xgboost`, `lightgbm`, `catboost`
  - **Topological / hardness analysis**: `ripser`, `persim`, `problexity`,
    `pymfe`, `gower`
  - **Visualization**: `matplotlib`, `seaborn`, `plotly`
  - **Utilities**: `tqdm`, `joblib`, `PyYAML`, `tabulate`, etc.

---

## 2. Datasets

Ten benchmark datasets spanning a range of sizes, dimensionalities, and
class-imbalance ratios (IR) are used. All characteristics below reflect the
datasets **after preprocessing**.

| Dataset Abbrev. | #Inst | #Feat | #Min | #Maj | IR |
|---|---|---|---|---|---|
| [BCWDD](https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic) | 569 | 30 | 212 | 357 | 1.68:1 |
| [HeartCleveland](https://archive.ics.uci.edu/dataset/45/heart+disease) | 297 | 13 | 137 | 160 | 1.16:1 |
| [Hepatitis](https://archive.ics.uci.edu/dataset/46/hepatitis) | 155 | 19 | 32 | 123 | 3.84:1 |
| [Hypothyroid](https://www.kaggle.com/datasets/yasserhessein/thyroid-disease-data-set) | 3772 | 5 | 291 | 3481 | 11.96:1 |
| [ILPD](https://archive.ics.uci.edu/dataset/225/ilpd+indian+liver+patient+dataset) | 583 | 10 | 167 | 416 | 2.49:1 |
| [NewThyroid1](https://sci2s.ugr.es/keel/dataset/data/imbalanced/new-thyroid1.zip) | 215 | 6 | 35 | 180 | 5.14:1 |
| [NewThyroid2](https://sci2s.ugr.es/keel/keel-dataset/datasets/imbalanced/imb_IRlowerThan9/new-thyroid2.zip) | 215 | 6 | 35 | 180 | 5.14:1 |
| [Pima](https://sci2s.ugr.es/keel/keel-dataset/datasets/imbalanced/imb_IRlowerThan9/pima.zip) | 392 | 9 | 130 | 262 | 2.01:1 |
| [Thoracic](https://archive.ics.uci.edu/dataset/277/thoracic+surgery+data) | 470 | 16 | 70 | 400 | 5.72:1 |
| [Vertebral](https://archive.ics.uci.edu/dataset/212/vertebral+column) | 310 | 6 | 100 | 210 | 2.1:1 |

**Column definitions**: `#Inst` = number of instances, `#Feat` = number of
features, `#Min` / `#Maj` = number of minority / majority instances, `IR` =
imbalance ratio ($N_{maj} / N_{min}$ : 1).

Processed dataset files are expected at `data/processed/<DatasetName>.csv`.

---

## 3. Data Preprocessing

Preprocessing is implemented in `utils.py` (`preprocess_data` class) and
follows established practice from prior tabular generative modeling work
(TabDDPM, TabSyn, STaSy), adapted for HardTVAE:

1. **Numerical missing values**: imputed with the column mean.
2. **Categorical missing values**: treated as a new category
   (`missing_category`).
3. **Categorical encoding**: one-hot encoding via `OneHotEncoder`
   (`handle_unknown='ignore'`).
4. **Numerical scaling**: `QuantileTransformer` (`output_distribution='uniform'`,
   `n_quantiles=min(n_samples, 5000)`), applied **only to numerical
   features** (one-hot columns are left untouched).

The fitted preprocessor exposes:
- `n_numerical`: number of numerical output columns
- `cat_dims`: list of one-hot group sizes (in order)
- `data_info`: `{'n_numerical': ..., 'cat_dims': [...]}`, passed directly to
  the `TabularCVAE` model
- `feature_names_out`: output column names (numerical columns first, then
  `<original_col>_<category_value>` for one-hot columns)

The preprocessor is fit on the training split only and applied to
validation/test splits via `.transform()` to avoid data leakage.

> **Note on mixed-type handling**: TVAE and CTGAN are trained on the
> mixed-type (preprocessed but not fully numericized) training data via SDV.
> HardTVAE operates on the fully numerical/one-hot encoded representation
> produced by `preprocess_data`.

---

## 4. Generative Models

### TVAE and HardTVAE

Both models share a symmetric encoder-decoder architecture, detailed in the
table below. HardTVAE extends TVAE with a hardness-aware curriculum weighting
mechanism that dynamically adjusts per-sample weights across three training
stages based on instance learning difficulty.

| Parameter | Default Value | Description |
|---|---|---|
| `input_dim` | Variable | Dimensionality of input tabular data |
| `latent_dim` | 5 | Dimensionality of latent space |
| `condition_dim` | Variable | Dimensionality of conditioning variables |
| `hidden_dims` | `[128, 64]` | Hidden layer dimensions (encoder/decoder) |
| Dropout Rate | 0.2 | Dropout probability for regularization |
| Learning Rate | $1 \times 10^{-3}$ | Adam optimizer learning rate |
| Beta ($\beta$) | 1.0 | KL divergence weighting factor |
| Activation Function | ReLU | Non-linear activation for hidden layers |
| Normalization | BatchNorm1d | Batch normalization for training stability |
| Loss Function | MSE + KL | Reconstruction loss + KL divergence |
| Stage cutoffs | $(0.3T,\ 0.3T,\ 0.4T)$ | Stage boundaries for the easy→hard transition in curriculum weighting |
| Annealing weight ($\epsilon$) | 0.1 | Annealing weight for self-paced weighting |

($T$ = total number of training epochs.)

### CTGAN

CTGAN is implemented via the [`sdv`](https://github.com/sdv-dev/SDV) Python
package with default architectural hyperparameters, since its
generator-discriminator design is not directly comparable to the
encoder-decoder structure of TVAE-based models.

### Training Protocol

All three models (CTGAN, TVAE, HardTVAE) are trained with a **fixed number of
epochs ($T = 150$)** and **batch size of 32**, applied consistently across all
datasets. This is a deliberate methodological choice: since the evaluation is
multi-objective and the target use case is class-imbalance correction (rather
than general synthesis benchmarking), tuning each model on a single metric
would introduce optimization bias inconsistent with the evaluation framework.
The fixed-setting protocol ensures observed differences are attributable to
architectural properties rather than tuning advantages.

---

## 5. Experimental Pipeline (`experiments.py`)

The full pipeline is implemented in `experiments.py` and run per
`(dataset, seed)` combination.

### 5.1 Setup

- A `MASTER_SEED = 42` is used to generate the list of random seeds
  (`random_seeds`) used across all experiments via `random.sample`.
- `N_EPOCHS = 150`; curriculum stage cutoffs are derived as
  `(0.3·N_EPOCHS, 0.3·N_EPOCHS, 0.4·N_EPOCHS)`.
- Results are written under a top-level `RESULTS/` directory:
  - `RESULTS/fidelity/fidelity_summary.csv` — fidelity results summary
  - `RESULTS/fidelity/artifacts/<dataset>/seed_<seed>/<model>/` — detailed
    per-run JSON artifacts (`complexity_details.json`,
    `hardness_details.json`)
  - `RESULTS/fidelity/plots/<model>/<dataset>/` — diagnostic plots from
    hardness/topological fidelity evaluation
  - `RESULTS/utility/utility_summary.csv` — utility evaluation summary
  - `RESULTS/utility/best_params.csv` — best hyperparameters from grid search

### 5.2 Per-(dataset, seed) loop

For each `(dataset_name, seed)` pair:

1. **Set random seed** (`set_seed(seed)`) for full reproducibility.
2. **Load data**: `load_data(DATA_PATH, random_state=seed)` returns
   `X_train, y_train, X_val, y_val, X_test, y_test`.
3. **Preprocess**: fit `preprocess_data` on `X_train`; transform validation
   and test splits (see Section 3).
4. **Identify minority/majority classes** and compute
   `n_samples_needed = |majority| - |minority|` (the number of synthetic
   minority samples required to balance the training set).

#### Step 1 — CTGAN baseline

- Train CTGAN (`ctgan(...)`, `epochs=N_EPOCHS`, `batch_size=32`,
  `seed=seed`) and generate synthetic minority samples.
- Evaluate **fidelity** across four views (see Section 6):
  distributional, complexity, hardness, topological.
- Evaluate **utility**: augment `X_train`/`y_train` with the synthetic
  minority samples and run `ClassificationEvaluator` (XGBoost, with grid
  search — see Section 7).
- Append results to `fidelity_summary.csv` and `utility_summary.csv`; save
  detailed JSON artifacts and best hyperparameters.

#### Step 2 — TVAE / HardTVAE configurations

For each `hardness_metric` in the configured list (`None` for plain TVAE, or
one of the 17 supported hardness measures — see Section 8) and each
`strategy`:

- If `hardness_metric is None`: only the `static` strategy is run, and
  `model_name = "TVAE"`.
- Otherwise: all three strategies (`curriculum`, `static`, `self_paced`) are
  run, and `model_name = "HardTVAE_<hardness_metric>_<strategy>"`.

For each configuration:

1. Instantiate `TabularCVAE` (`latent_dim=5`, `condition_dim=1`,
   `hidden_dims=[128, 64, 32]`, `data_info` from the preprocessor).
2. Instantiate `HardnessAwareCVAETrainer` with a `HardnessCalculator` and
   `CVAEHardnessIntegrator(hardness_strategy=strategy)`.
3. Compute hardness scores for the chosen `hardness_metric` (skipped/no-op
   if `hardness_metric is None`).
4. Train for `N_EPOCHS` epochs via `trainer.train_epoch(...)`.
5. Generate `n_samples_needed` synthetic minority samples
   (`trainer.generate_samples(...)`).
6. Evaluate **fidelity** (distributional, complexity, hardness,
   topological) against the real minority/training data.
7. Evaluate **utility**: augment the training set with the synthetic
   samples and run `ClassificationEvaluator`.
8. Append results to `fidelity_summary.csv` and `utility_summary.csv`; save
   detailed JSON artifacts and best hyperparameters to
   `best_params.csv`.

### 5.3 Running the pipeline

```bash
python experiments.py
```

By default, `datasets` and `hardness_metrics` in `main()` are restricted to a
small subset for testing (e.g. `['Thoracic']`, `['TD_P']`). To run the full
study, set:

```python
datasets = [
    'BCWDD', 'HeartCleveland', 'Hepatitis', 'Hypothyroid',
    'ILPD', 'NewThyroid1', 'NewThyroid2', 'Pima', 'Thoracic', 'Vertebral'
]
hardness_metrics = [None, 'kDN', 'DS', 'DCP',
                     'TD_U', 'CL', 'CLD', 'MV', 'CB', 'N1', 'N2', 'LSC',
                     'LSR', 'Harmfulness', 'F1', 'F2', 'F3', 'F4']
```

> **Note**: `TD_P` (Tree Depth Pruned) is excluded from the generative
> configurations (see Section 8) but retained for the hardness fidelity
> evaluation.

---

## 6. Multi-View Fidelity Evaluation

For each trained model, synthetic minority data is evaluated against real
data from four complementary perspectives:

- **Distributional fidelity** (`distributional_fidelity_calculate`):
  marginal alignment between real and synthetic features (e.g., KS
  statistics).
- **Complexity fidelity** (`complexity_fidelity_calculate`): similarity of
  data-complexity meta-features between real and synthetic data
  (`k=3` nearest neighbors).
- **Hardness fidelity** (`hardness_fidelity_calculate`): transfer of
  instance-hardness patterns from real to synthetic data (`k=3`); also
  produces diagnostic plots.
- **Topological fidelity** (`topological_fidelity_calculate`): preservation
  of manifold/topological structure (persistent homology,
  `dimensions_to_test=[3]`).

Results for all four views are flattened into a single row per
`(dataset, seed, model, hardness_metric, strategy)` combination and appended
to `fidelity_summary.csv`.

---

## 7. Utility Evaluation

The downstream classification utility of synthetic minority data is evaluated
using **XGBoost** (`ClassificationEvaluator`).

### Procedure

1. The dataset is split into **training, validation, and test** subsets.
2. The minority class in the **training subset only** is augmented (with
   synthetic samples) until the training subset is balanced; validation and
   test subsets remain representative of the real (imbalanced) data.
3. The balanced training subset is used to fit the classifier.
4. Hyperparameters are tuned on the validation subset via
   `GridSearchCV` (predefined `0.8 / 0.1 / 0.1` train/val/test split),
   selecting the best configuration by area under the precision-recall curve
   (`aucpr`).
5. The best model is evaluated on the held-out test subset, reporting
   **F1-score, specificity, recall, and precision**.

### XGBoost hyperparameters

| Parameter | Value / Search Space | Type |
|---|---|---|
| `n_estimators` | 500 | Fixed |
| `early_stopping_rounds` | 15 | Fixed |
| `eval_metric` | `aucpr` | Fixed |
| `learning_rate` | $\{0.01, 0.05, 0.1\}$ | Tuned |
| `min_child_weight` | $\{1, 3, 5\}$ | Tuned |
| `max_depth` | $\{3, 4, 5\}$ | Tuned |
| `gamma` | $\{0.0, 0.1, 0.5\}$ | Tuned |
| `subsample` | $\{0.7, 0.8\}$ | Tuned |
| `colsample_bytree` | $\{0.7, 0.8\}$ | Tuned |

Subsampling and column sampling are included given the small-to-medium size
of the datasets, to mitigate overfitting. Best hyperparameters per
configuration are logged to `RESULTS/utility/best_params.csv`.

### Utility Gain (UG) and Gain Index (GI)

To quantify the downstream impact of HardTVAE relative to the TVAE and CTGAN
baselines, two quantities are computed for each metric $m$ (F1-score,
specificity, recall, precision):

$$
UG_{i,k}(m) = m^{\mathrm{HardTVAE}}_{i,k} - m^{\mathrm{Baseline}}_{i,k}
$$

$$
GI_{k}(m) = \frac{1}{n} \sum_{i=1}^{n} UG_{i,k}(m)
$$

where $k$ indexes the configuration `{weighting strategy, hardness metric,
dataset, classifier}`, $i$ indexes the run (one of $n = 10$ random seeds),
and `Baseline` $\in$ `{TVAE, CTGAN}`.

- **UG** measures the per-run difference in metric $m$ between HardTVAE and a
  given baseline for configuration $k$.
- **GI** summarizes the average improvement (or degradation) across the $n$
  runs for configuration $k$.

These are computed separately against **TVAE** and **CTGAN** as baselines
(see `gain_indices_vs_TVAE.csv` and `gain_indices_vs_CTGAN.csv` in the
companion results).

---

## 8. Hardness Measures

HardTVAE supports **18 PyHard hardness measures**, grouped as follows:

- **Neighborhood-based**: kDN, N1, N2, LSC, LSR
- **Likelihood-based**: DS, DCP, CL, CLD
- **Tree-based**: TD_P (Tree Depth Pruned), TD_U (Tree Depth Unpruned)
- **Class-balance**: MV (Minority Value), CB (Class Balance)
- **Fisher criterion**: F1, F2, F3, F4
- Additional: Harmfulness

> **Exclusion of TD_P from generative configurations**: On the `Thoracic`
> dataset, the pruned decision tree underlying TD_P collapsed to a trivial
> (near-root) tree, yielding a null hardness score for all instances (no
> usable difficulty signal for instance weighting). To keep the
> configuration space consistent across datasets, TD_P is **excluded** from
> the HardTVAE generative configurations, giving:
>
> $$17 \text{ measures} \times 3 \text{ strategies} = 51 \text{ configurations}$$
>
> TD_P **is** retained for the hardness fidelity evaluation.

Detailed mathematical definitions of all 18 hardness measures are provided in
the Supplementary Appendix (Section A).

---

## 9. Stability and Reproducibility

- Each HardTVAE configuration is evaluated over **10 independent
  repetitions** (`n_runs = 10`), using a predefined list of random states
  generated from a fixed master seed (`MASTER_SEED = 42`).
- The experimental space consists of:
  - 3 weighting strategies: `static`, `curriculum`, `self_paced`
  - 17 hardness measures (excluding TD_P; one classifier — XGBoost)
  - 10 datasets
  - 10 random repetitions per configuration
  - → $3 \times 17 = 51$ HardTVAE variants, and
    $10 \times 17 \times 3 = 510$ HardTVAE executions per dataset
- Baselines (TVAE, CTGAN) are run with the same 10 seeds for direct
  comparability.
- All random states, seeds, raw results, and the full experimental loop
  (including the random-state generator) are available in this repository
  and on the companion webpage, enabling complete replication of the
  pipeline.

---

## 10. Output Summary

| File | Description |
|---|---|
| `RESULTS/fidelity/fidelity_summary.csv` | One row per `(dataset, seed, model, hardness_metric, strategy)` with all four fidelity-view metrics |
| `RESULTS/fidelity/artifacts/<dataset>/seed_<seed>/<model>/complexity_details.json` | Detailed complexity-fidelity results |
| `RESULTS/fidelity/artifacts/<dataset>/seed_<seed>/<model>/hardness_details.json` | Detailed hardness-fidelity results |
| `RESULTS/fidelity/plots/<model>/<dataset>/` | Diagnostic plots (hardness/topological fidelity) |
| `RESULTS/utility/utility_summary.csv` | Per-run classification metrics (F1, specificity, recall, precision) |
| `RESULTS/utility/best_params.csv` | Best XGBoost hyperparameters per configuration (grid search) |
