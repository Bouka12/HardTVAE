# HardTVAE: Hardness-Aware Tabular Variational Autoencoders

Companion code and results for *"Hardness-Aware Tabular Variational
Autoencoders for Synthetic Data Generation in Imbalanced Learning"*.

HardTVAE integrates instance hardness directly into the VAE training
objective, giving harder (borderline, rare, noisy) minority instances higher
weight during reconstruction. A multi-view fidelity framework (distributional,
topological, hardness-based, complexity-based) and a utility evaluation
protocol (XGBoost + Utility Gain / Gain Index) are used to assess the
synthetic data against TVAE and CTGAN baselines.

## Installation

```bash
pip install -r requirements.txt
```

PyTorch (CUDA build recommended) must be installed separately.

## Repository Structure

```
data/processed/        # Preprocessed dataset CSVs
utils.py                # Preprocessing pipeline
experiments.py          # Main experimental pipeline
models/                  # HardTVAE, CTGAN wrapper, hardness calculators
evaluation/              # Fidelity and utility evaluation modules
RESULTS/                 # Output: fidelity & utility summaries, artifacts, plots
```

## Running Experiments

```bash
python experiments.py
```

Edit the `datasets` and `hardness_metrics` lists in `experiments.py` to run
the full study (10 datasets, 17 hardness measures, 3 weighting strategies, 10
seeds). Results are written to `RESULTS/fidelity/` and `RESULTS/utility/`.

## Datasets

10 benchmark datasets (e.g. BCWDD, Hypothyroid, Pima, Thoracic, Vertebral,
...) with imbalance ratios from 1.16:1 to 11.96:1. See the Supplementary
Appendix and `EXPERIMENTAL_REPRODUCIBILITY_GUIDE.md` for full details.

## Reproducibility

All experiments use a fixed master seed and a predefined list of random
states across 10 repetitions per configuration, ensuring full
reproducibility. See `EXPERIMENTAL_REPRODUCIBILITY_GUIDE.md` for the complete
experimental protocol, model configurations, and evaluation metrics.

## Companion Website

Interactive exploration of fidelity and utility results is available on the
[companion website](#), including:
- Multi-View Fidelity Index (MFI) results
- Gain Index results vs. TVAE and CTGAN baselines



