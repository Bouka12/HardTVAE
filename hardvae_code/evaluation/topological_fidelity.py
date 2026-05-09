"""
topological_fidelity.py
=======================
Topological fidelity view for sensitivity analysis over IsUMap embedding
dimensions. Measures how well synthetic minority samples preserve the
topological structure of the real data using Persistent Homology (TDA).

Returns a single flat dict — one value per (dimension × metric) combination —
ready to be saved as a CSV row directly.

    result = topological_fidelity_calculate(X_real, y_real, X_synth, y_synth)
    # e.g. result['d3_bottleneck_H0'], result['d5_wasserstein_H1_similarity']

Pipeline
--------
For each embedding dimension d in dimensions_to_test (default: [3, 4, 5]):
  1. Embed X_real_minority and X_synth with IsUMap (k neighbours)
  2. Compute Vietoris-Rips persistence diagrams via ripser (H0 and H1)
  3. Compute Bottleneck and Wasserstein distances between diagrams
  4. Compute similarity scores: 1 / (1 + distance)  →  mapped to (0, 1]

Dependencies
------------
  pip install ripser persim
  IsUMap: place isumap/src/ relative to project root (see path logic below)
"""

import os
import sys
import warnings
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Optional TDA dependencies ─────────────────────────────────────────────────
TDA_AVAILABLE = True
_missing = []

try:
    from ripser import ripser
except ImportError:
    _missing.append("ripser")
    TDA_AVAILABLE = False

try:
    from persim import plot_diagrams
    try:
        from persim import bottleneck, wasserstein
    except ImportError:
        from persim.distances import bottleneck, wasserstein
except ImportError:
    _missing.append("persim")
    TDA_AVAILABLE = False

# ── IsUMap ────────────────────────────────────────────────────────────────────
project_root    = Path(__file__).resolve().parent.parent.parent
isumap_src_path = project_root / "isumap" / "src"
isumap_function = None

if isumap_src_path.exists():
    if str(isumap_src_path) not in sys.path:
        sys.path.append(str(isumap_src_path))
    try:
        import isumap as _isumap_module
        isumap_function = _isumap_module.isumap
        print("IsUMap function loaded successfully.")
    except ImportError as e:
        print(f"Error: isumap.py not found in {isumap_src_path}. {e}")
        TDA_AVAILABLE = False
    except AttributeError:
        print("Error: isumap.py exists but has no isumap() function.")
        TDA_AVAILABLE = False
else:
    print(f"Warning: isumap source path not found at {isumap_src_path}")
    TDA_AVAILABLE = False

if not TDA_AVAILABLE:
    warnings.warn(
        f"[topological_fidelity] Missing packages/modules: {_missing}. "
        "topological_fidelity() will return NaN values. "
        f"Install with: pip install {' '.join(_missing)}",
        ImportWarning,
        stacklevel=2,
    )

# ── Metric templates per homology dimension ───────────────────────────────────
HOMOLOGY_DIMS = [0, 1]   # H0 = connected components, H1 = loops


def _all_keys_for_dim(d_val: int) -> list:
    """Return all expected flat dict keys for one embedding dimension."""
    keys = []
    for h in HOMOLOGY_DIMS:
        keys += [
            f'd{d_val}_bottleneck_H{h}',
            f'd{d_val}_bottleneck_H{h}_similarity',
            f'd{d_val}_wasserstein_H{h}',
            f'd{d_val}_wasserstein_H{h}_similarity',
        ]
    keys.append(f'd{d_val}_status')
    return keys


# =============================================================================
# Internal helpers
# =============================================================================

def _to_array(data) -> np.ndarray:
    if isinstance(data, (pd.DataFrame, pd.Series)):
        return data.values.astype(np.float64)
    return np.array(data, dtype=np.float64)


def _get_embedding(isumap_result, d: int) -> np.ndarray:
    """Extract the (n, d)-shaped embedding array from isumap output."""
    if isinstance(isumap_result, np.ndarray) and isumap_result.ndim == 2:
        return isumap_result
    for item in isumap_result:
        if isinstance(item, np.ndarray) and item.ndim == 2 and item.shape[1] == d:
            return item
    raise ValueError(
        f"Could not find embedded array with shape (n, {d}) "
        f"in isumap output types: {[type(x) for x in isumap_result]}"
    )


def _compute_diagrams(X_embedded: np.ndarray) -> list:
    """Compute Vietoris-Rips persistence diagrams. Returns list indexed by H dim."""
    result = ripser(X_embedded, maxdim=max(HOMOLOGY_DIMS))
    return result['dgms']   # dgms[0] = H0, dgms[1] = H1


def _save_persistence_plots(dgm_real: list, dgm_synth: list,
                             save_path: str, label: str, d_val: int) -> None:
    try:
        os.makedirs(save_path, exist_ok=True)
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        plot_diagrams(dgm_real,  ax=axes[0],
                      title=f'Real Data (d={d_val}) — {label}')
        plot_diagrams(dgm_synth, ax=axes[1],
                      title=f'Synthetic Data (d={d_val}) — {label}')
        plt.tight_layout()
        fname = os.path.join(save_path, f'persistence_d{d_val}_{label}.png')
        plt.savefig(fname, dpi=100, bbox_inches='tight')
        plt.close()
        print(f"  [topological_fidelity] Plot saved: {fname}")
    except Exception as e:
        print(f"  [topological_fidelity] Could not save plot: {e}")


# =============================================================================
# Public API
# =============================================================================

def topological_fidelity_calculate(
    X_real,
    y_real,
    X_synth,
    y_synth,
    minority_class=None,
    dimensions_to_test=None,
    isumap_k: int = 15,
    save_path: str = None,
    dataset_name: str = "dataset",
    random_state: int = 42,
) -> dict:
    """
    Topological fidelity sensitivity analysis over IsUMap embedding dimensions.

    Parameters
    ----------
    X_real : array-like (n_total, n_features)
        Full training set (minority + majority). Minority subset is extracted
        automatically using y_real.
    y_real : array-like (n_total,)
    X_synth : array-like (n_synth, n_features)
        Synthetic minority-class samples only.
    y_synth : array-like (n_synth,)
    minority_class : int/str, optional
        Auto-detected as least frequent class if None.
    dimensions_to_test : list[int], default=[3, 4, 5]
        Each entry is one embedding dimension — the sensitivity analysis
        reports one set of metrics per dimension.
    isumap_k : int, default=15
        Number of neighbours for IsUMap.
    save_path : str, optional
        Directory for persistence diagram plots. No plots saved if None.
    dataset_name : str, default='dataset'
    random_state : int, default=42

    Returns
    -------
    result : dict
        Single flat dict, one row worth of data, with keys:

          d{d}_bottleneck_H0              Bottleneck distance H0
          d{d}_bottleneck_H0_similarity   1/(1+distance), in (0,1]
          d{d}_bottleneck_H1
          d{d}_bottleneck_H1_similarity
          d{d}_wasserstein_H0
          d{d}_wasserstein_H0_similarity
          d{d}_wasserstein_H1
          d{d}_wasserstein_H1_similarity
          d{d}_status                     'ok' or 'failed: <reason>'

        for each d in dimensions_to_test.

        Lower distance  = topologically more similar.
        Higher similarity = topologically more similar.
        Failed/unavailable dimensions have NaN metric values.
    """
    np.random.seed(random_state)

    if dimensions_to_test is None:
        dimensions_to_test = [3, 4, 5]

    # Initialise all expected keys to NaN / 'failed' up front
    result = {}
    for d_val in dimensions_to_test:
        for key in _all_keys_for_dim(d_val):
            result[key] = np.nan
        result[f'd{d_val}_status'] = 'failed'

    if not TDA_AVAILABLE:
        warnings.warn(
            f"[topological_fidelity] Returning NaN — missing: {_missing}",
            RuntimeWarning,
        )
        return result

    # ── Type conversion ───────────────────────────────────────────────────────
    X_real  = _to_array(X_real)
    y_real  = np.array(y_real).ravel()
    X_synth = _to_array(X_synth)

    # ── Auto-detect minority class ────────────────────────────────────────────
    unique_classes, class_counts = np.unique(y_real, return_counts=True)
    print(f"[topological_fidelity] classes={unique_classes}  counts={class_counts}")

    if len(unique_classes) < 2:
        raise ValueError(
            f"Need at least 2 classes, got {len(unique_classes)}: {unique_classes}"
        )
    if minority_class is None:
        minority_class = unique_classes[np.argmin(class_counts)]
        print(f"[topological_fidelity] Auto-detected minority class: {minority_class}")

    X_real_min = X_real[y_real == minority_class]
    print(f"[topological_fidelity] Real minority: {X_real_min.shape}  "
          f"Synthetic: {X_synth.shape}")

    # ── Per-dimension loop ────────────────────────────────────────────────────
    for d_val in dimensions_to_test:
        prefix = f'd{d_val}'
        print(f"\n[topological_fidelity] ─── IsUMap d={d_val} ───")

        try:
            # 1. Embed both real minority and synthetic with IsUMap
            emb_real  = isumap_function(data=X_real_min, d=d_val,
                                        k=isumap_k, verbose=False)
            emb_synth = isumap_function(data=X_synth,    d=d_val,
                                        k=isumap_k, verbose=False)
            X_real_emb  = _get_embedding(emb_real,  d_val)
            X_synth_emb = _get_embedding(emb_synth, d_val)
            print(f"  Embeddings — real: {X_real_emb.shape}  "
                  f"synth: {X_synth_emb.shape}")

            # 2. Compute Vietoris-Rips persistence diagrams
            dgm_real  = _compute_diagrams(X_real_emb)
            dgm_synth = _compute_diagrams(X_synth_emb)
            print(f"  Diagrams — H0 real: {dgm_real[0].shape}  "
                  f"H0 synth: {dgm_synth[0].shape}")

            # 3. Bottleneck & Wasserstein for each homology dimension
            for h in HOMOLOGY_DIMS:
                if h >= len(dgm_real) or h >= len(dgm_synth):
                    print(f"  H{h} not available for d={d_val} — skipping")
                    continue

                bn    = float(bottleneck(dgm_real[h],  dgm_synth[h]))
                ws    = float(wasserstein(dgm_real[h], dgm_synth[h]))
                bn_sim = 1.0 / (1.0 + bn)
                ws_sim = 1.0 / (1.0 + ws)

                result[f'{prefix}_bottleneck_H{h}']            = bn
                result[f'{prefix}_bottleneck_H{h}_similarity'] = bn_sim
                result[f'{prefix}_wasserstein_H{h}']           = ws
                result[f'{prefix}_wasserstein_H{h}_similarity'] = ws_sim

            result[f'{prefix}_status'] = 'ok'
            print(f"  d={d_val} → " + "  ".join(
                f"bn_H{h}={result[f'{prefix}_bottleneck_H{h}']:.4f}  "
                f"ws_H{h}={result[f'{prefix}_wasserstein_H{h}']:.4f}"
                for h in HOMOLOGY_DIMS
                if not np.isnan(result.get(f'{prefix}_bottleneck_H{h}', np.nan))
            ))

            # 4. Optional persistence diagram plots
            if save_path:
                _save_persistence_plots(
                    dgm_real, dgm_synth, save_path,
                    label=f'{dataset_name}_seed{random_state}',
                    d_val=d_val,
                )

        except Exception as exc:
            print(f"  Warning: d={d_val} failed — {exc}")
            traceback.print_exc()
            result[f'{prefix}_status'] = f'failed: {exc}'

    return result