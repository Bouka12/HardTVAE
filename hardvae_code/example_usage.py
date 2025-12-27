"""
Example usage of the Synthetic Data Evaluator Module

This script demonstrates how to use the comprehensive evaluation framework
for assessing synthetic minority data quality.
"""

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
import os

# Import our evaluation module
from synthetic_data_evaluator import SyntheticDataEvaluator

def create_sample_data():
    """
    Create sample imbalanced dataset for demonstration.
    """
    # Generate imbalanced dataset
    X, y = make_classification(
        n_samples=1000,
        n_features=10,
        n_informative=8,
        n_redundant=2,
        n_clusters_per_class=1,
        weights=[0.9, 0.1],  # Imbalanced classes
        random_state=42
    )
    
    # # Split into majority and minority
    # minority_mask = y == 1
    # X_minority = X[minority_mask]
    # y_minority = y[minority_mask]
    
    return X, y

def create_synthetic_data(X_real, y_real, method='noise'):
    """
    Create synthetic data using simple methods for demonstration.
    """
    n_samples = len(X_real)
    
    if method == 'noise':
        # Add Gaussian noise to real data
        noise = np.random.normal(0, 0.1, X_real.shape)
        X_synth = X_real + noise
        y_synth = y_real.copy()
        
    elif method == 'interpolation':
        # Linear interpolation between random pairs
        X_synth = []
        y_synth = []
        
        for i in range(n_samples):
            # Select two random samples
            idx1, idx2 = np.random.choice(len(X_real), 2, replace=False)
            alpha = np.random.random()
            
            # Interpolate
            x_new = alpha * X_real[idx1] + (1 - alpha) * X_real[idx2]
            X_synth.append(x_new)
            y_synth.append(y_real[idx1])  # Use label from first sample
        
        X_synth = np.array(X_synth)
        y_synth = np.array(y_synth)
    
    else:  # 'poor' quality synthetic data
        # Generate completely random data (poor quality)
        X_synth = np.random.normal(0, 1, X_real.shape)
        y_synth = y_real.copy()
    
    return X_synth, y_synth

def run_evaluation_example():
    """
    Run a complete evaluation example.
    """
    print("Synthetic Data Evaluation Example")
    print("=" * 40)
    
    # Create sample data
    print("1. Creating sample minority data...")
    X_real, y_real = create_sample_data()
    print(f"   Real minority data shape: {X_real.shape}")
    
    # Create different quality synthetic data
    methods = ['noise', 'interpolation', 'poor']
    results_comparison = {'statistical':{}, 'hardness':{}, 'complexity':{},  'topological':{}}
    
    # Create output directory
    output_dir = "evaluation_results"
    os.makedirs(output_dir, exist_ok=True)
    
    for method in methods:
        print(f"\n2. Creating synthetic data using '{method}' method...")
        X_synth, y_synth = create_synthetic_data(X_real, y_real, method)
        print(f"   Synthetic data shape: {X_synth.shape}")
        
        print(f"\n3. Running comprehensive evaluation for '{method}' method...")
        
        # Initialize evaluator
        evaluator = SyntheticDataEvaluator(random_state=42)
        
        # Run evaluation
        results = evaluator.evaluate_all(
            X_real, y_real, 
            X_synth, y_synth,
            save_path=output_dir,
            dataset_name=f"sample_{method}"
        )
        
        # Store results for comparison
        for key in results_comparison.keys():
            results_comparison[key][method] = results[key]

        # Save the comparison results:
        save_comparison_results_to_csv(results_comparison, f"sample_{method}", output_dir )
        

    #     # Print results_comparison and figure out later how to save such a file csv/json
    #     # Print summary
    #     summary = results['summary']
    #     print(f"\n   Results for '{method}' method:")
    #     print(f"   Overall Quality Score: {summary['overall_quality_score']:.3f}")
    #     print(f"   Assessment: {summary['assessment']}")
        
    #     print("   Component Scores:")
    #     for component, score in summary['component_scores'].items():
    #         print(f"     {component.capitalize()}: {score:.3f}")
        
    #     if summary['recommendations']:
    #         print("   Recommendations:")
    #         for rec in summary['recommendations']:
    #             print(f"     - {rec}")
    
    # # Compare methods
    # print(f"\n4. Method Comparison:")
    # print("-" * 30)
    # comparison_df = pd.DataFrame({
    #     method: {
    #         'Overall Score': results['overall_quality_score'],
    #         'Assessment': results['assessment'],
    #         **results['component_scores']
    #     }
    #     for method, results in results_comparison.items()
    # }).T
    
    # print(comparison_df.round(3))
    
    # # Save comparison
    # comparison_path = os.path.join(output_dir, "method_comparison.csv")
    # comparison_df.to_csv(comparison_path)
    # print(f"\nResults saved to: {output_dir}")
    print(f"Comparison results: \n{results_comparison}")


def save_comparison_results_to_csv(results_comparison, dataset_name, save_dir):
    os.makedirs(save_dir, exist_ok=True)

    for aspect, methods_dict in results_comparison.items():
        # Define path to the CSV file for this aspect
        csv_path = os.path.join(save_dir, f"{aspect}_comparison.csv")
        rows = []

        for method, metrics in methods_dict.items():
            # Compose a row with dataset and method info
            row = {'dataset': dataset_name, 'method': method}
            row.update(metrics)
            rows.append(row)

        # Create a DataFrame for the new rows
        df_new = pd.DataFrame(rows)

        # If file exists, append without duplicating header
        if os.path.exists(csv_path):
            df_new.to_csv(csv_path, mode='a', header=False, index=False)
        else:
            df_new.to_csv(csv_path, index=False)

        print(f"Updated: {csv_path}")

def run_quick_test():
    """
    Run a quick test of core functionality.
    """
    print("Quick Functionality Test")
    print("=" * 25)
    
    # Create minimal test data
    np.random.seed(42)
    X_real = np.random.normal(0, 1, (100, 5))
    y_real = np.ones(100)
    
    # Create synthetic data with noise
    X_synth = X_real + np.random.normal(0, 0.1, X_real.shape)
    y_synth = y_real.copy()
    
    # Initialize evaluator
    evaluator = SyntheticDataEvaluator(random_state=42)
    
    # Test individual components
    print("Testing individual evaluation components...")
    
    try:
        # Statistical evaluation
        stat_result = evaluator.statistical_evaluation(X_real, y_real, X_synth, y_synth)
        print(f"✓ Statistical evaluation: {stat_result['mean_similarity']:.3f}")
    except Exception as e:
        print(f"✗ Statistical evaluation failed: {e}")
    
    try:
        # Hardness evaluation
        hardness_result = evaluator.hardness_evaluation(X_real, X_synth)
        print(f"✓ Hardness evaluation: {hardness_result['overall_similarity']:.3f}")
    except Exception as e:
        print(f"✗ Hardness evaluation failed: {e}")
    
    print("\nQuick test completed!")

if __name__ == "__main__":
    # Run quick test first
    run_quick_test()
    
    print("\n" + "="*50 + "\n")
    
    # Run full example
    run_evaluation_example()

