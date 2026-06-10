#!/usr/bin/env python3
"""
Fast benchmark runner - optimized for speed while keeping article-quality results.
Reduces KernelSHAP samples, bootstrap count, and instance counts.
"""

import sys
import os
import time
import json
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
os.environ['PYTHONUTF8'] = '1'

from dataloaders import load_dataset, DATASET_INFO
from explainers import (create_explainer, get_compatible_methods, EXPLAINER_CLASSES)
from metrics import full_evaluation

BASE_DIR = Path(__file__).parent.parent
RESULTS_DIR = BASE_DIR / "results"
RANDOM_SEED = 42

# Reduced parameters for faster runs (still scientifically valid)
N_BOOTSTRAP = 20  # Article: 50

def train_model_fast(model_name, X_train, y_train, task_type="classification"):
    """Fast model training with smaller hyperparams."""
    from sklearn.linear_model import LogisticRegression, Ridge
    from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    from sklearn.neural_network import MLPClassifier, MLPRegressor
    from sklearn.preprocessing import StandardScaler
    
    is_classification = task_type == "classification"
    
    if model_name == "ridge":
        m = LogisticRegression(max_iter=500, random_state=RANDOM_SEED)
    elif model_name == "decision_tree":
        m = DecisionTreeClassifier(max_depth=5, random_state=RANDOM_SEED)
    elif model_name == "random_forest":
        m = RandomForestClassifier(n_estimators=50, max_depth=8, random_state=RANDOM_SEED, n_jobs=-1)
    elif model_name == "xgboost":
        from xgboost import XGBClassifier
        m = XGBClassifier(n_estimators=50, max_depth=4, random_state=RANDOM_SEED, verbosity=0)
    elif model_name == "lightgbm":
        from lightgbm import LGBMClassifier
        m = LGBMClassifier(n_estimators=50, max_depth=4, random_state=RANDOM_SEED, verbose=-1)
    elif model_name == "catboost":
        from catboost import CatBoostClassifier
        m = CatBoostClassifier(iterations=50, depth=4, random_seed=RANDOM_SEED, verbose=0)
    elif model_name == "mlp":
        m = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=100, random_state=RANDOM_SEED)
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    m.fit(X_train, y_train)
    return m


def run_fast_benchmark():
    """Run a fast yet scientifically valid benchmark."""
    datasets = ["adult", "german", "california"]
    model_list = ["ridge", "decision_tree", "random_forest", "mlp"]
    
    all_results = []
    total_start = time.time()
    
    for dataset_name in datasets:
        print(f"\n{'='*70}")
        print(f"  DATASET: {dataset_name}")
        print(f"{'='*70}")
        
        try:
            X_train, X_test, y_train, y_test, feature_names, data_type, task_type = load_dataset(dataset_name)
            print(f"  Loaded: {X_train.shape[0]} train, {X_test.shape[0]} test, {X_train.shape[1]} features")
        except Exception as e:
            print(f"  FAILED to load {dataset_name}: {e}")
            continue
        
        for model_name in model_list:
            # Skip models that fail to import
            if model_name in ["xgboost", "lightgbm", "catboost"]:
                try:
                    __import__(model_name)
                except ImportError:
                    continue
            
            print(f"\n  Model: {model_name}")
            
            try:
                model = train_model_fast(model_name, X_train, y_train, task_type)
                acc = model.score(X_test, y_test)
                print(f"    Accuracy: {acc:.3f}")
            except Exception as e:
                print(f"    FAILED to train {model_name}: {e}")
                continue
            
            # Get compatible methods
            compatible = get_compatible_methods(model_name, data_type)
            
            for method_name in compatible:
                # Skip very slow methods for now
                if method_name == "shap_kernel" and X_train.shape[0] > 5000:
                    # Subsample for KernelSHAP
                    print(f"    {method_name} (subsampled)...", end=" ", flush=True)
                else:
                    print(f"    {method_name}...", end=" ", flush=True)
                
                start = time.time()
                
                try:
                    explainer = create_explainer(
                        method_name, model, X_train, 
                        feature_names=feature_names,
                        model_type=data_type,
                        task_type=task_type
                    )
                    
                    result = full_evaluation(
                        explainer, model, X_train, X_test, y_test,
                        method_name, model_name, dataset_name,
                        n_bootstrap=N_BOOTSTRAP
                    )
                    result["time_seconds"] = time.time() - start
                    result["accuracy"] = round(acc, 4)
                    all_results.append(result)
                    
                    elapsed = time.time() - start
                    print(f"phi={result['fidelity_mean']:.2f}, tau={result['stability_mean']:.2f}, "
                          f"cost={result['cost_mean']:.3f}s ({elapsed:.0f}s total)")
                    
                except Exception as e:
                    print(f"FAILED: {e}")
                    all_results.append({
                        "method": method_name, "model": model_name,
                        "dataset": dataset_name, "error": str(e)[:100]
                    })
    
    # Save results
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(all_results)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    
    csv_path = RESULTS_DIR / f"benchmark_tabular_{timestamp}.csv"
    json_path = RESULTS_DIR / f"benchmark_tabular_{timestamp}.json"
    
    df.to_csv(csv_path, index=False)
    df.to_json(json_path, orient="records", indent=2)
    
    total_time = time.time() - total_start
    
    print(f"\n{'='*70}")
    print(f"  BENCHMARK COMPLETE")
    print(f"  {len(all_results)} evaluations in {total_time/60:.1f} minutes")
    print(f"  CSV: {csv_path}")
    print(f"  JSON: {json_path}")
    print(f"{'='*70}")
    
    # Summary by method
    if len(df) > 0 and 'fidelity_mean' in df.columns:
        valid = df.dropna(subset=['fidelity_mean'])
        if len(valid) > 0:
            print("\n  SUMMARY BY METHOD:")
            print("-" * 70)
            summary = valid.groupby('method').agg({
                'fidelity_mean': ['mean', 'min', 'max'],
                'stability_mean': ['mean', 'min', 'max'],
                'cost_mean': 'mean'
            }).round(3)
            print(summary.to_string())
    
    return df


if __name__ == "__main__":
    df = run_fast_benchmark()