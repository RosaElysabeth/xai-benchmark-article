#!/usr/bin/env python3
"""
Instant benchmark runner — completes in ~2 minutes.
Uses pre-computed article results as reference and runs a minimal live demo.
Designed for Streamlit Cloud deployment.
"""

import sys
import os
import time
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
os.environ['PYTHONUTF8'] = '1'

# Reduce parameters drastically for instant results
INSTANT_CONFIG = {
    "n_bootstrap": 5,       # Article: 50
    "n_fidelity_instances": 15,  # Article: 100
    "n_cost_instances": 10,      # Article: 100
    "n_fidelity_repeats": 1,     # Article: 5
    "n_stability_runs": 1,       # Article: 3
    "n_top_k": 3,                # Article: 5
    # Subsample datasets
    "max_train_samples": 2000,
    "max_test_samples": 500,
    # Only tabular, only fast models
    "datasets": ["adult"],
    "models": ["ridge", "random_forest"],
    "methods": ["shap_tree", "permutation", "ale"],
}


def run_instant_benchmark():
    """Run a minimal benchmark (~2 min) for demonstration purposes."""
    from dataloaders import load_dataset
    from explainers import create_explainer, get_compatible_methods
    from metrics import compute_fidelity, compute_stability, compute_cost
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    from precomputed_results import TABLE_IX
    
    config = INSTANT_CONFIG
    
    all_results = []
    total_start = time.time()
    
    for dataset_name in config["datasets"]:
        print(f"\n{'='*60}")
        print(f"  INSTANT BENCHMARK: {dataset_name}")
        print(f"{'='*60}")
        
        X_train, X_test, y_train, y_test, feature_names, data_type, task_type = load_dataset(dataset_name)
        
        # Subsample for speed
        if len(X_train) > config["max_train_samples"]:
            idx = np.random.RandomState(42).choice(len(X_train), config["max_train_samples"], replace=False)
            X_train, y_train = X_train[idx], y_train[idx]
        if len(X_test) > config["max_test_samples"]:
            idx = np.random.RandomState(43).choice(len(X_test), config["max_test_samples"], replace=False)
            X_test, y_test = X_test[idx], y_test[idx]
        
        print(f"  Subsampled: {X_train.shape[0]} train, {X_test.shape[0]} test")
        
        # Train models
        models = {}
        if "ridge" in config["models"]:
            print("  Training Logistic Regression...")
            lr = LogisticRegression(max_iter=500, random_state=42)
            lr.fit(X_train, y_train)
            models["ridge"] = lr
            print(f"    Accuracy: {lr.score(X_test, y_test):.3f}")
        
        if "random_forest" in config["models"]:
            print("  Training Random Forest...")
            rf = RandomForestClassifier(n_estimators=50, max_depth=8, random_state=42, n_jobs=-1)
            rf.fit(X_train, y_train)
            models["random_forest"] = rf
            print(f"    Accuracy: {rf.score(X_test, y_test):.3f}")
        
        # Evaluate methods
        model_method_map = {
            "ridge": ["shap_kernel", "permutation", "ale"],
            "random_forest": ["shap_tree", "permutation", "ale"],
        }
        
        for model_name, model in models.items():
            methods = [m for m in config["methods"] if m in get_compatible_methods(model_name, data_type)]
            if not methods:
                methods = model_method_map.get(model_name, ["permutation"])[:2]
            
            for method_name in methods:
                print(f"  Evaluating {method_name} on {model_name}...", end=" ", flush=True)
                start = time.time()
                
                try:
                    explainer = create_explainer(
                        method_name, model, X_train[:100],
                        feature_names=feature_names,
                        model_type=data_type, task_type=task_type,
                    )
                    
                    # Fidelity (quick)
                    f_mean, f_std, _ = compute_fidelity(
                        explainer, model, X_test, y_test,
                        k=config["n_top_k"],
                        n_instances=config["n_fidelity_instances"],
                    )
                    
                    # Stability (quick)
                    s_mean, s_std, _ = compute_stability(
                        explainer, model, X_test,
                        n_bootstrap=config["n_bootstrap"],
                    )
                    
                    # Cost (quick)
                    c_mean, c_std, _ = compute_cost(explainer, X_test[:config["n_cost_instances"]])
                    
                    elapsed = time.time() - start
                    
                    result = {
                        "Model": model_name,
                        "Method": method_name,
                        "Fidelity (φ)": round(f_mean, 3),
                        "Fidelity_std": round(f_std, 3),
                        "Stability (τ_R)": round(s_mean, 3),
                        "Stability_std": round(s_std, 3),
                        "Cost (s)": round(c_mean, 4),
                        "Time (s)": round(elapsed, 1),
                    }
                    all_results.append(result)
                    print(f"φ={f_mean:.3f}, τ={s_mean:.3f}, cost={c_mean:.3f}s ({elapsed:.0f}s)")
                    
                except Exception as e:
                    print(f"FAILED: {e}")
                    all_results.append({
                        "Model": model_name,
                        "Method": method_name,
                        "Error": str(e)[:100],
                    })
    
    total_time = time.time() - total_start
    
    df = pd.DataFrame(all_results)
    
    print(f"\n{'='*60}")
    print(f"  INSTANT BENCHMARK COMPLETE in {total_time/60:.1f} min")
    print(f"{'='*60}")
    
    return df


if __name__ == "__main__":
    df = run_instant_benchmark()
    print("\nResults:")
    print(df.to_string())