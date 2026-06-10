#!/usr/bin/env python3
"""
=============================================================================
 ARTICLE 4 — BENCHMARK XAI COMPARAISON
 Run script: orchestrate datasets, models, explainers, metrics
=============================================================================
"""

import os
import sys
import json
import time
import argparse
import numpy as np
import pandas as pd
from pathlib import Path

# Add scripts to path
sys.path.insert(0, str(Path(__file__).parent))

from dataloaders import load_dataset, DATASET_INFO
from explainers import (create_explainer, get_compatible_methods, 
                         EXPLAINER_CLASSES, SCOPE, AGNOSTIC, CAUSAL,
                         SHAPTreeExplainer, SHAPKernelExplainer)
from metrics import compute_fidelity, compute_stability, compute_cost, full_evaluation

# Try importing PyTorch models (optional)
try:
    from pytorch_models import ResNet18Wrapper, LSTMWrapper, MLPWrapper, create_model as create_pytorch_model
    HAS_PYTORCH = True
except ImportError:
    HAS_PYTORCH = False
    print("PyTorch models not available. Install torch to enable CNN/LSTM.")

# Try importing deep explainers (optional)
try:
    from deep_explainers import GradCAMExplainer, LRPExplainer, DeepSHAPExplainer, ImageLIMEExplainer
    HAS_DEEP_EXPLAINERS = True
except ImportError:
    HAS_DEEP_EXPLAINERS = False

BASE_DIR = Path(__file__).parent.parent
RESULTS_DIR = BASE_DIR / "results"
FIGURES_DIR = BASE_DIR / "figures"
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"

N_BOOTSTRAP = 50
RANDOM_SEED = 42


def train_model(model_name, X_train, y_train, task_type="classification"):
    """Train a model and return the fitted estimator."""
    from sklearn.linear_model import LogisticRegression, Ridge
    from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    from sklearn.neural_network import MLPClassifier, MLPRegressor
    from sklearn.preprocessing import StandardScaler
    
    is_classification = task_type == "classification"
    
    models = {
        "ridge": LogisticRegression(max_iter=1000, random_state=RANDOM_SEED) if is_classification 
                 else Ridge(alpha=1.0),
        "decision_tree": DecisionTreeClassifier(max_depth=5, random_state=RANDOM_SEED) if is_classification
                         else DecisionTreeRegressor(max_depth=5, random_state=RANDOM_SEED),
        "random_forest": RandomForestClassifier(n_estimators=100, max_depth=10, random_state=RANDOM_SEED, n_jobs=-1) if is_classification
                        else RandomForestRegressor(n_estimators=100, max_depth=10, random_state=RANDOM_SEED, n_jobs=-1),
        "xgboost": None,  # Import separately to handle missing package
        "lightgbm": None,  # Import separately
        "catboost": None,  # Import separately
        "ebm": None,       # Import separately
        "mlp": MLPClassifier(hidden_layer_sizes=(128, 64, 32), max_iter=200, random_state=RANDOM_SEED) if is_classification
               else MLPRegressor(hidden_layer_sizes=(128, 64, 32), max_iter=200, random_state=RANDOM_SEED),
    }
    
    # Optional imports
    try:
        from xgboost import XGBClassifier, XGBRegressor
        models["xgboost"] = XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.1, random_state=RANDOM_SEED) if is_classification \
                           else XGBRegressor(n_estimators=200, max_depth=6, learning_rate=0.1, random_state=RANDOM_SEED)
    except ImportError:
        print("    ⚠ XGBoost not available")
    
    try:
        from lightgbm import LGBMClassifier, LGBMRegressor
        models["lightgbm"] = LGBMClassifier(n_estimators=200, max_depth=6, learning_rate=0.1, random_state=RANDOM_SEED, verbose=-1) if is_classification \
                            else LGBMRegressor(n_estimators=200, max_depth=6, learning_rate=0.1, random_state=RANDOM_SEED, verbose=-1)
    except ImportError:
        print("    ⚠ LightGBM not available")
    
    try:
        from catboost import CatBoostClassifier, CatBoostRegressor
        models["catboost"] = CatBoostClassifier(iterations=200, depth=6, learning_rate=0.1, random_seed=RANDOM_SEED, verbose=0) if is_classification \
                            else CatBoostRegressor(iterations=200, depth=6, learning_rate=0.1, random_seed=RANDOM_SEED, verbose=0)
    except ImportError:
        print("    ⚠ CatBoost not available")
    
    try:
        from interpret.glassbox import ExplainableBoostingClassifier, ExplainableBoostingRegressor
        models["ebm"] = ExplainableBoostingClassifier(random_state=RANDOM_SEED) if is_classification \
                       else ExplainableBoostingRegressor(random_state=RANDOM_SEED)
    except ImportError:
        print("    ⚠ interpret (EBM) not available")
    
    if model_name not in models or models[model_name] is None:
        raise ValueError(f"Model {model_name} not available. Install required packages.")
    
    model = models[model_name]
    print(f"    Training {model_name}...")
    model.fit(X_train, y_train)
    return model


def run_benchmark(datasets=None, quick=False, n_bootstrap=N_BOOTSTRAP):
    """Run the full benchmark."""
    if datasets is None:
        datasets = list(DATASET_INFO.keys())
    
    if quick:
        datasets = ["adult"]
        model_names = ["ridge", "random_forest"]
        method_names = ["shap_tree", "lime", "permutation"]
        n_bootstrap = 5
    else:
        model_names = ["ridge", "decision_tree", "ebm", "random_forest", 
                      "xgboost", "lightgbm", "catboost", "mlp", 
                      "cnn_resnet18", "lstm"]
        method_names = list(EXPLAINER_CLASSES.keys())
    
    all_results = []
    
    for dataset_name in datasets:
        print(f"\n{'='*70}")
        print(f"  DATASET: {dataset_name} (n={DATASET_INFO[dataset_name]['n']}, "
              f"p={DATASET_INFO[dataset_name]['p']}, "
              f"type={DATASET_INFO[dataset_name]['type']})")
        print(f"{'='*70}")
        
        try:
            X_train, X_test, y_train, y_test, feature_names, data_type, task_type = load_dataset(dataset_name)
        except Exception as e:
            print(f"  ⚠ Skipping {dataset_name}: {e}")
            continue
        
        # Select models for this data type
        if data_type == "tabular":
            current_models = [m for m in model_names if m in 
                           ["ridge", "decision_tree", "ebm", "random_forest", 
                            "xgboost", "lightgbm", "catboost", "mlp"]]
        elif data_type == "image":
            current_models = ["cnn_resnet18"] if HAS_PYTORCH else []
        else:  # timeseries
            current_models = ["ridge", "mlp"]
            if HAS_PYTORCH:
                current_models.append("lstm")
        
        for model_name in current_models:
            print(f"\n  Model: {model_name}")
            
            try:
                model = train_model(model_name, X_train, y_train, task_type)
            except Exception as e:
                print(f"    ⚠ Skipping {model_name}: {e}")
                continue
            
            # Get compatible methods
            compatible_methods = get_compatible_methods(model_name, data_type)
            
            for method_name in compatible_methods:
                if quick and method_name not in method_names:
                    continue
                
                print(f"    {method_name}...", end=" ", flush=True)
                start_time = time.time()
                
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
                        n_bootstrap=n_bootstrap
                    )
                    result["time_seconds"] = time.time() - start_time
                    all_results.append(result)
                    phi_str = f"{result['fidelity_mean']:.2f}" if result.get('fidelity_mean') is not None else "N/A"
                    tau_str = f"{result['stability_mean']:.2f}" if result.get('stability_mean') is not None else "N/A"
                    cost_str = f"{result['cost_mean']:.3f}" if result.get('cost_mean') is not None else "N/A"
                    print(f"phi={phi_str}, tau={tau_str}, "
                          f"cost={cost_str}s ({result['time_seconds']:.1f}s total)")
                    
                except Exception as e:
                    print(f"FAILED: {e}")
                    all_results.append({
                        "method": method_name, "model": model_name,
                        "dataset": dataset_name, "error": str(e)
                    })
    
    # Save results
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(all_results)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    
    csv_path = RESULTS_DIR / f"benchmark_results_{timestamp}.csv"
    json_path = RESULTS_DIR / f"benchmark_results_{timestamp}.json"
    
    df.to_csv(csv_path, index=False)
    df.to_json(json_path, orient="records", indent=2)
    
    print(f"\n{'='*70}")
    print(f"  RESULTS SAVED")
    print(f"  CSV: {csv_path}")
    print(f"  JSON: {json_path}")
    print(f"  {len(all_results)} evaluations completed")
    print(f"{'='*70}")
    
    # Print summary
    print("\n  SUMMARY TABLE:")
    print("-" * 80)
    if len(df) > 0 and 'fidelity_mean' in df.columns:
        pivot = df.pivot_table(
            values=['fidelity_mean', 'stability_mean', 'cost_mean'],
            index='method', columns='dataset', aggfunc='mean'
        )
        print(pivot.to_string())
    
    return df


def generate_figures_from_results(results_csv=None):
    """Generate figures from existing results."""
    from generate_figures import generate_all_figures, generate_latex_table_ix
    
    if results_csv:
        print(f"Loading results from {results_csv}")
    
    generate_all_figures()
    generate_latex_table_ix()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="XAI Benchmark for Article 4: Comparative Evaluation"
    )
    parser.add_argument("--quick", action="store_true", 
                        help="Quick test: 1 dataset, 2 models, 3 methods, 5 bootstrap")
    parser.add_argument("--tabular", action="store_true", 
                        help="Tabular datasets only")
    parser.add_argument("--image", action="store_true", 
                        help="Image datasets only")
    parser.add_argument("--timeseries", action="store_true", 
                        help="Time-series datasets only")
    parser.add_argument("--figures", action="store_true", 
                        help="Generate figures from article data (no benchmark needed)")
    parser.add_argument("--from-csv", type=str, default=None,
                        help="Load results from CSV and generate figures")
    args = parser.parse_args()
    
    if args.figures:
        generate_figures_from_results(args.from_csv)
    else:
        datasets = None
        if args.tabular:
            datasets = ["adult", "german", "california"]
        elif args.image:
            datasets = ["cifar10", "isic2019"]
        elif args.timeseries:
            datasets = ["physionet", "electricity"]
        
        df = run_benchmark(datasets=datasets, quick=args.quick)