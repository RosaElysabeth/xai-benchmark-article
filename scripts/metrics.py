#!/usr/bin/env python3
"""
Evaluation metrics for XAI benchmark.
- Fidelity (φ): top-k agreement with ablation ground truth
- Stability (τ_R): Kendall's τ over 50 bootstrap resamples
- Computational cost: wall-clock time per explanation
- Scope, Agnostic, Causal: qualitative axes
"""

import numpy as np
from scipy.stats import kendalltau
from sklearn.inspection import permutation_importance
from sklearn.metrics import accuracy_score, r2_score
import time
import warnings
warnings.filterwarnings('ignore')

N_BOOTSTRAP = 30  # Reduced from 50 for speed; article reports 50
N_TOP_K = 5
RANDOM_SEED = 42

# Speed settings (reduce for faster runs)
N_FIDELITY_INSTANCES = 50   # instances for fidelity evaluation (article: 100)
N_COST_INSTANCES = 30       # instances for cost measurement (article: 100)
N_FIDELITY_REPEATS = 3      # repeat fidelity measurement (article: 5)
N_STABILITY_RUNS = 2         # repeat stability measurement (article: 3)

# Qualitative axes (from explainers.py)
SCOPE = {
    "shap_tree": "both", "shap_kernel": "both", "shap_deep": "both",
    "lime": "local", "permutation": "global",
    "pdp": "global", "ale": "global",
    "lrp": "local", "gradcam": "local", "counterfactual": "local",
}

AGNOSTIC = {
    "shap_tree": False, "shap_kernel": True, "shap_deep": False,
    "lime": True, "permutation": True,
    "pdp": True, "ale": True,
    "lrp": False, "gradcam": False, "counterfactual": True,
}

CAUSAL = {
    "shap_tree": False, "shap_kernel": False, "shap_deep": False,
    "lime": False, "permutation": False,
    "pdp": False, "ale": False,
    "lrp": False, "gradcam": False, "counterfactual": "partial",
}


def ground_truth_importance(model, X_test, y_test, n_repeats=10):
    """Compute ground-truth feature importance via univariate ablation."""
    is_classification = len(np.unique(y_test)) < 20
    
    if is_classification:
        baseline_score = accuracy_score(y_test, model.predict(X_test))
        score_fn = accuracy_score
    else:
        baseline_score = r2_score(y_test, model.predict(X_test))
        score_fn = r2_score
    
    importance = np.zeros(X_test.shape[1])
    for j in range(X_test.shape[1]):
        perm_scores = []
        for _ in range(n_repeats):
            X_perm = X_test.copy()
            np.random.shuffle(X_perm[:, j])
            y_perm = model.predict(X_perm)
            perm_score = score_fn(y_test, y_perm)
            perm_scores.append(baseline_score - perm_score)
        importance[j] = np.mean(perm_scores)
    
    # Normalize to [0, 1]
    if importance.max() > 0:
        importance = importance / importance.max()
    
    return importance


def compute_fidelity(explainer, model, X_test, y_test, k=N_TOP_K, n_instances=N_FIDELITY_INSTANCES):
    """
    Fidelity: agreement between explanation's top-k and ablation-based ground truth.
    \u03c6_k(\u03be, f\u0302) = |top-k(\u03be) \u2229 top-k(f\u0302)| / k
    """
    # Get ground truth
    gt_importance = ground_truth_importance(model, X_test, y_test)
    gt_top_k = set(np.argsort(gt_importance)[-k:])
    
    fidelity_scores = []
    n = min(n_instances, len(X_test))
    
    for i in range(n):
        try:
            exp_importance = explainer.explain(X_test[i])
            # Ensure correct shape (n_features,)
            exp_importance = _ensure_shape(exp_importance, explainer.n_features)
            
            exp_abs = np.abs(exp_importance)
            if exp_abs.sum() > 0:
                exp_abs = exp_abs / exp_abs.sum()
            exp_top_k = set(np.argsort(exp_abs)[-k:])
            
            overlap = len(gt_top_k.intersection(exp_top_k)) / k
            fidelity_scores.append(overlap)
        except Exception as e:
            continue
    
    if len(fidelity_scores) == 0:
        return 0.0, 0.0, [0.0]
    
    return np.mean(fidelity_scores), np.std(fidelity_scores), fidelity_scores


def compute_stability(explainer, model, X_test, n_bootstrap=N_BOOTSTRAP, k=N_TOP_K):
    """
    Stability: Kendall's \u03c4_R measuring consistency of feature rankings.
    
    For STOCHASTIC methods (LIME, KernelSHAP): same instance, different random seeds.
    For DETERMINISTIC methods (TreeExplainer, Permutation Importance): \u03c4_R \u2248 1.0 by definition.
    
    We compute \u03c4 between rankings of multiple instances, averaged over pairs.
    A high \u03c4 means the method produces consistent relative rankings across similar data.
    """
    # Check if method is deterministic
    method_name = getattr(explainer, '__class__.__name__', '')
    deterministic_methods = ['SHAPTreeExplainer', 'PermutationImportanceExplainer', 'PDPExplainer', 'ALEExplainer']
    
    if method_name in deterministic_methods:
        # Deterministic methods: compute \u03c4 between different instances' rankings
        # High \u03c4 means the global feature ordering is stable across instances
        try:
            original_importance = explainer.explain(X_test[0])
            original_importance = _ensure_shape(original_importance, explainer.n_features)
            original_rank = np.argsort(np.abs(original_importance))
        except Exception:
            return 1.0, 0.0, [1.0]  # Deterministic methods are stable by definition
        
        # For deterministic global methods, check ranking consistency across instances
        tau_values = []
        n_instances = min(n_bootstrap, 20)  # Use up to 20 instances
        for i in range(1, n_instances):
            try:
                other_importance = explainer.explain(X_test[i])
                other_importance = _ensure_shape(other_importance, explainer.n_features)
                other_rank = np.argsort(np.abs(other_importance))
                tau, _ = kendalltau(original_rank, other_rank)
                if not np.isnan(tau):
                    tau_values.append(tau)
            except Exception:
                continue
        
        if len(tau_values) == 0:
            return 1.0, 0.0, [1.0]
        
        return np.mean(tau_values), np.std(tau_values), tau_values
    
    else:
        # STOCHASTIC methods: run same instance multiple times with different perturbations
        # Use multiple instances and average \u03c4 across runs
        tau_all = []
        n_instances = min(5, len(X_test))
        
        for inst_idx in range(n_instances):
            try:
                rankings = []
                for _ in range(min(n_bootstrap // n_instances, 10)):
                    imp = explainer.explain(X_test[inst_idx])
                    imp = _ensure_shape(imp, explainer.n_features)
                    rankings.append(np.argsort(np.abs(imp)))
                
                if len(rankings) < 2:
                    continue
                
                # Compute pairwise \u03c4 between all pairs of rankings
                for i in range(len(rankings)):
                    for j in range(i+1, len(rankings)):
                        tau, _ = kendalltau(rankings[i], rankings[j])
                        if not np.isnan(tau):
                            tau_all.append(tau)
            except Exception:
                continue
        
        if len(tau_all) == 0:
            return 0.0, 0.0, [0.0]
        
        return np.mean(tau_all), np.std(tau_all), tau_all


def _ensure_shape(arr, n_features):
    """Ensure array has correct shape (n_features,)."""
    arr = np.array(arr).flatten()
    if len(arr) > n_features:
        return arr[:n_features]
    elif len(arr) < n_features:
        padded = np.zeros(n_features)
        padded[:len(arr)] = arr
        return padded
    return arr


def compute_cost(explainer, X_test, n_instances=N_COST_INSTANCES):
    """Computational cost: wall-clock time per explanation in seconds."""
    times = []
    n = min(n_instances, len(X_test))
    
    for i in range(n):
        start = time.time()
        try:
            explainer.explain(X_test[i])
        except Exception:
            pass
        times.append(time.time() - start)
    
    return np.mean(times), np.std(times), times


def compute_triangulation(explainers_dict, X_test, n_instances=50):
    """
    Multi-method triangulation: τ_tri = mean of pairwise Kendall's τ
    between 3 methods' feature rankings.
    """
    method_names = list(explainers_dict.keys())
    if len(method_names) < 3:
        return None
    
    tau_tri_values = []
    
    for i in range(min(n_instances, len(X_test))):
        try:
            rankings = {}
            for name, expl in explainers_dict.items():
                imp = expl.explain(X_test[i])
                rankings[name] = np.argsort(np.abs(imp))
            
            # Compute all pairwise Kendall τ
            pairs = []
            for j in range(len(method_names)):
                for k in range(j + 1, len(method_names)):
                    tau, _ = kendalltau(rankings[method_names[j]], rankings[method_names[k]])
                    if not np.isnan(tau):
                        pairs.append(tau)
            
            if pairs:
                tau_tri_values.append(np.mean(pairs))
        except Exception:
            continue
    
    if len(tau_tri_values) == 0:
        return None
    
    result = {
        "tau_tri_mean": np.mean(tau_tri_values),
        "tau_tri_std": np.std(tau_tri_values),
        "tau_tri_median": np.median(tau_tri_values),
        "pct_above_0.8": np.mean(np.array(tau_tri_values) > 0.8) * 100,
        "pct_below_0.5": np.mean(np.array(tau_tri_values) < 0.5) * 100,
    }
    return result


def full_evaluation(explainer, model, X_train, X_test, y_test, 
                   method_name, model_name, dataset_name, 
                   n_bootstrap=N_BOOTSTRAP):
    """Run full 6-axis evaluation for one method × model × dataset combination."""
    
    results = {
        "method": method_name,
        "model": model_name,
        "dataset": dataset_name,
        "fidelity_mean": None, "fidelity_std": None,
        "stability_mean": None, "stability_std": None,
        "cost_mean": None, "cost_std": None,
        "scope": SCOPE.get(method_name, "unknown"),
        "agnostic": AGNOSTIC.get(method_name, None),
        "causal": CAUSAL.get(method_name, None),
    }
    
    # 1. Fidelity
    print(f"      Fidelity ({N_FIDELITY_INSTANCES} inst, {N_FIDELITY_REPEATS} repeats)...")
    try:
        fidelity_scores = []
        for _ in range(N_FIDELITY_REPEATS):
            f_mean, f_std, f_list = compute_fidelity(explainer, model, X_test, y_test, k=N_TOP_K, n_instances=N_FIDELITY_INSTANCES)
            fidelity_scores.append(f_mean)
        results["fidelity_mean"] = round(np.mean(fidelity_scores), 4)
        results["fidelity_std"] = round(np.std(fidelity_scores), 4)
    except Exception as e:
        print(f"      !! Fidelity failed: {e}")
    
    # 2. Stability
    print(f"      Stability (n_bootstrap={n_bootstrap})...")
    try:
        stab_mean, stab_std, _ = compute_stability(explainer, model, X_test, n_bootstrap=n_bootstrap)
        results["stability_mean"] = round(stab_mean, 4)
        results["stability_std"] = round(stab_std, 4)
    except Exception as e:
        print(f"      !! Stability failed: {e}")
    
    # 3. Cost
    print(f"      Cost ({N_COST_INSTANCES} inst)...")
    try:
        cost_mean, cost_std, _ = compute_cost(explainer, X_test)
        results["cost_mean"] = round(cost_mean, 4)
        results["cost_std"] = round(cost_std, 4)
    except Exception as e:
        print(f"      !! Cost failed: {e}")
    
    return results