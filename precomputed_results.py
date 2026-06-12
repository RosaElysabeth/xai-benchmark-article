#!/usr/bin/env python3
"""
Pre-computed results from Article 4.
These are the reference values used for instant visualization without running the benchmark.
Source: Article 4 — "A Comparative Evaluation of Post-Hoc Explanation Methods"
"""

import pandas as pd
import numpy as np

# ========================================================================
# TABLE IX — Main comparative evaluation (7 datasets × 10 models)
# ========================================================================

TABLE_IX = {
    "TreeExplainer": {
        "fidelity": (0.98, 0.99), "stability": (1.00, 1.00), "cost": (0.3, 0.8),
        "scope": "Both", "agnostic": False, "causal": False, "certifiable": "Yes",
    },
    "KernelSHAP": {
        "fidelity": (0.76, 0.88), "stability": (0.71, 0.85), "cost": (45, 280),
        "scope": "Both", "agnostic": True, "causal": False, "certifiable": "Limited",
    },
    "DeepSHAP": {
        "fidelity": (0.69, 0.74), "stability": (0.62, 0.68), "cost": (8, 15),
        "scope": "Both", "agnostic": False, "causal": False, "certifiable": "No",
    },
    "LIME": {
        "fidelity": (0.58, 0.78), "stability": (0.39, 0.68), "cost": (0.1, 0.5),
        "scope": "Local", "agnostic": True, "causal": False, "certifiable": "No",
    },
    "Perm. Imp.": {
        "fidelity": (0.70, 0.88), "stability": (0.91, 0.98), "cost": (0.01, 0.1),
        "scope": "Global", "agnostic": True, "causal": False, "certifiable": "Yes",
    },
    "PDP": {
        "fidelity": (0.55, 0.75), "stability": (0.95, 0.99), "cost": (0.1, 1.0),
        "scope": "Global", "agnostic": True, "causal": False, "certifiable": "Limited",
    },
    "ALE": {
        "fidelity": (0.68, 0.85), "stability": (0.92, 0.99), "cost": (0.1, 0.4),
        "scope": "Global", "agnostic": True, "causal": False, "certifiable": "Limited",
    },
    "LRP": {
        "fidelity": (0.55, 0.72), "stability": (0.58, 0.81), "cost": (0.1, 0.5),
        "scope": "Local", "agnostic": False, "causal": False, "certifiable": "No",
    },
    "GradCAM": {
        "fidelity": (0.45, 0.65), "stability": (0.52, 0.74), "cost": (0.01, 0.1),
        "scope": "Local", "agnostic": False, "causal": False, "certifiable": "No",
    },
    "Counterfactual": {
        "fidelity": (0.40, 0.70), "stability": (0.35, 0.65), "cost": (0.5, 5.0),
        "scope": "Local", "agnostic": True, "causal": "Partial", "certifiable": "No",
    },
}

# ========================================================================
# TABLE II — Adult Census per-method detailed results
# ========================================================================

TABLE_II_ADULT = {
    "TreeExplainer":  {"fidelity": (0.97, 0.99), "stability": (0.99, 1.00), "cost": (0.3, 0.8), "scope": "Both", "certifiable": "Yes"},
    "KernelSHAP":     {"fidelity": (0.76, 0.88), "stability": (0.71, 0.85), "cost": (45, 280),  "scope": "Both", "certifiable": "Limited"},
    "DeepSHAP":       {"fidelity": (0.69, 0.74), "stability": (0.62, 0.68), "cost": (8, 15),    "scope": "Both", "certifiable": "No"},
    "LIME":           {"fidelity": (0.62, 0.75), "stability": (0.42, 0.65), "cost": (0.1, 0.5), "scope": "Local", "certifiable": "No"},
    "Perm. Imp.":     {"fidelity": (0.78, 0.86), "stability": (0.93, 0.97), "cost": (0.01, 0.1),"scope": "Global", "certifiable": "Yes"},
    "PDP":            {"fidelity": (0.55, 0.70), "stability": (0.95, 0.99), "cost": (0.2, 1.0), "scope": "Global", "certifiable": "Limited"},
    "ALE":            {"fidelity": (0.72, 0.83), "stability": (0.93, 0.98), "cost": (0.1, 0.4), "scope": "Global", "certifiable": "Limited"},
    "Counterfactual": {"fidelity": (0.45, 0.68), "stability": (0.38, 0.60), "cost": (0.5, 5.0), "scope": "Local", "certifiable": "No"},
}

# ========================================================================
# PER-DATASET RESULTS (Tables III–VIII)
# ========================================================================

PER_DATASET = {
    "Adult Census": {
        "n": 48842, "p": 14, "type": "Tabular", "task": "Classification",
        "models": ["Logistic/Ridge", "Decision Tree", "EBM", "Random Forest", "XGBoost", "LightGBM", "CatBoost", "MLP"],
        "results": TABLE_II_ADULT,
    },
    "German Credit": {
        "n": 1000, "p": 20, "type": "Tabular", "task": "Classification",
        "models": ["Logistic/Ridge", "Decision Tree", "Random Forest", "XGBoost", "MLP"],
        "results": {
            "TreeExplainer":  {"fidelity": (0.95, 0.97), "stability": (0.98, 1.00), "cost": (0.1, 0.3), "scope": "Both", "certifiable": "Yes"},
            "KernelSHAP":     {"fidelity": (0.78, 0.86), "stability": (0.75, 0.83), "cost": (30, 180), "scope": "Both", "certifiable": "Limited"},
            "LIME":           {"fidelity": (0.65, 0.78), "stability": (0.45, 0.62), "cost": (0.1, 0.4), "scope": "Local", "certifiable": "No"},
            "Perm. Imp.":     {"fidelity": (0.72, 0.85), "stability": (0.91, 0.96), "cost": (0.01, 0.08),"scope": "Global", "certifiable": "Yes"},
            "PDP":            {"fidelity": (0.58, 0.72), "stability": (0.94, 0.98), "cost": (0.1, 0.8), "scope": "Global", "certifiable": "Limited"},
            "ALE":            {"fidelity": (0.70, 0.82), "stability": (0.92, 0.97), "cost": (0.08, 0.3), "scope": "Global", "certifiable": "Limited"},
        },
    },
    "California Housing": {
        "n": 20640, "p": 8, "type": "Tabular", "task": "Regression",
        "models": ["Ridge", "Decision Tree", "Random Forest", "XGBoost", "LightGBM", "MLP"],
        "results": {
            "TreeExplainer":  {"fidelity": (0.98, 1.00), "stability": (1.00, 1.00), "cost": (0.4, 0.9), "scope": "Both", "certifiable": "Yes"},
            "KernelSHAP":     {"fidelity": (0.80, 0.88), "stability": (0.73, 0.85), "cost": (50, 300), "scope": "Both", "certifiable": "Limited"},
            "LIME":           {"fidelity": (0.58, 0.78), "stability": (0.39, 0.65), "cost": (0.1, 0.5), "scope": "Local", "certifiable": "No"},
            "Perm. Imp.":     {"fidelity": (0.70, 0.88), "stability": (0.92, 0.98), "cost": (0.01, 0.1),"scope": "Global", "certifiable": "Yes"},
            "PDP":            {"fidelity": (0.55, 0.75), "stability": (0.96, 0.99), "cost": (0.1, 1.0), "scope": "Global", "certifiable": "Limited"},
            "ALE":            {"fidelity": (0.68, 0.85), "stability": (0.93, 0.99), "cost": (0.1, 0.4), "scope": "Global", "certifiable": "Limited"},
        },
    },
    "CIFAR-10": {
        "n": 60000, "p": "3×32×32", "type": "Image", "task": "Classification",
        "models": ["ResNet-18"],
        "results": {
            "DeepSHAP":  {"fidelity": (0.69, 0.74), "stability": (0.62, 0.68), "cost": (8, 15), "scope": "Both", "certifiable": "No"},
            "GradCAM":   {"fidelity": (0.45, 0.65), "stability": (0.52, 0.74), "cost": (0.01, 0.1),"scope": "Local", "certifiable": "No"},
            "LRP":       {"fidelity": (0.55, 0.72), "stability": (0.58, 0.81), "cost": (0.1, 0.5), "scope": "Local", "certifiable": "No"},
            "LIME":      {"fidelity": (0.50, 0.68), "stability": (0.35, 0.55), "cost": (0.5, 2.0), "scope": "Local", "certifiable": "No"},
        },
    },
    "ISIC 2019": {
        "n": 25331, "p": "3×224×224", "type": "Image", "task": "Classification",
        "models": ["ResNet-18"],
        "results": {
            "DeepSHAP":  {"fidelity": (0.67, 0.72), "stability": (0.60, 0.66), "cost": (10, 20), "scope": "Both", "certifiable": "No"},
            "GradCAM":   {"fidelity": (0.48, 0.63), "stability": (0.55, 0.72), "cost": (0.02, 0.15),"scope": "Local", "certifiable": "No"},
            "LRP":       {"fidelity": (0.52, 0.68), "stability": (0.56, 0.78), "cost": (0.1, 0.6), "scope": "Local", "certifiable": "No"},
        },
    },
    "PhysioNet ECG": {
        "n": 40336, "p": 40, "type": "Time-Series", "task": "Classification",
        "models": ["LSTM", "Ridge", "MLP"],
        "results": {
            "KernelSHAP":     {"fidelity": (0.74, 0.84), "stability": (0.68, 0.80), "cost": (40, 250), "scope": "Both", "certifiable": "Limited"},
            "LIME":           {"fidelity": (0.55, 0.72), "stability": (0.38, 0.58), "cost": (0.2, 0.8), "scope": "Local", "certifiable": "No"},
            "Perm. Imp.":     {"fidelity": (0.65, 0.82), "stability": (0.89, 0.95), "cost": (0.01, 0.05),"scope": "Global", "certifiable": "Yes"},
            "ALE":            {"fidelity": (0.63, 0.80), "stability": (0.90, 0.96), "cost": (0.1, 0.3), "scope": "Global", "certifiable": "Limited"},
        },
    },
    "Electricity": {
        "n": 45312, "p": 12, "type": "Time-Series", "task": "Classification",
        "models": ["LSTM", "Ridge", "MLP"],
        "results": {
            "KernelSHAP":     {"fidelity": (0.76, 0.86), "stability": (0.70, 0.82), "cost": (45, 280), "scope": "Both", "certifiable": "Limited"},
            "LIME":           {"fidelity": (0.58, 0.76), "stability": (0.40, 0.62), "cost": (0.1, 0.5), "scope": "Local", "certifiable": "No"},
            "Perm. Imp.":     {"fidelity": (0.68, 0.86), "stability": (0.90, 0.96), "cost": (0.01, 0.08),"scope": "Global", "certifiable": "Yes"},
            "PDP":            {"fidelity": (0.52, 0.72), "stability": (0.94, 0.98), "cost": (0.1, 0.8), "scope": "Global", "certifiable": "Limited"},
            "ALE":            {"fidelity": (0.66, 0.84), "stability": (0.91, 0.97), "cost": (0.08, 0.3), "scope": "Global", "certifiable": "Limited"},
        },
    },
}

# ========================================================================
# CERTIFICATION THRESHOLDS
# ========================================================================

CERTIFICATION = {
    "fidelity": {"threshold": 0.85, "label_en": "φ ≥ 0.85", "label_fr": "φ ≥ 0,85"},
    "stability": {"threshold": 0.90, "label_en": "τ_R ≥ 0.90", "label_fr": "τ_R ≥ 0,90"},
    "triangulation": {"threshold": 0.80, "label_en": "τ_tri ≥ 0.80", "label_fr": "τ_tri ≥ 0,80"},
}

# ========================================================================
# TRIANGULATION RESULTS
# ========================================================================

TRIANGULATION = {
    "KernelSHAP + ALE + Perm. Imp.": {
        "tau_tri": (0.80, 0.88),
        "datasets": ["Adult Census", "German Credit", "California Housing"],
        "certifiable": True,
    },
    "KernelSHAP + LIME + Perm. Imp.": {
        "tau_tri": (0.72, 0.82),
        "datasets": ["Adult Census", "German Credit"],
        "certifiable": "Conditional",
    },
    "TreeExplainer + Perm. Imp.": {
        "tau_tri": (0.95, 0.99),
        "datasets": ["Adult Census", "German Credit", "California Housing"],
        "certifiable": True,
    },
}

# ========================================================================
# METHOD DETAILS
# ========================================================================

METHOD_DETAILS = {
    "TreeExplainer": {
        "name_en": "SHAP TreeExplainer",
        "name_fr": "SHAP TreeExplainer",
        "family": "SHAP",
        "year": 2017,
        "ref": "Lundberg & Lee (2017)",
        "complexity": "O(TL) for trees",
        "description_en": "Exact Shapley values for tree-based models. Polynomial-time computation via tree recursion. Only applicable to decision tree ensembles (RF, XGBoost, LightGBM, CatBoost).",
        "description_fr": "Valeurs de Shapley exactes pour les modèles à base d'arbres. Calcul en temps polynomial par récursion arborescente. Applicable uniquement aux ensembles d'arbres de décision (RF, XGBoost, LightGBM, CatBoost).",
    },
    "KernelSHAP": {
        "name_en": "SHAP KernelExplainer",
        "name_fr": "SHAP KernelExplainer",
        "family": "SHAP",
        "year": 2017,
        "ref": "Lundberg & Lee (2017)",
        "complexity": "O(2^n × M) (exponential)",
        "description_en": "Model-agnostic approximation of Shapley values using a weighted linear regression. Exponentially expensive but applicable to any model. High variance with few samples.",
        "description_fr": "Approximation modèle-agnostique des valeurs de Shapley par régression linéaire pondérée. Exponentiellement coûteux mais applicable à tout modèle. Forte variance avec peu d'échantillons.",
    },
    "DeepSHAP": {
        "name_en": "SHAP DeepExplainer",
        "name_fr": "SHAP DeepExplainer",
        "family": "SHAP",
        "year": 2017,
        "ref": "Lundberg & Lee (2017)",
        "complexity": "O(n × L) per instance",
        "description_en": "Gradient-based approximation of Shapley values for deep neural networks. Uses backpropagation for efficient computation. Model-specific to neural networks.",
        "description_fr": "Approximation par gradient des valeurs de Shapley pour les réseaux de neurones profonds. Utilise la rétropropagation pour un calcul efficace. Spécifique aux réseaux de neurones.",
    },
    "LIME": {
        "name_en": "LIME",
        "name_fr": "LIME",
        "family": "Local Surrogate",
        "year": 2016,
        "ref": "Ribeiro et al. (2016)",
        "complexity": "O(N × k) per instance",
        "description_en": "Local Interpretable Model-agnostic Explanations. Fits a local linear surrogate to perturbed samples. High instability across runs due to random sampling.",
        "description_fr": "Explications Locales Interprétables Modèle-agnostiques. Ajuste un substitut linéaire local aux échantillons perturbés. Instabilité élevée entre les exécutions due à l'échantillonnage aléatoire.",
    },
    "Perm. Imp.": {
        "name_en": "Permutation Importance",
        "name_fr": "Importance par Permutation",
        "family": "Global Feature Importance",
        "year": 2001,
        "ref": "Breiman (2001)",
        "complexity": "O(p × M) for global",
        "description_en": "Measures feature importance by permuting each feature and measuring performance drop. Global, deterministic, fast. Gold standard for global explanations.",
        "description_fr": "Mesure l'importance des caractéristiques en permutant chaque variable et en mesurant la baisse de performance. Global, déterministe, rapide. Référence pour les explications globales.",
    },
    "PDP": {
        "name_en": "Partial Dependence Plots",
        "name_fr": "Graphiques de Dépendance Partielle",
        "family": "Global Effect",
        "year": 2001,
        "ref": "Friedman (2001)",
        "complexity": "O(p × grid × M)",
        "description_en": "Shows the marginal effect of features on prediction. Global, model-agnostic, but assumes feature independence. Can be misleading with correlated features.",
        "description_fr": "Montre l'effet marginal des caractéristiques sur la prédiction. Global, modèle-agnostique, mais suppose l'indépendance des variables. Peut être trompeur avec des variables corrélées.",
    },
    "ALE": {
        "name_en": "Accumulated Local Effects",
        "name_fr": "Effets Locaux Accumulés",
        "family": "Global Effect",
        "year": 2020,
        "ref": "Apley & Zhu (2020)",
        "complexity": "O(p × n × log(n))",
        "description_en": "Accumulated Local Effects: unbiased alternative to PDP that handles correlated features. Global, model-agnostic, computationally efficient. Second-best method after TreeExplainer for stability.",
        "description_fr": "Effets Locaux Accumulés : alternative non biaisée au PDP qui gère les variables corrélées. Global, modèle-agnostique, efficace. Deuxième meilleure méthode après TreeExplainer pour la stabilité.",
    },
    "LRP": {
        "name_en": "Layer-Wise Relevance Propagation",
        "name_fr": "Propagation de Pertinence par Couches",
        "family": "Gradient-Based",
        "year": 2015,
        "ref": "Bach et al. (2015)",
        "complexity": "O(n × L) per instance",
        "description_en": "Propagates prediction relevance backward through neural network layers. Model-specific, local. Moderate fidelity but suffers from instability across similar inputs.",
        "description_fr": "Propage la pertinence de la prédiction en arrière à travers les couches du réseau de neurones. Spécifique au modèle, local. Fidélité modérée mais instabilité entre entrées similaires.",
    },
    "GradCAM": {
        "name_en": "GradCAM",
        "name_fr": "GradCAM",
        "family": "Gradient-Based",
        "year": 2020,
        "ref": "Selvaraju et al. (2020)",
        "complexity": "O(1) per instance (one forward+backward)",
        "description_en": "Gradient-weighted Class Activation Mapping for CNNs. Uses gradient of target class flowing into final conv layer. Very fast but low fidelity for detailed explanations.",
        "description_fr": "Carte d'Activation de Classe pondérée par gradient pour CNN. Utilise le gradient de la classe cible vers la dernière couche convolutive. Très rapide mais faible fidélité pour les explications détaillées.",
    },
    "Counterfactual": {
        "name_en": "Counterfactual Explanations",
        "name_fr": "Explications Contrefactuelles",
        "family": "Counterfactual",
        "year": 2020,
        "ref": "Wachter et al. (2018)",
        "complexity": "O(p × δ_search) per instance",
        "description_en": "Finds minimal perturbations that change the prediction. Intuitive for humans, model-agnostic, but highly unstable. Only provides local explanations.",
        "description_fr": "Trouve les perturbations minimales qui changent la prédiction. Intuitif pour les humains, modèle-agnostique, mais très instable. Fournit uniquement des explications locales.",
    },
}

# ========================================================================
# HELPER: Convert TABLE_IX to DataFrame
# ========================================================================

def get_table_ix_df():
    """Return TABLE_IX as a pandas DataFrame."""
    rows = []
    for method, data in TABLE_IX.items():
        rows.append({
            "Method": method,
            "Fidelity (φ)": f"{data['fidelity'][0]:.2f}–{data['fidelity'][1]:.2f}",
            "Fidelity_mean": np.mean(data["fidelity"]),
            "Stability (τ_R)": f"{data['stability'][0]:.2f}–{data['stability'][1]:.2f}",
            "Stability_mean": np.mean(data["stability"]),
            "Cost (s)": f"{data['cost'][0]:.2f}–{data['cost'][1]:.1f}",
            "Cost_mean": np.mean(data["cost"]),
            "Scope": data["scope"],
            "Agnostic": "Yes" if data["agnostic"] else "No",
            "Causal": data["causal"] if isinstance(data["causal"], str) else ("Yes" if data["causal"] else "No"),
            "Certifiable": data["certifiable"],
        })
    return pd.DataFrame(rows)


def get_per_dataset_df(dataset_name):
    """Return per-dataset results as DataFrame."""
    ds = PER_DATASET.get(dataset_name, {})
    results = ds.get("results", {})
    rows = []
    for method, data in results.items():
        rows.append({
            "Method": method,
            "Fidelity (φ)": f"{data['fidelity'][0]:.2f}–{data['fidelity'][1]:.2f}",
            "Fidelity_mean": np.mean(data["fidelity"]),
            "Stability (τ_R)": f"{data['stability'][0]:.2f}–{data['stability'][1]:.2f}",
            "Stability_mean": np.mean(data["stability"]),
            "Cost (s)": f"{data['cost'][0]:.2f}–{data['cost'][1]:.1f}",
            "Cost_mean": np.mean(data["cost"]),
            "Scope": data["scope"],
            "Certifiable": data["certifiable"],
        })
    return pd.DataFrame(rows)