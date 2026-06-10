#!/usr/bin/env python3
"""
Translations for the Streamlit app (French / English).
"""

TRANSLATIONS = {
    # --- App title & navigation ---
    "app_title": {
        "en": "XAI Benchmark — Article 4",
        "fr": "Benchmark XAI — Article 4",
    },
    "app_subtitle": {
        "en": "A Comparative Evaluation of Post-Hoc Explanation Methods for Predictive Modeling",
        "fr": "Évaluation Comparative des Méthodes d'Explication Post-Hoc pour la Modélisation Prédictive",
    },
    "nav_overview": {"en": "📊 Overview", "fr": "📊 Vue d'ensemble"},
    "nav_table_ix": {"en": "📋 Table IX", "fr": "📋 Tableau IX"},
    "nav_radar": {"en": "🎯 Radar Chart", "fr": "🎯 Radar"},
    "nav_fidelity": {"en": "📈 Fidelity", "fr": "📈 Fidélité"},
    "nav_stability": {"en": "📐 Stability", "fr": "📐 Stabilité"},
    "nav_per_dataset": {"en": "🗂️ Per Dataset", "fr": "🗂️ Par Dataset"},
    "nav_decision": {"en": "🌳 Decision Framework", "fr": "🌳 Cadre Décisionnel"},
    "nav_benchmark": {"en": "⚡ Live Benchmark", "fr": "⚡ Benchmark en Direct"},
    "nav_method": {"en": "📖 Method Details", "fr": "📖 Détails des Méthodes"},
    "nav_article": {"en": "📄 Article", "fr": "📄 Article"},
    
    # --- Overview page ---
    "overview_title": {"en": "Overview", "fr": "Vue d'ensemble"},
    "overview_intro": {
        "en": "This dashboard presents the results of **Article 4**: a systematic comparison of **10 post-hoc explanation methods** across **7 datasets** and **9 predictive models**, evaluated on **6 axes**: Fidelity, Stability, Cost, Scope, Agnosticité, and Causality.",
        "fr": "Ce tableau de bord présente les résultats de l'**Article 4** : une comparaison systématique de **10 méthodes d'explication post-hoc** sur **7 datasets** et **9 modèles prédictifs**, évaluées sur **6 axes** : Fidélité, Stabilité, Coût, Portée, Agnosticité et Causalité.",
    },
    "overview_methods": {"en": "10 Explanation Methods", "fr": "10 Méthodes d'Explication"},
    "overview_datasets": {"en": "7 Datasets", "fr": "7 Datasets"},
    "overview_models": {"en": "9 Models", "fr": "9 Modèles"},
    "overview_axes": {"en": "6 Evaluation Axes", "fr": "6 Axes d'Évaluation"},
    "overview_certifiable_title": {"en": "Certifiable Methods", "fr": "Méthodes Certifiables"},
    "overview_certifiable_text": {
        "en": "Only **2 methods** are certifiable standalone:\n1. **TreeExplainer** (τ_R = 1.00) — but only for tree-based models\n2. **Permutation Importance** (τ_R ≥ 0.91) — model-agnostic, global\n\nMethods like **KernelSHAP + ALE + Perm. Imp.** can achieve certification via triangulation (τ_tri ≥ 0.80).",
        "fr": "Seules **2 méthodes** sont certifiables seules :\n1. **TreeExplainer** (τ_R = 1,00) — mais uniquement pour les modèles à base d'arbres\n2. **Importance par Permutation** (τ_R ≥ 0,91) — modèle-agnostique, globale\n\nDes combinaisons comme **KernelSHAP + ALE + Perm. Imp.** peuvent atteindre la certification par triangulation (τ_tri ≥ 0,80).",
    },
    
    # --- Table IX ---
    "table_ix_title": {"en": "Table IX — Comparative Evaluation", "fr": "Tableau IX — Évaluation Comparative"},
    "table_ix_subtitle": {
        "en": "Across 7 datasets and 9 models (mean range; 95% CI ±0.02–0.05)",
        "fr": "Sur 7 datasets et 9 modèles (intervalle moyen ; IC 95% ±0,02–0,05)",
    },
    "table_ix_note": {
        "en": "*TreeExplainer requires tree-based models but supports RF, XGBoost, LightGBM, CatBoost.",
        "fr": "*TreeExplainer nécessite des modèles à base d'arbres mais supporte RF, XGBoost, LightGBM, CatBoost.",
    },
    "col_method": {"en": "Method", "fr": "Méthode"},
    "col_fidelity": {"en": "Fidelity (φ)", "fr": "Fidélité (φ)"},
    "col_stability": {"en": "Stability (τ_R)", "fr": "Stabilité (τ_R)"},
    "col_cost": {"en": "Cost (s)", "fr": "Coût (s)"},
    "col_scope": {"en": "Scope", "fr": "Portée"},
    "col_agnostic": {"en": "Agnostic", "fr": "Agnostique"},
    "col_causal": {"en": "Causal", "fr": "Causal"},
    "col_certifiable": {"en": "Certifiable?", "fr": "Certifiable ?"},
    
    # --- Radar ---
    "radar_title": {"en": "Six-Axis Radar Comparison", "fr": "Comparaison Radar à Six Axes"},
    "radar_subtitle": {
        "en": "Normalized scores (0–1) across all evaluation axes. Cost axis is inverted (lower cost = higher score).",
        "fr": "Scores normalisés (0–1) sur tous les axes d'évaluation. L'axe Coût est inversé (coût plus faible = score plus élevé).",
    },
    "radar_select": {"en": "Select methods to compare:", "fr": "Sélectionner les méthodes à comparer :"},
    
    # --- Fidelity ---
    "fidelity_title": {"en": "Fidelity Scores (φ)", "fr": "Scores de Fidélité (φ)"},
    "fidelity_subtitle": {
        "en": "Top-k agreement between explanation and ablation-based ground truth. Green line = certification threshold (φ ≥ 0.85).",
        "fr": "Accord top-k entre l'explication et la vérité terrain par ablation. Ligne verte = seuil de certification (φ ≥ 0,85).",
    },
    "certification_threshold_fidelity": {"en": "Certification threshold (φ ≥ 0.85)", "fr": "Seuil de certification (φ ≥ 0,85)"},
    
    # --- Stability ---
    "stability_title": {"en": "Stability Scores (Kendall's τ_R)", "fr": "Scores de Stabilité (τ_R de Kendall)"},
    "stability_subtitle": {
        "en": "Ranking consistency across bootstrap resamples. Green line = certification threshold (τ_R ≥ 0.90).",
        "fr": "Cohérence du classement à travers les rééchantillonnages bootstrap. Ligne verte = seuil de certification (τ_R ≥ 0,90).",
    },
    "certification_threshold_stability": {"en": "Certification threshold (τ_R ≥ 0.90)", "fr": "Seuil de certification (τ_R ≥ 0,90)"},
    
    # --- Per Dataset ---
    "per_dataset_title": {"en": "Results per Dataset", "fr": "Résultats par Dataset"},
    "per_dataset_select": {"en": "Select a dataset:", "fr": "Sélectionner un dataset :"},
    "per_dataset_info": {
        "en": "Samples: {n} | Features: {p} | Type: {type} | Task: {task}",
        "fr": "Échantillons : {n} | Variables : {p} | Type : {type} | Tâche : {task}",
    },
    
    # --- Decision Framework ---
    "decision_title": {"en": "Decision Framework for XAI Method Selection", "fr": "Cadre Décisionnel pour la Sélection de Méthodes XAI"},
    "decision_intro": {
        "en": "Use this decision tree to select the appropriate XAI method based on your context.",
        "fr": "Utilisez cet arbre de décision pour sélectionner la méthode XAI appropriée selon votre contexte.",
    },
    "decision_q1": {"en": "What type of data?", "fr": "Quel type de données ?"},
    "decision_tabular": {"en": "Tabular Data", "fr": "Données Tabulaires"},
    "decision_image": {"en": "Image Data", "fr": "Données Image"},
    "decision_timeseries": {"en": "Time-Series", "fr": "Séries Temporelles"},
    "decision_tree_models": {"en": "Tree Models? → TreeExplainer + Perm. Imp. + ALE (Grade A)", "fr": "Modèles Arbres ? → TreeExplainer + Perm. Imp. + ALE (Grade A)"},
    "decision_any_model": {"en": "Any Model? → KernelSHAP + ALE + Perm. Imp. (Verify τ_R ≥ 0.85)", "fr": "Tout Modèle ? → KernelSHAP + ALE + Perm. Imp. (Vérifier τ_R ≥ 0,85)"},
    "decision_image_rec": {"en": "GradCAM + cross-validation (check with occlusion)", "fr": "GradCAM + validation croisée (vérifier par occlusion)"},
    "decision_ts_rec": {"en": "KernelSHAP (time-aware) + ALE + LSTM validation", "fr": "KernelSHAP (temporel) + ALE + validation LSTM"},
    "decision_high_stakes": {"en": "HIGH-STAKES DECISION? → Multi-method triangulation (SHAP + LIME + Perm. Imp., τ_tri ≥ 0.80)", "fr": "DÉCISION À ENJEUX ÉLEVÉS ? → Triangulation multi-méthodes (SHAP + LIME + Perm. Imp., τ_tri ≥ 0,80)"},
    "decision_thresholds_title": {"en": "Certification Thresholds", "fr": "Seuils de Certification"},
    "decision_thresholds": {
        "en": "• Fidelity: φ ≥ 0.85\n• Stability: τ_R ≥ 0.90\n• Triangulation: τ_tri ≥ 0.80",
        "fr": "• Fidélité : φ ≥ 0,85\n• Stabilité : τ_R ≥ 0,90\n• Triangulation : τ_tri ≥ 0,80",
    },
    "decision_non_certifiable": {
        "en": "NON-CERTIFIABLE (standalone): LIME, LRP, GradCAM, Counterfactual",
        "fr": "NON-CERTIFIABLE (seul) : LIME, LRP, GradCAM, Contrefactuel",
    },
    
    # --- Live Benchmark ---
    "benchmark_title": {"en": "⚡ Live Benchmark — Instant Demo", "fr": "⚡ Benchmark en Direct — Démo Instantanée"},
    "benchmark_intro": {
        "en": "Run a quick benchmark on the Adult Census dataset (~2 min). This demonstrates the evaluation pipeline with 2 models and 3 methods.",
        "fr": "Exécutez un benchmark rapide sur le dataset Adult Census (~2 min). Ceci démontre le pipeline d'évaluation avec 2 modèles et 3 méthodes.",
    },
    "benchmark_warning": {
        "en": "⚠️ This is a **quick demo** with reduced parameters. Full benchmark (7 datasets × 9 models × 10 methods) takes ~48h. Article results use full parameters.",
        "fr": "⚠️ Ceci est une **démo rapide** avec des paramètres réduits. Le benchmark complet (7 datasets × 9 modèles × 10 méthodes) prend ~48h. Les résultats de l'article utilisent les paramètres complets.",
    },
    "benchmark_button": {"en": "🚀 Run Quick Benchmark", "fr": "🚀 Lancer le Benchmark Rapide"},
    "benchmark_running": {"en": "Running benchmark...", "fr": "Exécution du benchmark..."},
    "benchmark_done": {"en": "Benchmark complete!", "fr": "Benchmark terminé !"},
    "benchmark_results_title": {"en": "Results", "fr": "Résultats"},
    "benchmark_compare_title": {"en": "Comparison with Article Results", "fr": "Comparaison avec les Résultats de l'Article"},
    
    # --- Method Details ---
    "method_title": {"en": "Method Details", "fr": "Détails des Méthodes"},
    "method_select": {"en": "Select a method:", "fr": "Sélectionner une méthode :"},
    "method_family": {"en": "Family", "fr": "Famille"},
    "method_year": {"en": "Year", "fr": "Année"},
    "method_reference": {"en": "Reference", "fr": "Référence"},
    "method_complexity": {"en": "Complexity", "fr": "Complexité"},
    "method_description": {"en": "Description", "fr": "Description"},
    "method_performance": {"en": "Performance", "fr": "Performance"},
    
    # --- Article ---
    "article_title": {"en": "Article Reference", "fr": "Référence de l'Article"},
    "article_citation": {"en": "Citation (BibTeX)", "fr": "Citation (BibTeX)"},
    "article_abstract_title": {"en": "Abstract", "fr": "Résumé"},
    "article_abstract": {
        "en": "This paper presents a systematic comparative evaluation of ten post-hoc explanation methods for predictive modeling: SHAP (TreeExplainer, KernelExplainer, DeepExplainer), LIME, Permutation Importance, Partial Dependence Plots, Accumulated Local Effects, Layer-Wise Relevance Propagation, GradCAM, and Counterfactual Explanations. We evaluate these methods across seven benchmark datasets (tabular, image, time-series) and nine predictive models using six evaluation axes: fidelity, stability, computational cost, scope, model-agnosticity, and causality. Our results show that only TreeExplainer (τ_R = 1.00) and Permutation Importance (τ_R ≥ 0.91) meet certification thresholds for standalone use. We propose a triangulation-based certification framework and a decision framework for XAI method selection in practical applications.",
        "fr": "Cet article présente une évaluation comparative systématique de dix méthodes d'explication post-hoc pour la modélisation prédictive : SHAP (TreeExplainer, KernelExplainer, DeepExplainer), LIME, Importance par Permutation, Graphiques de Dépendance Partielle, Effets Locaux Accumulés, Propagation de Pertinence par Couches, GradCAM et Explications Contrefactuelles. Nous évaluons ces méthodes sur sept datasets de référence (tabulaires, images, séries temporelles) et neuf modèles prédictifs selon six axes d'évaluation : fidélité, stabilité, coût computationnel, portée, agnosticité et causalité. Nos résultats montrent que seuls TreeExplainer (τ_R = 1,00) et l'Importance par Permutation (τ_R ≥ 0,91) atteignent les seuils de certification pour une utilisation autonome. Nous proposons un cadre de certification par triangulation et un cadre décisionnel pour la sélection de méthodes XAI dans les applications pratiques.",
    },
    
    # --- Scope labels ---
    "scope_both": {"en": "Both", "fr": "Les deux"},
    "scope_local": {"en": "Local", "fr": "Local"},
    "scope_global": {"en": "Global", "fr": "Global"},
    
    # --- General ---
    "yes": {"en": "Yes", "fr": "Oui"},
    "no": {"en": "No", "fr": "Non"},
    "partial": {"en": "Partial", "fr": "Partiel"},
    "limited": {"en": "Limited", "fr": "Limité"},
    "download_pdf": {"en": "📥 Download PDF", "fr": "📥 Télécharger PDF"},
    "download_csv": {"en": "📥 Download CSV", "fr": "📥 Télécharger CSV"},
}


def t(key, lang="en"):
    """Translate a key to the specified language."""
    if key in TRANSLATIONS:
        return TRANSLATIONS[key].get(lang, TRANSLATIONS[key].get("en", key))
    return key