#!/usr/bin/env python3
"""
XAI Benchmark Dashboard — Article 4
Streamlit app with French/English toggle, instant pre-computed results, and live demo.
"""

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
import sys
import os
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / "scripts"))

from precomputed_results import (
    TABLE_IX, TABLE_II_ADULT, PER_DATASET, CERTIFICATION,
    TRIANGULATION, METHOD_DETAILS, get_table_ix_df, get_per_dataset_df,
)
from translations import TRANSLATIONS, t

# ========================================================================
# PAGE CONFIG
# ========================================================================

st.set_page_config(
    page_title="XAI Benchmark — Article 4",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ========================================================================
# LANGUAGE TOGGLE (in sidebar)
# ========================================================================

if "lang" not in st.session_state:
    st.session_state.lang = "en"

with st.sidebar:
    lang_label = "🇬🇧 English" if st.session_state.lang == "fr" else "🇫🇷 Français"
    if st.button(lang_label, key="lang_toggle"):
        st.session_state.lang = "en" if st.session_state.lang == "fr" else "fr"
        st.rerun()
    lang = st.session_state.lang

# ========================================================================
# NAVIGATION
# ========================================================================

with st.sidebar:
    page = st.radio(
        "Navigation" if lang == "en" else "Navigation",
        [
            t("nav_overview", lang),
            t("nav_table_ix", lang),
            t("nav_radar", lang),
            t("nav_fidelity", lang),
            t("nav_stability", lang),
            t("nav_per_dataset", lang),
            t("nav_decision", lang),
            t("nav_benchmark", lang),
            t("nav_method", lang),
            t("nav_article", lang),
        ],
        index=0,
    )

# ========================================================================
# METHOD COLORS
# ========================================================================

METHOD_COLORS = {
    "TreeExplainer": "#2196F3", "KernelSHAP": "#64B5F6", "DeepSHAP": "#90CAF9",
    "LIME": "#FF9800", "Perm. Imp.": "#4CAF50", "PDP": "#9C27B0",
    "ALE": "#7B1FA2", "LRP": "#F44336", "GradCAM": "#E91E63", "Counterfactual": "#795548",
}

# ========================================================================
# PAGE: OVERVIEW
# ========================================================================

if page == t("nav_overview", lang):
    st.title(t("app_title", lang))
    st.markdown(f"### {t('app_subtitle', lang)}")
    st.markdown("---")
    
    st.markdown(t("overview_intro", lang))
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric(t("overview_methods", lang), "10")
    col2.metric(t("overview_datasets", lang), "7")
    col3.metric(t("overview_models", lang), "9")
    col4.metric(t("overview_axes", lang), "6")
    
    st.markdown("---")
    st.markdown(f"### {t('overview_certifiable_title', lang)}")
    st.markdown(t("overview_certifiable_text", lang))
    
    # Quick summary table
    df = get_table_ix_df()
    st.dataframe(
        df[["Method", "Fidelity (φ)", "Stability (τ_R)", "Cost (s)", "Scope", "Certifiable"]],
        use_container_width=True,
        hide_index=True,
    )

# ========================================================================
# PAGE: TABLE IX
# ========================================================================

elif page == t("nav_table_ix", lang):
    st.title(t("table_ix_title", lang))
    st.markdown(f"*{t('table_ix_subtitle', lang)}*")
    
    df = get_table_ix_df()
    
    # Format for display
    display_cols = {
        "en": ["Method", "Fidelity (φ)", "Stability (τ_R)", "Cost (s)", "Scope", "Agnostic", "Causal", "Certifiable"],
        "fr": ["Méthode", "Fidélité (φ)", "Stabilité (τ_R)", "Coût (s)", "Portée", "Agnostique", "Causal", "Certifiable"],
    }
    col_map = {
        "Method": t("col_method", lang),
        "Fidelity (φ)": t("col_fidelity", lang),
        "Stability (τ_R)": t("col_stability", lang),
        "Cost (s)": t("col_cost", lang),
        "Scope": t("col_scope", lang),
        "Agnostic": t("col_agnostic", lang),
        "Causal": t("col_causal", lang),
        "Certifiable": t("col_certifiable", lang),
    }
    
    display_df = df[["Method", "Fidelity (φ)", "Stability (τ_R)", "Cost (s)", "Scope", "Agnostic", "Causal", "Certifiable"]].copy()
    display_df = display_df.rename(columns=col_map)
    
    # Highlight certifiable
    def highlight_certifiable(val):
        if val == "Yes" or val == "Oui":
            return "background-color: #c8e6c9; font-weight: bold"
        elif val == "Limited" or val == "Limité":
            return "background-color: #fff9c4"
        elif val == "No" or val == "Non":
            return "background-color: #ffcdd2"
        return ""
    
    styled = display_df.style.applymap(highlight_certifiable, subset=[col_map["Certifiable"]])
    st.dataframe(styled, use_container_width=True, hide_index=True)
    
    st.markdown(f"*{t('table_ix_note', lang)}*")
    
    # Download CSV
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button(
        t("download_csv", lang),
        data=csv,
        file_name="table_ix_results.csv",
        mime="text/csv",
    )

# ========================================================================
# PAGE: RADAR
# ========================================================================

elif page == t("nav_radar", lang):
    st.title(t("radar_title", lang))
    st.markdown(t("radar_subtitle", lang))
    
    # Method selection
    all_methods = list(TABLE_IX.keys())
    default_methods = ["TreeExplainer", "KernelSHAP", "LIME", "Perm. Imp.", "ALE", "GradCAM"]
    selected = st.multiselect(
        t("radar_select", lang),
        all_methods,
        default=default_methods,
    )
    
    if len(selected) < 2:
        st.warning("Select at least 2 methods." if lang == "en" else "Sélectionnez au moins 2 méthodes.")
    else:
        # Normalize data to 0-1
        categories = {
            "en": ["Fidelity", "Stability", "Cost\n(inverted)", "Scope", "Agnostic", "Causal"],
            "fr": ["Fidélité", "Stabilité", "Coût\n(inversé)", "Portée", "Agnosticité", "Causalité"],
        }
        cats = categories[lang]
        
        fig = go.Figure()
        
        for method in selected:
            data = TABLE_IX[method]
            # Invert cost: lower cost = better → normalize as 1 - (cost/max_cost)
            max_cost = 300  # max cost in the table
            cost_score = max(0, 1 - np.mean(data["cost"]) / max_cost)
            
            scope_map = {"Both": 0.75, "Global": 0.5, "Local": 0.25}
            agnostic_score = 1.0 if data["agnostic"] else 0.0
            causal_score = 0.5 if data["causal"] == "Partial" else (1.0 if data["causal"] else 0.0)
            
            values = [
                np.mean(data["fidelity"]),
                np.mean(data["stability"]),
                cost_score,
                scope_map.get(data["scope"], 0.25),
                agnostic_score,
                causal_score,
            ]
            values.append(values[0])  # close the polygon
            
            fig.add_trace(go.Scatterpolar(
                r=values,
                theta=cats + [cats[0]],
                fill='toself',
                name=method,
                line=dict(color=METHOD_COLORS.get(method, "#888")),
                opacity=0.6,
            ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(visible=True, range=[0, 1]),
            ),
            showlegend=True,
            height=600,
            title=t("radar_title", lang),
        )
        
        st.plotly_chart(fig, use_container_width=True)

# ========================================================================
# PAGE: FIDELITY
# ========================================================================

elif page == t("nav_fidelity", lang):
    st.title(t("fidelity_title", lang))
    st.markdown(t("fidelity_subtitle", lang))
    
    methods = list(TABLE_IX.keys())
    fidelity_means = [np.mean(TABLE_IX[m]["fidelity"]) for m in methods]
    fidelity_lows = [np.mean(TABLE_IX[m]["fidelity"]) - TABLE_IX[m]["fidelity"][0] for m in methods]
    fidelity_highs = [TABLE_IX[m]["fidelity"][1] - np.mean(TABLE_IX[m]["fidelity"]) for m in methods]
    colors = [METHOD_COLORS.get(m, "#888") for m in methods]
    
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=methods,
        y=fidelity_means,
        error_y=dict(type="data", symmetric=False, array=fidelity_highs, arrayminus=fidelity_lows),
        marker_color=colors,
        opacity=0.85,
    ))
    
    # Certification threshold
    fig.add_hline(
        y=0.85, line_dash="dash", line_color="green", line_width=2,
        annotation_text=t("certification_threshold_fidelity", lang),
        annotation_position="top left",
    )
    
    fig.update_layout(
        yaxis=dict(title="Fidelity (φ)" if lang == "en" else "Fidélité (φ)", range=[0, 1.1]),
        xaxis=dict(title="Method" if lang == "en" else "Méthode"),
        height=500,
    )
    
    st.plotly_chart(fig, use_container_width=True)

# ========================================================================
# PAGE: STABILITY
# ========================================================================

elif page == t("nav_stability", lang):
    st.title(t("stability_title", lang))
    st.markdown(t("stability_subtitle", lang))
    
    methods = list(TABLE_IX.keys())
    stab_means = [np.mean(TABLE_IX[m]["stability"]) for m in methods]
    stab_lows = [np.mean(TABLE_IX[m]["stability"]) - TABLE_IX[m]["stability"][0] for m in methods]
    stab_highs = [TABLE_IX[m]["stability"][1] - np.mean(TABLE_IX[m]["stability"]) for m in methods]
    colors = [METHOD_COLORS.get(m, "#888") for m in methods]
    
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=methods,
        y=stab_means,
        error_y=dict(type="data", symmetric=False, array=stab_highs, arrayminus=stab_lows),
        marker_color=colors,
        opacity=0.85,
    ))
    
    # Certification threshold
    fig.add_hline(
        y=0.90, line_dash="dash", line_color="green", line_width=2,
        annotation_text=t("certification_threshold_stability", lang),
        annotation_position="top left",
    )
    
    fig.update_layout(
        yaxis=dict(title="Stability (τ_R)" if lang == "en" else "Stabilité (τ_R)", range=[0, 1.1]),
        xaxis=dict(title="Method" if lang == "en" else "Méthode"),
        height=500,
    )
    
    st.plotly_chart(fig, use_container_width=True)

# ========================================================================
# PAGE: PER DATASET
# ========================================================================

elif page == t("nav_per_dataset", lang):
    st.title(t("per_dataset_title", lang))
    
    dataset_names = list(PER_DATASET.keys())
    selected_ds = st.selectbox(t("per_dataset_select", lang), dataset_names)
    
    ds = PER_DATASET[selected_ds]
    info = t("per_dataset_info", lang).format(
        n=ds["n"], p=ds["p"], type=ds["type"], task=ds["task"]
    )
    st.markdown(f"**{info}**")
    st.markdown(f"Models: {', '.join(ds['models'])}")
    
    df = get_per_dataset_df(selected_ds)
    st.dataframe(df, use_container_width=True, hide_index=True)
    
    # Fidelity + Stability bars for this dataset
    results = ds["results"]
    methods = list(results.keys())
    
    col1, col2 = st.columns(2)
    
    with col1:
        fid_means = [np.mean(results[m]["fidelity"]) for m in methods]
        fid_lows = [np.mean(results[m]["fidelity"]) - results[m]["fidelity"][0] for m in methods]
        fid_highs = [results[m]["fidelity"][1] - np.mean(results[m]["fidelity"]) for m in methods]
        colors = [METHOD_COLORS.get(m, "#888") for m in methods]
        
        fig_fid = go.Figure()
        fig_fid.add_trace(go.Bar(
            x=methods, y=fid_means,
            error_y=dict(type="data", symmetric=False, array=fid_highs, arrayminus=fid_lows),
            marker_color=colors, opacity=0.85,
        ))
        fig_fid.add_hline(y=0.85, line_dash="dash", line_color="green", line_width=2)
        fig_fid.update_layout(
            title="Fidelity (φ)" if lang == "en" else "Fidélité (φ)",
            yaxis=dict(range=[0, 1.1]),
            height=400,
        )
        st.plotly_chart(fig_fid, use_container_width=True)
    
    with col2:
        stab_means = [np.mean(results[m]["stability"]) for m in methods]
        stab_lows = [np.mean(results[m]["stability"]) - results[m]["stability"][0] for m in methods]
        stab_highs = [results[m]["stability"][1] - np.mean(results[m]["stability"]) for m in methods]
        
        fig_stab = go.Figure()
        fig_stab.add_trace(go.Bar(
            x=methods, y=stab_means,
            error_y=dict(type="data", symmetric=False, array=stab_highs, arrayminus=stab_lows),
            marker_color=colors, opacity=0.85,
        ))
        fig_stab.add_hline(y=0.90, line_dash="dash", line_color="green", line_width=2)
        fig_stab.update_layout(
            title="Stability (τ_R)" if lang == "en" else "Stabilité (τ_R)",
            yaxis=dict(range=[0, 1.1]),
            height=400,
        )
        st.plotly_chart(fig_stab, use_container_width=True)

# ========================================================================
# PAGE: DECISION FRAMEWORK
# ========================================================================

elif page == t("nav_decision", lang):
    st.title(t("decision_title", lang))
    st.markdown(t("decision_intro", lang))
    
    # Interactive decision tree
    st.markdown("---")
    
    # Step 1: Data type
    data_type = st.radio(
        t("decision_q1", lang),
        [t("decision_tabular", lang), t("decision_image", lang), t("decision_timeseries", lang)],
        horizontal=True,
    )
    
    st.markdown("---")
    
    if data_type == t("decision_tabular", lang):
        # Tree models?
        tree_models = st.checkbox(
            "Tree-based models (RF, XGBoost, LightGBM, CatBoost)?"
            if lang == "en" else "Modèles à base d'arbres (RF, XGBoost, LightGBM, CatBoost) ?",
            value=True,
        )
        if tree_models:
            st.success(f"✅ **{t('decision_tree_models', lang)}**")
        else:
            st.warning(f"⚠️ **{t('decision_any_model', lang)}**")
    
    elif data_type == t("decision_image", lang):
        st.info(f"📷 **{t('decision_image_rec', lang)}**")
    
    elif data_type == t("decision_timeseries", lang):
        st.info(f"📈 **{t('decision_ts_rec', lang)}**")
    
    st.markdown("---")
    
    # High-stakes?
    high_stakes = st.checkbox(
        "High-stakes decision (medical, legal, financial)?"
        if lang == "en" else "Décision à enjeux élevés (médical, juridique, financier) ?",
        value=False,
    )
    if high_stakes:
        st.error(f"🔴 **{t('decision_high_stakes', lang)}**")
    
    st.markdown("---")
    
    # Certification thresholds
    st.markdown(f"### {t('decision_thresholds_title', lang)}")
    thresholds = t("decision_thresholds", lang)
    for line in thresholds.split("\n"):
        st.markdown(f"**{line}**")
    
    st.markdown("---")
    st.markdown(f"⚠️ {t('decision_non_certifiable', lang)}")
    
    # Static decision framework figure
    st.markdown("---")
    figures_dir = Path(__file__).parent / "figures"
    decision_png = figures_dir / "decision_framework.png"
    if decision_png.exists():
        st.image(str(decision_png), caption=t("decision_title", lang))

# ========================================================================
# PAGE: LIVE BENCHMARK
# ========================================================================

elif page == t("nav_benchmark", lang):
    st.title(t("benchmark_title", lang))
    st.markdown(t("benchmark_intro", lang))
    st.warning(t("benchmark_warning", lang))
    
    if st.button(t("benchmark_button", lang), type="primary"):
        with st.spinner(t("benchmark_running", lang)):
            progress = st.progress(0, text="Loading dataset..." if lang == "en" else "Chargement du dataset...")
            
            try:
                from dataloaders import load_dataset
                from explainers import create_explainer, get_compatible_methods
                from metrics import compute_fidelity, compute_stability, compute_cost
                from sklearn.linear_model import LogisticRegression
                from sklearn.ensemble import RandomForestClassifier
                
                # Load Adult Census
                progress.progress(10, text="Loading Adult Census..." if lang == "en" else "Chargement d'Adult Census...")
                X_train, X_test, y_train, y_test, feature_names, data_type, task_type = load_dataset("adult")
                
                # Train 2 models
                progress.progress(20, text="Training Logistic Regression..." if lang == "en" else "Entraînement de la Régression Logistique...")
                model_lr = LogisticRegression(max_iter=500, random_state=42)
                model_lr.fit(X_train, y_train)
                
                progress.progress(35, text="Training Random Forest..." if lang == "en" else "Entraînement de Random Forest...")
                model_rf = RandomForestClassifier(n_estimators=50, max_depth=8, random_state=42, n_jobs=-1)
                model_rf.fit(X_train, y_train)
                
                models = {"Logistic Regression": ("shap_kernel", model_lr), "Random Forest": ("shap_tree", model_rf)}
                
                # Quick eval: subsample for speed
                X_test_sub = X_test[:100]
                y_test_sub = y_test[:100]
                
                all_results = []
                methods_done = 0
                total_methods = 3  # shap_tree, shap_kernel, permutation
                
                for model_name, (primary_method, model) in models.items():
                    compatible = get_compatible_methods(
                        "random_forest" if "Random" in model_name else "ridge", data_type
                    )
                    # Pick top 3
                    eval_methods = [m for m in ["shap_tree", "shap_kernel", "permutation", "lime"] if m in compatible][:2]
                    
                    for method_name in eval_methods:
                        methods_done += 1
                        progress.progress(
                            40 + int(50 * methods_done / total_methods),
                            text=f"Evaluating {method_name} on {model_name}..." if lang == "en" else f"Évaluation de {method_name} sur {model_name}..."
                        )
                        try:
                            explainer = create_explainer(
                                method_name, model, X_train[:200],
                                feature_names=feature_names,
                                model_type=data_type, task_type=task_type,
                            )
                            
                            # Fidelity (quick: 20 instances)
                            f_mean, f_std, f_list = compute_fidelity(explainer, model, X_test_sub, y_test_sub, k=5, n_instances=20)
                            # Stability (quick: 10 resamples)
                            s_mean, s_std, s_list = compute_stability(explainer, model, X_test_sub, n_bootstrap=10)
                            # Cost (quick: 10 instances)
                            c_mean, c_std, c_list = compute_cost(explainer, X_test_sub[:10])
                            
                            all_results.append({
                                "Model": model_name,
                                "Method": method_name,
                                "Fidelity (φ)": round(f_mean, 3),
                                "Stability (τ_R)": round(s_mean, 3),
                                "Cost (s)": round(c_mean, 3),
                            })
                        except Exception as e:
                            all_results.append({
                                "Model": model_name,
                                "Method": method_name,
                                "Error": str(e)[:80],
                            })
                
                progress.progress(95, text="Compiling results..." if lang == "en" else "Compilation des résultats...")
                
                if all_results:
                    df_results = pd.DataFrame(all_results)
                    st.success(t("benchmark_done", lang))
                    
                    st.markdown(f"### {t('benchmark_results_title', lang)}")
                    st.dataframe(df_results, use_container_width=True, hide_index=True)
                    
                    # Compare with article
                    st.markdown(f"### {t('benchmark_compare_title', lang)}")
                    article_data = TABLE_II_ADULT
                    compare_rows = []
                    for method in ["shap_tree", "shap_kernel", "permutation"]:
                        key_map = {"shap_tree": "TreeExplainer", "shap_kernel": "KernelSHAP", "permutation": "Perm. Imp."}
                        article_key = key_map.get(method, method)
                        if article_key in article_data:
                            compare_rows.append({
                                "Method": article_key,
                                "Article Fidelity": f"{article_data[article_key]['fidelity'][0]:.2f}–{article_data[article_key]['fidelity'][1]:.2f}",
                                "Article Stability": f"{article_data[article_key]['stability'][0]:.2f}–{article_data[article_key]['stability'][1]:.2f}",
                            })
                    
                    if compare_rows:
                        st.dataframe(pd.DataFrame(compare_rows), use_container_width=True, hide_index=True)
                
                progress.progress(100, text="Done!" if lang == "en" else "Terminé !")
                
            except Exception as e:
                st.error(f"Error: {e}")
                import traceback
                st.code(traceback.format_exc())

# ========================================================================
# PAGE: METHOD DETAILS
# ========================================================================

elif page == t("nav_method", lang):
    st.title(t("method_title", lang))
    
    method_names = list(METHOD_DETAILS.keys())
    selected_method = st.selectbox(t("method_select", lang), method_names)
    
    details = METHOD_DETAILS[selected_method]
    perf = TABLE_IX[selected_method]
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"### {details[f'name_{lang}']}")
        st.markdown(f"**{t('method_family', lang)}:** {details['family']}")
        st.markdown(f"**{t('method_year', lang)}:** {details['year']}")
        st.markdown(f"**{t('method_reference', lang)}:** {details['ref']}")
        st.markdown(f"**{t('method_complexity', lang)}:** `{details['complexity']}`")
        st.markdown("---")
        st.markdown(f"**{t('method_description', lang)}:**")
        st.markdown(details[f"description_{lang}"])
    
    with col2:
        st.markdown(f"### {t('method_performance', lang)}")
        
        # Performance metrics
        fid_val = f"{perf['fidelity'][0]:.2f}–{perf['fidelity'][1]:.2f}"
        stab_val = f"{perf['stability'][0]:.2f}–{perf['stability'][1]:.2f}"
        cost_val = f"{perf['cost'][0]:.1f}–{perf['cost'][1]:.1f}s"
        
        fid_cert = "✅" if np.mean(perf["fidelity"]) >= 0.85 else "❌"
        stab_cert = "✅" if np.mean(perf["stability"]) >= 0.90 else "❌"
        
        st.metric("Fidelity (φ)", fid_val, 
                  delta=f"Certification: {fid_cert}",
                  delta_color="normal" if "✅" in fid_cert else "inverse")
        st.metric("Stability (τ_R)", stab_val,
                  delta=f"Certification: {stab_cert}",
                  delta_color="normal" if "✅" in stab_cert else "inverse")
        st.metric("Cost", cost_val)
        st.metric("Scope", perf["scope"])
        st.metric("Agnostic", t("yes", lang) if perf["agnostic"] else t("no", lang))
        st.metric("Causal", t("partial", lang) if perf["causal"] == "Partial" else (t("yes", lang) if perf["causal"] else t("no", lang)))
        st.metric("Certifiable", perf["certifiable"])

# ========================================================================
# PAGE: ARTICLE
# ========================================================================

elif page == t("nav_article", lang):
    st.title(t("article_title", lang))
    
    st.markdown(f"### {t('article_abstract_title', lang)}")
    st.markdown(t("article_abstract", lang))
    
    st.markdown("---")
    st.markdown(f"### {t('article_citation', lang)}")
    
    bibtex = """@article{ralinirina2024comparative,
  title={A Comparative Evaluation of Post-Hoc Explanation Methods for Predictive Modeling: Strengths, Limitations, and Selection Guidelines},
  author={Ralinirina, Rosa Elysabeth and Ralaivao, Jean Christian and Ralaivao, Niaiko Michaël and Ratovondrahona, Alain Josué and Mahatody, Thomas},
  journal={IEEE Access},
  year={2024}
}"""
    st.code(bibtex, language="bibtex")
    
    # Triangulation
    st.markdown("---")
    tri_title = "Triangulation Results" if lang == "en" else "Résultats de Triangulation"
    st.markdown(f"### {tri_title}")
    
    tri_rows = []
    for name, data in TRIANGULATION.items():
        cert_label = t("yes", lang) if data["certifiable"] == True else (
            t("limited", lang) if data["certifiable"] == "Conditional" else t("no", lang)
        )
        tri_rows.append({
            "Combination" if lang == "en" else "Combinaison": name,
            "τ_tri": f"{data['tau_tri'][0]:.2f}–{data['tau_tri'][1]:.2f}",
            "Datasets": ", ".join(data["datasets"]),
            "Certifiable": cert_label,
        })
    
    st.dataframe(pd.DataFrame(tri_rows), use_container_width=True, hide_index=True)
    
    # Download figures
    st.markdown("---")
    dl_title = "Download Figures" if lang == "en" else "Télécharger les Figures"
    st.markdown(f"### {dl_title}")
    
    figures_dir = Path(__file__).parent / "figures"
    for fig_name in ["radar_comparison.png", "fidelity_comparison.png", "stability_comparison.png", "decision_framework.png"]:
        fig_path = figures_dir / fig_name
        if fig_path.exists():
            with open(str(fig_path), "rb") as f:
                st.download_button(
                    f"📥 {fig_name}",
                    data=f.read(),
                    file_name=fig_name,
                    mime="image/png",
                    key=fig_name,
                )

# ========================================================================
# FOOTER
# ========================================================================

st.markdown("---")
footer_text = {
    "en": "XAI Benchmark — Article 4 | [GitHub](https://github.com/RosaElysabeth/xai-benchmark-article)",
    "fr": "Benchmark XAI — Article 4 | [GitHub](https://github.com/RosaElysabeth/xai-benchmark-article)",
}
st.markdown(footer_text[lang])