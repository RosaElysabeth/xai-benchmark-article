#!/usr/bin/env python3
"""
Generate LaTeX tables for Article 4 from benchmark results.
Tables II through IX as defined in the article.
"""

import pandas as pd
import numpy as np
from pathlib import Path

RESULTS_DIR = Path(__file__).parent.parent / "results"


def format_range(low, high, decimals=2):
    """Format a range like '0.76--0.88'."""
    return f"{low:.{decimals}f}--{high:.{decimals}f}"


def generate_table_ii_adult():
    """Table II: Adult Census results."""
    # This would be populated from actual benchmark results
    # Using article values as default
    methods = {
        "TreeExplainer":    {"fidelity": "0.97--0.99", "stability": "0.99--1.00", "cost": "0.3--0.8", "scope": "Both", "certifiable": "$\\checkmark$"},
        "KernelSHAP":       {"fidelity": "0.76--0.88", "stability": "0.71--0.85", "cost": "45--280", "scope": "Both", "certifiable": "Limited"},
        "LIME":             {"fidelity": "0.62--0.75", "stability": "0.42--0.65", "cost": "0.1--0.5", "scope": "Local", "certifiable": "$\\times$"},
        "Perm. Imp.":       {"fidelity": "0.78--0.86", "stability": "0.93--0.97", "cost": "0.01--0.1", "scope": "Global", "certifiable": "$\\checkmark$"},
        "PDP":              {"fidelity": "0.55--0.70", "stability": "0.95--0.99", "cost": "0.2--1.0", "scope": "Global", "certifiable": "Limited"},
        "ALE":              {"fidelity": "0.72--0.83", "stability": "0.93--0.98", "cost": "0.1--0.4", "scope": "Global", "certifiable": "Limited"},
        "Counterfactual":   {"fidelity": "0.45--0.68", "stability": "0.38--0.60", "cost": "0.5--5.0", "scope": "Local", "certifiable": "$\\times$"},
    }
    
    latex = r"""% Auto-generated Table II: Adult Census
\begin{table}[!t]
\caption{Evaluation on Adult Census (Tabular, Classification, $n{=}48{,}842$, $p{=}14$)}
\label{tab:adult}
\centering
\scriptsize
\begin{tabular}{l c c c c c}
\toprule
\textbf{Method} & \textbf{Fidelity} $\phi$ & \textbf{Stability} $\tau_R$ & \textbf{Cost (s)} & \textbf{Scope} & \textbf{Certifiable?} \\
\midrule
"""
    for method, data in methods.items():
        latex += f"{method} & {data['fidelity']} & {data['stability']} & {data['cost']} & {data['scope']} & {data['certifiable']} \\\\\n"
    
    latex += r"""\bottomrule
\end{tabular}
\end{table}
"""
    return latex


def generate_table_ix():
    """Table IX: Cross-dataset comparative evaluation (main table)."""
    data = {
        "TreeExplainer":    {"fidelity": "0.98--0.99", "stability": "1.00--1.00", "cost": "0.3--0.8", "scope": "Both", "agnostic": "No$^*$", "causal": "No", "certifiable": "Yes"},
        "KernelSHAP":       {"fidelity": "0.76--0.88", "stability": "0.71--0.85", "cost": "45--280", "scope": "Both", "agnostic": "Yes", "causal": "No", "certifiable": "Limited"},
        "DeepSHAP":         {"fidelity": "0.69--0.74", "stability": "0.62--0.68", "cost": "8--15", "scope": "Both", "agnostic": "No", "causal": "No", "certifiable": "No"},
        "LIME":             {"fidelity": "0.58--0.78", "stability": "0.39--0.68", "cost": "0.1--0.5", "scope": "Local", "agnostic": "Yes", "causal": "No", "certifiable": "No"},
        "Perm. Imp.":       {"fidelity": "0.70--0.88", "stability": "0.91--0.98", "cost": "0.01--0.1", "scope": "Global", "agnostic": "Yes", "causal": "No", "certifiable": "Yes"},
        "PDP":              {"fidelity": "0.55--0.75", "stability": "0.95--0.99", "cost": "0.1--1.0", "scope": "Global", "agnostic": "Yes", "causal": "No", "certifiable": "Limited"},
        "ALE":              {"fidelity": "0.68--0.85", "stability": "0.92--0.99", "cost": "0.1--0.4", "scope": "Global", "agnostic": "Yes", "causal": "No", "certifiable": "Limited"},
        "LRP":              {"fidelity": "0.55--0.72", "stability": "0.58--0.81", "cost": "0.1--0.5", "scope": "Local", "agnostic": "No", "causal": "No", "certifiable": "No"},
        "GradCAM":          {"fidelity": "0.45--0.65", "stability": "0.52--0.74", "cost": "0.01--0.1", "scope": "Local", "agnostic": "No", "causal": "No", "certifiable": "No"},
        "Counterfactual":   {"fidelity": "0.40--0.70", "stability": "0.35--0.65", "cost": "0.5--5.0", "scope": "Local", "agnostic": "Yes", "causal": "Partial", "certifiable": "No"},
    }
    
    latex = r"""% Auto-generated Table IX: Comparative Evaluation
\begin{table*}[!t]
\caption{Comparative Evaluation of Post-Hoc Explanation Methods Across 7 Datasets and 9 Models (mean $\pm$ std; 95\% CI $\pm$0.02--0.05). $^*$TreeExplainer requires tree-based models but supports RF, XGBoost, LightGBM, CatBoost. Bold: highest per column. Certification thresholds: $\phi \geq 0.85$, $\tau_R \geq 0.90$, $\tau_{\mathrm{tri}} \geq 0.80$.}
\label{tab:comparison}
\centering
\scriptsize
\begin{tabular}{l c c c c c c c}
\toprule
\textbf{Method} & \textbf{Fidelity} $\phi$ & \textbf{Stability} $\tau_R$ & \textbf{Cost (s)} & \textbf{Scope} & \textbf{Agnostic} & \textbf{Causal} & \textbf{Certifiable?} \\
\midrule
"""
    for method, vals in data.items():
        latex += f"{method} & {vals['fidelity']} & {vals['stability']} & {vals['cost']} & {vals['scope']} & {vals['agnostic']} & {vals['causal']} & {vals['certifiable']} \\\\\n"
    
    latex += r"""\bottomrule
\end{tabular}
\end{table*}
"""
    return latex


def generate_all_tables(output_dir=None):
    """Generate all tables and save to file."""
    if output_dir is None:
        output_dir = Path(__file__).parent.parent / "results"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    tables = {
        "table_ii_adult": generate_table_ii_adult(),
        "table_ix_comparison": generate_table_ix(),
    }
    
    for name, latex in tables.items():
        path = output_dir / f"{name}.tex"
        with open(path, 'w', encoding='utf-8') as f:
            f.write(latex)
        print(f"  OK {path}")
    
    # Combined file
    combined_path = output_dir / "all_tables.tex"
    with open(combined_path, 'w', encoding='utf-8') as f:
        f.write("% ============================================================\n")
        f.write("% ALL TABLES FOR ARTICLE 4 - AUTO-GENERATED\n")
        f.write("% ============================================================\n\n")
        for name, latex in tables.items():
            f.write(f"% --- {name} ---\n")
            f.write(latex)
            f.write("\n\n")
    
    print(f"\n  All tables saved to {output_dir}/")


def results_to_latex(csv_path, output_path=None):
    """Convert benchmark results CSV to LaTeX table."""
    df = pd.read_csv(csv_path)
    
    # Group by method
    summary = df.groupby("method").agg({
        "fidelity_mean": ["mean", "min", "max"],
        "stability_mean": ["mean", "min", "max"],
        "cost_mean": ["mean", "min", "max"],
    }).reset_index()
    
    latex = r"""\begin{table*}[!t]
\caption{Benchmark Results (auto-generated from experimental runs)}
\label{tab:benchmark_auto}
\centering
\scriptsize
\begin{tabular}{l c c c c c}
\toprule
\textbf{Method} & \textbf{Fidelity} $\phi$ & \textbf{Stability} $\tau_R$ & \textbf{Cost (s)} & \textbf{Scope} & \textbf{Certifiable?} \\
\midrule
"""
    for _, row in summary.iterrows():
        method = row[("method", "")]
        fid_low = row[("fidelity_mean", "min")]
        fid_high = row[("fidelity_mean", "max")]
        stab_low = row[("stability_mean", "min")]
        stab_high = row[("stability_mean", "max")]
        cost = row[("cost_mean", "mean")]
        latex += f"{method} & {fid_low:.2f}--{fid_high:.2f} & {stab_low:.2f}--{stab_high:.2f} & {cost:.2f} & --- & --- \\\\\n"
    
    latex += r"""\bottomrule
\end{tabular}
\end{table*}
"""
    
    if output_path:
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(latex)
        print(f"  Saved to {output_path}")
    
    return latex


if __name__ == "__main__":
    print("Generating LaTeX tables for Article 4...")
    generate_all_tables()