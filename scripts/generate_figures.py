#!/usr/bin/env python3
"""
Generate all 4 figures and 5 tables for Article 4.
- Figure 1: radar_comparison.png (6-axis radar)
- Figure 2: fidelity_comparison.png (bar chart with error bars)
- Figure 3: stability_comparison.png (bar chart with error bars)
- Figure 4: decision_framework.png (decision tree diagram)
- Table IX: Comparative evaluation summary
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from pathlib import Path

FIGURES_DIR = Path(__file__).parent.parent / "figures"
RESULTS_DIR = Path(__file__).parent.parent / "results"

# ========================================================================
# DATA FROM ARTICLE 4 (Tables II-VIII and Table IX)
# ========================================================================

# Table IX: Comparative Evaluation
TABLE_IX = {
    "TreeExplainer":    {"fidelity": (0.98, 0.99), "stability": (1.00, 1.00), "cost": (0.3, 0.8), "scope": "Both", "agnostic": "No*", "causal": "No", "certifiable": "Yes"},
    "KernelSHAP":       {"fidelity": (0.76, 0.88), "stability": (0.71, 0.85), "cost": (45, 280), "scope": "Both", "agnostic": "Yes", "causal": "No", "certifiable": "Limited"},
    "DeepSHAP":         {"fidelity": (0.69, 0.74), "stability": (0.62, 0.68), "cost": (8, 15), "scope": "Both", "agnostic": "No", "causal": "No", "certifiable": "No"},
    "LIME":             {"fidelity": (0.58, 0.78), "stability": (0.39, 0.68), "cost": (0.1, 0.5), "scope": "Local", "agnostic": "Yes", "causal": "No", "certifiable": "No"},
    "Perm. Imp.":       {"fidelity": (0.70, 0.88), "stability": (0.91, 0.98), "cost": (0.01, 0.1), "scope": "Global", "agnostic": "Yes", "causal": "No", "certifiable": "Yes"},
    "PDP":              {"fidelity": (0.55, 0.75), "stability": (0.95, 0.99), "cost": (0.1, 1), "scope": "Global", "agnostic": "Yes", "causal": "No", "certifiable": "Limited"},
    "ALE":              {"fidelity": (0.68, 0.85), "stability": (0.92, 0.99), "cost": (0.1, 0.4), "scope": "Global", "agnostic": "Yes", "causal": "No", "certifiable": "Limited"},
    "LRP":              {"fidelity": (0.55, 0.72), "stability": (0.58, 0.81), "cost": (0.1, 0.5), "scope": "Local", "agnostic": "No", "causal": "No", "certifiable": "No"},
    "GradCAM":          {"fidelity": (0.45, 0.65), "stability": (0.52, 0.74), "cost": (0.01, 0.1), "scope": "Local", "agnostic": "No", "causal": "No", "certifiable": "No"},
    "Counterfactual":    {"fidelity": (0.40, 0.70), "stability": (0.35, 0.65), "cost": (0.5, 5), "scope": "Local", "agnostic": "Yes", "causal": "Partial", "certifiable": "No"},
}

# Colors for methods
METHOD_COLORS = {
    "TreeExplainer": "#2196F3", "KernelSHAP": "#64B5F6", "DeepSHAP": "#90CAF9",
    "LIME": "#FF9800", "Perm. Imp.": "#4CAF50", "PDP": "#9C27B0",
    "ALE": "#7B1FA2", "LRP": "#F44336", "GradCAM": "#E91E63", "Counterfactual": "#795548",
}


def generate_radar_comparison(output_dir=FIGURES_DIR):
    """Figure 1: 6-axis radar comparison."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    categories = ['Fidelity', 'Stability', 'Cost\n(inverted)', 'Scope', 'Agnostic', 'Causal']
    N = len(categories)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]
    
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
    
    # Method data (normalized to 0-1 scale)
    methods_data = {
        "TreeExplainer":    [0.99, 1.0, 0.95, 0.75, 0.0, 0.0],
        "KernelSHAP":       [0.82, 0.78, 0.30, 0.75, 1.0, 0.0],
        "LIME":             [0.68, 0.50, 0.90, 0.25, 1.0, 0.0],
        "Perm. Imp.":       [0.79, 0.95, 0.98, 0.50, 1.0, 0.0],
        "ALE":              [0.77, 0.96, 0.90, 0.50, 1.0, 0.0],
        "GradCAM":          [0.55, 0.63, 0.98, 0.25, 0.0, 0.0],
    }
    
    for method, values in methods_data.items():
        values_plot = values + values[:1]
        ax.plot(angles, values_plot, 'o-', linewidth=2, label=method, markersize=6)
        ax.fill(angles, values_plot, alpha=0.1)
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=12)
    ax.set_ylim(0, 1.05)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=9)
    ax.legend(loc='upper right', bbox_to_anchor=(1.35, 1.15), fontsize=11)
    ax.set_title('Six-Axis Radar Comparison of\nPost-Hoc Explanation Methods', fontsize=14, pad=20)
    
    plt.tight_layout()
    plt.savefig(output_dir / "radar_comparison.png", dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / "radar_comparison.pdf", bbox_inches='tight')
    plt.close()
    print(f"  OK radar_comparison.png saved")


def generate_fidelity_comparison(output_dir=FIGURES_DIR):
    """Figure 2: Fidelity scores with certification threshold."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    methods = list(TABLE_IX.keys())
    fidelity_means = [np.mean(TABLE_IX[m]["fidelity"]) for m in methods]
    fidelity_errs = [(np.mean(TABLE_IX[m]["fidelity"]) - TABLE_IX[m]["fidelity"][0],
                     TABLE_IX[m]["fidelity"][1] - np.mean(TABLE_IX[m]["fidelity"])) for m in methods]
    
    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(methods))
    colors = [METHOD_COLORS.get(m, "#888888") for m in methods]
    
    bars = ax.bar(x, fidelity_means, yerr=list(zip(*fidelity_errs)), 
                  color=colors, alpha=0.8, edgecolor='black', linewidth=0.5, capsize=5)
    
    # Certification threshold
    ax.axhline(y=0.85, color='green', linestyle='--', linewidth=2, label='Certification threshold (φ ≥ 0.85)')
    
    ax.set_ylabel('Fidelity (φ)', fontsize=13)
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=30, ha='right', fontsize=11)
    ax.set_ylim(0, 1.1)
    ax.legend(fontsize=11, loc='upper left')
    ax.set_title('Fidelity Scores Across Post-Hoc Methods', fontsize=14)
    ax.grid(axis='y', alpha=0.3)
    
    for bar, val in zip(bars, fidelity_means):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.02, 
                f'{val:.2f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(output_dir / "fidelity_comparison.png", dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / "fidelity_comparison.pdf", bbox_inches='tight')
    plt.close()
    print(f"  OK fidelity_comparison.png saved")


def generate_stability_comparison(output_dir=FIGURES_DIR):
    """Figure 3: Stability scores (Kendall's τ_R) with certification threshold."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    methods = list(TABLE_IX.keys())
    stability_means = [np.mean(TABLE_IX[m]["stability"]) for m in methods]
    stability_errs = [(np.mean(TABLE_IX[m]["stability"]) - TABLE_IX[m]["stability"][0],
                       TABLE_IX[m]["stability"][1] - np.mean(TABLE_IX[m]["stability"])) for m in methods]
    
    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(methods))
    colors = [METHOD_COLORS.get(m, "#888888") for m in methods]
    
    bars = ax.bar(x, stability_means, yerr=list(zip(*stability_errs)),
                  color=colors, alpha=0.8, edgecolor='black', linewidth=0.5, capsize=5)
    
    # Certification threshold
    ax.axhline(y=0.90, color='green', linestyle='--', linewidth=2, label='Certification threshold (τ_R ≥ 0.90)')
    
    ax.set_ylabel("Stability (Kendall's τ_R)", fontsize=13)
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=30, ha='right', fontsize=11)
    ax.set_ylim(0, 1.1)
    ax.legend(fontsize=11, loc='upper left')
    ax.set_title("Stability Scores (Kendall's τ_R) Across Post-Hoc Methods", fontsize=14)
    ax.grid(axis='y', alpha=0.3)
    
    for bar, val in zip(bars, stability_means):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.02,
                f'{val:.2f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(output_dir / "stability_comparison.png", dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / "stability_comparison.pdf", bbox_inches='tight')
    plt.close()
    print(f"  OK stability_comparison.png saved")


def generate_decision_framework(output_dir=FIGURES_DIR):
    """Figure 4: Decision framework diagram."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    fig, ax = plt.subplots(figsize=(10, 12))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 14)
    ax.axis('off')
    
    # Title
    ax.text(5, 13.5, 'Decision Framework for XAI Method Selection', 
            ha='center', fontsize=16, fontweight='bold')
    
    # Decision nodes
    decision_style = dict(boxstyle="round,pad=0.5", facecolor='#E3F2FD', edgecolor='#1976D2', linewidth=2)
    result_style = dict(boxstyle="round,pad=0.5", facecolor='#E8F5E9', edgecolor='#388E3C', linewidth=2)
    
    # Node 1: Data type
    ax.text(5, 12.5, 'What type of data?', fontsize=13, ha='center', 
            bbox=decision_style, fontweight='bold')
    
    # Branches
    branches = [
        (1.5, 11, 'Tabular Data', '#E3F2FD'),
        (5, 11, 'Image Data', '#FFF3E0'),
        (8.5, 11, 'Time-Series', '#F3E5F5'),
    ]
    
    for x, y, label, color in branches:
        ax.text(x, y, label, fontsize=12, ha='center', 
                bbox=dict(boxstyle="round,pad=0.4", facecolor=color, edgecolor='gray'))
        ax.annotate('', xy=(x, y+0.6), xytext=(5, 12.2),
                    arrowprops=dict(arrowstyle='->', color='gray', lw=1.5))
    
    # Tabular sub-decisions
    tab_results = [
        (1.5, 9.5, 'Tree Models?\n→ TreeExplainer\n+ Perm. Imp. + ALE\n(Grade A: τ≥0.9)', '#C8E6C9'),
        (4, 9.5, 'Any Model?\n→ KernelSHAP + ALE\n+ Perm. Imp.\n(Verify τ_R ≥ 0.85)', '#FFF9C4'),
    ]
    
    for x, y, label, color in tab_results:
        ax.text(x, y, label, fontsize=10, ha='center', va='center',
                bbox=dict(boxstyle="round,pad=0.4", facecolor=color, edgecolor='#388E3C', linewidth=1.5))
    
    # Image
    ax.text(5, 8, 'GradCAM + validation\n(cross-check with occlusion)', fontsize=10, ha='center',
            bbox=dict(boxstyle="round,pad=0.4", facecolor='#FFF9C4', edgecolor='#F57F17'))
    
    # Time-series
    ax.text(7.5, 9.5, 'KernelSHAP\n(time-aware)\n+ ALE + LSTM validation', fontsize=10, ha='center',
            bbox=dict(boxstyle="round,pad=0.4", facecolor='#F3E5F5', edgecolor='#7B1FA2'))
    
    # High-stakes decision
    ax.text(5, 6.5, 'HIGH-STAKES DECISION? → Multi-method triangulation\n(SHAP + LIME + Perm. Imp., τ_tri ≥ 0.80)', 
            fontsize=11, ha='center', fontweight='bold',
            bbox=dict(boxstyle="round,pad=0.5", facecolor='#FFEBEE', edgecolor='#C62828', linewidth=2))
    
    # Certification thresholds
    ax.text(5, 4.5, 'CERTIFICATION THRESHOLDS', fontsize=12, ha='center', fontweight='bold',
            bbox=dict(boxstyle="round,pad=0.3", facecolor='#E3F2FD', edgecolor='#1565C0', linewidth=2))
    
    ax.text(5, 3.3, '• Fidelity: φ ≥ 0.85\n• Stability: τ_R ≥ 0.90\n• Triangulation: τ_tri ≥ 0.80',
            fontsize=11, ha='center', va='center',
            bbox=dict(boxstyle="round,pad=0.4", facecolor='white', edgecolor='#1565C0'))
    
    # Non-certifiable methods
    ax.text(5, 1.5, 'NON-CERTIFIABLE (standalone): LIME, LRP, GradCAM, Counterfactual',
            fontsize=10, ha='center', color='#C62828',
            bbox=dict(boxstyle="round,pad=0.3", facecolor='#FFEBEE', edgecolor='#C62828'))
    
    plt.tight_layout()
    plt.savefig(output_dir / "decision_framework.png", dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / "decision_framework.pdf", bbox_inches='tight')
    plt.close()
    print(f"  OK decision_framework.png saved")


def generate_all_figures(output_dir=FIGURES_DIR):
    """Generate all 4 figures."""
    print("\nGenerating figures...")
    generate_radar_comparison(output_dir)
    generate_fidelity_comparison(output_dir)
    generate_stability_comparison(output_dir)
    generate_decision_framework(output_dir)
    print(f"\nAll figures saved to {output_dir}/")


def generate_latex_table_ix():
    """Generate LaTeX code for Table IX."""
    print("\n" + "="*80)
    print("TABLE IX: Comparative Evaluation of Post-Hoc Explanation Methods")
    print("="*80)
    
    header = """\\begin{table*}[!t]
\\caption{Comparative Evaluation of Post-Hoc Explanation Methods}
\\label{tab:comparison}
\\centering
\\scriptsize
\\begin{tabular}{l c c c c c c c}
\\toprule
\\textbf{Method} & \\textbf{Fidelity} & \\textbf{Stability} & \\textbf{Cost (s)} & \\textbf{Scope} & \\textbf{Agnostic} & \\textbf{Causal} & \\textbf{Certifiable?} \\\\
\\midrule"""
    
    rows = []
    for method, data in TABLE_IX.items():
        fid = f"{data['fidelity'][0]:.2f}--{data['fidelity'][1]:.2f}"
        stab = f"{data['stability'][0]:.2f}--{data['stability'][1]:.2f}"
        cost = f"{data['cost'][0]:.2f}--{data['cost'][1]:.1f}"
        rows.append(f"{method} & {fid} & {stab} & {cost} & {data['scope']} & {data['agnostic']} & {data['causal']} & {data['certifiable']} \\\\")
    
    footer = """\\bottomrule
\\end{tabular}
\\\\ \\vspace{2mm}
\\scriptsize $^*$TreeExplainer requires tree-based models but supports RF, XGBoost, LightGBM, CatBoost.
\\end{table*}"""
    
    print(header)
    for row in rows:
        print(row)
    print(footer)


if __name__ == "__main__":
    generate_all_figures()
    generate_latex_table_ix()