# BENCHMARK-XAI-ARTICLE4

Code de reproduction pour l'Article 4 :

**"A Comparative Evaluation of Post-Hoc Explanation Methods for Predictive Modeling:**
**Strengths, Limitations, and Selection Guidelines"**

Auteurs : Ralinirina, Ralaivao, Ralaivao, Ratovondrahona, Mahatody

---

## Structure

```
BENCHMARK-XAI-ARTICLE4/
├── README.md                         # Ce fichier
├── requirements.txt                  # Dépendances Python
├── scripts/
│   ├── run_benchmark.py              # Script principal d'orchestration
│   ├── dataloaders.py                # Chargeurs 7 datasets
│   ├── explainers.py                 # 10 wrappers XAI (SHAP×3, LIME, PermImp, PDP, ALE, LRP, GradCAM, Counterfactual)
│   ├── deep_explainers.py            # DeepSHAP, GradCAM, LRP (PyTorch)
│   ├── pytorch_models.py             # ResNet-18, LSTM, MLP (PyTorch)
│   ├── metrics.py                    # Fidelity, Stability, Cost, Triangulation
│   ├── generate_figures.py           # 4 figures de l'article
│   ├── generate_tables.py            # Tables LaTeX (II–IX)
│   └── preprocess_datasets.py        # Préprocessing ISIC, PhysioNet, Électricité
├── data/                             # Datasets (auto-téléchargés ou manuels)
├── models/                           # Modèles entraînés (cache)
├── results/                          # Résultats CSV/JSON/LaTeX
└── figures/                          # Figures PNG/PDF
```

## Installation

```bash
pip install -r requirements.txt
```

Dépendances optionnelles :
```bash
pip install torch torchvision          # Pour CNN/ResNet-18 + DeepSHAP
pip install wfdb                        # Pour PhysioNet ECG
pip install interpret                    # Pour EBM (ExplainableBoostingMachine)
```

## Utilisation

### Benchmark rapide (test, ~2 min)
```bash
python scripts/run_benchmark.py --quick
```

### Benchmark tabulaire seulement (~8h CPU)
```bash
python scripts/run_benchmark.py --tabular
```

### Benchmark complet (~48h GPU + 120h CPU)
```bash
python scripts/run_benchmark.py
```

### Générer les figures (à partir des données de l'article)
```bash
python scripts/run_benchmark.py --figures
```

### Préprocessing des datasets manuels
```bash
# ISIC 2019 (téléchargement manuel requis)
python scripts/preprocess_datasets.py --dataset isic2019

# PhysioNet ECG (nécessite wfdb)
python scripts/preprocess_datasets.py --dataset physionet

# Électricité (auto-téléchargement)
python scripts/preprocess_datasets.py --dataset electricity

# Tous les datasets
python scripts/preprocess_datasets.py --dataset all
```

## Résultats attendus (Article 4)

### Table IX : Évaluation comparative

| Méthode | Fidelity (φ) | Stability (τ_R) | Cost (s) | Scope | Agnostic | Causal | Certifiable? |
|---------|-------------|-----------------|----------|-------|----------|--------|-------------|
| TreeExplainer | 0.98–0.99 | 1.00–1.00 | 0.3–0.8 | Both | No* | No | **Yes** |
| KernelSHAP | 0.76–0.88 | 0.71–0.85 | 45–280 | Both | Yes | No | Limited |
| DeepSHAP | 0.69–0.74 | 0.62–0.68 | 8–15 | Both | No | No | No |
| LIME | 0.58–0.78 | 0.39–0.68 | 0.1–0.5 | Local | Yes | No | No |
| Perm. Imp. | 0.70–0.88 | 0.91–0.98 | 0.01–0.1 | Global | Yes | No | **Yes** |
| PDP | 0.55–0.75 | 0.95–0.99 | 0.1–1 | Global | Yes | No | Limited |
| ALE | 0.68–0.85 | 0.92–0.99 | 0.1–0.4 | Global | Yes | No | Limited |
| LRP | 0.55–0.72 | 0.58–0.81 | 0.1–0.5 | Local | No | No | No |
| GradCAM | 0.45–0.65 | 0.52–0.74 | 0.01–0.1 | Local | No | No | No |
| Counterfactual | 0.40–0.70 | 0.35–0.65 | 0.5–5 | Local | Yes | Partial | No |

*TreeExplainer requires tree-based models but supports RF, XGBoost, LightGBM, CatBoost.

### Certification Thresholds

- **Fidelity**: φ ≥ 0.85
- **Stability**: τ_R ≥ 0.90
- **Triangulation**: τ_tri ≥ 0.80

### Méthodes certifiables (standalone)
1. **TreeExplainer** (τ_R = 1.00) — seulement pour modèles à base d'arbres
2. **Permutation Importance** (τ_R ≥ 0.91) — model-agnostic, global

### Méthodes nécessitant triangulation
- KernelSHAP + ALE + Perm. Imp. → τ_tri ≥ 0.80

## 8 Méthodes × 9 Modèles × 7 Datasets

### Datasets
| Dataset | Type | n | p | Tâche |
|---------|------|---|---|-------|
| Adult Census | Tabular | 48,842 | 14 | Classification |
| German Credit | Tabular | 1,000 | 20 | Classification |
| California Housing | Tabular | 20,640 | 8 | Régression |
| CIFAR-10 | Image | 60,000 | 3×32×32 | Classification |
| ISIC 2019 | Image | 25,331 | 3×224×224 | Classification |
| PhysioNet | Time-series | 40,336 | 40 | Classification |
| Electricity | Time-series | 45,312 | 12 | Classification |

### Modèles
- **Tabulaires**: Logistic/Ridge, Decision Tree, EBM, Random Forest, XGBoost, LightGBM, CatBoost, MLP
- **Images**: CNN (ResNet-18)
- **Time-series**: LSTM

### Méthodes XAI
- **SHAP**: TreeExplainer, KernelSHAP, DeepSHAP
- **LIME**: Tabular, Image
- **Permutation Importance**
- **PDP** (Partial Dependence Plots)
- **ALE** (Accumulated Local Effects)
- **LRP** (Layer-Wise Relevance Propagation)
- **GradCAM** (Gradient-weighted Class Activation Mapping)
- **Counterfactual Explanations**

## 6 Axes d'évaluation

1. **Fidelity** (φ) : accord top-k entre explication et ground-truth par ablation
2. **Stability** (τ_R) : Kendall's τ sur 50 bootstrap resamples
3. **Computational Cost** : temps par explication (secondes)
4. **Scope** : local / global / les deux
5. **Agnosticity** : model-agnostic ou model-specific
6. **Causality** : garanties causales (aucune méthode ne les offre complètement)

## Citation

```bibtex
@article{ralinirina2024comparative,
  title={A Comparative Evaluation of Post-Hoc Explanation Methods for Predictive Modeling: Strengths, Limitations, and Selection Guidelines},
  author={Ralinirina, Rosa Elysabeth and Ralaivao, Jean Christian and Ralaivao, Niaiko Michaël and Ratovondrahona, Alain Josué and Mahatody, Thomas},
  journal={IEEE Access},
  year={2024}
}
```

## License

MIT License — Code will be released upon article acceptance.