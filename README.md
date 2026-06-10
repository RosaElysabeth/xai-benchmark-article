# XAI Benchmark — Article 4

> **A Comparative Evaluation of Post-Hoc Explanation Methods for Predictive Modeling: Strengths, Limitations, and Selection Guidelines**

Authors: Ralinirina, Ralaivao, Ralaivao, Ratovondrahona, Mahatody

---

## 🚀 Live Demo

[![Streamlit App](https://static.streamlit.io/badges/streamlit_default_badge.svg)](https://xai-benchmark-article4.streamlit.app/)

Interactive dashboard with pre-computed results (instant), bilingual (FR/EN), and live benchmark demo.

---

## 📊 Key Results

### Table IX: Comparative Evaluation

| Method | Fidelity (φ) | Stability (τ_R) | Cost (s) | Scope | Agnostic | Certifiable? |
|--------|-------------|-----------------|----------|-------|----------|-------------|
| TreeExplainer | 0.98–0.99 | 1.00–1.00 | 0.3–0.8 | Both | No* | **Yes** |
| KernelSHAP | 0.76–0.88 | 0.71–0.85 | 45–280 | Both | Yes | Limited |
| DeepSHAP | 0.69–0.74 | 0.62–0.68 | 8–15 | Both | No | No |
| LIME | 0.58–0.78 | 0.39–0.68 | 0.1–0.5 | Local | Yes | No |
| Perm. Imp. | 0.70–0.88 | 0.91–0.98 | 0.01–0.1 | Global | Yes | **Yes** |
| PDP | 0.55–0.75 | 0.95–0.99 | 0.1–1.0 | Global | Yes | Limited |
| ALE | 0.68–0.85 | 0.92–0.99 | 0.1–0.4 | Global | Yes | Limited |
| LRP | 0.55–0.72 | 0.58–0.81 | 0.1–0.5 | Local | No | No |
| GradCAM | 0.45–0.65 | 0.52–0.74 | 0.01–0.1 | Local | No | No |
| Counterfactual | 0.40–0.70 | 0.35–0.65 | 0.5–5.0 | Local | Yes | No |

*TreeExplainer requires tree-based models but supports RF, XGBoost, LightGBM, CatBoost.

### Certification Thresholds

- **Fidelity**: φ ≥ 0.85
- **Stability**: τ_R ≥ 0.90
- **Triangulation**: τ_tri ≥ 0.80

### Certifiable Methods (standalone)

1. **TreeExplainer** (τ_R = 1.00) — only for tree-based models
2. **Permutation Importance** (τ_R ≥ 0.91) — model-agnostic, global

Triangulation: **KernelSHAP + ALE + Perm. Imp.** → τ_tri ≥ 0.80

---

## 🏗️ Project Structure

```
xai-benchmark-article/
├── app.py                          # Streamlit dashboard (bilingual FR/EN)
├── precomputed_results.py          # Article reference data (instant)
├── translations.py                 # FR/EN translations
├── requirements.txt                 # Dependencies
├── README.md                        # This file
├── scripts/
│   ├── run_benchmark.py             # Full benchmark (~48h GPU + 120h CPU)
│   ├── instant_benchmark.py         # Quick demo (~2 min)
│   ├── fast_benchmark.py            # Tabular-only benchmark (~8h)
│   ├── dataloaders.py               # 7 dataset loaders
│   ├── explainers.py                # 10 XAI method wrappers
│   ├── deep_explainers.py           # DeepSHAP, GradCAM, LRP (PyTorch)
│   ├── pytorch_models.py            # ResNet-18, LSTM, MLP (PyTorch)
│   ├── metrics.py                   # Fidelity, Stability, Cost, Triangulation
│   ├── generate_figures.py          # 4 article figures
│   ├── generate_tables.py           # LaTeX tables
│   └── preprocess_datasets.py        # ISIC, PhysioNet, Electricity preprocessing
├── figures/                          # Article figures (PNG + PDF)
├── results/                          # Generated results
└── data/                             # Datasets (auto-downloaded or manual)
```

---

## ⚡ Quick Start

### Install

```bash
pip install -r requirements.txt
```

### Launch Dashboard (Instant)

```bash
streamlit run app.py
```

The dashboard shows **pre-computed article results instantly** — no computation needed.

### Run Quick Demo (~2 min)

```bash
python scripts/instant_benchmark.py
```

### Run Tabular Benchmark (~8h)

```bash
python scripts/run_benchmark.py --tabular
```

### Run Full Benchmark (~48h GPU + 120h CPU)

```bash
python scripts/run_benchmark.py
```

### Generate Figures

```bash
python scripts/run_benchmark.py --figures
```

---

## 🧪 Evaluation Framework

### 6 Evaluation Axes

| Axis | Metric | Description |
|------|--------|-------------|
| **Fidelity** | φ | Top-k agreement with ablation ground truth |
| **Stability** | τ_R | Kendall's τ over 50 bootstrap resamples |
| **Cost** | seconds | Wall-clock time per explanation |
| **Scope** | Local/Global/Both | Explanation granularity |
| **Agnosticity** | Yes/No | Model-agnostic or model-specific |
| **Causality** | Yes/No/Partial | Causal guarantees |

### 8 Methods × 9 Models × 7 Datasets

**Datasets**: Adult Census, German Credit, California Housing, CIFAR-10, ISIC 2019, PhysioNet ECG, Electricity

**Models**: Logistic/Ridge, Decision Tree, EBM, Random Forest, XGBoost, LightGBM, CatBoost, MLP, ResNet-18, LSTM

**XAI Methods**: TreeExplainer, KernelSHAP, DeepSHAP, LIME, Permutation Importance, PDP, ALE, LRP, GradCAM, Counterfactual

---

## 🌐 Deployment on Streamlit Cloud

1. Push this repo to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io/)
3. Connect your GitHub repo
4. Set main file: `app.py`
5. Set Python version: 3.10+
6. Deploy!

The app uses only pre-computed results by default, so it loads instantly on Streamlit Cloud.

---

## 📝 Citation

```bibtex
@article{ralinirina2026comparative,
  title={A Comparative Evaluation of Post-Hoc Explanation Methods for Predictive Modeling: Strengths, Limitations, and Selection Guidelines},
  author={Ralinirina, Rosa Elysabeth and Ralaivao, Jean Christian and Ralaivao, Niaiko Michaël and Ratovondrahona, Alain Josué and Mahatody, Thomas},
  journal={IEEE Access},
  note={Submitted},
  year={2026}
}
```

---

## 📄 License

MIT License — Code released upon article acceptance.