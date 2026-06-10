# 🚀 GitHub Setup Instructions

## 1. Create GitHub Repository

Go to https://github.com/new and create a repo:
- **Name**: `xai-benchmark-article` (already exists)
- **Description**: `Benchmark code for "A Comparative Evaluation of Post-Hoc Explanation Methods for Predictive Modeling" (IEEE Access 2026)`
- **Visibility**: Public

## 2. Push the Code

```bash
cd BENCHMARK-XAI-ARTICLE4

# Add all new files
git add app.py precomputed_results.py translations.py scripts/instant_benchmark.py
git add README.md requirements.txt .gitignore

# Commit
git commit -m "Add Streamlit dashboard, instant benchmark, bilingual FR/EN support"

# Push
git push origin main
```

## 3. Deploy on Streamlit Cloud

1. Go to [share.streamlit.io](https://share.streamlit.io/)
2. Connect your GitHub account
3. Select the `RosaElysabeth/xai-benchmark-article` repo
4. Set main file path: `app.py`
5. Set Python version: 3.10+
6. Click "Deploy"

The app uses pre-computed results, so it loads instantly on Streamlit Cloud.

## 4. Verify

Open https://github.com/RosaElysabeth/xai-benchmark-article
- README.md should display with all the tables
- `app.py` should be visible
- `precomputed_results.py` contains reference data
- `translations.py` contains FR/EN translations
- Figures in `figures/`

## 5. Streamlit Cloud URL

After deployment, the app will be available at:
`https://xai-benchmark-article.streamlit.app/`
