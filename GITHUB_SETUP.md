# 🚀 Instructions pour créer le repo GitHub

## 1. Créer le repo sur GitHub

Aller sur https://github.com/new et créer un repo :
- **Nom** : `xai-benchmark-article4`
- **Description** : `Benchmark code for "A Comparative Evaluation of Post-Hoc Explanation Methods for Predictive Modeling" (IEEE Access 2026)`
- **Visibilité** : Public
- **Ne PAS** initialiser avec README (on a déjà tout)

## 2. Push le code local

```bash
cd "03-DONNEES-RECHERCHE/BENCHMARK-XAI-ARTICLE4"

# Ajouter le remote
git remote add origin https://github.com/RosaElysabeth/xai-benchmark-article4.git

# Push
git branch -M main
git push -u origin main
```

## 3. Vérifier

Ouvrir https://github.com/RosaElysabeth/xai-benchmark-article4
- README.md doit s'afficher
- Les scripts doivent être dans `scripts/`
- Les figures dans `figures/`