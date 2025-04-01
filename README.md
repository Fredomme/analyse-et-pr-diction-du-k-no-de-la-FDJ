# 🎯 Analyse et Prédiction du Kéno FDJ avec IA, Statistiques et Simulations

Projet personnel de recherche, modélisation et prédiction autour des tirages du **Kéno de la Française des Jeux (FDJ)** à l'aide de **statistiques avancées**, **intelligence artificielle (IA)**, **réseaux de neurones (LSTM, Transformers)** et **simulations Monte Carlo**.

---

## 📌 Objectifs

- Reconstituer un modèle prédictif des tirages Kéno à partir des historiques FDJ
- Identifier d'éventuels biais ou régularités dans les données
- Simuler et évaluer des stratégies de sélection de numéros
- Explorer la possibilité de reconstruire le générateur pseudo-aléatoire utilisé par la FDJ
- Créer un pipeline complet d’analyse, d'entraînement et de prédiction

---

## 🧠 Contenu du projet

| Fichier / dossier | Rôle |
|-------------------|------|
| `main.py` | Exécution complète du pipeline (prétraitement, entraînement, prédiction) |
| `main_train.py` | Entraînement des modèles LSTM et Transformer |
| `main_eval.py` | Évaluation des modèles et application du filtre FDJ |
| `main_preproc.py` | Prétraitement des tirages, extraction des features |
| `data_loader.py` | Chargement, encodage, extraction des tirages |
| `models.py` | Modèles PyTorch utilisés (LSTM, Transformer) |
| `predictor.py` | Fonctions d'entraînement et de prédiction |
| `metrics.py` | Métriques personnalisées (Jaccard, Precision@20, Recall@20) |
| `fdj_filter.py` | Filtres statistiques appliqués aux prédictions finales |
| `config.yaml` | Paramètres globaux (entraînement, structure, etc.) |

---

## 🛠️ Technologies utilisées

- Python 3.12
- PyTorch
- Pandas / NumPy
- Matplotlib / Seaborn
- YAML
- Git + SSH
- Kali Linux 😎

---

## 🚀 Installation

```bash
git clone git@github.com:Fredomme/analyse-et-pr-diction-du-k-no-de-la-FDJ.git
cd analyse-et-pr-diction-du-k-no-de-la-FDJ
python3 -m venv keno_env
source keno_env/bin/activate
pip install -r requirements.txt
