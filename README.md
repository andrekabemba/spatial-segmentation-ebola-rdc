# Spatial Segmentation Ebola - RDC

Ce projet vise à réaliser une segmentation spatio-temporelle et éco-épidémiologique des zones de santé affectées par Ébola dans les provinces du Nord-Kivu et de l'Ituri (RDC).

## Structure du Projet

- `data/` : Contient les jeux de données bruts et préparés.
- `notebooks/` : Notebooks Jupyter pour l'exploration et le clustering.
- `src/` : Code source (modèles de clustering).
- `models/` : Modèles sauvegardés au format Pickle (.pkl).
- `main.py` : Script principal pour exécuter tout le pipeline.

## Installation

Assurez-vous d'avoir Python installé ainsi que les bibliothèques suivantes :
```bash
pip install pandas numpy scikit-learn matplotlib seaborn fastapi uvicorn flask requests
```

## Utilisation

### 1. Entraîner le Modèle
```bash
python main.py
```

### 2. Lancer l'Architecture de Prédiction
Ce projet utilise une architecture découplée :
- **Backend (API)** : Gère la logique ML avec FastAPI.
- **Frontend (Web)** : Interface utilisateur avec Flask.

**Étape A : Lancer l'API (Backend)**
```bash
python api.py
```
*(L'API sera disponible sur http://127.0.0.1:8000)*

**Étape B : Lancer l'Interface (Frontend)**
Ouvrez un nouveau terminal et lancez :
```bash
python flask_app.py
```
*(L'interface sera disponible sur http://127.0.0.1:5000)*

Les résultats seront sauvegardés dans `data/segmentation_resultats.csv` et le modèle dans `models/ebola_clustering_final.pkl`.

## Méthodologie

Le projet utilise deux approches de clustering non supervisé :
1. **K-Means** : Pour une partition robuste basée sur les centroïdes.
2. **Clustering Hiérarchique Ascendant (CHA)** : Pour comprendre les relations structurelles profondes.

Les zones sont classifiées en profils de vulnérabilité :
- **Épicentre Urbain** : Forte densité et charge épidémique élevée.
- **Réservoir Sylvatique** : Forte couverture forestière (risque de spillover).
- **Interface Lacustre** : Proximité des eaux et flux migratoires.

## Auteur
Projet corrigé et complété pour l'examen ML.
