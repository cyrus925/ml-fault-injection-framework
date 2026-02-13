# MLOps Resilience Lab

**MLOps Resilience Lab** est un framework Python permettant d’évaluer la **robustesse des pipelines ML** via :

- l’ingestion et transformation des données (architecture en médaillon : Bronze → Silver → Gold)  
- la création de features et le training d’un modèle prédictif (Random Forest)  
- l’injection de fautes contrôlées sur les données et le pipeline ML  
- le logging unifié pour suivre l’impact des anomalies sur le modèle et les prédictions  

Ce projet a été conçu pour démontrer  **la résilience des pipelines ML** face à des erreurs de données, des corruptions, ou des changements de schéma, dans un contexte DevSecMLOps.

Dans ce projet, nous nous intéressons à la prédiction de la valeur marchande d’un joueur de football à partir de ses attributs personnels et contextuels. Cette valeur est une estimation commerciale largement utilisée par les clubs, les agents et les analystes pour évaluer l'importance et la performance d’un joueur sur le marché des transferts.

 
Nous exploitons pour cela un jeu de données complet issu du dataset [Football Data From Transfermarkt](https://www.kaggle.com/datasets/davidcariboo/player-scores), disponible sur Kaggle et comprenant plusieurs fichiers CSV décrivant les joueurs, leurs caractéristiques, leurs clubs, les compétitions et l’historique des valorisations au fil du temps.

## Architecture
```
data/
├── raw/                     # Données brutes CSV (input)
├── silver/                  # Données transformées après nettoyage et calcul d'attributs
├── gold/                    # Données prêtes pour le ML (production)
├── features/                # Features vectorisées (X pour le modèle)
├── models/                  # Modèles entraînés
├── ml/
│   ├── fault/               # Données corrompues / injectées pour tests de robustesse
│   ├── features/            # Features pour le modèle ML à partir du gold
│   ├── models/              # Modèles entraînés sur features
│   └── predict/             # Outputs / prédictions du modèle
│
ingestion/
├── schema/                  # Schemas YAML pour chaque couche / table
├── bronze_to_silver.py      # Transformation des données brutes → Silver
├── inject_players_fault.py  # Injection de fautes dans les données gold + logging
├── silver_to_gold.py        # Transformation Silver → Gold (prête pour ML)
├── log_ingestion.py         # Logger ingestion / pipelines data
├── utilites.py              # Fonctions utilitaires pour ingestion et transformations
│
logs/
├── ingestion_log.csv        # Logs ingestion (bronze → gold)
├── ml_log.csv               # Logs ML (features, training, predict)
├── fault_log.csv            # Logs Fault Injection (données corrompues)
│
ml/
├── schema/                  # Schema YAML des features pour chaque table
├── create_features.py       # Création des features à partir du gold
├── fit_model.py             # Entraînement et évaluation du modèle ML
├── log_ml.py                # Logger pour les étapes ML
└── predict.py               # Génération des prédictions
orchestrator.py              # Lancement de la pipeline
```

#  Fonctionnalités principales

### 1 - Pipeline Data

- Lecture des fichiers CSV
- Transformation Raw → Silver → Gold
- Calcul d’attributs dérivés (ex: `age` depuis `date_of_birth`)
- Validation de schéma et logging des anomalies

### 2 - Feature Engineering

- Sélection de features numériques et catégorielles
- Encodage des variables catégorielles
- Construction de dataset prêt à l’entraînement
- Logging complet des statistiques 

### 3 - Training & Evaluation

- Entraînement d’un modèle `RandomForestRegressor`
- Séparation train/test
- Calcul de métriques (RMSE, MAE)
- Logging des performances du modèle

### 4 - Fault Injection Engine

- Corruption aléatoire des valeurs (`age`, `height`, `position`, etc.)
- Suppression de colonnes
- Drift simulé et anomalies
- Logging détaillé 

## Installation

### En local
```bash
git clone https://github.com/cyrus925/ml-fault-injection-framework.git
cd mlops-resilience-lab
python -m venv venv
source venv/bin/activate  # Linux / Mac
venv\Scripts\activate     # Windows
pip install -r requirements.txt
```

Assurez-vous que vos fichiers CSV sont dans data/raw/.

Pour lancer le pipeline :
```bash
python orchestrator.py
```

### Avec Docker

```bash
docker build -t python-app .    
docker run -it `
  -v ${PWD}/data:/fault-injection-app/data `
  -v ${PWD}/logs:/fault-injection-app/logs `
  python-app
```


## Améliorations & Perspectives

### BI et monitoring avancé

Toutes les étapes du pipeline sont loggées dans des fichiers CSV/Parquet (ingestion_log.csv, ml_log.csv, fault_log.csv). Ces logs peuvent être directement visualisés et explorés dans Power BI ou tout outil de BI pour :

- suivre la qualité des données en temps réel

- mesurer visuellement l’impact des anomalies injectées


### Archive & historisation

Chaque dataset (raw, silver, gold, features, modèles) peut être historisé pour une raison de maintenance. L'archive est une solution pour économiser de la mémoire.



### Exploiter plus que la table players

Actuellement, le modèle utilise seulement la table players. La base de données est riche et des liens entre les tables sont possible.
L'ajout de ces tables permettraient un modèle de machine learning plus efficace.

