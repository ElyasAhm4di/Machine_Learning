# Machine Learning — apprentissage pas à pas

Dépôt où je regroupe mes exercices de machine learning en Python, du plus simple au plus avancé. Chaque dossier correspond à un projet.

## Projets

| Dossier | Sujet | Statut |
|---|---|---|
| `Firs_project/` | Prédire le résultat d'un match de foot à partir de ses stats | ✅ Terminé |

## Projet 1 — Prédiction de résultats de matchs

**Objectif :** deviner si un match se termine par une victoire à domicile, un nul ou une victoire à l'extérieur, à partir de 7 stats : possession, tirs, tirs cadrés et corners.

**Données :** 1 000 matchs simulés avec des règles réalistes. Les buts sont exclus des features pour éviter une fuite de données.

**Démarche :**
1. Exploration des données (répartition des résultats, moyennes par résultat)
2. Découpe train / test (80 / 20, stratifiée)
3. Baseline avec un modèle naïf
4. Régression logistique et arbre de décision
5. Analyse des erreurs avec une matrice de confusion
6. Mise en évidence du surapprentissage et recherche de la meilleure profondeur d'arbre
7. Prédiction sur des matchs inventés

**Résultats (accuracy sur le test) :**

| Modèle | Score |
|---|---|
| Baseline (toujours « victoire à domicile ») | 0.47 |
| Régression logistique | 0.585 |
| Arbre de décision (profondeur 3) | 0.615 |
| Arbre sans limite (surapprentissage) | 0.495 |

**Ce que j'en retiens :** les nuls sont très difficiles à prédire, et un modèle plus simple généralise mieux qu'un modèle qui apprend par cœur.

## Lancer un projet

```bash
pip install numpy pandas scikit-learn
python Firs_project/ML1.py
```

## Outils

Python · pandas · NumPy · scikit-learn

## Note

Des IA (Claude) m'ont servi d'aide pendant l'apprentissage : explications des notions, indices quand je bloquais, et génération des données simulées. Le code d'analyse et des modèles a été écrit par moi.
