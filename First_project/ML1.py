import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier


#  Données : une saison inventée de 1000 matchs
# Ce bloc de génération des données a été écrit par Claude
# Tout le reste du code (exploration, modèles, analyse)
rng = np.random.default_rng(42)
n = 1000
possession = np.clip(rng.normal(52, 8, n), 30, 75).round(0)
force = (possession - 50) / 10
tirs_dom = rng.poisson(np.clip(12 + 3*force, 3, None))
tirs_ext = rng.poisson(np.clip(10 - 3*force, 3, None))
cadres_dom = rng.binomial(tirs_dom, 0.38)
cadres_ext = rng.binomial(tirs_ext, 0.35)
corners_dom = rng.poisson(np.clip(5 + 1.5*force, 1, None))
corners_ext = rng.poisson(np.clip(4.5 - 1.5*force, 1, None))
buts_dom = rng.binomial(cadres_dom, 0.32)
buts_ext = rng.binomial(cadres_ext, 0.32)
resultat = np.where(buts_dom > buts_ext, "victoire_dom",
           np.where(buts_dom < buts_ext, "victoire_ext", "nul"))
df = pd.DataFrame({"possession_dom": possession, "tirs_dom": tirs_dom,
    "tirs_ext": tirs_ext, "cadres_dom": cadres_dom, "cadres_ext": cadres_ext,
    "corners_dom": corners_dom, "corners_ext": corners_ext,
    "buts_dom": buts_dom, "buts_ext": buts_ext, "resultat": resultat})


#  Exploration : victoires à domicile et leurs tirs cadrés
def stats_victoires_dom(matchs):
    nb_victoires = 0
    total_cadres = 0
    for res, cadres in zip(matchs["resultat"], matchs["cadres_dom"]):
        if res == "victoire_dom":
            nb_victoires = nb_victoires + 1
            total_cadres = total_cadres + cadres

    if nb_victoires == 0:  # évite la division par zéro
        return 0, 0

    moyenne = round(total_cadres / nb_victoires, 3)
    return nb_victoires, moyenne

print(stats_victoires_dom(df))


# --- X = les stats, y = le résultat (pas les buts, sinon c'est de la triche) ---
features = ["possession_dom", "tirs_dom", "tirs_ext", "cadres_dom",
            "cadres_ext", "corners_dom", "corners_ext"]
x = df[features]
y = df["resultat"]

# 800 matchs pour apprendre, 200 pour l'examen
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.20, random_state=42, stratify=y)


#  Le modèle bête : il dit toujours "victoire_dom"
modele_bete = DummyClassifier(strategy="most_frequent")
modele_bete.fit(x_train, y_train)
pred_bete = modele_bete.predict(x_test)
print("Bête :", accuracy_score(y_test, pred_bete))


#  Régression logistique : un poids par stat
reg_log = LogisticRegression(max_iter=800)
reg_log.fit(x_train, y_train)
pred_reg = reg_log.predict(x_test)
print("Régression :", accuracy_score(y_test, pred_reg))


#  Arbre limité à 3 questions
arbre = DecisionTreeClassifier(max_depth=3, random_state=42)
arbre.fit(x_train, y_train)
pred_arbre = arbre.predict(x_test)
print("Arbre (prof. 3) :", accuracy_score(y_test, pred_arbre))


# Où la régression se trompe (lignes = vrai, colonnes = deviné)
print(confusion_matrix(y_test, pred_reg,
                       labels=["victoire_dom", "nul", "victoire_ext"]))


# --- Arbre sans limite : il apprend par cœur ---
arbre_libre = DecisionTreeClassifier(random_state=42)
arbre_libre.fit(x_train, y_train)
pred_libre_test = arbre_libre.predict(x_test)
pred_libre_train = arbre_libre.predict(x_train)
print("test :", accuracy_score(y_test, pred_libre_test))
print("train :", accuracy_score(y_train, pred_libre_train))


# --- On teste chaque profondeur pour trouver la meilleure ---
for profondeur in range(1, 11):
    arbre_essai = DecisionTreeClassifier(random_state=42, max_depth=profondeur)
    arbre_essai.fit(x_train, y_train)
    score_test = accuracy_score(y_test, arbre_essai.predict(x_test))
    score_train = accuracy_score(y_train, arbre_essai.predict(x_train))
    print("Profondeur", profondeur, "| test :", score_test, "| train :", score_train)


#  Deux matchs inventés
# Le domicile écrase le match
match_domine = pd.DataFrame({
    "possession_dom": [75],
    "tirs_dom": [15],
    "tirs_ext": [3],
    "cadres_dom": [12],
    "cadres_ext": [1],
    "corners_dom": [20],
    "corners_ext": [2]
})
print(arbre.classes_)  # l'ordre des probas
print(arbre.predict_proba(match_domine))

# Un match à peu près équilibré
match_equilibre = pd.DataFrame({
    "possession_dom": [51],
    "tirs_dom": [10],
    "tirs_ext": [10],
    "cadres_dom": [5],
    "cadres_ext": [5],
    "corners_dom": [10],
    "corners_ext": [5]
})
print(arbre.predict(match_equilibre))