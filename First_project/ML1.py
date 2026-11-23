import statistics

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.dummy import DummyClassifier
from sklearn.tree import DecisionTreeClassifier


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



def vic(x):
     victoire = 0
     tire_cadres = 0
     for i, j in zip(x["resultat"], x["cadres_dom"]):
         if i == "victoire_dom":
             victoire = victoire + 1
             tire_cadres = tire_cadres + j

     if victoire == 0:
         return 0, 0

     moyenne = round(tire_cadres / victoire, 3)
     return victoire, moyenne





x = df[["possession_dom", "tirs_dom" , "tirs_ext" , "cadres_dom" , "cadres_ext" , "corners_dom" , "corners_ext"]]
y = df["resultat"]

x_train, x_test, y_train, y_test = train_test_split(
    x,y, test_size=0.20, random_state=42, stratify=y)



bete = DummyClassifier(strategy="most_frequent")
bete.fit(x_train,y_train)
predict = bete.predict(x_test)
accuracy_score(y_test,predict)


reger = LogisticRegression(max_iter=800)
reger.fit(x_train,y_train)
predict1 = reger.predict(x_test)
print(predict1)
print(accuracy_score(predict1,y_test))



arbre = DecisionTreeClassifier(max_depth=3 , random_state=42)
arbre.fit(x_train,y_train)
arbre1 = arbre.predict(x_test)
print(accuracy_score(arbre1,y_test))








