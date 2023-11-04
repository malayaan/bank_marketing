import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV, train_test_split

# Charger les données
data = pd.read_csv('bank_marketing_encoded.csv')

# Séparation des données en features et target
X = data.drop('class', axis=1)
y = data['class']

# Diviser les données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Créer et entraîner un modèle de base RandomForestClassifier
rf_base = RandomForestClassifier(random_state=42)
rf_base.fit(X_train, y_train)

# Prédire les labels sur l'ensemble de test
y_pred_base = rf_base.predict(X_test)

# Calculer la précision
accuracy_base = accuracy_score(y_test, y_pred_base)
print("accuracy_base=", accuracy_base)

# Paramètres pour la recherche
param_grid_rf = {
    'n_estimators': [10, 50, 100, 200],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'bootstrap': [True, False]
}

# Créer le modèle de recherche
grid_search_rf = GridSearchCV(RandomForestClassifier(random_state=42), param_grid_rf, 
                              cv=3, n_jobs=-1, verbose=1)

# Lancer la recherche
grid_search_rf.fit(X_train, y_train)

# Meilleurs paramètres
best_params_rf = grid_search_rf.best_params_
best_score_rf = grid_search_rf.best_score_

print(best_params_rf, best_score_rf)
