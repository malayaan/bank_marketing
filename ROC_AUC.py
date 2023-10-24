from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt

data = pd.read_csv('7 bank_marketing (2).csv', sep=';')

# Codage à chaud des variables catégorielles pour éviter d'avoir un ordre implicite
data_encoded = pd.get_dummies(data, columns=['marital', 'education', 'default', 'housing', 'loan', 'contact', 'poutcome'], drop_first=True)

# Séparer les variables indépendantes et dépendantes
X = data_encoded.drop('class', axis=1)
y = data_encoded['class'].replace({'no': 0, 'yes': 1})

# Définir les meilleurs hyperparamètres à partir de l'information fournie
best_hyperparameters = [
    {'ccp_alpha': 0.001, 'criterion': 'entropy', 'max_depth': 2, 'min_samples_leaf': 6, 'min_samples_split': 13},
    {'ccp_alpha': 0.010, 'criterion': 'gini', 'max_depth': 3, 'min_samples_leaf': 1, 'min_samples_split': 5}
]

# Séparation des données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

def plot_roc_curve(params, X_train, y_train, X_test, y_test):
    # Former le modèle avec les hyperparamètres donnés
    clf = DecisionTreeClassifier(**params)
    clf.fit(X_train, y_train)
    
    # Prévoir les probabilités pour l'ensemble de test
    y_prob = clf.predict_proba(X_test)[:,1]
    
    # Calculer la courbe ROC
    fpr, tpr, thresholds = roc_curve(y_test, y_prob)
    
    # Calculer l'AUC
    auc_value = auc(fpr, tpr)
    
    # Afficher la courbe ROC
    plt.figure(figsize=(10, 7))
    plt.plot(fpr, tpr, label=f'AUC = {auc_value:.2f}')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'Courbe ROC pour {params}')
    plt.legend()
    plt.show()

    return auc_value

# Afficher la courbe ROC pour les meilleurs hyperparamètres et stocker les valeurs AUC
auc_values = []
for params in best_hyperparameters:
    auc_value = plot_roc_curve(params, X_train, y_train, X_test, y_test)
    auc_values.append(auc_value)

print(auc_values)
