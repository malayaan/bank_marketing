from sklearn.tree import plot_tree
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt

data = pd.read_csv('7 bank_marketing (2).csv', sep=';')

# Codage à chaud des variables catégorielles pour éviter d'avoir un ordre implicite
data_encoded = pd.get_dummies(data, columns=['marital', 'education', 'default', 'housing', 'loan', 'contact', 'poutcome'], drop_first=True)

# Séparer les variables indépendantes et dépendantes
X = data_encoded.drop('class', axis=1)
y = data_encoded['class']

# Définir les meilleurs hyperparamètres à partir de l'information fournie
best_hyperparameters = [
    {'ccp_alpha': 0.001, 'criterion': 'entropy', 'max_depth': 2, 'min_samples_leaf': 6, 'min_samples_split': 13},
    {'ccp_alpha': 0.010, 'criterion': 'gini', 'max_depth': 3, 'min_samples_leaf': 1, 'min_samples_split': 5}
]

# Fonction pour visualiser l'arbre de décision
def visualize_tree(params, X, y):
    # Former le modèle avec les hyperparamètres donnés
    clf = DecisionTreeClassifier(**params)
    clf.fit(X, y)
    
    # Afficher l'arbre
    plt.figure(figsize=(20, 10))
    plot_tree(clf, feature_names=X.columns, class_names=['no', 'yes'], filled=True, rounded=True, fontsize=10)
    plt.title(f"Arbre de décision avec {params}")
    plt.show()

# Visualiser les arbres pour les meilleurs hyperparamètres
for params in best_hyperparameters:
    visualize_tree(params, X, y)
