from sklearn.tree import plot_tree
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE

data_encoded = pd.read_csv('data/bank_marketing_encoded.csv')

# Séparer les variables indépendantes et dépendantes
X = data_encoded.drop('class', axis=1)
y = data_encoded['class']

# Initialisation de SMOTE
smote = SMOTE(random_state=42)

X, y = smote.fit_resample(X, y)

# Définir les meilleurs hyperparamètres à partir de l'information fournie 
#attention les profondeurs ont étés réduites
best_hyperparameters = [
    {'ccp_alpha': 0, 'class_weight': None, 'criterion': 'entropy', 'max_depth': 3, 'max_features': None, 'max_leaf_nodes': None, 'min_impurity_decrease': 0.0, 'min_samples_leaf': 1, 'min_samples_split': 2, 'min_weight_fraction_leaf': 0.0, 'random_state': None, 'splitter': 'best'},
    {'ccp_alpha': 0, 'class_weight': None, 'criterion': 'entropy', 'max_depth': 5, 'max_features': None, 'max_leaf_nodes': None, 'min_impurity_decrease': 0.0, 'min_samples_leaf': 1, 'min_samples_split': 2, 'min_weight_fraction_leaf': 0.0, 'random_state': None, 'splitter': 'best'}
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