import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D

# Function to display confusion matrix visually
def plot_confusion_matrix(cm, title="Confusion Matrix"):
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(title)
    plt.show()

data = pd.read_csv('7 bank_marketing (2).csv', sep=';')

# Codage à chaud des variables catégorielles pour éviter d'avoir un ordre implicite
data_encoded = pd.get_dummies(data, columns=['marital', 'education', 'default', 'housing', 'loan', 'contact', 'poutcome'], drop_first=True)
# file_path = "bank_marketing_encoded.csv"
# data_encoded.to_csv(file_path, index=False)
# print(data_encoded.head())

# Séparer les variables indépendantes et dépendantes
X = data_encoded.drop('class', axis=1)
y = data_encoded['class']

# Paramètres à optimiser, including ccp_alpha for pruning
param_grid = {
    'criterion': ['gini', 'entropy'],
    'max_depth': [None, 1, 2, 3, 4, 5],
    'min_samples_split': [3, 5, 7, 9, 11, 13, 15, 17],
    'min_samples_leaf': [1, 2, 5, 6],
    'ccp_alpha': [0, 0.001, 0.01, 0.1, 0.2, 0.3]  # Added values for ccp_alpha
}

# Initialisation de GridSearchCV avec un arbre de décision, les paramètres élargis, et k-fold (par exemple k=5)
grid_search_extended = GridSearchCV(DecisionTreeClassifier(), param_grid, cv=5, scoring='f1_macro', n_jobs=-1, verbose=1)

# Entraînement de GridSearchCV sur toutes les données
grid_search_extended.fit(X, y)

# Récupération de la matrice de résultats
results = pd.DataFrame(grid_search_extended.cv_results_)

# Affichage des deux meilleures combinaisons d'hyperparamètres
top_2_combinations = results[['params', 'mean_test_score', 'std_test_score', 'rank_test_score']].sort_values(by='rank_test_score').head(2)

# Pour chaque meilleure combinaison, affichez la matrice de confusion
confusion_matrices = []
for idx, row in top_2_combinations.iterrows():
    # Entraînez le modèle avec les hyperparamètres optimaux
    model = DecisionTreeClassifier(**row['params'])
    model.fit(X, y)
    # Prédiction sur toutes les données (car nous n'avons pas de division train/test)
    y_pred = model.predict(X)
    cm = confusion_matrix(y, y_pred)
    confusion_matrices.append(cm)

# Display confusion matrices visually
for idx, confusion in enumerate(confusion_matrices):
    plot_confusion_matrix(confusion, title=f"Confusion Matrix for Combination {idx + 1}")

# Displaying top 2 hyperparameter combinations in a more structured and readable format
formatted_combinations = []

for idx, row in top_2_combinations.iterrows():
    formatted_combination = {}
    for key, value in row['params'].items():
        formatted_combination[key] = value
    formatted_combination['mean_test_score'] = row['mean_test_score']
    formatted_combination['std_test_score'] = row['std_test_score']
    formatted_combinations.append(formatted_combination)

# Converting the structured combinations into DataFrame for better visual representation
formatted_combinations_df = pd.DataFrame(formatted_combinations)

# Filter results for visualization
filtered_results = results[(results['param_ccp_alpha'] == 0)][['param_criterion', 'param_max_depth', 'mean_test_score', 'std_test_score']]
filtered_results['param_max_depth'].replace({None: 0}, inplace=True)
filtered_results['param_criterion_num'] = filtered_results['param_criterion'].map({'gini': 0, 'entropy': 1})

# Create a 3D plot for mean_test_score
fig = plt.figure(figsize=(12, 6))

# F1 Score 3D plot
ax1 = fig.add_subplot(121, projection='3d')
scatter = ax1.scatter(filtered_results['param_max_depth'], 
            filtered_results['param_criterion_num'], 
            filtered_results['mean_test_score'], 
            c=filtered_results['mean_test_score'], 
            cmap='viridis', s=60, depthshade=False)
ax1.set_xlabel('Max Depth')
ax1.set_ylabel('Criterion (0: gini, 1: entropy)')
ax1.set_zlabel('F1 Score')
ax1.set_title('F1 Score vs. Hyperparameters')
ax1.set_yticks([0, 1])

# Standard deviation 3D plot
ax2 = fig.add_subplot(122, projection='3d')
scatter_std = ax2.scatter(filtered_results['param_max_depth'], 
            filtered_results['param_criterion_num'], 
            filtered_results['std_test_score'], 
            c=filtered_results['std_test_score'], 
            cmap='viridis', s=60, depthshade=False)
ax2.set_xlabel('Max Depth')
ax2.set_ylabel('Criterion (0: gini, 1: entropy)')
ax2.set_zlabel('Standard Deviation')
ax2.set_title('Standard Deviation vs. Hyperparameters')
ax2.set_yticks([0, 1])

plt.tight_layout()
plt.show()

# Return the top 2 combinations for the user
print(formatted_combinations_df)

# les deux meilleurs arbres sans élagages retenues sont ceux d'hypermatres:
# criterion  max_depth  min_samples_leaf  min_samples_split  mean_test_score  std_test_score
# 0   entropy          1                 1                 17         0.727602        0.071393
# 1      gini          2                 6                 13         0.727602        0.071393

# les deux meilleurs arbres avec élagages retenues sont ceux d'hypermatres:
#    ccp_alpha criterion  max_depth  min_samples_leaf  min_samples_split  mean_test_score  std_test_score  precision rappel 
# 0      0.001   entropy          2                 6                 13         0.727602        0.071393  0.6146     0.5226
# 1      0.010      gini          3                 1                  5         0.727602        0.071393  0.6146     0.5226
