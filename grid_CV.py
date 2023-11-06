import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV, train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from imblearn.over_sampling import SMOTE
from mpl_toolkits.mplot3d import Axes3D  # Import this for 3D plotting

# Function to display confusion matrix visually
def plot_confusion_matrix(cm, title="Confusion Matrix"):
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(title)
    plt.show()

# Load the data
data_encoded = pd.read_csv('bank_marketing_encoded.csv')

# Separate independent and dependent variables
X = data_encoded.drop('class', axis=1)
y = data_encoded['class']

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Initialize SMOTE and apply on training data only
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

#si on veut travailler sur les données unbalanced:
#X_train_smote, y_train_smote = X_train, y_train

# Define parameters for GridSearchCV
param_grid = {
    'criterion': ['gini', 'entropy'],
    'max_depth': [None, 1, 5, 10, 20],
    'min_samples_split': [2, 3, 5, 7, 9],
    'min_samples_leaf': [1, 2, 4, 6],
    'ccp_alpha': [0, 0.001, 0.01, 0.1, 0.2, 0.3]
}

# Initialize GridSearchCV
grid_search = GridSearchCV(DecisionTreeClassifier(), param_grid, cv=5, scoring='f1_macro', n_jobs=-1, verbose=1)

# Fit GridSearchCV on the training data only
grid_search.fit(X_train_smote, y_train_smote)

# Get the best estimator and predict on the test set
best_tree = grid_search.best_estimator_
y_pred = best_tree.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
plot_confusion_matrix(cm, title="Confusion Matrix for Best Model")

# Display top 2 hyperparameter combinations
results = pd.DataFrame(grid_search.cv_results_)
top_2_combinations = results.nlargest(2, 'mean_test_score')

# For each of the top 2 combinations, display the confusion matrix
for idx, row in top_2_combinations.iterrows():
    params = row['params']
    model = DecisionTreeClassifier(**params).fit(X_train_smote, y_train_smote)
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    plot_confusion_matrix(cm, title=f"Confusion Matrix for Top {idx+1} Combination")

# Displaying the top 2 hyperparameter combinations in a structured format
top_2_params = top_2_combinations['params'].apply(pd.Series)
top_2_params['mean_test_score'] = top_2_combinations['mean_test_score']
top_2_params['std_test_score'] = top_2_combinations['std_test_score']
print(top_2_params)


#résultats:

# pour les données unbalanced:

# les deux meilleurs arbres sans élagages retenues sont ceux d'hypermatres:
# criterion  max_depth  min_samples_leaf  min_samples_split  mean_test_score  std_test_score
# 0   entropy          1                 1                 17         0.727602        0.071393
# 1      gini          2                 6                 13         0.727602        0.071393

# les deux meilleurs arbres avec élagages retenues sont ceux d'hypermatres:
#    ccp_alpha criterion  max_depth  min_samples_leaf  min_samples_split  mean_test_score  std_test_score  precision rappel 
# 0      0.001   entropy          2                 6                 13         0.727602        0.071393  0.6146     0.5226
# 1      0.010      gini          3                 1                  5         0.727602        0.071393  0.6146     0.5226


# pour les données équilibrées:

#oversampling (dupplication aléatoire de données de la class sous représentée)
# les deux meilleurs arbres sans élagages retenues sont ceux d'hypermatres:
#    criterion max_depth  min_samples_leaf  min_samples_split  mean_test_score  std_test_score
#0     entropy      None                 1                  3         0.851780        0.078330
#1        gini      None                 1                  3         0.848698        0.081209
