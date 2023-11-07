import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, f1_score
from sklearn.model_selection import GridSearchCV, train_test_split
import seaborn as sns
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE

# Function to display confusion matrix visually
def plot_confusion_matrix(cm, title="Confusion Matrix"):
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(title)
    plt.show()

# Load the data
data_encoded = pd.read_csv('data/bank_marketing_encoded.csv')

# Separate independent and dependent variables
X = data_encoded.drop('class', axis=1)
y = data_encoded['class']

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Initialize SMOTE and apply on training data only
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

#if we want to work on the unbalance data
#X_train_smote, y_train_smote = X_train, y_train

# Define parameters for GridSearchCV
param_grid = {
    'criterion': ['entropy'],
    'max_depth': [None, 20,50,70],
    'min_samples_split': [2,5,10],
    'min_samples_leaf': [1,5,10],
    'ccp_alpha': [0, 0.001, 0.01, 0.1]
}

# Initialize GridSearchCV
grid_search = GridSearchCV(DecisionTreeClassifier(), param_grid, cv=5, scoring='f1_macro', n_jobs=-1, verbose=1)

# Fit GridSearchCV on the training data only
grid_search.fit(X_train_smote, y_train_smote)

# Get the best two estimators
best_params = grid_search.cv_results_['params'][grid_search.best_index_]
second_best_index = (grid_search.cv_results_['rank_test_score'] == 2).argmax()
second_best_params = grid_search.cv_results_['params'][second_best_index]

# Create the top two models
top_models = [DecisionTreeClassifier(**best_params).fit(X_train_smote, y_train_smote),
              DecisionTreeClassifier(**second_best_params).fit(X_train_smote, y_train_smote)]

# Evaluate the top two models
for i, model in enumerate(top_models, 1):
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='macro')
    precision, recall, fscore, support = precision_recall_fscore_support(y_test, y_pred)
    
    # Display the confusion matrix
    plot_confusion_matrix(cm, title=f"Confusion Matrix for Top {i} Model")
    
    # Print model performance
    print(f"Top {i} Model Hyperparameters: {model.get_params()}")
    print(f"Top {i} Model Test F1 Score: {f1}")
    print(f"Top {i} Model Precision, Recall, and F1 Score by Class:")
    for class_id in range(len(precision)):
        print(f"Class {class_id}: Precision: {precision[class_id]}, Recall: {recall[class_id]}, F1 Score: {fscore[class_id]}")
    print("\n")
    
    
# Top 1 Model Hyperparameters: {'ccp_alpha': 0, 'class_weight': None, 'criterion': 'entropy', 'max_depth': None, 'max_features': None, 'max_leaf_nodes': None, 'min_impurity_decrease': 0.0, 'min_samples_leaf': 1, 'min_samples_split': 2, 'min_weight_fraction_leaf': 0.0, 'random_state': None, 'splitter': 'best'}
# Top 1 Model Test F1 Score: 0.6412389039164437
# Top 1 Model Precision, Recall, and F1 Score by Class:
# Class 0: Precision: 0.8463343108504399, Recall: 0.7924217462932455, F1 Score: 0.8184912081678957
# Class 1: Precision: 0.42290076335877863, Recall: 0.5139146567717996, F1 Score: 0.46398659966499156


# Top 2 Model Hyperparameters: {'ccp_alpha': 0, 'class_weight': None, 'criterion': 'entropy', 'max_depth': 70, 'max_features': None, 'max_leaf_nodes': None, 'min_impurity_decrease': 0.0, 'min_samples_leaf': 1, 'min_samples_split': 2, 'min_weight_fraction_leaf': 0.0, 'random_state': None, 'splitter': 'best'}
# Top 2 Model Test F1 Score: 0.6335949920508743
# Top 2 Model Precision, Recall, and F1 Score by Class:
# Class 0: Precision: 0.841399416909621, Recall: 0.7924217462932455, F1 Score: 0.8161764705882353
# Class 1: Precision: 0.413953488372093, Recall: 0.49536178107606677, F1 Score: 0.4510135135135135