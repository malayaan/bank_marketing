import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from scipy.stats import mode
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import accuracy_score
from sklearn.utils import resample
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix, roc_auc_score, auc, roc_curve, precision_recall_curve, average_precision_score

# Load the data
file_path = 'bank_marketing_encoded.csv'
data = pd.read_csv(file_path)

# Convert the 'class' column to a binary numeric column where 'yes' is 1 and 'no' is 0
data['class'] = data['class'].apply(lambda x: 1 if x == 'yes' else 0)

# Separate the features and the target variable
X = data.drop('class', axis=1)
y = data['class']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Paramètres de la forêt
n_trees = 90
bootstrap_size = int(0.8 * len(X_train))

# Fonction pour faire des prédictions avec la forêt
def predict_forest(forest, X):
    predictions = np.array([tree.predict(X) for tree in forest])
    predictions_mode = mode(predictions, axis=0, keepdims=True)[0][0]  # If you want to keep the dimensions
    return predictions_mode

# Optimisation des hyperparamètres
alpha_values = np.logspace(-4, -1, 11)
alpha_scores = []

kf = KFold(n_splits=5, shuffle=True, random_state=42)

for alpha in alpha_values:
    print("alpha",alpha)
    fold_scores = []
    for train_index, val_index in kf.split(X_train):
        # Diviser les données
        X_train_fold, X_val_fold = X_train.iloc[train_index], X_train.iloc[val_index]
        y_train_fold, y_val_fold = y_train.iloc[train_index], y_train.iloc[val_index]

        # Réinitialiser la forêt pour le pli actuel
        current_forest = []
        for i in range(n_trees):
            # Bootstrapping
            X_sample, y_sample = resample(X_train_fold, y_train_fold, n_samples=bootstrap_size, random_state=i)
            tree = DecisionTreeClassifier(max_depth=10, min_samples_leaf=2, min_samples_split=5, random_state=42+i, ccp_alpha=alpha)
            tree.fit(X_sample, y_sample)
            current_forest.append(tree)

        # Faire des prédictions sur le pli de validation
        y_val_pred = predict_forest(current_forest, X_val_fold)

        # Calculer le score pour le pli actuel
        fold_score = accuracy_score(y_val_fold, y_val_pred)
        fold_scores.append(fold_score)
    
    # Calculer la moyenne des scores de validation croisée
    mean_cv_score = np.mean(fold_scores)
    print("mean_cv_score", mean_cv_score)
    alpha_scores.append(mean_cv_score)

# Trouver le meilleur ccp_alpha
best_alpha_index = np.argmax(alpha_scores)
best_alpha = alpha_values[best_alpha_index]
best_alpha_score = alpha_scores[best_alpha_index]
print("Best ccp_alpha:", best_alpha)
print("Best ccp_alpha score:", best_alpha_score)


# Train the optimized model with the best ccp_alpha
best_rf_classifier = DecisionTreeClassifier(ccp_alpha=best_alpha, random_state=42)
best_rf_classifier.fit(X_train, y_train)

# Predict on the test data using the best model
y_pred_best = best_rf_classifier.predict(X_test)

# Calculate accuracy for the best model
accuracy_best = accuracy_score(y_test, y_pred_best)
report_best = classification_report(y_test, y_pred_best)

print("performance_optimized", accuracy_best)
print("Classification Report for the Optimized Model:")
print(report_best)

# Confusion matrix for the optimized model
conf_matrix_best = confusion_matrix(y_test, y_pred_best)
print("Confusion Matrix for the Optimized Model:")
print(conf_matrix_best)

# ROC-AUC score for the optimized model
roc_auc_best = roc_auc_score(y_test, best_rf_classifier.predict_proba(X_test)[:, 1])
print("ROC-AUC Score for the Optimized Model:", roc_auc_best)

# Calculate precision-recall curve
precision, recall, thresholds = precision_recall_curve(y_test, best_rf_classifier.predict_proba(X_test)[:, 1])
pr_auc = auc(recall, precision)
print("Precision-Recall AUC for the Optimized Model:", pr_auc)

# Calculate the average precision score
average_precision = average_precision_score(y_test, best_rf_classifier.predict_proba(X_test)[:, 1])
print("Average Precision for the Optimized Model:", average_precision)

# Calculate ROC curve
fpr, tpr, roc_thresholds = roc_curve(y_test, best_rf_classifier.predict_proba(X_test)[:, 1])

# Plot ROC curve
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 8))
plt.plot(fpr, tpr, label=f'Optimized Model ROC curve (area = {roc_auc_best:.2f})')
plt.plot([0, 1], [0, 1], 'k--')  # Dashed diagonal
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc="lower right")
plt.show()

# Plot Precision-Recall curve
plt.figure(figsize=(10, 8))
plt.plot(recall, precision, label=f'Optimized Model PR curve (area = {pr_auc:.2f})')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend(loc="lower left")
plt.show()