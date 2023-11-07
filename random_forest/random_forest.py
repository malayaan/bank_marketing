import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report,
    confusion_matrix, roc_auc_score, auc,
    roc_curve, precision_recall_curve, average_precision_score
)
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_fscore_support

# Load the data
file_path = 'data/bank_marketing_encoded.csv'
data = pd.read_csv(file_path)

# Data preprocessing
data['class'] = data['class'].map({'yes': 1, 'no': 0})  # Convert 'class' to binary
X = data.drop('class', axis=1)  # Features
y = data['class']  # Target variable

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Apply SMOTE
smote = SMOTE(random_state=42)
X_train, y_train = smote.fit_resample(X_train, y_train)

# Initialize the Random Forest classifier
rf_classifier = RandomForestClassifier(random_state=42)

# Define the parameter grid for GridSearchCV
param_grid = {
    'n_estimators': [50,100, 150],
    'max_depth': [ 30,50,70],
    'min_samples_split': [2],
    'min_samples_leaf': [2]
}

# Initialize the GridSearchCV object with F1 score metric
grid_search = GridSearchCV(estimator=rf_classifier, param_grid=param_grid, scoring='f1', cv=5, n_jobs=-1, verbose=2)

# Fit the grid search to the data
grid_search.fit(X_train, y_train)

# Get the best parameters and the best F1 score
best_params = grid_search.best_params_
best_f1_score = grid_search.best_score_
print("Best parameters found by grid search:", best_params)

# Train the classifier with the best found parameters
best_rf_classifier = RandomForestClassifier(**best_params, random_state=42)
best_rf_classifier.fit(X_train, y_train)

# Predict on the test data using the best model
y_pred_best = best_rf_classifier.predict(X_test)

# Evaluate the optimized model
accuracy_best = accuracy_score(y_test, y_pred_best)
f1_best = f1_score(y_test, y_pred_best, average='binary')

# Display results for the optimized model
print(f"\nAccuracy of the optimized model: {accuracy_best:.2f}")
print(f"F1 Score of the optimized model: {f1_best:.2f}")
print("\nClassification Report for the Optimized Model:")
print(classification_report(y_test, y_pred_best))

# Confusion matrix for the optimized model
conf_matrix_best = confusion_matrix(y_test, y_pred_best)
print("Confusion Matrix for the Optimized Model:")
print(conf_matrix_best)

# ROC-AUC score for the optimized model
roc_auc_best = roc_auc_score(y_test, best_rf_classifier.predict_proba(X_test)[:, 1])
print("ROC-AUC Score for the Optimized Model:", roc_auc_best)

# Calculate precision-recall curve and AUC
precision, recall, thresholds = precision_recall_curve(y_test, best_rf_classifier.predict_proba(X_test)[:, 1])
pr_auc = auc(recall, precision)
print("Precision-Recall AUC for the Optimized Model:", pr_auc)

# Average precision score
average_precision = average_precision_score(y_test, best_rf_classifier.predict_proba(X_test)[:, 1])
print("Average Precision for the Optimized Model:", average_precision)

# Plot ROC curve
fpr, tpr, roc_thresholds = roc_curve(y_test, best_rf_classifier.predict_proba(X_test)[:, 1])
plt.figure(figsize=(10, 8))
plt.plot(fpr, tpr, label=f'Optimized Model ROC curve (area = {roc_auc_best:.2f})')
plt.plot([0, 1], [0, 1], 'k--')  # Dashed diagonal
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve for the Optimized Model')
plt.legend(loc="lower right")
plt.show()

# Plot Precision-Recall curve
plt.figure(figsize=(10, 8))
plt.plot(recall, precision, label=f'Optimized Model PR curve (area = {pr_auc:.2f})')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve for the Optimized Model')
plt.legend(loc="lower left")
plt.show()

# Calcul des métriques de précision, rappel, score F1 et support pour chaque classe
precision_cls, recall_cls, f1_score_cls, support_cls = precision_recall_fscore_support(y_test, y_pred_best)

# Affichage des métriques détaillées par classe
print("\nDetailed Accuracy by Class:")
class_labels = best_rf_classifier.classes_
for i, (precision, recall, f1, support) in enumerate(zip(precision_cls, recall_cls, f1_score_cls, support_cls)):
    print(f"Class {class_labels[i]} - Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}, Support: {support}")
    
#results:
# Best parameters found by grid search: {'max_depth': 30, 'min_samples_leaf': 2, 'min_samples_split': 2, 'n_estimators': 100}

# Accuracy of the optimized model: 0.77
# F1 Score of the optimized model: 0.54

# Classification Report for the Optimized Model:
#               precision    recall  f1-score   support

#            0       0.87      0.83      0.85      1821
#            1       0.50      0.58      0.54       539

#     accuracy                           0.77      2360
#    macro avg       0.69      0.70      0.69      2360
# weighted avg       0.79      0.77      0.78      2360

# Confusion Matrix for the Optimized Model:
# [[1513  308]
#  [ 227  312]]
# ROC-AUC Score for the Optimized Model: 0.7570678713300507
# Precision-Recall AUC for the Optimized Model: 0.4810757213471932
# Average Precision for the Optimized Model: 0.48229470999150925

# Detailed Accuracy by Class:
# Class 0 - Precision: 0.8695, Recall: 0.8309, F1 Score: 0.8498, Support: 1821
# Class 1 - Precision: 0.5032, Recall: 0.5788, F1 Score: 0.5384, Support: 539