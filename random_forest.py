import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import GridSearchCV
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

# Check the shape of the resulting dataframes
(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

# Initialize the Random Forest classifier
rf_classifier = RandomForestClassifier(random_state=42)

# Fit the classifier to the training data
rf_classifier.fit(X_train, y_train)

# Predict on the test data
y_pred = rf_classifier.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print("performance_ref", accuracy, report)
#Ces résultats montrent que le modèle est meilleur pour prédire les clients qui ne souscriront pas (classe 0) par rapport à ceux qui souscriront (classe 1). Cela pourrait être dû à un déséquilibre dans les classes de l'ensemble de données.

# Define the parameter grid
param_grid = {
    'n_estimators': [90, 100, 110],
    'max_depth': [5, 10,20],
    'min_samples_split': [2, 5, 15],
    'min_samples_leaf': [2]
}

# Initialize the GridSearchCV object
grid_search = GridSearchCV(estimator=rf_classifier, param_grid=param_grid, 
                           cv=5, n_jobs=-1, verbose=2)

# Fit the grid search to the data
grid_search.fit(X_train, y_train)

# Get the best parameters
best_params = grid_search.best_params_
print("performance_ref", accuracy, report)
print("Best parameters:", best_params)

# Train the classifier with the best found parameters
best_rf_classifier = RandomForestClassifier(**best_params, random_state=42)
best_rf_classifier.fit(X_train, y_train)

# Predict on the test data using the best model
y_pred_best = best_rf_classifier.predict(X_test)

# Calculate accuracy for the best model
accuracy_best = accuracy_score(y_test, y_pred_best)
report_best = classification_report(y_test, y_pred_best)

print("performance_optimized", accuracy_best, report_best)

# Print the classification report for the optimized model
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
