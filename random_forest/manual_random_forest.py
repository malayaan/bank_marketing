import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from scipy.stats import mode
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import accuracy_score, f1_score, classification_report,precision_recall_fscore_support
from sklearn.metrics import confusion_matrix, roc_auc_score, auc, roc_curve, precision_recall_curve, average_precision_score
from sklearn.utils import resample
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils import resample
from sklearn.model_selection import KFold
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    roc_curve, precision_recall_curve, average_precision_score
)
import matplotlib.pyplot as plt

# Load the data
file_path = 'data/bank_marketing_encoded.csv'
data = pd.read_csv(file_path)

# Convert the 'class' column to a binary numeric column where 'yes' is 1 and 'no' is 0
data['class'] = data['class'].apply(lambda x: 1 if x == 'yes' else 0)


# Separate the features and the target variable
X = data.drop('class', axis=1)
y = data['class']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
smote = SMOTE(random_state=42)
X_train, y_train = smote.fit_resample(X_train, y_train)

# Nombre d'arbres dans la forêt
n_trees = 100
# Taille de l'échantillon pour le bootstrapping
bootstrap_size = len(X_train)

# Valeurs de alpha à tester
alpha_values = [0.001, 0.01, 0.1]

# Dictionnaire pour enregistrer les scores de performance pour chaque alpha
alpha_scores = {alpha: [] for alpha in alpha_values}

# Cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Boucle sur les valeurs de alpha
for alpha in alpha_values:
    # Boucle sur les plis de validation croisée
    for train_index, test_index in kf.split(X_train):
        # Séparer les données en plis d'entraînement et de test
        X_train_fold, X_test_fold = X_train.iloc[train_index], X_train.iloc[test_index]
        y_train_fold, y_test_fold = y_train[train_index], y_train[test_index]

        # Liste pour stocker les arbres de la forêt
        forest = []

        # Création de la forêt
        for i in range(n_trees):
            # Bootstrapping
            X_sample, y_sample = resample(X_train_fold, y_train_fold, n_samples=bootstrap_size, random_state=i)
            tree = DecisionTreeClassifier(max_depth=30, min_samples_leaf=2, min_samples_split=2, random_state=42+i, ccp_alpha=alpha)
            tree.fit(X_sample, y_sample)
            forest.append(tree)

        # Évaluation de la forêt sur le pli de test
        y_pred_forest = np.mean([tree.predict(X_test_fold) for tree in forest], axis=0)
        y_pred_forest = np.round(y_pred_forest)  # Pour obtenir des prédictions binaires

        # Calcul des métriques
        f1 = f1_score(y_test_fold, y_pred_forest)
        alpha_scores[alpha].append(f1)
        print(alpha)

# Calculer la moyenne des scores F1 pour chaque alpha
mean_scores = {alpha: np.mean(scores) for alpha, scores in alpha_scores.items()}
print(mean_scores)

# Après avoir calculé les scores F1 pour chaque alpha
mean_f1_scores = {alpha: np.mean(f1_scores) for alpha, f1_scores in alpha_scores.items()}
best_alpha = max(mean_f1_scores, key=mean_f1_scores.get)

# Entraînez une forêt finale avec le meilleur alpha
final_forest = []
for i in range(n_trees):
    X_sample, y_sample = resample(X_train, y_train, n_samples=bootstrap_size, random_state=i)
    tree = DecisionTreeClassifier(max_depth=30, min_samples_leaf=2, min_samples_split=2, random_state=42+i, ccp_alpha=best_alpha)
    tree.fit(X_sample, y_sample)
    final_forest.append(tree)

# Prédire sur le jeu de données de test
y_pred_forest = np.mean([tree.predict(X_test) for tree in final_forest], axis=0)
y_pred_forest = np.round(y_pred_forest)  # Pour obtenir des prédictions binaires

# Calculer et afficher toutes les métriques
accuracy = accuracy_score(y_test, y_pred_forest)
f1 = f1_score(y_test, y_pred_forest)
print(f"Accuracy: {accuracy:.2f}")
print(f"F1 Score: {f1:.2f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred_forest))
conf_matrix = confusion_matrix(y_test, y_pred_forest)
print("Confusion Matrix:")
print(conf_matrix)
roc_auc = roc_auc_score(y_test, np.mean([tree.predict_proba(X_test)[:, 1] for tree in final_forest], axis=0))
print(f"ROC-AUC Score: {roc_auc:.2f}")

# Calcul des métriques de précision, rappel, score F1 et support pour chaque classe
precision_cls, recall_cls, f1_score_cls, support_cls = precision_recall_fscore_support(y_test, y_pred_forest)

# Affichage des métriques détaillées par classe
print("\nDetailed Accuracy by Class:")
class_labels = np.unique(y) 
for i in range(len(class_labels)):
    print(f"Class {class_labels[i]} - Precision: {precision_cls[i]:.4f}, Recall: {recall_cls[i]:.4f}, Support: {support_cls[i]}")

# Calculer la courbe ROC
fpr, tpr, _ = roc_curve(y_test, np.mean([tree.predict_proba(X_test)[:, 1] for tree in final_forest], axis=0))
plt.figure()
plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc="lower right")
plt.show()

# Calculer la courbe Precision-Recall
precision, recall, _ = precision_recall_curve(y_test, np.mean([tree.predict_proba(X_test)[:, 1] for tree in final_forest], axis=0))
average_precision = average_precision_score(y_test, np.mean([tree.predict_proba(X_test)[:, 1] for tree in final_forest], axis=0))
plt.figure()
plt.plot(recall, precision, label='Precision-Recall curve (area = %0.2f)' % average_precision)
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend(loc="lower left")
plt.show()