import pandas as pd

data = pd.read_csv('data/7 bank_marketing (2).csv', sep=';')

# Codage à chaud des variables catégorielles pour éviter d'avoir un ordre implicite
data_encoded = pd.get_dummies(data, columns=['marital', 'education', 'default', 'housing', 'loan', 'contact', 'poutcome'], drop_first=True)
file_path = "data/bank_marketing_encoded.csv"
data_encoded.to_csv(file_path, index=False)
print(data_encoded.head())
