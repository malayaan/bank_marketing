import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Charger le fichier CSV avec le bon séparateur
data = pd.read_csv("data/7 bank_marketing (2).csv", sep=';')

# Afficher les premières lignes du jeu de données
# print(data.head())

"""
Voilà à quoi ressemble notre jeu de données :

age : âge du client.
marital : statut marital du client.
education : niveau d'éducation du client.
default : si le client a un défaut de crédit ou non.
balance : solde annuel moyen en euros.
housing : si le client a un prêt immobilier ou non.
loan : si le client a un prêt personnel ou non.
contact : type de communication de contact.
poutcome : résultat de la campagne de marketing précédente.
class : si le client a souscrit un dépôt à terme ou non.
"""
# Obtenir les informations sur le jeu de données
data_info = data.info()

# Obtenir le nombre de valeurs uniques pour chaque colonne
unique_counts = data.nunique()

# print(unique_counts)
"""
Voici la typologie des variables :

Variables numériques :

age: avec 70 valeurs uniques.
balance: avec 3,095 valeurs uniques.

Variables catégorielles :

marital: 3 catégories (par exemple, marié, célibataire, etc.).
education: 3 catégories.
default: 2 catégories (oui ou non).
housing: 2 catégories.
loan: 2 catégories.
contact: 2 catégories.
poutcome: 3 catégories.
class: 2 catégories (oui ou non).

il n'y a pas de valuers manquantes
"""

# Statistiques descriptives pour les variables numériques
numeric_stats = data[['age', 'balance']].describe()

# Distribution des variables catégorielles
categorical_distribution = data.select_dtypes(include=['object']).apply(pd.Series.value_counts)

# print(numeric_stats, categorical_distribution)
"""
Voici les statistiques descriptives pour les variables numériques :

Age :

Moyenne: 40.8 ans
Écart-type: 11.3
Minimum: 18 ans
Maximum: 89 ans
25e percentile: 32 ans
Médiane: 38 ans
75e percentile: 47 ans

Balance :

Moyenne: 1553.18 euros
Écart-type: 3082.33 euros
Minimum: -1884 euros
Maximum: 81204 euros
25e percentile: 162.75 euros
Médiane: 596 euros
75e percentile: 1740 euros

Concernant la distribution des variables catégorielles, voici quelques points notables :

Marital : La majorité des personnes sont mariées (4515), suivies de personnes célibataires (2459) et de personnes divorcées (890).
Education : La plupart des personnes ont un niveau d'éducation secondaire (4210), suivies de celles ayant un niveau tertiaire (2639) et primaire (1015).
Default : Très peu de personnes ont un défaut de crédit (56).
Housing : Un grand nombre de personnes ont un prêt immobilier (4947).
Loan : La plupart des personnes n'ont pas de prêt personnel (6774).
Contact : La majorité des contacts se font par téléphone portable (cellular) (7274).
Poutcome : La plupart des précédentes campagnes de marketing ont échoué (4694).
Class : La plupart des personnes n'ont pas souscrit de dépôt à terme (6068) soit 77%
"""

# Violin plots pour les variables numériques
plt.figure(figsize=(14, 6))

# Violin plot pour 'age'
plt.subplot(1, 2, 1)
sns.violinplot(data['age'], color='skyblue')
plt.title('Violin plot pour Age')

# Violin plot pour 'balance'
plt.subplot(1, 2, 2)
sns.violinplot(data['balance'], color='lightgreen')
plt.title('Violin plot pour Balance')

plt.tight_layout()
plt.show()

categorical_columns = ['marital', 'education', 'default', 'housing', 'loan', 'contact', 'poutcome', 'class']

# Violin plots pour les variables catégorielles
plt.figure(figsize=(20, 15))
for i, column in enumerate(categorical_columns, 1):
    plt.subplot(3, 3, i)
    sns.violinplot(x=data[column], y=data['balance'], palette='pastel')
    plt.title(f'Violin plot pour Balance par {column}')
    plt.xticks(rotation=45)

plt.tight_layout()
plt.show()

plt.figure(figsize=(20, 15))
for i, column in enumerate(categorical_columns, 1):
    plt.subplot(3, 3, i)
    sns.violinplot(x=data[column], y=data['age'], palette='pastel')
    plt.title(f'Violin plot pour Age par {column}')
    plt.xticks(rotation=45)

plt.tight_layout()
plt.show()

"""
Violin plots pour les variables numériques :

Age : Cette visualisation confirme nos observations précédentes sur l'âge. La majorité des clients ont entre 30 et 50 ans.
Balance : La majorité des clients ont un solde inférieur à 5000 euros, mais il y a quelques clients avec des soldes bien supérieurs.
Violin plots pour les variables catégorielles avec Balance comme variable continue :
Ces tracés montrent la distribution du solde (balance) pour chaque catégorie des variables catégorielles.

Par exemple, pour le statut marital, vous pouvez voir que les clients divorcés ont généralement un solde plus bas que les clients mariés ou célibataires.
Autre exemple, dans le tracé de marital vs age, nous pouvons observer que les personnes divorcées ont tendance à être plus âgées que celles qui sont célibataires ou mariées.
De même, pour la variable education, les personnes avec un niveau d'éducation primaire semblent être plus âgées que celles avec des niveaux secondaire ou tertiaire.
"""