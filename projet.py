print("hello world")
import os
import pandas as pd

# Se placer dans le bon dossier
#os.chdir("Projet Etude") 

# Affiche la liste des fichiers pour vérification
print("Fichiers disponibles :", os.listdir())

# Lecture du fichier CSV bien formaté
df = pd.read_csv("agenda-nantes.csv", encoding='utf-8', sep=';')
print(df.head())
# garder les colonnes utiles et nettoyer
evenements = df[['Date', 'Heure de début', 'Lieu : quartier', 'Types', 'Thèmes',
                 'Géolocalisation (latitude, longitude)', 'Longitude', 'Latitude', 'Lieu : adresse']].copy()

evenements = evenements.dropna(subset=['Date'])  # supprimer lignes sans date
evenements['Date'] = pd.to_datetime(evenements['Date'], errors='coerce')
evenements = evenements.dropna(subset=['Date'])  # supprimer si date non reconnue

# Ajouter des colonnes temporelles
evenements['Jour_semaine'] = evenements['Date'].dt.day_name()
evenements['Mois'] = evenements['Date'].dt.month
evenements['Annee'] = evenements['Date'].dt.year
evenements['Evenement'] = 1  # présence événement = 1

print(evenements.head())
#Regrouper les événements par date et quartier
base = evenements.groupby(['Date', 'Lieu : quartier']).agg({'Evenement': 'sum'}).reset_index()
base['Evenement'] = (base['Evenement'] > 0).astype(int)  # 1 si ≥1 événement
base['Jour_semaine'] = base['Date'].dt.day_name()
base['Mois'] = base['Date'].dt.month
base['Annee'] = base['Date'].dt.year
print (base.head ())



from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report


from sklearn.preprocessing import LabelEncoder

# Créer les encodages sur 'Jour_semaine' et 'Lieu : quartier'
le_jour = LabelEncoder()
le_quartier = LabelEncoder()

base['Jour_enc'] = le_jour.fit_transform(base['Jour_semaine'])
base['Quartier_enc'] = le_quartier.fit_transform(base['Lieu : quartier'])

X = base[['Jour_enc', 'Mois', 'Annee', 'Quartier_enc']]
y = base['Evenement']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
print(classification_report(y_test, y_pred))

# Exemple : prédire s'il y aura un événement à "Centre-ville" le 20 mai 2025
import datetime

date_ex = datetime.datetime(2025, 5, 20)
jour = le_jour.transform([date_ex.strftime("%A")])[0]
mois = date_ex.month
annee = date_ex.year
quartier = le_quartier.transform(["Centre-ville"])[0]

# Créer l’entrée
X_new = [[jour, mois, annee, quartier]]
prediction = clf.predict(X_new)

if prediction[0] == 1:
    print("✅ Il y aura probablement un événement ce jour-là.")
else:
    print("❌ Aucun événement prévu.")



