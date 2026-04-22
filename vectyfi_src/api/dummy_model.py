import pickle
from sklearn.ensemble import RandomForestClassifier
import numpy as np

#fictive data
X = np.random.rand(100, 3)  # 100 exemples, 3 features
y = np.random.randint(0, 2, 100)  # labels binaires

model = RandomForestClassifier()
model.fit(X, y)

# Sauvegarde du modèle
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

print("model.pkl créé !")
