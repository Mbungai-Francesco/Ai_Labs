from sklearn.datasets import make_blobs
from matplotlib.colors import ListedColormap
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report,confusion_matrix

# Paramètres (modifiable)
n_samples = 300  # nombre total d'échantillons
centers = 4     # nombre de nuages/classes / labels
n_features = 2  # dimension d = 2 (2D)
random_state = 0

# Génération des données
X, y = make_blobs(n_samples=n_samples, centers=centers, n_features=n_features, random_state=random_state)
print("X.shape:", X.shape)
print("y.shape:", y.shape)
print("Premiers échantillons X:", X[:10])
print("Premiers labels:", y[:10])

# Visualisation
# plt.figure(figsize=(6,6))
# scatter = plt.scatter(X[:, 0], X[:, 1], c=y, cmap='tab10', s=30, edgecolor='k')
# plt.title(f"make_blobs: n_samples={n_samples}, centers={centers}")
# plt.xlabel("X0")
# plt.ylabel("X1")
# plt.gca().set_aspect('equal', adjustable='box')
# plt.colorbar(scatter, ticks=range(centers), label='Classe (label)')
# plt.show()

#Split des datas en train et test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state)
print("X_train.shape:", X_train.shape)
print("X_test.shape:", X_test.shape)
print("y_train.shape:", y_train.shape)
print("y_test.shape:", y_test.shape)
print("Premiers échantillons X_train:", X_train[:10])
print("Premiers labels y_train:", y_train[:10])
print("Premiers échantillons X_test:", X_test[:10])
print("Premiers labels y_test:", y_test[:10])

# Création et entraînement du modèle KNN
k = 5  # nombre de voisins
knn = KNeighborsClassifier(n_neighbors=k,n_jobs=1)
knn.fit(X_train, y_train)

# Prédiction sur les données d'entraînement
y_pred_train = knn.predict(X_train)
print("Prédiction sur les données d'entraînement:", y_pred_train[:10])

# Prédiction sur les données de test
y_pred_test = knn.predict(X_test)
print("Prédiction sur les données de test:", y_pred_test[:10])

# Calcul de la précision
accuracy_train = knn.score(X_train, y_train)
print(f"Précision sur les données d'entraînement: {accuracy_train * 100:.2f}%")

accuracy_test = knn.score(X_test, y_test)
print(f"Précision sur les données de test: {accuracy_test * 100:.2f}%")

# Visualisation des résultats
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap='viridis')
plt.title('Données d\'entraînement')

plt.subplot(1, 2, 2)
plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap='viridis')
plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap='viridis', marker='o', edgecolor='k', s=50, label='Train')
plt.scatter(X_test[:, 0], X_test[:, 1], c=y_pred_test, cmap='cool', marker='x')
plt.title('Données de test avec prédictions')

plt.show()

# Rapport de classification et matrice de confusion
print("Rapport de classification sur les données de test:")
print(classification_report(y_test, y_pred_test))
print("Matrice de confusion sur les données de test:")
print(confusion_matrix(y_test, y_pred_test))

# Visualisation des frontières de décision
classifier = knn
def plot_boundaries(classifier,X,Y,h=0.2):
    x0_min, x0_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x1_min, x1_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    x0, x1 = np.meshgrid(np.arange(x0_min, x0_max,h),
                         np.arange(x1_min, x1_max,h))
    dataset = np.c_[x0.ravel(),x1.ravel()]
    Z = classifier.predict(dataset)

    # Put the result into a color plot
    Z = Z.reshape(x0.shape)
    plt.figure()
    plt.pcolormesh(x0, x1, Z)

    # Plot also the training points
    plt.scatter(X[:, 0], X[:, 1], c=Y,
                edgecolor='k', s=20)
    plt.xlim(x0.min(), x0.max())
    plt.ylim(x1.min(), x1.max())
    
plot_boundaries(classifier,X_train,y_train)


