import numpy as np
from sklearn.cluster import DBSCAN, SpectralClustering
from sklearn.metrics import silhouette_score

train_test_dataset = np.load('embeddings-audio-lab3.npz')
X_train, X_test = train_test_dataset['X_train'], train_test_dataset['X_test']
print(f"Shape of X_train: {X_train.shape}"), print(f"Shape of X_test: {X_test.shape}")

# spectral clustering
sc = SpectralClustering(n_clusters=20, random_state=0)
y_pred = sc.fit_predict(X_train)

silhouette = silhouette_score(X_train, y_pred)

# Print the silhouette score
print(f"Silhouette score of Spectral Clustering(Xtrain): {silhouette:.4f}")
print(f"Predicted labels from Spectral Clustering: {y_pred}")

# Calcul d'une "inertie" approchée : somme des carrés des distances au centroïde de chaque cluster
labels = y_pred
unique_labels = np.unique(labels)

# traiter l'éventuel label bruit (e.g. -1) s'il existe (utile pour DBSCAN)
valid_labels = unique_labels  # filtre si besoin : valid_labels = unique_labels[unique_labels != -1]

inertia_approx = 0.0
centers = []
for lab in valid_labels:
    mask = (labels == lab)
    if np.sum(mask) == 0:
        # cluster vide, ignorer
        centers.append(np.zeros(X_train.shape[1]))
        continue
    center = X_train[mask].mean(axis=0)
    centers.append(center)
    diff = X_train[mask] - center
    inertia_approx += np.sum((diff ** 2).sum(axis=1))

centers = np.array(centers)
print(f"Approx. inertia (sum squared distances to empirical centroids): {inertia_approx:.4f}")
