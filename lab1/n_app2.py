import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform

# Charger les embeddings depuis le fichier .npz
data = np.load('embeddings-audio-lab1.npz')
print("Clés dans le fichier .npz :", data.files)
print("Contenu des données :", {key: data[key].shape for key in data.files})
embeddings = data['embeddings']  # Shape: (80, d) — 80 échantillons, d dimensions

print(f"Shape des embeddings : {embeddings.shape}")

from scipy.spatial.distance import pdist, squareform

# Calculer la matrice de distance euclidienne (pairwise)
distance_matrix = squareform(pdist(embeddings, metric='euclidean'))

print(f"Shape de la matrice de distance : {distance_matrix.shape}")

# Visualiser la matrice des distances
plt.figure(figsize=(8, 6))
plt.matshow(distance_matrix, fignum=1, cmap='viridis')
plt.title("Matrice de distance euclidienne entre les embeddings audio")
plt.colorbar(label="Distance euclidienne")
plt.xlabel("Échantillon i")
plt.ylabel("Échantillon j")
plt.show()

## experience
# Supposons que les 40 premiers sont "Dog barking", les 40 suivants "Fireworks"
n_dog = 40
n_fireworks = 40

# Masquer les valeurs hors des blocs pour mieux visualiser
mask = np.ones_like(distance_matrix, dtype=bool)
mask[:n_dog, n_dog:] = False  # Masque entre classes
mask[n_dog:, :n_dog] = False  # Masque entre classes

plt.figure(figsize=(5, 5))
plt.matshow(distance_matrix * mask, fignum=1, cmap='viridis')
plt.title("Distances intra-classes seulement")
plt.colorbar()
plt.show()

# Interprétation:
# La matrice de distance euclidienne calculée sur les embeddings montre une structure clairement séparée en deux blocs : 
# les échantillons de la même classe (“Dog barking” ou “Fireworks”) sont proches les uns des autres dans l’espace latent,
# tandis que les échantillons de classes différentes sont éloignés. 
# Cela démontre que les embeddings appris par le modèle CNN14 capturent des caractéristiques discriminantes pertinentes pour distinguer les deux types de sons. 
# Contrairement aux données brutes (qui sont très complexes et non structurées),
# les embeddings permettent une séparation visuelle et quantitative nette des classes, 
# ce qui facilite grandement les tâches ultérieures comme la classification ou la recherche similaire.