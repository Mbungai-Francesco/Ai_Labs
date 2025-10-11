import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform

# 1. Générer deux nuages de points (100 points chacun en 2D)
np.random.seed(0)  # pour la reproductibilité
cloud1 = np.random.randn(100, 2) + np.array([-2, -2])   # centré autour de (0,0)
print("cloud1.shape:", cloud1.shape)
print("x (5 premiers):", cloud1[:5, 0])
print("y (5 premiers):", cloud1[:5, 1])
cloud2 = np.random.randn(100, 2) + np.array([2, 2])   # centré autour de (5,5)
print("cloud2.shape:", cloud2.shape)
print("x (5 premiers):", cloud2[:5, 0])
print("y (5 premiers):", cloud2[:5, 1])

# Fusionner les deux nuages
data = np.vstack([cloud1, cloud2])  # shape (200,2)
print("data.shape:", data.shape)
print("x (5 premiers):", data[:5, 0])
print("x (100-105 ieme elements):", data[100:105, 0])
print("y (5 premiers):", data[:5, 1])

# 2. Visualiser les deux nuages
plt.figure(figsize=(5,5))
plt.scatter(cloud1[:,0], cloud1[:,1], color='blue', label='Cloud 1')
plt.scatter(cloud2[:,0], cloud2[:,1], color='red', label='Cloud 2')
plt.legend()
plt.title("Deux nuages de points en 2D")
plt.xlabel("X")
plt.ylabel("Y")
plt.show()

# 3. Calculer la matrice des distances euclidiennes
distances = pdist(data, metric='euclidean')
print("len(distances):", len(distances))  # doit être 200*199/2 = 19900
print("Exemples (5 premiers) distances condensées:", distances[:5])
print("Exemples (5 derniers) distances condensées:", distances[-5:])
distance_matrix = squareform(distances)  # matrice 200x200

# 4. Visualiser la matrice des distances
plt.figure(figsize=(6,6))
plt.matshow(distance_matrix, fignum=1, cmap='viridis')
plt.colorbar(label="Distance euclidienne")
plt.title("Matrice des distances pairwise")
plt.show()
