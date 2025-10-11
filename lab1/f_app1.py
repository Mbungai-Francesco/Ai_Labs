import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform

# Generate 10 numbers from a normal distribution with mean=0 and std=1
cloud1 = np.random.normal(loc=[0,0], scale=1, size=(100,2))
cloud2 = np.random.normal(loc=[5,5], scale=1, size=(100,2))

plt.figure(figsize=(6, 6))
plt.scatter(cloud1[:, 0], cloud1[:, 1], color='blue', label='Cloud 1')
plt.scatter(cloud2[:, 0], cloud2[:, 1], color='red', label='Cloud 2')
plt.legend()
plt.title("Two Clouds of Points in 2D")
plt.xlabel("X-axis")
plt.ylabel("Y-axis")
plt.grid(True)
plt.show()

points = np.vstack((cloud1, cloud2))

dist_matrix = squareform(pdist(points, metric='euclidean'))

print(dist_matrix)

# print(cloud.shape)
# print("Arraay from normal distribution:\n")
# print(cloud1)
# print(cloud2)