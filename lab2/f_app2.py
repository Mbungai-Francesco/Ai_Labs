import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

# Generate 3 clouds (clusters) of 2D points
X, y = make_blobs(
    n_samples=300,      # total number of points
    n_features=2,       # dimension = 2 (x, y)
    centers=2,          # number of clusters
    cluster_std=1.0,    # spread of each cluster
    random_state=0      # reproducibility
)

X_train, X_test, y_train, y_test = train_test_split(X[:, 0],X[:, 1],train_size=0.80,test_size=0.20,random_state=0)
# x_test.shape, y_train.shape, y_test.shape

print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

k = 1
classifier = KNeighborsClassifier(n_neighbors=k, n_jobs=1)

# Train the classifier
classifier.fit(X_train, y_train)

# Evaluate accuracy
train_acc = classifier.score(X_train, y_train)
test_acc = classifier.score(X_test, y_test)

print(f"K = {k}")
print(f"Training accuracy: {train_acc:.3f}")
print(f"Test accuracy: {test_acc:.3f}")

# Visualize the generated data
# Uncomment the lines below to see the plot

# plt.figure(figsize=(6, 6))
# plt.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis', s=40)
# plt.title("Synthetic 2D Data Generated with make_blobs")
# plt.xlabel("Feature 1 (x)")
# plt.ylabel("Feature 2 (y)")
# plt.grid(True)
# plt.show()
