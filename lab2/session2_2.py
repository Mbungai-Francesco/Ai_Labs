import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.datasets import make_classification
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Create synthetic data
X, y = make_classification(
    n_samples=200,      # total data points
    n_features=2,       # only 2 features (for 2D visualization)
    n_informative=2,    # both features useful
    n_redundant=0,      # no duplicate features
    n_clusters_per_class=1,
    random_state=0
)
print("X shape:", X.shape)
print("y shape:", y.shape)

# Create model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

rf = RandomForestClassifier(n_estimators=10, random_state=0)
rf.fit(X_train, y_train)

y_pred = rf.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))

# Evaluate
accuracy = rf.score(X_test, y_test)
print(" \n Accuracy:", accuracy)
print(classification_report(y_test, y_pred))

plt.scatter(X[:, 0], X[:, 1], c=y, cmap='bwr')
plt.title("Synthetic Binary Classification Dataset")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.show()