import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
train_test_dataset = np.load('embeddings-audio-lab2.npz')

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_dataset['X_train'], train_test_dataset['X_test'], train_test_dataset['y_train'], train_test_dataset['y_test']

# Test print statements to verify data loading
# class_names = np.unique(y_train)
# print(f"Class names : {class_names}")
# print(f"Shape of X_train: {X_train.shape}"), print(f"Shape of y_test: {y_test.shape}")

#Fit the model
rf = RandomForestClassifier(n_estimators=150, random_state=0)
rf.fit(X_train, y_train)

# Predict on the test set
y_pred = rf.predict(X_test)

# Print the classification report
print(classification_report(y_test, y_pred))

# Compute and print the confusion matrix (counts)
conf_mat = confusion_matrix(y_test, y_pred)
print(f"Confusion matrix (counts): \n{conf_mat}")

# Create a heatmap of the confusion matrix
sns.heatmap(conf_mat, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion matrix')
plt.show()


#Print the accuracy
accuracy = rf.score(X_test, y_test)
print(f"Accuracy: {accuracy * 100:.2f}%")

plt.figure(figsize=(8, 6))
sns.heatmap(conf_mat, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.xlabel('Predicted Accent')
plt.ylabel('True Accent')
plt.title('Confusion Matrix: Random Forest on Audio Embeddings')
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=150)