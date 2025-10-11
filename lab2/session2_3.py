import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

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

# Print the confusion matrix
print(f"Confusion matrix: \n{rf.predict_proba(X_test)}")

#Print the accuracy
accuracy = rf.score(X_test, y_test)
print(f"Accuracy: {accuracy * 100:.2f}%")
