import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from LogisticRegression import LogisticRegression

# Load data
data = load_iris()
X = data.data[:, :2]  # Select first two features (sepal length, sepal width)
y = (data.target == 1).astype(int)  # Binary classification (class 1 vs rest)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

# Train model
model = LogisticRegression()
model.fit(X_train, y_train)

# Save model
np.savez("classifier2.npz", weights=model.weights, bias=model.bias)
print("Classifier 2 trained and saved.")
