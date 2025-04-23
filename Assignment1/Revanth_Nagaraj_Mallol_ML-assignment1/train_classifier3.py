import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from LogisticRegression import LogisticRegression

# Load data
data = load_iris()
X = data.data  # Use all four features
y = (data.target == 2).astype(int)  # Binary classification (class 2 vs rest)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

# Train model
model = LogisticRegression()
model.fit(X_train, y_train)

# Save model
np.savez("classifier3.npz", weights=model.weights, bias=model.bias)
print("Classifier 3 trained and saved.")
