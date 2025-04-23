import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from LogisticRegression import LogisticRegression

# Load data
data = load_iris()
X = data.data[:, :2]
y = (data.target == 0).astype(int)  # Binary classification (class 0 vs rest)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

# Train model
model = LogisticRegression()
model.fit(X_train, y_train)

# Save model
np.savez("classifier1.npz", weights=model.weights, bias=model.bias)
