import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from LogisticRegression import LogisticRegression # importing the classifier
from mlxtend.plotting import plot_decision_regions
import matplotlib.pyplot as plt

# Load the Iris dataset
iris = load_iris()
X = iris.data
y = (iris.target == 2).astype(int)  # Binary classification of given class with other classes

# Standardize the input features
skale = StandardScaler()
X_standardized = skale.fit_transform(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_standardized, y, test_size=0.1, random_state=42,stratify=y)

# fitting the model
model = LogisticRegression(lr=0.01, epoch=100)
model.fit(X_train, y_train) # fitting for all 4 features
print("\nActual values",y_test)

# predictions
y_pred_p = model.predict(X_test)
print("\nPredicted values:", y_pred_p)
# Evaluate acc
acc = np.mean(y_pred_p == y_test)
print(f"\nAccuracy: {acc}")

