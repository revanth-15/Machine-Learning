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
y = (iris.target == 2).astype(int)  # Binary classification of class 2 vs. rest

# Standardize the input features
skale = StandardScaler()
X_standardized = skale.fit_transform(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_standardized, y, test_size=0.1, random_state=42, stratify=y)

# fitting the model
model = LogisticRegression(lr=0.01, epoch=100)
model.fit(X_train[:, 0], y_train) # fitting for input sepal length
print("\nActual values",y_test)

# predictions
y_pred_p = model.predict(X_test[:, 0])
print("\nPredicted values:", y_pred_p)
# Evaluate acc
acc = np.mean(y_pred_p == y_test)
print(f"\nAccuracy: {acc}")


# plotting
plot_decision_regions(X_test[:, 0].reshape(-1, 1), y_test, clf=model, legend=2)
plt.title('Logistic Regression Decision Regions for classes')
plt.show()