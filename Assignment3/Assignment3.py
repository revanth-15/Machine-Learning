import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import time
from collections import Counter
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import cross_val_score
import xgboost as xgb
import lightgbm as lgb

# Step 1: Load Titanic dataset
data = pd.read_csv("https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv")

# Step 2: Data Preprocessing
def preprocess_data(df):
    df = df.copy()
    df = df.drop(["Name", "Ticket", "Cabin"], axis=1)
    df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])
    df["Age"] = df["Age"].fillna(df["Age"].median())
    df["Fare"] = df["Fare"].fillna(df["Fare"].median())
    df = pd.get_dummies(df, columns=["Sex", "Embarked"], drop_first=True)
    return df

# Apply preprocessing
data = preprocess_data(data)

# Step 3: Split dataset
def train_test_split(X, y, test_size=0.2, random_state=42):
    np.random.seed(random_state)
    indices = np.arange(len(X))
    np.random.shuffle(indices)
    split = int(len(X) * (1 - test_size))
    train_idx, test_idx = indices[:split], indices[split:]
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]

X = data.drop("Survived", axis=1).values
y = data["Survived"].values
X_train, X_test, y_train, y_test = train_test_split(X, y)

# Step 4: Implement Decision Tree from Scratch
class DecisionTree:
    def __init__(self, max_depth=5, min_samples_split=2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.tree = None

    def fit(self, X, y):
        self.tree = self._build_tree(X, y, depth=0)

    def _build_tree(self, X, y, depth):
        if depth >= self.max_depth or len(set(y)) == 1 or len(y) < self.min_samples_split:
            return Counter(y).most_common(1)[0][0]
        
        feature, threshold = self._best_split(X, y)
        if feature is None:
            return Counter(y).most_common(1)[0][0]
        
        left_mask = X[:, feature] <= threshold
        right_mask = ~left_mask
        left_subtree = self._build_tree(X[left_mask], y[left_mask], depth + 1)
        right_subtree = self._build_tree(X[right_mask], y[right_mask], depth + 1)
        
        return {"feature": feature, "threshold": threshold, "left": left_subtree, "right": right_subtree}

    def _best_split(self, X, y):
        best_gini, best_feature, best_threshold = float('inf'), None, None
        for feature in range(X.shape[1]):
            thresholds = np.unique(X[:, feature])
            for threshold in thresholds:
                left_mask = X[:, feature] <= threshold
                right_mask = ~left_mask
                gini = self._gini_index(y[left_mask], y[right_mask])
                if gini < best_gini:
                    best_gini, best_feature, best_threshold = gini, feature, threshold
        return best_feature, best_threshold

    def _gini_index(self, left, right):
        def gini(y):
            m = len(y)
            if m == 0:
                return 0
            return 1.0 - sum((np.sum(y == c) / m) ** 2 for c in np.unique(y))
        
        n = len(left) + len(right)
        return (len(left) / n) * gini(left) + (len(right) / n) * gini(right)

    def predict(self, X):
        return np.array([self._predict_single(x, self.tree) for x in X])

    def _predict_single(self, x, node):
        if isinstance(node, dict):
            if x[node["feature"]] <= node["threshold"]:
                return self._predict_single(x, node["left"])
            else:
                return self._predict_single(x, node["right"])
        return node

# Step 5: Train and Evaluate Decision Tree
dt = DecisionTree(max_depth=5)
dt.fit(X_train, y_train)
dt_preds = dt.predict(X_test)

# Step 6: Print Accuracy
def accuracy(y_true, y_pred):
    return np.mean(y_true == y_pred)

print("Decision Tree Accuracy:", accuracy(y_test, dt_preds))

# Step 7: Compare with Scikit-Learn's Decision Tree
sklearn_dt = DecisionTreeClassifier(max_depth=5, random_state=42)
sklearn_dt.fit(X_train, y_train)
sklearn_dt_preds = sklearn_dt.predict(X_test)

print("Scikit-Learn Decision Tree Accuracy:", accuracy(y_test, sklearn_dt_preds))

# Step 8: Plot Decision Tree
plt.figure(figsize=(20, 10))
plot_tree(sklearn_dt, feature_names=data.drop("Survived", axis=1).columns, filled=True)
plt.show()

# Step 9: Compare with Scikit-Learn's Random Forest
sklearn_rf = RandomForestClassifier(n_estimators=100, random_state=42)
sklearn_rf.fit(X_train, y_train)
sklearn_rf_preds = sklearn_rf.predict(X_test)

print("Scikit-Learn Random Forest Accuracy:", accuracy(y_test, sklearn_rf_preds))

# Step 10: Compare with Scikit-Learn's AdaBoost
sklearn_adaboost = AdaBoostClassifier(estimator=DecisionTreeClassifier(max_depth=1), n_estimators=50, random_state=42)
sklearn_adaboost.fit(X_train, y_train)
sklearn_adaboost_preds = sklearn_adaboost.predict(X_test)

print("Scikit-Learn AdaBoost Accuracy:", accuracy(y_test, sklearn_adaboost_preds))

# Step 11: Compare with Scikit-Learn's Gradient Boosting
sklearn_gb = GradientBoostingClassifier(n_estimators=100, random_state=42)
sklearn_gb.fit(X_train, y_train)
sklearn_gb_preds = sklearn_gb.predict(X_test)

print("Scikit-Learn Gradient Boosting Accuracy:", accuracy(y_test, sklearn_gb_preds))

# Step 12: Compare with XGBoost
xgb_model = xgb.XGBClassifier(n_estimators=100, random_state=42)
xgb_model.fit(X_train, y_train)
xgb_preds = xgb_model.predict(X_test)

print("XGBoost Accuracy:", accuracy(y_test, xgb_preds))

# Step 13: Compare with LightGBM
lgb_model = lgb.LGBMClassifier(n_estimators=100, random_state=42)
lgb_model.fit(X_train, y_train)
lgb_preds = lgb_model.predict(X_test)

print("LightGBM Accuracy:", accuracy(y_test, lgb_preds))

# Step 15: Compare Training Time
def train_model(model, X_train, y_train, X_test, y_test, model_name):
    start_time = time.time()
    model.fit(X_train, y_train)
    training_time = time.time() - start_time
    
    preds = model.predict(X_test)
    accuracy = np.mean(y_test == preds)
    
    return {
        "Model": model_name,
        "Accuracy": accuracy,
        "Training Time (s)": training_time
    }

# Define models
models = [
    (DecisionTree(max_depth=5), "Decision Tree"),
    (RandomForestClassifier(n_estimators=100, random_state=42), "Random Forest"),
    (AdaBoostClassifier(estimator=DecisionTreeClassifier(max_depth=1), n_estimators=50, random_state=42), "AdaBoost"),
    (GradientBoostingClassifier(n_estimators=100, random_state=42), "Gradient Boosting"),
    (xgb.XGBClassifier(n_estimators=100, random_state=42), "XGBoost"),
    (lgb.LGBMClassifier(n_estimators=100, random_state=42), "LightGBM"),
]

# Train and evaluate models
results = [train_model(model, X_train, y_train, X_test, y_test, name) for model, name in models]

# Convert results to DataFrame
results_df = pd.DataFrame(results)
print(results_df)

# Step 16: Plot Feature Importances for Decision Trees and Random Forest
feature_names = data.drop("Survived", axis=1).columns

def plot_feature_importances(model, feature_names, model_name):
    if model_name == "Decision Tree":
        importances = np.zeros(len(feature_names))
        for node in model.tree.values():
            if isinstance(node, dict):
                importances[node["feature"]] += 1
    elif model_name == "Random Forest":
        importances = model.feature_importances_
    else:
        return
    
    indices = np.argsort(importances)[::-1]
    plt.figure(figsize=(10, 5))
    sns.barplot(x=importances[indices], y=np.array(feature_names)[indices])
    plt.title(f"{model_name} Feature Importances")
    plt.show()

plot_feature_importances(models[0][0], feature_names, "Decision Tree")
plot_feature_importances(models[1][0], feature_names, "Random Forest")

# Step 17: Plot ROC Curve
def plot_roc_curve(y_test, preds, model_name):
    fpr, tpr, _ = roc_curve(y_test, preds)
    auc_score = auc(fpr, tpr)
    
    plt.figure(figsize=(6, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f"{model_name} (AUC = {auc_score:.2f})")
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'{model_name} ROC Curve')
    plt.legend(loc="lower right")
    plt.show()

for result, (model, name) in zip(results, models):
    plot_roc_curve(y_test, model.predict(X_test), result["Model"])

# Step 18: Compare Cross-Validation Score
def train_model_cv(model, X_train, y_train, model_name):
    cross_val = cross_val_score(model, X_train, y_train, cv=5).mean()
    return {
        "Model": model_name,
        "Cross-Validation Score": cross_val
    }
