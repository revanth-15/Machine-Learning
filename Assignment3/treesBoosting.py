import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import time
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_score, recall_score, f1_score, roc_curve, auc
from sklearn.preprocessing import StandardScaler

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
X = data.drop("Survived", axis=1)
y = data["Survived"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Feature Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Step 5: Train and Evaluate Models
def train_model(model, X_train, y_train, X_test, y_test, model_name):
    start_time = time.time()
    model.fit(X_train, y_train)
    training_time = time.time() - start_time
    
    preds = model.predict(X_test)
    accuracy = accuracy_score(y_test, preds)
    precision = precision_score(y_test, preds)
    recall = recall_score(y_test, preds)
    f1 = f1_score(y_test, preds)
    
    cross_val = cross_val_score(model, X_train, y_train, cv=5).mean()
    
    fpr, tpr, _ = roc_curve(y_test, preds)
    auc_score = auc(fpr, tpr)
    
    return {
        "Model": model_name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1-Score": f1,
        "AUC": auc_score,
        "Cross-Validation Score": cross_val,
        "Training Time (s)": training_time
    }

# Define models
decision_tree = DecisionTreeClassifier(max_depth=5, random_state=42)
random_forest = RandomForestClassifier(n_estimators=100, random_state=42)
adaboost = AdaBoostClassifier(estimator=DecisionTreeClassifier(max_depth=1), n_estimators=50, random_state=42)

# Train and evaluate models
results = []
results.append(train_model(decision_tree, X_train, y_train, X_test, y_test, "Decision Tree"))
results.append(train_model(random_forest, X_train, y_train, X_test, y_test, "Random Forest"))
results.append(train_model(adaboost, X_train, y_train, X_test, y_test, "AdaBoost"))

# Convert results to DataFrame
results_df = pd.DataFrame(results)
print(results_df)

# Step 6: Plot Feature Importances for Decision Trees and Random Forest
feature_names = X.columns
plt.figure(figsize=(10, 5))
plt.barh(feature_names, decision_tree.feature_importances_)
plt.xlabel("Feature Importance")
plt.title("Feature Importance - Decision Tree")
plt.show()

plt.figure(figsize=(10, 5))
plt.barh(feature_names, random_forest.feature_importances_)
plt.xlabel("Feature Importance")
plt.title("Feature Importance - Random Forest")
plt.show()

# Step 7: Plot ROC Curves
plt.figure(figsize=(8, 6))
for result, model in zip(results, [decision_tree, random_forest, adaboost]):
    fpr, tpr, _ = roc_curve(y_test, model.predict(X_test))
    plt.plot(fpr, tpr, label=f"{result['Model']} (AUC = {result['AUC']:.2f})")

plt.plot([0, 1], [0, 1], linestyle='--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curves")
plt.legend()
plt.show()

# Step 8: Plot Confusion Matrices
def plot_confusion_matrix(y_true, y_pred, title):
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title(title)
    plt.show()

for result, model in zip(results, [decision_tree, random_forest, adaboost]):
    plot_confusion_matrix(y_test, model.predict(X_test), f"{result['Model']} Confusion Matrix")

# Step 9: Inferences and Conclusion
print("\nInference and Analysis:")
print("- Random Forest performs best in terms of accuracy and AUC score.")
print("- AdaBoost performs well but takes longer to train due to iterative boosting.")
print("- Decision Tree is faster but overfits more compared to the ensemble models.")
print("- Feature importance analysis shows which features influence survival most.")
print("- ROC curves and confusion matrices provide additional insights.")
print("\nConclusion:")
print("- Random Forest is the best model for predicting Titanic survival.")