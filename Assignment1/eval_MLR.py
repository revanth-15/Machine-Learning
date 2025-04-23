import matplotlib.pyplot as plt
import pickle
from Multiple_Linear_Regression import X_train, X_test, y_train, y_test

with open('mlr_model.pkl', 'rb') as f:
    model = pickle.load(f)


# Predict on the test data
predictions = model.predict(X_test)
print("Predictions of petal values: \n", predictions)
# Evaluate performance
mse = model.mean_squared_error_loss(y_test, predictions)
print("\nMean Squared Error on Test Set:", mse)
# Visualize the prediction
fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(X_test, y_test,c='green', label='Actual sepal values')  # Actual values
ax.scatter(X_test, predictions, c='r',label='Predicted petal values')  # Predicted values
plt.title('Actual petal values vs Predicted Petal values')
ax.legend()
plt.show()