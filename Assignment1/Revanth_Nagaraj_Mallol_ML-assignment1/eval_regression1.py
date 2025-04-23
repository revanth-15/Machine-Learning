# importing
import pickle
import matplotlib.pyplot as plt
from train_regression1 import model_1_X_train , model_1_X_test, model_1_y_train,reg, model_1_y_test,model_1_X, model_1_y


# loading the model
with open('reg_model_1.pkl', 'rb') as f:
    model = pickle.load(f)

# Prediction
model_1_y_pred = reg.predict(model_1_X_test)
print("Predictions for sepal width from sepal length are:\n", model_1_y_pred)

# MSE
print("MSE is:", reg.score(model_1_X_test, model_1_y_test))


# Visualize the prediction
fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(model_1_X_test, model_1_y_test,label='Test values')
ax.plot(model_1_X_test, model_1_y_pred, c='r',label='Predicted Regression line')
plt.xlabel('Sepal Length')
plt.ylabel('Sepal Width')
ax.legend()
plt.title('Actual values of Sepal Length and Sepal width predicted regression line')
plt.show()