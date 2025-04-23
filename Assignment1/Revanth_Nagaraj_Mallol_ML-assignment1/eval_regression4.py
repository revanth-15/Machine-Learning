# importing
import pickle
import matplotlib.pyplot as plt
from train_regression4 import model_1_X_train , model_1_X_test, model_1_y_train, model_1_y_test,reg,model_1_X, model_1_y

# loading the model
with open('reg_model_4.pkl', 'rb') as f:
    model = pickle.load(f)

# Prediction
model_1_y_pred = reg.predict(model_1_X_test)
print("Predictions for sepal length from petal width are:\n", model_1_y_pred)

# MSE
print("MSE is:\n",reg.score(model_1_X_test, model_1_y_test))


# Visualize the prediction
fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(model_1_X_test, model_1_y_test,label="Test Values")
ax.plot(model_1_X_test, model_1_y_pred, c='r',label="Predicted Line")
plt.xlabel('Petal Length')
plt.ylabel('Petal width')
plt.title('Actual values of petal Length and petal width predicted regression line')
plt.show()
plt.show()
