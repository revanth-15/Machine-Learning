import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import StratifiedShuffleSplit # to get even classes in the test split
from sklearn.model_selection import train_test_split
from LinearRegression import LinearRegression
#%%
iris = load_iris()
iris
#%%
# Splitting data into test set with even classes
lr_data = iris.data

# model_1_X = np.vstack((lr_data[:,0], lr_data[:,2])).transpose()
model_1_X = lr_data[:,0] # sepal length
model_1_y = lr_data[:,2] # target= petal length

# # Use StratifiedShuffleSplit to select 10% of the data with an even split
# sss = StratifiedShuffleSplit(n_splits=1, test_size=0.1, random_state=42)
#
# for train_index, test_index in sss.split(model_1_X, model_1_y):
#     X_subset, y_subset = model_1_X[test_index], model_1_y[test_index]
# Now, X_subset and y_subset contain the randomly selected 10% of the Iris dataset with an even split of each class
#%%
model_1_X_train , model_1_X_test, model_1_y_train, model_1_y_test = train_test_split(model_1_X, model_1_y, test_size=0.1, random_state=42)

#%%
reg = LinearRegression()
reg.fit(model_1_X_train, model_1_y_train)
b = reg.bias
m = reg.weights[0]

with open('reg_model_2.pkl', 'wb') as f:
    pickle.dump(reg,f)