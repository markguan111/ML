''' Using the Diabetes dataset that is in scikit-learn, answer the questions below and create a scatterplot
graph with a regression line '''

import matplotlib.pylab as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn import datasets
from sklearn.datasets import load_diabetes 

#how many sameples and How many features?

diabetes = datasets.load_diabetes()

print(diabetes.data.shape)


# What does feature s6 represent?
print(diabetes.data.shape)
#print out the coefficient

from sklearn.model_selection import train_test_split

X_train , X_test, y_train, y_test = train_test_split(
    diabetes.data,diabetes.target ,random_state=11
)
#1
mymodel = LinearRegression()

#2 use fit to train our model
mymodel.fit(X_train,y_train)

#priint out the coefficient
print(mymodel.coef_)

#print out the intercept
print(mymodel.intercept_)

# 3 use predict to test your model
predicted = mymodel.predict(X_test)
expected = y_test


# create a scatterplot with regression line
plt.plot(expected,predicted,".")



x = np.linspace(0,330,100)
print(x)
y = x


plt.plot(x,y)
plt.show()
