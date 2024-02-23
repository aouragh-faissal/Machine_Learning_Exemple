import numpy as np
from sklearn.datasets import make_regression
import matplotlib.pyplot as plt


#to always reproduce the same dataset
np.random.seed(0) 
x, y = make_regression(n_samples=100, n_features=1, noise=10)
#show dataset
plt.scatter(x, y) 

#resize x & y
y = y.reshape(y.shape[0], 1)
X = np.hstack((x, np.ones(x.shape)))

np.random.seed(0) 
theta = np.random.randn(2, 1)

def model(X, theta):
    return X.dot(theta)

plt.scatter(x, y , c='b')
plt.plot(x, model(X, theta), c='r')
plt.show()

#Cost function
def cost_function(X, y, theta):
    m = len(y)
    return 1/(2*m) * np.sum((model(X, theta) - y)**2)

#Gradient
def grad(X, y, theta):
    m = len(y)
    return 1/m * X.T.dot(model(X, theta) - y)


#Gradient_descent
def gradient_descent(X, y, theta, learning_rate, n_iterations):
    
    #storage table to record the evolution of the cost of the model
    cost_history = np.zeros(n_iterations) 
    
    for i in range(0, n_iterations):
        theta = theta - learning_rate * grad(X, y, theta) 
        cost_history[i] = cost_function(X, y, theta) 
        
    return theta, cost_history


#coefficient of determination R**2
def coef_determination(y, pred):
    u = ((y - pred)**2).sum()
    v = ((y - y.mean())**2).sum()
    return 1 - u/v

########################################################################################

n_iterations = 1000
learning_rate = 0.01
theta_final, cost_history = gradient_descent(X, y, theta, learning_rate, n_iterations)
predictions = model(X, theta_final)

# Displays the prediction results compared to our Dataset 
plt.scatter(x, y)
plt.plot(x, predictions, c='r')
plt.show()

#learning curve
plt.plot(range(n_iterations), cost_history)
plt.show()

print(coef_determination(y, predictions))