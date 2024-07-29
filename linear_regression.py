
#Logistic regression
#prediction of a class of values
#approximation using linear function
#y = ax+b
#y - slope
#b - bias
#calculate 'w' and 'b'
#we define a cost function
#diff b/w actual and approximate value
#we train the samples using the 'cost function' formula
#we then calculate the gradient of the cost function w.r.t 'w' & 'b', weights and biases respectively
#check cost function details
#we then do a gradient descent
#we check the steepest descent and we do this iteratively
#until we get a minimum value
#check links in the video for formulas
#Linear regression is basically an ML where we find an approximation function which fits right in the middle
#of a cluster of linearly spread sample points

import numpy as np
class LinearRegression:

      def __init__(self, lr = 0.001, n_iters = 1000):
            self.lr = lr
            self. n_iters = n_iters
            self.weights = None
            self.bias = None

#we need to implement the gradient descent method in this section
#the gradient descent is a parabola (a U shaped curve), and we need the cost function which is 
#a straight line to begin from a point of contact to the parabola.
#we follow this line all the way through the curve into it's lowest point
            
      def fit(self, X, y):    
            #it's an iterative process and the 'why' part of the gradient descent is in the notes
            #basically we need to calculate the cost function, the cost fn is something that finds the optimal fit between the actual and predicted target values
            #the cost function is calculated using certain parameters which are being defined below

            n_samples, n_features = X.shape
            self.weights = np.zeros(n_features)
            self.bias = 0
            y_predicted = np.dot(X, self.weights) + self.bias
            dw = (1/n_samples) * np.dot(X.T, (y_predicted - y))  #derivative formaula for weight
            db = (1/n_samples) * np.sum(y_predicted - y)

            self.weights -= self.lr * dw
            self.bias -= self.lr * db

      def predict(self, X):
            y_predicted = np.dot(X, self.weights) + self.bias
            return y_predicted

