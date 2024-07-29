
#importing the necessary libraries
from collections import Counter
import numpy as np

#calc euclidean distance by defining a formula in the form of a function
def euclidean_distance(x1, x2):
    return np.sqrt(np.sum((x1 - x2) ** 2))

#defining KNN algortihm as a class
#this can alo be called from the sci-kit learn library

class KNN:
    def __init__(self, k=3):
        self.k = k

#refer the fit() method and it's working in the notes
#where each row represents a sample (an individual iris flower) and each column represents a feature (a specific measurement of the iris flowers).
# The Iris dataset has four features for each sample:
# Sepal length in cm
# Sepal width in cm
# Petal length in cm
# Petal width in cm

#y is a 1D numpy array where each element corresponds to the class label of the respective sample in X.
# The Iris dataset has three classes (species of iris flowers):
# 0: Iris-setosa
# 1: Iris-versicolor
# 2: Iris-virginica
        
    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

#refer to predict() method in the notes
    def predict(self, X):
        y_pred = [self._predict(x) for x in X]  #the 'x' is like the 'ith' element inside 'X'
        return np.array(y_pred)

    def _predict(self, x):
        # Compute distances between 'x' in test datapoint to every 'x' in every training datapoint
        distances = [euclidean_distance(x, x_train) for x_train in self.X_train]  #every data point 'x' is trained from self.X_train and corresponding matrix is named as x_train

#We use x to represent a single test data point when calculating distances to all training data points. This allows the _predict method to find the nearest neighbors for that specific test data point and determine its predicted class label. 
#The use of x ensures that the distances are calculated between one test data point and each training data point, which is the         
# In the KNN algorithm, the process of prediction for each test data point involves comparing that single test data point to every training data point. 
# This is because the KNN algorithm is based on finding the nearest neighbors of a given point from the training data to classify it.
# Why Compare a Single Test Data Point to All Training Data Points:
# The KNN algorithm relies on the principle of proximity, meaning that a data point is classified based on the majority class of its nearest neighbors in the training data. Therefore, the comparison needs to be between each test data point and all training data points. Here’s why:

# Nearest Neighbors Concept:

# For each test data point x, you need to determine which training data points are closest to it (nearest neighbors).
# The "closeness" is determined by calculating the distance (e.g., Euclidean distance) between the test data point x and each training data point x_train.
# Majority Voting for Classification:

# After identifying the nearest neighbors for a test data point x, the algorithm performs a majority vote among the classes of these neighbors to predict the class of x.
# This requires knowing the distances from x to all training data points to accurately identify which are the nearest. 
              
        # Sort by distance and return indices of the first k neighbors
        k_idx = np.argsort(distances)[: self.k]
 
        # Extract the labels of the k nearest neighbor training samples
        k_neighbor_labels = [self.y_train[i] for i in k_idx]
 
        # return the most common class label
        most_common = Counter(k_neighbor_labels).most_common(1)
        return most_common[0][0]


if __name__ == "__main__":
    # Imports
    from matplotlib.colors import ListedColormap
    from sklearn import datasets
    from sklearn.model_selection import train_test_split

    cmap = ListedColormap(["#FF0000", "#00FF00", "#0000FF"])

#defining accuracy as a metric in order to gague the model's trueness
    def accuracy(y_true, y_pred):
        accuracy = np.sum(y_true == y_pred) / len(y_true)
        return accuracy

    iris = datasets.load_iris()
    X, y = iris.data, iris.target

#splitting the dataset into test and train
#for testing or validation of the model, 20% is reserved whereas the rest 80% is used for model training
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=1234
    )

    k = 5
    clf = KNN(k=k)
    clf.fit(X_train, y_train)
    predictions = clf.predict(X_test)
    print("KNN classification accuracy", accuracy(y_test, predictions))