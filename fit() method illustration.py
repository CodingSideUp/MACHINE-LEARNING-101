from sklearn.linear_model import LinearRegression

# Sample data
X = [[1], [2], [3], [4]]
y = [0, 1, 2, 3]

# Create a Linear Regression model
model = LinearRegression()

# Train the model
model.fit(X, y)

from sklearn.linear_model import LinearRegression

# Sample data
X_train = [[1], [2], [3], [4]]
y_train = [0, 1, 2, 3]
X_test = [[5], [6]]

# Create and train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
predictions = model.predict(X_test)
print(predictions)
