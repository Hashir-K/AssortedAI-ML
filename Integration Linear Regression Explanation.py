# Introduction
# In this walkthrough, we’ll review the steps from the linear regression lab, explaining each part of the process and providing the correct solution. This guide will help you verify your work and understand the reasoning behind each step of implementing a linear regression model. By following this guide, you will better understand how to build and evaluate a model that predicts house prices based on square footage, a fundamental task in machine learning regression problems.

# By the end of this walkthrough, you will be able to:
# Understand the dataset.
# Train the model.
# Evaluate the model.
# Visualize the results.

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# 1. Loading and preparing the data
# In the lab, we worked with a dataset that included house prices based on square footage. The first step was to load the dataset and create a Pandas DataFrame:
data = {
    'SquareFootage': [1500, 1800, 2400, 3000, 3500, 4000, 4500],
    'Price': [200000, 250000, 300000, 350000, 400000, 500000, 600000]
}

df = pd.DataFrame(data)
print(df.head())

# Here, we created a simple dataset with two columns: SquareFootage (our input feature) and Price (our target variable). 
# This dataset is a good fit for a linear regression problem since the relationship between square footage and price is likely linear.


# 2. Splitting the data
# Next, we split the data into training and testing sets. This is crucial for building a model that generalizes well to unseen data:
X = df[['SquareFootage']]  # Features
y = df['Price']            # Target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Training set: 80% of the data, used to train the model
# Test set: 20% of the data, used to evaluate the model's performance

# By splitting the data, we ensure that the model isn’t overfitting to the training set and can predict values for new, unseen data.


# 3. Training the linear regression model
# We then initialized and trained the linear regression model using the training data:
model = LinearRegression()
model.fit(X_train, y_train)

print(f"Intercept: {model.intercept_}")
print(f"Coefficient: {model.coef_[0]}")

# Intercept: This is the value of the target variable when the input feature is zero. In this case, it represents the base price of a house without considering its square footage.
# Coefficient: This is the slope of the linear regression line, representing the rate at which the house price increases for each additional square foot.

# For example, if the coefficient is 100, it means that for every additional square foot, the house price increases by $100.


# 4. Making predictions
# Once the model was trained, we used it to make predictions on the test set:
y_pred = model.predict(X_test)

print("Predicted Prices:", y_pred)
print("Actual Prices:", y_test.values)

# The predicted prices are the model’s estimates based on the square footage from the test set. 
# Comparing these predictions to the actual house prices gives us insight into how well the model performed.


# 5. Evaluating the model
# To evaluate the model’s performance, we used mean squared error (MSE) and R-squared (R²) metrics:
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse}")
print(f"R-squared: {r2}")

# MSE: This metric measures the average squared difference between the predicted and actual values. A lower MSE indicates a more accurate model.
# R²: This metric tells us how well the model explains the variance in the target variable. An R² value close to 1 means the model explains most of the variance, while a value closer to 0 indicates poor performance.

# Sample output:
# MSE: 625000000.0
# R²: 0.96

# In this case, an R-squared of 0.96 means the model explains 96% of the variance in house prices, which indicates a strong fit.


# 6. Visualizing the results
# Finally, we plotted the actual house prices and the regression line to visualize how well the model fits the data:
plt.scatter(X_test, y_test, color='blue', label='Actual Data')
plt.plot(X_test, y_pred, color='red', label='Regression Line')
plt.xlabel('Square Footage')
plt.ylabel('Price')
plt.title('House Prices vs. Square Footage')
plt.legend()
plt.show()

# The plot shows:
# Blue dots representing the actual house prices.
# A red line representing the regression line learned by the model.
# A good fit will show the red line passing closely through the blue points, indicating that the model’s predictions align with the actual data.


# 7. Interpretation of results
# In this lab, we successfully trained a linear regression model to predict house prices based on square footage. 
# The model’s R-squared value of 0.96 indicates that the linear relationship between square footage and price is well-captured by the model. The low mean squared error further supports that the model’s predictions are close to the actual values.
# The coefficients learned by the model help interpret the relationship between the features and the target variable. In this case, the coefficient tells us how much the house price increases with each additional square foot.

# Conclusion
# In this lab, you successfully trained a linear regression model to predict house prices based on square footage. The model’s R-squared value of 0.96 indicates that the linear relationship between square footage and price is well-captured. The low mean squared error further supports the accuracy of the model’s predictions.

# Key takeaways:
# Always split your data into training and test sets to prevent overfitting.
# Use evaluation metrics such as MSE and R² to measure model performance.
# Visualizing the results helps provide a clear understanding of how well the model fits the data.