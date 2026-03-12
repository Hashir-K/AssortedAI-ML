# Introduction
# In this activity, you will learn how to implement a logistic regression model using Python and the popular machine learning library Scikit-learn. Logistic regression is commonly used for classification tasks, where the goal is to categorize data into distinct classes (e.g., spam vs. not spam, pass vs. fail). We’ll use logistic regression to predict binary outcomes and evaluate the model’s performance.

# By the end of this activity, you will be able to:

# Set up and train a logistic regression model using Scikit-learn.

# Interpret model outputs and performance metrics such as accuracy and a confusion matrix, also known as an error matrix.

# Visualize the logistic regression curve and predicted probabilities with Matplotlib.

# 1. Setting up your environment
# Ensure that you have the necessary libraries installed. If you haven’t installed them yet, use the following command to install the required packages:
#python -m pip install numpy pandas scikit-learn matplotlib

#2. Importing required libraries
#Start by importing the libraries we’ll need for this activity:

# Import necessary libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt

# NumPy and pandas will help us handle numerical and tabular data.
# Scikit-learn's LogisticRegression will be used to build the model.
# Matplotlib will allow us to visualize the results.

# 3. Loading and preparing the data
# We’ll use a sample dataset to classify whether students pass or fail based on the number of their study hours. You can use this dataset or substitute it with your own.

# Sample dataset: Study hours and whether students passed or failed
data = {
    'StudyHours': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'Pass': [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]  # 0 = Fail, 1 = Pass
}

# Convert to DataFrame
df = pd.DataFrame(data)

# Display the first few rows of the data
print(df.head())

# Here, StudyHours is our feature, and Pass is the target label, where 0 indicates failure and 1 indicates passing.

# 4. Splitting the data into training and testing sets
# We will split the dataset into training and testing sets, allowing us to train the model on one portion of the data and evaluate it on another:

# Features (X) and Target (y)
X = df[['StudyHours']]  # Feature(s)
y = df['Pass']          # Target variable (0 = Fail, 1 = Pass)

# Split data into 80% training and 20% testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Display the shape of the training and testing sets
print(f"Training data: {X_train.shape}, {y_train.shape}")
print(f"Testing data: {X_test.shape}, {y_test.shape}")

#This will split the dataset into 80% for training and 20% for testing, ensuring the model is evaluated on unseen data.


# 5. Training the logistic regression model
# Now we’ll initialize and train the logistic regression model using the training data:

# Initialize the Logistic Regression model
model = LogisticRegression()

# Train the model on the training data
model.fit(X_train, y_train)

# Display the model's learned coefficients
print(f"Intercept: {model.intercept_}")
print(f"Coefficient: {model.coef_[0]}")

# Intercept: This is the bias term in the logistic regression equation.
# Coefficient: This value indicates how much the log odds of passing change with each additional hour of study.


# 6. Making predictions
# Once the model is trained, we can use it to predict whether students pass or fail based on the number of their study hours:

# Make predictions on the testing set
y_pred = model.predict(X_test)

# Display the predictions
print("Predicted Outcomes (Pass/Fail):", y_pred)
print("Actual Outcomes:", y_test.values)


# 7. Evaluating the model
# To evaluate how well the logistic regression model performed, we’ll use several metrics, including accuracy, a confusion matrix, and a classification report (which includes precision, recall, and F1 score):


# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)

# Generate confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)

# Generate classification report
class_report = classification_report(y_test, y_pred)

# Display evaluation metrics
print(f"Accuracy: {accuracy}")
print("Confusion Matrix:")
print(conf_matrix)
print("Classification Report:")
print(class_report)

# Accuracy: The percentage of correctly predicted outcomes out of all predictions
# Confusion matrix: A table that shows the number of correct and incorrect predictions categorized by true positives, true negatives, false positives, and false negatives
# Classification report: A report that provides detailed metrics such as precision, recall, and F1 score for each class


# 8. Visualizing the results
# Logistic regression produces probabilities for each outcome. We can visualize the sigmoid function, 
# which is the key characteristic of logistic regression, and plot the model’s predictions against the actual data points:
# Create a range of study hours for plotting
study_hours_range = np.linspace(X.min(), X.max(), 100)

# Calculate predicted probabilities using the sigmoid function
y_prob = model.predict_proba(study_hours_range.reshape(-1, 1))[:, 1]

# Plot the actual data points
plt.scatter(X_test, y_test, color='blue', label='Actual Data')

# Plot the logistic regression curve
plt.plot(study_hours_range, y_prob, color='red', label='Logistic Regression Curve')

# Add labels and title
plt.xlabel('Study Hours')
plt.ylabel('Probability of Passing')
plt.title('Logistic Regression: Study Hours vs. Pass/Fail')
plt.legend()

# Show the plot
plt.show()

# This visualization helps you understand the relationship between the number of study hours and the likelihood of passing. The sigmoid curve shows the probability of passing as the number of study hours increases.

# Conclusion
# In this activity, you successfully implemented a logistic regression model to predict whether students would pass or fail based on the number of their study hours. 

# Key takeaways include:
# Logistic regression is a powerful tool for binary classification problems, where the goal is to predict a categorical outcome (e.g., pass/fail, yes/no).
# Model evaluation metrics such as accuracy, confusion matrices, and classification reports provide insights into the performance of the model.
# Visualization of the sigmoid function gives a clearer picture of how logistic regression estimates probabilities.

# By following these steps, you will have the knowledge to apply logistic regression to other classification problems in your own machine learning projects.
