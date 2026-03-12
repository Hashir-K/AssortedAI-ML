# Introduction
# In this reading, we’ll walk through the steps you followed during the logistic regression activity, providing explanations and insights into each part of the process. This will help you verify your work and understand the reasons behind each step, ensuring that you have successfully implemented and evaluated the logistic regression model. We’ll cover data preparation, model training, predictions, and evaluation.

# By the end of this walkthrough, you will be able to:
# Follow the steps for implementing a logistic regression model, including data preparation, training, and making predictions.
# Assess the model's effectiveness with metrics such as accuracy, a confusion matrix, and a classification report.
# Create visualizations of the logistic regression curve and interpret the relationship between study hours and the probability of passing.

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt

# Step-by-step guide
# Step 1: Load and prepare the data
# In this activity, we used a dataset in which we aimed to predict whether students pass or fail based on the number of their study hours. The dataset included two columns: StudyHours (the feature) and Pass (the target label).

# Here’s how we loaded the data and displayed the first few rows:
data = {
    'StudyHours': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'Pass': [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]  # 0 = Fail, 1 = Pass
}

df = pd.DataFrame(data)
print(df.head())

# This dataset is appropriate for logistic regression because it has a binary target variable (0 = Fail, 1 = Pass), making it ideal for classification.

# Step 2: Split the data
# To ensure we can evaluate the model’s performance on unseen data, we split the dataset into training and testing sets:
X = df[['StudyHours']]  # Feature(s)
y = df['Pass']          # Target variable (0 = Fail, 1 = Pass)

# Split data into 80% training and 20% testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Training data: {X_train.shape}, {y_train.shape}")
print(f"Testing data: {X_test.shape}, {y_test.shape}")

# Training set (80%): This subset is used to train the logistic regression model.
# Test set (20%): This subset is used to evaluate the model's ability to generalize to new, unseen data.


# Step 3: Train the logistic regression model
# After splitting the data, we initialized and trained the logistic regression model:
model = LogisticRegression()
model.fit(X_train, y_train)

print(f"Intercept: {model.intercept_}")
print(f"Coefficient: {model.coef_[0]}")

# Intercept: The bias term, which indicates the log odds of the target variable (Pass/Fail) when the feature (StudyHours) is zero.
# Coefficient: This value represents the rate of change in the log odds of passing as study hours increase. In logistic regression, the relationship between the feature and the outcome is modeled using the logistic function.

# For example, if the coefficient is 0.8, it means that each additional hour of study increases the log odds of passing by 0.8.

# Step 4: Make predictions
# Once the model was trained, we used it to predict whether students in the test set would pass or fail:
y_pred = model.predict(X_test)

print("Predicted Outcomes (Pass/Fail):", y_pred)
print("Actual Outcomes:", y_test.values)

# The model’s predictions were compared against the actual outcomes, showing how well it performed on unseen data.

# Step 5: Evaluate the model
# To evaluate the logistic regression model, we used several performance metrics:
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

print(f"Accuracy: {accuracy}")
print("Confusion Matrix:")
print(conf_matrix)
print("Classification Report:")
print(class_report)

# Here’s what each metric tells us:

# Accuracy: The proportion of correctly predicted outcomes out of all predictions. A high accuracy score indicates that the model predicted most outcomes correctly.
# Confusion matrix: A matrix that shows the counts of true positives (TP), true negatives (TN), false positives (FP), and false negatives (FN). This helps us understand where the model made mistakes (e.g., predicting a fail when the student actually passed).
# Classification report: A report that provides detailed metrics for each class:
# Precision: The proportion of positive predictions that were actually correct
# Recall: The proportion of actual positives that were correctly identified
# F1 score: The harmonic mean of precision and recall, providing a balanced measure of both


# Step 6: Visualize the results
# To better understand the relationship between study hours and the probability of passing, we plotted the logistic regression curve (sigmoid function):
study_hours_range = np.linspace(X.min(), X.max(), 100)
y_prob = model.predict_proba(study_hours_range.reshape(-1, 1))[:, 1]

plt.scatter(X_test, y_test, color='blue', label='Actual Data')
plt.plot(study_hours_range, y_prob, color='red', label='Logistic Regression Curve')
plt.xlabel('Study Hours')
plt.ylabel('Probability of Passing')
plt.title('Logistic Regression: Study Hours vs. Pass/Fail')
plt.legend()
plt.show()


# The plot shows:
# Blue dots representing the actual outcomes (pass or fail).
# A red curve representing the logistic regression’s predicted probability of passing, which increases as the number of study hours increases.
# The sigmoid curve is characteristic of logistic regression, showing the probability that a student passes as the number of study hours increases.

# Step 7: Interpret the results
# In this activity, we successfully implemented a logistic regression model to predict whether students pass or fail based on the number of their study hours. 
# The model performed well, achieving 100% accuracy on the test set. 

# Here’s what we learned:
# Logistic regression is well-suited for binary classification problems where the goal is to predict one of two outcomes (e.g., pass/fail).
# The model’s coefficients provide insights into how changes in the input feature (number of study hours) affect the likelihood of the outcome (passing).
# Performance metrics such as accuracy, a confusion matrix, and a classification report help evaluate the model’s effectiveness.

# Conclusion
# By following the steps outlined in the activity, you’ve learned how to implement, train, and evaluate a logistic regression model. The model performed well in predicting whether students would pass based on the number of their study hours, achieving perfect accuracy on the test data.

# Key takeaways:
# Logistic regression is a simple yet powerful tool for binary classification tasks.
# Model evaluation metrics such as accuracy and confusion matrices provide valuable insights into model performance.
# Visualization of the sigmoid function helps interpret the probability estimates produced by logistic regression.
