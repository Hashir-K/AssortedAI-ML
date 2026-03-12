# Introduction
# In this activity, you will implement a decision tree model using Python and the Scikit-learn library. Decision trees are versatile machine learning algorithms used for both classification and regression tasks. We’ll focus on using decision trees for classification, and you’ll learn how to build, train, and evaluate a decision tree model.

# By the end of this activity, you will be able to:
# Build and train a decision tree model using Scikit-learn for classification tasks.
# Assess the model's performance using accuracy and a confusion matrix. 
# Assess the model’s performance by visualizing the decision tree structure.
# Tune a decision tree to prevent overfitting and improve its generalization to new data.

# 1. Setting up your environment
# Before starting, ensure that you have the required libraries installed. If you haven’t done so, use the following command to install the necessary packages:
# python -m pip install numpy pandas scikit-learn matplotlib

# 2. Importing required libraries
# Start by importing the libraries we’ll need:
# Import necessary libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
from sklearn import tree

# NumPy and pandas for handling data
# Scikit-learn's DecisionTreeClassifier to build the decision tree model
# Matplotlib to visualize the results
# tree from Scikit-learn to visualize the decision tree structure

# 3. Loading and preparing the data
# We’ll use a simple dataset where the goal is to classify whether a student passes or fails based on the number of their study hours
# and their previous exam scores. You can use this dataset or apply the same procedure to another dataset.

# Sample dataset: Study hours, previous exam scores, and pass/fail labels
data = {
    'StudyHours': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'PrevExamScore': [30, 40, 45, 50, 60, 65, 70, 75, 80, 85],
    'Pass': [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]  # 0 = Fail, 1 = Pass
}

# Convert to DataFrame
df = pd.DataFrame(data)

# Display the first few rows of the data
print(df.head())

# The features in this dataset are StudyHours and PrevExamScore, while Pass is the target variable, where 0 indicates a student has failed and 1 indicates they have passed.

# 4. Splitting the data into training and testing sets
# We will split the dataset into training and testing sets to train the decision tree on one part of the data and evaluate it on the other part.
# Features (X) and Target (y)
X = df[['StudyHours', 'PrevExamScore']]  # Features
y = df['Pass']                           # Target variable (0 = Fail, 1 = Pass)

# Split data into 80% training and 20% testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Display the shape of the training and testing sets
print(f"Training data: {X_train.shape}, {y_train.shape}")
print(f"Testing data: {X_test.shape}, {y_test.shape}")

# This will split the data into 80% for training and 20% for testing, ensuring the model is evaluated on unseen data.

# 5. Training the decision tree model
# Now we’ll create and train the decision tree model using the training data:
# Initialize the Decision Tree Classifier
model = DecisionTreeClassifier(random_state=42)

# Train the model on the training data
model.fit(X_train, y_train)

# Display the model's parameters
print(f"Tree depth: {model.get_depth()}")
print(f"Number of leaves: {model.get_n_leaves()}")

# The decision tree will automatically learn the best splits of the data based on the features (StudyHours and PrevExamScore) to classify whether a student passes or fails.
# Tree depth: Indicates the maximum depth of the decision tree
# Number of leaves: Represents the number of terminal nodes (leaves) in the tree


# 6. Making predictions
# Once the model is trained, we can use it to predict whether students in the test set will pass or fail:
# Make predictions on the testing set
y_pred = model.predict(X_test)

# Display the predictions
print("Predicted Outcomes (Pass/Fail):", y_pred)
print("Actual Outcomes:", y_test.values)

# The model's predictions can then be compared with the actual outcomes from the test data to assess its performance.


# 7. Evaluating the Model
# To evaluate how well the decision tree model performed, we’ll use several metrics, including accuracy, a confusion matrix, and a classification report:
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

# Accuracy: The proportion of correct predictions out of all predictions
# Confusion matrix: A matrix that shows the counts of true positives (TP), true negatives (TN), false positives (FP), and false negatives (FN), helping us see where the model made mistakes
# Classification report: A report that provides metrics such as precision, recall, and F1 score for each class (pass or fail)


# 8. Visualizing the decision tree
# One of the key advantages of decision trees is their interpretability. We can visualize the structure of the decision tree using the plot_tree function from Scikit-learn:
# Visualize the decision tree
plt.figure(figsize=(12,8))
tree.plot_tree(model, feature_names=['StudyHours', 'PrevExamScore'], class_names=['Fail', 'Pass'], filled=True)
plt.title('Decision Tree for Classifying Pass/Fail')
plt.show()

# The plot will display the structure of the decision tree, showing how the model splits the data at each node based on the number of study hours and previous exam scores to classify pass or fail.


# 9. Tuning the decision tree
# Decision trees can be prone to overfitting, especially if they are allowed to grow too deep. You can control the complexity of the tree 
# by limiting its depth or the number of samples required to make a split:
# Limit the tree depth to avoid overfitting
model_tuned = DecisionTreeClassifier(max_depth=3, random_state=42)

# Train the model on the training data
model_tuned.fit(X_train, y_train)

# Make predictions with the tuned model
y_pred_tuned = model_tuned.predict(X_test)

# Evaluate the tuned model
accuracy_tuned = accuracy_score(y_test, y_pred_tuned)
print(f"Accuracy (Tuned Model): {accuracy_tuned}")

# By limiting the tree depth, we can reduce the likelihood of overfitting and improve the model's generalization to new data.

# Conclusion
# In this activity, you have successfully implemented a decision tree classifier to predict whether students pass or fail based on the number of study hours 
# and their previous exam scores. Key takeaways include:
# Decision trees are powerful algorithms that can model complex decision-making processes based on multiple features.
# Model evaluation metrics such as accuracy and a confusion matrix help assess how well the model performs on unseen data.
# Visualization of the decision tree structure provides an interpretable way to understand how the model makes decisions.

# By following these steps, you now have the knowledge to implement decision trees for both classification and regression tasks and explore tuning techniques 
# to improve their performance.









