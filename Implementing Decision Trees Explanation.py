# Introduction
# In this reading, we’ll walk through the steps you followed during the decision tree activity, providing explanations and insights into each part of the process. 
# This will help you verify your work and understand the reasons behind each step, ensuring that you have successfully implemented and evaluated the decision tree model. 
# We’ll cover data preparation, model training, predictions, evaluation, and visualization of the decision tree structure.

# By the end of this walkthrough, you will be able to:
# Understand the process of building and training a decision tree using Scikit-Learn, including data preparation and feature selection.
# Assess the decision tree's effectiveness on the test data with metrics such as accuracy, a confusion matrix, and a classification report.
# Visualize the structure of the decision tree to interpret how it makes decisions, and apply tuning techniques to prevent overfitting.

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
from sklearn import tree

# Step-by-step guide:
# Step 1: Load and prepare the data
# In this activity, we used a dataset where the goal was to predict whether students pass or fail based on their StudyHours and PrevExamScore. 
# Here’s how we loaded the data into a Pandas DataFrame:
data = {
    'StudyHours': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'PrevExamScore': [30, 40, 45, 50, 60, 65, 70, 75, 80, 85],
    'Pass': [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]  # 0 = Fail, 1 = Pass
}

df = pd.DataFrame(data)
print(df.head())

# This dataset has two input features:

# StudyHours: the number of hours a student studied
# PrevExamScore: the student's score from a previous exam
# The target variable is Pass, which indicates whether the student passed (1) or failed (0).


# Step 2: Split the data
# To ensure that the model can generalize to unseen data, we split the dataset into training and testing sets:

X = df[['StudyHours', 'PrevExamScore']]  # Features
y = df['Pass']  # Target

# Split data into 80% training and 20% testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Training data: {X_train.shape}, {y_train.shape}")
print(f"Testing data: {X_test.shape}, {y_test.shape}")

# Training set (80%): Used to train the decision tree model
# Testing set (20%): Used to evaluate the model’s performance on new data

# Step 3: Train the decision tree model
# We then created and trained the decision tree model:
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)

print(f"Tree depth: {model.get_depth()}")
print(f"Number of leaves: {model.get_n_leaves()}")

# Tree depth refers to the number of levels in the decision tree.
# Number of leaves represents the number of terminal nodes in the tree where predictions are made.
# The decision tree learns how to split the data based on the features (StudyHours and PrevExamScore) to predict whether a student will pass or fail.

# Step 4: Make predictions
# Once the model was trained, we used it to make predictions on the test data:
y_pred = model.predict(X_test)

print("Predicted Outcomes (Pass/Fail):", y_pred)
print("Actual Outcomes:", y_test.values)

# The predicted outcomes (0 = fail, 1 = pass) were compared with the actual outcomes from the test set, allowing us to assess the model’s accuracy.

# Step 5: Evaluate the model
# To evaluate the performance of the decision tree model, we used accuracy, a confusion matrix, and a classification report:
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

print(f"Accuracy: {accuracy}")
print("Confusion Matrix:")
print(conf_matrix)
print("Classification Report:")
print(class_report)

# Here’s what each metric tells us:
# Accuracy: The proportion of correct predictions out of the total number of predictions. Higher accuracy indicates better model performance.
# Confusion matrix: A table that shows the number of true positives (TP), true negatives (TN), false positives (FP), and false negatives (FN). It helps us understand where the model made errors.
# Classification report: Provides detailed metrics for each class (pass/fail), including:
# Precision: The proportion of positive predictions that were actually correct
# Recall: The proportion of actual positives that were correctly predicted
# F1 score: The harmonic mean of precision and recall, giving a balanced measure of both


# Step 6: Visualize the decision tree
# One of the main benefits of decision trees is their interpretability. We visualized the tree structure to understand how the model makes decisions:
plt.figure(figsize=(12,8))
tree.plot_tree(model, feature_names=['StudyHours', 'PrevExamScore'], class_names=['Fail', 'Pass'], filled=True)
plt.title('Decision Tree for Classifying Pass/Fail')
plt.show()

# This plot displays the entire structure of the decision tree, showing how the model splits the data based on StudyHours and PrevExamScore at each level. Each leaf node represents a final decision (pass or fail), and the branches leading to these nodes show the conditions the model used to make its decisions.

# Step 7: Tune the decision tree
# Decision trees can sometimes overfit the training data if they are allowed to grow too deep. To prevent overfitting, we can limit the depth of the tree:
model_tuned = DecisionTreeClassifier(max_depth=3, random_state=42)
model_tuned.fit(X_train, y_train)

# Making predictions with the tuned model
y_pred_tuned = model_tuned.predict(X_test)

# Evaluating the tuned model
accuracy_tuned = accuracy_score(y_test, y_pred_tuned)
print(f"Accuracy (Tuned Model): {accuracy_tuned}")

# By limiting the tree depth to 3, we simplify the model and reduce the risk of overfitting. In this case, if the original model was overfitting, 
# the tuned model may show improved performance on the test set.

# Conclusion
# In this activity, you successfully implemented a decision tree model to predict whether students would pass or fail based on the number of study hours 
# and their previous exam scores. 
# Here’s what you’ve learned:
# Decision trees are intuitive and easy to interpret; they are powerful algorithms for classification tasks.
# Model evaluation metrics such as accuracy, the confusion matrix, and the classification report provide a thorough understanding of how well the model performed.
# Visualization of the decision tree offers a clear picture of the model’s decision-making process.

# Now that you’ve successfully completed the activity, you can apply decision trees to other classification problems and explore different datasets and scenarios.

# These skills are valuable for classification problems, and you can apply them to other datasets and scenarios to further explore decision trees
# and other machine learning models.



