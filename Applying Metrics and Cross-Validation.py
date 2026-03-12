# In this activity, you will apply evaluation metrics and use cross-validation to assess the performance of supervised learning models. Cross-validation is a powerful technique that ensures your model’s performance is reliable and not dependent on any particular train-test split. 

# By the end of this activity, you will be able to:
# Implement cross-validation: use cross-validation to ensure reliable model performance and avoid reliance on a single train-test split.
# Apply evaluation metrics: calculate and interpret key metrics such as accuracy, precision, recall, F1 score, and R-squared for classification and regression models.
# Assess model performance: use cross-validation with multiple metrics to gain a comprehensive understanding of how well a model generalizes to unseen data.

# 1. Setting up your environment
# Before starting, ensure you have the required libraries installed. You will need NumPy, pandas, and Scikit-learn. 
# If you haven’t installed them yet, run the following command:

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, r2_score

#Scikit-Learn provides the models and metrics you’ll be using.


# 3. Loading and preparing the data
# You’ll use a sample dataset to predict whether students will pass or fail a presumptive future exam (not shown) based on their study hours and previous exam scores—a binary classification problem. You can use your own dataset, but the steps will remain the same:
# Sample dataset: Study hours, previous exam scores, and pass/fail labels
data = {
    'StudyHours': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'PrevExamScore': [30, 40, 45, 50, 60, 65, 70, 75, 80, 85],
    'Pass': [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]  # 0 = Fail, 1 = Pass
}

df = pd.DataFrame(data)

# Features and target variable
X = df[['StudyHours', 'PrevExamScore']]
y = df['Pass']

#Use StudyHours and PrevExamScore as features and Pass (0 = Fail, 1 = Pass) as the target variable.

# 4. Applying evaluation metrics without cross-validation
# First, train the model on a single train-test split and apply evaluation metrics. Use a logistic regression model to predict whether students will pass or fail:
from sklearn.linear_model import LogisticRegression

# Split the data into training and testing sets (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the Logistic Regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

#Next, calculate the model’s accuracy, precision, recall, and F1 score using the test set predictions:

# Calculate metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(f'Accuracy: {accuracy}')
print(f'Precision: {precision}')
print(f'Recall: {recall}')
print(f'F1-Score: {f1}')

# These metrics give you a snapshot of how well the model performed on the test set:
# Accuracy measures the proportion of correct predictions.
# Precision indicates how many predicted positives were correct.
# Recall measures how many actual positives were correctly predicted.
# F1 score is a balance between precision and recall.

# 5. Introducing cross-validation
# While the above method works, it’s limited by the single train-test split, which could lead to overfitting or underfitting. To get a more reliable performance estimate, use cross-validation. (cross-validation allows you to split the dataset into multiple subsets and reliably calculate model performance).

# Cross-validation involves splitting the data into multiple folds, training the model on some folds, and testing it on the remaining folds. The process is repeated for each fold, and the average performance is taken across all folds.

# 6. Performing k-fold cross-validation
# You will use k-fold cross-validation, where the dataset is split into k equal parts (folds). 
# Each fold is used as a test set while the remaining folds are used for training:
from sklearn.model_selection import cross_val_score

# Initialize the model
model = LogisticRegression()

# Perform 5-fold cross-validation and calculate accuracy for each fold
cv_scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')

# Display the accuracy for each fold and the mean accuracy
print(f'Cross-validation accuracies: {cv_scores}')
print(f'Mean cross-validation accuracy: {np.mean(cv_scores)}')

# Here, the cross_val_score function automatically splits the data into five folds, trains the model on four folds, and tests it on the remaining fold. 
# This process is repeated five times, and it reports the accuracy for each fold.

# 7. Cross-validation with multiple metrics
# Calculate multiple metrics during cross-validation using the scoring parameter. Use k-fold cross-validation to calculate accuracy, precision, recall, and F1 score:
from sklearn.model_selection import cross_validate

# Define multiple scoring metrics
scoring = ['accuracy', 'precision', 'recall', 'f1']

# Perform cross-validation
cv_results = cross_validate(model, X, y, cv=5, scoring=scoring)

# Print results for each metric
print(f"Cross-validation Accuracy: {np.mean(cv_results['test_accuracy'])}")
print(f"Cross-validation Precision: {np.mean(cv_results['test_precision'])}")
print(f"Cross-validation Recall: {np.mean(cv_results['test_recall'])}")
print(f"Cross-validation F1-Score: {np.mean(cv_results['test_f1'])}")

# This approach presents the average metrics for each fold, providing a better picture of how the model performs across different data splits.

# 8. Cross-validation with a regression model
# For regression tasks, use metrics such as mean absolute error (MAE), mean squared error (MSE), and R-squared. 
# Apply these metrics with cross-validation for a regression model:
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Sample dataset for regression
X_reg = df[['StudyHours']]
y_reg = df['PrevExamScore']

# Initialize a linear regression model
reg_model = LinearRegression()

# Perform 5-fold cross-validation using R-squared as the metric
cv_scores_r2 = cross_val_score(reg_model, X_reg, y_reg, cv=5, scoring='r2')

print(f'Cross-validation R-squared scores: {cv_scores_r2}')
print(f'Mean R-squared score: {np.mean(cv_scores_r2)}')

# For regression models:
# R-squared indicates how well the model explains the variance in the target variable.
# MSE and MAE measure the average error between the predicted and actual values.

# Conclusion
# In this activity, you learned how to apply evaluation metrics and use cross-validation to reliably assess the performance of machine learning models. 

# Key takeaways:
# Cross-validation provides a more robust performance estimate by training and testing on multiple data splits.
# You can calculate multiple evaluation metrics during cross-validation to get a comprehensive view of model performance.
# Both classification and regression models benefit from cross-validation, which helps avoid overfitting and ensures reliable performance.








# #Explanations:
# 1. Loading and preparing the data
# You used a dataset with the features StudyHours and PrevExamScore to predict whether students would pass or fail a presumptive future exam (not shown). The first step was to load the data into a pandas DataFrame:

# StudyHours and PrevExamScore were the features. Pass (0 = Fail, 1 = Pass) was the target variable. This data was well suited for a binary classification problem using models like logistic regression.

# 2. Splitting the data and training a logistic regression model
# You began by splitting the dataset into training and testing sets and applying a logistic regression model to the training set:

# You used train_test_split to randomly assign 80 percent of the data for training and 20 percent for testing. After training the logistic regression model, you made predictions on the testing set.

# 3. Applying evaluation metrics
# Next, you calculated the model’s accuracy, precision, recall, and F1 score to evaluate its performance:

# Here’s what these metrics show:
# Accuracy: the proportion of correct predictions out of all predictions made. In this case, the model's accuracy is the percentage of students it correctly classified as pass or fail.
# Precision: the proportion of positive predictions (Pass) that were correct.
# Recall: the proportion of actual positive cases (Pass) that were correctly identified.
# F1 score: the harmonic mean of precision and recall, providing a balanced measure of the two.

# 4. Introducing cross-validation
# Cross-validation was then introduced to provide a more robust evaluation of the model's performance. You used five-fold cross-validation, which splits the dataset into five parts (folds), trains the model on four folds, and tests it on the remaining fold. This process is repeated for each fold, and the average performance is calculated:

# The cross_val_score function calculates the accuracy for each fold. The mean of these scores gives a more reliable estimate of how well the model generalizes to unseen data.

# 5. Cross-validation with multiple metrics
# In addition to accuracy, you calculated precision, recall, and F1 score using cross-validation to assess the model’s performance across various metrics:

# This approach provided a more comprehensive evaluation of the model. Accuracy, precision, recall, and F1 score were calculated for each fold, and the average was reported for each metric.

# 6. Cross-validation with a regression model
# For regression tasks, you used metrics such as R-squared and mean absolute error (MAE) to evaluate the model's performance. Here’s how you applied cross-validation to a regression model:

# In this case, the R-squared metric measured how much of the variance in the target variable was explained by the features. A higher R-squared value indicates a better fit.  You could also calculate mean squared error (MSE) or MAE using the same process.

# Conclusion
# You successfully applied evaluation metrics and cross-validation to a classification problem using logistic regression. You also explored how cross-validation could be used with regression models to obtain more robust performance estimates.

# Key takeaways:
# Cross-validation provides a more reliable evaluation by testing the model on multiple data splits.
# Accuracy, precision, recall, and F1 score are essential metrics for evaluating classification models.
# For regression models, R-squared, MSE, and MAE are commonly used metrics.

# Using cross-validation with multiple metrics helps ensure that the model’s performance is well-rounded and not dependent on a single evaluation metric.