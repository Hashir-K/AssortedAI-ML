# Introduction
# In this walkthrough, we will review the correct solution to the least absolute shrinkage and selection operator (LASSO) activity. 
# LASSO is a regularization technique that helps with feature selection by shrinking the coefficients of less important features to zero. 
# This guide will take you through each step of the activity, from data preparation to model training and evaluation, while explaining how LASSO effectively 
# elects important features.

# By the end of this walkthrough, you'll be able to:
# Apply LASSO regression correctly: Understand how to implement LASSO for feature selection and regularization, including setting up the model and interpreting 
# the coefficients.
# Analyze the impact of regularization: Evaluate how different values of the regularization parameter alpha affect the model's complexity and performance.
# Interpret LASSO coefficients: Determine which features are retained or eliminated based on their coefficients, enhancing model interpretability and efficiency.



# 1. Loading and preparing the data
# We used a simple dataset that contains two features, StudyHours and PrevExamScore, to predict whether a student passes an exam (Pass: 0 = Fail, 1 = Pass). 
# The first step was to load the dataset and prepare the feature matrix (X) and the target variable (y).

import pandas as pd

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

# In this example:
# StudyHours and PrevExamScore are the features.
# Pass is the target variable.



# 2. Splitting the data
# Next, we split the data into training and testing sets to evaluate the model’s performance. 
# We used the training set to fit the LASSO model and the testing set to assess the model’s ability to generalize to new data.
from sklearn.model_selection import train_test_split

# Split the data into training and testing sets (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)



# 3. Applying LASSO regression
# In the core part of the activity, we applied the LASSO regression model. 
# LASSO applies L1 regularization, which adds a penalty to the loss function that shrinks less important feature coefficients to zero, 
# effectively selecting the most important features.

# Here’s how we implemented LASSO:
from sklearn.linear_model import Lasso
from sklearn.metrics import r2_score

# Initialize the LASSO model with alpha (regularization parameter)
lasso_model = Lasso(alpha=0.1)

# Train the LASSO model
lasso_model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = lasso_model.predict(X_test)

# Evaluate the model's performance using R-squared
r2 = r2_score(y_test, y_pred)
print(f'R-squared score: {r2}')

# Explanation

#alpha is the regularization strength. Higher values of alpha will shrink more coefficients to zero, simplifying the model. Lower values will keep more features in the model.
# The r2_score function evaluates how well the model explains the variance in the target variable. A higher R-squared value means the model is a better fit.



# 4. Interpreting the coefficients
# After fitting the LASSO model, it’s important to look at the feature coefficients to understand which features were selected and which shrunk to zero.

# Display the coefficients of the features
print(f'LASSO Coefficients: {lasso_model.coef_}')

# Example output
# LASSO Coefficients: [0.0, 0.022]

# In this case:
# StudyHours has a coefficient of 0, meaning it was removed from the model.
# PrevExamScore has a nonzero coefficient, meaning it was retained as an important feature.



# 5. Experimenting with different alpha values
# To better understand the effect of regularization, it’s useful to experiment with different values of alpha. Higher alpha values apply stronger regularization,
# shrinking more coefficients to zero, while lower values allow more features to remain in the model.

# Experiment with different alpha values
for alpha in [0.01, 0.05, 0.1, 0.5, 1.0]:
    lasso_model = Lasso(alpha=alpha)
    lasso_model.fit(X_train, y_train)
    y_pred = lasso_model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    print(f'Alpha: {alpha}, R-squared score: {r2}, Coefficients: {lasso_model.coef_}')



# Explanation
# Lower alpha values keep more features in the model but may lead to overfitting.
# Higher alpha values shrink more feature coefficients to zero, reducing the complexity of the model but potentially underfitting the data.



# Conclusion
# You successfully implemented LASSO regression to perform feature selection and regularization. 
# By adjusting the regularization parameter alpha, you saw how LASSO can reduce the number of features in the model, improving interpretability while maintaining
# predictive power.

# Key takeaways
# LASSO uses L1 regularization to shrink less important feature coefficients to zero, helping with feature selection.
# Adjusting the alpha parameter controls how aggressively LASSO shrinks coefficients. Higher alpha values result in simpler models with fewer features.
# Experimenting with different alpha values allows you to balance model complexity and performance.
