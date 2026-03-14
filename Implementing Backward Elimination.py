# Introduction
# In this activity, you will implement backward elimination, a feature selection technique that helps you identify the most important features for your ML model. Backward elimination starts with all features and progressively removes the least significant ones, leading to a more efficient model. The goal of this activity is to help you apply this technique to a dataset and refine your model by eliminating irrelevant features.

# By the end of this activity, you'll be able to:
# Implement backward elimination: identify and remove the least significant features from a dataset.
# Apply statistical modeling: fit a linear regression model using the statsmodels library and interpret the p-values to determine feature significance.
# Refine and simplify models: analyze the impact of removing irrelevant features on model performance and interpret the results to improve model efficiency.

# Step-by-step instructions
# Step 1: Import the required libraries
# Before starting, make sure you have the necessary libraries installed. You will be using Python along with the following libraries:
# pandas will be used to handle the dataset.
# statsmodels will help you perform statistical modeling, which is required for backward elimination.

import pandas as pd
import statsmodels.api as sm
from sklearn.model_selection import train_test_split


# Step 2: Load and prepare the data
# You’ll use a sample dataset to predict whether a student passes a speculative future assignment (not shown) based on their study hours and previous exam scores. 
# Alternatively, you can apply the same steps to your own dataset:

# Sample dataset
data = {
    'StudyHours': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'PrevExamScore': [30, 40, 45, 50, 60, 65, 70, 75, 80, 85],
    'Pass': [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]  # 0 = Fail, 1 = Pass
}

df = pd.DataFrame(data)

# Features and target variable
X = df[['StudyHours', 'PrevExamScore']]
y = df['Pass']

# In this example, StudyHours and PrevExamScore are the features, and Pass is the target variable (0 = Fail, 1 = Pass).


# Step 3: Add a constant to the model
# In statsmodels, you need to add a constant to your feature matrix for the intercept term. 
# This constant will be necessary for the linear regression model used in backward elimination.

# Add a constant to the model (for the intercept)
X = sm.add_constant(X)


# Step 4: Fit the initial model
# Now, you will fit the initial model using all the available features: 

# Fit the model using Ordinary Least Squares (OLS) regression
model = sm.OLS(y, X).fit()

# Display the summary, including p-values for each feature
print(model.summary())

# The goal is to start with all features and then progressively remove the least significant ones. The output will show a summary of the model, including the p-values for each feature. The p-value helps you determine the statistical significance of each feature: features with high p-values are considered less significant and should be removed.


# Step 5: Implement backward elimination
# The main idea behind backward elimination is to iteratively remove the feature with the highest p-value—greater than 0.05 in this case—and refit the model until all remaining features have a p-value less than 0.05.

# Step-by-step process:
# Fit the model with all features.
# Identify the feature with the highest p-value.
# Remove the feature with the highest p-value.
# Refit the model and repeat until all remaining features are statistically significant.

# Here’s a simple implementation of this process:
# Define a significance level
significance_level = 0.05

# Perform backward elimination
while True:
    # Fit the model
    model = sm.OLS(y, X).fit()
    # Get the highest p-value in the model
    max_p_value = model.pvalues.max()
    
    # Check if the highest p-value is greater than the significance level
    if max_p_value > significance_level:
        # Identify the feature with the highest p-value
        feature_to_remove = model.pvalues.idxmax()
        print(f"Removing feature: {feature_to_remove} with p-value: {max_p_value}")
        
        # Drop the feature
        X = X.drop(columns=[feature_to_remove])
    else:
        break

# Display the final model summary
print(model.summary())

# Step 6: Analyze the results
# Once you’ve completed the backward elimination process, review the final model summary. The remaining features should all have p-values less than the significance level, meaning they are statistically significant predictors of the target variable.

# Questions to consider:
# Which features were removed during the backward elimination process?
# How did the model’s performance improve as irrelevant features were removed?
# Can you interpret the coefficients of the remaining features?


# Conclusion
# In this activity, you applied backward elimination to progressively remove the least significant features from a dataset. 
# This technique helps simplify your model by keeping only the most relevant features, which can improve performance and reduce overfitting.

# Backward elimination is particularly useful when:
# You have many features, and not all are relevant.
# You want to improve model interpretability.
# You want to focus on the features that have the most impact on the target variable.