# In this walkthrough, we’ll review the correct solution to the activity where you implemented backward elimination as a feature selection method. 
# Backward elimination starts with all available features in a dataset and progressively removes those that are statistically insignificant. 

# This step-by-step guide will help you understand the process of removing irrelevant features and refining your ML model.

# By the end of this walkthrough, you'll be able to:
# Implement backward elimination correctly: follow the step-by-step process of applying backward elimination, 
# including adding a constant, fitting the model, and iteratively removing statistically insignificant features.
# Interpret model summary: analyze the p-values and coefficients from the model summary to determine which features are significant predictors.
# Refine and simplify models: understand how removing irrelevant features can lead to a more efficient and interpretable model, reducing the risk of overfitting.

# Step-by-step walkthrough
# Step 1: Load and prepare the data
# The first step was to load a sample dataset and prepare the feature matrix (X) and the target variable (y). 
# For this activity, we used a simple dataset of study hours, previous exam scores, and whether the student passed or failed.

import pandas as pd

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

# In this example:

# StudyHours and PrevExamScore were the input features.
# Pass (0 = Fail, 1 = Pass) was the target variable.


# Step 2: Add a constant to the model
# Before performing backward elimination, we added a constant (intercept) to the feature matrix. This is necessary for fitting the model with statsmodels.

import statsmodels.api as sm

# Add a constant (for the intercept)
X = sm.add_constant(X)

# The constant helps in modeling the intercept, which is crucial in linear regression models.

# Step 3: Fit the initial model
# The next step was to fit a model using ordinary least squares (OLS) regression, including all the features in the dataset. 
# This initial model serves as the starting point for backward elimination.

# Fit the model using OLS regression
model = sm.OLS(y, X).fit()

# Display the model summary (including p-values)
print(model.summary())

# The output of the model summary included p-values for each feature. The p-value indicates the statistical significance of each feature:

# A p-value less than 0.05 typically suggests the feature is statistically significant.
# A p-value greater than 0.05 indicates that the feature may not contribute much to the model.


# Step 4: Implement backward elimination
# Backward elimination progressively removes the feature with the highest p-value (greater than 0.05) and refits the model with the remaining features. The goal is to continue this process until all remaining features have a p-value below the significance threshold.

# Here’s how we implemented the backward elimination process:

# Set a significance level
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

# Explanation of the process
# We first fit the model with all features and evaluated their p-values.
# The feature with the highest p-value (indicating the least statistical significance) was identified and removed from the feature matrix.
# The model was refitted with the remaining features, and this process was repeated until all p-values were below the significance threshold (0.05).


# Step 5: Analyze the results
# After completing the backward elimination process, we examined the final model summary. 
# The remaining features in the model should all have p-values below 0.05, indicating that they are statistically significant predictors of the target variable.

# Sample output

# # Here’s an example of what the output might look like after backward elimination:

# Removing feature: StudyHours with p-value: 0.10
# Final Model Summary:
#                             OLS Regression Results                            
# ==============================================================================
# Dep. Variable:                   Pass   R-squared:                       0.970
# Model:                            OLS   Adj. R-squared:                  0.966
# Method:                 Least Squares   F-statistic:                     215.9
# Date:                Sun, 03 Sep 2023   Prob (F-statistic):           0.000102
# Time:                        11:45:26   Log-Likelihood:                -2.6013
# No. Observations:                  10   AIC:                             9.203
# Df Residuals:                       8   BIC:                             9.808
# Df Model:                           1                                         
# Covariance Type:            nonrobust                                         
# ==============================================================================
#                  coef    std err          t      P>|t|      [0.025      0.975]
# ------------------------------------------------------------------------------
# const         -0.9135      0.170     -5.358      0.001      -1.318      -0.509
# PrevExamScore  0.0225      0.002     14.694      0.000       0.019       0.026
# ==============================================================================
# Omnibus:                        2.040   Durbin-Watson:                   2.453
# Prob(Omnibus):                  0.361   Jarque-Bera (JB):                1.203
# Skew:                          -0.758   Prob(JB):                        0.548
# Kurtosis:                       1.967   Cond. No.                         535.
# ==============================================================================


# In this case, PrevExamScore was the remaining feature after backward elimination, as it had a statistically significant p-value (less than 0.05). 
# The feature StudyHours was removed because its p-value exceeded the significance threshold.


# Conclusion
# Backward elimination helps in simplifying ML models by removing irrelevant features, which can lead to improved performance and interpretability. 

# After completing the backward elimination process:
# You should have a model that includes only the most significant features.
# The model should be less prone to overfitting, as irrelevant or weak predictors have been removed.
# You can now evaluate the refined model’s performance using metrics such as R-squared or other relevant evaluation methods.