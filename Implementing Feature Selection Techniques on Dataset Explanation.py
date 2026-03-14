# Introduction
# In this walkthrough, we will review the correct implementation and solution for the activity on feature selection techniques. 
# By applying backward elimination, forward selection, and the least absolute shrinkage and selection operator (LASSO), you learned how to identify and 
# retain the most significant features in a dataset. This guide will walk through each technique and explain the solution to the activity, 
# helping you understand how feature selection improves model performance and generalization.

# By the end of this walkthrough, you'll be able to:
# Implement cross-validation techniques: Use cross-validation to assess and enhance the reliability of supervised learning models.
# Apply key evaluation metrics: Accurately calculate and interpret metrics such as accuracy, precision, recall, F1-score, and R-squared.
# Analyze and interpret results: Evaluate the generalizability and performance of models using cross-validation combined with various evaluation metrics.


# 1. Loading and preparing the data
# We started by loading the dataset containing two features, StudyHours and PrevExamScore, with Pass (0 = Fail, 1 = Pass) as the target variable. 
# Here’s how we set up the feature matrix (X) and the target variable (y):

import pandas as pd

# Sample dataset: Study hours, previous exam scores, and pass/fail labels
data = {'StudyHours': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        'PrevExamScore': [30, 40, 45, 50, 60, 65, 70, 75, 80, 85],
        'Pass': [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]}  # 0 = Fail, 1 = Pass

df = pd.DataFrame(data)

# Define features and target variable
X = df[['StudyHours', 'PrevExamScore']]
y = df['Pass']



# 2. Implementing backward elimination
# Backward elimination starts by fitting the model with all features and progressively removes those with high p-values (i.e., features that are statistically insignificant).

# Solution

# Here’s how backward elimination was implemented:
import statsmodels.api as sm

# Add a constant (intercept) to the features
X = sm.add_constant(X)

# Fit the Ordinary Least Squares (OLS) regression model
model = sm.OLS(y, X).fit()

# Display the model summary (including p-values)
print(model.summary())

# Remove feature with highest p-value (if greater than 0.05)
if model.pvalues['StudyHours'] > 0.05:
    X = X.drop(columns='StudyHours')
    model = sm.OLS(y, X).fit()

# Final model after backward elimination
print(model.summary())

# Explanation:
# After fitting the model, the p-value of StudyHours was checked. If its p-value was greater than 0.05, we removed it from the feature set and refitted the model.
# In this case, PrevExamScore was the significant feature, and StudyHours was removed.




# 3. Implementing forward selection
# Forward selection adds features one by one based on their contribution to the model’s performance (measured by R-squared). The feature that improves 
# the model the most is added at each step.

# Solution

# Here’s how forward selection was implemented:
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

def forward_selection(X, y):
    remaining_features = set(X.columns)
    selected_features = []
    current_score = 0.0
    
    while remaining_features:
        scores_with_candidates = []
        
        for feature in remaining_features:
            features_to_test = selected_features + [feature]
            X_train, X_test, y_train, y_test = train_test_split(X[features_to_test], y, test_size=0.2, random_state=42)
            
            # Train the model
            model = LinearRegression()
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            score = r2_score(y_test, y_pred)
            
            scores_with_candidates.append((score, feature))
        
        # Select the feature with the highest R-squared score
        scores_with_candidates.sort(reverse=True)
        best_score, best_feature = scores_with_candidates[0]
        
        if current_score < best_score:
            remaining_features.remove(best_feature)
            selected_features.append(best_feature)
            current_score = best_score
        else:
            break
    
    return selected_features

# Run forward selection
best_features = forward_selection(X, y)
print(f"Selected features using Forward Selection: {best_features}")



# Explanation
# We started with an empty model and progressively added features, evaluating the R-squared score for each combination.
# In this case, PrevExamScore was selected first because it had the highest positive impact on model performance.
# StudyHours was not added because it did not significantly improve the model.




# 4. Implementing LASSO
# LASSO is a regularization technique that shrinks the coefficients of less important features to zero, effectively selecting the most significant features automatically.

# Solution

# Here’s how LASSO was implemented:
from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the LASSO model with a regularization parameter (alpha)
lasso_model = Lasso(alpha=0.1)

# Train the LASSO model
lasso_model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = lasso_model.predict(X_test)

# Evaluate the model's performance using R-squared
r2 = r2_score(y_test, y_pred)
print(f'R-squared score: {r2}')

# Display the LASSO coefficients
print(f'LASSO Coefficients: {lasso_model.coef_}')


# Explanation
# LASSO automatically shrinks the coefficient for StudyHours to zero, meaning it was removed from the model, while PrevExamScore was retained with a nonzero coefficient.
# The final model with LASSO was simplified, retaining only the most important feature.





# Conclusion
# You applied three different feature selection techniques:
# Backward elimination: Removed statistically insignificant features based on p-values
# Forward selection: Added features one by one to improve model performance
# LASSO: Automatically performed feature selection by shrinking less important features to zero

# Key takeaways
# Backward elimination is useful when you want to remove features manually based on statistical significance.
# Forward selection helps identify which features most improve model performance.
# LASSO is an efficient automatic feature selection method that balances model complexity and predictive power.




