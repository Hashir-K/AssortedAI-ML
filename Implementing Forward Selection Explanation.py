# Introduction
# In this walkthrough, we will review the correct solution to the activity in which you implemented forward selection. 
# Forward selection is a step-by-step feature selection method where we start with no features and progressively add the most significant ones based on their contribution to the model’s performance. 

# This guide will walk you through each step of the process and explain how you can achieve the correct solution.

# By the end of this walkthrough, you'll be able to:
# Implement forward selection: Follow the step-by-step process of applying forward selection to add the most significant features to a model.
# Evaluate feature impact: Use the R-squared metric to assess the contribution of each feature to the model's performance.
# Build and interpret efficient models: Analyze the selected features, and understand how they contribute to building a more interpretable and efficient ML model.

# 1. Loading and preparing the data
# We started by loading the dataset that contains StudyHours and PrevExamScore as the input features, with Pass (0 = Fail, 1 = Pass) 
# as the target variable. You can either use this dataset or apply the same process to your own dataset.

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


# 2. Implementing forward selection
# The key task of forward selection is to start with no features and progressively add the feature that improves the model’s performance the most. For each iteration, we fit a regression model and calculate the R-squared value, which tells us how much of the variance in the target variable is explained by the features.

# Here’s the step-by-step process
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

def forward_selection(X, y):
    remaining_features = set(X.columns)
    selected_features = []
    current_score = 0.0
    best_score = 0.0
    
    while remaining_features:
        scores_with_candidates = []
        
        # Loop through remaining features
        for feature in remaining_features:
            features_to_test = selected_features + [feature]
            X_train, X_test, y_train, y_test = train_test_split(X[features_to_test], y, test_size=0.2, random_state=42)
            
            # Train the model
            model = LinearRegression()
            model.fit(X_train, y_train)
            
            # Make predictions and calculate R-squared
            y_pred = model.predict(X_test)
            score = r2_score(y_test, y_pred)
            
            # Record the score with the current feature
            scores_with_candidates.append((score, feature))
        
        # Sort candidates by score (highest score first)
        scores_with_candidates.sort(reverse=True)
        best_score, best_feature = scores_with_candidates[0]
        
        # If adding the feature improves the score, add it to the model
        if current_score < best_score:
            remaining_features.remove(best_feature)
            selected_features.append(best_feature)
            current_score = best_score
        else:
            break
    
    return selected_features

# Run forward selection
best_features = forward_selection(X, y)
print("Selected features using Forward Selection:", best_features)

# Explanation

# Initial setup: We start with an empty model (no features) and iteratively add the feature that improves the model’s R-squared the most.
# Evaluation: For each iteration, the model is trained using the selected features, and the R-squared value is calculated.
# Feature selection: The feature that results in the highest R-squared improvement is added to the model.
# Termination: The process continues until no further improvement in R-squared can be achieved by adding more features.



# 3. Results analysis
# Once the forward selection process is complete, the most relevant features are selected and printed. 
# Below is an example output you may encounter:
Selected features using Forward Selection: ['PrevExamScore']

# In this case, PrevExamScore was the feature that provided the most improvement in the model's performance. 
# The forward selection algorithm determined that adding StudyHours did not significantly improve the model’s R-squared, so only PrevExamScore was selected.

# Key observations:
# PrevExamScore had a strong correlation with the target variable, so it was selected first.
# StudyHours might not have contributed significantly to the prediction of the outcome (whether a student passes or fails), so it wasn’t included in the final model.



# 4. Evaluating the model performance
# To further analyze the model, we can evaluate how well the selected features explain the variance in the target variable using the R-squared metric. 
# You can do this by calculating the R-squared value for the model trained on the selected features.

X_train, X_test, y_train, y_test = train_test_split(X[best_features], y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
final_r2_score = r2_score(y_test, y_pred)

print(f'Final R-squared score with selected features: {final_r2_score}')

# This will give you a final R-squared value that reflects the performance of the model using only the selected features.



# Conclusion
# You successfully implemented forward selection and identified the most relevant features for predicting the target variable. 
# Forward selection is a useful technique when you want to build a more interpretable model by selecting only the most impactful features.

# Key takeaways:
# Forward selection starts with no features and adds the most significant ones based on their contribution to model performance.
# The R-squared metric helps evaluate the model’s performance and determines which features to include.
# Forward selection can help improve model interpretability by keeping only the most relevant features.