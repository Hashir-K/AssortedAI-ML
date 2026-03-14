# Introduction
# In this activity, you will implement forward selection, a feature selection technique in which you start with no features 
# and progressively add the most significant ones. Forward selection helps to build a more efficient ML model by identifying the most relevant features. The goal of this activity is to guide you through the steps required to apply forward selection to a dataset and refine your model by including only the most impactful features.

# By the end of this activity, you'll be able to:
# Implement forward selection: Apply forward selection to add the most significant features to an ML model, enhancing its predictive power.
# Evaluate model performance: Use the R-squared metric to assess the impact of each feature on the model's performance during the selection process.
# Build an efficient model: Identify and include only the most relevant features, improving model efficiency and interpretability.


# 1. Setting up your environment
# Before starting, ensure that you have all the necessary libraries installed. In this activity, you’ll use Python’s 
# Scikit-learn and pandas for data manipulation and model training. If you haven’t installed these libraries, run the following command:

# pip install pandas scikit-learn

# 2. Importing the required libraries
# Let’s start by importing the necessary libraries for implementing forward selection:
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

# pandas is used for loading and handling the dataset.
# LinearRegression from Scikit-learn is used to create the regression model.
# R-squared is the metric you'll use to evaluate how well the model performs with the selected features.


# 3. Loading and preparing the data
# You’ll use a simple dataset where you aim to predict whether a student passes based on their study hours and previous exam scores. 
# You can use this dataset or apply the same steps to your own dataset.
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


# In this dataset:
# StudyHours and PrevExamScore are the features.
# Pass (0 = Fail, 1 = Pass) is the target variable.




# 4. Implementing forward selection
# In forward selection, you start with no features and iteratively add the feature that provides the most improvement to the model’s performance. 
# After each feature is added, you evaluate the model’s performance using the R-squared metric, which tells you how well the model explains the variance 
# in the target variable.

# Step-by-step process
# Start with an empty model (no features).
# For each feature, train a model, and evaluate its performance using R-squared.
# Add the feature that improves the R-squared value the most.
# Repeat the process, adding features one by one until no further improvement is made.

# Here’s how you can implement this process:
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


# 5. Explaining the process
# How forward selection works:
# Initialize with no features: The process begins with an empty set of features.
# Evaluate each feature: At each iteration, the model is trained with one additional feature at a time, and the R-squared value is calculated.
# Add the best feature: The feature that provides the highest improvement in R-squared is added to the model.
# Repeat: The process repeats by adding the next best feature until no further improvement can be made.

# Metrics used
# R-squared (coefficient of determination): This metric shows how much of the variance in the target variable is explained by the features. The higher the R-squared, the better the model explains the data.



# 6. Analyzing the results
# Once you’ve completed the forward selection process, you should see the most important features selected by the algorithm. 
# The selected features should be the ones that provide the highest predictive power for the model.

# Questions to consider:
# Which features were selected in the forward selection process?
# How did the model’s performance improve as more features were added?
# How can you interpret the significance of the selected features?

# Conclusion
# In this activity, you implemented forward selection, a step-by-step approach to building a more efficient model by selecting the most significant features. 
# Forward selection allows you to incrementally improve your model by adding features that contribute the most to the model’s predictive power.

# Forward selection is especially useful when:
# You have a large number of features, and you need to identify the most relevant ones.
# You want to build a more interpretable model.
# You are looking for a data-driven approach to feature selection.