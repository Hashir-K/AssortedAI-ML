# In this activity, you will implement the least absolute shrinkage and selection operator (LASSO), a powerful regularization technique used 
# for feature selection and model tuning. 

# LASSO not only reduces the magnitude of less important feature coefficients but can also shrink some of them to zero, effectively selecting only the most 
# relevant features for your model. 

# The goal of this activity is to help you apply LASSO to a dataset, experiment with regularization strength, and interpret the results.

# By the end of this activity, you'll be able to:
# Implement LASSO regression: Apply LASSO to perform feature selection and regularization in an ML model.
# Adjust regularization strength: Experiment with different values of the regularization parameter (alpha) to understand its impact on model complexity and performance.
# Interpret LASSO results: Analyze the coefficients of the features to identify which ones are most relevant and how LASSO helps in simplifying the model.



# 1. Setting up your environment
# Before you begin, ensure that you have the necessary Python libraries installed. You will use pandas for data handling and Scikit-learn for model building. 
# If you haven’t installed these libraries yet, you can do so with the following command:
pip install pandas scikit-learn


# 2. Importing the required libraries
# Let’s start by importing the necessary libraries to perform LASSO regression:
import pandas as pd
from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

# pandas for data manipulation.
# Lasso from Scikit-learn to apply LASSO regression.
# train_test_split to split the dataset into training and testing sets.
# r2_score to evaluate the model's performance.



# 3. Loading and preparing the data
# We will use a dataset with two features, StudyHours and PrevExamScore, to predict whether a student passes an exam. 
# You can apply the same process to your own dataset.

# Sample dataset: Study hours, previous exam scores, and pass/fail labels
data = {
    'StudyHours': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'PrevExamScore': [30, 40, 45, 50, 60, 65, 70, 75, 80, 85],
    'Pass': [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]  # 0 = Fail, 1 = Pass
}

df = pd.DataFrame(data)

# Features and target variable
X = df[['StudyHours', 'PrevExamScore']]  # Features
y = df['Pass']  # Target variable



# 4. Splitting the data
# To evaluate the model’s performance, we will split the dataset into training and testing sets. 
# This allows us to train the model on one part of the data and test it on the remaining part.

# Split the data into training and testing sets (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)




# 5. Applying LASSO
# LASSO applies L1 regularization, which adds a penalty term to the loss function. 
# This penalization causes less important feature coefficients to shrink to zero, effectively selecting only the most relevant features for the model.

# Steps for applying LASSO
# Initialize the LASSO model: Specify a value for the regularization parameter alpha.
# Train the model: Use the training data to fit the LASSO model.
# Evaluate the model: Use the test data to make predictions and calculate the performance of the model using R-squared.

# Here is the code to implement LASSO:

# Initialize the LASSO model with alpha (regularization parameter)
lasso_model = Lasso(alpha=0.1)

# Train the model on the training data
lasso_model.fit(X_train, y_train)

# Make predictions on the test data
y_pred = lasso_model.predict(X_test)

# Evaluate the model's performance using R-squared
r2 = r2_score(y_test, y_pred)
print(f'R-squared score: {r2}')



# 6. Analyzing the results
# Once you train the LASSO model, some feature coefficients are reduced to zero. 
# This means that the feature has been effectively removed from the model, helping to simplify the model while keeping only the most significant features.

# To view the coefficients of the features after applying LASSO, run the following code:

# Display the coefficients of the features
print(f'LASSO Coefficients: {lasso_model.coef_}')

#Example output:
#LASSO Coefficients: [0.0, 0.022]

# In this example:
# The coefficient for StudyHours is 0, meaning it was removed from the model.
# The coefficient for PrevExamScore is nonzero, meaning it was retained in the model.


# 7. Tuning the regularization parameter
# The regularization parameter alpha controls how much the coefficients shrunk. Higher values of alpha lead to more aggressive shrinkage 
# (more features are set to zero), while lower values allow more features to remain in the model. 
# Experiment with different values of alpha to see how it affects the model's performance:

# Try different alpha values and compare the results
for alpha in [0.01, 0.05, 0.1, 0.5, 1.0]:
    lasso_model = Lasso(alpha=alpha)
    lasso_model.fit(X_train, y_train)
    y_pred = lasso_model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    print(f'Alpha: {alpha}, R-squared score: {r2}, Coefficients: {lasso_model.coef_}')


# Lower alpha values keep more features in the model but may lead to overfitting.
# Higher alpha values simplify the model but may reduce its accuracy.



# Conclusion
# You successfully applied LASSO regression to perform feature selection and regularization. 
# By adjusting the alpha parameter, you learned how to balance model complexity and performance.

# Key takeaways
# LASSO uses L1 regularization to shrink feature coefficients, which helps in feature selection.
# Increasing alpha results in more aggressive feature elimination.
# By selecting only the most important features, LASSO helps improve model interpretability and reduces the risk of overfitting.

# Experiment with different alpha values and datasets to further explore the impact of regularization on model performance.  



