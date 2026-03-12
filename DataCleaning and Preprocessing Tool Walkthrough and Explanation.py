import pandas as pd
import numpy as np


#Step 1: Review Dummy Data

# Create a dummy dataset
np.random.seed(0)
dummy_data = {
    'Feature1': np.random.normal(100, 10, 100).tolist() + [np.nan, 200],  # Normally distributed with an outlier
    'Feature2': np.random.randint(0, 100, 102).tolist(),  # Random integers
    'Category': ['A', 'B', 'C', 'D'] * 25 + [np.nan, 'A'],  # Categorical with some missing values
    'Target': np.random.choice([0, 1], 102).tolist()  # Binary target variable
}

# Convert the dictionary to a pandas DataFrame
df_dummy = pd.DataFrame(dummy_data)

# Display the first few rows of the dummy dataset
print(df_dummy.head())

#Explanation
# This dataset includes numeric and categorical features, missing values, and an outlier, 
# which simulates common challenges encountered in real-world datasets.




#Step 2: Apply Preprocessing tool

#Step 2.1: Handle missing values 
# The first step in preprocessing is addressing any missing values in the dataset. 
# This can be done by either removing missing data or filling in the missing values with an appropriate statistic, such as the mean.

# Fill missing values with the mean for numeric columns
df_filled = df_dummy.fillna(df_dummy.mean())

# Fill missing categorical data with the mode (most frequent value)
df_filled['Category'].fillna(df_filled['Category'].mode()[0], inplace=True)

print(df_filled.isnull().sum())  # Verify that there are no missing values

# Explanation
# In this solution, numeric missing values are filled with the mean of the respective column, 
# while missing values in the categorical column are filled with the most frequent category (mode). 
# This ensures that no data is lost due to missing values.


#Step 2.2: Remove outliers 
# Outliers can distort the analysis and negatively impact model performance. 
# You remove them using the Z-score method, which measures how far each data point is from the mean.

from scipy import stats

# Calculate Z-scores for numerical features
z_scores = np.abs(stats.zscore(df_filled.select_dtypes(include=[np.number])))

# Remove rows with any Z-scores greater than 3 (commonly used threshold for outliers)
df_no_outliers = df_filled[(z_scores < 3).all(axis=1)]

print(df_no_outliers.describe())  # Verify that outliers have been removed

#Explanation
# Z-scores are calculated for all numeric columns, and any rows with Z-scores greater than three are considered outliers and removed.
#  This step ensures that the dataset is free from extreme values that could skew the model.

#Step 2.3: Scale the data 
# Scaling ensures that all numeric features contribute equally to the analysis, which is important for many ML algorithms,
#  especially those that rely on distance metrics.

from sklearn.preprocessing import StandardScaler

# Scale numeric features using StandardScaler (Z-score normalization)
scaler = StandardScaler()
df_no_outliers[df_no_outliers.select_dtypes(include=[np.number]).columns] = scaler.fit_transform(df_no_outliers.select_dtypes(include=[np.number]))

print(df_no_outliers.head())  # Verify that the data has been scaled

#Explanation
# The StandardScaler scales numeric features so that they have a mean of zero and a standard deviation of one. 
# This transformation helps improve the performance of many ML algorithms.

#Step 2.4: Encode categorical variables 
# ML models require numeric input, so categorical variables must be converted into a numerical format using one-hot encoding.

# One-hot encode the categorical feature
df_encoded = pd.get_dummies(df_no_outliers, columns=['Category'])

print(df_encoded.head())  # Verify that the categorical variable has been encoded

#Explanation
# The pd.get_dummies() function converts the categorical column into multiple binary columns, each representing a category. This allows the categorical data to be used in machine learning models.


#Step 2.5: Save the preprocessed data
# Finally, the cleaned and preprocessed dataset is saved to a new comma-separated values (CSV) file, 
# making it ready for use in further analysis or model training.

# Save the preprocessed DataFrame to a CSV file
df_encoded.to_csv('preprocessed_dummy_data.csv', index=False)

print('Preprocessed data saved as preprocessed_dummy_data.csv')

# Explanation
# The preprocessed data is saved to a file named preprocessed_dummy_data.csv. This file can now be used as input for ML algorithms,
#  ensuring that the data is clean, consistent, and properly formatted.




#Step 3: Verify the data and perform a quality check 
# After completing the preprocessing steps, it's important to verify that the data has been processed correctly. You should check for the following:

#No missing values
#Ensure that all missing values have been handled.
print(df_encoded.isnull().sum())

#No outliers
# Confirm that outliers have been removed.
print(df_encoded.describe())

#Proper scaling
# Check that the numeric features have been scaled appropriately.
print(df_encoded.head())

#Correct encoding
# Ensure that categorical variables have been properly encoded.
print(df_encoded.columns)
