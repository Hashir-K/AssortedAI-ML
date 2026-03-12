import pandas as pd
import numpy as np

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

#Explanation: This code generates a dummy dataset with 100 rows and 4 columns: two numeric features, one categorical feature, and a binary target variable. 
# The dataset includes some missing values and an outlier to simulate real-world data challenges.



def load_data(df):
    return df

def handle_missing_values(df):
    return df.fillna(df.mean())  # For numeric data, fill missing values with the mean

def remove_outliers(df):
    z_scores = np.abs(stats.zscore(df.select_dtypes(include=[np.number])))
    return df[(z_scores < 3).all(axis=1)]  # Remove rows with any outliers

def scale_data(df):
    scaler = StandardScaler()
    df[df.select_dtypes(include=[np.number]).columns] = scaler.fit_transform(df.select_dtypes(include=[np.number]))
    return df

def encode_categorical(df, categorical_columns):
    return pd.get_dummies(df, columns=categorical_columns)

def save_data(df, output_filepath):
    df.to_csv(output_filepath, index=False)

#Explanation: These functions encapsulate the core preprocessing tasks, making them reusable across different datasets. They will be applied to our dummy data.



# Load the data
df_preprocessed = load_data(df_dummy)

# Handle missing values
df_preprocessed = handle_missing_values(df_preprocessed)

# Remove outliers
df_preprocessed = remove_outliers(df_preprocessed)

# Scale the data
df_preprocessed = scale_data(df_preprocessed)

# Encode categorical variables
df_preprocessed = encode_categorical(df_preprocessed, ['Category'])

# Display the preprocessed data
print(df_preprocessed.head())

#Explanation: This code applies the preprocessing steps to the dummy data. It handles missing values by filling them with the mean, 
# removes outliers using the Z-score method, scales the numeric data, and encodes the categorical variables using one-hot encoding.




# Save the cleaned and preprocessed DataFrame to a CSV file
save_data(df_preprocessed, 'preprocessed_dummy_data.csv')

print('Preprocessing complete. Preprocessed data saved as preprocessed_dummy_data.csv')

#Explanation: Saving the preprocessed data to a new file ensures that it’s ready for use in training ML models. 
# This step makes it easy to use the cleaned and processed data in future analysis or modeling efforts.


print(df_preprocessed.isnull().sum())
#Explanation: This checks that all missing values. have been handled properly.

print(df_preprocessed.describe())
#Explanation: This summarizes the dataset and confirms that any extreme values (outliers). have been removed.

print(df_preprocessed.head())
#Explanation: This ensures that the numeric features have been scaled properly, making them ready for ML algorithms.

print(df_preprocessed.columns)
#Explanation: This confirms that the categorical variables have been encoded into numerical values correctly.


