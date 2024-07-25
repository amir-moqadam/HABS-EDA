import pandas as pd

def load_dataset(file_path):
    """
    Load the dataset from a CSV file.
    """
    return pd.read_csv(file_path)

def handle_missing_values(df):
    """
    Handle missing values in the dataframe.
    For simplicity, we will fill numerical missing values with the mean and categorical missing values with the mode.
    """
    for column in df.columns:
        if df[column].dtype == 'object':
            df[column].fillna(df[column].mode()[0], inplace=True)
        else:
            df[column].fillna(df[column].mean(), inplace=True)
    print("\nMissing values handled.")

def correct_data_types(df):
    """
    Correct data types if necessary.
    Example: Convert 'date' columns from string to datetime.
    """
    for column in df.columns:
        if 'date' in column.lower():
            df[column] = pd.to_datetime(df[column])
    print("\nData types corrected.")

def check_duplicates(df):
    """
    Check for and handle duplicate rows.
    """
    duplicates = df.duplicated().sum()
    if duplicates > 0:
        df.drop_duplicates(inplace=True)
        print("\n{} duplicate rows found and removed.".format(duplicates))
    else:
        print("\nNo duplicate rows found.")

def standardize_categorical_variables(df):
    """
    Standardize categorical variables.
    Convert categorical text data to lowercase and strip leading/trailing whitespace.
    """
    for column in df.select_dtypes(include=['object', 'category']).columns:
        df[column] = df[column].str.lower().str.strip()
    print("\nCategorical variables standardized.")

def main(file_path):
    # Load the dataset
    df = load_dataset(file_path)
    
    # Handle missing values
    handle_missing_values(df)
    
    # Correct data types
    correct_data_types(df)
    
    # Check for duplicates
    check_duplicates(df)
    
    # Standardize categorical variables
    standardize_categorical_variables(df)
    
    # Save the cleaned dataset
    cleaned_file_path = "cleaned_" + file_path
    df.to_csv(cleaned_file_path, index=False)
    print("\nCleaned dataset saved to {}".format(cleaned_file_path))

# Example usage:
if __name__ == "__main__":
    file_path = "path_to_your_dataset.csv"  # Replace with the path to your dataset file
    main(file_path)
