import pandas as pd

def load_dataset(file_path):
    """
    Load the dataset from a CSV file.
    """
    return pd.read_csv(file_path)

def generate_summary_statistics(df):
    """
    Generate summary statistics for numerical features.
    """
    print("\nSummary statistics for numerical features:")
    print(df.describe())

def generate_categorical_summary(df):
    """
    Generate frequency counts for categorical features.
    """
    print("\nSummary statistics for categorical features:")
    for column in df.select_dtypes(include=['object', 'category']).columns:
        print("\nColumn: {}".format(column))
        print(df[column].value_counts())

def get_dataset_overview(df):
    """
    Get an overview of the dataset.
    """
    print("\nDataset Overview:")
    print("Number of rows: {}".format(df.shape[0]))
    print("Number of columns: {}".format(df.shape[1]))
    print("Column names:")
    print(df.columns.tolist())
    print("\nData types:")
    print(df.dtypes)
    print("\nMissing values in the dataset:")
    missing_values = df.isnull().sum()
    print(missing_values[missing_values > 0])

def main(file_path):
    # Load the dataset
    df = load_dataset(file_path)
    
    # Get dataset overview
    get_dataset_overview(df)
    
    # Generate summary statistics for numerical features
    generate_summary_statistics(df)
    
    # Generate summary statistics for categorical features
    generate_categorical_summary(df)

# Example usage:
if __name__ == "__main__":
    file_path = "path_to_your_dataset.csv"  # Replace with the path to your dataset file
    main(file_path)
