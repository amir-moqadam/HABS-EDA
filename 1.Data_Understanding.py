import pandas as pd

def load_dataset(file_path):
    """
    Load the dataset from a CSV file.
    """
    return pd.read_csv(file_path)

def display_first_rows(df, num_rows=5):
    """
    Display the first few rows of the dataframe.
    """
    print("First {} rows of the dataset:".format(num_rows))
    print(df.head(num_rows))

def get_overview(df):
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

def summarize_features(df):
    """
    Summarize key features of the dataset.
    """
    print("\nSummary of numerical features:")
    print(df.describe())
    
    print("\nSummary of categorical features:")
    for column in df.select_dtypes(include=['object', 'category']).columns:
        print("\nColumn: {}".format(column))
        print(df[column].value_counts())

def check_missing_values(df):
    """
    Check for missing values in the dataset.
    """
    print("\nMissing values in the dataset:")
    missing_values = df.isnull().sum()
    print(missing_values[missing_values > 0])

# Main function to execute the EDA tasks
def main(file_path):
    # Load the dataset
    df = load_dataset(file_path)
    
    # Display the first few rows
    display_first_rows(df)
    
    # Get an overview of the dataset
    get_overview(df)
    
    # Summarize key features
    summarize_features(df)
    
    # Check for missing values
    check_missing_values(df)

# Example usage:
if __name__ == "__main__":
    file_path = "path_to_your_dataset.csv"  # Replace with the path to your dataset file
    main(file_path)
