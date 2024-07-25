import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder

def load_dataset(file_path):
    """
    Load the dataset from a CSV file.
    """
    return pd.read_csv(file_path)

def create_new_features(df):
    """
    Create new features from existing data.
    Example: Age groups, interaction terms, etc.
    """
    # Example: Create age groups
    df['AgeGroup'] = pd.cut(df['Age'], bins=[60, 70, 80, 90], labels=['60-70', '70-80', '80-90'])
    print("\nNew features created: AgeGroup")
    return df

def normalize_features(df, feature_columns):
    """
    Normalize numerical features using MinMaxScaler.
    """
    scaler = MinMaxScaler()
    df[feature_columns] = scaler.fit_transform(df[feature_columns])
    print("\nNumerical features normalized using MinMaxScaler.")
    return df

def standardize_features(df, feature_columns):
    """
    Standardize numerical features using StandardScaler.
    """
    scaler = StandardScaler()
    df[feature_columns] = scaler.fit_transform(df[feature_columns])
    print("\nNumerical features standardized using StandardScaler.")
    return df

def encode_categorical_variables(df):
    """
    Encode categorical variables using OneHotEncoder.
    """
    categorical_columns = df.select_dtypes(include=['object', 'category']).columns
    encoder = OneHotEncoder(sparse=False, drop='first')
    encoded_data = encoder.fit_transform(df[categorical_columns])
    encoded_df = pd.DataFrame(encoded_data, columns=encoder.get_feature_names_out(categorical_columns))
    
    # Drop original categorical columns and concatenate the encoded columns
    df.drop(categorical_columns, axis=1, inplace=True)
    df = pd.concat([df, encoded_df], axis=1)
    print("\nCategorical variables encoded using OneHotEncoder.")
    return df

def main(file_path):
    # Load the dataset
    df = load_dataset(file_path)
    
    # Create new features
    df = create_new_features(df)
    
    # Define numerical features for normalization/standardization
    numerical_columns = df.select_dtypes(include=['number']).columns
    
    # Normalize features
    df = normalize_features(df, numerical_columns)
    
    # Standardize features
    df = standardize_features(df, numerical_columns)
    
    # Encode categorical variables
    df = encode_categorical_variables(df)
    
    # Save the processed dataset
    processed_file_path = "processed_" + file_path
    df.to_csv(processed_file_path, index=False)
    print("\nProcessed dataset saved to {}".format(processed_file_path))

# Example usage:
if __name__ == "__main__":
    file_path = "path_to_your_dataset.csv"  # Replace with the path to your dataset file
    main(file_path)
