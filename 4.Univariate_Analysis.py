import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def load_dataset(file_path):
    """
    Load the dataset from a CSV file.
    """
    return pd.read_csv(file_path)

def plot_histograms(df):
    """
    Plot histograms for numerical variables.
    """
    numerical_columns = df.select_dtypes(include=['number']).columns
    df[numerical_columns].hist(figsize=(15, 15), bins=20, edgecolor='black')
    plt.suptitle('Histograms of Numerical Features', fontsize=20)
    plt.show()

def plot_boxplots(df):
    """
    Plot box plots for numerical variables.
    """
    numerical_columns = df.select_dtypes(include=['number']).columns
    plt.figure(figsize=(15, 10))
    df[numerical_columns].boxplot()
    plt.title('Box plots of Numerical Features', fontsize=20)
    plt.xticks(rotation=90)
    plt.show()

def plot_bar_charts(df):
    """
    Plot bar charts for categorical variables.
    """
    categorical_columns = df.select_dtypes(include=['object', 'category']).columns
    for column in categorical_columns:
        plt.figure(figsize=(10, 5))
        sns.countplot(y=column, data=df, palette='viridis')
        plt.title(f'Bar Chart of {column}', fontsize=15)
        plt.show()

def main(file_path):
    # Load the dataset
    df = load_dataset(file_path)
    
    # Plot histograms for numerical variables
    plot_histograms(df)
    
    # Plot box plots for numerical variables
    plot_boxplots(df)
    
    # Plot bar charts for categorical variables
    plot_bar_charts(df)

# Example usage:
if __name__ == "__main__":
    file_path = "path_to_your_dataset.csv"  # Replace with the path to your dataset file
    main(file_path)
