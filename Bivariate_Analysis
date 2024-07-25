import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def load_dataset(file_path):
    """
    Load the dataset from a CSV file.
    """
    return pd.read_csv(file_path)

def plot_scatter_plots(df):
    """
    Plot scatter plots for pairs of numerical variables.
    """
    numerical_columns = df.select_dtypes(include=['number']).columns
    sns.pairplot(df[numerical_columns], diag_kind='kde')
    plt.suptitle('Scatter Plots of Numerical Features', y=1.02, fontsize=20)
    plt.show()

def plot_box_plots(df):
    """
    Plot box plots for numerical variables across different categories.
    """
    numerical_columns = df.select_dtypes(include=['number']).columns
    categorical_columns = df.select_dtypes(include=['object', 'category']).columns
    
    for num_col in numerical_columns:
        for cat_col in categorical_columns:
            plt.figure(figsize=(12, 6))
            sns.boxplot(x=cat_col, y=num_col, data=df, palette='viridis')
            plt.title(f'Box Plot of {num_col} by {cat_col}', fontsize=15)
            plt.xticks(rotation=45)
            plt.show()

def plot_correlation_heatmap(df):
    """
    Plot a correlation heatmap for numerical variables.
    """
    numerical_columns = df.select_dtypes(include=['number']).columns
    correlation_matrix = df[numerical_columns].corr()
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm', square=True, linewidths=.5)
    plt.title('Correlation Heatmap', fontsize=20)
    plt.show()

def main(file_path):
    # Load the dataset
    df = load_dataset(file_path)
    
    # Plot scatter plots for numerical variables
    plot_scatter_plots(df)
    
    # Plot box plots for numerical variables across different categories
    plot_box_plots(df)
    
    # Plot correlation heatmap
    plot_correlation_heatmap(df)

# Example usage:
if __name__ == "__main__":
    file_path = "path_to_your_dataset.csv"  # Replace with the path to your dataset file
    main(file_path)
