import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def load_dataset(file_path):
    """
    Load the dataset from a CSV file.
    """
    return pd.read_csv(file_path)

def detect_outliers_zscore(df, threshold=3):
    """
    Detect outliers using Z-scores.
    """
    z_scores = np.abs((df - df.mean()) / df.std())
    outliers = (z_scores > threshold).sum().sum()
    print(f"\nNumber of outliers detected using Z-score method: {outliers}")
    return z_scores > threshold

def plot_box_plots(df):
    """
    Plot box plots for numerical variables to visualize outliers.
    """
    numerical_columns = df.select_dtypes(include=['number']).columns
    plt.figure(figsize=(15, 10))
    df[numerical_columns].boxplot()
    plt.title('Box plots of Numerical Features', fontsize=20)
    plt.xticks(rotation=90)
    plt.show()

def plot_scatter_plots(df, x_column, y_column):
    """
    Plot scatter plots to visualize outliers.
    """
    plt.figure(figsize=(10, 7))
    sns.scatterplot(x=x_column, y=y_column, data=df)
    plt.title(f'Scatter Plot of {x_column} vs {y_column}', fontsize=15)
    plt.show()

def main(file_path):
    # Load the dataset
    df = load_dataset(file_path)
    
    # Detect outliers using Z-scores
    outliers_zscore = detect_outliers_zscore(df)
    
    # Plot box plots for numerical variables
    plot_box_plots(df)
    
    # Plot scatter plots for pairs of numerical variables
    numerical_columns = df.select_dtypes(include=['number']).columns
    for i in range(len(numerical_columns)):
        for j in range(i + 1, len(numerical_columns)):
            plot_scatter_plots(df, numerical_columns[i], numerical_columns[j])
    
    # Highlight outliers in scatter plots
    for i in range(len(numerical_columns)):
        for j in range(i + 1, len(numerical_columns)):
            plt.figure(figsize=(10, 7))
            sns.scatterplot(x=numerical_columns[i], y=numerical_columns[j], data=df, hue=outliers_zscore[numerical_columns[i]] | outliers_zscore[numerical_columns[j]], palette={True: 'red', False: 'blue'})
            plt.title(f'Scatter Plot of {numerical_columns[i]} vs {numerical_columns[j]} with Outliers Highlighted', fontsize=15)
            plt.show()

# Example usage:
if __name__ == "__main__":
    file_path = "path_to_your_dataset.csv"  # Replace with the path to your dataset file
    main(file_path)
