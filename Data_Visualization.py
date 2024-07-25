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

def plot_box_plots(df):
    """
    Plot box plots for numerical variables.
    """
    numerical_columns = df.select_dtypes(include=['number']).columns
    plt.figure(figsize=(15, 10))
    df[numerical_columns].boxplot()
    plt.title('Box plots of Numerical Features', fontsize=20)
    plt.xticks(rotation=90)
    plt.show()

def plot_scatter_plots(df, x_column, y_column):
    """
    Plot scatter plots for pairs of numerical variables.
    """
    plt.figure(figsize=(10, 7))
    sns.scatterplot(x=x_column, y=y_column, data=df)
    plt.title(f'Scatter Plot of {x_column} vs {y_column}', fontsize=15)
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

def plot_violin_plots(df, categorical_column, numerical_column):
    """
    Plot violin plots for numerical variables across different categories.
    """
    plt.figure(figsize=(12, 6))
    sns.violinplot(x=categorical_column, y=numerical_column, data=df, palette='viridis')
    plt.title(f'Violin Plot of {numerical_column} by {categorical_column}', fontsize=15)
    plt.xticks(rotation=45)
    plt.show()

def main(file_path):
    # Load the dataset
    df = load_dataset(file_path)
    
    # Plot histograms for numerical variables
    plot_histograms(df)
    
    # Plot box plots for numerical variables
    plot_box_plots(df)
    
    # Plot scatter plots for pairs of numerical variables
    numerical_columns = df.select_dtypes(include=['number']).columns
    for i in range(len(numerical_columns)):
        for j in range(i + 1, len(numerical_columns)):
            plot_scatter_plots(df, numerical_columns[i], numerical_columns[j])
    
    # Plot correlation heatmap
    plot_correlation_heatmap(df)
    
    # Plot violin plots for numerical variables across different categories
    categorical_columns = df.select_dtypes(include=['object', 'category']).columns
    for num_col in numerical_columns:
        for cat_col in categorical_columns:
            plot_violin_plots(df, cat_col, num_col)

# Example usage:
if __name__ == "__main__":
    file_path = "path_to_your_dataset.csv"  # Replace with the path to your dataset file
    main(file_path)
