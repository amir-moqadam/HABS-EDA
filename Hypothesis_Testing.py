import pandas as pd
import scipy.stats as stats
import numpy as np

def load_dataset(file_path):
    """
    Load the dataset from a CSV file.
    """
    return pd.read_csv(file_path)

def t_test(df, group_col, value_col):
    """
    Perform a t-test to compare the means of two groups.
    """
    groups = df[group_col].unique()
    if len(groups) != 2:
        print(f"T-test requires exactly 2 groups. Found {len(groups)} groups in {group_col}.")
        return
    
    group1 = df[df[group_col] == groups[0]][value_col]
    group2 = df[df[group_col] == groups[1]][value_col]
    
    t_stat, p_value = stats.ttest_ind(group1, group2, nan_policy='omit')
    print(f"T-test between {groups[0]} and {groups[1]} for {value_col}:")
    print(f"T-statistic: {t_stat:.4f}, P-value: {p_value:.4f}\n")

def chi_square_test(df, col1, col2):
    """
    Perform a chi-square test for independence between two categorical variables.
    """
    contingency_table = pd.crosstab(df[col1], df[col2])
    chi2_stat, p_val, dof, expected = stats.chi2_contingency(contingency_table)
    print(f"Chi-square test between {col1} and {col2}:")
    print(f"Chi2 Statistic: {chi2_stat:.4f}, P-value: {p_val:.4f}\n")

def correlation_test(df, col1, col2):
    """
    Perform a Pearson correlation test between two numerical variables.
    """
    corr, p_val = stats.pearsonr(df[col1].dropna(), df[col2].dropna())
    print(f"Pearson correlation test between {col1} and {col2}:")
    print(f"Correlation coefficient: {corr:.4f}, P-value: {p_val:.4f}\n")

def main(file_path):
    # Load the dataset
    df = load_dataset(file_path)
    
    # Perform t-tests
    t_test(df, 'GroupColumn', 'ValueColumn')  # Replace with actual column names

    # Perform chi-square tests
    chi_square_test(df, 'CategoricalColumn1', 'CategoricalColumn2')  # Replace with actual column names
    
    # Perform correlation tests
    correlation_test(df, 'NumericalColumn1', 'NumericalColumn2')  # Replace with actual column names

# Example usage:
if __name__ == "__main__":
    file_path = "path_to_your_dataset.csv"  # Replace with the path to your dataset file
    main(file_path)
