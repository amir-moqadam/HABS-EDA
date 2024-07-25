import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from fpdf import FPDF
import os

def load_dataset(file_path):
    """
    Load the dataset from a CSV file.
    """
    return pd.read_csv(file_path)

def generate_summary_statistics(df):
    """
    Generate summary statistics for numerical features.
    """
    summary = df.describe().T
    return summary

def generate_categorical_summary(df):
    """
    Generate frequency counts for categorical features.
    """
    categorical_summary = {}
    for column in df.select_dtypes(include=['object', 'category']).columns:
        categorical_summary[column] = df[column].value_counts()
    return categorical_summary

def plot_histograms(df, output_dir):
    """
    Plot histograms for numerical variables and save them.
    """
    numerical_columns = df.select_dtypes(include=['number']).columns
    for col in numerical_columns:
        plt.figure(figsize=(10, 6))
        sns.histplot(df[col], kde=True, bins=20)
        plt.title(f'Histogram of {col}', fontsize=15)
        plt.xlabel(col)
        plt.ylabel('Frequency')
        plt.savefig(os.path.join(output_dir, f'{col}_histogram.png'))
        plt.close()

def plot_box_plots(df, output_dir):
    """
    Plot box plots for numerical variables and save them.
    """
    numerical_columns = df.select_dtypes(include=['number']).columns
    for col in numerical_columns:
        plt.figure(figsize=(10, 6))
        sns.boxplot(x=df[col])
        plt.title(f'Box plot of {col}', fontsize=15)
        plt.xlabel(col)
        plt.savefig(os.path.join(output_dir, f'{col}_boxplot.png'))
        plt.close()

def plot_correlation_heatmap(df, output_dir):
    """
    Plot a correlation heatmap for numerical variables and save it.
    """
    numerical_columns = df.select_dtypes(include=['number']).columns
    correlation_matrix = df[numerical_columns].corr()
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm', square=True, linewidths=.5)
    plt.title('Correlation Heatmap', fontsize=20)
    plt.savefig(os.path.join(output_dir, 'correlation_heatmap.png'))
    plt.close()

def generate_pdf_report(df, output_dir, pdf_path):
    """
    Generate a PDF report with key statistics, visualizations, and findings.
    """
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    
    # Add a title page
    pdf.add_page()
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(200, 10, 'Data Analysis Report', ln=True, align='C')
    
    # Add summary statistics
    pdf.add_page()
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(200, 10, 'Summary Statistics', ln=True)
    
    summary = generate_summary_statistics(df)
    pdf.set_font("Arial", '', 10)
    for i in range(len(summary)):
        pdf.cell(0, 10, str(summary.iloc[i]), ln=True)
    
    # Add categorical summary
    pdf.add_page()
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(200, 10, 'Categorical Summary', ln=True)
    
    categorical_summary = generate_categorical_summary(df)
    pdf.set_font("Arial", '', 10)
    for column, counts in categorical_summary.items():
        pdf.cell(0, 10, f'{column}:', ln=True)
        for category, count in counts.items():
            pdf.cell(0, 10, f'    {category}: {count}', ln=True)
    
    # Add histograms
    plot_histograms(df, output_dir)
    pdf.add_page()
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(200, 10, 'Histograms', ln=True)
    
    for col in df.select_dtypes(include=['number']).columns:
        pdf.add_page()
        pdf.image(os.path.join(output_dir, f'{col}_histogram.png'), x=10, y=30, w=190)
    
    # Add box plots
    plot_box_plots(df, output_dir)
    pdf.add_page()
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(200, 10, 'Box Plots', ln=True)
    
    for col in df.select_dtypes(include=['number']).columns:
        pdf.add_page()
        pdf.image(os.path.join(output_dir, f'{col}_boxplot.png'), x=10, y=30, w=190)
    
    # Add correlation heatmap
    plot_correlation_heatmap(df, output_dir)
    pdf.add_page()
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(200, 10, 'Correlation Heatmap', ln=True)
    pdf.add_page()
    pdf.image(os.path.join(output_dir, 'correlation_heatmap.png'), x=10, y=30, w=190)
    
    # Save the PDF report
    pdf.output(pdf_path)
    print(f'Report saved to {pdf_path}')

def main(file_path):
    # Load the dataset
    df = load_dataset(file_path)
    
    # Create output directory for plots
    output_dir = 'plots'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Generate PDF report
    pdf_path = 'data_analysis_report.pdf'
    generate_pdf_report(df, output_dir, pdf_path)

# Example usage:
if __name__ == "__main__":
    file_path = "path_to_your_dataset.csv"  # Replace with the path to your dataset file
    main(file_path)
