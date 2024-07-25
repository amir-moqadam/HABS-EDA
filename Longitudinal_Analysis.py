import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from lifelines import KaplanMeierFitter

def load_dataset(file_path):
    """
    Load the dataset from a CSV file.
    """
    return pd.read_csv(file_path)

def plot_time_series(df, id_column, time_column, value_columns):
    """
    Plot time-series graphs for key variables.
    """
    plt.figure(figsize=(14, 7))
    for value_column in value_columns:
        for key, grp in df.groupby([id_column]):
            plt.plot(grp[time_column], grp[value_column], label=f"{value_column} for {key}")
        plt.title(f'Time Series Plot for {value_column}', fontsize=15)
        plt.xlabel(time_column)
        plt.ylabel(value_column)
        plt.legend(loc='best')
        plt.show()

def perform_survival_analysis(df, time_column, event_column):
    """
    Perform survival analysis and plot the Kaplan-Meier curve.
    """
    kmf = KaplanMeierFitter()
    kmf.fit(durations=df[time_column], event_observed=df[event_column])
    
    plt.figure(figsize=(10, 7))
    kmf.plot_survival_function()
    plt.title('Kaplan-Meier Survival Curve', fontsize=15)
    plt.xlabel('Time')
    plt.ylabel('Survival Probability')
    plt.show()

def main(file_path):
    # Load the dataset
    df = load_dataset(file_path)
    
    # Define the columns for the analysis
    id_column = "ParticipantID"  # Replace with the actual column name for participant ID
    time_column = "Time"  # Replace with the actual column name for time
    value_columns = ["CognitiveScore", "ImagingMeasure"]  # Replace with the actual column names for values to plot

    # Plot time-series graphs
    plot_time_series(df, id_column, time_column, value_columns)
    
    # Perform survival analysis if applicable
    event_column = "EventOccurred"  # Replace with the actual column name for event occurrence (0/1)
    perform_survival_analysis(df, time_column, event_column)

# Example usage:
if __name__ == "__main__":
    file_path = "path_to_your_dataset.csv"  # Replace with the path to your dataset file
    main(file_path)
