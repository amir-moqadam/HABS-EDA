import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

def load_dataset(file_path):
    """
    Load the dataset from a CSV file.
    """
    return pd.read_csv(file_path)

def plot_pair_plots(df):
    """
    Plot pair plots for numerical variables.
    """
    numerical_columns = df.select_dtypes(include=['number']).columns
    sns.pairplot(df[numerical_columns])
    plt.suptitle('Pair Plots of Numerical Features', y=1.02, fontsize=20)
    plt.show()

def perform_pca(df, n_components=2):
    """
    Perform Principal Component Analysis (PCA) and plot the results.
    """
    numerical_columns = df.select_dtypes(include=['number']).columns
    x = df[numerical_columns].dropna()  # Drop rows with missing values for PCA
    x = StandardScaler().fit_transform(x)  # Standardize the data

    pca = PCA(n_components=n_components)
    principal_components = pca.fit_transform(x)
    pca_df = pd.DataFrame(data=principal_components, columns=[f'PC{i+1}' for i in range(n_components)])

    plt.figure(figsize=(10, 7))
    sns.scatterplot(x='PC1', y='PC2', data=pca_df, palette='viridis')
    plt.title('PCA: First two principal components', fontsize=15)
    plt.show()
    print("Explained variance ratio by each principal component:", pca.explained_variance_ratio_)

def perform_kmeans_clustering(df, n_clusters=3):
    """
    Perform K-means clustering and plot the results.
    """
    numerical_columns = df.select_dtypes(include=['number']).columns
    x = df[numerical_columns].dropna()  # Drop rows with missing values for clustering
    x = StandardScaler().fit_transform(x)  # Standardize the data

    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(x)
    clusters = kmeans.labels_
    df['Cluster'] = clusters

    pca = PCA(n_components=2)
    principal_components = pca.fit_transform(x)
    pca_df = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2'])
    pca_df['Cluster'] = clusters

    plt.figure(figsize=(10, 7))
    sns.scatterplot(x='PC1', y='PC2', hue='Cluster', data=pca_df, palette='viridis')
    plt.title(f'K-means Clustering with {n_clusters} Clusters', fontsize=15)
    plt.show()

def main(file_path):
    # Load the dataset
    df = load_dataset(file_path)
    
    # Plot pair plots for numerical variables
    plot_pair_plots(df)
    
    # Perform PCA
    perform_pca(df)
    
    # Perform K-means clustering
    perform_kmeans_clustering(df)

# Example usage:
if __name__ == "__main__":
    file_path = "path_to_your_dataset.csv"  # Replace with the path to your dataset file
    main(file_path)
