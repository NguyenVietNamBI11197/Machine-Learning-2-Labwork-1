import os
import pandas as pd
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Specify file path
file_path = 'C:/Users/Bin/Desktop/Machine Learning 2/wine.data'

# Check if the file exists
if os.path.exists(file_path):
    # Load Wine dataset
    wine_data = pd.read_csv(file_path, header=None, names=['Class', 'Alcohol', 'Malic Acid', 'Ash', 'Alcalinity of Ash', 'Magnesium', 'Total Phenols', 'Flavanoids', 'Nonflavanoid Phenols', 'Proanthocyanins', 'Color Intensity', 'Hue', 'OD280/OD315 of Diluted Wines', 'Proline'])

    # Separate features and target variable
    X = wine_data.drop('Class', axis=1)
    y = wine_data['Class']

    # Apply PCA
    pca = PCA()
    X_pca = pca.fit_transform(X)

    # Print cumulative explained variance ratio for the first two components
    cumulative_explained_variance_ratio = pca.explained_variance_ratio_.cumsum()
    print(f"Cumulative Explained Variance Ratio for the first two components (Wine Dataset): {cumulative_explained_variance_ratio[1]}")

    # Plot explained variance ratio
    plt.bar(range(1, len(pca.explained_variance_ratio_) + 1), pca.explained_variance_ratio_)
    plt.xlabel('Principal Component')
    plt.ylabel('Explained Variance Ratio')
    plt.title('Explained Variance Ratio of Principal Components (Wine Dataset)')
    plt.show()
else:
    print(f"The file {file_path} does not exist.")
