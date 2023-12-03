import os
import pandas as pd

# Specify file path
file_path = 'C:/Users/Bin/Desktop/Machine Learning 2//wine.data'

# Check if the file exists
if os.path.exists(file_path):
    # Load Wine dataset
    wine_data = pd.read_csv(file_path, header=None, names=['Class', 'Alcohol', 'Malic Acid', 'Ash', 'Alcalinity of Ash', 'Magnesium', 'Total Phenols', 'Flavanoids', 'Nonflavanoid Phenols', 'Proanthocyanins', 'Color Intensity', 'Hue', 'OD280/OD315 of Diluted Wines', 'Proline'])

    # Drop the 'Class' column
    wine_data_numeric = wine_data.drop('Class', axis=1)

    # Calculate mean and variance
    wine_mean = wine_data_numeric.mean()
    wine_variance = wine_data_numeric.var()

    # Calculate covariance matrix
    wine_covariance = wine_data_numeric.cov()

    # Calculate correlation matrix
    wine_correlation = wine_data_numeric.corr()

    # Display the results
    print("Wine Dataset Mean:")
    print(wine_mean)

    print("\nWine Dataset Variance:")
    print(wine_variance)

    print("\nWine Dataset Covariance Matrix:")
    print(wine_covariance)

    print("\nWine Dataset Correlation Matrix:")
    print(wine_correlation)
else:
    print(f"The file {file_path} does not exist.")
