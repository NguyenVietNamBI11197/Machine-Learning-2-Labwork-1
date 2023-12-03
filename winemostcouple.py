import os
import pandas as pd

# Specify file path
file_path = 'C:/Users/Bin/Desktop/Machine Learning 2/wine/wine.data'

# Check if the file exists
if os.path.exists(file_path):
    # Load Wine dataset
    wine_data = pd.read_csv(file_path, header=None, names=['Class', 'Alcohol', 'Malic Acid', 'Ash', 'Alcalinity of Ash', 'Magnesium', 'Total Phenols', 'Flavanoids', 'Nonflavanoid Phenols', 'Proanthocyanins', 'Color Intensity', 'Hue', 'OD280/OD315 of Diluted Wines', 'Proline'])

    # Drop the 'Class' column
    wine_data_numeric = wine_data.drop('Class', axis=1)

    # Calculate correlation matrix
    wine_correlation = wine_data_numeric.corr()

    # Find the most correlated couple of features
    most_correlated_pair = (wine_correlation.abs().stack().idxmax())
    feature1, feature2 = most_correlated_pair

    # Display the results
    print(f"Wine Dataset Most Correlated Features: {feature1} and {feature2}")
    print(f"Correlation Coefficient: {wine_correlation.loc[feature1, feature2]}")
else:
    print(f"The file {file_path} does not exist.")
