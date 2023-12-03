import os
import pandas as pd

# Specify file path
file_path = 'C:/Users/Bin/Desktop/Machine Learning 2/iris/iris.data'

# Check if the file exists
if os.path.exists(file_path):
    # Load Iris dataset
    iris_data = pd.read_csv(file_path, header=None, names=['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'class'])

    # Calculate correlation matrix
    iris_correlation = iris_data.corr()

    # Find the most correlated couple of features
    most_correlated_pair = (iris_correlation.abs().stack().idxmax())
    feature1, feature2 = most_correlated_pair

    # Display the results
    print(f"Iris Dataset Most Correlated Features: {feature1} and {feature2}")
    print(f"Correlation Coefficient: {iris_correlation.loc[feature1, feature2]}")
else:
    print(f"The file {file_path} does not exist.")
