import os
import pandas as pd

# Specify file path
file_path = 'C:/Users/Bin/Desktop/Machine Learning 2/iris.data'

# Check if the file exists
if os.path.exists(file_path):
    # Load Iris dataset
    iris_data = pd.read_csv(file_path, header=None, names=['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'class'])

    # Drop the 'class' column
    iris_data_numeric = iris_data.drop('class', axis=1)

    # Calculate mean and variance
    iris_mean = iris_data_numeric.mean()
    iris_variance = iris_data_numeric.var()

    # Calculate covariance matrix
    iris_covariance = iris_data_numeric.cov()

    # Calculate correlation matrix
    iris_correlation = iris_data_numeric.corr()

    # Display the results
    print("Iris Dataset Mean:")
    print(iris_mean)

    print("\nIris Dataset Variance:")
    print(iris_variance)

    print("\nIris Dataset Covariance Matrix:")
    print(iris_covariance)

    print("\nIris Dataset Correlation Matrix:")
    print(iris_correlation)
else:
    print(f"The file {file_path} does not exist.")