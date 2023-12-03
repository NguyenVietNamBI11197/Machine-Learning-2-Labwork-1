import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Specify file path
file_path = 'C:/Users/Bin/Desktop/Machine Learning 2/iris.data'

# Check if the file exists
if os.path.exists(file_path):
    # Load Iris dataset
    iris_data = pd.read_csv(file_path, header=None, names=['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'class'])

    # Plot scatter plot for sepal length vs. sepal width
    plt.figure(figsize=(10, 8))
    sns.scatterplot(x='sepal_length', y='sepal_width', hue='class', data=iris_data, palette='viridis')
    plt.title('Scatter Plot of Sepal Length vs. Sepal Width (Iris Dataset)')
    plt.show()

    # Plot scatter plot for petal length vs. petal width
    plt.figure(figsize=(10, 8))
    sns.scatterplot(x='petal_length', y='petal_width', hue='class', data=iris_data, palette='viridis')
    plt.title('Scatter Plot of Petal Length vs. Petal Width (Iris Dataset)')
    plt.show()
else:
    print(f"The file {file_path} does not exist.")
