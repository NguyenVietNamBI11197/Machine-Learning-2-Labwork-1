import os
import pandas as pd
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

# Specify file path
file_path = 'C:/Users/Bin/Desktop/Machine Learning 2/iris.data'

# Check if the file exists
if os.path.exists(file_path):
    # Load Iris dataset
    iris_data = pd.read_csv(file_path, header=None, names=['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'class'])

    # Separate features and target variable
    X = iris_data.drop('class', axis=1)
    y = iris_data['class']

    # Apply PCA
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)

    # Create a DataFrame for visualization
    pca_df = pd.DataFrame(data=X_pca, columns=['PC1', 'PC2'])
    pca_df['class'] = y

    # Plot the scatter plot
    plt.figure(figsize=(10, 8))
    sns.scatterplot(x='PC1', y='PC2', hue='class', data=pca_df, palette='viridis')
    plt.title('Scatter Plot of First Two Principal Components (Iris Dataset)')
    plt.show()
else:
    print(f"The file {file_path} does not exist.")
