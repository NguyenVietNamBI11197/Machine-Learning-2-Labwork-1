import os
import pandas as pd
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

# Specify file path
file_path = 'C:/Users/Bin/Desktop/Machine Learning 2/wine.data'

# Check if the file exists
if os.path.exists(file_path):
    # Load Wine dataset
    wine_data = pd.read_csv(file_path, header=None, names=['Class', 'Alcohol', 'Malic Acid', 'Ash', 'Alcalinity of Ash', 'Magnesium', 'Total Phenols', 'Flavanoids', 'Nonflavanoid Phenols', 'Proanthocyanins', 'Color Intensity', 'Hue', 'OD280/OD315 of Diluted Wines', 'Proline'])

    # Separate features and target variable
    X = wine_data.drop('Class', axis=1)
    y = wine_data['Class']

    # Apply PCA with more components (e.g., 6 components)
    num_components = 6
    pca = PCA(n_components=num_components)
    X_pca = pca.fit_transform(X)

    # Create a DataFrame for visualization
    pca_df = pd.DataFrame(data=X_pca, columns=[f'PC{i}' for i in range(1, num_components + 1)])
    pca_df['class'] = y

    # Plot the scatter plot
    plt.figure(figsize=(12, 10))
    sns.scatterplot(x='PC1', y='PC2', hue='class', data=pca_df, palette='viridis')
    plt.title(f'Scatter Plot of First Two Principal Components (Wine Dataset)')
    plt.show()
else:
    print(f"The file {file_path} does not exist.")
