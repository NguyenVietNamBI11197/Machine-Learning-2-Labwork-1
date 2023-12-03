import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Specify file path
file_path = 'C:/Users/Bin/Desktop/Machine Learning 2/wine.data'

# Check if the file exists
if os.path.exists(file_path):
    # Load Wine dataset
    wine_data = pd.read_csv(file_path, header=None, names=['Class', 'Alcohol', 'Malic Acid', 'Ash', 'Alcalinity of Ash', 'Magnesium', 'Total Phenols', 'Flavanoids', 'Nonflavanoid Phenols', 'Proanthocyanins', 'Color Intensity', 'Hue', 'OD280/OD315 of Diluted Wines', 'Proline'])

    # Plot scatter plot for Alcohol vs. Color Intensity
    plt.figure(figsize=(10, 8))
    sns.scatterplot(x='Alcohol', y='Color Intensity', hue='Class', data=wine_data, palette='viridis')
    plt.title('Scatter Plot of Alcohol vs. Color Intensity (Wine Dataset)')
    plt.show()

    # Plot scatter plot for Flavanoids vs. Total Phenols
    plt.figure(figsize=(10, 8))
    sns.scatterplot(x='Flavanoids', y='Total Phenols', hue='Class', data=wine_data, palette='viridis')
    plt.title('Scatter Plot of Flavanoids vs. Total Phenols (Wine Dataset)')
    plt.show()
else:
    print(f"The file {file_path} does not exist.")
