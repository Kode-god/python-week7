import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris

# 1 Load and Explore the Dataset
# Load the Iris dataset
iris = load_iris()
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
df['species'] = iris.target
df['species'] = df['species'].map({0: 'setosa', 1: 'versicolor', 2: 'virginica'})

# Display the first few rows
print("First few rows of the dataset:")
print(df.head())

# Explore the structure of the dataset
print("\nDataset info:")
print(df.info())

# Check for missing values
print("\nMissing values:")
print(df.isnull().sum())

# Clean the dataset and drop missing values if any
df.dropna(inplace=True) 

# 2 Basic Data Analysis
# Compute basic statistics
print("\nBasic statistics:")
print(df.describe())

# Group by species and compute mean of numerical columns
print("\nMean of numerical columns grouped by species:")
print(df.groupby('species').mean())

# 3 Data Visualization
sns.set(style="whitegrid")

# 1. Bar chart: Average sepal length per species
plt.figure(figsize=(8, 5))
df.groupby('species')['sepal length (cm)'].mean().plot(kind='bar', color='skyblue')
plt.title('Average Sepal Length per Species')
plt.xlabel('Species')
plt.ylabel('Average Sepal Length (cm)')
plt.show()

# 2. Histogram: Distribution of petal length
plt.figure(figsize=(8, 5))
sns.histplot(df['petal length (cm)'], bins=20, kde=True, color='orange')
plt.title('Distribution of Petal Length')
plt.xlabel('Petal Length (cm)')
plt.ylabel('Frequency')
plt.show()

# 3. Scatter plot: Sepal length vs. petal length
plt.figure(figsize=(8, 5))
sns.scatterplot(x='sepal length (cm)', y='petal length (cm)', hue='species', data=df, palette='viridis')
plt.title('Sepal Length vs. Petal Length')
plt.xlabel('Sepal Length (cm)')
plt.ylabel('Petal Length (cm)')
plt.legend(title='Species')
plt.show()

# Findings and Observations
print("\nFindings and Observations:")
print("1. Setosa has the smallest average sepal and petal lengths, while virginica has the largest.")
print("2. The scatter plot shows a clear separation between species based on sepal and petal lengths.")