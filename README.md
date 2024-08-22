Clustering Analysis Project
Description
This project involves clustering analysis on two datasets: a call center dataset and a mall customer dataset. The primary goal is to use K-Means clustering to identify distinct groups within the data.

Datasets:
Call Center Data: Contains information on experience in months and the number of calls attended per day.
Mall Customer Data: Includes customer demographics and spending scores.
Installation
Ensure you have Python installed, and then install the required packages using pip. You can use the following command to install the necessary libraries:

bash
Copy code
pip install pandas numpy matplotlib seaborn scikit-learn
Usage
Call Center Data Analysis
Load Data: The dataset is loaded from a URL and includes experience in months and calls attended per day.
Apply K-Means Clustering:
Use the elbow method (knee rule) to determine the optimal number of clusters.
Fit the K-Means model with the chosen number of clusters.
Visualize Results:
Scatter plot of the data points colored by their cluster labels.
Mall Customer Data Analysis
Load Data: The dataset includes customer demographics and spending scores.
Pairwise Plot:
Visualize pairwise relationships in the dataset.
Apply K-Means Clustering:
Determine the optimal number of clusters using the elbow method.
Fit the K-Means model and visualize the clusters with pairwise plots.
Advanced Visualization:
Scatter plot with varying sizes based on age.
3D scatter plot using age, annual income, and spending score.
Code Example
Call Center Data
python
Copy code
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans

# Load data
df = pd.read_csv('https://raw.githubusercontent.com/bipulshahi/Dataset/main/Call%20Center%20Data.csv')

# Determine optimal number of clusters using the elbow method
losses = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i)
    kmeans.fit(df)
    losses.append(kmeans.inertia_)
plt.plot(range(1, 11), losses, marker='*')
plt.show()

# Fit K-Means with 4 clusters
model1 = KMeans(n_clusters=4)
model1.fit(df)
df['Labels'] = model1.predict(df)

# Visualize clusters
plt.scatter(df['Experience in months'], df['Call Attended in a day'], c=df['Labels'])
plt.show()
Mall Customer Data
python
Copy code
# Load data
df_mall = pd.read_csv('https://raw.githubusercontent.com/bipulshahi/Dataset/main/mall.csv')
dfm = df_mall.loc[:, 'Age':'Spending Score (1-100)']

# Pairwise plot
sns.pairplot(dfm)
plt.show()

# Determine optimal number of clusters using the elbow method
dfm2 = dfm.copy()
losses = []
for i in range(1, 11):
    model2 = KMeans(n_clusters=i)
    model2.fit(dfm2)
    losses.append(model2.inertia_)
plt.plot(range(1, 11), losses, marker='*')
plt.show()

# Fit K-Means with 6 clusters
model3 = KMeans(n_clusters=6)
model3.fit(dfm2)
dfm2['Labels'] = model3.predict(dfm2)

# Visualize clusters
sns.pairplot(dfm2, hue='Labels')
plt.show()

# 3D Scatter Plot
from mpl_toolkits.mplot3d import Axes3D
plt.figure(figsize=(10, 8))
ax = plt.figure().add_subplot(projection='3d')
ax.scatter(dfm2['Annual Income (k$)'], dfm2['Spending Score (1-100)'], zs=dfm2['Age'], zdir='z', c=dfm2['Labels'])
ax.set_xlabel('Annual Income (k$)')
ax.set_ylabel('Spending Score (1-100)')
ax.set_zlabel('Age')
plt.show()
Acknowledgements
Data sources: Call Center Data, Mall Data
Libraries used: Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn
