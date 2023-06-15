import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform

category_data = pd.read_csv("./data/others/category_one_hot.csv")

correlations = category_data.corr()

# Convert correlation matrix to distance matrix
distance_matrix = 1 - np.abs(correlations)

# Convert distance matrix to condensed form
condensed_distance = squareform(distance_matrix)
print(condensed_distance)
# Perform hierarchical clustering
Z = linkage(condensed_distance, method='average')

# Define the number of clusters
num_clusters = 10

# Assign categories to clusters
labels = fcluster(Z, num_clusters, criterion='maxclust')

print(labels)
categories = "Drama	Action	Sci-Fi	Comedy	Adventure	Fantasy	Mystery	Psychological	Ecchi	Josei	Military	Romance	Demons	Dementia	Music	Game	Cars	Mecha	Horror	School	Historical	Kids	Shounen	Shoujo	Magic	Harem	Martial Arts	Sports	Slice of Life	Seinen	Parody	Police	Thriller	Supernatural	Samurai	Super Power	Vampire	Shoujo Ai	Shounen Ai	Space"
print(len(categories.split(sep='\t')))
df = pd.DataFrame({'Category': categories.split(sep='\t'), 'Cluster': labels})

df.to_csv("clustered.csv")

# top_correlations = {}
# for column in correlations.columns:
#     # Sort correlation values in descending order (excluding self-correlation)
#     sorted_correlations = correlations[column].drop(column).sort_values(ascending=False)
#     # Select top 5 correlated columns
#     top_correlations[column] = sorted_correlations[:5].index.tolist()

# # Print the top 5 correlated columns for each column
# for column, correlated_columns in top_correlations.items():
#     print(f"Top 5 correlated genre for {column}: {correlated_columns}")