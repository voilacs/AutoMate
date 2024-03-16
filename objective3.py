import numpy as np
from numpy.linalg import eig
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import fetch_california_housing

# Load the California housing dataset
california_housing = fetch_california_housing()
# Extract latitude and longitude data
lat_long = california_housing.data[:, -2:]

# Specify the number of clusters (k) for KMeans clustering
k = 5

# Initialize KMeans with the desired number of clusters and fit it to the data
kmeans = KMeans(n_clusters=k)
kmeans.fit(lat_long)

# Extract the centroids of the clusters and the labels for each data point
centroids = kmeans.cluster_centers_
labels = kmeans.labels_

# Initialize a list to hold data points for each cluster
clusters = [[] for _ in range(k)]

# Assign each data point to its corresponding cluster
for i, label in enumerate(labels):
    clusters[label].append(lat_long[i])

# Convert cluster lists to numpy arrays for easier manipulation
clusters = [np.array(cluster) for cluster in clusters]

# Lists to store the direction vectors (a) and points on the line (b) for each cluster
a_list = []
b_list = []

# Iterate over each cluster to find the efficient line
for index, cluster in enumerate(clusters):
    # Calculate the mean of the cluster's points to center the data
    mean_lat_long = np.mean(cluster, axis=0)
    centered_lat_long = cluster - mean_lat_long
    
    # Compute the covariance matrix of the centered data
    cov_matrix = np.cov(centered_lat_long.T)
    
    # Perform eigen decomposition to find the eigenvectors and eigenvalues
    eigenvalues, eigenvectors = eig(cov_matrix)
    
    # The eigenvector corresponding to the largest eigenvalue is the direction vector for the efficient line
    max_eigval_index = np.argmax(eigenvalues)
    direction_vector_a = eigenvectors[:, max_eigval_index]
    
    # Store the direction vector and the mean point for each cluster
    a_list.append(direction_vector_a)
    b_list.append(mean_lat_long)

    # Plot the points of the cluster
    plt.scatter(cluster[:, 0], cluster[:, 1], alpha=0.5, label=f'Cluster {index+1}')

    # Calculate and plot the efficient line for the cluster
    point1 = mean_lat_long + direction_vector_a * 20  # Extend the line in one direction
    point2 = mean_lat_long - direction_vector_a * 20  # Extend the line in the opposite direction
    plt.plot([point1[0], point2[0]], [point1[1], point2[1]], '-', lw=2, label=f'Line {index+1}')

# Set the plot limits to focus on the region of interest
plt.ylim(-125, -114)
plt.xlim(32, 42)

# Add a legend and title to the plot
plt.legend()
plt.title('Houses and Efficient Lines per Cluster')

# Display the plot
plt.show()
