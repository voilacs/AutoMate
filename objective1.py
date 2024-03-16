import numpy as np
from numpy.linalg import eig
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing

# Load California housing dataset
california_housing = fetch_california_housing()
# Extract latitude and longitude from the dataset
lat_long = california_housing.data[:, -2:]

# Calculate the mean of latitude and longitude to center the data
mean_lat_long = np.mean(lat_long, axis=0)

# Center the data by subtracting the mean from each data point
centered_lat_long = lat_long - mean_lat_long

# Calculate the covariance matrix of the centered data
cov_matrix = np.cov(centered_lat_long.T)

# Perform eigen decomposition on the covariance matrix to find eigenvectors and eigenvalues
eigenvalues, eigenvectors = eig(cov_matrix)

# Find the index of the maximum eigenvalue
max_eigval_index = np.argmax(eigenvalues)

# The eigenvector corresponding to the largest eigenvalue represents the direction of the efficient line
direction_vector_a = eigenvectors[:, max_eigval_index]

# Plot original latitude and longitude data points
plt.scatter(lat_long[:, 0], lat_long[:, 1], alpha=0.8, label='Houses')

# Calculate two points on the efficient line for plotting
# These points are determined by moving a certain distance along the direction vector from the mean point
point1 = mean_lat_long + direction_vector_a * 20  # Extend the line in one direction
point2 = mean_lat_long - direction_vector_a * 20  # Extend the line in the opposite direction

# Plot the efficient line
plt.plot([point1[0], point2[0]], [point1[1], point2[1]], 'r-', lw=2, label='Efficient Line')

# Add labels and title to the plot
plt.xlabel('Latitude')
plt.ylabel('Longitude')
plt.legend()
plt.title('Houses and Efficient Line')

# Display the plot
plt.show()
