import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing

# Function to calculate the perpendicular distance from a point to a line
def point_to_line(point, line_point, line_direction):
    # Compute vector from point on the line to the point in question
    line_vector = line_point - point
    # Project this vector onto the line direction to find the closest point on the line
    projected_vector = np.dot(line_vector, line_direction) / np.dot(line_direction, line_direction) * line_direction
    # The distance is the norm of the vector from the point to its projection on the line
    distance = np.linalg.norm(line_vector - projected_vector)
    return distance

# Normalize the data for better numerical stability and performance in gradient descent
def normalize(data):
    return (data - np.mean(data, axis=0)) / np.std(data, axis=0)

# Calculate the cost function, which is the maximum distance of any point to the line
def calculate_cost(a, b, points):
    distances = [point_to_line(point, b, a) for point in points]
    return np.max(distances)

# Perform gradient descent to minimize the cost function
def gradient_descent(points, initial_a, initial_b, learning_rate=0.01, max_iter=1000, tol=1e-6):
    # Initialize the direction vector and a point on the line
    a = initial_a.astype(float)
    b = initial_b.astype(float)

    for _ in range(max_iter):
        # Compute distances from all points to the line and find the farthest point
        distances = np.array([point_to_line(point, b, a) for point in points])
        max_distance_index = np.argmax(distances)
        farthest_point = points[max_distance_index]

        # Calculate gradient based on the farthest point
        line_vector = b - farthest_point
        projected_vector = np.dot(line_vector, a) / np.dot(a, a) * a
        gradient = 2 * (line_vector - projected_vector)

        # Update the line parameters using the gradient
        a -= learning_rate * gradient
        b -= learning_rate * gradient

        # Stop if the change is smaller than the tolerance
        if np.linalg.norm(gradient) < tol:
            break

    # Return the normalized direction vector and the updated point on the line
    return a / np.linalg.norm(a), b

# Load the California housing dataset and extract latitude and longitude
california_housing = fetch_california_housing()
lat_long = california_housing.data[:, -2:]

# Normalize latitude and longitude
normalized_lat_long = normalize(lat_long)

# Set initial values for the direction vector and point on the line
initial_a = np.array([1, 1]) / np.sqrt(2)
initial_b = np.array([0, 0])

# Optimize the line parameters using gradient descent
optimized_a, optimized_b = gradient_descent(normalized_lat_long, initial_a, initial_b)

# Translate the optimized point on the line back to the original space of latitude and longitude
mean = np.mean(lat_long, axis=0)
std = np.std(lat_long, axis=0)
translated_b = optimized_b * std + mean

# Plot the actual latitude and longitude data
plt.scatter(lat_long[:, 0], lat_long[:, 1], alpha=0.5, label='Houses')

# Plot the fair line in the original space
plt.plot([np.min(lat_long[:, 0]), np.max(lat_long[:, 0])], y_values, 'r-', lw=2, label='Fair Line')

# Add labels and a legend to the plot
plt.xlabel('Latitude')
plt.ylabel('Longitude')
plt.legend()
plt.title('Fair Line Minimizing Maximum Distance to Houses in Actual Space')

# Display the plot
plt.show()
