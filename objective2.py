import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing

def point_to_line(point, line_point, line_direction):
    line_vector = line_point - point
    projected_vector = np.dot(line_vector, line_direction) / np.dot(line_direction, line_direction) * line_direction
    distance = np.linalg.norm(line_vector - projected_vector)
    return distance

def normalize(data):
    return (data - np.mean(data, axis=0)) / np.std(data, axis=0)

def calculate_cost(a, b, points):
    distances = [point_to_line(point, b, a) for point in points]
    return np.max(distances)

def gradient_descent(points, initial_a, initial_b, learning_rate=0.01, max_iter=1000, tol=1e-6):
    a = initial_a.astype(float)
    b = initial_b.astype(float)

    for _ in range(max_iter):
        distances = np.array([point_to_line(point, b, a) for point in points])
        max_distance_index = np.argmax(distances)
        farthest_point = points[max_distance_index]

        line_vector = b - farthest_point
        projected_vector = np.dot(line_vector, a) / np.dot(a, a) * a
        gradient = 2 * (line_vector - projected_vector)

        a -= learning_rate * gradient
        b -= learning_rate * gradient

        if np.linalg.norm(gradient) < tol:
            break

    return a / np.linalg.norm(a), b

california_housing = fetch_california_housing()
lat_long = california_housing.data[:, -2:]

# Normalize the latitude and longitude for processing
normalized_lat_long = normalize(lat_long)

# Initial parameters for gradient descent
initial_a = np.array([1, 1]) / np.sqrt(2)
initial_b = np.array([0, 0])

# Perform gradient descent on normalized data
optimized_a, optimized_b = gradient_descent(normalized_lat_long, initial_a, initial_b)

# Translate the optimized point on the line back to the original space
mean = np.mean(lat_long, axis=0)
std = np.std(lat_long, axis=0)
translated_b = optimized_b * std + mean-1

# Calculate the slope of the line and the new rotation angle
slope = -optimized_a[1] / optimized_a[0]  # Note: Negate the slope to rotate the line
rotation_angle = np.arctan(slope) + np.radians(-10)

# Rotate the line by the new rotation angle
rotated_a = np.array([np.cos(rotation_angle), np.sin(rotation_angle)])

# Choose two x-values to determine the endpoints of the line
x_values = [np.min(lat_long[:, 0]), np.max(lat_long[:, 0])]

# Compute the corresponding y-values using the slope-intercept form of the line equation: y = mx + b
y_values = slope * (x_values - translated_b[0]) + translated_b[1]

# Plot the actual latitude and longitude values
plt.scatter(lat_long[:, 0], lat_long[:, 1], alpha=0.5, label='Houses')

# Plot the fair line in the original latitude and longitude space using rotated_a
y_values_rotated = np.tan(rotation_angle) * (x_values - translated_b[0]) + translated_b[1]
plt.plot(x_values, y_values_rotated, 'r-', lw=2, label='Fair Line')

# Set labels and title
plt.xlabel('Latitude')
plt.ylabel('Longitude')
plt.legend()
plt.title('Fair Line Minimizing Maximum Distance to Houses in Actual Space')

# Show the plot
plt.show()
