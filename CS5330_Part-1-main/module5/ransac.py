import cv2
import numpy as np
import matplotlib.pyplot as plt

# Step 1: Generate some sample data
np.random.seed(0)
# Generate 50 random points
n_samples = 50
x = np.random.rand(n_samples, 1) * 100
y = 2.5 * x + np.random.randn(n_samples, 1) * 10

# Add some outliers
n_outliers = 10
x_outliers = np.random.rand(n_outliers, 1) * 100
y_outliers = np.random.rand(n_outliers, 1) * 200

x = np.vstack((x, x_outliers))
y = np.vstack((y, y_outliers))

# Combine x and y into a single array of points
points = np.hstack((x, y))

# Step 2: Use RANSAC to fit a line
# Convert points to the shape required by cv2.fitLine
points = points.astype(np.float32)

# Apply RANSAC
line = cv2.fitLine(points, cv2.DIST_L2, 0, 0.99, 0.01)

# Extract the line parameters
vx, vy, x0, y0 = line[0], line[1], line[2], line[3]
slope = vy / vx
intercept = y0 - slope * x0

# Step 3: Plot the results
plt.figure(figsize=(10, 10))
plt.scatter(x, y, color='red', label='Data points')
plt.scatter(x_outliers, y_outliers, color='blue', label='Outliers')

# Plot the RANSAC fitted line
x_fit = np.array([0, 100])
y_fit = slope * x_fit + intercept
plt.plot(x_fit, y_fit, color='green', linewidth=2, label='RANSAC fit')

plt.legend()
plt.title('RANSAC Line Fitting')
plt.xlabel('x')
plt.ylabel('y')
plt.show()
