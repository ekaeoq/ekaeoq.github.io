import matplotlib.pyplot as plt
import numpy as np

# Create a 5x5 grid (all black initially)
grid = np.zeros((5, 5))

# Define points (x, y, z)
points = [(1, 1, 0), (2, 2, 0.2), (3, 3, 0), (2, 3, 0.1)]
min_z, max_z = 0, 0.2

# Map points to the grid
for x, y, z in points:
    normalized_z = (z - min_z) / (max_z - min_z)  # Normalize z to 0-1
    grid[y, x] = normalized_z * 255  # Set pixel value (0-255)

# Display the grayscale image
plt.imshow(grid, cmap='gray', vmin=0, vmax=255)
plt.title("2D Grayscale Height Map")
plt.xlabel("X")
plt.ylabel("Y")
plt.colorbar(label="Grayscale Intensity (Height)")
plt.show()
