import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import Delaunay
from scipy.spatial import ConvexHull

# Generate 10 random points
points = np.random.rand(5, 2)

# Delaunay triangulation
tri = Delaunay(points)

# Create figure
fig, ax = plt.subplots()
fig.patch.set_facecolor("black")
ax.set_facecolor("black")

ratio = 0.4  # distance ratio from centroid to vertex
dot_size = 10  # smaller dot size

# Dictionaries: white point -> cyan dots, red point -> cyan dots
groups_white = {tuple(p): [] for p in points}
groups_red = {}

# Process each triangle
for simplex in tri.simplices:
    triangle = points[simplex]
    centroid = triangle.mean(axis=0)
    centroid_key = tuple(np.round(centroid, 6))  # safe hashable key
    ax.scatter(*centroid, color='red', s=dot_size)

    if centroid_key not in groups_red:
        groups_red[centroid_key] = []

    cyan_points = []
    for vertex in triangle:
        # Pick point on the line at given ratio
        point_on_line = (1 - ratio) * centroid + ratio * vertex
        ax.scatter(*point_on_line, color='cyan', s=dot_size)

        # Add this cyan dot to both its groups
        groups_white[tuple(vertex)].append(point_on_line)
        groups_red[centroid_key].append(point_on_line)

        cyan_points.append(point_on_line)

    # ---- Fill polygon formed by cyan points around red centroid ----
    cyan_points = np.array(cyan_points)
    ax.fill(cyan_points[:, 0], cyan_points[:, 1], color="cyan", alpha=0.6)

# ---- Connect & fill cyan dots that share the same white dot ----
for white, cyan_list in groups_white.items():
    if len(cyan_list) > 1:
        cyan_array = np.array(cyan_list)
        for i in range(len(cyan_array)):
            for j in range(i + 1, len(cyan_array)):
                ax.plot([cyan_array[i, 0], cyan_array[j, 0]],
                        [cyan_array[i, 1], cyan_array[j, 1]],
                        color='magenta', linewidth=1.5, alpha=0.8)

        # ---- Fill polygon around white dot ----
        if len(cyan_array) >= 3:
            hull = ConvexHull(cyan_array)
            hull_points = cyan_array[hull.vertices]
            ax.fill(hull_points[:, 0], hull_points[:, 1], color="magenta", alpha=0.6)

# ---- Connect cyan dots that share the same red dot ----
for red, cyan_list in groups_red.items():
    if len(cyan_list) > 1:
        cyan_array = np.array(cyan_list)
        for i in range(len(cyan_array)):
            for j in range(i + 1, len(cyan_array)):
                ax.plot([cyan_array[i, 0], cyan_array[j, 0]],
                        [cyan_array[i, 1], cyan_array[j, 1]],
                        color='cyan', linewidth=1.5, alpha=0.8)

# Optional: plot original white points smaller too
ax.scatter(points[:, 0], points[:, 1], color="white", s=dot_size)

# Remove axes
ax.set_xticks([])
ax.set_yticks([])
ax.axis("off")

plt.show()