import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import Delaunay, ConvexHull
from matplotlib.widgets import Slider

# ==========================
# USER SETTINGS
# ==========================
n_points = 9   # Total number of white points
mode = 1       # 0 = grid, 1 = random
ratio_init = 0.4     # initial distance ratio from centroid to vertex
dot_size = 10        # smaller dot size

# ==========================
# Generate points
# ==========================
if mode == 0:
    grid_size = int(np.ceil(np.sqrt(n_points)))
    x = np.linspace(0, 1, grid_size)
    y = np.linspace(0, 1, grid_size)
    xx, yy = np.meshgrid(x, y)
    points = np.vstack([xx.ravel(), yy.ravel()]).T
    points = points[:n_points]
else:
    points = np.random.rand(n_points, 2)

# ==========================
# Delaunay triangulation
# ==========================
tri = Delaunay(points)

# ==========================
# Plot setup
# ==========================
fig, ax = plt.subplots()
plt.subplots_adjust(bottom=0.25)  # space for slider
fig.patch.set_facecolor("black")
ax.set_facecolor("black")

slider_ax = plt.axes([0.2, 0.1, 0.6, 0.03], facecolor='gray')
ratio_slider = Slider(slider_ax, 'Ratio', 0.0, 1.0, valinit=ratio_init)

# ==========================
# Drawing function
# ==========================
def draw_plot(ratio):
    ax.clear()
    ax.set_facecolor("black")

    groups_white = {tuple(p): [] for p in points}
    groups_red = {}

    # Process triangles
    for simplex in tri.simplices:
        triangle = points[simplex]
        centroid = triangle.mean(axis=0)
        centroid_key = tuple(np.round(centroid, 6))
        ax.scatter(*centroid, color='red', s=dot_size)

        if centroid_key not in groups_red:
            groups_red[centroid_key] = []

        cyan_points = []
        for vertex in triangle:
            point_on_line = (1 - ratio) * centroid + ratio * vertex
            ax.scatter(*point_on_line, color='cyan', s=dot_size)

            groups_white[tuple(vertex)].append(point_on_line)
            groups_red[centroid_key].append(point_on_line)

            cyan_points.append(point_on_line)

        cyan_points = np.array(cyan_points)
        ax.fill(cyan_points[:, 0], cyan_points[:, 1], color="cyan", alpha=0.6)

    # Connect cyan dots sharing same white dot
    for white, cyan_list in groups_white.items():
        if len(cyan_list) > 1:
            cyan_array = np.array(cyan_list)
            for i in range(len(cyan_array)):
                for j in range(i + 1, len(cyan_array)):
                    ax.plot([cyan_array[i, 0], cyan_array[j, 0]],
                            [cyan_array[i, 1], cyan_array[j, 1]],
                            color='magenta', linewidth=1.5, alpha=0.8)
            if len(cyan_array) >= 3:
                hull = ConvexHull(cyan_array)
                hull_points = cyan_array[hull.vertices]
                ax.fill(hull_points[:, 0], hull_points[:, 1], color="magenta", alpha=0.6)

    # Connect cyan dots sharing same red dot
    for red, cyan_list in groups_red.items():
        if len(cyan_list) > 1:
            cyan_array = np.array(cyan_list)
            for i in range(len(cyan_array)):
                for j in range(i + 1, len(cyan_array)):
                    ax.plot([cyan_array[i, 0], cyan_array[j, 0]],
                            [cyan_array[i, 1], cyan_array[j, 1]],
                            color='cyan', linewidth=1.5, alpha=0.8)

    # Plot original white points
    ax.scatter(points[:, 0], points[:, 1], color="white", s=dot_size)

    # Remove axes
    ax.set_xticks([])
    ax.set_yticks([])
    ax.axis("off")
    fig.canvas.draw_idle()

# Initial draw
draw_plot(ratio_init)

# ==========================
# Slider update
# ==========================
def update(val):
    draw_plot(ratio_slider.val)

ratio_slider.on_changed(update)

plt.show()
