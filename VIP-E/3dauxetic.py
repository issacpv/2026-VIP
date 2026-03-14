import numpy as np
from scipy.spatial import Delaunay, ConvexHull
import plotly.graph_objects as go
import trimesh
import os


cube_size = 20
num_points = 2            # random interior points
ratio = 0.5               # green division points: centroid + ratio*(vertex-centroid)
sphere_radius = 0.5       # <<< adjust sphere radius here
n_edge_divs = 2           # boundary support points per edge (includes endpoints)
sample_spheres_max = 200  # cap spheres for performance; set None for all


points = np.random.rand(num_points, 3) * cube_size

cube_corners = np.array([
    [0, 0, 0],
    [cube_size, 0, 0],
    [0, cube_size, 0],
    [0, 0, cube_size],
    [cube_size, cube_size, 0],
    [cube_size, 0, cube_size],
    [0, cube_size, cube_size],
    [cube_size, cube_size, cube_size]
])

C = {
    "000": 0, "x00": 1, "0y0": 2, "00z": 3,
    "xy0": 4, "x0z": 5, "0yz": 6, "xyz": 7
}

edge_pairs = [
    (C["000"], C["x00"]), (C["x00"], C["xy0"]),
    (C["xy0"], C["0y0"]), (C["0y0"], C["000"]),  # bottom
    (C["00z"], C["x0z"]), (C["x0z"], C["xyz"]),
    (C["xyz"], C["0yz"]), (C["0yz"], C["00z"]),  # top
    (C["000"], C["00z"]), (C["x00"], C["x0z"]),
    (C["xy0"], C["xyz"]), (C["0y0"], C["0yz"])   # verticals
]

edge_midpoints = np.array([
    (cube_corners[a] + cube_corners[b]) / 2 for a, b in edge_pairs
])


edge_support = []
t_vals = np.linspace(0.0, 1.0, n_edge_divs)
for a, b in edge_pairs:
    A, B = cube_corners[a], cube_corners[b]
    for t in t_vals:
        p = (1 - t) * A + t * B
        edge_support.append(p)
edge_support = np.array(edge_support)

# Deduplicate (to avoid duplicates with corners/midpoints)
def unique_rows(arr, tol=1e-12):
    r = np.round(arr / tol) * tol
    _, idx = np.unique(r, axis=0, return_index=True)
    return arr[np.sort(idx)]

edge_support = unique_rows(edge_support)


all_points = np.vstack([points, cube_corners, edge_midpoints, edge_support])
all_points = unique_rows(all_points)

tri = Delaunay(all_points)

centroids = np.mean(all_points[tri.simplices], axis=1)
red_to_green = {i: [] for i in range(len(all_points))}
blue_to_green = {i: [] for i in range(len(centroids))}
connection_points = []

for simplex_idx, (simplex, centroid) in enumerate(zip(tri.simplices, centroids)):
    for vertex_index in simplex:
        vertex = all_points[vertex_index]
        division_point = centroid + ratio * (vertex - centroid)
        red_to_green[vertex_index].append(division_point)
        blue_to_green[simplex_idx].append(division_point)
        connection_points.append(division_point)

connection_points = np.array(connection_points)

def hull_faces(segment_points):
    if segment_points is None or len(segment_points) < 4:
        return [], []
    try:
        hull = ConvexHull(segment_points)
    except Exception:
        hull = ConvexHull(segment_points, qhull_options='QJ')
    return hull.simplices, segment_points

x_faces, y_faces, z_faces = [], [], []

# Build faces from each red vertex + its green points
for red_idx, greens in red_to_green.items():
    seg = np.vstack([all_points[red_idx]] + greens) if len(greens) else None
    simplices, seg_pts = hull_faces(seg)
    for simplex in simplices:
        pts = seg_pts[simplex]
        x_faces.append(pts[:, 0])
        y_faces.append(pts[:, 1])
        z_faces.append(pts[:, 2])

# Build faces from each blue centroid + its green points
for blue_idx, greens in blue_to_green.items():
    seg = np.vstack([centroids[blue_idx]] + greens) if len(greens) else None
    simplices, seg_pts = hull_faces(seg)
    for simplex in simplices:
        pts = seg_pts[simplex]
        x_faces.append(pts[:, 0])
        y_faces.append(pts[:, 1])
        z_faces.append(pts[:, 2])

cube_edges_poly = np.array([
    [0, 0, 0],
    [cube_size, 0, 0],
    [cube_size, cube_size, 0],
    [0, cube_size, 0],
    [0, 0, 0],
    [0, 0, cube_size],
    [cube_size, 0, cube_size],
    [cube_size, 0, 0],
    [cube_size, cube_size, cube_size],
    [cube_size, cube_size, 0],
    [0, cube_size, cube_size],
    [0, cube_size, 0],
    [cube_size, cube_size, cube_size],
    [cube_size, 0, cube_size],
    [0, cube_size, cube_size],
    [0, 0, cube_size]
])

fig = go.Figure()

# Cube wireframe
fig.add_trace(go.Scatter3d(
    x=cube_edges_poly[:, 0], y=cube_edges_poly[:, 1], z=cube_edges_poly[:, 2],
    mode='lines', line=dict(color='black', width=2), name='Cube Boundary'
))

# Red points (all input points)
fig.add_trace(go.Scatter3d(
    x=all_points[:, 0], y=all_points[:, 1], z=all_points[:, 2],
    mode='markers', marker=dict(size=3, color='red'),
    name='Points (random + corners + midpoints + edge support)'
))

# Blue centroids
fig.add_trace(go.Scatter3d(
    x=centroids[:, 0], y=centroids[:, 1], z=centroids[:, 2],
    mode='markers', marker=dict(size=3, color='blue'), name='Centroids'
))

# Translucent green faces
for x, y, z in zip(x_faces, y_faces, z_faces):
    fig.add_trace(go.Mesh3d(
        x=x, y=y, z=z, color='lightgreen', opacity=0.35, showscale=False
    ))

theta = np.linspace(0, 2 * np.pi, 15)
phi = np.linspace(0, np.pi, 10)
theta, phi = np.meshgrid(theta, phi)
ux = np.sin(phi) * np.cos(theta)
uy = np.sin(phi) * np.sin(theta)
uz = np.cos(phi)

if sample_spheres_max is None or len(connection_points) <= sample_spheres_max:
    sample = connection_points
else:
    step = max(1, len(connection_points) // sample_spheres_max)
    sample = connection_points[::step]

for cx, cy, cz in sample:
    x = cx + sphere_radius * ux
    y = cy + sphere_radius * uy
    z = cz + sphere_radius * uz
    fig.add_trace(go.Mesh3d(
        x=x.ravel(), y=y.ravel(), z=z.ravel(),
        alphahull=0, color='lightgreen', opacity=0.35, showscale=False
    ))

fig.update_layout(
    scene=dict(
        xaxis=dict(range=[0, cube_size], title='X'),
        yaxis=dict(range=[0, cube_size], title='Y'),
        zaxis=dict(range=[0, cube_size], title='Z'),
        aspectmode='cube'
    ),
    title="3D Structure (Corners + Edge Midpoints + Outer-Edge Support) with Transparent Spheres",
    width=1000, height=800
)
fig.show()


triangles = []
for x, y, z in zip(x_faces, y_faces, z_faces):
    pts = np.column_stack([x, y, z])
    if len(pts) < 3:
        continue
    for i in range(1, len(pts) - 1):
        tri = [pts[0], pts[i], pts[i + 1]]
        triangles.append(tri)

# Also add spheres if desired
add_spheres = True
if add_spheres:
    for cx, cy, cz in sample:
        sphere = trimesh.creation.icosphere(subdivisions=2, radius=sphere_radius)
        sphere.apply_translation([cx, cy, cz])
        triangles.extend(np.array(sphere.triangles))

# Convert to Trimesh object
mesh = trimesh.Trimesh(
    vertices=np.vstack(triangles).reshape(-1, 3),
    faces=np.arange(len(triangles) * 3).reshape(-1, 3),
    process=True
)

# Export to STL
output_dir = os.path.dirname(os.path.abspath(__file__))
output_path = os.path.join(output_dir, "structure.stl")
mesh.export(output_path)
print(f"STL file saved at: {output_path}")
