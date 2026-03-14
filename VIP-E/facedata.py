import numpy as np
from scipy.spatial import Delaunay, ConvexHull
import os

# =============================================================
# 1. GENERATE GRID POINTS + TETRAHEDRALIZATION
# =============================================================

# Create a 3D grid: 0, 2, 4, ..., 20 in each axis
grid_values = np.arange(0, 6, 2)

points = []
for x in grid_values:
    for y in grid_values:
        for z in grid_values:
            points.append([x, y, z])

points = np.array(points, dtype=float)

num_points = len(points)

# Tetrahedralize
tri = Delaunay(points)

# Compute centroid of each tetra
centroids = np.mean(points[tri.simplices], axis=1)

# Ratio for division points
ratio = 0.2


# =============================================================
# 2. GENERATE GREEN DIVISION POINTS
# =============================================================
red_to_green = {i: [] for i in range(num_points)}
blue_to_green = {i: [] for i in range(len(centroids))}

for simplex_idx, (simplex, centroid) in enumerate(zip(tri.simplices, centroids)):
    for vertex_index in simplex:
        vertex = points[vertex_index]
        division_point = centroid + ratio * (vertex - centroid)
        red_to_green[vertex_index].append(division_point)
        blue_to_green[simplex_idx].append(division_point)


# =============================================================
# 3. BUILD STRUCTURES (WITHOUT CENTER)
# =============================================================
positive_structures = []
for red_idx, greens in red_to_green.items():
    if len(greens) >= 4:
        positive_structures.append(np.vstack(greens))

blue_structures = []
for blue_idx, greens in blue_to_green.items():
    if len(greens) >= 4:
        blue_structures.append(np.vstack(greens))


# =============================================================
# 4. ORIENTATION FUNCTIONS
# =============================================================
def poly_centroid(poly):
    return np.mean(poly, axis=0)

def oriented_faces(poly, centroid):
    try:
        hull = ConvexHull(poly)
    except:
        hull = ConvexHull(poly, qhull_options='QJ')

    faces_out = []
    for simplex in hull.simplices:
        v0, v1, v2 = poly[simplex]
        normal = np.cross(v1 - v0, v2 - v0)
        face_centroid = (v0 + v1 + v2) / 3
        direction = face_centroid - centroid

        # left-hand alignment
        if np.dot(normal, direction) < 0:
            simplex = simplex[::-1]

        faces_out.append(tuple(simplex))
    return faces_out
# =============================================================
# EXPORT ALL STRUCTURE POINTS INTO ONE TXT FILE
# =============================================================

output_file = r"C:\Users\enkhl\OneDrive\Documents\B. SUNY Courses\Sophomore Fall 2025\vip295\seperatepolyhedrons\all_structures.txt"

def write_all_points(filename, structures):
    """Write points from all structures (red + blue) into one txt file."""
    with open(filename, "w") as f:
        f.write("# Auto-generated point catalog for all structures\n")
        f.write("# ------------------------------------------------\n\n")

        for struct_type, struct_list in structures:
            for index, poly in enumerate(struct_list):

                # Get clean vertex list
                verts = localize_structure(poly)

                # Header for each structure
                f.write(f"# Structure type: {struct_type}\n")
                f.write(f"# Structure index: {index}\n")
                f.write("points = [\n")

                # Write points
                for v in verts:
                    f.write(f"  [{v[0]:.6f}, {v[1]:.6f}, {v[2]:.6f}],\n")

                f.write("];\n\n")  # space between structures

    print(f"Saved all structures to {filename}")


# =============================================================
# Helper: extract unique local vertices for each structure
# =============================================================
def localize_structure(poly):
    centroid = poly_centroid(poly)
    faces = oriented_faces(poly, centroid)

    local_vertices = []
    local_map = {}

    def add_local(v):
        key = tuple(v.round(6))
        if key not in local_map:
            local_map[key] = len(local_vertices)
            local_vertices.append(v)
        return local_map[key]

    # Add vertices in the order used by faces
    for f in faces:
        for i in f:
            add_local(poly[i])

    return local_vertices


# =============================================================
# BUILD STRUCTURE LIST AND EXPORT
# =============================================================
structures = [
    ("RED", positive_structures),
    ("BLUE", blue_structures)
]

write_all_points(output_file, structures)
