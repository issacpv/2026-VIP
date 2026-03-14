import numpy as np
from scipy.spatial import ConvexHull

def orient_clockwise(points, face):
    """Ensure triangular face vertices are clockwise when viewed from outside."""
    p = np.array(points)
    a, b, c = p[face]
    normal = np.cross(b - a, c - a)
    centroid = np.mean(p, axis=0)

    # Flip if the normal points inward
    if np.dot(normal, centroid - a) > 0:
        face = face[::-1]

    return face.tolist()


def generate_faces(points):
    hull = ConvexHull(points)
    oriented_faces = []
    for face in hull.simplices:
        oriented_faces.append(orient_clockwise(points, face))
    return oriented_faces


def print_openscad_block(points, faces):
    print("polyhedron(")
    print("  points = [")
    for p in points:
        print(f"    [{p[0]}, {p[1]}, {p[2]}],")
    print("  ],")
    print("  faces = [")
    for f in faces:
        print(f"    [{f[0]}, {f[1]}, {f[2]}],")
    print("  ]")
    print(");\n")


# ============================================================
# INPUT: EXACTLY 5 POLYHEDRON DATASETS
# ============================================================

poly_sets = [
    # Polyhedron 0
    [
  [0.800000, 0.000000, 0.800000],
  [0.800000, 0.400000, 0.400000],
  [0.000000, 0.800000, 0.800000],
  [0.400000, 0.800000, 0.400000],
    ],

    # Polyhedron 1
    [
  [0.400000, 1.200000, 1.200000],
  [0.400000, 0.800000, 0.800000],
  [0.000000, 0.800000, 1.200000],
  [0.400000, 0.400000, 2.800000],
  [0.800000, 1.200000, 1.600000],
  [0.800000, 0.000000, 1.200000],
  [0.800000, 0.400000, 0.800000],
  [1.200000, 0.400000, 1.200000],
  [0.800000, 0.000000, 2.800000],
  [1.200000, 0.800000, 1.600000],
    ],

    # Polyhedron 2
    [
  [0.400000, 0.800000, 3.600000],
  [0.800000, 0.000000, 3.200000],
  [0.800000, 0.800000, 4.000000],
  [0.400000, 0.400000, 3.200000],
    ],

    # Polyhedron 3
    [
  [0.400000, 2.800000, 0.400000],
  [0.400000, 1.200000, 0.400000],
  [0.000000, 1.200000, 0.800000],
  [0.400000, 1.600000, 0.800000],
  [0.400000, 2.400000, 0.800000],
    ],

    # Polyhedron 4
    [
  [1.200000, 1.200000, 2.400000],
  [0.800000, 2.400000, 1.600000],
  [1.200000, 1.600000, 2.800000],
  [0.800000, 2.400000, 2.400000],
  [0.800000, 0.800000, 2.400000],
  [0.400000, 0.800000, 2.800000],
  [0.000000, 1.200000, 1.200000],
  [0.400000, 1.200000, 3.200000],
  [0.400000, 2.400000, 2.800000],
  [0.800000, 1.600000, 3.200000],
  [0.400000, 2.400000, 1.200000],
  [0.400000, 1.600000, 1.200000],
  [0.800000, 1.600000, 1.600000],
    ]
]


# ============================================================
# PROCESS AND OUTPUT ALL 5 POLYHEDRONS IN OPESCAD FORMAT
# ============================================================

for pts in poly_sets:
    faces = generate_faces(pts)
    print_openscad_block(pts, faces)
