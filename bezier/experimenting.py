import numpy as np
import matplotlib.pyplot as plt
from itertools import product

# ============================================================
# 1) Basic vector helpers
# ============================================================
def v(x, y, z):
    return np.array([x, y, z], dtype=float)

def unit(x):
    x = np.asarray(x, dtype=float)
    n = np.linalg.norm(x)
    if n < 1e-12:
        return x * 0.0
    return x / n

# ============================================================
# 2) Cubic Bezier utilities
# ============================================================
def bezier3(P0, P1, P2, P3, t):
    t = np.asarray(t, dtype=float)
    if t.ndim == 0:
        t = np.array([t], dtype=float)

    u = 1.0 - t
    return (
        (u**3)[:, None] * P0
        + (3 * u**2 * t)[:, None] * P1
        + (3 * u * t**2)[:, None] * P2
        + (t**3)[:, None] * P3
    )

def sample_bezier3(P0, P1, P2, P3, n=80):
    t = np.linspace(0.0, 1.0, n)
    return bezier3(P0, P1, P2, P3, t)

def bezier_strut(P_start, P_end, bend_dir, bend=0.12):
    """
    Build a curved cubic Bezier member between P_start and P_end.
    bend_dir is projected to be perpendicular to the member direction.
    """
    P_start = np.asarray(P_start, float)
    P_end = np.asarray(P_end, float)

    d = P_end - P_start
    L = np.linalg.norm(d)
    if L < 1e-12:
        return P_start, P_start, P_end, P_end

    d_hat = d / L
    bd = unit(bend_dir)

    # Remove any component of bend_dir parallel to member direction
    bd = bd - np.dot(bd, d_hat) * d_hat
    bd = unit(bd)

    # fallback if bend_dir accidentally parallel to strut
    if np.linalg.norm(bd) < 1e-12:
        trial = np.cross(d_hat, v(0, 0, 1))
        if np.linalg.norm(trial) < 1e-12:
            trial = np.cross(d_hat, v(0, 1, 0))
        bd = unit(trial)

    P0 = P_start
    P3 = P_end
    offset = bend * L * bd
    P1 = P0 + d / 3.0 + offset
    P2 = P0 + 2.0 * d / 3.0 + offset
    return P0, P1, P2, P3

# ============================================================
# 3) Geometry: skewed 3D "parallelogram cell" (parallelepiped)
# ============================================================
def parallelepiped_vertices(origin, a, b, c):
    """
    8 vertices of a skewed 3D cell.
    """
    O = np.asarray(origin, float)
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    c = np.asarray(c, float)

    verts = {
        "000": O,
        "100": O + a,
        "010": O + b,
        "110": O + a + b,
        "001": O + c,
        "101": O + a + c,
        "011": O + b + c,
        "111": O + a + b + c,
    }
    return verts

def edge_midpoint(P, Q):
    return 0.5 * (P + Q)

def face_center(P, Q, R, S):
    return 0.25 * (P + Q + R + S)

def cell_center(verts):
    P = np.stack(list(verts.values()), axis=0)
    return P.mean(axis=0)

# ============================================================
# 4) Build one median-connected 3D parallelogram cell
# ============================================================
def build_auxetic_cell(origin, a, b, c):
    """
    Returns:
      nodes: dict[name] = point
      members: list of (start_name, end_name, kind)
    Geometry:
      - outer cell edges
      - face centers
      - edge midpoints
      - median connections on each face
      - center-to-face-center links
    """
    verts = parallelepiped_vertices(origin, a, b, c)
    nodes = dict(verts)

    # --- edge list of the parallelepiped ---
    edges = [
        ("000", "100"), ("000", "010"), ("000", "001"),
        ("100", "110"), ("100", "101"),
        ("010", "110"), ("010", "011"),
        ("001", "101"), ("001", "011"),
        ("110", "111"),
        ("101", "111"),
        ("011", "111"),
    ]

    # create midpoint nodes for every edge
    for i, (u, w) in enumerate(edges):
        nodes[f"M{i}"] = edge_midpoint(nodes[u], nodes[w])

    # face definitions: each face is a parallelogram
    faces = {
        "F_bottom": ("000", "100", "110", "010"),
        "F_top":    ("001", "101", "111", "011"),
        "F_x0":     ("000", "010", "011", "001"),
        "F_x1":     ("100", "110", "111", "101"),
        "F_y0":     ("000", "100", "101", "001"),
        "F_y1":     ("010", "110", "111", "011"),
    }

    for fname, (p, q, r, s) in faces.items():
        nodes[fname] = face_center(nodes[p], nodes[q], nodes[r], nodes[s])

    nodes["C"] = cell_center(verts)

    members = []

    # outer frame edges
    for e in edges:
        members.append((e[0], e[1], "frame"))

    # map edges to midpoint ids
    edge_to_mid = {}
    for i, (u, w) in enumerate(edges):
        edge_to_mid[tuple(sorted((u, w)))] = f"M{i}"

    def get_mid(u, w):
        return edge_to_mid[tuple(sorted((u, w)))]

    # median connections on each face:
    # connect face center to edge midpoints
    for fname, (p, q, r, s) in faces.items():
        mids = [
            get_mid(p, q),
            get_mid(q, r),
            get_mid(r, s),
            get_mid(s, p),
        ]
        for m in mids:
            members.append((fname, m, "median"))

    # connect cell center to each face center
    for fname in faces.keys():
        members.append(("C", fname, "spoke"))

    return nodes, members

# ============================================================
# 5) Tile multiple cells into a lattice
# ============================================================
def build_auxetic_lattice(nx=2, ny=2, nz=2,
                          a=v(1.0, 0.2, 0.0),
                          b=v(0.25, 1.0, 0.0),
                          c=v(0.15, 0.25, 1.0)):
    """
    Build repeated skewed cells.
    We merge coincident nodes so neighboring cells share geometry.
    """
    global_nodes = {}
    global_members = set()

    def key_from_point(P, tol=8):
        return tuple(np.round(P, tol))

    for i, j, k in product(range(nx), range(ny), range(nz)):
        origin = i * a + j * b + k * c
        cell_nodes, cell_members = build_auxetic_cell(origin, a, b, c)

        local_to_global = {}

        # merge nodes with same coordinates
        for lname, P in cell_nodes.items():
            key = key_from_point(P)
            if key not in global_nodes:
                gname = f"N{len(global_nodes)}"
                global_nodes[key] = (gname, P)
            local_to_global[lname] = global_nodes[key][0]

        # convert members to global names
        for u, w, kind in cell_members:
            gu = local_to_global[u]
            gw = local_to_global[w]
            if gu != gw:
                pair = tuple(sorted((gu, gw)))
                global_members.add((pair[0], pair[1], kind))

    # unpack nodes to dict[name] = point
    nodes = {name: P for (_, (name, P)) in global_nodes.items()}
    members = list(global_members)
    return nodes, members

# ============================================================
# 6) Convert straight members into Bezier curves
# ============================================================
def build_bezier_members(nodes, members, bend_frame=0.05, bend_median=0.12, bend_spoke=0.08):
    """
    Creates sampled curves for each member.
    Different member types get different curvature amounts.
    """
    curves = []

    # approximate lattice center for outward/inward curvature directions
    P_all = np.stack(list(nodes.values()), axis=0)
    center = P_all.mean(axis=0)

    for u, w, kind in members:
        P0 = nodes[u]
        P3 = nodes[w]
        mid = 0.5 * (P0 + P3)

        # Bend direction chosen so members curve relative to lattice center
        radial = mid - center

        if kind == "frame":
            bend = bend_frame
            bend_dir = radial + v(0, 0, 1)
        elif kind == "median":
            bend = bend_median
            bend_dir = -radial + v(0.3, 0.0, 0.5)
        else:  # "spoke"
            bend = bend_spoke
            bend_dir = np.cross(P3 - P0, radial + v(0, 0, 1))
            if np.linalg.norm(bend_dir) < 1e-12:
                bend_dir = radial + v(0, 1, 0)

        B0, B1, B2, B3 = bezier_strut(P0, P3, bend_dir=bend_dir, bend=bend)
        pts = sample_bezier3(B0, B1, B2, B3, n=90)
        curves.append((pts, kind))

    return curves

# ============================================================
# 7) Plotting
# ============================================================
def _set_axes_equal(ax):
    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    y_range = abs(y_limits[1] - y_limits[0])
    z_range = abs(z_limits[1] - z_limits[0])

    x_middle = np.mean(x_limits)
    y_middle = np.mean(y_limits)
    z_middle = np.mean(z_limits)

    plot_radius = 0.5 * max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])

def plot_lattice(nodes, curves, show_nodes=True, show_labels=False, title="3D auxetic-style lattice"):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax.set_title(title)

    for pts, kind in curves:
        if kind == "frame":
            lw = 1.8
            alpha = 0.9
        elif kind == "median":
            lw = 2.2
            alpha = 0.95
        else:
            lw = 1.4
            alpha = 0.8

        ax.plot(pts[:, 0], pts[:, 1], pts[:, 2], linewidth=lw, alpha=alpha)

    if show_nodes:
        P = np.stack(list(nodes.values()), axis=0)
        ax.scatter(P[:, 0], P[:, 1], P[:, 2], s=16)

    if show_labels:
        for name, p in nodes.items():
            ax.text(p[0], p[1], p[2], name, fontsize=7)

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    _set_axes_equal(ax)
    plt.tight_layout()
    plt.show()

# ============================================================
# 8) Optional affine deformation
# ============================================================
def apply_affine_to_nodes(nodes, F):
    return {k: (np.asarray(p) @ F.T) for k, p in nodes.items()}

# ============================================================
# 9) Export
# ============================================================
def export_curves_xyz(curves, filename="auxetic_bezier_lattice.xyz"):
    with open(filename, "w") as f:
        for pts, kind in curves:
            f.write(f"# {kind}\n")
            for p in pts:
                f.write(f"{p[0]:.6f} {p[1]:.6f} {p[2]:.6f}\n")
            f.write("\n")

# ============================================================
# 10) Main
# ============================================================
if __name__ == "__main__":
    # skewed basis vectors -> gives true 3D parallelogram faces
    a = v(1.00, 0.25, 0.00)
    b = v(0.30, 1.00, 0.00)
    c = v(0.15, 0.20, 1.00)

    nodes, members = build_auxetic_lattice(
        nx=2, ny=2, nz=2,
        a=a, b=b, c=c
    )

    curves = build_bezier_members(
        nodes, members,
        bend_frame=0.03,
        bend_median=0.10,
        bend_spoke=0.06
    )

    plot_lattice(
        nodes, curves,
        show_nodes=True,
        show_labels=False,
        title="Median-connected 3D parallelogram auxetic-style lattice"
    )

    # example deformation
    F = np.array([
        [1.06, 0.00, 0.00],
        [0.00, 1.06, 0.00],
        [0.00, 0.00, 0.92],
    ], dtype=float)

    nodes_def = apply_affine_to_nodes(nodes, F)
    curves_def = build_bezier_members(
        nodes_def, members,
        bend_frame=0.03,
        bend_median=0.10,
        bend_spoke=0.06
    )

    plot_lattice(
        nodes_def, curves_def,
        show_nodes=True,
        show_labels=False,
        title="Deformed lattice"
    )

    export_curves_xyz(curves, "auxetic_bezier_lattice.xyz")
    print("Exported to auxetic_bezier_lattice.xyz")
