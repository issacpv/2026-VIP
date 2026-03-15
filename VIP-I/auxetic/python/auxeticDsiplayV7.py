import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay, ConvexHull
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
try:
    from solid2 import polyhedron, sphere, hull, union, scad_render_to_file, translate
    HAS_SOLID = True
except ImportError:
    HAS_SOLID = False
try:
    from stl import mesh as stl_mesh
    HAS_STL = True
except ImportError:
    HAS_STL = False

# ==========================
# USER SETTINGS
# ==========================
mode = 6         # 1=random 2D, 2=random 2.5D, 3=random 3D
                 # 4=grid 2D,   5=grid 2.5D,   6=grid 3D
n_points = 9
ratio = 0.4
nx, ny, nz = 1, 1, 1
cell = 1.0
nz_layers = 3    # number of layers in 2.5D extrusion (modes 2, 5 only)

# --- Bezier toggle ---
use_bezier = False   # True = Bezier curved struts/edges, False = straight lines/flat edges

# --- Auxetic Bezier tuning (only applies when use_bezier = True) ---
bend_reentrant = 0.18  # bow for 2-point struts connecting shrunk simplex corners
bend_ngon      = 0.14  # bow for n-gon/polyhedron perimeter edges (shared-vertex hubs)
bend_triangle  = 0.12  # bow for magenta simplex face edges
bend_vertical  = 0.08  # bow for vertical struts in 2.5D modes

# --- Intersection check ---
intersect_threshold = 0.05   # fraction of mean strut length; raise = more sensitive
intersect_check     = True   # set False to skip (faster for large lattices)

# --- Export settings ---
export_scad = True          # generate a .scad file (OpenSCAD)
export_stl  = True          # generate a .stl file (direct STL, no OpenSCAD needed)
export_name = 'auxetic_lattice'  # output filename (no extension)
strut_radius = 0.02          # physical radius of struts in exported geometry (world units)
strut_segments = 6           # cross-section polygon sides for struts (higher = rounder, slower)

# ==========================
# Helpers
# ==========================
def v(x, y, z):
    return np.array([x, y, z], dtype=float)

def unit(a):
    a = np.asarray(a, dtype=float)
    n = np.linalg.norm(a)
    return np.zeros_like(a) if n < 1e-12 else a / n

def bezier3(P0, P1, P2, P3, t):
    t = np.asarray(t, dtype=float)
    if t.ndim == 0:
        t = np.array([t], dtype=float)
    u = 1.0 - t
    return (
        (u**3)[:, None] * P0
        + 3 * (u**2 * t)[:, None] * P1
        + 3 * (u * t**2)[:, None] * P2
        + (t**3)[:, None] * P3
    )

def sample_bezier3(P0, P1, P2, P3, n=40):
    return bezier3(P0, P1, P2, P3, np.linspace(0.0, 1.0, n))

def bezier_strut(P0, P3, bend_dir, bend=0.10):
    P0 = np.asarray(P0, float)
    P3 = np.asarray(P3, float)
    d  = P3 - P0
    L  = np.linalg.norm(d)
    if L < 1e-12:
        return P0, P0, P3, P3
    d_hat = d / L
    bd = unit(bend_dir)
    bd = bd - np.dot(bd, d_hat) * d_hat
    bd = unit(bd)
    if np.linalg.norm(bd) < 1e-12:
        trial = np.cross(d_hat, v(0, 0, 1))
        if np.linalg.norm(trial) < 1e-12:
            trial = np.cross(d_hat, v(0, 1, 0))
        bd = unit(trial)
    offset = bend * L * bd
    P1 = P0 + d / 3.0 + offset
    P2 = P0 + 2.0 * d / 3.0 + offset
    return P0, P1, P2, P3

def draw_bezier_strut(ax, p0, p1, bend_dir, bend,
                      color='cyan', alpha=0.9, lw=1.8,
                      registry=None, label='strut'):
    if use_bezier:
        B0, B1, B2, B3 = bezier_strut(p0, p1, bend_dir=bend_dir, bend=bend)
        pts = sample_bezier3(B0, B1, B2, B3)
    else:
        pts = np.array([p0, p1])
    ax.plot(pts[:, 0], pts[:, 1], pts[:, 2], color=color, alpha=alpha, linewidth=lw)
    if registry is not None:
        registry.append((pts, label))

def draw_bezier_filled_polygon(ax, verts_3d, centroid_3d, bend,
                                color, alpha_face, alpha_edge, lw,
                                registry=None, label='polygon'):
    """Filled polygon. When use_bezier=True, edges are Bezier struts.
    Either way, no outline is drawn on the face itself."""
    ax.add_collection3d(Poly3DCollection([verts_3d], color=color,
                                          alpha=alpha_face, edgecolor='none'))
    if use_bezier:
        n = len(verts_3d)
        for i in range(n):
            j        = (i + 1) % n
            p0       = verts_3d[i]
            p1       = verts_3d[j]
            mid      = 0.5 * (p0 + p1)
            bend_dir = unit(centroid_3d - mid)
            draw_bezier_strut(ax, p0, p1, bend_dir=bend_dir, bend=bend,
                              color=color, alpha=alpha_edge, lw=lw,
                              registry=registry, label=f'{label}_edge{i}')

# ==========================
# Intersection checker
# ==========================
def count_bezier_intersections(curve_list, threshold_frac=0.05):
    """
    Returns the number of non-adjacent curve pairs whose minimum
    sampled distance falls below threshold_frac * mean_strut_length.
    Does NOT print individual warnings — only the count is used externally.
    """
    if len(curve_list) < 2:
        return 0
    lengths   = [np.linalg.norm(c[0][-1] - c[0][0]) for c in curve_list]
    mean_len  = np.mean(lengths) if lengths else 1.0
    threshold = threshold_frac * mean_len
    count = 0
    n     = len(curve_list)
    for i in range(n):
        pts_i  = curve_list[i][0]
        ends_i = {tuple(np.round(pts_i[0],  8)),
                  tuple(np.round(pts_i[-1], 8))}
        for j in range(i + 1, n):
            pts_j  = curve_list[j][0]
            ends_j = {tuple(np.round(pts_j[0],  8)),
                      tuple(np.round(pts_j[-1], 8))}
            if ends_i & ends_j:          # adjacent — skip
                continue
            diff     = pts_i[:, None, :] - pts_j[None, :, :]
            min_dist = np.sqrt((diff ** 2).sum(axis=2)).min()
            if min_dist < threshold:
                count += 1
    return count


# ==========================
# 1) Point generation
# ==========================
def generate_points(n_points, mode):
    """Returns (points, tri) for 2D modes or (points_3d, tri) for 3D modes."""
    if mode in [3]:          # random 3D
        points = np.random.rand(n_points, 3)
        tri    = Delaunay(points)
        return points, tri
    elif mode in [6]:        # grid 3D
        cbrt = max(2, round(n_points ** (1/3)))
        x = np.linspace(0, 1, cbrt)
        xx, yy, zz = np.meshgrid(x, x, x)
        points = np.vstack([xx.ravel(), yy.ravel(), zz.ravel()]).T
        tri    = Delaunay(points)
        return points, tri
    elif mode in [4, 5]:     # grid 2D / 2.5D
        def factor_pair(n):
            for i in range(int(np.sqrt(n)), 0, -1):
                if n % i == 0:
                    return i, n // i
            return 1, n
        nx2d, ny2d = factor_pair(n_points)
        x  = np.linspace(0, 1, nx2d)
        y  = np.linspace(0, 1, ny2d)
        xx, yy = np.meshgrid(x, y)
        points = np.vstack([xx.ravel(), yy.ravel()]).T
        tri    = Delaunay(points)
        return points, tri
    else:                    # random 2D / 2.5D
        points = np.random.rand(n_points, 2)
        tri    = Delaunay(points)
        return points, tri

points_nd, tri_nd = generate_points(n_points, mode)

# ==========================
# 2) Generate background 3D lattice nodes
# ==========================
def generate_3d_nodes(nx, ny, nz, cell):
    nodes = {}
    for i in range(nx + 1):
        for j in range(ny + 1):
            for k in range(nz + 1):
                nodes[f"G_{i}_{j}_{k}"] = np.array(
                    [i*cell/nx, j*cell/ny, k*cell/nz], dtype=float
                )
    return nodes

nodes_3d = generate_3d_nodes(nx, ny, nz, cell)

# ==========================
# 3) Add lattice points to node dict
# ==========================
def add_lattice_to_3d(nodes_3d, points_nd, mode, nz_layers=3):
    nodes    = nodes_3d.copy()
    pts      = points_nd
    mins     = pts.min(axis=0)
    maxs     = pts.max(axis=0)
    pts_norm = (pts - mins) / (maxs - mins + 1e-12)

    if mode in [3, 6]:       # already 3D
        for idx, pt in enumerate(pts_norm):
            nodes[f"L3D_{idx}"] = pt.astype(float)
    elif mode in [1, 4]:
        for idx, pt in enumerate(pts_norm):
            nodes[f"L2D_{idx}_0"] = np.array([pt[0], pt[1], 0.0])
    elif mode in [2, 5]:
        for z_idx in range(nz_layers):
            z = z_idx / (nz_layers - 1)
            for idx, pt in enumerate(pts_norm):
                nodes[f"L2D_{idx}_{z_idx}"] = np.array([pt[0], pt[1], z])
    return nodes

nodes_combined = add_lattice_to_3d(nodes_3d, points_nd, mode, nz_layers=nz_layers)

# ==========================
# 4) Geometry builders (shared between dry-run and plot)
# ==========================

def build_simplex_curves_2d(points_2d, tri, ratio, z_val, bends, mode, nz_layers, z_idx):
    """
    Returns list of (pts, label) for all Bezier curves in one Z layer.
    No axes — pure geometry for the intersection dry-run.
    """
    curves = []

    def fake_draw(p0, p1, bend_dir, bend, label='s'):
        B0, B1, B2, B3 = bezier_strut(p0, p1, bend_dir=bend_dir, bend=bend)
        curves.append((sample_bezier3(B0, B1, B2, B3), label))

    def fake_polygon(verts_3d, centroid_3d, bend, label='p'):
        n = len(verts_3d)
        for i in range(n):
            j        = (i + 1) % n
            p0       = verts_3d[i]
            p1       = verts_3d[j]
            mid      = 0.5 * (p0 + p1)
            bend_dir = unit(centroid_3d - mid)
            fake_draw(p0, p1, bend_dir, bend, label=f'{label}_e{i}')

    # Magenta triangles
    for simplex in tri.simplices:
        triangle    = points_2d[simplex]
        centroid_2d = triangle.mean(axis=0)
        tri_pts_2d  = np.array([(1-ratio)*triangle[i] + ratio*centroid_2d for i in range(3)])
        if mode in [1, 4]:
            tri_pts_3d  = np.hstack([tri_pts_2d, np.zeros((3, 1))])
            centroid_3d = np.append(centroid_2d, 0.0)
            fake_polygon(tri_pts_3d, centroid_3d, bends['triangle'], label=f'tri{tuple(simplex)}')
        elif mode in [2, 5]:
            for zi in range(nz_layers - 1):
                zb = zi / (nz_layers - 1)
                zt = (zi + 1) / (nz_layers - 1)
                bottom_pts = np.hstack([tri_pts_2d, np.full((3, 1), zb)])
                top_pts    = np.hstack([tri_pts_2d, np.full((3, 1), zt)])
                fake_polygon(top_pts, np.append(centroid_2d, zt),
                             bends['triangle'], label=f'tri_top_{zi}')
                for i in range(3):
                    j    = (i + 1) % 3
                    quad = np.array([bottom_pts[i], bottom_pts[j], top_pts[j], top_pts[i]])
                    fake_polygon(quad, quad.mean(axis=0),
                                 bends['triangle'], label=f'tri_side{i}_{zi}')

    # Cyan connectors
    groups_white    = {tuple(p): [] for p in points_2d}
    strut_centroids = {tuple(p): [] for p in points_2d}
    for simplex in tri.simplices:
        triangle    = points_2d[simplex]
        centroid_2d = triangle.mean(axis=0)
        centroid_3d = np.append(centroid_2d, z_val)
        tri_pts_2d  = np.array([(1-ratio)*triangle[i] + ratio*centroid_2d for i in range(3)])
        tri_pts_3d  = np.hstack([tri_pts_2d, np.full((3, 1), z_val)])
        for i, vertex in enumerate(triangle):
            key = tuple(vertex)
            groups_white[key].append(tri_pts_3d[i])
            strut_centroids[key].append(centroid_3d)

    for white, pts_list in groups_white.items():
        centroids_list = strut_centroids[white]
        if len(pts_list) == 2:
            p0, p1   = pts_list[0], pts_list[1]
            bend_dir = unit(unit(centroids_list[0] - p0) + unit(centroids_list[1] - p1))
            if mode in [1, 4]:
                fake_draw(p0, p1, bend_dir, bends['reentrant'], label=f'strut_{white}')
            else:
                zb = z_idx / (nz_layers - 1)
                zt = (z_idx + 1) / (nz_layers - 1) if z_idx < nz_layers - 1 else zb
                p0b = np.array([p0[0], p0[1], zb]); p1b = np.array([p1[0], p1[1], zb])
                p0t = np.array([p0[0], p0[1], zt]); p1t = np.array([p1[0], p1[1], zt])
                fake_draw(p0b, p1b, bend_dir, bends['reentrant'])
                fake_draw(p0t, p1t, bend_dir, bends['reentrant'])
                vd = unit(np.append(centroids_list[0][:2] - 0.5*(p0[:2]+p1[:2]), 0))
                fake_draw(p0b, p0t, vd, bends['vertical'])
                fake_draw(p1b, p1t, vd, bends['vertical'])
        elif len(pts_list) >= 3:
            try:
                pts_arr    = np.array(pts_list)
                hub_center = pts_arr.mean(axis=0)
                hull       = ConvexHull(pts_arr[:, :2])
                vertices   = pts_arr[hull.vertices, :2]
                n_verts    = len(vertices)
                if mode in [1, 4]:
                    verts_3d = np.hstack([vertices, np.full((n_verts, 1), z_val)])
                    fake_polygon(verts_3d, hub_center, bends['ngon'], label=f'hub_{white}')
                else:
                    zb = z_idx / (nz_layers - 1)
                    zt = (z_idx + 1) / (nz_layers - 1) if z_idx < nz_layers - 1 else zb
                    tv = np.hstack([vertices, np.full((n_verts, 1), zt)])
                    bv = np.hstack([vertices, np.full((n_verts, 1), zb)])
                    fake_polygon(tv, np.append(hub_center[:2], zt), bends['ngon'])
                    fake_polygon(bv, np.append(hub_center[:2], zb), bends['ngon'])
                    for i in range(n_verts):
                        vd = unit(np.append(hub_center[:2] - vertices[i], 0))
                        fake_draw(np.append(vertices[i], zb),
                                  np.append(vertices[i], zt), vd, bends['vertical'])
            except Exception:
                pass
    return curves


def build_simplex_curves_3d(points_3d, tri, ratio, bends):
    """
    Dry-run geometry builder for 3D modes (3, 6).
    Each simplex is a tetrahedron. Returns list of (pts, label).
    """
    curves = []

    def fake_draw(p0, p1, bend_dir, bend, label='s'):
        B0, B1, B2, B3 = bezier_strut(p0, p1, bend_dir=bend_dir, bend=bend)
        curves.append((sample_bezier3(B0, B1, B2, B3), label))

    def fake_polygon(verts_3d, centroid_3d, bend, label='p'):
        n = len(verts_3d)
        for i in range(n):
            j        = (i + 1) % n
            p0       = verts_3d[i]
            p1       = verts_3d[j]
            mid      = 0.5 * (p0 + p1)
            bend_dir = unit(centroid_3d - mid)
            fake_draw(p0, p1, bend_dir, bend, label=f'{label}_e{i}')

    # Normalize points to [0,1]^3
    mins     = points_3d.min(axis=0)
    maxs     = points_3d.max(axis=0)
    pts_norm = (points_3d - mins) / (maxs - mins + 1e-12)

    groups = {tuple(p): [] for p in pts_norm}
    centroids_map = {tuple(p): [] for p in pts_norm}

    for simplex in tri.simplices:
        tet         = pts_norm[simplex]          # (4, 3)
        centroid    = tet.mean(axis=0)
        # Shrunk tet vertices
        shrunk      = np.array([(1-ratio)*tet[i] + ratio*centroid for i in range(4)])

        # 4 triangular faces of the tetrahedron
        face_indices = [(0,1,2), (0,1,3), (0,2,3), (1,2,3)]
        for fi in face_indices:
            face_verts = shrunk[list(fi)]
            face_cent  = face_verts.mean(axis=0)
            fake_polygon(face_verts, face_cent, bends['triangle'],
                         label=f'tet{tuple(simplex)}_face{fi}')

        # Group shrunk corners by original vertex for connector logic
        for i, vertex in enumerate(pts_norm[simplex]):
            key = tuple(vertex)
            groups[key].append(shrunk[i])
            centroids_map[key].append(centroid)

    # Connectors between shrunk corners at shared vertices
    for key, pts_list in groups.items():
        centroids_list = centroids_map[key]
        if len(pts_list) == 2:
            p0, p1   = pts_list[0], pts_list[1]
            bend_dir = unit(unit(centroids_list[0] - p0) + unit(centroids_list[1] - p1))
            fake_draw(p0, p1, bend_dir, bends['reentrant'], label=f'strut_{key}')
        elif len(pts_list) >= 3:
                pts_arr    = np.array(pts_list)
                hub_center = pts_arr.mean(axis=0)
                n_pts      = len(pts_arr)
                for a in range(n_pts):
                    for b in range(a + 1, n_pts):
                        p0       = pts_arr[a]
                        p1       = pts_arr[b]
                        mid      = 0.5 * (p0 + p1)
                        bend_dir = unit(hub_center - mid)
                        fake_draw(p0, p1, bend_dir, bends['ngon'],
                                  label=f'hub_{key}_e{a}{b}')
    return curves

# ==========================
# 5) Plot function
# ==========================
def plot_3d_lattice(nodes, points_nd, tri, ratio, mode, nz_layers=3, show_nodes=False):
    # --- Dry-run for intersection check / auto-adjust ---
    initial_bends = {
        'reentrant': bend_reentrant,
        'ngon':      bend_ngon,
        'triangle':  bend_triangle,
        'vertical':  bend_vertical,
    }

    if intersect_check and use_bezier:
        print("\n--- Intersection Check ---")
        active_bends = dict(initial_bends)

        if mode in [3, 6]:
            def curve_builder(bends):
                return build_simplex_curves_3d(points_nd, tri, ratio, bends)
        else:
            layers = [0] if mode in [1, 4] else list(range(nz_layers))
            def curve_builder(bends):
                all_curves = []
                for z_idx in layers:
                    z_val = 0.0 if mode in [1, 4] else z_idx / (nz_layers - 1)
                    all_curves += build_simplex_curves_2d(
                        points_nd, tri, ratio, z_val, bends, mode, nz_layers, z_idx)
                return all_curves

        dry_curves = curve_builder(active_bends)
        n_total    = len(dry_curves)
        n_inter    = count_bezier_intersections(dry_curves, intersect_threshold)
        ratio_val  = n_inter / n_total if n_total > 0 else 0.0

        print(f"Curves checked : {n_total}")
        print(f"Intersections  : {n_inter} ({100*ratio_val:.1f}%)")
        print("--------------------------\n")
    else:
        active_bends = initial_bends

    # --- Actual plot ---
    fig = plt.figure(figsize=(8, 6))
    ax  = fig.add_subplot(111, projection='3d')
    registry = [] if (intersect_check and use_bezier) else None

    # ---- 3D modes (3, 6) ----
    if mode in [3, 6]:
        mins     = points_nd.min(axis=0)
        maxs     = points_nd.max(axis=0)
        pts_norm = (points_nd - mins) / (maxs - mins + 1e-12)

        groups        = {tuple(p): [] for p in pts_norm}
        centroids_map = {tuple(p): [] for p in pts_norm}

        for simplex in tri.simplices:
            tet      = pts_norm[simplex]
            centroid = tet.mean(axis=0)
            shrunk   = np.array([(1-ratio)*tet[i] + ratio*centroid for i in range(4)])

            # Draw 4 triangular faces of each shrunken tetrahedron (magenta)
            face_indices = [(0,1,2), (0,1,3), (0,2,3), (1,2,3)]
            for fi in face_indices:
                face_verts = shrunk[list(fi)]
                face_cent  = face_verts.mean(axis=0)
                draw_bezier_filled_polygon(
                    ax, face_verts, face_cent,
                    bend=active_bends['triangle'],
                    color='magenta', alpha_face=0.20, alpha_edge=0.85, lw=1.6,
                    registry=registry,
                    label=f'tet{tuple(simplex)}_face{fi}')

            for i, vertex in enumerate(pts_norm[simplex]):
                key = tuple(vertex)
                groups[key].append(shrunk[i])
                centroids_map[key].append(centroid)

        # Cyan scatter dots at every shrunk tet corner
        all_shrunk = [pt for pts_list in groups.values() for pt in pts_list]
        for pt in all_shrunk:
            ax.scatter(*pt, color='cyan', s=20)

        # Cyan connectors
        for key, pts_list in groups.items():
            centroids_list = centroids_map[key]
            if len(pts_list) == 2:
                p0, p1   = pts_list[0], pts_list[1]
                bend_dir = unit(unit(centroids_list[0] - p0) +
                                unit(centroids_list[1] - p1))
                draw_bezier_strut(ax, p0, p1, bend_dir=bend_dir,
                                  bend=active_bends['reentrant'],
                                  registry=registry, label=f'strut_{key}')
            elif len(pts_list) >= 3:
                # Use all unique pairs instead of ConvexHull —
                # ConvexHull fails when points are coplanar (common in grid mode)
                pts_arr    = np.array(pts_list)
                hub_center = pts_arr.mean(axis=0)
                n_pts      = len(pts_arr)
                for a in range(n_pts):
                    for b in range(a + 1, n_pts):
                        p0       = pts_arr[a]
                        p1       = pts_arr[b]
                        mid      = 0.5 * (p0 + p1)
                        bend_dir = unit(hub_center - mid)
                        draw_bezier_strut(ax, p0, p1, bend_dir=bend_dir,
                                          bend=active_bends['ngon'],
                                          registry=registry,
                                          label=f'hub_{key}_e{a}{b}')

    # ---- 2D / 2.5D modes (1, 2, 4, 5) ----
    else:
        points_2d = points_nd

        # Magenta triangles
        for simplex in tri.simplices:
            triangle    = points_2d[simplex]
            centroid_2d = triangle.mean(axis=0)
            tri_pts_2d  = np.array([(1-ratio)*triangle[i] + ratio*centroid_2d for i in range(3)])

            if mode in [1, 4]:
                tri_pts_3d  = np.hstack([tri_pts_2d, np.zeros((3, 1))])
                centroid_3d = np.append(centroid_2d, 0.0)
                draw_bezier_filled_polygon(
                    ax, tri_pts_3d, centroid_3d,
                    bend=active_bends['triangle'], color='magenta',
                    alpha_face=0.25, alpha_edge=0.9, lw=1.8,
                    registry=registry, label=f'tri{tuple(simplex)}_z0')

            elif mode in [2, 5]:
                for z_idx in range(nz_layers - 1):
                    zb = z_idx / (nz_layers - 1)
                    zt = (z_idx + 1) / (nz_layers - 1)
                    bottom_pts = np.hstack([tri_pts_2d, np.full((3, 1), zb)])
                    top_pts    = np.hstack([tri_pts_2d, np.full((3, 1), zt)])
                    draw_bezier_filled_polygon(
                        ax, top_pts, np.append(centroid_2d, zt),
                        bend=active_bends['triangle'], color='magenta',
                        alpha_face=0.25, alpha_edge=0.9, lw=1.8,
                        registry=registry,
                        label=f'tri{tuple(simplex)}_top_z{z_idx}')
                    for i in range(3):
                        j    = (i + 1) % 3
                        quad = np.array([bottom_pts[i], bottom_pts[j],
                                         top_pts[j],    top_pts[i]])
                        draw_bezier_filled_polygon(
                            ax, quad, quad.mean(axis=0),
                            bend=active_bends['triangle'], color='magenta',
                            alpha_face=0.15, alpha_edge=0.85, lw=1.5,
                            registry=registry,
                            label=f'tri{tuple(simplex)}_side{i}_z{z_idx}')

        # Cyan connectors
        layers = [0] if mode in [1, 4] else list(range(nz_layers))
        for z_idx in layers:
            z_val = 0.0 if mode in [1, 4] else z_idx / (nz_layers - 1)

            groups_white    = {tuple(p): [] for p in points_2d}
            strut_centroids = {tuple(p): [] for p in points_2d}

            for simplex in tri.simplices:
                triangle    = points_2d[simplex]
                centroid_2d = triangle.mean(axis=0)
                centroid_3d = np.append(centroid_2d, z_val)
                tri_pts_2d  = np.array([(1-ratio)*triangle[i] + ratio*centroid_2d for i in range(3)])
                tri_pts_3d  = np.hstack([tri_pts_2d, np.full((3, 1), z_val)])
                for pt in tri_pts_3d:
                    ax.scatter(*pt, color='cyan', s=20)
                for i, vertex in enumerate(triangle):
                    key = tuple(vertex)
                    groups_white[key].append(tri_pts_3d[i])
                    strut_centroids[key].append(centroid_3d)

            for white, pts_list in groups_white.items():
                centroids_list = strut_centroids[white]
                if len(pts_list) == 2:
                    p0, p1   = pts_list[0], pts_list[1]
                    bend_dir = unit(unit(centroids_list[0] - p0) +
                                    unit(centroids_list[1] - p1))
                    if mode in [1, 4]:
                        draw_bezier_strut(ax, p0, p1, bend_dir=bend_dir,
                                          bend=active_bends['reentrant'],
                                          registry=registry,
                                          label=f'strut_{white}')
                    else:
                        zb = z_idx / (nz_layers - 1)
                        zt = (z_idx+1)/(nz_layers-1) if z_idx < nz_layers-1 else zb
                        p0b = np.array([p0[0], p0[1], zb])
                        p1b = np.array([p1[0], p1[1], zb])
                        p0t = np.array([p0[0], p0[1], zt])
                        p1t = np.array([p1[0], p1[1], zt])
                        draw_bezier_strut(ax, p0b, p1b, bend_dir=bend_dir,
                                          bend=active_bends['reentrant'],
                                          registry=registry,
                                          label=f'strut_{white}_bot_z{z_idx}')
                        draw_bezier_strut(ax, p0t, p1t, bend_dir=bend_dir,
                                          bend=active_bends['reentrant'],
                                          registry=registry,
                                          label=f'strut_{white}_top_z{z_idx}')
                        vd = unit(np.append(centroids_list[0][:2] - 0.5*(p0[:2]+p1[:2]), 0))
                        draw_bezier_strut(ax, p0b, p0t, bend_dir=vd,
                                          bend=active_bends['vertical'],
                                          registry=registry,
                                          label=f'vert0_{white}_z{z_idx}')
                        draw_bezier_strut(ax, p1b, p1t, bend_dir=vd,
                                          bend=active_bends['vertical'],
                                          registry=registry,
                                          label=f'vert1_{white}_z{z_idx}')

                elif len(pts_list) >= 3:
                    try:
                        pts_arr    = np.array(pts_list)
                        hub_center = pts_arr.mean(axis=0)
                        hull       = ConvexHull(pts_arr[:, :2])
                        vertices   = pts_arr[hull.vertices, :2]
                        n_verts    = len(vertices)
                        if mode in [1, 4]:
                            verts_3d = np.hstack([vertices, np.full((n_verts, 1), z_val)])
                            draw_bezier_filled_polygon(
                                ax, verts_3d, hub_center,
                                bend=active_bends['ngon'], color='cyan',
                                alpha_face=0.25, alpha_edge=0.9, lw=1.8,
                                registry=registry,
                                label=f'hub_{white}_z{z_idx}')
                        else:
                            zb = z_idx / (nz_layers - 1)
                            zt = (z_idx+1)/(nz_layers-1) if z_idx < nz_layers-1 else zb
                            tv = np.hstack([vertices, np.full((n_verts, 1), zt)])
                            bv = np.hstack([vertices, np.full((n_verts, 1), zb)])
                            draw_bezier_filled_polygon(
                                ax, tv, np.append(hub_center[:2], zt),
                                bend=active_bends['ngon'], color='cyan',
                                alpha_face=0.25, alpha_edge=0.9, lw=1.8,
                                registry=registry,
                                label=f'hub_{white}_top_z{z_idx}')
                            draw_bezier_filled_polygon(
                                ax, bv, np.append(hub_center[:2], zb),
                                bend=active_bends['ngon'], color='cyan',
                                alpha_face=0.25, alpha_edge=0.9, lw=1.8,
                                registry=registry,
                                label=f'hub_{white}_bot_z{z_idx}')
                            for i in range(n_verts):
                                vd = unit(np.append(hub_center[:2] - vertices[i], 0))
                                draw_bezier_strut(
                                    ax, np.append(vertices[i], zb),
                                    np.append(vertices[i], zt),
                                    bend_dir=vd, bend=active_bends['vertical'],
                                    registry=registry,
                                    label=f'hub_{white}_vert{i}_z{z_idx}')
                    except Exception:
                        pass

    # --- View settings ---
    ax.set_box_aspect([1, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_zlim(0, 1)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    elev, azim = 30, 45
    ax.view_init(elev=elev, azim=azim)

    def on_key(event):
        nonlocal elev, azim
        step = 10
        if event.key == 'up':      elev += step
        elif event.key == 'down':  elev -= step
        elif event.key == 'left':  azim -= step
        elif event.key == 'right': azim += step
        elif event.key == 'x':     elev, azim = 0, 0
        elif event.key == 'y':     elev, azim = 0, 90
        elif event.key == 'z':     elev, azim = 90, -90
        ax.view_init(elev=elev, azim=azim)
        fig.canvas.draw_idle()

    fig.canvas.mpl_connect('key_press_event', on_key)
    plt.show()


# ==========================
# 7) OpenSCAD / STL export
# ==========================

# --- Export settings (edit these) ---
export_enabled   = True        # set True to generate .scad and .stl on run
# Paths are relative to the script location; use absolute paths to write elsewhere
export_scad_path = "auxetic_lattice.scad"
export_stl_path  = "auxetic_lattice.stl"
export_png_path  = None          # set to e.g. "auxetic_lattice.png" to render a preview
strut_radius     = 0.02         # physical tube radius for struts (in lattice units)
face_thickness   = 0.015        # extrusion thickness for solid faces (in lattice units)
scad_segments    = 8            # $fn for cylinders — raise for smoother tubes (slower)

def _fmt(v):
    """Format a 3-vector as an OpenSCAD string."""
    return f"[{v[0]:.6f},{v[1]:.6f},{v[2]:.6f}]"

def scad_strut(p0, p1, radius, fn):
    """
    OpenSCAD snippet for a cylinder swept from p0 to p1.
    Aligns the Z-axis of a cylinder with the strut direction using
    rotate(a=angle, v=axis) where angle = arccos(dot(Z, d_hat)).
    """
    p0 = np.asarray(p0, float)
    p1 = np.asarray(p1, float)
    d  = p1 - p0
    L  = np.linalg.norm(d)
    if L < 1e-10:
        return ""

    z_hat = np.array([0.0, 0.0, 1.0])
    d_hat = d / L
    dot   = np.clip(np.dot(z_hat, d_hat), -1.0, 1.0)
    angle = np.degrees(np.arccos(dot))
    axis  = np.cross(z_hat, d_hat)
    a_norm = np.linalg.norm(axis)
    if a_norm < 1e-10:
        axis = np.array([1.0, 0.0, 0.0])  # arbitrary perp when already aligned
    else:
        axis = axis / a_norm

    return (
        f"translate({_fmt(p0)})\n"
        f"  rotate(a={angle:.6f}, v={_fmt(axis)})\n"
        f"    cylinder(h={L:.6f}, r={radius:.6f}, $fn={fn});\n"
    )

def scad_bezier_tube(pts, radius, fn):
    """
    Approximate a Bezier curve as a union of short cylinder segments.
    pts: (N, 3) sampled curve points.
    """
    lines = []
    for i in range(len(pts) - 1):
        lines.append(scad_strut(pts[i], pts[i+1], radius, fn))
    return "".join(lines)

def scad_face(verts_3d, thickness, fn):
    """
    Extrude a polygon face into a solid prism along its normal.
    All faces are fully triangulated so CGAL never sees non-planar polygons.
    """
    verts_3d = np.asarray(verts_3d, float)
    n = len(verts_3d)
    if n < 3:
        return ""

    # Newell normal
    normal = np.zeros(3)
    for i in range(n):
        j = (i + 1) % n
        normal[0] += (verts_3d[i][1] - verts_3d[j][1]) * (verts_3d[i][2] + verts_3d[j][2])
        normal[1] += (verts_3d[i][2] - verts_3d[j][2]) * (verts_3d[i][0] + verts_3d[j][0])
        normal[2] += (verts_3d[i][0] - verts_3d[j][0]) * (verts_3d[i][1] + verts_3d[j][1])
    nn = np.linalg.norm(normal)
    if nn < 1e-10:
        return ""
    normal = normal / nn

    # bottom verts = indices 0..n-1, top verts = indices n..2n-1
    bottom = verts_3d
    top    = verts_3d + normal * thickness
    all_verts = list(bottom) + list(top)
    vert_strs = ", ".join(_fmt(v) for v in all_verts)

    tris = []
    # Bottom cap — fan from vertex 0, reversed winding for outward normal
    for i in range(1, n - 1):
        tris.append([0, i + 1, i])
    # Top cap — fan from vertex n
    for i in range(1, n - 1):
        tris.append([n, n + i, n + i + 1])
    # Side walls — each edge becomes two triangles
    for i in range(n):
        j = (i + 1) % n
        tris.append([i,     j,     j + n])
        tris.append([i,     j + n, i + n])

    face_strs = ", ".join("[" + ",".join(str(x) for x in t) + "]" for t in tris)
    return f"polyhedron(points=[{vert_strs}], faces=[{face_strs}], convexity=4);\n"

def collect_export_geometry(points_nd, tri, ratio, mode, nz_layers, active_bends):
    """
    Walk the same geometry logic as the plotter but collect:
      strut_curves : list of (N,3) point arrays  — one per strut
      face_verts   : list of (M,3) vertex arrays — one per face
    No matplotlib involved.
    """
    strut_curves = []
    face_verts_list = []

    def add_strut(p0, p1, bend_dir, bend):
        if use_bezier:
            B0, B1, B2, B3 = bezier_strut(p0, p1, bend_dir=bend_dir, bend=bend)
            pts = sample_bezier3(B0, B1, B2, B3, n=20)
        else:
            pts = np.array([p0, p1], dtype=float)
        strut_curves.append(pts)

    def add_face(verts_3d, centroid_3d, bend):
        face_verts_list.append(np.asarray(verts_3d, float))
        if use_bezier:
            n = len(verts_3d)
            for i in range(n):
                j        = (i + 1) % n
                p0       = verts_3d[i]
                p1       = verts_3d[j]
                mid      = 0.5 * (p0 + p1)
                bend_dir = unit(centroid_3d - mid)
                add_strut(p0, p1, bend_dir, bend)

    # ---- 3D modes ----
    if mode in [3, 6]:
        mins     = points_nd.min(axis=0)
        maxs     = points_nd.max(axis=0)
        pts_norm = (points_nd - mins) / (maxs - mins + 1e-12)
        groups        = {tuple(p): [] for p in pts_norm}
        centroids_map = {tuple(p): [] for p in pts_norm}

        for simplex in tri.simplices:
            tet      = pts_norm[simplex]
            centroid = tet.mean(axis=0)
            shrunk   = np.array([(1-ratio)*tet[i] + ratio*centroid for i in range(4)])
            for fi in [(0,1,2),(0,1,3),(0,2,3),(1,2,3)]:
                fv   = shrunk[list(fi)]
                fc   = fv.mean(axis=0)
                add_face(fv, fc, active_bends['triangle'])
            for i, vertex in enumerate(pts_norm[simplex]):
                key = tuple(vertex)
                groups[key].append(shrunk[i])
                centroids_map[key].append(centroid)

        for key, pts_list in groups.items():
            centroids_list = centroids_map[key]
            if len(pts_list) == 2:
                p0, p1   = pts_list[0], pts_list[1]
                bend_dir = unit(unit(centroids_list[0]-p0)+unit(centroids_list[1]-p1))
                add_strut(p0, p1, bend_dir, active_bends['reentrant'])
            elif len(pts_list) >= 3:
                pts_arr    = np.array(pts_list)
                hub_center = pts_arr.mean(axis=0)
                n_pts      = len(pts_arr)
                for a in range(n_pts):
                    for b in range(a+1, n_pts):
                        p0  = pts_arr[a]; p1 = pts_arr[b]
                        mid = 0.5*(p0+p1)
                        add_strut(p0, p1, unit(hub_center-mid), active_bends['ngon'])

    # ---- 2D / 2.5D modes ----
    else:
        points_2d = points_nd
        layers = [0] if mode in [1,4] else list(range(nz_layers))

        for simplex in tri.simplices:
            triangle    = points_2d[simplex]
            centroid_2d = triangle.mean(axis=0)
            tri_pts_2d  = np.array([(1-ratio)*triangle[i]+ratio*centroid_2d for i in range(3)])
            if mode in [1,4]:
                tri_pts_3d  = np.hstack([tri_pts_2d, np.zeros((3,1))])
                centroid_3d = np.append(centroid_2d, 0.0)
                add_face(tri_pts_3d, centroid_3d, active_bends['triangle'])
            elif mode in [2,5]:
                for z_idx in range(nz_layers-1):
                    zb = z_idx/(nz_layers-1); zt = (z_idx+1)/(nz_layers-1)
                    bp = np.hstack([tri_pts_2d, np.full((3,1),zb)])
                    tp = np.hstack([tri_pts_2d, np.full((3,1),zt)])
                    add_face(tp, np.append(centroid_2d,zt), active_bends['triangle'])
                    for i in range(3):
                        j    = (i+1)%3
                        quad = np.array([bp[i],bp[j],tp[j],tp[i]])
                        add_face(quad, quad.mean(axis=0), active_bends['triangle'])

        for z_idx in layers:
            z_val = 0.0 if mode in [1,4] else z_idx/(nz_layers-1)
            groups_white    = {tuple(p): [] for p in points_2d}
            strut_centroids = {tuple(p): [] for p in points_2d}
            for simplex in tri.simplices:
                triangle    = points_2d[simplex]
                centroid_2d = triangle.mean(axis=0)
                centroid_3d = np.append(centroid_2d, z_val)
                tri_pts_2d  = np.array([(1-ratio)*triangle[i]+ratio*centroid_2d for i in range(3)])
                tri_pts_3d  = np.hstack([tri_pts_2d, np.full((3,1),z_val)])
                for i, vertex in enumerate(triangle):
                    key = tuple(vertex)
                    groups_white[key].append(tri_pts_3d[i])
                    strut_centroids[key].append(centroid_3d)

            for white, pts_list in groups_white.items():
                centroids_list = strut_centroids[white]
                if len(pts_list) == 2:
                    p0, p1   = pts_list[0], pts_list[1]
                    bend_dir = unit(unit(centroids_list[0]-p0)+unit(centroids_list[1]-p1))
                    if mode in [1,4]:
                        add_strut(p0, p1, bend_dir, active_bends['reentrant'])
                    else:
                        zb = z_idx/(nz_layers-1)
                        zt = (z_idx+1)/(nz_layers-1) if z_idx<nz_layers-1 else zb
                        p0b=np.array([p0[0],p0[1],zb]); p1b=np.array([p1[0],p1[1],zb])
                        p0t=np.array([p0[0],p0[1],zt]); p1t=np.array([p1[0],p1[1],zt])
                        vd = unit(np.append(centroids_list[0][:2]-0.5*(p0[:2]+p1[:2]),0))
                        add_strut(p0b, p1b, bend_dir, active_bends['reentrant'])
                        add_strut(p0t, p1t, bend_dir, active_bends['reentrant'])
                        add_strut(p0b, p0t, vd, active_bends['vertical'])
                        add_strut(p1b, p1t, vd, active_bends['vertical'])
                elif len(pts_list) >= 3:
                    try:
                        pts_arr    = np.array(pts_list)
                        hub_center = pts_arr.mean(axis=0)
                        hull       = ConvexHull(pts_arr[:,:2])
                        vertices   = pts_arr[hull.vertices,:2]
                        n_verts    = len(vertices)
                        if mode in [1,4]:
                            verts_3d = np.hstack([vertices, np.full((n_verts,1),z_val)])
                            add_face(verts_3d, hub_center, active_bends['ngon'])
                            # Always add edge struts regardless of use_bezier
                            for i in range(n_verts):
                                j   = (i+1) % n_verts
                                p0  = np.append(vertices[i], z_val)
                                p1  = np.append(vertices[j], z_val)
                                mid = 0.5*(p0+p1)
                                add_strut(p0, p1, unit(hub_center-mid), active_bends['ngon'])
                        else:
                            zb = z_idx/(nz_layers-1)
                            zt = (z_idx+1)/(nz_layers-1) if z_idx<nz_layers-1 else zb
                            tv = np.hstack([vertices, np.full((n_verts,1),zt)])
                            bv = np.hstack([vertices, np.full((n_verts,1),zb)])
                            add_face(tv, np.append(hub_center[:2],zt), active_bends['ngon'])
                            add_face(bv, np.append(hub_center[:2],zb), active_bends['ngon'])
                            for i in range(n_verts):
                                j   = (i+1) % n_verts
                                p0b = np.append(vertices[i], zb)
                                p1b = np.append(vertices[j], zb)
                                p0t = np.append(vertices[i], zt)
                                p1t = np.append(vertices[j], zt)
                                mid = 0.5*(p0b+p1b)
                                bd  = unit(hub_center-mid)
                                vd  = unit(np.append(hub_center[:2]-vertices[i],0))
                                add_strut(p0b, p1b, bd, active_bends['ngon'])
                                add_strut(p0t, p1t, bd, active_bends['ngon'])
                                add_strut(p0b, p0t, vd, active_bends['vertical'])
                    except Exception:
                        pass

    return strut_curves, face_verts_list


def export_to_scad(scad_path, strut_curves, face_verts_list):
    """Write a .scad file unioning all struts and faces."""
    lines = [
        f"// Auto-generated by lattice_viz.py\n",
        f"// mode={mode}  n_points={n_points}  ratio={ratio}  use_bezier={use_bezier}\n\n",
        f"$fn = {scad_segments};\n\n",
        "render(convexity=10)\n",
        "union() {\n",
    ]

    # Struts
    if strut_curves:
        lines.append("  // --- struts ---\n")
        for pts in strut_curves:
            lines.append(scad_bezier_tube(pts, strut_radius, scad_segments))

    # Faces
    if face_verts_list:
        lines.append("  // --- faces ---\n")
        for fv in face_verts_list:
            lines.append(scad_face(fv, face_thickness, scad_segments))

    lines.append("}\n")

    with open(scad_path, "w") as f:
        f.writelines(lines)
    print(f"  Wrote: {scad_path}  ({len(strut_curves)} struts, {len(face_verts_list)} faces)")


def find_openscad():
    """Find the OpenSCAD binary, checking PATH and common install locations."""
    import shutil, os
    # Check PATH first
    found = shutil.which("openscad") or shutil.which("OpenSCAD")
    if found:
        return found
    # Common Windows install locations
    candidates = [
        r"C:\Program Files\OpenSCAD\openscad.exe",
        r"C:\Program Files (x86)\OpenSCAD\openscad.exe",
        os.path.expanduser(r"~\AppData\Local\Programs\OpenSCAD\openscad.exe"),
    ]
    # Common macOS locations
    candidates += [
        "/Applications/OpenSCAD.app/Contents/MacOS/OpenSCAD",
        "/usr/local/bin/openscad",
        "/opt/homebrew/bin/openscad",
    ]
    for path in candidates:
        if os.path.isfile(path):
            return path
    return None


def render_scad(scad_path, out_path, extra_args=None):
    """
    Call OpenSCAD CLI to render a .scad file.
    out_path extension determines output type: .stl, .off, .png, .svg, etc.
    """
    import subprocess, os
    openscad_bin = find_openscad()
    if openscad_bin is None:
        print("  OpenSCAD not found on PATH — .scad written but not rendered.")
        print("  Install from https://openscad.org and re-run, or open the .scad manually.")
        return False
    ext = os.path.splitext(out_path)[1].lower()
    fmt_args = ["--export-format", "binstl"] if ext == ".stl" else []
    cmd = [openscad_bin] + fmt_args + ["-o", out_path, scad_path]
    if extra_args:
        cmd += extra_args
    print(f"  OpenSCAD -> {out_path}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode == 0:
        print(f"  Done: {out_path}")
        return True
    else:
        print(f"  OpenSCAD error:\n{result.stderr}")
        return False


def export_stl_direct(stl_path, strut_curves, face_verts_list):
    """
    Write STL directly from Python using numpy-stl.
    No OpenSCAD required.
    Faces are written as thin solid prisms; struts as tube meshes.
    """
    import os
    try:
        from stl import mesh as stl_mesh
    except ImportError:
        print("  numpy-stl not installed. Run: pip install numpy-stl")
        return False

    all_triangles = []

    # --- Faces: triangulate each polygon as a thin prism ---
    for poly in face_verts_list:
        poly = np.asarray(poly, float)
        n = len(poly)
        if n < 3:
            continue

        # Newell normal for offset direction
        normal = np.zeros(3)
        for i in range(n):
            j = (i + 1) % n
            normal[0] += (poly[i][1] - poly[j][1]) * (poly[i][2] + poly[j][2])
            normal[1] += (poly[i][2] - poly[j][2]) * (poly[i][0] + poly[j][0])
            normal[2] += (poly[i][0] - poly[j][0]) * (poly[i][1] + poly[j][1])
        nn = np.linalg.norm(normal)
        if nn < 1e-10:
            continue
        normal = normal / nn

        bottom = poly
        top    = poly + normal * face_thickness

        # Bottom cap (reversed winding)
        for i in range(1, n - 1):
            all_triangles.append([bottom[0], bottom[i + 1], bottom[i]])
        # Top cap
        for i in range(1, n - 1):
            all_triangles.append([top[0], top[i], top[i + 1]])
        # Side walls
        for i in range(n):
            j = (i + 1) % n
            all_triangles.append([bottom[i], bottom[j], top[j]])
            all_triangles.append([bottom[i], top[j],   top[i]])

    # --- Struts: tube mesh along each curve ---
    def tube_mesh(path, radius, segments=6):
        path = np.asarray(path, float)
        if len(path) < 2:
            return []
        rings = []
        for idx, pt in enumerate(path):
            if idx == 0:
                tang = path[1] - path[0]
            elif idx == len(path) - 1:
                tang = path[-1] - path[-2]
            else:
                tang = path[idx + 1] - path[idx - 1]
            tang_n = np.linalg.norm(tang)
            if tang_n < 1e-12:
                continue
            tang = tang / tang_n
            perp = np.cross(tang, np.array([0, 0, 1]))
            if np.linalg.norm(perp) < 1e-6:
                perp = np.cross(tang, np.array([0, 1, 0]))
            perp = perp / np.linalg.norm(perp)
            perp2 = np.cross(tang, perp)
            ring = [pt + radius * (np.cos(2*np.pi*s/segments) * perp +
                                   np.sin(2*np.pi*s/segments) * perp2)
                    for s in range(segments)]
            rings.append(ring)

        tris = []
        for i in range(len(rings) - 1):
            r0, r1 = rings[i], rings[i + 1]
            seg = len(r0)
            for s in range(seg):
                s1 = (s + 1) % seg
                tris.append([r0[s], r1[s],  r1[s1]])
                tris.append([r0[s], r1[s1], r0[s1]])
        # End caps
        c0 = path[0]
        for s in range(len(rings[0])):
            s1 = (s + 1) % len(rings[0])
            tris.append([c0, rings[0][s1], rings[0][s]])
        c1 = path[-1]
        for s in range(len(rings[-1])):
            s1 = (s + 1) % len(rings[-1])
            tris.append([c1, rings[-1][s], rings[-1][s1]])
        return tris

    for path in strut_curves:
        all_triangles += tube_mesh(path, strut_radius, scad_segments)

    if not all_triangles:
        print("  No geometry to export.")
        return False

    m = stl_mesh.Mesh(np.zeros(len(all_triangles), dtype=stl_mesh.Mesh.dtype))
    for i, tri in enumerate(all_triangles):
        for j in range(3):
            m.vectors[i][j] = np.asarray(tri[j], float)
    m.save(stl_path)
    size_kb = os.path.getsize(stl_path) // 1024
    print(f"  STL ready: {stl_path}  ({size_kb} KB)")
    return True


def run_export(points_nd, tri, ratio, mode, nz_layers):
    """Entry point: collect geometry, write .scad and .stl directly from Python."""
    import os
    print("\n--- Export ---")

    script_dir = os.path.dirname(os.path.abspath(__file__))

    def resolve(p):
        return p if os.path.isabs(p) else os.path.join(script_dir, p)

    scad_abs = resolve(export_scad_path)
    stl_abs  = resolve(export_stl_path)

    active_bends = {
        'reentrant': bend_reentrant,
        'ngon':      bend_ngon,
        'triangle':  bend_triangle,
        'vertical':  bend_vertical,
    }

    strut_curves, face_verts_list = collect_export_geometry(
        points_nd, tri, ratio, mode, nz_layers, active_bends
    )
    print(f"  Geometry: {len(strut_curves)} struts, {len(face_verts_list)} faces")

    export_to_scad(scad_abs, strut_curves, face_verts_list)
    export_stl_direct(stl_abs, strut_curves, face_verts_list)

    print("--------------\n")


# ==========================
# 8) Run — single execution
# ==========================

# Export runs first so it doesn't block on the plot window
if export_enabled:
    run_export(points_nd, tri_nd, ratio, mode, nz_layers)

plot_3d_lattice(nodes_combined, points_nd, tri_nd, ratio, mode,
                nz_layers=nz_layers, show_nodes=False)
