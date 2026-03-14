import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay, ConvexHull
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# ==========================
# USER SETTINGS
# ==========================
mode = 5         # 1=random 2D, 2=random 2.5D, 4=grid 2D, 5=grid 2.5D
n_points = 9
ratio = 0.4
nx, ny, nz = 1, 1, 1
cell = 1.0
nz_layers = 3    # number of layers in 2.5D extrusion

# --- Auxetic Bezier tuning ---
bend_reentrant = 0.18  # bow for 2-point struts connecting shrunk triangle corners
bend_ngon      = 0.08  # bow for n-gon perimeter edges (shared-vertex hubs)
bend_triangle  = 0.08  # bow for magenta triangle edges
bend_vertical  = 0.08  # bow for vertical struts in 2.5D modes

# --- Intersection warning ---
# Curves whose sampled points come within this fraction of the avg strut length
# of each other will trigger a warning. Raise to be more sensitive, lower to
# suppress warnings for near-misses you don't care about.
intersect_threshold = 0.05   # fraction of mean strut length
# Set to False to skip the check entirely (faster for large lattices)
intersect_check = True

# ==========================
# Helpers
# ==========================
def v(x, y, z):
    return np.array([x, y, z], dtype=float)

def unit(a):
    a = np.asarray(a, dtype=float)
    n = np.linalg.norm(a)
    if n < 1e-12:
        return np.zeros_like(a)
    return a / n

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
    t = np.linspace(0.0, 1.0, n)
    return bezier3(P0, P1, P2, P3, t)

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

def draw_bezier_strut(ax, p0, p1, bend_dir, bend, color='cyan', alpha=0.9, lw=1.8,
                      registry=None, label='strut'):
    B0, B1, B2, B3 = bezier_strut(p0, p1, bend_dir=bend_dir, bend=bend)
    pts = sample_bezier3(B0, B1, B2, B3)
    ax.plot(pts[:, 0], pts[:, 1], pts[:, 2], color=color, alpha=alpha, linewidth=lw)
    if registry is not None:
        registry.append((pts, label))

def draw_bezier_filled_polygon(ax, verts_3d, centroid_3d, bend, color, alpha_face, alpha_edge, lw,
                                registry=None, label='polygon'):
    """
    Draw a filled polygon whose edges are each replaced by an inward-bowing
    Bezier strut. The flat face is still drawn for fill; the edges are curves.
    verts_3d: (N, 3) array of polygon vertices in order
    centroid_3d: (3,) center point — edges bow toward this
    """
    n = len(verts_3d)
    # Flat filled face
    ax.add_collection3d(Poly3DCollection([verts_3d], color=color, alpha=alpha_face,
                                          edgecolor='none'))
    # Bezier on every edge, bowing inward toward centroid
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
def check_bezier_intersections(curve_list, threshold_frac=0.05):
    """
    curve_list: list of (pts_array, label_string) where pts_array is (N, 3).
    Checks every non-adjacent pair. If the minimum distance between any two
    curves falls below threshold_frac * mean_strut_length, prints a warning.
    Adjacent curves (sharing an endpoint) are skipped — they meet by design.
    Returns the number of warnings issued.
    """
    if len(curve_list) < 2:
        return 0

    # Estimate mean strut length from all curve endpoint distances
    lengths = [np.linalg.norm(c[0][-1] - c[0][0]) for c in curve_list]
    mean_len = np.mean(lengths) if lengths else 1.0
    threshold = threshold_frac * mean_len

    warnings_issued = 0
    n = len(curve_list)

    for i in range(n):
        pts_i, label_i = curve_list[i]
        ends_i = {tuple(np.round(pts_i[0],  8)),
                  tuple(np.round(pts_i[-1], 8))}

        for j in range(i + 1, n):
            pts_j, label_j = curve_list[j]
            ends_j = {tuple(np.round(pts_j[0],  8)),
                      tuple(np.round(pts_j[-1], 8))}

            # Skip adjacent curves — shared endpoint is intentional
            if ends_i & ends_j:
                continue

            # Minimum distance via broadcasted pairwise norms
            # Shape: (len_i, 1, 3) - (1, len_j, 3) -> (len_i, len_j)
            diff     = pts_i[:, None, :] - pts_j[None, :, :]
            min_dist = np.sqrt((diff ** 2).sum(axis=2)).min()

            if min_dist < threshold:
                warnings_issued += 1
                print(
                    f"  WARNING: curves may intersect — "
                    f"'{label_i}' and '{label_j}' "
                    f"(min dist {min_dist:.4f}, threshold {threshold:.4f})"
                )

    return warnings_issued

# ==========================
# 1) Generate 2D points
# ==========================
def generate_2d_points(n_points, grid=False):
    if grid:
        def factor_pair(n):
            for i in range(int(np.sqrt(n)), 0, -1):
                if n % i == 0:
                    return i, n // i
            return 1, n
        nx2d, ny2d = factor_pair(n_points)
        x = np.linspace(0, 1, nx2d)
        y = np.linspace(0, 1, ny2d)
        xx, yy = np.meshgrid(x, y)
        points = np.vstack([xx.ravel(), yy.ravel()]).T
    else:
        points = np.random.rand(n_points, 2)
    tri = Delaunay(points)
    return points, tri

points_2d, tri_2d = generate_2d_points(n_points, grid=(mode in [4, 5]))

# ==========================
# 2) Generate 3D lattice nodes
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
# 3) Add 2D/2.5D lattice points to 3D nodes
# ==========================
def add_lattice_to_3d(nodes_3d, points_2d, mode, nz_layers=3):
    nodes = nodes_3d.copy()
    pts_norm = (points_2d - points_2d.min(axis=0)) / (
        points_2d.max(axis=0) - points_2d.min(axis=0) + 1e-12
    )
    if mode in [1, 4]:
        for idx, pt in enumerate(pts_norm):
            nodes[f"L2D_{idx}_0"] = np.array([pt[0], pt[1], 0.0])
    elif mode in [2, 5]:
        for z_idx in range(nz_layers):
            z = z_idx / (nz_layers - 1)
            for idx, pt in enumerate(pts_norm):
                nodes[f"L2D_{idx}_{z_idx}"] = np.array([pt[0], pt[1], z])
    return nodes

nodes_combined = add_lattice_to_3d(nodes_3d, points_2d, mode, nz_layers=nz_layers)

# ==========================
# 4) Plotting
# ==========================
def plot_3d_lattice(nodes, points_2d, tri, ratio, mode, nz_layers=3, show_nodes=False):
    fig = plt.figure(figsize=(8, 6))
    ax  = fig.add_subplot(111, projection='3d')

    # Registry collects every drawn Bezier curve for intersection checking
    registry = [] if intersect_check else None

    # ==========================
    # Magenta triangles — Bezier on every edge, bowing inward to centroid
    # ==========================
    for simplex in tri.simplices:
        triangle    = points_2d[simplex]
        centroid_2d = triangle.mean(axis=0)
        tri_pts_2d  = np.array([(1-ratio)*triangle[i] + ratio*centroid_2d for i in range(3)])

        if mode in [1, 4]:
            tri_pts_3d  = np.hstack([tri_pts_2d, np.zeros((3, 1))])
            centroid_3d = np.append(centroid_2d, 0.0)
            draw_bezier_filled_polygon(ax, tri_pts_3d, centroid_3d,
                                       bend=bend_triangle,
                                       color='magenta',
                                       alpha_face=0.25, alpha_edge=0.9, lw=1.8,
                                       registry=registry,
                                       label=f'tri{tuple(simplex)}_z{0}')

        elif mode in [2, 5]:
            for z_idx in range(nz_layers - 1):
                z_bottom    = z_idx / (nz_layers - 1)
                z_top       = (z_idx + 1) / (nz_layers - 1)
                bottom_pts  = np.hstack([tri_pts_2d, np.full((3, 1), z_bottom)])
                top_pts     = np.hstack([tri_pts_2d, np.full((3, 1), z_top)])
                cent_bottom = np.append(centroid_2d, z_bottom)
                cent_top    = np.append(centroid_2d, z_top)

                # Top face
                draw_bezier_filled_polygon(ax, top_pts, cent_top,
                                           bend=bend_triangle,
                                           color='magenta',
                                           alpha_face=0.25, alpha_edge=0.9, lw=1.8,
                                           registry=registry,
                                           label=f'tri{tuple(simplex)}_top_z{z_idx}')
                # Side quads — Bezier on each vertical and lateral edge
                for i in range(3):
                    j    = (i + 1) % 3
                    quad = np.array([bottom_pts[i], bottom_pts[j],
                                     top_pts[j],    top_pts[i]])
                    quad_cent = quad.mean(axis=0)
                    draw_bezier_filled_polygon(ax, quad, quad_cent,
                                               bend=bend_triangle,
                                               color='magenta',
                                               alpha_face=0.15, alpha_edge=0.85, lw=1.5,
                                               registry=registry,
                                               label=f'tri{tuple(simplex)}_side{i}_z{z_idx}')

    # ==========================
    # Cyan Bezier connectors
    # ==========================
    layers = [0] if mode in [1, 4] else list(range(nz_layers))

    for z_idx in layers:
        z_val = 0.0 if mode in [1, 4] else z_idx / (nz_layers - 1)

        groups_white   = {tuple(p): [] for p in points_2d}
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
                p0, p1 = pts_list[0], pts_list[1]
                inward_0 = unit(centroids_list[0] - p0)
                inward_1 = unit(centroids_list[1] - p1)
                bend_dir = unit(inward_0 + inward_1)

                if mode in [1, 4]:
                    draw_bezier_strut(ax, p0, p1,
                                      bend_dir=bend_dir, bend=bend_reentrant,
                                      registry=registry, label=f'strut_{white}')
                else:
                    z_bottom = z_idx / (nz_layers - 1)
                    z_top    = (z_idx + 1) / (nz_layers - 1) if z_idx < nz_layers - 1 else z_bottom
                    p0_bot = np.array([p0[0], p0[1], z_bottom])
                    p1_bot = np.array([p1[0], p1[1], z_bottom])
                    p0_top = np.array([p0[0], p0[1], z_top])
                    p1_top = np.array([p1[0], p1[1], z_top])
                    draw_bezier_strut(ax, p0_bot, p1_bot, bend_dir=bend_dir, bend=bend_reentrant,
                                      registry=registry, label=f'strut_{white}_bot_z{z_idx}')
                    draw_bezier_strut(ax, p0_top, p1_top, bend_dir=bend_dir, bend=bend_reentrant,
                                      registry=registry, label=f'strut_{white}_top_z{z_idx}')
                    mid_xy   = 0.5 * (p0[:2] + p1[:2])
                    vert_dir = unit(np.append(centroids_list[0][:2] - mid_xy, 0))
                    draw_bezier_strut(ax, p0_bot, p0_top, bend_dir=vert_dir, bend=bend_vertical,
                                      registry=registry, label=f'vert0_{white}_z{z_idx}')
                    draw_bezier_strut(ax, p1_bot, p1_top, bend_dir=vert_dir, bend=bend_vertical,
                                      registry=registry, label=f'vert1_{white}_z{z_idx}')

            elif len(pts_list) >= 3:
                try:
                    pts_arr    = np.array(pts_list)
                    hub_center = pts_arr.mean(axis=0)
                    hull       = ConvexHull(pts_arr[:, :2])
                    vertices   = pts_arr[hull.vertices, :2]
                    n_verts    = len(vertices)

                    if mode in [1, 4]:
                        verts_3d = np.hstack([vertices, np.full((n_verts, 1), z_val)])
                        draw_bezier_filled_polygon(ax, verts_3d, hub_center,
                                                   bend=bend_ngon,
                                                   color='cyan',
                                                   alpha_face=0.25, alpha_edge=0.9, lw=1.8,
                                                   registry=registry,
                                                   label=f'hub_{white}_z{z_idx}')
                    else:
                        z_bottom    = z_idx / (nz_layers - 1)
                        z_top       = (z_idx + 1) / (nz_layers - 1) if z_idx < nz_layers - 1 else z_bottom
                        top_verts   = np.hstack([vertices, np.full((n_verts, 1), z_top)])
                        bot_verts   = np.hstack([vertices, np.full((n_verts, 1), z_bottom)])
                        cent_top    = np.append(hub_center[:2], z_top)
                        cent_bot    = np.append(hub_center[:2], z_bottom)

                        draw_bezier_filled_polygon(ax, top_verts, cent_top,
                                                   bend=bend_ngon, color='cyan',
                                                   alpha_face=0.25, alpha_edge=0.9, lw=1.8,
                                                   registry=registry,
                                                   label=f'hub_{white}_top_z{z_idx}')
                        draw_bezier_filled_polygon(ax, bot_verts, cent_bot,
                                                   bend=bend_ngon, color='cyan',
                                                   alpha_face=0.25, alpha_edge=0.9, lw=1.8,
                                                   registry=registry,
                                                   label=f'hub_{white}_bot_z{z_idx}')
                        for i in range(n_verts):
                            j        = (i + 1) % n_verts
                            p0_bot   = np.append(vertices[i], z_bottom)
                            p1_bot   = np.append(vertices[j], z_bottom)
                            p0_top   = np.append(vertices[i], z_top)
                            vert_dir = unit(np.append(hub_center[:2] - vertices[i], 0))
                            draw_bezier_strut(ax, p0_bot, p0_top,
                                              bend_dir=vert_dir, bend=bend_vertical,
                                              registry=registry,
                                              label=f'hub_{white}_vert{i}_z{z_idx}')
                except Exception:
                    pass

    # --- Intersection check ---
    if intersect_check and registry:
        print(f"\nChecking {len(registry)} Bezier curves for intersections "
              f"(threshold = {intersect_threshold} x mean strut length)...")
        n_warnings = check_bezier_intersections(registry, threshold_frac=intersect_threshold)
        if n_warnings == 0:
            print("  No intersections detected.")
        else:
            print(f"  {n_warnings} potential intersection(s) found. "
                  f"Consider reducing bend values in USER SETTINGS.")

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
# 5) Run — single execution
# ==========================
plot_3d_lattice(nodes_combined, points_2d, tri_2d, ratio, mode, nz_layers=nz_layers, show_nodes=False)
