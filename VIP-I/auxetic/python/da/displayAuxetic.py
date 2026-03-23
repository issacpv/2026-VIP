import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay, ConvexHull
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# ==========================
# USER SETTINGS
# ==========================import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay, ConvexHull
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# ==========================
# USER SETTINGS
# ==========================
mode = 2         # 1=random 2D, 2=random 2.5D, 4=grid 2D, 5=grid 2.5D
n_points = 9
ratio = 0.4
nx, ny, nz = 1, 1, 1
cell = 1.0
nz_layers = 3     # number of layers in 2.5D extrusion

# ==========================
# Bezier helpers
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

def sample_bezier3(P0, P1, P2, P3, n=60):
    t = np.linspace(0.0, 1.0, n)
    return bezier3(P0, P1, P2, P3, t)

def bezier_curve_between(p0, p1, bend_factor=0.1):
    mid = (p0 + p1) / 2
    d = p1 - p0
    bend_dir = np.array([-d[1], d[0]])  # perpendicular in XY
    bend_dir = unit(np.append(bend_dir, 0))
    offset = bend_factor * np.linalg.norm(d) * bend_dir
    P0 = p0
    P1 = p0 + d/3 + offset
    P2 = p0 + 2*d/3 + offset
    P3 = p1
    return sample_bezier3(P0, P1, P2, P3, n=30)

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

points_2d, tri_2d = generate_2d_points(n_points, grid=(mode in [4,5]))

# ==========================
# 2) Generate 3D lattice nodes
# ==========================
def generate_3d_nodes(nx, ny, nz, cell):
    nodes = {}
    for i in range(nx+1):
        for j in range(ny+1):
            for k in range(nz+1):
                nodes[f"G_{i}_{j}_{k}"] = np.array([i*cell/nx, j*cell/ny, k*cell/nz], dtype=float)
    return nodes

nodes_3d = generate_3d_nodes(nx, ny, nz, cell)

# ==========================
# 3) Add 2D/2.5D lattice points to 3D nodes
# ==========================
def add_lattice_to_3d(nodes_3d, points_2d, mode, nz_layers=3):
    nodes = nodes_3d.copy()
    pts_norm = (points_2d - points_2d.min(axis=0)) / (points_2d.max(axis=0) - points_2d.min(axis=0) + 1e-12)
    
    if mode in [1,4]:  # simple 2D lattice
        for idx, pt in enumerate(pts_norm):
            nodes[f"L2D_{idx}_0"] = np.array([pt[0], pt[1], 0.0])
    elif mode in [2,5]:  # extrude in Z
        for z_idx in range(nz_layers):
            z = z_idx / (nz_layers - 1)
            for idx, pt in enumerate(pts_norm):
                nodes[f"L2D_{idx}_{z_idx}"] = np.array([pt[0], pt[1], z])
    return nodes

nodes_combined = add_lattice_to_3d(nodes_3d, points_2d, mode, nz_layers=nz_layers)

# ==========================
# 4) Plotting 3D lattice with Bezier curves
# ==========================
def plot_3d_lattice(nodes, points_2d, tri, ratio, mode, show_nodes=False):
    fig = plt.figure(figsize=(8,6))
    ax = fig.add_subplot(111, projection='3d')

    # --- Magenta triangles ---
    for simplex in tri.simplices:
        triangle = points_2d[simplex]
        centroid = triangle.mean(axis=0)
        tri_pts_2d = np.array([(1-ratio)*triangle[i] + ratio*centroid for i in range(3)])
        
        if mode in [1,4]:
            tri_pts_3d = np.hstack([tri_pts_2d, np.zeros((3,1))])
            ax.add_collection3d(Poly3DCollection([tri_pts_3d], color='magenta', alpha=0.5))
        elif mode in [2,5]:
            for z_idx in range(nz_layers-1):
                z_bottom = z_idx / (nz_layers-1)
                z_top = (z_idx+1) / (nz_layers-1)
                bottom_pts = np.hstack([tri_pts_2d, np.full((3,1), z_bottom)])
                top_pts = np.hstack([tri_pts_2d, np.full((3,1), z_top)])
                ax.add_collection3d(Poly3DCollection([top_pts], color='magenta', alpha=0.5))
                for i in range(3):
                    j = (i+1)%3
                    quad = [bottom_pts[i], bottom_pts[j], top_pts[j], top_pts[i]]
                    ax.add_collection3d(Poly3DCollection([quad], color='magenta', alpha=0.5))

    # --- Cyan points and Bezier connections ---
    layers = [0] if mode in [1,4] else list(range(nz_layers))
    for z_idx in layers:
        z_val = 0.0 if mode in [1,4] else z_idx / (nz_layers-1)
        groups_white = {tuple(p): [] for p in points_2d}
        for simplex in tri.simplices:
            triangle = points_2d[simplex]
            centroid = triangle.mean(axis=0)
            tri_pts_2d = np.array([(1-ratio)*triangle[i] + ratio*centroid for i in range(3)])
            tri_pts_3d = np.hstack([tri_pts_2d, np.full((3,1), z_val)])
            for i, vertex in enumerate(triangle):
                groups_white[tuple(vertex)].append(tri_pts_3d[i])

        # Determine connections
        for white, pts_list in groups_white.items():
            if len(pts_list) == 2:
                # Bezier curve for line
                p0, p1 = pts_list
                bez_pts = bezier_curve_between(p0, p1)
                if mode in [1,4]:
                    ax.plot(bez_pts[:,0], bez_pts[:,1], bez_pts[:,2], color='cyan', alpha=0.8)
                else:
                    # extrude along Z
                    for z_idx in range(nz_layers-1):
                        z_bottom = z_idx / (nz_layers-1)
                        z_top = (z_idx+1) / (nz_layers-1)
                        bottom = np.hstack([bez_pts[:,:2], np.full((bez_pts.shape[0],1), z_bottom)])
                        top = np.hstack([bez_pts[:,:2], np.full((bez_pts.shape[0],1), z_top)])
                        for i in range(len(bez_pts)-1):
                            quad = [bottom[i], bottom[i+1], top[i+1], top[i]]
                            ax.add_collection3d(Poly3DCollection([quad], color='cyan', alpha=0.3))
            elif len(pts_list) >= 3:
                # convex hull extruded prism
                try:
                    hull = ConvexHull(np.array(pts_list)[:,:2])
                    vertices = np.array(pts_list)[hull.vertices,:2]
                    for i in range(len(vertices)):
                        j = (i+1)%len(vertices)
                        bottom = np.hstack([vertices[i], 0])
                        bottom_next = np.hstack([vertices[j], 0])
                        top = np.hstack([vertices[i], 1])
                        top_next = np.hstack([vertices[j], 1])
                        quad = [bottom, bottom_next, top_next, top]
                        ax.add_collection3d(Poly3DCollection([quad], color='cyan', alpha=0.3))
                    top_face = np.hstack([vertices, np.ones((len(vertices),1))])
                    bottom_face = np.hstack([vertices, np.zeros((len(vertices),1))])
                    ax.add_collection3d(Poly3DCollection([top_face], color='cyan', alpha=0.3))
                    ax.add_collection3d(Poly3DCollection([bottom_face], color='cyan', alpha=0.3))
                except:
                    pass

    # --------------------------
    # 3D view settings
    # --------------------------
    ax.set_box_aspect([1,1,1])
    ax.set_xlim(0,1)
    ax.set_ylim(0,1)
    ax.set_zlim(0,1)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    elev, azim = 30, 45
    ax.view_init(elev=elev, azim=azim)

    # --------------------------
    # interactive rotation
    # --------------------------
    def on_key(event):
        nonlocal elev, azim
        step = 10
        if event.key == 'up':
            elev += step
        elif event.key == 'down':
            elev -= step
        elif event.key == 'left':
            azim -= step
        elif event.key == 'right':
            azim += step
        elif event.key == 'x':
            elev, azim = 0, 0
        elif event.key == 'y':
            elev, azim = 0, 90
        elif event.key == 'z':
            elev, azim = 90, -90
        ax.view_init(elev=elev, azim=azim)
        fig.canvas.draw_idle()

    fig.canvas.mpl_connect('key_press_event', on_key)
    plt.show()

# ==========================
# 5) Run
# ==========================
plot_3d_lattice(nodes_combined, points_2d, tri_2d, ratio, mode, show_nodes=False)
mode = 5         # 1=random 2D, 2=random 2.5D, 4=grid 2D, 5=grid 2.5D
n_points = 9
ratio = 0.4
nx, ny, nz = 1, 1, 1
cell = 1.0
nz_layers = 3     # number of layers in 2.5D extrusion

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

points_2d, tri_2d = generate_2d_points(n_points, grid=(mode in [4,5]))

# ==========================
# 2) Generate 3D lattice nodes
# ==========================
def generate_3d_nodes(nx, ny, nz, cell):
    nodes = {}
    for i in range(nx+1):
        for j in range(ny+1):
            for k in range(nz+1):
                nodes[f"G_{i}_{j}_{k}"] = np.array([i*cell/nx, j*cell/ny, k*cell/nz], dtype=float)
    return nodes

nodes_3d = generate_3d_nodes(nx, ny, nz, cell)

# ==========================
# 3) Add 2D/2.5D lattice points to 3D nodes
# ==========================
def add_lattice_to_3d(nodes_3d, points_2d, mode, nz_layers=3):
    nodes = nodes_3d.copy()
    pts_norm = (points_2d - points_2d.min(axis=0)) / (points_2d.max(axis=0) - points_2d.min(axis=0) + 1e-12)
    
    if mode in [1,4]:  # simple 2D lattice
        for idx, pt in enumerate(pts_norm):
            nodes[f"L2D_{idx}_0"] = np.array([pt[0], pt[1], 0.0])
    elif mode in [2,5]:  # extrude in Z
        for z_idx in range(nz_layers):
            z = z_idx / (nz_layers - 1)
            for idx, pt in enumerate(pts_norm):
                nodes[f"L2D_{idx}_{z_idx}"] = np.array([pt[0], pt[1], z])
    return nodes

nodes_combined = add_lattice_to_3d(nodes_3d, points_2d, mode, nz_layers=nz_layers)

# ==========================
# 4) Plotting 3D lattice
# ==========================
def plot_3d_lattice(nodes, points_2d, tri, ratio, mode, show_nodes=False):
    fig = plt.figure(figsize=(8,6))
    ax = fig.add_subplot(111, projection='3d')

    # --- Magenta triangles (unchanged) ---
    for simplex in tri.simplices:
        triangle = points_2d[simplex]
        centroid = triangle.mean(axis=0)
        tri_pts_2d = np.array([(1-ratio)*triangle[i] + ratio*centroid for i in range(3)])
        
        if mode in [1,4]:
            tri_pts_3d = np.hstack([tri_pts_2d, np.zeros((3,1))])
            ax.add_collection3d(Poly3DCollection([tri_pts_3d], color='magenta', alpha=0.5))
        elif mode in [2,5]:
            for z_idx in range(nz_layers-1):
                z_bottom = z_idx / (nz_layers-1)
                z_top = (z_idx+1) / (nz_layers-1)
                bottom_pts = np.hstack([tri_pts_2d, np.full((3,1), z_bottom)])
                top_pts = np.hstack([tri_pts_2d, np.full((3,1), z_top)])
                ax.add_collection3d(Poly3DCollection([top_pts], color='magenta', alpha=0.5))
                for i in range(3):
                    j = (i+1)%3
                    quad = [bottom_pts[i], bottom_pts[j], top_pts[j], top_pts[i]]
                    ax.add_collection3d(Poly3DCollection([quad], color='magenta', alpha=0.5))

    # --- Cyan points and extrusions ---
    if mode in [1,2,4,5]:
        layers = [0] if mode in [1,4] else list(range(nz_layers))
        for z_idx in layers:
            z_val = 0.0 if mode in [1,4] else z_idx / (nz_layers-1)
            groups_white = {tuple(p): [] for p in points_2d}
            for simplex in tri.simplices:
                triangle = points_2d[simplex]
                centroid = triangle.mean(axis=0)
                tri_pts_2d = np.array([(1-ratio)*triangle[i] + ratio*centroid for i in range(3)])
                tri_pts_3d = np.hstack([tri_pts_2d, np.full((3,1), z_val)])
                for pt in tri_pts_3d:
                    ax.scatter(*pt, color='cyan', s=20)
                for i, vertex in enumerate(triangle):
                    groups_white[tuple(vertex)].append(tri_pts_3d[i])

            # Determine connections
            for white, pts_list in groups_white.items():
                if len(pts_list) == 2:
                    # two points → extrude rectangle
                    p0, p1 = pts_list
                    if mode in [1,4]:
                        ax.plot([p0[0], p1[0]], [p0[1], p1[1]], [p0[2], p1[2]], color='cyan', alpha=0.8)
                    else:
                        # extrude as rectangle along z
                        rect = np.array([
                            [p0[0], p0[1], 0],
                            [p1[0], p1[1], 0],
                            [p1[0], p1[1], 1],
                            [p0[0], p0[1], 1]
                        ])
                        ax.add_collection3d(Poly3DCollection([rect], color='cyan', alpha=0.3))
                        ax.plot([p0[0], p1[0]], [p0[1], p1[1]], [0,0], color='cyan', alpha=0.8)
                        ax.plot([p0[0], p1[0]], [p0[1], p1[1]], [1,1], color='cyan', alpha=0.8)
                elif len(pts_list) >= 3:
                    # 3+ points → convex hull extruded prism
                    try:
                        hull = ConvexHull(np.array(pts_list)[:,:2])
                        vertices = np.array(pts_list)[hull.vertices,:2]
                        for i in range(len(vertices)):
                            j = (i+1)%len(vertices)
                            bottom = np.hstack([vertices[i], 0])
                            bottom_next = np.hstack([vertices[j], 0])
                            top = np.hstack([vertices[i], 1])
                            top_next = np.hstack([vertices[j], 1])
                            # vertical wall
                            quad = [bottom, bottom_next, top_next, top]
                            ax.add_collection3d(Poly3DCollection([quad], color='cyan', alpha=0.3))
                        # top and bottom faces
                        top_face = np.hstack([vertices, np.ones((len(vertices),1))])
                        bottom_face = np.hstack([vertices, np.zeros((len(vertices),1))])
                        ax.add_collection3d(Poly3DCollection([top_face], color='cyan', alpha=0.3))
                        ax.add_collection3d(Poly3DCollection([bottom_face], color='cyan', alpha=0.3))
                    except:
                        pass

    # --------------------------
    # 3D view settings
    # --------------------------
    ax.set_box_aspect([1,1,1])
    ax.set_xlim(0,1)
    ax.set_ylim(0,1)
    ax.set_zlim(0,1)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    elev, azim = 30, 45
    ax.view_init(elev=elev, azim=azim)

    # --------------------------
    # interactive rotation
    # --------------------------
    def on_key(event):
        nonlocal elev, azim
        step = 10
        if event.key == 'up':
            elev += step
        elif event.key == 'down':
            elev -= step
        elif event.key == 'left':
            azim -= step
        elif event.key == 'right':
            azim += step
        elif event.key == 'x':
            elev, azim = 0, 0
        elif event.key == 'y':
            elev, azim = 0, 90
        elif event.key == 'z':
            elev, azim = 90, -90
        ax.view_init(elev=elev, azim=azim)
        fig.canvas.draw_idle()

    fig.canvas.mpl_connect('key_press_event', on_key)
    plt.show()

# ==========================
# 5) Run
# ==========================
plot_3d_lattice(nodes_combined, points_2d, tri_2d, ratio, mode, show_nodes=False)
