
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# ============================================================
# 1) Basic helpers
# ============================================================
def v(x, y, z):
    return np.array([x, y, z], dtype=float)

def unit(a):
    a = np.asarray(a, dtype=float)
    n = np.linalg.norm(a)
    if n < 1e-12:
        return np.zeros_like(a)
    return a / n

# ============================================================
# 2) Bezier helpers
# ============================================================
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

def bezier_strut(P0, P3, bend_dir, bend=0.10):
    P0 = np.asarray(P0, float)
    P3 = np.asarray(P3, float)

    d = P3 - P0
    L = np.linalg.norm(d)
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

# ============================================================
# 3) Connected lattice generator
# ============================================================
def build_connected_auxetic_lattice(nx=1, ny=1, nz=1, cell=1.0, reentrant=0.28):
    nodes = {}
    members = []

    def corner_name(i, j, k):
        return f"G_{i}_{j}_{k}"

    for i in range(nx + 1):
        for j in range(ny + 1):
            for k in range(nz + 1):
                nodes[corner_name(i, j, k)] = v(i * cell, j * cell, k * cell)

    added_edges = set()

    def add_member(a, b, kind):
        if a == b:
            return
        key = tuple(sorted((a, b)) + [kind])
        if key not in added_edges:
            members.append((a, b, kind))
            added_edges.add(key)

    for i in range(nx + 1):
        for j in range(ny + 1):
            for k in range(nz + 1):
                if i < nx:
                    add_member(corner_name(i, j, k), corner_name(i + 1, j, k), "frame")
                if j < ny:
                    add_member(corner_name(i, j, k), corner_name(i, j + 1, k), "frame")
                if k < nz:
                    add_member(corner_name(i, j, k), corner_name(i, j, k + 1), "frame")

    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                c000 = corner_name(i,     j,     k)
                c100 = corner_name(i + 1, j,     k)
                c010 = corner_name(i,     j + 1, k)
                c110 = corner_name(i + 1, j + 1, k)
                c001 = corner_name(i,     j,     k + 1)
                c101 = corner_name(i + 1, j,     k + 1)
                c011 = corner_name(i,     j + 1, k + 1)
                c111 = corner_name(i + 1, j + 1, k + 1)

                corners = {
                    "000": nodes[c000], "100": nodes[c100], "010": nodes[c010], "110": nodes[c110],
                    "001": nodes[c001], "101": nodes[c101], "011": nodes[c011], "111": nodes[c111],
                }

                center = sum(corners.values()) / 8.0
                r = reentrant * cell

                face_defs = {
                    "xm": ("000", "010", "011", "001"),
                    "xp": ("100", "110", "111", "101"),
                    "ym": ("000", "100", "101", "001"),
                    "yp": ("010", "110", "111", "011"),
                    "zm": ("000", "100", "110", "010"),
                    "zp": ("001", "101", "111", "011"),
                }

                face_nodes = {}
                for fname, ids in face_defs.items():
                    fc = sum(corners[s] for s in ids) / 4.0
                    inward = unit(center - fc)
                    p = fc + r * inward
                    nname = f"F_{i}_{j}_{k}_{fname}"
                    nodes[nname] = p
                    face_nodes[fname] = nname

                    for cid in ids:
                        gname = {
                            "000": c000, "100": c100, "010": c010, "110": c110,
                            "001": c001, "101": c101, "011": c011, "111": c111
                        }[cid]
                        add_member(nname, gname, "reentrant")

                add_member(face_nodes["xm"], face_nodes["xp"], "inner")
                add_member(face_nodes["ym"], face_nodes["yp"], "inner")
                add_member(face_nodes["zm"], face_nodes["zp"], "inner")

                add_member(face_nodes["xm"], face_nodes["ym"], "inner_diag")
                add_member(face_nodes["xm"], face_nodes["zm"], "inner_diag")
                add_member(face_nodes["xp"], face_nodes["yp"], "inner_diag")
                add_member(face_nodes["xp"], face_nodes["zp"], "inner_diag")
                add_member(face_nodes["ym"], face_nodes["zm"], "inner_diag")
                add_member(face_nodes["yp"], face_nodes["zp"], "inner_diag")

    return nodes, members

# ============================================================
# 4) Build Bezier curves from members
# ============================================================
def build_bezier_curves(nodes, members,
                        bend_frame=0.02,
                        bend_reentrant=0.10,
                        bend_inner=0.06,
                        bend_inner_diag=0.08):
    curves = []
    all_pts = np.stack(list(nodes.values()), axis=0)
    global_center = all_pts.mean(axis=0)

    for a, b, kind in members:
        P0 = nodes[a]
        P3 = nodes[b]
        mid = 0.5 * (P0 + P3)
        radial = mid - global_center

        if kind == "frame":
            bend = bend_frame
            bend_dir = np.cross(P3 - P0, v(0, 0, 1))
            if np.linalg.norm(bend_dir) < 1e-12:
                bend_dir = v(0, 1, 0)
        elif kind == "reentrant":
            bend = bend_reentrant
            bend_dir = -radial + v(0.2, 0.3, 0.4)
        elif kind == "inner":
            bend = bend_inner
            bend_dir = radial + v(0.0, 1.0, 0.5)
        else:
            bend = bend_inner_diag
            bend_dir = np.cross(P3 - P0, radial + v(0, 1, 0))
            if np.linalg.norm(bend_dir) < 1e-12:
                bend_dir = radial + v(1, 0, 0)

        B0, B1, B2, B3 = bezier_strut(P0, P3, bend_dir=bend_dir, bend=bend)
        pts = sample_bezier3(B0, B1, B2, B3, n=70)
        curves.append((pts, kind))

    return curves

# ============================================================
# 5) Filled geometry helpers
# ============================================================
def get_cell_geometry(nodes, i, j, k):
    c000 = nodes[f"G_{i}_{j}_{k}"]
    c100 = nodes[f"G_{i+1}_{j}_{k}"]
    c010 = nodes[f"G_{i}_{j+1}_{k}"]
    c110 = nodes[f"G_{i+1}_{j+1}_{k}"]
    c001 = nodes[f"G_{i}_{j}_{k+1}"]
    c101 = nodes[f"G_{i+1}_{j}_{k+1}"]
    c011 = nodes[f"G_{i}_{j+1}_{k+1}"]
    c111 = nodes[f"G_{i+1}_{j+1}_{k+1}"]

    corners = {
        "000": c000, "100": c100, "010": c010, "110": c110,
        "001": c001, "101": c101, "011": c011, "111": c111,
    }

    face_nodes = {
        "xm": nodes[f"F_{i}_{j}_{k}_xm"],
        "xp": nodes[f"F_{i}_{j}_{k}_xp"],
        "ym": nodes[f"F_{i}_{j}_{k}_ym"],
        "yp": nodes[f"F_{i}_{j}_{k}_yp"],
        "zm": nodes[f"F_{i}_{j}_{k}_zm"],
        "zp": nodes[f"F_{i}_{j}_{k}_zp"],
    }

    cube_faces = [
        [c000, c010, c011, c001],
        [c100, c110, c111, c101],
        [c000, c100, c101, c001],
        [c010, c110, c111, c011],
        [c000, c100, c110, c010],
        [c001, c101, c111, c011],
    ]

    face_defs = {
        "xm": ("000", "010", "011", "001"),
        "xp": ("100", "110", "111", "101"),
        "ym": ("000", "100", "101", "001"),
        "yp": ("010", "110", "111", "011"),
        "zm": ("000", "100", "110", "010"),
        "zp": ("001", "101", "111", "011"),
    }

    face_pyramids = []
    for fname, ids in face_defs.items():
        fp = face_nodes[fname]
        quad = [corners[s] for s in ids]
        for a in range(4):
            face_pyramids.append([fp, quad[a], quad[(a + 1) % 4]])

    octa_faces = [
        [face_nodes["xp"], face_nodes["yp"], face_nodes["zp"]],
        [face_nodes["xp"], face_nodes["zp"], face_nodes["ym"]],
        [face_nodes["xp"], face_nodes["ym"], face_nodes["zm"]],
        [face_nodes["xp"], face_nodes["zm"], face_nodes["yp"]],
        [face_nodes["xm"], face_nodes["yp"], face_nodes["zp"]],
        [face_nodes["xm"], face_nodes["zp"], face_nodes["ym"]],
        [face_nodes["xm"], face_nodes["ym"], face_nodes["zm"]],
        [face_nodes["xm"], face_nodes["zm"], face_nodes["yp"]],
    ]

    return cube_faces, face_pyramids, octa_faces

# ============================================================
# 6) Simple extent / apparent Poisson check
# ============================================================
def extents(nodes):
    P = np.stack(list(nodes.values()), axis=0)
    mins = P.min(axis=0)
    maxs = P.max(axis=0)
    return maxs - mins

def compare_states(nx, ny, nz, cell, r1, r2):
    n1, _ = build_connected_auxetic_lattice(nx, ny, nz, cell=cell, reentrant=r1)
    n2, _ = build_connected_auxetic_lattice(nx, ny, nz, cell=cell, reentrant=r2)

    e1 = extents(n1)
    e2 = extents(n2)

    eps_x = (e2[0] - e1[0]) / (e1[0] + 1e-12)
    eps_z = (e2[2] - e1[2]) / (e1[2] + 1e-12)
    nu = -eps_x / (eps_z + 1e-12)

    return e1, e2, eps_x, eps_z, nu

# ============================================================
# 7) Plotting
# ============================================================
def set_axes_equal(ax):
    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    y_range = abs(y_limits[1] - y_limits[0])
    z_range = abs(z_limits[1] - z_limits[0])

    x_mid = np.mean(x_limits)
    y_mid = np.mean(y_limits)
    z_mid = np.mean(z_limits)

    r = 0.5 * max([x_range, y_range, z_range])

    ax.set_xlim3d([x_mid - r, x_mid + r])
    ax.set_ylim3d([y_mid - r, y_mid + r])
    ax.set_zlim3d([z_mid - r, z_mid + r])

def plot_lattice(nodes, curves, nx, ny, nz,
                 title="Connected 3D auxetic-style lattice",
                 show_nodes=False,
                 fill_external=True,
                 fill_internal=True):
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax.set_title(title)

    # --- Camera vectors ---
    camera_dir = np.array([1, 1, 1], dtype=float)
    camera_up = np.array([0, 0, 1], dtype=float)
    step_deg = 10

    def rotate_vector(v, axis, angle_deg):
        """Rotate vector v about axis by angle_deg degrees."""
        angle = np.radians(angle_deg)
        axis = axis / np.linalg.norm(axis)
        cos = np.cos(angle)
        sin = np.sin(angle)
        cross = np.cross(axis, v)
        dot = np.dot(axis, v)
        return v * cos + cross * sin + axis * dot * (1 - cos)

    # --- Key handler using arrow keys ---
    def on_key(event):
        nonlocal camera_dir, camera_up
        event.guiEvent = None  # override matplotlib default bindings

        right = np.cross(camera_dir, camera_up)

        if event.key == 'up':
            # Rotate downward
            camera_dir = rotate_vector(camera_dir, right, -step_deg)
            camera_up = rotate_vector(camera_up, right, -step_deg)
        elif event.key == 'down':
            # Rotate upward
            camera_dir = rotate_vector(camera_dir, right, step_deg)
            camera_up = rotate_vector(camera_up, right, step_deg)
        elif event.key == 'left':
            # Rotate left
            camera_dir = rotate_vector(camera_dir, camera_up, step_deg)
        elif event.key == 'right':
            # Rotate right
            camera_dir = rotate_vector(camera_dir, camera_up, -step_deg)
        elif event.key == 'x':
            camera_dir = np.array([1, 0, 0])
            camera_up = np.array([0, 0, 1])
        elif event.key == 'y':
            camera_dir = np.array([0, 1, 0])
            camera_up = np.array([0, 0, 1])
        elif event.key == 'z':
            camera_dir = np.array([0, 0, 1])
            camera_up = np.array([0, 1, 0])
        elif event.key == 'q':
            plt.close('all')
            exit()

        # Apply updated view
        azim = np.degrees(np.arctan2(camera_dir[1], camera_dir[0]))
        elev = np.degrees(np.arcsin(camera_dir[2] / np.linalg.norm(camera_dir)))
        ax.view_init(elev=elev, azim=azim)
        plt.draw()

    fig.canvas.mpl_connect('key_press_event', on_key)

    # --- Draw geometry ---
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                cube_faces, face_pyramids, octa_faces = get_cell_geometry(nodes, i, j, k)

                if fill_external:
                    ax.add_collection3d(
                        Poly3DCollection(
                            cube_faces,
                            facecolor="lightgray",
                            edgecolor="black",
                            linewidths=0.6,
                            alpha=0.08
                        )
                    )

                if fill_internal:
                    ax.add_collection3d(
                        Poly3DCollection(
                            face_pyramids,
                            facecolor="cornflowerblue",
                            edgecolor="none",
                            alpha=0.12
                        )
                    )
                    ax.add_collection3d(
                        Poly3DCollection(
                            octa_faces,
                            facecolor="crimson",
                            edgecolor="none",
                            alpha=0.28
                        )
                    )

    for pts, kind in curves:
        if kind == "frame":
            lw = 1.0
            alpha = 0.30
        elif kind == "reentrant":
            lw = 2.0
            alpha = 0.85
        elif kind == "inner":
            lw = 1.5
            alpha = 0.75
        else:
            lw = 1.2
            alpha = 0.65

        ax.plot(pts[:, 0], pts[:, 1], pts[:, 2], linewidth=lw, alpha=alpha)

    if show_nodes:
        P = np.stack(list(nodes.values()), axis=0)
        ax.scatter(P[:, 0], P[:, 1], P[:, 2], s=10)

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    # Make cube proportions correct
    set_axes_equal(ax)
    ax.set_box_aspect([1,1,1])
    plt.tight_layout()
    plt.show()

# ============================================================
# 8) Run
# ============================================================
if __name__ == "__main__":
    nx, ny, nz = 1, 1, 1
    cell = 1.0
    reentrant = 0.30

    nodes, members = build_connected_auxetic_lattice(
        nx=nx, ny=ny, nz=nz, cell=cell, reentrant=reentrant
    )

    curves = build_bezier_curves(nodes, members)

    plot_lattice(
        nodes, curves, nx, ny, nz,
        title=f"Single auxetic unit cell (nx={nx}, ny={ny}, nz={nz}, r={reentrant})",
        show_nodes=False,
        fill_external=True,
        fill_internal=True
    )

    e1, e2, eps_x, eps_z, nu = compare_states(
        nx=nx, ny=ny, nz=nz, cell=cell, r1=0.18, r2=0.34
    )
    print("State 1 extents [x, y, z]:", e1)
    print("State 2 extents [x, y, z]:", e2)
    print("Lateral strain x:", eps_x)
    print("Axial strain z:", eps_z)
    print("Apparent Poisson ratio:", nu)
