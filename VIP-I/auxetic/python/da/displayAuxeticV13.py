import numpy as np
import matplotlib.pyplot as plt
from itertools import permutations
from matplotlib.colors import to_rgba
from scipy.spatial import Delaunay, ConvexHull
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

try:
    from stl import mesh as stl_mesh
    HAS_STL = True
except ImportError:
    HAS_STL = False

# ==========================
# USER SETTINGS
# ==========================
mode        = 6      # 1=random 2D, 2=random 2.5D, 3=random 3D
                     # 4=grid 2D,   5=grid 2.5D,   6=grid 3D
n_points    = 16
ratio       = 0.5
nx, ny, nz  = 1, 1, 1
cell        = 1.0
nz_layers   = 2      # layers in 2.5D extrusion (modes 2, 5 only)

# --- Shape settings ---
ngon_thickness = 0.03   # extrusion depth of n-gon solids (lattice units)
hub_scale      = 0.45   # truncated octahedron size relative to mean strut reach
                         # ~0.45 = gaps between hubs, ~0.5 = nearly touching

# --- Bezier curve settings ---
bezier_bulge   = 0.18   # how far edges bow outward (0 = straight, 0.3 = strongly curved)
bezier_samples = 12     # number of segments per curved edge (higher = smoother)

# --- Export settings ---
export_enabled = True
export_scad    = False
export_stl     = True
export_obj     = True
export_scad_path = "auxetic_lattice.scad"
export_stl_path  = "auxetic_lattice.stl"
export_obj_path  = "auxetic_lattice.obj"

strut_radius   = 0.02   # tube radius for exported struts (world units)
face_thickness = 0.015  # prism thickness for exported faces (world units)
scad_segments  = 8      # cylinder sides in SCAD ($fn)

# --- Visualization ---
show_plot = False   # set True to open the matplotlib window


# ==========================
# BEZIER CURVE UTILITIES
# ==========================

def quadratic_bezier(p0, p1, p_ctrl, n_samples):
    """
    Quadratic Bézier from p0 to p1 with control point p_ctrl.
    Returns (n_samples+1) points including both endpoints.
    """
    t = np.linspace(0.0, 1.0, n_samples + 1)
    t  = t[:, np.newaxis]
    pts = (1 - t)**2 * p0 + 2 * (1 - t) * t * p_ctrl + t**2 * p1
    return pts


def bezier_edge_3d(p0, p1, centroid, bulge=0.18, n_samples=12):
    """
    Curved edge between p0 and p1 that bows *away* from centroid.
    The control point is the edge midpoint pushed outward by `bulge`
    (as a fraction of the edge length, in the direction away from centroid).
    Returns array of shape (n_samples+1, 3).
    """
    p0 = np.asarray(p0, float)
    p1 = np.asarray(p1, float)
    centroid = np.asarray(centroid, float)

    mid = 0.5 * (p0 + p1)
    outward = mid - centroid
    out_len = np.linalg.norm(outward)

    if out_len < 1e-12:
        # fallback: bow along edge normal in the plane
        edge = p1 - p0
        outward = np.array([-edge[1], edge[0], 0.0] if len(edge) >= 3 else [-edge[1], edge[0]])
        out_len = np.linalg.norm(outward)
        if out_len < 1e-12:
            outward = np.zeros_like(mid)
            out_len = 1.0

    edge_len = np.linalg.norm(p1 - p0)
    ctrl = mid + (outward / out_len) * bulge * edge_len

    return quadratic_bezier(p0, p1, ctrl, n_samples)


def bezier_polygon_rings_3d(verts_3d, bulge=0.18, n_samples=12):
    """
    Replace each edge of a polygon with a Bézier curve.
    Returns a flat list of 3D points tracing the curved outline (open, no repeat of first).
    Also returns segments: list of (n_samples+1, 3) arrays, one per edge.
    """
    verts_3d = np.asarray(verts_3d, float)
    n = len(verts_3d)
    centroid = verts_3d.mean(axis=0)
    segments = []
    for i in range(n):
        j = (i + 1) % n
        seg = bezier_edge_3d(verts_3d[i], verts_3d[j], centroid, bulge, n_samples)
        segments.append(seg)
    # Chain: for each segment skip the last point (avoid duplicate at junction)
    chain = np.vstack([seg[:-1] for seg in segments])
    return chain, segments


def bezier_polygon_fans(verts_3d, bulge=0.18, n_samples=12):
    """
    Tessellate a curved polygon into triangles by fan from centroid.
    Each 'slice' between adjacent Bézier samples on an edge becomes a triangle
    with the polygon centroid.
    Returns list of triangles (each a (3,3) array).
    """
    chain, segments = bezier_polygon_rings_3d(verts_3d, bulge, n_samples)
    centroid = np.asarray(verts_3d, float).mean(axis=0)
    triangles = []
    for seg in segments:
        for k in range(len(seg) - 1):
            triangles.append(np.array([centroid, seg[k], seg[k + 1]]))
    return triangles


# ==========================
# TRIANGULATION
# ==========================

def triangulate_3d_grid_symmetric(nx, ny, nz):
    """
    6-tetrahedra-per-cube BCC decomposition.
    Body diagonal of each cube points TOWARD the global lattice center,
    giving 3-fold symmetry per cube and consistent hub geometry at all nodes.
    """
    x = np.linspace(0, 1, nx)
    y = np.linspace(0, 1, ny)
    z = np.linspace(0, 1, nz)
    xx, yy, zz = np.meshgrid(x, y, z, indexing='ij')
    points = np.vstack([xx.ravel(), yy.ravel(), zz.ravel()]).T

    def idx(i, j, k):
        return i * (ny * nz) + j * nz + k

    cx, cy, cz = (nx - 1) / 2.0, (ny - 1) / 2.0, (nz - 1) / 2.0
    tetrahedra = []

    for i in range(nx - 1):
        for j in range(ny - 1):
            for k in range(nz - 1):
                c000 = idx(i,   j,   k  )
                c001 = idx(i,   j,   k+1)
                c010 = idx(i,   j+1, k  )
                c011 = idx(i,   j+1, k+1)
                c100 = idx(i+1, j,   k  )
                c101 = idx(i+1, j,   k+1)
                c110 = idx(i+1, j+1, k  )
                c111 = idx(i+1, j+1, k+1)

                sx = (i + 0.5) >= cx
                sy = (j + 0.5) >= cy
                sz = (k + 0.5) >= cz

                if sx and sy and sz:
                    a, b = c111, c000
                    ring = [c110, c010, c011, c001, c101, c100]
                elif not sx and sy and sz:
                    a, b = c011, c100
                    ring = [c001, c000, c010, c110, c111, c101]
                elif sx and not sy and sz:
                    a, b = c101, c010
                    ring = [c100, c000, c001, c011, c111, c110]
                elif sx and sy and not sz:
                    a, b = c110, c001
                    ring = [c111, c011, c010, c000, c100, c101]
                elif not sx and not sy and sz:
                    a, b = c001, c110
                    ring = [c000, c100, c101, c111, c011, c010]
                elif not sx and sy and not sz:
                    a, b = c010, c101
                    ring = [c011, c111, c110, c100, c000, c001]
                elif sx and not sy and not sz:
                    a, b = c100, c011
                    ring = [c110, c111, c101, c001, c000, c010]
                else:
                    a, b = c000, c111
                    ring = [c100, c110, c010, c011, c001, c101]

                for r in range(6):
                    tetrahedra.append([a, b, ring[r], ring[(r + 1) % 6]])

    class MockTri:
        def __init__(self, s): self.simplices = np.array(s)

    return points, MockTri(tetrahedra)


def triangulate_grid_symmetric(nx, ny):
    """
    2D grid triangulation with center-aligned diagonals (pinwheel pattern).
    """
    x = np.linspace(0, 1, nx)
    y = np.linspace(0, 1, ny)
    xx, yy = np.meshgrid(x, y)
    points = np.vstack([xx.ravel(), yy.ravel()]).T

    def idx(i, j): return j * nx + i

    center = np.array([0.5, 0.5])
    triangles = []

    for i in range(nx - 1):
        for j in range(ny - 1):
            bl, br = idx(i, j), idx(i + 1, j)
            tl, tr = idx(i, j + 1), idx(i + 1, j + 1)
            qc = points[[bl, br, tl, tr]].mean(axis=0)
            to_c = center - qc
            if to_c[0] * to_c[1] > 0:
                triangles += [[bl, br, tr], [bl, tr, tl]]
            else:
                triangles += [[br, tr, tl], [br, tl, bl]]

    return points, np.array(triangles)


# ==========================
# POINT GENERATION
# ==========================

def generate_points(n_points, mode):
    if mode == 3:
        pts = np.random.rand(n_points, 3)
        return pts, Delaunay(pts)
    elif mode == 6:
        cbrt = max(2, round(n_points ** (1 / 3)))
        return triangulate_3d_grid_symmetric(cbrt, cbrt, cbrt)
    elif mode in [4, 5]:
        def factor_pair(n):
            for i in range(int(np.sqrt(n)), 0, -1):
                if n % i == 0: return i, n // i
            return 1, n
        nx2d, ny2d = factor_pair(n_points)
        pts, tris = triangulate_grid_symmetric(nx2d, ny2d)

        class MockTri:
            def __init__(self, s): self.simplices = np.array(s)

        return pts, MockTri(tris)
    else:
        pts = np.random.rand(n_points, 2)
        return pts, Delaunay(pts)


# ==========================
# TRUNCATED OCTAHEDRON
# ==========================

def make_truncated_octahedron(center, scale):
    """
    24 vertices, 6 square faces, 8 hex faces of a truncated octahedron.
    Canonical vertices = all signed permutations of (0, ±1, ±2).
    """
    raw = set()
    for perm in permutations([0, 1, 2]):
        for sx in [1, -1]:
            for sy in [1, -1]:
                for sz in [1, -1]:
                    raw.add((perm[0]*sx, perm[1]*sy, perm[2]*sz))

    verts = np.array(sorted(raw), dtype=float)
    verts = verts * (scale / np.sqrt(2.0))
    verts += np.asarray(center, float)

    s = 2.0 * scale / np.sqrt(2.0)
    center = np.asarray(center, float)

    square_faces = []
    for axis in range(3):
        for sign in [1, -1]:
            face = [vi for vi, v in enumerate(verts)
                    if abs((v - center)[axis] - sign * s) < 1e-9]
            if len(face) == 4:
                pts = verts[face]
                fc  = pts.mean(axis=0)
                other = [a for a in range(3) if a != axis]
                angles = np.arctan2((pts - fc)[:, other[1]], (pts - fc)[:, other[0]])
                square_faces.append([face[i] for i in np.argsort(angles)])

    hex_faces = []
    for sx in [1, -1]:
        for sy in [1, -1]:
            for sz in [1, -1]:
                diag = np.array([sx, sy, sz], float)
                dots = (verts - center) @ diag
                face = [vi for vi, d in enumerate(dots)
                        if abs(d - dots.max()) < 1e-9]
                if len(face) == 6:
                    pts  = verts[face]
                    fc   = pts.mean(axis=0)
                    dn   = diag / np.linalg.norm(diag)
                    u    = np.cross(dn, [1, 0, 0])
                    if np.linalg.norm(u) < 1e-6:
                        u = np.cross(dn, [0, 1, 0])
                    u /= np.linalg.norm(u)
                    v_ax = np.cross(dn, u)
                    off  = pts - fc
                    angles = np.arctan2(off @ v_ax, off @ u)
                    hex_faces.append([face[i] for i in np.argsort(angles)])

    return verts, square_faces, hex_faces


def is_central_hub(pts_list):
    """True when shrunk corners arrive from all 8 octants (full hub node)."""
    if len(pts_list) < 8:
        return False
    pts      = np.array(pts_list)
    centroid = pts.mean(axis=0)
    offsets  = pts - centroid
    octants  = set()
    for off in offsets:
        sx = 1 if off[0] > 1e-9 else (-1 if off[0] < -1e-9 else 0)
        sy = 1 if off[1] > 1e-9 else (-1 if off[1] < -1e-9 else 0)
        sz = 1 if off[2] > 1e-9 else (-1 if off[2] < -1e-9 else 0)
        if sx != 0 and sy != 0 and sz != 0:
            octants.add((sx, sy, sz))
    return len(octants) == 8


# ==========================
# GEOMETRY HELPERS
# ==========================

def unit(a):
    a = np.asarray(a, float)
    n = np.linalg.norm(a)
    return np.zeros_like(a) if n < 1e-12 else a / n


def convex_order_3d(pts):
    """
    Sort roughly-coplanar 3D points into convex polygon order.
    Uses SVD for the normal, largest-offset point for a stable u-axis.
    """
    pts = np.asarray(pts, float)
    if len(pts) < 3:
        return None
    centroid = pts.mean(axis=0)
    offsets  = pts - centroid
    _, _, Vt = np.linalg.svd(offsets)
    normal   = Vt[-1]
    mags     = np.linalg.norm(offsets, axis=1)
    u        = offsets[np.argmax(mags)]
    u        = u - np.dot(u, normal) * normal
    if np.linalg.norm(u) < 1e-12:
        return None
    u  = u / np.linalg.norm(u)
    v  = np.cross(normal, u)
    if np.linalg.norm(v) < 1e-12:
        return None
    v  = v / np.linalg.norm(v)
    coords = offsets @ np.stack([u, v], axis=1)
    return pts[np.argsort(np.arctan2(coords[:, 1], coords[:, 0]))]


def order_hub_ring_xy(vertices_xy, z_plane):
    xy = np.asarray(vertices_xy, float)
    if len(xy) < 3:
        return xy
    v3 = np.hstack([xy, np.full((len(xy), 1), z_plane)])
    o  = convex_order_3d(v3)
    return o[:, :2] if o is not None else xy


def newell_normal(poly):
    """Newell method unit normal for a polygon."""
    n = len(poly)
    normal = np.zeros(3)
    for i in range(n):
        j = (i + 1) % n
        normal[0] += (poly[i][1] - poly[j][1]) * (poly[i][2] + poly[j][2])
        normal[1] += (poly[i][2] - poly[j][2]) * (poly[i][0] + poly[j][0])
        normal[2] += (poly[i][0] - poly[j][0]) * (poly[i][1] + poly[j][1])
    nn = np.linalg.norm(normal)
    return normal / nn if nn > 1e-10 else None


# ==========================
# DRAW HELPERS (matplotlib)
# ==========================

def draw_strut(ax, p0, p1, color='cyan', alpha=0.9, lw=1.8,
               registry=None, label='strut'):
    p0 = np.asarray(p0, float)
    p1 = np.asarray(p1, float)
    if np.linalg.norm(p1 - p0) < 1e-12:
        return
    pts = np.array([p0, p1])
    ax.plot(pts[:, 0], pts[:, 1], pts[:, 2],
            color=color, alpha=alpha, linewidth=lw)
    if registry is not None:
        registry.append((pts, label))


def draw_filled_polygon(ax, verts_3d, centroid_3d,
                        color, alpha_face, alpha_edge, lw,
                        registry=None, label='polygon'):
    """
    Draw a polygon with Bézier-curved edges.
    Face is tessellated as fan triangles from centroid along the curved boundary.
    Edges are drawn as smooth curves.
    """
    verts_3d = np.asarray(verts_3d, float)
    centroid_3d = np.asarray(centroid_3d, float)
    n = len(verts_3d)

    _, segments = bezier_polygon_rings_3d(verts_3d, bezier_bulge, bezier_samples)

    # Draw curved edges
    for seg in segments:
        ax.plot(seg[:, 0], seg[:, 1], seg[:, 2],
                color=color, alpha=alpha_edge, linewidth=lw)

    # Fill face with Bézier-fan triangles
    tris = bezier_polygon_fans(verts_3d, bezier_bulge, bezier_samples)
    ax.add_collection3d(Poly3DCollection(
        tris, color=color, alpha=alpha_face, edgecolor='none'))


def draw_solid_polygon(ax, verts_3d, thickness, color,
                       alpha_face, alpha_edge, lw,
                       registry=None, label='solid'):
    """Extruded polygon prism along its Newell normal, with Bézier-curved edges."""
    verts_3d = np.asarray(verts_3d, float)
    normal   = newell_normal(verts_3d)
    if normal is None:
        return
    top    = verts_3d
    bottom = verts_3d - normal * thickness
    fc  = to_rgba(color, alpha_face)
    ec  = to_rgba(color, alpha_edge)
    ewl = max(lw * 0.4, 0.4)

    # Top and bottom curved edges + fan fills
    for layer_verts in [top, bottom]:
        chain, segments = bezier_polygon_rings_3d(layer_verts, bezier_bulge, bezier_samples)
        for seg in segments:
            ax.plot(seg[:, 0], seg[:, 1], seg[:, 2],
                    color=color, alpha=alpha_edge, linewidth=ewl)
        tris = bezier_polygon_fans(layer_verts, bezier_bulge, bezier_samples)
        ax.add_collection3d(Poly3DCollection(
            tris, facecolors=[fc] * len(tris), edgecolors=['none'] * len(tris)))

    # Side quads: connect top and bottom Bézier chains
    _, top_segs    = bezier_polygon_rings_3d(top,    bezier_bulge, bezier_samples)
    _, bottom_segs = bezier_polygon_rings_3d(bottom, bezier_bulge, bezier_samples)
    n = len(top_segs)
    for i in range(n):
        ts = top_segs[i]
        bs = bottom_segs[i]
        # Draw vertical edges at corners
        ax.plot([ts[0, 0], bs[0, 0]], [ts[0, 1], bs[0, 1]], [ts[0, 2], bs[0, 2]],
                color=color, alpha=alpha_edge, linewidth=ewl)
        # Fill side quads along the Bézier edge
        for k in range(len(ts) - 1):
            quad = np.array([ts[k], ts[k+1], bs[k+1], bs[k]])
            ax.add_collection3d(Poly3DCollection(
                [quad], facecolors=[fc], edgecolors=['none']))


def draw_prism_between_rings(ax, bv, tv, color, alpha_face, alpha_edge, lw,
                              registry=None, label='prism'):
    bv = np.asarray(bv, float)
    tv = np.asarray(tv, float)
    n  = len(bv)
    if n < 3 or len(tv) != n or np.max(np.linalg.norm(tv - bv, axis=1)) < 1e-9:
        return
    fc  = to_rgba(color, alpha_face)
    ec  = to_rgba(color, alpha_edge)
    ewl = max(lw * 0.4, 0.4)

    # Top and bottom caps with Bézier curves
    for layer_verts in [tv, bv]:
        chain, segments = bezier_polygon_rings_3d(layer_verts, bezier_bulge, bezier_samples)
        for seg in segments:
            ax.plot(seg[:, 0], seg[:, 1], seg[:, 2],
                    color=color, alpha=alpha_edge, linewidth=ewl)
        tris = bezier_polygon_fans(layer_verts, bezier_bulge, bezier_samples)
        ax.add_collection3d(Poly3DCollection(
            tris, facecolors=[fc] * len(tris), edgecolors=['none'] * len(tris)))

    # Side quads with curved top/bottom edges
    _, bsegs = bezier_polygon_rings_3d(bv, bezier_bulge, bezier_samples)
    _, tsegs = bezier_polygon_rings_3d(tv, bezier_bulge, bezier_samples)
    for i in range(n):
        bs = bsegs[i]
        ts = tsegs[i]
        ax.plot([bs[0, 0], ts[0, 0]], [bs[0, 1], ts[0, 1]], [bs[0, 2], ts[0, 2]],
                color=color, alpha=alpha_edge, linewidth=ewl)
        for k in range(len(bs) - 1):
            quad = np.array([bs[k], bs[k+1], ts[k+1], ts[k]])
            ax.add_collection3d(Poly3DCollection(
                [quad], facecolors=[fc], edgecolors=['none']))


def draw_truncated_octahedron(ax, center, scale,
                               alpha_face=0.18, alpha_edge=0.85,
                               registry=None, label='hub'):
    verts, sq_faces, hex_faces = make_truncated_octahedron(center, scale)
    fc_sq  = to_rgba('cyan',    alpha_face)
    fc_hex = to_rgba('magenta', alpha_face)

    for fi, face in enumerate(sq_faces):
        pts = verts[face]
        # Bézier-curved edges
        _, segments = bezier_polygon_rings_3d(pts, bezier_bulge, bezier_samples)
        for seg in segments:
            ax.plot(seg[:, 0], seg[:, 1], seg[:, 2],
                    color='cyan', alpha=alpha_edge, linewidth=1.4)
        # Fan-filled face
        tris = bezier_polygon_fans(pts, bezier_bulge, bezier_samples)
        ax.add_collection3d(Poly3DCollection(
            tris, facecolors=[fc_sq] * len(tris), edgecolors=['none'] * len(tris)))

    for fi, face in enumerate(hex_faces):
        pts = verts[face]
        _, segments = bezier_polygon_rings_3d(pts, bezier_bulge, bezier_samples)
        for seg in segments:
            ax.plot(seg[:, 0], seg[:, 1], seg[:, 2],
                    color='magenta', alpha=alpha_edge, linewidth=1.4)
        tris = bezier_polygon_fans(pts, bezier_bulge, bezier_samples)
        ax.add_collection3d(Poly3DCollection(
            tris, facecolors=[fc_hex] * len(tris), edgecolors=['none'] * len(tris)))


# ==========================
# HUB DISPATCH (shared logic)
# ==========================

def _hub_scale_from_pts(hub_center, pts_list):
    mean_dist = np.mean([np.linalg.norm(p - hub_center) for p in pts_list])
    return mean_dist * hub_scale


def dispatch_hub_draw(ax, key, pts_list, registry):
    """Draw the correct hub type for a given vertex."""
    hub_center = np.array(key, float)
    if is_central_hub(pts_list):
        scale = _hub_scale_from_pts(hub_center, pts_list)
        draw_truncated_octahedron(ax, hub_center, scale,
                                   registry=registry, label=f'hub_{key}')
    else:
        pts_arr = np.array(pts_list)
        ordered = convex_order_3d(pts_arr)
        if ordered is not None:
            draw_solid_polygon(ax, ordered, ngon_thickness,
                               color='cyan', alpha_face=0.35, alpha_edge=0.9, lw=1.8,
                               registry=registry, label=f'hub_{key}')


def dispatch_hub_export(key, pts_list, face_verts_list):
    """Add the correct hub geometry to face_verts_list for export."""
    hub_center = np.array(key, float)
    if is_central_hub(pts_list):
        scale = _hub_scale_from_pts(hub_center, pts_list)
        verts, sq_faces, hex_faces = make_truncated_octahedron(hub_center, scale)
        for face in sq_faces + hex_faces:
            face_verts_list.append(verts[face])
    else:
        pts_arr = np.array(pts_list)
        ordered = convex_order_3d(pts_arr)
        if ordered is not None:
            face_verts_list.append(ordered)


# ==========================
# 3D GROUP BUILDER
# ==========================

def build_3d_groups(pts_norm, tri, ratio):
    """
    For each tetrahedron: shrink it, record the shrunk corners grouped
    by original vertex key, and return (groups, face_list).
    face_list: list of (verts_3d, centroid) for tet faces.
    groups: dict key→list of shrunk corner positions.
    """
    groups      = {tuple(p): [] for p in pts_norm}
    tet_faces   = []

    for simplex in tri.simplices:
        tet      = pts_norm[simplex]
        centroid = tet.mean(axis=0)
        shrunk   = np.array([(1 - ratio) * tet[i] + ratio * centroid
                              for i in range(4)])
        for fi in [(0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3)]:
            fv = shrunk[list(fi)]
            tet_faces.append((fv, fv.mean(axis=0)))
        for i, vertex in enumerate(pts_norm[simplex]):
            groups[tuple(vertex)].append(shrunk[i])

    return groups, tet_faces


# ==========================
# PLOT
# ==========================

def plot_3d_lattice(points_nd, tri, ratio, mode, nz_layers=2):
    fig = plt.figure(figsize=(8, 6))
    ax  = fig.add_subplot(111, projection='3d')

    if mode in [3, 6]:
        pts_norm = points_nd
        groups, tet_faces = build_3d_groups(pts_norm, tri, ratio)

        for fv, fc in tet_faces:
            draw_filled_polygon(ax, fv, fc,
                                color='magenta', alpha_face=0.20,
                                alpha_edge=0.85, lw=1.6)

        for pt in [p for pts in groups.values() for p in pts]:
            ax.scatter(*pt, color='cyan', s=20)

        for key, pts_list in groups.items():
            if len(pts_list) == 2:
                draw_strut(ax, pts_list[0], pts_list[1], label=f'strut_{key}')
            elif len(pts_list) >= 3:
                dispatch_hub_draw(ax, key, pts_list, registry=None)

    else:
        points_2d = points_nd
        layers    = [0] if mode in [1, 4] else list(range(nz_layers))

        for simplex in tri.simplices:
            tri_pts = points_2d[simplex]
            c2d     = tri_pts.mean(axis=0)
            t2d     = np.array([(1 - ratio) * tri_pts[i] + ratio * c2d
                                 for i in range(3)])
            if mode in [1, 4]:
                draw_filled_polygon(ax,
                                    np.hstack([t2d, np.zeros((3, 1))]),
                                    np.append(c2d, 0.0),
                                    color='magenta', alpha_face=0.25,
                                    alpha_edge=0.9, lw=1.8)
            elif mode in [2, 5]:
                for z_idx in range(nz_layers - 1):
                    zb = z_idx / (nz_layers - 1)
                    zt = (z_idx + 1) / (nz_layers - 1)
                    bp = np.hstack([t2d, np.full((3, 1), zb)])
                    tp = np.hstack([t2d, np.full((3, 1), zt)])
                    draw_filled_polygon(ax, tp, np.append(c2d, zt),
                                        color='magenta', alpha_face=0.25,
                                        alpha_edge=0.9, lw=1.8)
                    for i in range(3):
                        j    = (i + 1) % 3
                        quad = np.array([bp[i], bp[j], tp[j], tp[i]])
                        draw_filled_polygon(ax, quad, quad.mean(axis=0),
                                            color='magenta', alpha_face=0.15,
                                            alpha_edge=0.85, lw=1.5)

        for z_idx in layers:
            z_val  = 0.0 if mode in [1, 4] else z_idx / (nz_layers - 1)
            groups = {tuple(p): [] for p in points_2d}

            for simplex in tri.simplices:
                tri_pts = points_2d[simplex]
                c2d     = tri_pts.mean(axis=0)
                t2d     = np.array([(1 - ratio) * tri_pts[i] + ratio * c2d
                                     for i in range(3)])
                t3d     = np.hstack([t2d, np.full((3, 1), z_val)])
                for pt in t3d:
                    ax.scatter(*pt, color='cyan', s=20)
                for i, v in enumerate(tri_pts):
                    groups[tuple(v)].append(t3d[i])

            for key, pts_list in groups.items():
                if len(pts_list) == 2:
                    p0, p1 = pts_list
                    if mode in [1, 4]:
                        draw_strut(ax, p0, p1)
                    else:
                        zb = z_idx / (nz_layers - 1)
                        zt = (z_idx + 1) / (nz_layers - 1) if z_idx < nz_layers - 1 else zb
                        draw_strut(ax, [p0[0], p0[1], zb], [p1[0], p1[1], zb])
                        draw_strut(ax, [p0[0], p0[1], zt], [p1[0], p1[1], zt])
                        draw_strut(ax, [p0[0], p0[1], zb], [p0[0], p0[1], zt])
                        draw_strut(ax, [p1[0], p1[1], zb], [p1[0], p1[1], zt])
                elif len(pts_list) >= 3:
                    try:
                        pts_arr = np.array(pts_list)
                        hull    = ConvexHull(pts_arr[:, :2])
                        verts   = pts_arr[hull.vertices, :2]
                        if mode in [1, 4]:
                            ring = order_hub_ring_xy(verts, z_val)
                            v3d  = np.hstack([ring, np.full((len(ring), 1), z_val)])
                            draw_solid_polygon(ax, v3d, ngon_thickness,
                                               color='cyan', alpha_face=0.25,
                                               alpha_edge=0.9, lw=1.8)
                        else:
                            zb   = z_idx / (nz_layers - 1)
                            zt   = (z_idx + 1) / (nz_layers - 1) if z_idx < nz_layers - 1 else zb
                            ring = order_hub_ring_xy(verts, zb)
                            nr   = len(ring)
                            bv   = np.hstack([ring, np.full((nr, 1), zb)])
                            tv   = np.hstack([ring, np.full((nr, 1), zt)])
                            if abs(zt - zb) < 1e-9:
                                draw_solid_polygon(ax, bv, ngon_thickness,
                                                   color='cyan', alpha_face=0.25,
                                                   alpha_edge=0.9, lw=1.8)
                            else:
                                draw_prism_between_rings(ax, bv, tv, color='cyan',
                                                          alpha_face=0.25,
                                                          alpha_edge=0.9, lw=1.8)
                    except Exception:
                        pass

    ax.set_box_aspect([1, 1, 1])
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_zlim(0, 1)
    ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
    elev, azim = 30, 45
    ax.view_init(elev=elev, azim=azim)

    def on_key(event):
        nonlocal elev, azim
        step = 10
        if   event.key == 'up':    elev += step
        elif event.key == 'down':  elev -= step
        elif event.key == 'left':  azim -= step
        elif event.key == 'right': azim += step
        elif event.key == 'x':     elev, azim = 0,  0
        elif event.key == 'y':     elev, azim = 0,  90
        elif event.key == 'z':     elev, azim = 90, -90
        ax.view_init(elev=elev, azim=azim)
        fig.canvas.draw_idle()

    fig.canvas.mpl_connect('key_press_event', on_key)
    plt.show()


# ==========================
# EXPORT — GEOMETRY COLLECT
# ==========================

def collect_export_geometry(points_nd, tri, ratio, mode, nz_layers):
    strut_curves    = []
    face_verts_list = []

    def add_strut(p0, p1):
        p0 = np.asarray(p0, float); p1 = np.asarray(p1, float)
        if np.linalg.norm(p1 - p0) > 1e-9:
            strut_curves.append(np.array([p0, p1]))

    def add_face(verts_3d):
        face_verts_list.append(np.asarray(verts_3d, float))

    if mode in [3, 6]:
        pts_norm          = points_nd
        groups, tet_faces = build_3d_groups(pts_norm, tri, ratio)

        for fv, _ in tet_faces:
            add_face(fv)

        for key, pts_list in groups.items():
            if len(pts_list) == 2:
                add_strut(pts_list[0], pts_list[1])
            elif len(pts_list) >= 3:
                dispatch_hub_export(key, pts_list, face_verts_list)

    else:
        points_2d = points_nd
        layers    = [0] if mode in [1, 4] else list(range(nz_layers))

        for simplex in tri.simplices:
            tri_pts = points_2d[simplex]
            c2d     = tri_pts.mean(axis=0)
            t2d     = np.array([(1 - ratio) * tri_pts[i] + ratio * c2d
                                 for i in range(3)])
            if mode in [1, 4]:
                add_face(np.hstack([t2d, np.zeros((3, 1))]))
            elif mode in [2, 5]:
                for z_idx in range(nz_layers - 1):
                    zb = z_idx / (nz_layers - 1)
                    zt = (z_idx + 1) / (nz_layers - 1)
                    bp = np.hstack([t2d, np.full((3, 1), zb)])
                    tp = np.hstack([t2d, np.full((3, 1), zt)])
                    add_face(tp)
                    for i in range(3):
                        j = (i + 1) % 3
                        add_face(np.array([bp[i], bp[j], tp[j], tp[i]]))

        for z_idx in layers:
            z_val  = 0.0 if mode in [1, 4] else z_idx / (nz_layers - 1)
            groups = {tuple(p): [] for p in points_2d}

            for simplex in tri.simplices:
                tri_pts = points_2d[simplex]
                c2d     = tri_pts.mean(axis=0)
                t2d     = np.array([(1 - ratio) * tri_pts[i] + ratio * c2d
                                     for i in range(3)])
                t3d     = np.hstack([t2d, np.full((3, 1), z_val)])
                for i, v in enumerate(tri_pts):
                    groups[tuple(v)].append(t3d[i])

            for key, pts_list in groups.items():
                if len(pts_list) == 2:
                    p0, p1 = pts_list
                    if mode in [1, 4]:
                        add_strut(p0, p1)
                    else:
                        zb = z_idx / (nz_layers - 1)
                        zt = (z_idx + 1) / (nz_layers - 1) if z_idx < nz_layers - 1 else zb
                        add_strut([p0[0], p0[1], zb], [p1[0], p1[1], zb])
                        add_strut([p0[0], p0[1], zt], [p1[0], p1[1], zt])
                        add_strut([p0[0], p0[1], zb], [p0[0], p0[1], zt])
                        add_strut([p1[0], p1[1], zb], [p1[0], p1[1], zt])
                elif len(pts_list) >= 3:
                    try:
                        pts_arr = np.array(pts_list)
                        hull    = ConvexHull(pts_arr[:, :2])
                        verts   = pts_arr[hull.vertices, :2]
                        if mode in [1, 4]:
                            ring = order_hub_ring_xy(verts, z_val)
                            add_face(np.hstack([ring, np.full((len(ring), 1), z_val)]))
                        else:
                            zb   = z_idx / (nz_layers - 1)
                            zt   = (z_idx + 1) / (nz_layers - 1) if z_idx < nz_layers - 1 else zb
                            ring = order_hub_ring_xy(verts, zb)
                            nr   = len(ring)
                            bv   = np.hstack([ring, np.full((nr, 1), zb)])
                            tv   = np.hstack([ring, np.full((nr, 1), zt)])
                            add_face(tv); add_face(bv)
                            if abs(zt - zb) >= 1e-9:
                                for i in range(nr):
                                    j = (i + 1) % nr
                                    add_face(np.array([bv[i], bv[j], tv[j], tv[i]]))
                    except Exception:
                        pass

    return strut_curves, face_verts_list


# ==========================
# EXPORT — MESH BUILD
# ==========================

def build_export_triangles(strut_curves, face_verts_list):
    """
    Build export triangles using Bézier-curved polygon outlines for faces,
    and tube meshes for struts.
    """
    all_triangles = []

    for poly in face_verts_list:
        poly = np.asarray(poly, float)
        n    = len(poly)
        if n < 3:
            continue
        normal = newell_normal(poly)
        if normal is None:
            continue

        bottom = poly
        top    = poly + normal * face_thickness

        # Generate Bézier-tessellated triangles for top and bottom faces
        bot_tris = bezier_polygon_fans(bottom, bezier_bulge, bezier_samples)
        top_tris = bezier_polygon_fans(top,    bezier_bulge, bezier_samples)

        # Bottom face (winding reversed for outward normal)
        for tri in bot_tris:
            all_triangles.append([tri[0], tri[2], tri[1]])
        # Top face
        for tri in top_tris:
            all_triangles.append(tri)

        # Side walls: connect matching Bézier chains on top and bottom
        _, bot_segs = bezier_polygon_rings_3d(bottom, bezier_bulge, bezier_samples)
        _, top_segs = bezier_polygon_rings_3d(top,    bezier_bulge, bezier_samples)
        for i in range(n):
            bs = bot_segs[i]
            ts = top_segs[i]
            for k in range(len(bs) - 1):
                b0, b1 = bs[k],   bs[k+1]
                t0, t1 = ts[k],   ts[k+1]
                all_triangles.append([b0, b1, t1])
                all_triangles.append([b0, t1, t0])

    def tube_mesh(path, radius, segments):
        path  = np.asarray(path, float)
        rings = []
        for k, pt in enumerate(path):
            tang = path[1] - path[0] if k == 0 else \
                   path[-1] - path[-2] if k == len(path) - 1 else \
                   path[k + 1] - path[k - 1]
            tn = np.linalg.norm(tang)
            if tn < 1e-12: continue
            tang = tang / tn
            perp = np.cross(tang, [0, 0, 1])
            if np.linalg.norm(perp) < 1e-6:
                perp = np.cross(tang, [0, 1, 0])
            perp  = perp / np.linalg.norm(perp)
            perp2 = np.cross(tang, perp)
            rings.append([pt + radius * (np.cos(2 * np.pi * s / segments) * perp +
                                          np.sin(2 * np.pi * s / segments) * perp2)
                           for s in range(segments)])
        tris = []
        for i in range(len(rings) - 1):
            r0, r1 = rings[i], rings[i + 1]
            for s in range(segments):
                s1 = (s + 1) % segments
                tris += [[r0[s], r1[s], r1[s1]], [r0[s], r1[s1], r0[s1]]]
        for s in range(segments):
            s1 = (s + 1) % segments
            tris.append([path[0],  rings[0][-1 - s1], rings[0][-1 - s]])
            tris.append([path[-1], rings[-1][s],       rings[-1][s1]])
        return tris

    for path in strut_curves:
        all_triangles += tube_mesh(path, strut_radius, scad_segments)

    return all_triangles


# ==========================
# EXPORT — WRITERS
# ==========================

def _fmt(v):
    return f"[{v[0]:.6f},{v[1]:.6f},{v[2]:.6f}]"


def export_to_scad(scad_path, strut_curves, face_verts_list):
    def scad_cylinder(p0, p1, radius, fn):
        d   = p1 - p0
        L   = np.linalg.norm(d)
        if L < 1e-10: return ""
        dh  = d / L
        dot = np.clip(dh[2], -1, 1)
        ang = np.degrees(np.arccos(dot))
        ax  = np.cross([0, 0, 1], dh)
        an  = np.linalg.norm(ax)
        ax  = ax / an if an > 1e-10 else np.array([1, 0, 0])
        return (f"translate({_fmt(p0)})\n"
                f"  rotate(a={ang:.6f},v={_fmt(ax)})\n"
                f"    cylinder(h={L:.6f},r={radius:.6f},$fn={fn});\n")

    def scad_face(verts_3d):
        """
        Export a face as Bézier-tessellated polyhedron triangles in SCAD.
        """
        verts_3d = np.asarray(verts_3d, float)
        n = len(verts_3d)
        if n < 3: return ""
        normal = newell_normal(verts_3d)
        if normal is None: return ""

        bottom = verts_3d
        top    = verts_3d + normal * face_thickness

        bot_tris = bezier_polygon_fans(bottom, bezier_bulge, bezier_samples)
        top_tris = bezier_polygon_fans(top,    bezier_bulge, bezier_samples)
        _, bot_segs = bezier_polygon_rings_3d(bottom, bezier_bulge, bezier_samples)
        _, top_segs = bezier_polygon_rings_3d(top,    bezier_bulge, bezier_samples)

        all_verts = []
        vi_map = {}
        lines = []

        def get_vi(pt):
            key = tuple(np.round(pt, 9))
            if key not in vi_map:
                vi_map[key] = len(all_verts)
                all_verts.append(pt)
            return vi_map[key]

        faces = []
        for tri in bot_tris:
            faces.append([get_vi(tri[0]), get_vi(tri[2]), get_vi(tri[1])])
        for tri in top_tris:
            faces.append([get_vi(tri[0]), get_vi(tri[1]), get_vi(tri[2])])
        for i in range(n):
            bs = bot_segs[i]; ts = top_segs[i]
            for k in range(len(bs) - 1):
                b0, b1, t0, t1 = bs[k], bs[k+1], ts[k], ts[k+1]
                faces.append([get_vi(b0), get_vi(b1), get_vi(t1)])
                faces.append([get_vi(b0), get_vi(t1), get_vi(t0)])

        vs = ", ".join(_fmt(v) for v in all_verts)
        fs = ", ".join("[" + ",".join(str(x) for x in f) + "]" for f in faces)
        return f"polyhedron(points=[{vs}],faces=[{fs}],convexity=4);\n"

    lines = [f"// auxetic lattice  mode={mode}  n_points={n_points}  ratio={ratio}\n",
             f"// bezier_bulge={bezier_bulge}  bezier_samples={bezier_samples}\n",
             f"$fn={scad_segments};\nrender(convexity=10)\nunion(){{\n",
             "  // struts\n"]
    for pts in strut_curves:
        lines.append(scad_cylinder(pts[0], pts[1], strut_radius, scad_segments))
    lines.append("  // faces\n")
    for fv in face_verts_list:
        lines.append(scad_face(fv))
    lines.append("}\n")
    with open(scad_path, "w") as f:
        f.writelines(lines)
    print(f"  SCAD: {scad_path}  ({len(strut_curves)} struts, {len(face_verts_list)} faces)")


def export_stl_direct(stl_path, triangles):
    try:
        from stl import mesh as stl_mesh
    except ImportError:
        print("  numpy-stl not installed — skipping STL"); return
    import os
    m = stl_mesh.Mesh(np.zeros(len(triangles), dtype=stl_mesh.Mesh.dtype))
    for i, tri in enumerate(triangles):
        for j in range(3):
            m.vectors[i][j] = np.asarray(tri[j], float)
    m.save(stl_path)
    print(f"  STL: {stl_path}  ({os.path.getsize(stl_path)//1024} KB)")


def export_obj_direct(obj_path, triangles):
    import os
    lines   = ["# auxetic lattice\n"]
    v_count = 0
    n_count = 0
    for tri in triangles:
        a, b, c = (np.asarray(tri[k], float) for k in range(3))
        nv = np.cross(b - a, c - a)
        nn = np.linalg.norm(nv)
        if nn < 1e-14: continue
        nv = nv / nn
        vb = v_count + 1
        for p in (a, b, c):
            lines.append(f"v {p[0]:.9g} {p[1]:.9g} {p[2]:.9g}\n")
        v_count += 3
        n_count += 1
        lines.append(f"vn {nv[0]:.9g} {nv[1]:.9g} {nv[2]:.9g}\n")
        lines.append(f"f {vb}//{n_count} {vb+1}//{n_count} {vb+2}//{n_count}\n")
    with open(obj_path, "w", encoding="utf-8", newline="\n") as f:
        f.writelines(lines)
    print(f"  OBJ: {obj_path}  ({os.path.getsize(obj_path)//1024} KB, {n_count} tris)")


def run_export(points_nd, tri, ratio, mode, nz_layers):
    import os
    print("\n--- Export ---")
    base = os.path.dirname(os.path.abspath(__file__))
    resolve = lambda p: p if os.path.isabs(p) else os.path.join(base, p)

    strut_curves, face_verts_list = collect_export_geometry(
        points_nd, tri, ratio, mode, nz_layers)
    print(f"  Geometry: {len(strut_curves)} struts, {len(face_verts_list)} faces")

    if export_scad:
        export_to_scad(resolve(export_scad_path), strut_curves, face_verts_list)

    if export_stl or export_obj:
        triangles = build_export_triangles(strut_curves, face_verts_list)
        if export_stl:
            export_stl_direct(resolve(export_stl_path), triangles)
        if export_obj:
            export_obj_direct(resolve(export_obj_path), triangles)

    print("--------------\n")


# ==========================
# ENTRY POINT
# ==========================

points_nd, tri_nd = generate_points(n_points, mode)

if export_enabled:
    run_export(points_nd, tri_nd, ratio, mode, nz_layers)

if show_plot:
    plot_3d_lattice(points_nd, tri_nd, ratio, mode, nz_layers=nz_layers)
