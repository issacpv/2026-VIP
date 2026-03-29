import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
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
n_points = 120
ratio = 0.5
nx, ny, nz = 1, 1, 1
cell = 1.0
nz_layers = 2    # number of layers in 2.5D extrusion (modes 2, 5 only)

# --- Shape thickness ---
ngon_thickness = 0.03   # extrusion depth of cyan n-gon solids (in lattice units)

# --- Intersection check ---z
intersect_threshold = 0.05   # fraction of mean strut length; raise = more sensitive
intersect_check     = False   # set False to skip (faster for large lattices)

# --- Export settings ---
export_scad = False          # generate a .scad file (OpenSCAD)
export_stl  = True           # generate a .stl file (direct STL, no OpenSCAD needed)
export_obj  = True           # generate a .obj file (Wavefront; no extra dependencies)
export_name = 'auxetic_lattice'  # output filename (no extension)
strut_radius = 0.02          # physical radius of struts in exported geometry (world units)
strut_segments = 6           # cross-section polygon sides for struts (higher = rounder, slower)
hub_scale = 0.45 
# ==========================
# Helpers
# ==========================
def make_truncated_octahedron(center, scale):
    """
    Returns the 24 vertices, 6 square faces, and 8 hex faces
    of a truncated octahedron centered at `center` with given `scale`.

    Canonical vertices are all permutations of (0, ±1, ±2) — 24 total.
    Scaled so that edge length = scale.

    Square faces:  6, each perpendicular to a coordinate axis.
    Hex faces:     8, each facing a body diagonal direction.
    """
    # All permutations of (0, ±1, ±2)
    from itertools import permutations
    raw = set()
    for perm in permutations([0, 1, 2]):
        for sx in [1, -1]:
            for sy in [1, -1]:
                for sz in [1, -1]:
                    v = (perm[0]*sx, perm[1]*sy, perm[2]*sz)
                    raw.add(v)
    # Remove (0,0,0) duplicates from sign flips on zero
    verts = np.array(sorted(raw), dtype=float)
    # Normalize: edge length of canonical form = sqrt(2)*(2-1)= sqrt(2)
    # Scale to desired size
    edge_len = np.sqrt(2.0)
    verts = verts * (scale / edge_len)
    verts += np.asarray(center, float)

    # --- Square faces (6): perpendicular to each axis ---
    # Each square face has all vertices with one coordinate = ±2*scale/edge_len
    s = 2.0 * scale / edge_len
    square_faces = []
    for axis in range(3):
        for sign in [1, -1]:
            face = []
            for vi, v in enumerate(verts):
                # Find vertices where this axis coordinate = sign*s
                if abs((v - center)[axis] - sign * s) < 1e-9:
                    face.append(vi)
            if len(face) == 4:
                # Order the 4 vertices as a convex quad
                face_pts = verts[face]
                fc = face_pts.mean(axis=0)
                # Project onto plane perpendicular to axis
                other = [a for a in range(3) if a != axis]
                angles = np.arctan2(
                    (face_pts - fc)[:, other[1]],
                    (face_pts - fc)[:, other[0]]
                )
                ordered = [face[i] for i in np.argsort(angles)]
                square_faces.append(ordered)

    # --- Hex faces (8): one per octant ---
    # Each hex face has vertices whose coordinates sum to ±3*scale/edge_len
    # (the face perpendicular to body diagonal (±1,±1,±1))
    hex_faces = []
    for sx in [1, -1]:
        for sy in [1, -1]:
            for sz in [1, -1]:
                target_sum = (sx + sy + sz) * s  # = ±3s but body diag faces have sum=±(1+1+2)variants
                # Hex face: vertices where the dot product with (sx,sy,sz) is maximized
                diag = np.array([sx, sy, sz], float)
                dots = (verts - center) @ diag
                max_dot = dots.max()
                face = [vi for vi, d in enumerate(dots) if abs(d - max_dot) < 1e-9]
                if len(face) == 6:
                    face_pts = verts[face]
                    fc = face_pts.mean(axis=0)
                    # Sort by angle in the plane perpendicular to diag
                    diag_n = diag / np.linalg.norm(diag)
                    u = np.cross(diag_n, [1, 0, 0])
                    if np.linalg.norm(u) < 1e-6:
                        u = np.cross(diag_n, [0, 1, 0])
                    u /= np.linalg.norm(u)
                    v_ax = np.cross(diag_n, u)
                    offsets = face_pts - fc
                    angles = np.arctan2(offsets @ v_ax, offsets @ u)
                    ordered = [face[i] for i in np.argsort(angles)]
                    hex_faces.append(ordered)

    return verts, square_faces, hex_faces


def draw_truncated_octahedron(ax, center, scale, alpha_face=0.18, alpha_edge=0.85,
                               registry=None, label='hub'):
    """Draw a truncated octahedron hub solid at the given center."""
    verts, sq_faces, hex_faces = make_truncated_octahedron(center, scale)

    fc_sq  = to_rgba('cyan',    alpha_face)
    fc_hex = to_rgba('magenta', alpha_face)
    ec     = to_rgba('cyan',    alpha_edge)

    # Draw square faces (cyan)
    for fi, face in enumerate(sq_faces):
        pts = verts[face]
        tris = [[pts[0], pts[i], pts[i+1]] for i in range(1, len(pts)-1)]
        ax.add_collection3d(Poly3DCollection(tris, facecolors=[fc_sq]*len(tris),
                                              edgecolors=['none']*len(tris)))
        # Draw edges
        for i in range(len(pts)):
            j = (i+1) % len(pts)
            draw_strut(ax, pts[i], pts[j], color='cyan', alpha=alpha_edge,
                       lw=1.4, registry=registry, label=f'{label}_sq{fi}_e{i}')

    # Draw hex faces (magenta)
    for fi, face in enumerate(hex_faces):
        pts = verts[face]
        tris = [[pts[0], pts[i], pts[i+1]] for i in range(1, len(pts)-1)]
        ax.add_collection3d(Poly3DCollection(tris, facecolors=[fc_hex]*len(tris),
                                              edgecolors=['none']*len(tris)))
        for i in range(len(pts)):
            j = (i+1) % len(pts)
            draw_strut(ax, pts[i], pts[j], color='magenta', alpha=alpha_edge,
                       lw=1.4, registry=registry, label=f'{label}_hex{fi}_e{i}')

def draw_truncated_octahedron(ax, center, scale, alpha_face=0.18, alpha_edge=0.85,
                               registry=None, label='hub'):
    """Draw a truncated octahedron hub solid at the given center."""
    verts, sq_faces, hex_faces = make_truncated_octahedron(center, scale)

    fc_sq  = to_rgba('cyan',    alpha_face)
    fc_hex = to_rgba('magenta', alpha_face)
    ec     = to_rgba('cyan',    alpha_edge)

    # Draw square faces (cyan)
    for fi, face in enumerate(sq_faces):
        pts = verts[face]
        tris = [[pts[0], pts[i], pts[i+1]] for i in range(1, len(pts)-1)]
        ax.add_collection3d(Poly3DCollection(tris, facecolors=[fc_sq]*len(tris),
                                              edgecolors=['none']*len(tris)))
        # Draw edges
        for i in range(len(pts)):
            j = (i+1) % len(pts)
            draw_strut(ax, pts[i], pts[j], color='cyan', alpha=alpha_edge,
                       lw=1.4, registry=registry, label=f'{label}_sq{fi}_e{i}')

    # Draw hex faces (magenta)
    for fi, face in enumerate(hex_faces):
        pts = verts[face]
        tris = [[pts[0], pts[i], pts[i+1]] for i in range(1, len(pts)-1)]
        ax.add_collection3d(Poly3DCollection(tris, facecolors=[fc_hex]*len(tris),
                                              edgecolors=['none']*len(tris)))
        for i in range(len(pts)):
            j = (i+1) % len(pts)
            draw_strut(ax, pts[i], pts[j], color='magenta', alpha=alpha_edge,
                       lw=1.4, registry=registry, label=f'{label}_hex{fi}_e{i}')
                       
def triangulate_3d_grid_symmetric(nx, ny, nz, center_aligned=True):
    """
    6-tetrahedra-per-cube decomposition where the body diagonal of each
    cube points TOWARD the global center of the lattice.

    For a 2x2x2 arrangement of cubes (3x3x3 points), this means:
      - The 8 cubes each pick the diagonal whose midpoint is closest
        to the global center — i.e. the diagonal that runs inward.
      - Adjacent cubes in different octants get complementary diagonals,
        so shared faces still match.

    The 4 possible body diagonals per cube:
      D0: (0,0,0)-(1,1,1)  midpoint offset (+,+,+)
      D1: (1,0,0)-(0,1,1)  midpoint offset (-,+,+)  [flipped x]
      D2: (0,1,0)-(1,0,1)  midpoint offset (+,-,+)  [flipped y]
      D3: (0,0,1)-(1,1,0)  midpoint offset (+,+,-)  [flipped z]
    """
    x = np.linspace(0, 1, nx)
    y = np.linspace(0, 1, ny)
    z = np.linspace(0, 1, nz)
    xx, yy, zz = np.meshgrid(x, y, z, indexing='ij')
    points = np.vstack([xx.ravel(), yy.ravel(), zz.ravel()]).T

    def idx(i, j, k):
        return i * (ny * nz) + j * nz + k

    # Global center of the whole grid
    cx = (nx - 1) / 2.0
    cy = (ny - 1) / 2.0
    cz = (nz - 1) / 2.0

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

                # Cube center relative to grid center
                # sign tells us which octant this cube is in
                dx = (i + 0.5) - cx
                dy = (j + 0.5) - cy
                dz = (k + 0.5) - cz

                # Pick the diagonal whose direction most aligns
                # with the inward vector (toward global center).
                # Each diagonal has a direction vector:
                #   D0: (1,1,1)   points toward (+,+,+) octant
                #   D1: (-1,1,1)  points toward (-,+,+) octant
                #   D2: (1,-1,1)  points toward (+,-,+) octant
                #   D3: (1,1,-1)  points toward (+,+,-) octant
                #
                # We want the diagonal pointing FROM the cube's octant
                # TOWARD center, so we pick by sign of (dx, dy, dz).
                #
                # Cube in (+,+,+) octant → inward diagonal runs (1,1,1)→(0,0,0)
                #   i.e. D0 with a=c111, b=c000
                # Cube in (-,+,+) octant → inward diagonal runs (0,1,1)→(1,0,0)
                #   i.e. D1 with a=c011, b=c100
                # etc.

                sx = dx >= 0  # True = cube is in positive-x half
                sy = dy >= 0
                sz = dz >= 0

                if sx and sy and sz:
                    # Octant (+,+,+): diagonal from c111 toward c000
                    a, b = c111, c000
                    ring = [c110, c010, c011, c001, c101, c100]
                elif not sx and sy and sz:
                    # Octant (-,+,+): diagonal from c011 toward c100
                    a, b = c011, c100
                    ring = [c001, c000, c010, c110, c111, c101]
                elif sx and not sy and sz:
                    # Octant (+,-,+): diagonal from c101 toward c010
                    a, b = c101, c010
                    ring = [c100, c000, c001, c011, c111, c110]
                elif sx and sy and not sz:
                    # Octant (+,+,-): diagonal from c110 toward c001
                    a, b = c110, c001
                    ring = [c111, c011, c010, c000, c100, c101]
                elif not sx and not sy and sz:
                    # Octant (-,-,+): diagonal from c001 toward c110
                    a, b = c001, c110
                    ring = [c000, c100, c101, c111, c011, c010]
                elif not sx and sy and not sz:
                    # Octant (-,+,-): diagonal from c010 toward c101
                    a, b = c010, c101
                    ring = [c011, c111, c110, c100, c000, c001]
                elif sx and not sy and not sz:
                    # Octant (+,-,-): diagonal from c100 toward c011
                    a, b = c100, c011
                    ring = [c110, c111, c101, c001, c000, c010]
                else:
                    # Octant (-,-,-): diagonal from c000 toward c111
                    a, b = c000, c111
                    ring = [c100, c110, c010, c011, c001, c101]

                for r in range(6):
                    tetrahedra.append([a, b, ring[r], ring[(r+1) % 6]])

    class MockTri:
        def __init__(self, simplices):
            self.simplices = np.array(simplices)

    return points, MockTri(tetrahedra)

def triangulate_grid_symmetric(nx, ny, center_aligned=True):
    """
    Manually triangulate a grid with controlled diagonal orientation.
    
    nx, ny: grid dimensions (number of points per axis)
    center_aligned: if True, diagonals point away from center (pinwheel);
                    if False, all diagonals have the same global orientation
    
    Returns: (points, triangles)
        points: (nx*ny, 2) array of 2D coordinates
        triangles: (N, 3) array of vertex indices forming triangles
    """
    # Generate grid points
    x = np.linspace(0, 1, nx)
    y = np.linspace(0, 1, ny)
    xx, yy = np.meshgrid(x, y)
    points = np.vstack([xx.ravel(), yy.ravel()]).T  # (nx*ny, 2)
    
    # Helper: convert (i, j) grid coords to flat index
    def idx(i, j):
        return j * nx + i
    
    triangles = []
    center = np.array([0.5, 0.5])  # center of the unit square
    
    for i in range(nx - 1):
        for j in range(ny - 1):
            # Four corners of the quad
            bl = idx(i, j)          # bottom-left
            br = idx(i + 1, j)      # bottom-right
            tl = idx(i, j + 1)      # top-left
            tr = idx(i + 1, j + 1)  # top-right
            
            quad_center = points[[bl, br, tl, tr]].mean(axis=0)
            
            if center_aligned:
                # Choose diagonal based on quadrant relative to structure center
                # This creates a pinwheel pattern
                to_center = center - quad_center
                
                # Use the sign of the cross product to pick diagonal direction
                # This creates rotational symmetry around the center
                if to_center[0] * to_center[1] > 0:
                    # Diagonal from bottom-left to top-right
                    triangles.append([bl, br, tr])
                    triangles.append([bl, tr, tl])
                else:
                    # Diagonal from bottom-right to top-left
                    triangles.append([br, tr, tl])
                    triangles.append([br, tl, bl])
            else:
                # All diagonals point the same way (e.g., always SW-NE)
                # This creates a uniform chevron pattern
                triangles.append([bl, br, tr])
                triangles.append([bl, tr, tl])
    
    return points, np.array(triangles)

def unit(a):
    a = np.asarray(a, dtype=float)
    n = np.linalg.norm(a)
    return np.zeros_like(a) if n < 1e-12 else a / n

def draw_strut(ax, p0, p1, color='cyan', alpha=0.9, lw=1.8,
               registry=None, label='strut'):
    p0 = np.asarray(p0, float)
    p1 = np.asarray(p1, float)
    if np.linalg.norm(p1 - p0) < 1e-12:
        return
    pts = np.array([p0, p1])
    ax.plot(pts[:, 0], pts[:, 1], pts[:, 2], color=color, alpha=alpha, linewidth=lw)
    if registry is not None:
        registry.append((pts, label))

def draw_filled_polygon(ax, verts_3d, centroid_3d,
                        color, alpha_face, alpha_edge, lw,
                        registry=None, label='polygon'):
    """Filled polygon (fan-triangulated) with straight edge segments."""
    n = len(verts_3d)
    triangles = [
        [verts_3d[0], verts_3d[i], verts_3d[i + 1]]
        for i in range(1, n - 1)
    ]
    ax.add_collection3d(Poly3DCollection(triangles, color=color,
                                          alpha=alpha_face, edgecolor='none'))
    for i in range(n):
        j = (i + 1) % n
        draw_strut(ax, verts_3d[i], verts_3d[j], color=color, alpha=alpha_edge,
                   lw=lw, registry=registry, label=f'{label}_edge{i}')

def convex_order_3d(pts):
    """
    Order a set of roughly coplanar 3D points into a convex polygon.
    
    Fixes vs original:
    - Builds a stable local frame using the point with the largest
      deviation from centroid as the reference direction (not SVD,
      which can flip arbitrarily between calls).
    - Projects all points onto this consistent frame and sorts by angle.
    - Returns None if fewer than 3 non-degenerate points.
    """
    pts = np.asarray(pts, float)
    if len(pts) < 3:
        return None

    centroid = pts.mean(axis=0)
    offsets  = pts - centroid

    # Best-fit normal via SVD (last right singular vector)
    _, _, Vt = np.linalg.svd(offsets)
    normal   = Vt[-1]

    # --- Stable reference direction ---
    # Pick the offset with the largest magnitude as the u-axis seed.
    # This is deterministic for a given point set and doesn't flip.
    magnitudes = np.linalg.norm(offsets, axis=1)
    ref_idx    = np.argmax(magnitudes)
    u          = offsets[ref_idx]
    u          = u - np.dot(u, normal) * normal   # project onto plane
    u_norm     = np.linalg.norm(u)
    if u_norm < 1e-12:
        return None
    u = u / u_norm
    v = np.cross(normal, u)
    v_norm = np.linalg.norm(v)
    if v_norm < 1e-12:
        return None
    v = v / v_norm

    # Project and sort by angle
    coords = offsets @ np.stack([u, v], axis=1)   # (N, 2)
    angles = np.arctan2(coords[:, 1], coords[:, 0])
    return pts[np.argsort(angles)]


def order_hub_ring_xy(vertices_xy, z_plane):
    """Counterclockwise xy rows for a planar hub (shared vertex / hull corner ring)."""
    xy = np.asarray(vertices_xy, float)
    if len(xy) < 3:
        return xy
    v3 = np.hstack([xy, np.full((len(xy), 1), z_plane)])
    o = convex_order_3d(v3)
    return o[:, :2] if o is not None else xy


def draw_solid_polygon(ax, verts_3d, centroid_3d, thickness,
                       color, alpha_face, alpha_edge, lw,
                       registry=None, label='solid',
                       draw_tube_outline=True, rim_outline=False):
    """
    Draw a solid extruded polygon (prism) by offsetting vertices along the
    face normal by `thickness`. Top / bottom / side quads optional rim lines.
    draw_tube_outline: if True, draw 3D line segments (not used for hub solids
    when False — avoids “thick tube” look; mesh export has no cylinders there).
    rim_outline: if True, draw thin polygon edges so faces meet cleanly at the border.
    """
    verts_3d = np.asarray(verts_3d, float)
    n = len(verts_3d)
    if n < 3:
        return

    # Newell normal
    normal = np.zeros(3)
    for i in range(n):
        j = (i + 1) % n
        normal[0] += (verts_3d[i][1] - verts_3d[j][1]) * (verts_3d[i][2] + verts_3d[j][2])
        normal[1] += (verts_3d[i][2] - verts_3d[j][2]) * (verts_3d[i][0] + verts_3d[j][0])
        normal[2] += (verts_3d[i][0] - verts_3d[j][0]) * (verts_3d[i][1] + verts_3d[j][1])
    nn = np.linalg.norm(normal)
    if nn < 1e-10:
        return
    normal = normal / nn

    top    = verts_3d
    bottom = verts_3d - normal * thickness

    fc = to_rgba(color, alpha_face)
    ec = to_rgba(color, alpha_edge) if rim_outline else 'none'
    ewl = max(lw * 0.4, 0.4) if rim_outline else 0.0

    # Top face
    ax.add_collection3d(Poly3DCollection(
        [top], facecolors=[fc], edgecolors=[ec], linewidths=ewl))
    # Bottom face
    ax.add_collection3d(Poly3DCollection(
        [bottom], facecolors=[fc], edgecolors=[ec], linewidths=ewl))
    # Side quads
    for i in range(n):
        j    = (i + 1) % n
        quad = np.array([top[i], top[j], bottom[j], bottom[i]])
        ax.add_collection3d(Poly3DCollection(
            [quad], facecolors=[fc], edgecolors=[ec], linewidths=ewl))

    if not draw_tube_outline:
        return
    for i in range(n):
        j = (i + 1) % n
        draw_strut(ax, top[i], top[j], color=color, alpha=alpha_edge, lw=lw,
                   registry=registry, label=f'{label}_top_e{i}')
        draw_strut(ax, bottom[i], bottom[j], color=color, alpha=alpha_edge, lw=lw,
                   registry=registry, label=f'{label}_bot_e{i}')
        draw_strut(ax, top[i], bottom[i], color=color, alpha=alpha_edge, lw=lw,
                   registry=registry, label=f'{label}_side_e{i}')


def draw_prism_between_rings(ax, bv, tv, color, alpha_face, alpha_edge, lw,
                             registry=None, label='prism',
                             draw_tube_outline=False, rim_outline=True):
    """
    Closed prism between two parallel n-gons (same vertex order): triangulated
    caps plus one quad per side. For 2.5D hubs; no thick line tubes by default.
    """
    bv = np.asarray(bv, float)
    tv = np.asarray(tv, float)
    n = len(bv)
    if n < 3 or len(tv) != n:
        return
    if np.max(np.linalg.norm(tv - bv, axis=1)) < 1e-9:
        return

    fc = to_rgba(color, alpha_face)
    ec = to_rgba(color, alpha_edge) if rim_outline else 'none'
    ewl = max(lw * 0.4, 0.4) if rim_outline else 0.0

    for i in range(1, n - 1):
        tri = [bv[0], bv[i + 1], bv[i]]
        ax.add_collection3d(Poly3DCollection(
            [tri], facecolors=[fc], edgecolors=[ec], linewidths=ewl))
    for i in range(1, n - 1):
        tri = [tv[0], tv[i], tv[i + 1]]
        ax.add_collection3d(Poly3DCollection(
            [tri], facecolors=[fc], edgecolors=[ec], linewidths=ewl))
    for i in range(n):
        j = (i + 1) % n
        quad = np.array([bv[i], bv[j], tv[j], tv[i]])
        ax.add_collection3d(Poly3DCollection(
            [quad], facecolors=[fc], edgecolors=[ec], linewidths=ewl))

    if not draw_tube_outline:
        return
    for i in range(n):
        j = (i + 1) % n
        draw_strut(ax, bv[i], bv[j], color=color, alpha=alpha_edge, lw=lw,
                   registry=registry, label=f'{label}_bot_e{i}')
        draw_strut(ax, tv[i], tv[j], color=color, alpha=alpha_edge, lw=lw,
                   registry=registry, label=f'{label}_top_e{i}')
        draw_strut(ax, bv[i], tv[i], color=color, alpha=alpha_edge, lw=lw,
                   registry=registry, label=f'{label}_side_v{i}')


# ==========================
# Intersection checker
# ==========================
def count_curve_intersections(curve_list, threshold_frac=0.05):
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
    if mode in [3]:          # random 3D
        points = np.random.rand(n_points, 3)
        tri    = Delaunay(points)
        return points, tri

    elif mode in [6]:        # grid 3D
        cbrt = max(2, round(n_points ** (1/3)))
        # triangulate_3d_grid_symmetric already returns (points, MockTri)
        # do NOT wrap in another MockTri
        points, tri = triangulate_3d_grid_symmetric(cbrt, cbrt, cbrt, center_aligned=True)
        return points, tri

    elif mode in [4, 5]:     # grid 2D / 2.5D
        def factor_pair(n):
            for i in range(int(np.sqrt(n)), 0, -1):
                if n % i == 0:
                    return i, n // i
            return 1, n
        nx2d, ny2d = factor_pair(n_points)
        # triangulate_grid_symmetric already returns (points, MockTri)
        points, tri = triangulate_grid_symmetric(nx2d, ny2d, center_aligned=True)
        return points, tri

    else:                    # random 2D / 2.5D
        points = np.random.rand(n_points, 2)
        tri    = Delaunay(points)
        return points, tri

# ==========================
# 1b) Generate lattice points
# ==========================
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
    
    # Points are already normalized in generate_points for grid modes
    # For random modes, normalize them here
    if mode in [1, 2]:  # random 2D modes
        mins     = pts.min(axis=0)
        maxs     = pts.max(axis=0)
        pts_norm = (pts - mins) / (maxs - mins + 1e-12)
    elif mode == 3:  # random 3D mode
        mins     = pts.min(axis=0)
        maxs     = pts.max(axis=0)
        pts_norm = (pts - mins) / (maxs - mins + 1e-12)
    else:  # grid modes (4, 5, 6) - already normalized to [0,1]
        pts_norm = pts

    if mode in [3, 6]:       # already 3D
        for idx, pt in enumerate(pts_norm):
            nodes[f"L3D_{idx}"] = pt.astype(float)
    elif mode in [1, 4]:     # 2D, single layer at z=0
        for idx, pt in enumerate(pts_norm):
            nodes[f"L2D_{idx}_0"] = np.array([pt[0], pt[1], 0.0])
    elif mode in [2, 5]:     # 2.5D, multiple z layers
        for z_idx in range(nz_layers):
            z = z_idx / (nz_layers - 1) if nz_layers > 1 else 0.0
            for idx, pt in enumerate(pts_norm):
                nodes[f"L2D_{idx}_{z_idx}"] = np.array([pt[0], pt[1], z])
    return nodes

nodes_combined = add_lattice_to_3d(nodes_3d, points_nd, mode, nz_layers=nz_layers)

# ==========================
# 4) Geometry builders (shared between dry-run and plot)
# ==========================

def build_simplex_curves_3d(points_3d, tri, ratio):
    """
    Dry-run geometry builder for 3D modes (3, 6).
    Each simplex is a tetrahedron. Returns list of (pts, label).
    """
    curves = []

    def fake_draw(p0, p1, label='s'):
        p0 = np.asarray(p0, float)
        p1 = np.asarray(p1, float)
        if np.linalg.norm(p1 - p0) < 1e-12:
            return
        curves.append((np.array([p0, p1]), label))

    def fake_polygon(verts_3d, _centroid_3d, label='p'):
        n = len(verts_3d)
        for i in range(n):
            j = (i + 1) % n
            fake_draw(verts_3d[i], verts_3d[j], label=f'{label}_e{i}')

    # Points should already be normalized to [0,1]^3 for grid modes
    # For random mode 3, normalize them
    pts_norm = points_3d
    if pts_norm.min() < -0.1 or pts_norm.max() > 1.1:  # check if not normalized
        mins     = pts_norm.min(axis=0)
        maxs     = pts_norm.max(axis=0)
        pts_norm = (pts_norm - mins) / (maxs - mins + 1e-12)

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
            fake_polygon(face_verts, face_cent, label=f'tet{tuple(simplex)}_face{fi}')

        # Group shrunk corners by original vertex for connector logic
        for i, vertex in enumerate(pts_norm[simplex]):
            key = tuple(vertex)
            groups[key].append(shrunk[i])
            centroids_map[key].append(centroid)

    # Connectors between shrunk corners at shared vertices
    for key, pts_list in groups.items():
        centroids_list = centroids_map[key]
        if len(pts_list) == 2:
            p0, p1 = pts_list[0], pts_list[1]
            fake_draw(p0, p1, label=f'strut_{key}')
        elif len(pts_list) >= 3:
            pts_arr    = np.array(pts_list)
            hub_center = pts_arr.mean(axis=0)
            ordered    = convex_order_3d(pts_arr)
            if ordered is not None:
                fake_polygon(ordered, hub_center, label=f'hub_{key}')
    return curves


def build_simplex_curves_3d(points_3d, tri, ratio):
    """
    Dry-run geometry builder for 3D modes (3, 6).
    Each simplex is a tetrahedron. Returns list of (pts, label).
    """
    curves = []

    def fake_draw(p0, p1, label='s'):
        p0 = np.asarray(p0, float)
        p1 = np.asarray(p1, float)
        if np.linalg.norm(p1 - p0) < 1e-12:
            return
        curves.append((np.array([p0, p1]), label))

    def fake_polygon(verts_3d, _centroid_3d, label='p'):
        n = len(verts_3d)
        for i in range(n):
            j = (i + 1) % n
            fake_draw(verts_3d[i], verts_3d[j], label=f'{label}_e{i}')

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
            fake_polygon(face_verts, face_cent, label=f'tet{tuple(simplex)}_face{fi}')

        # Group shrunk corners by original vertex for connector logic
        for i, vertex in enumerate(pts_norm[simplex]):
            key = tuple(vertex)
            groups[key].append(shrunk[i])
            centroids_map[key].append(centroid)

    # Connectors between shrunk corners at shared vertices
    for key, pts_list in groups.items():
        centroids_list = centroids_map[key]
        if len(pts_list) == 2:
            p0, p1 = pts_list[0], pts_list[1]
            fake_draw(p0, p1, label=f'strut_{key}')
        elif len(pts_list) >= 3:
                pts_arr    = np.array(pts_list)
                hub_center = pts_arr.mean(axis=0)
                ordered    = convex_order_3d(pts_arr)
                if ordered is not None:
                    fake_polygon(ordered, hub_center, label=f'hub_{key}')
    return curves

# ==========================
# 5) Plot function
# ==========================
def plot_3d_lattice(nodes, points_nd, tri, ratio, mode, nz_layers=3, show_nodes=False):
    if intersect_check:
        print("\n--- Intersection Check ---")
        if mode in [3, 6]:
            def curve_builder():
                return build_simplex_curves_3d(points_nd, tri, ratio)
        else:
            layers = [0] if mode in [1, 4] else list(range(nz_layers))
            def curve_builder():
                all_curves = []
                for z_idx in layers:
                    z_val = 0.0 if mode in [1, 4] else z_idx / (nz_layers - 1)
                    all_curves += build_simplex_curves_2d(
                        points_nd, tri, ratio, z_val, mode, nz_layers, z_idx)
                return all_curves

        dry_curves = curve_builder()
        n_total    = len(dry_curves)
        n_inter    = count_curve_intersections(dry_curves, intersect_threshold)
        ratio_val  = n_inter / n_total if n_total > 0 else 0.0

        print(f"Segments checked : {n_total}")
        print(f"Intersections    : {n_inter} ({100*ratio_val:.1f}%)")
        print("--------------------------\n")

    fig = plt.figure(figsize=(8, 6))
    ax  = fig.add_subplot(111, projection='3d')
    registry = [] if intersect_check else None

    # ---- 3D modes (3, 6) ----
    if mode in [3, 6]:
        # Points should already be normalized for grid mode
        pts_norm = points_nd
        if pts_norm.min() < -0.1 or pts_norm.max() > 1.1:  # check if not normalized
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
                draw_filled_polygon(
                    ax, face_verts, face_cent,
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
        # Instead of the existing hub polygon block:
        for key, pts_list in groups.items():
            if len(pts_list) == 2:
                p0, p1 = pts_list[0], pts_list[1]
                draw_strut(ax, p0, p1, registry=registry, label=f'strut_{key}')
            elif len(pts_list) >= 3:
                # hub_center is the original lattice node position
                hub_center = np.array(key, float)  # key is tuple of original vertex coords
                # Scale: half the mean strut length so hubs don't overlap
                all_pts = np.array(pts_list)
                mean_strut = np.mean([np.linalg.norm(p - hub_center) for p in all_pts])
                scale = mean_strut * 0.45
                draw_truncated_octahedron(ax, hub_center, scale,
                                        registry=registry, label=f'hub_{key}')

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
                draw_filled_polygon(
                    ax, tri_pts_3d, centroid_3d,
                    color='magenta',
                    alpha_face=0.25, alpha_edge=0.9, lw=1.8,
                    registry=registry, label=f'tri{tuple(simplex)}_z0')

            elif mode in [2, 5]:
                for z_idx in range(nz_layers - 1):
                    zb = z_idx / (nz_layers - 1)
                    zt = (z_idx + 1) / (nz_layers - 1)
                    bottom_pts = np.hstack([tri_pts_2d, np.full((3, 1), zb)])
                    top_pts    = np.hstack([tri_pts_2d, np.full((3, 1), zt)])
                    draw_filled_polygon(
                        ax, top_pts, np.append(centroid_2d, zt),
                        color='magenta',
                        alpha_face=0.25, alpha_edge=0.9, lw=1.8,
                        registry=registry,
                        label=f'tri{tuple(simplex)}_top_z{z_idx}')
                    for i in range(3):
                        j    = (i + 1) % 3
                        quad = np.array([bottom_pts[i], bottom_pts[j],
                                         top_pts[j],    top_pts[i]])
                        draw_filled_polygon(
                            ax, quad, quad.mean(axis=0),
                            color='magenta',
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
                    p0, p1 = pts_list[0], pts_list[1]
                    if mode in [1, 4]:
                        draw_strut(ax, p0, p1, registry=registry,
                                   label=f'strut_{white}')
                    else:
                        zb = z_idx / (nz_layers - 1)
                        zt = (z_idx+1)/(nz_layers-1) if z_idx < nz_layers-1 else zb
                        p0b = np.array([p0[0], p0[1], zb])
                        p1b = np.array([p1[0], p1[1], zb])
                        p0t = np.array([p0[0], p0[1], zt])
                        p1t = np.array([p1[0], p1[1], zt])
                        draw_strut(ax, p0b, p1b, registry=registry,
                                   label=f'strut_{white}_bot_z{z_idx}')
                        draw_strut(ax, p0t, p1t, registry=registry,
                                   label=f'strut_{white}_top_z{z_idx}')
                        draw_strut(ax, p0b, p0t, registry=registry,
                                   label=f'vert0_{white}_z{z_idx}')
                        draw_strut(ax, p1b, p1t, registry=registry,
                                   label=f'vert1_{white}_z{z_idx}')

                elif len(pts_list) >= 3:
                    try:
                        pts_arr    = np.array(pts_list)
                        hub_center = pts_arr.mean(axis=0)
                        hull       = ConvexHull(pts_arr[:, :2])
                        vertices   = pts_arr[hull.vertices, :2]
                        n_verts    = len(vertices)
                        if mode in [1, 4]:
                            base2 = order_hub_ring_xy(vertices, z_val)
                            verts_3d = np.hstack([
                                base2, np.full((len(base2), 1), z_val)])
                            draw_solid_polygon(
                                ax, verts_3d, hub_center,
                                thickness=ngon_thickness,
                                color='cyan',
                                alpha_face=0.25, alpha_edge=0.9, lw=1.8,
                                registry=registry,
                                label=f'hub_{white}_z{z_idx}',
                                draw_tube_outline=False,
                                rim_outline=True)
                        else:
                            zb = z_idx / (nz_layers - 1)
                            zt = (z_idx+1)/(nz_layers-1) if z_idx < nz_layers-1 else zb
                            base2 = order_hub_ring_xy(vertices, zb)
                            nr = len(base2)
                            bv = np.hstack([base2, np.full((nr, 1), zb)])
                            tv = np.hstack([base2, np.full((nr, 1), zt)])
                            if abs(zt - zb) < 1e-9:
                                draw_solid_polygon(
                                    ax, bv, np.append(hub_center[:2], zb),
                                    thickness=ngon_thickness,
                                    color='cyan',
                                    alpha_face=0.25, alpha_edge=0.9, lw=1.8,
                                    registry=registry,
                                    label=f'hub_{white}_z{z_idx}',
                                    draw_tube_outline=False,
                                    rim_outline=True)
                            else:
                                draw_prism_between_rings(
                                    ax, bv, tv,
                                    color='cyan',
                                    alpha_face=0.25, alpha_edge=0.9, lw=1.8,
                                    registry=registry,
                                    label=f'hub_{white}_z{z_idx}')
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

def add_truncated_octahedron_faces(center, scale, face_verts_list):
    """Add all 14 faces of a truncated octahedron to face_verts_list for mesh export."""
    verts, sq_faces, hex_faces = make_truncated_octahedron(center, scale)
    for face in sq_faces + hex_faces:
        face_verts_list.append(verts[face])


def is_central_hub(pts_list):
    """
    True when shrunk corners arrive from all 8 octants — i.e. this is
    a full truncated-octahedron hub, not a simpler edge/face hub.
    """
    if len(pts_list) < 8:
        return False
    pts = np.array(pts_list)
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
# 7) OpenSCAD / mesh export (STL, OBJ)
# ==========================

# --- Export settings (edit these) ---
export_enabled   = True         # set True to generate .scad and .stl on run
# Paths are relative to the script location; use absolute paths to write elsewhere
export_scad_path = "auxetic_lattice.scad"
export_stl_path  = "auxetic_lattice.stl"
export_obj_path  = "auxetic_lattice.obj"
export_png_path  = None          # set to e.g. "auxetic_lattice.png" to render a preview
# Fusion Insert Mesh treats both STL and OBJ as mesh bodies. OBJ often carries
# explicit normals; disable STL (export_stl = False in USER SETTINGS) if you only want OBJ.
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

def scad_polyline_tube(pts, radius, fn):
    """
    Union of cylinder segments along a polyline (straight struts: one segment).
    pts: (N, 3) points along the strut.
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

def collect_export_geometry(points_nd, tri, ratio, mode, nz_layers):
    """
    Walk the same geometry logic as the plotter but collect:
      strut_curves : list of (N,3) point arrays  — one per strut
      face_verts   : list of (M,3) vertex arrays — one per face
    No matplotlib involved.
    """
    strut_curves = []
    face_verts_list = []

    def add_strut(p0, p1):
        p0 = np.asarray(p0, float)
        p1 = np.asarray(p1, float)
        if np.linalg.norm(p1 - p0) < 1e-9:
            return
        strut_curves.append(np.array([p0, p1], dtype=float))

    def add_face(verts_3d, _centroid_3d):
        face_verts_list.append(np.asarray(verts_3d, float))

    # ---- 3D modes ----
    if mode in [3, 6]:
        # Points should already be normalized for grid mode
        pts_norm = points_nd
        if pts_norm.min() < -0.1 or pts_norm.max() > 1.1:  # check if not normalized
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
                add_face(fv, fc)
            for i, vertex in enumerate(pts_norm[simplex]):
                key = tuple(vertex)
                groups[key].append(shrunk[i])
                centroids_map[key].append(centroid)

        for key, pts_list in groups.items():
            centroids_list = centroids_map[key]
            if len(pts_list) == 2:
                p0, p1 = pts_list[0], pts_list[1]
                add_strut(p0, p1)
            for key, pts_list in groups.items():
                if len(pts_list) == 2:
                    p0, p1 = pts_list[0], pts_list[1]
                    add_strut(p0, p1)
                elif len(pts_list) >= 3:
                    hub_center = np.array(key, float)
                    if is_central_hub(pts_list):
                        # Full truncated octahedron — same logic as the plotter
                        mean_dist = np.mean([np.linalg.norm(p - hub_center)
                                                for p in pts_list])
                        scale = mean_dist * 0.45
                        add_truncated_octahedron_faces(hub_center, scale, face_verts_list)
                    else:
                        # Normal hub — convex polygon extruded as a thin prism
                        pts_arr = np.array(pts_list)
                        ordered = convex_order_3d(pts_arr)
                        if ordered is not None:
                            add_face(ordered, hub_center)

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
                add_face(tri_pts_3d, centroid_3d)
            elif mode in [2,5]:
                for z_idx in range(nz_layers-1):
                    zb = z_idx/(nz_layers-1); zt = (z_idx+1)/(nz_layers-1)
                    bp = np.hstack([tri_pts_2d, np.full((3,1),zb)])
                    tp = np.hstack([tri_pts_2d, np.full((3,1),zt)])
                    add_face(tp, np.append(centroid_2d,zt))
                    for i in range(3):
                        j    = (i+1)%3
                        quad = np.array([bp[i],bp[j],tp[j],tp[i]])
                        add_face(quad, quad.mean(axis=0))

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
                    p0, p1 = pts_list[0], pts_list[1]
                    if mode in [1,4]:
                        add_strut(p0, p1)
                    else:
                        zb = z_idx/(nz_layers-1)
                        zt = (z_idx+1)/(nz_layers-1) if z_idx<nz_layers-1 else zb
                        p0b=np.array([p0[0],p0[1],zb]); p1b=np.array([p1[0],p1[1],zb])
                        p0t=np.array([p0[0],p0[1],zt]); p1t=np.array([p1[0],p1[1],zt])
                        add_strut(p0b, p1b)
                        add_strut(p0t, p1t)
                        add_strut(p0b, p0t)
                        add_strut(p1b, p1t)
                elif len(pts_list) >= 3:
                    try:
                        pts_arr    = np.array(pts_list)
                        hub_center = pts_arr.mean(axis=0)
                        hull       = ConvexHull(pts_arr[:,:2])
                        vertices   = pts_arr[hull.vertices,:2]
                        n_verts    = len(vertices)
                        if mode in [1,4]:
                            base2 = order_hub_ring_xy(vertices, z_val)
                            verts_3d = np.hstack([
                                base2, np.full((len(base2), 1), z_val)])
                            add_face(verts_3d, hub_center)
                        else:
                            zb = z_idx/(nz_layers-1)
                            zt = (z_idx+1)/(nz_layers-1) if z_idx<nz_layers-1 else zb
                            base2 = order_hub_ring_xy(vertices, zb)
                            nr = len(base2)
                            bv = np.hstack([base2, np.full((nr, 1), zb)])
                            tv = np.hstack([base2, np.full((nr, 1), zt)])
                            add_face(tv, np.append(hub_center[:2],zt))
                            add_face(bv, np.append(hub_center[:2],zb))
                            if abs(zt - zb) >= 1e-9:
                                for i in range(nr):
                                    j = (i + 1) % nr
                                    side = np.array([bv[i], bv[j], tv[j], tv[i]])
                                    add_face(side, side.mean(axis=0))
                    except Exception:
                        pass

    return strut_curves, face_verts_list


def export_to_scad(scad_path, strut_curves, face_verts_list):
    """Write a .scad file unioning all struts and faces."""
    lines = [
        f"// Auto-generated by lattice_viz.py\n",
        f"// mode={mode}  n_points={n_points}  ratio={ratio}\n\n",
        f"$fn = {scad_segments};\n\n",
        "render(convexity=10)\n",
        "union() {\n",
    ]

    # Struts
    if strut_curves:
        lines.append("  // --- struts ---\n")
        for pts in strut_curves:
            lines.append(scad_polyline_tube(pts, strut_radius, scad_segments))

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


def build_export_triangles(strut_curves, face_verts_list):
    """
    Build the same triangle soup used for static mesh exports (STL / OBJ).
    Each triangle is a list of three 3D points.
    """
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

        if len(rings) < 2:
            return []

        tris = []
        for i in range(len(rings) - 1):
            r0, r1 = rings[i], rings[i + 1]
            seg = len(r0)
            for s in range(seg):
                s1 = (s + 1) % seg
                tris.append([r0[s], r1[s],  r1[s1]])
                tris.append([r0[s], r1[s1], r0[s1]])
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

    return all_triangles


def export_stl_direct(stl_path, strut_curves, face_verts_list, triangles=None):
    """
    Write STL directly from Python using numpy-stl.
    No OpenSCAD required.
    Faces are written as thin solid prisms; struts as tube meshes.
    Pass triangles= to reuse a soup from build_export_triangles (avoids duplicate work with OBJ).
    """
    import os
    try:
        from stl import mesh as stl_mesh
    except ImportError:
        print("  numpy-stl not installed. Run: pip install numpy-stl")
        return False

    all_triangles = triangles
    if all_triangles is None:
        all_triangles = build_export_triangles(strut_curves, face_verts_list)
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


def export_obj_direct(obj_path, strut_curves, face_verts_list, triangles=None):
    """Write an ASCII Wavefront .obj (triangles + per-face normals). No extra packages."""
    import os

    all_triangles = triangles
    if all_triangles is None:
        all_triangles = build_export_triangles(strut_curves, face_verts_list)
    if not all_triangles:
        print("  No geometry to export (OBJ).")
        return False

    v_count = 0
    n_count = 0
    lines = ["# auxetic lattice export\n",
             f"# mode={mode} n_points={n_points} ratio={ratio}\n"]
    for tri in all_triangles:
        a, b, c = np.asarray(tri[0], float), np.asarray(tri[1], float), np.asarray(tri[2], float)
        nvec = np.cross(b - a, c - a)
        nn = np.linalg.norm(nvec)
        if nn < 1e-14:
            continue
        nvec = nvec / nn
        v_base = v_count + 1
        for p in (a, b, c):
            lines.append(f"v {p[0]:.9g} {p[1]:.9g} {p[2]:.9g}\n")
            v_count += 1
        n_count += 1
        lines.append(f"vn {nvec[0]:.9g} {nvec[1]:.9g} {nvec[2]:.9g}\n")
        lines.append(
            f"f {v_base}//{n_count} {v_base + 1}//{n_count} {v_base + 2}//{n_count}\n")

    with open(obj_path, "w", encoding="utf-8", newline="\n") as f:
        f.writelines(lines)
    size_kb = os.path.getsize(obj_path) // 1024
    print(f"  OBJ ready: {obj_path}  ({size_kb} KB, {n_count} tris)")
    return True


def run_export(points_nd, tri, ratio, mode, nz_layers):
    """Entry point: collect geometry; write .scad / .stl / .obj per USER SETTINGS flags."""
    import os
    print("\n--- Export ---")

    script_dir = os.path.dirname(os.path.abspath(__file__))

    def resolve(p):
        return p if os.path.isabs(p) else os.path.join(script_dir, p)

    scad_abs = resolve(export_scad_path)
    stl_abs  = resolve(export_stl_path)
    obj_abs  = resolve(export_obj_path)

    strut_curves, face_verts_list = collect_export_geometry(
        points_nd, tri, ratio, mode, nz_layers
    )
    print(f"  Geometry: {len(strut_curves)} struts, {len(face_verts_list)} faces")

    if export_scad:
        export_to_scad(scad_abs, strut_curves, face_verts_list)

    mesh_triangles = None
    if export_stl or export_obj:
        mesh_triangles = build_export_triangles(strut_curves, face_verts_list)
    if export_stl:
        export_stl_direct(stl_abs, strut_curves, face_verts_list, triangles=mesh_triangles)
    if export_obj:
        export_obj_direct(obj_abs, strut_curves, face_verts_list, triangles=mesh_triangles)

    print("--------------\n")

# ==========================
# 8) Run — single execution
# ==========================

# Export runs first so it doesn't block on the plot window
if export_enabled:
    run_export(points_nd, tri_nd, ratio, mode, nz_layers)

plot_3d_lattice(nodes_combined, points_nd, tri_nd, ratio, mode,
                nz_layers=nz_layers, show_nodes=False)
