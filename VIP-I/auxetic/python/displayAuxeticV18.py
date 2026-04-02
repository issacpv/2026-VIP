import numpy as np
from itertools import permutations
from scipy.spatial import Delaunay, ConvexHull

# ==========================
# USER SETTINGS
# ==========================
mode        = 6
n_points    = 16
ratio       = 0.5
nz_layers   = 2

# --- Shape settings ---
ngon_thickness  = 0.03
hub_size_factor = 0.75

# --- Export settings ---
export_scad      = False
export_stl       = True
export_obj       = True
export_scad_path = "auxetic_lattice.scad"
export_stl_path  = "auxetic_lattice.stl"
export_obj_path  = "auxetic_lattice.obj"

strut_radius   = 0.02
face_thickness = 0.015
scad_segments  = 8


# ==========================
# TRIANGULATION
# ==========================

def triangulate_3d_grid_symmetric(nx, ny, nz):
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
# TRUNCATED CUBOCTAHEDRON
# ==========================

def make_truncated_cuboctahedron(center, scale):
    a, b, c = 1.0, 1.0 + np.sqrt(2), 1.0 + 2 * np.sqrt(2)
    raw = set()
    for perm in permutations([a, b, c]):
        for sx in [1, -1]:
            for sy in [1, -1]:
                for sz in [1, -1]:
                    raw.add((perm[0]*sx, perm[1]*sy, perm[2]*sz))

    verts   = np.array(sorted(raw), dtype=float)
    mean_r  = np.mean(np.linalg.norm(verts, axis=1))
    verts   = verts * (scale / mean_r)
    center  = np.asarray(center, float)
    verts  += center

    def collect_face(normal_axis, sign, expected_count):
        coords    = (verts - center)[:, normal_axis]
        threshold = sign * (coords.max() if sign > 0 else -coords.min()) * 0.999
        face      = [i for i, v in enumerate(verts - center)
                     if sign * v[normal_axis] >= threshold]
        return face if len(face) == expected_count else []

    oct_faces = []
    for axis in range(3):
        for sign in [1, -1]:
            face = collect_face(axis, sign, 8)
            if face:
                pts   = verts[face]
                fc    = pts.mean(axis=0)
                other = [a for a in range(3) if a != axis]
                angles = np.arctan2(
                    (pts - fc)[:, other[1]],
                    (pts - fc)[:, other[0]])
                oct_faces.append([face[i] for i in np.argsort(angles)])

    hex_faces = []
    for sx in [1, -1]:
        for sy in [1, -1]:
            for sz in [1, -1]:
                diag      = np.array([sx, sy, sz], float)
                dots      = (verts - center) @ diag
                threshold = dots.max() * 0.999
                face      = [i for i, d in enumerate(dots) if d >= threshold]
                if len(face) == 6:
                    pts  = verts[face]
                    fc   = pts.mean(axis=0)
                    dn   = diag / np.linalg.norm(diag)
                    u    = np.cross(dn, [1, 0, 0])
                    if np.linalg.norm(u) < 1e-6:
                        u = np.cross(dn, [0, 1, 0])
                    u   /= np.linalg.norm(u)
                    v_ax = np.cross(dn, u)
                    off  = pts - fc
                    angles = np.arctan2(off @ v_ax, off @ u)
                    hex_faces.append([face[i] for i in np.argsort(angles)])

    sq_faces    = []
    edge_normals = []
    for i in range(3):
        for j in range(3):
            if i != j:
                for si in [1, -1]:
                    for sj in [1, -1]:
                        n = np.zeros(3)
                        n[i] = si; n[j] = sj
                        edge_normals.append(n / np.linalg.norm(n))
    seen = set()
    for n in edge_normals:
        nkey = tuple(np.round(n, 6))
        if nkey in seen: continue
        seen.add(nkey)
        dots      = (verts - center) @ n
        threshold = dots.max() * 0.999
        face      = [i for i, d in enumerate(dots) if d >= threshold]
        if len(face) == 4:
            pts  = verts[face]
            fc   = pts.mean(axis=0)
            u    = np.cross(n, [1, 0, 0])
            if np.linalg.norm(u) < 1e-6:
                u = np.cross(n, [0, 1, 0])
            u   /= np.linalg.norm(u)
            v_ax = np.cross(n, u)
            off  = pts - fc
            angles = np.arctan2(off @ v_ax, off @ u)
            sq_faces.append([face[i] for i in np.argsort(angles)])

    return verts, oct_faces, hex_faces, sq_faces


def is_central_hub(pts_list):
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

def unit(v):
    v = np.asarray(v, float)
    n = np.linalg.norm(v)
    return np.zeros_like(v) if n < 1e-12 else v / n


def convex_order_3d(pts):
    pts      = np.asarray(pts, float)
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
    v      = v / np.linalg.norm(v)
    coords = offsets @ np.stack([u, v], axis=1)
    return pts[np.argsort(np.arctan2(coords[:, 1], coords[:, 0]))]


def newell_normal(poly):
    n      = len(poly)
    normal = np.zeros(3)
    for i in range(n):
        j         = (i + 1) % n
        normal[0] += (poly[i][1] - poly[j][1]) * (poly[i][2] + poly[j][2])
        normal[1] += (poly[i][2] - poly[j][2]) * (poly[i][0] + poly[j][0])
        normal[2] += (poly[i][0] - poly[j][0]) * (poly[i][1] + poly[j][1])
    nn = np.linalg.norm(normal)
    return normal / nn if nn > 1e-10 else None


def order_hub_ring_xy(vertices_xy, z_plane):
    xy = np.asarray(vertices_xy, float)
    if len(xy) < 3:
        return xy
    v3 = np.hstack([xy, np.full((len(xy), 1), z_plane)])
    o  = convex_order_3d(v3)
    return o[:, :2] if o is not None else xy


# ==========================
# SOLID MESH BUILDERS
# ==========================

def triangles_for_solid_tetrahedron(tet_verts):
    v          = np.asarray(tet_verts, float)
    triangles  = []
    face_indices = [(1, 2, 3), (0, 3, 2), (0, 1, 3), (0, 2, 1)]
    for fi in face_indices:
        a, b, c  = v[fi[0]], v[fi[1]], v[fi[2]]
        opposite = v[list(set(range(4)) - set(fi))[0]]
        normal   = np.cross(b - a, c - a)
        if np.dot(normal, a - opposite) < 0:
            b, c = c, b
        triangles.append([a, b, c])
    return triangles


def triangles_for_convex_solid(pts):
    pts = np.asarray(pts, float)
    if len(pts) < 4:
        return []
    try:
        hull = ConvexHull(pts)
    except Exception:
        return []
    centroid  = pts.mean(axis=0)
    triangles = []
    for simplex in hull.simplices:
        a, b, c = pts[simplex[0]], pts[simplex[1]], pts[simplex[2]]
        normal  = np.cross(b - a, c - a)
        if np.dot(normal, a - centroid) < 0:
            b, c = c, b
        triangles.append([a, b, c])
    return triangles


def _extrude_polygon_solid(verts_3d):
    verts_3d   = np.asarray(verts_3d, float)
    normal     = newell_normal(verts_3d)
    if normal is None:
        return []
    top        = verts_3d
    bottom     = verts_3d - normal * ngon_thickness
    n          = len(verts_3d)
    tris       = []
    centroid_t = top.mean(axis=0)
    for i in range(1, n - 1):
        t0, ti, ti1 = top[0], top[i], top[i + 1]
        if np.dot(np.cross(ti - t0, ti1 - t0), normal) < 0:
            tris.append([t0, ti1, ti])
        else:
            tris.append([t0, ti, ti1])
        b0, bi, bi1 = bottom[0], bottom[i], bottom[i + 1]
        if np.dot(np.cross(bi - b0, bi1 - b0), -normal) < 0:
            tris.append([b0, bi1, bi])
        else:
            tris.append([b0, bi, bi1])
    for i in range(n):
        j        = (i + 1) % n
        a, b, c, d = top[i], top[j], bottom[j], bottom[i]
        mid      = (a + b + c + d) / 4.0
        n1       = np.cross(b - a, d - a)
        if np.dot(n1, mid - centroid_t) < 0:
            tris.append([a, d, b])
            tris.append([b, d, c])
        else:
            tris.append([a, b, d])
            tris.append([b, c, d])
    return tris


# ==========================
# HUB EXPORT
# ==========================

def _hub_scale_for_tcoh(hub_center, pts_list):
    mean_dist = np.mean([np.linalg.norm(p - hub_center) for p in pts_list])
    a, b, c   = 1.0, 1.0 + np.sqrt(2), 1.0 + 2 * np.sqrt(2)
    all_verts = []
    for perm in permutations([a, b, c]):
        for sx in [1, -1]:
            for sy in [1, -1]:
                for sz in [1, -1]:
                    all_verts.append([perm[0]*sx, perm[1]*sy, perm[2]*sz])
    all_verts = np.array(all_verts, dtype=float)
    mean_r    = np.mean(np.linalg.norm(all_verts, axis=1))
    oct_fc_dist = c / mean_r
    return (mean_dist / oct_fc_dist) * hub_size_factor


def dispatch_hub_export(key, pts_list, all_triangles):
    hub_center = np.array(key, float)
    if is_central_hub(pts_list):
        scale                         = _hub_scale_for_tcoh(hub_center, pts_list)
        verts, oct_faces, hex_faces, sq_faces = make_truncated_cuboctahedron(hub_center, scale)
        all_triangles.extend(triangles_for_convex_solid(verts))
    else:
        pts_arr = np.array(pts_list)
        ordered = convex_order_3d(pts_arr)
        if ordered is not None:
            all_triangles.extend(_extrude_polygon_solid(ordered))


# ==========================
# 3D GROUP BUILDER
# ==========================

def build_3d_groups(pts_norm, tri, ratio):
    groups = {tuple(p): [] for p in pts_norm}
    for simplex in tri.simplices:
        tet      = pts_norm[simplex]
        centroid = tet.mean(axis=0)
        shrunk   = np.array([(1 - ratio) * tet[i] + ratio * centroid
                              for i in range(4)])
        for i, vertex in enumerate(pts_norm[simplex]):
            groups[tuple(vertex)].append(shrunk[i])
    return groups


# ==========================
# GEOMETRY COLLECTION
# ==========================

def collect_export_geometry(points_nd, tri, ratio, mode, nz_layers):
    strut_curves  = []
    all_triangles = []

    def add_strut(p0, p1):
        p0 = np.asarray(p0, float)
        p1 = np.asarray(p1, float)
        if np.linalg.norm(p1 - p0) > 1e-9:
            strut_curves.append(np.array([p0, p1]))

    if mode in [3, 6]:
        pts_norm = points_nd
        groups   = build_3d_groups(pts_norm, tri, ratio)

        for simplex in tri.simplices:
            tet      = pts_norm[simplex]
            centroid = tet.mean(axis=0)
            shrunk   = np.array([(1 - ratio) * tet[i] + ratio * centroid
                                  for i in range(4)])
            all_triangles.extend(triangles_for_solid_tetrahedron(shrunk))

        for key, pts_list in groups.items():
            if len(pts_list) == 2:
                add_strut(pts_list[0], pts_list[1])
            elif len(pts_list) >= 3:
                dispatch_hub_export(key, pts_list, all_triangles)

    else:
        points_2d = points_nd
        layers    = [0] if mode in [1, 4] else list(range(nz_layers))

        for simplex in tri.simplices:
            tri_pts = points_2d[simplex]
            c2d     = tri_pts.mean(axis=0)
            t2d     = np.array([(1 - ratio) * tri_pts[i] + ratio * c2d
                                 for i in range(3)])
            if mode in [1, 4]:
                face3 = np.hstack([t2d, np.zeros((3, 1))])
                all_triangles.extend(_extrude_polygon_solid(face3))
            elif mode in [2, 5]:
                for z_idx in range(nz_layers - 1):
                    zb = z_idx / (nz_layers - 1)
                    zt = (z_idx + 1) / (nz_layers - 1)
                    bp = np.hstack([t2d, np.full((3, 1), zb)])
                    tp = np.hstack([t2d, np.full((3, 1), zt)])
                    all_triangles.extend(_extrude_polygon_solid(tp))
                    for i in range(3):
                        j    = (i + 1) % 3
                        quad = np.array([bp[i], bp[j], tp[j], tp[i]])
                        all_triangles.extend(_extrude_polygon_solid(quad))

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
                            v3d  = np.hstack([ring, np.full((len(ring), 1), z_val)])
                            all_triangles.extend(_extrude_polygon_solid(v3d))
                        else:
                            zb   = z_idx / (nz_layers - 1)
                            zt   = (z_idx + 1) / (nz_layers - 1) if z_idx < nz_layers - 1 else zb
                            ring = order_hub_ring_xy(verts, zb)
                            nr   = len(ring)
                            bv   = np.hstack([ring, np.full((nr, 1), zb)])
                            tv   = np.hstack([ring, np.full((nr, 1), zt)])
                            if abs(zt - zb) >= 1e-9:
                                all_triangles.extend(triangles_for_convex_solid(np.vstack([bv, tv])))
                            else:
                                all_triangles.extend(_extrude_polygon_solid(bv))
                    except Exception:
                        pass

    return strut_curves, all_triangles


# ==========================
# MESH BUILD
# ==========================

def build_export_triangles(strut_curves, all_triangles):
    result = list(all_triangles)

    def tube_mesh(path, radius, segments):
        path  = np.asarray(path, float)
        rings = []
        for k, pt in enumerate(path):
            tang = path[1] - path[0] if k == 0 else \
                   path[-1] - path[-2] if k == len(path) - 1 else \
                   path[k + 1] - path[k - 1]
            tn = np.linalg.norm(tang)
            if tn < 1e-12: continue
            tang  = tang / tn
            perp  = np.cross(tang, [0, 0, 1])
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
            tris.append([path[0],  rings[0][-1 - s1],  rings[0][-1 - s]])
            tris.append([path[-1], rings[-1][s],        rings[-1][s1]])
        return tris

    for path in strut_curves:
        result += tube_mesh(path, strut_radius, scad_segments)

    return result


# ==========================
# WRITERS
# ==========================

def _fmt(v):
    return f"[{v[0]:.6f},{v[1]:.6f},{v[2]:.6f}]"


def export_to_scad(scad_path, strut_curves, triangles):
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

    lines = [f"// auxetic lattice  mode={mode}  n_points={n_points}  ratio={ratio}\n",
             f"$fn={scad_segments};\nrender(convexity=10)\nunion(){{\n",
             "  // struts\n"]
    for pts in strut_curves:
        lines.append(scad_cylinder(pts[0], pts[1], strut_radius, scad_segments))
    if triangles:
        all_v, all_f, vi = [], [], 0
        for tri in triangles:
            a, b, c = (np.asarray(tri[k], float) for k in range(3))
            all_v  += [a, b, c]
            all_f.append([vi, vi + 1, vi + 2])
            vi     += 3
        vs = ", ".join(_fmt(v) for v in all_v)
        fs = ", ".join("[" + ",".join(str(x) for x in f) + "]" for f in all_f)
        lines.append(f"polyhedron(points=[{vs}],faces=[{fs}],convexity=10);\n")
    lines.append("}\n")
    with open(scad_path, "w") as f:
        f.writelines(lines)
    print(f"  SCAD: {scad_path}  ({len(strut_curves)} struts, {len(triangles)} tris)")


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
    print(f"  STL: {stl_path}  ({os.path.getsize(stl_path) // 1024} KB)")


def export_obj_direct(obj_path, triangles):
    import os
    lines   = ["# auxetic lattice\n"]
    v_count = 0
    n_count = 0
    for tri in triangles:
        a, b, c = (np.asarray(tri[k], float) for k in range(3))
        nv      = np.cross(b - a, c - a)
        nn      = np.linalg.norm(nv)
        if nn < 1e-14: continue
        nv      = nv / nn
        vb      = v_count + 1
        for p in (a, b, c):
            lines.append(f"v {p[0]:.9g} {p[1]:.9g} {p[2]:.9g}\n")
        v_count += 3
        n_count += 1
        lines.append(f"vn {nv[0]:.9g} {nv[1]:.9g} {nv[2]:.9g}\n")
        lines.append(f"f {vb}//{n_count} {vb+1}//{n_count} {vb+2}//{n_count}\n")
    with open(obj_path, "w", encoding="utf-8", newline="\n") as f:
        f.writelines(lines)
    print(f"  OBJ: {obj_path}  ({os.path.getsize(obj_path) // 1024} KB, {n_count} tris)")


# ==========================
# ENTRY POINT
# ==========================

def run_export(points_nd, tri, ratio, mode, nz_layers):
    import os
    print("\n--- Export ---")
    base    = os.path.dirname(os.path.abspath(__file__))
    resolve = lambda p: p if os.path.isabs(p) else os.path.join(base, p)

    strut_curves, all_triangles = collect_export_geometry(
        points_nd, tri, ratio, mode, nz_layers)
    print(f"  Geometry: {len(strut_curves)} struts, {len(all_triangles)} solid tris")

    triangles = build_export_triangles(strut_curves, all_triangles)
    print(f"  Final mesh: {len(triangles)} triangles (incl. strut tubes)")

    if export_scad:
        export_to_scad(resolve(export_scad_path), strut_curves, triangles)
    if export_stl:
        export_stl_direct(resolve(export_stl_path), triangles)
    if export_obj:
        export_obj_direct(resolve(export_obj_path), triangles)

    print("--------------\n")


points_nd, tri_nd = generate_points(n_points, mode)
run_export(points_nd, tri_nd, ratio, mode, nz_layers)
