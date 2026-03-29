# Auxetic Lattice Generator

This Python script generates auxetic-style lattices from triangulated or tetrahedral point sets (random or grid-based, 2D–3D) and exports them as solid meshes for CAD or 3D printing.

## Features

- Multiple lattice modes:
  - `1`: random 2D triangulated lattice.
  - `2`: random 2.5D lattice (stacked layers).
  - `3`: random 3D Delaunay tetrahedral lattice.
  - `4`: grid-based 2D triangulated lattice.
  - `5`: grid-based 2.5D lattice.
  - `6`: grid-based 3D lattice with symmetric 6-tetrahedra-per-cube decomposition.
- Struts and polygonal hubs built by shrinking simplices toward their centroids with a user-controlled `ratio`.
- Automatic detection of high-connectivity “central hubs” and replacement with truncated-octahedron solids.
- Export options:
  - Binary STL via `numpy-stl`.
  - OBJ with per-triangle normals.
  - OpenSCAD (`.scad`) using cylinders and extruded polyhedra.
- Optional 3D visualization with Matplotlib.

## Dependencies

Required:
- `numpy`
- `matplotlib`
- `scipy` (for `Delaunay` and `ConvexHull`)

Optional:
- `numpy-stl` (for STL export)

Install with:
```bash
pip install numpy matplotlib scipy numpy-stl
```

## Basic usage

Edit the user settings near the top of the script:

```python
mode        = 6      # 1–6: lattice mode
n_points    = 25     # number of seed points
ratio       = 0.5    # shrink factor toward centroid
nz_layers   = 2      # for 2.5D modes only

ngon_thickness = 0.03
hub_scale      = 0.45

export_enabled = True
export_scad    = False
export_stl     = True
export_obj     = True

export_scad_path = "auxetic_lattice.scad"
export_stl_path  = "auxetic_lattice.stl"
export_obj_path  = "auxetic_lattice.obj"

strut_radius   = 0.02
face_thickness = 0.015
scad_segments  = 8

show_plot = False
```

Run:

```bash
python auxetic_lattice.py
```

When `export_enabled = True`, the script writes output files to the same directory as the script (unless absolute paths are given).

## Output

Depending on the export flags you will get one or more of:

- `auxetic_lattice.stl` – solid mesh with tube struts and thickened faces.
- `auxetic_lattice.obj` – OBJ mesh with vertex normals.
- `auxetic_lattice.scad` – OpenSCAD model using cylinders and extruded polygons.

Geometry consists of:

- **Struts**: tubular edges along shrunken simplex connections (`strut_radius`, `scad_segments`).
- **Faces/hubs**: polygons extruded along their Newell normal by `face_thickness`, plus truncated-octahedron hubs where connectivity is high.

## Mode overview

- **1, 2 (random 2D/2.5D)**: random points in the unit square, 2D Delaunay triangulation, then shrink-and-hub construction; mode 2 stacks layers in z and connects vertically.
- **3 (random 3D)**: random points in the unit cube, 3D Delaunay tetrahedra, then shrink-and-hub in 3D.
- **4, 5 (grid 2D/2.5D)**: structured 2D grid with center-aligned diagonals for symmetry, with optional vertical stacking and connections in mode 5.
- **6 (grid 3D)**: structured 3D grid; each cube is decomposed into 6 tetrahedra whose body diagonals point toward the global lattice center, improving hub symmetry.

## Tuning parameters

- `ratio`:
  - Smaller values keep hubs near original vertices, shortening struts.
  - Larger values move hubs toward simplex centroids, lengthening struts.
- `hub_scale`:
  - Controls truncated-octahedron hub size relative to mean strut reach; around `0.45` yields small gaps, near `0.5` almost touches neighbors.
- `strut_radius`, `face_thickness`:
  - Increase for more robust prints, decrease for lighter structures.

## File Structure

```text
auxetic/
├── README.md
├── python/da/                  — all Python source files
|   ├──displayAuxeticV01.py         — prototype
|   ├──displayAuxeticV02.py         — add mode 1, 2, 4, & 5
|   ├──displayAuxeticV03.py         — add bezier curve + overlap detection
|   ├──displayAuxeticV04.py         — add mode 6
|   ├──displayAuxeticV05.py         — add mode 3
|   ├──displayAuxeticV06.py         — prototype stl generation
|   ├──displayAuxeticV07.py         — prototype stl generation #2
|   ├──displayAuxeticV08.py         — fix n-gon shape
|   ├──displayAuxeticV09.py         — prototype stl generation #2
|   ├──displayAuxeticV10.py         — corrected mode 6 symmetry at small n
|   ├──displayAuxeticV11.py         — mode 6 fully corrected (mode 3  broken)
|   └──displayAuxeticV11.py         — refactored (mode 3  broken)
└── media/                     — reference images and videos of expected output
```
