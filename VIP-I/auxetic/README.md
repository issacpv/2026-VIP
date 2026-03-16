# Lattice Visualization Tool

A Python tool for generating and visualizing 2D and 2.5D Delaunay-triangulated lattice structures in an interactive 3D matplotlib window.

---

## Requirements
```bash
pip install numpy matplotlib scipy
```

---

## Configuration

All settings are controlled by the **USER SETTINGS** block at the top of `displayAuxeticV#.py`. No command-line arguments are needed.

### Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `mode` | `1` | Visualization mode (1–6). See Modes below. |
| `n_points` | `9` | Number of lattice points. In grid modes, rounded to the nearest factorable grid size. |
| `ratio` | `0.4` | Simplex shrink factor toward centroid. `0.0` = full size, `1.0` = collapsed to a point. Controls strut width. Values between `0.2`–`0.5` work best. |
| `nx, ny, nz` | `1, 1, 1` | Background 3D node grid dimensions. Typically leave at default. |
| `cell` | `1.0` | Physical size of the unit cell. Scales the background grid. |
| `nz_layers` | `3` | Number of Z layers in 2.5D modes (2 and 5) only. Minimum of 2. Values above 5 may slow rendering. |
| `use_bezier` | `False` | Enables Bezier curved struts and edges. When `False`, straight lines and flat faces are used and the intersection check is skipped entirely. |
| `bend_reentrant` | `0.18` | Bezier bow amount for 2-point struts connecting shrunk simplex corners. Higher values deepen the reentrant angle and strengthen the auxetic effect. Only applies when `use_bezier = True`. |
| `bend_ngon` | `0.14` | Bezier bow amount for n-gon/polyhedron perimeter edges at shared-vertex hubs. Controls curvature of the flexible hinge regions. Only applies when `use_bezier = True`. |
| `bend_triangle` | `0.12` | Bezier bow amount for the edges of the magenta solid simplex face regions. Only applies when `use_bezier = True`. |
| `bend_vertical` | `0.08` | Bezier bow amount for vertical struts in 2.5D modes only. Only applies when `use_bezier = True`. |
| `intersect_threshold` | `0.05` | Sensitivity of the intersection check, as a fraction of mean strut length. Raise to catch more near-misses, lower to suppress minor ones. Ignored when `use_bezier = False`. |
| `intersect_check` | `True` | Set to `False` to skip intersection checking entirely. Ignored when `use_bezier = False`. |
| `export_enabled` | `False` | Set to `True` to write `.scad` and `.stl` files on run. Export runs before the plot window opens. |
| `export_scad_path` | `"auxetic_lattice.scad"` | Output path for the OpenSCAD file. Relative paths are resolved to the script's directory. |
| `export_stl_path` | `"auxetic_lattice.stl"` | Output path for the STL file. Relative paths are resolved to the script's directory. |
| `export_png_path` | `None` | Set to a filename to render a PNG preview via OpenSCAD. Requires OpenSCAD to be installed. Set to `None` to skip. |
| `strut_radius` | `0.02` | Physical tube radius for struts in exported geometry, in lattice units. |
| `face_thickness` | `0.015` | Extrusion thickness for solid faces in exported geometry, in lattice units. |
| `scad_segments` | `8` | Number of polygon sides for strut cross-sections. Higher values produce rounder tubes but increase file size and render time. |
---

## Modes

Change the mode by editing the `mode` variable at the top of `lattice_viz.py`:
```python
mode = 1   # change this value
```

| Mode | Name | Description |
|------|------|-------------|
| `1` | Random 2D | Randomly placed points triangulated into a flat lattice at z = 0. Produces organic, irregular patterns. Re-run for a new random layout. |
| `2` | Random 2.5D | Same as Mode 1 but extruded along the Z-axis across `nz_layers` slices, creating a volumetric structure with randomly placed struts. |
| `3` | Random 3D | Points with fully random x, y, and z coordinates, tetrahedralized via 3D Delaunay triangulation. Produces a true volumetric auxetic structure with irregular geometry. |
| `4` | Grid 2D | Points arranged in a regular rectangular grid, then triangulated. Produces a uniform, symmetric flat lattice. |
| `5` | Grid 2.5D | Same as Mode 4 but extruded along Z. Produces a clean, periodic 3D lattice. |
| `6` | Grid 3D | Points arranged in a regular cubic grid, tetrahedralized in 3D. Produces a clean, periodic volumetric auxetic structure. |

> **Note for grid modes (4, 5, 6):** `n_points` should be a composite number so it factors cleanly into a rectangular or cubic grid. Prime numbers fall back to a degenerate layout.

---

## Export

Set `export_enabled = True` to generate files on run. Both files are written to the same folder as `lattice_viz.py` by default, or to the absolute path you specify.

**OpenSCAD is not required.** The STL is written directly from Python using `numpy-stl`. The `.scad` file is also written and can be opened in OpenSCAD for further editing, boolean operations, or rendering a cleaner manifold solid if needed.

Export runs before the plot window opens, so files are available even if you close the plot immediately.

---

## Interactive Controls

The plot window must be focused for key bindings to work.

### Viewport Rotation

| Key | Action |
|-----|--------|
| `↑` Arrow Up | Rotate view upward (increase elevation) |
| `↓` Arrow Down | Rotate view downward (decrease elevation) |
| `←` Arrow Left | Rotate view left (decrease azimuth) |
| `→` Arrow Right | Rotate view right (increase azimuth) |
| `X` | Snap to X-axis view (elev = 0°, azim = 0°) |
| `Y` | Snap to Y-axis view (elev = 0°, azim = 90°) |
| `Z` | Snap to top-down view (elev = 90°, azim = −90°) |

### Closing the Window

- Click the **X** button on the window title bar, or
- Press **Q** while the plot window is focused (default matplotlib shortcut), or
- Press **Ctrl+W** on most systems

---

## File Structure
```
auxetic/
├── README.md
├── python/                    — all Python source files
|   ├──displayAuxeticV1.py         — prototype
|   ├──displayAuxeticV2.py         — add mode 1, 2, 4, & 5
|   ├──displayAuxeticV3.py         — add bezier curve + overlap detection
|   ├──displayAuxeticV4.py         — add mode 6
|   ├──displayAuxeticV5.py         — add mode 3
|   ├──displayAuxeticV6.py         — prototype stl generation
|   ├──displayAuxeticV7.py         — prototype stl generation #2
|   └──displayAuxeticV8.py         — fix n-gon shape
└── media/                     — reference images and videos of expected output
```
