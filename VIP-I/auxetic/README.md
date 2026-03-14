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
| `mode` | `1` | Visualization mode (1, 2, 4, or 5). See Modes below. |
| `n_points` | `9` | Number of lattice points. In grid modes, rounded to the nearest factorable grid size. |
| `ratio` | `0.4` | Triangle shrink factor toward centroid. `0.0` = full size, `1.0` = collapsed to a point. Controls strut width. Values between `0.2`–`0.5` work best. |
| `nx, ny, nz` | `1, 1, 1` | Background 3D node grid dimensions. Typically leave at default. |
| `cell` | `1.0` | Physical size of the unit cell. Scales the background grid. |
| `nz_layers` | `3` | Number of Z layers in 2.5D modes. Minimum of 2. Values above 5 may slow rendering. |

---

## Modes

Change the mode by editing the `mode` variable at the top of `displayAuxeticV#.py`:
```python
mode = 1   # change this value
```

| Mode | Name | Description |
|------|------|-------------|
| `1` | Random 2D | Randomly placed points triangulated into a flat lattice at z = 0. Produces organic, irregular patterns. Re-run for a new random layout. |
| `2` | Random 2.5D | Same as Mode 1 but extruded along the Z-axis across `nz_layers` slices, creating a volumetric structure with randomly placed struts. |
| `4` | Grid 2D | Points arranged in a regular rectangular grid, then triangulated. Produces a uniform, symmetric flat lattice. |
| `5` | Grid 2.5D | Same as Mode 4 but extruded along Z. Produces a clean, periodic 3D lattice. |

> **Note for grid modes (4, 5):** `n_points` should be a composite number (e.g. 4, 6, 9, 12) so it factors cleanly into a rectangular grid. Prime numbers fall back to a 1×N single row.

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
|   ├──displayAuxeticV1.py         — first prototype
|   ├──displayAuxeticV2.py         — add mode 1, 2, 4, & 5
|   └──displayAuxeticV3.py         — add bezier curve + overlap detection
└── media/                     — reference images and videos of expected output
```
