# Tsunami Wave Focusing at Mavericks

2D shallow-water tsunami simulation showing how the underwater ridge at
Mavericks, California focuses incoming wave energy onto that stretch of
coastline compared to adjacent areas.

## Scientific question

Does the continental shelf bathymetry along the northern California coast
preferentially focus long wavelength wave energy onto the Mavericks location?

Running the Mavericks simulation should demonstrate that the answer to this question: yes, peak coastal amplitude occurs within 5 km of Mavericks,
driven by refraction over the offshore ridge.

## Files

**`tsunami_utils.py`** — Shared numerical kernels (MacCormack solver, sponge functions)

**`tsunami_demo.ipynb`** — Full walkthrough: Green's law validation for simple case, validation of sqrt(gH) 
showing wave speed slowing as wave approaches coast in simple case, convergence test in simple case, Mavericks wave focusing when using real continental shelf bathymetry

**`mavericks_bathy.nc`** — GEBCO 2026 bathymetry, northern California shelf (36.5-38.5°N, 124.5-121.5°W)

## Installation

```bash
git clone https://github.com/nataliaberrios/tsunami-wave-focusing
cd tsunami-wave-focusing
pip install -r requirements.txt
```

## Usage

```bash
jupyter notebook tsunami_demo.ipynb
```

Run cells top to bottom. Switches at the top of each section control
the bathymetry type, source shape, and coast orientation.

The notebook includes an optional grid convergence study (5 resolutions,
N=300 to N=1500) that takes ~20 minutes. Set `RUN_CONVERGENCE = False`
at the top of that cell to skip it during normal runs.

## Physics

The simulation solves the 2D linear shallow-water equations using the
MacCormack predictor-corrector scheme (2nd order in space and time).
The depth-dependent wave speed c = sqrt(gH) causes refraction: as the
wavefront crosses the continental shelf, the portion over the shallower
Mavericks ridge slows relative to adjacent areas, bending the wavefront
and concentrating energy at that location.

## Performance

The MacCormack stepper is JIT-compiled with Numba (parallel=True),
reducing runtime for simple case simulations with large grids and for the Mavericks
simulation (700x275 grid, 4000 sec simulation). Note that sometimes the wave height in the Mavericks simulation appear to blowup, but this only happens if the kernel was not restarted before running the notebook from top to bottom.

## Data

Bathymetry from the GEBCO_2026 Grid, covering the northern California
continental shelf (36.5–38.5°N, 124.5–121.5°W).

GEBCO Compilation Group (2026) GEBCO 2026 Grid
doi: 10.5285/4f68d5c7-45eb-f999-e063-7086abc036fa

The GEBCO Grid is in the public domain and free to use with attribution.
