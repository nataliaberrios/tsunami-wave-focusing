# Tsunami propagation and wave focusing at Mavericks, CA

2D shallow-water tsunami simulation that attempts to reproduce theoretical tsunami propagation, and show how bathymetric variations at
Mavericks, California focus incoming wave energy onto that stretch of
coastline compared to adjacent areas.

## Scientific question
How well do finite-difference shallow water models reproduce theoretical tsunami propagation?

To explore this, the jupyter notebook creates figures comparing the simulation outputs to theoretical expressions for the wave speed (c = sqrt(gH)) and Green's law (which predicts how wave amplitude should increase as water depth decreases).

Does the continental shelf bathymetry along the northern California coast
preferentially focus waves onto the Mavericks location?

Running the Mavericks simulation in the Jupyter notebook demonstrates that the answer to this question: yes, peak coastal amplitude occurs within 5 km of Mavericks,
probably due to waves traveling over an offshore ridge as they approach the coast.

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
Wave speed depends on depth through c = sqrt(gH). As the
wavefront crosses the continental shelf, the portion over a shallower
bathymetric ridge slows relative to adjacent areas, bending the wavefront
and concentrating energy at that location.

## Performance

The MacCormack stepper is JIT-compiled with Numba (parallel=True),
reducing runtime for simple case simulations with large grids and for the Mavericks
simulation (700x275 grid, 4000 second simulation). Note that sometimes the wave height in the Mavericks simulation appear to blowup, but this seems to only happen if the kernel was not restarted before running the notebook from top to bottom.

## Data

Bathymetry from the GEBCO_2026 Grid, covering the northern California
continental shelf (36.5–38.5°N, 124.5–121.5°W).

GEBCO Compilation Group (2026) GEBCO 2026 Grid
doi: 10.5285/4f68d5c7-45eb-f999-e063-7086abc036fa

The GEBCO Grid is in the public domain and free to use with attribution.
