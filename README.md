# TerraE

**TerraE** (*terrae*, Latin: "of the earth") is a large-scale hydrological land model that pays homage to [ModelE](https://www.giss.nasa.gov/tools/modelE/), NASA GISS's climate model. This Python prototype targets AI co-scientist workflows and modern land-surface modeling.

## Overview

TerraE simulates water and heat transport in soil. This prototype implements a **single soil column** as a stepping stone toward full multi-grid-cell coupling. We start here to validate physics, numerics, and mass balance before scaling to distributed runs.

- **Phase 1**: Explicit Richards equation solver (port from GHY.f)
- **Phase 2**: Mass-conservative implicit Richards solver (Celia et al. 1990, Zeng & Decker 2009)
- **Stack**: Python, NumPy, Matplotlib (JAX optional for future autodiff/GPU)

## Installation

```bash
pip install -e .
```

## Quick Start

```bash
# Run 14-day simulation
PYTHONPATH=src python scripts/sample_run.py

# Compare explicit vs implicit solvers
PYTHONPATH=src python scripts/compare_explicit_implicit.py

# Visualize output
PYTHONPATH=src python scripts/visualize_run.py
```

## Project Structure

```
├── src/
│   ├── terrae/                    # Core Python model
│   │   ├── constants.py           # Physical constants (ModelE compatible)
│   │   ├── types.py               # Data structures (NGM, IMT, etc.)
│   │   ├── driver.py              # Time stepping, advance_bare_soil
│   │   └── soil/
│   │       ├── hydraulic.py       # van Genuchten–Mualem (ROSETTA params)
│   │       ├── hydrology.py       # reth, hydra, fl, runoff, ImplicitRichards
│   │       ├── heat.py            # Thermal conductivity, heat flux
│   │       └── properties.py      # get_soil_properties (thets, thetm, shc)
│   └── terrae_biome_interface.f90 # BiomeE coupling interface (Section 9)
├── scripts/                       # Sample runs and visualization
├── doc/                           # Technical documentation
├── tests/
└── utilities/                     # Soil layer generators, ModelE input tools
```

## Soil Hydraulics

Uses **van Genuchten–Mualem** (standard in CLM, JULES, HYDRUS) with **ROSETTA** pedotransfer parameters. Replaces legacy GHY.f cubic θ(h) formulation.

**Solvers:**
- **Explicit**: Adaptive sub-stepping, unit-gradient bottom BC
- **Implicit**: Celia (1990) modified Picard, mass-conservative, same bottom BC

Both solvers use identical boundary conditions for direct comparison.

## BiomeE Coupling

TerraE provides a **BiomeE interface** (`src/terrae_biome_interface.f90`) for coupling with the BiomeE ecosystem model. The design minimizes changes to BiomeE: add `USE terrae_biome_interface` and replace BiomeE's bucket soil update with `biome_set_soil_from_terrae()`.

**Interface routines (called by TerraE Python via f2py/cffi):**
- `biome_set_forcings` — Push atmospheric + soil state to BiomeE
- `biome_get_exports` — Pull canopy properties + transpiration from BiomeE
- `biome_run` — Advance BiomeE one GCM timestep

**Compilation:** `gfortran -c terrae_biome_interface.f90`; link with BiomeE objects and TerraE Python extension. See [TerraE Technical Description v2, Section 9](https://docs.google.com/document/d/1SbJ_u2jiyf-StvHdLiyalWJMrme8LZCUNlxqdWnR1eI/edit?tab=t.5ql9wgu3mmls) for full interface specification and Section 9.7 for the planned psi-based water-stress improvement.

## Technical Documentation

- **TerraE Technical Description v2**: [Google Doc](https://docs.google.com/document/d/1SbJ_u2jiyf-StvHdLiyalWJMrme8LZCUNlxqdWnR1eI/edit?tab=t.5ql9wgu3mmls) — see especially **Section 9** (BiomeE interface)
- **LaTeX for Overleaf**: `doc/terrae_technical_description.tex` — import into [Overleaf](https://www.overleaf.com) for editing and PDF export

## Development

```bash
pip install -e ".[dev]"
pytest
```

## Next Development Steps

1. **Testing with observational data**  
   Validate soil moisture and temperature against in-situ (e.g., FLUXNET, SMAP) and satellite products. Calibrate parameters and assess bias.

2. **Subgrid fractions and processes**  
   Add details on different land-cover fractions (bare soil, vegetated) and their processes: interception, transpiration, vegetated soil evaporation, snow. Implement tile-based water and energy balances.

3. **Coupling with BiomeE** *(interface initiated)*  
   The `terrae_biome_interface.f90` module provides the coupling layer. Remaining work: f2py/cffi bindings, psi-based water stress (Section 9.7), and full integration testing.

4. **Coupling with AI co-scientist**  
   Expose model for AI-driven experimentation: parameter sweeps, surrogate modeling, inverse problems, and automated hypothesis testing.

5. **Multi-grid-cell prototype**  
   Extend from single column to multiple grid cells (e.g., regional domain). Add lateral routing, subgrid heterogeneity, and parallel execution. The current single-column setup serves as the validated prototype for this step.

## References

- Rosenzweig, C. & Abramopoulos, F. (1997). Land-surface model development for the GISS GCM. *J. Climate*, 10:2040–2054.
- Celia, M.A., et al. (1990). A general mass-conservative numerical solution for the unsaturated flow equation. *Water Resour. Res.*, 26:1483–1496.
- Zeng, X. & Decker, M. (2009). Improving the numerical solution of soil moisture-based Richards equation. *J. Hydrometeor.*, 10:308–319.
- NASA GISS ModelE: https://www.giss.nasa.gov/tools/modelE/

## License

MIT License — see [LICENSE](LICENSE).
