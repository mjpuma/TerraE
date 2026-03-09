# TerraE

**TerraE** (*terrae*, Latin: "of the earth") is a large-scale hydrological land model that pays homage to [ModelE](https://github.com/nasa-giss/modelE), NASA GISS's climate model. This Python prototype targets AI co-scientist workflows and modern land-surface modeling.

## Overview

TerraE simulates water and heat transport in soil. This prototype implements a **single soil column** as a stepping stone toward full multi-grid-cell coupling. We start here to validate physics, numerics, and mass balance before scaling to distributed runs.

- **Phase 1**: Explicit Richards equation solver (port from GHY.f)
- **Phase 2**: Mass-conservative implicit Richards solver (Celia et al. 1990, Zeng & Decker 2009)
- **Stack**: Python, NumPy, Matplotlib (JAX optional for future autodiff/GPU)

## Installation

```bash
cd terrae_py
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
terrae_py/
├── src/terrae/
│   ├── constants.py      # Physical constants (ModelE compatible)
│   ├── types.py          # Data structures (NGM, IMT, etc.)
│   ├── driver.py         # Time stepping, advance_bare_soil
│   └── soil/
│       ├── hydraulic.py  # van Genuchten–Mualem (ROSETTA params)
│       ├── hydrology.py  # reth, hydra, fl, runoff, ImplicitRichards
│       ├── heat.py       # Thermal conductivity, heat flux
│       └── properties.py # get_soil_properties (thets, thetm, shc)
├── scripts/
│   ├── sample_run.py
│   ├── compare_explicit_implicit.py
│   └── visualize_run.py
├── doc/
│   └── terrae_technical_description.tex  # Overleaf LaTeX
└── tests/
```

## Soil Hydraulics

Uses **van Genuchten–Mualem** (standard in CLM, JULES, HYDRUS) with **ROSETTA** pedotransfer parameters. Replaces legacy GHY.f cubic θ(h) formulation.

**Solvers:**
- **Explicit**: Adaptive sub-stepping, unit-gradient bottom BC
- **Implicit**: Celia (1990) modified Picard, mass-conservative, same bottom BC

Both solvers use identical boundary conditions for direct comparison.

## Technical Documentation

- **TerraE Technical Description v2**: [Google Doc](https://docs.google.com/document/d/1SbJ_u2jiyf-StvHdLiyalWJMrme8LZCUNlxqdWnR1eI/edit?usp=sharing)
- **LaTeX for Overleaf**: `terrae_py/doc/terrae_technical_description.tex` — import into [Overleaf](https://www.overleaf.com) for editing and PDF export

## Development

```bash
cd terrae_py
pip install -e ".[dev]"
pytest
```

## Next Development Steps

1. **Testing with observational data**  
   Validate soil moisture and temperature against in-situ (e.g., FLUXNET, SMAP) and satellite products. Calibrate parameters and assess bias.

2. **Subgrid fractions and processes**  
   Add details on different land-cover fractions (bare soil, vegetated) and their processes: interception, transpiration, vegetated soil evaporation, snow. Implement tile-based water and energy balances.

3. **Coupling with BiomeE**  
   Integrate with the BiomeE ecosystem model for vegetation dynamics, phenology, and carbon–water feedbacks. Replace/supplement Ent TBM interface.

4. **Coupling with AI co-scientist**  
   Expose model for AI-driven experimentation: parameter sweeps, surrogate modeling, inverse problems, and automated hypothesis testing.

5. **Multi-grid-cell prototype**  
   Extend from single column to multiple grid cells (e.g., regional domain). Add lateral routing, subgrid heterogeneity, and parallel execution. The current single-column setup serves as the validated prototype for this step.

## References

- Rosenzweig, C. & Abramopoulos, F. (1997). Land-surface model development for the GISS GCM. *J. Climate*, 10:2040–2054.
- Celia, M.A., et al. (1990). A general mass-conservative numerical solution for the unsaturated flow equation. *Water Resour. Res.*, 26:1483–1496.
- Zeng, X. & Decker, M. (2009). Improving the numerical solution of soil moisture-based Richards equation. *J. Hydrometeor.*, 10:308–319.
- NASA GISS ModelE: https://github.com/nasa-giss/modelE

## License

MIT License — see [LICENSE](LICENSE).
