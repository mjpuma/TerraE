# TerraE Scripts

## Sample run and visualization

```bash
# From terrae_py/ directory
PYTHONPATH=src python scripts/sample_run.py      # Run 14-day simulation
PYTHONPATH=src python scripts/visualize_run.py   # Generate figures (runs sample if needed)
```

**Output:**
- `output/sample_run.npz` – time series (theta, tp, w_tot, runoff, pr)
- `output/figures/terrae_sample_run.png` – 6-panel figure
- `output/figures/water_balance.png` – water balance check

**Figure panels:**
1. Soil moisture by layer (θ time series)
2. Soil temperature by layer
3. Precipitation and cumulative runoff
4. Column-integrated water
5. Hovmöller (θ vs depth and time)
6. Moisture profiles at selected times
