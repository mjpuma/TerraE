#!/usr/bin/env python3
"""
TerraE sample run: 14-day simulation with realistic forcing.

Forcing: diurnal temperature/radiation cycle, precipitation events.
Supports single-column (bare soil) or multi-column (land cover) modes.
Output: time series saved to NPZ, used by visualization.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from terrae import types
from terrae.driver import advance_bare_soil, advance_cell, init_step
from terrae.soil.heat import retp


def diurnal_forcing(hour: float, day: int) -> dict:
    """Diurnal cycle: temperature, radiation (simplified)."""
    t_rad = 2 * np.pi * (hour / 24.0 - 0.25)  # Peak at noon
    ts = 288.0 + 8.0 * np.sin(t_rad)  # 280-296 K
    srht = max(0, 400 * np.sin(t_rad))  # W/m²
    trht = 300.0 + 50 * np.sin(t_rad)  # Longwave
    return {"ts": ts, "srht": srht, "trht": trht}


def precip_event(day: int, hour: float) -> float:
    """Rain events: 5 mm on days 2, 5, 8, 11."""
    if day in (2, 5, 8, 11) and 6 <= hour <= 8:
        return 5e-3 / 7200.0  # 5 mm over 2 h, m/s
    return 0.0


def run_simulation(
    n_days: int = 14,
    dt_hours: float = 1.0,
    output_dir: Path | None = None,
) -> tuple[np.ndarray, dict]:
    """
    Run TerraE bare soil for n_days. Returns (data arrays, metadata).
    """
    output_dir = output_dir or Path(__file__).parent.parent / "output"
    output_dir.mkdir(exist_ok=True)

    # Soil setup: 6 layers, pure sand
    q = np.zeros((types.IMT, types.NGM))
    q[0, :] = 1.0
    qk = np.ones((types.IMT, types.NGM))
    dz = np.array([0.05, 0.1, 0.2, 0.4, 0.8, 1.6])  # m, finer near surface

    n, zb, zc, thets, thetm, shc, ws = init_step(dz, q)

    # Initial state: moderately wet
    w = np.array([0.015, 0.03, 0.06, 0.1, 0.15, 0.25])
    ht = np.zeros(types.NGM + 1)
    for k in range(n):
        ht[k] = 1e5 * (1 + 0.1 * k)  # ~15-20°C

    dt_sec = dt_hours * 3600.0
    n_steps = int(n_days * 24 / dt_hours)

    # Storage
    time_h = np.zeros(n_steps + 1)
    theta_all = np.zeros((n_steps + 1, n))
    tp_all = np.zeros((n_steps + 1, n))
    w_tot = np.zeros(n_steps + 1)
    runoff_cum = np.zeros(n_steps + 1)
    pr_all = np.zeros(n_steps + 1)

    time_h[0] = 0
    theta_all[0] = w[:n] / dz[:n]
    tp0, _ = retp(w, ht, shc, n)
    tp_all[0] = tp0[:n]
    w_tot[0] = np.sum(w[:n])
    runoff_cum[0] = 0.0

    total_runoff = 0.0
    deep_perc_cum_arr = np.zeros(n_steps + 1)
    for step in range(n_steps):
        hour = (step * dt_hours) % 24
        day = int(step * dt_hours / 24)

        forc = diurnal_forcing(hour, day)
        pr = precip_event(day, hour)
        htpr = pr * 4185 * 288 if pr > 0 else 0.0  # Rain at 15°C

        w, ht, run, evap, _, deep_perc = advance_bare_soil(
            w,
            ht,
            dz,
            q,
            qk,
            sl=0.02,
            pr=pr,
            htpr=htpr,
            srht=forc["srht"],
            trht=forc["trht"],
            ts=forc["ts"],
            rho=1.2,
            ch=0.01,
            vs=4.0,
            dt=dt_sec,
        )

        total_runoff += run
        deep_perc_cum_arr[step + 1] = deep_perc_cum_arr[step] + deep_perc
        time_h[step + 1] = (step + 1) * dt_hours
        theta_all[step + 1] = w[:n] / dz[:n]
        tp, _ = retp(w, ht, shc, n)
        tp_all[step + 1] = tp[:n]
        w_tot[step + 1] = np.sum(w[:n])
        runoff_cum[step + 1] = total_runoff
        pr_all[step + 1] = pr

    np.savez(
        output_dir / "sample_run.npz",
        time_h=time_h,
        theta=theta_all,
        tp=tp_all,
        w_tot=w_tot,
        runoff_cum=runoff_cum,
        deep_perc_cum=deep_perc_cum_arr,
        pr=pr_all,
        zc=zc[:n],
        dz=dz[:n],
        thets=thets[:n],
        thetm=thetm[:n],
    )

    meta = {
        "n_days": n_days,
        "dt_hours": dt_hours,
        "n_layers": n,
        "dz": dz[:n],
        "zc": zc[:n],
    }
    return (time_h, theta_all, tp_all, w_tot, runoff_cum, pr_all, zc[:n], dz[:n]), meta


def run_simulation_multi_column(
    n_days: int = 14,
    dt_hours: float = 1.0,
    output_dir: Path | None = None,
    fractions: np.ndarray | None = None,
    irrig_rates: np.ndarray | None = None,
) -> tuple[tuple, dict]:
    """
    Run TerraE with multiple land cover columns.

    fractions: (n_cols) area fractions for tree, grass, urban, suburban, rainfed_ag, irrigated_ag.
               Must sum to 1. Default: mixed urban/suburban/ag.
    irrig_rates: (n_cols) irrigation flux m/s for irrigated_ag column only.
    """
    output_dir = output_dir or Path(__file__).parent.parent / "output"
    output_dir.mkdir(exist_ok=True)

    n_cols = types.N_COLS
    if fractions is None:
        fractions = np.array([0.15, 0.20, 0.10, 0.25, 0.15, 0.15])  # tree, grass, urban, suburban, rainfed, irrig
        fractions /= fractions.sum()
    if irrig_rates is None:
        irrig_rates = np.zeros(n_cols)
        irrig_rates[types.I_IRRIGATED_AG] = 1e-7  # 8.6 mm/day to irrigated ag

    # Soil setup: shared 6 layers, pure sand
    q = np.zeros((types.IMT, types.NGM))
    q[0, :] = 1.0
    qk = np.ones((types.IMT, types.NGM))
    dz = np.array([0.05, 0.1, 0.2, 0.4, 0.8, 1.6])

    n, zb, zc, thets, thetm, shc, ws = init_step(dz, q)

    # Per-column slope: urban/suburban higher runoff
    sl = np.full(n_cols, 0.02)
    sl[types.I_URBAN] = 0.05
    sl[types.I_SUBURBAN] = 0.03

    # Initial state: (n_layers, n_cols)
    w = np.tile(np.array([0.015, 0.03, 0.06, 0.1, 0.15, 0.25])[:, np.newaxis], (1, n_cols))
    ht = np.zeros((types.NGM + 1, n_cols))
    for j in range(n_cols):
        for k in range(n):
            ht[k, j] = 1e5 * (1 + 0.1 * k)

    dt_sec = dt_hours * 3600.0
    n_steps = int(n_days * 24 / dt_hours)

    time_h = np.zeros(n_steps + 1)
    theta_all = np.zeros((n_steps + 1, n, n_cols))
    tp_all = np.zeros((n_steps + 1, n, n_cols))
    w_tot = np.zeros(n_steps + 1)
    w_tot_cols = np.zeros((n_steps + 1, n_cols))
    runoff_cum = np.zeros(n_steps + 1)
    irrig_cum = np.zeros(n_steps + 1)
    pr_all = np.zeros(n_steps + 1)

    # Area-weighted irrigation rate (m/s) per step
    irrig_rate_cell = np.sum(fractions * irrig_rates)

    time_h[0] = 0
    theta_all[0] = w[:n, :] / dz[:n, np.newaxis]
    for j in range(n_cols):
        tp0, _ = retp(w[:, j], ht[:, j], shc, n)
        tp_all[0, :, j] = tp0[:n]
    w_tot[0] = np.sum(w[:n, :] * fractions)
    w_tot_cols[0] = np.sum(w[:n, :], axis=0)
    runoff_cum[0] = 0.0
    pr_all[0] = 0.0

    total_runoff = 0.0
    deep_perc_cum_arr = np.zeros(n_steps + 1)
    for step in range(n_steps):
        hour = (step * dt_hours) % 24
        day = int(step * dt_hours / 24)

        forc = diurnal_forcing(hour, day)
        pr = precip_event(day, hour)
        htpr = pr * 4185 * 288 if pr > 0 else 0.0

        w, ht, run, evap, deep_perc = advance_cell(
            w, ht, fractions, dz, q, qk, sl,
            pr=pr, htpr=htpr, srht=forc["srht"], trht=forc["trht"], ts=forc["ts"],
            rho=1.2, ch=0.01, vs=4.0, dt=dt_sec, irrig=irrig_rates,
        )

        total_runoff += run
        deep_perc_cum_arr[step + 1] = deep_perc_cum_arr[step] + deep_perc
        time_h[step + 1] = (step + 1) * dt_hours
        irrig_cum[step + 1] = irrig_cum[step] + irrig_rate_cell * dt_sec
        theta_all[step + 1] = w[:n, :] / dz[:n, np.newaxis]
        for j in range(n_cols):
            tp, _ = retp(w[:, j], ht[:, j], shc, n)
            tp_all[step + 1, :, j] = tp[:n]
        w_tot[step + 1] = np.sum(w[:n, :] * fractions)
        w_tot_cols[step + 1] = np.sum(w[:n, :], axis=0)
        runoff_cum[step + 1] = total_runoff
        pr_all[step + 1] = pr

    np.savez(
        output_dir / "sample_run_multi.npz",
        time_h=time_h,
        theta=theta_all,
        tp=tp_all,
        w_tot=w_tot,
        w_tot_cols=w_tot_cols,
        runoff_cum=runoff_cum,
        deep_perc_cum=deep_perc_cum_arr,
        irrig_cum=irrig_cum,
        pr=pr_all,
        fractions=fractions,
        zc=zc[:n],
        dz=dz[:n],
        thets=thets[:n],
        thetm=thetm[:n],
        land_cover_types=np.array(types.LAND_COVER_TYPES, dtype=object),
    )

    meta = {
        "n_days": n_days,
        "dt_hours": dt_hours,
        "n_layers": n,
        "n_cols": n_cols,
        "land_cover_types": types.LAND_COVER_TYPES,
        "fractions": fractions,
        "dz": dz[:n],
        "zc": zc[:n],
    }
    return (time_h, theta_all, tp_all, w_tot, w_tot_cols, runoff_cum, pr_all, zc[:n], dz[:n]), meta


if __name__ == "__main__":
    multi = "--multi" in sys.argv
    if multi:
        data, meta = run_simulation_multi_column(n_days=14, dt_hours=1.0)
        print(f"Run complete: {meta['n_days']} days, {len(data[0])} steps, {meta['n_cols']} columns")
        print(f"Output: output/sample_run_multi.npz")
        print(f"Columns: {meta['land_cover_types']}")
        print(f"Final total water (area-weighted): {data[3][-1]:.4f} m")
        print(f"Total runoff: {data[5][-1]:.6f} m")
    else:
        data, meta = run_simulation(n_days=14, dt_hours=1.0)
        print(f"Run complete: {meta['n_days']} days, {len(data[0])} steps")
        print(f"Output: output/sample_run.npz")
        print(f"Final total water: {data[3][-1]:.4f} m")
        print(f"Total runoff: {data[4][-1]:.6f} m")
