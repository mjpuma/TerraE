#!/usr/bin/env python3
"""
Debug implicit solver mass balance: find worst steps and trace causes.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from terrae import types
from terrae.driver import init_step
from terrae.soil.hydraulic import h_from_theta, weighted_params
from terrae.soil.hydrology import ImplicitRichards, reth


def precip_event(day: int, hour: float) -> float:
    if day in (2, 5, 8, 11) and 6 <= hour <= 8:
        return 5e-3 / 7200.0
    return 0.0


def main() -> None:
    q = np.zeros((types.IMT, types.NGM))
    q[0, :] = 1.0
    qk = np.ones((types.IMT, types.NGM))
    dz = np.array([0.05, 0.1, 0.2, 0.4, 0.8, 1.6])
    n, zb, zc, thets, thetm, shc, ws = init_step(dz, q)
    solver = ImplicitRichards(dz, zc, q, n)

    w0 = np.array([0.015, 0.03, 0.06, 0.1, 0.15, 0.25])
    dt_hours = 1.0
    dt_sec = dt_hours * 3600.0
    n_steps = 14 * 24  # 14 days

    w = w0.copy()
    Ks_surf = weighted_params(q, 0).Ks

    worst_steps = []
    for step in range(n_steps):
        hour = (step * dt_hours) % 24
        day = int(step * dt_hours / 24)
        pr = precip_event(day, hour)
        flux_top = min(pr, Ks_surf)
        theta_old = reth(w, solver.dz, n)
        psi0 = np.array([float(h_from_theta(theta_old[k], solver._vg[k])) for k in range(n)])
        S = np.zeros(n)

        psi_new, theta_new, diag = solver.solve(
            psi0, theta_old, dt_sec, flux_top, S, z_wt=None,
            tol_psi=1e-4, tol_theta=1e-6, max_iter=20,
        )

        delta_S = np.sum(solver.dz * (theta_new - theta_old))
        inflow = flux_top * dt_sec
        outflow = diag["deep_percolation"]
        residual = inflow - outflow - delta_S

        # Saturation clipping
        w_new = theta_new * solver.dz
        rnf_sat = 0.0
        for k in range(n):
            if w_new[k] > ws[k]:
                excess = w_new[k] - ws[k]
                rnf_sat += excess / dt_sec

        worst_steps.append({
            "step": step,
            "day": day,
            "hour": hour,
            "pr": pr,
            "flux_top": flux_top,
            "inflow": inflow,
            "outflow": outflow,
            "delta_S": delta_S,
            "residual": residual,
            "rnf_sat": rnf_sat,
            "theta_old": theta_old.copy(),
            "theta_new": theta_new.copy(),
            "converged": diag["converged"],
            "picard": diag["picard_iterations"],
        })
        w[:n] = np.clip(w_new, solver.dz * thetm, ws)

    # Sort by |residual|
    worst_steps.sort(key=lambda x: abs(x["residual"]), reverse=True)

    print("=== Top 5 worst mass balance steps ===\n")
    for i, s in enumerate(worst_steps[:5]):
        print(f"--- Step {s['step']} (day {s['day']:.0f}, hour {s['hour']:.0f}) ---")
        print(f"  pr={s['pr']*1e6:.2f} mm/h  flux_top={s['flux_top']*1e6:.2f} mm/h")
        print(f"  inflow={s['inflow']*1000:.4f} mm  outflow={s['outflow']*1000:.4f} mm")
        print(f"  delta_S={s['delta_S']*1000:.4f} mm  residual={s['residual']*1000:.4f} mm")
        print(f"  rnf_sat={s['rnf_sat']*1e6:.2f} mm/h  converged={s['converged']}  picard={s['picard']}")
        print(f"  theta_old: {s['theta_old']}")
        print(f"  theta_new: {s['theta_new']}")
        print(f"  theta_new > thets? {np.any(s['theta_new'] > thets[0])}")
        print()

    # Check: does residual correlate with precip?
    pr_steps = [s for s in worst_steps if s["pr"] > 0]
    dry_steps = [s for s in worst_steps if s["pr"] == 0]
    print("=== Precip vs dry steps ===")
    print(f"Precip steps: mean |residual| = {np.mean([abs(s['residual']) for s in pr_steps])*1000:.4f} mm")
    print(f"Dry steps:    mean |residual| = {np.mean([abs(s['residual']) for s in dry_steps])*1000:.4f} mm")


if __name__ == "__main__":
    main()
