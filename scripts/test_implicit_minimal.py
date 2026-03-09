#!/usr/bin/env python3
"""
Minimal unit test for ImplicitRichards: steady drainage only.
No precip, no evap. Should drain and satisfy mass balance.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from terrae import types
from terrae.driver import init_step
from terrae.soil.hydraulic import h_from_theta
from terrae.soil.hydrology import ImplicitRichards, reth


def main() -> None:
    q = np.zeros((types.IMT, types.NGM))
    q[0, :] = 1.0
    dz = np.array([0.05, 0.1, 0.2, 0.4, 0.8, 1.6])
    n, zb, zc, thets, thetm, shc, ws = init_step(dz, q)
    solver = ImplicitRichards(dz, zc, q, n)

    # Initial: moderately wet
    theta_old = np.array([0.25, 0.22, 0.20, 0.18, 0.16, 0.15])
    w_old = theta_old * solver.dz
    psi0 = np.array([float(h_from_theta(theta_old[k], solver._vg[k])) for k in range(n)])
    S = np.zeros(n)
    flux_top = 0.0  # No precip
    dt = 3600.0  # 1 hour
    z_wt = None  # Free drainage

    print("=== Steady drainage test ===")
    print("theta_old:", theta_old)
    print("flux_top:", flux_top, "dt:", dt)
    print()

    # Single solve
    psi_new, theta_new, diag = solver.solve(
        psi0, theta_old, dt, flux_top, S, z_wt=z_wt,
        tol_psi=1e-4, tol_theta=1e-6, max_iter=50,
    )

    print("Converged:", diag["converged"])
    print("Picard iters:", diag["picard_iterations"])
    print("theta_new:", theta_new)
    print("delta_theta:", theta_new - theta_old)
    print("delta_S (m):", np.sum(solver.dz * (theta_new - theta_old)))
    print("deep_perc (m):", diag["deep_percolation"])

    # Mass balance: inflow - outflow - sink = delta_S
    inflow = flux_top * dt
    outflow = diag["deep_percolation"]
    sink = np.sum(S * solver.dz) * dt
    delta_S = np.sum(solver.dz * (theta_new - theta_old))
    residual = inflow - outflow - sink - delta_S
    print()
    print("Mass balance:")
    print("  inflow:", inflow)
    print("  outflow:", outflow)
    print("  sink:", sink)
    print("  delta_S:", delta_S)
    print("  residual:", residual)
    print("  |residual| < 1e-4:", abs(residual) < 1e-4)

    try:
        r = solver.mass_balance_check(
            theta_old, theta_new, dt, flux_top,
            diag["deep_percolation"], S,
        )
        print("  mass_balance_check OK, residual:", r)
    except AssertionError as e:
        print("  mass_balance_check FAILED:", e)


if __name__ == "__main__":
    main()
