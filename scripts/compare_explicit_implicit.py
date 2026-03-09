#!/usr/bin/env python3
"""
Compare explicit (fl + fllmt) vs implicit (ImplicitRichards) Richards solvers.

Runs both with identical forcing and plots multipanel diagnostics:
  - Theta time series by layer
  - Hovmöller (theta vs depth-time)
  - Runoff and precipitation
  - Deep percolation
  - Mass balance residual
  - Picard iterations (implicit only)
  - Difference plots (explicit - implicit)
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from terrae import constants, types
from terrae.driver import advance_bare_soil, init_step
from terrae.soil.heat import flh, retp, xklh
from terrae.soil.hydraulic import h_from_theta, weighted_params
from terrae.soil.hydrology import (
    ImplicitRichards,
    hydra,
    reth,
    runoff,
)


def precip_event(day: int, hour: float) -> float:
    """Rain events: 5 mm on days 2, 5, 8, 11."""
    if day in (2, 5, 8, 11) and 6 <= hour <= 8:
        return 5e-3 / 7200.0  # 5 mm over 2 h, m/s
    return 0.0


def advance_bare_soil_implicit(
    w: np.ndarray,
    ht: np.ndarray,
    dz: np.ndarray,
    q: np.ndarray,
    qk: np.ndarray,
    sl: float,
    pr: float,
    htpr: float,
    srht: float,
    trht: float,
    ts: float,
    rho: float,
    ch: float,
    vs: float,
    dt: float,
    solver: ImplicitRichards,
    z_wt: float | None = None,
) -> tuple[np.ndarray, np.ndarray, float, float, float, dict]:
    """
    Single-step advance using ImplicitRichards.
    Returns w, ht, total_runoff, total_evap, total_heat_runoff, diagnostics.
    """
    n = solver.n
    thets = np.array([solver._vg[k].theta_s for k in range(n)])
    thetm = np.array([solver._vg[k].theta_r for k in range(n)])
    ws = thets * solver.dz
    evapdl = np.zeros(n)
    zb = np.zeros(n + 1)
    zb[0] = 0
    for k in range(n):
        zb[k + 1] = zb[k] - solver.dz[k]

    from terrae.soil.properties import get_soil_properties
    _, _, shc = get_soil_properties(q, solver.dz)
    shc = np.array(shc)[:n]
    theta_old = reth(w, solver.dz, n)
    tp, fice = retp(w, ht, shc, n)

    # Surface BC: limit infiltration by Ks
    Ks_surf = weighted_params(q, 0).Ks
    flux_top = min(pr, Ks_surf)
    hortonian_runoff = max(0.0, pr - flux_top)

    S = evapdl
    psi0 = np.array([float(h_from_theta(theta_old[k], solver._vg[k])) for k in range(n)])

    psi_new, theta_new, diag = solver.solve(
        psi0, theta_old, dt, flux_top, S, z_wt=z_wt
    )

    diag["mass_residual"] = solver.mass_balance_check(
        theta_old, theta_new, dt, flux_top, diag["deep_percolation"], S
    )

    w_new = theta_new * solver.dz
    rnf_sat = 0.0
    for k in range(n):
        if w_new[k] > ws[k]:
            excess = w_new[k] - ws[k]
            w_new[k] = ws[k]
            rnf_sat += excess / dt
        w_new[k] = np.clip(w_new[k], solver.dz[k] * thetm[k], ws[k])

    h, xk, d, xku = hydra(theta_new, thets, thetm, fice[:n], solver.zc, q, qk, n)
    xinfc = xk[1] * abs(h[0]) / (abs(solver.zc[0]) + 1e-30) if n >= 1 else 0.0
    f = np.zeros(n + 1)
    f[0] = -flux_top
    for k in range(1, n):
        dz_int = abs(solver.zc[k - 1] - solver.zc[k])
        if dz_int > 1e-30:
            f[k] = -xk[k + 1] * (h[k - 1] - h[k]) / dz_int
    f[n] = 0.0

    rnf_sat_step, rnff = runoff(w_new, ws, f, xinfc, xku, solver.dz, sl, n)
    rnf_sat += rnf_sat_step
    total_runoff = (hortonian_runoff + rnf_sat) * dt

    w[:n] = w_new
    total_evap = 0.0
    xkh, xkhm = xklh(theta_new, thets, w_new, fice[:n], solver.dz, zb, solver.zc, q, n)
    fh = flh(xkhm, tp, f, solver.zc, n)
    fh[0] = htpr - srht - trht

    for k in range(n):
        ht[k] += (fh[k + 1] - fh[k] - constants.SHW_VOL * max(tp[k], 0) * rnff[k]) * dt

    heat_runoff = constants.SHW_VOL * (total_runoff / dt * max(tp[0], 0)) * dt if dt > 0 else 0.0
    return w, ht, total_runoff, total_evap, heat_runoff, diag


def run_comparison(
    n_days: int = 14,
    dt_hours: float = 1.0,
) -> tuple[dict, dict]:
    """Run both explicit and implicit; return (explicit_data, implicit_data)."""
    q = np.zeros((types.IMT, types.NGM))
    q[0, :] = 1.0
    qk = np.ones((types.IMT, types.NGM))
    dz = np.array([0.05, 0.1, 0.2, 0.4, 0.8, 1.6])

    n, zb, zc, thets, thetm, shc, ws = init_step(dz, q)
    solver = ImplicitRichards(dz, zc, q, n)

    w0 = np.array([0.015, 0.03, 0.06, 0.1, 0.15, 0.25])
    ht0 = np.zeros(types.NGM + 1)
    for k in range(n):
        ht0[k] = 1e5 * (1 + 0.1 * k)

    dt_sec = dt_hours * 3600.0
    n_steps = int(n_days * 24 / dt_hours)

    def diurnal(hour: float) -> dict:
        t = 2 * np.pi * (hour / 24.0 - 0.25)
        return {
            "ts": 288.0 + 8.0 * np.sin(t),
            "srht": max(0, 400 * np.sin(t)),
            "trht": 300.0 + 50 * np.sin(t),
        }

    # --- Explicit run ---
    w_ex = w0.copy()
    ht_ex = ht0.copy()
    theta_ex = np.zeros((n_steps + 1, n))
    runoff_ex = np.zeros(n_steps + 1)
    deep_perc_ex = np.zeros(n_steps + 1)
    pr_all = np.zeros(n_steps + 1)
    theta_ex[0] = w0[:n] / dz[:n]
    runoff_ex[0] = 0.0

    for step in range(n_steps):
        hour = (step * dt_hours) % 24
        day = int(step * dt_hours / 24)
        forc = diurnal(hour)
        pr = precip_event(day, hour)
        htpr = pr * 4185 * 288 if pr > 0 else 0.0

        w_ex, ht_ex, run, _, _, deep_ex = advance_bare_soil(
            w_ex, ht_ex, dz, q, qk, sl=0.02,
            pr=pr, htpr=htpr, srht=forc["srht"], trht=forc["trht"], ts=forc["ts"],
            rho=1.2, ch=0.01, vs=4.0, dt=dt_sec,
        )

        theta_ex[step + 1] = w_ex[:n] / dz[:n]
        runoff_ex[step + 1] = runoff_ex[step] + run
        deep_perc_ex[step] = deep_ex
        pr_all[step + 1] = pr

    # --- Implicit run ---
    w_im = w0.copy()
    ht_im = ht0.copy()
    theta_im = np.zeros((n_steps + 1, n))
    runoff_im = np.zeros(n_steps + 1)
    deep_perc_im = np.zeros(n_steps + 1)
    picard_iters = np.zeros(n_steps)
    mass_resid = np.zeros(n_steps)
    theta_im[0] = w0[:n] / dz[:n]
    runoff_im[0] = 0.0

    for step in range(n_steps):
        hour = (step * dt_hours) % 24
        day = int(step * dt_hours / 24)
        forc = diurnal(hour)
        pr = precip_event(day, hour)
        htpr = pr * 4185 * 288 if pr > 0 else 0.0

        w_im, ht_im, run, _, _, diag = advance_bare_soil_implicit(
            w_im, ht_im, dz, q, qk, sl=0.02,
            pr=pr, htpr=htpr, srht=forc["srht"], trht=forc["trht"], ts=forc["ts"],
            rho=1.2, ch=0.01, vs=4.0, dt=dt_sec,
            solver=solver, z_wt=None,
        )

        theta_im[step + 1] = w_im[:n] / dz[:n]
        runoff_im[step + 1] = runoff_im[step] + run
        deep_perc_im[step] = diag["deep_percolation"]
        picard_iters[step] = diag["picard_iterations"]
        mass_resid[step] = diag.get("mass_residual") or 0.0

    time_h = np.arange(n_steps + 1) * dt_hours

    explicit_data = {
        "time_h": time_h,
        "theta": theta_ex,
        "runoff_cum": runoff_ex,
        "deep_perc": deep_perc_ex,
        "pr": pr_all,
        "zc": zc[:n],
        "dz": dz[:n],
        "thets": thets[:n],
        "thetm": thetm[:n],
    }

    implicit_data = {
        "time_h": time_h,
        "theta": theta_im,
        "runoff_cum": runoff_im,
        "deep_perc": deep_perc_im,
        "picard_iters": picard_iters,
        "mass_resid": mass_resid,
        "pr": pr_all,
        "zc": zc[:n],
        "dz": dz[:n],
        "thets": thets[:n],
        "thetm": thetm[:n],
    }

    return explicit_data, implicit_data


def plot_diagnostics(explicit_data: dict, implicit_data: dict, outpath: Path) -> None:
    """Create multipanel diagnostic figure."""
    time_d = explicit_data["time_h"] / 24.0
    theta_ex = explicit_data["theta"]
    theta_im = implicit_data["theta"]
    runoff_ex = explicit_data["runoff_cum"]
    runoff_im = implicit_data["runoff_cum"]
    pr = explicit_data["pr"]
    zc = explicit_data["zc"]
    dz = explicit_data["dz"]
    thets = explicit_data["thets"][0]
    thetm = explicit_data["thetm"][0]
    n_layers = theta_ex.shape[1]
    depth_edges = np.concatenate([[0], np.cumsum(dz)])
    dt_d = time_d[1] - time_d[0] if len(time_d) > 1 else 1.0 / 24.0
    time_edges = np.concatenate([time_d, [time_d[-1] + dt_d]])

    fig = plt.figure(figsize=(14, 18))
    fig.suptitle(
        "Explicit vs Implicit Richards Solver Comparison",
        fontsize=14,
        fontweight="bold",
    )

    # --- Row 1: Theta time series (layer 1 = shallowest, TerraE 1-based) ---
    ax1 = fig.add_subplot(4, 3, 1)
    for k in range(n_layers):
        ax1.plot(time_d, theta_ex[:, k], ls="-", lw=1.2, label=f"L{k+1}")
    ax1.set_title("(a) Explicit: θ by layer")
    ax1.set_ylabel("θ (m³/m³)")
    ax1.axhline(thets, color="gray", ls="--", alpha=0.5)
    ax1.axhline(thetm, color="gray", ls=":", alpha=0.5)
    ax1.legend(loc="upper right", fontsize=7, ncol=2)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, time_d[-1])

    ax2 = fig.add_subplot(4, 3, 2)
    for k in range(n_layers):
        ax2.plot(time_d, theta_im[:, k], ls="-", lw=1.2, label=f"L{k+1}")
    ax2.set_title("(b) Implicit: θ by layer")
    ax2.set_ylabel("θ (m³/m³)")
    ax2.axhline(thets, color="gray", ls="--", alpha=0.5)
    ax2.axhline(thetm, color="gray", ls=":", alpha=0.5)
    ax2.legend(loc="upper right", fontsize=7, ncol=2)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, time_d[-1])

    ax3 = fig.add_subplot(4, 3, 3)
    diff = theta_ex - theta_im
    dt_d = time_d[1] - time_d[0] if len(time_d) > 1 else 1.0 / 24.0
    time_edges = np.concatenate([time_d, [time_d[-1] + dt_d]])
    im3 = ax3.pcolormesh(
        time_edges,
        depth_edges,
        diff.T,
        shading="flat",
        cmap="RdBu_r",
        vmin=-0.05,
        vmax=0.05,
    )
    ax3.set_title("(c) Difference (Explicit − Implicit)")
    ax3.set_ylabel("Depth (m)")
    ax3.invert_yaxis()
    ax3.set_xlabel("Time (days)")
    plt.colorbar(im3, ax=ax3, label="Δθ")

    # --- Row 2: Hovmöller (depth 0 at top) ---
    ax4 = fig.add_subplot(4, 3, 4)
    im4 = ax4.pcolormesh(
        time_edges, depth_edges, theta_ex.T,
        shading="flat", cmap="YlGnBu", vmin=thetm, vmax=thets,
    )
    ax4.set_title("(d) Explicit: θ Hovmöller")
    ax4.set_ylabel("Depth (m)")
    ax4.invert_yaxis()
    plt.colorbar(im4, ax=ax4, label="θ")

    ax5 = fig.add_subplot(4, 3, 5)
    im5 = ax5.pcolormesh(
        time_edges, depth_edges, theta_im.T,
        shading="flat", cmap="YlGnBu", vmin=thetm, vmax=thets,
    )
    ax5.set_title("(e) Implicit: θ Hovmöller")
    ax5.set_ylabel("Depth (m)")
    ax5.invert_yaxis()
    plt.colorbar(im5, ax=ax5, label="θ")

    ax6 = fig.add_subplot(4, 3, 6)
    im6 = ax6.pcolormesh(
        time_edges, depth_edges, diff.T,
        shading="flat", cmap="RdBu_r", vmin=-0.03, vmax=0.03,
    )
    ax6.set_title("(f) Difference Hovmöller")
    ax6.set_ylabel("Depth (m)")
    ax6.invert_yaxis()
    plt.colorbar(im6, ax=ax6, label="Δθ")

    # --- Row 3: Runoff, precip, deep percolation ---
    ax7 = fig.add_subplot(4, 3, 7)
    pr_mm = pr * 1000 * 3600
    ax7.bar(time_d, pr_mm, width=0.02, color="steelblue", alpha=0.6, label="Precip")
    ax7.set_ylabel("Precip (mm/h)", color="steelblue")
    ax7.tick_params(axis="y", labelcolor="steelblue")
    ax7b = ax7.twinx()
    ax7b.plot(time_d, runoff_ex * 1000, "g-", lw=1.5, label="Runoff explicit")
    ax7b.plot(time_d, runoff_im * 1000, "orange", ls="--", lw=1.5, label="Runoff implicit")
    ax7b.set_ylabel("Cum. runoff (mm)", color="green")
    ax7b.legend(loc="upper right", fontsize=8)
    ax7.set_title("(g) Precip and runoff (explicit vs implicit)")
    ax7.set_xlim(0, time_d[-1])

    ax8 = fig.add_subplot(4, 3, 8)
    deep_ex_mm = np.asarray(explicit_data["deep_perc"]).flatten() * 1000
    deep_im_mm = np.asarray(implicit_data["deep_perc"]).flatten() * 1000
    cum_deep_ex = np.cumsum(deep_ex_mm)
    cum_deep_im = np.cumsum(deep_im_mm)
    t_deep = np.linspace(time_d[1], time_d[-1], len(cum_deep_ex))
    ax8.plot(t_deep, cum_deep_ex, "g-", lw=1.5, label="Explicit")
    ax8.plot(t_deep, cum_deep_im, "orange", ls="--", lw=1.5, label="Implicit")
    ax8.set_xlabel("Time (days)")
    ax8.set_ylabel("Cum. deep perc. (mm)")
    ax8.set_title("(h) Deep percolation (explicit vs implicit)")
    ax8.legend(loc="upper left", fontsize=8)
    ax8.grid(True, alpha=0.3)

    ax9 = fig.add_subplot(4, 3, 9)
    picard = np.asarray(implicit_data["picard_iters"]).flatten()
    t_pic = np.linspace(time_d[1], time_d[-1], len(picard))
    ax9.bar(t_pic, picard, width=0.02, color="teal", alpha=0.7)
    ax9.set_xlabel("Time (days)")
    ax9.set_ylabel("Picard iterations")
    ax9.set_title("(i) Implicit: Picard iterations")
    ax9.axhline(np.nanmean(picard), color="red", ls="--", label=f"Mean={np.nanmean(picard):.1f}")
    ax9.legend(fontsize=8)

    # --- Row 4: Mass balance, profiles, summary ---
    ax10 = fig.add_subplot(4, 3, 10)
    mass_r = np.asarray(implicit_data["mass_resid"]).flatten()
    t_mass = np.linspace(time_d[1], time_d[-1], len(mass_r))
    ax10.plot(t_mass, np.nan_to_num(mass_r, nan=0.0) * 1000, "k-", lw=1)
    ax10.axhline(0, color="gray", ls="--")
    ax10.set_xlabel("Time (days)")
    ax10.set_ylabel("Mass residual (mm)")
    ax10.set_title("(j) Implicit: Mass balance residual")
    ax10.grid(True, alpha=0.3)

    ax11 = fig.add_subplot(4, 3, 11)
    t_select = [0, 3, 7, 10, 13]
    colors = plt.cm.viridis(np.linspace(0, 1, len(t_select)))
    depth_centers = np.cumsum(dz) - 0.5 * dz
    for i, td in enumerate(t_select):
        idx = np.argmin(np.abs(time_d - td))
        ax11.plot(theta_ex[idx, :], depth_centers, "o-", color=colors[i])
        ax11.plot(theta_im[idx, :], depth_centers, "s--", color=colors[i], alpha=0.7)
    # Proxy for legend: explicit = circles solid, implicit = squares dashed
    from matplotlib.lines import Line2D
    ax11.legend(
        handles=[
            Line2D([0], [0], marker="o", color="gray", linestyle="-", label="Explicit"),
            Line2D([0], [0], marker="s", color="gray", linestyle="--", label="Implicit"),
        ],
        loc="lower left",
        fontsize=8,
    )
    ax11.set_xlabel("θ (m³/m³)")
    ax11.set_ylabel("Depth (m)")
    ax11.set_title("(k) θ profiles at selected times (explicit vs implicit)")
    ax11.invert_yaxis()
    ax11.grid(True, alpha=0.3)

    ax12 = fig.add_subplot(4, 3, 12)
    ax12.axis("off")
    rmse = np.sqrt(np.nanmean((theta_ex - theta_im) ** 2))
    diff_by_layer = np.nanmax(np.abs(theta_ex - theta_im), axis=0)
    layer_str = " ".join(f"L{i+1}:{d:.3f}" for i, d in enumerate(diff_by_layer))
    txt = (
        f"Summary\n"
        f"───────\n"
        f"RMSE θ: {rmse:.4f} m³/m³\n"
        f"Max |Δθ|: {np.nanmax(np.abs(theta_ex - theta_im)):.4f}\n"
        f"Max |Δθ| by layer:\n  {layer_str}\n"
        f"(deep L{n_layers-1}–L{n_layers} often largest)\n"
        f"Runoff diff: {(runoff_ex[-1]-runoff_im[-1])*1000:.2f} mm\n"
        f"Mean Picard: {np.nanmean(picard):.1f}\n"
        f"Max |mass resid|: {np.nanmax(np.abs(np.nan_to_num(mass_r, nan=0)))*1000:.4f} mm"
    )
    ax12.text(0.1, 0.5, txt, fontsize=8, family="monospace", verticalalignment="center")

    plt.tight_layout()
    plt.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {outpath}")


def main() -> None:
    outdir = Path(__file__).resolve().parents[1] / "output"
    outdir.mkdir(exist_ok=True)
    figdir = outdir / "figures"
    figdir.mkdir(exist_ok=True)

    print("Running explicit solver...")
    print("Running implicit solver...")
    explicit_data, implicit_data = run_comparison(n_days=14, dt_hours=1.0)

    np.savez(
        outdir / "compare_explicit_implicit.npz",
        **{f"explicit_{k}": v for k, v in explicit_data.items()},
        **{f"implicit_{k}": v for k, v in implicit_data.items()},
    )

    plot_diagnostics(explicit_data, implicit_data, figdir / "compare_explicit_implicit.png")
    print("Done.")


if __name__ == "__main__":
    main()
