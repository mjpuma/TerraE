#!/usr/bin/env python3
"""
Detailed visualization of TerraE sample run.

Creates multi-panel figure: soil moisture, temperature, runoff, profiles, Hovmöller.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

OUTPUT_DIR = Path(__file__).resolve().parents[1] / "output"
FIG_DIR = OUTPUT_DIR / "figures"


def load_run() -> dict:
    """Load sample_run.npz."""
    data = np.load(OUTPUT_DIR / "sample_run.npz")
    return {k: data[k] for k in data.files}


def plot_all(data: dict, figpath: Path | None = None) -> None:
    """Create comprehensive multi-panel visualization."""
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    figpath = figpath or FIG_DIR / "terrae_sample_run.png"

    time_h = data["time_h"]
    theta = data["theta"]
    tp = data["tp"]
    w_tot = data["w_tot"]
    runoff_cum = data["runoff_cum"]
    pr = data["pr"]
    zc = data["zc"]
    dz = data["dz"]
    thets = data["thets"]
    thetm = data["thetm"]

    n_layers = theta.shape[1]
    time_d = time_h / 24.0

    fig = plt.figure(figsize=(14, 16))
    fig.suptitle("TerraE Land Model: 14-Day Bare Soil Simulation", fontsize=14, fontweight="bold")

    # --- Panel 1: Volumetric water content (theta) time series ---
    ax1 = fig.add_subplot(4, 2, 1)
    for k in range(n_layers):
        ax1.plot(time_d, theta[:, k], label=f"Layer {k+1} (z={-zc[k]:.2f}m)", lw=1.5)
    ax1.axhline(thets[0], color="gray", ls="--", alpha=0.7, label="θs")
    ax1.axhline(thetm[0], color="gray", ls=":", alpha=0.7, label="θr")
    ax1.set_xlabel("Time (days)")
    ax1.set_ylabel("θ (m³/m³)")
    ax1.set_title("(a) Soil moisture by layer")
    ax1.legend(loc="upper right", fontsize=8)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, time_d[-1])

    # --- Panel 2: Temperature time series ---
    ax2 = fig.add_subplot(4, 2, 2)
    for k in range(min(4, n_layers)):
        ax2.plot(time_d, tp[:, k], label=f"Layer {k+1} (z={-zc[k]:.2f}m)", lw=1.5)
    ax2.set_xlabel("Time (days)")
    ax2.set_ylabel("Temperature (°C)")
    ax2.set_title("(b) Soil temperature by layer")
    ax2.legend(loc="upper right", fontsize=8)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, time_d[-1])

    # --- Panel 3: Precipitation and runoff ---
    ax3 = fig.add_subplot(4, 2, 3)
    pr_mm = pr * 1000 * 3600  # mm/h (assuming 1h steps)
    ax3.bar(time_d, pr_mm, width=0.03, color="steelblue", alpha=0.7, label="Precip (mm/h)")
    ax3.set_ylabel("Precip (mm/h)", color="steelblue")
    ax3.tick_params(axis="y", labelcolor="steelblue")
    ax3.set_xlim(0, time_d[-1])

    ax3b = ax3.twinx()
    ax3b.plot(time_d, runoff_cum * 1000, color="darkgreen", lw=2, label="Cum. runoff (mm)")
    ax3b.set_ylabel("Cum. runoff (mm)", color="darkgreen")
    ax3b.tick_params(axis="y", labelcolor="darkgreen")
    ax3.set_xlabel("Time (days)")
    ax3.set_title("(c) Precipitation and cumulative runoff")

    # --- Panel 4: Total soil water ---
    ax4 = fig.add_subplot(4, 2, 4)
    ax4.plot(time_d, w_tot * 1000, "b-", lw=2, label="Total water")
    ax4.set_xlabel("Time (days)")
    ax4.set_ylabel("Total water (mm)")
    ax4.set_title("(d) Column-integrated water")
    ax4.grid(True, alpha=0.3)
    ax4.set_xlim(0, time_d[-1])

    # --- Panel 5: Hovmöller - theta vs depth and time ---
    ax5 = fig.add_subplot(4, 2, 5)
    depth = -zc
    im = ax5.pcolormesh(
        time_d,
        depth,
        theta.T,
        shading="auto",
        cmap="YlGnBu",
        vmin=thetm[0],
        vmax=thets[0],
    )
    ax5.set_xlabel("Time (days)")
    ax5.set_ylabel("Depth (m)")
    ax5.set_title("(e) Soil moisture Hovmöller")
    ax5.invert_yaxis()
    plt.colorbar(im, ax=ax5, label="θ (m³/m³)")

    # --- Panel 6: Vertical moisture profiles at selected times ---
    ax6 = fig.add_subplot(4, 2, 6)
    t_select = [0, 2, 5, 7, 10, 13]
    colors = plt.cm.viridis(np.linspace(0, 1, len(t_select)))
    for i, td in enumerate(t_select):
        idx = np.argmin(np.abs(time_d - td))
        ax6.plot(theta[idx, :], -zc, "o-", color=colors[i], label=f"Day {time_d[idx]:.1f}")
    ax6.axvline(thets[0], color="gray", ls="--", alpha=0.7)
    ax6.axvline(thetm[0], color="gray", ls=":", alpha=0.7)
    ax6.set_xlabel("θ (m³/m³)")
    ax6.set_ylabel("Depth (m)")
    ax6.set_title("(f) Moisture profiles at selected times")
    ax6.legend(loc="lower left", fontsize=8)
    ax6.grid(True, alpha=0.3)
    ax6.invert_yaxis()

    # --- Panel 7: Temperature Hovmöller ---
    ax7 = fig.add_subplot(4, 2, 7)
    im7 = ax7.pcolormesh(time_d, depth, tp.T, shading="auto", cmap="RdYlBu_r")
    ax7.set_xlabel("Time (days)")
    ax7.set_ylabel("Depth (m)")
    ax7.set_title("(g) Soil temperature Hovmöller")
    ax7.invert_yaxis()
    plt.colorbar(im7, ax=ax7, label="T (°C)")

    # --- Panel 8: Vertical temperature profiles at selected times ---
    ax8 = fig.add_subplot(4, 2, 8)
    for i, td in enumerate(t_select):
        idx = np.argmin(np.abs(time_d - td))
        ax8.plot(tp[idx, :], -zc, "o-", color=colors[i], label=f"Day {time_d[idx]:.1f}")
    ax8.set_xlabel("Temperature (°C)")
    ax8.set_ylabel("Depth (m)")
    ax8.set_title("(h) Temperature profiles at selected times")
    ax8.legend(loc="upper right", fontsize=8)
    ax8.grid(True, alpha=0.3)
    ax8.invert_yaxis()

    plt.tight_layout()
    plt.savefig(figpath, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {figpath}")


def load_run_multi() -> dict:
    """Load sample_run_multi.npz."""
    data = np.load(OUTPUT_DIR / "sample_run_multi.npz", allow_pickle=True)
    return {k: data[k] for k in data.files}


def plot_subgrid_fractions(data: dict, figpath: Path | None = None) -> None:
    """Create per-column visualization for each land cover fraction."""
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    figpath = figpath or FIG_DIR / "terrae_subgrid_fractions.png"

    time_h = data["time_h"]
    theta = data["theta"]  # (n_steps+1, n_layers, n_cols)
    tp = data["tp"]
    w_tot_cols = data["w_tot_cols"]  # (n_steps+1, n_cols)
    runoff_cum = data["runoff_cum"]
    pr = data["pr"]
    fractions = data["fractions"]
    zc = data["zc"]
    thets = data["thets"]
    thetm = data["thetm"]

    # land_cover_types can be stored as object array
    lct = data.get("land_cover_types")
    if lct is not None:
        col_names = [str(x) for x in lct]
    else:
        col_names = [f"Col {j}" for j in range(w_tot_cols.shape[1])]

    n_cols = w_tot_cols.shape[1]
    n_layers = theta.shape[1]
    time_d = time_h / 24.0

    # Shared y-axis ranges for fair comparison
    theta_min = max(0, float(thetm[0]) - 0.02)
    theta_max = min(1, float(thets[0]) + 0.02)
    w_mm = w_tot_cols * 1000
    w_min = max(0, np.min(w_mm) - 10)
    w_max = np.max(w_mm) + 10

    # Colors for each land cover type
    colors = plt.cm.Set2(np.linspace(0, 1, n_cols))

    fig, axes = plt.subplots(2, 3, figsize=(14, 9), sharex=True)
    axes = axes.flatten()
    fig.suptitle("TerraE: Soil State by Land Cover Fraction", fontsize=14, fontweight="bold")

    for j, ax in enumerate(axes):
        if j >= n_cols:
            ax.set_visible(False)
            continue

        # Top layer theta
        ax.plot(time_d, theta[:, 0, j], color=colors[j], lw=2, label="θ surface")
        ax.axhline(thets[0], color="gray", ls="--", alpha=0.5)
        ax.axhline(thetm[0], color="gray", ls=":", alpha=0.5)
        ax.set_ylim(theta_min, theta_max)

        ax.set_ylabel("θ (m³/m³)", color=colors[j])
        ax.tick_params(axis="y", labelcolor=colors[j])
        ax.set_title(f"{col_names[j]} (frac={fractions[j]:.0%})")
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, time_d[-1])

        ax2 = ax.twinx()
        ax2.plot(time_d, w_tot_cols[:, j] * 1000, "k--", lw=1, alpha=0.7, label="Total water")
        ax2.set_ylim(w_min, w_max)
        ax2.set_ylabel("Total water (mm)", color="gray")
        ax2.tick_params(axis="y", labelcolor="gray")

    for ax in axes[:3]:
        ax.set_xticklabels([])
    for ax in axes[3:]:
        ax.set_xlabel("Time (days)")

    plt.tight_layout()
    plt.savefig(figpath, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {figpath}")


def plot_subgrid_hovmoller(data: dict) -> None:
    """Hovmöller (theta vs depth-time) for each land cover column."""
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    time_h = data["time_h"]
    theta = data["theta"]
    zc = data["zc"]
    thets = data["thets"]
    thetm = data["thetm"]
    fractions = data["fractions"]

    lct = data.get("land_cover_types")
    col_names = [str(x) for x in lct] if lct is not None else [f"Col {j}" for j in range(theta.shape[2])]

    n_cols = theta.shape[2]
    time_d = time_h / 24.0
    depth = -zc

    # Same color scale for all Hovmöller panels
    vmin, vmax = float(thetm[0]), float(thets[0])

    fig, axes = plt.subplots(2, 3, figsize=(14, 8), sharex=True, sharey=True)
    axes = axes.flatten()
    fig.suptitle("TerraE: Soil Moisture Hovmöller by Land Cover", fontsize=14, fontweight="bold")

    for j, ax in enumerate(axes):
        if j >= n_cols:
            ax.set_visible(False)
            continue

        im = ax.pcolormesh(
            time_d,
            depth,
            theta[:, :, j].T,
            shading="auto",
            cmap="YlGnBu",
            vmin=vmin,
            vmax=vmax,
        )
        ax.set_title(f"{col_names[j]} ({fractions[j]:.0%})")
        ax.invert_yaxis()
        ax.set_xlabel("Time (days)")
        ax.set_ylabel("Depth (m)")
        plt.colorbar(im, ax=ax, label="θ")

    plt.tight_layout()
    plt.savefig(FIG_DIR / "terrae_subgrid_hovmoller.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {FIG_DIR / 'terrae_subgrid_hovmoller.png'}")


def plot_subgrid_temp_hovmoller(data: dict) -> None:
    """Hovmöller (temperature vs depth-time) for each land cover column."""
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    time_h = data["time_h"]
    tp = data["tp"]
    zc = data["zc"]
    fractions = data["fractions"]

    lct = data.get("land_cover_types")
    col_names = [str(x) for x in lct] if lct is not None else [f"Col {j}" for j in range(tp.shape[2])]

    n_cols = tp.shape[2]
    time_d = time_h / 24.0
    depth = -zc

    fig, axes = plt.subplots(2, 3, figsize=(14, 8), sharex=True, sharey=True)
    axes = axes.flatten()
    fig.suptitle("TerraE: Soil Temperature Hovmöller by Land Cover", fontsize=14, fontweight="bold")

    for j, ax in enumerate(axes):
        if j >= n_cols:
            ax.set_visible(False)
            continue

        im = ax.pcolormesh(
            time_d,
            depth,
            tp[:, :, j].T,
            shading="auto",
            cmap="RdYlBu_r",
        )
        ax.set_title(f"{col_names[j]} ({fractions[j]:.0%})")
        ax.invert_yaxis()
        ax.set_xlabel("Time (days)")
        ax.set_ylabel("Depth (m)")
        plt.colorbar(im, ax=ax, label="T (°C)")

    plt.tight_layout()
    plt.savefig(FIG_DIR / "terrae_subgrid_temp_hovmoller.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {FIG_DIR / 'terrae_subgrid_temp_hovmoller.png'}")


def plot_subgrid_comparison(data: dict) -> None:
    """Side-by-side comparison: total water, theta surface, and temperature for all columns."""
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    time_h = data["time_h"]
    theta = data["theta"]
    tp = data["tp"]
    w_tot_cols = data["w_tot_cols"]
    fractions = data["fractions"]

    lct = data.get("land_cover_types")
    col_names = [str(x) for x in lct] if lct is not None else [f"Col {j}" for j in range(w_tot_cols.shape[1])]

    n_cols = w_tot_cols.shape[1]
    time_d = time_h / 24.0
    colors = plt.cm.Set2(np.linspace(0, 1, n_cols))

    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

    # Panel 1: Total water per column
    ax1 = axes[0]
    for j in range(n_cols):
        ax1.plot(time_d, w_tot_cols[:, j] * 1000, color=colors[j], lw=1.5, label=f"{col_names[j]} ({fractions[j]:.0%})")
    ax1.set_ylabel("Total water (mm)")
    ax1.set_title("(a) Column-integrated water by land cover")
    ax1.legend(loc="upper right", fontsize=9)
    ax1.grid(True, alpha=0.3)

    # Panel 2: Surface theta per column
    ax2 = axes[1]
    for j in range(n_cols):
        ax2.plot(time_d, theta[:, 0, j], color=colors[j], lw=1.5, label=col_names[j])
    ax2.set_ylabel("θ surface (m³/m³)")
    ax2.set_title("(b) Surface layer moisture by land cover")
    ax2.legend(loc="upper right", fontsize=9)
    ax2.grid(True, alpha=0.3)

    # Panel 3: Surface temperature per column
    ax3 = axes[2]
    for j in range(n_cols):
        ax3.plot(time_d, tp[:, 0, j], color=colors[j], lw=1.5, label=col_names[j])
    ax3.set_xlabel("Time (days)")
    ax3.set_ylabel("T surface (°C)")
    ax3.set_title("(c) Surface layer temperature by land cover")
    ax3.legend(loc="upper right", fontsize=9)
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(FIG_DIR / "terrae_subgrid_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {FIG_DIR / 'terrae_subgrid_comparison.png'}")


def plot_water_balance(data: dict) -> None:
    """Water balance check: precip (+ irrig) - runoff - change in storage."""
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    time_h = data["time_h"]
    w_tot = data["w_tot"]
    runoff_cum = data["runoff_cum"]
    pr = data["pr"]
    irrig_cum = data.get("irrig_cum")

    dt_sec = np.diff(time_h) * 3600
    pr_cum = np.concatenate([[0], np.cumsum(pr[1:] * dt_sec)])
    if irrig_cum is not None:
        water_in = pr_cum + irrig_cum
    else:
        water_in = pr_cum
    storage_change = w_tot - w_tot[0]
    balance = water_in - runoff_cum - storage_change

    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    axes[0].plot(time_h / 24, pr_cum * 1000, label="Cum. precip (mm)")
    if irrig_cum is not None:
        axes[0].plot(time_h / 24, irrig_cum * 1000, label="Cum. irrig (mm)")
    axes[0].plot(time_h / 24, runoff_cum * 1000, label="Cum. runoff (mm)")
    axes[0].plot(time_h / 24, storage_change * 1000, label="Δ storage (mm)")
    axes[0].set_ylabel("mm")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[0].set_title("Water balance components")

    axes[1].plot(time_h / 24, balance * 1000, "r-", lw=1.5)
    axes[1].axhline(0, color="k", ls="--", alpha=0.5)
    axes[1].set_xlabel("Time (days)")
    axes[1].set_ylabel("Balance error (mm)")
    axes[1].set_title("Conservation residual: (precip+irrig) - runoff - Δstorage")
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(FIG_DIR / "water_balance.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {FIG_DIR / 'water_balance.png'}")
    print(f"Max balance error: {np.max(np.abs(balance))*1000:.4f} mm")


if __name__ == "__main__":
    multi = "--multi" in sys.argv
    sys.path.insert(0, str(Path(__file__).resolve().parent))

    if multi:
        if not (OUTPUT_DIR / "sample_run_multi.npz").exists():
            print("Running multi-column simulation first...")
            from sample_run import run_simulation_multi_column

            run_simulation_multi_column(n_days=14, dt_hours=1.0)

        data = load_run_multi()
        plot_subgrid_fractions(data)
        plot_subgrid_hovmoller(data)
        plot_subgrid_temp_hovmoller(data)
        plot_subgrid_comparison(data)
        plot_water_balance(data)
    else:
        if not (OUTPUT_DIR / "sample_run.npz").exists():
            print("Running sample simulation first...")
            from sample_run import run_simulation

            run_simulation(n_days=14, dt_hours=1.0)

        data = load_run()
        plot_all(data)
        plot_water_balance(data)

    print("Visualization complete.")
