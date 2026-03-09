"""
Time-stepping driver: advnc (single cell, bare soil), gdtm, check_water, check_energy.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from terrae import constants, types
from terrae.soil.heat import flh, retp, xklh
from terrae.soil.hydrology import fl, fllmt, hydra, reth, runoff
from terrae.soil.properties import get_soil_properties


def init_step(
    dz: NDArray[np.float64],
    q: NDArray[np.float64],
) -> tuple[int, NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Initialize layer geometry and soil properties."""
    n = 0
    for k in range(types.NGM):
        if dz[k] <= 0:
            break
        n = k + 1

    if n <= 0:
        raise ValueError("GHY: init_step n <= 0")

    zb = np.zeros(n + 1)
    zb[0] = 0
    for k in range(n):
        zb[k + 1] = zb[k] - dz[k]

    zc = np.zeros(n)
    for k in range(n):
        zc[k] = 0.5 * (zb[k] + zb[k + 1])

    thets, thetm, shc = get_soil_properties(q, dz)
    ws = thets * dz

    return n, zb, zc, thets, thetm, shc, ws


def gdtm(
    d: NDArray[np.float64],
    dz: NDArray[np.float64],
    shc: NDArray[np.float64],
    xkh: NDArray[np.float64],
    w: NDArray[np.float64],
    fice: NDArray[np.float64],
    tp: NDArray[np.float64],
    n: int,
    ch: float,
    vs: float,
    rho: float,
    ts: float,
) -> float:
    """Maximum stable timestep from CFL limits."""
    sgmm = 1.0
    dldz2 = np.max(d[:n] / (dz[:n] ** 2 + 1e-30))
    dtm = sgmm / (dldz2 + 1e-12)

    shw = constants.SHW_VOL
    shi = constants.SHI_VOL
    for k in range(n):
        xk1 = xkh[k]
        ak1 = (shc[k] + ((1 - fice[k]) * shw + fice[k] * shi) * w[k]) / dz[k]
        dtm = min(dtm, 0.5 * ak1 * dz[k] ** 2 / (xk1 + 1e-12))

    rho3 = 0.001 * rho
    cna = ch * vs
    for k in [0]:
        ak2 = shc[k] + ((1 - fice[k]) * shw + fice[k] * shi) * w[k]
        xk2 = constants.SHA * rho * cna + 8 * constants.STBO * (tp[k] + constants.TF) ** 3
        dtm = min(dtm, 0.5 * ak2 / (xk2 + 1e-12))

    return max(dtm, 5.0)


def apply_fluxes(
    w: NDArray[np.float64],
    ht: NDArray[np.float64],
    f: NDArray[np.float64],
    fh: NDArray[np.float64],
    rnf: float,
    rnff: NDArray[np.float64],
    evapdl: NDArray[np.float64],
    tp: NDArray[np.float64],
    dz: NDArray[np.float64],
    thetm: NDArray[np.float64],
    ws: NDArray[np.float64],
    dts: float,
    n: int,
) -> float:
    """
    Explicit flux update for w and ht.
    Conservative: any oversaturation is added to runoff to preserve water balance.
    Returns updated rnf (for accumulation).
    """
    shw = constants.SHW_VOL
    fd = 1.0

    w[0] = w[0] - rnf * dts
    ht[0] = ht[0] - shw * max(tp[0], 0) * rnf * dts

    for k in range(n):
        w[k] = w[k] + (f[k + 1] - f[k] - rnff[k] - fd * evapdl[k]) * dts
        ht[k] = ht[k] + (fh[k + 1] - fh[k] - shw * max(tp[k], 0) * rnff[k]) * dts

    # Conservative bounds: excess goes to runoff; energy adjusted to preserve balance
    for k in range(n):
        if w[k] > ws[k]:
            excess = w[k] - ws[k]
            w[k] = ws[k]
            ht[k] = ht[k] - shw * max(tp[k], 0) * excess
            if k == 0:
                rnf = rnf + excess / dts
            else:
                rnff[k] = rnff[k] + excess / dts
        elif w[k] < dz[k] * thetm[k]:
            deficit = dz[k] * thetm[k] - w[k]
            w[k] = dz[k] * thetm[k]
            ht[k] = ht[k] + shw * max(tp[k], 0) * deficit
            if k == 0:
                rnf = max(0.0, rnf - deficit / dts)
            else:
                rnff[k] = max(0.0, rnff[k] - deficit / dts)

    return rnf


def advance_bare_soil(
    w: NDArray[np.float64],
    ht: NDArray[np.float64],
    dz: NDArray[np.float64],
    q: NDArray[np.float64],
    qk: NDArray[np.float64],
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
    irrig: float = 0.0,
) -> tuple[NDArray[np.float64], NDArray[np.float64], float, float, float]:
    """
    Advance bare soil one GCM timestep.

    Returns:
        w, ht, total_runoff, total_evap, total_heat_runoff, total_deep_perc.
    Bottom BC: unit gradient free drainage (matches implicit solver).
    irrig: irrigation flux (m/s), added to precip.
    """
    n, zb, zc, thets, thetm, shc, ws = init_step(dz, q)
    evapdl = np.zeros(n)
    dts_rem = dt
    total_runoff = 0.0
    total_evap = 0.0
    total_heat_runoff = 0.0
    total_deep_perc = 0.0
    pr_tot = pr + irrig
    htpr_tot = htpr + (irrig * 4185 * 288 if irrig > 0 else 0.0)

    while dts_rem > 1e-12:
        theta = reth(w, dz, n)
        tp, fice = retp(w, ht, shc, n)
        h, xk, d, xku = hydra(theta, thets, thetm, fice, zc, q, qk, n)

        xkh, xkhm = xklh(theta, thets, w, fice, dz, zb, zc, q, n)
        dtm = gdtm(d, dz, shc, xkh, w, fice, tp, n, ch, vs, rho, ts)

        if dtm >= dts_rem:
            dts = dts_rem
            dts_rem = 0.0
        else:
            dts = min(dtm, 0.5 * dts_rem)
            dts_rem -= dts

        f, xinfc = fl(h, xk, zc, n)
        f[0] = -pr_tot  # Surface flux: positive = upward; precip+irrig downward
        f[n] = -xku[n - 1]  # Bottom: unit gradient free drainage (match implicit BC)

        rnf, rnff = runoff(w, ws, f, xinfc, xku, dz, sl, n)
        rnf = fllmt(w, ws, thetm, dz, f, rnf, rnff, evapdl, dts, n)

        fh = flh(xkhm, tp, f, zc, n)
        fh[0] = htpr_tot - srht - trht

        rnf = apply_fluxes(w, ht, f, fh, rnf, rnff, evapdl, tp, dz, thetm, ws, dts, n)

        total_runoff += (rnf + np.sum(rnff[:n])) * dts
        total_heat_runoff += constants.SHW_VOL * (
            rnf * max(tp[0], 0) + np.sum(rnff[:n] * np.maximum(tp[:n], 0))
        ) * dts
        total_deep_perc += max(0.0, -f[n]) * dts  # f[n] negative = downward drainage

        theta = reth(w, dz, n)
        tp, fice = retp(w, ht, shc, n)

    return w, ht, total_runoff, total_evap, total_heat_runoff, total_deep_perc


def advance_cell(
    w: NDArray[np.float64],
    ht: NDArray[np.float64],
    fractions: NDArray[np.float64],
    dz: NDArray[np.float64],
    q: NDArray[np.float64],
    qk: NDArray[np.float64],
    sl: NDArray[np.float64],
    pr: float,
    htpr: float,
    srht: float,
    trht: float,
    ts: float,
    rho: float,
    ch: float,
    vs: float,
    dt: float,
    irrig: NDArray[np.float64] | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.float64], float, float, float]:
    """
    Advance grid cell with multiple land cover columns. Area-weighted aggregation.

    w: (n_layers, n_cols) water content per column
    ht: (n_layers+1, n_cols) heat content per column
    fractions: (n_cols) area fractions, must sum to 1
    dz: (n_layers) shared layer thicknesses
    q, qk: (IMT, NGM) shared soil texture
    sl: (n_cols) slope per column (urban can have higher runoff)
    irrig: (n_cols) irrigation flux m/s per column, or None for zeros

    Returns updated w, ht, area-weighted total runoff, total evap, total deep percolation.
    """
    types.validate_fractions(fractions)
    n_cols = fractions.shape[0]
    n = w.shape[0]
    if irrig is None:
        irrig = np.zeros(n_cols)

    total_runoff = 0.0
    total_evap = 0.0
    total_deep_perc = 0.0

    for j in range(n_cols):
        if fractions[j] < 1e-12:
            continue
        w_j = w[:, j].copy()
        ht_j = ht[:, j].copy()
        w_j, ht_j, run, evap, _, deep_perc = advance_bare_soil(
            w_j, ht_j, dz, q, qk, sl[j], pr, htpr, srht, trht, ts, rho, ch, vs, dt, irrig=irrig[j]
        )
        w[:, j] = w_j
        ht[:, j] = ht_j
        total_runoff += fractions[j] * run
        total_evap += fractions[j] * evap
        total_deep_perc += fractions[j] * deep_perc

    return w, ht, total_runoff, total_evap, total_deep_perc


def check_water(
    w_before: float,
    w_after: NDArray[np.float64],
    pr: float,
    evap: float,
    rnf: float,
    rnff: NDArray[np.float64],
    dts: float,
    n: int,
) -> float:
    """Water conservation error (m/s)."""
    total_after = np.sum(w_after[:n])
    flux = pr - evap - rnf - np.sum(rnff[:n])
    return (total_after - w_before) / dts - flux


def check_energy(
    e_before: float,
    ht_after: NDArray[np.float64],
    htpr: float,
    evap: float,
    rnf: float,
    rnff: NDArray[np.float64],
    tp: NDArray[np.float64],
    srht: float,
    trht: float,
    dts: float,
    n: int,
) -> float:
    """Energy conservation error (W/m²). Flux into soil = -fh[0] = srht + trht - htpr - ELH*evap - heat_runoff."""
    shw = constants.SHW_VOL
    total_after = np.sum(ht_after[:n])
    heat_runoff = shw * (rnf * max(tp[0], 0) + np.sum(rnff[:n] * np.maximum(tp[:n], 0)))
    flux = srht + trht - htpr - constants.ELH * evap - heat_runoff
    return (total_after - e_before) / dts - flux
