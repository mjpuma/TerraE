"""
Soil heat: xklh (De Vries conductivity), flh (heat flux), retp (temperature from heat).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from terrae import constants, types
from terrae.soil.hydraulic import VG_PARAMS


def xklh(
    theta: NDArray[np.float64],
    thets: NDArray[np.float64],
    w: NDArray[np.float64],
    fice: NDArray[np.float64],
    dz: NDArray[np.float64],
    zb: NDArray[np.float64],
    zc: NDArray[np.float64],
    q: NDArray[np.float64],
    n: int,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """
    De Vries heat conductivity. Returns xkh (in-layer) and xkhm (between layers).
    """
    ALAMW = 0.573345
    ALAMI = 2.1762
    ALAMA = 0.025
    ALAMBR = 2.9
    ALAMS = np.array([8.8, 2.9, 2.9, 0.25])  # sand, silt, clay, peat
    BA = ALAMA / ALAMW - 1.0

    gabc = np.array([0.125, 0.125, 0.75])
    hcwtw = 1.0
    hcwti = np.mean(1.0 / (1.0 + (ALAMI / ALAMW - 1) * gabc))
    hcwt = np.array([np.mean(1.0 / (1.0 + (ALAMS[i] / ALAMW - 1) * gabc)) for i in range(4)])
    hcwtb = 1.0

    xsha = np.zeros(n + 1)
    xsh = np.zeros(n + 1)
    for k in range(n):
        for i in range(types.IMT - 1):
            xs = (1.0 - VG_PARAMS[i].theta_s) * q[i, k]
            xsha[k] += xs * hcwt[i] * ALAMS[i]
            xsh[k] += xs * hcwt[i]

    xkh = np.zeros(n + 1)
    for k in range(n):
        gaa = 0.298 * theta[k] / (thets[k] + 1e-6) + 0.035
        gca = 1.0 - 2 * gaa
        hcwta = (2 / (1 + BA * gaa) + 1 / (1 + BA * gca)) / 3
        xw = w[k] * (1 - fice[k]) / dz[k]
        xi = w[k] * fice[k] / dz[k]
        xa = thets[k] - theta[k]
        xb = q[types.IMT - 1, k] if k < n else 0
        xnum = xw * hcwtw * ALAMW + xi * hcwti * ALAMI + xa * hcwta * ALAMA + xsha[k] + xb * hcwtb * ALAMBR
        xden = xw * hcwtw + xi * hcwti + xa * hcwta + xsh[k] + xb * hcwtb
        xkh[k] = xnum / (xden + 1e-30)

    xkhm = np.zeros(n + 1)
    for k in range(1, n):
        xkhm[k] = ((zb[k] - zc[k - 1]) * xkh[k] + (zc[k] - zb[k]) * xkh[k - 1]) / (zc[k] - zc[k - 1] + 1e-30)

    return xkh, xkhm


def flh(
    xkhm: NDArray[np.float64],
    tp: NDArray[np.float64],
    f: NDArray[np.float64],
    zc: NDArray[np.float64],
    n: int,
) -> NDArray[np.float64]:
    """Heat flux between layers: conduction + advection."""
    fh = np.zeros(n + 1)
    fh[n] = 0.0
    shw = constants.SHW_VOL
    for k in range(1, n):
        dz = zc[k - 1] - zc[k]
        if abs(dz) > 1e-30:
            fh[k] = -xkhm[k] * (tp[k - 1] - tp[k]) / dz
            if f[k] > 0:
                fh[k] += f[k] * tp[k] * shw
            else:
                fh[k] += f[k] * tp[k - 1] * shw
    return fh


def retp(
    w: NDArray[np.float64],
    ht: NDArray[np.float64],
    shc: NDArray[np.float64],
    n: int,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """
    Temperature and ice fraction from heat content.
    ht = heat in layer, w = water. Handles freezing.
    """
    fsn = constants.FSN
    shw = constants.SHW_VOL
    shi = constants.SHI_VOL
    tp = np.zeros(n + 1)
    fice = np.zeros(n + 1)

    for k in range(n):
        if w[k] < 1e-12:
            tp[k] = ht[k] / (shc[k] + 1e-30)
            fice[k] = 0.0
        elif fsn * w[k] + ht[k] < 0:
            tp[k] = (ht[k] + w[k] * fsn) / (shc[k] + w[k] * shi)
            fice[k] = 1.0
        elif ht[k] > 0:
            tp[k] = ht[k] / (shc[k] + w[k] * shw)
            fice[k] = 0.0
        else:
            fice[k] = -ht[k] / (fsn * w[k] + 1e-30)
            fice[k] = np.clip(fice[k], 0, 1)
            tp[k] = 0.0

    return tp, fice
