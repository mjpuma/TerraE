"""
Van Genuchten–Mualem soil hydraulic model.

Standard formulation used in CLM, JULES, HYDRUS, and most modern LSMs.
Parameters from ROSETTA pedotransfer (Schaap et al. 2001) and Carsel & Parrish (1988).

References:
- van Genuchten (1980) Soil Sci. Soc. Am. J. 44:892–898
- Mualem (1976) Water Resour. Res. 12:513–522
- Schaap et al. (2001) J. Hydrol. 251:163–176
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from numpy.typing import NDArray

from terrae import types

# ----- Van Genuchten–Mualem equations -----
# θ(h) = θr + (θs − θr) / (1 + |αh|^n)^m,  m = 1 − 1/n
# K(Se) = Ks × √Se × [1 − (1 − Se^(1/m))^m]²
# Units: h [m], α [1/m], Ks [m/s], θ [m³/m³]


@dataclass
class VgParams:
    """Van Genuchten parameters for one soil texture."""

    theta_r: float  # Residual water content (m³/m³)
    theta_s: float  # Saturated water content (m³/m³)
    alpha: float  # Inverse of air-entry pressure (1/m)
    n: float  # Pore-size distribution index
    Ks: float  # Saturated hydraulic conductivity (m/s)

    @property
    def m(self) -> float:
        return 1.0 - 1.0 / self.n


# ROSETTA / Carsel & Parrish class-average parameters.
# alpha [1/m] = alpha_cm * 100; Ks [m/s] = Ks_cm_day / 86400 / 100
# Order: Sand, Silt (Loam proxy), Clay, Peat, Loam (5th texture for compatibility)
VG_PARAMS: tuple[VgParams, ...] = (
    VgParams(theta_r=0.045, theta_s=0.43, alpha=14.5, n=2.68, Ks=8.25e-5),   # Sand 712.8 cm/day
    VgParams(theta_r=0.078, theta_s=0.43, alpha=3.6, n=1.56, Ks=2.88e-6),    # Silt/Loam 249 cm/day
    VgParams(theta_r=0.098, theta_s=0.43, alpha=0.8, n=1.09, Ks=7.22e-7),    # Clay 62.4 cm/day
    VgParams(theta_r=0.07, theta_s=0.61, alpha=1.8, n=1.53, Ks=5.21e-7),     # Peat
    VgParams(theta_r=0.078, theta_s=0.43, alpha=3.6, n=1.56, Ks=2.88e-6),   # Loam
)


def theta_from_h(h: float | NDArray, params: VgParams) -> float | NDArray:
    """
    Volumetric water content from matric potential (van Genuchten).

    Args:
        h: Matric potential [m], h ≤ 0 (suction positive)
        params: Van Genuchten parameters

    Returns:
        θ [m³/m³]
    """
    h = np.asarray(h, dtype=float)
    h_neg = np.minimum(h, 0.0)  # Only negative h (suction)
    m = params.m
    Se = 1.0 / (1.0 + (params.alpha * np.abs(h_neg)) ** params.n) ** m
    return params.theta_r + (params.theta_s - params.theta_r) * Se


def h_from_theta(theta: float | NDArray, params: VgParams) -> float | NDArray:
    """
    Matric potential from water content (inverse van Genuchten).

    Args:
        theta: Volumetric water content [m³/m³]
        params: Van Genuchten parameters

    Returns:
        h [m], negative for unsaturated
    """
    theta = np.asarray(theta, dtype=float)
    theta = np.clip(theta, params.theta_r + 1e-12, params.theta_s - 1e-12)
    Se = (theta - params.theta_r) / (params.theta_s - params.theta_r)
    m = params.m
    # Se = 1/(1+(α|h|)^n)^m  =>  (α|h|)^n = Se^(-1/m) - 1
    inner = Se ** (-1.0 / m) - 1.0
    inner = np.maximum(inner, 1e-30)
    h = -(inner ** (1.0 / params.n)) / params.alpha
    return h


def K_from_Se(Se: float | NDArray, params: VgParams) -> float | NDArray:
    """
    Hydraulic conductivity from effective saturation (Mualem).

    K = Ks × √Se × [1 − (1 − Se^(1/m))^m]²
    """
    Se = np.asarray(Se, dtype=float)
    Se = np.clip(Se, 1e-12, 1.0)
    m = params.m
    term = 1.0 - (1.0 - Se ** (1.0 / m)) ** m
    term = np.maximum(term, 1e-30)  # Avoid 0^2
    return params.Ks * np.sqrt(Se) * term**2


def K_from_theta(theta: float | NDArray, params: VgParams) -> float | NDArray:
    """Hydraulic conductivity from water content."""
    theta = np.asarray(theta, dtype=float)
    theta = np.clip(theta, params.theta_r + 1e-12, params.theta_s - 1e-12)
    Se = (theta - params.theta_r) / (params.theta_s - params.theta_r)
    return K_from_Se(Se, params)


def D_from_theta(theta: float | NDArray, params: VgParams, dtheta: float = 1e-6) -> float | NDArray:
    """
    Hydraulic diffusivity D = K × dθ/dh.

    Uses numerical derivative for robustness.
    """
    theta = np.asarray(theta, dtype=float)
    K = K_from_theta(theta, params)
    theta_hi = np.minimum(theta + dtheta, params.theta_s - 1e-10)
    theta_lo = np.maximum(theta - dtheta, params.theta_r + 1e-10)
    h_hi = h_from_theta(theta_hi, params)
    h_lo = h_from_theta(theta_lo, params)
    dtheta_dh = (theta_hi - theta_lo) / (h_hi - h_lo + 1e-30)
    return K / (np.abs(dtheta_dh) + 1e-30)


def weighted_params(q: NDArray[np.float64], layer: int) -> VgParams:
    """
    Van Genuchten parameters for a mixed-texture layer.

    Uses linear weighting by texture fraction. For proper mixing,
    consider harmonic mean for Ks and more sophisticated averaging.
    """
    theta_r = sum(q[i, layer] * VG_PARAMS[i].theta_r for i in range(types.IMT))
    theta_s = sum(q[i, layer] * VG_PARAMS[i].theta_s for i in range(types.IMT))
    # Geometric mean for alpha, n; harmonic for Ks (common practice)
    alpha_prod = 1.0
    n_prod = 1.0
    Ks_inv = 0.0
    for i in range(types.IMT):
        if q[i, layer] > 1e-10:
            alpha_prod *= VG_PARAMS[i].alpha ** q[i, layer]
            n_prod *= VG_PARAMS[i].n ** q[i, layer]
            Ks_inv += q[i, layer] / (VG_PARAMS[i].Ks + 1e-30)
    alpha = alpha_prod
    n = n_prod
    Ks = 1.0 / (Ks_inv + 1e-30) if Ks_inv > 0 else VG_PARAMS[0].Ks
    return VgParams(theta_r=theta_r, theta_s=theta_s, alpha=alpha, n=n, Ks=Ks)
