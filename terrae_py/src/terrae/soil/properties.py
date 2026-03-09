"""
Soil hydraulic properties using van Genuchten–Mualem.

Replaces legacy GHY.f cubic θ(h) with standard van Genuchten formulation.
Parameters from ROSETTA pedotransfer (Schaap et al. 2001).

References:
- van Genuchten (1980) Soil Sci. Soc. Am. J. 44:892–898
- Schaap et al. (2001) J. Hydrol. 251:163–176
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from terrae import types
from terrae.soil.hydraulic import VG_PARAMS

# Specific heat capacity of soil texture (J/K/m³), imt textures
# Sand, Silt, Clay, Peat, Loam
SHC_SOIL_TEXTURE = np.array([2e6, 2e6, 2e6, 2.5e6, 2.4e6])


def get_soil_properties(
    q: NDArray[np.float64],
    dz: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """
    Compute saturated/minimum theta and heat capacity per layer.

    Uses van Genuchten parameters (θs, θr) weighted by texture fraction.
    Compatible with TerraE driver interface.

    Args:
        q: Texture fractions (imt, ngm), q[i,k] = fraction of texture i in layer k
        dz: Layer thickness (ngm), m

    Returns:
        thets: Saturated theta per layer (ngm)
        thetm: Residual (minimum) theta per layer (ngm)
        shc: Heat capacity per layer (ngm), J/K/m²
    """
    n = 0
    for k in range(types.NGM):
        if dz[k] <= 0:
            break
        n = k + 1

    thets = np.zeros(types.NGM)
    thetm = np.zeros(types.NGM)
    shc = np.zeros(types.NGM)

    for k in range(n):
        if dz[k] <= 0:
            break
        # Weighted θs and θr from van Genuchten params
        for i in range(types.IMT):
            thets[k] += q[i, k] * VG_PARAMS[i].theta_s
            thetm[k] += q[i, k] * VG_PARAMS[i].theta_r
        # Heat capacity: (1 - porosity) * mineral heat cap * thickness
        for i in range(types.IMT):
            shc[k] += q[i, k] * SHC_SOIL_TEXTURE[i]
        shc[k] = (1.0 - thets[k]) * shc[k] * dz[k]

    return thets, thetm, shc
