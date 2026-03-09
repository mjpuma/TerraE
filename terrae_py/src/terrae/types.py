"""
Data structures and parameters for TerraE land model.

Matches giss_LSM/GHY_H.f for compatibility with Fortran reference.
Supports arbitrary land cover columns within a grid cell.
"""

from __future__ import annotations

from dataclasses import dataclass

# ----- Grid parameters (from ghy_h) -----
# Number of soil layers
NGM: int = 6

# Number of soil texture classes (sand, silt, clay, peat, +1)
IMT: int = 5

# Number of snow layers
NLSN: int = 3

# Land surface fractions: 1=bare, 2=vegetated, 3=reserved (legacy GHY)
LS_NFRAC: int = 3

# Layer index for bare soil
I_BARE: int = 1

# Layer index for vegetated soil
I_VEGE: int = 2

# ----- Land cover columns (within grid cell) -----
# Default column types; can be extended via LAND_COVER_TYPES
LAND_COVER_TYPES: tuple[str, ...] = (
    "tree",
    "grass",
    "urban",
    "suburban",
    "rainfed_ag",
    "irrigated_ag",
)
N_COLS: int = len(LAND_COVER_TYPES)

# Column indices for default types
I_TREE: int = 0
I_GRASS: int = 1
I_URBAN: int = 2
I_SUBURBAN: int = 3
I_RAINFED_AG: int = 4
I_IRRIGATED_AG: int = 5


def validate_fractions(f: tuple[float, ...] | list[float], n: int | None = None) -> None:
    """Raise if fractions don't sum to 1 or have wrong length."""
    f = list(f)
    if n is not None and len(f) != n:
        raise ValueError(f"Expected {n} fractions, got {len(f)}")
    if abs(sum(f) - 1.0) > 1e-10:
        raise ValueError(f"Fractions must sum to 1, got {sum(f)}")
    if any(x < -1e-12 for x in f):
        raise ValueError("Fractions must be non-negative")


@dataclass
class SoilState:
    """Prognostic soil state for one grid cell (bare + vegetated)."""

    # Water content w(k, ibv) in m, k=0..NGM (0=canopy for vegetated)
    w: tuple[tuple[float, ...], ...]  # shape (NGM+1, LS_NFRAC)

    # Heat content ht(k, ibv) in J/m²
    ht: tuple[tuple[float, ...], ...]  # shape (NGM+1, LS_NFRAC)

    # Layer thickness dz(k) in m
    dz: tuple[float, ...]  # shape (NGM,)


@dataclass
class SoilProperties:
    """Soil texture and hydraulic properties."""

    # Texture fractions q(imt, ngm) - sand, silt, clay, peat, etc.
    q: tuple[tuple[float, ...], ...]  # shape (IMT, NGM)

    # Hydraulic conductivity qk(imt, ngm)
    qk: tuple[tuple[float, ...], ...]  # shape (IMT, NGM)

    # Slope (for runoff)
    sl: float


@dataclass
class LandForcing:
    """Atmospheric forcing for land model."""

    pr: float  # Precipitation (m/s)
    htpr: float  # Precipitation heat flux (W/m²)
    prs: float  # Snow precipitation (m/s)
    htprs: float  # Snow precipitation heat
    srht: float  # Shortwave radiation (W/m²)
    trht: float  # Longwave radiation (W/m²)
    ts: float  # Surface air temperature (K)
    qs: float  # Surface specific humidity (kg/kg)
    pres: float  # Surface pressure (Pa)
    rho: float  # Air density (kg/m³)
    ch: float  # Bulk transfer coefficient * wind speed (m/s)
