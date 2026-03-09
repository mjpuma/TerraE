"""
Physical constants for TerraE land model.

Values match NASA GISS ModelE Constants_mod.F90 for compatibility with
Fortran reference implementation.
"""

from __future__ import annotations

# ----- Radiation -----
# Stefan-Boltzmann constant (W/m^2 K^4), CODATA 2010
STBO: float = 5.67037321e-8

# ----- Thermodynamic -----
# Freezing point of water at 1 atm (K)
TF: float = 273.15

# Latent heat of evaporation at 0°C (J/kg)
LHE: float = 2.5e6

# Latent heat of melt at 0°C (J/kg)
LHM: float = 3.34e5

# Latent heat of sublimation at 0°C (J/kg)
LHS: float = LHE + LHM

# ----- Densities (kg/m³) -----
RHOW: float = 1e3  # Pure water
RHOWS: float = 1030.0  # Sea water
RHOI: float = 916.6  # Pure ice

# ----- Temperatures -----
# Freezing point (K) - alias for TF, used as tfrz in GHY.f
TFRZ: float = TF

# ----- Specific heats (J/kg/K) -----
SHW: float = 4185.0  # Water at 20°C
SHI: float = 2060.0  # Ice at 0°C

# Gas constant (J/K/mol)
GASC: float = 8.314510

# Molar mass of dry air (g/mol)
MAIR: float = 28.9655

# Gas constant for dry air (J/K/kg) = 1000 * GASC / MAIR
RGAS: float = 1e3 * GASC / MAIR  # ≈ 287.05

# Specific heat of dry air at constant pressure (J/kg/K) = RGAS / kapa
# kapa = (gamma - 1) / gamma, gamma = 1.401
KAPA: float = (1.401 - 1.0) / 1.401
SHA: float = RGAS / KAPA  # ≈ 1003.3

# ----- Derived for land model (per m³) -----
# Heat capacity of water (J/m³/K) = SHW * RHOW
SHW_VOL: float = SHW * RHOW

# Heat capacity of ice (J/m³/K) = SHI * RHOW
SHI_VOL: float = SHI * RHOW

# Latent heat of fusion per unit volume (J/m³) = LHM * RHOW
FSN: float = LHM * RHOW

# Latent heat of evaporation per unit volume (J/m³) = LHE * RHOW
ELH: float = LHE * RHOW

# ----- Gravity -----
GRAV: float = 9.80665

# ----- Numerical -----
ZERO: float = 0.0
ONE: float = 1.0
TEENY: float = 1e-30  # Avoid 0/0
UNDEF: float = -1e30
