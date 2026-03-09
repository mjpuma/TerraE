"""
Soil hydrology: reth, hydra, fl, runoff, fllmt, ImplicitRichards.

Explicit Richards (fl, fllmt) + mass-conservative implicit (ImplicitRichards).
Bare-soil only (no canopy, no snow) for Phase 1.

References:
  - Celia et al. (1990) Water Resour. Res. 26:1483–1496 — Modified Picard, mixed-form
  - Zeng & Decker (2009) J. Hydrometeor. 10:308–319 — Hydrostatic equilibrium fix
  - Oleson et al. (2020) CLM5 Technical Note — Variable layer thickness
  - TerraE Technical Description v2 — Sign conventions, Darcy discretization (Appendix B)
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from terrae import constants, types
from terrae.soil.hydraulic import (
    D_from_theta,
    K_from_theta,
    VG_PARAMS,
    h_from_theta,
    theta_from_h,
    weighted_params,
)

# ----- SIGN CONVENTION (TerraE Technical Description, Appendix B) -----
# zc:    elevation positive UPWARD, z=0 at surface → zc[k] < 0 for soil layers
# psi:   matric potential only, psi < 0 unsaturated (NOT total head)
# Total head: h[k] = psi[k] + zc[k]
# Darcy flux: F = -K * (H(l) - H(l-1)) / (Z(l) - Z(l-1))  [Igor Aleinov, GHY]
#   → Flux positive = UPWARD (TerraE convention)
#   → f(k) = -xk(k)*(h(k-1)-h(k))/(zc(k-1)-zc(k)) in fl()
# ImplicitRichards uses flux positive = DOWNWARD (Celia/Richards convention) internally;
# flux_top and flux_bottom are converted at BC application.


def reth(
    w: NDArray[np.float64],
    dz: NDArray[np.float64],
    n: int,
) -> NDArray[np.float64]:
    """
    Compute theta (volumetric water content) from water depth.

    theta(k) = w(k) / dz(k)
    """
    theta = np.zeros_like(w)
    for k in range(n):
        if dz[k] > 0:
            theta[k] = w[k] / dz[k]
    return theta


def hydra(
    theta: NDArray[np.float64],
    thets: NDArray[np.float64],
    thetm: NDArray[np.float64],
    fice: NDArray[np.float64],
    zc: NDArray[np.float64],
    q: NDArray[np.float64],  # (imt, ngm)
    qk: NDArray[np.float64],  # (imt, ngm)
    n: int,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """
    Compute matric potential h, conductivity xk, diffusivity d for mixed soil.

    Uses van Genuchten for each layer (weighted params). Adds gravitational
    potential. Geometric mean for inter-layer K.
    """
    h = np.zeros(n + 1)
    xk = np.zeros(n + 1)
    d = np.zeros(n)
    xku = np.zeros(n + 1)

    XKUD = 2.78e-5  # Surface conductance factor (m/s)

    for k in range(n):
        th = np.clip(theta[k], thetm[k] + 1e-12, thets[k] - 1e-12)
        params = weighted_params(q, k)
        h[k] = h_from_theta(th, params)
        xku[k] = K_from_theta(th, params) * (1.0 - fice[k])
        d[k] = D_from_theta(th, params) * (1.0 - fice[k])

    xk[n] = 0.0

    # Top layer: special treatment for surface conductance
    if n >= 1:
        xk1 = sum(
            (qk[i, 0] if i < qk.shape[0] else 1.0) * K_from_theta(thets[0], VG_PARAMS[i])
            for i in range(types.IMT - 1)
        )
        xkl = xk1 / (1.0 + xk1 / (abs(zc[0]) * XKUD + 1e-30))
        xkl = xkl * (1.0 - fice[0] * theta[0] / (thets[0] + 1e-12))
        xk[1] = max(0.0, xkl)

    for k in range(2, n + 1):
        xk[k] = np.sqrt(xku[k - 1] * xku[k - 2])

    # Add gravitational potential
    for k in range(n):
        h[k] = h[k] + zc[k]

    return h, xk, d, xku


def fl(
    h: NDArray[np.float64],
    xk: NDArray[np.float64],
    zc: NDArray[np.float64],
    n: int,
) -> tuple[NDArray[np.float64], float]:
    """
    Darcy flux between layers. f[k] = flux at bottom of layer k-1 (top of layer k).
    Positive = upward. f[0] = surface (set by driver), f[n] = bottom (set by driver).
    Driver sets f[n] = -xku[n-1] for unit gradient free drainage (matches implicit BC).
    GHY: f(k) = -xk(k)*(h(k-1)-h(k))/(zc(k-1)-zc(k)) for k=2..n.
    """
    f = np.zeros(n + 1)
    f[n] = 0.0  # Overwritten by driver with -xku[n-1] for unit gradient BC
    # f[k] = flux at interface between layer k-1 and k; xk[k+1] = conductivity there
    for k in range(1, n):
        dz = zc[k - 1] - zc[k]
        if abs(dz) > 1e-30:
            f[k] = -xk[k + 1] * (h[k - 1] - h[k]) / dz

    xinfc = xk[1] * abs(h[0]) / (abs(zc[0]) + 1e-30) if n >= 1 else 0.0
    return f, xinfc


def runoff(
    w: NDArray[np.float64],
    ws: NDArray[np.float64],
    f: NDArray[np.float64],
    xinfc: float,
    xku: NDArray[np.float64],
    dz: NDArray[np.float64],
    sl: float,
    n: int,
    prfr: float = 0.2,
) -> tuple[float, NDArray[np.float64]]:
    """
    Surface and subsurface runoff.

    Simplified: satfrac = (w/ws)^rosmp, rnf = satfrac * max(-f(0),0).
    f[0] = surface flux (positive upward); water_down = downward flux.
    rnff(k) = xku(k) * sl * dz(k) / sdstnc.
    """
    ROSMP = 8.0
    SDSTNC = 100.0

    if ws[0] > 1e-16:
        satfrac = min((w[0] / ws[0]) ** ROSMP, 0.6)
    else:
        satfrac = 0.0
    water_down = max(-f[0], 0.0)
    rnf = satfrac * water_down
    water_down = (1.0 - satfrac) * water_down
    if water_down * 30.0 > xinfc * prfr:
        rnf += water_down * np.exp(-xinfc * prfr / (water_down + 1e-30))

    rnff = np.zeros(n + 1)
    for k in range(n):
        rnff[k] = xku[k] * sl * dz[k] / SDSTNC

    return rnf, rnff


# -----------------------------------------------------------------------------
# ImplicitRichards — mass-conservative implicit solver (replaces fl + fllmt)
# -----------------------------------------------------------------------------


class ImplicitRichards:
    """
    Mass-conservative implicit Richards solver.

    Numerical scheme: Celia et al. (1990) modified Picard iteration.
    Hydrostatic equilibrium fix: Zeng & Decker (2009) for shallow water table.
    Variable layer geometry: Oleson et al. (2020) CLM5 Technical Note.

    Replaces explicit fl() + fllmt(). Preserves hydra(), runoff(), reth().
    """

    def __init__(self, dz: NDArray[np.float64], zc: NDArray[np.float64],
                 q: NDArray[np.float64], n: int) -> None:
        """
        Args:
            dz:  Layer thicknesses [m], shape (n,), geometric progression
            zc:  Node elevations [m], shape (n,), negative (upward positive)
            q:   Texture fractions (imt, ngm)
            n:   Number of active layers
        Precompute all geometry here — never inside timestep loop.
        """
        self.dz = dz[:n].copy()
        self.zc = zc[:n].copy()
        self.q = q
        self.n = n

        # Inter-node distances — Celia/Puma SoilWaterFlow MATRIX.FOR
        # dz_half[k] = (dz[k-1]+dz[k])/2 = distance for interface between k-1 and k
        self.dz_half = np.zeros(n + 1)
        self.dz_half[0] = self.dz[0]  # surface
        for k in range(1, n):
            self.dz_half[k] = 0.5 * (self.dz[k - 1] + self.dz[k])
        self.dz_half[n] = self.dz[n - 1]  # bottom

        # Cache VG params per layer — weighted_params is slow
        self._vg = [weighted_params(q, k) for k in range(n)]

    # ----------------------------------------------------------------
    # Van Genuchten functions (analytical, vectorized over scalar psi)
    # ----------------------------------------------------------------

    def _theta(self, psi: float, k: int) -> float:
        """theta(psi) via existing theta_from_h."""
        return float(theta_from_h(np.array([psi]), self._vg[k])[0])

    def _C(self, psi: float, k: int) -> float:
        """
        Specific moisture capacity C = dtheta/dpsi [1/m].
        Analytical derivative — Celia (1990) eq. 5.
        C = (theta_s-theta_r)*m*n*alpha*(alpha|psi|)^(n-1)
            / (1+(alpha|psi|)^n)^(m+1)    for psi < 0
        C = 0                              for psi >= 0
        """
        if psi >= 0:
            return 0.0
        vg = self._vg[k]
        ap = vg.alpha * abs(psi)
        apn = ap ** vg.n
        return ((vg.theta_s - vg.theta_r) * vg.m * vg.n
                * vg.alpha * (ap ** (vg.n - 1.0))
                / (1.0 + apn) ** (vg.m + 1.0))

    def _K(self, psi: float, k: int) -> float:
        """K(psi) via existing K_from_theta."""
        theta = self._theta(psi, k)
        return float(K_from_theta(theta, self._vg[k]))

    def _K_interface(self, psi: NDArray[np.float64], k: int) -> float:
        """
        Geometric mean inter-node K at interface k | k+1.
        Oleson CLM5 Tech Note — preferred at sharp wetting fronts.
        Consistent with geometric mean already used in hydra().
        """
        return np.sqrt(self._K(psi[k], k) * self._K(psi[k + 1], k + 1))

    # ----------------------------------------------------------------
    # Zeng & Decker (2009) equilibrium state
    # ----------------------------------------------------------------

    def _equilibrium(self, z_wt: float) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """
        Hydrostatic equilibrium matric head and theta profiles.
        Zeng & Decker (2009): psi_E[k] = z_wt - zc[k]
        Above water table: psi_E < 0 (unsaturated, correct)
        At/below water table: psi_E >= 0 → clip to 0 (saturated)

        Args:
            z_wt: Water table elevation [m], negative (upward positive)
        Returns:
            psi_E:   shape (n,)
            theta_E: shape (n,)
        """
        psi_E = np.minimum(z_wt - self.zc, 0.0)  # clip at saturation
        theta_E = np.array([self._theta(float(psi_E[k]), k) for k in range(self.n)])
        return psi_E, theta_E

    # ----------------------------------------------------------------
    # Tridiagonal system (Celia 1990 + Zeng & Decker 2009)
    # ----------------------------------------------------------------

    def _build_tridiagonal(
        self,
        psi_j: NDArray[np.float64],
        theta_old: NDArray[np.float64],
        dt: float,
        flux_top: float,
        S: NDArray[np.float64],
        z_wt: float | None = None,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
        """
        Build tridiagonal system A*psi = rhs for Picard iterate j.

        Aligned with Celia/Puma SoilWaterFlow MATRIX.FOR (block-centered finite diff).
        Flux: q = K*(dψ/dz - 1), diff = K/dz_half, flux = diff*(Δψ - dz_half).

        Args:
            psi_j:     Current Picard iterate psi^{m+1,j}, shape (n,)
            theta_old: theta^m, shape (n,)
            dt:        Timestep [s]
            flux_top:  Infiltration rate [m/s], positive downward into soil
            S:         Sink per layer [m/s] (evap, root uptake placeholder)
            z_wt:      Water table elevation [m] or None (deep/free drainage)
        """
        n = self.n
        lower = np.zeros(n)
        main = np.zeros(n)
        upper = np.zeros(n)
        rhs = np.zeros(n)

        theta_j = np.array([self._theta(float(psi_j[k]), k) for k in range(n)])
        C_j = np.array([self._C(float(psi_j[k]), k) for k in range(n)])

        # Interface conductivities — geometric mean (Celia uses harmonic; geometric OK)
        K_int = np.zeros(n + 1)
        K_int[0] = self._K(float(psi_j[0]), 0)
        for k in range(n - 1):
            K_int[k + 1] = self._K_interface(psi_j, k)
        K_int[n] = self._K(float(psi_j[n - 1]), n - 1)

        # Bottom BC flux
        if z_wt is not None:
            flux_bottom = 0.0  # Zeng & Decker: zero flux at equilibrium
        else:
            flux_bottom = K_int[n]  # Unit gradient free drainage

        # Interior and boundary blocks — MATRIX.FOR structure
        for k in range(n):
            dz1 = self.dz_half[k]      # interface above (between k-1 and k)
            dz2 = self.dz_half[k + 1]  # interface below (between k and k+1)

            diff1 = K_int[k] / (dz1 + 1e-30) if k > 0 else 0.0
            diff2 = K_int[k + 1] / (dz2 + 1e-30) if k < n - 1 else 0.0

            # Main diagonal: dz/dt*C + diff1 + diff2 (Celia eq.)
            # Top block: no diff1 (nbct=2 flux BC); bottom block: no diff2 (nbcb=2 flux BC)
            main[k] = self.dz[k] * C_j[k] / dt + diff1 + diff2
            lower[k] = -diff1 if k > 0 else 0.0
            upper[k] = -diff2 if k < n - 1 else 0.0

            # RHS: dz*C/dt*ψ_j - dz/dt*(θ_j - θ_old) + diff1*dz1 - diff2*dz2 + S*dz
            # (Celia linearization: LHS has flux terms in ψ; RHS has known terms)
            moisture = self.dz[k] * C_j[k] / dt * psi_j[k] - self.dz[k] / dt * (theta_j[k] - theta_old[k])
            flux_const = diff1 * dz1 - diff2 * dz2
            rhs[k] = moisture + flux_const + S[k] * self.dz[k]

        # Top BC: flux_top into domain (positive downward) → add to RHS
        # (flux_in = flux_top increases storage; larger RHS → larger psi[0] → more water)
        rhs[0] += flux_top
        # Bottom BC: flux_bottom out of domain → subtract from RHS
        # (drives flux_out = flux_bottom; subtract to remove water from bottom cell)
        rhs[n - 1] -= flux_bottom

        return lower, main, upper, rhs

    # ----------------------------------------------------------------
    # Thomas algorithm — O(N), no numpy.linalg.solve
    # ----------------------------------------------------------------

    def _thomas(
        self,
        lower: NDArray[np.float64],
        main: NDArray[np.float64],
        upper: NDArray[np.float64],
        rhs: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """
        Thomas algorithm for tridiagonal system. O(N).
        lower[0] and upper[n-1] unused (boundary nodes).
        """
        n = len(rhs)
        c_ = np.zeros(n)
        d_ = np.zeros(n)
        x = np.zeros(n)

        c_[0] = upper[0] / (main[0] + 1e-30)
        d_[0] = rhs[0] / (main[0] + 1e-30)
        for k in range(1, n):
            denom = main[k] - lower[k] * c_[k - 1]
            c_[k] = upper[k] / (denom + 1e-30)
            d_[k] = (rhs[k] - lower[k] * d_[k - 1]) / (denom + 1e-30)

        x[n - 1] = d_[n - 1]
        for k in range(n - 2, -1, -1):
            x[k] = d_[k] - c_[k] * x[k + 1]
        return x

    # ----------------------------------------------------------------
    # Main solve — Celia (1990) Picard + Zeng & Decker (2009) BC
    # ----------------------------------------------------------------

    def solve(
        self,
        psi0: NDArray[np.float64],
        theta_old: NDArray[np.float64],
        dt: float,
        flux_top: float,
        S: NDArray[np.float64],
        z_wt: float | None = None,
        tol_psi: float = 1e-4,
        tol_theta: float = 1e-6,
        max_iter: int = 20,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], dict]:
        """
        Modified Picard iteration — Celia et al. (1990).
        Adaptive dt on non-convergence.

        Args:
            psi0:      Initial guess psi^m [m]
            theta_old: theta^m [m³/m³]
            dt:        Timestep [s]
            flux_top:  Surface infiltration [m/s], positive into soil
            S:         Sink per layer [m/s]
            z_wt:      Water table elevation [m] or None
                       None  → unit gradient BC (urban, bare/desert)
                       float → Zeng & Decker equilibrium BC (ag, suburban)

        Returns:
            psi_new, theta_new, diagnostics dict
        """
        psi_j = psi0.copy()

        for j in range(max_iter):
            lower, main, upper, rhs = self._build_tridiagonal(
                psi_j, theta_old, dt, flux_top, S, z_wt
            )
            psi_sol = self._thomas(lower, main, upper, rhs)
            # Under-relaxation for stability when Picard oscillates
            alpha = 0.7
            psi_new = psi_j + alpha * (psi_sol - psi_j)
            # Clamp psi to unsaturated range (avoid saturated blowup)
            psi_new = np.clip(psi_new, -1e3, 0.0)
            delta = psi_new - psi_j
            theta_new = np.array([self._theta(float(psi_new[k]), k)
                                 for k in range(self.n)])
            theta_prev = np.array([self._theta(float(psi_j[k]), k)
                                  for k in range(self.n)])

            # Celia (1990) section 4 convergence criteria — both must pass
            if (np.max(np.abs(delta)) < tol_psi and
                    np.max(np.abs(theta_new - theta_prev)) < tol_theta):
                # Bottom flux: unit gradient (free drainage) or zero (z_wt)
                K_bot = self._K(float(psi_new[-1]), self.n - 1)
                if z_wt is not None:
                    deep_perc = 0.0  # near zero at equilibrium
                else:
                    deep_perc = K_bot * dt  # unit gradient

                return psi_new, theta_new, {
                    'converged': True,
                    'picard_iterations': j + 1,
                    'deep_percolation': deep_perc,
                    'psi': psi_new,
                    'theta': theta_new,
                    'mass_residual': None,  # filled by mass_balance_check
                }

            psi_j = psi_new

        # Non-convergence: halve dt and retry (adaptive stepping)
        if dt > 60.0:  # 1-minute floor
            return self.solve(psi0, theta_old, dt / 2.0,
                             flux_top, S, z_wt,
                             tol_psi, tol_theta, max_iter)

        # At dt floor: accept with warning flag
        K_bot = self._K(float(psi_j[-1]), self.n - 1)
        return psi_j, np.array([self._theta(float(psi_j[k]), k)
                               for k in range(self.n)]), {
            'converged': False,
            'picard_iterations': max_iter,
            'deep_percolation': K_bot * dt if z_wt is None else 0.0,
            'psi': psi_j,
            'theta': np.array([self._theta(float(psi_j[k]), k)
                              for k in range(self.n)]),
            'mass_residual': None,
        }

    # ----------------------------------------------------------------
    # Mass balance — run every timestep, non-negotiable
    # ----------------------------------------------------------------

    def mass_balance_check(
        self,
        theta_old: NDArray[np.float64],
        theta_new: NDArray[np.float64],
        dt: float,
        flux_top: float,
        deep_perc: float,
        S: NDArray[np.float64],
    ) -> float:
        """
        Water balance closure per timestep.
        Tolerance: 1e-4 m (0.1 mm). Failure = bug in Picard loop.
        Tighten to 1e-6 once formulation validated.

        If variable thickness dz vs dz_int were conflated in
        _build_tridiagonal, this will catch it immediately.
        """
        delta_S = np.sum(self.dz * (theta_new - theta_old))
        inflow = flux_top * dt
        outflow = deep_perc
        sink_total = np.sum(S * self.dz) * dt
        residual = inflow - outflow - sink_total - delta_S
        assert abs(residual) < 1e-4, (
            f"Mass balance FAILED: residual={residual:.2e} m  "
            f"inflow={inflow:.4f}  outflow={outflow:.4f}  "
            f"dS={delta_S:.4f}  sink={sink_total:.4f}"
        )
        return residual


def fllmt(
    w: NDArray[np.float64],
    ws: NDArray[np.float64],
    thetm: NDArray[np.float64],
    dz: NDArray[np.float64],
    f: NDArray[np.float64],
    rnf: float,
    rnff: NDArray[np.float64],
    evapdl: NDArray[np.float64],
    dts: float,
    n: int,
    fd: float = 1.0,
) -> float:
    """
    # LEGACY EXPLICIT — Limit fluxes to prevent over/undersaturation.
    Modifies f, rnff in place. Returns updated rnf.
    Replaced by ImplicitRichards for mass-conservative implicit solve.
    """
    trunc = 0.0
    evap = evapdl * fd

    for k in range(n - 1, 1, -1):
        wn = w[k] + (f[k + 1] - f[k] - rnff[k] - evap[k]) * dts
        if wn - ws[k] > trunc:
            f[k] = f[k] + (wn - ws[k] + trunc) / dts
        if wn - dz[k] * thetm[k] < trunc:
            rnff[k] = rnff[k] + (wn - dz[k] * thetm[k] - trunc) / dts
            if rnff[k] < 0:
                f[k] = f[k] + rnff[k]
                rnff[k] = 0

    wn = w[0] + (f[1] - f[0] - rnf - rnff[0] - evap[0]) * dts
    if wn - ws[0] > trunc:
        rnf = rnf + (wn - ws[0] + trunc) / dts
    if wn - dz[0] * thetm[0] < trunc:
        rnf = rnf + (wn - dz[0] * thetm[0] - trunc) / dts

    return max(0.0, rnf)
