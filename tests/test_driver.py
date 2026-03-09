"""Tests for driver and hydrology (Steps 3-10)."""

import numpy as np
import pytest

from terrae import constants, types
from terrae.driver import advance_bare_soil, advance_cell, check_energy, check_water, init_step
from terrae.soil.hydrology import fl, hydra, reth, runoff
from terrae.soil.properties import get_soil_properties


def test_reth() -> None:
    """reth computes theta from w/dz."""
    from terrae.soil.hydrology import reth

    w = np.array([0.05, 0.1, 0.15, 0.2, 0.2, 0.3])
    dz = np.array([0.1, 0.2, 0.3, 0.5, 0.5, 1.0])
    theta = reth(w, dz, 6)
    assert theta[0] == pytest.approx(0.5)
    assert theta[1] == pytest.approx(0.5)


def test_init_step() -> None:
    """init_step returns valid geometry."""
    q = np.zeros((types.IMT, types.NGM))
    q[0, :] = 1.0
    dz = np.array([0.1, 0.2, 0.3, 0.5, 0.5, 1.0])
    n, zb, zc, thets, thetm, shc, ws = init_step(dz, q)
    assert n == 6
    assert zb[0] == 0
    assert zb[6] == pytest.approx(-2.6)
    assert thets[0] == pytest.approx(0.43)


def test_advance_bare_soil_single_step() -> None:
    """advance_bare_soil runs without error."""
    q = np.zeros((types.IMT, types.NGM))
    q[0, :] = 1.0
    qk = np.ones((types.IMT, types.NGM))
    dz = np.array([0.1, 0.2, 0.3, 0.5, 0.5, 1.0])
    w = np.array([0.03, 0.06, 0.09, 0.12, 0.12, 0.18])
    ht = np.zeros(types.NGM + 1)
    ht[:6] = 1e4

    w_out, ht_out, run, evap, _, _ = advance_bare_soil(
        w.copy(),
        ht.copy(),
        dz,
        q,
        qk,
        sl=0.01,
        pr=1e-7,
        htpr=0.0,
        srht=100.0,
        trht=300.0,
        ts=288.0,
        rho=1.2,
        ch=0.01,
        vs=5.0,
        dt=3600.0,
    )

    assert np.all(w_out >= 0)
    assert np.all(w_out <= 0.5)


def test_check_water_identity() -> None:
    """check_water returns 0 when conserved."""
    w_before = 0.5
    w_after = np.array([0.5])
    err = check_water(w_before, w_after, pr=0.0, evap=0.0, rnf=0.0, rnff=np.array([0.0]), dts=1.0, n=1)
    assert abs(err) < 1e-12


def test_water_balance_single_step() -> None:
    """Water balance for one GCM step with no precip, zero slope (no runoff)."""
    q = np.zeros((types.IMT, types.NGM))
    q[0, :] = 1.0
    qk = np.ones((types.IMT, types.NGM))
    dz = np.array([0.05, 0.1, 0.2, 0.4, 0.8, 1.6])
    w = np.array([0.015, 0.03, 0.06, 0.1, 0.15, 0.25])
    ht = np.zeros(types.NGM + 1)
    ht[:6] = 1e5

    w_init = np.sum(w[:6])
    w_out, ht_out, run, evap, _, deep_perc = advance_bare_soil(
        w.copy(), ht.copy(), dz, q, qk,
        sl=0.0, pr=0.0, htpr=0.0, srht=200.0, trht=300.0,
        ts=288.0, rho=1.2, ch=0.01, vs=4.0, dt=3600.0,
    )
    w_final = np.sum(w_out[:6])
    err = w_final - w_init - (0.0 - evap * 3600 - run - deep_perc)
    assert abs(err) < 1e-12, f"Water balance error {err:.2e} m (must be < 1e-12)"


def test_water_balance_machine_precision() -> None:
    """Water balance must hold to machine precision over a full run."""
    q = np.zeros((types.IMT, types.NGM))
    q[0, :] = 1.0
    qk = np.ones((types.IMT, types.NGM))
    dz = np.array([0.05, 0.1, 0.2, 0.4, 0.8, 1.6])
    w = np.array([0.015, 0.03, 0.06, 0.1, 0.15, 0.25])
    ht = np.zeros(types.NGM + 1)
    ht[:6] = 1e5

    w_init = np.sum(w[:6])
    total_pr = 0.0
    total_evap = 0.0
    total_runoff = 0.0
    total_deep_perc = 0.0

    dt_sec = 3600.0
    n_steps = 24 * 3  # 3 days

    for step in range(n_steps):
        hour = step % 24
        day = step // 24
        pr = 5e-3 / 7200.0 if (day in (0, 1, 2) and 6 <= hour <= 8) else 0.0
        htpr = pr * 4185 * 288 if pr > 0 else 0.0

        w, ht, run, evap, _, deep_perc = advance_bare_soil(
            w, ht, dz, q, qk,
            sl=0.02, pr=pr, htpr=htpr, srht=200.0, trht=300.0,
            ts=288.0, rho=1.2, ch=0.01, vs=4.0, dt=dt_sec,
        )

        total_pr += pr * dt_sec
        total_evap += evap * dt_sec
        total_runoff += run
        total_deep_perc += deep_perc

    w_final = np.sum(w[:6])
    delta_w = w_final - w_init
    flux_balance = total_pr - total_evap - total_runoff - total_deep_perc
    err = delta_w - flux_balance

    assert abs(err) < 1e-12, f"Water balance error {err:.2e} m (must be < 1e-12)"


def test_energy_balance_single_step() -> None:
    """Energy balance for one GCM step with no precip, zero slope."""
    q = np.zeros((types.IMT, types.NGM))
    q[0, :] = 1.0
    qk = np.ones((types.IMT, types.NGM))
    dz = np.array([0.05, 0.1, 0.2, 0.4, 0.8, 1.6])
    w = np.array([0.015, 0.03, 0.06, 0.1, 0.15, 0.25])
    ht = np.zeros(types.NGM + 1)
    ht[:6] = 1e5

    srht, trht = 200.0, 300.0
    e_before = np.sum(ht[:6])
    w_out, ht_out, run, evap, _, _ = advance_bare_soil(
        w.copy(), ht.copy(), dz, q, qk,
        sl=0.0, pr=0.0, htpr=0.0, srht=srht, trht=trht,
        ts=288.0, rho=1.2, ch=0.01, vs=4.0, dt=3600.0,
    )
    e_after = np.sum(ht_out[:6])
    from terrae.soil.heat import retp
    from terrae.soil.properties import get_soil_properties
    _, _, shc = get_soil_properties(q, dz)
    tp, _ = retp(w_out, ht_out, shc, 6)

    # Flux: htpr - ELH*evap - heat_in_runoff - srht - trht (no precip, evap; runoff≈0 with sl=0)
    # run is total runoff (m); check_energy expects rnf (m/s), so rnf = run/dt
    rnf = run / 3600.0 if run > 0 else 0.0
    err = check_energy(
        e_before, ht_out, htpr=0.0, evap=0.0, rnf=rnf, rnff=np.zeros(6),
        tp=tp, srht=srht, trht=trht, dts=3600.0, n=6,
    )
    assert abs(err) < 1.0, f"Energy balance error {err:.2e} W/m² (single step, sl=0)"


def test_energy_balance_machine_precision() -> None:
    """Energy balance must hold to machine precision over a full run."""
    from terrae.soil.heat import retp
    from terrae.soil.properties import get_soil_properties

    q = np.zeros((types.IMT, types.NGM))
    q[0, :] = 1.0
    qk = np.ones((types.IMT, types.NGM))
    dz = np.array([0.05, 0.1, 0.2, 0.4, 0.8, 1.6])
    w = np.array([0.015, 0.03, 0.06, 0.1, 0.15, 0.25])
    ht = np.zeros(types.NGM + 1)
    ht[:6] = 1e5

    _, _, shc = get_soil_properties(q, dz)
    e_init = np.sum(ht[:6])
    total_flux = 0.0

    dt_sec = 3600.0
    n_steps = 24 * 3
    srht, trht = 200.0, 300.0

    for step in range(n_steps):
        hour = step % 24
        day = step // 24
        pr = 5e-3 / 7200.0 if (day in (0, 1, 2) and 6 <= hour <= 8) else 0.0
        htpr = pr * 4185 * 288 if pr > 0 else 0.0

        w, ht, run, evap, heat_runoff, _ = advance_bare_soil(
            w, ht, dz, q, qk,
            sl=0.0, pr=pr, htpr=htpr, srht=srht, trht=trht,
            ts=288.0, rho=1.2, ch=0.01, vs=4.0, dt=dt_sec,
        )

        flux_in = (srht + trht - htpr) * dt_sec - constants.ELH * evap * dt_sec - heat_runoff
        total_flux += flux_in

    e_final = np.sum(ht[:6])
    err = (e_final - e_init) - total_flux
    # Float64 accumulation over 72 steps; relative error ~1e-13
    assert abs(err) < 1e-7, f"Energy balance error {err:.2e} J/m² (must be < 1e-7)"


def test_advance_cell_multi_column() -> None:
    """advance_cell runs with multiple columns and area-weighted aggregation."""
    n_cols = types.N_COLS
    fractions = np.ones(n_cols) / n_cols  # equal fractions
    dz = np.array([0.05, 0.1, 0.2, 0.4, 0.8, 1.6])
    q = np.zeros((types.IMT, types.NGM))
    q[0, :] = 1.0
    qk = np.ones((types.IMT, types.NGM))
    sl = np.full(n_cols, 0.02)

    w = np.tile(np.array([0.015, 0.03, 0.06, 0.1, 0.15, 0.25])[:, np.newaxis], (1, n_cols))
    ht = np.zeros((types.NGM + 1, n_cols))
    ht[:6, :] = 1e5

    w_out, ht_out, run, evap, _ = advance_cell(
        w.copy(), ht.copy(), fractions, dz, q, qk, sl,
        pr=0.0, htpr=0.0, srht=200.0, trht=300.0, ts=288.0,
        rho=1.2, ch=0.01, vs=4.0, dt=3600.0,
    )

    assert w_out.shape == (6, n_cols)
    assert ht_out.shape == (7, n_cols)
    assert run >= 0
    assert evap >= 0
