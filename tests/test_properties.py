"""Step 2 validation: soil properties (van Genuchten + ROSETTA)."""

import numpy as np
import pytest

from terrae import types
from terrae.soil.hydraulic import (
    VG_PARAMS,
    D_from_theta,
    K_from_theta,
    h_from_theta,
    theta_from_h,
)
from terrae.soil.properties import get_soil_properties, SHC_SOIL_TEXTURE


def test_vg_params_count() -> None:
    """Five texture classes for compatibility with types.IMT."""
    assert len(VG_PARAMS) == types.IMT


def test_theta_from_h_saturation() -> None:
    """At h=0, theta = theta_s."""
    for params in VG_PARAMS:
        th = theta_from_h(0.0, params)
        assert th == pytest.approx(params.theta_s, rel=1e-10)


def test_theta_from_h_dry() -> None:
    """At very dry (large |h|), theta approaches theta_r."""
    for params in VG_PARAMS:
        th = theta_from_h(-100.0, params)
        assert th >= params.theta_r
        assert th < params.theta_s


def test_h_from_theta_inverse() -> None:
    """h_from_theta and theta_from_h are inverses."""
    params = VG_PARAMS[0]
    for th in [0.1, 0.2, 0.3, 0.4]:
        h = h_from_theta(th, params)
        th_back = theta_from_h(h, params)
        assert th_back == pytest.approx(th, rel=1e-6)


def test_K_from_theta_saturation() -> None:
    """At saturation, K = Ks."""
    for params in VG_PARAMS:
        K = K_from_theta(params.theta_s, params)
        assert K == pytest.approx(params.Ks, rel=1e-4)


def test_D_from_theta_positive() -> None:
    """Diffusivity is positive for valid theta."""
    params = VG_PARAMS[0]
    for th in [0.15, 0.25, 0.35]:
        D = D_from_theta(th, params)
        assert D > 0


def test_get_soil_properties_pure_sand() -> None:
    """get_soil_properties with 100% sand."""
    q = np.zeros((types.IMT, types.NGM))
    q[0, :] = 1.0
    dz = np.array([0.1, 0.2, 0.3, 0.5, 0.5, 1.0])

    thets, thetm, shc = get_soil_properties(q, dz)

    assert thets[0] == pytest.approx(0.43, rel=1e-10)  # Sand theta_s
    assert thetm[0] == pytest.approx(0.045, rel=1e-10)  # Sand theta_r
    assert shc[0] == pytest.approx(0.57 * 2e6 * 0.1, rel=1e-6)  # (1-0.43)*2e6*0.1


def test_get_soil_properties_mixed_texture() -> None:
    """get_soil_properties with 50% sand, 50% silt."""
    q = np.zeros((types.IMT, types.NGM))
    q[0, 0] = 0.5
    q[1, 0] = 0.5
    dz = np.array([0.1] + [0.0] * (types.NGM - 1))
    dz[1] = 0.2

    thets, thetm, shc = get_soil_properties(q, dz)

    expected_thets = 0.5 * 0.43 + 0.5 * 0.43  # Both 0.43
    assert thets[0] == pytest.approx(expected_thets, rel=1e-10)


def test_get_soil_properties_peat() -> None:
    """Peat has high theta_s."""
    q = np.zeros((types.IMT, types.NGM))
    q[3, 0] = 1.0  # Peat
    dz = np.array([0.1] + [0.0] * (types.NGM - 1))

    thets, thetm, shc = get_soil_properties(q, dz)

    assert thets[0] == pytest.approx(0.61, rel=1e-10)
    assert thetm[0] == pytest.approx(0.07, rel=1e-10)
