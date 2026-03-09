"""Step 1 validation: import test."""

import pytest


def test_import_terrae() -> None:
    """Terrae package imports successfully."""
    import terrae  # noqa: F401
    assert terrae.__version__ == "0.1.0"


def test_import_constants() -> None:
    """Constants module has expected values."""
    from terrae import constants

    assert constants.TF == 273.15
    assert constants.RHOW == 1000.0
    assert constants.LHE == 2.5e6
    assert constants.SHW_VOL == constants.SHW * constants.RHOW


def test_import_types() -> None:
    """Types module has expected parameters."""
    from terrae import types

    assert types.NGM == 6
    assert types.IMT == 5
    assert types.NLSN == 3
    assert types.LS_NFRAC == 3
