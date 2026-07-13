"""Package-level sanity tests for the Phase 0 skeleton."""

import meridian


def test_version_is_semver() -> None:
    major, minor, patch = meridian.__version__.split(".")
    assert major.isdigit() and minor.isdigit() and patch.isdigit()


def test_version_matches_phase_zero() -> None:
    assert meridian.__version__ == "0.0.1"
