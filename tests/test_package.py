"""Package-level sanity tests."""

import tomllib
from pathlib import Path

import meridian


def test_version_is_semver() -> None:
    major, minor, patch = meridian.__version__.split(".")
    assert major.isdigit() and minor.isdigit() and patch.isdigit()


def test_version_matches_pyproject() -> None:
    """``meridian.__version__`` must track ``pyproject.toml``.

    Guards the drift that previously left the packaged version at 0.0.1 while the repo
    was tagged v0.10.0.
    """
    pyproject = Path(__file__).resolve().parents[1] / "pyproject.toml"
    declared = tomllib.loads(pyproject.read_text())["project"]["version"]
    assert meridian.__version__ == declared
