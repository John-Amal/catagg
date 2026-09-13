"""Checks that the distribution is assembled correctly.

These are not smoke tests: each one fails on a specific, real packaging
mistake that is otherwise silent until a user hits it.
"""

from __future__ import annotations

import re
from importlib.metadata import version as dist_version
from importlib.resources import files

import catagg

PEP440 = re.compile(r"^\d+\.\d+\.\d+(?:(?:a|b|rc)\d+)?(?:\.post\d+)?(?:\.dev\d+)?$")


def test_version_is_pep440() -> None:
    assert PEP440.match(catagg.__version__), catagg.__version__


def test_module_version_matches_distribution_metadata() -> None:
    """The single source of truth must survive the build.

    Fails if ``__init__.__version__`` and the built metadata drift apart --
    the usual cause of a release tagged with one number and published under
    another.
    """
    assert catagg.__version__ == dist_version("catagg")


def test_py_typed_marker_is_installed() -> None:
    """Without this file downstream type checkers silently ignore the package."""
    assert files("catagg").joinpath("py.typed").is_file()
