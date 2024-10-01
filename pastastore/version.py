# ruff: noqa: D100
from importlib import import_module, metadata
from platform import python_version

import pastas as ps
from packaging.version import parse as parse_version

PASTAS_VERSION = parse_version(ps.__version__)
PASTAS_LEQ_022 = PASTAS_VERSION <= parse_version("0.22.0")
PASTAS_GEQ_150 = PASTAS_VERSION >= parse_version("1.5.0")

__version__ = "1.7.0"


def show_versions(optional=False) -> None:
    """Print the version of dependencies.

    Parameters
    ----------
    optional : bool, optional
        Print the version of optional dependencies, by default False
    """
    msg = (
        f"Pastastore version : {__version__}\n\n"
        f"Python version     : {python_version()}\n"
        f"Pandas version     : {metadata.version('pandas')}\n"
        f"Matplotlib version : {metadata.version('matplotlib')}\n"
        f"Pastas version     : {metadata.version('pastas')}\n"
        f"PyYAML version     : {metadata.version('pyyaml')}\n"
    )
    if optional:
        msg += "\nArcticDB version   : "
        try:
            import_module("arcticdb")
            msg += f"{metadata.version('arctidb')}"
        except ImportError:
            msg += "Not Installed"

    print(msg)
