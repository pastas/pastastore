from importlib.metadata import version

from packaging.version import parse as parse_version

PASTAS_VERSION = version("pastas")
PASTAS_LEQ_022 = parse_version(PASTAS_VERSION) <= parse_version("0.22.0")

__version__ = "0.10b.0"
