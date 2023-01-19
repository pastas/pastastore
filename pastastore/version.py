from importlib.metadata import version

from packaging.version import parse as parse_version

PASTAS_VERSION = version("pastas")
PASTAS_LEQ_021 = parse_version(PASTAS_VERSION) <= parse_version("0.21.0")

__version__ = "0.10b.0"
