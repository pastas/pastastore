import pastas as ps
from packaging.version import parse as parse_version

PASTAS_VERSION = parse_version(ps.__version__)
PASTAS_LEQ_022 = PASTAS_VERSION <= parse_version("0.22.0")

__version__ = "1.4.0"
