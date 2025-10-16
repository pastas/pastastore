# ruff: noqa: F401 D104
import logging

from pastastore import connectors, styling, util
from pastastore.connectors import (
    ArcticDBConnector,
    DictConnector,
    PasConnector,
)
from pastastore.store import PastaStore
from pastastore.util import get_color_logger
from pastastore.version import __version__, show_versions

logger = get_color_logger("INFO", logger_name=__name__)
try:
    from pastastore import extensions
except ModuleNotFoundError:
    logging.warning("Could not import extensions module. Update pastas to >=1.3.0!")
