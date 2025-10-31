# ruff: noqa: D104 F401
import logging

from pastastore.extensions.accessor import (
    register_pastastore_accessor as register_pastastore_accessor,
)

logger = logging.getLogger(__name__)


def activate_hydropandas_extension():
    """Register HydroPandas extension for downloading time series data."""
    from pastastore.extensions.hpd import HydroPandasExtension as _

    logger.info(
        "Registered HydroPandas extension in PastaStore, "
        "e.g. `pstore.hpd.download_bro_gmw()`."
    )
