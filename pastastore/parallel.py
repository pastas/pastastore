"""Importable worker helpers for parallel processing.

These functions are defined in a regular Python module so they remain
importable by multiprocessing workers started with ``spawn`` or
``forkserver`` on Python 3.14+.
"""

from typing import Any

import pastas as ps

import pastastore as pst
from pastastore import connectors as pst_connectors
from pastastore.base import BaseConnector


def _resolve_connector(connector: BaseConnector | None = None) -> BaseConnector:
    """Return the connector for the current worker context.

    Parameters
    ----------
    connector : BaseConnector | None, optional
        Picklable connector passed in by the caller. When omitted, the
        worker-local connector created by the multiprocessing initializer is
        used instead.

    Returns
    -------
    BaseConnector
        Connector bound to the appropriate worker context.
    """
    if connector is None:
        connector = pst_connectors.get_worker_connector()
    return connector


def model_statistic(
    model_name, statistic: str, connector: BaseConnector | None = None
) -> Any:
    """Compute a statistic of a Pastas model."""
    resolved_connector = _resolve_connector(connector)
    ml = resolved_connector.get_models(model_name)
    return getattr(ml.stats, statistic)()


def rsq(model_name: str, connector: BaseConnector | None = None) -> float:
    """Compute the R-squared value of a Pastas model."""
    return model_statistic(model_name, "rsq", connector)


def prediction_interval(
    model_name: str,
    connector: BaseConnector | None = None,
    **kwargs: Any,
):
    """Compute the prediction interval for a Pastas model."""
    resolved_connector = _resolve_connector(connector)
    ml = resolved_connector.get_models(model_name)
    return ml.solver.prediction_interval(**kwargs)


def two_step_solve(
    name: str,
    connector: BaseConnector | None = None,
    noisemodel=None,
    **kwargs: Any,
) -> None:
    """Solve a Pastas model in two steps and store the result."""
    resolved_connector = _resolve_connector(connector)
    pstore = pst.PastaStore(resolved_connector)
    ml = pstore.create_model(name)
    report = kwargs.pop("report", False)
    ml.solve(report=False, **kwargs)
    if noisemodel is not None:
        ml.add_noisemodel(noisemodel())
    else:
        # support for older pastas versions
        try:
            noisemodel = ps.ArNoiseModel
        except AttributeError:
            noisemodel = ps.NoiseModel
        ml.add_noisemodel(noisemodel())
    ml.solve(initial=False, report=report, **kwargs)
    resolved_connector.add_model(ml, overwrite=True)


def get_model(model_name: str, connector: BaseConnector | None = None):
    """Load a model from the active worker connector."""
    resolved_connector = _resolve_connector(connector)
    return resolved_connector.get_model(model_name)
