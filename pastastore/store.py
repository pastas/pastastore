"""Module containing the PastaStore object for managing time series and models."""

import json
import logging
import os
import warnings
from functools import partial
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import pastas as ps
from pastas.io.pas import pastas_hook
from tqdm.auto import tqdm

from pastastore.base import BaseConnector
from pastastore.connectors import ArcticDBConnector, DictConnector, PasConnector
from pastastore.plotting import Maps, Plots
from pastastore.styling import boolean_styler
from pastastore.typing import FrameOrSeriesUnion, PastasLibs, TimeSeriesLibs
from pastastore.util import ZipUtils, _custom_warning
from pastastore.version import PASTAS_GEQ_150
from pastastore.yaml_interface import PastastoreYAML

warnings.showwarning = _custom_warning

logger = logging.getLogger(__name__)


class PastaStore:
    """PastaStore object for managing pastas time series and models.

    Requires a Connector object to provide the interface to
    the database. Different Connectors are available, e.g.:

    - PasConnector for storing all data as .pas (JSON) files on disk (recommended)
    - ArcticDBConnector for saving data on disk using arcticdb package
    - DictConnector for storing all data in dictionaries (in-memory)

    Parameters
    ----------
    connector : Connector object
        object that provides the interface to the
        database, e.g. ArcticConnector (see pastastore.connectors)
    name : str, optional
        name of the PastaStore, by default takes the name of the Connector object
    """

    _accessors = set()

    def __init__(
        self,
        connector: Optional[BaseConnector] = None,
        name: Optional[str] = None,
    ):
        """Initialize PastaStore for managing pastas time series and models.

        Parameters
        ----------
        connector : Connector object, optional
            object that provides the connection to the database. Default is None, which
            will create a DictConnector. This default Connector does not store data on
            disk.
        name : str, optional
            name of the PastaStore, if not provided uses the Connector name
        """
        if isinstance(connector, str):
            raise DeprecationWarning(
                "PastaStore expects the connector as the first argument since v1.1!"
            )
        if connector is None:
            connector = DictConnector("pastas_db")
        self.conn = connector
        self.validator = self.conn.validator
        self.name = name if name is not None else self.conn.name
        self._register_connector_methods()

        # register map, plot and yaml classes
        self.maps = Maps(self)
        self.plots = Plots(self)
        self.yaml = PastastoreYAML(self)
        self.zip = ZipUtils(self)

    @classmethod
    def from_pastastore_config_file(cls, fname, update_path: bool = True):
        """Create a PastaStore from a pastastore config file.

        Parameters
        ----------
        fname : str
            path to the pastastore config file
        update_path : bool, optional
            when True, use path derived from location of the config file instead of
            the stored path in the config file. If a PastaStore is moved, the path
            in the config file will probably still refer to the old location. Set to
            False to read the file from the path listed in the config file. In that
            case config files do not need to be stored within the correct directory.

        Returns
        -------
        PastaStore
            PastaStore
        """
        with open(fname, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        conn_type = cfg.pop("connector_type")
        if conn_type == "pas":
            # allow loading of pas files from config file that was moved
            if update_path:
                # update path to the config file
                cfg["path"] = Path(fname).parent.parent
            conn = PasConnector(**cfg)
        elif conn_type == "arcticdb":
            if update_path:
                prefix, _ = cfg["uri"].split("://")
                if prefix.lower() == "lmbd":
                    cfg["uri"] = prefix + "://" + str(Path(fname).parent.parent)
            conn = ArcticDBConnector(**cfg)
        else:
            raise ValueError(
                f"Cannot load connector type: '{conn_type}'. "
                "This is only supported for PasConnector and ArcticDBConnector."
            )
        return cls(conn)

    @property
    def empty(self) -> bool:
        """Check if the PastaStore is empty."""
        return self.conn.empty

    def _register_connector_methods(self):
        """Register connector methods (internal method)."""
        methods = [
            func
            for func in dir(self.conn)
            if callable(getattr(self.conn, func)) and not func.startswith("_")
        ]
        for meth in methods:
            setattr(self, meth, getattr(self.conn, meth))

    @property
    def oseries(self):
        """
        Returns the oseries metadata as dataframe.

        Returns
        -------
        oseries
            oseries metadata as dataframe
        """
        return self.conn.oseries

    @property
    def stresses(self):
        """
        Returns the stresses metadata as dataframe.

        Returns
        -------
        stresses
            stresses metadata as dataframe
        """
        return self.conn.stresses

    @property
    def models(self):
        """Return the ModelAccessor object.

        The ModelAccessor object allows dictionary-like assignment and access to models.
        In addition it provides some useful utilities for working with stored models
        in the database.

        Examples
        --------
        Get a model by name::

        >>> model = pstore.models["my_model"]

        Store a model in the database::

        >>> pstore.models["my_model_v2"] = model

        Get model metadata dataframe::

        >>> pstore.models.metadata

        Number of models::

        >>> len(pstore.models)

        Random model::

        >>> model = pstore.models.random()

        Iterate over stored models::

        >>> for ml in pstore.models:
        >>>     ml.solve()

        Returns
        -------
        ModelAccessor
            ModelAccessor object
        """
        return self.conn.models

    @property
    def oseries_names(self):
        """Return list of oseries names.

        Returns
        -------
        list
            list of oseries names
        """
        return self.conn.oseries_names

    @property
    def stresses_names(self):
        """Return list of streses names.

        Returns
        -------
        list
            list of stresses names
        """
        return self.conn.stresses_names

    @property
    def model_names(self):
        """Return list of model names.

        Returns
        -------
        list
            list of model names
        """
        return self.conn.model_names

    @property
    def _modelnames_cache(self):
        # Backward-compat shim; prefer public property model_names
        return self.conn.model_names

    @property
    def n_oseries(self):
        """Return number of oseries.

        Returns
        -------
        int
            number of oseries
        """
        return self.conn.n_oseries

    @property
    def n_stresses(self):
        """Return number of stresses.

        Returns
        -------
        int
            number of stresses
        """
        return self.conn.n_stresses

    @property
    def n_models(self):
        """Return number of models.

        Returns
        -------
        int
            number of models
        """
        return self.conn.n_models

    @property
    def oseries_models(self):
        """Return dictionary of models per oseries.

        Returns
        -------
        dict
            dictionary containing list of models (values) for each oseries (keys).
        """
        return self.conn.oseries_models

    @property
    def oseries_with_models(self):
        """Return list of oseries for which models are contained in the database.

        Returns
        -------
        list
            list of oseries names for which models are contained in the database.
        """
        return self.conn.oseries_with_models

    @property
    def stresses_models(self):
        """Return dictionary of models per stress.

        Returns
        -------
        dict
            dictionary containing list of models (values) for each stress (keys).
        """
        return self.conn.stresses_models

    @property
    def stresses_with_models(self):
        """Return list of stresses for which models are contained in the database.

        Returns
        -------
        list
            list of stress names for which models are contained in the database.
        """
        return self.conn.stresses_with_models

    def __repr__(self):
        """Representation string of the object."""
        return f"<PastaStore> {self.name}: \n - " + self.conn.__str__()

    def get_oseries_distances(
        self, names: Optional[Union[list, str]] = None
    ) -> FrameOrSeriesUnion:
        """Get the distances in meters between the oseries.

        Parameters
        ----------
        names: str or list of str
            names of the oseries to calculate distances between

        Returns
        -------
        distances: pandas.DataFrame
            Pandas DataFrame with the distances between the oseries
        """
        oseries_df = self.conn.oseries
        other_df = self.conn.oseries

        names = self.conn.parse_names(names)

        xo = pd.to_numeric(oseries_df.loc[names, "x"])
        xt = pd.to_numeric(other_df.loc[:, "x"])
        yo = pd.to_numeric(oseries_df.loc[names, "y"])
        yt = pd.to_numeric(other_df.loc[:, "y"])

        xh, xi = np.meshgrid(xt, xo)
        yh, yi = np.meshgrid(yt, yo)

        distances = pd.DataFrame(
            np.sqrt((xh - xi) ** 2 + (yh - yi) ** 2),
            index=names,
            columns=other_df.index,
        )

        return distances

    def get_nearest_oseries(
        self,
        names: Optional[Union[list, str]] = None,
        n: int = 1,
        maxdist: Optional[float] = None,
    ) -> FrameOrSeriesUnion:
        """Get the nearest (n) oseries.

        Parameters
        ----------
        names: str or list of str
            string or list of strings with the name(s) of the oseries
        n: int
            number of oseries to obtain
        maxdist : float, optional
            maximum distance to consider

        Returns
        -------
        oseries:
            list with the names of the oseries.
        """
        distances = self.get_oseries_distances(names)
        if maxdist is not None:
            distances = distances.where(distances <= maxdist, np.nan)

        data = pd.DataFrame(columns=np.arange(n))

        for series_name in distances.index:
            others = distances.loc[series_name].dropna().sort_values().index.tolist()
            # remove self
            others.remove(series_name)
            series = pd.DataFrame(
                index=[series_name], columns=data.columns, data=[others[:n]]
            )
            data = pd.concat([data, series], axis=0)
        return data

    def get_distances(
        self,
        oseries: Optional[Union[list, str]] = None,
        stresses: Optional[Union[list, str]] = None,
        kind: Optional[Union[str, List[str]]] = None,
    ) -> FrameOrSeriesUnion:
        """Get the distances in meters between the oseries and stresses.

        Parameters
        ----------
        oseries: str or list of str
            name(s) of the oseries
        stresses: str or list of str
            name(s) of the stresses
        kind: str, list of str
            string or list of strings representing which kind(s) of
            stresses to consider

        Returns
        -------
        distances: pandas.DataFrame
            Pandas DataFrame with the distances between the oseries (index)
            and the stresses (columns).
        """
        oseries_df = self.conn.oseries
        stresses_df = self.conn.stresses

        oseries = self.conn.parse_names(oseries)

        if stresses is None and kind is None:
            stresses = stresses_df.index
        elif stresses is not None and kind is not None:
            if isinstance(kind, str):
                kind = [kind]
            mask = stresses_df.kind.isin(kind)
            stresses = stresses_df.loc[stresses].loc[mask].index
        elif stresses is None:
            if isinstance(kind, str):
                kind = [kind]
            stresses = stresses_df.loc[stresses_df.kind.isin(kind)].index

        xo = pd.to_numeric(oseries_df.loc[oseries, "x"])
        xt = pd.to_numeric(stresses_df.loc[stresses, "x"])
        yo = pd.to_numeric(oseries_df.loc[oseries, "y"])
        yt = pd.to_numeric(stresses_df.loc[stresses, "y"])

        xh, xi = np.meshgrid(xt, xo)
        yh, yi = np.meshgrid(yt, yo)

        distances = pd.DataFrame(
            np.sqrt((xh - xi) ** 2 + (yh - yi) ** 2),
            index=oseries,
            columns=stresses,
        )

        return distances

    def get_nearest_stresses(
        self,
        oseries: Optional[Union[list, str]] = None,
        stresses: Optional[Union[list, str]] = None,
        kind: Optional[Union[list, str]] = None,
        n: int = 1,
        maxdist: Optional[float] = None,
    ) -> FrameOrSeriesUnion:
        """Get the nearest (n) stresses of a specific kind.

        Parameters
        ----------
        oseries: str
            string with the name of the oseries
        stresses: str or list of str
            string with the name of the stresses
        kind: str, list of str, optional
            string or list of str with the name of the kind(s)
            of stresses to consider
        n: int
            number of stresses to obtain
        maxdist : float, optional
            maximum distance to consider

        Returns
        -------
        stresses:
            list with the names of the stresses.
        """
        distances = self.get_distances(oseries, stresses, kind)
        if maxdist is not None:
            distances = distances.where(distances <= maxdist, np.nan)

        data = pd.DataFrame(columns=np.arange(n))

        for series in distances.index:
            series = pd.DataFrame(
                [distances.loc[series].dropna().sort_values().index[:n]]
            )
            data = pd.concat([data, series], axis=0)
        return data

    def get_signatures(
        self,
        names: list[str] | str | None = None,
        signatures: list[str] | None = None,
        libname: TimeSeriesLibs = "oseries",
        progressbar: bool = False,
        ignore_errors: bool = False,
    ) -> pd.DataFrame | pd.Series:
        """Get groundwater signatures.

        NaN-values are returned when the signature cannot be computed.

        Parameters
        ----------
        names : str, list of str, or None, optional
            names of the time series, by default None which
            uses all the time series in the library
        signatures : list of str, optional
            list of groundwater signatures to compute, if None all groundwater
            signatures in ps.stats.signatures.__all__ are used, by default None
        libname : str
            name of the library containing the time series
            ('oseries' or 'stresses'), by default "oseries"
        progressbar : bool, optional
            show progressbar, by default False
        ignore_errors : bool, optional
            ignore errors when True, i.e. when non-existent timeseries is
            encountered in names, by default False

        Returns
        -------
        signatures_df : pandas.DataFrame or pandas.Series
            Containing the time series (columns) and the signatures (index).

        Note
        ----
        Names is set as the first argument to allow parallelization.
        """
        names = self.conn.parse_names(names, libname=libname)

        signatures = (
            ps.stats.signatures.__all__.copy() if signatures is None else signatures
        )

        sigs = []
        # loop through oseries names
        for name in (
            tqdm(names, desc="Get groundwater signatures") if progressbar else names
        ):
            sig = pd.Series(np.nan, index=signatures, name=name)
            try:
                s = (
                    self.conn.get_oseries(name)
                    if libname == "oseries"
                    else self.conn.get_stresses(name)
                )
            except Exception as e:
                if ignore_errors:
                    return sig
                else:
                    raise e

            try:
                i_signatures: pd.Series = ps.stats.signatures.summary(
                    s.squeeze(), signatures
                ).squeeze("columns")
            except Exception as e:
                if ignore_errors:
                    i_signatures = []
                    for signature in signatures:
                        try:
                            sign_val = getattr(ps.stats.signatures, signature)(
                                s.squeeze()
                            )
                        except Exception as _:
                            sign_val = np.nan
                        i_signatures.append(sign_val)
                else:
                    raise e
            sig.loc[signatures] = i_signatures.values
            sigs.append(sig)
        if len(sigs) == 1:
            return sigs[0]
        else:
            return pd.concat(sigs, axis=1)

    def get_tmin_tmax(
        self,
        libname: Optional[PastasLibs] = None,
        names: Union[str, List[str], None] = None,
        progressbar: bool = False,
    ) -> pd.DataFrame:
        """Get tmin and tmax for time series and/or models.

        Parameters
        ----------
        libname : str, optional
            name of the library containing the time series ('oseries', 'stresses',
            'models', or None), by default None which returns tmin/tmax for all
            libraries
        names : str, list of str, or None, optional
            names of the time series, by default None which
            uses all the time series in the library
        progressbar : bool, optional
            show progressbar, by default False

        Returns
        -------
        tmintmax : pd.dataframe
            Dataframe containing tmin and tmax per time series and/or model
        """
        results = {}
        if libname is None:
            libs = ["oseries", "stresses", "models"]
        else:
            libs = [libname]

        for lib in libs:
            _names = self.conn.parse_names(names, libname=lib)
            tmintmax = pd.DataFrame(
                index=names, columns=["tmin", "tmax"], dtype="datetime64[ns]"
            )
            desc = f"Get tmin/tmax {lib}"
            for n in tqdm(_names, desc=desc) if progressbar else _names:
                if lib == "models":
                    mld = self.conn.get_models(
                        n,
                        return_dict=True,
                    )
                    tmintmax.loc[n, "tmin"] = mld["settings"]["tmin"]
                    tmintmax.loc[n, "tmax"] = mld["settings"]["tmax"]
                else:
                    s = (
                        self.conn.get_oseries(n)
                        if lib == "oseries"
                        else self.conn.get_stresses(n)
                    )
                    tmintmax.loc[n, "tmin"] = s.first_valid_index()
                    tmintmax.loc[n, "tmax"] = s.last_valid_index()

            results[lib] = tmintmax

        if len(results) == 1:
            return results[lib]
        else:
            return pd.concat(results, axis=0)

    def get_extent(self, libname, names=None, buffer=0.0):
        """Get extent [xmin, xmax, ymin, ymax] from library.

        Parameters
        ----------
        libname : str
            name of the library containing the time series
            ('oseries', 'stresses', 'models')
        names : str, list of str, or None, optional
            list of names to include for computing the extent
        buffer : float, optional
            add this distance to the extent, by default 0.0

        Returns
        -------
        extent : list
            extent [xmin, xmax, ymin, ymax]
        """
        names = self.conn.parse_names(names, libname=libname)
        if libname in ["oseries", "stresses"]:
            df = getattr(self, libname)
        elif libname == "models":
            df = self.oseries
        else:
            raise ValueError(f"Cannot get extent for library '{libname}'.")

        extent = [
            df.loc[names, "x"].min() - buffer,
            df.loc[names, "x"].max() + buffer,
            df.loc[names, "y"].min() - buffer,
            df.loc[names, "y"].max() + buffer,
        ]
        return extent

    def get_parameters(
        self,
        parameters: Optional[List[str]] = None,
        modelnames: Optional[List[str]] = None,
        param_value: Optional[str] = "optimal",
        progressbar: Optional[bool] = False,
        ignore_errors: Optional[bool] = True,
    ) -> FrameOrSeriesUnion:
        """Get model parameters.

        NaN-values are returned when the parameters are not present in the model or the
        model is not optimized.

        Parameters
        ----------
        parameters : list of str, optional
            names of the parameters, by default None which uses all
            parameters from each model
        modelnames : str or list of str, optional
            name(s) of model(s), by default None in which case all models
            are used
        param_value : str, optional
            which column to use from the model parameters dataframe, by
            default "optimal" which retrieves the optimized parameters.
        progressbar : bool, optional
            show progressbar, default is False
        ignore_errors : bool, optional
            ignore errors when True, i.e. when non-existent model is
            encountered in modelnames, by default True

        Returns
        -------
        p : pandas.DataFrame
            DataFrame containing the parameters (columns) per model (rows)
        """
        modelnames = self.conn.parse_names(modelnames, libname="models")

        # create dataframe for results
        p = pd.DataFrame(index=modelnames, columns=parameters)

        # loop through model names and store results
        desc = "Get model parameters"
        for mlname in tqdm(modelnames, desc=desc) if progressbar else modelnames:
            try:
                mldict = self.get_models(mlname, return_dict=True, progressbar=False)
            except Exception as e:
                if ignore_errors:
                    p.loc[mlname, :] = np.nan
                    continue
                else:
                    raise e
            if parameters is None:
                pindex = mldict["parameters"].index
            else:
                pindex = parameters

            for c in pindex:
                if c in mldict["parameters"].index:
                    p.loc[mlname, c] = mldict["parameters"].loc[c, param_value]
                else:
                    p.loc[mlname, c] = np.nan

        return p.astype(float).squeeze()

    def get_statistics(
        self,
        statistics: Union[str, List[str]],
        modelnames: Optional[List[str]] = None,
        parallel: bool = False,
        progressbar: Optional[bool] = False,
        ignore_errors: Optional[bool] = False,
        fancy_output: bool = True,
        **kwargs,
    ) -> FrameOrSeriesUnion:
        """Get model statistics.

        Parameters
        ----------
        statistics : str or list of str
            statistic or list of statistics to calculate, e.g. ["evp", "rsq", "rmse"],
            for a full list see `pastas.modelstats.Statistics.ops`.
        modelnames : list of str, optional
            modelnames to calculates statistics for, by default None, which
            uses all models in the store
        progressbar : bool, optional
            show progressbar, by default False
        ignore_errors : bool, optional
            ignore errors when True, i.e. when trying to calculate statistics
            for non-existent model in modelnames, default is False
        parallel : bool, optional
            use parallel processing, by default False
        fancy_output : bool, optional
            only read if parallel=True, if True, return as DataFrame with statistics,
            otherwise return list of results
        **kwargs
            any arguments that can be passed to the methods for calculating
            statistics

        Returns
        -------
        s : pandas.DataFrame
        """
        modelnames = self.conn.parse_names(modelnames, libname="models")

        # if statistics is str
        if isinstance(statistics, str):
            statistics = [statistics]

        if parallel:
            kwargs["statistics"] = statistics
            if self.conn.conn_type == "pas":
                kwargs["connector"] = self.conn
            return self.apply(
                "models",
                self.conn._get_statistics,
                modelnames,
                kwargs=kwargs,
                parallel=parallel,
                progressbar=progressbar,
                fancy_output=fancy_output,
            ).T  # transpose to match serial output
        else:
            # create dataframe for results
            s = pd.DataFrame(index=modelnames, columns=statistics, data=np.nan)

            # loop through model names
            desc = "Get model statistics"
            for mlname in tqdm(modelnames, desc=desc) if progressbar else modelnames:
                try:
                    ml = self.conn.get_models(mlname, progressbar=False)
                except Exception as e:
                    if ignore_errors:
                        continue
                    else:
                        raise e
                for stat in statistics:
                    value = getattr(ml.stats, stat)(**kwargs)
                    s.loc[mlname, stat] = value

            return s.astype(float).squeeze()

    def create_model(
        self,
        name: str,
        modelname: Optional[str] = None,
        add_recharge: bool = True,
        add_ar_noisemodel: bool = False,
        recharge_name: str = "recharge",
    ) -> ps.Model:
        """Create a pastas Model.

        Parameters
        ----------
        name : str
            name of the oseries to create a model for
        modelname : str, optional
            name of the model, default is None, which uses oseries name
        add_recharge : bool, optional
            add recharge to the model by looking for the closest
            precipitation and evaporation time series in the stresses
            library, by default True
        add_ar_noisemodel : bool, optional
            add AR(1) noise model to the model, by default False
        recharge_name : str
            name of the RechargeModel

        Returns
        -------
        pastas.Model
            model for the oseries

        Raises
        ------
        KeyError
            if data is stored as dataframe and no column is provided
        ValueError
            if time series is empty
        """
        # get oseries metadata
        meta = self.conn.get_metadata("oseries", name, as_frame=False)
        ts = self.conn.get_oseries(name)

        # convert to time series and create model
        if not ts.dropna().empty:
            if modelname is None:
                modelname = name
            ml = ps.Model(ts, name=modelname, metadata=meta)
            if add_recharge:
                self.add_recharge(ml, recharge_name=recharge_name)
            if add_ar_noisemodel and PASTAS_GEQ_150:
                ml.add_noisemodel(ps.ArNoiseModel())
            return ml
        else:
            raise ValueError("Empty time series!")

    def create_models_bulk(
        self,
        oseries: Optional[Union[list, str]] = None,
        add_recharge: bool = True,
        solve: bool = False,
        store_models: bool = True,
        ignore_errors: bool = False,
        suffix: Optional[str] = None,
        progressbar: bool = True,
        **kwargs,
    ) -> Union[Tuple[dict, dict], dict]:
        """Bulk creation of pastas models.

        Parameters
        ----------
        oseries : list of str, optional
            names of oseries to create models for, by default None,
            which creates models for all oseries
        add_recharge : bool, optional
            add recharge to the models based on closest
            precipitation and evaporation time series, by default True
        solve : bool, optional
            solve the model, by default False
        store_models : bool, optional
            if False, return a list of models, by default True, which will
            store the models in the database.
        ignore_errors : bool, optional
            ignore errors while creating models, by default False
        suffix : str, optional
            add suffix to oseries name to create model name, by default None
        progressbar : bool, optional
            show progressbar, by default True

        Returns
        -------
        models : dict, if return_models is True
            dictionary of models
        errors : list, always returned
            list of model names that could not be created
        """
        if oseries is None:
            oseries = self.conn.oseries.index
        elif isinstance(oseries, str):
            oseries = [oseries]

        models = {}
        errors = {}
        desc = "Bulk creation models"
        for o in tqdm(oseries, desc=desc) if progressbar else oseries:
            try:
                if suffix is not None:
                    modelname = f"{o}{suffix}"
                else:
                    modelname = o
                iml = self.create_model(
                    o, modelname=modelname, add_recharge=add_recharge
                )
            except Exception as e:
                if ignore_errors:
                    errors[o] = e
                    continue
                else:
                    raise e
            if solve:
                iml.solve(**kwargs)
            if store_models:
                self.add_model(iml, overwrite=True)
            else:
                models[o] = iml
        if len(errors) > 0:
            logger.warning(
                "Warning! Some models were not created. See errors dictionary "
                "for more information."
            )
        if store_models:
            return errors
        else:
            return models, errors

    def add_recharge(
        self,
        ml: ps.Model,
        rfunc=None,
        recharge=None,
        recharge_name: str = "recharge",
    ) -> None:
        """Add recharge to a pastas model.

        Uses closest precipitation and evaporation time series in database.
        These are assumed to be labeled with kind = 'prec' or 'evap'.

        Parameters
        ----------
        ml : pastas.Model
            pastas.Model object
        rfunc : pastas.rfunc, optional
            response function to use for recharge in model, by default None
            which uses ps.Exponential() (for different response functions, see pastas
            documentation)
        recharge : ps.RechargeModel
            recharge model to use, default is ps.rch.Linear()
        recharge_name : str
            name of the RechargeModel
        """
        if recharge is None:
            recharge = ps.rch.Linear()
        if rfunc is None:
            rfunc = ps.Exponential

        self.add_stressmodel(
            ml,
            stresses={"prec": "nearest", "evap": "nearest"},
            rfunc=rfunc,
            stressmodel=ps.RechargeModel,
            stressmodel_name=recharge_name,
            recharge=recharge,
        )

    def _parse_stresses(
        self,
        stresses: Union[str, List[str], Dict[str, str]],
        kind: Optional[str],
        stressmodel,
        oseries: Optional[str] = None,
    ):
        # parse stresses for RechargeModel, allow list of len 2 or 3 and
        # set correct kwarg names
        if stressmodel._name == "RechargeModel":
            if isinstance(stresses, list):
                if len(stresses) == 2:
                    stresses = {
                        "prec": stresses[0],
                        "evap": stresses[1],
                    }
                elif len(stresses) == 3:
                    stresses = {
                        "prec": stresses[0],
                        "evap": stresses[1],
                        "temp": stresses[2],
                    }
                else:
                    raise ValueError(
                        "RechargeModel requires 2 or 3 stress names, "
                        f"got: {len(stresses)}!"
                    )
        # if stresses is list, create dictionary normally
        elif isinstance(stresses, list):
            stresses = {"stress": stresses}
        # if stresses is str, make it a list of len 1
        elif isinstance(stresses, str):
            stresses = {"stress": [stresses]}

        # check if stresses is a dictionary, else raise TypeError
        if not isinstance(stresses, dict):
            raise TypeError("stresses must be a list, string or dictionary!")

        # if no kind specified, set to well for WellModel
        if stressmodel._name == "WellModel":
            if kind is None:
                kind = "well"

        # store a copy of the user input for kind
        if isinstance(kind, list):
            _kind = kind.copy()
        else:
            _kind = kind

        # create empty list for gathering metadata
        metadata = []
        # loop over stresses keys/values
        for i, (k, v) in enumerate(stresses.items()):
            # if entry in dictionary is str, make it list of len 1
            if isinstance(v, str):
                v = [v]
            # parse value
            if isinstance(v, list):
                for item in v:
                    names = []  # empty list for names
                    # parse nearest
                    if item.startswith("nearest"):
                        # check oseries defined if nearest option is used
                        if not oseries:
                            raise ValueError(
                                "Getting nearest stress(es) requires oseries name!"
                            )
                        try:
                            if len(item.split()) == 3:  # nearest <n> <kind>
                                n = int(item.split()[1])
                                kind = item.split()[2]
                            elif len(item.split()) == 2:  # nearest <n> | <kind>
                                try:
                                    n = int(item.split()[1])  # try converting to <n>
                                except ValueError:
                                    n = 1
                                    kind = item.split()[1]  # interpret as <kind>
                            else:  # nearest
                                n = 1
                                # if RechargeModel, we can infer kind
                                if (
                                    _kind is None
                                    and stressmodel._name == "RechargeModel"
                                ):
                                    kind = k
                                elif _kind is None:  # catch no kind with bare nearest
                                    raise ValueError(
                                        "Bare 'nearest' found but no kind specified."
                                    )
                                elif isinstance(_kind, list):
                                    kind = _kind[i]  # if multiple kind, select i-th
                        except Exception as e:
                            # raise if nearest parsing failed
                            raise ValueError(
                                f"Could not parse stresses: '{item}'! "
                                "When using option 'nearest', use 'nearest' and specify"
                                " kind, or 'nearest <kind>' or 'nearest <n> <kind>'!"
                            ) from e
                        # check if kind exists at all
                        if kind not in self.stresses.kind.values:
                            raise ValueError(
                                f"Could not find stresses with kind='{kind}'!"
                            )
                        # get stress names of <n> nearest <kind> stresses
                        inames = self.get_nearest_stresses(
                            oseries, kind=kind, n=n
                        ).iloc[0]
                        # check if any NaNs in result
                        if inames.isna().any():
                            nkind = (self.stresses.kind == kind).sum()
                            raise ValueError(
                                f"Could not find {n} nearest stress(es) for '{kind}'! "
                                f"There are only {nkind} '{kind}' stresses."
                            )
                        # append names
                        names += inames.tolist()
                    else:
                        # assume name is name of stress
                        names.append(item)
                # get stresses and metadata
                stress_series, imeta = self.get_stresses(
                    names, return_metadata=True, squeeze=True
                )
                # replace stress name(s) with time series
                if len(names) > 1:
                    stresses[k] = list(stress_series.values())
                else:
                    stresses[k] = stress_series
                # gather metadata
                if isinstance(imeta, list):
                    metadata += imeta
                else:
                    metadata.append(imeta)

        return stresses, metadata

    def get_stressmodel(
        self,
        stresses: Union[str, List[str], Dict[str, str]],
        stressmodel=ps.StressModel,
        stressmodel_name: Optional[str] = None,
        rfunc=ps.Exponential,
        rfunc_kwargs: Optional[dict] = None,
        kind: Optional[Union[List[str], str]] = None,
        oseries: Optional[str] = None,
        **kwargs,
    ):
        """Get a Pastas stressmodel from stresses time series in Pastastore.

        Supports "nearest" selection. Any stress name can be replaced by
        "nearest [<n>] <kind>" where <n> is optional and represents the number of
        nearest stresses and <kind> and represents the kind of stress to
        consider. <kind> can also be specified directly with the `kind` kwarg.

        Note: the 'nearest' option requires the oseries name to be provided.
        Additionally, 'x' and 'y' metadata must be stored for oseries and stresses.

        Parameters
        ----------
        stresses : str, list of str, or dict
            name(s) of the time series to use for the stressmodel, or dictionary
            with key(s) and value(s) as time series name(s). Options include:
               - name of stress: `"prec_stn"`
               - list of stress names: `["prec_stn", "evap_stn"]`
               - dict for RechargeModel: `{"prec": "prec_stn", "evap": "evap_stn"}`
               - dict for StressModel: `{"stress": "well1"}`
               - nearest, specifying kind: `"nearest well"`
               - nearest specifying number and kind: `"nearest 2 well"`
        stressmodel : str or class
            stressmodel class to use, by default ps.StressModel
        stressmodel_name : str, optional
            name of the stressmodel, by default None, which uses the stress name,
            if there is 1 stress otherwise the name of the stressmodel type. For
            RechargeModels, the name defaults to 'recharge'.
        rfunc : str or class
            response function class to use, by default ps.Exponential
        rfunc_kwargs : dict, optional
            keyword arguments to pass to the response function, by default None
        kind : str or list of str, optional
            specify kind of stress(es) to use, by default None, useful in combination
            with 'nearest' option for defining stresses
        oseries : str, optional
            name of the oseries to use for the stressmodel, by default None, used when
            'nearest' option is used for defining stresses.
        **kwargs
            additional keyword arguments to pass to the stressmodel

        Returns
        -------
        stressmodel : pastas.StressModel
            pastas StressModel that can be added to pastas Model.
        """
        # get stressmodel class, if str was provided
        if isinstance(stressmodel, str):
            stressmodel = getattr(ps, stressmodel)

        # parse stresses names to get time series and metadata
        stresses, metadata = self._parse_stresses(
            stresses=stresses, stressmodel=stressmodel, kind=kind, oseries=oseries
        )

        # get stressmodel name if not provided
        if stressmodel_name is None:
            if stressmodel._name == "RechargeModel":
                stressmodel_name = "recharge"
            elif len(metadata) == 1:
                stressmodel_name = stresses["stress"].squeeze().name
            else:
                stressmodel_name = stressmodel._name

        # check if metadata is list of len 1 and unpack
        if isinstance(metadata, list) and len(metadata) == 1:
            metadata = metadata[0]

        # get stressmodel time series settings
        if kind and "settings" not in kwargs:
            # try using kind to get predefined settings options
            if isinstance(kind, str):
                kwargs["settings"] = ps.rcParams["timeseries"].get(kind, None)
            else:
                kwargs["settings"] = [
                    ps.rcParams["timeseries"].get(ikind, None) for ikind in kind
                ]
        elif kind is None and "settings" not in kwargs:
            # try using kind stored in metadata to get predefined settings options
            if isinstance(metadata, list):
                kwargs["settings"] = [
                    ps.rcParams["timeseries"].get(imeta.get("kind", None), None)
                    for imeta in metadata
                ]
            elif isinstance(metadata, dict):
                kwargs["settings"] = ps.rcParams["timeseries"].get(
                    metadata.get("kind", None), None
                )

        # get rfunc class if str was provided
        if isinstance(rfunc, str):
            rfunc = getattr(ps, rfunc)

        # create empty rfunc_kwargs if not provided
        if rfunc_kwargs is None:
            rfunc_kwargs = {}

        # special for WellModels
        if stressmodel._name == "WellModel":
            if isinstance(stresses["stress"], list):
                names = [s.squeeze().name for s in stresses["stress"]]
            else:
                names = [stresses["stress"].squeeze().name]
                stresses["stress"] = [stresses["stress"]]  # ugly fix for WellModel
            # check oseries is provided
            if oseries is None:
                raise ValueError("WellModel requires 'oseries' to compute distances!")
            # compute distances and add to kwargs
            distances = (
                self.get_distances(oseries=oseries, stresses=names)
                .T.squeeze(axis=1)
                .values
            )
            kwargs["distances"] = distances
            # set settings to well
            if "settings" not in kwargs:
                kwargs["settings"] = "well"
            # override rfunc and set to HantushWellModel
            rfunc = ps.HantushWellModel

        kwargs["metadata"] = metadata

        return stressmodel(
            **stresses,
            rfunc=rfunc(**rfunc_kwargs),
            name=stressmodel_name,
            **kwargs,
        )

    def add_stressmodel(
        self,
        ml: Union[ps.Model, str],
        stresses: Union[str, List[str], Dict[str, str]],
        stressmodel=ps.StressModel,
        stressmodel_name: Optional[str] = None,
        rfunc=ps.Exponential,
        rfunc_kwargs: Optional[dict] = None,
        kind: Optional[Union[List[str], str]] = None,
        **kwargs,
    ):
        """Add a pastas StressModel from stresses time series in Pastastore.

        Supports "nearest" selection. Any stress name can be replaced by
        "nearest [<n>] <kind>" where <n> is optional and represents the number of
        nearest stresses and <kind> and represents the kind of stress to
        consider. <kind> can also be specified directly with the `kind` kwarg.

        Note: the 'nearest' option requires the oseries name to be provided.
        Additionally, 'x' and 'y' metadata must be stored for oseries and stresses.

        Parameters
        ----------
        ml : pastas.Model or str
            pastas.Model object to add StressModel to, if passed as string,
            model is loaded from store, the stressmodel is added and then written
            back to the store.
        stresses : str, list of str, or dict
            name(s) of the time series to use for the stressmodel, or dictionary
            with key(s) and value(s) as time series name(s). Options include:
               - name of stress: `"prec_stn"`
               - list of stress names: `["prec_stn", "evap_stn"]`
               - dict for RechargeModel: `{"prec": "prec_stn", "evap": "evap_stn"}`
               - dict for StressModel: `{"stress": "well1"}`
               - nearest, specifying kind: `"nearest well"`
               - nearest specifying number and kind: `"nearest 2 well"`
        stressmodel : str or class
            stressmodel class to use, by default ps.StressModel
        stressmodel_name : str, optional
            name of the stressmodel, by default None, which uses the stress name,
            if there is 1 stress otherwise the name of the stressmodel type. For
            RechargeModels, the name defaults to 'recharge'.
        rfunc : str or class
            response function class to use, by default ps.Exponential
        rfunc_kwargs : dict, optional
            keyword arguments to pass to the response function, by default None
        kind : str or list of str, optional
            specify kind of stress(es) to use, by default None, useful in combination
            with 'nearest' option for defining stresses
        **kwargs
            additional keyword arguments to pass to the stressmodel
        """
        sm = self.get_stressmodel(
            stresses=stresses,
            stressmodel=stressmodel,
            stressmodel_name=stressmodel_name,
            rfunc=rfunc,
            rfunc_kwargs=rfunc_kwargs,
            kind=kind,
            oseries=ml if isinstance(ml, str) else ml.oseries.name,
            **kwargs,
        )
        if isinstance(ml, str):
            ml = self.conn.get_model(ml)
            ml.add_stressmodel(sm)
            self.conn.add_model(ml, overwrite=True)
            logger.info(
                "Stressmodel '%s' added to model '%s' and stored in database.",
                sm.name,
                ml.name,
            )
        else:
            ml.add_stressmodel(sm)

    def solve_models(
        self,
        modelnames: Union[List[str], str, None] = None,
        report: bool = False,
        ignore_solve_errors: bool = False,
        progressbar: bool = True,
        parallel: bool = False,
        max_workers: Optional[int] = None,
        **kwargs,
    ) -> None:
        """Solves the models in the store.

        Parameters
        ----------
        modelnames : list of str, optional
            list of model names, if None all models in the pastastore
            are solved.
        report : boolean, optional
            determines if a report is printed when the model is solved,
            default is False
        ignore_solve_errors : boolean, optional
            if True, errors emerging from the solve method are ignored,
            default is False which will raise an exception when a model
            cannot be optimized
        progressbar : bool, optional
            show progressbar, default is True.
        parallel: bool, optional
            if True, solve models in parallel using ProcessPoolExecutor
        max_workers: int, optional
            maximum number of workers to use in parallel solving, default is
            None which will use the number of cores available on the machine
        **kwargs : dictionary
            arguments are passed to the solve method.

        Notes
        -----
        Users should be aware that parallel solving is platform dependent
        and may not always work. The current implementation works well for Linux users.
        For Windows users, parallel solving does not work when called directly from
        Jupyter Notebooks or IPython. To use parallel solving on Windows, the following
        code should be used in a Python file::

            from multiprocessing import freeze_support

            if __name__ == "__main__":
                freeze_support()
                pstore.solve_models(parallel=True)
        """
        if "mls" in kwargs:
            modelnames = kwargs.pop("mls")
            logger.warning("Argument `mls` is deprecated, use `modelnames` instead.")

        modelnames = self.conn.parse_names(modelnames, libname="models")

        # prepare parallel
        if parallel and self.conn.conn_type == "dict":
            parallel = False
            logger.error(
                "Parallel solving only supported for PasConnector and "
                "ArcticDBConnector databases. Setting parallel to `False`"
            )
        if parallel:
            if self.conn.conn_type == "arcticdb":
                solve_model = partial(
                    self.conn._solve_model,
                    report=report,
                    ignore_solve_errors=ignore_solve_errors,
                    **kwargs,
                )
                self.conn._parallel(
                    solve_model,
                    modelnames,
                    max_workers=max_workers,
                    chunksize=None,
                    progressbar=progressbar,
                    desc="Solving models (parallel)",
                )
            elif self.conn.conn_type == "pas":
                solve_model = partial(
                    self.conn._solve_model,
                    connector=self.conn,
                    report=report,
                    ignore_solve_errors=ignore_solve_errors,
                    **kwargs,
                )
                self.conn._parallel(
                    solve_model,
                    modelnames,
                    max_workers=max_workers,
                    chunksize=None,
                    progressbar=progressbar,
                    desc="Solving models (parallel)",
                )
        else:
            solve_model = partial(
                self.conn._solve_model,
                connector=self.conn,
                report=report,
                ignore_solve_errors=ignore_solve_errors,
                **kwargs,
            )
            for ml_name in (
                tqdm(modelnames, desc="Solving models") if progressbar else modelnames
            ):
                solve_model(ml_name=ml_name)

    def check_models(self, checklist=None, modelnames=None, style_output: bool = False):
        """Check models against checklist.

        Parameters
        ----------
        checklist : dict, optional
            dictionary containing model check methods, by default None which
            uses the ps.checks.checks_brakenhoff_2022 checklist. This includes:
               - fit metric R² >= 0.6
               - runs test for autocorrelation
               - t95 response < half length calibration period
               - |model parameters| < 1.96 * σ (std deviation)
               - model parameters are not on bounds
        modelnames : list of str, optional
            list of modelnames to perform checks on, by default None
        style_output : bool, optional
            if True, return styled dataframe with pass/fail colors,
            by default False

        Returns
        -------
        pd.DataFrame
            DataFrame containing pass True/False for each check for each model
        """
        if checklist is None:
            checklist = ps.check.checks_brakenhoff_2022

        names = self.conn.parse_names(modelnames, libname="models")

        check_dfs = []
        for n in names:
            cdf = ps.check.checklist(self.models[n], checklist, report=False)["pass"]
            cdf.name = n
            check_dfs.append(cdf)
        chkdf = pd.concat(check_dfs, axis=1)
        chkdf.columns.name = "models"
        if style_output:
            return chkdf.style.map(boolean_styler)
        else:
            return chkdf

    def to_zip(self, fname: str, overwrite=False, progressbar: bool = True):
        """Write data to zipfile.

        Parameters
        ----------
        fname : str
            name of zipfile
        overwrite : bool, optional
            if True, overwrite existing file
        progressbar : bool, optional
            show progressbar, by default True
        """
        from zipfile import ZIP_DEFLATED, ZipFile

        fname = Path(fname)
        if fname.exists() and not overwrite:
            raise FileExistsError(
                "File already exists! Use 'overwrite=True' to force writing file."
            )
        elif fname.exists():
            warnings.warn(f"Overwriting file '{fname.name}'", stacklevel=1)

        with ZipFile(fname, "w", compression=ZIP_DEFLATED) as archive:
            # oseries
            self.zip.series_to_archive(archive, "oseries", progressbar=progressbar)
            # stresses
            self.zip.series_to_archive(archive, "stresses", progressbar=progressbar)
            # models
            self.zip.models_to_archive(archive, progressbar=progressbar)

    def export_model_series_to_csv(
        self,
        names: Optional[Union[list, str]] = None,
        exportdir: Path | str = ".",
        exportmeta: bool = True,
    ):  # pragma: no cover
        """Export model time series to csv files.

        Parameters
        ----------
        names : Optional[Union[list, str]], optional
            names of models to export, by default None, which uses retrieves
            all models from database
        exportdir : str, optional
            directory to export csv files to, default is current directory
        exportmeta : bool, optional
            export metadata for all time series as csv file, default is True
        """
        names = self.conn.parse_names(names, libname="models")
        exportdir = Path(exportdir)
        for name in names:
            mldict = self.get_models(name, return_dict=True)

            oname = mldict["oseries"]["name"]
            o = self.get_oseries(oname)
            o.to_csv(exportdir / f"{oname}.csv")

            if exportmeta:
                metalist = [self.conn.get_metadata("oseries", oname)]

            for sm in mldict["stressmodels"]:
                if mldict["stressmodels"][sm]["stressmodel"] == "RechargeModel":
                    for istress in ["prec", "evap"]:
                        istress = mldict["stressmodels"][sm][istress]
                        stress_name = istress["name"]
                        ts = self.conn.get_stresses(stress_name)
                        ts.to_csv(exportdir / f"{stress_name}.csv")
                        if exportmeta:
                            tsmeta = self.get_metadata("stresses", stress_name)
                            metalist.append(tsmeta)
                else:
                    for istress in mldict["stressmodels"][sm]["stress"]:
                        stress_name = istress["name"]
                        ts = self.conn.get_stresses(stress_name)
                        ts.to_csv(exportdir / f"{stress_name}.csv")
                        if exportmeta:
                            tsmeta = self.conn.get_metadata("stresses", stress_name)
                            metalist.append(tsmeta)

            if exportmeta:
                pd.concat(metalist, axis=0).to_csv(exportdir / f"metadata_{name}.csv")

    @classmethod
    def from_zip(
        cls,
        fname: str,
        conn: Optional[BaseConnector] = None,
        storename: Optional[str] = None,
        progressbar: bool = True,
        series_ext_json: bool = False,
    ):
        """Load PastaStore from zipfile.

        Parameters
        ----------
        fname : str
            pathname of zipfile
        conn : Connector object, optional
            connector for storing loaded data, default is None which creates a
            DictConnector. This Connector does not store data on disk.
        storename : str, optional
            name of the PastaStore, by default None, which
            defaults to the name of the Connector.
        progressbar : bool, optional
            show progressbar, by default True
        series_ext_json : bool, optional
            if True, series are expected to have a .json extension, by default False,
            which assumes a .pas extension. Set this option to true for reading
            zipfiles created with older versions of pastastore <1.8.0.

        Returns
        -------
        pastastore.PastaStore
            return PastaStore containing data from zipfile
        """
        from zipfile import ZipFile

        if conn is None:
            conn = DictConnector("pastas_db")

        if series_ext_json:
            ext = "json"
        else:
            ext = "pas"

        # short-circuit for PasConnector when zipfile was written using pas files
        if conn.conn_type == "pas" and not series_ext_json:
            with ZipFile(fname, "r") as archive:
                archive.extractall(conn.path)
            if storename is None:
                storename = conn.name
            return cls(conn, storename)

        with ZipFile(fname, "r") as archive:
            namelist = [
                fi
                for fi in archive.namelist()
                if not fi.endswith(f"_meta.{ext}") and not fi.endswith(os.sep)
            ]
            for f in tqdm(namelist, desc="Reading zip") if progressbar else namelist:
                if f.endswith("_meta.json"):
                    raise (
                        ValueError(
                            "The zipfile was created using pastastore <1.8.0."
                            " Please pass `series_ext_json=True` to `from_zip()`"
                        )
                    )
                libname, fjson = os.path.split(f)
                libname = os.path.split(libname)[-1]  # in case zip is one level deeper
                if libname in ["stresses", "oseries"]:
                    s = pd.read_json(archive.open(f), dtype=float, orient="columns")
                    if not isinstance(s.index, pd.DatetimeIndex):
                        s.index = pd.to_datetime(s.index, unit="ms")
                    s = s.sort_index()
                    meta = json.load(archive.open(f.replace(f".{ext}", f"_meta.{ext}")))
                    conn._add_series(libname, s, fjson.split(".")[0], metadata=meta)
                elif libname in ["models"]:
                    ml = json.load(archive.open(f), object_hook=pastas_hook)
                    conn.add_model(ml)
        if storename is None:
            storename = conn.name
        return cls(conn, storename)

    def search(
        self,
        s: Optional[Union[list, str]] = None,
        libname: Optional[PastasLibs] = None,
        case_sensitive: bool = True,
        sort=True,
    ):
        """Search for names of time series or models containing string `s`.

        Parameters
        ----------
        libname : str
            name of the library to search in
        s : str, lst
            find names with part of this string or strings in list
        case_sensitive : bool, optional
            whether search should be case sensitive, by default True
        sort : bool, optional
            sort list of names

        Returns
        -------
        matches : list
            list of names that match search result
        """
        if isinstance(s, str) and s in [None, "models", "stresses", "oseries"]:
            warnings.warn(
                "The order of arguments 's' and 'libname' has been flipped since v1.11."
                " Please update your code to use pstore.search(s=..., libname=...).",
                DeprecationWarning,
                stacklevel=2,
            )
            temp = str(s)  # create copy
            s = str(libname)  # create copy
            libname = temp
        if libname == "models":
            lib_names = {"models": self.model_names}
        elif libname == "stresses":
            lib_names = {"stresses": self.stresses_names}
        elif libname == "oseries":
            lib_names = {"oseries": self.oseries_names}
        elif libname is None:
            lib_names = {
                "oseries": self.oseries_names,
                "stresses": self.stresses_names,
                "models": self.model_names,
            }
        else:
            raise ValueError(
                "Provide valid libname: 'models', 'stresses', 'oseries' or None"
                " to seach within all libraries."
            )

        result = {}
        for lib, names in lib_names.items():
            if isinstance(s, str):
                if case_sensitive:
                    matches = [n for n in names if s in n]
                else:
                    matches = [n for n in names if s.lower() in n.lower()]
            elif isinstance(s, list):
                m = np.array([])
                for sub in s:
                    if case_sensitive:
                        m = np.append(m, [n for n in names if sub in n])
                    else:
                        m = np.append(m, [n for n in names if sub.lower() in n.lower()])
                matches = list(np.unique(m))
            else:
                raise TypeError("s must be str or list of str!")
            if sort:
                matches.sort()
            result[lib] = matches

        if len(result) == 1:
            return result[libname]
        else:
            return result

    def apply(
        self,
        libname: PastasLibs,
        func: callable,
        names: Optional[Union[str, List[str]]] = None,
        kwargs: Optional[dict] = None,
        progressbar: bool = True,
        parallel: bool = False,
        max_workers: Optional[int] = None,
        fancy_output: bool = True,
    ) -> Union[dict, pd.Series, pd.DataFrame]:
        """Apply function to items in library.

        Supported libraries are oseries, stresses, and models.

        Parameters
        ----------
        libname : str
            library name, supports "oseries", "stresses" and "models"
        func : callable
            function that accepts a string corresponding to the name of an item in
            the library as its first argument. Additional keyword arguments can be
            specified. The function can return any result, or update an item in the
            database without returning anything.
        names : str, list of str, optional
            apply function to these names, by default None which loops over all stored
            items in library
        kwargs : dict, optional
            keyword arguments to pass to func, by default None
        progressbar : bool, optional
            show progressbar, by default True
        parallel : bool, optional
            run apply in parallel, default is False.
        max_workers : int, optional
            max no. of workers, only used if parallel is True
        fancy_output : bool, optional
            if True, try returning result as pandas Series or DataFrame, by default
            False

        Returns
        -------
        dict
            dict of results of func, with names as keys and results as values

        Notes
        -----
        Users should be aware that parallel solving is platform dependent
        and may not always work. The current implementation works well for Linux users.
        For Windows users, parallel solving does not work when called directly from
        Jupyter Notebooks or IPython. To use parallel solving on Windows, the following
        code should be used in a Python file::

            from multiprocessing import freeze_support

            if __name__ == "__main__":
                freeze_support()
                pstore.apply("models", some_func, parallel=True)
        """
        names = self.conn.parse_names(names, libname)
        if kwargs is None:
            kwargs = {}
        if libname not in ("oseries", "stresses", "models"):
            raise ValueError(
                "'libname' must be one of ['oseries', 'stresses', 'models']!"
            )
        if parallel:
            result = self.conn._parallel(
                func,
                kwargs=kwargs,
                names=names,
                progressbar=progressbar,
                max_workers=max_workers,
                chunksize=None,
                desc=f"Computing {func.__name__} (parallel)",
            )
        else:
            result = []
            for n in tqdm(
                names, desc=f"Computing {func.__name__}", disable=not progressbar
            ):
                result.append(func(n, **kwargs))
        if fancy_output:
            return PastaStore._fancy_output(result, names, func.__name__)
        else:
            return result

    @staticmethod
    def _fancy_output(
        result: Iterable,
        names: List[str],
        label: Optional[str] = None,
    ) -> Union[pd.Series, pd.DataFrame, dict]:
        """Convert apply result to pandas Series, DataFrame or dict.

        Parameters
        ----------
        result : Iterable
            result of apply function
        names : list
            list of names
        label : str, optional
            label for columns, by default None

        Returns
        -------
        pd.Series, pd.DataFrame, dict
            Series, DataFrame or dict with results
        """
        if not isinstance(result, list):
            result = list(result)
        if isinstance(result[0], (float, int, np.integer)):
            return pd.Series(result, index=names)
        elif isinstance(result[0], (pd.Series, pd.DataFrame)):
            if isinstance(result[0], pd.Series):
                if names[0] == result[0].name:
                    df = pd.concat(result, axis=1)
                else:
                    df = pd.concat(dict(zip(names, result, strict=True)), axis=1)
            else:
                df = pd.concat(dict(zip(names, result, strict=True)), axis=1)
            if label is not None:
                df.columns.name = label
            return df
        elif result[0] is None:
            return None  # return None if first result is None?
        else:
            return dict(zip(names, result, strict=True))

    def within(
        self,
        extent: list,
        names: Optional[list[str]] = None,
        libname: PastasLibs = "oseries",
    ):
        """Get names of items within extent.

        Parameters
        ----------
        extent : list
            list with [xmin, xmax, ymin, ymax]
        names : str, list of str, optional
            list of names to include, by default None
        libname : str, optional
            name of library, must be one of ('oseries', 'stresses', 'models'), by
            default "oseries"

        Returns
        -------
        list
            list of items within extent
        """
        xmin, xmax, ymin, ymax = extent
        names = self.conn.parse_names(names, libname)
        if libname == "oseries":
            df = self.oseries.loc[names]
        elif libname == "stresses":
            df = self.stresses.loc[names]
        elif libname == "models":
            onames = np.unique(
                [
                    self.get_models(modelname, return_dict=True)["oseries"]["name"]
                    for modelname in names
                ]
            )
            df = self.oseries.loc[onames]
        else:
            raise ValueError(
                "libname must be one of ['oseries', 'stresses', 'models']"
                f", got '{libname}'"
            )
        mask = (
            (df["x"] <= xmax)
            & (df["x"] >= xmin)
            & (df["y"] >= ymin)
            & (df["y"] <= ymax)
        )
        return df.loc[mask].index.tolist()
