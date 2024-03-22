import json
import os
import warnings
from typing import List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import pastas as ps
from packaging.version import parse as parse_version
from pastas.io.pas import pastas_hook
from tqdm.auto import tqdm

from pastastore.base import BaseConnector
from pastastore.connectors import DictConnector
from pastastore.plotting import Maps, Plots
from pastastore.util import _custom_warning
from pastastore.yaml_interface import PastastoreYAML

FrameorSeriesUnion = Union[pd.DataFrame, pd.Series]
warnings.showwarning = _custom_warning


class PastaStore:
    """PastaStore object for managing pastas time series and models.

    Requires a Connector object to provide the interface to
    the database. Different Connectors are available, e.g.:

    - PasConnector for storing all data as .pas (JSON) files on disk (recommended)
    - DictConenctor for storing all data in dictionaries (in-memory)
    - ArcticConnector for saving data to MongoDB using the Arctic module
    - PystoreConnector for saving data to disk using the Pystore module

    Parameters
    ----------
    connector : Connector object
        object that provides the interface to the
        database, e.g. ArcticConnector (see pastastore.connectors)
    name : str, optional
        name of the PastaStore, by default takes the name of the Connector object
    """

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
        self.name = name if name is not None else self.conn.name
        self._register_connector_methods()

        # register map, plot and yaml classes
        self.maps = Maps(self)
        self.plots = Plots(self)
        self.yaml = PastastoreYAML(self)

    def _register_connector_methods(self):
        """Internal method for registering connector methods."""
        methods = [
            func
            for func in dir(self.conn)
            if callable(getattr(self.conn, func)) and not func.startswith("_")
        ]
        for meth in methods:
            setattr(self, meth, getattr(self.conn, meth))

    @property
    def oseries(self):
        return self.conn.oseries

    @property
    def stresses(self):
        return self.conn.stresses

    @property
    def models(self):
        return self.conn.models

    @property
    def oseries_names(self):
        return self.conn.oseries_names

    @property
    def stresses_names(self):
        return self.conn.stresses_names

    @property
    def model_names(self):
        return self.conn.model_names

    @property
    def _modelnames_cache(self):
        return self.conn._modelnames_cache

    @property
    def n_oseries(self):
        return self.conn.n_oseries

    @property
    def n_stresses(self):
        return self.conn.n_stresses

    @property
    def n_models(self):
        return self.conn.n_models

    @property
    def oseries_models(self):
        return self.conn.oseries_models

    @property
    def oseries_with_models(self):
        return self.conn.oseries_with_models

    def __repr__(self):
        """Representation string of the object."""
        return f"<PastaStore> {self.name}: \n - " + self.conn.__str__()

    def get_oseries_distances(
        self, names: Optional[Union[list, str]] = None
    ) -> FrameorSeriesUnion:
        """Method to obtain the distances in meters between the oseries.

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

        names = self.conn._parse_names(names)

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
    ) -> FrameorSeriesUnion:
        """Method to obtain the nearest (n) oseries.

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
    ) -> FrameorSeriesUnion:
        """Method to obtain the distances in meters between the oseries and
        stresses.

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

        oseries = self.conn._parse_names(oseries)

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
    ) -> FrameorSeriesUnion:
        """Method to obtain the nearest (n) stresses of a specific kind.

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
        signatures=None,
        names=None,
        libname="oseries",
        progressbar=False,
        ignore_errors=False,
    ):
        """Get groundwater signatures. NaN-values are returned when the
        signature could not be computed.

        Parameters
        ----------
        signatures : list of str, optional
            list of groundwater signatures to compute, if None all groundwater
            signatures in ps.stats.signatures.__all__ are used, by default None
        names : str, list of str, or None, optional
            names of the time series, by default None which
            uses all the time series in the library
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
        signatures_df : pandas.DataFrame
            DataFrame containing the signatures (columns) per time series (rows)
        """
        names = self.conn._parse_names(names, libname=libname)

        if signatures is None:
            signatures = ps.stats.signatures.__all__.copy()

        # create dataframe for results
        signatures_df = pd.DataFrame(index=names, columns=signatures, data=np.nan)

        # loop through oseries names
        desc = "Get groundwater signatures"
        for name in tqdm(names, desc=desc) if progressbar else names:
            try:
                if libname == "oseries":
                    s = self.conn.get_oseries(name)
                else:
                    s = self.conn.get_stresses(name)
            except Exception as e:
                if ignore_errors:
                    signatures_df.loc[name, :] = np.nan
                    continue
                else:
                    raise e

            try:
                i_signatures = ps.stats.signatures.summary(s.squeeze(), signatures)
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
            signatures_df.loc[name, signatures] = i_signatures.squeeze()

        return signatures_df

    def get_tmin_tmax(self, libname, names=None, progressbar=False):
        """Get tmin and tmax for time series.

        Parameters
        ----------
        libname : str
            name of the library containing the time series
            ('oseries' or 'stresses')
        names : str, list of str, or None, optional
            names of the time series, by default None which
            uses all the time series in the library
        progressbar : bool, optional
            show progressbar, by default False

        Returns
        -------
        tmintmax : pd.dataframe
            Dataframe containing tmin and tmax per time series
        """

        names = self.conn._parse_names(names, libname=libname)
        tmintmax = pd.DataFrame(
            index=names, columns=["tmin", "tmax"], dtype="datetime64[ns]"
        )
        desc = f"Get tmin/tmax {libname}"
        for n in tqdm(names, desc=desc) if progressbar else names:
            if libname == "oseries":
                s = self.conn.get_oseries(n)
            else:
                s = self.conn.get_stresses(n)
            tmintmax.loc[n, "tmin"] = s.first_valid_index()
            tmintmax.loc[n, "tmax"] = s.last_valid_index()
        return tmintmax

    def get_extent(self, libname, names=None, buffer=0.0):
        names = self.conn._parse_names(names, libname=libname)
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
        ignore_errors: Optional[bool] = False,
    ) -> FrameorSeriesUnion:
        """Get model parameters. NaN-values are returned when the parameters
        are not present in the model or the model is not optimized.

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
            encountered in modelnames, by default False

        Returns
        -------
        p : pandas.DataFrame
            DataFrame containing the parameters (columns) per model (rows)
        """
        modelnames = self.conn._parse_names(modelnames, libname="models")

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
                p.loc[mlname, c] = mldict["parameters"].loc[c, param_value]

        p = p.squeeze()
        return p.astype(float)

    def get_statistics(
        self,
        statistics: Union[str, List[str]],
        modelnames: Optional[List[str]] = None,
        progressbar: Optional[bool] = False,
        ignore_errors: Optional[bool] = False,
        **kwargs,
    ) -> FrameorSeriesUnion:
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
        **kwargs
            any arguments that can be passed to the methods for calculating
            statistics

        Returns
        -------
        s : pandas.DataFrame
        """

        modelnames = self.conn._parse_names(modelnames, libname="models")

        # if statistics is str
        if isinstance(statistics, str):
            statistics = [statistics]

        # create dataframe for results
        s = pd.DataFrame(index=modelnames, columns=statistics, data=np.nan)

        # loop through model names
        desc = "Get model statistics"
        for mlname in tqdm(modelnames, desc=desc) if progressbar else modelnames:
            try:
                ml = self.get_models(mlname, progressbar=False)
            except Exception as e:
                if ignore_errors:
                    continue
                else:
                    raise e
            for stat in statistics:
                value = ml.stats.__getattribute__(stat)(**kwargs)
                s.loc[mlname, stat] = value

        s = s.squeeze()
        return s.astype(float)

    def create_model(
        self,
        name: str,
        modelname: str = None,
        add_recharge: bool = True,
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

        # convert to Timeseries and create model
        if not ts.dropna().empty:
            if modelname is None:
                modelname = name
            ml = ps.Model(ts, name=modelname, metadata=meta)
            if add_recharge:
                self.add_recharge(ml, recharge_name=recharge_name)
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
                iml = self.create_model(o, add_recharge=add_recharge)
            except Exception as e:
                if ignore_errors:
                    errors[o] = e
                    continue
                else:
                    raise e
            if solve:
                iml.solve(**kwargs)
            if store_models:
                self.conn.add_model(iml, overwrite=True)
            else:
                models[o] = iml
        if len(errors) > 0:
            print("Warning! Errors occurred while creating models!")
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
        # get nearest prec and evap stns
        if "prec" not in self.stresses.kind.values:
            raise ValueError(
                "No stresses with kind='prec' found in store. "
                "add_recharge() requires stresses with kind='prec'!"
            )
        if "evap" not in self.stresses.kind.values:
            raise ValueError(
                "No stresses with kind='evap' found in store. "
                "add_recharge() requires stresses with kind='evap'!"
            )
        names = []
        for var in ("prec", "evap"):
            try:
                name = self.get_nearest_stresses(ml.oseries.name, kind=var).iloc[0, 0]
            except AttributeError:
                msg = "No precipitation or evaporation time series found!"
                raise Exception(msg)
            if isinstance(name, float):
                if np.isnan(name):
                    raise ValueError(
                        f"Unable to find nearest '{var}' stress! "
                        "Check x and y coordinates."
                    )
            else:
                names.append(name)
        if len(names) == 0:
            msg = "No precipitation or evaporation time series found!"
            raise Exception(msg)

        # get data
        tsdict = self.conn.get_stresses(names)
        metadata = self.conn.get_metadata("stresses", names, as_frame=False)
        # add recharge to model
        rch = ps.RechargeModel(
            tsdict[names[0]],
            tsdict[names[1]],
            rfunc=rfunc,
            name=recharge_name,
            recharge=recharge,
            settings=("prec", "evap"),
            metadata=metadata,
        )
        ml.add_stressmodel(rch)

    def solve_models(
        self,
        mls: Optional[Union[ps.Model, list, str]] = None,
        report: bool = False,
        ignore_solve_errors: bool = False,
        store_result: bool = True,
        progressbar: bool = True,
        **kwargs,
    ) -> None:
        """Solves the models in the store.

        Parameters
        ----------
        mls : list of str, optional
            list of model names, if None all models in the pastastore
            are solved.
        report : boolean, optional
            determines if a report is printed when the model is solved,
            default is False
        ignore_solve_errors : boolean, optional
            if True, errors emerging from the solve method are ignored,
            default is False which will raise an exception when a model
            cannot be optimized
        store_result : bool, optional
            if True save optimized models, default is True
        progressbar : bool, optional
            show progressbar, default is True
        **kwargs :
            arguments are passed to the solve method.
        """
        if mls is None:
            mls = self.conn.model_names
        elif isinstance(mls, ps.Model):
            mls = [mls.name]

        desc = "Solving models"
        for ml_name in tqdm(mls, desc=desc) if progressbar else mls:
            ml = self.conn.get_models(ml_name)

            m_kwargs = {}
            for key, value in kwargs.items():
                if isinstance(value, pd.Series):
                    m_kwargs[key] = value.loc[ml_name]
                else:
                    m_kwargs[key] = value
            # Convert timestamps
            for tstamp in ["tmin", "tmax"]:
                if tstamp in m_kwargs:
                    m_kwargs[tstamp] = pd.Timestamp(m_kwargs[tstamp])

            try:
                ml.solve(report=report, **m_kwargs)
                if store_result:
                    self.conn.add_model(ml, overwrite=True)
            except Exception as e:
                if ignore_solve_errors:
                    warning = "solve error ignored for -> {}".format(ml.name)
                    ps.logger.warning(warning)
                else:
                    raise e

    def model_results(
        self,
        mls: Optional[Union[ps.Model, list, str]] = None,
        progressbar: bool = True,
    ):  # pragma: no cover
        """Get pastas model results.

        Parameters
        ----------
        mls : list of str, optional
            list of model names, by default None which means results for
            all models will be calculated
        progressbar : bool, optional
            show progressbar, by default True

        Returns
        -------
        results : pd.DataFrame
            dataframe containing parameters and other statistics
            for each model

        Raises
        ------
        ModuleNotFoundError
            if the art_tools module is not available
        """
        try:
            from art_tools import pastas_get_model_results
        except Exception:
            raise ModuleNotFoundError("You need 'art_tools' to use this method!")

        if mls is None:
            mls = self.conn.models
        elif isinstance(mls, ps.Model):
            mls = [mls.name]

        results_list = []
        desc = "Get model results"
        for mlname in tqdm(mls, desc=desc) if progressbar else mls:
            try:
                iml = self.conn.get_models(mlname)
            except Exception as e:
                print("{1}: '{0}' could not be parsed!".format(mlname, e))
                continue
            iresults = pastas_get_model_results(
                iml, par_selection="all", stats=("evp",), stderrors=True
            )
            results_list.append(iresults)

        return pd.concat(results_list, axis=1).transpose()

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

        if os.path.exists(fname) and not overwrite:
            raise FileExistsError(
                "File already exists! " "Use 'overwrite=True' to " "force writing file."
            )
        elif os.path.exists(fname):
            warnings.warn(f"Overwriting file '{os.path.basename(fname)}'")

        with ZipFile(fname, "w", compression=ZIP_DEFLATED) as archive:
            # oseries
            self.conn._series_to_archive(archive, "oseries", progressbar=progressbar)
            # stresses
            self.conn._series_to_archive(archive, "stresses", progressbar=progressbar)
            # models
            self.conn._models_to_archive(archive, progressbar=progressbar)

    def export_model_series_to_csv(
        self,
        names: Optional[Union[list, str]] = None,
        exportdir: str = ".",
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
        names = self.conn._parse_names(names, libname="models")
        for name in names:
            mldict = self.get_models(name, return_dict=True)

            oname = mldict["oseries"]["name"]
            o = self.get_oseries(oname)
            o.to_csv(os.path.join(exportdir, f"{oname}.csv"))

            if exportmeta:
                metalist = [self.get_metadata("oseries", oname)]

            for sm in mldict["stressmodels"]:
                if mldict["stressmodels"][sm]["stressmodel"] == "RechargeModel":
                    for istress in ["prec", "evap"]:
                        istress = mldict["stressmodels"][sm][istress]
                        stress_name = istress["name"]
                        ts = self.get_stresses(stress_name)
                        ts.to_csv(os.path.join(exportdir, f"{stress_name}.csv"))
                        if exportmeta:
                            tsmeta = self.get_metadata("stresses", stress_name)
                            metalist.append(tsmeta)
                else:
                    for istress in mldict["stressmodels"][sm]["stress"]:
                        stress_name = istress["name"]
                        ts = self.get_stresses(stress_name)
                        ts.to_csv(os.path.join(exportdir, f"{stress_name}.csv"))
                        if exportmeta:
                            tsmeta = self.get_metadata("stresses", stress_name)
                            metalist.append(tsmeta)

            if exportmeta:
                pd.concat(metalist, axis=0).to_csv(
                    os.path.join(exportdir, f"metadata_{name}.csv")
                )

    @classmethod
    def from_zip(
        cls,
        fname: str,
        conn: Optional[BaseConnector] = None,
        storename: Optional[str] = None,
        progressbar: bool = True,
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

        Returns
        -------
        pastastore.PastaStore
            return PastaStore containing data from zipfile
        """
        from zipfile import ZipFile

        if conn is None:
            conn = DictConnector("pastas_db")

        with ZipFile(fname, "r") as archive:
            namelist = [
                fi for fi in archive.namelist() if not fi.endswith("_meta.json")
            ]
            for f in tqdm(namelist, desc="Reading zip") if progressbar else namelist:
                libname, fjson = os.path.split(f)
                if libname in ["stresses", "oseries"]:
                    s = pd.read_json(archive.open(f), dtype=float, orient="columns")
                    if not isinstance(s.index, pd.DatetimeIndex):
                        s.index = pd.to_datetime(s.index, unit="ms")
                    s = s.sort_index()
                    meta = json.load(archive.open(f.replace(".json", "_meta.json")))
                    conn._add_series(libname, s, fjson.split(".")[0], metadata=meta)
                elif libname in ["models"]:
                    ml = json.load(archive.open(f), object_hook=pastas_hook)
                    conn.add_model(ml)
        if storename is None:
            storename = conn.name
        return cls(conn, storename)

    def search(
        self,
        libname: str,
        s: Optional[Union[list, str]] = None,
        case_sensitive: bool = True,
        sort=True,
    ):
        """Search for names of time series or models starting with `s`.

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

        if libname == "models":
            lib_names = getattr(self, "model_names")
        elif libname == "stresses":
            lib_names = getattr(self, "stresses_names")
        elif libname == "oseries":
            lib_names = getattr(self, "oseries_names")
        else:
            raise ValueError("Provide valid libname: 'models', 'stresses' or 'oseries'")

        if isinstance(s, str):
            if case_sensitive:
                matches = [n for n in lib_names if s in n]
            else:
                matches = [n for n in lib_names if s.lower() in n.lower()]
        if isinstance(s, list):
            m = np.array([])
            for sub in s:
                if case_sensitive:
                    m = np.append(m, [n for n in lib_names if sub in n])
                else:
                    m = np.append(m, [n for n in lib_names if sub.lower() in n.lower()])
            matches = list(np.unique(m))
        if sort:
            matches.sort()
        return matches

    def get_model_timeseries_names(
        self,
        modelnames: Optional[Union[list, str]] = None,
        dropna: bool = True,
        progressbar: bool = True,
    ) -> FrameorSeriesUnion:
        """Get time series names contained in model.

        Parameters
        ----------
        modelnames : Optional[Union[list, str]], optional
            list or name of models to get time series names for,
            by default None which will use all modelnames
        dropna : bool, optional
            drop stresses from table if stress is not included in any
            model, by default True
        progressbar : bool, optional
            show progressbar, by default True

        Returns
        -------
        structure : pandas.DataFrame
            returns DataFrame with oseries name per model, and a flag
            indicating whether a stress is contained within a time series
            model.
        """

        model_names = self.conn._parse_names(modelnames, libname="models")
        structure = pd.DataFrame(
            index=model_names, columns=["oseries"] + self.stresses_names
        )

        for mlnam in (
            tqdm(model_names, desc="Get model time series names")
            if progressbar
            else model_names
        ):
            iml = self.get_models(mlnam, return_dict=True)

            PASFILE_LEQ_022 = parse_version(
                iml["file_info"]["pastas_version"]
            ) <= parse_version("0.22.0")

            # oseries
            structure.loc[mlnam, "oseries"] = iml["oseries"]["name"]

            for sm in iml["stressmodels"].values():
                class_key = "stressmodel" if PASFILE_LEQ_022 else "class"
                if sm[class_key] == "RechargeModel":
                    pnam = sm["prec"]["name"]
                    enam = sm["evap"]["name"]
                    structure.loc[mlnam, pnam] = 1
                    structure.loc[mlnam, enam] = 1
                elif "stress" in sm:
                    smstress = sm["stress"]
                    if isinstance(smstress, dict):
                        smstress = [smstress]
                    for s in smstress:
                        structure.loc[mlnam, s["name"]] = 1
        if dropna:
            return structure.dropna(how="all", axis=1)
        else:
            return structure

    def apply(self, libname, func, names=None, progressbar=True):
        """Apply function to items in library.

        Supported libraries are oseries, stresses, and models.

        Parameters
        ----------
        libname : str
            library name, supports "oseries", "stresses" and "models"
        func : callable
            function that accepts items from one of the supported libraries as input
        names : str, list of str, optional
            apply function to these names, by default None which loops over all stored
            items in library
        progressbar : bool, optional
            show progressbar, by default True

        Returns
        -------
        dict
            dict of results of func, with names as keys and results as values
        """
        names = self.conn._parse_names(names, libname)
        result = {}
        if libname not in ("oseries", "stresses", "models"):
            raise ValueError(
                "'libname' must be one of ['oseries', 'stresses', 'models']!"
            )
        getter = getattr(self.conn, f"get_{libname}")
        for n in (
            tqdm(names, desc=f"Applying {func.__name__}") if progressbar else names
        ):
            result[n] = func(getter(n))
        return result

    def within(self, extent, names=None, libname="oseries"):
        xmin, xmax, ymin, ymax = extent
        names = self.conn._parse_names(names, libname)
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
