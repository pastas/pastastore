import json
import os
import warnings
from typing import Optional, Tuple, Union

import numpy as np
import pandas as pd
import pastas as ps
from pastas.io.pas import pastas_hook
from tqdm import tqdm

from .util import _custom_warning

FrameorSeriesUnion = Union[pd.DataFrame, pd.Series]
warnings.showwarning = _custom_warning


class PastaStore:
    """Pastas project for managing pastas timeseries and models.

    Requires a Connector object to provide the interface to
    the database. Different Connectors are available, e.g.:

        - ArcticConnector for saving data to MongoDB using the Arctic module
        - PystoreConnector for saving data to disk using the Pystore module

    Parameters
    ----------
    name : str
        name of the project
    connector : Connector object
        object that provides the interface to the
        database, e.g. ArcticConnector (see pastastore.connectors)
    """

    def __init__(self, name: str, connector):
        """Initialize PastaStore for managing pastas timeseries and models.

        Parameters
        ----------
        name : str
            name of the project
        connector : Connector object
            object that provides the interface to the
            database
        """
        self.name = name
        self.conn = connector
        self._register_connector_methods()

    def _register_connector_methods(self):
        """Internal method for registering connector methods."""
        methods = [func for func in dir(self.conn) if
                   callable(getattr(self.conn, func)) and
                   not func.startswith("_")]
        for meth in methods:
            setattr(self, meth, getattr(self.conn, meth))

    def __repr__(self):
        """Representation string of the object."""
        return f"<PastaStore> {self.name}: \n - " + self.conn.__str__()

    def get_oseries_distances(self, names: Optional[Union[list, str]] = None) \
            -> FrameorSeriesUnion:
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

        distances = pd.DataFrame(np.sqrt((xh - xi) ** 2 + (yh - yi) ** 2),
                                 index=names, columns=other_df.index)

        return distances

    def get_nearest_oseries(self, names: Optional[Union[list, str]] = None,
                            n: int = 1, maxdist: Optional[float] = None) \
            -> FrameorSeriesUnion:
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

        for series in distances.index:
            series = pd.Series(
                distances.loc[series].dropna().sort_values().index[:n],
                name=series)
            data = data.append(series)
        return data

    def get_distances(self, oseries: Optional[Union[list, str]] = None,
                      stresses: Optional[Union[list, str]] = None,
                      kind: Optional[str] = None) -> FrameorSeriesUnion:
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

        distances = pd.DataFrame(np.sqrt((xh - xi) ** 2 + (yh - yi) ** 2),
                                 index=oseries, columns=stresses)

        return distances

    def get_nearest_stresses(self, oseries: Optional[Union[list, str]] = None,
                             stresses: Optional[Union[list, str]] = None,
                             kind: Optional[Union[list, str]] = None,
                             n: int = 1,
                             maxdist: Optional[float] = None) -> \
            FrameorSeriesUnion:
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
            series = pd.Series(
                distances.loc[series].dropna().sort_values().index[:n],
                name=series)
            data = data.append(series)
        return data

    def get_tmin_tmax(self, libname, names=None, progressbar=False):
        """Get tmin and tmax for timeseries.

        Parameters
        ----------
        libname : str
            name of the library containing the timeseries
            ('oseries' or 'stresses')
        names : str, list of str, or None, optional
            names of the timeseries, by default None which
            uses all the timeseries in the library
        progressbar : bool, optional
            show progressbar, by default False

        Returns
        -------
        tmintmax : pd.dataframe
            Dataframe containing tmin and tmax per timeseries
        """

        names = self.conn._parse_names(names, libname=libname)
        tmintmax = pd.DataFrame(index=names, columns=["tmin", "tmax"],
                                dtype='datetime64[ns]')
        desc = f"Get tmin/tmax {libname}"
        for n in (tqdm(names, desc=desc) if progressbar else names):
            if libname == "oseries":
                s = self.conn.get_oseries(n)
            else:
                s = self.conn.get_stresses(n)
            tmintmax.loc[n, "tmin"] = s.first_valid_index()
            tmintmax.loc[n, "tmax"] = s.last_valid_index()
        return tmintmax

    def get_parameters(self, parameters=None, modelnames=None,
                       param_value="optimal", progressbar=False):
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
        for mlname in (tqdm(modelnames, desc=desc)
                       if progressbar else modelnames):
            mldict = self.get_models(mlname, return_dict=True,
                                     progressbar=False)
            if parameters is None:
                pindex = mldict["parameters"].index
            else:
                pindex = parameters

            for c in pindex:
                p.loc[mlname, c] = \
                    mldict["parameters"].loc[c, param_value]

        p = p.squeeze()
        return p.astype(float)

    def get_statistics(self, statistics, modelnames=None, progressbar=False,
                       **kwargs):
        """Get model statistics.

        Parameters
        ----------
        statistics : list of str
            list of statistics to calculate, e.g. ["evp", "rsq", "rmse"], for
            a full list see `pastas.modelstats.Statistics.ops`.
        modelnames : list of str, optional
            modelnames to calculates statistics for, by default None, which
            uses all models in the store
        progressbar : bool, optional
            show progressbar, by default False
        **kwargs
            any arguments that can be passed to the methods for calculating
            statistics

        Returns
        -------
        s : pandas.DataFrame
        """

        modelnames = self.conn._parse_names(modelnames, libname="models")

        # create dataframe for results
        s = pd.DataFrame(index=modelnames, columns=statistics)

        # loop through model names
        desc = "Get model statistics"
        for mlname in (tqdm(modelnames, desc=desc)
                       if progressbar else modelnames):
            ml = self.get_models(mlname, progressbar=False)
            for stat in statistics:
                value = ml.stats.__getattribute__(stat)(**kwargs)
                s.loc[mlname, stat] = value

        s = s.squeeze()
        return s.astype(float)

    def create_model(self, name: str, modelname: str = None,
                     add_recharge: bool = True) -> ps.Model:
        """Create a pastas Model.

        Parameters
        ----------
        name : str
            name of the oseries to create a model for
        modelname : str, optional
            name of the model, default is None, which uses oseries name
        add_recharge : bool, optional
            add recharge to the model by looking for the closest
            precipitation and evaporation timeseries in the stresses
            library, by default True

        Returns
        -------
        pastas.Model
            model for the oseries

        Raises
        ------
        KeyError
            if data is stored as dataframe and no column is provided
        ValueError
            if timeseries is empty
        """
        # get oseries metadata
        meta = self.conn.get_metadata("oseries", name, as_frame=False)
        ts = self.conn.get_oseries(name)

        # convert to Timeseries and create model
        if not ts.dropna().empty:
            ts = ps.TimeSeries(ts, name=name, settings="oseries",
                               metadata=meta)
            if modelname is None:
                modelname = name
            ml = ps.Model(ts, name=modelname, metadata=meta)

            if add_recharge:
                self.add_recharge(ml)
            return ml
        else:
            raise ValueError("Empty timeseries!")

    def create_models_bulk(self, oseries: Optional[Union[list, str]] = None,
                           add_recharge: bool = True, store: bool = True,
                           solve: bool = False, progressbar: bool = True,
                           return_models: bool = False,
                           ignore_errors: bool = False,
                           **kwargs) -> Union[Tuple[dict, dict], dict]:
        """Bulk creation of pastas models.

        Parameters
        ----------
        oseries : list of str, optional
            names of oseries to create models for, by default None,
            which creates models for all oseries
        add_recharge : bool, optional
            add recharge to the models based on closest
            precipitation and evaporation timeseries, by default True
        store : bool, optional
            store the model, by default True
        solve : bool, optional
            solve the model, by default False
        progressbar : bool, optional
            show progressbar, by default True
        return_models : bool, optional
            if True, return a list of models, by default False
        ignore_errors : bool, optional
            ignore errors while creating models, by default False

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
        for o in (tqdm(oseries, desc=desc) if progressbar else oseries):
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
            if store:
                self.conn.add_model(iml, overwrite=True)
            if return_models:
                models[o] = iml
        if len(errors) > 0:
            print("Warning! Errors occurred while creating models!")
        if return_models:
            return models, errors
        else:
            return errors

    def add_recharge(self, ml: ps.Model, rfunc=ps.Gamma,
                     recharge=ps.rch.Linear()) -> None:
        """Add recharge to a pastas model.

        Uses closest precipitation and evaporation timeseries in database.
        These are assumed to be labeled with kind = 'prec' or 'evap'.

        Parameters
        ----------
        ml : pastas.Model
            pastas.Model object
        rfunc : pastas.rfunc, optional
            response function to use for recharge in model,
            by default ps.Gamma (for different response functions, see
            pastas documentation)
        recharge : ps.RechargeModel
            recharge model to use, default is ps.rch.Linear()
        """
        # get nearest prec and evap stns
        names = []
        for var in ("prec", "evap"):
            try:
                name = self.get_nearest_stresses(
                    ml.oseries.name, kind=var).iloc[0, 0]
            except AttributeError:
                msg = "No precipitation or evaporation timeseries found!"
                raise Exception(msg)
            if isinstance(name, float):
                if np.isnan(name):
                    raise ValueError(f"Unable to find nearest '{var}' stress! "
                                     "Check X and Y coordinates.")
            else:
                names.append(name)
        if len(names) == 0:
            msg = "No precipitation or evaporation timeseries found!"
            raise Exception(msg)

        # get data
        tsdict = self.conn.get_stresses(names)
        stresses = []
        for (k, s), setting in zip(tsdict.items(), ("prec", "evap")):
            metadata = self.conn.get_metadata("stresses", k, as_frame=False)
            stresses.append(ps.TimeSeries(s, name=k, settings=setting,
                                          metadata=metadata))

        # add recharge to model
        rch = ps.RechargeModel(stresses[0], stresses[1], rfunc,
                               name="recharge", recharge=recharge,
                               settings=("prec", "evap"),
                               metadata=[i.metadata for i in stresses])
        ml.add_stressmodel(rch)

    def solve_models(self, mls: Optional[Union[ps.Model, list, str]] = None,
                     report: bool = False, ignore_solve_errors: bool = False,
                     store_result: bool = True, progressbar: bool = True,
                     **kwargs) -> None:
        """Solves the models in the store.

        Parameters
        ----------
        mls : list of str, optional
            list of model names, if None all models in the project
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
            mls = self.conn.models
        elif isinstance(mls, ps.Model):
            mls = [mls.name]

        desc = "Solving models"
        for ml_name in (tqdm(mls, desc=desc) if progressbar else mls):
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

    def model_results(self, mls: Optional[Union[ps.Model, list, str]] = None,
                      progressbar: bool = True):  # pragma: no cover
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
            raise ModuleNotFoundError(
                "You need 'art_tools' to use this method!")

        if mls is None:
            mls = self.conn.models
        elif isinstance(mls, ps.Model):
            mls = [mls.name]

        results_list = []
        desc = "Get model results"
        for mlname in (tqdm(mls, desc=desc) if progressbar else mls):
            try:
                iml = self.conn.get_models(mlname)
            except Exception as e:
                print("{1}: '{0}' could not be parsed!".format(mlname, e))
                continue
            iresults = pastas_get_model_results(
                iml, par_selection='all', stats=('evp',), stderrors=True)
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
            raise FileExistsError("File already exists! "
                                  "Use 'overwrite=True' to "
                                  "force writing file.")
        elif os.path.exists(fname):
            warnings.warn(f"Overwriting file '{os.path.basename(fname)}'")

        with ZipFile(fname, "w", compression=ZIP_DEFLATED) as archive:
            # oseries
            self.conn._series_to_archive(archive, "oseries",
                                         progressbar=progressbar)
            # stresses
            self.conn._series_to_archive(archive, "stresses",
                                         progressbar=progressbar)
            # models
            self.conn._models_to_archive(archive, progressbar=progressbar)

    def export_model_series_to_csv(self,
                                   names: Optional[Union[list, str]] = None,
                                   exportdir: str = ".",
                                   exportmeta: bool = True):
        """Export model timeseries to csv files.

        Parameters
        ----------
        names : Optional[Union[list, str]], optional
            names of models to export, by default None, which uses retrieves
            all models from database
        exportdir : str, optional
            directory to export csv files to, default is current directory
        exportmeta : bool, optional
            export metadata for all timeseries as csv file, default is True
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
                        ts.to_csv(os.path.join(
                            exportdir, f"{stress_name}.csv"))
                        if exportmeta:
                            tsmeta = self.get_metadata("stresses", stress_name)
                            metalist.append(tsmeta)
                else:
                    for istress in mldict["stressmodels"][sm]["stress"]:
                        stress_name = istress["name"]
                        ts = self.get_stresses(stress_name)
                        ts.to_csv(os.path.join(
                            exportdir, f"{stress_name}.csv"))
                        if exportmeta:
                            tsmeta = self.get_metadata("stresses", stress_name)
                            metalist.append(tsmeta)

            if exportmeta:
                pd.concat(metalist, axis=0).to_csv(
                    os.path.join(exportdir, f"metadata_{name}.csv"))

    @ classmethod
    def from_zip(cls, fname: str, conn, storename: Optional[str] = None):
        """Load PastaStore from zipfile.

        Parameters
        ----------
        fname : str
            pathname of zipfile
        conn : Connector object
            connector for storing loaded data
        storename : str, optional
            name of the PastaStore, by default None, which
            defaults to the name of the Connector.

        Returns
        -------
        pastastore.PastaStore
            return PastaStore containing data from zipfile
        """
        from zipfile import ZipFile
        with ZipFile(fname, "r") as archive:
            namelist = [fi for fi in archive.namelist()
                        if not fi.endswith("_meta.json")]
            for f in namelist:
                libname, fjson = os.path.split(f)
                if libname in ["stresses", "oseries"]:
                    s = pd.read_json(archive.open(f),
                                     orient="columns")
                    if not isinstance(s.index, pd.DatetimeIndex):
                        s.index = pd.to_datetime(s.index, unit='ms')
                        s = s.sort_index()
                    meta = json.load(archive.open(
                        f.replace(".json", "_meta.json")))
                    conn._add_series(libname, s, fjson.split(".")[0],
                                     metadata=meta)
                elif libname in ["models"]:
                    ml = json.load(archive.open(f), object_hook=pastas_hook)
                    conn.add_model(ml)
        if storename is None:
            storename = conn.name
        return cls(storename, conn)

    @ property
    def oseries(self):
        return self.conn.oseries

    @ property
    def stresses(self):
        return self.conn.stresses

    @ property
    def models(self):
        return self.conn.models
