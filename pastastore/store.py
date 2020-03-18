from typing import Union, Tuple, Optional

from tqdm import tqdm
import numpy as np
import pandas as pd

import pastas as ps

FrameorSeriesUnion = Union[pd.DataFrame, pd.Series]


class PastaStore:
    """
    Pastas project for managing pastas timeseries and models.

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
        """
        Initialize PastaStore for managing pastas timeseries and models.

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

    def __repr__(self):
        """
        Representation string of the object.

        """
        return f"<PastasProject> {self.name}: \n - " + self.conn.__str__()

    def get_oseries_distances(self, names: Optional[Union[list, str]] = None) \
            -> FrameorSeriesUnion:
        """
        Method to obtain the distances in meters between the oseries.

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
                            n: int = 1) -> FrameorSeriesUnion:
        """
        Method to obtain the nearest (n) oseries.

        Parameters
        ----------
        names: str or list of str
            string or list of strings with the name(s) of the oseries
        n: int
            number of oseries to obtain

        Returns
        -------
        oseries:
            list with the names of the oseries.

        """

        distances = self.get_oseries_distances(names)

        data = pd.DataFrame(columns=np.arange(n))

        for series in distances.index:
            series = pd.Series(distances.loc[series].sort_values().index[:n],
                               name=series)
            data = data.append(series)
        return data

    def get_distances(self, oseries: Optional[Union[list, str]] = None,
                      stresses: Optional[Union[list, str]] = None,
                      kind: Optional[str] = None) -> FrameorSeriesUnion:
        """
        Method to obtain the distances in meters between the oseries and
        stresses.

        Parameters
        ----------
        oseries: str or list of str
            name(s) of the oseries
        stresses: str or list of str
            name(s) of the stresses
        kind: str
            string representing which kind of stresses to consider

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
            mask = stresses_df.kind == kind
            stresses = stresses_df.loc[stresses].loc[mask].index
        elif stresses is None:
            stresses = stresses_df.loc[stresses_df.kind == kind].index

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
                             kind: Optional[str] = None, n: int = 1) -> \
            FrameorSeriesUnion:
        """
        Method to obtain the nearest (n) stresses of a specific kind.

        Parameters
        ----------
        oseries: str
            string with the name of the oseries
        stresses: str or list of str
            string with the name of the stresses
        kind:
            string with the name of the stresses
        n: int
            number of stresses to obtain

        Returns
        -------
        stresses:
            list with the names of the stresses.

        """

        distances = self.get_distances(oseries, stresses, kind)

        data = pd.DataFrame(columns=np.arange(n))

        for series in distances.index:
            series = pd.Series(distances.loc[series].sort_values().index[:n],
                               name=series)
            data = data.append(series)
        return data

    def get_tmin_tmax(self, libname, names=None, progressbar=False):
        """
        Get tmin and tmax for timeseries.

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

        lib = self.conn.get_library(libname)
        names = self.conn._parse_names(names, libname=libname)
        tmintmax = pd.DataFrame(index=names, columns=["tmin", "tmax"],
                                dtype='datetime64[ns]')
        for n in (tqdm(names) if progressbar else names):
            s = lib.read(n)
            tmintmax.loc[n, "tmin"] = s.data.first_valid_index()
            tmintmax.loc[n, "tmax"] = s.data.last_valid_index()
        return tmintmax

    def create_model(self, name: str, add_recharge: bool = True) -> ps.Model:
        """
        Create a new pastas Model.

        Parameters
        ----------
        name : str
            name of the oseries to create a model for
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
        # get right column of data
        ts = self.conn._get_dataframe_values(
            "oseries", name, ts, metadata=meta)

        # convert to Timeseries and create model
        if not ts.dropna().empty:
            ts = ps.TimeSeries(ts, name=name, settings="oseries",
                               metadata=meta)
            ml = ps.Model(ts, name=name, metadata=meta)

            if add_recharge:
                self.add_recharge(ml)
            return ml
        else:
            raise ValueError("Empty timeseries!")

    def create_models(self, oseries: Optional[Union[list, str]] = None,
                      add_recharge: bool = True, store: bool = False,
                      solve: bool = False, progressbar: bool = True,
                      return_models: bool = False, ignore_errors: bool = True,
                      **kwargs) -> Union[Tuple[dict, list], list]:
        """
        Bulk creation of pastas models.

        Parameters
        ----------
        oseries : list of str, optional
            names of oseries to create models for, by default None,
            which creates models for all oseries
        add_recharge : bool, optional
            add recharge to the models based on closest
            precipitation and evaporation timeseries, by default True
        store : bool, optional
            store the model, by default False
        solve : bool, optional
            solve the model, by default False
        progressbar : bool, optional
            show progressbar, by default True
        return_models : bool, optional
            if True, return a list of models, by default False
        ignore_errors : bool, optional
            ignore errors while creating models, by default True

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
        errors = []
        for o in (tqdm(oseries) if progressbar else oseries):
            try:
                iml = self.create_model(o, add_recharge=add_recharge)
            except Exception as e:
                if ignore_errors:
                    errors.append(o)
                    continue
                else:
                    raise e
            if solve:
                iml.solve(**kwargs)
            if store:
                self.conn.add_model(iml, add_version=True)
            if return_models:
                models[o] = iml
        if return_models:
            return models, errors
        else:
            return errors

    def add_recharge(self, ml: ps.Model, rfunc=ps.Gamma) -> None:
        """
        Add recharge to a pastas model.

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

        """
        # get nearest prec and evap stns
        names = []
        for var in ("prec", "evap"):
            name = self.get_nearest_stresses(
                ml.oseries.name, kind=var).iloc[0, 0]
            # names.append(str(self.stresses.loc[name, "station"]))
            names.append(name)

        # get data
        tsdict = self.conn.get_stresses(names)
        stresses = []
        for k, s in tsdict.items():
            # TODO: two possible calls to retrieve metadata here
            # 1 for metadata
            # 2 if data is dataframe and data column needs to be found
            metadata = self.conn.get_metadata("stresses", k, as_frame=False)
            s = self.conn._get_dataframe_values("stresses", k, s,
                                                metadata=metadata)
            stresses.append(ps.TimeSeries(s, name=k, metadata=metadata))

        # add recharge to model
        rch = ps.StressModel2(stresses, rfunc, name="recharge",
                              metadata=[i.metadata for i in stresses],
                              settings=("prec", "evap"))
        ml.add_stressmodel(rch)

    def solve_models(self, mls: Optional[Union[ps.Model, list, str]] = None,
                     report: bool = False, ignore_solve_errors: bool = False,
                     store_result: bool = True, progressbar: bool = True,
                     **kwargs) -> None:
        """
        Solves the models in the store

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

        for ml_name in (tqdm(mls) if progressbar else mls):
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
                    self.conn.add_model(ml, add_version=True)
            except Exception as e:
                if ignore_solve_errors:
                    warning = "solve error ignored for -> {}".format(ml.name)
                    ps.logger.warning(warning)
                else:
                    raise e

    def model_results(self, mls: Optional[Union[ps.Model, list, str]] = None,
                      progressbar: bool = True):  # pragma: no cover
        """
        Get pastas model results.

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
            from art_tools import (
                pastas_get_model_results)
        except:
            raise ModuleNotFoundError(
                "You need 'art_tools' to use this method!")

        if mls is None:
            mls = self.conn.models
        elif isinstance(mls, ps.Model):
            mls = [mls.name]

        results_list = []
        for mlname in (tqdm(mls) if progressbar else mls):
            try:
                iml = self.conn.get_models(mlname)
            except Exception as e:
                print("{1}: '{0}' could not be parsed!".format(mlname, e))
                continue
            iresults = pastas_get_model_results(
                iml, par_selection='all', stats=('evp',), stderrors=True)
            results_list.append(iresults)

        return pd.concat(results_list, axis=1).transpose()
