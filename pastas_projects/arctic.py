import functools
import sys
from collections.abc import Iterable

import arctic
import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import Point
from tqdm import tqdm

import pastas as ps

from .base import BaseProject
from .util import query_yes_no


class ArcticPastas(BaseProject):
    """Class that connects to a running Mongo database that can be used
    to store timeseries and models for Pastas projects.

    Parameters
    ----------
    connstr : str
        connection string
    projectname : name
        name of the project

    """

    def __init__(self, connstr, projectname):
        """Create an ArcticPastas object that connects to a
        running Mongo database.

        Parameters
        ----------
        connstr : str
            connection string
        projectname : name
            name of the project

        """
        self.connstr = connstr
        self.projectname = projectname

        self.arc = arctic.Arctic(connstr)
        self._initialize()

    def __repr__(self):
        """representation string of the object.
        """
        noseries = len(self.lib_oseries.list_symbols())
        nstresses = len(self.lib_stresses.list_symbols())
        nmodels = len(self.lib_models.list_symbols())
        return "<ArcticPastas object> '{0}': {1} oseries, {2} stresses, {3} models".format(
            self.projectname, noseries, nstresses, nmodels
        )

    def _initialize(self):
        """internal method to initalize the libraries.
        """
        for libname in ["oseries", "stresses", "models"]:
            if self.projectname + "." + libname not in self.arc.list_libraries():
                self.arc.initialize_library(self.projectname + "." + libname)
            setattr(self, "lib_" + libname.replace(".", ""),
                    self.get_library(libname))

    def get_library(self, libname):
        """Get Arctic library handle

        Parameters
        ----------
        libname : str
            name of the library

        Returns
        -------
        Arctic.library handle
            handle to the library

        """
        # get library
        lib = self.arc.get_library(self.projectname + "." + libname)
        return lib

    def _add_series(self, libname, series, name, metadata=None,
                    add_version=False):
        """internal method to add series to database

        Parameters
        ----------
        libname : str
            name of the library to add the series to
        series : pandas.Series or pandas.DataFrame
            data to add
        name : str
            name of the timeseries
        metadata : dict, optional
            dictionary containing metadata, by default None
        add_version : bool, optional
            if True, add a new version of the dataset to the database,
            by default False

        Raises
        ------
        Exception
            if add_version is False and name is already in the database
            raises an Exception.

        """
        lib = self.get_library(libname)
        if name not in lib.list_symbols() or add_version:
            lib.write(name, series, metadata=metadata)
        else:
            raise Exception("Item with name '{0}' already"
                            " in '{1}' library!".format(name, libname))

    def add_oseries(self, series, name, metadata=None, add_version=False):
        """add oseries to the database

        Parameters
        ----------
        series : pandas.Series or pandas.DataFrame
            data to add
        name : str
            name of the timeseries
        metadata : dict, optional
            dictionary containing metadata, by default None
        add_version : bool, optional
            if True, add a new version of the dataset to the database,
            by default False

        """
        self._add_series("oseries", series, name=name,
                         metadata=metadata, add_version=add_version)
        ArcticPastas.oseries.fget.cache_clear()

    def add_stress(self, series, name, kind, metadata=None, add_version=False):
        """add stress to the database

        Parameters
        ----------
        series : pandas.Series or pandas.DataFrame
            data to add
        name : str
            name of the timeseries
        kind : str
            category to identify type of stress, this label is added to the
            metadata dictionary.
        metadata : dict, optional
            dictionary containing metadata, by default None
        add_version : bool, optional
            if True, add a new version of the dataset to the database,
            by default False

        """
        if metadata is None:
            metadata = {}

        metadata["kind"] = kind
        self._add_series("stresses", series, name=name,
                         metadata=metadata, add_version=add_version)
        ArcticPastas.stresses.fget.cache_clear()

    def add_model(self, ml, add_version=False):
        """add model to the database

        Parameters
        ----------
        ml : pastas.Model
            pastas Model to add to the database
        add_version : bool, optional
            if True, add new version of existing model, by default False

        Raises
        ------
        Exception
            if add_version is False and model is already in the database
            raises an Exception.

        """
        lib = self.lib_models
        if ml.name not in lib.list_symbols() or add_version:
            mldict = ml.to_dict(series=False)
            lib.write(ml.name, mldict, metadata=ml.oseries.metadata)
        else:
            raise Exception("Model with name '{}' already in store!".format(
                ml.name))
        ArcticPastas.models.fget.cache_clear()

    def del_model(self, name):
        """delete model from the database

        Parameters
        ----------
        name : str
            name of the model to delete
        """
        self.lib_models.delete(name)

    def del_oseries(self, name):
        """delete oseries from the database

        Parameters
        ----------
        name : str
            name of the oseries to delete
        """
        self.lib_oseries.delete(name)
        ArcticPastas.oseries.fget.cache_clear()

    def del_stress(self, name):
        """delete stress from the database

        Parameters
        ----------
        name : str
            name of the stress to delete
        """
        self.lib_stresses.delete(name)
        ArcticPastas.stresses.fget.cache_clear()

    def delete_library(self, lib=None):
        """Delete entire library

        Warning: this breaks a lot of methods in this object!

        Parameters
        ----------
        lib : str, optional
            name of the library to delete, by default None, which
            deletes all libraries

        """
        if libs is None:
            libs = ["oseries", "stresses", "models"]
        elif isinstance(libs, str):
            libs = [libs]
        query_yes_no("Delete library(s): '{}'?".format(", ".join(libs)))
        for lib in libs:
            self.arc.delete_library(self.projectname + "." + lib)
            print("... deleted library '{}'!".format(lib))

    def _get_timeseries(self, libname, names, progressbar=True):
        """internal method to get timeseries

        Parameters
        ----------
        libname : str
            name of the library
        names : str or list of str
            names of the timeseries to load
        progressbar : bool, optional
            show progressbar, by default True

        Returns
        -------
        pandas.DataFrame or dict of pandas.DataFrames
            either returns timeseries as pandas.DataFrame or
            dictionary containing the timeseries.

        """
        lib = self.get_library(libname)

        if isinstance(names, str):
            ts = lib.read(names).data
        elif isinstance(names, Iterable):
            ts = {}
            for n in (tqdm(names) if progressbar else names):
                ts[n] = lib.read(n).data
        return ts

    def get_metadata(self, libname, names):
        """read the metadata

        Parameters
        ----------
        libname : str
            name of the library containing the dataset
        names : str or list of str
            names of the datasets for which to read the metadata

        Returns
        -------
        dict or pandas.DataFrame
            returns metadata dictionary (for one item) or DataFrame
            of metadata (for several datasets)

        """
        lib = self.get_library(libname)

        # read only metadata
        if isinstance(names, str):
            meta = lib.read_metadata(names).metadata
        elif isinstance(names, Iterable):
            metalist = []
            for n in names:
                metalist.append(lib.read_metadata(n).metadata)
            meta = pd.DataFrame(metalist)
            if len({"x", "y"}.difference(meta.columns)) == 0:
                meta["x"] = meta["x"].astype(float)
                meta["y"] = meta["y"].astype(float)
                meta = gpd.GeoDataFrame(meta, geometry=[Point(
                    (s["x"], s["y"])) for i, s in meta.iterrows()])
            if "name" in meta.columns:
                meta.set_index("name", inplace=True)
        return meta

    def get_oseries(self, names, progressbar=False):
        """get oseries from database

        Parameters
        ----------
        names : str or list of str
            names of the oseries to load
        progressbar : bool, optional
            show progressbar, by default False

        Returns
        -------
        pandas.DataFrame or dict of DataFrames
            returns timeseries as DataFrame or dictionary of DataFrames if
            multiple names were passed

        """
        return self._get_timeseries("oseries", names, progressbar=progressbar)

    def get_stresses(self, names, progressbar=False):
        """get stresses from database

        Parameters
        ----------
        names : str or list of str
            names of the stresses to load
        progressbar : bool, optional
            show progressbar, by default False

        Returns
        -------
        pandas.DataFrame or dict of DataFrames
            returns timeseries as DataFrame or dictionary of DataFrames if
            multiple names were passed

        """
        return self._get_timeseries("stresses", names, progressbar=progressbar)

    def get_models(self, names, progressbar=False):
        """load models from database

        Parameters
        ----------
        names : str or list of str
            names of the models to load
        progressbar : bool, optional
            show progressbar, by default False

        Returns
        -------
        pastas.Model or list of pastas.Model
            return pastas model, or list of models if multiple names were
            passed

        """
        if isinstance(names, str):
            names = [names]

        models = []

        for n in (tqdm(names) if progressbar else names):
            item = self.lib_models.read(n)
            data = item.data

            if 'series' not in data['oseries']:
                name = data["oseries"]['name']
                if name not in self.oseries.index:
                    msg = 'oseries {} not present in project'.format(name)
                    raise(LookupError(msg))
                s = self.get_oseries(name)
                data['oseries']['series'] = s.value
            for ts in data["stressmodels"].values():
                if "stress" in ts.keys():
                    for stress in ts["stress"]:
                        if 'series' not in stress:
                            name = stress['name']
                            symbol = str(self.stresses.loc[name, "station"])
                            if name in self.stresses.index:
                                s = self.get_stresses(symbol)
                            else:
                                msg = 'stress {} not present in project'.format(
                                    name)
                            stress['series'] = s

            ml = ps.io.base.load_model(data)
            models.append(ml)
        if len(models) == 1:
            return models[0]
        else:
            return models

    def create_model(self, name, add_recharge=True):
        """create a new pastas Model

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
        ValueError
            if timeseries is empty

        """
        # get oseries metadata
        meta = self.get_metadata("oseries", name)
        ts = self.get_oseries(name)

        # convert to Timeseries and create model
        if not ts.value.dropna().empty:
            ts = ps.TimeSeries(ts.value, name=name, settings="oseries",
                               metadata=meta)
            ml = ps.Model(ts, name=name, metadata=meta)

            if add_recharge:
                self.add_recharge(ml)
            return ml
        else:
            raise ValueError("Empty timeseries!")

    def create_models(self, oseries=None, add_recharge=True, store=False,
                      solve=False, progressbar=True, return_models=False,
                      ignore_errors=True, **kwargs):
        """batch creation of models

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
            oseries = self.oseries.index
        elif isinstance(oseries, str):
            oseries = [oseries]

        models = {}
        errors = []
        for o in (tqdm(oseries)
                  if progressbar else oseries):
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
                self.add_model(iml, add_version=True)
            if return_models:
                models[o] = iml
        if return_models:
            return models, errors
        else:
            return errors

    def add_recharge(self, ml, rfunc=ps.Gamma):
        """add recharge to a pastas model using
        closest precipitation and evaporation timeseries in database

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
            names.append(str(self.stresses.loc[name, "station"]))

        # get data
        tsdict = self.get_stresses(names)
        stresses = []
        for k, s in tsdict.items():
            metadata = self.get_metadata("stresses", k)
            stresses.append(ps.TimeSeries(s, name=metadata["name"],
                                          metadata=metadata))

        # add recharge to model
        rch = ps.StressModel2(stresses, rfunc, name="recharge",
                              metadata=[i.metadata for i in stresses],
                              settings=("prec", "evap"))
        ml.add_stressmodel(rch)

    def solve_models(self, mls=None, report=False, ignore_solve_errors=False,
                     progressbar=True, store_result=True, **kwargs):
        """Solves the models in the library

        mls: list of str, optional
            list of model names, if None all models in the project are solved.
        report: boolean, optional
            determines if a report is printed when the model is solved.
        ignore_solve_errors: boolean, optional
            if True errors emerging from the solve method are ignored.
        **kwargs:
            arguments are passed to the solve method.

        """
        if mls is None:
            mls = self.models
        elif isinstance(mls, ps.Model):
            mls = [mls.name]

        for ml_name in (tqdm(mls) if progressbar else mls):
            ml = self.get_models(ml_name)

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
                    self.add_model(ml, add_version=True)
            except Exception as e:
                if ignore_solve_errors:
                    warning = "solve error ignored for -> {}".format(ml.name)
                    ps.logger.warning(warning)
                else:
                    raise e

    @property
    @functools.lru_cache()
    def oseries(self):
        lib = self.lib_oseries
        return gpd.GeoDataFrame(self.get_metadata("oseries",
                                                  lib.list_symbols()))

    @property
    @functools.lru_cache()
    def stresses(self):
        lib = self.lib_stresses
        return self.get_metadata("stresses",
                                 lib.list_symbols())

    @property
    @functools.lru_cache()
    def models(self):
        lib = self.lib_models
        return lib.list_symbols()
