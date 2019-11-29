import functools
import os
from collections.abc import Iterable
from importlib import import_module

import json
import numpy as np
import pandas as pd
import pystore
from geopandas import GeoDataFrame
from tqdm import tqdm

import pastas as ps
from pastas.io.pas import PastasEncoder

from .base import BaseProject


class PystorePastas(BaseProject):
    """Class that connects to a PyStore (location on disk) that can be
    used to store timeseries and models for Pastas projects.

    Parameters
    ----------
    name : str
        name of the Project
    path : str
        path to the pystore directory
    oseries_store : str, optional
        name of the oseries store, by default None
    stresses_store : str, optional
        name of the stresses store, by default None
    model_store : str, optional
        name of the model store, by default None

    """

    def __init__(self, name, path, oseries_store=None, stresses_store=None,
                 model_store=None, new_store=False):
        """Create a PystorePastas object that points to a Pystore.

        Parameters
        ----------
        name : str
            name of the Project
        path : str
            path to the pystore directory
        oseries_store : str, optional
            name of the oseries store, by default None
        stresses_store : str, optional
            name of the stresses store, by default None
        model_store : str, optional
            name of the model store, by default None

        """
        self.name = name
        self.path = path

        pystore.set_path(self.path)

        self._initialize(oseries_store, stresses_store, model_store, new_store)

    def __repr__(self):
        """string representation of object
        """
        storename = self.name
        noseries = len(self.lib_oseries.list_collections()) if self.lib_oseries else 0
        nstresses = len(self.lib_stresses.list_collections()) if self.lib_stresses else 0
        nmodels = len(self.models)
        return (f"<PystorePastas object> '{storename}': {noseries} oseries, "
                f"{nstresses} stresses, {nmodels} models")

    def _initialize(self, oseries_store=None, stresses_store=None,
                    model_store=None, new_store=False):
        """internal method to initialize links to stores (libraries)

        Parameters
        ----------
        oseries_store : str, optional
            name of the oseries store, by default None
        stresses_store : str, optional
            name of the stresses store, by default None
        model_store : str, optional
            name of the models store, by default None

        """
        if oseries_store is None and new_store:
            oseries_store = "oseries"
            self.lib_oseries = pystore.store(oseries_store)
        elif oseries_store is not None:
            self.lib_oseries = pystore.store(oseries_store)
        else:
            self.lib_oseries = None

        if stresses_store is None and new_store:
            stresses_store = "stresses"
            self.lib_stresses = pystore.store(stresses_store)
        elif stresses_store is not None:
            self.lib_stresses = pystore.store(stresses_store)
        else:
            self.lib_stresses = None

        if model_store is None and new_store:
            model_store = "models"
            self.lib_models = pystore.store(model_store)
        elif model_store is not None:
            self.lib_models = pystore.store(model_store)
        else:
            self.lib_models = None

        self.info = {oseries_store: "oseries",
                     stresses_store: "stresses",
                     model_store: "models"}

    def _add_series(self, libname, series, collection, item, metadata=None,
                    overwrite=True):
        """internal method to add series to a library/store

        Parameters
        ----------
        libname : str
            name of the library
        series : pandas.DataFrame or pandas.Series
            data to write to the pystore
        collection : str
            name of the collection
        item : str
            name of the item
        metadata : dict, optional
            dictionary containing metadata, by default None
        overwrite : bool, optional
            overwrite existing dataset with the same name,
            by default True

        """
        lib = self.get_store(libname)
        coll = lib.collection(collection)
        coll.write(item, series, metadata=metadata, overwrite=overwrite)

    def add_oseries(self, series, collection, item, metadata=None,
                    overwrite=True):
        """add oseries to the pystore

        Parameters
        ----------
        series : pandas.DataFrame of pandas.Series
            oseries data to write to the store
        collection : str
            name of the collection to store the data in
        item : str
            name of the item to store the data as
        metadata : dict, optional
            dictionary containing metadata, by default None
        overwrite : bool, optional
            overwrite existing dataset with the same name,
            by default True

        """
        self._add_series("oseries", series, collection, item,
                         metadata=metadata, overwrite=overwrite)
        PystorePastas.oseries.fget.cache_clear()

    def add_stress(self, series, collection, item, metadata=None,
                   overwrite=True):
        """add stresses to the pystore

        Parameters
        ----------
        series : pandas.DataFrame of pandas.Series
            stress data to write to the store
        collection : str
            name of the collection to store the data in
        item : str
            name of the item to store the data as
        metadata : dict, optional
            dictionary containing metadata, by default None
        overwrite : bool, optional
            overwrite existing dataset with the same name,
            by default True

        """
        self._add_series("stresses", series, collection, item,
                         metadata=metadata, overwrite=overwrite)
        PystorePastas.stresses.fget.cache_clear()

    def add_model(self, ml, overwrite=True):
        """add model to the pystore

        Parameters
        ----------
        ml : pastas.Model
            model to write to the store
        overwrite : bool, optional
            overwrite existing store model if it already exists,
            by default True

        """
        mldict = ml.to_dict(series=False)
        jsondict = json.loads(json.dumps(mldict, cls=PastasEncoder, indent=4))

        lib = self.get_store("models")
        coll = lib.collection(ml.name)
        coll.write(ml.name, pd.DataFrame(), metadata=jsondict,
                   overwrite=overwrite)

    def _del_item(self, libname, collection, item):
        """internal method to delete data from the store

        Parameters
        ----------
        libname : str
            name of the library
        collection : str
            name of the collection to delete the item from
        item : str
            name of the item to delete

        """
        lib = self.get_store(libname)
        c = lib.collection(collection)
        c.delete_item(item)

    def _del_collection(self, libname, collection):
        """internal method to delete a collection

        Parameters
        ----------
        libname : str
            name of the library containing the collection
        collection : str
            name of the collection to delete

        """
        lib = self.get_store(libname)
        lib.delete_collection(collection)

    def del_oseries(self, name, item=None):
        """delete oseries

        Parameters
        ----------
        name : str
            name of the collection containing the data
        item : str, optional
            if provided, delete item, by default None,
            which deletes the whole collection

        """
        if item is None:
            self._del_collection("oseries", name)
        else:
            self._del_item("oseries", name, item)
        PystorePastas.oseries.fget.cache_clear()

    def del_stress(self, name, item=None):
        """delete stress

        Parameters
        ----------
        name : str
            name of the collection containing the data
        item : str, optional
            if provided delete item, by default None,
            which deletes the whole collection
        """
        if item is None:
            self._del_collection("stresses", name)
        else:
            self._del_item("stresses", name, item)
        PystorePastas.stresses.fget.cache_clear()

    def del_model(self, name):
        """delete model from store

        Parameters
        ----------
        name : str
            name of the model to delete

        """
        self._del_collection("models", name)
        # os.remove(os.path.join(self.lib_models, name + ".pas"))

    def get_store(self, storename):
        """get store handle

        Parameters
        ----------
        storename : str
            name of the store

        Returns
        -------
        pystore.store
            handle to the store

        """
        return getattr(self, f"lib_{storename}")

    def _get_timeseries(self, storename, names, item=None,
                        progressbar=True):
        """internal method to load timeseries data

        Parameters
        ----------
        storename : str
            name of the store to load data from
        names : str or list of str
            name(s) of the timeseries to load
        item : str, optional
            name of the item to load in each collection, by default None,
            which defaults to the first item in the collection
        progressbar : bool, optional
            show progressbar, by default True

        Returns
        -------
        pandas.DataFrame or dict of pandas.DataFrames
            returns data DataFrame or dictionary of DataFrames
            if multiple names are provided

        Raises
        ------
        FileNotFoundError
            if no collection can be found with name(s)

        """
        store = self.get_store(storename)

        if isinstance(names, str):
            if names not in store.list_collections():
                raise FileNotFoundError(f"No data for {names}")
            collection = store.collection(names)
            if item is None:
                itemname = list(collection.list_items())[0]
            else:
                itemname = item
            ts = collection.item(itemname).to_pandas()
        elif isinstance(names, Iterable):
            ts = {}
            for n in (tqdm(names) if progressbar else names):
                collection = store.collection(n)
                if item is None:
                    itemname = list(collection.list_items())[0]
                else:
                    itemname = item
                ts[n] = collection.item(itemname).to_pandas()
        return ts

    def get_metadata(self, names, item=None, kind="oseries", progressbar=True):
        """read metadata for dataset

        Parameters
        ----------
        names : str or list of str
            name(s) of collections to load metadata from
        item : str, optional
            name of the item to load from each collection, by default None,
            which uses the first item in each collection
        kind : str, optional
            name of the store to load metadata from, by default "oseries"
        progressbar : bool, optional
            show progressbar, by default True

        Returns
        -------
        dict or dict of dicts
            dictionary containing metadata

        """
        store = self.get_store(kind)

        if isinstance(names, str):
            c = store.collection(names)
            if item is None:
                itemname = list(c.list_items())[0]
            else:
                itemname = item
            meta = pystore.utils.read_metadata(c._item_path(itemname))
        elif isinstance(names, Iterable):
            meta = {}
            for n in (tqdm(names) if progressbar else names):
                c = store.collection(n)
                if item is None:
                    itemname = list(c.list_items())[0]
                else:
                    itemname = item
                meta[n] = pystore.utils.read_metadata(c._item_path(itemname))

        return meta

    def get_oseries(self, names, item=None, progressbar=True):
        """load oseries from store

        Parameters
        ----------
        names : str or list of str
            name(s) of collections to load oseries data from
        item : str, optional
            name of the item to load from each collection, by default None,
            which takes the first item from each collection
        progressbar : bool, optional
            show progressbar, by default True

        Returns
        -------
        pandas.DataFrame or dict of pandas.DataFrames
            returns data as a DataFrame or a dictionary of DataFrames
            if multiple names are passed

        """
        return self._get_timeseries("oseries", names, item,
                                    progressbar=progressbar)

    def get_stresses(self, names, item=None, progressbar=True):
        """load stresses from store

        Parameters
        ----------
        names : str or list of str
            name(s) of collections to load stresses data from
        item : str, optional
            name of the item to load from each collection, by default None,
            which takes the first item from each collection
        progressbar : bool, optional
            show progressbar, by default True

        Returns
        -------
        pandas.DataFrame or dict of pandas.DataFrames
            returns data as a DataFrame or a dictionary of DataFrames
            if multiple names are passed

        """
        return self._get_timeseries("stresses", names, item,
                                    progressbar=progressbar)

    def get_models(self, names, progressbar=False):
        """load models from store

        Parameters
        ----------
        names : str or list of str
            name(s) of the models to load
        progressbar : bool, optional
            show progressbar, by default False

        Returns
        -------
        list
            list of models

        """
        if isinstance(names, str):
            names = [names]

        lib = self.get_store("models")

        models = []
        load_mod = import_module("pastas.io.pas")
        for n in (tqdm(names) if progressbar else names):

            c = lib.collection(n)

            jsonpath = c._item_path(n).joinpath("metadata.json")
            data = load_mod.load(jsonpath)

            if 'series' not in data['oseries']:
                name = data["oseries"]['name']
                param_id = data["oseries"]["metadata"]["parameterId"]
                if name not in self.oseries.index:
                    msg = 'oseries {} not present in project'.format(name)
                    raise(LookupError(msg))
                s = self.get_oseries(name, item=param_id, progressbar=False)
                data['oseries']['series'] = s.value
            for ts in data["stressmodels"].values():
                if "stress" in ts.keys():
                    for stress in ts["stress"]:
                        if 'series' not in stress:
                            name = stress['name']
                            symbol = str(
                                self.stresses.loc[self.stresses.name == name,
                                                  "station"].iloc[0])
                            if symbol in self.stresses.index:
                                s = self.get_stresses(symbol, item=None,
                                                      progressbar=False)
                                s.columns = [symbol]
                            else:
                                msg = 'stress {} not present in project'.format(
                                    name)
                                raise KeyError(msg)
                            stress['series'] = s

            ml = ps.io.base.load_model(data)
            models.append(ml)

        if len(models) == 1:
            return models[0]
        else:
            return models

    def add_recharge(self, ml, rfunc=ps.Gamma):
        """method to add recharge to pastas Model

        Parameters
        ----------
        ml : pastas.Model
            model to add the recharge to
        rfunc : pastas.rfunc, optional
            response function for recharge, by default ps.Gamma

        """
        # get nearest prec and evap stns
        names = []
        for var in ("prec", "evap"):
            name = self.get_nearest_stresses(
                ml.oseries.name, kind=var).iloc[0, 0]
            names.append(str(self.stresses.loc[name, "station"]))

        # get data
        tsdict = self.get_stresses(names, progressbar=False)
        stresses = []
        for k, s in tsdict.items():
            metadata = self.get_metadata(k, item=None, kind="stresses",
                                         progressbar=False)
            stresses.append(ps.TimeSeries(s, name=metadata["name"],
                                          metadata=metadata))

        # add recharge to model
        rch = ps.StressModel2(stresses, rfunc, name="recharge",
                              metadata=[i.metadata for i in stresses],
                              settings=("prec", "evap"))
        ml.add_stressmodel(rch)

    def create_model(self, name, item, add_recharge=True):
        """create model for oseries

        Parameters
        ----------
        name : str
            name of the collection to load the oseries data from
        item : str
            name of the item
        add_recharge : bool, optional
            add recharge to the model based on the closest
            evaporation and precipitation timeseries in the
            pystore, by default True

        Returns
        -------
        pastas.Model
            model for the oseries

        """
        # get oseries metadata
        meta = self.get_metadata(name, item=item, kind="oseries",
                                 progressbar=False)
        ts = self.get_oseries(name, item, progressbar=False)

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

    def create_models(self, oseries=None, item=None, add_recharge=True, store=False,
                      solve=False, progressbar=True, return_models=False,
                      ignore_errors=True, **kwargs):
        """create multiple models

        Parameters
        ----------
        oseries : list of str, optional
            list of oseries to create models for, by default None,
            which creates models for all the oseries in the database
        item : str
            name of item to use from each collection
        add_recharge : bool, optional
            add recharge to the models, by default True, uses the
            closest precipitation and evaporation data in the pystore
        store : bool, optional
            store the models, by default False
        solve : bool, optional
            solve the models, by default False
        progressbar : bool, optional
            show progressbar, by default True
        return_models : bool, optional
            return dict containing models, by default False
        ignore_errors : bool, optional
            ignore errors when creating models, by default True

        Returns
        -------
        dictionary of pastas.Models
            if return_models is True, return models
        errors : list
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
                iml = self.create_model(o, item, add_recharge=add_recharge)
            except Exception as e:
                if ignore_errors:
                    errors.append(o)
                    continue
                else:
                    raise e
            if solve:
                iml.solve(**kwargs)
            if store:
                self.add_model(iml)
            if return_models:
                models[o] = iml
        if return_models:
            return models, errors
        else:
            return errors

    def solve_models(self, mls=None, report=False, ignore_solve_errors=False,
                     progressbar=True, store_result=True, **kwargs):
        """Solves the models in the library

        mls: list of str, optional
            list of model names, if None all models in the
            project are solved.
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
                    self.add_model(ml)
            except Exception as e:
                if ignore_solve_errors:
                    warning = "Solve error ignored for -> {}".format(ml.name)
                    ps.logger.warning(warning)
                else:
                    raise e

    @property
    @functools.lru_cache()
    def oseries(self):
        if self.lib_oseries is not None:
            lib = self.lib_oseries
            return GeoDataFrame(self.get_metadata(lib.list_collections(),
                                                kind="oseries",
                                                progressbar=False)).transpose()
        else:
            return GeoDataFrame()

    @property
    @functools.lru_cache()
    def stresses(self):
        if self.lib_stresses is not None:
            lib = self.lib_stresses
            return GeoDataFrame(self.get_metadata(lib.list_collections(),
                                                kind="stresses",
                                                progressbar=False)).transpose()
        else:
            return GeoDataFrame()

    @property
    # @functools.lru_cache()
    def models(self):
        if self.lib_models is not None:
            lib = self.get_store("models")
            mls = lib.list_collections()
        else:
            mls = []
        return mls
