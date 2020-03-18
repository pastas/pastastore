import functools
import json
from importlib import import_module
from typing import Optional, Union

import numpy as np
import pandas as pd
from tqdm import tqdm

import pastas as ps
from pastas import Model
from pastas.io.pas import PastasEncoder

from .base import BaseConnector, ConnectorUtil

FrameorSeriesUnion = Union[pd.DataFrame, pd.Series]


class ArcticConnector(BaseConnector, ConnectorUtil):
    """Object to serve as the interface between MongoDB and Python
    using the Arctic module. Provides all the methods to read, write,
    or delete data from the database.

    Create an ArcticConnector object that connects to a
    running MongoDB database via Arctic.

    Parameters
    ----------
    connstr : str
        connection string
    projectname : str
        name of the project
    library_map: dict, optional
        dictionary containing the default library names as
        keys ('oseries', 'stresses', 'models') and the user
        specified library names as corresponding values.
        Allows user defined library names.
    """

    conn_type = "arctic"

    def __init__(self, name: str, connstr: str,
                 library_map: Optional[dict] = None):
        """Create an ArcticConnector object that connects to a
        running MongoDB database via Arctic.

        Parameters
        ----------
        connstr : str
            connection string
        projectname : name
            name of the project
        library_map: dict, optional
            dictionary containing the default library names as
            keys ('oseries', 'stresses', 'models') and the user
            specified library names as corresponding values.
            Allows user defined library names.
        """
        try:
            import arctic
        except ModuleNotFoundError as e:
            print("Please install arctic (also requires "
                  "a MongoDB instance running somewhere, e.g. "
                  "MongoDB Community: \n"
                  "https://docs.mongodb.com/manual/administration"
                  "/install-community/)!")
            raise e
        self.connstr = connstr
        self.name = name

        self.libs: dict = {}
        self.arc = arctic.Arctic(connstr)
        self._initialize(library_map)

    def __repr__(self):
        """Representation string of the object.
        """
        noseries = len(self.get_library("oseries").list_symbols())
        nstresses = len(self.get_library("stresses").list_symbols())
        nmodels = len(self.get_library("models").list_symbols())
        return ("<ArcticConnector object> '{0}': {1} oseries, "
                "{2} stresses, {3} models".format(
                    self.name, noseries, nstresses, nmodels))

    def _initialize(self, library_map: Optional[dict]) -> None:
        """Internal method to initalize the libraries.
        """
        if library_map is None:
            libmap = {i: i for i in self._default_library_names}
        else:
            libmap = library_map

        self.library_map = libmap

        for libname in libmap.values():
            if self._library_name(libname) not in self.arc.list_libraries():
                self.arc.initialize_library(self._library_name(libname))
            self.libs[libname] = self.get_library(libname)

    def _library_name(self, libname: str) -> str:
        """Internal method to get full library name according to Arctic"""
        return ".".join([self.name, libname])

    def get_library(self, libname: str):
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
        # get custom library name if necessary
        real_libname = self.library_map[libname]

        # get library handle
        lib = self.arc.get_library(self._library_name(real_libname))
        return lib

    def _add_series(self, libname: str, series: FrameorSeriesUnion, name: str,
                    metadata: Optional[dict] = None,
                    add_version: bool = False) -> None:
        """Internal method to add series to database

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
            self._clear_cache(libname)
        else:
            raise Exception("Item with name '{0}' already"
                            " in '{1}' library!".format(name, libname))

    def add_oseries(self, series: FrameorSeriesUnion, name: str,
                    metadata: Optional[dict] = None,
                    add_version: bool = False) -> None:
        """Add oseries to the database

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
        if isinstance(series, pd.DataFrame) and len(series.columns) > 1:
            if metadata is None:
                print("Data contains multiple columns, "
                      "assuming values in column 0!")
                metadata = {"value_col": 0}
            elif not "value_col" in metadata.keys():
                print("Data contains multiple columns, "
                      "assuming values in column 0!")
                metadata["value_col"] = 0

        self._add_series("oseries", series, name=name,
                         metadata=metadata, add_version=add_version)

    def add_stress(self, series: FrameorSeriesUnion, name: str, kind: str,
                   metadata: Optional[dict] = None,
                   add_version: bool = False) -> None:
        """Add stress to the database

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
            dictionary containing metadata, by default None. Also used to
            point to column containing timeseries if DataFrame is passed
            using the "value_col" key.
        add_version : bool, optional
            if True, add a new version of the dataset to the database,
            by default False
        """
        if metadata is None:
            metadata = {}

        if isinstance(series, pd.DataFrame) and len(series.columns) > 1:
            print("Data contains multiple columns, "
                  "assuming values in column 0!")
            metadata["value_col"] = 0

        metadata["kind"] = kind
        self._add_series("stresses", series, name=name,
                         metadata=metadata, add_version=add_version)

    def add_model(self, ml: ps.Model, add_version: bool = False) -> None:
        """Add model to the database

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
        lib = self.get_library("models")
        if ml.name not in lib.list_symbols() or add_version:
            mldict = ml.to_dict(series=False)
            lib.write(ml.name, mldict, metadata=ml.oseries.metadata)
        else:
            raise Exception("Model with name '{}' already in store!".format(
                ml.name))
        self._clear_cache("models")

    def _del_item(self, libname: str, name: str) -> None:
        """Internal method to delete items (series or models)

        Parameters
        ----------
        libname : str
            name of library to delete item from
        name : str
            name of item to delete
        """
        lib = self.get_library(libname)
        lib.delete(name)

    def del_models(self, names: Union[list, str]) -> None:
        """Delete model(s) from the database

        Parameters
        ----------
        names : str or list of str
            name(s) of the model to delete
        """
        for n in self._parse_names(names, libname="models"):
            self._del_item("models", n)
        self._clear_cache("models")

    def del_oseries(self, names: Union[list, str]):
        """Delete oseries from the database

        Parameters
        ----------
        names : str or list of str
            name(s) of the oseries to delete
        """
        for n in self._parse_names(names, libname="oseries"):
            self._del_item("oseries", n)
        self._clear_cache("oseries")

    def del_stress(self, names: Union[list, str]):
        """Delete stress from the database

        Parameters
        ----------
        names : str or list of str
            name(s) of the stress to delete
        """
        for n in self._parse_names(names, libname="stresses"):
            self._del_item("stresses", n)
        self._clear_cache("stresses")

    def _get_series(self, libname: str, names: Union[list, str],
                    progressbar: bool = True) -> FrameorSeriesUnion:
        """Internal method to get timeseries

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

        ts = {}
        names = self._parse_names(names, libname=libname)
        for n in (tqdm(names) if progressbar else names):
            ts[n] = lib.read(n).data
        # return frame if len == 1
        if len(ts) == 1:
            return ts[n]
        else:
            return ts

    def get_metadata(self, libname: str, names: Union[list, str],
                     progressbar: bool = False, as_frame: bool = True) -> \
            Union[dict, pd.DataFrame]:
        """Read metadata from database

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

        metalist = []
        names = self._parse_names(names, libname=libname)
        for n in (tqdm(names) if progressbar else names):
            imeta = lib.read_metadata(n).metadata
            if "name" not in imeta.keys():
                imeta["name"] = n
            metalist.append(imeta)
        if as_frame:
            meta = self._meta_list_to_frame(metalist, names=names)
            return meta
        else:
            if len(metalist) == 1:
                return metalist[0]
            else:
                return metalist

    def get_oseries(self, names: Union[list, str],
                    progressbar: bool = False) -> FrameorSeriesUnion:
        """Get oseries from database

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
        return self._get_series("oseries", names, progressbar=progressbar)

    def get_stresses(self, names: Union[list, str],
                     progressbar: bool = False) -> FrameorSeriesUnion:
        """Get stresses from database

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
        return self._get_series("stresses", names, progressbar=progressbar)

    def get_models(self, names: Union[list, str],
                   progressbar: bool = False) -> Union[ps.Model, dict]:
        """Load models from database

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
        lib = self.get_library("models")
        models = []
        names = self._parse_names(names, libname="models")

        for n in (tqdm(names) if progressbar else names):
            item = lib.read(n)
            data = item.data
            ml = self._parse_model_dict(data)
            models.append(ml)
        if len(models) == 1:
            return models[0]
        else:
            return models

    @staticmethod
    def _clear_cache(libname: str) -> None:
        """Clear cached property
        """
        getattr(ArcticConnector, libname).fget.cache_clear()

    @property  # type: ignore
    @functools.lru_cache()
    def oseries(self):
        """DataFrame with overview of oseries
        """
        lib = self.get_library("oseries")
        df = self.get_metadata("oseries", lib.list_symbols())
        return df

    @property  # type: ignore
    @functools.lru_cache()
    def stresses(self):
        """DataFrame with overview of stresses
        """
        lib = self.get_library("stresses")
        return self.get_metadata("stresses",
                                 lib.list_symbols())

    @property  # type: ignore
    @functools.lru_cache()
    def models(self):
        """List of model names
        """
        lib = self.get_library("models")
        return lib.list_symbols()


class PystoreConnector(BaseConnector, ConnectorUtil):
    """Object to serve as the interface between storage and Python
    using the Pystore module. Provides all the methods to read, write,
    or delete data from the pystore.

    Create as PystoreConnector object that connects to a folder on disk
    containing a Pystore.

    Parameters
    ----------
    name : str
        name of the store
    path : str
        path to the pystore directory
    library_map: dict, optional
        dictionary containing the default library names as
        keys ('oseries', 'stresses', 'models') and the user
        specified library names as corresponding values.
        Allows user defined library names.
    """
    conn_type = "pystore"

    def __init__(self, name: str, path: str,
                 library_map: Optional[dict] = None):
        """Create a PystoreConnector object that points to a Pystore.

        Parameters
        ----------
        name : str
            name of the store
        path : str
            path to the pystore directory
        library_map : dict, optional
            dictionary containing the default library names as
            keys ('oseries', 'stresses', 'models') and the user
            specified library names as corresponding values.
            Allows user defined library names.
        """
        try:
            import pystore
        except ModuleNotFoundError as e:
            print("Install pystore, follow instructions at "
                  "https://github.com/ranaroussi/pystore#dependencies")
            raise e
        self.name = name
        self.path = path
        pystore.set_path(self.path)
        self.store = pystore.store(self.name)
        self.libs: dict = {}
        self._initialize(library_map)

    def __repr__(self):
        """Representation string of the object
        """
        storename = self.name
        noseries = len(self.get_library("oseries").list_items())
        nstresses = len(self.get_library("stresses").list_items())
        nmodels = len(self.get_library("models").list_items())
        return (f"<PystoreConnector object> '{storename}': {noseries} oseries,"
                f" {nstresses} stresses, {nmodels} models")

    def _initialize(self, library_map: Optional[dict]):
        """Internal method to initalize the libraries (stores).
        """
        if library_map is None:
            self.library_map = {i: i for i in self._default_library_names}
        else:
            self.library_map = library_map

        for libname in self.library_map.values():
            lib = self.store.collection(libname)
            self.libs[libname] = lib

    def get_library(self, libname: str):
        """Get Pystore library handle

        Parameters
        ----------
        libname : str
            name of the library

        Returns
        -------
        Pystore.Collection handle
            handle to the library
        """
        # get custom library name if necessary
        real_libname = self.library_map[libname]

        # get library handle
        lib = self.store.collection(real_libname)
        return lib

    def _add_series(self, libname: str, series: FrameorSeriesUnion, name: str,
                    metadata: Optional[dict] = None,
                    overwrite=True):
        """Internal method to add series to a library/store

        Parameters
        ----------
        libname : str
            name of the library
        series : pandas.DataFrame or pandas.Series
            data to write to the pystore
        name : str
            name of the series
        metadata : dict, optional
            dictionary containing metadata, by default None
        overwrite : bool, optional
            overwrite existing dataset with the same name,
            by default True
        """
        lib = self.get_library(libname)
        lib.write(name, series, metadata=metadata, overwrite=overwrite)
        self._clear_cache(libname)

    def add_oseries(self, series: FrameorSeriesUnion, name: str,
                    metadata: Optional[dict] = None,
                    overwrite=True):
        """Add oseries to the pystore

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
        if isinstance(series, pd.DataFrame) and len(series.columns) > 1:
            if metadata is None:
                print("Data contains multiple columns, "
                      "assuming values in column 0!")
                metadata = {"value_col": 0}
            elif not "value_col" in metadata.keys():
                print("Data contains multiple columns, "
                      "assuming values in column 0!")
                metadata["value_col"] = 0
        self._add_series("oseries", series, name,
                         metadata=metadata, overwrite=overwrite)

    def add_stress(self, series: FrameorSeriesUnion, name: str, kind,
                   metadata: Optional[dict] = None,
                   overwrite=True):
        """Add stresses to the pystore

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
        if metadata is None:
            metadata = {}
        if kind not in metadata.keys():
            metadata["kind"] = kind
        self._add_series("stresses", series, name,
                         metadata=metadata, overwrite=overwrite)

    def add_model(self, ml: ps.Model, add_version: bool = True):
        """Add model to the pystore

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

        collection = self.get_library("models")
        collection.write(ml.name, pd.DataFrame(), metadata=jsondict,
                         overwrite=add_version)
        self._clear_cache("models")

    def _del_series(self, libname: str, name):
        """Internal method to delete data from the store

        Parameters
        ----------
        libname : str
            name of the library
        name : str
            name of the series to delete
        """
        lib = self.get_library(libname)
        lib.delete_item(name)
        self._clear_cache(libname)

    def del_oseries(self, names: Union[list, str]):
        """Delete oseries from pystore

        Parameters
        ----------
        name : str
            name of the collection containing the data
        names : str or list of str, optional
            name(s) of oseries to delete
        """
        for n in self._parse_names(names, libname="oseries"):
            self._del_series("oseries", names)

    def del_stress(self, names: Union[list, str]):
        """Delete stresses from pystore

        Parameters
        ----------
        names : str or list of str
            name(s) of the series to delete
        """
        for n in self._parse_names(names, libname="stresses"):
            self._del_series("stresses", names)

    def del_models(self, names: Union[list, str]):
        """Delete model(s) from pystore

        Parameters
        ----------
        names : str
            name(s) of the model(s) to delete
        """
        for n in self._parse_names(names, libname="models"):
            self._del_series("models", names)

    def _get_series(self, libname: str, names: Union[list, str],
                    progressbar: bool = True):
        """Internal method to load timeseries data

        Parameters
        ----------
        libname : str
            name of the store to load data from
        names : str or list of str
            name(s) of the timeseries to load
        progressbar : bool, optional
            show progressbar, by default True

        Returns
        -------
        pandas.DataFrame or dict of pandas.DataFrames
            returns data DataFrame or dictionary of DataFrames
            if multiple names are provided
        """
        lib = self.get_library(libname)

        ts = {}
        names = self._parse_names(names, libname=libname)
        for n in (tqdm(names) if progressbar else names):
            ts[n] = lib.item(n).to_pandas()
        # return frame if len == 1
        if len(ts) == 1:
            return ts[n]
        else:
            return ts

    def get_metadata(self, libname: str, names: Union[list, str],
                     progressbar: bool = False, as_frame=True) \
            -> Union[dict, pd.DataFrame]:
        """Read metadata from pystore

        Parameters
        ----------
        libname : str
            name of the library the series are in,
            usually ("oseries" or "stresses")
        names : str or list of str
            name(s) of series to load metadata for
        progressbar : bool, optional
            show progressbar, by default True
        as_frame : bool, optional
            return metadata as dataframe, default is
            True, otherwise return as dict or list of
            dict

        Returns
        -------
        list or pandas.DataFrame
            list or pandas.DataFrame containing metadata
        """
        import pystore
        lib = self.get_library(libname)

        metalist = []
        names = self._parse_names(names, libname=libname)
        for n in (tqdm(names) if progressbar else names):
            imeta = pystore.utils.read_metadata(lib._item_path(n))
            if "name" not in imeta.keys():
                imeta["name"] = n
            metalist.append(imeta)
        if as_frame:
            meta = self._meta_list_to_frame(metalist, names=names)
            return meta
        else:
            if len(metalist) == 1:
                return metalist[0]
            else:
                return metalist

    def get_oseries(self, names: Union[list, str],
                    progressbar: bool = False) -> FrameorSeriesUnion:
        """Retrieve oseries from pystore

        Parameters
        ----------
        names : str or list of str
            name(s) of collections to load oseries data from
        progressbar : bool, optional
            show progressbar, by default True

        Returns
        -------
        pandas.DataFrame or dict of pandas.DataFrames
            returns data as a DataFrame or a dictionary of DataFrames
            if multiple names are passed
        """
        return self._get_series("oseries", names, progressbar=progressbar)

    def get_stresses(self, names: Union[list, str],
                     progressbar: bool = False) -> FrameorSeriesUnion:
        """Retrieve stresses from pystore

        Parameters
        ----------
        names : str or list of str
            name(s) of collections to load stresses data from
        progressbar : bool, optional
            show progressbar, by default True

        Returns
        -------
        pandas.DataFrame or dict of pandas.DataFrames
            returns data as a DataFrame or a dictionary of DataFrames
            if multiple names are passed
        """
        return self._get_series("stresses", names, progressbar=progressbar)

    def get_models(self, names: Union[list, str],
                   progressbar: bool = False) -> Union[ps.Model, dict]:
        """Load models from pystore

        Parameters
        ----------
        names : str or list of str
            name(s) of the models to load
        progressbar : bool, optional
            show progressbar, by default False

        Returns
        -------
        list or pastas.Model
            model or list of models

        """
        lib = self.get_library("models")

        models = []
        load_mod = import_module("pastas.io.pas")  # type: ignore
        names = self._parse_names(names, libname="models")
        for n in (tqdm(names) if progressbar else names):

            jsonpath = lib._item_path(n).joinpath("metadata.json")
            data = load_mod.load(jsonpath)  # type: ignore

            ml = self._parse_model_dict(data)
            models.append(ml)

        if len(models) == 1:
            return models[0]
        else:
            return models

    @staticmethod
    def _clear_cache(libname: str) -> None:
        """Clear cached property
        """
        getattr(PystoreConnector, libname).fget.cache_clear()

    @property  # type: ignore
    @functools.lru_cache()
    def oseries(self):
        """DataFrame with overview of oseries
        """
        lib = self.get_library("oseries")
        df = self.get_metadata("oseries", lib.list_items())
        return df

    @property  # type: ignore
    @functools.lru_cache()
    def stresses(self):
        """DataFrame with overview of stresses
        """
        lib = self.get_library("stresses")
        df = self.get_metadata("stresses", lib.list_items())
        return df

    @property  # type: ignore
    @functools.lru_cache()
    def models(self):
        """List of model names
        """
        lib = self.get_library("models")
        if lib is not None:
            mls = lib.list_items()
        else:
            mls = []
        return mls


class DictConnector(BaseConnector, ConnectorUtil):
    """Object to store timeseries and pastas models in-memory. Provides
    methods to read, write, or delete data from the object. Data is stored
    in dictionaries.

    Parameters
    ----------
    name : str
        user-specified name of the connector
    library_map : dict, optional
        dictionary containing the default library names as
        keys ('oseries', 'stresses', 'models') and the user
        specified library names as corresponding values.
        Allows user defined library names.
    """
    conn_type = "dict"

    def __init__(self, name: str, library_map: Optional[dict] = None):
        """Create DictConnector object that stores all data in dictionaries
        (in-memory).

        Parameters
        ----------
        name : str
            user-specified name of the connector
        library_map : dict, optional
            dictionary containing the default library names as
            keys ('oseries', 'stresses', 'models') and the user
            specified library names as corresponding values.
            Allows user defined library names.
        """
        self.name = name

        # allow custom library names
        if library_map is None:
            libmap = {i: i for i in self._default_library_names}
        else:
            libmap = library_map

        self.library_map = libmap

        # set empty dictionaries for series
        for val in self.library_map.values():
            setattr(self, "lib_" + val, {})

    def __repr__(self):
        """Representation string of the object.
        """
        noseries = len(self.get_library("oseries").keys())
        nstresses = len(self.get_library("stresses").keys())
        nmodels = len(self.get_library("models").keys())
        return "<DictConnector object> '{0}': {1} oseries, {2} stresses, {3} models".format(
            self.name, noseries, nstresses, nmodels
        )

    def get_library(self, libname: str):
        """Get reference to dictionary holding data

        Parameters
        ----------
        libname : str
            name of the library
        """
        # get custom library name
        real_libname = "lib_" + self.library_map[libname]
        return getattr(self, real_libname)

    def _add_series(self, libname: str, series: FrameorSeriesUnion,
                    name: str, metadata: Union[dict, None] = None) -> None:
        """Internal method to obtain series

        Parameters
        ----------
        libname : str
            name of library
        series : FrameorSeriesUnion
            pandas.Series or pandas.DataFrame containing data
        name : str
            name of the series
        metadata : dict, optional
            dictionary containing metadata, by default None
        """
        lib = self.get_library(libname)
        lib[name] = (metadata, series)
        self._clear_cache(libname)

    def add_oseries(self, series: FrameorSeriesUnion, name: str,
                    metadata: Union[dict, None] = None, **kwargs) -> None:
        """Add oseries to object

        Parameters
        ----------
        series : FrameorSeriesUnion
            pandas.Series or pandas.DataFrame containing data
        name : str
            name of the oseries
        metadata : dict, optional
            dictionary with metadata, by default None
        """
        self._add_series("oseries", series, name, metadata=metadata)

    def add_stress(self, series: FrameorSeriesUnion, name: str, kind: str,
                   metadata: Union[dict, None] = None, **kwargs) -> None:
        """Add stress to object

        Parameters
        ----------
        series : FrameorSeriesUnion
            pandas.Series or pandas.DataFrame containing data
        name : str
            name of the stress
        kind : str
            type of stress (i.e. 'prec', 'evap', 'well', etc.)
        metadata : dict, optional
            dictionary containing metadata, by default None
        """
        if metadata is None:
            metadata = {}
        if kind not in metadata.keys():
            metadata["kind"] = kind
        self._add_series("stresses", series, name, metadata=metadata)

    def add_model(self, ml: Model, **kwargs) -> None:
        """Add model to object

        Parameters
        ----------
        ml : Model
            pastas.Model to add
        """
        lib = self.get_library("models")
        mldict = ml.to_dict(series=False)
        lib[ml.name] = mldict
        self._clear_cache("models")

    def del_models(self, names: Union[list, str]) -> None:
        """Delete models from object

        Parameters
        ----------
        names : Union[list, str]
            str or list of str of model names to remove
        """
        lib = self.get_library("models")
        for n in self._parse_names(names, libname="models"):
            _ = lib.pop(n)
        self._clear_cache("models")

    def del_oseries(self, names: Union[list, str]) -> None:
        """Delete oseries from object

        Parameters
        ----------
        names : Union[list, str]
            str or list of str of oseries to remove
        """
        lib = self.get_library("oseries")
        for n in self._parse_names(names, libname="oseries"):
            _ = lib.pop(n)
        self._clear_cache("oseries")

    def del_stress(self, names: Union[list, str]) -> None:
        """Delete stresses from object

        Parameters
        ----------
        names : Union[list, str]
            str or list of str of stresses to remove
        """
        lib = self.get_library("stresses")
        for n in self._parse_names(names, libname="stresses"):
            _ = lib.pop(n)
        self._clear_cache("stresses")

    def _get_series(self, libname: str, names: Union[list, str],
                    progressbar: bool = True) -> FrameorSeriesUnion:
        """Internal method to get oseries or stresses

        Parameters
        ----------
        libname : str
            name of library
        names : Union[list, str]
            str or list of string
        progressbar : bool, optional
            show progressbar, by default True

        Returns
        -------
        dict, FrameorSeriesUnion
            returns DataFrame or Series if only one name is passed, else
            returns dict with all the data

        """
        lib = self.get_library(libname)
        ts = {}
        names = self._parse_names(names, libname=libname)
        for n in (tqdm(names) if progressbar else names):
            ts[n] = lib[n][1]
        # return frame if len == 1
        if len(ts) == 1:
            return ts[n]
        else:
            return ts

    def get_metadata(self, libname: str, names: Union[list, str],
                     progressbar: bool = False, as_frame: bool = True) \
            -> Union[pd.DataFrame, list]:
        """Get metadata from object

        Parameters
        ----------
        libname : str
            name of library
        names : Union[list, str]
            str or list of str of names to get metadata for
        progressbar : bool, optional
            show progressbar, by default False
        as_frame : bool, optional
            return as DataFrame, by default True

        Returns
        -------
        Union[pd.DataFrame, list]
            returns list of metadata or pandas.DataFrame depending on value
            of `as_frame`
        """
        lib = self.get_library(libname)
        metalist = []
        names = self._parse_names(names, libname=libname)
        for n in (tqdm(names) if progressbar else names):
            imeta = lib[n][0]
            if "name" not in imeta.keys():
                imeta["name"] = n
            metalist.append(imeta)

        if as_frame:
            meta = self._meta_list_to_frame(metalist, names=names)
            return meta
        else:
            if len(metalist) == 1:
                return metalist[0]
            else:
                return metalist

    def get_oseries(self, names: Union[list, str],
                    progressbar: bool = False) -> FrameorSeriesUnion:
        """Retrieve oseries from object

        Parameters
        ----------
        names : Union[list, str]
            name or list of names to retrieve
        progressbar : bool, optional
            show progressbar, by default False

        Returns
        -------
        dict, FrameorSeriesUnion
            returns dictionary or DataFrame/Series depending on number of
            names passed
        """
        return self._get_series("oseries", names, progressbar=progressbar)

    def get_stresses(self, names: Union[list, str],
                     progressbar: bool = False) -> FrameorSeriesUnion:
        """Retrieve stresses from object

        Parameters
        ----------
        names : Union[list, str]
            name or list of names of stresses to retrieve
        progressbar : bool, optional
            show progressbar, by default False

        Returns
        -------
        dict, FrameorSeriesUnion
            returns dictionary or DataFrame/Series depending on number of
            names passed
        """
        return self._get_series("stresses", names, progressbar=progressbar)

    def get_models(self, names: Union[list, str],
                   progressbar: bool = False) -> Union[Model, dict]:
        """Load models from object

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
        lib = self.get_library("models")
        models = []
        names = self._parse_names(names, libname="models")

        for n in (tqdm(names) if progressbar else names):
            data = lib[n]
            ml = self._parse_model_dict(data)
            models.append(ml)
        if len(models) == 1:
            return models[0]
        else:
            return models

    @staticmethod
    def _clear_cache(libname: str) -> None:
        """Clear cached property
        """
        getattr(DictConnector, libname).fget.cache_clear()

    @property  # type: ignore
    @functools.lru_cache()
    def oseries(self):
        """DataFrame showing overview of oseries
        """
        lib = self.get_library("oseries")
        return self.get_metadata("oseries", names=list(lib.keys()))

    @property  # type: ignore
    @functools.lru_cache()
    def stresses(self):
        """DataFrame showing overview of stresses
        """
        lib = self.get_library("stresses")
        return self.get_metadata("stresses", names=list(lib.keys()))

    @property  # type: ignore
    @functools.lru_cache()
    def models(self):
        """List of model names
        """
        lib = self.get_library("models")
        return list(lib.keys())
