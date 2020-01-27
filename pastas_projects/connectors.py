import functools
import json
from collections.abc import Iterable
from importlib import import_module
from typing import Union, Optional

import numpy as np
import pandas as pd
from tqdm import tqdm

import pastas as ps
from pastas.io.pas import PastasEncoder

from .base import BaseConnector

FrameorSeriesUnion = Union[pd.DataFrame, pd.Series]


class ArcticConnector(BaseConnector):
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
    # _default_library_names = ["oseries", "stresses", "models"]
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
                  "a MongoDB instance running somewhere, e.g. MongoDB Community: \n"
                  "https://docs.mongodb.com/manual/administration/install-community/)!")
            raise e
        self.connstr = connstr
        self.name = name

        self.libs = {}
        self.arc = arctic.Arctic(connstr)
        self._initialize(library_map)

    def __repr__(self):
        """representation string of the object.
        """
        noseries = len(self.get_library("oseries").list_symbols())
        nstresses = len(self.get_library("stresses").list_symbols())
        nmodels = len(self.get_library("models").list_symbols())
        return "<ArcticConnector object> '{0}': {1} oseries, {2} stresses, {3} models".format(
            self.name, noseries, nstresses, nmodels
        )

    def _initialize(self, library_map: Optional[dict]) -> None:
        """internal method to initalize the libraries.
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
        return ".".join([self.name, libname])

    def _parse_names(self, names: Union[list, str]) -> list:
        """internal method to parse names kwarg,
        returns iterable with name(s)

        """
        if names is None:
            return self.oseries.index
        elif isinstance(names, str):
            return [names]
        elif isinstance(names, Iterable):
            return names
        else:
            raise NotImplementedError(f"Cannot parse 'names': {names}")

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
                    metadata: Optional[dict] = None, add_version: bool = False) \
            -> None:
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

    def add_oseries(self, series: FrameorSeriesUnion, name: str,
                    metadata: Optional[dict] = None,
                    add_version: bool = False) -> None:
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
        if isinstance(series, pd.DataFrame) and len(series.columns) > 1:
            print("Data contains multiple columns, assuming values in column 0!")
            metadata = {"value_col": 0}
        self._add_series("oseries", series, name=name,
                         metadata=metadata, add_version=add_version)
        self._clear_cache("oseries")

    def add_stress(self, series: FrameorSeriesUnion, name: str, kind: str, metadata: Optional[dict] = None, add_version: bool = False) -> None:
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
            print("Data contains multiple columns, assuming values in column 0!")
            metadata["value_col"] = 0

        metadata["kind"] = kind
        self._add_series("stresses", series, name=name,
                         metadata=metadata, add_version=add_version)
        self._clear_cache("stresses")

    def add_model(self, ml: ps.Model, add_version: bool = False) -> None:
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
        lib = self.get_library("models")
        if ml.name not in lib.list_symbols() or add_version:
            mldict = ml.to_dict(series=False)
            lib.write(ml.name, mldict, metadata=ml.oseries.metadata)
        else:
            raise Exception("Model with name '{}' already in store!".format(
                ml.name))
        self._clear_cache("models")

    def _del_item(self, libname: str, name: str) -> None:
        """internal method to delete items (series or models)

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
        """delete model(s) from the database

        Parameters
        ----------
        names : str or list of str
            name(s) of the model to delete
        """
        for n in self._parse_names(names):
            self._del_item("models", n)
        self._clear_cache("models")

    def del_oseries(self, names: Union[list, str]):
        """delete oseries from the database

        Parameters
        ----------
        names : str or list of str
            name(s) of the oseries to delete
        """
        for n in self._parse_names(names):
            self._del_item("oseries", n)
        self._clear_cache("oseries")

    def del_stress(self, names: Union[list, str]):
        """delete stress from the database

        Parameters
        ----------
        names : str or list of str
            name(s) of the stress to delete
        """
        for n in self._parse_names(names):
            self._del_item("stresses", n)
        self._clear_cache("stresses")

    def _get_series(self, libname: str, names: Union[list, str],
                    progressbar: bool = True) -> FrameorSeriesUnion:
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

        ts = {}
        names = self._parse_names(names)
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

        metalist = []
        names = self._parse_names(names)
        for n in (tqdm(names) if progressbar else names):
            imeta = lib.read_metadata(n).metadata
            if "name" not in imeta.keys():
                imeta["name"] = n
            metalist.append(imeta)
        if as_frame:
            # convert to frame if len > 1 else return dict
            if len(metalist) > 1:
                meta = pd.DataFrame(metalist)
                if len({"x", "y"}.difference(meta.columns)) == 0:
                    meta["x"] = meta["x"].astype(float)
                    meta["y"] = meta["y"].astype(float)
                    # meta = gpd.GeoDataFrame(meta, geometry=[Point(
                    #     (s["x"], s["y"])) for i, s in meta.iterrows()])
            elif len(metalist) == 1:
                meta = pd.DataFrame(metalist)
            elif len(metalist) == 0:
                meta = pd.DataFrame()
            if "name" in meta.columns:
                meta.set_index("name", inplace=True)
            else:
                meta.index = names
            return meta
        else:
            if len(metalist) == 1:
                return metalist[0]
            else:
                return metalist

    def get_oseries(self, names: Union[list, str],
                    progressbar: bool = False) -> FrameorSeriesUnion:
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
        return self._get_series("oseries", names, progressbar=progressbar)

    def get_stresses(self, names: Union[list, str],
                     progressbar: bool = False) -> FrameorSeriesUnion:
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
        return self._get_series("stresses", names, progressbar=progressbar)

    def _get_values(self, libname: str, name: str, series: FrameorSeriesUnion,
                    metadata: Optional[dict] = None) -> FrameorSeriesUnion:
        """internal method to get column containing values from data

        Parameters
        ----------
        libname : str
            name of the library the data is stored in
        name : str
            name of the series
        series : pd.Series or pd.DataFrame
            data
        metadata : dict, optional
            if passed avoids retrieving dict from database,
            default is None, in which case data will be read
            from database.

        """
        if isinstance(series, pd.Series):
            return series
        elif len(series.columns) == 1:
            return series.iloc[:, 0]
        else:
            if metadata is None:
                meta = self.get_metadata(libname, name, as_frame=False)
            else:
                meta = metadata
            if "value_col" not in meta.keys():
                raise KeyError("Please provide 'value_col' to metadata "
                               "dictionary to point to column containing "
                               "timeseries values!")
            value_col = meta["value_col"]
            if isinstance(value_col, str):
                series = series.loc[:, value_col]
            elif isinstance(value_col, int):
                series = series.iloc[:, value_col]
            return series

    def get_models(self, names: Union[list, str],
                   progressbar: bool = False) -> Union[ps.Model, dict]:
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
        lib = self.get_library("models")
        models = []
        names = self._parse_names(names)

        for n in (tqdm(names) if progressbar else names):
            item = lib.read(n)
            data = item.data

            if 'series' not in data['oseries']:
                name = data["oseries"]['name']
                if name not in self.oseries.index:
                    msg = 'oseries {} not present in project'.format(name)
                    raise(LookupError(msg))
                s = self.get_oseries(name)
                data['oseries']['series'] = self._get_values(
                    "oseries", name, s)
            for ts in data["stressmodels"].values():
                if "stress" in ts.keys():
                    for stress in ts["stress"]:
                        if 'series' not in stress:
                            name = stress['name']
                            if name in self.stresses.index:
                                s = self.get_stresses(name)
                                stress['series'] = self._get_values(
                                    "stresses", name, s)
                            else:
                                msg = "stress '{}' not present in project".format(
                                    name)
                                raise KeyError(msg)

            ml = ps.io.base.load_model(data)
            models.append(ml)
        if len(models) == 1:
            return models[0]
        else:
            return models

    @staticmethod
    def _clear_cache(libname: str) -> None:
        """clear cache
        """
        getattr(ArcticConnector, libname).fget.cache_clear()

    @property
    @functools.lru_cache()
    def oseries(self):
        lib = self.get_library("oseries")
        df = self.get_metadata("oseries", lib.list_symbols())
        return df

    @property
    @functools.lru_cache()
    def stresses(self):
        lib = self.get_library("stresses")
        return self.get_metadata("stresses",
                                 lib.list_symbols())

    @property
    @functools.lru_cache()
    def models(self):
        lib = self.get_library("models")
        return lib.list_symbols()


class PystoreConnector(BaseConnector):

    conn_type = "pystore"

    def __init__(self, name: str, path: str,
                 library_map: Optional[dict] = None):
        """Create a PystoreConnector object that points to a Pystore.

        Parameters
        ----------
        name : str
            name of the Project
        path : str
            path to the pystore directory
        library_map : dict, optional
            dictionary containing the default library names as
            keys ('oseries', 'stresses', 'models') and the user
            specified library names as corresponding values.
            Allows user defined library names.

        Note
        ----
        - Pystore uses the terminology Store > Collection > Item,
          but in this code 'Collection' is referred to as 'Library'
          (same as Arctic).

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
        self.libs = {}
        self._initialize(library_map)

    def __repr__(self):
        """string representation of object
        """
        storename = self.name
        noseries = len(self.get_library("oseries").list_items())
        nstresses = len(self.get_library("stresses").list_items())
        nmodels = len(self.get_library("models").list_items())
        return (f"<PystoreConnector object> '{storename}': {noseries} oseries, "
                f"{nstresses} stresses, {nmodels} models")

    def _initialize(self, library_map: Optional[dict]):
        """internal method to initalize the libraries (stores).
        """
        if library_map is None:
            self.library_map = {i: i for i in self._default_library_names}
        else:
            self.library_map = library_map

        for libname in self.library_map.values():
            lib = self.store.collection(libname)
            self.libs[libname] = lib

    def _parse_names(self, names: Union[list, str]):
        """internal method to parse names kwarg,
        returns iterable with name(s)

        """
        if names is None:
            return self.oseries.index
        elif isinstance(names, str):
            return [names]
        elif isinstance(names, Iterable):
            return names
        else:
            raise NotImplementedError(f"Cannot parse 'names': {names}")

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
        """internal method to add series to a library/store

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
        self._add_series("oseries", series, name,
                         metadata=metadata, overwrite=overwrite)

    def add_stress(self, series: FrameorSeriesUnion, name: str, kind,
                   metadata: Optional[dict] = None,
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
        if kind not in metadata.keys():
            metadata["kind"] = kind
        self._add_series("stresses", series, name,
                         metadata=metadata, overwrite=overwrite)

    def add_model(self, ml: ps.Model, add_version: bool = True):
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

        collection = self.get_library("models")
        collection.write(ml.name, pd.DataFrame(), metadata=jsondict,
                         overwrite=add_version)
        self._clear_cache("models")

    def _del_series(self, libname: str, name):
        """internal method to delete data from the store

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
        """delete oseries

        Parameters
        ----------
        name : str
            name of the collection containing the data
        item : str, optional
            if provided, delete item, by default None,
            which deletes the whole collection

        """
        for n in self._parse_names(names):
            self._del_series("oseries", names)

    def del_stress(self, names: Union[list, str]):
        """delete stresses

        Parameters
        ----------
        names : str or list of str
            name(s) of the series to delete
        """
        for n in self._parse_names(names):
            self._del_series("stresses", names)

    def del_models(self, names: Union[list, str]):
        """delete model from store

        Parameters
        ----------
        names : str
            name(s) of the model(s) to delete

        """
        for n in self._parse_names(names):
            self._del_series("models", names)

    def _get_series(self, libname: str, names: Union[list, str],
                    progressbar: bool = True):
        """internal method to load timeseries data

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
        names = self._parse_names(names)
        for n in (tqdm(names) if progressbar else names):
            ts[n] = lib.item(n).to_pandas()
        # return frame if len == 1
        if len(ts) == 1:
            return ts[n]
        else:
            return ts

    def _get_values(self, libname: str, name, series: FrameorSeriesUnion,
                    metadata=None) -> FrameorSeriesUnion:
        """internal method to get column containing values from data

        Parameters
        ----------
        libname : str
            name of the library the data is stored in
        name : str
            name of the series
        series : pd.Series or pd.DataFrame
            data
        metadata : dict, optional
            if passed avoids retrieving dict from database,
            default is None, in which case data will be read
            from database.

        """
        if isinstance(series, pd.Series):
            return series
        elif len(series.columns) == 1:
            return series.iloc[:, 0]
        else:
            if metadata is None:
                meta = self.get_metadata(libname, name, as_frame=False)
            else:
                meta = metadata
            if "value_col" not in meta.keys():
                raise KeyError("Please provide 'value_col' to metadata "
                               "dictionary to point to column containing "
                               "timeseries values!")
            value_col = meta["value_col"]
            if isinstance(value_col, str):
                series = series.loc[:, value_col]
            elif isinstance(value_col, int):
                series = series.iloc[:, value_col]
            return series

    def get_metadata(self, libname: str, names: Union[list, str],
                     progressbar: bool = False, as_frame=True) -> Union[dict, pd.DataFrame]:
        """read metadata for dataset

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
        dict or dict of dicts
            dictionary containing metadata

        """
        import pystore
        lib = self.get_library(libname)

        metalist = []
        names = self._parse_names(names)
        for n in (tqdm(names) if progressbar else names):
            imeta = pystore.utils.read_metadata(lib._item_path(n))
            if "name" not in imeta.keys():
                imeta["name"] = n
            metalist.append(imeta)
        if as_frame:
            # convert to frame
            if len(metalist) > 1:
                meta = pd.DataFrame(metalist)
                if len({"x", "y"}.difference(meta.columns)) == 0:
                    meta["x"] = meta["x"].astype(float)
                    meta["y"] = meta["y"].astype(float)
                    # meta = gpd.GeoDataFrame(meta, geometry=[Point(
                    #     (s["x"], s["y"])) for i, s in meta.iterrows()])
            elif len(metalist) == 1:
                meta = pd.DataFrame(metalist)
            elif len(metalist) == 0:
                meta = pd.DataFrame()
            if "name" in meta.columns:
                meta.set_index("name", inplace=True)
            else:
                meta.index = names
            return meta
        else:
            if len(metalist) == 1:
                return metalist[0]
            else:
                return metalist

    def get_oseries(self, names: Union[list, str],
                    progressbar: bool = False) -> FrameorSeriesUnion:
        """load oseries from store

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
        """load stresses from store

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
        lib = self.get_library("models")

        models = []
        load_mod = import_module("pastas.io.pas")
        names = self._parse_names(names)
        for n in (tqdm(names) if progressbar else names):

            jsonpath = lib._item_path(n).joinpath("metadata.json")
            data = load_mod.load(jsonpath)

            if 'series' not in data['oseries']:
                name = data["oseries"]['name']
                if name not in self.oseries.index:
                    msg = 'oseries {} not present in project'.format(name)
                    raise(LookupError(msg))
                s = self.get_oseries(name, progressbar=False)
                data['oseries']['series'] = self._get_values(
                    "oseries", name, s)
            for ts in data["stressmodels"].values():
                if "stress" in ts.keys():
                    for stress in ts["stress"]:
                        if 'series' not in stress:
                            name = stress['name']
                            if name in self.stresses.index:
                                s = self.get_stresses(
                                    name, progressbar=False)
                            else:
                                msg = 'stress {} not present in project'.format(
                                    name)
                                raise KeyError(msg)
                            stress['series'] = self._get_values(
                                "stresses", name, s)

            ml = ps.io.base.load_model(data)
            models.append(ml)

        if len(models) == 1:
            return models[0]
        else:
            return models

    @staticmethod
    def _clear_cache(libname: str) -> None:
        """clear cache
        """
        getattr(PystoreConnector, libname).fget.cache_clear()

    @property
    @functools.lru_cache()
    def oseries(self):
        lib = self.get_library("oseries")
        df = self.get_metadata("oseries", lib.list_items())
        return df

    @property
    @functools.lru_cache()
    def stresses(self):
        lib = self.get_library("stresses")
        df = self.get_metadata("stresses", lib.list_items())
        return df

    @property
    @functools.lru_cache()
    def models(self):
        lib = self.get_library("models")
        if lib is not None:
            mls = lib.list_items()
        else:
            mls = []
        return mls
