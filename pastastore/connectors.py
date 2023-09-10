import json
import os
import warnings
from copy import deepcopy
from importlib import import_module
from typing import Dict, Optional, Union

import pandas as pd
from pastas.io.pas import PastasEncoder, pastas_hook

from pastastore.base import BaseConnector, ConnectorUtil, ModelAccessor
from pastastore.util import _custom_warning

FrameorSeriesUnion = Union[pd.DataFrame, pd.Series]
warnings.showwarning = _custom_warning


class ArcticConnector(BaseConnector, ConnectorUtil):  # pragma: no cover
    conn_type = "arctic"

    def __init__(self, name: str, connstr: str):
        """Create an ArcticConnector object that connects to a running MongoDB
        database via Arctic.

        Parameters
        ----------
        name : str
            name of the database
        connstr : str
            connection string (e.g. 'mongodb://localhost:27017/')
        """
        try:
            import arctic
        except ModuleNotFoundError as e:
            print(
                "Please install arctic (also requires "
                "a MongoDB instance running somewhere, e.g. "
                "MongoDB Community: \n"
                "https://docs.mongodb.com/manual/administration"
                "/install-community/)!"
            )
            raise e
        self.connstr = connstr
        self.name = name

        self.libs: dict = {}
        self.arc = arctic.Arctic(connstr)
        self._initialize()
        self.models = ModelAccessor(self)
        # for older versions of PastaStore, if oseries_models library is empty
        # populate oseries - models database
        self._update_all_oseries_model_links()

    def _initialize(self) -> None:
        """Internal method to initalize the libraries."""

        for libname in self._default_library_names:
            if self._library_name(libname) not in self.arc.list_libraries():
                self.arc.initialize_library(self._library_name(libname))
            else:
                print(
                    f"ArcticConnector: library "
                    f"'{self._library_name(libname)}'"
                    " already exists. Linking to existing library."
                )
            self.libs[libname] = self._get_library(libname)

    def _library_name(self, libname: str) -> str:
        """Internal method to get full library name according to Arctic."""
        return ".".join([self.name, libname])

    def _get_library(self, libname: str):
        """Get Arctic library handle.

        Parameters
        ----------
        libname : str
            name of the library

        Returns
        -------
        lib : arctic.Library handle
            handle to the library
        """
        # get library handle
        lib = self.arc.get_library(self._library_name(libname))
        return lib

    def _add_item(
        self,
        libname: str,
        item: Union[FrameorSeriesUnion, Dict],
        name: str,
        metadata: Optional[Dict] = None,
        **_,
    ) -> None:
        """Internal method to add item to library (time series or model).

        Parameters
        ----------
        libname : str
            name of the library
        item : Union[FrameorSeriesUnion, Dict]
            item to add, either time series or pastas.Model as dictionary
        name : str
            name of the item
        metadata : Optional[Dict], optional
            dictionary containing metadata, by default None
        """
        lib = self._get_library(libname)
        lib.write(name, item, metadata=metadata)

    def _get_item(self, libname: str, name: str) -> Union[FrameorSeriesUnion, Dict]:
        """Internal method to retrieve item from library.

        Parameters
        ----------
        libname : str
            name of the library
        name : str
            name of the item

        Returns
        -------
        item : Union[FrameorSeriesUnion, Dict]
            time series or model dictionary
        """
        lib = self._get_library(libname)
        return lib.read(name).data

    def _del_item(self, libname: str, name: str) -> None:
        """Internal method to delete items (series or models).

        Parameters
        ----------
        libname : str
            name of library to delete item from
        name : str
            name of item to delete
        """
        lib = self._get_library(libname)
        lib.delete(name)

    def _get_metadata(self, libname: str, name: str) -> dict:
        """Internal method to retrieve metadata for an item.

        Parameters
        ----------
        libname : str
            name of the library
        name : str
            name of the item

        Returns
        -------
        dict
            dictionary containing metadata
        """
        lib = self._get_library(libname)
        return lib.read_metadata(name).metadata

    @property
    def oseries_names(self):
        """List of oseries names.

        Returns
        -------
        list
            list of oseries in library
        """
        return self._get_library("oseries").list_symbols()

    @property
    def stresses_names(self):
        """List of stresses names.

        Returns
        -------
        list
            list of stresses in library
        """
        return self._get_library("stresses").list_symbols()

    @property
    def model_names(self):
        """List of model names.

        Returns
        -------
        list
            list of models in library
        """
        return self._get_library("models").list_symbols()

    @property
    def oseries_with_models(self):
        """List of oseries with models."""
        return self._get_library("oseries_models").list_symbols()


class ArcticDBConnector(BaseConnector, ConnectorUtil):
    conn_type = "arcticdb"

    def __init__(self, name: str, uri: str):
        """Create an ArcticDBConnector object using ArcticDB to store data.

        Parameters
        ----------
        name : str
            name of the database
        uri : str
            URI connection string (e.g. 'lmdb://<your path here>')
        """
        try:
            import arcticdb
        except ModuleNotFoundError as e:
            print("Please install arcticdb with `pip install arcticdb`!")
            raise e
        self.uri = uri
        self.name = name

        self.libs: dict = {}
        self.arc = arcticdb.Arctic(uri)
        self._initialize()
        self.models = ModelAccessor(self)
        # for older versions of PastaStore, if oseries_models library is empty
        # populate oseries - models database
        self._update_all_oseries_model_links()

    def _initialize(self) -> None:
        """Internal method to initalize the libraries."""

        for libname in self._default_library_names:
            if self._library_name(libname) not in self.arc.list_libraries():
                self.arc.create_library(self._library_name(libname))
            else:
                print(
                    f"ArcticDBConnector: library "
                    f"'{self._library_name(libname)}'"
                    " already exists. Linking to existing library."
                )
            self.libs[libname] = self._get_library(libname)

    def _library_name(self, libname: str) -> str:
        """Internal method to get full library name according to ArcticDB."""
        return ".".join([self.name, libname])

    def _get_library(self, libname: str):
        """Get ArcticDB library handle.

        Parameters
        ----------
        libname : str
            name of the library

        Returns
        -------
        lib : arcticdb.Library handle
            handle to the library
        """
        # get library handle
        lib = self.arc.get_library(self._library_name(libname))
        return lib

    def _add_item(
        self,
        libname: str,
        item: Union[FrameorSeriesUnion, Dict],
        name: str,
        metadata: Optional[Dict] = None,
        **_,
    ) -> None:
        """Internal method to add item to library (time series or model).

        Parameters
        ----------
        libname : str
            name of the library
        item : Union[FrameorSeriesUnion, Dict]
            item to add, either time series or pastas.Model as dictionary
        name : str
            name of the item
        metadata : Optional[Dict], optional
            dictionary containing metadata, by default None
        """
        lib = self._get_library(libname)
        # only normalizable datatypes can be written with write, else use write_pickle
        # normalizable: Series, DataFrames, Numpy Arrays
        if isinstance(item, (dict, list)):
            lib.write_pickle(name, item, metadata=metadata)
        else:
            lib.write(name, item, metadata=metadata)

    def _get_item(self, libname: str, name: str) -> Union[FrameorSeriesUnion, Dict]:
        """Internal method to retrieve item from library.

        Parameters
        ----------
        libname : str
            name of the library
        name : str
            name of the item

        Returns
        -------
        item : Union[FrameorSeriesUnion, Dict]
            time series or model dictionary
        """
        lib = self._get_library(libname)
        return lib.read(name).data

    def _del_item(self, libname: str, name: str) -> None:
        """Internal method to delete items (series or models).

        Parameters
        ----------
        libname : str
            name of library to delete item from
        name : str
            name of item to delete
        """
        lib = self._get_library(libname)
        lib.delete(name)

    def _get_metadata(self, libname: str, name: str) -> dict:
        """Internal method to retrieve metadata for an item.

        Parameters
        ----------
        libname : str
            name of the library
        name : str
            name of the item

        Returns
        -------
        dict
            dictionary containing metadata
        """
        lib = self._get_library(libname)
        return lib.read_metadata(name).metadata

    @property
    def oseries_names(self):
        """List of oseries names.

        Returns
        -------
        list
            list of oseries in library
        """
        return self._get_library("oseries").list_symbols()

    @property
    def stresses_names(self):
        """List of stresses names.

        Returns
        -------
        list
            list of stresses in library
        """
        return self._get_library("stresses").list_symbols()

    @property
    def model_names(self):
        """List of model names.

        Returns
        -------
        list
            list of models in library
        """
        return self._get_library("models").list_symbols()

    @property
    def oseries_with_models(self):
        """List of oseries with models."""
        return self._get_library("oseries_models").list_symbols()


class PystoreConnector(BaseConnector, ConnectorUtil):  # pragma: no cover
    conn_type = "pystore"

    def __init__(self, name: str, path: str):
        """Create a PystoreConnector object that points to a Pystore.

        Parameters
        ----------
        name : str
            name of the store
        path : str
            path to the pystore directory
        """
        try:
            import pystore
        except ModuleNotFoundError as e:
            print(
                "Install pystore, follow instructions at "
                "https://github.com/ranaroussi/pystore#dependencies"
            )
            raise e
        self.name = name
        self.path = path
        pystore.set_path(self.path)
        self.store = pystore.store(self.name)
        self.libs: dict = {}
        self._initialize()
        self.models = ModelAccessor(self)
        # for older versions of PastaStore, if oseries_models library is empty
        # populate oseries - models database
        self._update_all_oseries_model_links()

    def _initialize(self) -> None:
        """Internal method to initalize the libraries (stores)."""
        for libname in self._default_library_names:
            if libname in self.store.list_collections():
                print(
                    f"PystoreConnector: library '{self.path}/{libname}' "
                    "already exists. Linking to existing library."
                )
            lib = self.store.collection(libname)
            self.libs[libname] = lib

    def _get_library(self, libname: str):
        """Get Pystore library handle.

        Parameters
        ----------
        libname : str
            name of the library

        Returns
        -------
        Pystore.Collection handle
            handle to the library
        """
        # get library handle
        lib = self.store.collection(libname)
        return lib

    def _add_item(
        self,
        libname: str,
        item: Union[FrameorSeriesUnion, Dict],
        name: str,
        metadata: Optional[Dict] = None,
        overwrite: bool = False,
    ) -> None:
        """Internal method to add item to library (time series or model).

        Parameters
        ----------
        libname : str
            name of the library
        item : Union[FrameorSeriesUnion, Dict]
            item to add, either time series or pastas.Model as dictionary
        name : str
            name of the item
        metadata : Optional[Dict], optional
            dictionary containing metadata, by default None
        overwrite : bool, optional
            overwrite item if it already exists, by default False.
        """
        # convert to DataFrame because pystore doesn't accept pandas.Series
        # (maybe has an easy fix, but converting w to_frame for now)
        if isinstance(item, pd.Series):
            s = item.to_frame(name=name)
            is_type = "series"
        elif isinstance(item, dict):
            s = pd.DataFrame()  # empty DataFrame as placeholder
            jsondict = json.loads(json.dumps(item, cls=PastasEncoder, indent=4))
            metadata = jsondict  # model dict is stored in metadata
            is_type = "series"
        elif isinstance(item, list):
            s = pd.Series(item).to_frame(name="modelnames")
            is_type = "list"
        elif isinstance(item, pd.DataFrame):
            s = item
            is_type = "dataframe"

        # store info about input type to ensure same type is returned
        if metadata is None:
            metadata = {"_is_type": is_type}
        else:
            metadata["_is_type"] = is_type

        lib = self._get_library(libname)
        lib.write(name, s, metadata=metadata, overwrite=overwrite)

    def _get_item(self, libname: str, name: str) -> Union[FrameorSeriesUnion, Dict]:
        """Internal method to retrieve item from pystore library.

        Parameters
        ----------
        libname : str
            name of the library
        name : str
            name of the item

        Returns
        -------
        item : Union[FrameorSeriesUnion, Dict]
            time series or model dictionary
        """
        load_mod = import_module("pastas.io.pas")  # type: ignore
        lib = self._get_library(libname)
        # hack for storing models, stored as metadata
        if libname == "models":
            jsonpath = lib._item_path(name).joinpath("metadata.json")
            s = load_mod.load(jsonpath)  # type: ignore
        else:
            # read series and convert to pandas
            item = lib.item(name)
            s = item.to_pandas()
            # remove _is_type key and return correct type
            is_type = item.metadata.pop("_is_type")
            if is_type == "series":
                s = s.squeeze()
            elif is_type == "list":
                s = s["modelnames"].tolist()
        return s

    def _del_item(self, libname: str, name: str) -> None:
        """Internal method to delete data from the store.

        Parameters
        ----------
        libname : str
            name of the library
        name : str
            name of the item to delete
        """
        lib = self._get_library(libname)
        lib.delete_item(name)
        self._clear_cache(libname)

    def _get_metadata(self, libname: str, name: str) -> dict:
        """Internal method to read metadata from pystore.

        Parameters
        ----------
        libname : str
            name of the library the series are in ("oseries" or "stresses")
        name : str
            name of item to load metadata for

        Returns
        -------
        imeta : dict
            dictionary containing metadata
        """
        from pystore.utils import read_metadata

        lib = self._get_library(libname)
        imeta = read_metadata(lib._item_path(name))
        if "name" not in imeta.keys():
            imeta["name"] = name
        if "_is_type" in imeta.keys():
            imeta.pop("_is_type")
        return imeta

    @property
    def oseries_names(self):
        """List of oseries names.

        Returns
        -------
        list
            list of oseries in library
        """
        return list(self._get_library("oseries").list_items())

    @property
    def stresses_names(self):
        """List of stresses names.

        Returns
        -------
        list
            list of stresses in library
        """
        return list(self._get_library("stresses").list_items())

    @property
    def model_names(self):
        """List of model names.

        Returns
        -------
        list
            list of models in library
        """
        return list(self._get_library("models").list_items())

    @property
    def oseries_with_models(self):
        """List of oseries with models."""
        return list(self._get_library("oseries_models").list_items())


class DictConnector(BaseConnector, ConnectorUtil):
    conn_type = "dict"

    def __init__(self, name: str = "pastas_db"):
        """Create DictConnector object that stores data in dictionaries.

        Parameters
        ----------
        name : str, optional
            user-specified name of the connector
        """
        self.name = name

        # create empty dictionaries for series and models
        for val in self._default_library_names:
            setattr(self, "lib_" + val, {})
        self.models = ModelAccessor(self)
        # for older versions of PastaStore, if oseries_models library is empty
        # populate oseries - models database
        self._update_all_oseries_model_links()

    def _get_library(self, libname: str):
        """Get reference to dictionary holding data.

        Parameters
        ----------
        libname : str
            name of the library

        Returns
        -------
        lib : dict
            library handle
        """
        return getattr(self, f"lib_{libname}")

    def _add_item(
        self,
        libname: str,
        item: Union[FrameorSeriesUnion, Dict],
        name: str,
        metadata: Optional[Dict] = None,
        **_,
    ) -> None:
        """Internal method to add item (time series or models).

        Parameters
        ----------
        libname : str
            name of library
        item : FrameorSeriesUnion
            pandas.Series or pandas.DataFrame containing data
        name : str
            name of the item
        metadata : dict, optional
            dictionary containing metadata, by default None
        """
        lib = self._get_library(libname)
        if libname in ["models", "oseries_models"]:
            lib[name] = item
        else:
            lib[name] = (metadata, item)

    def _get_item(self, libname: str, name: str) -> Union[FrameorSeriesUnion, Dict]:
        """Internal method to retrieve item from pystore library.

        Parameters
        ----------
        libname : str
            name of the library
        name : str
            name of the item

        Returns
        -------
        item : Union[FrameorSeriesUnion, Dict]
            time series or model dictionary
        """
        lib = self._get_library(libname)
        if libname in ["models", "oseries_models"]:
            item = deepcopy(lib[name])
        else:
            item = deepcopy(lib[name][1])
        return item

    def _del_item(self, libname: str, name: str) -> None:
        """Internal method to delete items (series or models).

        Parameters
        ----------
        libname : str
            name of library to delete item from
        name : str
            name of item to delete
        """
        lib = self._get_library(libname)
        _ = lib.pop(name)

    def _get_metadata(self, libname: str, name: str) -> dict:
        """Internal method to read metadata.

        Parameters
        ----------
        libname : str
            name of the library the series are in ("oseries" or "stresses")
        name : str
            name of item to load metadata for

        Returns
        -------
        imeta : dict
            dictionary containing metadata
        """
        lib = self._get_library(libname)
        imeta = deepcopy(lib[name][0])
        return imeta

    @property
    def oseries_names(self):
        """List of oseries names."""
        lib = self._get_library("oseries")
        return list(lib.keys())

    @property
    def stresses_names(self):
        """List of stresses names."""
        lib = self._get_library("stresses")
        return list(lib.keys())

    @property
    def model_names(self):
        """List of model names."""
        lib = self._get_library("models")
        return list(lib.keys())

    @property
    def oseries_with_models(self):
        """List of oseries with models."""
        lib = self._get_library("oseries_models")
        return list(lib.keys())


class PasConnector(BaseConnector, ConnectorUtil):
    conn_type = "pas"

    def __init__(self, name: str, path: str):
        """Create PasConnector object that stores data as JSON files on disk.

        Uses Pastas export format (pas-files) to store files.

        Parameters
        ----------
        name : str
            user-specified name of the connector, this will be the name of the
            directory in which the data will be stored
        path : str
            path to directory for storing the data
        """
        self.name = name
        self.path = os.path.abspath(os.path.join(path, self.name))
        self.relpath = os.path.relpath(self.path)
        self._initialize()
        self.models = ModelAccessor(self)
        # for older versions of PastaStore, if oseries_models library is empty
        # populate oseries_models library
        self._update_all_oseries_model_links()

    def _initialize(self) -> None:
        """Internal method to initialize the libraries."""
        for val in self._default_library_names:
            libdir = os.path.join(self.path, val)
            if not os.path.exists(libdir):
                print(f"PasConnector: library '{val}' created in '{libdir}'")
                os.makedirs(libdir)
            else:
                print(
                    f"PasConnector: library '{val}' already exists. "
                    f"Linking to existing directory: '{libdir}'"
                )
            setattr(self, f"lib_{val}", os.path.join(self.path, val))

    def _get_library(self, libname: str):
        """Get path to directory holding data.

        Parameters
        ----------
        libname : str
            name of the library

        Returns
        -------
        lib : str
            path to library
        """
        return getattr(self, "lib_" + libname)

    def _add_item(
        self,
        libname: str,
        item: Union[FrameorSeriesUnion, Dict],
        name: str,
        metadata: Optional[Dict] = None,
        **_,
    ) -> None:
        """Internal method to add item (time series or models).

        Parameters
        ----------
        libname : str
            name of library
        item : FrameorSeriesUnion
            pandas.Series or pandas.DataFrame containing data
        name : str
            name of the item
        metadata : dict, optional
            dictionary containing metadata, by default None
        """
        lib = self._get_library(libname)

        # time series
        if isinstance(item, pd.Series):
            item = item.to_frame()
        if isinstance(item, pd.DataFrame):
            sjson = item.to_json(orient="columns")
            fname = os.path.join(lib, f"{name}.pas")
            with open(fname, "w") as f:
                f.write(sjson)
            if metadata is not None:
                mjson = json.dumps(metadata, cls=PastasEncoder, indent=4)
                fname_meta = os.path.join(lib, f"{name}_meta.pas")
                with open(fname_meta, "w") as m:
                    m.write(mjson)
        # pastas model dict
        elif isinstance(item, dict):
            jsondict = json.dumps(item, cls=PastasEncoder, indent=4)
            fmodel = os.path.join(lib, f"{name}.pas")
            with open(fmodel, "w") as fm:
                fm.write(jsondict)
        # oseries_models list
        elif isinstance(item, list):
            jsondict = json.dumps(item)
            fname = os.path.join(lib, f"{name}.pas")
            with open(fname, "w") as fm:
                fm.write(jsondict)

    def _get_item(self, libname: str, name: str) -> Union[FrameorSeriesUnion, Dict]:
        """Internal method to retrieve item.

        Parameters
        ----------
        libname : str
            name of the library
        name : str
            name of the item

        Returns
        -------
        item : Union[FrameorSeriesUnion, Dict]
            time series or model dictionary
        """
        lib = self._get_library(libname)
        fjson = os.path.join(lib, f"{name}.pas")
        if not os.path.exists(fjson):
            msg = f"Item '{name}' not in '{libname}' library."
            raise FileNotFoundError(msg)
        # model
        if libname == "models":
            with open(fjson, "r") as ml_json:
                item = json.load(ml_json, object_hook=pastas_hook)
        # list of models per oseries
        elif libname == "oseries_models":
            with open(fjson, "r") as f:
                item = json.load(f)
        # time series
        else:
            item = self._series_from_json(fjson)
        return item

    def _del_item(self, libname: str, name: str) -> None:
        """Internal method to delete items (series or models).

        Parameters
        ----------
        libname : str
            name of library to delete item from
        name : str
            name of item to delete
        """
        lib = self._get_library(libname)
        os.remove(os.path.join(lib, f"{name}.pas"))
        # remove metadata for time series
        if libname != "models":
            try:
                os.remove(os.path.join(lib, f"{name}_meta.pas"))
            except FileNotFoundError:
                # Nothing to delete
                pass

    def _get_metadata(self, libname: str, name: str) -> dict:
        """Internal method to read metadata.

        Parameters
        ----------
        libname : str
            name of the library the series are in ("oseries" or "stresses")
        name : str
            name of item to load metadata for

        Returns
        -------
        imeta : dict
            dictionary containing metadata
        """
        lib = self._get_library(libname)
        mjson = os.path.join(lib, f"{name}_meta.pas")
        if os.path.isfile(mjson):
            imeta = self._metadata_from_json(mjson)
        else:
            imeta = {}
        return imeta

    @property
    def oseries_names(self):
        """List of oseries names."""
        lib = self._get_library("oseries")
        return [
            i[:-4]
            for i in os.listdir(lib)
            if i.endswith(".pas")
            if not i.endswith("_meta.pas")
        ]

    @property
    def stresses_names(self):
        """List of stresses names."""
        lib = self._get_library("stresses")
        return [
            i[:-4]
            for i in os.listdir(lib)
            if i.endswith(".pas")
            if not i.endswith("_meta.pas")
        ]

    @property
    def model_names(self):
        """List of model names."""
        lib = self._get_library("models")
        return [i[:-4] for i in os.listdir(lib) if i.endswith(".pas")]

    @property
    def oseries_with_models(self):
        """List of oseries with models."""
        lib = self._get_library("oseries_models")
        return [i[:-4] for i in os.listdir(lib) if i.endswith(".pas")]
