"""Module containing classes for connecting to different data stores."""

import json
import logging
import os
import warnings
from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy
from functools import partial
from pathlib import Path

# import weakref
from typing import Callable, Dict, List, Optional, Tuple, Union

import pandas as pd
from pastas.io.pas import PastasEncoder, pastas_hook
from tqdm.auto import tqdm
from tqdm.contrib.concurrent import process_map

from pastastore.base import BaseConnector, ModelAccessor
from pastastore.typing import AllLibs, FrameOrSeriesUnion, TimeSeriesLibs
from pastastore.util import _custom_warning, metadata_from_json, series_from_json
from pastastore.validator import Validator

warnings.showwarning = _custom_warning

logger = logging.getLogger(__name__)

# Global connector for multiprocessing workaround
# This is required for connectors (like ArcticDBConnector) that cannot be pickled.
# The initializer function in _parallel() sets this global variable in each worker
# process, allowing unpicklable connectors to be used with multiprocessing.
# See: https://docs.python.org/3/library/concurrent.futures.html#processpoolexecutor
# Note: Using simple None type to avoid circular import issues
conn = None


class ParallelUtil:
    """Mix-in class for storing parallelizable methods."""

    @staticmethod
    def _solve_model(
        ml_name: str,
        connector: Optional[BaseConnector] = None,
        report: bool = False,
        ignore_solve_errors: bool = False,
        **kwargs,
    ) -> None:
        """Solve a model in the store (internal method).

        ml_name : list of str, optional
            name of a model in the pastastore
        connector : PasConnector, optional
            Connector to use, by default None which gets the global ArcticDB
            Connector. Otherwise parse a PasConnector.
        report : boolean, optional
            determines if a report is printed when the model is solved,
            default is False
        ignore_solve_errors : boolean, optional
            if True, errors emerging from the solve method are ignored,
            default is False which will raise an exception when a model
            cannot be optimized
        **kwargs : dictionary
            arguments are passed to the solve method.
        """
        if connector is not None:
            _conn = connector
        else:
            _conn = globals()["conn"]

        ml = _conn.get_models(ml_name)
        m_kwargs = {}
        for key, value in kwargs.items():
            if isinstance(value, pd.Series):
                m_kwargs[key] = value.loc[ml.name]
            else:
                m_kwargs[key] = value
        # Convert timestamps
        for tstamp in ["tmin", "tmax"]:
            if tstamp in m_kwargs:
                m_kwargs[tstamp] = pd.Timestamp(m_kwargs[tstamp])

        try:
            ml.solve(report=report, **m_kwargs)
        except Exception as e:  # pylint: disable=broad-except
            if ignore_solve_errors:
                warning = f"Solve error ignored for '{ml.name}': {e}"
                logger.warning(warning)
            else:
                raise e
        # store the updated model back in the database
        _conn.add_model(ml, overwrite=True)

    @staticmethod
    def _get_statistics(
        name: str,
        statistics: List[str],
        connector: Union[None, BaseConnector] = None,
        **kwargs,
    ) -> pd.Series:
        """Get statistics for a model in the store (internal method).

        This function was made to be run in parallel mode. For the odd user
        that wants to run this function directly in sequential model using
        an ArcticDBDConnector the connector argument must be passed in the kwargs
        of the apply method.
        """
        if connector is not None:
            _conn = connector
        else:
            _conn = globals()["conn"]

        ml = _conn.get_model(name)
        series = pd.Series(index=statistics, dtype=float)
        for stat in statistics:
            # Note: ml.stats is part of pastas.Model public API
            series.loc[stat] = getattr(ml.stats, stat)(**kwargs)
        return series

    @staticmethod
    def _get_max_workers_and_chunksize(
        max_workers: int, njobs: int, chunksize: int = None
    ) -> Tuple[int, int]:
        """Get the maximum workers and chunksize for parallel processing.

        From: https://stackoverflow.com/a/42096963/10596229
        """
        max_workers = (
            min(32, os.cpu_count() + 4) if max_workers is None else max_workers
        )
        if chunksize is None:
            # 14 chunks per worker balances overhead vs granularity
            # from stackoverflow link posted in docstring.
            CHUNKS_PER_WORKER = 14
            num_chunks = max_workers * CHUNKS_PER_WORKER
            chunksize = max(njobs // num_chunks, 1)
        return max_workers, chunksize


class ArcticDBConnector(BaseConnector, ParallelUtil):
    """ArcticDBConnector object using ArcticDB to store data."""

    _conn_type = "arcticdb"

    def __init__(self, name: str, uri: str, verbose: bool = True):
        """Create an ArcticDBConnector object using ArcticDB to store data.

        Parameters
        ----------
        name : str
            name of the database
        uri : str
            URI connection string (e.g. 'lmdb://<your path here>')
        verbose : bool, optional
            whether to log messages when database is initialized, by default True
        """
        try:
            import arcticdb

        except ModuleNotFoundError as e:
            logger.error("Please install arcticdb with `pip install arcticdb`!")
            raise e

        # avoid warn on all metadata writes
        from arcticdb_ext import set_config_string

        set_config_string("PickledMetadata.LogLevel", "DEBUG")

        self.uri = uri
        self.name = name

        # initialize validator class to check inputs
        self._validator = Validator(self)

        # create libraries
        self.libs: dict = {}
        self.arc = arcticdb.Arctic(uri)
        self._initialize(verbose=verbose)
        self.models = ModelAccessor(self)
        # for older versions of PastaStore, if oseries_models library is empty
        # populate oseries - models database
        if (self.n_models > 0) and (
            len(self.oseries_models) == 0 or len(self.stresses_models) == 0
        ):
            self._update_time_series_model_links(recompute=False, progressbar=True)
        # write pstore file to store database info that can be used to load pstore
        if "lmdb" in self.uri:
            self.write_pstore_config_file()

    def _initialize(self, verbose: bool = True) -> None:
        """Initialize the libraries (internal method)."""
        if "lmdb" in self.uri.lower():  # only check for LMDB
            self.validator.check_config_connector_type(
                Path(self.uri.split("://")[1]) / self.name
            )
        for libname in self._default_library_names:
            if self._library_name(libname) not in self.arc.list_libraries():
                self.arc.create_library(self._library_name(libname))
            else:
                if verbose:
                    logger.info(
                        "ArcticDBConnector: library '%s' already exists. "
                        "Linking to existing library.",
                        self._library_name(libname),
                    )
            self.libs[libname] = self._get_library(libname)

    def write_pstore_config_file(self, path: str = None) -> None:
        """Write pstore configuration file to store database info."""
        # NOTE: method is not private as theoretically an ArcticDB
        # database could also be hosted in the cloud, in which case,
        # writing this config in the folder holding the database
        # is no longer possible. For those situations, the user can
        # write this config file and specify the path it should be
        # written to.
        config = {
            "connector_type": self.conn_type,
            "name": self.name,
            "uri": self.uri,
        }
        if path is None and "lmdb" in self.uri:
            path = Path(self.uri.split("://")[1])
        elif path is None and "lmdb" not in self.uri:
            raise ValueError("Please provide a path to write the pastastore file!")

        with (path / self.name / f"{self.name}.pastastore").open(
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(config, f)

    def _library_name(self, libname: AllLibs) -> str:
        """Get full library name according to ArcticDB (internal method)."""
        return ".".join([self.name, libname])

    def _get_library(self, libname: AllLibs):
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
        if libname in self.libs:
            return self.libs[libname]
        else:
            return self.arc.get_library(self._library_name(libname))

    def _add_item(
        self,
        libname: AllLibs,
        item: Union[FrameOrSeriesUnion, Dict],
        name: str,
        metadata: Optional[Dict] = None,
        **_,
    ) -> None:
        """Add item to library (time series or model) (internal method).

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

        # check file name for illegal characters
        name = self.validator.check_filename_illegal_chars(libname, name)

        # only normalizable datatypes can be written with write, else use write_pickle
        # normalizable: Series, DataFrames, Numpy Arrays
        if isinstance(item, (dict, list)):
            logger.debug(
                "Writing pickled item '%s' to ArcticDB library '%s'.", name, libname
            )
            lib.write_pickle(name, item, metadata=metadata)
        else:
            logger.debug("Writing item '%s' to ArcticDB library '%s'.", name, libname)
            lib.write(name, item, metadata=metadata)

    def _get_item(self, libname: AllLibs, name: str) -> Union[FrameOrSeriesUnion, Dict]:
        """Retrieve item from library (internal method).

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

    def _del_item(self, libname: AllLibs, name: str, force: bool = False) -> None:
        """Delete items (series or models) (internal method).

        Parameters
        ----------
        libname : str
            name of library to delete item from
        name : str
            name of item to delete
        force : bool, optional
            force deletion even if series is used in models, by default False
        """
        lib = self._get_library(libname)
        if self.validator.PROTECT_SERIES_IN_MODELS and not force:
            self.validator.check_series_in_models(libname, name)
        lib.delete(name)

    def _get_metadata(self, libname: TimeSeriesLibs, name: str) -> dict:
        """Retrieve metadata for an item (internal method).

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

    def _parallel(
        self,
        func: Callable,
        names: List[str],
        kwargs: Optional[Dict] = None,
        progressbar: Optional[bool] = True,
        max_workers: Optional[int] = None,
        chunksize: Optional[int] = None,
        desc: str = "",
        initializer: Callable = None,
        initargs: Optional[tuple] = None,
    ):
        """Parallel processing of function.

        Does not return results, so function must store results in database.

        Note
        ----
        ArcticDB connection objects cannot be pickled, which is required for
        multiprocessing. This implementation uses an initializer function that
        creates a new ArcticDBConnector instance in each worker process and stores
        it in the global `conn` variable. User-provided functions can access this
        connector via the global `conn` variable.

        This is the standard Python multiprocessing pattern for unpicklable objects.
        See: https://docs.python.org/3/library/concurrent.futures.html#processpoolexecutor

        For a connector that supports direct method passing (no global variable
        required), use PasConnector instead.

        Parameters
        ----------
        func : function
            function to apply in parallel
        names : list
            list of names to apply function to
        kwargs : dict, optional
            keyword arguments to pass to function
        progressbar : bool, optional
            show progressbar, by default True
        max_workers : int, optional
            maximum number of workers, by default None
        chunksize : int, optional
            chunksize for parallel processing, by default None
        desc : str, optional
            description for progressbar, by default ""
        initializer : Callable, optional
            function to initialize each worker process, by default None
        initargs : tuple, optional
            arguments to pass to initializer function, by default None
        """
        max_workers, chunksize = self._get_max_workers_and_chunksize(
            max_workers, len(names), chunksize
        )
        if initializer is None:

            def initializer(*args):
                # assign to module-level variable without using 'global' statement
                globals()["conn"] = ArcticDBConnector(*args)

            initargs = (self.name, self.uri, False)

        if initargs is None:
            initargs = ()

        if kwargs is None:
            kwargs = {}

        if progressbar:
            result = []
            with tqdm(total=len(names), desc=desc) as pbar:
                with ProcessPoolExecutor(
                    max_workers=max_workers, initializer=initializer, initargs=initargs
                ) as executor:
                    for item in executor.map(
                        partial(func, **kwargs), names, chunksize=chunksize
                    ):
                        result.append(item)
                        pbar.update()
        else:
            with ProcessPoolExecutor(
                max_workers=max_workers, initializer=initializer, initargs=initargs
            ) as executor:
                result = executor.map(
                    partial(func, **kwargs), names, chunksize=chunksize
                )

        # update links if models were stored
        self._trigger_links_update_if_needed(modelnames=names)

        return result

    def _list_symbols(self, libname: AllLibs) -> List[str]:
        """List symbols in a library (internal method).

        Parameters
        ----------
        libname : str
            name of the library

        Returns
        -------
        list
            list of symbols in the library
        """
        return self._get_library(libname).list_symbols()


class DictConnector(BaseConnector, ParallelUtil):
    """DictConnector object that stores timeseries and models in dictionaries."""

    _conn_type = "dict"

    def __init__(self, name: str = "pastas_db"):
        """Create DictConnector object that stores data in dictionaries.

        Parameters
        ----------
        name : str, optional
            user-specified name of the connector
        """
        super().__init__()
        self.name = name

        # create empty dictionaries for series and models
        for val in self._default_library_names:
            setattr(self, "lib_" + val, {})
        self._validator = Validator(self)
        self.models = ModelAccessor(self)
        # for older versions of PastaStore, if oseries_models library is empty
        # populate oseries - models database
        if (self.n_models > 0) and (
            len(self.oseries_models) == 0 or len(self.stresses_models) == 0
        ):
            self._update_time_series_model_links(recompute=False, progressbar=True)

    def _get_library(self, libname: AllLibs):
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
        item: Union[FrameOrSeriesUnion, Dict],
        name: str,
        metadata: Optional[Dict] = None,
        **_,
    ) -> None:
        """Add item (time series or models) (internal method).

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

        # check file name for illegal characters
        name = self.validator.check_filename_illegal_chars(libname, name)

        if libname in ["models", "oseries_models", "stresses_models"]:
            lib[name] = item
        else:
            lib[name] = (metadata, item)

    def _get_item(self, libname: AllLibs, name: str) -> Union[FrameOrSeriesUnion, Dict]:
        """Retrieve item from database (internal method).

        Parameters
        ----------
        libname : str
            name of the library
        name : str
            name of the item

        Returns
        -------
        item : Union[FrameorSeriesUnion, Dict]
            time series or model dictionary, modifying the returned object will not
            affect the stored data, like in a real database
        """
        lib = self._get_library(libname)
        # deepcopy calls are needed to ensure users cannot change "stored" items
        if libname in ["models", "oseries_models", "stresses_models"]:
            item = deepcopy(lib[name])
        else:
            item = deepcopy(lib[name][1])
        return item

    def _del_item(self, libname: AllLibs, name: str, force: bool = False) -> None:
        """Delete items (series or models) (internal method).

        Parameters
        ----------
        libname : str
            name of library to delete item from
        name : str
            name of item to delete
        force : bool, optional
            if True, force delete item and do not perform check if series
            is used in a model, by default False
        """
        if self.validator.PROTECT_SERIES_IN_MODELS and not force:
            self.validator.check_series_in_models(libname, name)
        lib = self._get_library(libname)
        _ = lib.pop(name)

    def _get_metadata(self, libname: TimeSeriesLibs, name: str) -> dict:
        """Read metadata (internal method).

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

    def _parallel(self, *args, **kwargs) -> None:
        """Parallel implementation method.

        Raises
        ------
        NotImplementedError
            DictConnector uses in-memory storage that cannot be shared across
            processes. Use PasConnector or ArcticDBConnector for parallel operations.
        """
        raise NotImplementedError(
            "DictConnector does not support parallel processing,"
            " use PasConnector or ArcticDBConnector."
        )

    def _list_symbols(self, libname: AllLibs) -> List[str]:
        """List symbols in a library (internal method).

        Parameters
        ----------
        libname : str
            name of the library

        Returns
        -------
        list
            list of symbols in the library
        """
        lib = self._get_library(libname)
        return list(lib.keys())


class PasConnector(BaseConnector, ParallelUtil):
    """PasConnector object that stores time series and models as JSON files on disk."""

    _conn_type = "pas"

    def __init__(self, name: str, path: str, verbose: bool = True):
        """Create PasConnector object that stores data as JSON files on disk.

        Uses Pastas export format (pas-files) to store files.

        Parameters
        ----------
        name : str
            user-specified name of the connector, this will be the name of the
            directory in which the data will be stored
        path : str
            path to directory for storing the data
        verbose : bool, optional
            whether to print message when database is initialized, by default True
        """
        # set shared memory flags for parallel processing
        super().__init__()
        self.name = name
        self.parentdir = Path(path)
        self.path = (self.parentdir / self.name).absolute()
        self.relpath = os.path.relpath(self.parentdir)
        self._validator = Validator(self)
        self._initialize(verbose=verbose)
        self.models = ModelAccessor(self)

        # for older versions of PastaStore, if oseries_models library is empty
        # populate oseries_models library
        if (self.n_models > 0) and (
            len(self.oseries_models) == 0 or len(self.stresses_models) == 0
        ):
            self._update_time_series_model_links(recompute=False, progressbar=True)
        # write pstore file to store database info that can be used to load pstore
        self._write_pstore_config_file()

    def _initialize(self, verbose: bool = True) -> None:
        """Initialize the libraries (internal method)."""
        self.validator.check_config_connector_type(self.path)
        for val in self._default_library_names:
            libdir = self.path / val
            if not libdir.exists():
                if verbose:
                    logger.info(
                        "PasConnector: library '%s' created in '%s'", val, libdir
                    )
                libdir.mkdir(parents=True, exist_ok=False)
            else:
                if verbose:
                    logger.info(
                        "PasConnector: library '%s' already exists. "
                        "Linking to existing directory: '%s'",
                        val,
                        libdir,
                    )
            setattr(self, f"lib_{val}", self.path / val)

    def _write_pstore_config_file(self):
        """Write pstore configuration file to store database info."""
        config = {
            "connector_type": self.conn_type,
            "name": self.name,
            "path": str(self.parentdir.absolute()),
        }
        with (self.path / f"{self.name}.pastastore").open("w", encoding="utf-8") as f:
            json.dump(config, f)

    def _get_library(self, libname: AllLibs) -> Path:
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
        return Path(getattr(self, "lib_" + libname))

    def _add_item(
        self,
        libname: str,
        item: Union[FrameOrSeriesUnion, Dict],
        name: str,
        metadata: Optional[Dict] = None,
        **_,
    ) -> None:
        """Add item (time series or models) (internal method).

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

        # check file name for illegal characters
        name = self.validator.check_filename_illegal_chars(libname, name)

        # time series
        if isinstance(item, pd.Series):
            item = item.to_frame()
        if isinstance(item, pd.DataFrame):
            sjson = item.to_json(orient="columns")
            if name.endswith("_meta"):
                raise ValueError(
                    "Time series name cannot end with '_meta'. "
                    "Please use a different name for your time series."
                )
            fname = lib / f"{name}.pas"
            with fname.open("w", encoding="utf-8") as f:
                logger.debug("Writing time series '%s' to disk at '%s'.", name, fname)
                f.write(sjson)
            if metadata is not None:
                mjson = json.dumps(metadata, cls=PastasEncoder, indent=4)
                fname_meta = lib / f"{name}_meta.pas"
                with fname_meta.open("w", encoding="utf-8") as m:
                    logger.debug(
                        "Writing metadata '%s' to disk at '%s'.", name, fname_meta
                    )
                    m.write(mjson)
        # pastas model dict
        elif isinstance(item, dict):
            jsondict = json.dumps(item, cls=PastasEncoder, indent=4)
            fmodel = lib / f"{name}.pas"
            with fmodel.open("w", encoding="utf-8") as fm:
                logger.debug("Writing model '%s' to disk at '%s'.", name, fmodel)
                fm.write(jsondict)
        # oseries_models or stresses_models list
        elif isinstance(item, list):
            jsondict = json.dumps(item)
            fname = lib / f"{name}.pas"
            with fname.open("w", encoding="utf-8") as fm:
                logger.debug("Writing link list '%s' to disk at '%s'.", name, fname)
                fm.write(jsondict)

    def _get_item(self, libname: AllLibs, name: str) -> Union[FrameOrSeriesUnion, Dict]:
        """Retrieve item (internal method).

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
        fjson = lib / f"{name}.pas"
        if not fjson.exists():
            msg = f"Item '{name}' not in '{libname}' library."
            raise FileNotFoundError(msg)
        # model
        if libname == "models":
            with fjson.open("r", encoding="utf-8") as ml_json:
                item = json.load(ml_json, object_hook=pastas_hook)
        # list of models per oseries
        elif libname in ["oseries_models", "stresses_models"]:
            with fjson.open("r", encoding="utf-8") as f:
                item = json.load(f)
        # time series
        else:
            item = series_from_json(fjson)
        return item

    def _del_item(self, libname: AllLibs, name: str, force: bool = False) -> None:
        """Delete items (series or models) (internal method).

        Parameters
        ----------
        libname : str
            name of library to delete item from
        name : str
            name of item to delete
        force : bool, optional
            if True, force delete item and do not perform check if series
            is used in a model, by default False
        """
        lib = self._get_library(libname)
        if self.validator.PROTECT_SERIES_IN_MODELS and not force:
            self.validator.check_series_in_models(libname, name)
        (lib / f"{name}.pas").unlink()
        # remove metadata for time series
        if libname in ["oseries", "stresses"]:
            try:
                (lib / f"{name}_meta.pas").unlink()
            except FileNotFoundError:
                # Nothing to delete
                pass

    def _get_metadata(self, libname: TimeSeriesLibs, name: str) -> dict:
        """Read metadata (internal method).

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
        mjson = lib / f"{name}_meta.pas"
        if mjson.is_file():
            imeta = metadata_from_json(mjson)
        else:
            imeta = {}
        return imeta

    def _parallel(
        self,
        func: Callable,
        names: List[str],
        kwargs: Optional[dict] = None,
        progressbar: Optional[bool] = True,
        max_workers: Optional[int] = None,
        chunksize: Optional[int] = None,
        desc: str = "",
        initializer: Callable = None,
        initargs: Optional[tuple] = None,
    ):
        """Parallel processing of function.

        Does not return results, so function must store results in database.

        Parameters
        ----------
        func : function
            function to apply in parallel
        names : list
            list of names to apply function to
        progressbar : bool, optional
            show progressbar, by default True
        max_workers : int, optional
            maximum number of workers, by default None
        chunksize : int, optional
            chunksize for parallel processing, by default None
        desc : str, optional
            description for progressbar, by default ""
        initializer : Callable, optional
            function to initialize each worker process, by default None
        initargs : tuple, optional
            arguments to pass to initializer function, by default None
        """
        max_workers, chunksize = self._get_max_workers_and_chunksize(
            max_workers, len(names), chunksize
        )

        if kwargs is None:
            kwargs = {}

        if progressbar:
            if initializer is not None:
                result = []
                with tqdm(total=len(names), desc=desc) as pbar:
                    with ProcessPoolExecutor(
                        max_workers=max_workers,
                        initializer=initializer,
                        initargs=initargs,
                    ) as executor:
                        for item in executor.map(
                            partial(func, **kwargs), names, chunksize=chunksize
                        ):
                            result.append(item)
                            pbar.update()
            else:
                result = process_map(
                    partial(func, **kwargs),
                    names,
                    max_workers=max_workers,
                    chunksize=chunksize,
                    desc=desc,
                    total=len(names),
                )
        else:
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                result = executor.map(
                    partial(func, **kwargs), names, chunksize=chunksize
                )

        # update links if models were stored
        self._trigger_links_update_if_needed(modelnames=names)

        return result

    def _list_symbols(self, libname: AllLibs) -> List[str]:
        """List symbols in a library (internal method).

        Parameters
        ----------
        libname : str
            name of the library

        Returns
        -------
        list
            list of symbols in the library
        """
        lib = self._get_library(libname)
        return [i.stem for i in lib.glob("*.pas") if not i.stem.endswith("_meta")]
