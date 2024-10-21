"""Module containing classes for connecting to different data stores."""

import json
import os
import warnings
from copy import deepcopy
from typing import Dict, Optional, Union

import pandas as pd
from pastas.io.pas import PastasEncoder, pastas_hook

from pastastore.base import BaseConnector, ConnectorUtil, ModelAccessor
from pastastore.util import _custom_warning

FrameorSeriesUnion = Union[pd.DataFrame, pd.Series]
warnings.showwarning = _custom_warning


class ConnectorUtil:
    """Mix-in class for general Connector helper functions.

    Only for internal methods, and not methods that are related to CRUD operations on
    database.
    """

    def _parse_names(
        self,
        names: Optional[Union[list, str]] = None,
        libname: Optional[str] = "oseries",
    ) -> list:
        """Parse names kwarg, returns iterable with name(s) (internal method).

        Parameters
        ----------
        names : Union[list, str], optional
            str or list of str or None or 'all' (last two options
            retrieves all names)
        libname : str, optional
            name of library, default is 'oseries'

        Returns
        -------
        list
            list of names
        """
        if not isinstance(names, str) and isinstance(names, Iterable):
            return names
        elif isinstance(names, str) and names != "all":
            return [names]
        elif names is None or names == "all":
            if libname == "oseries":
                return self.oseries_names
            elif libname == "stresses":
                return self.stresses_names
            elif libname == "models":
                return self.model_names
            elif libname == "oseries_models":
                return self.oseries_with_models
            else:
                raise ValueError(f"No library '{libname}'!")
        else:
            raise NotImplementedError(f"Cannot parse 'names': {names}")

    @staticmethod
    def _meta_list_to_frame(metalist: list, names: list):
        """Convert list of metadata dictionaries to DataFrame.

        Parameters
        ----------
        metalist : list
            list of metadata dictionaries
        names : list
            list of names corresponding to data in metalist

        Returns
        -------
        pandas.DataFrame
            DataFrame containing overview of metadata
        """
        # convert to dataframe
        if len(metalist) > 1:
            meta = pd.DataFrame(metalist)
            if len({"x", "y"}.difference(meta.columns)) == 0:
                meta["x"] = meta["x"].astype(float)
                meta["y"] = meta["y"].astype(float)
        elif len(metalist) == 1:
            meta = pd.DataFrame(metalist)
        elif len(metalist) == 0:
            meta = pd.DataFrame()

        meta.index = names
        meta.index.name = "name"
        return meta

    def _parse_model_dict(self, mdict: dict, update_ts_settings: bool = False):
        """Parse dictionary describing pastas models (internal method).

        Parameters
        ----------
        mdict : dict
            dictionary describing pastas.Model
        update_ts_settings : bool, optional
            update stored tmin and tmax in time series settings
            based on time series loaded from store.

        Returns
        -------
        ml : pastas.Model
            time series analysis model
        """
        PASFILE_LEQ_022 = parse_version(
            mdict["file_info"]["pastas_version"]
        ) <= parse_version("0.22.0")

        # oseries
        if "series" not in mdict["oseries"]:
            name = str(mdict["oseries"]["name"])
            if name not in self.oseries.index:
                msg = "oseries '{}' not present in library".format(name)
                raise LookupError(msg)
            mdict["oseries"]["series"] = self.get_oseries(name).squeeze()
            # update tmin/tmax from time series
            if update_ts_settings:
                mdict["oseries"]["settings"]["tmin"] = mdict["oseries"]["series"].index[
                    0
                ]
                mdict["oseries"]["settings"]["tmax"] = mdict["oseries"]["series"].index[
                    -1
                ]

        # StressModel, WellModel
        for ts in mdict["stressmodels"].values():
            if "stress" in ts.keys():
                # WellModel
                classkey = "stressmodel" if PASFILE_LEQ_022 else "class"
                if ts[classkey] == "WellModel":
                    for stress in ts["stress"]:
                        if "series" not in stress:
                            name = str(stress["name"])
                            if name in self.stresses.index:
                                stress["series"] = self.get_stresses(name).squeeze()
                                # update tmin/tmax from time series
                                if update_ts_settings:
                                    stress["settings"]["tmin"] = stress["series"].index[
                                        0
                                    ]
                                    stress["settings"]["tmax"] = stress["series"].index[
                                        -1
                                    ]
                # StressModel
                else:
                    for stress in ts["stress"] if PASFILE_LEQ_022 else [ts["stress"]]:
                        if "series" not in stress:
                            name = str(stress["name"])
                            if name in self.stresses.index:
                                stress["series"] = self.get_stresses(name).squeeze()
                                # update tmin/tmax from time series
                                if update_ts_settings:
                                    stress["settings"]["tmin"] = stress["series"].index[
                                        0
                                    ]
                                    stress["settings"]["tmax"] = stress["series"].index[
                                        -1
                                    ]

            # RechargeModel, TarsoModel
            if ("prec" in ts.keys()) and ("evap" in ts.keys()):
                for stress in [ts["prec"], ts["evap"]]:
                    if "series" not in stress:
                        name = str(stress["name"])
                        if name in self.stresses.index:
                            stress["series"] = self.get_stresses(name).squeeze()
                            # update tmin/tmax from time series
                            if update_ts_settings:
                                stress["settings"]["tmin"] = stress["series"].index[0]
                                stress["settings"]["tmax"] = stress["series"].index[-1]
                        else:
                            msg = "stress '{}' not present in library".format(name)
                            raise KeyError(msg)

        # hack for pcov w dtype object (when filled with NaNs on store?)
        if "fit" in mdict:
            if "pcov" in mdict["fit"]:
                pcov = mdict["fit"]["pcov"]
                if pcov.dtypes.apply(lambda dtyp: isinstance(dtyp, object)).any():
                    mdict["fit"]["pcov"] = pcov.astype(float)

        # check pastas version vs pas-file version
        file_version = mdict["file_info"]["pastas_version"]

        # check file version and pastas version
        # if file<0.23  and pastas>=1.0 --> error
        PASTAS_GT_023 = parse_version(ps.__version__) > parse_version("0.23.1")
        if PASFILE_LEQ_022 and PASTAS_GT_023:
            raise UserWarning(
                f"This file was created with Pastas v{file_version} "
                f"and cannot be loaded with Pastas v{ps.__version__} Please load and "
                "save the file with Pastas 0.23 first to update the file "
                "format."
            )

        try:
            # pastas>=0.15.0
            ml = ps.io.base._load_model(mdict)
        except AttributeError:
            # pastas<0.15.0
            ml = ps.io.base.load_model(mdict)
        return ml

    @staticmethod
    def _validate_input_series(series):
        """Check if series is pandas.DataFrame or pandas.Series.

        Parameters
        ----------
        series : object
            object to validate

        Raises
        ------
        TypeError
            if object is not of type pandas.DataFrame or pandas.Series
        """
        if not (isinstance(series, pd.DataFrame) or isinstance(series, pd.Series)):
            raise TypeError("Please provide pandas.DataFrame or pandas.Series!")
        if isinstance(series, pd.DataFrame):
            if series.columns.size > 1:
                raise ValueError("Only DataFrames with one column are supported!")

    @staticmethod
    def _set_series_name(series, name):
        """Set series name to match user defined name in store.

        Parameters
        ----------
        series : pandas.Series or pandas.DataFrame
            set name for this time series
        name : str
            name of the time series (used in the pastastore)
        """
        if isinstance(series, pd.Series):
            series.name = name
            # empty string on index name causes trouble when reading
            # data from ArcticDB: TODO: check if still an issue?
            if series.index.name == "":
                series.index.name = None

        if isinstance(series, pd.DataFrame):
            series.columns = [name]
            # check for hydropandas objects which are instances of DataFrame but
            # do have a name attribute
            if hasattr(series, "name"):
                series.name = name
        return series

    @staticmethod
    def _check_stressmodels_supported(ml):
        supported_stressmodels = [
            "StressModel",
            "StressModel2",
            "RechargeModel",
            "WellModel",
            "TarsoModel",
            "Constant",
            "LinearTrend",
            "StepModel",
        ]
        if isinstance(ml, ps.Model):
            smtyps = [sm._name for sm in ml.stressmodels.values()]
        elif isinstance(ml, dict):
            classkey = "stressmodel" if PASTAS_LEQ_022 else "class"
            smtyps = [sm[classkey] for sm in ml["stressmodels"].values()]
        check = isin(smtyps, supported_stressmodels)
        if not all(check):
            unsupported = set(smtyps) - set(supported_stressmodels)
            raise NotImplementedError(
                "PastaStore does not support storing models with the "
                f"following stressmodels: {unsupported}"
            )

    @staticmethod
    def _check_model_series_names_for_store(ml):
        prec_evap_model = ["RechargeModel", "TarsoModel"]

        if isinstance(ml, ps.Model):
            series_names = [
                istress.series.name
                for sm in ml.stressmodels.values()
                for istress in sm.stress
            ]

        elif isinstance(ml, dict):
            # non RechargeModel, Tarsomodel, WellModel stressmodels
            classkey = "stressmodel" if PASTAS_LEQ_022 else "class"
            if PASTAS_LEQ_022:
                series_names = [
                    istress["name"]
                    for sm in ml["stressmodels"].values()
                    if sm[classkey] not in (prec_evap_model + ["WellModel"])
                    for istress in sm["stress"]
                ]
            else:
                series_names = [
                    sm["stress"]["name"]
                    for sm in ml["stressmodels"].values()
                    if sm[classkey] not in (prec_evap_model + ["WellModel"])
                ]

            # WellModel
            if isin(
                ["WellModel"],
                [i[classkey] for i in ml["stressmodels"].values()],
            ).any():
                series_names += [
                    istress["name"]
                    for sm in ml["stressmodels"].values()
                    if sm[classkey] in ["WellModel"]
                    for istress in sm["stress"]
                ]

            # RechargeModel, TarsoModel
            if isin(
                prec_evap_model,
                [i[classkey] for i in ml["stressmodels"].values()],
            ).any():
                series_names += [
                    istress["name"]
                    for sm in ml["stressmodels"].values()
                    if sm[classkey] in prec_evap_model
                    for istress in [sm["prec"], sm["evap"]]
                ]

        else:
            raise TypeError("Expected pastas.Model or dict!")
        if len(series_names) - len(set(series_names)) > 0:
            msg = (
                "There are multiple stresses series with the same name! "
                "Each series name must be unique for the PastaStore!"
            )
            raise ValueError(msg)

    def _check_oseries_in_store(self, ml: Union[ps.Model, dict]):
        """Check if Model oseries are contained in PastaStore (internal method).

        Parameters
        ----------
        ml : Union[ps.Model, dict]
            pastas Model
        """
        if isinstance(ml, ps.Model):
            name = ml.oseries.name
        elif isinstance(ml, dict):
            name = str(ml["oseries"]["name"])
        else:
            raise TypeError("Expected pastas.Model or dict!")
        if name not in self.oseries.index:
            msg = (
                f"Cannot add model because oseries '{name}' "
                "is not contained in store."
            )
            raise LookupError(msg)
        # expensive check
        if self.CHECK_MODEL_SERIES_VALUES and isinstance(ml, ps.Model):
            s_org = self.get_oseries(name).squeeze().dropna()
            if PASTAS_LEQ_022:
                so = ml.oseries.series_original
            else:
                so = ml.oseries._series_original
            try:
                assert_series_equal(
                    so.dropna(),
                    s_org,
                    atol=self.SERIES_EQUALITY_ABSOLUTE_TOLERANCE,
                    rtol=self.SERIES_EQUALITY_RELATIVE_TOLERANCE,
                )
            except AssertionError as e:
                raise ValueError(
                    f"Cannot add model because model oseries '{name}'"
                    " is different from stored oseries! See stacktrace for differences."
                ) from e

    def _check_stresses_in_store(self, ml: Union[ps.Model, dict]):
        """Check if stresses time series are contained in PastaStore (internal method).

        Parameters
        ----------
        ml : Union[ps.Model, dict]
            pastas Model
        """
        prec_evap_model = ["RechargeModel", "TarsoModel"]
        if isinstance(ml, ps.Model):
            for sm in ml.stressmodels.values():
                if sm._name in prec_evap_model:
                    stresses = [sm.prec, sm.evap]
                else:
                    stresses = sm.stress
                for s in stresses:
                    if str(s.name) not in self.stresses.index:
                        msg = (
                            f"Cannot add model because stress '{s.name}' "
                            "is not contained in store."
                        )
                        raise LookupError(msg)
                    if self.CHECK_MODEL_SERIES_VALUES:
                        s_org = self.get_stresses(s.name).squeeze()
                        if PASTAS_LEQ_022:
                            so = s.series_original
                        else:
                            so = s._series_original
                        try:
                            assert_series_equal(
                                so,
                                s_org,
                                atol=self.SERIES_EQUALITY_ABSOLUTE_TOLERANCE,
                                rtol=self.SERIES_EQUALITY_RELATIVE_TOLERANCE,
                            )
                        except AssertionError as e:
                            raise ValueError(
                                f"Cannot add model because model stress "
                                f"'{s.name}' is different from stored stress! "
                                "See stacktrace for differences."
                            ) from e
        elif isinstance(ml, dict):
            for sm in ml["stressmodels"].values():
                classkey = "stressmodel" if PASTAS_LEQ_022 else "class"
                if sm[classkey] in prec_evap_model:
                    stresses = [sm["prec"], sm["evap"]]
                elif sm[classkey] in ["WellModel"]:
                    stresses = sm["stress"]
                else:
                    stresses = sm["stress"] if PASTAS_LEQ_022 else [sm["stress"]]
                for s in stresses:
                    if str(s["name"]) not in self.stresses.index:
                        msg = (
                            f"Cannot add model because stress '{s['name']}' "
                            "is not contained in store."
                        )
                        raise LookupError(msg)
        else:
            raise TypeError("Expected pastas.Model or dict!")

    def _stored_series_to_json(
        self,
        libname: str,
        names: Optional[Union[list, str]] = None,
        squeeze: bool = True,
        progressbar: bool = False,
    ):
        """Write stored series to JSON.

        Parameters
        ----------
        libname : str
            library name
        names : Optional[Union[list, str]], optional
            names of series, by default None
        squeeze : bool, optional
            return single entry as json string instead
            of list, by default True
        progressbar : bool, optional
            show progressbar, by default False

        Returns
        -------
        files : list or str
            list of series converted to JSON string or single string
            if single entry is returned and squeeze is True
        """
        names = self._parse_names(names, libname=libname)
        files = []
        for n in tqdm(names, desc=libname) if progressbar else names:
            s = self._get_series(libname, n, progressbar=False)
            if isinstance(s, pd.Series):
                s = s.to_frame()
            try:
                sjson = s.to_json(orient="columns")
            except ValueError as e:
                msg = (
                    f"DatetimeIndex of '{n}' probably contains NaT "
                    "or duplicate timestamps!"
                )
                raise ValueError(msg) from e
            files.append(sjson)
        if len(files) == 1 and squeeze:
            return files[0]
        else:
            return files

    def _stored_metadata_to_json(
        self,
        libname: str,
        names: Optional[Union[list, str]] = None,
        squeeze: bool = True,
        progressbar: bool = False,
    ):
        """Write metadata from stored series to JSON.

        Parameters
        ----------
        libname : str
            library containing series
        names : Optional[Union[list, str]], optional
            names to parse, by default None
        squeeze : bool, optional
            return single entry as json string instead of list, by default True
        progressbar : bool, optional
            show progressbar, by default False

        Returns
        -------
        files : list or str
            list of json string
        """
        names = self._parse_names(names, libname=libname)
        files = []
        for n in tqdm(names, desc=libname) if progressbar else names:
            meta = self.get_metadata(libname, n, as_frame=False)
            meta_json = json.dumps(meta, cls=PastasEncoder, indent=4)
            files.append(meta_json)
        if len(files) == 1 and squeeze:
            return files[0]
        else:
            return files

    def _series_to_archive(
        self,
        archive,
        libname: str,
        names: Optional[Union[list, str]] = None,
        progressbar: bool = True,
    ):
        """Write DataFrame or Series to zipfile (internal method).

        Parameters
        ----------
        archive : zipfile.ZipFile
            reference to an archive to write data to
        libname : str
            name of the library to write to zipfile
        names : str or list of str, optional
            names of the time series to write to archive, by default None,
            which writes all time series to archive
        progressbar : bool, optional
            show progressbar, by default True
        """
        names = self._parse_names(names, libname=libname)
        for n in tqdm(names, desc=libname) if progressbar else names:
            sjson = self._stored_series_to_json(
                libname, names=n, progressbar=False, squeeze=True
            )
            meta_json = self._stored_metadata_to_json(
                libname, names=n, progressbar=False, squeeze=True
            )
            archive.writestr(f"{libname}/{n}.json", sjson)
            archive.writestr(f"{libname}/{n}_meta.json", meta_json)

    def _models_to_archive(self, archive, names=None, progressbar=True):
        """Write pastas.Model to zipfile (internal method).

        Parameters
        ----------
        archive : zipfile.ZipFile
            reference to an archive to write data to
        names : str or list of str, optional
            names of the models to write to archive, by default None,
            which writes all models to archive
        progressbar : bool, optional
            show progressbar, by default True
        """
        names = self._parse_names(names, libname="models")
        for n in tqdm(names, desc="models") if progressbar else names:
            m = self.get_models(n, return_dict=True)
            jsondict = json.dumps(m, cls=PastasEncoder, indent=4)
            archive.writestr(f"models/{n}.pas", jsondict)

    @staticmethod
    def _series_from_json(fjson: str, squeeze: bool = True):
        """Load time series from JSON.

        Parameters
        ----------
        fjson : str
            path to file
        squeeze : bool, optional
            squeeze time series object to obtain pandas Series

        Returns
        -------
        s : pd.DataFrame
            DataFrame containing time series
        """
        s = pd.read_json(fjson, orient="columns", precise_float=True, dtype=False)
        if not isinstance(s.index, pd.DatetimeIndex):
            s.index = pd.to_datetime(s.index, unit="ms")
        s = s.sort_index()  # needed for some reason ...
        if squeeze:
            return s.squeeze()
        return s

    @staticmethod
    def _metadata_from_json(fjson: str):
        """Load metadata dictionary from JSON.

        Parameters
        ----------
        fjson : str
            path to file

        Returns
        -------
        meta : dict
            dictionary containing metadata
        """
        with open(fjson, "r") as f:
            meta = json.load(f)
        return meta

    def _get_model_orphans(self):
        """Get models whose oseries no longer exist in database.

        Returns
        -------
        dict
            dictionary with oseries names as keys and lists of model names
            as values
        """
        d = {}
        for mlnam in tqdm(self.model_names, desc="Identifying model orphans"):
            mdict = self.get_models(mlnam, return_dict=True)
            onam = mdict["oseries"]["name"]
            if onam not in self.oseries_names:
                if onam in d:
                    d[onam] = d[onam].append(mlnam)
                else:
                    d[onam] = [mlnam]
        return d

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
            conn = connector
        else:
            conn = globals()["conn"]

        ml = conn.get_models(ml_name)
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
        except Exception as e:
            if ignore_solve_errors:
                warning = "Solve error ignored for '%s': %s " % (ml.name, e)
                logger.warning(warning)
            else:
                raise e

        conn.add_model(ml, overwrite=True)


class ArcticDBConnector(BaseConnector, ConnectorUtil):
    """ArcticDBConnector object using ArcticDB to store data."""

    conn_type = "arcticdb"

    def __init__(self, name: str, uri: str, verbose: bool = True):
        """Create an ArcticDBConnector object using ArcticDB to store data.

        Parameters
        ----------
        name : str
            name of the database
        uri : str
            URI connection string (e.g. 'lmdb://<your path here>')
        verbose : bool, optional
            whether to print message when database is initialized, by default True
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
        self._initialize(verbose=verbose)
        self.models = ModelAccessor(self)
        # for older versions of PastaStore, if oseries_models library is empty
        # populate oseries - models database
        self._update_all_oseries_model_links()

    def _initialize(self, verbose: bool = True) -> None:
        """Initialize the libraries (internal method)."""
        for libname in self._default_library_names:
            if self._library_name(libname) not in self.arc.list_libraries():
                self.arc.create_library(self._library_name(libname))
            else:
                if verbose:
                    print(
                        f"ArcticDBConnector: library "
                        f"'{self._library_name(libname)}'"
                        " already exists. Linking to existing library."
                    )
            self.libs[libname] = self._get_library(libname)

    def _library_name(self, libname: str) -> str:
        """Get full library name according to ArcticDB (internal method)."""
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
        # only normalizable datatypes can be written with write, else use write_pickle
        # normalizable: Series, DataFrames, Numpy Arrays
        if isinstance(item, (dict, list)):
            lib.write_pickle(name, item, metadata=metadata)
        else:
            lib.write(name, item, metadata=metadata)

    def _get_item(self, libname: str, name: str) -> Union[FrameorSeriesUnion, Dict]:
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

    def _del_item(self, libname: str, name: str) -> None:
        """Delete items (series or models) (internal method).

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
        progressbar: Optional[bool] = True,
        max_workers: Optional[int] = None,
        chunksize: Optional[int] = None,
        desc: str = "",
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
        """

        def initializer(*args):
            global conn
            conn = ArcticDBConnector(*args)

        initargs = (self.name, self.uri, False)

        if progressbar:
            with tqdm(total=len(names), desc=desc) as pbar:
                with ProcessPoolExecutor(
                    max_workers=max_workers, initializer=initializer, initargs=initargs
                ) as executor:
                    for _ in executor.map(func, names, chunksize=chunksize):
                        pbar.update()
        else:
            with ProcessPoolExecutor(
                max_workers=max_workers, initializer=initializer, initargs=initargs
            ) as executor:
                executor.map(func, names, chunksize=chunksize)

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


class DictConnector(BaseConnector, ConnectorUtil):
    """DictConnector object that stores timeseries and models in dictionaries."""

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
        if libname in ["models", "oseries_models"]:
            lib[name] = item
        else:
            lib[name] = (metadata, item)

    def _get_item(self, libname: str, name: str) -> Union[FrameorSeriesUnion, Dict]:
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
            time series or model dictionary
        """
        lib = self._get_library(libname)
        if libname in ["models", "oseries_models"]:
            item = deepcopy(lib[name])
        else:
            item = deepcopy(lib[name][1])
        return item

    def _del_item(self, libname: str, name: str) -> None:
        """Delete items (series or models) (internal method).

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
        raise NotImplementedError(
            "DictConnector does not support parallel processing,"
            " use PasConnector or ArcticDBConnector."
        )

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
    """PasConnector object that stores time series and models as JSON files on disk."""

    conn_type = "pas"

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
        self.name = name
        self.path = os.path.abspath(os.path.join(path, self.name))
        self.relpath = os.path.relpath(self.path)
        self._initialize(verbose=verbose)
        self.models = ModelAccessor(self)
        # for older versions of PastaStore, if oseries_models library is empty
        # populate oseries_models library
        self._update_all_oseries_model_links()

    def _initialize(self, verbose: bool = True) -> None:
        """Initialize the libraries (internal method)."""
        for val in self._default_library_names:
            libdir = os.path.join(self.path, val)
            if not os.path.exists(libdir):
                if verbose:
                    print(f"PasConnector: library '{val}' created in '{libdir}'")
                os.makedirs(libdir)
            else:
                if verbose:
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
        """Delete items (series or models) (internal method).

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
        mjson = os.path.join(lib, f"{name}_meta.pas")
        if os.path.isfile(mjson):
            imeta = self._metadata_from_json(mjson)
        else:
            imeta = {}
        return imeta

    def _parallel(
        self,
        func: Callable,
        names: List[str],
        progressbar: Optional[bool] = True,
        max_workers: Optional[int] = None,
        chunksize: Optional[int] = None,
        desc: str = "",
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
        """
        if progressbar:
            process_map(
                func,
                names,
                max_workers=max_workers,
                chunksize=chunksize,
                desc=desc,
                total=len(names),
            )
        else:
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                executor.map(func, names, chunksize=chunksize)

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
