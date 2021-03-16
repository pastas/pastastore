import json
import warnings
from abc import ABC, abstractmethod, abstractproperty
from collections.abc import Iterable
from typing import Optional, Union

import pandas as pd
import pastas as ps
from pastas import Model
from pastas.io.pas import PastasEncoder
from tqdm import tqdm

from .util import _custom_warning

FrameorSeriesUnion = Union[pd.DataFrame, pd.Series]
warnings.showwarning = _custom_warning


class BaseConnector(ABC):  # pragma: no cover
    """Metaclass for connecting to data management sources.

    For example, MongoDB through Arctic, Pystore, or other databases.
    Create your own connection to a data source by writing a a class
    that inherits from this BaseConnector. Your class has to override
    each method and property.
    """
    _default_library_names = ["oseries", "stresses", "models"]

    @abstractmethod
    def get_library(self, libname: str):
        """Get library handle.

        Parameters
        ----------
        libname : str,
            name of the library
        """
        pass

    @abstractmethod
    def add_oseries(self, series: FrameorSeriesUnion, name: str,
                    metadata: Union[dict, None] = None,
                    overwrite: bool = False, **kwargs) -> None:
        """Add oseries.

        Parameters
        ----------
        series : FrameorSeriesUnion
            pandas.Series or pandas.DataFrame to add
        name : str
            name of the series
        metadata : Optional[dict], optional
            dictionary containing metadata, by default None
        overwrite: bool, optional
            overwrite existing dataset with the same name,
            by default False
        """
        pass

    @abstractmethod
    def add_stress(self, series: FrameorSeriesUnion, name: str, kind: str,
                   metadata: Optional[dict] = None,
                   overwrite: bool = False, **kwargs) -> None:
        """Add stress.

        Parameters
        ----------
        series : FrameorSeriesUnion
            pandas.Series or pandas.DataFrame to add
        name : str
            name of the series
        kind : str
            label specifying type of stress (i.e. 'prec' or 'evap')
        metadata : Optional[dict], optional
            dictionary containing metadata, by default None
        overwrite: bool, optional
            overwrite existing dataset with the same name,
            by default False
        """
        pass

    @abstractmethod
    def add_model(self, ml: Model, overwrite: bool = False, **kwargs) -> None:
        """Add model.

        Parameters
        ----------
        ml : Model
            pastas.Model
        """
        pass

    @abstractmethod
    def del_models(self, names: Union[list, str]) -> None:
        """Delete model.

        Parameters
        ----------
        names : Union[list, str]
            str or list of str of model names to delete
        """
        pass

    @abstractmethod
    def del_oseries(self, names: Union[list, str]) -> None:
        """Delete oseries.

        Parameters
        ----------
        names : Union[list, str]
            str or list of str of oseries names to delete
        """
        pass

    @abstractmethod
    def del_stress(self, names: Union[list, str]) -> None:
        """Delete stresses.

        Parameters
        ----------
        names : Union[list, str]
            str or list of str of stresses to delete
        """
        pass

    @abstractmethod
    def get_metadata(self, libname: str, names: Union[list, str],
                     progressbar: bool = False, as_frame: bool = True) \
            -> Union[pd.DataFrame, dict]:
        """Get metadata for oseries or stress.

        Parameters
        ----------
        libname : str
            name of library
        names : Union[list, str]
            str or list of str of series to get metadata for
        progressbar : bool, optional
            show progressbar, by default False
        as_frame : bool, optional
            return as dataframe, by default True

        Returns
        -------
        Union[pd.DataFrame, dict]
            dictionary or pandas.DataFrame depending on value of `as_frame`.
        """
        pass

    @abstractmethod
    def get_oseries(self, names: Union[list, str],
                    progressbar: bool = False) -> FrameorSeriesUnion:
        """Get oseries.

        Parameters
        ----------
        names : Union[list, str]
            str or list of str of names of oseries to retrieve
        progressbar : bool, optional
            show progressbar, by default False

        Returns
        -------
        dict, pandas.DataFrame
            return dictionary containing data if multiple names are passed,
            else return pandas.DataFrame or pandas.Series
        """
        pass

    @abstractmethod
    def get_stresses(self, names: Union[list, str],
                     progressbar: bool = False) -> FrameorSeriesUnion:
        """Get stresses.

        Parameters
        ----------
        names : Union[list, str]
            str or list of str of names of stresses to retrieve
        progressbar : bool, optional
            show progressbar, by default False

        Returns
        -------
        dict, pandas.DataFrame
            return dictionary containing data if multiple names are passed,
            else return pandas.DataFrame or pandas.Series
        """
        pass

    @abstractmethod
    def get_models(self, names: Union[list, str], progressbar: bool = False,
                   **kwargs) -> Union[Model, dict]:
        """Get models.

        Parameters
        ----------
        names : Union[list, str]
            str or list of str of models to retrieve
        progressbar : bool, optional
            show progressbar, by default False

        Returns
        -------
        Union[Model, dict]
            return pastas.Model if only one name is passed, else return dict
        """
        pass

    @abstractproperty
    def oseries(self):
        """Dataframe containing oseries overview."""
        pass

    @abstractproperty
    def stresses(self):
        """Dataframe containing stresses overview."""
        pass

    @abstractproperty
    def models(self):
        """List of model names."""
        pass


class ConnectorUtil:
    """Mix-in class for general Connector helper functions."""

    def _parse_names(self, names: Optional[Union[list, str]] = None,
                     libname: Optional[str] = "oseries") -> list:
        """Internal method to parse names kwarg, returns iterable with name(s).

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
                return getattr(self, "oseries").index.to_list()
            elif libname == "stresses":
                return getattr(self, "stresses").index.to_list()
            elif libname == "models":
                return getattr(self, "models")
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

    def _parse_model_dict(self, mdict: dict):
        """Internal method to parse dictionary describing pastas models.

        Parameters
        ----------
        mdict : dict
            dictionary describing pastas.Model

        Returns
        -------
        ml : pastas.Model
            timeseries analysis model
        """
        if 'series' not in mdict['oseries']:
            name = mdict["oseries"]['name']
            if name not in self.oseries.index:
                msg = 'oseries {} not present in project'.format(name)
                raise LookupError(msg)
            mdict['oseries']['series'] = self.get_oseries(name)
        for ts in mdict["stressmodels"].values():
            if "stress" in ts.keys():
                for stress in ts["stress"]:
                    if 'series' not in stress:
                        name = stress['name']
                        if name in self.stresses.index:
                            stress['series'] = self.get_stresses(name)
                        else:
                            msg = "stress '{}' not present in project".format(
                                name)
                            raise KeyError(msg)
        try:
            # pastas>=0.15.0
            ml = ps.io.base._load_model(mdict)
        except AttributeError:
            # pastas<0.15.0
            ml = ps.io.base.load_model(mdict)
        return ml

    @staticmethod
    def _validate_input_series(series):
        """check if series is pandas.DataFrame or pandas.Series.

        Parameters
        ----------
        series : object
            object to validate

        Raises
        ------
        TypeError
            if object is not of type pandas.DataFrame or pandas.Series
        """
        if not (isinstance(series, pd.DataFrame) or
                isinstance(series, pd.Series)):
            raise TypeError("Please provide pandas.DataFrame"
                            " or pandas.Series!")
        if isinstance(series, pd.DataFrame):
            if series.columns.size > 1:
                raise ValueError("Only DataFrames with one "
                                 "column are supported!")

    @staticmethod
    def _set_series_name(series, name):
        """Set series name to match user defined name in store.

        Parameters
        ----------
        series : pandas.Series or pandas.DataFrame
            set name for this timeseries
        name : str
            name of the timeseries (used in the pastastore)
        """
        if isinstance(series, pd.Series):
            series.name = name
            # empty string on index name causes trouble when reading
            # data from Arctic VersionStores
            if series.index.name == "":
                series.index.name = None

        if isinstance(series, pd.DataFrame):
            series.columns = [name]
        return series

    @staticmethod
    def _check_model_series_for_store(ml):
        if isinstance(ml, ps.Model):
            series_names = [istress.series.name
                            for sm in ml.stressmodels.values()
                            if sm._name != "RechargeModel"
                            for istress in sm.stress]
            if "RechargeModel" in [i._name for i in ml.stressmodels.values()]:
                series_names += [istress.series.name
                                 for sm in ml.stressmodels.values()
                                 if sm._name == "RechargeModel"
                                 for istress in sm.stress]
        elif isinstance(ml, dict):
            # non RechargeModel stressmodels
            series_names = [istress["name"] for sm in
                            ml["stressmodels"].values()
                            if sm["stressmodel"] != "RechargeModel"
                            for istress in sm["stress"]]
            # RechargeModel
            if "RechargeModel" in [i["stressmodel"] for i in
                                   ml["stressmodels"].values()]:
                series_names += [istress["name"] for sm in
                                 ml["stressmodels"].values()
                                 if sm["stressmodel"] == "RechargeModel"
                                 for istress in [sm["prec"], sm["evap"]]]
        else:
            raise TypeError("Expected pastas.Model or dict!")
        if len(series_names) - len(set(series_names)) > 0:
            msg = ("There are multiple stresses series with the same name! "
                   "Each series name must be unique for the PastaStore!")
            raise ValueError(msg)

    def _check_oseries_in_store(self, ml: Union[ps.Model, dict]):
        """Internal method, check if Model oseries are contained in PastaStore.

        Parameters
        ----------
        ml : Union[ps.Model, dict]
            pastas Model
        """
        if isinstance(ml, ps.Model):
            name = ml.oseries.name
        elif isinstance(ml, dict):
            name = ml["oseries"]["name"]
        else:
            raise TypeError("Expected pastas.Model or dict!")
        if name not in self.oseries.index:
            msg = (f"Cannot add model because oseries '{name}' "
                   "is not contained in store.")
            raise LookupError(msg)

    def _check_stresses_in_store(self, ml: Union[ps.Model, dict]):
        """Internal method, check if stresses timeseries are contained in
        PastaStore.

        Parameters
        ----------
        ml : Union[ps.Model, dict]
            pastas Model
        """
        if isinstance(ml, ps.Model):
            for sm in ml.stressmodels.values():
                if sm._name == "RechargeModel":
                    stresses = [sm.prec, sm.evap]
                else:
                    stresses = sm.stress
                for s in stresses:
                    if s.name not in self.stresses.index:
                        msg = (f"Cannot add model because stress '{s.name}' "
                               "is not contained in store.")
                        raise LookupError(msg)
        elif isinstance(ml, dict):
            for sm in ml["stressmodels"].values():
                if sm["stressmodel"] == "RechargeModel":
                    stresses = [sm["prec"], sm["evap"]]
                else:
                    stresses = sm["stress"]
                for s in stresses:
                    if s["name"] not in self.stresses.index:
                        msg = (f"Cannot add model because stress '{s.name}' "
                               "is not contained in store.")
                        raise LookupError(msg)
        else:
            raise TypeError("Expected pastas.Model or dict!")

    def _stored_series_to_json(self,
                               libname: str,
                               names: Optional[Union[list, str]] = None,
                               squeeze: bool = True,
                               progressbar: bool = False):
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
        for n in (tqdm(names, desc=libname) if progressbar else names):
            s = self._get_series(libname, n, progressbar=False)
            if isinstance(s, pd.Series):
                s = s.to_frame()
            sjson = s.to_json(orient="columns")
            files.append(sjson)
        if len(files) == 1 and squeeze:
            return files[0]
        else:
            return files

    def _stored_metadata_to_json(self,
                                 libname: str,
                                 names: Optional[Union[list, str]] = None,
                                 squeeze: bool = True,
                                 progressbar: bool = False):
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
        for n in (tqdm(names, desc=libname) if progressbar else names):
            meta = self.get_metadata(libname, n, as_frame=False)
            meta_json = json.dumps(meta, cls=PastasEncoder, indent=4)
            files.append(meta_json)
        if len(files) == 1 and squeeze:
            return files[0]
        else:
            return files

    def _series_to_archive(self, archive, libname: str,
                           names: Optional[Union[list, str]] = None,
                           progressbar: bool = True):
        """Internal method for writing DataFrame or Series to zipfile.

        Parameters
        ----------
        archive : zipfile.ZipFile
            reference to an archive to write data to
        libname : str
            name of the library to write to zipfile
        names : str or list of str, optional
            names of the timeseries to write to archive, by default None,
            which writes all timeseries to archive
        progressbar : bool, optional
            show progressbar, by default True
        """
        names = self._parse_names(names, libname=libname)
        for n in (tqdm(names, desc=libname) if progressbar else names):
            sjson = self._stored_series_to_json(
                libname, names=n, progressbar=False, squeeze=True)
            meta_json = self._stored_metadata_to_json(
                libname, names=n, progressbar=False, squeeze=True)
            archive.writestr(f"{libname}/{n}.json", sjson)
            archive.writestr(f"{libname}/{n}_meta.json", meta_json)

    def _models_to_archive(self, archive, names=None, progressbar=True):
        """Internal method for writing pastas.Model to zipfile.

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
        for n in (tqdm(names, desc="models") if progressbar else names):
            m = self.get_models(n, return_dict=True)
            jsondict = json.dumps(m, cls=PastasEncoder, indent=4)
            archive.writestr(f"models/{n}.pas", jsondict)

    @staticmethod
    def _series_from_json(fjson: str):
        """Load timeseries from JSON.

        Parameters
        ----------
        fjson : str
            path to file

        Returns
        -------
        s : pd.DataFrame
            DataFrame containing timeseries
        """
        s = pd.read_json(fjson, orient="columns")
        if not isinstance(s.index, pd.DatetimeIndex):
            s.index = pd.to_datetime(s.index, unit='ms')
        s = s.sort_index()  # needed for some reason ...
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
