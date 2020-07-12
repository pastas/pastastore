import json
from abc import ABC, abstractmethod, abstractproperty
from collections.abc import Iterable
from typing import Optional, Union
from warnings import warn

import pandas as pd
from tqdm import tqdm

import pastas as ps
from pastas import Model
from pastas.io.pas import PastasEncoder

FrameorSeriesUnion = Union[pd.DataFrame, pd.Series]


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
                    metadata: Union[dict, None] = None, **kwargs) -> None:
        """Add oseries.

        Parameters
        ----------
        series : FrameorSeriesUnion
            pandas.Series or pandas.DataFrame to add
        name : str
            name of the series
        metadata : Optional[dict], optional
            dictionary containing metadata, by default None
        """
        pass

    @abstractmethod
    def add_stress(self, series: FrameorSeriesUnion, name: str, kind: str,
                   metadata: Optional[dict] = None, **kwargs) -> None:
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
        """
        pass

    @abstractmethod
    def add_model(self, ml: Model, **kwargs) -> None:
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
                     progressbar: bool = False, as_frame: bool = True) -> Union[pd.DataFrame, dict]:
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
    def get_models(self, names: Union[list, str],
                   progressbar: bool = False) -> Union[Model, dict]:
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
            str or list of str or None (retrieves all names)
        libname : str, optional
            name of library, default is 'oseries'

        Returns
        -------
        list
            list of names
        """
        if names is None:
            if libname == "oseries":
                return getattr(self, "oseries").index.to_list()
            elif libname == "stresses":
                return getattr(self, "stresses").index.to_list()
            elif libname == "models":
                return getattr(self, "models")
            else:
                raise ValueError(f"No library '{libname}'!")
        elif isinstance(names, str):
            return [names]
        elif isinstance(names, Iterable):
            return names
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
            s = self.get_oseries(name)
            mdict['oseries']['series'] = self._get_dataframe_values(
                "oseries", name, s)
        for ts in mdict["stressmodels"].values():
            if "stress" in ts.keys():
                for stress in ts["stress"]:
                    if 'series' not in stress:
                        name = stress['name']
                        if name in self.stresses.index:
                            s = self.get_stresses(name)
                            stress['series'] = self._get_dataframe_values(
                                "stresses", name, s)
                        else:
                            msg = "stress '{}' not present in project".format(
                                name)
                            raise KeyError(msg)
        ml = ps.io.base.load_model(mdict)
        return ml

    def _get_dataframe_values(self, libname: str, name: str,
                              series: FrameorSeriesUnion,
                              metadata: Optional[dict] = None) \
            -> FrameorSeriesUnion:
        """Internal method to get column containing values from data.

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

    @staticmethod
    def _validate_metadata_multi_column(metadata):
        """Internal method for validating metadata dict. Checks whether user
        has identified which column is holding the relevant data, otherwise
        assumes relevant data is in column 0.

        Parameters
        ----------
        metadata : dict
            add "value_col" entry to metadata if not provided
        """
        if metadata is None:
            warn("'series' contains multiple columns, "
                 "assuming values in column 0! Pass metadata "
                 "and set 'value_col' to name or index number of "
                 "of data column to silence this warning.")
            metadata = {"value_col": 0}
        elif not "value_col" in metadata.keys():
            warn("'series' contains multiple columns, "
                 "assuming values in column 0! Pass metadata "
                 "and set 'value_col' to name or index number of "
                 "of data column to silence this warning.")
            metadata["value_col"] = 0
        return metadata

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

    def _check_oseries_in_store(self, ml: Union[ps.Model, dict]):
        """Internal method, check if Model oseries are contained in PastaStore.
        If not add series to the oseries library.

        Parameters
        ----------
        ml : Union[ps.Model, dict]
            pastas Model
        """
        if isinstance(ml, ps.Model):
            name = ml.oseries.name
            meta = ml.oseries.metadata
            s = ml.oseries.series.copy()
        elif isinstance(ml, dict):
            name = ml["oseries"]["name"]
            meta = ml["oseries"]["metadata"]
            try:
                s = ml["oseries"]["series"]
            except KeyError:
                s = None
        else:
            raise TypeError("Expected pastas.Model or dict!")
        if name not in self.oseries.index:
            warn(f"Timeseries '{name}' is not contained in store.")
            if s is not None:
                warn(f"Adding '{name}' to 'oseries' store!")
                s.index.name = None
                self.add_oseries(s, name, metadata=meta)

    def _check_stresses_in_store(self, ml: Union[ps.Model, dict]):
        """Internal method, check if stresses timeseries are contained in
        PastaStore. If not, add timeseries to the stresses library.

        Parameters
        ----------
        ml : Union[ps.Model, dict]
            pastas Model
        """
        if isinstance(ml, ps.Model):
            for sm in ml.stressmodels.values():
                for s in sm.stress:
                    if s.name not in self.stresses.index:
                        warn(f"Timeseries '{s.name}' "
                             "is not contained in store.")
                        warn(f"Adding '{s.name}' to 'stresses' store!")
                        ss = s.series
                        ss.index.name = None
                        self.add_stress(ss, s.name,
                                        kind=s.metadata["kind"],
                                        metadata=s.metadata)
        elif isinstance(ml, dict):
            for sm in ml["stressmodels"].values():
                for s in sm["stress"]:
                    if s["name"] not in self.stresses.index:
                        warn(f"Timeseries '{s['name']}' "
                             "is not contained in store.")
                        if "series" in s:
                            warn(f"Adding '{s['name']}' to 'stresses' store!")
                            ss = s["series"]
                            ss.index.name = None
                            self.add_stress(ss, s["name"],
                                            kind=s["metadata"]["kind"],
                                            metadata=s["metadata"])
        else:
            raise TypeError("Expected pastas.Model or dict!")

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
            s = self._get_series(libname, n, progressbar=False)
            if isinstance(s, pd.Series):
                s = s.to_frame()
            sjson = s.to_json(orient="columns")
            archive.writestr(f"{libname}/{n}.json", sjson)
            
            meta = self.get_metadata(libname, n, as_frame=False)
            meta_json = json.dumps(meta, cls=PastasEncoder, indent=4)
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


if __name__ == "__main__":

    import pastastore as pst
    import pandas as pd

    conn = pst.DictConnector("test")
    store = pst.PastaStore("test", conn)

    o = pd.read_csv("./tests/data/obs.csv", index_col=0, parse_dates=True)
    o.index.name = "oseries1"
    store.add_oseries(o, "oseries1", metadata={"name": "oseries1",
                                               "x": 100000,
                                               "y": 400000})

    ml = store.create_model("oseries1", add_recharge=False)

    store.del_oseries("oseries1")

    store.add_model(ml)
