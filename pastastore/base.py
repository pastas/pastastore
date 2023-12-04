import functools
import json
import warnings
from abc import ABC, abstractmethod, abstractproperty
from collections.abc import Iterable
from itertools import chain
from typing import Dict, List, Optional, Tuple, Union

import pandas as pd
import pastas as ps
from numpy import isin
from packaging.version import parse as parse_version
from pastas.io.pas import PastasEncoder
from tqdm.auto import tqdm

from pastastore.util import ItemInLibraryException, _custom_warning, validate_names
from pastastore.version import PASTAS_LEQ_022

FrameorSeriesUnion = Union[pd.DataFrame, pd.Series]
warnings.showwarning = _custom_warning


class BaseConnector(ABC):
    """Base Connector class.

    Class holds base logic for dealing with time series and Pastas
    Models. Create your own Connector to a data source by writing a a
    class that inherits from this BaseConnector. Your class has to
    override each abstractmethod and abstractproperty.
    """

    _default_library_names = [
        "oseries",
        "stresses",
        "models",
        "oseries_models",
    ]

    # whether to check model time series contents against stored copies
    CHECK_MODEL_SERIES_VALUES = True

    # whether to validate time series according to pastas rules
    # True for pastas>=0.23.0 and False for pastas<=0.22.0
    USE_PASTAS_VALIDATE_SERIES = False if PASTAS_LEQ_022 else True

    def __repr__(self):
        """Representation string of the object."""
        return (
            f"<{type(self).__name__}> '{self.name}': "
            f"{self.n_oseries} oseries, "
            f"{self.n_stresses} stresses, "
            f"{self.n_models} models"
        )

    @abstractmethod
    def _get_library(self, libname: str):
        """Get library handle.

        Must be overriden by subclass.

        Parameters
        ----------
        libname : str
            name of the library

        Returns
        -------
        lib : Any
            handle to the library
        """

    @abstractmethod
    def _add_item(
        self,
        libname: str,
        item: Union[FrameorSeriesUnion, Dict],
        name: str,
        metadata: Optional[Dict] = None,
        overwrite: bool = False,
    ) -> None:
        """Internal method to add item for both time series and pastas.Models.

        Must be overriden by subclass.

        Parameters
        ----------
        libname : str
            name of library to add item to
        item : FrameorSeriesUnion or dict
            item to add
        name : str
            name of the item
        metadata : dict, optional
            dictionary containing metadata, by default None
        """

    @abstractmethod
    def _get_item(self, libname: str, name: str) -> Union[FrameorSeriesUnion, Dict]:
        """Internal method to get item (series or pastas.Models).

        Must be overriden by subclass.

        Parameters
        ----------
        libname : str
            name of library
        name : str
            name of item

        Returns
        -------
        item : FrameorSeriesUnion or dict
            item (time series or pastas.Model)
        """

    @abstractmethod
    def _del_item(self, libname: str, name: str) -> None:
        """Internal method to delete items (series or models).

        Must be overriden by subclass.

        Parameters
        ----------
        libname : str
            name of library to delete item from
        name : str
            name of item to delete
        """

    @abstractmethod
    def _get_metadata(self, libname: str, name: str) -> Dict:
        """Internal method to get metadata.

        Must be overriden by subclass.

        Parameters
        ----------
        libname : str
            name of the library
        name : str
            name of the item

        Returns
        -------
        metadata : dict
            dictionary containing metadata
        """

    @abstractproperty
    def oseries_names(self):
        """List of oseries names.

        Property must be overriden by subclass.
        """

    @abstractproperty
    def stresses_names(self):
        """List of stresses names.

        Property must be overriden by subclass.
        """

    @abstractproperty
    def model_names(self):
        """List of model names.

        Property must be overriden by subclass.
        """

    def set_check_model_series_values(self, b: bool):
        """Turn CHECK_MODEL_SERIES_VALUES option on (True) or off (False).

        The default option is on (it is highly recommended to keep it that
        way). When turned on, the model time series
        (ml.oseries._series_original, and stressmodel.stress._series_original)
        values are checked against the stored copies in the database. If these
        do not match, an error is raised, and the model is not added to the
        database. This guarantees the stored model will be identical after
        loading from the database. This check is somewhat computationally
        expensive, which is why it can be turned on or off.

        Parameters
        ----------
        b : bool
            boolean indicating whether option should be turned on (True) or
            off (False). Option is on by default.
        """
        self.CHECK_MODEL_SERIES_VALUES = b
        print(f"Model time series checking set to: {b}.")

    def set_use_pastas_validate_series(self, b: bool):
        """Turn USE_PASTAS_VALIDATE_SERIES option on (True) or off (False).

        This will use pastas.validate_oseries() or pastas.validate_stresses()
        to test the time series. If they do not meet the criteria, an error is
        raised. Turning this option off will allow the user to store any time
        series but this will mean that time series models cannot be made from
        stored time series directly and will have to be modified before
        building the models. This in turn will mean that storing the models
        will not work as the stored time series copy is checked against the
        time series in the model to check if they are equal.

        Note: this option requires pastas>=0.23.0, otherwise it is turned off.

        Parameters
        ----------
        b : bool
            boolean indicating whether option should be turned on (True) or
            off (False). Option is on by default.
        """
        self.USE_PASTAS_VALIDATE_SERIES = b
        print(f"Model time series checking set to: {b}.")

    def _pastas_validate(self, validate):
        """Whether to validate time series.

        Parameters
        ----------
        validate : bool, NoneType
            value of validate keyword argument

        Returns
        -------
        b : bool
            return global or local setting (True or False)
        """
        if validate is None:
            return self.USE_PASTAS_VALIDATE_SERIES
        else:
            return validate

    def _add_series(
        self,
        libname: str,
        series: FrameorSeriesUnion,
        name: str,
        metadata: Optional[dict] = None,
        validate: Optional[bool] = None,
        overwrite: bool = False,
    ) -> None:
        """Internal method to add series to database.

        Parameters
        ----------
        libname : str
            name of the library to add the series to
        series : pandas.Series or pandas.DataFrame
            data to add
        name : str
            name of the time series
        metadata : dict, optional
            dictionary containing metadata, by default None
        validate: bool, optional
            use pastas to validate series, default is None, which will use the
            USE_PASTAS_VALIDATE_SERIES value (default is True).
        overwrite : bool, optional
            overwrite existing dataset with the same name,
            by default False

        Raises
        ------
        ItemInLibraryException
            if overwrite is False and name is already in the database
        """
        if not isinstance(name, str):
            name = str(name)
        self._validate_input_series(series)
        series = self._set_series_name(series, name)
        if self._pastas_validate(validate):
            if libname == "oseries":
                ps.validate_oseries(series)
            else:
                ps.validate_stress(series)
        in_store = getattr(self, f"{libname}_names")
        if name not in in_store or overwrite:
            self._add_item(
                libname, series, name, metadata=metadata, overwrite=overwrite
            )
            self._clear_cache(libname)
        else:
            raise ItemInLibraryException(
                f"Item with name '{name}' already" f" in '{libname}' library!"
            )

    def _update_series(
        self,
        libname: str,
        series: FrameorSeriesUnion,
        name: str,
        metadata: Optional[dict] = None,
        validate: Optional[bool] = None,
    ) -> None:
        """Internal method to update time series.

        Parameters
        ----------
        libname : str
            name of library
        series : FrameorSeriesUnion
            time series containing update values
        name : str
            name of the time series to update
        metadata : Optional[dict], optional
            optionally provide metadata dictionary which will also update
            the current stored metadata dictionary, by default None
        validate: bool, optional
            use pastas to validate series, default is None, which will use the
            USE_PASTAS_VALIDATE_SERIES value (default is True).
        """
        if libname not in ["oseries", "stresses"]:
            raise ValueError("Library must be 'oseries' or 'stresses'!")
        self._validate_input_series(series)
        series = self._set_series_name(series, name)
        stored = self._get_series(libname, name, progressbar=False)
        # get union of index
        idx_union = stored.index.union(series.index)
        # update series with new values
        update = stored.reindex(idx_union)
        update.update(series)
        # metadata
        update_meta = self._get_metadata(libname, name)
        if metadata is not None:
            update_meta.update(metadata)
        self._add_series(
            libname,
            update,
            name,
            metadata=update_meta,
            validate=validate,
            overwrite=True,
        )

    def _upsert_series(
        self,
        libname: str,
        series: FrameorSeriesUnion,
        name: str,
        metadata: Optional[dict] = None,
        validate: Optional[bool] = None,
    ) -> None:
        """Update or insert series depending on whether it exists in store.

        Parameters
        ----------
        libname : str
            name of library
        series : FrameorSeriesUnion
            time series to update/insert
        name : str
            name of the time series
        metadata : Optional[dict], optional
            metadata dictionary, by default None
        validate : bool, optional
            use pastas to validate series, default is None, which will use the
            USE_PASTAS_VALIDATE_SERIES value (default is True).
        """
        if libname not in ["oseries", "stresses"]:
            raise ValueError("Library must be 'oseries' or 'stresses'!")
        if name in getattr(self, f"{libname}_names"):
            self._update_series(
                libname, series, name, metadata=metadata, validate=validate
            )
        else:
            self._add_series(
                libname, series, name, metadata=metadata, validate=validate
            )

    def update_metadata(self, libname: str, name: str, metadata: dict) -> None:
        """Update metadata.

        Note: also retrieves and stores time series as updating only metadata
        is not supported for some Connectors.

        Parameters
        ----------
        libname : str
            name of library
        name : str
            name of the item for which to update metadata
        metadata : dict
            metadata dictionary that will be used to update the stored
            metadata
        """

        if libname not in ["oseries", "stresses"]:
            raise ValueError("Library must be 'oseries' or 'stresses'!")
        update_meta = self._get_metadata(libname, name)
        update_meta.update(metadata)
        # get series, since just updating metadata is not really defined
        # in all cases
        s = self._get_series(libname, name, progressbar=False)
        self._add_series(libname, s, name, metadata=update_meta, overwrite=True)

    def add_oseries(
        self,
        series: FrameorSeriesUnion,
        name: str,
        metadata: Optional[dict] = None,
        validate: Optional[bool] = None,
        overwrite: bool = False,
    ) -> None:
        """Add oseries to the database.

        Parameters
        ----------
        series : pandas.Series or pandas.DataFrame
            data to add
        name : str
            name of the time series
        metadata : dict, optional
            dictionary containing metadata, by default None.
        validate : bool, optional
            use pastas to validate series, default is None, which will use the
            USE_PASTAS_VALIDATE_SERIES value (default is True).
        overwrite : bool, optional
            overwrite existing dataset with the same name,
            by default False
        """
        series, metadata = self._parse_series_input(series, metadata)
        self._add_series(
            "oseries",
            series,
            name=name,
            metadata=metadata,
            validate=validate,
            overwrite=overwrite,
        )

    def add_stress(
        self,
        series: FrameorSeriesUnion,
        name: str,
        kind: str,
        metadata: Optional[dict] = None,
        validate: Optional[bool] = None,
        overwrite: bool = False,
    ) -> None:
        """Add stress to the database.

        Parameters
        ----------
        series : pandas.Series or pandas.DataFrame
            data to add, if pastas.Timeseries is passed, series_orignal
            and metadata is stored in database
        name : str
            name of the time series
        kind : str
            category to identify type of stress, this label is added to the
            metadata dictionary.
        metadata : dict, optional
            dictionary containing metadata, by default None.
        validate : bool, optional
            use pastas to validate series, default is True
        overwrite : bool, optional
            overwrite existing dataset with the same name,
            by default False
        """
        series, metadata = self._parse_series_input(series, metadata)
        if metadata is None:
            metadata = {}
        metadata["kind"] = kind
        self._add_series(
            "stresses",
            series,
            name=name,
            metadata=metadata,
            validate=validate,
            overwrite=overwrite,
        )

    def add_model(
        self,
        ml: Union[ps.Model, dict],
        overwrite: bool = False,
        validate_metadata: bool = False,
    ) -> None:
        """Add model to the database.

        Parameters
        ----------
        ml : pastas.Model or dict
            pastas Model or dictionary to add to the database
        overwrite : bool, optional
            if True, overwrite existing model, by default False
        validate_metadata, bool optional
            remove unsupported characters from metadata dictionary keys

        Raises
        ------
        TypeError
            if model is not pastas.Model or dict
        ItemInLibraryException
            if overwrite is False and model is already in the database
        """
        if isinstance(ml, ps.Model):
            mldict = ml.to_dict(series=False)
            name = ml.name
            if validate_metadata:
                metadata = validate_names(d=ml.oseries.metadata)
            else:
                metadata = ml.oseries.metadata
        elif isinstance(ml, dict):
            mldict = ml
            name = ml["name"]
            metadata = None
        else:
            raise TypeError("Expected pastas.Model or dict!")
        if not isinstance(name, str):
            name = str(name)
        if name not in self.model_names or overwrite:
            # check if stressmodels supported
            self._check_stressmodels_supported(ml)
            # check if oseries and stresses exist in store
            self._check_model_series_names_for_store(ml)
            self._check_oseries_in_store(ml)
            self._check_stresses_in_store(ml)
            # write model to store
            self._add_item(
                "models", mldict, name, metadata=metadata, overwrite=overwrite
            )
        else:
            raise ItemInLibraryException(
                f"Model with name '{name}' " "already in 'models' library!"
            )
        self._clear_cache("_modelnames_cache")
        self._add_oseries_model_links(str(mldict["oseries"]["name"]), name)

    @staticmethod
    def _parse_series_input(
        series: FrameorSeriesUnion,
        metadata: Optional[Dict] = None,
    ) -> Tuple[FrameorSeriesUnion, Optional[Dict]]:
        """Internal method to parse series input.

        Parameters
        ----------
        series : FrameorSeriesUnion,
            series object to parse
        metadata : dict, optional
            metadata dictionary or None, by default None

        Returns
        -------
        series, metadata : FrameorSeriesUnion, Optional[Dict]
            time series as pandas.Series or DataFrame and optionally
            metadata dictionary
        """
        if isinstance(series, ps.timeseries.TimeSeries):
            raise DeprecationWarning(
                "Pastas TimeSeries objects are no longer supported!"
            )
        s = series
        m = metadata
        return s, m

    def update_oseries(
        self,
        series: FrameorSeriesUnion,
        name: str,
        metadata: Optional[dict] = None,
    ) -> None:
        """Update oseries values.

        Parameters
        ----------
        series : FrameorSeriesUnion
            time series to update stored oseries with
        name : str
            name of the oseries to update
        metadata : Optional[dict], optional
            optionally provide metadata, which will update
            the stored metadata dictionary, by default None
        """
        series, metadata = self._parse_series_input(series, metadata)
        self._update_series("oseries", series, name, metadata=metadata)

    def upsert_oseries(
        self,
        series: FrameorSeriesUnion,
        name: str,
        metadata: Optional[dict] = None,
    ) -> None:
        """Update or insert oseries values depending on whether it exists.

        Parameters
        ----------
        series : FrameorSeriesUnion
            time series to update/insert
        name : str
            name of the oseries
        metadata : Optional[dict], optional
            optionally provide metadata, which will update
            the stored metadata dictionary if it exists, by default None
        """
        series, metadata = self._parse_series_input(series, metadata)
        self._upsert_series("oseries", series, name, metadata=metadata)

    def update_stress(
        self,
        series: FrameorSeriesUnion,
        name: str,
        metadata: Optional[dict] = None,
    ) -> None:
        """Update stresses values.

        Note: the 'kind' attribute of a stress cannot be updated! To update
        the 'kind' delete and add the stress again.

        Parameters
        ----------
        series : FrameorSeriesUnion
            time series to update stored stress with
        name : str
            name of the stress to update
        metadata : Optional[dict], optional
            optionally provide metadata, which will update
            the stored metadata dictionary, by default None
        """
        series, metadata = self._parse_series_input(series, metadata)
        self._update_series("stresses", series, name, metadata=metadata)

    def upsert_stress(
        self,
        series: FrameorSeriesUnion,
        name: str,
        kind: str,
        metadata: Optional[dict] = None,
    ) -> None:
        """Update or insert stress values depending on whether it exists.

        Parameters
        ----------
        series : FrameorSeriesUnion
            time series to update/insert
        name : str
            name of the stress
        metadata : Optional[dict], optional
            optionally provide metadata, which will update
            the stored metadata dictionary if it exists, by default None
        """
        series, metadata = self._parse_series_input(series, metadata)
        if metadata is None:
            metadata = {}
        metadata["kind"] = kind
        self._upsert_series("stresses", series, name, metadata=metadata)

    def del_models(self, names: Union[list, str]) -> None:
        """Delete model(s) from the database.

        Parameters
        ----------
        names : str or list of str
            name(s) of the model to delete
        """
        for n in self._parse_names(names, libname="models"):
            mldict = self.get_models(n, return_dict=True)
            oname = mldict["oseries"]["name"]
            self._del_item("models", n)
            self._del_oseries_model_link(oname, n)
        self._clear_cache("_modelnames_cache")

    def del_oseries(self, names: Union[list, str], remove_models: bool = False):
        """Delete oseries from the database.

        Parameters
        ----------
        names : str or list of str
            name(s) of the oseries to delete
        remove_models : bool, optional
            also delete models for deleted oseries, default is False
        """
        names = self._parse_names(names, libname="oseries")
        for n in names:
            self._del_item("oseries", n)
        self._clear_cache("oseries")
        # remove associated models from database
        if remove_models:
            modelnames = list(
                chain.from_iterable([self.oseries_models.get(n, []) for n in names])
            )
            self.del_models(modelnames)

    def del_stress(self, names: Union[list, str]):
        """Delete stress from the database.

        Parameters
        ----------
        names : str or list of str
            name(s) of the stress to delete
        """
        for n in self._parse_names(names, libname="stresses"):
            self._del_item("stresses", n)
        self._clear_cache("stresses")

    def _get_series(
        self,
        libname: str,
        names: Union[list, str],
        progressbar: bool = True,
        squeeze: bool = True,
    ) -> FrameorSeriesUnion:
        """Internal method to get time series.

        Parameters
        ----------
        libname : str
            name of the library
        names : str or list of str
            names of the time series to load
        progressbar : bool, optional
            show progressbar, by default True
        squeeze : bool, optional
            if True return DataFrame or Series instead of dictionary
            for single entry

        Returns
        -------
        pandas.DataFrame or dict of pandas.DataFrames
            either returns time series as pandas.DataFrame or
            dictionary containing the time series.
        """
        ts = {}
        names = self._parse_names(names, libname=libname)
        desc = f"Get {libname}"
        for n in tqdm(names, desc=desc) if progressbar else names:
            ts[n] = self._get_item(libname, n)
        # return frame if len == 1
        if len(ts) == 1 and squeeze:
            return ts[n]
        else:
            return ts

    def get_metadata(
        self,
        libname: str,
        names: Union[list, str],
        progressbar: bool = False,
        as_frame: bool = True,
        squeeze: bool = True,
    ) -> Union[dict, pd.DataFrame]:
        """Read metadata from database.

        Parameters
        ----------
        libname : str
            name of the library containing the dataset
        names : str or list of str
            names of the datasets for which to read the metadata
        squeeze : bool, optional
            if True return dict instead of list of dict
            for single entry

        Returns
        -------
        dict or pandas.DataFrame
            returns metadata dictionary or DataFrame of metadata
        """
        metalist = []
        names = self._parse_names(names, libname=libname)
        desc = f"Get metadata {libname}"
        for n in tqdm(names, desc=desc) if progressbar else names:
            imeta = self._get_metadata(libname, n)
            if imeta is None:
                imeta = {}
            if "name" not in imeta.keys():
                imeta["name"] = n
            metalist.append(imeta)
        if as_frame:
            meta = self._meta_list_to_frame(metalist, names=names)
            return meta
        else:
            if len(metalist) == 1 and squeeze:
                return metalist[0]
            else:
                return metalist

    def get_oseries(
        self,
        names: Union[list, str],
        return_metadata: bool = False,
        progressbar: bool = False,
        squeeze: bool = True,
    ) -> Union[Union[FrameorSeriesUnion, Dict], Optional[Union[Dict, List]]]:
        """Get oseries from database.

        Parameters
        ----------
        names : str or list of str
            names of the oseries to load
        return_metadata : bool, optional
            return metadata as dictionary or list of dictionaries,
            default is False
        progressbar : bool, optional
            show progressbar, by default False
        squeeze : bool, optional
            if True return DataFrame or Series instead of dictionary
            for single entry

        Returns
        -------
        oseries : pandas.DataFrame or dict of DataFrames
            returns time series as DataFrame or dictionary of DataFrames if
            multiple names were passed
        metadata : dict or list of dict
            metadata for each oseries, only returned if return_metadata=True
        """
        oseries = self._get_series(
            "oseries", names, progressbar=progressbar, squeeze=squeeze
        )
        if return_metadata:
            metadata = self.get_metadata(
                "oseries",
                names,
                progressbar=progressbar,
                as_frame=False,
                squeeze=squeeze,
            )
            return oseries, metadata
        else:
            return oseries

    def get_stresses(
        self,
        names: Union[list, str],
        return_metadata: bool = False,
        progressbar: bool = False,
        squeeze: bool = True,
    ) -> Union[Union[FrameorSeriesUnion, Dict], Optional[Union[Dict, List]]]:
        """Get stresses from database.

        Parameters
        ----------
        names : str or list of str
            names of the stresses to load
        return_metadata : bool, optional
            return metadata as dictionary or list of dictionaries,
            default is False
        progressbar : bool, optional
            show progressbar, by default False
        squeeze : bool, optional
            if True return DataFrame or Series instead of dictionary
            for single entry

        Returns
        -------
        stresses : pandas.DataFrame or dict of DataFrames
            returns time series as DataFrame or dictionary of DataFrames if
            multiple names were passed
        metadata : dict or list of dict
            metadata for each stress, only returned if return_metadata=True
        """
        stresses = self._get_series(
            "stresses", names, progressbar=progressbar, squeeze=squeeze
        )
        if return_metadata:
            metadata = self.get_metadata(
                "stresses",
                names,
                progressbar=progressbar,
                as_frame=False,
                squeeze=squeeze,
            )
            return stresses, metadata
        else:
            return stresses

    def get_models(
        self,
        names: Union[list, str],
        return_dict: bool = False,
        progressbar: bool = False,
        squeeze: bool = True,
        update_ts_settings: bool = False,
    ) -> Union[ps.Model, list]:
        """Load models from database.

        Parameters
        ----------
        names : str or list of str
            names of the models to load
        return_dict : bool, optional
            return model dictionary instead of pastas.Model (much
            faster for obtaining parameters, for example)
        progressbar : bool, optional
            show progressbar, by default False
        squeeze : bool, optional
            if True return Model instead of list of Models
            for single entry
        update_ts_settings : bool, optional
            update time series settings based on time series in store.
            overwrites stored tmin/tmax in model.

        Returns
        -------
        pastas.Model or list of pastas.Model
            return pastas model, or list of models if multiple names were
            passed
        """
        models = []
        names = self._parse_names(names, libname="models")
        desc = "Get models"
        for n in tqdm(names, desc=desc) if progressbar else names:
            data = self._get_item("models", n)
            if return_dict:
                ml = data
            else:
                ml = self._parse_model_dict(data, update_ts_settings=update_ts_settings)
            models.append(ml)
        if len(models) == 1 and squeeze:
            return models[0]
        else:
            return models

    def empty_library(
        self, libname: str, prompt: bool = True, progressbar: bool = True
    ):
        """Empty library of all its contents.

        Parameters
        ----------
        libname : str
            name of the library
        prompt : bool, optional
            prompt user for input before deleting
            contents, by default True. Default answer is
            "n", user must enter 'y' to delete contents
        progressbar : bool, optional
            show progressbar, by default True
        """
        if prompt:
            ui = input(
                f"Do you want to empty '{libname}'"
                " library of all its contents? [y/N] "
            )
            if ui.lower() != "y":
                return

        if libname == "models":
            # also delete linked modelnames linked to oseries
            libs = ["models", "oseries_models"]
        else:
            libs = [libname]

        # delete items and clear caches
        for libname in libs:
            names = self._parse_names(None, libname)
            for name in (
                tqdm(names, desc=f"Deleting items from {libname}")
                if progressbar
                else names
            ):
                self._del_item(libname, name)
            self._clear_cache(libname)
            print(f"Emptied library {libname} in {self.name}: " f"{self.__class__}")

    def _iter_series(self, libname: str, names: Optional[List[str]] = None):
        """Internal method iterate over time series in library.

        Parameters
        ----------
        libname : str
            name of library (e.g. 'oseries' or 'stresses')
        names : Optional[List[str]], optional
            list of names, by default None, which defaults to
            all stored series


        Yields
        -------
        pandas.Series or pandas.DataFrame
            time series contained in library
        """
        names = self._parse_names(names, libname)
        for nam in names:
            yield self._get_series(libname, nam, progressbar=False)

    def iter_oseries(self, names: Optional[List[str]] = None):
        """Iterate over oseries in library.

        Parameters
        ----------
        names : Optional[List[str]], optional
            list of oseries names, by default None, which defaults to
            all stored series


        Yields
        -------
        pandas.Series or pandas.DataFrame
            oseries contained in library
        """
        yield from self._iter_series("oseries", names=names)

    def iter_stresses(self, names: Optional[List[str]] = None):
        """Iterate over stresses in library.

        Parameters
        ----------
        names : Optional[List[str]], optional
            list of stresses names, by default None, which defaults to
            all stored series


        Yields
        -------
        pandas.Series or pandas.DataFrame
            stresses contained in library
        """
        yield from self._iter_series("stresses", names=names)

    def iter_models(
        self, modelnames: Optional[List[str]] = None, return_dict: bool = False
    ):
        """Iterate over models in library.

        Parameters
        ----------
        modelnames : Optional[List[str]], optional
            list of models to iterate over, by default None which uses
            all models
        return_dict : bool, optional
            if True, return model as dictionary, by default False,
            which returns a pastas.Model.

        Yields
        -------
        pastas.Model or dict
            time series model
        """

        modelnames = self._parse_names(modelnames, "models")
        for mlnam in modelnames:
            yield self.get_models(mlnam, return_dict=return_dict, progressbar=False)

    def _add_oseries_model_links(self, onam: str, mlnames: Union[str, List[str]]):
        """Add model name to stored list of models per oseries.

        Parameters
        ----------
        onam : str
            name of oseries
        mlnames : Union[str, List[str]]
            model name or list of model names for an oseries with name
            onam.
        """
        # get stored list of model names
        if str(onam) in self.oseries_with_models:
            modellist = self._get_item("oseries_models", onam)
        else:
            # else empty list
            modellist = []
        # if one model name, make list for loop
        if isinstance(mlnames, str):
            mlnames = [mlnames]
        # loop over model names
        for iml in mlnames:
            # if not present, add to list
            if iml not in modellist:
                modellist.append(iml)
        self._add_item("oseries_models", modellist, onam, overwrite=True)
        self._clear_cache("oseries_models")

    def _del_oseries_model_link(self, onam, mlnam):
        """Delete model name from stored list of models per oseries.

        Parameters
        ----------
        onam : str
            name of oseries
        mlnam : str
            name of model
        """
        modellist = self._get_item("oseries_models", onam)
        modellist.remove(mlnam)
        if len(modellist) == 0:
            self._del_item("oseries_models", onam)
        else:
            self._add_item("oseries_models", modellist, onam, overwrite=True)
        self._clear_cache("oseries_models")

    def _update_all_oseries_model_links(self):
        """Add all model names to oseries metadata dictionaries.

        Used for old PastaStore versions, where relationship between
        oseries and models was not stored. If there are any models in
        the database and if the oseries_models library is empty, loops
        through all models to determine which oseries each model belongs
        to.
        """
        # get oseries_models library if there are any contents, if empty
        # add all model links.
        if self.n_models > 0:
            if len(self.oseries_models) == 0:
                links = self._get_all_oseries_model_links()
                for onam, mllinks in tqdm(
                    links.items(),
                    desc="Store models per oseries",
                    total=len(links),
                ):
                    self._add_oseries_model_links(onam, mllinks)

    def _get_all_oseries_model_links(self):
        """Get all model names per oseries in dictionary.

        Returns
        -------
        links : dict
            dictionary with oseries names as keys and lists of model names as
            values
        """
        links = {}
        for mldict in tqdm(
            self.iter_models(return_dict=True),
            total=self.n_models,
            desc="Get models per oseries",
        ):
            onam = mldict["oseries"]["name"]
            mlnam = mldict["name"]
            if onam in links:
                links[onam].append(mlnam)
            else:
                links[onam] = [mlnam]
        return links

    @staticmethod
    def _clear_cache(libname: str) -> None:
        """Clear cached property."""
        if libname == "models":
            libname = "_modelnames_cache"
        getattr(BaseConnector, libname).fget.cache_clear()

    @property  # type: ignore
    @functools.lru_cache()
    def oseries(self):
        """Dataframe with overview of oseries."""
        return self.get_metadata("oseries", self.oseries_names)

    @property  # type: ignore
    @functools.lru_cache()
    def stresses(self):
        """Dataframe with overview of stresses."""
        return self.get_metadata("stresses", self.stresses_names)

    @property  # type: ignore
    @functools.lru_cache()
    def _modelnames_cache(self):
        """List of model names."""
        return self.model_names

    @property
    def n_oseries(self):
        return len(self.oseries_names)

    @property
    def n_stresses(self):
        return len(self.stresses_names)

    @property
    def n_models(self):
        return len(self.model_names)

    @property  # type: ignore
    @functools.lru_cache()
    def oseries_models(self):
        """List of model names per oseries.

        Returns
        -------
        d : dict
            dictionary with oseries names as keys and list of model names as
            values
        """
        d = {}
        for onam in self.oseries_with_models:
            d[onam] = self._get_item("oseries_models", onam)
        return d


class ConnectorUtil:
    """Mix-in class for general Connector helper functions.

    Only for internal methods, and not methods that are related to CRUD
    operations on database.
    """

    def _parse_names(
        self,
        names: Optional[Union[list, str]] = None,
        libname: Optional[str] = "oseries",
    ) -> list:
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
                return getattr(self, "oseries_names")
            elif libname == "stresses":
                return getattr(self, "stresses_names")
            elif libname == "models":
                return getattr(self, "model_names")
            elif libname == "oseries_models":
                return getattr(self, "oseries_with_models")
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
        return meta

    def _parse_model_dict(self, mdict: dict, update_ts_settings: bool = False):
        """Internal method to parse dictionary describing pastas models.

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
            mdict["oseries"]["series"] = self.get_oseries(name)
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
                                stress["series"] = self.get_stresses(name)
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
                                stress["series"] = self.get_stresses(name)
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
                            stress["series"] = self.get_stresses(name)
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
            # data from Arctic VersionStores
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
        """Internal method, check if Model oseries are contained in PastaStore.

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
            if not so.dropna().equals(s_org):
                raise ValueError(
                    f"Cannot add model because model oseries '{name}'"
                    " is different from stored oseries!"
                )

    def _check_stresses_in_store(self, ml: Union[ps.Model, dict]):
        """Internal method, check if stresses time series are contained in
        PastaStore.

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
                        if not so.equals(s_org):
                            raise ValueError(
                                f"Cannot add model because model stress "
                                f"'{s.name}' is different from stored stress!"
                            )
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
        """Internal method for writing DataFrame or Series to zipfile.

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
        for n in tqdm(names, desc="models") if progressbar else names:
            m = self.get_models(n, return_dict=True)
            jsondict = json.dumps(m, cls=PastasEncoder, indent=4)
            archive.writestr(f"models/{n}.pas", jsondict)

    @staticmethod
    def _series_from_json(fjson: str):
        """Load time series from JSON.

        Parameters
        ----------
        fjson : str
            path to file

        Returns
        -------
        s : pd.DataFrame
            DataFrame containing time series
        """
        s = pd.read_json(fjson, orient="columns", precise_float=True)
        if not isinstance(s.index, pd.DatetimeIndex):
            s.index = pd.to_datetime(s.index, unit="ms")
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


class ModelAccessor:
    """Object for managing access to stored models.

    Provides dict-like access to models (i.e. PastaStore.models["model1"]),
    or allows adding models to the PastaStore using dict-like assignment
    (i.e. PastaStore.models["model1"] = ml), and it can serve as an iterator
    (i.e. [ml for ml in pstore.models]).
    """

    def __init__(self, conn):
        """Initialize model accessor.

        Parameters
        ----------
        conn : pastastore.*Connector type
            connector
        """
        self.conn = conn

    def __repr__(self):
        """Representation of the object is a list of modelnames."""
        return self.conn._modelnames_cache.__repr__()

    def __getitem__(self, name: str):
        """Get model from store with model name as key.

        Parameters
        ----------
        name : str
            name of the model
        """
        return self.conn.get_models(name)

    def __setitem__(self, name: str, ml: ps.Model):
        """Set item.

        Parameters
        ----------
        name : str
            name of the model
        ml : pastas.Model or dict
            model to add to the pastastore
        """
        ml.name = name
        self.conn.add_model(ml, overwrite=True)

    def __iter__(self):
        """Iterate over models.

        Yields
        -------
        ml : pastas.Model
            model
        """
        yield from self.conn.iter_models()

    def __len__(self):
        """No.

        of models
        """
        return self.conn.n_models

    def random(self):
        from random import choice

        return self.conn.get_models(choice(self.conn._modelnames_cache))
