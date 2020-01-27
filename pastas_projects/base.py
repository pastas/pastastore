from abc import ABC, abstractmethod, abstractproperty
from typing import Union

import pandas as pd

from pastas import Model

FrameorSeriesUnion = Union[pd.DataFrame, pd.Series]


class BaseConnector(ABC):  # pragma: no cover
    """Metaclass for connecting to data management sources,
    i.e. MongoDB through Arctic, Pystore, or other databases.

    Create your own connection to a data source by writing a
    a class that inherits from this BaseConnector. Your class
    has to override each method and property.

    """
    _default_library_names = ["oseries", "stresses", "models"]

    @abstractmethod
    def get_library(self, libname: str):
        pass

    @abstractmethod
    def _add_series(self, libname: str, series: FrameorSeriesUnion,
                    name: str, metadata: Union[dict, None] = None,
                    add_version: bool = False) -> None:
        pass

    @abstractmethod
    def add_oseries(self, series: FrameorSeriesUnion, name: str,
                    metadata: Union[dict, None] = None,
                    add_version: bool = False) -> None:
        pass

    @abstractmethod
    def add_stress(self, series: FrameorSeriesUnion, name: str, kind: str,
                   metadata: Union[dict, None] = None,
                   add_version: bool = False) -> None:
        pass

    @abstractmethod
    def add_model(self, ml: Model, add_version: bool = False) -> None:
        pass

    @abstractmethod
    def del_models(self, names: Union[list, str]) -> None:
        pass

    @abstractmethod
    def del_oseries(self, names: Union[list, str]) -> None:
        pass

    @abstractmethod
    def del_stress(self, names: Union[list, str]) -> None:
        pass

    @abstractmethod
    def _get_series(self, libname: str, names: Union[list, str],
                    progressbar: bool = True) -> FrameorSeriesUnion:
        pass

    @abstractmethod
    def get_metadata(self, libname: str, names: Union[list, str],
                     progressbar: bool = False, as_frame: bool = True) -> \
            Union[pd.DataFrame, dict]:
        pass

    @abstractmethod
    def get_oseries(self, names: Union[list, str],
                    progressbar: bool = False) -> FrameorSeriesUnion:
        pass

    @abstractmethod
    def get_stresses(self, names: Union[list, str],
                     progressbar: bool = False) -> FrameorSeriesUnion:
        pass

    @abstractmethod
    def get_models(self, names: Union[list, str],
                   progressbar: bool = False) -> Union[Model, dict]:
        pass

    @abstractproperty
    def oseries(self):
        pass

    @abstractproperty
    def stresses(self):
        pass

    @abstractproperty
    def models(self):
        pass
