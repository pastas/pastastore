from abc import ABC, abstractmethod, abstractproperty


class BaseConnector(ABC):  # pragma: no cover
    """Metaclass for connecting to data management sources,
    i.e. MongoDB through Arctic, Pystore, or other databases.

    Create your own connection to a data source by writing a
    a class that inherits from this BaseConnector. Your class
    has to override each method and property.

    """
    _default_library_names = ["oseries", "stresses", "models"]

    @abstractmethod
    def get_library(self, libname):
        pass

    @abstractmethod
    def _add_series(self, libname, series, name, metadata=None,
                    add_version=False):
        pass

    @abstractmethod
    def add_oseries(self, series, name, metadata=None, add_version=False):
        pass

    @abstractmethod
    def add_stress(self, series, name, kind, metadata=None, add_version=False):
        pass

    @abstractmethod
    def add_model(self, ml, add_version=False):
        pass

    @abstractmethod
    def del_models(self, names):
        pass

    @abstractmethod
    def del_oseries(self, names):
        pass

    @abstractmethod
    def del_stress(self, names):
        pass

    @abstractmethod
    def _get_series(self, libname, names, progressbar=True):
        pass

    @abstractmethod
    def get_metadata(self, libname, names, progressbar=False, as_frame=True):
        pass

    @abstractmethod
    def get_oseries(self, names, progressbar=False):
        pass

    @abstractmethod
    def get_stresses(self, names, progressbar=False):
        pass

    @abstractmethod
    def get_models(self, names, progressbar=False):
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
