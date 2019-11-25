import functools
import sys
from collections.abc import Iterable

import arctic
import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import Point
from tqdm import tqdm

import pastas as ps

from .base import BaseProject
from .util import query_yes_no


class ArcticPastas(BaseProject):
    def __init__(self, connstr, projectname):

        self.connstr = connstr
        self.projectname = projectname

        self.arc = arctic.Arctic(connstr)
        self._initialize()

    def __repr__(self):
        noseries = len(self.lib_oseries.list_symbols())
        nstresses = len(self.lib_stresses.list_symbols())
        nmodels = len(self.lib_models.list_symbols())
        return "<ArcticPastas object> '{0}': {1} oseries, {2} stresses, {3} models".format(
            self.projectname, noseries, nstresses, nmodels
        )

    def _initialize(self):
        for libname in ["oseries", "stresses", "models"]:
            if self.projectname + "." + libname not in self.arc.list_libraries():
                self.arc.initialize_library(self.projectname + "." + libname)
            setattr(self, "lib_" + libname.replace(".", ""),
                    self.get_library(libname))

    def get_library(self, libname):
        # get library
        lib = self.arc.get_library(self.projectname + "." + libname)
        return lib

    def _add_series(self, libname, series, name, metadata=None,
                    add_version=False):
        lib = self.get_library(libname)
        if name not in lib.list_symbols() or add_version:
            lib.write(name, series, metadata=metadata)
        else:
            raise Exception("Item with name '{0}' already"
                            " in '{1}' library!".format(name, libname))

    def add_oseries(self, series, name, metadata=None, add_version=False):
        self._add_series("oseries", series, name=name,
                         metadata=metadata, add_version=add_version)
        ArcticPastas.oseries.fget.cache_clear()

    def add_stress(self, series, name, kind, metadata=None, add_version=False):
        if metadata is None:
            metadata = {}

        metadata["kind"] = kind
        self._add_series("stresses", series, name=name,
                         metadata=metadata, add_version=add_version)
        ArcticPastas.stresses.fget.cache_clear()

    def add_model(self, ml, add_version=False):
        lib = self.lib_models
        if ml.name not in lib.list_symbols() or add_version:
            mldict = ml.to_dict(series=False)
            lib.write(ml.name, mldict, metadata=ml.oseries.metadata)
        else:
            raise Exception("Model with name '{}' already in store!".format(
                ml.name))
        ArcticPastas.models.fget.cache_clear()

    def del_model(self, name):
        self.lib_models.delete(name)

    def del_oseries(self, name):
        self.lib_oseries.delete(name)
        ArcticPastas.oseries.fget.cache_clear()

    def del_stress(self, name):
        self.lib_stresses.delete(name)
        ArcticPastas.stresses.fget.cache_clear()

    def delete_library(self, lib=None):
        if libs is None:
            libs = ["oseries", "stresses", "models"]
        elif isinstance(libs, str):
            libs = [libs]
        query_yes_no("Delete library(s): '{}'?".format(", ".join(libs)))
        for lib in libs:
            self.arc.delete_library(self.projectname + "." + lib)
            print("... deleted library '{}'!".format(lib))

    def _get_timeseries(self, libname, names, progressbar=True):

        lib = self.get_library(libname)

        if isinstance(names, str):
            ts = lib.read(names).data
        elif isinstance(names, Iterable):
            ts = {}
            for n in (tqdm(names) if progressbar else names):
                ts[n] = lib.read(n).data
        return ts

    def get_metadata(self, libname, names):

        lib = self.get_library(libname)

        # read only metadata
        if isinstance(names, str):
            meta = lib.read_metadata(names).metadata
        elif isinstance(names, Iterable):
            metalist = []
            for n in names:
                metalist.append(lib.read_metadata(n).metadata)
            meta = pd.DataFrame(metalist)
            if len({"x", "y"}.difference(meta.columns)) == 0:
                meta["x"] = meta["x"].astype(float)
                meta["y"] = meta["y"].astype(float)
                meta = gpd.GeoDataFrame(meta, geometry=[Point(
                    (s["x"], s["y"])) for i, s in meta.iterrows()])
            if "name" in meta.columns:
                meta.set_index("name", inplace=True)
        return meta

    def get_oseries(self, names, progressbar=False):
        return self._get_timeseries("oseries", names, progressbar=progressbar)

    def get_stresses(self, names, progressbar=False):
        return self._get_timeseries("stresses", names, progressbar=progressbar)

    def get_models(self, names, progressbar=False):

        if isinstance(names, str):
            names = [names]

        models = []

        for n in (tqdm(names) if progressbar else names):
            item = self.lib_models.read(n)
            data = item.data

            if 'series' not in data['oseries']:
                name = data["oseries"]['name']
                if name not in self.oseries.index:
                    msg = 'oseries {} not present in project'.format(name)
                    raise(LookupError(msg))
                s = self.get_oseries(name)
                data['oseries']['series'] = s.value
            for ts in data["stressmodels"].values():
                if "stress" in ts.keys():
                    for stress in ts["stress"]:
                        if 'series' not in stress:
                            name = stress['name']
                            symbol = str(self.stresses.loc[name, "station"])
                            if name in self.stresses.index:
                                s = self.get_stresses(symbol)
                            else:
                                msg = 'stress {} not present in project'.format(
                                    name)
                            stress['series'] = s

            ml = ps.io.base.load_model(data)
            models.append(ml)
        if len(models) == 1:
            return models[0]
        else:
            return models

    def create_model(self, name, add_recharge=True):
        # get oseries metadata
        meta = self.get_metadata("oseries", name)
        ts = self.get_oseries(name)

        # convert to Timeseries and create model
        if not ts.value.dropna().empty:
            ts = ps.TimeSeries(ts.value, name=name, settings="oseries",
                               metadata=meta)
            ml = ps.Model(ts, name=name, metadata=meta)

            if add_recharge:
                self.add_recharge(ml)
            return ml
        else:
            raise ValueError("Empty timeseries!")

    def create_models(self, oseries=None, add_recharge=True, store=False,
                      solve=False, progressbar=True, return_models=False,
                      ignore_errors=True, **kwargs):
        if oseries is None:
            oseries = self.oseries.index
        elif isinstance(oseries, str):
            oseries = [oseries]

        models = {}
        errors = []
        for o in (tqdm(oseries)
                  if progressbar else oseries):
            try:
                iml = self.create_model(o, add_recharge=add_recharge)
            except Exception as e:
                if ignore_errors:
                    errors.append(o)
                    continue
                else:
                    raise e
            if solve:
                iml.solve(**kwargs)
            if store:
                self.add_model(iml, add_version=True)
            if return_models:
                models[o] = iml
        if return_models:
            return models, errors
        else:
            return errors

    def add_recharge(self, ml, rfunc=ps.Gamma):
        # get nearest prec and evap stns
        names = []
        for var in ("prec", "evap"):
            name = self.get_nearest_stresses(
                ml.oseries.name, kind=var).iloc[0, 0]
            names.append(str(self.stresses.loc[name, "station"]))

        # get data
        tsdict = self.get_stresses(names)
        stresses = []
        for k, s in tsdict.items():
            metadata = self.get_metadata("stresses", k)
            stresses.append(ps.TimeSeries(s, name=metadata["name"],
                                          metadata=metadata))

        # add recharge to model
        rch = ps.StressModel2(stresses, rfunc, name="recharge",
                              metadata=[i.metadata for i in stresses],
                              settings=("prec", "evap"))
        ml.add_stressmodel(rch)

    def solve_models(self, mls=None, report=False, ignore_solve_errors=False,
                     progressbar=True, store_result=True, **kwargs):
        """Solves the models in the library

        mls: list of str, optional
            list of model names, if None all models in the project are solved.
        report: boolean, optional
            determines if a report is printed when the model is solved.
        ignore_solve_errors: boolean, optional
            if True errors emerging from the solve method are ignored.
        **kwargs:
            arguments are passed to the solve method.

        """
        if mls is None:
            mls = self.models
        elif isinstance(mls, ps.Model):
            mls = [mls.name]

        for ml_name in (tqdm(mls) if progressbar else mls):
            ml = self.get_models(ml_name)

            m_kwargs = {}
            for key, value in kwargs.items():
                if isinstance(value, pd.Series):
                    m_kwargs[key] = value.loc[ml_name]
                else:
                    m_kwargs[key] = value
            # Convert timestamps
            for tstamp in ["tmin", "tmax"]:
                if tstamp in m_kwargs:
                    m_kwargs[tstamp] = pd.Timestamp(m_kwargs[tstamp])

            try:
                ml.solve(report=report, **m_kwargs)
                if store_result:
                    self.add_model(ml, add_version=True)
            except Exception as e:
                if ignore_solve_errors:
                    warning = "solve error ignored for -> {}".format(ml.name)
                    ps.logger.warning(warning)
                else:
                    raise e

    @property
    @functools.lru_cache()
    def oseries(self):
        lib = self.lib_oseries
        return gpd.GeoDataFrame(self.get_metadata("oseries",
                                                  lib.list_symbols()))

    @property
    @functools.lru_cache()
    def stresses(self):
        lib = self.lib_stresses
        return self.get_metadata("stresses",
                                 lib.list_symbols())

    @property
    @functools.lru_cache()
    def models(self):
        lib = self.lib_models
        return lib.list_symbols()
