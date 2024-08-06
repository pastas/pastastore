"""HydroPandas extension for PastaStore.

Features:

- Add `hpd.Obs` and `hpd.ObsCollection` to PastaStore.
- Download and store meteorological data from KNMI or groundwater observations from BRO.
- Update currently stored (KNMI or BRO) time series from last observation to tmax.
"""

import logging
from typing import List, Optional, Union

import hydropandas as hpd
import numpy as np
from pandas import DataFrame, Series, Timedelta, Timestamp
from tqdm.auto import tqdm

from pastastore.extensions.accessor import register_pastastore_accessor

logger = logging.getLogger("hydropandas")


TimeType = Optional[Union[str, Timestamp]]


@register_pastastore_accessor("hpd")
class HydroPandasExtension:
    """HydroPandas extension for PastaStore.

    Parameters
    ----------
    store: pastastore.store.PastaStore
        PastaStore object to extend with HydroPandas functionality
    """

    def __init__(self, store):
        """Initialize HydroPandasExtenstion.

        Parameters
        ----------
        store : pasta.store.PastaStore
            PastaStore object to extend with HydroPandas functionality
        """
        self._store = store

    def add_obscollection(
        self,
        libname: str,
        oc: hpd.ObsCollection,
        kind: Optional[str] = None,
        data_column: Optional[str] = None,
        unit_multiplier: float = 1.0,
        update: bool = False,
    ):
        """Add an ObsCollection to the PastaStore.

        Parameters
        ----------
        libname : str
            Name of the library to add the ObsCollection to ["oseries", "stresses"].
        oc : hpd.ObsCollection
            ObsCollection to add to the store.
        kind : str, optional
            kind identifier for observations, by default None. Required for adding
            stresses.
        data_column : str, optional
            name of column containing observation values, by default None.
        unit_multiplier : float, optional
            multiply unit by this value before saving it in the store
        update : bool, optional
            if True, update currently stored time series with new data
        """
        for name, row in oc.iterrows():
            obs = row["obs"]
            # metadata = row.drop("obs").to_dict()
            self.add_observation(
                libname,
                obs,
                name=name,
                kind=kind,
                data_column=data_column,
                unit_multiplier=unit_multiplier,
                update=update,
            )

    def add_observation(
        self,
        libname: str,
        obs: hpd.Obs,
        name: Optional[str] = None,
        kind: Optional[str] = None,
        data_column: Optional[str] = None,
        unit_multiplier: float = 1.0,
        update: bool = False,
    ):
        """Add an hydropandas observation series to the PastaStore.

        Parameters
        ----------
        libname : str
            Name of the library to add the observation to ["oseries", "stresses"].
        obs : hpd.Obs
            hydroPandas observation series to add to the store.
        name : str, optional
            Name of the observation, by default None. If None, the name of the
            observation is used.
        kind : str, optional
            kind identifier for observations, by default None. Required for adding
            stresses.
        data_column : str, optional
            name of column containing observation values, by default None.
        unit_multiplier : float, optional
            multiply unit by this value before saving it in the store
        update : bool, optional
            if True, update currently stored time series with new data
        """
        # if data_column is not None, use data_column
        if data_column is not None:
            if not obs.empty:
                o = obs[data_column]
            else:
                o = Series()
        # if data_column is None, check no. of columns in obs
        # if only one column, use that column
        elif isinstance(obs, DataFrame) and obs.columns.size == 1:
            o = obs.iloc[:, 0]
        elif isinstance(obs, Series):
            o = obs
        # else raise error
        else:
            raise ValueError("No data_column specified and obs has multiple columns.")

        # break if obs is empty
        if o.empty:
            logger.info("Observation '%s' is empty, not adding to store.", name)
            return

        # gather metadata from obs object
        metadata = {key: getattr(obs, key) for key in obs._metadata}

        # convert np dtypes to builtins
        for k, v in metadata.items():
            if isinstance(v, np.integer):
                metadata[k] = int(v)
            elif isinstance(v, np.floating):
                metadata[k] = float(v)

        metadata.pop("name", None)
        metadata.pop("meta", None)
        unit = metadata.get("unit", None)
        if unit == "m" and unit_multiplier == 1e3:
            metadata["unit"] = "mm"
        elif unit_multiplier != 1.0:
            metadata["unit"] = f"{unit_multiplier:e}*{unit}"

        source = metadata.get("source", "")
        if len(source) > 0:
            source = f"{source} "

        if update:
            action_msg = "updated in"
        else:
            action_msg = "added to"

        if libname == "oseries":
            self._store.upsert_oseries(o, name, metadata=metadata)
            logger.info(
                "%sobservation '%s' %s oseries library.", source, name, action_msg
            )
        elif libname == "stresses":
            if kind is None:
                raise ValueError("`kind` must be specified for stresses!")
            self._store.upsert_stress(
                o * unit_multiplier, name, kind, metadata=metadata
            )
            logger.info(
                "%sstress '%s' (kind='%s') %s stresses library.",
                source,
                name,
                kind,
                action_msg,
            )
        else:
            raise ValueError("libname must be 'oseries' or 'stresses'.")

    def download_knmi_precipitation(
        self,
        stns: Optional[list[int]] = None,
        meteo_var: str = "RH",
        tmin: TimeType = None,
        tmax: TimeType = None,
        unit_multiplier: float = 1e3,
        **kwargs,
    ):
        """Download precipitation data from KNMI and store in PastaStore.

        Parameters
        ----------
        stns : list of int/str, optional
            list of station numbers to download data for, by default None
        meteo_var : str, optional
            variable to download, by default "RH", valid options are ["RD", "RH"].
        tmin : TimeType, optional
            start time, by default None
        tmax : TimeType, optional
            end time, by default None
        unit_multiplier : float, optional
            multiply unit by this value before saving it in the store,
            by default 1e3 to convert m to mm
        """
        self.download_knmi_meteo(
            meteo_var=meteo_var,
            kind="prec",
            stns=stns,
            tmin=tmin,
            tmax=tmax,
            unit_multiplier=unit_multiplier,
            **kwargs,
        )

    def download_knmi_evaporation(
        self,
        stns: Optional[list[int]] = None,
        meteo_var: str = "EV24",
        tmin: TimeType = None,
        tmax: TimeType = None,
        unit_multiplier: float = 1e3,
        **kwargs,
    ):
        """Download evaporation data from KNMI and store in PastaStore.

        Parameters
        ----------
        stns : list of int/str, optional
            list of station numbers to download data for, by default None
        meteo_var : str, optional
            variable to download, by default "EV24"
        tmin : TimeType, optional
            start time, by default None
        tmax : TimeType, optional
            end time, by default None
        unit_multiplier : float, optional
            multiply unit by this value before saving it in the store,
            by default 1e3 to convert m to mm
        """
        self.download_knmi_meteo(
            meteo_var=meteo_var,
            kind="evap",
            stns=stns,
            tmin=tmin,
            tmax=tmax,
            unit_multiplier=unit_multiplier,
            **kwargs,
        )

    def download_knmi_meteo(
        self,
        meteo_var: str,
        kind: str,
        stns: Optional[list[int]] = None,
        tmin: TimeType = None,
        tmax: TimeType = None,
        unit_multiplier: float = 1.0,
        **kwargs,
    ):
        """Download meteorological data from KNMI and store in PastaStore.

        Parameters
        ----------
        meteo_var : str, optional
            variable to download, by default "RH", valid options are
            e.g. ["RD", "RH", "EV24", "T", "Q"].
        kind : str
            kind identifier for observations, usually "prec" or "evap".
        stns : list of int/str, optional
            list of station numbers to download data for, by default None
        tmin : TimeType, optional
            start time, by default None
        tmax : TimeType, optional
            end time, by default None
        unit_multiplier : float, optional
            multiply unit by this value before saving it in the store,
            by default 1.0 (no conversion)
        """
        # get tmin/tmax if not specified
        tmintmax = self._store.get_tmin_tmax("oseries")
        if tmin is None:
            tmin = tmintmax.loc[:, "tmin"].min() - Timedelta(days=10 * 365)
        if tmax is None:
            tmax = tmintmax.loc[:, "tmax"].max()

        if stns is None:
            locations = self._store.oseries.loc[:, ["x", "y"]]
        else:
            locations = None

        # download data
        knmi = hpd.read_knmi(
            locations=locations,
            stns=stns,
            meteo_vars=[meteo_var],
            starts=tmin,
            ends=tmax,
            **kwargs,
        )

        # add to store
        self.add_obscollection(
            libname="stresses",
            oc=knmi,
            kind=kind,
            data_column=meteo_var,
            unit_multiplier=unit_multiplier,
            update=False,
        )

    def update_knmi_meteo(
        self,
        names: Optional[List[str]] = None,
        tmin: TimeType = None,
        tmax: TimeType = None,
    ):
        """Update meteorological data from KNMI in PastaStore.

        Parameters
        ----------
        names : list of str, optional
            list of names of observations to update, by default None
        tmin : TimeType, optional
            start time, by default None, which uses current last observation timestamp
            as tmin
        tmax : TimeType, optional
            end time, by default None, which defaults to today
        """
        if names is None:
            names = self._store.stresses.loc[
                self._store.stresses["source"] == "KNMI"
            ].index.tolist()

        tmintmax = self._store.get_tmin_tmax("stresses", names=names)

        for name in tqdm(names, desc="Updating KNMI meteo stresses"):
            stn = self._store.stresses.loc[name, "station"]
            meteo_var = self._store.stresses.loc[name, "meteo_var"]
            unit = self._store.stresses.loc[name, "unit"]
            kind = self._store.stresses.loc[name, "kind"]

            if unit == "mm":
                unit_multiplier = 1e3
            else:
                unit_multiplier = 1.0

            if tmin is None:
                tmin = tmintmax.loc[name, "tmax"]

            knmi = hpd.read_knmi(
                stns=[stn],
                meteo_vars=[meteo_var],
                starts=tmin,
                ends=tmax,
            )

            self.add_observation(
                "stresses",
                knmi["obs"].iloc[0],
                name=name,
                kind=kind,
                data_column=meteo_var,
                unit_multiplier=unit_multiplier,
                update=True,
            )

    def download_bro_gmw(
        self,
        extent: Optional[List[float | int]] = None,
        tmin: TimeType = None,
        tmax: TimeType = None,
        update: bool = False,
        **kwargs,
    ):
        """Download groundwater monitoring well observations from BRO.

        Parameters
        ----------
        extent: tuple, optional
            Extent of the area to download observations from.
        tmin: pandas.Timestamp, optional
            Start date of the observations to download.
        tmax: pandas.Timestamp, optional
            End date of the observations to download.
        **kwargs: dict, optional
            Additional keyword arguments to pass to `hpd.read_bro()`
        """
        bro = hpd.read_bro(
            extent=extent,
            tmin=tmin,
            tmax=tmax,
            **kwargs,
        )
        self.add_obscollection("oseries", bro, data_column="values", update=update)

    def update_bro_gmw(
        self,
        names: Optional[List[str]] = None,
        tmin: TimeType = None,
        tmax: TimeType = None,
        **kwargs,
    ):
        """Update groundwater monitoring well observations from BRO.

        Parameters
        ----------
        names : list of str, optional
            list of names of observations to update, by default None which updates all
            stored oseries.
        tmin : TimeType, optional
            start time, by default None, which uses current last observation timestamp
            as tmin
        tmax : TimeType, optional
            end time, by default None, which defaults to today
        **kwargs : dict, optional
            Additional keyword arguments to pass to `hpd.GroundwaterObs.from_bro()`
        """
        if names is None:
            names = self._store.oseries.index.to_list()

        tmintmax = self._store.get_tmin_tmax("oseries")

        for obsnam in tqdm(names, desc="Updating BRO oseries"):
            bro_id, tube_number = obsnam.split("_")

            if tmin is None:
                _, tmin = tmintmax.loc[obsnam]  # tmin is stored tmax

            obs = hpd.GroundwaterObs.from_bro(
                bro_id, int(tube_number), tmin=tmin, tmax=tmax, **kwargs
            )
            self.add_observation(
                "oseries", obs, name=obsnam, data_column="values", update=True
            )
