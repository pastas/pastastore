import logging
from typing import Optional, Union

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
        """
        for name, row in oc.iterrows():
            obs = row["obs"]
            # metadata = row.drop("obs").to_dict()
            self.add_observation(
                libname, obs, name=name, kind=kind, data_column=data_column
            )

    def add_observation(
        self,
        libname: str,
        obs: hpd.Obs,
        name: Optional[str] = None,
        kind: Optional[str] = None,
        data_column: Optional[str] = None,
        unit_multiplier: float = 1.0,
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

        if libname == "oseries":
            self._store.add_oseries(o, name, metadata=metadata)
            logger.info("%sobservation '%s' added to oseries library.", source, name)
        elif libname == "stresses":
            if kind is None:
                raise ValueError("`kind` must be specified for stresses!")
            self._store.add_stress(o * unit_multiplier, name, kind, metadata=metadata)
            logger.info(
                "%sstress '%s' (kind='%s') added to stresses library.",
                source,
                name,
                kind,
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
        update: bool = False,
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
        update : bool, optional
            if True, update currently stored precipitation time series with new data
        """
        self.download_knmi_meteo(
            meteo_var=meteo_var,
            kind="prec",
            stns=stns,
            tmin=tmin,
            tmax=tmax,
            unit_multiplier=unit_multiplier,
            update=update,
            **kwargs,
        )

    def download_knmi_evaporation(
        self,
        stns: Optional[list[int]] = None,
        meteo_var: str = "EV24",
        tmin: TimeType = None,
        tmax: TimeType = None,
        unit_multiplier: float = 1e3,
        update: bool = False,
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
        update : bool, optional
            if True, update currently stored evaporation time series with new data
        """
        self.download_knmi_meteo(
            meteo_var=meteo_var,
            kind="evap",
            stns=stns,
            tmin=tmin,
            tmax=tmax,
            unit_multiplier=unit_multiplier,
            update=update,
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
        update: bool = False,
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
        update : bool, optional
            if True, update currently stored precipitation time series with new data
        """
        # get tmin/tmax if not specified
        if update:
            stressnames = self._store.stresses.loc[
                self._store.stresses["kind"] == kind
            ].index.tolist()
            tmintmax = self._store.get_tmin_tmax("stresses", names=stressnames)
            if tmin is None:
                tmin = tmintmax.loc[:, "tmax"].min()
            if tmax is None:
                tmax = Timestamp.now().normalize()
        else:
            tmintmax = self._store.get_tmin_tmax("oseries")
            if tmin is None:
                tmin = tmintmax.loc[:, "tmin"].min() - Timedelta(days=10 * 365)
            if tmax is None:
                tmax = tmintmax.loc[:, "tmax"].max()

        # if update, only download data for stations in store
        if update:
            locations = None
            if stns is None:
                stns = self._store.stresses.loc[stressnames, "station"].tolist()
            else:
                check = np.isin(
                    stns, self._store.stresses.loc[stressnames, "station"].values
                )
                if not check.all():
                    raise ValueError(
                        "Not all specified stations are in the store: "
                        f"{np.array(stns)[~check]}"
                    )
        elif stns is None:
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
        )

    def download_bro_gmw(
        self,
        extent=None,
        tmin=None,
        tmax=None,
        update=False,
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
        update: bool, optional
            If True, update existing observations in the store.
        **kwargs: dict, optional
            Additional keyword arguments to pass to `hpd.read_bro()`
        """
        if extent is not None and update:
            raise ValueError("Cannot specify extent AND update=True.")
        elif extent is None and not update:
            raise ValueError("Either extent or update=True must be specified.")

        if update:
            tmintmax = self._store.get_tmin_tmax("oseries")
            for obsnam in tqdm(
                self._store.oseries.index, desc="Updating oseries from BRO"
            ):
                bro_id, tube_number = obsnam.split("_")
                tmin, tmax = tmintmax.loc[obsnam]
                obs = hpd.GroundWaterObs.from_bro(
                    bro_id, int(tube_number), tmin=tmin, tmax=tmax, **kwargs
                )
                self.add_observation("oseries", obs, name=obsnam, data_column="values")
        else:
            bro = hpd.read_bro(
                extent=extent,
                tmin=tmin,
                tmax=tmax,
                **kwargs,
            )
            self.add_obscollection("oseries", bro, data_column="values")
