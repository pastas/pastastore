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
from hydropandas.io.knmi import _get_default_settings, download_knmi_data, get_stations
from pandas import DataFrame, Series, Timedelta, Timestamp
from pastas.timeseries_utils import timestep_weighted_resample
from tqdm.auto import tqdm

from pastastore.extensions.accessor import register_pastastore_accessor

logger = logging.getLogger("hydropandas_extension")


TimeType = Optional[Union[str, Timestamp]]


def _check_latest_measurement_date_de_bilt(meteo_var: str, **kwargs):
    # get measurements at de Bilt
    stn_de_bilt = 550 if meteo_var == "RD" else 260
    ts_de_bilt, _, _ = download_knmi_data(
        stn_de_bilt,
        meteo_var,
        start=Timestamp.today() - Timedelta(days=60),
        end=Timestamp.today(),
        settings=_get_default_settings(kwargs),
        stn_name="De Bilt",
    )
    return ts_de_bilt.index[-1]  # last measurement date


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

    def __repr__(self):
        """Return string representation of HydroPandasExtension."""
        methods = "".join(
            [f"\n - {meth}" for meth in dir(self) if not meth.startswith("_")]
        )
        return "HydroPandasExtension, available methods:" + methods

    def add_obscollection(
        self,
        libname: str,
        oc: hpd.ObsCollection,
        kind: Optional[str] = None,
        data_column: Optional[str] = None,
        unit_multiplier: float = 1.0,
        update: bool = False,
        normalize_datetime_index: bool = False,
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
        normalize_datetime_index : bool, optional
            if True, normalize the datetime so stress value at midnight represents
            the daily total, by default True.
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
                normalize_datetime_index=normalize_datetime_index,
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
        normalize_datetime_index: bool = False,
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
        normalize_datetime_index : bool, optional
            if True, normalize the datetime so stress value at midnight represents
            the daily total, by default True.
        """
        # if data_column is not None, use data_column
        if data_column is not None:
            if not obs.empty:
                o = obs[[data_column]]
            else:
                o = Series()
        elif isinstance(obs, Series):
            o = obs
        # else raise error
        elif isinstance(obs, DataFrame) and (obs.columns.size > 1):
            raise ValueError("No data_column specified and obs has multiple columns.")
        else:
            raise TypeError("obs must be a Series or DataFrame with a single column.")

        # break if obs is empty
        if o.empty:
            logger.info("Observation '%s' is empty, not adding to store.", name)
            return

        if normalize_datetime_index and o.index.size > 1:
            o = self._normalize_datetime_index(o).iloc[1:]  # remove first nan
        elif normalize_datetime_index and o.index.size <= 1:
            raise ValueError(
                "Must have minimum of 2 observations for timestep_weighted_resample."
            )

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
        if unit == "m" and np.allclose(unit_multiplier, 1e3):
            metadata["unit"] = "mm"
        elif unit_multiplier != 1.0:
            metadata["unit"] = f"{unit_multiplier:.1e}*{unit}"

        source = metadata.get("source", "")
        if len(source) > 0:
            source = f"{source} "

        if update:
            action_msg = "updated in"
        else:
            action_msg = "added to"

        if libname == "oseries":
            self._store.upsert_oseries(o.squeeze(axis=1), name, metadata=metadata)
            logger.info(
                "%sobservation '%s' %s oseries library.", source, name, action_msg
            )
        elif libname == "stresses":
            if kind is None:
                raise ValueError("`kind` must be specified for stresses!")
            self._store.upsert_stress(
                (o * unit_multiplier).squeeze(axis=1), name, kind, metadata=metadata
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

    def _get_tmin_tmax(self, tmin, tmax, oseries=None):
        """Get tmin and tmax from store if not specified.

        Parameters
        ----------
        tmin : TimeType
            start time
        tmax : TimeType
            end time
        oseries : str, optional
            name of the observation series to get tmin/tmax for, by default None

        Returns
        -------
        tmin, tmax : TimeType, TimeType
            tmin and tmax
        """
        # get tmin/tmax if not specified
        if tmin is None or tmax is None:
            tmintmax = self._store.get_tmin_tmax(
                "oseries", names=[oseries] if oseries else None
            )
        if tmin is None:
            tmin = tmintmax.loc[:, "tmin"].min() - Timedelta(days=10 * 365)
        if tmax is None:
            tmax = tmintmax.loc[:, "tmax"].max()
        return tmin, tmax

    @staticmethod
    def _normalize_datetime_index(obs):
        """Normalize observation datetime index (i.e. set observation time to midnight).

        Parameters
        ----------
        obs : pandas.Series
            observation series to normalize

        Returns
        -------
        hpd.Obs
            observation series with normalized datetime index
        """
        if isinstance(obs, hpd.Obs):
            metadata = {k: getattr(obs, k) for k in obs._metadata}
        else:
            metadata = {}
        return obs.__class__(
            timestep_weighted_resample(
                obs,
                obs.index.normalize(),
            ).rename(obs.name),
            **metadata,
        )

    def download_knmi_precipitation(
        self,
        stns: Optional[list[int]] = None,
        meteo_var: str = "RD",
        tmin: TimeType = None,
        tmax: TimeType = None,
        unit_multiplier: float = 1e3,
        fill_missing_obs: bool = True,
        normalize_datetime_index: bool = True,
        **kwargs,
    ):
        """Download precipitation data from KNMI and store in PastaStore.

        Parameters
        ----------
        stns : list of int/str, optional
            list of station numbers to download data for, by default None
        meteo_var : str, optional
            variable to download, by default "RD", valid options are ["RD", "RH"].
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
            fill_missing_obs=fill_missing_obs,
            normalize_datetime_index=normalize_datetime_index,
            **kwargs,
        )

    def download_knmi_evaporation(
        self,
        stns: Optional[list[int]] = None,
        meteo_var: str = "EV24",
        tmin: TimeType = None,
        tmax: TimeType = None,
        unit_multiplier: float = 1e3,
        fill_missing_obs: bool = True,
        normalize_datetime_index: bool = True,
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
        fill_missing_obs : bool, optional
            if True, fill missing observations by getting observations from nearest
            station with data.
        normalize_datetime_index : bool, optional
            if True, normalize the datetime so stress value at midnight represents
            the daily total, by default True.
        """
        self.download_knmi_meteo(
            meteo_var=meteo_var,
            kind="evap",
            stns=stns,
            tmin=tmin,
            tmax=tmax,
            unit_multiplier=unit_multiplier,
            fill_missing_obs=fill_missing_obs,
            normalize_datetime_index=normalize_datetime_index,
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
        normalize_datetime_index: bool = True,
        fill_missing_obs: bool = True,
        **kwargs,
    ):
        """Download meteorological data from KNMI and store in PastaStore.

        Parameters
        ----------
        meteo_var : str, optional
            variable to download, by default "RH", valid options are
            e.g. ["RD", "RH", "EV24", "T", "Q"].
        kind : str
            kind identifier for observations in pastastore, usually "prec" or "evap".
        stns : list of int/str, optional
            list of station numbers to download data for, by default None
        tmin : TimeType, optional
            start time, by default None
        tmax : TimeType, optional
            end time, by default None
        unit_multiplier : float, optional
            multiply unit by this value before saving it in the store,
            by default 1.0 (no conversion)
        fill_missing_obs : bool, optional
            if True, fill missing observations by getting observations from nearest
            station with data.
        normalize_datetime_index : bool, optional
            if True, normalize the datetime so stress value at midnight represents
            the daily total, by default True.
        """
        tmin, tmax = self._get_tmin_tmax(tmin, tmax)

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
            fill_missing_obs=fill_missing_obs,
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
            normalize_datetime_index=normalize_datetime_index,
        )

    def download_nearest_knmi_precipitation(
        self,
        oseries: str,
        meteo_var: str = "RD",
        tmin: Optional[TimeType] = None,
        tmax: Optional[TimeType] = None,
        unit_multiplier: float = 1e3,
        normalize_datetime_index: bool = True,
        fill_missing_obs: bool = True,
        **kwargs,
    ):
        """Download precipitation time series data from nearest KNMI station.

        Parameters
        ----------
        oseries : str
            download nearest precipitation information for this observation well
        meteo_var : str, optional
            variable to download, by default "RD", valid options are ["RD", "RH"].
        tmin : TimeType
            start time
        tmax : TimeType
            end time
        unit_multiplier : float, optional
            multiply unit by this value before saving it in the store,
            by default 1e3 (converting m to mm)
        fill_missing_obs : bool, optional
            if True, fill missing observations by getting observations from nearest
            station with data.
        fill_missing_obs : bool, optional
            if True, fill missing observations by getting observations from nearest
            station with data.
        """
        self.download_nearest_knmi_meteo(
            oseries=oseries,
            meteo_var=meteo_var,
            kind="prec",
            tmin=tmin,
            tmax=tmax,
            unit_multiplier=unit_multiplier,
            normalize_datetime_index=normalize_datetime_index,
            fill_missing_obs=fill_missing_obs,
            **kwargs,
        )

    def download_nearest_knmi_evaporation(
        self,
        oseries: str,
        meteo_var: str = "EV24",
        tmin: Optional[TimeType] = None,
        tmax: Optional[TimeType] = None,
        unit_multiplier: float = 1e3,
        normalize_datetime_index: bool = True,
        fill_missing_obs: bool = True,
        **kwargs,
    ):
        """Download evaporation time series data from nearest KNMI station.

        Parameters
        ----------
        oseries : str
            download nearest evaporation information for this observation well
        meteo_var : str, optional
            variable to download, by default "EV24", valid options are:
            ["EV24", "penman", "hargreaves", "makkink"].
        tmin : TimeType
            start time
        tmax : TimeType
            end time
        unit_multiplier : float, optional
            multiply unit by this value before saving it in the store,
            by default 1e3 (converting m to mm)
        fill_missing_obs : bool, optional
            if True, fill missing observations by getting observations from nearest
            station with data.
        fill_missing_obs : bool, optional
            if True, fill missing observations by getting observations from nearest
            station with data.
        """
        self.download_nearest_knmi_meteo(
            oseries=oseries,
            meteo_var=meteo_var,
            kind="evap",
            tmin=tmin,
            tmax=tmax,
            unit_multiplier=unit_multiplier,
            normalize_datetime_index=normalize_datetime_index,
            fill_missing_obs=fill_missing_obs,
            **kwargs,
        )

    def download_nearest_knmi_meteo(
        self,
        oseries: str,
        meteo_var: str,
        kind: str,
        tmin: Optional[TimeType] = None,
        tmax: Optional[TimeType] = None,
        unit_multiplier: float = 1.0,
        normalize_datetime_index: bool = True,
        fill_missing_obs: bool = True,
        **kwargs,
    ):
        """Download meteorological data from nearest KNMI station.

        Parameters
        ----------
        oseries : str
            download nearest meteorological information for this observation well
        meteo_var : str
            meteorological variable to download, e.g. "RD", "RH", "EV24", "T", "Q"
        kind : str
            kind identifier for observations in pastastore, usually "prec" or "evap".
        tmin : TimeType
            start time
        tmax : TimeType
            end time
        unit_multiplier : float, optional
            multiply unit by this value before saving it in the store,
            by default 1.0 (no conversion)
        fill_missing_obs : bool, optional
            if True, fill missing observations by getting observations from nearest
            station with data.
        fill_missing_obs : bool, optional
            if True, fill missing observations by getting observations from nearest
            station with data.
        """
        xy = self._store.oseries.loc[[oseries], ["x", "y"]].to_numpy()
        # download data
        tmin, tmax = self._get_tmin_tmax(tmin, tmax, oseries=oseries)
        knmi = hpd.read_knmi(
            xy=xy,
            meteo_vars=[meteo_var],
            starts=tmin,
            ends=tmax,
            fill_missing_obs=fill_missing_obs,
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
            normalize_datetime_index=normalize_datetime_index,
        )

    def update_knmi_meteo(
        self,
        names: Optional[List[str]] = None,
        tmin: TimeType = None,
        tmax: TimeType = None,
        fill_missing_obs: bool = True,
        normalize_datetime_index: bool = True,
        raise_on_error: bool = False,
        **kwargs,
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
        fill_missing_obs : bool, optional
            if True, fill missing observations by getting observations from nearest
            station with data.
        normalize_datetime_index : bool, optional
            if True, normalize the datetime so stress value at midnight represents
            the daily total, by default True.
        raise_on_error : bool, optional
            if True, raise error if an error occurs, by default False
        **kwargs : dict, optional
            Additional keyword arguments to pass to `hpd.read_knmi()`
        """
        if "source" not in self._store.stresses.columns:
            msg = (
                "Cannot update KNMI stresses! "
                "KNMI stresses cannot be identified if 'source' column is not defined."
            )
            logger.error(msg)
            if raise_on_error:
                raise ValueError(msg)
            else:
                return

        if names is None:
            names = self._store.stresses.loc[
                self._store.stresses["source"] == "KNMI"
            ].index.tolist()

        tmintmax = self._store.get_tmin_tmax("stresses", names=names)

        if tmax is not None:
            if tmintmax["tmax"].min() >= Timestamp(tmax):
                logger.info(f"All KNMI stresses are up to date till {tmax}.")
                return

        try:
            maxtmax_rd = _check_latest_measurement_date_de_bilt("RD")
            maxtmax_ev24 = _check_latest_measurement_date_de_bilt("EV24")
        except Exception as e:
            # otherwise use maxtmax 28 days (4 weeks) prior to today
            logger.warning(
                "Could not check latest measurement date in De Bilt: %s" % str(e)
            )
            maxtmax_rd = maxtmax_ev24 = Timestamp.today() - Timedelta(days=28)
            logger.info(
                "Using 28 days (4 weeks) prior to today as maxtmax: %s."
                % str(maxtmax_rd)
            )

        for name in tqdm(names, desc="Updating KNMI meteo stresses"):
            meteo_var = self._store.stresses.loc[name, "meteo_var"]
            if meteo_var == "RD":
                maxtmax = maxtmax_rd
            elif meteo_var == "EV24":
                maxtmax = maxtmax_ev24
            else:
                maxtmax = maxtmax_rd

            # 1 days extra to ensure computation of daily totals using
            # timestep_weighted_resample
            if tmin is None:
                itmin = tmintmax.loc[name, "tmax"] - Timedelta(days=1)
            else:
                itmin = tmin - Timedelta(days=1)

            # ensure 2 observations at least
            if itmin >= (maxtmax - Timedelta(days=1)):
                logger.debug("KNMI %s is already up to date." % name)
                continue

            if tmax is None:
                itmax = maxtmax
            else:
                itmax = Timestamp(tmax)

            # fix for duplicate station entry in metadata:
            stress_station = (
                self._store.stresses.at[name, "station"]
                if "station" in self._store.stresses.columns
                else None
            )
            if stress_station is not None and not isinstance(
                stress_station, (int, np.integer)
            ):
                stress_station = stress_station.squeeze().unique().item()

            unit = self._store.stresses.loc[name, "unit"]
            kind = self._store.stresses.loc[name, "kind"]
            if stress_station is not None:
                stn = stress_station
            else:
                stns = get_stations(meteo_var)
                stn_name = name.split("_")[-1].lower()
                mask = stns["name"].str.lower().str.replace(" ", "-") == stn_name
                if not mask.any():
                    logger.warning(
                        "Station '%s' not found in list of KNMI %s stations."
                        % (stn_name, meteo_var)
                    )
                    continue
                stn = stns.loc[mask].index[0]

            if unit == "m":
                unit_multiplier = 1.0
            elif unit == "mm":
                unit_multiplier = 1e3
            elif unit.count("m") == 1 and unit.endswith("m"):
                unit_multiplier = float(unit.replace("m", ""))
            else:
                unit_multiplier = 1.0
                logger.warning(
                    "Unit '%s' not recognized, using unit_multiplier=%.1e."
                    % (unit, unit_multiplier)
                )

            logger.debug("Updating KNMI %s from %s to %s" % (name, itmin, itmax))
            knmi = hpd.read_knmi(
                stns=[stn],
                meteo_vars=[meteo_var],
                starts=itmin,
                ends=itmax,
                fill_missing_obs=fill_missing_obs,
                **kwargs,
            )
            obs = knmi["obs"].iloc[0]

            try:
                self.add_observation(
                    "stresses",
                    obs,
                    name=name,
                    kind=kind,
                    data_column=meteo_var,
                    unit_multiplier=unit_multiplier,
                    update=True,
                    normalize_datetime_index=normalize_datetime_index,
                )
            except ValueError as e:
                logger.error("Error updating KNMI %s: %s" % (name, str(e)))
                if raise_on_error:
                    raise e

    def download_bro_gmw(
        self,
        extent: Optional[List[float]] = None,
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
