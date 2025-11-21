"""Useful utilities for pastastore."""

import json
import logging
import os
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd
from colorama import Back, Fore, Style
from numpy.lib._iotools import NameValidator
from pandas.testing import assert_series_equal
from pastas import Model
from pastas.io.pas import PastasEncoder
from pastas.stats.tests import runs_test, stoffer_toloi
from tqdm.auto import tqdm

from pastastore.styling import boolean_row_styler
from pastastore.typing import TimeSeriesLibs

logger = logging.getLogger(__name__)


def _custom_warning(message, category=UserWarning, filename="", lineno=-1, *args):
    print(f"{filename}:{lineno}: {category.__name__}: {message}")


class ZipUtils:
    """Utility class for zip file operations."""

    def __init__(self, pstore):
        self.pstore = pstore

    def _stored_series_to_json(
        self,
        libname: TimeSeriesLibs,
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
        names = self.pstore.parse_names(names, libname=libname)
        files = []
        for n in tqdm(names, desc=libname) if progressbar else names:
            s = self.pstore.conn._get_series(libname, n, progressbar=False)
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
        libname: TimeSeriesLibs,
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
        names = self.pstore.parse_names(names, libname=libname)
        files = []
        for n in tqdm(names, desc=libname) if progressbar else names:
            meta = self.pstore.get_metadata(libname, n, as_frame=False)
            meta_json = json.dumps(meta, cls=PastasEncoder, indent=4)
            files.append(meta_json)
        if len(files) == 1 and squeeze:
            return files[0]
        else:
            return files

    def series_to_archive(
        self,
        archive,
        libname: TimeSeriesLibs,
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
        names = self.pstore.parse_names(names, libname=libname)
        for n in tqdm(names, desc=libname) if progressbar else names:
            sjson = self._stored_series_to_json(
                libname, names=n, progressbar=False, squeeze=True
            )
            meta_json = self._stored_metadata_to_json(
                libname, names=n, progressbar=False, squeeze=True
            )
            archive.writestr(f"{libname}/{n}.pas", sjson)
            archive.writestr(f"{libname}/{n}_meta.pas", meta_json)

    def models_to_archive(self, archive, names=None, progressbar=True):
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
        names = self.pstore.parse_names(names, libname="models")
        for n in tqdm(names, desc="models") if progressbar else names:
            m = self.pstore.get_models(n, return_dict=True)
            jsondict = json.dumps(m, cls=PastasEncoder, indent=4)
            archive.writestr(f"models/{n}.pas", jsondict)


class ColoredFormatter(logging.Formatter):
    """Colored log formatter.

    Taken from
    https://gist.github.com/joshbode/58fac7ababc700f51e2a9ecdebe563ad
    """

    def __init__(
        self, *args, colors: Optional[Dict[str, str]] = None, **kwargs
    ) -> None:
        """Initialize the formatter with specified format strings."""
        super().__init__(*args, **kwargs)

        self.colors = colors if colors else {}

    def format(self, record) -> str:
        """Format the specified record as text."""
        record.color = self.colors.get(record.levelname, "")
        record.reset = Style.RESET_ALL

        return super().format(record)


def get_color_logger(level="INFO", logger_name=None):
    """Get a logger with colored output.

    Parameters
    ----------
    level : str, optional
        The logging level to set for the logger. Default is "INFO".

    Returns
    -------
    logger : logging.Logger
        The configured logger object.
    """
    if level == "DEBUG":
        FORMAT = "{color}{levelname}:{name}.{funcName}:{lineno}:{message}{reset}"
    else:
        FORMAT = "{color}{message}{reset}"
    formatter = ColoredFormatter(
        FORMAT,
        style="{",
        datefmt="%Y-%m-%d %H:%M:%S",
        colors={
            "DEBUG": Fore.CYAN,
            "INFO": Fore.GREEN,
            "WARNING": Fore.YELLOW,
            "ERROR": Fore.RED,
            "CRITICAL": Fore.RED + Back.WHITE + Style.BRIGHT,
        },
    )

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)

    logger = logging.getLogger(logger_name)
    logger.handlers[:] = []
    logger.addHandler(handler)
    logger.setLevel(getattr(logging, level))

    logging.captureWarnings(True)
    return logger


class ItemInLibraryException(Exception):
    """Exception when item is already in library."""


class SeriesUsedByModel(Exception):
    """Exception raised when a series is used by a model."""


def series_from_json(fjson: str, squeeze: bool = True):
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
        return s.squeeze(axis="columns")
    return s


def metadata_from_json(fjson: str):
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
    with open(fjson, "r", encoding="utf-8") as f:
        meta = json.load(f)
    return meta


def delete_arcticdb_connector(
    conn=None,
    uri: Optional[str] = None,
    name: Optional[str] = None,
    libraries: Optional[List[str]] = None,
) -> None:
    """Delete libraries from arcticDB database.

    Parameters
    ----------
    conn : pastastore.ArcticDBConnector
        ArcticDBConnector object
    uri : str, optional
        uri connection string to the database
    name : str, optional
        name of the database
    libraries : Optional[List[str]], optional
        list of library names to delete, by default None which deletes
        all libraries
    """
    import arcticdb

    if conn is not None:
        name = conn.name
        uri = conn.uri
    elif name is None or uri is None:
        raise ValueError("Provide 'name' and 'uri' OR 'conn'!")

    # connect to arcticdb
    arc = arcticdb.Arctic(uri)

    logger.info("Deleting ArcticDBConnector database: '%s' ... ", name)
    # get library names
    if libraries is None:
        libs = []
        for ilib in arc.list_libraries():
            if ilib.split(".")[0] == name:
                libs.append(ilib)
    elif name is not None:
        libs = [name + "." + ilib for ilib in libraries]
    else:
        raise ValueError("Provide 'name' and 'uri' OR 'conn'!")

    for lib in libs:
        arc.delete_library(lib)

        if libraries is not None:
            logger.info(" - deleted: %s", lib)

    # delete .pastastore file if entire pastastore is deleted
    remaining_libs = [
        ilib for ilib in arc.list_libraries() if ilib.split(".")[0] == name
    ]
    if remaining_libs == 0:
        os.unlink(uri.split("//")[-1] / f"{name}.pastastore")

    # check if any remaining libraries in lmdb dir, if none, delete entire folder
    remaining = arc.list_libraries()
    if len(remaining) == 0:
        shutil.rmtree(Path(conn.uri.split("://")[-1]))

    logger.info("Done!")


def delete_dict_connector(conn, libraries: Optional[List[str]] = None) -> None:
    """Delete DictConnector object."""
    logger.info("Deleting DictConnector: '%s' ... ", conn.name)
    if libraries is None:
        del conn
        logger.info("Done!")
    else:
        for lib in libraries:
            delattr(conn, f"lib_{conn.libname[lib]}")
            logger.info(" - deleted: %s", lib)
    logger.info("Done!")


def delete_pas_connector(conn, libraries: Optional[List[str]] = None) -> None:
    """Delete PasConnector object."""
    logger.info("Deleting PasConnector database: '%s' ... ", conn.name)
    if libraries is None:
        shutil.rmtree(conn.path)
        logger.info("Done!")
    else:
        for lib in libraries:
            shutil.rmtree(conn.path / lib)
            logger.info(" - deleted: %s", lib)
        logger.info("Done!")


def delete_pastastore(pstore, libraries: Optional[List[str]] = None) -> None:
    """Delete libraries from PastaStore.

    Note
    ----
    This deletes the original PastaStore object. To access
    data that has not been deleted, it is recommended to create a new
    PastaStore object with the same Connector settings. This also creates
    new empty libraries if they were deleted.

    Parameters
    ----------
    pstore : pastastore.PastaStore
        PastaStore object to delete (from)
    libraries : Optional[List[str]], optional
        list of library names to delete, by default None which deletes
        all libraries

    Raises
    ------
    TypeError
        when Connector type is not recognized
    """
    if pstore.conn.conn_type == "dict":
        delete_dict_connector(pstore)
    elif pstore.conn.conn_type == "arcticdb":
        delete_arcticdb_connector(conn=pstore.conn, libraries=libraries)
    elif pstore.conn.conn_type == "pas":
        delete_pas_connector(conn=pstore.conn, libraries=libraries)
    else:
        raise TypeError(
            f"Unrecognized pastastore Connector type: {pstore.conn.conn_type}"
        )


def validate_names(
    s: Optional[str] = None,
    d: Optional[dict] = None,
    replace_space: Optional[str] = "_",
    deletechars: Optional[str] = None,
    **kwargs,
) -> Union[str, Dict]:
    """Remove invalid characters from string or dictionary keys.

    Parameters
    ----------
    s : str, optional
        remove invalid characters from string
    d : dict, optional
        remove invalid characters from keys from dictionary
    replace_space : str, optional
        replace spaces by this character, by default "_"
    deletechars : str, optional
        a string combining invalid characters, by default None

    Returns
    -------
    str, dict
        string or dict with invalid characters removed
    """
    validator = NameValidator(
        replace_space=replace_space, deletechars=deletechars, **kwargs
    )
    if s is not None:
        new_str = validator(s)  # tuple
        if len(new_str) == 1:
            return new_str[0]
        else:
            return new_str
    elif d is not None:
        new_dict = {}
        for k, v in d.items():
            new_dict[validator(k)[0]] = v
        return new_dict
    else:
        raise ValueError("Provide one of 's' (string) or 'd' (dict)!")


def compare_models(
    ml1: Model,
    ml2: Model,
    stats: List[str] = None,
    detailed_comparison: bool = False,
    style_output: bool = False,
) -> pd.DataFrame:
    """Compare two Pastas models.

    Parameters
    ----------
    ml1 : pastas.Model
        first model to compare
    ml2 : pastas.Model
        second model to compare
    stats : list of str, optional
        if provided compare these model statistics
    detailed_comparison : bool, optional
        if True return DataFrame containing comparison details,
        by default False which returns True if models are equivalent
        or False if they are not
    style_output : bool, optional
        if True and detailed_comparison is True, return styled DataFrame
        with colored output, by default False

    Returns
    -------
    bool or pd.DataFrame
        returns True if models are equivalent when detailed_comparison=True
        else returns DataFrame containing comparison details.
    """
    df = pd.DataFrame(columns=["model 0", "model 1"])
    so1 = []  # for storing series_original
    ss1 = []  # for storing series

    for i, ml in enumerate([ml1, ml2]):
        counter = 0  # for counting stress time series
        df.loc["name:", f"model {i}"] = ml.name

        for k in ml.settings.keys():
            df.loc[f"- settings: {k}", f"model {i}"] = ml.settings.get(k)

        if i == 0:
            oso = ml.oseries._series_original  # noqa: SLF001
            df.loc["oseries: series_original", f"model {i}"] = True

            oss = ml.oseries.series
            df.loc["oseries: series", f"model {i}"] = True

        elif i == 1:
            try:
                assert_series_equal(oso, ml.oseries._series_original)  # noqa: SLF001
                compare_oso = True
            except (ValueError, AssertionError):
                # series are not identical in length or index does not match
                compare_oso = False

            try:
                assert_series_equal(oss, ml.oseries.series)
                compare_oss = True
            except (ValueError, AssertionError):
                # series are not identical in length or index does not match
                compare_oss = False

            df.loc["oseries: series_original", f"model {i}"] = compare_oso
            df.loc["oseries: series", f"model {i}"] = compare_oss

        for sm_name, sm in ml.stressmodels.items():
            df.loc[f"stressmodel: '{sm_name}'"] = sm_name
            df.loc["- rfunc"] = (
                type(sm.rfunc).__name__ if sm.rfunc is not None else "NA"
            )

            if type(sm).__name__ == "RechargeModel":
                stresses = [sm.prec, sm.evap]
            else:
                stresses = sm.stress

            for ts in stresses:
                df.loc[f"- time series: '{ts.name}'"] = ts.name
                for tsk in ts.settings.keys():
                    df.loc[f"  - {ts.name} settings: {tsk}", f"model {i}"] = (
                        ts.settings[tsk]
                    )

                if i == 0:
                    so1.append(ts._series_original.copy())  # noqa: SLF001
                    ss1.append(ts.series.copy())

                    df.loc[f"  - {ts.name}: series_original"] = True
                    df.loc[f"  - {ts.name}: series"] = True

                elif i == 1:
                    # ValueError if series cannot be compared,
                    # set result to False
                    try:
                        assert_series_equal(
                            so1[counter],
                            ts._series_original,  # noqa: SLF001
                        )
                        compare_so1 = True
                    except (ValueError, AssertionError):
                        compare_so1 = False

                    try:
                        assert_series_equal(ss1[counter], ts.series)
                        compare_ss1 = True
                    except (ValueError, AssertionError):
                        compare_ss1 = False
                    df.loc[f"  - {ts.name}: series_original"] = compare_so1
                    df.loc[f"  - {ts.name}: series"] = compare_ss1

                counter += 1

        for p in ml.parameters.index:
            df.loc[f"param: {p} (init)", f"model {i}"] = ml.parameters.loc[p, "initial"]
            df.loc[f"param: {p} (opt)", f"model {i}"] = ml.parameters.loc[p, "optimal"]

        if stats:
            stats_df = ml.stats.summary(stats=stats)
            for s in stats:
                df.loc[f"stats: {s}", f"model {i}"] = stats_df.loc[s, "Value"]

    # compare
    df["comparison"] = df.iloc[:, 0] == df.iloc[:, 1]

    # isclose for params
    param_mask = df.index.str.startswith("param: ")
    df.loc[param_mask, "comparison"] = np.isclose(
        df.loc[param_mask, "model 0"].astype(float).values,
        df.loc[param_mask, "model 1"].astype(float).values,
    )

    # ensure NaN == NaN is not counted as a difference
    nanmask = df.iloc[:, 0].isna() & df.iloc[:, 1].isna()
    df.loc[nanmask, "comparison"] = True

    # for stats comparison must be almost_equal
    if stats:
        stats_idx = [f"stats: {s}" for s in stats]
        b = np.isclose(
            df.loc[stats_idx, "model 0"].astype(float).values,
            df.loc[stats_idx, "model 1"].astype(float).values,
        )
        df.loc[stats_idx, "comparison"] = b

    if detailed_comparison:
        if style_output:
            return df.style.apply(
                boolean_row_styler, column="comparison", axis=1
            ).set_table_styles(
                [
                    {"selector": "th.col_heading", "props": [("text-align", "center")]},
                    {
                        "selector": "th.row_heading",
                        "props": [("text-align", "left"), ("white-space", "pre")],
                    },
                ],
            )
        else:
            return df
    else:
        return df["comparison"].iloc[1:].all()  # ignore name difference


def copy_database(
    conn1,
    conn2,
    libraries: Optional[List[str]] = None,
    overwrite: bool = False,
    progressbar: bool = True,
) -> None:
    """Copy libraries from one database to another.

    Parameters
    ----------
    conn1 : pastastore.*Connector
        source Connector containing link to current database containing data
    conn2 : pastastore.*Connector
        destination Connector with link to database to which you want to copy
    libraries : Optional[List[str]], optional
        list of str containing names of libraries to copy, by default None,
        which copies all libraries: ['oseries', 'stresses', 'models']
    overwrite : bool, optional
        overwrite data in destination database, by default False
    progressbar : bool, optional
        show progressbars, by default False

    Raises
    ------
    ValueError
        if library name is not understood
    """
    if libraries is None:
        libraries = ["oseries", "stresses", "models"]

    for lib in libraries:
        if lib == "oseries":
            for name in (
                tqdm(conn1.oseries_names, desc="copying oseries")
                if progressbar
                else conn1.oseries_names
            ):
                o, meta = conn1.get_oseries(name, return_metadata=True)
                conn2.add_oseries(o, name, metadata=meta, overwrite=overwrite)
        elif lib == "stresses":
            for name in (
                tqdm(conn1.stresses_names, desc="copying stresses")
                if progressbar
                else conn1.stresses_names
            ):
                s, meta = conn1.get_stresses(name, return_metadata=True)
                conn2.add_stress(
                    s,
                    name,
                    kind=meta["kind"],
                    metadata=meta,
                    overwrite=overwrite,
                )
        elif lib == "models":
            for name in (
                tqdm(conn1.model_names, desc="copying models")
                if progressbar
                else conn1.model_names
            ):
                mldict = conn1.get_models(name, return_dict=True)
                conn2.add_model(mldict, overwrite=overwrite)
        else:
            raise ValueError(f"Library name '{lib}' not recognized!")


def frontiers_checks(
    pstore,
    modelnames: Optional[List[str]] = None,
    oseries: Optional[List[str]] = None,
    check1_rsq: bool = True,
    check1_threshold: float = 0.7,
    check2_autocor: bool = True,
    check2_test: str = "runs",
    check2_pvalue: float = 0.05,
    check3_tmem: bool = True,
    check3_cutoff: float = 0.95,
    check4_gain: bool = True,
    check5_parambounds: bool = False,
    csv_dir: Optional[str] = None,
    progressbar: bool = False,
) -> pd.DataFrame:  # pragma: no cover
    """Check models in a PastaStore to see if they pass reliability criteria.

    The reliability criteria are taken from Brakenhoff et al. 2022 [bra_2022]_.
    These criteria were applied in a region with recharge, river levels and
    pumping wells as stresses. This is by no means an exhaustive list of
    reliability criteria but might serve as a reasonable starting point for
    model diagnostic checking.

    Parameters
    ----------
    pstore : pastastore.PastaStore
        reference to a PastaStore
    modelnames : list of str, optional
        list of model names to consider, if None checks 'oseries', if both are
        None, all stored models will be checked
    oseries :  list of str, optional
        list of oseries to consider, corresponding models will be picked up
        from pastastore. If None, uses all stored models are checked.
    check1 : bool, optional
        check if model fit is above a threshold of the coefficient
        of determination $R^2$ , by default True
    check1_threshold : float, optional
        threshold of the $R^2$ fit statistic, by default 0.7
    check2 : bool, optional
        check if the noise of the model has autocorrelation with
        statistical test, by default True
    check2_test : str, optional
        statistical test for autocorrelation. Available options are Runs
        test "runs", Stoffer-Toloi "stoffer" or "both", by default "runs"
    check2_pvalue : float, optional
        p-value for the statistical test to define the confindence
        interval, by default 0.05
    check3 : bool, optional
        check if the length of the response time is within the
        calibration period, by default True
    check3_cutoff : float, optional
        the cutoff of the response time, by default 0.95
    check4 : bool, optional
        check if the uncertainty of the gain, by default True
    check5 : bool, optional
        check if parameters hit parameter bounds, by default False
    csv_dir : string, optional
        directory to store CSV file with overview of checks for every
        model, by default None which will not store results
    progressbar : bool, optional
        show progressbar, by default False

    Returns
    -------
    df : pandas.DataFrame
        dataFrame with all models and whether or not they pass
        the reliability checks


    References
    ----------
    .. [bra_2022]
    Brakenhoff, D.A., Vonk M.A., Collenteur, R.A., van Baar, M., Bakker, M.:
    Application of Time Series Analysis to Estimate Drawdown From Multiple Well
    Fields. Front. Earth Sci., 14 June 2022 doi:10.3389/feart.2022.907609
    """
    df = pd.DataFrame(columns=["all_checks_passed"])

    if modelnames is not None:
        models = modelnames
        if oseries is not None:
            logger.warning(
                "Warning! Both 'modelnames' and 'oseries' provided,"
                " only using 'modelnames'!"
            )
    elif oseries is not None:
        models = []
        for o in oseries:
            models += pstore.oseries_models[o]
    else:
        models = pstore.model_names

    for mlnam in (
        tqdm(models, desc="Running model diagnostics") if progressbar else models
    ):
        ml = pstore.get_models(mlnam)

        if ml.parameters["optimal"].hasnans:
            logger.warning(
                "Warning! Skipping model '%s' because it is not solved!", mlnam
            )
            continue

        checks = pd.DataFrame(columns=["stat", "threshold", "units", "check_passed"])

        # Check 1 - Fit Statistic
        if check1_rsq:
            rsq = ml.stats.rsq()
            check_rsq_passed = rsq >= check1_threshold
            checks.loc["rsq >= threshold", :] = (
                rsq,
                check1_threshold,
                "-",
                check_rsq_passed,
            )

        # Check 2 - Autocorrelation Noise
        if check2_autocor:
            noise = ml.noise()
            if noise is None:
                noise = ml.residuals()
                logger.warning(
                    "Warning! Checking autocorrelation on the residuals not the noise!"
                )
            if check2_test == "runs" or check2_test == "both":
                _, p_runs = runs_test(noise.iloc[1:])
                if p_runs > check2_pvalue:  # No autocorrelation
                    check_runs_acf_passed = True
                else:  # Significant autocorrelation
                    check_runs_acf_passed = False
                checks.loc["ACF: Runs test", :] = (
                    p_runs,
                    check2_pvalue,
                    "-",
                    check_runs_acf_passed,
                )
            if check2_test == "stoffer" or check2_test == "both":
                _, p_stoffer = stoffer_toloi(
                    noise.iloc[1:], snap_to_equidistant_timestamps=True
                )
                if p_stoffer > check2_pvalue:
                    check_st_acf_passed = True
                else:
                    check_st_acf_passed = False
                checks.loc["ACF: Stoffer-Toloi test", :] = (
                    p_stoffer,
                    check2_pvalue,
                    "-",
                    check_st_acf_passed,
                )

        # Check 3 - Response Time
        if check3_tmem:
            len_oseries_calib = (ml.settings["tmax"] - ml.settings["tmin"]).days
            for sm_name, sm in ml.stressmodels.items():
                if type(sm).__name__ == "WellModel":
                    nwells = sm.distances.index.size
                    for iw in range(nwells):
                        p = sm.get_parameters(model=ml, istress=iw)
                        t = sm.rfunc.get_t(p, dt=1, cutoff=0.999)
                        step = sm.rfunc.step(p, cutoff=0.999) / sm.rfunc.gain(p)
                        tmem = np.interp(check3_cutoff, step, t)
                        check_tmem_passed = tmem < len_oseries_calib / 2
                        idxlbl = (
                            f"calib_period > 2*t_mem_95%: "
                            f"{sm_name}-{iw:02g} ({sm.distances.index[iw]})"
                        )
                        checks.loc[idxlbl, :] = (
                            tmem,
                            len_oseries_calib,
                            "days",
                            check_tmem_passed,
                        )
                else:
                    tmem = ml.get_response_tmax(sm_name, cutoff=check3_cutoff)
                    if tmem is None:  # no rfunc in stressmodel
                        tmem = 0
                    check_tmem_passed = tmem < len_oseries_calib / 2
                    checks.loc[f"calib_period > 2*t_mem_95%: {sm_name}", :] = (
                        tmem,
                        len_oseries_calib,
                        "days",
                        check_tmem_passed,
                    )

        # Check 4 - Uncertainty Gain
        if check4_gain:
            for sm_name, sm in ml.stressmodels.items():
                if type(sm).__name__ == "WellModel":
                    for iw in range(sm.distances.index.size):
                        p = sm.get_parameters(model=ml, istress=iw)
                        gain = sm.rfunc.gain(p)
                        gain_std = np.sqrt(sm.variance_gain(model=ml, istress=iw))
                        if gain_std is None:
                            gain_std = np.nan
                            check_gain_passed = pd.NA
                        elif np.isnan(gain_std):
                            check_gain_passed = pd.NA
                        else:
                            check_gain_passed = np.abs(gain) > 2 * gain_std
                        checks.loc[
                            (
                                f"gain > 2*std: {sm_name}-{iw:02g}"
                                f" ({sm.distances.index[iw]})"
                            ),
                            :,
                        ] = (
                            gain,
                            2 * gain_std,
                            "(unit head)/(unit well stress)",
                            check_gain_passed,
                        )
                    continue
                elif type(sm).__name__ == "LinearTrend":
                    gain = ml.parameters.loc[f"{sm_name}_a", "optimal"]
                    gain_std = ml.parameters.loc[f"{sm_name}_a", "stderr"]
                elif type(sm).__name__ == "StepModel":
                    gain = ml.parameters.loc[f"{sm_name}_d", "optimal"]
                    gain_std = ml.parameters.loc[f"{sm_name}_d", "stderr"]
                else:
                    gain = ml.parameters.loc[f"{sm_name}_A", "optimal"]
                    gain_std = ml.parameters.loc[f"{sm_name}_A", "stderr"]

                if gain_std is None:
                    gain_std = np.nan
                    check_gain_passed = pd.NA
                elif np.isnan(gain_std):
                    check_gain_passed = pd.NA
                else:
                    check_gain_passed = np.abs(gain) > 2 * gain_std
                checks.loc[f"gain > 2*std: {sm_name}", :] = (
                    gain,
                    2 * gain_std,
                    "(unit head)/(unit well stress)",
                    check_gain_passed,
                )

        # Check 5 - Parameter Bounds
        if check5_parambounds:
            upper, lower = ml._check_parameters_bounds()  # noqa: SLF001
            for param in ml.parameters.index:
                bounds = (
                    ml.parameters.loc[param, "pmin"],
                    ml.parameters.loc[param, "pmax"],
                )
                b = ~(upper.loc[param] or lower.loc[param])

                checks.loc[f"Parameter bounds: {param}", :] = (
                    ml.parameters.loc[param, "optimal"],
                    bounds,
                    "_",
                    b,
                )

        df.loc[mlnam, "all_checks_passed"] = checks["check_passed"].all()
        df.loc[mlnam, checks.index] = checks.loc[:, "check_passed"]

        if csv_dir:
            checks.to_csv(f"{csv_dir}/checks_{ml.name}.csv", na_rep="NaN")

    return df


def frontiers_aic_select(
    pstore,
    modelnames: Optional[List[str]] = None,
    oseries: Optional[List[str]] = None,
    full_output: bool = False,
) -> pd.DataFrame:  # pragma: no cover
    """Select the best model structure based on the minimum AIC.

    As proposed by Brakenhoff et al. 2022 [bra_2022]_.

    Parameters
    ----------
    pstore : pastastore.PastaStore
        reference to a PastaStore
    modelnames : list of str
        list of model names (that pass reliability criteria)
    oseries : list of oseries
        list of locations for which to select models, note that this uses all
        models associated with a specific location.
    full_output : bool, optional
        if set to True, returns a DataFrame including all models per location
        and their AIC values

    Returns
    -------
    pandas.DataFrame
        DataFrame with selected best model per location based on the AIC, or a
        DataFrame containing statistics for each of the models per location

    References
    ----------
    .. [bra_2022] Brakenhoff, D.A., Vonk M.A., Collenteur, R.A., van Baar, M.,
    Bakker, M.: Application of Time Series Analysis to Estimate Drawdown From
    Multiple Well Fields. Front. Earth Sci., 14 June 2022
    doi:10.3389/feart.2022.907609
    """
    if modelnames is None and oseries is None:
        modelnames = pstore.model_names
    elif modelnames is None and oseries is not None:
        modelnames = []
        for o in oseries:
            modelnames += pstore.oseries_models[o]
    elif oseries is not None:
        logger.warning(
            "Warning! Both 'modelnames' and 'oseries' provided, using only 'modelnames'"
        )

    # Dataframe of models with corresponding oseries
    df = pstore.get_model_timeseries_names(modelnames, progressbar=False).loc[
        :, ["oseries"]
    ]
    # AIC of models
    aic = pstore.get_statistics(["aic"], modelnames, progressbar=True)
    if full_output:
        # group models per location and obtain the AIC identify model
        # with lowest AIC per location
        collect = []
        gr = df.join(aic).groupby("oseries")
        for _, idf in gr:
            idf.index.name = "modelname"
            idf = (
                idf.sort_values("aic").reset_index().set_index(["oseries", "modelname"])
            )
            idf = idf.rename(columns={"aic": "AIC"})
            idf["dAIC"] = idf["AIC"] - idf["AIC"].min()
            idf = idf.replace(0.0, np.nan)
            collect.append(idf)
        return pd.concat(collect, axis=0)
    else:
        return (
            df.join(aic).groupby("oseries").idxmin().rename(columns={"aic": "min_aic"})
        )
