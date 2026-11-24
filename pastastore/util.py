"""Useful utilities for pastastore."""

import json
import logging
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.lib._iotools import NameValidator
from pandas.io.formats.style import Styler
from pandas.testing import assert_series_equal
from pastas import Model
from pastas.io.pas import PastasEncoder

from pastastore._tqdm import tqdm
from pastastore.styling import boolean_row_styler
from pastastore.typing import TimeSeriesLibs
from pastastore.version import PASTAS_GEQ_200

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
        names: list[str] | str | None = None,
        squeeze: bool = True,
        progressbar: bool = False,
    ):
        """Write stored series to JSON.

        Parameters
        ----------
        libname : str
            library name
        names : list[str] | str | None, optional
            names of series, by default None
        squeeze : bool, optional
            return single entry as json string instead
            of list, by default True
        progressbar : bool, optional
            show progressbar, by default False

        Returns
        -------
        files : list | str
            list of series converted to JSON string or single string
            if single entry is returned and squeeze is True
        """
        names = self.pstore.parse_names(names, libname=libname)
        files = []
        for n in tqdm(names, desc=libname) if progressbar else names:
            s = self.pstore.conn._get_series(libname, n, progressbar=False)
            try:
                if type(s) is pd.Series:
                    s = s.to_frame()
                if type(s) is pd.DataFrame:
                    sjson = s.to_json(orient="columns", date_format="iso", indent=2)
                else:
                    # workaround for subclasses of DataFrame that override to_json,
                    # looking at you hydropandas...
                    sjson = pd.DataFrame(s).to_json(
                        orient="columns", date_format="iso", indent=2
                    )
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
        names: list[str] | str | None = None,
        squeeze: bool = True,
        progressbar: bool = False,
    ):
        """Write metadata from stored series to JSON.

        Parameters
        ----------
        libname : str
            library containing series
        names : list[str] | str | None, optional
            names to parse, by default None
        squeeze : bool, optional
            return single entry as json string instead of list, by default True
        progressbar : bool, optional
            show progressbar, by default False

        Returns
        -------
        files : list | str
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
        names: list[str] | str | None = None,
        progressbar: bool = True,
    ):
        """Write DataFrame or Series to zipfile (internal method).

        Parameters
        ----------
        archive : zipfile.ZipFile
            reference to an archive to write data to
        libname : str
            name of the library to write to zipfile
        names : str | list[str], optional
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
        names : str | list[str], optional
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

    def __init__(self, *args, colors: dict[str, str] | None = None, **kwargs) -> None:
        """Initialize the formatter with specified format strings."""
        super().__init__(*args, **kwargs)

        self.colors = colors if colors else {}

    def format(self, record) -> str:
        """Format the specified record as text."""
        record.color = self.colors.get(record.levelname, "")
        record.reset = "\x1b[0m"

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
            "DEBUG": "\x1b[36m",
            "INFO": "\x1b[32m",
            "WARNING": "\x1b[33m",
            "ERROR": "\x1b[31m",
            "CRITICAL": "\x1b[31m" + "\x1b[47m" + "\x1b[1m",
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
    s = pd.read_json(
        fjson,
        orient="columns",
        precise_float=True,
        dtype=False,
    )
    # ensure column is numeric
    for col in s.columns:
        if s[col].dtype == "object":
            s[col] = pd.to_numeric(s[col], errors="coerce")
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
    uri: str | None = None,
    name: str | None = None,
    libraries: list[str] | None = None,
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
    libraries : list[str] | None, optional
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


def delete_dict_connector(conn, libraries: list[str] | None = None) -> None:
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


def delete_pas_connector(conn, libraries: list[str] | None = None) -> None:
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


def delete_pastastore(pstore, libraries: list[str] | None = None) -> None:
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
    libraries : list[str] | None, optional
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
    s: str | None = None,
    d: dict | None = None,
    replace_space: str | None = "_",
    deletechars: str | None = None,
    **kwargs,
) -> str | dict:
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
    stats: list[str] | None = None,
    detailed_comparison: bool = False,
    style_output: bool = False,
) -> pd.DataFrame | Styler:
    """Compare two Pastas models.

    Parameters
    ----------
    ml1 : pastas.Model
        first model to compare
    ml2 : pastas.Model
        second model to compare
    stats : list[str], optional
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
    bool or pd.DataFrame or pd.Styler
        returns True if models are equivalent when detailed_comparison=True
        else returns DataFrame containing comparison details.
    """
    df = pd.DataFrame(columns=pd.Index(["model 0", "model 1"]))
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

            if PASTAS_GEQ_200:
                stresses = list(sm.stresses)
            elif type(sm).__name__ == "RechargeModel":
                temp = [sm.temp] if sm.temp is not None else []
                stresses = [sm.prec, sm.evap] + temp
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
            df.loc[f"param: {p} (pmin)", f"model {i}"] = ml.parameters.loc[p, "pmin"]
            df.loc[f"param: {p} (pmax)", f"model {i}"] = ml.parameters.loc[p, "pmax"]
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
    libraries: list[str] | None = None,
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
    libraries : list[str] | None, optional
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
