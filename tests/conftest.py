# ruff: noqa: D100 D103
import importlib
from importlib import metadata

import pandas as pd
import pastas as ps
import pytest

import pastastore as pst

params = ["dict", "pas", "arcticdb"]


@pytest.fixture(scope="module")
def data1():
    d = {
        "oseries1": pd.read_csv("./tests/data/obs.csv", index_col=0, parse_dates=True),
        "oseries1_meta": {"x": 165000, "y": 424000},
        "prec1": pd.read_csv("./tests/data/rain.csv", index_col=0, parse_dates=True),
        "prec1_meta": {"x": 165050, "y": 424050},
        "evap1": pd.read_csv("./tests/data/evap.csv", index_col=0, parse_dates=True),
        "evap1_meta": {"x": 164500, "y": 424000},
    }
    return d


@pytest.fixture(scope="module")
def data2():
    d = {
        "oseries2": pd.read_csv(
            "./tests/data/head_nb1.csv", index_col=0, parse_dates=True
        ),
        "oseries2_meta": {"x": 164000, "y": 423000},
        "prec2": pd.read_csv(
            "./tests/data/rain_nb1.csv", index_col=0, parse_dates=True
        ),
        "prec2_meta": {"x": 164010, "y": 423000},
        "evap2": pd.read_csv(
            "./tests/data/evap_nb1.csv", index_col=0, parse_dates=True
        ),
        "evap2_meta": {"x": 164000, "y": 423030},
    }
    return d


@pytest.fixture(scope="module")
def data3():
    w = pd.read_csv("./tests/data/well_month_end.csv", index_col=0, parse_dates=True)
    w = ps.ts.timestep_weighted_resample(
        w,
        pd.date_range(w.index[0] - pd.offsets.MonthBegin(), w.index[-1], freq="D"),
    ).bfill()

    d = {
        "oseries3": pd.read_csv(
            "./tests/data/gw_obs.csv", index_col=0, parse_dates=True
        ),
        "oseries3_meta": {"x": 165554, "y": 422685},
        "well1": w,
        "well1_meta": {"x": 164691, "y": 423579},
        "well2": w + 10,
        "well2_meta": {"x": 164691 + 200, "y": 423579 + 200},
    }
    return d


def initialize_project(conn, data1, data2, data3):
    pstore = pst.PastaStore(conn, "test_project")

    # dataset 1
    pstore.add_oseries(data1["oseries1"], "oseries1", metadata=data1["oseries1_meta"])
    pstore.add_stress(
        data1["prec1"], "prec1", kind="prec", metadata=data1["prec1_meta"]
    )
    pstore.add_stress(
        data1["evap1"], "evap1", kind="evap", metadata=data1["evap1_meta"]
    )

    # dataset 2
    pstore.add_oseries(data2["oseries2"], "oseries2", metadata=data2["oseries2_meta"])
    pstore.add_stress(
        data2["prec2"], "prec2", kind="prec", metadata=data2["prec2_meta"]
    )
    pstore.add_stress(
        data2["evap2"], "evap2", kind="evap", metadata=data2["evap2_meta"]
    )

    # dataset 3
    pstore.add_oseries(data3["oseries3"], "oseries3", metadata=data3["oseries3_meta"])
    pstore.add_stress(
        data3["well1"], "well1", kind="well", metadata=data3["well1_meta"]
    )
    pstore.add_stress(
        data3["well2"], "well2", kind="well", metadata=data3["well2_meta"]
    )

    return pstore


@pytest.fixture(scope="module", params=params)
def conn(request):
    """Fixture that yields connection object."""
    name = f"test_{request.param}"
    # connect to dbase
    if request.param == "arcticdb":
        uri = "lmdb://./tests/data/arcticdb/"
        conn = pst.ArcticDBConnector(name, uri)
    elif request.param == "dict":
        conn = pst.DictConnector(name)
    elif request.param == "pas":
        conn = pst.PasConnector(name, "./tests/data")
    else:
        raise ValueError("Unrecognized parameter!")
    conn.type = request.param  # added here for defining test dependencies
    return conn


@pytest.fixture(scope="module", params=params)
def pstore(request, data1, data2, data3):
    if request.param == "arcticdb":
        name = "testdb"
        uri = "lmdb://./tests/data/arcticdb/"
        connector = pst.ArcticDBConnector(name, uri)
    elif request.param == "dict":
        name = "testdb"
        connector = pst.DictConnector(name)
    elif request.param == "pas":
        name = "testdb"
        connector = pst.PasConnector(name, "./tests/data/pas")
    else:
        raise ValueError("Unrecognized parameter!")
    pstore = initialize_project(connector, data1, data2, data3)
    pstore.type = request.param  # added here for defining test dependencies
    yield pstore
    pst.util.delete_pastastore(pstore)


_has_pkg_cache = {}


def has_pkg(pkg: str, strict: bool = True) -> bool:
    """
    Determine if the given Python package is installed.

    Parameters
    ----------
    pkg : str
        Name of the package to check.
    strict : bool
        If False, only check if package metadata is available.
        If True, try to import the package (all dependencies must be present).

    Returns
    -------
    bool
        True if the package is installed, otherwise False.

    Notes
    -----
    Originally written by Mike Toews (mwtoews@gmail.com) for FloPy.
    """

    def try_import():
        try:  # import name, e.g. "import shapefile"
            importlib.import_module(pkg)
            return True
        except ModuleNotFoundError:
            return False

    def try_metadata() -> bool:
        try:  # package name, e.g. pyshp
            metadata.distribution(pkg)
            return True
        except metadata.PackageNotFoundError:
            return False

    found = False
    if not strict:
        found = pkg in _has_pkg_cache or try_metadata()
    if not found:
        found = try_import()
    _has_pkg_cache[pkg] = found

    return _has_pkg_cache[pkg]


def requires_pkg(*pkgs):
    missing = {pkg for pkg in pkgs if not has_pkg(pkg, strict=True)}
    return pytest.mark.skipif(
        missing,
        reason=f"missing package{'s' if len(missing) != 1 else ''}: "
        + ", ".join(missing),
    )
