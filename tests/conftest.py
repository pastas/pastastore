# ruff: noqa: D100 D103
import importlib
from importlib import metadata

import pandas as pd
import pastas as ps
import pytest

import pastastore as pst

params = ["dict", "pas", "arcticdb"]


def initialize_project(conn):
    pstore = pst.PastaStore(conn, "test_project")

    # oseries 1
    o = pd.read_csv("./tests/data/obs.csv", index_col=0, parse_dates=True)
    pstore.add_oseries(o, "oseries1", metadata={"x": 165000, "y": 424000})

    # oseries 2
    o = pd.read_csv("./tests/data/head_nb1.csv", index_col=0, parse_dates=True)
    pstore.add_oseries(o, "oseries2", metadata={"x": 164000, "y": 423000})

    # oseries 3
    o = pd.read_csv("./tests/data/gw_obs.csv", index_col=0, parse_dates=True)
    pstore.add_oseries(o, "oseries3", metadata={"x": 165554, "y": 422685})

    # prec 1
    s = pd.read_csv("./tests/data/rain.csv", index_col=0, parse_dates=True)
    pstore.add_stress(s, "prec1", kind="prec", metadata={"x": 165050, "y": 424050})

    # prec 2
    s = pd.read_csv("./tests/data/rain_nb1.csv", index_col=0, parse_dates=True)
    pstore.add_stress(s, "prec2", kind="prec", metadata={"x": 164010, "y": 423000})

    # evap 1
    s = pd.read_csv("./tests/data/evap.csv", index_col=0, parse_dates=True)
    pstore.add_stress(s, "evap1", kind="evap", metadata={"x": 164500, "y": 424000})

    # evap 2
    s = pd.read_csv("./tests/data/evap_nb1.csv", index_col=0, parse_dates=True)
    pstore.add_stress(s, "evap2", kind="evap", metadata={"x": 164000, "y": 423030})

    # well 1
    s = pd.read_csv("./tests/data/well_month_end.csv", index_col=0, parse_dates=True)
    try:
        s = ps.ts.timestep_weighted_resample(
            s,
            pd.date_range(s.index[0] - pd.offsets.MonthBegin(), s.index[-1], freq="D"),
        ).bfill()
    except AttributeError:
        # pastas<=0.22.0
        pass
    pstore.add_stress(s, "well1", kind="well", metadata={"x": 164691, "y": 423579})
    # add second well
    pstore.add_stress(
        s + 10, "well2", kind="well", metadata={"x": 164691 + 200, "y": 423579_200}
    )

    return pstore


@pytest.fixture(scope="module", params=params)
def conn(request):
    """Fixture that yields connection object."""
    name = f"test_{request.param}"
    # connect to dbase
    if request.param == "arcticdb":
        uri = "lmdb://./arctic_db/"
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
def pstore(request):
    if request.param == "arcticdb":
        name = "test_project"
        uri = "lmdb://./arctic_db/"
        connector = pst.ArcticDBConnector(name, uri)
    elif request.param == "dict":
        name = "test_project"
        connector = pst.DictConnector(name)
    elif request.param == "pas":
        name = "test_project"
        connector = pst.PasConnector(name, "./tests/data/pas")
    else:
        raise ValueError("Unrecognized parameter!")
    pstore = initialize_project(connector)
    pstore.type = request.param  # added here for defining test dependencies
    yield pstore
    pst.util.delete_pastastore(pstore)


def delete_arcticdb_test_db():
    connstr = "lmdb://./arctic_db/"
    name = "test_project"
    connector = pst.ArcticDBConnector(name, connstr)
    pst.util.delete_arcticdb_connector(connector)
    print("ArcticDBConnector 'test_project' deleted.")


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
