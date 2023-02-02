import importlib

import pandas as pd
import pastas as ps
import pkg_resources
import pytest

import pastastore as pst

params = ["dict", "pas"]  # "arctic" and "pystore" removed for CI, can be tested locally


def initialize_project(conn):
    pstore = pst.PastaStore("test_project", conn)

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
    s = pd.read_csv("./tests/data/well.csv", index_col=0, parse_dates=True)
    try:
        s = ps.ts.timestep_weighted_resample(
            s, pd.date_range(s.index[0], s.index[-1], freq="D")
        )
    except AttributeError:
        # pastas<=0.22.0
        pass
    pstore.add_stress(s, "well1", kind="well", metadata={"x": 164691, "y": 423579})

    return pstore


@pytest.fixture(scope="module", params=params)
def conn(request):
    """Fixture that yields connection object."""
    name = f"test_{request.param}"
    # connect to dbase
    if request.param == "arctic":
        connstr = "mongodb://localhost:27017/"
        conn = pst.ArcticConnector(name, connstr)
    elif request.param == "pystore":
        path = "./tests/data/pystore"
        conn = pst.PystoreConnector(name, path)
    elif request.param == "dict":
        conn = pst.DictConnector(name)
    elif request.param == "pas":
        conn = pst.PasConnector(name, "./tests/data/pas")
    else:
        raise ValueError("Unrecognized parameter!")
    conn.type = request.param  # added here for defining test dependencies
    yield conn


@pytest.fixture(scope="module", params=params)
def pstore(request):
    if request.param == "arctic":
        connstr = "mongodb://localhost:27017/"
        name = "test_project"
        connector = pst.ArcticConnector(name, connstr)
    elif request.param == "pystore":
        name = "test_project"
        path = "./tests/data/pystore"
        import pystore

        pystore.set_path(path)
        connector = pst.PystoreConnector(name, path)
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


def delete_arctic_test_db():
    connstr = "mongodb://localhost:27017/"
    name = "test_project"
    connector = pst.ArcticConnector(name, connstr)
    pst.util.delete_arctic_connector(connector)
    print("ArcticConnector 'test_project' deleted.")


_has_pkg_cache = {}


def has_pkg(pkg):
    """
    Determines if the given Python package is installed.

    Originally written by Mike Toews (mwtoews@gmail.com) for FloPy.
    """
    if pkg not in _has_pkg_cache:
        # for some dependencies, package name and import name are different
        # (e.g. pyshp/shapefile, mfpymake/pymake, python-dateutil/dateutil)
        # pkg_resources expects package name, importlib expects import name
        try:
            _has_pkg_cache[pkg] = bool(importlib.import_module(pkg))
        except (ImportError, ModuleNotFoundError):
            try:
                _has_pkg_cache[pkg] = bool(pkg_resources.get_distribution(pkg))
            except pkg_resources.DistributionNotFound:
                _has_pkg_cache[pkg] = False

    return _has_pkg_cache[pkg]


def requires_pkg(*pkgs):
    missing = {pkg for pkg in pkgs if not has_pkg(pkg)}
    return pytest.mark.skipif(
        missing,
        reason=f"missing package{'s' if len(missing) != 1 else ''}: "
        + ", ".join(missing),
    )
