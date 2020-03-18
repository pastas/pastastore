import pytest
import pystore
import arctic
import pandas as pd

import pastastore as pst


def initialize_project(conn):

    prj = pst.PastaStore("test_project", conn)

    # oseries 1
    o = pd.read_csv("./tests/data/obs.csv", index_col=0, parse_dates=True)
    o.index.name = "oseries1"
    prj.conn.add_oseries(o, "oseries1", metadata={"x": 100000,
                                                "y": 400000})
    # oseries 2
    o = pd.read_csv("./tests/data/head_nb1.csv", index_col=0, parse_dates=True)
    o.index.name = "oseries2"
    prj.conn.add_oseries(o, "oseries2", metadata={"x": 100300,
                                                "y": 400400})

    # prec 1
    s = pd.read_csv("./tests/data/rain.csv", index_col=0, parse_dates=True)
    prj.conn.add_stress(s, "prec1", kind="prec", metadata={"x": 100000,
                                                         "y": 400000})

    # prec 2
    s = pd.read_csv("./tests/data/rain_nb1.csv", index_col=0, parse_dates=True)
    prj.conn.add_stress(s, "prec2", kind="prec", metadata={"x": 100300,
                                                         "y": 400400})

    # evap 1
    s = pd.read_csv("./tests/data/evap.csv", index_col=0, parse_dates=True)
    prj.conn.add_stress(s, "evap1", kind="evap", metadata={"x": 100000,
                                                         "y": 400000})

    # evap 2
    s = pd.read_csv("./tests/data/evap_nb1.csv", index_col=0, parse_dates=True)
    prj.conn.add_stress(s, "evap2", kind="evap", metadata={"x": 100300,
                                                         "y": 400400})
    return prj


@pytest.fixture(scope="module", params=["arctic", "pystore", "dict"])
def pr(request):
    """Fixture that yields connection object.
    """
    name = f"test_{request.param}"
    # connect to dbase
    if request.param == "arctic":
        connstr = "mongodb://localhost:27017/"
        pr = pst.ArcticConnector(name, connstr)
    elif request.param == "pystore":
        path = "./tests/data/pystore"
        pr = pst.PystoreConnector(name, path)
    elif request.param == "dict":
        pr = pst.DictConnector(name)
    pr.type = request.param  # added here for defining test dependencies
    yield pr


@pytest.fixture(scope="module", params=["arctic", "pystore", "dict"])
def prj(request):
    if request.param == "arctic":
        connstr = "mongodb://localhost:27017/"
        name = "test_project"
        arc = arctic.Arctic(connstr)
        if name in [lib.split(".")[0] for lib in arc.list_libraries()]:
            connector = pst.ArcticConnector(name, connstr)
            prj = pst.PastaStore(name, connector)
        else:
            connector = pst.ArcticConnector(name, connstr)
            prj = initialize_project(connector)
    elif request.param == "pystore":
        name = "test_project"
        path = "./tests/data/pystore"
        pystore.set_path(path)
        if name in pystore.list_stores():
            connector = pst.PystoreConnector(name, path)
            prj = pst.PastaStore(name, connector)
        else:
            connector = pst.PystoreConnector(name, path)
            prj = initialize_project(connector)
    elif request.param == "dict":
        name = "test_project"
        connector = pst.DictConnector(name)
        prj = initialize_project(connector)
    prj.type = request.param  # added here for defining test dependencies
    yield prj
