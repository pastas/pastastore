import pytest
import pystore
import arctic
import pandas as pd

import pastas_projects as pp


def initialize_project(conn):

    prj = pp.PastasProject("test_project", conn)

    # oseries 1
    o = pd.read_csv("./tests/data/obs.csv", index_col=0, parse_dates=True)
    o.index.name = "oseries1"
    prj.db.add_oseries(o, "oseries1", metadata={"x": 100000,
                                                "y": 400000})
    # oseries 2
    o = pd.read_csv("./tests/data/head_nb1.csv", index_col=0, parse_dates=True)
    o.index.name = "oseries2"
    prj.db.add_oseries(o, "oseries2", metadata={"x": 100300,
                                                "y": 400400})

    # prec 1
    s = pd.read_csv("./tests/data/rain.csv", index_col=0, parse_dates=True)
    prj.db.add_stress(s, "prec1", kind="prec", metadata={"x": 100000,
                                                         "y": 400000})

    # prec 2
    s = pd.read_csv("./tests/data/rain_nb1.csv", index_col=0, parse_dates=True)
    prj.db.add_stress(s, "prec2", kind="prec", metadata={"x": 100300,
                                                         "y": 400400})

    # evap 1
    s = pd.read_csv("./tests/data/evap.csv", index_col=0, parse_dates=True)
    prj.db.add_stress(s, "evap1", kind="evap", metadata={"x": 100000,
                                                         "y": 400000})

    # evap 2
    s = pd.read_csv("./tests/data/evap_nb1.csv", index_col=0, parse_dates=True)
    prj.db.add_stress(s, "evap2", kind="evap", metadata={"x": 100300,
                                                         "y": 400400})
    return prj


@pytest.fixture(scope="module", params=["arctic", "pystore"])
def pr(request):
    """ fixture that yields connection object
    """
    name = f"test_{request.param}"
    # connect to dbase
    if request.param == "arctic":
        connstr = "mongodb://localhost:27017/"
        pr = pp.ArcticConnector(name, connstr)
    elif request.param == "pystore":
        path = "./tests/data/pystore"
        pr = pp.PystoreConnector(name, path)
    pr.type = request.param  # added here for defining test dependencies
    yield pr


@pytest.fixture(scope="module", params=["arctic", "pystore"])
def prj(request):
    if request.param == "arctic":
        connstr = "mongodb://localhost:27017/"
        name = "test_project"
        arc = arctic.Arctic(connstr)
        if name in [lib.split(".")[0] for lib in arc.list_libraries()]:
            connector = pp.ArcticConnector(name, connstr)
            prj = pp.PastasProject(name, connector)
        else:
            connector = pp.ArcticConnector(name, connstr)
            prj = initialize_project(connector)
    elif request.param == "pystore":
        name = "test_project"
        path = "./tests/data/pystore"
        pystore.set_path(path)
        if name in pystore.list_stores():
            connector = pp.PystoreConnector(name, path)
            prj = pp.PastasProject(name, connector)
        else:
            connector = pp.PystoreConnector(name, path)
            prj = initialize_project(connector)
    prj.type = request.param  # added here for defining test dependencies
    yield prj
