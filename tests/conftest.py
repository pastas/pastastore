import pytest
import pystore
import arctic
import pandas as pd

import pastastore as pst


params = ["arctic", "pystore", "dict", "pas"]
# params = ["pas"]


def initialize_project(conn):

    prj = pst.PastaStore("test_project", conn)

    # oseries 1
    o = pd.read_csv("./tests/data/obs.csv", index_col=0, parse_dates=True)
    prj.add_oseries(o, "oseries1", metadata={"x": 100000,
                                             "y": 400000})
    # oseries 2
    o = pd.read_csv("./tests/data/head_nb1.csv", index_col=0, parse_dates=True)
    prj.add_oseries(o, "oseries2", metadata={"x": 100300,
                                             "y": 400400})

    # oseries 3
    o = pd.read_csv("./tests/data/gw_obs.csv", index_col=0, parse_dates=True)
    prj.add_oseries(o, "oseries3", metadata={"x": 165554,
                                             "y": 422685})

    # prec 1
    s = pd.read_csv("./tests/data/rain.csv", index_col=0, parse_dates=True)
    prj.add_stress(s, "prec1", kind="prec", metadata={"x": 100000,
                                                      "y": 400000})

    # prec 2
    s = pd.read_csv("./tests/data/rain_nb1.csv", index_col=0, parse_dates=True)
    prj.add_stress(s, "prec2", kind="prec", metadata={"x": 100300,
                                                      "y": 400400})

    # evap 1
    s = pd.read_csv("./tests/data/evap.csv", index_col=0, parse_dates=True)
    prj.add_stress(s, "evap1", kind="evap", metadata={"x": 100000,
                                                      "y": 400000})

    # evap 2
    s = pd.read_csv("./tests/data/evap_nb1.csv", index_col=0, parse_dates=True)
    prj.add_stress(s, "evap2", kind="evap", metadata={"x": 100300,
                                                      "y": 400400})

    # well 1
    s = pd.read_csv("./tests/data/well.csv", index_col=0, parse_dates=True)
    prj.add_stress(s, "well1", kind="well", metadata={"x": 164691,
                                                      "y": 423579})

    return prj


@pytest.fixture(scope="module", params=params)
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
    elif request.param == "pas":
        pr = pst.PasConnector(name, "./tests/data/pas")
    else:
        raise ValueError("Unrecognized parameter!")
    pr.type = request.param  # added here for defining test dependencies
    yield pr


@pytest.fixture(scope="module", params=params)
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
    elif request.param == "pas":
        name = "test_project"
        connector = pst.PasConnector(name, "./tests/data/pas")
        prj = initialize_project(connector)
    else:
        raise ValueError("Unrecognized parameter!")
    prj.type = request.param  # added here for defining test dependencies
    yield prj
