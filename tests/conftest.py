import pandas as pd
import pastastore as pst
import pystore
import pytest

params = ["arctic", "pystore", "dict", "pas"]
# params = ["dict"]


def initialize_project(conn):

    pstore = pst.PastaStore("test_project", conn)

    # oseries 1
    o = pd.read_csv("./tests/data/obs.csv", index_col=0, parse_dates=True)
    pstore.add_oseries(o, "oseries1", metadata={"x": 100000,
                                                "y": 400000})
    # oseries 2
    o = pd.read_csv("./tests/data/head_nb1.csv", index_col=0, parse_dates=True)
    pstore.add_oseries(o, "oseries2", metadata={"x": 100300,
                                                "y": 400400})

    # oseries 3
    o = pd.read_csv("./tests/data/gw_obs.csv", index_col=0, parse_dates=True)
    pstore.add_oseries(o, "oseries3", metadata={"x": 165554,
                                                "y": 422685})

    # prec 1
    s = pd.read_csv("./tests/data/rain.csv", index_col=0, parse_dates=True)
    pstore.add_stress(s, "prec1", kind="prec", metadata={"x": 100000,
                                                         "y": 400000})

    # prec 2
    s = pd.read_csv("./tests/data/rain_nb1.csv", index_col=0, parse_dates=True)
    pstore.add_stress(s, "prec2", kind="prec", metadata={"x": 100300,
                                                         "y": 400400})

    # evap 1
    s = pd.read_csv("./tests/data/evap.csv", index_col=0, parse_dates=True)
    pstore.add_stress(s, "evap1", kind="evap", metadata={"x": 100000,
                                                         "y": 400000})

    # evap 2
    s = pd.read_csv("./tests/data/evap_nb1.csv", index_col=0, parse_dates=True)
    pstore.add_stress(s, "evap2", kind="evap", metadata={"x": 100300,
                                                         "y": 400400})

    # well 1
    s = pd.read_csv("./tests/data/well.csv", index_col=0, parse_dates=True)
    pstore.add_stress(s, "well1", kind="well", metadata={"x": 164691,
                                                         "y": 423579})

    return pstore


@pytest.fixture(scope="module", params=params)
def conn(request):
    """Fixture that yields connection object.
    """
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
