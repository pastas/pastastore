import os
import warnings

import pandas as pd
import pytest
from pytest_dependency import depends

import pastas as ps

with warnings.catch_warnings():
    warnings.simplefilter(action="ignore", category=FutureWarning)
    from pastas_projects import util

ps.set_log_level("ERROR")


def test_get_library(pr):
    olib = pr.get_library("oseries")
    return olib


@pytest.mark.dependency()
def test_add_oseries(pr):
    o = pd.read_csv("./tests/data/obs.csv", index_col=0, parse_dates=True)
    o.index.name = "oseries1"
    pr.add_oseries(o, "oseries1", metadata={"name": "oseries1",
                                            "x": 100000,
                                            "y": 400000})
    return


@pytest.mark.dependency()
def test_add_stress(pr):
    s = pd.read_csv("./tests/data/rain.csv", index_col=0, parse_dates=True)
    pr.add_stress(s, "prec", kind="prec", metadata={"kind": "prec",
                                                    "x": 100001,
                                                    "y": 400001})
    return


@pytest.mark.dependency()
def test_get_oseries(request, pr):
    depends(request, [f"test_add_oseries[{pr.type}]"])
    o = pr.get_oseries("oseries1")
    o.index.name = "oseries1"
    return o


@pytest.mark.dependency()
def test_get_stress(request, pr):
    depends(request, [f"test_add_stress[{pr.type}]"])
    s = pr.get_stresses('prec')
    s.name = 'prec'
    return s


@pytest.mark.dependency()
def test_oseries_prop(request, pr):
    depends(request, [f"test_add_oseries[{pr.type}]"])
    return pr.oseries


@pytest.mark.dependency()
def test_stresses_prop(request, pr):
    depends(request, [f"test_add_stress[{pr.type}]"])
    return pr.stresses


def test_repr(pr):
    return pr.__repr__()


@pytest.mark.dependency()
def test_del_oseries(request, pr):
    depends(request, [f"test_add_oseries[{pr.type}]"])
    pr.del_oseries("oseries1")
    return


@pytest.mark.dependency()
def test_del_stress(request, pr):
    depends(request, [f"test_add_stress[{pr.type}]"])
    pr.del_stress("prec")
    return


@pytest.mark.dependency()
def test_delete(request, pr):
    if pr.conn_type == "arctic":
        util.delete_arctic(pr.connstr, pr.name, libraries=["oseries"])
        util.delete_arctic(pr.connstr, pr.name)
    elif pr.conn_type == "pystore":
        util.delete_pystore(pr.path, pr.name, libraries=["oseries"])
        util.delete_pystore(pr.path, pr.name)
    return
