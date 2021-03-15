import warnings

import pandas as pd
import pytest
from pytest_dependency import depends

import pastas as ps

with warnings.catch_warnings():
    warnings.simplefilter(action="ignore", category=FutureWarning)
    import pastastore as pst

ps.set_log_level("ERROR")


def test_get_library(pr):
    olib = pr.get_library("oseries")
    return olib


def test_add_get_series(request, pr):
    o1 = pd.Series(index=pd.date_range("2000", periods=10, freq="D"), data=1.0)
    o1.name = "test_series"
    pr.add_oseries(o1, "test_series", metadata=None)
    o2 = pr.get_oseries("test_series")
    # PasConnector has no logic for preserving Series
    if pr.conn_type == "pas":
        o2 = o2.squeeze()
    try:
        assert isinstance(o2, pd.Series)
        assert (o1 == o2).all()
    finally:
        pr.del_oseries("test_series")
    return


def test_add_get_dataframe(request, pr):
    o1 = pd.DataFrame(data=1.0, columns=["test_df"],
                      index=pd.date_range("2000", periods=10, freq="D"))
    o1.index.name = "test_idx"
    pr.add_oseries(o1, "test_df", metadata=None)
    o2 = pr.get_oseries("test_df")
    try:
        assert isinstance(o2, pd.DataFrame)
        assert (o1 == o2).all().all()
    finally:
        pr.del_oseries("test_df")
    return


@pytest.mark.dependency()
def test_add_oseries(pr):
    o = pd.read_csv("./tests/data/obs.csv", index_col=0, parse_dates=True)
    pr.add_oseries(o, "oseries1", metadata={"name": "oseries1",
                                            "x": 100000,
                                            "y": 400000},
                   overwrite=True)
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
        pst.util.delete_arctic_connector(
            pr.connstr, pr.name, libraries=["oseries"])
        pst.util.delete_arctic_connector(pr.connstr, pr.name)
    elif pr.conn_type == "pystore":
        pst.util.delete_pystore_connector(
            pr.path, pr.name, libraries=["oseries"])
        pst.util.delete_pystore_connector(pr.path, pr.name)
    return
