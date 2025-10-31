# ruff: noqa: D100 D103
import warnings

import numpy as np
import pandas as pd
import pastas as ps
import pytest
from pytest_dependency import depends

with warnings.catch_warnings():
    warnings.simplefilter(action="ignore", category=FutureWarning)
    import pastastore as pst

ps.set_log_level("ERROR")


def test_get_library(conn):
    _ = conn._get_library("oseries")


def test_add_get_series(request, conn):
    o1 = pd.Series(
        index=pd.date_range("2000", periods=10, freq="D"),
        data=0.0,
    )
    o1.name = "test_series"
    conn.add_oseries(o1, "test_series", metadata=None)
    o2 = conn.get_oseries("test_series")
    try:
        assert isinstance(o2, pd.Series)
        assert o1.equals(o2)
        assert o1.dtype == o2.dtype
    finally:
        conn.del_oseries("test_series")


def test_add_get_single_value_series(request, conn):
    o1 = pd.Series({pd.Timestamp(2025, 1, 1): 5.0})
    o1.name = "test_single_value_series"
    conn.add_oseries(o1, "test_single_value_series", metadata=None)
    o2 = conn.get_oseries("test_single_value_series")
    try:
        assert isinstance(o2, pd.Series)
        assert o1.equals(o2)
        assert o1.dtype == o2.dtype
    finally:
        conn.del_oseries("test_single_value_series")


def test_add_get_series_wnans(request, conn):
    o1 = pd.Series(
        index=pd.date_range("2000", periods=10, freq="D"),
        data=1.0,
        dtype=np.float64,
    )
    o1.iloc[-3:] = np.nan
    o1.name = "test_series_nans"
    conn.add_oseries(o1, "test_series_nans", metadata=None)
    o2 = conn.get_oseries("test_series_nans")
    try:
        assert isinstance(o2, pd.Series)
        assert o1.equals(o2)
    finally:
        conn.del_oseries("test_series_nans")


def test_add_get_dataframe(request, conn):
    o1 = pd.DataFrame(
        data=1.0,
        columns=["test_df"],
        index=pd.date_range("2000", periods=10, freq="D"),
    )
    o1.index.name = "test_idx"
    conn.add_oseries(o1, "test_df", metadata=None)
    o2 = conn.get_oseries("test_df")
    # PasConnector does not preserve DataFrames after load, so convert if needed.
    if conn.conn_type == "pas":
        o2 = o2.to_frame()
    try:
        assert isinstance(o2, pd.DataFrame)
        assert o1.equals(o2)
    finally:
        conn.del_oseries("test_df")


def test_add_pastas_timeseries(request, conn):
    o1 = pd.DataFrame(
        data=1.0,
        columns=["test_df"],
        index=pd.date_range("2000", periods=10, freq="D"),
    )
    o1.index.name = "test_idx"
    ts = ps.timeseries.TimeSeries(o1, metadata={"x": 100000.0, "y": 400000.0})
    try:
        conn.add_oseries(ts, "test_pastas_ts", metadata=None)
    except TypeError:
        pass


def test_add_series_illegal_filename(request, conn):
    o1 = pd.Series(
        index=pd.date_range("2000", periods=10, freq="D"),
        data=0.0,
    )
    o1.name = r"test\series/illegal_chars"
    conn.add_oseries(o1, o1.name, metadata=None)
    o2 = conn.get_oseries("testseriesillegal_chars")
    try:
        assert isinstance(o2, pd.Series)
        assert o1.equals(o2)
    finally:
        conn.del_oseries("testseriesillegal_chars")

    if conn.conn_type == "pas":
        with pytest.raises(ValueError, match="cannot end with '_meta'"):
            conn.add_oseries(o1, "illegal_meta", metadata=None)


def test_update_series(request, conn):
    o1 = pd.DataFrame(
        data=1.0,
        columns=["test_df"],
        index=pd.date_range("2000", periods=10, freq="D"),
    )
    o1.index.name = "test_idx"
    conn.add_oseries(o1, "test_df", metadata={"x": 100000.0})
    o2 = pd.DataFrame(
        data=2.0,
        columns=["test_df"],
        index=pd.date_range("2000-01-10", periods=2, freq="D"),
    )
    o2.index.name = "test_idx"
    conn.update_oseries(o2, "test_df", metadata={"x": 200000.0, "y": 400000})
    o3 = conn.get_oseries("test_df")
    try:
        assert (o3.iloc[-2:] == 2.0).all().all()
        assert o3.index.size == 11
    finally:
        conn.del_oseries("test_df")


def test_upsert_oseries(request, conn):
    o1 = pd.DataFrame(
        data=1.0,
        columns=["test_df"],
        index=pd.date_range("2000", periods=10, freq="D"),
    )
    o1.index.name = "test_idx"
    conn.upsert_oseries(o1, "test_df", metadata={"x": 100000.0})
    o2 = pd.DataFrame(
        data=2.0,
        columns=["test_df"],
        index=pd.date_range("2000-01-05", periods=10, freq="D"),
    )
    o2.index.name = "test_idx"
    conn.upsert_oseries(o2, "test_df", metadata={"x": 200000.0, "y": 400000})
    o3 = conn.get_oseries("test_df")
    try:
        assert (o3.iloc[-10:] == 2.0).all().all()
        assert o3.index.size == 14
    finally:
        conn.del_oseries("test_df")


def test_upsert_stress(request, conn):
    s1 = pd.DataFrame(
        data=1.0,
        columns=["test_df"],
        index=pd.date_range("2000", periods=10, freq="D"),
    )
    s1.index.name = "test_idx"
    conn.upsert_stress(s1, "test_df", kind="useless", metadata={"x": 100000.0})
    s2 = pd.DataFrame(
        data=2.0,
        columns=["test_df"],
        index=pd.date_range("2000-01-05", periods=10, freq="D"),
    )
    s2.index.name = "test_idx"
    conn.upsert_stress(
        s2,
        "test_df",
        kind="not useless",
        metadata={"x": 200000.0, "y": 400000},
    )
    s3 = conn.get_stresses("test_df")
    try:
        assert (s3.iloc[-10:] == 2.0).all().all()
        assert s3.index.size == 14
        assert conn.stresses.loc["test_df", "kind"] == "not useless"
    finally:
        conn.del_stress("test_df")


def test_update_metadata(request, conn):
    o1 = pd.DataFrame(
        data=1.1,
        columns=["test_df"],
        index=pd.date_range("2000", periods=10, freq="D"),
        dtype=float,
    )
    o1.index.name = "test_idx"
    conn.add_oseries(o1, "test_df", metadata={"x": 100000.0})
    conn.update_metadata("oseries", "test_df", {"x": 200000.0, "y": 400000.0})
    m = conn._get_metadata("oseries", "test_df")
    try:
        assert isinstance(m, dict)
        assert m["x"] == 200000.0
        assert m["y"] == 400000.0
    finally:
        conn.del_oseries("test_df")


@pytest.mark.dependency
def test_add_oseries(conn):
    o = pd.read_csv("./tests/data/obs.csv", index_col=0, parse_dates=True)
    conn.add_oseries(
        o,
        "oseries1",
        metadata={"name": "oseries1", "x": 100000, "y": 400000},
        overwrite=True,
    )


@pytest.mark.dependency
def test_add_stress(conn):
    s = pd.read_csv("./tests/data/rain.csv", index_col=0, parse_dates=True)
    conn.add_stress(
        s,
        "prec",
        kind="prec",
        metadata={"kind": "prec", "x": 100001, "y": 400001},
    )


@pytest.mark.dependency
def test_get_oseries(request, conn):
    depends(request, [f"test_add_oseries[{conn.type}]"])
    _ = conn.get_oseries("oseries1")


@pytest.mark.dependency
def test_get_oseries_and_metadata(request, conn):
    depends(request, [f"test_add_oseries[{conn.type}]"])
    _ = conn.get_oseries("oseries1", return_metadata=True)


@pytest.mark.dependency
def test_get_stress(request, conn):
    depends(request, [f"test_add_stress[{conn.type}]"])
    s = conn.get_stresses("prec")
    s.name = "prec"


@pytest.mark.dependency
def test_get_stress_and_metadata(request, conn):
    depends(request, [f"test_add_stress[{conn.type}]"])
    s, _ = conn.get_stresses("prec", return_metadata=True)
    s.name = "prec"


@pytest.mark.dependency
def test_oseries_prop(request, conn):
    depends(request, [f"test_add_oseries[{conn.type}]"])
    _ = conn.oseries


@pytest.mark.dependency
def test_stresses_prop(request, conn):
    depends(request, [f"test_add_stress[{conn.type}]"])
    _ = conn.stresses


def test_repr(conn):
    conn.__repr__()


@pytest.mark.dependency
def test_del_oseries(request, conn):
    depends(request, [f"test_add_oseries[{conn.type}]"])
    conn.del_oseries("oseries1")


@pytest.mark.dependency
def test_del_stress(request, conn):
    depends(request, [f"test_add_stress[{conn.type}]"])
    conn.del_stress("prec")


@pytest.mark.dependency
def test_empty_library(request, conn):
    s1 = pd.Series(
        index=pd.date_range("2000", periods=10, freq="D"),
        data=1.0,
        dtype=np.float64,
    )
    s1.name = "test_series"
    conn.add_oseries(s1, "test_series", metadata=None)
    conn.empty_library("oseries", prompt=False, progressbar=False)


@pytest.mark.dependency
def test_delete(request, conn):
    # No need to delete dictconnector (in memory)
    if conn.conn_type == "arcticdb":
        pst.util.delete_arcticdb_connector(conn, libraries=["oseries"])
        pst.util.delete_arcticdb_connector(conn)
    elif conn.conn_type == "pas":
        pst.util.delete_pas_connector(conn, libraries=["oseries"])
        pst.util.delete_pas_connector(conn)


def test_new_connector_in_occupied_dir():
    conn1 = pst.PasConnector("my_db", "./tests/data/pas")
    with pytest.raises(
        ValueError, match=f"Directory '{conn1.name}/' in use by another connector type!"
    ):
        pst.ArcticDBConnector("my_db", "lmdb://./tests/data/pas")

    pst.util.delete_pas_connector(conn1)

    conn1 = pst.ArcticDBConnector("my_db", "lmdb://./tests/data/arcticdb")
    with pytest.raises(
        ValueError, match=f"Directory '{conn1.name}/' in use by another connector type!"
    ):
        pst.PasConnector("my_db", "./tests/data/arcticdb")

    pst.util.delete_arcticdb_connector(conn1)
