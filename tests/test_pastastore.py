import os
import warnings
import pytest
from pytest_dependency import depends

with warnings.catch_warnings():
    warnings.simplefilter(action="ignore", category=FutureWarning)
    import pastastore as pst


@pytest.mark.dependency()
def test_create_model(prj):
    ml = prj.create_model("oseries1")
    return ml


@pytest.mark.dependency()
def test_properties(prj):
    _ = prj.oseries
    _ = prj.stresses
    _ = prj.models
    return


@pytest.mark.dependency()
def test_store_model(request, prj):
    depends(request, [f"test_create_model[{prj.type}]"])
    ml = test_create_model(prj)
    prj.conn.add_model(ml)
    return


@pytest.mark.dependency()
def test_store_model_missing_series(request, prj):
    depends(request, [f"test_create_model[{prj.type}]",
                      f"test_store_model[{prj.type}]"])
    ml = test_create_model(prj)
    prj.del_models("oseries1")
    prj.del_oseries("oseries1")
    prj.del_stress("prec1")
    prj.del_stress("evap1")
    prj.add_model(ml)
    return


@pytest.mark.dependency()
def test_get_model(request, prj):
    depends(request, [f"test_create_model[{prj.type}]",
                      f"test_store_model[{prj.type}]",
                      f"test_store_model_missing_series[{prj.type}]"])
    ml = prj.conn.get_models("oseries1")
    return ml


@pytest.mark.dependency()
def test_del_model(request, prj):
    depends(request, [f"test_create_model[{prj.type}]",
                      f"test_store_model[{prj.type}]",
                      f"test_store_model_missing_series[{prj.type}]",
                      f"test_get_model[{prj.type}]"])
    prj.conn.del_models("oseries1")
    return


@pytest.mark.dependency()
def test_create_models(prj):
    mls = prj.create_models(["oseries1", "oseries2"], store=True,
                            progressbar=False)
    _ = prj.conn.models
    return mls


@pytest.mark.dependency()
def test_get_parameters(request, prj):
    depends(request, [f"test_create_models[{prj.type}]"])
    p = prj.get_parameters(progressbar=False, param_value="initial")
    assert p.index.size == 2
    assert p.isna().sum().sum() == 0
    return p


@pytest.mark.dependency()
def test_solve_models_and_get_stats(request, prj):
    depends(request, [f"test_create_models[{prj.type}]"])
    mls = prj.solve_models(["oseries1", "oseries2"],
                           ignore_solve_errors=False,
                           progressbar=False,
                           store_result=True)
    stats = prj.get_statistics(["evp", "aic"], progressbar=False)
    assert stats.index.size == 2
    return mls, stats

# @pytest.mark.dependency()
# def test_model_results(request, prj):
#     depends(request, [f"test_create_models[{prj.type}]",
#                       f"test_solve_models[{prj.type}]"])
#     prj.model_results(["oseries1", "oseries2"], progressbar=False)
#     return


def test_oseries_distances(prj):
    _ = prj.get_nearest_oseries()
    return


def test_repr(prj):
    return prj.__repr__()


def test_to_from_zip(prj):
    zipname = f"test_{prj.type}.zip"
    prj.to_zip(zipname, progressbar=False)
    conn = pst.DictConnector("test")
    try:
        store = pst.PastaStore.from_zip(zipname, conn)
        assert not store.oseries.empty
    finally:
        os.remove(zipname)
    return store


def test_delete_db(prj):
    if prj.conn.conn_type == "arctic":
        pst.util.delete_arctic(prj.conn.connstr, prj.conn.name)
    elif prj.conn.conn_type == "pystore":
        pst.util.delete_pystore(prj.conn.path, prj.conn.name)
    return
