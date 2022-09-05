import os
import warnings

import numpy as np
import pandas as pd
import pastas as ps
import pytest
from numpy import allclose
from pytest_dependency import depends

with warnings.catch_warnings():
    warnings.simplefilter(action="ignore", category=FutureWarning)
    import pastastore as pst


@pytest.mark.dependency()
def test_iter_oseries(pstore):
    _ = list(pstore.iter_oseries())
    return


@pytest.mark.dependency()
def test_iter_stresses(pstore):
    _ = list(pstore.iter_stresses())
    return


@pytest.mark.dependency()
def test_get_tmintmax(pstore):
    _ = pstore.get_tmin_tmax("oseries")
    _ = pstore.get_tmin_tmax("stresses")
    return


@pytest.mark.dependency()
def test_search(pstore):
    results = pstore.search("oseries", "OSER", case_sensitive=False)
    assert len(results) == 3
    assert (len(set(results) - {"oseries1", "oseries2", "oseries3"}) == 0)
    return


@pytest.mark.dependency()
def test_create_model(pstore):
    ml = pstore.create_model("oseries1")
    return ml


@pytest.mark.dependency()
def test_properties(pstore):

    pstore.add_oseries(pd.Series(dtype=np.float64), "deleteme")
    pstore.add_stress(pd.Series(dtype=np.float64), "deleteme", kind="useless")

    _ = pstore.oseries
    _ = pstore.stresses
    _ = pstore.models

    try:
        assert pstore.n_oseries == pstore.conn.n_oseries
        assert pstore.n_stresses == pstore.conn.n_stresses
    finally:
        pstore.del_oseries("deleteme")
        pstore.del_stress("deleteme")

    return


@pytest.mark.dependency()
def test_store_model(request, pstore):
    depends(request, [f"test_create_model[{pstore.type}]"])
    ml = test_create_model(pstore)
    pstore.conn.add_model(ml)
    return


@pytest.mark.dependency()
def test_model_accessor(request, pstore):
    depends(request, [f"test_store_model[{pstore.type}]"])
    # repr
    pstore.models.__repr__()
    # getter
    ml = pstore.models["oseries1"]
    # setter
    pstore.models["oseries1_2"] = ml
    # iter
    mnames = [ml.name for ml in pstore.models]
    try:
        assert len(mnames) == 2
        assert mnames[0] in ["oseries1", "oseries1_2"]
        assert mnames[1] in ["oseries1", "oseries1_2"]
    finally:
        pstore.del_models("oseries1_2")
    return


@pytest.mark.dependency()
def test_oseries_model_accessor(request, pstore):
    depends(request, [f"test_store_model[{pstore.type}]"])
    # repr
    pstore.oseries_models.__repr__()
    # get model names
    ml = pstore.models["oseries1"]
    ml_list1 = pstore.oseries_models["oseries1"]
    assert len(ml_list1) == 1

    # add model
    pstore.models["oseries1_2"] = ml
    ml_list2 = pstore.oseries_models["oseries1"]
    assert len(ml_list2) == 2

    # delete model
    pstore.del_models("oseries1_2")
    ml_list3 = pstore.oseries_models["oseries1"]
    assert len(ml_list3) == 1
    return


@pytest.mark.dependency()
def test_store_model_missing_series(request, pstore):
    depends(request, [f"test_create_model[{pstore.type}]",
                      f"test_store_model[{pstore.type}]"])
    ml = test_create_model(pstore)
    o = pstore.get_oseries("oseries1")
    meta = pstore.get_metadata("oseries", "oseries1", as_frame=False)
    pstore.del_models("oseries1")
    pstore.del_oseries("oseries1")
    try:
        pstore.add_model(ml)
    except LookupError:
        pstore.add_oseries(o, "oseries1", metadata=meta)
        pstore.add_model(ml)
        return


@pytest.mark.dependency()
def test_get_model(request, pstore):
    depends(request, [f"test_create_model[{pstore.type}]",
                      f"test_store_model[{pstore.type}]",
                      f"test_store_model_missing_series[{pstore.type}]"])
    ml = pstore.conn.get_models("oseries1")
    return ml


@pytest.mark.dependency()
def test_del_model(request, pstore):
    depends(request, [f"test_create_model[{pstore.type}]",
                      f"test_store_model[{pstore.type}]",
                      f"test_store_model_missing_series[{pstore.type}]",
                      f"test_get_model[{pstore.type}]"])
    pstore.conn.del_models("oseries1")
    return


@pytest.mark.dependency()
def test_create_models(pstore):
    mls = pstore.create_models_bulk(["oseries1", "oseries2"], store=True,
                                    progressbar=False)
    _ = pstore.conn.models
    return mls


@pytest.mark.dependency()
def test_get_parameters(request, pstore):
    depends(request, [f"test_create_models[{pstore.type}]"])
    p = pstore.get_parameters(progressbar=False, param_value="initial")
    assert p.index.size == 2
    assert p.isna().sum().sum() == 0
    return p


@pytest.mark.dependency()
def test_iter_models(request, pstore):
    depends(request, [f"test_create_models[{pstore.type}]"])
    _ = list(pstore.iter_models())
    return


@pytest.mark.dependency()
def test_solve_models_and_get_stats(request, pstore):
    depends(request, [f"test_create_models[{pstore.type}]"])
    mls = pstore.solve_models(ignore_solve_errors=False,
                              progressbar=False,
                              store_result=True)
    stats = pstore.get_statistics(["evp", "aic"], progressbar=False)
    assert stats.index.size == 2
    return mls, stats


@pytest.mark.dependency()
def test_save_and_load_model(request, pstore):
    ml = pstore.create_model("oseries2")
    sm = ps.StressModel(pstore.get_stresses('well1'), ps.Gamma,
                        name='well1', settings="well")
    ml.add_stressmodel(sm)
    ml.solve(tmin='1993-1-1')
    evp_ml = ml.stats.evp()
    pstore.add_model(ml, overwrite=True)
    ml2 = pstore.get_models(ml.name)
    evp_ml2 = ml2.stats.evp()
    assert allclose(evp_ml, evp_ml2)
    assert pst.util.compare_models(ml, ml2)
    return ml, ml2


def test_update_ts_settings(request, pstore):
    pstore.set_check_model_series_values(False)

    o = pstore.get_oseries("oseries2")
    ml = ps.Model(o.loc[:"2013"], name="ml_oseries2")

    pnam = pstore.get_nearest_stresses("oseries2", kind="prec").iloc[0]
    p = pstore.get_stresses(pnam)
    enam = pstore.get_nearest_stresses("oseries2", kind="evap").iloc[0]
    e = pstore.get_stresses(enam)
    rm = ps.RechargeModel(p.loc[:"2013"], e.loc[:"2013"])
    ml.add_stressmodel(rm)
    tmax = p.index.intersection(e.index).max()

    p2 = pstore.get_stresses("prec1")
    sm = ps.StressModel(p2.loc[:"2013"], ps.Exponential, "prec")
    ml.add_stressmodel(sm)

    pstore.add_model(ml)

    ml2 = pstore.get_models(ml.name, update_ts_settings=True)

    try:
        assert ml2.oseries.settings["tmax"] == o.index[-1]
        assert ml2.stressmodels["recharge"].prec.settings["tmax"] == tmax
        assert ml2.stressmodels["recharge"].evap.settings["tmax"] == tmax
        assert ml2.stressmodels["prec"].stress[0].settings["tmax"] == \
            p2.index[-1]
    except AssertionError:
        pstore.del_models("ml_oseries2")
        pstore.set_check_model_series_values(True)
        raise
    return


# @pytest.mark.dependency()
# def test_model_results(request, pstore):
#     depends(request, [f"test_create_models[{pstore.type}]",
#                       f"test_solve_models[{pstore.type}]"])
#     pstore.model_results(["oseries1", "oseries2"], progressbar=False)
#     return


def test_oseries_distances(pstore):
    _ = pstore.get_nearest_oseries()
    return


def test_repr(pstore):
    return pstore.__repr__()


def test_copy_dbase(pstore):
    conn2 = pst.DictConnector("destination")
    pst.util.copy_database(pstore.conn, conn2, overwrite=False,
                           progressbar=True)
    return


def test_to_from_zip(pstore):
    zipname = f"test_{pstore.type}.zip"
    pstore.to_zip(zipname, progressbar=False, overwrite=True)
    conn = pst.DictConnector("test")
    try:
        store = pst.PastaStore.from_zip(zipname, conn)
        assert not store.oseries.empty
    finally:
        os.remove(zipname)
    return store


def test_example_pastastore():
    from pastastore.datasets import example_pastastore
    _ = example_pastastore()
    return
