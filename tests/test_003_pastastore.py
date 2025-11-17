# ruff: noqa: D100 D103
from pathlib import Path

import numpy as np
import pandas as pd
import pastas as ps
import pytest
from numpy import allclose
from packaging.version import parse
from pytest_dependency import depends

import pastastore as pst
from pastastore.util import SeriesUsedByModel


@pytest.mark.dependency
def test_iter_oseries(pstore):
    _ = list(pstore.iter_oseries())


@pytest.mark.dependency
def test_iter_stresses(pstore):
    _ = list(pstore.iter_stresses())


@pytest.mark.dependency
def test_get_tmintmax(pstore):
    ostt = pstore.get_tmin_tmax("oseries")
    assert ostt.at["oseries1", "tmin"] == pd.Timestamp("2010-01-14")
    sttt = pstore.get_tmin_tmax("stresses")
    assert sttt.at["evap2", "tmax"] == pd.Timestamp("2016-11-22")
    ml = pstore.create_model("oseries1")
    ml.solve(report=False)
    pstore.add_model(ml)
    mltt = pstore.get_tmin_tmax("models")
    assert mltt.at["oseries1", "tmax"] == pd.Timestamp("2015-06-28")
    pstore.del_model("oseries1")


@pytest.mark.dependency
def test_search(pstore):
    results = pstore.search("OSER", libname="oseries", case_sensitive=False)
    assert len(results) == 3
    assert len(set(results) - {"oseries1", "oseries2", "oseries3"}) == 0


@pytest.mark.dependency
def test_create_model(pstore):
    _ = pstore.create_model("oseries1")


@pytest.mark.dependency
def test_properties(pstore):
    pstore.add_oseries(pd.Series(dtype=np.float64), "deleteme", validate=False)
    pstore.add_stress(
        pd.Series(dtype=np.float64), "deleteme", kind="useless", validate=False
    )

    _ = pstore.oseries
    _ = pstore.stresses
    _ = pstore.models
    _ = pstore.oseries_models
    _ = pstore.stresses_models

    try:
        assert pstore.n_oseries == pstore.conn.n_oseries
        assert pstore.n_stresses == pstore.conn.n_stresses
    finally:
        pstore.del_oseries("deleteme")
        pstore.del_stress("deleteme")


@pytest.mark.dependency
def test_store_model(request, pstore):
    depends(request, [f"test_create_model[{pstore.type}]"])
    ml = pstore.create_model("oseries1")
    pstore.add_model(ml)


@pytest.mark.dependency
def test_del_oseries_used_by_model(request, pstore):
    depends(request, [f"test_store_model[{pstore.type}]"])
    oseries, ometa = pstore.get_oseries("oseries1", return_metadata=True)
    with pytest.raises(SeriesUsedByModel):
        pstore.del_oseries("oseries1")
    pstore.del_oseries("oseries1", force=True)
    pstore.add_oseries(oseries, "oseries1", metadata=ometa)
    pstore.validator.set_protect_series_in_models(False)
    pstore.del_oseries("oseries1")
    pstore.add_oseries(oseries, "oseries1", metadata=ometa)
    pstore.validator.set_protect_series_in_models(True)


@pytest.mark.dependency
def test_del_stress_used_by_model(request, pstore):
    depends(request, [f"test_store_model[{pstore.type}]"])
    stress, smeta = pstore.get_stress("prec1", return_metadata=True)
    with pytest.raises(SeriesUsedByModel):
        pstore.del_stress("prec1")
    pstore.del_stress("prec1", force=True)
    pstore.add_stress(stress, "prec1", kind="prec", metadata=smeta)
    pstore.validator.set_protect_series_in_models(False)
    pstore.del_stress("prec1")
    pstore.add_stress(stress, "prec1", kind="prec", metadata=smeta)
    pstore.validator.set_protect_series_in_models(True)


@pytest.mark.dependency
def test_model_accessor(request, pstore):
    depends(request, [f"test_store_model[{pstore.type}]"])
    pstore.models.__repr__()
    ml = pstore.models["oseries1"]
    pstore.models["oseries1_2"] = ml
    mnames = [ml.name for ml in pstore.models]
    try:
        assert len(mnames) == 2
        assert mnames[0] in ["oseries1", "oseries1_2"]
        assert mnames[1] in ["oseries1", "oseries1_2"]
    finally:
        pstore.del_models("oseries1_2")


@pytest.mark.dependency
def test_oseries_model_accessor(request, pstore):
    depends(request, [f"test_store_model[{pstore.type}]"])
    pstore.oseries_models.__repr__()
    ml = pstore.models["oseries1"]
    ml_list1 = pstore.oseries_models["oseries1"]
    assert len(ml_list1) == 1

    pstore.models["oseries1_2"] = ml
    ml_list2 = pstore.oseries_models["oseries1"]
    assert len(ml_list2) == 2

    pstore.del_models("oseries1_2")
    ml_list3 = pstore.oseries_models["oseries1"]
    assert len(ml_list3) == 1


@pytest.mark.dependency
def test_store_model_missing_series(request, pstore):
    depends(
        request,
        [
            f"test_create_model[{pstore.type}]",
            f"test_store_model[{pstore.type}]",
        ],
    )
    ml = pstore.create_model("oseries1")
    o = pstore.get_oseries("oseries1")
    meta = pstore.get_metadata("oseries", "oseries1", as_frame=False)
    pstore.del_models("oseries1")
    pstore.del_oseries("oseries1")
    try:
        pstore.add_model(ml)
    except LookupError:
        pstore.add_oseries(o, "oseries1", metadata=meta)
        pstore.add_model(ml)


@pytest.mark.dependency
def test_get_model(request, pstore):
    depends(
        request,
        [
            f"test_create_model[{pstore.type}]",
            f"test_store_model[{pstore.type}]",
            f"test_store_model_missing_series[{pstore.type}]",
        ],
    )
    _ = pstore.get_models("oseries1")


@pytest.mark.dependency
def test_del_model(request, pstore):
    depends(
        request,
        [
            f"test_create_model[{pstore.type}]",
            f"test_store_model[{pstore.type}]",
            f"test_store_model_missing_series[{pstore.type}]",
            f"test_get_model[{pstore.type}]",
        ],
    )
    pstore.del_models("oseries1")


@pytest.mark.dependency
def test_create_models(pstore):
    _ = pstore.create_models_bulk(
        ["oseries1", "oseries2"], store=True, progressbar=False
    )
    _ = pstore.models
    assert pstore.n_models == 2


@pytest.mark.dependency
def test_get_parameters(request, pstore):
    depends(request, [f"test_create_models[{pstore.type}]"])
    p = pstore.get_parameters(progressbar=False, param_value="initial")
    assert p.index.size == 2
    assert p.isna().sum().sum() == 0


@pytest.mark.dependency
def test_get_signatures(request, pstore):
    depends(request, [f"test_create_models[{pstore.type}]"])
    s = pstore.get_signatures(progressbar=False)
    assert s.shape[0] == len(ps.stats.signatures.__all__)


@pytest.mark.dependency
def test_iter_models(request, pstore):
    depends(request, [f"test_create_models[{pstore.type}]"])
    _ = list(pstore.iter_models())


@pytest.mark.dependency
def test_solve_models_and_get_stats(request, pstore):
    depends(request, [f"test_create_models[{pstore.type}]"])
    _ = pstore.solve_models(
        ignore_solve_errors=False, progressbar=False, parallel=False
    )
    stats = pstore.get_statistics(["evp", "aic"], progressbar=False)
    assert stats.index.size == 2


@pytest.mark.dependency
def test_check_models(request, pstore):
    depends(request, [f"test_solve_models_and_get_stats[{pstore.type}]"])
    if parse(ps.__version__) >= parse("1.8.0"):
        _ = pstore.check_models(style_output=True)


@pytest.mark.dependency
def test_solve_models_parallel(request, pstore):
    depends(request, [f"test_create_models[{pstore.type}]"])
    _ = pstore.solve_models(ignore_solve_errors=False, progressbar=False, parallel=True)


def test_apply(request, pstore):
    depends(request, [f"test_solve_models_and_get_stats[{pstore.type}]"])

    def func(ml_name):
        ml = pstore.conn.get_models(ml_name)
        return ml.parameters.loc["recharge_A", "optimal"]

    result = pstore.apply("models", func)
    assert len(result) == 2


@pytest.mark.dependency
def test_save_and_load_model(request, pstore):
    ml = pstore.create_model("oseries1")
    ml.solve()
    evp_ml = ml.stats.evp()
    pstore.add_model(ml, overwrite=True)
    ml2 = pstore.get_models(ml.name)
    evp_ml2 = ml2.stats.evp()
    assert allclose(evp_ml, evp_ml2)
    assert pst.util.compare_models(ml, ml2)


def test_update_ts_settings(request, pstore):
    pstore.validator.set_check_model_series_values(False)

    o = pstore.get_oseries("oseries2")
    ml = ps.Model(o.loc[:"2013"], name="ml_oseries2")

    pnam = pstore.get_nearest_stresses("oseries2", kind="prec").iloc[0]
    p = pstore.get_stresses(pnam)
    enam = pstore.get_nearest_stresses("oseries2", kind="evap").iloc[0]
    e = pstore.get_stresses(enam)
    rm = ps.RechargeModel(p.loc[:"2013"], e.loc[:"2013"])
    ml.add_stressmodel(rm)
    tmax = p.index.intersection(e.index).max()
    ml.solve()

    p2 = pstore.get_stresses("prec1")
    sm = ps.StressModel(p2.loc[:"2013"], ps.Exponential(), "prec")
    ml.add_stressmodel(sm)
    pstore.add_model(ml)

    ml2 = pstore.get_models(ml.name, update_ts_settings=True)

    assert ml2.oseries.settings["tmax"] == o.index[-1]
    assert ml2.stressmodels["recharge"].prec.settings["tmax"] == tmax
    assert ml2.stressmodels["recharge"].evap.settings["tmax"] == tmax
    assert ml2.stressmodels["prec"].stress[0].settings["tmax"] == p2.index[-1]
    pstore.del_models("ml_oseries2")
    pstore.validator.set_check_model_series_values(True)


def test_oseries_distances(pstore):
    _ = pstore.get_nearest_oseries()


def test_repr(pstore):
    pstore.__repr__()


def test_copy_dbase(pstore):
    conn2 = pst.DictConnector("destination")
    pst.util.copy_database(pstore.conn, conn2, overwrite=False, progressbar=True)


def test_to_from_zip(pstore):
    if pstore.type == "arcticdb" and parse(ps.__version__) < parse("1.1.0"):
        pytest.xfail("model datetime objects not supported")
    zipname = f"test_{pstore.type}.zip"
    pstore.to_zip(zipname, progressbar=False, overwrite=True)
    conn = pst.DictConnector("test")
    try:
        store = pst.PastaStore.from_zip(zipname, conn)
        assert not store.oseries.empty
    finally:
        Path(zipname).unlink()


def test_load_pastastore_from_config_file(pstore):
    if pstore.type == "pas" or pstore.type == "arcticdb":
        path = (
            pstore.conn.path
            if pstore.type == "pas"
            else Path(pstore.conn.uri.split("://")[-1]) / pstore.conn.name
        )
        fname = path / f"{pstore.conn.name}.pastastore"
        pstore2 = pst.PastaStore.from_pastastore_config_file(fname)
        assert not pstore2.empty


def test_example_pastastore():
    from pastastore.datasets import example_pastastore

    _ = example_pastastore()


def test_validate_names():
    from pastastore.util import validate_names

    assert validate_names(s="(test)") == "test"
    assert validate_names(d={"(test)": 2})["test"] == 2


def test_meta_with_name(pstore):
    s = pd.Series(
        index=pd.date_range("2020-01-01", periods=10, freq="D"),
        data=np.arange(10),
        dtype=float,
    )
    smeta = {"name": "not_what_i_want"}
    pstore.add_stress(s, "what_i_want", kind="special", metadata=smeta)
    assert "what_i_want" in pstore.stresses.index, "This is not right."
    pstore.del_stress("what_i_want")


@pytest.mark.dependency
def test_models_metadata(request, pstore):
    pstore.create_models_bulk(["oseries1", "oseries2"], store=True, progressbar=False)
    df = pstore.models.metadata
    assert df.index.size == 2
    assert (df["n_stressmodels"] == 1).all()


def test_pstore_validator_settings(pstore):
    _ = pstore.validator.settings
    _ = pstore.conn.validation_settings
