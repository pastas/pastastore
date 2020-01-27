import os
import warnings
import arctic
import pandas as pd
import pytest
from pytest_dependency import depends

import pastas as ps

with warnings.catch_warnings():
    warnings.simplefilter(action="ignore", category=FutureWarning)
    from pastas_projects import PastasProject, ArcticConnector, util


@pytest.mark.dependency()
def test_create_model(prj):
    ml = prj.create_model("oseries1")
    return ml


@pytest.mark.dependency()
def test_store_model(request, prj):
    depends(request, [f"test_create_model[{prj.type}]"])
    ml = test_create_model(prj)
    prj.db.add_model(ml)
    return


@pytest.mark.dependency()
def test_get_model(request, prj):
    depends(request, [f"test_create_model[{prj.type}]",
                      f"test_store_model[{prj.type}]"])
    ml = prj.db.get_models("oseries1")
    return ml


@pytest.mark.dependency()
def test_del_model(request, prj):
    depends(request, [f"test_create_model[{prj.type}]",
                      f"test_store_model[{prj.type}]",
                      f"test_get_model[{prj.type}]"])
    prj.db.del_models("oseries1")
    return


@pytest.mark.dependency()
def test_create_models(prj):
    mls = prj.create_models(["oseries1", "oseries2"], store=True,
                            progressbar=False)
    return mls


@pytest.mark.dependency()
def test_solve_models(request, prj):
    depends(request, [f"test_create_models[{prj.type}]"])
    mls = prj.solve_models(["oseries1", "oseries2"],
                           ignore_solve_errors=False,
                           progressbar=False,
                           store_result=True)
    return mls


# @pytest.mark.dependency()
# def test_model_results(request, prj):
#     depends(request, [f"test_create_models[{prj.type}]",
#                       f"test_solve_models[{prj.type}]"])
#     prj.model_results(["oseries1", "oseries2"], progressbar=False)
#     return


def test_oseries_distances(prj):
    nearest = prj.get_nearest_oseries()
    return


def test_repr(prj):
    return prj.__repr__()


def test_delete_db(prj):
    if prj.db.conn_type == "arctic":
        util.delete_arctic(prj.db.connstr, prj.db.name)
    elif prj.db.conn_type == "pystore":
        util.delete_pystore(prj.db.path, prj.db.name)
    return
