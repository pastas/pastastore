import os
import pastas as ps
from pastas_projects import ArcticPastas

ps.set_log_level("ERROR")


def test_init():
    # which dbase to use
    collection = "aaenmaas"

    # connect to dbase (make sure docker container is up and running)
    connstr = "mongodb://localhost:27017/"
    pr = ArcticPastas(connstr, collection)
    pr.__repr__()
    return pr


def test_get_oseries():
    pr = test_init()
    o = pr.get_oseries("103JVM_boven_O")
    o.index.name = "103JVM_boven_O"
    return o


def test_add_oseries():
    pr = test_init()
    o = test_get_oseries()
    pr.add_oseries(o, o.index.name + "__2")
    return


def test_del_oseries():
    pr = test_init()
    pr.del_oseries("103JVM_boven_O__2")
    return


def test_get_stress():
    pr = test_init()
    s = pr.get_stresses('980')
    s.name = '980'
    return s


def test_add_stress():
    pr = test_init()
    s = test_get_stress()
    pr.add_stress(s, s.name + "__2", "prec")
    return


def test_del_stress():
    pr = test_init()
    pr.del_stress("980__2")
    return


def test_create_model():
    pr = test_init()
    ml = pr.create_model("103JVM_boven_O")
    return ml


def test_store_model():
    pr = test_init()
    ml = test_create_model()
    ml.name += "__2"
    pr.add_model(ml)
    return


def test_get_model():
    pr = test_init()
    ml = pr.get_models("103JVM_boven_O__2")
    return ml


def test_del_model():
    pr = test_init()
    pr.del_model("103JVM_boven_O__2")
    return


def test_get_library():
    pr = test_init()
    olib = pr.get_library("oseries")
    return olib


def test_create_models():
    pr = test_init()
    mls = pr.create_models(["WIJB020_G", "STRA001_G"], progressbar=False)
    return mls


def test_solve_models():
    pr = test_init()
    pr.solve_models(["WIJB020_G", "STRA001_G"],
                    ignore_solve_errors=True,
                    progressbar=False,
                    store_result=False)
    return

def test_model_results():
    pr = test_init()
    pr.model_results(["WIJB020_G", "STRA001_G"], progressbar=False)
    return


def test_oseries_prop():
    pr = test_init()
    return pr.oseries


def test_stresses_prop():
    pr = test_init()
    return pr.stresses


def test_models_prop():
    pr = test_init()
    return pr.models
