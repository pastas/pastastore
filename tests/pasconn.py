import os
import pastastore as pst
import pandas as pd
from conftest import initialize_project

try:
    pst.util.delete_arctic_connector(connstr="mongodb://localhost:27017/", name="test")
except Exception as e:
    print(e)
    
pr = pst.ArcticConnector("test", "mongodb://localhost:27017/")


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


# 1 / 0
# os.chdir("..")
# conn = pst.PasConnector("test", path="./tests/data/pas")
# pr = initialize_project(conn)
# ml = pr.create_model("oseries1")
# pr.add_model(ml)
# pr.to_zip("test_pas.zip", overwrite=True)

# store = pst.PastaStore.from_zip("./test_pas.zip", pst.DictConnector("test"))

# p = store.get_parameters(progressbar=False, param_value="initial")

# 1 / 0

# mls = store.create_models_bulk(["oseries1", "oseries2"], store=True,
#                                progressbar=False)
# mls = store.solve_models(["oseries1", "oseries2"],
#                          ignore_solve_errors=False,
#                          progressbar=False,
#                          store_result=True)
# stats = store.get_statistics(["evp", "aic"], progressbar=False)
