from timeit import default_timer

import arctic
import pandas as pd
import tqdm
from shapely.geometry import Point

import observations as obs

# %% settings
connstr = "mongodb://localhost:27017/"
libname = "aaenmaas"
pystore_path = r"C:\GitHub\traval\extracted_data\pystore"

# %% load data
oc = obs.ObsCollection.from_pystore("knmi",
                                    pystore_path,
                                    obs.KnmiObs,
                                    verbose=False,
                                    progressbar=True)

# %% add kind information to stresses
for i, irow in oc.iterrows():
    if irow.obs.meta["name"].startswith("RD"):
        kind = "prec"
    elif irow.obs.meta["name"].startswith("EV24"):
        kind = "evap"
    irow.obs.meta["kind"] = kind
    oc.loc[i, "kind"] = kind

# %% write to arctic library
libname = "{}.stresses".format(libname)
oc.to_arctic(connstr, libname, verbose=True)
