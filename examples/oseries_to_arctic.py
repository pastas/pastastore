from timeit import default_timer

import arctic
import geopandas as gpd
import pandas as pd
import tqdm
from shapely.geometry import Point

import observations as obs

# %% settings
connstr = "mongodb://localhost:27017/"
storename = 'vitens'
items = ["GW.meting.totaalcorrectie"]

# %% connect to database
arc = arctic.Arctic(connstr)

# %% write data to library oseries

# read in data
oc = obs.ObsCollection.from_pystore(storename,
                                    "../../traval/extracted_data/pystore",
                                    nameby="collection",
                                    item_names=items,
                                    verbose=False,
                                    progressbar=True)
# write to arctic
libname = "{}.oseries".format(storename)
oc.to_arctic(connstr, libname, verbose=True)
