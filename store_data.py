from timeit import default_timer

import arctic
import geopandas as gpd
import pandas as pd
import tqdm
from shapely.geometry import Point

import observations as obs

# start database by typing mongod in console

# connect to database
arc = arctic.Arctic('localhost:27017')

libname = "time.ts"
# create library
if not libname in arc.list_libraries():
    arc.initialize_library("time.ts")

    # get library
    timelib = arc["time.ts"]

    # read in data
    oc = obs.ObsCollection.from_pystore("aaenmaas",
                                        "../traval/extracted_data/pystore",
                                        nameby="collection",
                                        item_names=["GW.meting.totaalcorrectie"],
                                        verbose=False, progressbar=True)

    # write data to library
    for o in tqdm.tqdm(oc.obs.values):
        metadata = o.meta
        timelib.write(o.name, o, metadata=metadata)
else:
    # get library
    timelib = arc["time.ts"]

# read data from library
collect = []
rows_read = 0
start = default_timer()

for sym in tqdm.tqdm(timelib.list_symbols()):
    item = timelib.read(sym)
    collect.append(item.data)
    rows_read += len(item.data.index)
end = default_timer()
print("Symbols: {0:.0f}  Rows: {1:,.0f}  "
      "Time: {2:.2f}s  Rows/s: {3:,.1f}".format(len(timelib.list_symbols()),
                                                rows_read,
                                                (end -
                                                 start),
                                                rows_read / (end - start)))

# read only metadata
metalist = []
for sym in tqdm.tqdm(timelib.list_symbols()):
    metalist.append(timelib.read_metadata(sym).metadata)

metadf = pd.DataFrame(metalist)
metadf["x"] = metadf["x"].astype(float)
metadf["y"] = metadf["y"].astype(float)
metadf = gpd.GeoDataFrame(metadf, geometry=[Point(
    (s["x"], s["y"])) for i, s in metadf.iterrows()])

# %%

arc.initialize_library("time.models")
