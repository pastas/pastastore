# %% read data from library
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

# %% read only metadata
metalist = []
for sym in tqdm.tqdm(timelib.list_symbols()):
    metalist.append(timelib.read_metadata(sym).metadata)

metadf = pd.DataFrame(metalist)
metadf["x"] = metadf["x"].astype(float)
metadf["y"] = metadf["y"].astype(float)
metadf = gpd.GeoDataFrame(metadf, geometry=[Point(
    (s["x"], s["y"])) for i, s in metadf.iterrows()])