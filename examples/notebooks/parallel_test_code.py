# %%
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from multiprocessing import Process, Queue
from timeit import default_timer as timer

import pandas as pd
import pastas as ps
from tqdm.contrib.concurrent import process_map, thread_map

import pastastore as pst

ps.set_log_level("ERROR")

# %%

conn = pst.PasConnector("zeeland_bro", path="/home/david/github/gwdatalens/pastasdb")
pstore = pst.PastaStore(conn)
model_names = pstore.model_names[:100]

# %%
conn2 = pst.ArcticDBConnector(
    "zeeland_bro2", uri="lmdb:///home/david/github/gwdatalens/pastasdb"
)
pstore2 = pst.PastaStore(conn2)

# %%


def connect_and_read_symbol(name, timings_=None):
    if timings_ is not None:
        start = pd.to_datetime("now")
    conn.get_model(name)
    # .write(
    #     symbol,
    #     pd.DataFrame(
    #         np.random.randn(nrows, ncols),
    #         columns=[f'c{i}' for i in range(ncols)]
    #     )
    # )
    if timings_ is not None:
        timings_.put((start, pd.to_datetime("now")))


def initializer(connector):
    """Create global connector object so each worker has one connection to database."""
    global conn
    conn = pst.ArcticDBConnector(*connector, verbose=False)
    return conn


connector = (conn2.name, conn2.uri)

read_model = partial(connect_and_read_symbol)
with ProcessPoolExecutor(
    max_workers=11, initializer=initializer, initargs=(connector,)
) as executor:
    executor.map(read_model, model_names, chunksize=pstore.n_models // 11)


# %%
for ipstore in [pstore, pstore2]:
    print(f"--- {ipstore.conn.conn_type} ---")
    start = timer()
    ipstore.solve_models(model_names, parallel=False, progressbar=False)
    end = timer()
    print(f"Single core: {(end - start):.2f} s")

    start = timer()
    ipstore.solve_models(model_names, parallel=False, progressbar=True)
    end = timer()
    print(f"Single core (progressbar): {(end - start):.2f} s")

    start = timer()
    ipstore.solve_models(model_names, parallel=True, progressbar=False)
    end = timer()
    print(f"Multi core:  {(end - start):.2f} s")

    start = timer()
    ipstore.solve_models(model_names, parallel=True, progressbar=True)
    end = timer()
    print(f"Multi core (progressbar):  {(end - start):.2f} s")

# %%
pstore2.solve_models(model_names, parallel=True, progressbar=True)

# %%
pstore.solve_models(model_names, parallel=True, progressbar=True)

# %%

timings = Queue()


# %%
concurrent_writers = {
    Process(target=connect_and_read_symbol, args=(name, (conn.name, conn.uri), timings))
    for name in model_names
}

# Start all processes
for proc in concurrent_writers:
    proc.start()
# Wait for them to complete
for proc in concurrent_writers:
    proc.join()

assert set(conn.arc["models"].list_symbols()) == model_names

# plot timings
timings_list = []
while not timings.empty():
    timings_list.append(timings.get())

pd.DataFrame(timings_list, columns=["start", "end"]).transpose().plot()

# %%

# %%

read_model = partial(connect_and_read_symbol, conn=(conn.name, conn.uri))
process_map(read_model, model_names, max_workers=11, chunksize=1)

# %%
read_model = partial(connect_and_read_symbol, conn=(conn.name, conn.uri))
thread_map(read_model, model_names, max_workers=11, chunksize=1)

# %%
read_model = partial(connect_and_read_symbol, conn=(conn.name, conn.uri))
with ProcessPoolExecutor(max_workers=11) as executor:
    executor.map(read_model, model_names, chunksize=pstore.n_models // 11)

# %%
read_model = partial(connect_and_read_symbol, conn=(conn.name, conn.uri))
with ProcessPoolExecutor(max_workers=11) as executor:
    executor.submit(read_model, model_names)

# %%
