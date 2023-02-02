Welcome to pastastore's documentation!
======================================

`pastastore` is a module for managing
`Pastas <https://pastas.readthedocs.io/en/latest/>`_ time series and models.

The module supports storing and managing data with a database or on disk.
This gives the user a simple way to manage Pastas projects, and allows the user
to pick up where they left off, without having to load all data into memory.

An small example using pastastore::

  import pastastore as pst
  import pandas as pd
  
  # initialize a connector and a pastastore
  pstore = pst.PastaStore("my_store", pst.PasConnector("my_dbase", "./path_to_folder"))

  # read some data
  series = pd.read_csv("some.csv", index_col=[0], parse_dates=True)

  # add data to store
  pstore.add_oseries(series, "my_oseries", metadata={"x": 10., "y": 20.})


See the table of contents to get started with `pastastore`.

.. toctree::
   :maxdepth: 3
   :caption: Contents

   Getting started <getting_started>
   Example usage <examples>
   User guide <user_guide>
   API-docs <modules>


Indices and tables
==================

* :ref:`genindex`
