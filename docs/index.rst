Welcome to pastastore's documentation!
======================================

`pastastore` is a module for managing
`Pastas <https://pastas.readthedocs.io/en/latest/>`_ timeseries and models.

The module supports storing and managing data with a database or on disk.
This gives the user a simple way to manage Pastas projects, and allows the user
to pick up where they left off, without having to load all data into memory.
For users who have used `pastas` before, this module is similar to
`pastas.Project`, but much more extensive in terms of functionality.

The connection to the data/database/disk is managed by a connector object.
Currently, three connectors are included. The first connector `DictConnector`
stores all data in-memory using dictionaries. The other two implementations
`ArcticConnector` and `PystoreConnector` store data in a database or on disk. Both
implementations are designed to have fast read/write operations, while also
compressing the stored data. These connectors are implemented using the following
two modules:

* `Arctic <https://arctic.readthedocs.io/en/latest/>`_ is a timeseries/dataframe
  database that sits atop `MongoDB <https://www.mongodb.com>`_. Arctic supports
  pandas.DataFrames.
* `PyStore <https://github.com/ranaroussi/pystore>`_ is a datastore (inspired
  by Arctic) created for storing pandas dataframes (especially timeseries) on
  disk. Data is stored using fastparquet and compressed with Snappy.

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
