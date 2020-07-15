========
Examples
========
This page provides some short examples and an example application in a
Jupyter Notebook. The following snippets show typical usage.

The general idea is to first define the connector object. This object manages
the communication between the user and the data store. The the next step is to
pass that connector to the `PastaStore` to access all of the useful methods
for creating and solving timeseries models.

Using Arctic
------------

The following snippet shows how to create an `ArcticConnector` and initialize
a `PastaStore` object. Please note that a MongoDB instance must be running for
this to work::

   import pastastore as pst

   # define arctic connector
   connstr = "mongodb://localhost:27017/"
   conn = pst.ArcticConnector("my_connector", connstr)

   # create project for managing Pastas data and models
   store = pst.PastasProject("my_project", conn)


Using Pystore
-------------

To use the `PystoreConnector` the steps are identical, the only difference is
that the user has to provide a path to a location on disk instead of a
connection string to a database::

   # define pystore connector
   path = "./data/pystore"
   conn = pst.PystoreConnector("my_connector", path)

   # create project for managing Pastas data and models
   store = pst.PastasProject("my_project", conn)


The PastaStore object
---------------------

The `PastaStore` object provides useful methods e.g. for creating models and
determining which timeseries are closest to one another. The database
read/write/delete methods are always accessed through `pr.conn` i.e.::

   # add an observation timeseries
   series = pd.Series(index=pd.date_range("2019", "2020", freq="D"), data=1.0)
   store.conn.add_oseries(series, "my_oseries", metadata={"x": 100, "y": 200})

   # retrieve the oseries
   oseries = store.conn.get_oseries("my_oseries")

To create a timeseries model use `store.create_model()`. Note that this does
not automatically add the model to the database. To store the model, it has to
be explicitly added to the database::

   # create a timeseries model
   ml = store.create_model("my_oseries", add_recharge=False)

   # add to the database
   store.conn.add_model(ml)

   # retrieve model from database
   ml = store.conn.get_models("my_oseries")


Example Notebooks
-----------------

The links below link to Jupyter Notebooks with explanation and examples of the
usage of the `pastastore` module:

.. toctree::
  :maxdepth: 1
  :glob:

  examples/*
