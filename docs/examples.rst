========
Examples
========
This page provides some short examples and example applications in 
Jupyter Notebooks. The following snippets show typical usage.

The general idea is to first define the connector object. This object manages
the communication between the user and the data store. The the next step is to
pass that connector to the `PastaStore` to access all of the useful methods
for building time series models.

In-memory
---------

The following snippet shows how to use PastaStore with in-memory storage of 
time series and models. This is the simplest implementation because everything
is stored in-memory (in dictionaries)::

   import pastastore as pst

   # define dict connector
   conn = pst.DictConnector("my_db")

   # create project for managing Pastas data and models
   store = pst.PastaStore(conn)


Using Pastas
------------

The following snippet shows how to use PastaStore with storage of 
time series and models on disk as pas-files. This is the simplest implementation 
that writes data to disk as no external dependencies are required::

   import pastastore as pst

   # define pas connector
   path = "./data/pastas_db"
   conn = pst.PasConnector("my_db", path)

   # create project for managing Pastas data and models
   store = pst.PastaStore(conn)


Using ArcticDB
--------------

The following snippet shows how to create an `ArcticDBConnector` and initialize
a `PastaStore` object::

   import pastastore as pst

   # define ArcticDB connector
   uri = "lmdb://./my_path_here/"
   conn = pst.ArcticDBConnector("my_db", uri)

   # create project for managing Pastas data and models
   store = pst.PastaStore(conn)


The PastaStore object
---------------------

The `PastaStore` object provides useful methods e.g. for creating models and
determining which time series are closest to one another. The database 
read/write/delete methods can be accessed directly from the `PastaStore` 
object::

   # create a new time series
   series = pd.Series(index=pd.date_range("2019", "2020", freq="D"), data=1.0)
   
   # add an observation time series
   store.add_oseries(series, "my_oseries", metadata={"x": 100, "y": 200})

   # retrieve the oseries
   oseries = store.get_oseries("my_oseries")

To create a Pastas time series model use `store.create_model()`. Note that this does
not automatically add the model to the database. To store the model, it has to
be explicitly added to the database::

   # create a time series model
   ml = store.create_model("my_oseries", add_recharge=False)

   # add to the database
   store.add_model(ml)

   # retrieve model from database
   ml = store.get_models("my_oseries")


The links below link to Jupyter Notebooks with explanation and examples of the
usage of the `pastastore` module:

.. toctree::
  :maxdepth: 3
  :caption: Notebooks

  notebooks/index.rst
