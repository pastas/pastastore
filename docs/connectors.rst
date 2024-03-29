=================
Connector objects
=================
The structure and some background on the different types of Connectors is
detailed below.

Each connector makes a distinction between the following datasets:

* observation time series (the series to be simulated)
* stresses time series (the forcing series on the system)
* models (the time series models)

In-memory
---------
The :ref:`DictConnector` is a very simple object that stores all
data and models in dictionaries. The data is stored in-memory and not on disk
and is therefore not persistent, i.e. you cannot pick up where you left off
last time. Once you exit Python your data is lost. For small projects, this
connector can be useful as it is extremely simple and fast.

Pas-files
---------
The :ref:`PasConnector` is an object that stores Pastas time series and models
on disk as pas-files. These are JSON files (with a .pas extension) and make 
use of Pastas methods to store models on disk. There is no compression of files 
and the files are stored in directories on the harddrive which means all files 
are human-readable. The advantage of this Connector is that no external 
dependencies are required. The downside of this storage method is that it takes 
up more diskspace and is slower than the other Connectors.

The PasConnector uses the following structure:

.. code-block::

   +-- directory
   |   +-- sub-directories (i.e. oseries, stresses, models)
   |   |   +-- pas-files... (i.e. individual time series or models)

The data is stored within these sub-directories. Observations and stresses 
tim eseries are stored as JSON files. Models are stored as JSON as well but 
*do not* contain the time series themselves. These are picked up from
the other directories when the model is loaded from the database.

ArcticDB
--------
Note: this Connector uses ArcticDB the next-generation version of Arctic. Requires arcticdb Python package.

The :ref:`ArcticDBConnector` is an object that creates a
local database. This can be an existing or a new database.
For each of the datasets a collection or library is created. These are named
using the following convention: `<database name>.<library name>`.

The ArcticDB implementation uses the following structure:

.. code-block::

   +-- database
   |   +-- libraries (i.e. oseries, stresses, models)
   |   |   +-- items... (i.e. individual time series or models)

The data is stored within these libraries. Observations and stresses time series
are stored as pandas.DataFrames. Models are stored as pickled dictionaries 
and *do not* contain the time series themselves. These are picked up from
the other libraries when the model is loaded from the database.

Arctic
------
Note: this Connector is not actively tested!

The :ref:`ArcticConnector` is an object that creates a
connection with a MongoDB database. This can be an existing or a new database.
For each of the datasets a collection or library is created. These are named
using the following convention: `<database name>.<library name>`.

The Arctic implementation uses the following structure:

.. code-block::

   +-- database
   |   +-- collections or libraries (i.e. oseries, stresses, models)
   |   |   +-- documents... (i.e. individual time series or models)

The data is stored within these libraries. Observations and stresses time series
are stored as pandas.DataFrames. Models are stored in JSON (actually binary
JSON) and *do not* contain the time series themselves. These are picked up from
the other libraries when the model is loaded from the database.

Pystore
-------
Note: this Connector is not actively tested!

The :ref:`PystoreConnector` is an object that links
to a location on disk. This can either be an existing or a new Pystore. A new
store is created with collections (or libraries) that hold the different 
datasets:

* observation time series
* stresses time series
* models

The Pystores have the following structure:

.. code-block:: bash

   +-- store
   |   +-- collections or libraries... (i.e. oseries, stresses, models)
   |   |   +-- items... (i.e. individual time series or models)


The time series data is stored as Dask DataFrames which can be easily converted
to pandas DataFrames. The models are stored as JSON (not including the
time series) in the metadata file belonging to an item. The actual data in the
item is an empty DataFrame serving as a placeholder. This slightly 'hacky'
design allows the models to be saved in a PyStore. The time series are picked
up from their respective stores when the model is loaded from disk.

Custom Connectors
-----------------
It should be relatively straightforward to write your own custom connector
object. The :ref:`Base` submodule contains the
`BaseConnector` class that defines which methods and properties *must*
be defined. The `ConnectorUtil` mix-in class contains some general methods that
are used by each connector. Each Connector object should inherit from these two
classes.

The `BaseConnector` class also shows the expected call signature for each
method. Following the same call signature should ensure that your new connector
works directly with `PastaStore`. Extra keyword arguments can be
added in the custom class.

Below is a small snippet showing a custom Connector class::

   class MyCustomConnector(BaseConnector, ConnectorUtil):
      """Must override each method and property in BaseConnector, e.g."""

      def _get_item(self, name, progressbar=False):
         # your code here for getting an item from your database
         pass
