# pastas_projects

This module contains a tool to manage [Pastas](https://pastas.readthedocs.io/en/latest/) timeseries and models on disk. The connection to the database/disk is managed by a connector object. Currently, two connectors are included:
- [Arctic](https://arctic.readthedocs.io/en/latest/) is a timeseries/dataframe database that sits atop [MongoDB](https://www.mongodb.com). Arctic supports pandas.DataFrames.
- [PyStore](https://github.com/ranaroussi/pystore) is a datastore (inspired by Arctic) created for storing pandas dataframes (especially timeseries) on disk. Data is stored using fastparquet and compressed with Snappy.

The implementation is similar to pastas.Project, but instead of holding all the data in memory, all data is stored in a database or on disk instead. This gives the user a simple way to manage Pastas projects, and allows the user to pick up where they left off, without having to load everything into memory. Both implementations are designed to have fast read/write operations, while also compressing the stored data.

## Dependencies
This module has several dependencies (depending on which connector is used):

If using Pystore:
- PyStore uses [Snappy](http://google.github.io/snappy/), a fast and efficient compression/decompression library from Google. You'll need to install Snappy on your system before installing PyStore. See links for installation instructions here: https://github.com/ranaroussi/pystore#dependencies

If using Arctic:
- Arctic requires MongoDB, e.g. install the Community edition ([Windows](https://fastdl.mongodb.org/win32/mongodb-win32-x86_64-2012plus-4.2.1-signed.msi), [MacOS](https://fastdl.mongodb.org/osx/mongodb-macos-x86_64-4.2.1.tgz)).

Optional: if using Docker for running MongoDB:
- Install [Docker](https://www.docker.com/products/docker-desktop)
- Pull the mongodb image by typing `docker pull mongo` in a terminal.
- To ensure a persistent volume (i.e. your data is kept even if you shutdown your docker container running mongodb) run the following command before starting the mongo database, e.g. `docker volume create --name=mongodata`
- Start the docker container on localhost by running e.g. `docker run --mongodb -v mongodata:/data/db -d -p 27017:27017 mongo`
- View your running containers with `docker ps -a`

## Installation
Install the module by typing `pip install pastas_projects`. Please note that pystore and arctic are not automatically installed as dependencies

_For installing in development mode, clone the repository and install by typing `pip install -e .` from the module root directory._

## Usage
The following snippets show typical usage. The general idea is to first define the connector object. Then, the next step is to pass that connector to `PastasProject`.

### Using Arctic

```python
import pastas_projects as pp

# define arctic connector
connstr = "mongodb://localhost:27017/"
conn = pp.ArcticConnector("my_connector", connstr)

# create project for managing Pastas data and models
pr = pp.PastasProject("my_project", conn)
```
### Using Pystore

```python
import pastas_projects as pp

# define pystore connector
path = "./data/pystore"
conn = pp.PystoreConnector("my_connector", path)

# create project for managing Pastas data and models
pr = pp.PastasProject("my_project", conn)
```

The database read/write/delete methods are always accessed through `pr.db` i.e.:
```python
series = pr.db.get_oseries("my_oseries")
```

## Types of Connectors

The structure and some background on the different types of Projects is detailed below.

### ArcticConnector
The ArcticConnector is an object that creates a connection with a MongoDB database. This can be an existing or a new database. A database is created to hold the different datasets: observation timeseries, stresses timeseries and models. For each of these datasets a collection or library is created. These are named using the following convention: `<database name>.<collection name>`.

The Arctic implementation uses the following structure: `database / collections or libraries / documents`. The data is stored within these libraries. Observations and stresses timeseries are stored as pandas.DataFrames. Models are stored in JSON (actually binary JSON) and *do not* contain the timeseries themselves. These are picked up from the other libraries when the model is loaded from the database.

The ArcticPastas object allows the user to add different versions for datasets, which can be used to keep a history of older models for example.

### PystoreConnector
The PystoreConnector is an object that links to a location on disk. This can either be an existing or a new Pystore. A new store is created with collections that hold the different datasets: observation timeseries, stresses timeseries, and models.

The Pystores have the following structure: `store / collections / items`. The timeseries data is stored as Dask DataFrames which can be easily converted to pandas DataFrames. The models are stored as JSON (not including the timeseries) in the metadata file belonging to an item. The actual data in the item is an empty DataFrame serving as a placeholder. This slightly 'hacky' design allows the models to be saved in a PyStore. The timeseries are picked up from their respective stores when the model is loaded from disk.

PyStore supports so-called snapshots (which store the current state of the store) but this has not been actively implemented in this module. PystorePastas does not have the same versioning capabilities as Arctic.

### Custom Connectors
It should be relatively straightforward to write your own custom connector object. The
pastas_project.base module contains the BaseConnector class that defines which methods and properties must be defined. Each Connector object must inherit from this class. The BaseConnector class also shows the expected call signature for each method. Extra keyword arguments can be used in the custom class.

```python
class MyCustomConnector(BaseConnector):
    """Must override each method and property in BaseConnector, e.g."""

    def get_oseries(self, name, progressbar=False):
        # your code to get oseries from database here
        pass
```

## Notes
- The tests are run using a store/database on a local PC.
- Test the versioning of data in Arctic.
