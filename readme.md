![pastastore](https://github.com/pastas/pastastore/workflows/pastastore/badge.svg)
[![Documentation Status](https://readthedocs.org/projects/pastastore/badge/?version=latest)](https://pastastore.readthedocs.io/en/latest/?badge=latest)
[![Codacy Badge](https://api.codacy.com/project/badge/Grade/81b1e0294f5247cfa4eca657a8eebc61)](https://www.codacy.com/gh/pastas/pastastore?utm_source=github.com&utm_medium=referral&utm_content=pastas/pastastore&utm_campaign=Badge_Grade)
[![Codacy Badge](https://app.codacy.com/project/badge/Coverage/81b1e0294f5247cfa4eca657a8eebc61)](https://www.codacy.com/gh/pastas/pastastore/dashboard?utm_source=github.com&utm_medium=referral&utm_content=pastas/pastastore&utm_campaign=Badge_Coverage)
![PyPI](https://img.shields.io/pypi/v/pastastore)

# pastastore

This module contains a tool to manage
[Pastas](https://pastas.readthedocs.io/en/latest/) timeseries and models in a
database.

Storing timeseries and models in a database gives the user 
a simple way to manage Pastas projects with the added bonus of allowing the user 
to pick upwhere they left off, without having to (re)load everything into memory.

The connection to database/disk/memory is managed by a connector object.
Currently, four connectors are included. The first implementation is an 
in-memory connector. The other three store data on disk or in a database. 
The PasConnector implementation writes human-readable JSON files to disk. 
The ArcticConnector and PystoreConnector implementations are designed to have 
fast read/write operations, while also compressing the stored data.

-   In-memory: uses dictionaries to hold timeseries and pastas Models in-memory.
      Does not require any additional packages to use. 

-   Pastas: uses Pastas write and read methods to store data as JSON files on
      disk. Does not require any additional packages to use.

-   [Arctic](https://arctic.readthedocs.io/en/latest/) is a timeseries/dataframe
      database that sits atop [MongoDB](https://www.mongodb.com). Arctic supports
      pandas.DataFrames.

-   [PyStore](https://github.com/ranaroussi/pystore) is a datastore (inspired
      by Arctic) created for storing pandas dataframes (especially timeseries) on
      disk. Data is stored using fastparquet and compressed with Snappy.

## Installation

Install the module by typing `pip install pastastore`.

For installing in development mode, clone the repository and install by
typing `pip install -e .` from the module root directory. 

Pastastore has some optional dependencies which can be usefull if one want to 
use all functionalities of this package. The current optional dependencies
are adjustText, pyproj and contextily. There are external dependencies when 
using the `pystore` or `arctic` connectors as well. To install these
dependencies read (see [Connector Dependencies section](#dependencies))! 
since these are _not_ automatically installed. The adjustText, pyproj and 
contextily package can be automatically installed using `pip install 
pastastore[full]` or `pip install -e .[full]` when using Windows. MacOS or 
Linux users can use `pastastore[full,contextily]` to install contextily 
(and importantly rasterio) as well. Windows users are asked to install 
rasterio and contextily seperately.


## Usage

The following snippets show typical usage. The general idea is to first define
the connector object. The next step is to pass that connector to
`PastaStore`.

### Using in-memory dictionaries

This works out of the box after installing with `pip` without installing any 
additional Python dependencies or external software.

```python
import pastastore as pst

# define connector
conn = pst.DictConnector("my_connector")

# create project for managing Pastas data and models
store = pst.PastaStore("my_project", conn)
```

### Using Pastas read/load methods

Store data on disk as JSON files (with .pas extension) using Pastas read and 
load methods. This works out of the box after installing with `pip` without 
installing any additional Python dependencies or external software.

```python
import pastastore as pst

# define connector
path = "./data/pas"
conn = pst.PasConnector("my_connector")

# create project for managing Pastas data and models
store = pst.PastaStore("my_project", conn)
```

### Using Arctic

Store data in MongoDB using Arctic. Only works if there is an instance of 
MongoDB running somewhere.

```python
import pastastore as pst

# define arctic connector
connstr = "mongodb://localhost:27017/"  # local instance of mongodb
conn = pst.ArcticConnector("my_connector", connstr)

# create project for managing Pastas data and models
store = pst.PastaStore("my_project", conn)
```

### Using Pystore

Store data on disk as parquet files using compression. Only works if 
`python-snappy` and `pystore` are installed.

```python
import pastastore as pst

# define pystore connector
path = "./data/pystore"  # path to a directory
conn = pst.PystoreConnector("my_connector", path)

# create project for managing Pastas data and models
store = pst.PastaStore("my_project", conn)
```

The database read/write/delete methods can be accessed through the reference
to the connector object. For easy access, the
most common methods are registered to the `store` object. E.g.

```python
series = store.conn.get_oseries("my_oseries")
```

is equivalent to:

```python
series = store.get_oseries("my_oseries")
```

## Connector Dependencies

This module has several dependencies (depending on which connector is used):

If using `Dictconnector` or `PasConnector`:

-   No additional dependencies are required.

If using `ArcticConnector`:

-   Arctic requires MongoDB, e.g. install the Community edition
    ([Windows](https://fastdl.mongodb.org/win32/mongodb-win32-x86_64-2012plus-4.2.1-signed.msi),
    [MacOS](https://fastdl.mongodb.org/osx/mongodb-macos-x86_64-4.2.1.tgz)).

-   OR, if you wish to use Docker for running MongoDB see the installation instructions [here](https://github.com/pastas/pastastore/tree/master/dockerfiles#running-mongodb-from-docker).

If using `PystoreConnector`:

-   PyStore uses [Snappy](http://google.github.io/snappy/), a fast and
    efficient compression/decompression library from Google. You'll need to
    install Snappy on your system before installing PyStore. See links for
    installation instructions here:
    <https://github.com/ranaroussi/pystore#dependencies>
