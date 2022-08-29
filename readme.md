![pastastore](https://github.com/pastas/pastastore/workflows/pastastore/badge.svg)
[![Documentation Status](https://readthedocs.org/projects/pastastore/badge/?version=latest)](https://pastastore.readthedocs.io/en/latest/?badge=latest)
[![Codacy Badge](https://api.codacy.com/project/badge/Grade/81b1e0294f5247cfa4eca657a8eebc61)](https://www.codacy.com/gh/pastas/pastastore?utm_source=github.com&utm_medium=referral&utm_content=pastas/pastastore&utm_campaign=Badge_Grade)
[![Codacy Badge](https://app.codacy.com/project/badge/Coverage/81b1e0294f5247cfa4eca657a8eebc61)](https://www.codacy.com/gh/pastas/pastastore/dashboard?utm_source=github.com&utm_medium=referral&utm_content=pastas/pastastore&utm_campaign=Badge_Coverage)
![PyPI](https://img.shields.io/pypi/v/pastastore)

# pastastore

This module stores 
[Pastas](https://pastas.readthedocs.io/en/latest/) timeseries and models in a
database.

Storing timeseries and models in a database allows the user to manage time
series and Pastas models on disk, which allows the user to pick up where they
left off without having to reload everything.

## Installation

Install the module with `pip install pastastore`.

For installing in development mode, clone the repository and install by typing
`pip install -e .` from the module root directory.

For plotting background maps, the `contextily` and `pyproj` packages are
required. For a full install, including an optional dependency for plotting and
labeling data on maps, use: `pip install pastastore[full]` or `pip install
.[full]` when on MacOS or Linux. Windows users are asked to install `rasterio`
themselves since it often cannot be installed using `pip`. `rasterio` is a
dependency of `contextily`. Windows users can install `pastastore` with the
optional labeling package adjustText using `pip install pastastore[adjusttext]`
or `.[adjusttext]`.

_Note: There are external dependencies when using the `pystore` or `arctic`
connectors. To install these dependencies read (see [Connector Dependencies
section](#dependencies))! since these are \_not_ automatically installed.\_

## Usage

The following snippets show typical usage. The first step is to define a
so-called `Connector` object. This object contains methods to store time series
or models to the database, or read objects from the database.

The following code creates a PasConnector, which uses Pastas JSON-styled
"`.pas`-files" to save models in a folder on your computer (in this case a
folder called `pastas_db` in the current directory).

```python
import pastastore as pst

# create connector instance
conn = pst.PasConnector("my_db", path="./pastas_db")
```

The next step is to pass that connector to the `PastaStore` object. This object
contains all kinds of useful methods to analyze and visualize time series, and
build and analyze models.

```python
# create PastaStore instance
pstore = pst.PastaStore("my_project", conn)
```

Now the user can add time series, models or analyze or visualize existing
objects in the database. Some examples showing the functionality of the
PastaStore object are shown below:

```python
import pandas as pd

# load oseries from CSV and add to database
oseries = pd.read_csv("oseries.csv")
pstore.add_oseries(oseries, "my_oseries", metadata={"x": 100_000, "y": 400_000})

# read oseries from database
oseries = pstore.get_oseries("my_oseries")

# view oseries metadata DataFrame
pstore.oseries

# plot oseries location on map
ax = pstore.maps.oseries()
pstore.maps.add_background_map(ax)  # add a background map

# plot my_oseries time series
ax2 = pstore.plot.oseries(names=["my_oseries"])

# export whole database to a zip file
pstore.to_zip("my_db_backup.zip")
```

For more elaborate examples, refer to the
[Notebooks](https://pastastore.readthedocs.io/en/latest/examples.html#example-notebooks).

### Which Connector should I choose?

There are currently four Connectors included in `pastastore`. Each of the
Connectors are briefly described below.

-   PasConnector (works out of the box, **preferred choice**)
-   DictConnector (works out of the box)
-   ArcticConnector (requires `arctic` and MongoDB, **best performance**)
-   PystoreConnector (requires `pystore` and `python-snappy`)

#### PasConnector

For most people the `PasConnector` is the best choice, as it does not require
any other (external) dependencies, and uses human-readable files to store time
series and pastas Models on disk.

```python
# requires a name, and path to a folder
conn = pst.PasConnector("my_db", path="./pastas_db")
```

#### DictConnector

The `DictConnector` does not store files on disk, storing everything in memory.
This is usually not what you'd want, but it can be useful as a temporary
storage container. All "stored" data will be lost if you restart the kernel.

```python
# requires a name
conn = pst.DictConnector("my_temporary_db")
```

#### ArcticConnector

Store data in MongoDB using Arctic. Only works if there is an instance of
MongoDB running and the `arctic` python package is installed. This Connector
has the best performance, both in terms of read/write speeds and data
compression.

```python
# provide a name and a connection string to a running instance of MongoDB
connstr = "mongodb://localhost:27017/"  # local instance of mongodb
conn = pst.ArcticConnector("my_db", connstr)
```

#### PystoreConnector

Store data on disk as parquet files using compression. Only works if
`python-snappy` and `pystore` are installed. Does not require separate database
software running somewhere, but installation of `python-snappy` is a little
more challenging. Slightly less performant than ArcticConnector, but faster
than PasConnector.

```python
# provide a name and a path to a folder on disk
conn = pst.PystoreConnector("my_db", path="./pastas_db")
```

## Connector Dependencies

This module has several dependencies (depending on which connector is used):

If using `Dictconnector` or `PasConnector`:

-   No additional dependencies are required.

If using `ArcticConnector`:

-   Arctic requires MongoDB, e.g. install the Community edition
    ([Windows](https://fastdl.mongodb.org/win32/mongodb-win32-x86_64-2012plus-4.2.1-signed.msi),
    [MacOS](https://fastdl.mongodb.org/osx/mongodb-macos-x86_64-4.2.1.tgz)).

-   OR, if you wish to use Docker for running MongoDB see the installation
    instructions [here](https://github.com/pastas/pastastore/tree/master/dockerfiles#running-mongodb-from-docker).

If using `PystoreConnector`:

-   PyStore uses [Snappy](http://google.github.io/snappy/), a fast and efficient
    compression/decompression library from Google. You'll need to install Snappy on
    your system before installing PyStore. See links for installation instructions
    here: <https://github.com/ranaroussi/pystore#dependencies>
