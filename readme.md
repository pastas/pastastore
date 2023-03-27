![pastastore](https://github.com/pastas/pastastore/workflows/pastastore/badge.svg)
[![Documentation Status](https://readthedocs.org/projects/pastastore/badge/?version=latest)](https://pastastore.readthedocs.io/en/latest/?badge=latest)
[![Codacy Badge](https://api.codacy.com/project/badge/Grade/81b1e0294f5247cfa4eca657a8eebc61)](https://www.codacy.com/gh/pastas/pastastore?utm_source=github.com&utm_medium=referral&utm_content=pastas/pastastore&utm_campaign=Badge_Grade)
[![Codacy Badge](https://app.codacy.com/project/badge/Coverage/81b1e0294f5247cfa4eca657a8eebc61)](https://www.codacy.com/gh/pastas/pastastore/dashboard?utm_source=github.com&utm_medium=referral&utm_content=pastas/pastastore&utm_campaign=Badge_Coverage)
![PyPI](https://img.shields.io/pypi/v/pastastore)

# pastastore

This module stores 
[Pastas](https://pastas.readthedocs.io/en/latest/) time series and models in a
database.

Storing time series and models in a database allows the user to manage time
series and Pastas models on disk, which allows the user to pick up where they
left off without having to reload everything.

## Installation

Install the module with `pip install pastastore`.

For installing in development mode, clone the repository and install by typing
`pip install -e .` from the module root directory.

For plotting background maps, the `contextily` and `pyproj` packages are
required. For a full install, including optional dependencies for plotting and
labeling data on maps, use: `pip install pastastore[optional]` Windows users
are asked to install `rasterio` themselves since it often cannot be installed
using `pip`. `rasterio` is a dependency of `contextily`.

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
conn = pst.PasConnector(name="pastas_db", path=".")
```

The next step is to pass that connector to the `PastaStore` object. This object
contains all kinds of useful methods to analyze and visualize time series, and
build and analyze models.

```python
# create PastaStore instance
pstore = pst.PastaStore(conn)
```

Now the user can add time series, models or analyze or visualize existing
objects in the database. Some examples showing the functionality of the
PastaStore object are shown below:

```python
import pandas as pd
import pastas as ps

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

# create a model with pastas
ml = ps.Model(oseries, name="my_model")

# add model to database
pstore.add_model(ml)

# load model from database
ml2 = pstore.get_models("my_model")

# export whole database to a zip file
pstore.to_zip("my_backup.zip")
```

For more elaborate examples, refer to the
[Notebooks](https://pastastore.readthedocs.io/en/latest/examples.html#example-notebooks).
