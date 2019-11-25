# pastas_projects

This module contains objects to manage [Pastas](https://pastas.readthedocs.io/en/latest/) timeseries and models with [Arctic](https://arctic.readthedocs.io/en/latest/) or [PyStore](https://github.com/ranaroussi/pystore). These objects are similar to pastas.Project, but do not hold all the data in memory, instead writing all data to a database or disk instead. This gives the user a simple way to manage Pastas projects, and allows the user to pick up where they left off, without having to load everything into memory. Both implementations are designed to have fast read/write operations, while also compressing the stored data.

There are two flavors of Project in this module:
- ArcticPastas is based on Arctic. Arctic is a timeseries/dataframe database that sits atop [MongoDB](https://www.mongodb.com). Arctic supports pandas.DataFrames.
- PystorePastas is based on Pystore: PyStore is a datastore (inspired by Arctic) created for storing pandas dataframes (especially timeseries) on disk. Data is stored using fastparquet and compressed with Snappy.

## Dependencies
This module has several dependencies:

If using Pystore:
- PyStore uses [Snappy](http://google.github.io/snappy/), a fast and efficient compression/decompression library from Google. You'll need to install Snappy on your system before installing PyStore.

If using Arctic:
- Arctic uses MongoDB. Install the Community edition ([Windows](https://fastdl.mongodb.org/win32/mongodb-win32-x86_64-2012plus-4.2.1-signed.msi), [MacOS](https://fastdl.mongodb.org/osx/mongodb-macos-x86_64-4.2.1.tgz)).

If using Docker for running MongoDB:
- Install [Docker](https://www.docker.com/products/docker-desktop)
- Pull the mongodb image: `docker pull mongo`
- To ensure a persistent volume (i.e. your data is kept even if you shutdown your docker container running mongodb) run the following command before starting the mongo database, e.g. `docker volume create --name=mongodata`
- Start the docker container on localhost by running e.g. `docker run --mongodb -v mongodata:/data/db -d -p 27017:27017 mongo`
- View your running containers with `docker ps -a`

## Installation
After installing Snappy, install the module by typing `pip install -e .` from the module root directory.

## Projects

The structure and some background on the different types of Projects is detailed below.

### ArcticPastas
ArcticPastas is an object that creates a connection with a MongoDB database. This can be an existing or a new database. A collection is created to hold the different datasets: observation timeseries, stresses timeseries and models. For each of these datasets a library is created. These are named using the following convention: `<collection name>.<library name>`.

The data is stored within these libraries. Observations and stresses timeseries are stored as pandas.DataFrames. Models are stored in JSON (actually binary JSON) and *do not* contain the timeseries themselves. These are picked up from the other libraries when the model is loaded from the database.

The ArcticPastas object allows the user to add different versions for datasets, which can be used, for example, to keep a history of older models.

### PystorePastas
PystorePastas is an object that links to a location on disk. This can either be an existing or a new Pystore. Three different stores are created to hold the data:
observation series, stresses timeseries, and models.

The Pystores have the following structure: `store / collections / items`. Compared to the ArcticPastas object, there is an extra level which allows several different timeseries to be stored for one location. This also means that when loading or writing data both the 'collection' and 'item' names have to be passed.

The timeseries data is stored as Dask DataFrames which can be easily converted to pandas DataFrames. The models are stored as JSON (not including the timeseries) in the metadata file belonging to an item. The actual data in the item is an empty DataFrame serving as a placeholder. This slightly 'hacky' design allows the models to be saved in a PyStore. The timeseries are picked up from their respective stores when the model is loaded from disk.

PyStore supports so-called snapshots (which store the current state of the store) but this has not been actively implemented in PystorePastas. PystorePastas does not have the same versioning capabilities as ArcticPastas.

## Notes
- There may be some hard-coded references to certain DataFrame columns as this project was developed using one data source. The timeseries data is assumed to be stored in a column with name 'value' and the comments are stored in a column with the name 'comments'. This should be improved in future updates.
- The tests are run using an existing store/database on a local PC. In future updates the testing data and databases should be moved to the ./tests/data directory.
- Test the versioning in ArcticPastas.
