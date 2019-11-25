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
- To ensure a persistant volume (i.e. your data is kept even if you shutdown your docker container running mongodb) run the following command before starting the mongo database, e.g. `docker volume create --name=mongodata`
- Start the docker container on localhost by running e.g. `docker run --mongodb -v mongodata:/data/db -d -p 27017:27017 mongo`
- View your running containers with `docker ps -a`

## Installation
After installing Snappy, install the module by typing `pip install -e .` from the module root directory.


