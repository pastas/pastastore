# Which Connector should you pick?

There are currently four Connectors included in `pastastore`. Each of the
Connectors are briefly described below.

- PasConnector (works out of the box, **preferred choice**)
- DictConnector (works out of the box)
- ArcticConnector (not actively tested, requires `arctic` and MongoDB, **best performance**)
- PystoreConnector (not actively tested, requires `pystore` and `python-snappy`)

## PasConnector

For most people the `PasConnector` is the best choice, as it does not require
any other (external) dependencies, and uses human-readable files to store time
series and pastas Models on disk.

```python
# requires a name, and path to a folder
conn = pst.PasConnector("my_db", path="./pastas_db")
```

## DictConnector

The `DictConnector` does not store files on disk, storing everything in memory.
This is usually not what you'd want, but it can be useful as a temporary
storage container. All "stored" data will be lost if you restart the kernel.

```python
# requires a name
conn = pst.DictConnector("my_temporary_db")
```

## ArcticConnector

Store data in MongoDB using Arctic. Only works if there is an instance of
MongoDB running and the `arctic` python package is installed. This Connector
has the best performance, both in terms of read/write speeds and data
compression. Currently, `arctic` does not work with pandas>1.1 because it
has reference to the deprecated `pandas.Panel` object. This can be fixed by
installing a custom version of arctic in which this bug has been addressed.
This can be done by cloning a Pull Request and installing that version of arctic.

```python
# provide a name and a connection string to a running instance of MongoDB
connstr = "mongodb://localhost:27017/"  # local instance of mongodb
conn = pst.ArcticConnector("my_db", connstr)
```

## PystoreConnector

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

- No additional dependencies are required.

If using `ArcticConnector`:

- Arctic requires MongoDB, e.g. install the Community edition
    ([Windows](https://fastdl.mongodb.org/win32/mongodb-win32-x86_64-2012plus-4.2.1-signed.msi),
    [MacOS](https://fastdl.mongodb.org/osx/mongodb-macos-x86_64-4.2.1.tgz)).
- OR, if you wish to use Docker for running MongoDB see the installation
    instructions [here](https://github.com/pastas/pastastore/tree/master/dockerfiles#running-mongodb-from-docker).

If using `PystoreConnector`:

- PyStore uses [Snappy](http://google.github.io/snappy/), a fast and efficient
    compression/decompression library from Google. You'll need to install Snappy on
    your system before installing PyStore. See links for installation instructions
    here: <https://github.com/ranaroussi/pystore#dependencies>
