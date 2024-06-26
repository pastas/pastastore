# Which Connector should you pick?

There are currently four Connectors included in `pastastore`. Each of the
Connectors are briefly described below.

- PasConnector (works out of the box, **preferred choice**)
- DictConnector (works out of the box)
- ArcticDBConnector (requires `arcticdb`, **best performance**)

## PasConnector

For most people the `PasConnector` is the best choice, as it does not require
any other (external) dependencies, and uses human-readable files to store time
series and pastas Models on disk.

```python
# requires a name, and path to a folder
conn = pst.PasConnector("./pastas_db", name="my_db")
```

## DictConnector

The `DictConnector` does not store files on disk, storing everything in memory.
This is usually not what you'd want, but it can be useful as a temporary
storage container. All "stored" data will be lost if you restart the kernel.

```python
# requires a name
conn = pst.DictConnector("my_temporary_db")
```

## ArcticDBConnector

Store on disk using ArcticDB. Only works if the `arcticdb` python package is installed.
This Connector has the best overall performance in terms of read/write speeds and data
compression.

```python
# requires a name, and path to a folder
path = "lmdb://./pastas_db/"  # path to folder
conn = pst.ArcticDBConnector(path, name="my_db")
```
