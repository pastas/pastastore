===============
Getting Started
===============
On this page you will find all the information to get started with `pastastore`.

Getting Python
--------------
To install `pastastore`, a working version of Python 3.7 or higher has to be
installed on your computer. We recommend using the
`Anaconda Distribution <https://www.continuum.io/downloads>`_
of Python.

Installing `pastastore`
-----------------------
Install the module by typing::
  
    pip install pastastore

Please note that the `pystore` and `arctic` modules are not automatically 
installed (see `Dependencies`_ section below)!

_For installing in development mode, clone the repository and install by
typing `pip install -e .` from the module root directory._

Using `pastastore`
------------------
Start Python and import the module::

    import pastastore as pst
    conn = pst.DictConnector("my_connector")
    store = pst.PastaStore("my_store", conn)

See the :ref:`examples` section for some quick examples on how to get started.

Dependencies
------------
This module has several optional dependencies. These are required for storing 
timeseries and models in a database using compression. Without these, only 
the in-memory option (DictConnector) and storing data on disk without 
compression (PasConnector) are available.

It is up to the user to determine which dependencies they want to install. 
Both the `PystoreConnector` and `ArcticConnector` are dependent on external 
software. It is recommended but not required to install these dependencies.

* Using Pystore requires Snappy:
   * `Snappy <http://google.github.io/snappy/>`_ is a fast and efficient
     compression/decompression library from Google. You'll need to install
     Snappy on your system before installing PyStore. See links for installation
     instructions here: `<https://github.com/ranaroussi/pystore#dependencies>`_
   * After installing Snappy, install pystore by typing `pip install pystore`
     in a terminal.
* Using Arctic requires MongoDB:
   * The recommended method of obtaining MongoDB is using
     `Docker <https://www.docker.com/products/docker-desktop>`_.
     The instructions for this are shown below.
   * Alternatively, get MongoDB by installing the Community edition
     (`Windows <https://fastdl.mongodb.org/win32/mongodb-win32-x86_64-2012plus-4.2.1-signed.msi>`_,
     `MacOS <https://fastdl.mongodb.org/osx/mongodb-macos-x86_64-4.2.1.tgz>`_).

Running MongoDB from docker
^^^^^^^^^^^^^^^^^^^^^^^^^^^
Follow these steps to get the MongoDB docker container up and running
using `docker-compose`:

#. Install Docker (i.e.
   `Docker Desktop <https://www.docker.com/products/docker-desktop>`_)
#. Open a terminal and navigate to `./dockerfiles`.
#. Run `docker-compose up -d` to start the docker container running MongoDB.
   The `-d` flag runs the container in the background. This command uses the
   `docker-compose.yml` file by default.
#. View your running containers with `docker ps -a`.
#. If you are done and wish to stop the container, run `docker-compose stop` in a terminal.


