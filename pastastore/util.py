from typing import Optional, List
from tqdm import tqdm


def _custom_warning(message, category=UserWarning, filename='', lineno=-1,
                    *args):
    print(f"{filename}:{lineno}: {category.__name__}: {message}")


def delete_pystore_connector(path: Optional[str] = None,
                             name: Optional[str] = None,
                             conn=None,
                             libraries: Optional[List[str]] = None) -> None:
    """Delete libraries from pystore.

    Parameters
    ----------
    path : str, optional
        path to pystore
    name : str, optional
        name of the pystore
    conn : PystoreConnector, optional
        PystoreConnector object
    libraries : Optional[List[str]], optional
        list of library names to delete, by default None which deletes
        all libraries
    """
    try:
        import pystore
    except ModuleNotFoundError as e:
        print("Please install `pystore`!")
        raise e

    if conn is not None:
        name = conn.name
        path = conn.path
    elif name is None or path is None:
        raise ValueError("Please provide 'name' and 'path' OR 'conn'!")

    print(f"Deleting pystore: '{name}' ...", end="")
    pystore.set_path(path)
    if libraries is None:
        pystore.delete_store(name)
        print(" Done!")
    else:
        store = pystore.store(name)
        for lib in libraries:
            print()
            store.delete_collection(lib)
            print(f" - deleted: {lib}")


def delete_arctic_connector(connstr: Optional[str] = None,
                            name: Optional[str] = None,
                            conn=None,
                            libraries: Optional[List[str]] = None) -> None:
    """Delete libraries from arctic database.

    Parameters
    ----------
    connstr : str, optional
        connection string to the database
    name : str, optional
        name of the database
    conn : pastastore.ArcticConnector
        ArcticConnector object
    libraries : Optional[List[str]], optional
        list of library names to delete, by default None which deletes
        all libraries
    """
    try:
        import arctic
    except ModuleNotFoundError as e:
        print("Please install `arctic`!")
        raise e

    if conn is not None:
        name = conn.name
        connstr = conn.connstr
    elif name is None or connstr is None:
        raise ValueError("Please provide 'name' and 'connstr' OR 'conn'!")

    arc = arctic.Arctic(connstr)

    print(f"Deleting database: '{name}' ...")
    # get library names
    if libraries is None:
        libs = []
        for ilib in arc.list_libraries():
            if ilib.split(".")[0] == name:
                libs.append(ilib)
    else:
        libs = [name + "." + ilib for ilib in libraries]

    for l in libs:
        arc.delete_library(l)
        print(f" - deleted: {l}")
    print("... Done!")


def delete_dict_connector(conn, libraries: Optional[List[str]] = None) -> None:
    print(f"Deleting DictConnector: '{conn.name}' ...", end="")
    if libraries is None:
        del conn
        print(" Done!")
    else:
        for l in libraries:
            print()
            delattr(conn, f"lib_{conn.libname[l]}")
            print(f" - deleted: {l}")
    print("... Done!")


def delete_pastastore(pstore, libraries: Optional[List[str]] = None) -> None:
    """Delete libraries from PastaStore.

    Note
    ----
    This deletes the original PastaStore object. To access
    data that has not been deleted, it is recommended to create a new
    PastaStore object with the same Connector settings. This also creates
    new empty libraries if they were deleted.

    Parameters
    ----------
    pstore : pastastore.PastaStore
        PastaStore object to delete (from)
    libraries : Optional[List[str]], optional
        list of library names to delete, by default None which deletes
        all libraries

    Raises
    ------
    TypeError
        when Connector type is not recognized
    """
    if pstore.conn.conn_type == "pystore":
        delete_pystore_connector(
            conn=pstore.conn, libraries=libraries)
    elif pstore.conn.conn_type == "dict":
        delete_dict_connector(pstore)
    elif pstore.conn.conn_type == "arctic":
        delete_arctic_connector(conn=pstore.conn, libraries=libraries)
    else:
        raise TypeError("Unrecognized pastastore Connector type: "
                        f"{pstore.conn.conn_type}")


def empty_library(pstore, libname: str,
                  progressbar: Optional[bool] = True) -> None:
    """Empty an entire library in a PastaStore.

    Parameters
    ----------
    pstore : pastastore.PastaStore
        PastaStore object to delete library contents from
    libname : str
        name of the library to delete all items from
    progressbar : bool, optional
        if True show progressbar (default)
    """
    conn = pstore.conn
    lib = conn.get_library(libname)
    names = conn._parse_names(None, libname)

    if pstore.conn.conn_type == "arctic":
        for name in (tqdm(names, desc=f"Deleting items from {libname}")
                     if progressbar else names):
            lib.delete(name)
        conn._clear_cache(libname)
    elif pstore.conn.conn_type == "pystore":
        for name in (tqdm(names, desc=f"Deleting items from {libname}")
                     if progressbar else names):
            lib.delete_item(name)
        conn._clear_cache(libname)
    elif pstore.conn.conn_type == "dict":
        for name in (tqdm(names, desc=f"Deleting items from {libname}")
                     if progressbar else names):
            _ = lib.pop(name)
    print(f"Emptied library {libname} in {conn.name}: {conn.__class__}")
