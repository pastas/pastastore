import pystore
import arctic

from typing import Optional, List


def delete_pystore(path: str, name: str, libraries: Optional[List[str]] = None) \
        -> None:
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


def delete_arctic(connstr: str, name: str,
                  libraries: Optional[List[str]] = None) -> None:

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
