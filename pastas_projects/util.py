import pystore
import arctic


def delete_pystore(path, name, libraries=None):
    pystore.set_path(path)
    if libraries is None:
        pystore.delete_store(name)
    else:
        store = pystore.store(name)
        for lib in libraries:
            store.delete_collection(lib)


def delete_arctic(connstr, name, libraries=None):

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
