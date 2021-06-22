=========
Utilities
=========

The `pastastore.util` submodule contains useful functions, i.e. for deleting
databases, connector objects, and PastaStore objects, emptying a library of
all its contents or copying all data to a new database:


* :meth:`pastastore.util.delete_pastastore`
* :meth:`pastastore.util.delete_dict_connector`
* :meth:`pastastore.util.delete_pas_connector`
* :meth:`pastastore.util.delete_pystore_connector`
* :meth:`pastastore.util.delete_arctic_connector`
* :meth:`pastastore.util.empty_library`
* :meth:`pastastore.util.copy_database`


It also contains a method for making a detailed comparison between two 
pastas.Models:

* :meth:`pastastore.util.compare_models`