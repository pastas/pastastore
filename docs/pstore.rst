=================
PastaStore object
=================

The `PastaStore` object is essentially a class for working with timeseries and
pastas Models. A connector has to be passed to the object which manages the
retrieval and storage of data.

Methods are available for the following tasks:

* Calculating distances between locations of timeseries, i.e. getting the
  nearest timeseries to a location::

    # get 3 nearest oseries
    store.get_nearest_oseries("my_oseries", n=3)

    # get nearest precipitation series
    store.get_nearest_stress("my_oseries", kind="prec")


* Creating pastas Models, optionally adding a recharge stressmodel::

    # create model
    ml = store.create_model("my_oseries", add_recharge=True)

Bulk operations are also provided for:

* Creating and storing pastas Models::

    # create models and store in database
    store.create_models(add_recharge=True, store=True)

* Optimizing pastas Models and storing the results::

    # solve models and store result in database
    store.solve_models(ignore_solver_errors=True, store_result=True)
