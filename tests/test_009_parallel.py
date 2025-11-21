# ruff: noqa: D100 D103
import pytest
from conftest import for_connectors

import pastastore as pst

pst.get_color_logger("DEBUG", logger_name="pastastore")


@pytest.fixture(scope="module")
def pstore_with_models(pstore):
    """
    Fixture that creates models for oseries1 and oseries2.

    This provides a pastastore with two models ready for testing parallel-safe
    operations and time series link updates.
    """
    # Create and add model for oseries1
    ml1 = pstore.create_model("oseries1")
    ml1.solve(report=False)
    pstore.add_model(ml1)

    # Create and add model for oseries2
    ml2 = pstore.create_model("oseries2")
    ml2.solve(report=False)
    pstore.add_model(ml2)
    # trigger update to start fresh
    _ = pstore.conn.oseries_models
    # ensure clean start
    pstore.conn._trigger_links_update_if_needed()
    yield pstore

    # Cleanup
    if "oseries1" in pstore.model_names:
        pstore.del_model("oseries1")
    if "oseries2" in pstore.model_names:
        pstore.del_model("oseries2")


class TestSingleThreadOperations:
    """Test single-thread add_model operations for all connector types."""

    def test_oseries_models_triggers_update(self, pstore_with_models):
        """Test that accessing oseries_models triggers update of added models."""
        # Initially, _added_models should be empty
        assert len(pstore_with_models.conn._added_models) == 0
        # add model
        ml = pstore_with_models.create_model("oseries3")
        pstore_with_models.add_model(ml)
        # Access oseries_models to trigger update
        _ = pstore_with_models.conn.oseries_models
        # After update, _added_models should be empty
        assert len(pstore_with_models.conn._added_models) == 0
        # cleanup
        pstore_with_models.del_model("oseries3")

    def test_stresses_models_triggers_update(self, pstore_with_models):
        """Test that accessing stresses_models triggers update of added models."""
        # initially, _added_models should be empty
        assert len(pstore_with_models.conn._added_models) == 0
        # add model
        ml = pstore_with_models.create_model("oseries3")
        pstore_with_models.add_model(ml)
        # access stresses_models to trigger update
        _ = pstore_with_models.conn.stresses_models
        # after update, _added_models should be empty
        assert len(pstore_with_models.conn._added_models) == 0
        # cleanup
        pstore_with_models.del_model("oseries3")

    def test_oseries_with_models_triggers_update(self, pstore_with_models):
        """Test that accessing oseries_with_models triggers update of added models."""
        # initially, _added_models should be empty
        assert len(pstore_with_models.conn._added_models) == 0
        # add model
        ml = pstore_with_models.create_model("oseries3")
        pstore_with_models.add_model(ml)
        # access oseries_models to trigger update
        _ = pstore_with_models.conn.oseries_with_models
        # after update, _added_models should be empty
        assert len(pstore_with_models.conn._added_models) == 0
        # cleanup
        pstore_with_models.del_model("oseries3")

    def test_stresses_with_models_triggers_update(self, pstore_with_models):
        """Test that accessing stresses_with_models triggers update of added models."""
        # initially, _added_models should be empty
        assert len(pstore_with_models.conn._added_models) == 0
        # add model
        ml = pstore_with_models.create_model("oseries3")
        pstore_with_models.add_model(ml)
        # access stresses_with_models to trigger update
        _ = pstore_with_models.conn.stresses_with_models
        # after update, _added_models should be empty
        assert len(pstore_with_models.conn._added_models) == 0
        # cleanup
        pstore_with_models.del_model("oseries3")

    def test_add_model_appends_to_added_models_list(self, pstore_with_models):
        """Test that add_model appends to _added_models list."""
        # Check internal state, should contain two models
        assert len(pstore_with_models.conn._added_models) == 0
        # Create a new model
        ml = pstore_with_models.create_model("oseries3")
        # Add model
        pstore_with_models.add_model(ml)
        # Verify the model name was added to _added_models
        assert "oseries3" in pstore_with_models.conn._added_models
        # check oseries_models (triggers link update)
        assert "oseries3" in pstore_with_models.oseries_models
        # check that _added_models is now empty after update
        assert len(pstore_with_models.conn._added_models) == 0
        # Cleanup
        pstore_with_models.del_model("oseries3")

    def test_del_model_deletes_from_added_models_list(self, pstore_with_models):
        """Test that del_model deletes from non-empty _added_models list."""
        # Check internal state, should be empty
        assert len(pstore_with_models.conn._added_models) == 0
        # Create a new model
        ml = pstore_with_models.create_model("oseries3")
        # Add model
        pstore_with_models.add_model(ml)
        pstore_with_models.del_model("oseries3")
        # check that _added_models is now empty after update
        assert len(pstore_with_models.conn._added_models) == 0


class TestParallelButNotReally:
    """Test internal update flag behavior (for parallel)."""

    @for_connectors(connectors=["pas", "arcticdb"])
    def test_parallel_add_model_sets_update_flags(self, pstore_with_models):
        """Test parallel_safe adds set update flags for parallel connectors."""

        def add_model(name):
            """Add model using global pstore."""
            ml = pstore_with_models.get_models(name)
            ml.solve(report=False)
            pstore_with_models.add_model(ml, overwrite=True)

        pstore_with_models.apply(
            func=add_model,
            names=["oseries1", "oseries2"],
            libname="models",
            parallel=False,
        )

        # check update parameters are set to True
        assert pstore_with_models.conn._oseries_links_need_update.value is True
        assert pstore_with_models.conn._stresses_links_need_update.value is True

        # trigger update
        pstore_with_models.conn._trigger_links_update_if_needed()

        # after update, flags should be False
        assert pstore_with_models.conn._oseries_links_need_update.value is False
        assert pstore_with_models.conn._stresses_links_need_update.value is False

    @for_connectors(connectors=["pas", "arcticdb"])
    def test_oseries_models_triggers_update(self, pstore_with_models):
        """Test that accessing oseries_models triggers update of added models."""
        # Initially, _added_models should be empty
        assert len(pstore_with_models.conn._added_models) == 0

        def add_model(name):
            """Add model using pstore."""
            ml = pstore_with_models.create_model(name)
            pstore_with_models.add_model(ml, overwrite=True)

        pstore_with_models.apply(
            func=add_model,
            names=["oseries1", "oseries2"],
            libname="oseries",
            parallel=False,
        )

        # check update parameters are set to True
        assert pstore_with_models.conn._oseries_links_need_update.value is True
        assert pstore_with_models.conn._stresses_links_need_update.value is True

        # trigger update
        om = pstore_with_models.oseries_models

        # check if result is correct
        assert om["oseries1"] == ["oseries1"]
        assert om["oseries2"] == ["oseries2"]

        # after update, flags should be False
        assert pstore_with_models.conn._oseries_links_need_update.value is False
        assert pstore_with_models.conn._stresses_links_need_update.value is False

    @for_connectors(connectors=["pas", "arcticdb"])
    def test_stresses_models_triggers_update(self, pstore_with_models):
        """Test that accessing stresses_models triggers update of added models."""
        # Initially, _added_models should be empty
        assert len(pstore_with_models.conn._added_models) == 0

        def add_model(name):
            """Add model using pstore."""
            ml = pstore_with_models.create_model(name)
            pstore_with_models.add_model(ml, overwrite=True)

        pstore_with_models.apply(
            func=add_model,
            names=["oseries1", "oseries2"],
            libname="oseries",
            parallel=False,
        )

        # check update parameters are set to True
        assert pstore_with_models.conn._oseries_links_need_update.value is True
        assert pstore_with_models.conn._stresses_links_need_update.value is True

        # trigger update
        sm = pstore_with_models.stresses_models

        # check if result is correct
        assert sm["prec1"] == ["oseries1"]
        assert sm["prec2"] == ["oseries2"]
        assert sm["evap1"] == ["oseries1"]
        assert sm["evap2"] == ["oseries2"]

        # after update, flags should be False
        assert pstore_with_models.conn._oseries_links_need_update.value is False
        assert pstore_with_models.conn._stresses_links_need_update.value is False

    @for_connectors(connectors=["pas", "arcticdb"])
    def test_oseries_with_models_triggers_update(self, pstore_with_models):
        """Test that accessing oseries_with_models triggers update of added models."""
        # Initially, _added_models should be empty
        assert len(pstore_with_models.conn._added_models) == 0

        def add_model(name):
            """Add model using pstore."""
            ml = pstore_with_models.create_model(name)
            pstore_with_models.add_model(ml, overwrite=True)

        pstore_with_models.apply(
            func=add_model,
            names=["oseries1", "oseries2"],
            libname="oseries",
            parallel=False,
        )

        # check update parameters are set to True
        assert pstore_with_models.conn._oseries_links_need_update.value is True
        assert pstore_with_models.conn._stresses_links_need_update.value is True

        # trigger update
        owm = pstore_with_models.oseries_with_models

        # check if result is correct
        assert "oseries1" in owm
        assert "oseries2" in owm

        # after update, flags should be False
        assert pstore_with_models.conn._oseries_links_need_update.value is False
        assert pstore_with_models.conn._stresses_links_need_update.value is False

    @for_connectors(connectors=["pas", "arcticdb"])
    def test_stresses_with_models_triggers_update(self, pstore_with_models):
        """Test that accessing stresses_with_models triggers update of added models."""
        # Initially, _added_models should be empty
        assert len(pstore_with_models.conn._added_models) == 0

        def add_model(name):
            """Add model using pstore."""
            ml = pstore_with_models.create_model(name)
            pstore_with_models.add_model(ml, overwrite=True)

        pstore_with_models.apply(
            func=add_model,
            names=["oseries1", "oseries2"],
            libname="oseries",
            parallel=False,
        )

        # check update parameters are set to True
        assert pstore_with_models.conn._oseries_links_need_update.value is True
        assert pstore_with_models.conn._stresses_links_need_update.value is True

        # check if result is correct
        swm = pstore_with_models.stresses_with_models

        assert "prec1" in swm
        assert "prec2" in swm
        assert "evap1" in swm
        assert "evap2" in swm

        # after update, flags should be False
        assert pstore_with_models.conn._oseries_links_need_update.value is False
        assert pstore_with_models.conn._stresses_links_need_update.value is False


def add_model_pas(name):
    """Add model using pstore."""
    ml = ppstore.create_model(name)
    ppstore.add_model(ml, overwrite=True)


def add_model_arcticdb(name):
    """Add model using pstore."""
    ppstore = pst.PastaStore(conn)
    ml = ppstore.create_model(name)
    conn.add_model(ml, overwrite=True)


def setup_parallel_pstore(conn_type, data1, data2):
    global conn
    global ppstore

    if conn_type == "arcticdb":
        name = "paralleldb"
        uri = "lmdb://./tests/data/arcticdb/"
        conn = pst.ArcticDBConnector(name, uri)
    elif conn_type == "pas":
        name = "paralleldb"
        conn = pst.PasConnector(name, "./tests/data/pas")
    else:
        raise ValueError("Unrecognized parameter!")

    ppstore = pst.PastaStore(conn)
    # dataset 1
    ppstore.add_oseries(data1["oseries1"], "oseries1", metadata=data1["oseries1_meta"])
    ppstore.add_stress(
        data1["prec1"], "prec1", kind="prec", metadata=data1["prec1_meta"]
    )
    ppstore.add_stress(
        data1["evap1"], "evap1", kind="evap", metadata=data1["evap1_meta"]
    )

    # dataset 2
    ppstore.add_oseries(data2["oseries2"], "oseries2", metadata=data2["oseries2_meta"])
    ppstore.add_stress(
        data2["prec2"], "prec2", kind="prec", metadata=data2["prec2_meta"]
    )
    ppstore.add_stress(
        data2["evap2"], "evap2", kind="evap", metadata=data2["evap2_meta"]
    )
    return ppstore


@pytest.mark.parametrize("conn_type", ["pas", "arcticdb"])
def test_parallel_add_model(conn_type, data1, data2):
    """Test that accessing stresses_with_models triggers update of added models."""

    ppstore = setup_parallel_pstore(conn_type, data1, data2)

    if conn_type == "arcticdb":
        func = add_model_arcticdb
    elif conn_type == "pas":
        func = add_model_pas
    else:
        raise ValueError("Unrecognized parameter!")
    try:
        # Initially, _added_models should be empty
        assert len(ppstore.conn._added_models) == 0

        ppstore.apply(
            func=func,
            names=["oseries1", "oseries2"],
            libname="oseries",
            parallel=True,
            max_workers=1,
        )

        # check update parameters are set to False, since after parallel these are
        # recomputed automatically
        assert ppstore.conn._oseries_links_need_update.value is False
        assert ppstore.conn._stresses_links_need_update.value is False

        # check if result is correct
        om = ppstore.oseries_models
        owm = ppstore.oseries_with_models
        sm = ppstore.stresses_models
        swm = ppstore.stresses_with_models
        assert "oseries1" in owm
        assert "oseries2" in owm
        assert "oseries1" in om
        assert "oseries2" in om
        assert "oseries1" in om["oseries1"]
        assert "oseries2" in om["oseries2"]
        assert "prec1" in swm
        assert "prec2" in swm
        assert "evap1" in swm
        assert "evap2" in swm
        assert "oseries1" in sm["prec1"]
        assert "oseries2" in sm["prec2"]
        assert "oseries1" in sm["evap1"]
        assert "oseries2" in sm["evap2"]

    finally:
        pst.util.delete_pastastore(ppstore)
