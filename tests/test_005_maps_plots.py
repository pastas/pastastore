import matplotlib.pyplot as plt
import pytest
from conftest import requires_pkg
from pytest_dependency import depends

# %% plots


def test_plot_oseries(pstore):
    ax = pstore.plots.oseries()
    plt.close(ax.figure)


def test_plot_stresses(pstore):
    ax = pstore.plots.stresses()
    plt.close(ax.figure)


def test_plot_stresses_availability(pstore):
    ax = pstore.plots.data_availability("stresses", kind="prec", set_yticks=True)
    plt.close(ax.figure)


@pytest.mark.dependency()
def test_cumulative_hist(request, pstore):
    ml1 = pstore.create_model("oseries1")
    pstore.add_model(ml1)
    ml2 = pstore.create_model("oseries2")
    pstore.add_model(ml2)
    ax = pstore.plots.cumulative_hist()
    plt.close(ax.figure)


# %% maps


@pytest.mark.bgmap
def test_map_oseries_w_bgmap(pstore):
    ax = pstore.maps.oseries()
    # only test bgmap once for pas
    if pstore.type == "pas":
        pstore.maps.add_background_map(ax)
    plt.close(ax.figure)


@requires_pkg("adjustText")
def test_map_stresses(pstore):
    ax = pstore.maps.stresses(kind="prec", adjust=True)
    plt.close(ax.figure)


def test_map_stresslinks(pstore):
    ml = pstore.create_model("oseries1", modelname="ml1")
    pstore.add_model(ml)
    ax = pstore.maps.stresslinks()
    plt.close(ax.figure)


@pytest.mark.dependency()
def test_map_models(request, pstore):
    ax = pstore.maps.models()
    plt.close(ax.figure)


@pytest.mark.dependency()
def test_map_model(request, pstore):
    depends(request, [f"test_map_models[{pstore.type}]"])
    ax = pstore.maps.model("oseries1")
    plt.close(ax.figure)


@pytest.mark.dependency()
def test_map_modelstat(request, pstore):
    ax = pstore.maps.modelstat("evp")
    plt.close(ax.figure)


@pytest.mark.dependency()
def test_list_ctx_providers(request, pstore):
    pstore.maps._list_contextily_providers()
