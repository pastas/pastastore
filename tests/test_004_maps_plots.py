import warnings

import matplotlib.pyplot as plt
import pytest
from pytest_dependency import depends

with warnings.catch_warnings():
    warnings.simplefilter(action="ignore", category=FutureWarning)
    import pastastore as pst

# %% plots


def test_plot_oseries(pstore):
    ax = pstore.plots.oseries()
    plt.close(ax.figure)
    return


def test_plot_stresses(pstore):
    ax = pstore.plots.oseries()
    plt.close(ax.figure)
    return


def test_plot_stresses_availability(pstore):
    ax = pstore.plots.data_availability("stresses", kind="prec",
                                        set_yticks=True)
    plt.close(ax.figure)
    return


# %% maps

def test_map_oseries_w_bgmap(pstore):
    ax = pstore.maps.oseries()
    # only test bgmap once for arctic
    if pstore.type == "arctic":
        pstore.maps.add_background_map(ax)
    plt.close(ax.figure)
    return


def test_map_stresses(pstore):
    ax = pstore.maps.stresses(kind="prec")
    plt.close(ax.figure)
    return

def test_map_stresses(pstore):
    ax = pstore.maps.stresslinks()
    plt.close(ax.figure)
    return

@pytest.mark.dependency()
def test_map_models(request, pstore):
    ml = pstore.create_model("oseries1")
    pstore.add_model(ml)
    ax = pstore.maps.models()
    plt.close(ax.figure)
    return


@pytest.mark.dependency()
def test_map_model(request, pstore):
    depends(request, [f"test_map_models[{pstore.type}]"])
    ax = pstore.maps.model("oseries1")
    plt.close(ax.figure)
    return


def test_delete_db(pstore):
    pst.util.delete_pastastore(pstore)
    return
