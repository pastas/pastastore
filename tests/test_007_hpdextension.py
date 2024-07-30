# ruff: noqa: D100 D103
import pytest

import pastastore as pst
from pastastore.extensions import activate_hydropandas_extension


@pytest.mark.pastas150()
def test_hpd_download_from_bro():
    activate_hydropandas_extension()
    pstore = pst.PastaStore()
    pstore.hpd.download_bro_gmw(
        extent=(117850, 118180, 439550, 439900), tmin="2022-01-01", tmax="2022-01-02"
    )
    assert pstore.n_oseries == 3


@pytest.mark.pastas150()
def test_hpd_download_precipitation_from_knmi():
    activate_hydropandas_extension()
    pstore = pst.PastaStore()
    pstore.hpd.download_knmi_precipitation(
        stns=[260], tmin="2022-01-01", tmax="2022-01-31"
    )
    assert pstore.n_stresses == 1


@pytest.mark.pastas150()
def test_hpd_download_evaporation_from_knmi():
    activate_hydropandas_extension()
    pstore = pst.PastaStore()
    pstore.hpd.download_knmi_evaporation(
        stns=[260], tmin="2022-01-01", tmax="2022-01-31"
    )
    assert pstore.n_stresses == 1
