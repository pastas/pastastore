# ruff: noqa: D100 D103
import pytest
from pandas import Timestamp

import pastastore as pst


@pytest.mark.pastas150
def test_hpd_download_from_bro():
    from pastastore.extensions import activate_hydropandas_extension

    activate_hydropandas_extension()
    pstore = pst.PastaStore()
    pstore.hpd.download_bro_gmw(
        extent=(117850, 118180, 439550, 439900), tmin="2022-01-01", tmax="2022-01-02"
    )
    assert pstore.n_oseries == 3


@pytest.mark.pastas150
def test_hpd_download_precipitation_from_knmi():
    from pastastore.extensions import activate_hydropandas_extension

    activate_hydropandas_extension()
    pstore = pst.PastaStore()
    pstore.hpd.download_knmi_precipitation(
        stns=[260], tmin="2022-01-01", tmax="2022-01-31"
    )
    assert pstore.n_stresses == 1


@pytest.mark.pastas150
def test_hpd_download_evaporation_from_knmi():
    from pastastore.extensions import activate_hydropandas_extension

    activate_hydropandas_extension()
    pstore = pst.PastaStore()
    pstore.hpd.download_knmi_evaporation(
        stns=[260], tmin="2022-01-01", tmax="2022-01-31"
    )
    assert pstore.n_stresses == 1


@pytest.mark.pastas150
def test_update_oseries():
    from pastastore.extensions import activate_hydropandas_extension

    activate_hydropandas_extension()

    pstore = pst.PastaStore.from_zip("tests/data/test_hpd_update.zip")
    pstore.hpd.update_bro_gmw(tmax="2024-01-31")
    tmintmax = pstore.get_tmin_tmax("oseries")
    assert tmintmax.loc["GMW000000036319_1", "tmax"] >= Timestamp("2024-01-30")
    assert tmintmax.loc["GMW000000036327_1", "tmax"] >= Timestamp("2024-01-20")


@pytest.mark.pastas150
def test_update_stresses():
    from pastastore.extensions import activate_hydropandas_extension

    activate_hydropandas_extension()

    pstore = pst.PastaStore.from_zip("tests/data/test_hpd_update.zip")
    pstore.hpd.update_knmi_meteo(tmax="2024-01-31")
    tmintmax = pstore.get_tmin_tmax("stresses")
    assert (tmintmax["tmax"] >= Timestamp("2024-01-31")).all()
