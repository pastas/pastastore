# ruff: noqa: D100 D103
import pastas as ps
import pytest


def test_stressmodel_time_series_name(pstore):
    pstore.get_stressmodel("evap1")


def test_stressmodel_override_settings(pstore):
    sm = pstore.get_stressmodel("prec1", settings="evap")
    assert sm.stress[0].settings["fill_nan"] == "interpolate"

    # override settings by passing kind: bit weird, but good to check
    sm = pstore.get_stressmodel("prec1", kind="evap")
    assert sm.stress[0].settings["fill_nan"] == "interpolate"


def test_stressmodel_rfunc_kwargs(pstore):
    sm = pstore.get_stressmodel("well1", rfunc=ps.Hantush, rfunc_kwargs={"quad": True})
    assert sm.rfunc.quad


def test_stressmodel_nearest_kind_no_oseries_specified(pstore):
    with pytest.raises(ValueError, match=r"Getting nearest stress*"):
        pstore.get_stressmodel("nearest evap")  # error oseries


def test_stressmodel_nearest_no_kind_specified(pstore):
    with pytest.raises(ValueError, match=r"Could not parse stresses*"):
        pstore.get_stressmodel("nearest", oseries="oseries1")

    # also in dictionary mode
    with pytest.raises(ValueError, match=r"Could not parse stresses*"):
        pstore.get_stressmodel({"stress": ["nearest"]}, oseries="oseries1")


def test_stressmodel_nearest_kind(pstore):
    # nearest kind
    sm = pstore.get_stressmodel("nearest evap", oseries="oseries1")
    assert sm.stress[0].name == "evap1"

    # nearest kind in dict
    sm = pstore.get_stressmodel({"stress": "nearest evap"}, oseries="oseries1")
    assert sm.stress[0].name == "evap1"

    # nearest and kind separate
    sm = pstore.get_stressmodel({"stress": "nearest"}, kind="evap", oseries="oseries1")
    assert sm.stress[0].name == "evap1"

    # nearest in dict and kind separate
    sm = pstore.get_stressmodel(
        {"stress": ["nearest"]}, kind="evap", oseries="oseries1"
    )
    assert sm.stress[0].name == "evap1"


def test_recharge_model(pstore):
    # test list of stress names
    rm = pstore.get_stressmodel(["prec1", "evap1"], stressmodel="RechargeModel")
    assert rm.stress[0].name == "prec1"
    assert rm.stress[1].name == "evap1"

    # test list of nearest
    rm = pstore.get_stressmodel(
        ["nearest prec", "nearest evap"],
        stressmodel="RechargeModel",
        oseries="oseries1",
    )
    assert rm.stress[0].name == "prec1"
    assert rm.stress[1].name == "evap1"

    # test dict, no kind specified
    rm = pstore.get_stressmodel(
        {"prec": "nearest", "evap": "nearest"},
        stressmodel="RechargeModel",
        oseries="oseries1",
    )
    assert rm.stress[0].name == "prec1"
    assert rm.stress[1].name == "evap1"

    # test list, bare nearest with kind specified
    rm = pstore.get_stressmodel(
        ["nearest", "nearest"],
        kind=["prec", "evap"],
        stressmodel="RechargeModel",
        oseries="oseries1",
    )
    assert rm.stress[0].name == "prec1"
    assert rm.stress[1].name == "evap1"


def test_wellmodel(pstore):
    # test nearest <n> <kind>
    wm = pstore.get_stressmodel(
        "nearest 2 well",
        stressmodel="WellModel",
        oseries="oseries1",
    )
    assert wm.stress[0].name == "well1"
    assert wm.stress[1].name == "well2"

    # test nearest with no kind specified
    wm = pstore.get_stressmodel(
        "nearest 2",
        stressmodel="WellModel",
        oseries="oseries1",
    )
    assert wm.stress[0].name == "well1"
    assert wm.stress[1].name == "well2"

    # test nearest n, with non-existing kind specified
    with pytest.raises(ValueError, match=r"Could not find stresses*"):
        pstore.get_stressmodel(
            "nearest 2",
            kind="well2",
            stressmodel="WellModel",
            oseries="oseries1",
        )

    # test nearest n with n exceeded
    with pytest.raises(ValueError, match=r"Could not find*"):
        pstore.get_stressmodel(
            "nearest 3",
            kind="well",
            stressmodel="WellModel",
            oseries="oseries1",
        )
