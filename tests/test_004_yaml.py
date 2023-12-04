import os
import tempfile
from contextlib import contextmanager

import pytest
from pytest_dependency import depends

import pastastore as pst


@contextmanager
def tempyaml(yaml):
    temp = tempfile.NamedTemporaryFile(delete=False)
    temp.write(yaml.encode("utf-8"))
    temp.close()
    try:
        yield temp.name
    finally:
        os.unlink(temp.name)


@pytest.mark.dependency()
def test_load_yaml_rechargemodel(pstore):
    yamlstr = """
    my_first_model:                   # model name
      oseries: oseries2               # head time series name, obtained from pastastore
      stressmodels:                   # stressmodels dictionary
        recharge:                     # name of the recharge stressmodel
          class: RechargeModel        # type of pastas StressModel
          prec: prec2                 # name of precipitation stress, obtained from pastastore
          evap: evap2                 # name of evaporation stress, obtained from pastastore
          recharge: Linear            # pastas recharge type
          rfunc: Exponential          # response function
    """
    with tempyaml(yamlstr) as f:
        ml = pstore.yaml.load(f)[0]
    pstore.add_model(ml)


@pytest.mark.dependency()
def test_load_yaml_stressmodel(pstore):
    yamlstr = """
    my_second_model:                  # model name
      oseries: oseries2               # head time series name, obtained from pastastore
      stressmodels:                   # stressmodels dictionary
        prec:                         # name of the recharge stressmodel
          class: StressModel          # type of pastas StressModel
          stress: prec2               # name of precipitation stress, obtained from pastastore
          rfunc: Gamma                # response function
    """
    with tempyaml(yamlstr) as f:
        ml = pstore.yaml.load(f)[0]
    pstore.add_model(ml)


@pytest.mark.dependency()
def test_load_yaml_wellmodel(pstore):
    yamlstr = """
    my_third_model:                   # model name
      oseries: oseries1               # head time series name, obtained from pastastore
      stressmodels:                   # stressmodels dictionary
        well:                         # name of the recharge stressmodel
          class: WellModel            # type of pastas StressModel
          stress: well1               # name of well stress, obtained from pastastore
          distances: [100]

    """
    with tempyaml(yamlstr) as f:
        ml = pstore.yaml.load(f)[0]
    pstore.add_model(ml)


@pytest.mark.dependency()
def test_write_load_compare_yaml(request, pstore):
    depends(
        request,
        [
            f"test_load_yaml_rechargemodel[{pstore.type}]",
            f"test_load_yaml_stressmodel[{pstore.type}]",
            f"test_load_yaml_wellmodel[{pstore.type}]",
        ],
    )
    pstore.yaml.export_models(modelnames=["my_first_model"])
    ml1 = pstore.models["my_first_model"]
    ml2 = pstore.yaml.load("my_first_model.yaml")[0]
    assert (
        pst.util.compare_models(ml1, ml2, detailed_comparison=True).iloc[1:, -1].all()
    )
    os.remove("my_first_model.yaml")


@pytest.mark.dependency()
def test_write_yaml_per_oseries(request, pstore):
    depends(
        request,
        [
            f"test_load_yaml_rechargemodel[{pstore.type}]",
            f"test_load_yaml_stressmodel[{pstore.type}]",
            f"test_load_yaml_wellmodel[{pstore.type}]",
        ],
    )
    pstore.yaml.export_stored_models_per_oseries()
    os.remove("oseries1.yaml")
    os.remove("oseries2.yaml")


@pytest.mark.dependency()
def test_write_yaml_minimal(request, pstore):
    depends(
        request,
        [
            f"test_load_yaml_rechargemodel[{pstore.type}]",
            f"test_load_yaml_stressmodel[{pstore.type}]",
            f"test_load_yaml_wellmodel[{pstore.type}]",
        ],
    )
    ml = pstore.models["my_first_model"]
    pstore.yaml.export_model(ml, minimal_yaml=True)
    os.remove("my_first_model.yaml")


@pytest.mark.dependency()
def test_write_yaml_minimal_nearest(request, pstore):
    depends(
        request,
        [
            f"test_load_yaml_rechargemodel[{pstore.type}]",
            f"test_load_yaml_stressmodel[{pstore.type}]",
            f"test_load_yaml_wellmodel[{pstore.type}]",
        ],
    )
    ml = pstore.models["my_third_model"]
    pstore.yaml.export_model(ml, minimal_yaml=True, use_nearest=True)
    os.remove("my_third_model.yaml")
