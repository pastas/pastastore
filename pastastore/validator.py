"""Module containing Validator class for checking input data for connectors."""

import json
import logging
import os
import shutil
import warnings

# import weakref
from typing import TYPE_CHECKING, Union

import pandas as pd
import pastas as ps
from numpy import isin
from pandas.testing import assert_series_equal

from pastastore.util import SeriesUsedByModel, _custom_warning, validate_names

if TYPE_CHECKING:
    from pastastore.base import BaseConnector

FrameorSeriesUnion = Union[pd.DataFrame, pd.Series]
warnings.showwarning = _custom_warning

logger = logging.getLogger(__name__)


class Validator:
    """Validator class for checking input data and model consistency.

    This class provides validation methods for time series data, models,
    and metadata to ensure data integrity when storing in PastaStore
    connectors.

    Parameters
    ----------
    connector : BaseConnector
        The connector instance used for validation.

    Attributes
    ----------
    CHECK_MODEL_SERIES_VALUES : bool
        Whether to check model time series contents against stored copies.
    USE_PASTAS_VALIDATE_SERIES : bool
        Whether to validate time series according to pastas rules.
    SERIES_EQUALITY_ABSOLUTE_TOLERANCE : float
        Absolute tolerance for series equality comparison.
    SERIES_EQUALITY_RELATIVE_TOLERANCE : float
        Relative tolerance for series equality comparison.
    """

    # whether to check model time series contents against stored copies
    CHECK_MODEL_SERIES_VALUES = True

    # whether to validate time series according to pastas rules
    USE_PASTAS_VALIDATE_SERIES = True

    # protect series in models from being deleted or modified
    PROTECT_SERIES_IN_MODELS = True

    # set series equality comparison settings (using assert_series_equal)
    SERIES_EQUALITY_ABSOLUTE_TOLERANCE = 1e-10
    SERIES_EQUALITY_RELATIVE_TOLERANCE = 0.0

    def __init__(self, connector: "BaseConnector"):
        """Initialize Validator with connector reference.

        Parameters
        ----------
        connector : BaseConnector
            The connector instance to validate against.
        """
        self.connector = connector

    @property
    def settings(self):
        """Return current connector settings as dictionary."""
        return {
            "CHECK_MODEL_SERIES_VALUES": self.CHECK_MODEL_SERIES_VALUES,
            "USE_PASTAS_VALIDATE_SERIES": self.USE_PASTAS_VALIDATE_SERIES,
            "PROTECT_SERIES_IN_MODELS": self.PROTECT_SERIES_IN_MODELS,
            "SERIES_EQUALITY_ABSOLUTE_TOLERANCE": (
                self.SERIES_EQUALITY_ABSOLUTE_TOLERANCE
            ),
            "SERIES_EQUALITY_RELATIVE_TOLERANCE": (
                self.SERIES_EQUALITY_RELATIVE_TOLERANCE
            ),
        }

    def set_check_model_series_values(self, b: bool):
        """Turn CHECK_MODEL_SERIES_VALUES option on (True) or off (False).

        The default option is on (it is highly recommended to keep it that
        way). When turned on, the model time series
        (ml.oseries._series_original, and stressmodel.stress._series_original)
        values are checked against the stored copies in the database. If these
        do not match, an error is raised, and the model is not added to the
        database. This guarantees the stored model will be identical after
        loading from the database. This check is somewhat computationally
        expensive, which is why it can be turned on or off.

        Parameters
        ----------
        b : bool
            boolean indicating whether option should be turned on (True) or
            off (False). Option is on by default.
        """
        self.CHECK_MODEL_SERIES_VALUES = b
        logger.info("Model time series checking set to: %s.", b)

    def set_use_pastas_validate_series(self, b: bool):
        """Turn USE_PASTAS_VALIDATE_SERIES option on (True) or off (False).

        This will use pastas.validate_oseries() or pastas.validate_stresses()
        to test the time series. If they do not meet the criteria, an error is
        raised. Turning this option off will allow the user to store any time
        series but this will mean that time series models cannot be made from
        stored time series directly and will have to be modified before
        building the models. This in turn will mean that storing the models
        will not work as the stored time series copy is checked against the
        time series in the model to check if they are equal.

        Note: this option requires pastas>=0.23.0, otherwise it is turned off.

        Parameters
        ----------
        b : bool
            boolean indicating whether option should be turned on (True) or
            off (False). Option is on by default.
        """
        self.USE_PASTAS_VALIDATE_SERIES = b
        logger.info("Pastas time series validation set to: %s.", b)

    def set_protect_series_in_models(self, b: bool):
        """Turn PROTECT_SERIES_IN_MODELS option on (True) or off (False).

        The default option is on. When turned on, deleting a time series that
        is used in a model will raise an error. This prevents models from
        breaking because a required time series has been deleted. If you really
        want to delete such a time series, use the force=True option in
        del_oseries() or del_stress().

        Parameters
        ----------
        b : bool
            boolean indicating whether option should be turned on (True) or
            off (False). Option is on by default.
        """
        self.PROTECT_SERIES_IN_MODELS = b
        logger.info("Protect series in models set to: %s.", b)

    def pastas_validation_status(self, validate):
        """Whether to validate time series.

        Parameters
        ----------
        validate : bool, NoneType
            value of validate keyword argument

        Returns
        -------
        b : bool
            return global or local setting (True or False)
        """
        if validate is None:
            return self.USE_PASTAS_VALIDATE_SERIES
        else:
            return validate

    @staticmethod
    def check_filename_illegal_chars(libname: str, name: str) -> str:
        """Check filename for invalid characters (internal method).

        Parameters
        ----------
        libname : str
            library name
        name : str
            name of the item

        Returns
        -------
        str
            validated name
        """
        # check name for invalid characters in name
        new_name = validate_names(name, deletechars=r"\/" + os.sep, replace_space=None)
        if new_name != name:
            warning = (
                f"{libname} name '{name}' contained invalid characters "
                f"and was changed to '{new_name}'"
            )
            logger.warning(warning)
            name = new_name
        return name

    @staticmethod
    def validate_input_series(series):
        """Check if series is pandas.DataFrame or pandas.Series.

        Parameters
        ----------
        series : object
            object to validate

        Raises
        ------
        TypeError
            if object is not of type pandas.DataFrame or pandas.Series
        """
        if not (isinstance(series, pd.DataFrame) or isinstance(series, pd.Series)):
            raise TypeError("Please provide pandas.DataFrame or pandas.Series!")
        if isinstance(series, pd.DataFrame):
            if series.columns.size > 1:
                raise ValueError("Only DataFrames with one column are supported!")

    @staticmethod
    def set_series_name(series, name):
        """Set series name to match user defined name in store.

        Parameters
        ----------
        series : pandas.Series or pandas.DataFrame
            set name for this time series
        name : str
            name of the time series (used in the pastastore)
        """
        if isinstance(series, pd.Series):
            series.name = name
            # empty string on index name causes trouble when reading
            # data from ArcticDB: TODO: check if still an issue?
            if series.index.name == "":
                series.index.name = None

        if isinstance(series, pd.DataFrame):
            series.columns = [name]
            # check for hydropandas objects which are instances of DataFrame but
            # do have a name attribute
            if hasattr(series, "name"):
                series.name = name
        return series

    @staticmethod
    def check_stressmodels_supported(ml):
        """Check if all stressmodels in the model are supported."""
        supported_stressmodels = [
            "StressModel",
            "StressModel2",
            "RechargeModel",
            "WellModel",
            "TarsoModel",
            "Constant",
            "LinearTrend",
            "StepModel",
        ]
        if isinstance(ml, ps.Model):
            # Use type().__name__ instead of protected _name attribute
            smtyps = [type(sm).__name__ for sm in ml.stressmodels.values()]
        elif isinstance(ml, dict):
            classkey = "class"
            smtyps = [sm[classkey] for sm in ml["stressmodels"].values()]
        else:
            raise TypeError("Expected pastas.Model or dict!")
        check = set(smtyps).issubset(supported_stressmodels)
        if not check:
            unsupported = set(smtyps) - set(supported_stressmodels)
            raise NotImplementedError(
                "PastaStore does not support storing models with the "
                f"following stressmodels: {unsupported}"
            )

    @staticmethod
    def check_model_series_names_duplicates(ml):
        """Check for duplicate series names in the model."""
        prec_evap_model = ["RechargeModel", "TarsoModel"]

        if isinstance(ml, ps.Model):
            series_names = [
                istress.series.name
                for sm in ml.stressmodels.values()
                for istress in sm.stress
            ]

        elif isinstance(ml, dict):
            # non RechargeModel, Tarsomodel, WellModel stressmodels
            classkey = "class"
            series_names = [
                sm["stress"]["name"]
                for sm in ml["stressmodels"].values()
                if sm[classkey] not in (prec_evap_model + ["WellModel"])
            ]

            # WellModel
            if isin(
                ["WellModel"],
                [i[classkey] for i in ml["stressmodels"].values()],
            ).any():
                series_names += [
                    istress["name"]
                    for sm in ml["stressmodels"].values()
                    if sm[classkey] in ["WellModel"]
                    for istress in sm["stress"]
                ]

            # RechargeModel, TarsoModel
            if isin(
                prec_evap_model,
                [i[classkey] for i in ml["stressmodels"].values()],
            ).any():
                series_names += [
                    istress["name"]
                    for sm in ml["stressmodels"].values()
                    if sm[classkey] in prec_evap_model
                    for istress in [sm["prec"], sm["evap"]]
                ]

        else:
            raise TypeError("Expected pastas.Model or dict!")
        if len(series_names) - len(set(series_names)) > 0:
            msg = (
                "There are multiple stresses series with the same name! "
                "Each series name must be unique for the PastaStore!"
            )
            raise ValueError(msg)

    def check_oseries_in_store(self, ml: Union[ps.Model, dict]):
        """Check if Model oseries are contained in PastaStore (internal method).

        Parameters
        ----------
        ml : Union[ps.Model, dict]
            pastas Model
        """
        if isinstance(ml, ps.Model):
            name = ml.oseries.name
        elif isinstance(ml, dict):
            name = str(ml["oseries"]["name"])
        else:
            raise TypeError("Expected pastas.Model or dict!")
        if name not in self.connector.oseries.index:
            msg = (
                f"Cannot add model because oseries '{name}' is not contained in store."
            )
            raise LookupError(msg)
        # expensive check
        if self.CHECK_MODEL_SERIES_VALUES and isinstance(ml, ps.Model):
            s_org = self.connector.get_oseries(name).squeeze().dropna()
            # Access to _series_original is necessary for validation with Pastas models
            so = ml.oseries._series_original  # noqa: SLF001
            try:
                assert_series_equal(
                    so.dropna(),
                    s_org,
                    atol=self.SERIES_EQUALITY_ABSOLUTE_TOLERANCE,
                    rtol=self.SERIES_EQUALITY_RELATIVE_TOLERANCE,
                )
            except AssertionError as e:
                raise ValueError(
                    f"Cannot add model because model oseries '{name}'"
                    " is different from stored oseries! See stacktrace for differences."
                ) from e

    def check_stresses_in_store(self, ml: Union[ps.Model, dict]):
        """Check if stresses time series are contained in PastaStore (internal method).

        Parameters
        ----------
        ml : Union[ps.Model, dict]
            pastas Model
        """
        prec_evap_model = ["RechargeModel", "TarsoModel"]
        if isinstance(ml, ps.Model):
            for sm in ml.stressmodels.values():
                # Check class name using type instead of protected _name attribute
                if type(sm).__name__ in prec_evap_model:
                    stresses = [sm.prec, sm.evap]
                else:
                    stresses = sm.stress
                for s in stresses:
                    if str(s.name) not in self.connector.stresses.index:
                        msg = (
                            f"Cannot add model because stress '{s.name}' "
                            "is not contained in store."
                        )
                        raise LookupError(msg)
                    if self.CHECK_MODEL_SERIES_VALUES:
                        s_org = self.connector.get_stresses(s.name).squeeze()
                        # Access to _series_original needed for Pastas validation
                        so = s._series_original  # noqa: SLF001
                        try:
                            assert_series_equal(
                                so,
                                s_org,
                                atol=self.SERIES_EQUALITY_ABSOLUTE_TOLERANCE,
                                rtol=self.SERIES_EQUALITY_RELATIVE_TOLERANCE,
                            )
                        except AssertionError as e:
                            raise ValueError(
                                f"Cannot add model because model stress "
                                f"'{s.name}' is different from stored stress! "
                                "See stacktrace for differences."
                            ) from e
        elif isinstance(ml, dict):
            for sm in ml["stressmodels"].values():
                classkey = "class"
                if sm[classkey] in prec_evap_model:
                    stresses = [sm["prec"], sm["evap"]]
                elif sm[classkey] in ["WellModel"]:
                    stresses = sm["stress"]
                else:
                    stresses = [sm["stress"]]
                for s in stresses:
                    if str(s["name"]) not in self.connector.stresses.index:
                        msg = (
                            f"Cannot add model because stress '{s['name']}' "
                            "is not contained in store."
                        )
                        raise LookupError(msg)
        else:
            raise TypeError("Expected pastas.Model or dict!")

    def check_config_connector_type(self, path: str) -> None:
        """Check if config file connector type matches connector instance.

        Parameters
        ----------
        path : str
            path to directory containing the pastastore config file
        """
        if path.exists() and path.is_dir():
            config_file = list(path.glob("*.pastastore"))
            if len(config_file) > 0:
                with config_file[0].open("r", encoding="utf-8") as f:
                    cfg = json.load(f)
                stored_connector_type = cfg.pop("connector_type")
                if stored_connector_type != self.connector.conn_type:
                    # NOTE: delete _arctic_cfg that is created on ArcticDB init
                    if self.connector.conn_type == "arcticdb":
                        shutil.rmtree(path.parent / "_arctic_cfg")
                    raise ValueError(
                        f"Directory '{self.connector.name}/' in use by another "
                        f"connector type! Either create a '{stored_connector_type}' "
                        "connector to load the current pastastore or change the "
                        f"directory name to create a new '{self.connector.conn_type}' "
                        "connector."
                    )

    def check_series_in_models(self, libname, name):
        """Check if time series is used in any model (internal method).

        Parameters
        ----------
        libname : str
            library name ('oseries' or 'stresses')
        name : str
            name of the time series
        """
        msg = (
            "{libname} '{name}' is used in {n_models} model(s)! Either "
            "delete model(s) first, or use force=True."
        )
        if libname == "oseries":
            if name in self.connector.oseries_models:
                n_models = len(self.connector.oseries_models[name])
                raise SeriesUsedByModel(
                    msg.format(libname=libname, name=name, n_models=n_models)
                )
        elif libname == "stresses":
            if name in self.connector.stresses_models:
                n_models = len(self.connector.stresses_models[name])
                raise SeriesUsedByModel(
                    msg.format(libname=libname, name=name, n_models=n_models)
                )
