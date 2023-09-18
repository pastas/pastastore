import datetime
import logging
import os
from copy import deepcopy
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
import pastas as ps
import yaml

from pastastore.version import PASTAS_LEQ_022

ps.logger.setLevel("ERROR")

logging.basicConfig(level="INFO")
logger = logging.getLogger(__name__)


def _convert_dict_dtypes_for_yaml(d: Dict[str, Any]):
    """Internal method to convert dictionary values for storing in YAML format.

    Parameters
    ----------
    d : dict
        dictionary to parse iteratively
    """
    for k, v in d.items():
        if isinstance(v, dict):
            _convert_dict_dtypes_for_yaml(v)
        elif isinstance(v, list) and k == "stress":
            for iv in v:
                if isinstance(iv, dict):
                    _convert_dict_dtypes_for_yaml(iv)
        elif isinstance(v, pd.Timestamp):
            d[k] = v.strftime("%Y-%m-%d %H:%M:%S")
        elif isinstance(v, datetime.datetime):
            d[k] = pd.to_datetime(v).strftime("%Y-%m-%d %H:%M:%S")
        elif isinstance(v, pd.Timedelta):
            d[k] = v.to_timedelta64().__str__()
        elif isinstance(v, datetime.timedelta):
            d[k] = pd.to_timedelta(v).to_timedelta64().__str__()
        elif isinstance(v, np.int64):
            d[k] = int(v)
        elif isinstance(v, np.float64):
            d[k] = float(v)
        elif isinstance(v, pd.DataFrame):
            d[k] = v.reset_index().to_dict(orient="records")


def replace_ts_with_name(d, nearest=False):
    """Replace time series dict with its name in pastas model dict.

    Parameters
    ----------
    d : dict
        pastas model dictionary
    nearest : bool, optional
        replace time series with "nearest" option. Warning, this does not
        check whether the time series are actually the nearest ones!
    """
    for k, v in d.items():
        if k in ["oseries", "prec", "evap", "stress"]:
            if isinstance(v, dict):
                if nearest and k != "oseries":
                    if k == "stress":
                        d[k] = "nearest <kind>"
                    else:
                        d[k] = f"nearest {k}"
                else:
                    d[k] = v["name"]
            elif isinstance(v, list):
                if nearest:
                    d[k] = f"nearest {len(v)} well"
                else:
                    d[k] = [iv["name"] for iv in v]
        elif isinstance(v, dict):
            replace_ts_with_name(v, nearest=nearest)


def reduce_to_minimal_dict(d, keys=None):
    """Reduce pastas model dictionary to a minimal form.

    This minimal form strives to keep the minimal information that still
    allows a model to be constructed. Users are warned, reducing a model
    dictionary with this function can lead to a different model than
    the original!

    Parameters
    ----------
    d : dict
        pastas model in dictionary form
    keys : list, optional
        list of keys to keep, by default None, which defaults to:
        ["name", "oseries", "settings", "tmin", "tmax", "noise",
        "stressmodels", "rfunc", "stress", "prec", "evap", "stressmodel"]
    """

    if keys is None:
        keys = [
            "name",
            "oseries",
            "settings",
            "tmin",
            "tmax",
            "noise",
            "stressmodels",
            "rfunc",
            "stress",
            "prec",
            "evap",
            "stressmodel" if PASTAS_LEQ_022 else "class",
        ]

    # also keep stressmodels by adding names to keys list
    if "stressmodels" in d:
        keys += list(d["stressmodels"].keys())

    iter_d = d.copy()  # copy dictionary to use as iterator

    # delete keys if not in keys list or if value is None
    for k, v in iter_d.items():
        if (k not in keys) or (v is None):
            del d[k]
        elif isinstance(v, dict):
            reduce_to_minimal_dict(v, keys=keys)


class PastastoreYAML:
    """Class for reading/writing Pastas models in YAML format.

    This class provides a more human-readable form of Pastas models in
    comparison to Pastas default .pas (JSON) files. The goal is to provide
    users with a simple mini-language to quickly build/test different model
    structures. A PastaStore is required as input, which contains existing
    models or time series required to build new models. This
    class also introduces some shortcuts to simplify building models.
    Shortcuts include the option to pass 'nearest' as the name of a stress,
    which will automatically select the closest stress of a particular type.
    Other shortcuts include certain default options when certain information
    is not listed in the YAML file, that will work well in many cases.

    Usage
    -----
    Instantiate the PastastoreYAML class::

        pyaml = PastastoreYAML(pstore)

    Export a Pastas model to a YAML file::

        pyaml.export_model_to_yaml(ml)

    Load a Pastas model from a YAML file::

        models = pyaml.load_yaml("my_first_model.yaml")

    Example YAML file using 'nearest'::

      my_first_model:  # this is the name of the model
        oseries: "oseries1"  # name of oseries stored in PastaStore
        stressmodels:
            recharge:  # recognized as RechargeModel by name
              prec: "nearest"  # use nearest stress with kind="prec"
              evap: "EV24_DEELEN"  # specific station
            river:
              stress: "nearest riv"  # nearest stress with kind="riv"
            wells:
              stress: "nearest 3"  # nearest 3 stresses with kind="well"
              stressmodel: WellModel  # provide StressModel type
    """

    def __init__(self, pstore):
        """Constructor for PastasstoreYAML class.

        Parameters
        ----------
        pstore : pastastore.PastaStore
            PastaStore object containing models to be exported as YAML files
            or containing time series that are referenced in YAML files.
        """
        self.pstore = pstore

    def _parse_rechargemodel_dict(self, d: Dict, onam: Optional[str] = None) -> Dict:
        """Internal method to parse RechargeModel dictionary.

        Note: supports 'nearest' as input to 'prec' and 'evap',
        which will automatically select nearest stress with kind="prec" or
        "evap". Requires "x" and "y" locations to be present in both oseries
        and stresses metadata.

        Parameters
        ----------
        d : dict
            dictionary containing RechargeModel information
        onam : str, optional
            name of oseries used when 'nearest' is provided as prec or evap,
            by default None

        Returns
        -------
        d : dict
            dictionary that can be read by ps.io.base.load(),
            containing stresses obtained from PastaStore, and setting
            defaults if they were not already provided.
        """
        # precipitation
        prec_val = d.get("prec", "nearest")
        if isinstance(prec_val, dict):
            pnam = prec_val["name"]
            p = self.pstore.get_stresses(pnam)
            prec_val["series"] = p
            prec = prec_val
        elif prec_val.startswith("nearest"):
            if onam is None:
                raise ValueError("Provide oseries name when using nearest!")
            if len(prec_val.split()) > 1:
                kind = prec_val.split()[-1]
            else:
                kind = "prec"
            pnam = self.pstore.get_nearest_stresses(onam, kind=kind).iloc[0, 0]
            logger.info(f"  | using nearest stress with kind='{kind}': '{pnam}'")
            p, pmeta = self.pstore.get_stresses(pnam, return_metadata=True)
            prec = {
                "name": pnam,
                "settings": "prec",
                "metadata": pmeta,
                "series": p,
            }
        elif isinstance(prec_val, str):
            pnam = d["prec"]
            p, pmeta = self.pstore.get_stresses(pnam, return_metadata=True)
            prec = {
                "name": pnam,
                "settings": "prec",
                "metadata": pmeta,
                "series": p,
            }
        else:
            raise NotImplementedError(f"Could not parse prec value: '{prec_val}'")
        d["prec"] = prec

        # evaporation
        evap_val = d.get("evap", "nearest")
        if isinstance(evap_val, dict):
            enam = evap_val["name"]
            e = self.pstore.get_stresses(enam)
            evap_val["series"] = e
            evap = evap_val
        elif evap_val.startswith("nearest"):
            if onam is None:
                raise ValueError("Provide oseries name when using nearest!")
            if len(evap_val.split()) > 1:
                kind = evap_val.split()[-1]
            else:
                kind = "evap"
            enam = self.pstore.get_nearest_stresses(onam, kind=kind).iloc[0, 0]
            logger.info(f"  | using nearest stress with kind='{kind}': '{enam}'")
            e, emeta = self.pstore.get_stresses(enam, return_metadata=True)
            evap = {
                "name": enam,
                "settings": "evap",
                "metadata": emeta,
                "series": e,
            }
        elif isinstance(evap_val, str):
            enam = d["evap"]
            e, emeta = self.pstore.get_stresses(enam, return_metadata=True)
            evap = {
                "name": enam,
                "settings": "evap",
                "metadata": emeta,
                "series": e,
            }
        else:
            raise NotImplementedError(f"Could not parse evap value: '{evap_val}'")
        d["evap"] = evap

        # rfunc
        if "rfunc" not in d:
            logger.info("  | no 'rfunc' provided, using 'Exponential'")
        # for pastas >= 0.23.0, convert rfunc value to dictionary with 'class' key
        elif not isinstance(d["rfunc"], dict) and not PASTAS_LEQ_022:
            d["rfunc"] = {"class": d["rfunc"]}

        # stressmodel
        classkey = "stressmodel" if PASTAS_LEQ_022 else "class"
        if classkey not in d:
            d[classkey] = "RechargeModel"

        # recharge type (i.e. Linear, FlexModel, etc.)
        if ("recharge" not in d) and (d[classkey] == "RechargeModel"):
            logger.info("  | no 'recharge' type provided, using 'Linear'")
        # if pastas >= 0.23.0, recharge value must be dict with class key
        elif not isinstance(d["recharge"], dict) and not PASTAS_LEQ_022:
            d["recharge"] = {"class": d["recharge"]}

        # tarsomodel logic
        if d[classkey] == "TarsoModel":
            dmin = d.get("dmin", None)
            dmax = d.get("dmin", None)
            oseries = d.get("oseries", None)
            if ((dmin is None) or (dmax is None)) and (oseries is None):
                logger.info(
                    "  | no 'dmin/dmax' or 'oseries' provided,"
                    f" filling in 'oseries': '{onam}'"
                )
                d["oseries"] = onam

        if "oseries" in d:
            onam = d["oseries"]
            if isinstance(onam, str):
                o = self.pstore.get_oseries(onam)
                d["oseries"] = o

        return d

    def _parse_stressmodel_dict(self, d: Dict, onam: Optional[str] = None) -> Dict:
        """Internal method to parse StressModel dictionary.

        Note: supports 'nearest' or 'nearest <kind>' as input to 'stress',
        which will automatically select nearest stress with kind=<kind>.
        Requires "x" and "y" locations to be present in both oseries and
        stresses metadata.

        Parameters
        ----------
        d : dict
            dictionary containing WellModel information
        onam : str, optional
            name of oseries used when 'nearest <kind>' is provided as stress,
            by default None

        Returns
        -------
        d : dict
            dictionary that can be read by ps.io.base.load(),
            containing stresses obtained from PastaStore, and setting
            defaults if they were not already provided.
        """

        # get stress
        snam = d.pop("stress")

        # if list, obtain first and only entry
        if isinstance(snam, list):
            snam = snam[0]
        # if str, either name of single series or 'nearest <kind>'
        if snam.startswith("nearest"):
            if len(snam.split()) > 1:
                kind = snam.split()[-1]
            else:
                kind = None
            if kind == "oseries":
                snam = self.pstore.get_nearest_oseries(onam).iloc[0, 0]
            else:
                snam = self.pstore.get_nearest_stresses(onam, kind=kind).iloc[0, 0]
            logger.info(f"  | using nearest stress with kind='{kind}': {snam}")

        s, smeta = self.pstore.get_stresses(snam, return_metadata=True)
        s = {
            "name": snam,
            "settings": d.pop("settings", None),
            "metadata": smeta,
            "series": s,
        }
        d["stress"] = [s] if PASTAS_LEQ_022 else s

        # use stress name if not provided
        if "name" not in d:
            d["name"] = snam

        # rfunc
        if "rfunc" not in d:
            logger.info("  | no 'rfunc' provided, using 'Gamma'")
            d["rfunc"] = "Gamma" if PASTAS_LEQ_022 else {"class": "Gamma"}
        # for pastas >= 0.23.0, convert rfunc value to dictionary with 'class' key
        elif not isinstance(d["rfunc"], dict) and not PASTAS_LEQ_022:
            d["rfunc"] = {"class": d["rfunc"]}

        return d

    def _parse_wellmodel_dict(self, d: Dict, onam: Optional[str] = None) -> Dict:
        """Internal method to parse WellModel dictionary.

        Note: supports 'nearest' or 'nearest <number> <kind>' as input to
        'stress', which will automatically select nearest or <number> of
        nearest stress(es) with kind=<kind>. Requires "x" and "y" locations to
        be present in both oseries and stresses metadata.

        Parameters
        ----------
        d : dict
            dictionary containing WellModel information
        onam : str, optional
            name of oseries used when 'nearest' is provided as stress,
            by default None

        Returns
        -------
        d : dict
            dictionary that can be read by ps.io.base.load(),
            containing stresses obtained from PastaStore, and setting
            defaults if they were not already provided.
        """

        # parse stress
        snames = d.pop("stress")

        # if str, either name of single series or 'nearest <n> <kind>'
        if isinstance(snames, str):
            if snames.startswith("nearest"):
                if len(snames.split()) == 3:
                    n = int(snames.split()[1])
                    kind = snames.split()[2]
                elif len(snames.split()) == 2:
                    try:
                        n = int(snames.split()[1])
                    except ValueError:
                        raise ValueError(
                            f"Could not parse: '{snames}'! "
                            "When using option 'nearest' for WellModel,  "
                            "use 'nearest <n>' or 'nearest <n> <kind>'!"
                        )
                    kind = "well"
                elif len(snames.split()) == 1:
                    n = 1
                    kind = "well"
                snames = (
                    self.pstore.get_nearest_stresses(onam, kind=kind, n=n)
                    .iloc[0]
                    .values
                )
                logger.info(
                    f"  | using {n} nearest stress(es) with kind='{kind}': " f"{snames}"
                )
            else:
                snames = [snames]

        # get time series
        slist = []
        for snam in snames:
            s, smeta = self.pstore.get_stresses(snam, return_metadata=True)
            sdict = {
                "name": snam,
                "settings": "well",
                "metadata": smeta,
                "series": s,
            }
            slist.append(sdict)
        d["stress"] = slist

        # get distances
        if "distances" not in d:
            d["distances"] = self.pstore.get_distances(
                oseries=onam, stresses=snames
            ).values.squeeze()

        # use default name if not provided
        if "name" not in d:
            d["name"] = "wells"

        # rfunc
        if "rfunc" not in d:
            logger.info("  | no 'rfunc' provided, using 'HantushWellModel'")
            d["rfunc"] = (
                "HantushWellModel" if PASTAS_LEQ_022 else {"class": "HantushWellModel"}
            )
        # for pastas >= 0.23.0, convert rfunc value to dictionary with 'class' key
        elif not isinstance(d["rfunc"], dict) and not PASTAS_LEQ_022:
            d["rfunc"] = {"class": d["rfunc"]}

        if "up" not in d:
            logger.info(
                "  | no 'up' provided, set to 'False', "
                "(i.e. pumping rate is positive for extraction)."
            )
            d["up"] = False

        return d

    def construct_mldict(self, mlyml: dict, mlnam: str) -> dict:
        # get oseries + metadata
        if isinstance(mlyml["oseries"], dict):
            onam = str(mlyml["oseries"]["name"])
            _ = mlyml.pop("oseries")
        else:
            onam = str(mlyml.pop("oseries"))

        logger.info(f"Building model '{mlnam}' for oseries '{onam}'")
        o, ometa = self.pstore.get_oseries(onam, return_metadata=True)

        # create model to obtain default model settings
        ml = ps.Model(o, name=mlnam, metadata=ometa)
        mldict = ml.to_dict(series=True)

        # update with stored model settings
        if "settings" in mlyml:
            mldict["settings"].update(mlyml["settings"])

        # stressmodels
        for smnam, smyml in mlyml["stressmodels"].items():
            # set name if not provided
            if smyml is not None:
                name = smyml.get("name", smnam)
            else:
                name = smnam
            logger.info(f"| parsing stressmodel: '{name}'")

            # check whether smtyp is defined
            classkey = "stressmodel" if PASTAS_LEQ_022 else "class"
            if smyml is not None:
                if PASTAS_LEQ_022:
                    if "class" in smyml:
                        smyml["stressmodel"] = smyml.pop("class")
                if classkey in smyml:
                    smtyp = True
                else:
                    smtyp = False
            else:
                smtyp = False

            # check if RechargeModel based on name if smtyp not defined
            if (
                smnam.lower() in ["rch", "rech", "recharge", "rechargemodel"]
            ) and not smtyp:
                logger.info("| assuming RechargeModel based on stressmodel name.")
                # check if stressmodel dictionary is empty, create (nearly
                # empty) dict so defaults are used
                if smyml is None:
                    mlyml["stressmodels"][smnam] = {"name": "recharge"}
                    smyml = mlyml["stressmodels"][smnam]
                if "name" not in smyml:
                    smyml["name"] = smnam
                smtyp = smyml.get(classkey, "RechargeModel")
            else:
                # if no info is provided, raise error,
                # cannot make any assumptions for non-RechargeModels
                if smyml is None:
                    raise ValueError(
                        "Insufficient information " f"for stressmodel '{name}'!"
                    )
                # get stressmodel type, with default StressModel
                if classkey in smyml:
                    smtyp = smyml[classkey]
                else:
                    logger.info(
                        "| no stressmodel class type provided, " "using 'StressModel'"
                    )
                    smtyp = "StressModel"

            # parse dictionary based on smtyp
            if smtyp in ["RechargeModel", "TarsoModel"]:
                # parse RechargeModel
                sm = self._parse_rechargemodel_dict(smyml, onam=onam)

                # turn off constant for TarsoModel
                if smtyp == "TarsoModel":
                    mldict["constant"] = False
            elif smtyp == "StressModel":
                # parse StressModel
                sm = self._parse_stressmodel_dict(smyml, onam=onam)
            elif smtyp == "WellModel":
                # parse WellModel
                sm = self._parse_wellmodel_dict(smyml, onam=onam)
            else:
                raise NotImplementedError(
                    "PastaStore.yaml interface does " f"not (yet) support '{smtyp}'!"
                )

            # add to list
            smyml.update(sm)

        # update model dict w/ default settings with loaded data
        mldict.update(mlyml)

        # add name to dictionary if not already present
        if "name" not in mldict:
            mldict["name"] = mlnam

        # convert warmup and time_offset to panads.Timedelta
        if "warmup" in mldict["settings"]:
            mldict["settings"]["warmup"] = pd.Timedelta(mldict["settings"]["warmup"])
        if "time_offset" in mldict["settings"]:
            mldict["settings"]["time_offset"] = pd.Timedelta(
                mldict["settings"]["time_offset"]
            )
        return mldict

    def load(self, fyaml: str) -> List[ps.Model]:
        """Load Pastas YAML file.

        Note: currently supports RechargeModel, StressModel and WellModel.

        Parameters
        ----------
        fyaml : str
            path to file

        Returns
        -------
        models : list
            list containing pastas model(s)

        Raises
        ------
        ValueError
            if insufficient information is provided to construct pastas model
        NotImplementedError
            if unsupported stressmodel is encountered
        """

        with open(fyaml, "r") as f:
            yml = yaml.load(f, Loader=yaml.CFullLoader)

        models = []

        for mlnam in yml.keys():
            mlyml = yml[mlnam]

            mldict = self.construct_mldict(mlyml, mlnam)

            # load model
            ml = ps.io.base._load_model(mldict)
            models.append(ml)

        return models

    def export_stored_models_per_oseries(
        self,
        oseries: Optional[Union[List[str], str]] = None,
        outdir: Optional[str] = ".",
        minimal_yaml: Optional[bool] = False,
        use_nearest: Optional[bool] = False,
    ):
        """Export store models grouped per oseries (location) to YAML file(s).

        Note: The oseries names are used as file names.

        Parameters
        ----------
        oseries : list of str, optional
            list of oseries (location) names, by default None, which uses
            all stored oseries for which there are models.
        outdir : str, optional
            path to output directory, by default "." (current directory)
        minimal_yaml : bool, optional
            reduce yaml file to include the minimum amount of information
            that will still construct a model. Users are warned, using this
            option does not guarantee the same model will be constructed
            as the one that was exported! Default is False.
        use_nearest : bool, optional
            if True, replaces time series with "nearest <kind>", filling in
            kind where possible. Warning! This does not check whether
            the time series are actually the nearest ones! Only used
            when minimal_yaml=True. Default is False.
        """

        onames = self.pstore.conn._parse_names(oseries, "oseries")

        for onam in onames:
            try:
                onam = int(onam)
            except ValueError:
                pass
            # check if any models exist for oseries
            if onam not in self.pstore.oseries_models:
                continue

            model_list = self.pstore.get_models(
                self.pstore.oseries_models[onam],
                return_dict=True,
                squeeze=False,
            )

            model_dicts = {}
            for d in model_list:
                if minimal_yaml:
                    replace_ts_with_name(d, nearest=use_nearest)
                    reduce_to_minimal_dict(d)
                _convert_dict_dtypes_for_yaml(d)
                name = d.pop("name")
                model_dicts[name] = d

            with open(os.path.join(outdir, f"{onam}.yaml"), "w") as f:
                yaml.dump(model_dicts, f, Dumper=yaml.CDumper)

    def export_models(
        self,
        models: Optional[Union[List[ps.Model], List[Dict]]] = None,
        modelnames: Optional[Union[List[str], str]] = None,
        outdir: Optional[str] = ".",
        minimal_yaml: Optional[bool] = False,
        use_nearest: Optional[bool] = False,
        split: Optional[bool] = True,
        filename: Optional[str] = "pastas_models.yaml",
    ):
        """Export (stored) models to yaml file(s).

        Parameters
        ----------
        models : list of ps.Model or dict, optional
            pastas Models to write to yaml file(s), if not provided,
            uses modelnames to collect stored models to export.
        modelnames : list of str, optional
            list of model names to export, by default None, which uses
            all stored models.
        outdir : str, optional
            path to output directory, by default "." (current directory)
        minimal_yaml : bool, optional
            reduce yaml file to include the minimum amount of information
            that will still construct a model. Users are warned, using this
            option does not guarantee the same model will be constructed
            as the one that was exported! Default is False.
        use_nearest : bool, optional
            if True, replaces time series with "nearest <kind>", filling in
            kind where possible. Warning! This does not check whether
            the time series are actually the nearest ones! Only used
            when minimal_yaml=True. Default is False.
        split : bool, optional
            if True, split into separate yaml files, otherwise store all
            in the same file. The model names are used as file names.
        filename : str, optional
            filename for YAML file, only used if `split=False`
        """
        if models is None:
            modelnames = self.pstore.conn._parse_names(modelnames, "models")
            model_list = self.pstore.get_models(
                modelnames, return_dict=True, squeeze=False
            )
        else:
            model_list = [
                iml.to_dict(series=False) if isinstance(iml, ps.Model) else iml
                for iml in models
            ]

        # each model in separate file
        if split:
            for ml in model_list:
                self.export_model(ml, outdir=outdir, minimal_yaml=minimal_yaml)
        # all models in same file
        else:
            model_dicts = {}
            for d in model_list:
                if minimal_yaml:
                    replace_ts_with_name(d, nearest=use_nearest)
                    reduce_to_minimal_dict(d)
                _convert_dict_dtypes_for_yaml(d)
                name = d.pop("name")
                model_dicts[name] = d

            with open(os.path.join(outdir, filename), "w") as f:
                yaml.dump(model_dicts, f, Dumper=yaml.CDumper)

    @staticmethod
    def export_model(
        ml: Union[ps.Model, dict],
        outdir: Optional[str] = ".",
        minimal_yaml: Optional[bool] = False,
        use_nearest: Optional[bool] = False,
    ):
        """Write single pastas model to YAML file.

        Parameters
        ----------
        ml : ps.Model or dict
            pastas model instance or dictionary representing a pastas model
        outdir : str, optional
            path to output directory, by default "." (current directory)
        minimal_yaml : bool, optional
            reduce yaml file to include the minimum amount of information
            that will still construct a model. Users are warned, using this
            option does not guarantee the same model will be constructed
            as the one that was exported! Default is False.
        use_nearest : bool, optional
            if True, replaces time series with "nearest <kind>", filling in
            kind where possible. Warning! This does not check whether
            the time series are actually the nearest ones! Only used
            when minimal_yaml=True. Default is False.
        """
        if isinstance(ml, dict):
            name = ml["name"]
        else:
            name = ml.name
        with open(os.path.join(outdir, f"{name}.yaml"), "w") as f:
            if isinstance(ml, ps.Model):
                mldict = deepcopy(ml.to_dict(series=False))
            elif isinstance(ml, dict):
                mldict = ml
            else:
                raise TypeError("Only accepts dictionary or pastas.Model!")
            mlname = mldict.pop("name")
            if minimal_yaml:
                replace_ts_with_name(mldict, nearest=use_nearest)
                reduce_to_minimal_dict(mldict)
            _convert_dict_dtypes_for_yaml(mldict)
            yaml.dump({mlname: mldict}, f, Dumper=yaml.CDumper)
