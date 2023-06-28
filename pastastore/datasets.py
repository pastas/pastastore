import os

import pandas as pd
from hydropandas import Obs, ObsCollection

try:
    from pastas.timeseries_utils import timestep_weighted_resample
except ModuleNotFoundError:
    from pastas.utils import timestep_weighted_resample

import pastastore as pst
from pastastore.base import BaseConnector


def example_pastastore(conn="DictConnector"):
    """Example dataset loaded into PastaStore.

    Parameters
    ----------
    conn : str or Connector, optional
        name of Connector type, by default "DictConnector", which
        initializes a default Connector. If an Connector instance is passed,
        use that Connector.

    Returns
    -------
    pstore : pastastore.PastaStore
        PastaStore containing example dataset
    """

    # check it test dataset is available
    datadir = os.path.join(os.path.dirname(__file__), "../tests/data")
    if not os.path.exists(datadir):
        raise FileNotFoundError(
            "Test datasets not available! Clone repository from GitHub."
        )

    # initialize default connector if conn is str
    if not isinstance(conn, BaseConnector):
        conn = _default_connector(conn)

    # initialize PastaStore
    pstore = pst.PastaStore(conn, "example")

    # add data

    # oseries 1
    o = pd.read_csv(os.path.join(datadir, "obs.csv"), index_col=0, parse_dates=True)
    pstore.add_oseries(o, "oseries1", metadata={"x": 165000, "y": 424000})
    # oseries 2
    o = pd.read_csv(
        os.path.join(datadir, "head_nb1.csv"), index_col=0, parse_dates=True
    )
    pstore.add_oseries(o, "oseries2", metadata={"x": 164000, "y": 423000})

    # oseries 3
    o = pd.read_csv(os.path.join(datadir, "gw_obs.csv"), index_col=0, parse_dates=True)
    pstore.add_oseries(o, "oseries3", metadata={"x": 165554, "y": 422685})

    # prec 1
    s = pd.read_csv(os.path.join(datadir, "rain.csv"), index_col=0, parse_dates=True)
    pstore.add_stress(s, "prec1", kind="prec", metadata={"x": 165050, "y": 424050})

    # prec 2
    s = pd.read_csv(
        os.path.join(datadir, "rain_nb1.csv"), index_col=0, parse_dates=True
    )
    pstore.add_stress(s, "prec2", kind="prec", metadata={"x": 164010, "y": 423000})

    # evap 1
    s = pd.read_csv(os.path.join(datadir, "evap.csv"), index_col=0, parse_dates=True)
    pstore.add_stress(s, "evap1", kind="evap", metadata={"x": 164500, "y": 424000})

    # evap 2
    s = pd.read_csv(
        os.path.join(datadir, "evap_nb1.csv"), index_col=0, parse_dates=True
    )
    pstore.add_stress(s, "evap2", kind="evap", metadata={"x": 164000, "y": 423030})

    # well 1
    s = pd.read_csv(os.path.join(datadir, "well.csv"), index_col=0, parse_dates=True)
    s = timestep_weighted_resample(s, pd.date_range(s.index[0], s.index[-1], freq="D"))
    pstore.add_stress(s, "well1", kind="well", metadata={"x": 164691, "y": 423579})

    # river notebook data (nb5)
    oseries = pd.read_csv(
        os.path.join(datadir, "nb5_head.csv"), parse_dates=True, index_col=0
    ).squeeze("columns")
    pstore.add_oseries(oseries, "head_nb5", metadata={"x": 200_000, "y": 450_000.0})

    rain = pd.read_csv(
        os.path.join(datadir, "nb5_prec.csv"), parse_dates=True, index_col=0
    ).squeeze("columns")
    pstore.add_stress(
        rain, "prec_nb5", kind="prec", metadata={"x": 200_000, "y": 450_000.0}
    )
    evap = pd.read_csv(
        os.path.join(datadir, "nb5_evap.csv"), parse_dates=True, index_col=0
    ).squeeze("columns")
    pstore.add_stress(
        evap, "evap_nb5", kind="evap", metadata={"x": 200_000, "y": 450_000.0}
    )
    waterlevel = pd.read_csv(
        os.path.join(datadir, "nb5_riv.csv"), parse_dates=True, index_col=0
    ).squeeze("columns")
    pstore.add_stress(
        waterlevel,
        "riv_nb5",
        kind="riv",
        metadata={"x": 200_000, "y": 450_000.0},
    )
    # TODO: temporary fix for older version of hydropandas that does not
    # read Menyanthes time series names correctly.
    # multiwell notebook data
    fname = os.path.join(datadir, "MenyanthesTest.men")
    # meny = ps.read.MenyData(fname)
    meny = ObsCollection.from_menyanthes(fname, Obs)

    oseries = meny.loc["Obsevation well", "obs"]
    ometa = {
        "x": oseries.meta["x"],
        "y": oseries.meta["y"],
    }
    pstore.add_oseries(oseries.dropna(), "head_mw", metadata=ometa)

    prec = meny.loc["Precipitation", "obs"]
    prec.index = prec.index.round("D")
    prec.name = "prec"
    pmeta = {
        "x": prec.meta["x"],
        "y": prec.meta["y"],
    }
    pstore.add_stress(prec, "prec_mw", kind="prec", metadata=pmeta)
    evap = meny.loc["Evaporation", "obs"]
    evap.index = evap.index.round("D")
    evap.name = "evap"
    emeta = {
        "x": evap.meta["x"],
        "y": evap.meta["y"],
    }
    pstore.add_stress(evap, "evap_mw", kind="evap", metadata=emeta)

    pressure = meny.loc["Air Pressure", "obs"]
    pressure.index = pressure.index.round("D")
    pressure.name = "pressure"
    pres_meta = {
        "x": pressure.meta["x"],
        "y": pressure.meta["y"],
    }
    pstore.add_stress(pressure, "pressure_mw", kind="pressure", metadata=pres_meta)

    extraction_names = [
        "Extraction 1",
        "Extraction 2",
        "Extraction 3",
        "Extraction 4",
    ]
    for extr in extraction_names:
        ts = meny.loc[extr, "obs"]
        wmeta = {"x": ts.meta["x"], "y": ts.meta["y"]}
        # replace spaces in names for Pastas
        name = extr.replace(" ", "_").lower()
        # resample to daily timestep
        ts_d = timestep_weighted_resample(
            ts, pd.date_range(ts.index[0], ts.index[-1], freq="D")
        )
        pstore.add_stress(ts_d, name, kind="well", metadata=wmeta)

    return pstore


def _default_connector(conntype: str):
    """Get default connector based on name.

    Parameters
    ----------
    conntype : str
        name of connector (DictConnector, PasConnector,
        ArcticConnector, ArcticDBConnector or PystoreConnector)

    Returns
    -------
    conn : Connector
        default Connector based on type.
    """
    Conn = getattr(pst, conntype)
    if Conn.conn_type == "arctic":
        connstr = "mongodb://localhost:27017/"
        conn = Conn("my_db", connstr)
    elif Conn.conn_type == "arcticdb":
        uri = "lmdb://./arctic_db"
        conn = Conn("my_db", uri)
    elif Conn.conn_type == "pystore":
        conn = Conn("my_db", "./pystore_db")
    elif Conn.conn_type == "dict":
        conn = Conn("my_db")
    elif Conn.conn_type == "pas":
        conn = Conn("my_db", "./pas_db")
    else:
        raise ValueError(f"Unrecognized connector type! '{conntype}'")
    return conn
