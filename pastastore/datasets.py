import os

import pandas as pd
import pastas as ps

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
        raise FileNotFoundError("Test datasets not available! "
                                "Clone repository from GitHub.")

    # initialize default connector if conn is str
    if not isinstance(conn, BaseConnector):
        conn = _default_connector(conn)

    # initialize PastaStore
    pstore = pst.PastaStore("example", conn)

    # add data

    # oseries 1
    o = pd.read_csv(os.path.join(datadir, "obs.csv"),
                    index_col=0, parse_dates=True)
    pstore.add_oseries(o, "oseries1", metadata={"x": 165000,
                                                "y": 424000})
    # oseries 2
    o = pd.read_csv(os.path.join(datadir, "head_nb1.csv"),
                    index_col=0, parse_dates=True)
    pstore.add_oseries(o, "oseries2", metadata={"x": 164000,
                                                "y": 423000})

    # oseries 3
    o = pd.read_csv(os.path.join(datadir, "gw_obs.csv"),
                    index_col=0, parse_dates=True)
    pstore.add_oseries(o, "oseries3", metadata={"x": 165554,
                                                "y": 422685})

    # prec 1
    s = pd.read_csv(os.path.join(datadir, "rain.csv"),
                    index_col=0, parse_dates=True)
    pstore.add_stress(s, "prec1", kind="prec", metadata={"x": 165050,
                                                         "y": 424050})

    # prec 2
    s = pd.read_csv(os.path.join(datadir, "rain_nb1.csv"),
                    index_col=0, parse_dates=True)
    pstore.add_stress(s, "prec2", kind="prec", metadata={"x": 164010,
                                                         "y": 423000})

    # evap 1
    s = pd.read_csv(os.path.join(datadir, "evap.csv"),
                    index_col=0, parse_dates=True)
    pstore.add_stress(s, "evap1", kind="evap", metadata={"x": 164500,
                                                         "y": 424000})

    # evap 2
    s = pd.read_csv(os.path.join(datadir, "evap_nb1.csv"),
                    index_col=0, parse_dates=True)
    pstore.add_stress(s, "evap2", kind="evap", metadata={"x": 164000,
                                                         "y": 423030})

    # well 1
    s = pd.read_csv(os.path.join(datadir, "well.csv"),
                    index_col=0, parse_dates=True)
    pstore.add_stress(s, "well1", kind="well", metadata={"x": 164691,
                                                         "y": 423579})

    # river notebook data (nb5)
    oseries = pd.read_csv(os.path.join(datadir, "nb5_head.csv"),
                          parse_dates=True, index_col=0).squeeze("columns")
    pstore.add_oseries(oseries, "head_nb5", metadata={
        "x": 200_000, "y": 450_000.})

    rain = pd.read_csv(os.path.join(datadir, "nb5_prec.csv"), parse_dates=True,
                       index_col=0).squeeze("columns")
    pstore.add_stress(rain, "prec_nb5", kind="prec",
                      metadata={"x": 200_000, "y": 450_000.})
    evap = pd.read_csv(os.path.join(datadir, "nb5_evap.csv"), parse_dates=True,
                       index_col=0).squeeze("columns")
    pstore.add_stress(evap, "evap_nb5", kind="evap",
                      metadata={"x": 200_000, "y": 450_000.})
    waterlevel = pd.read_csv(os.path.join(datadir, "nb5_riv.csv"),
                             parse_dates=True, index_col=0).squeeze("columns")
    pstore.add_stress(waterlevel, "riv_nb5", kind="riv",
                      metadata={"x": 200_000, "y": 450_000.})

    # multiwell notebook data
    fname = os.path.join(datadir, 'MenyanthesTest.men')
    meny = ps.read.MenyData(fname)

    oseries = meny.H['Obsevation well']['values'].dropna()
    ometa = {"x": meny.H["Obsevation well"]['xcoord'],
             "y": meny.H["Obsevation well"]['ycoord']}
    pstore.add_oseries(oseries, "head_mw", metadata=ometa)

    prec = meny.IN['Precipitation']['values']
    prec.index = prec.index.round("D")
    prec.name = "prec"
    pmeta = {"x": meny.IN['Precipitation']['xcoord'],
             "y": meny.IN['Precipitation']['ycoord']}
    pstore.add_stress(prec, "prec_mw", kind="prec", metadata=pmeta)
    evap = meny.IN['Evaporation']['values']
    evap.index = evap.index.round("D")
    evap.name = "evap"
    emeta = {"x": meny.IN['Evaporation']['xcoord'],
             "y": meny.IN['Evaporation']['ycoord']}
    pstore.add_stress(evap, "evap_mw", kind="evap", metadata=emeta)

    extraction_names = ['Extraction 2', 'Extraction 3']
    for extr in extraction_names:
        wmeta = {"x": meny.IN[extr]["xcoord"],
                 "y": meny.IN[extr]["ycoord"]}
        # replace spaces in names for Pastas
        name = extr.replace(" ", "_").lower()
        ts = meny.IN[extr]["values"]
        pstore.add_stress(ts, name, kind="well", metadata=wmeta)

    return pstore


def _default_connector(conntype: str):
    """Get default connector based on name.

    Parameters
    ----------
    conntype : str
        name of connector (DictConnector, PasConnector,
        ArcticConnector or PystoreConnector)

    Returns
    -------
    conn : Connector
        default Connector based on type.
    """
    Conn = getattr(pst, conntype)
    if Conn.conn_type == "arctic":
        connstr = "mongodb://localhost:27017/"
        conn = Conn("my_db", connstr)
    elif Conn.conn_type == "pystore":
        conn = Conn("my_db", "./pystore_db")
    elif Conn.conn_type == "dict":
        conn = Conn("my_db")
    elif Conn.conn_type == "pas":
        conn = Conn("my_db", "./pas_db")
    else:
        raise ValueError("Unrecognized parameter!")
    return conn
