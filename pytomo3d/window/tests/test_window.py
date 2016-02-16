import os
import inspect
from obspy import read, read_inventory, readEvents
import pyflex
from pyflex.window import Window
import pytomo3d.window.window as win
import json
from pytomo3d.window.write_window import write_jsonfile

# Most generic way to get the data folder path.
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(
    inspect.getfile(inspect.currentframe()))), "data")
obsfile = os.path.join(DATA_DIR, "IU.KBL.obs.proc.mseed")
synfile = os.path.join(DATA_DIR, "IU.KBL.syn.proc.mseed")
staxml = os.path.join(DATA_DIR, "IU.KBL.xml")
inv = read_inventory(staxml)


def test_read_config():
    config_file = os.path.join(DATA_DIR, "27_60.BHZ.config.yaml")
    config = win.load_window_config_yaml(config_file)
    assert isinstance(config, pyflex.Config)
    assert config.max_period == 60.0
    assert config.min_period == 27.0
    assert config.stalta_waterlevel == 0.10


def test_window_on_trace():
    obs_tr = read(obsfile).select(channel="*R")[0]
    syn_tr = read(synfile).select(channel="*R")[0]

    config_file = os.path.join(DATA_DIR, "27_60.BHZ.config.yaml")
    config = win.load_window_config_yaml(config_file)

    quakeml = os.path.join(DATA_DIR, "C201009031635A.xml")
    cat = readEvents(quakeml)

    windows = win.window_on_trace(obs_tr, syn_tr, config, station=inv,
                                  event=cat, _verbose=False,
                                  figure_mode=False)

    winfile_bm = os.path.join(DATA_DIR, "benchmark",
                              "IU.KBL..BHR.window.json")
    with open(winfile_bm) as fh:
        windows_json = json.load(fh)
    for _win, _win_json_bm in zip(windows, windows_json):
        _win_bm = Window._load_from_json_content(_win_json_bm)
        assert _win == _win_bm
