import os
import inspect
from obspy import read, read_inventory, readEvents
import pyflex
from pyflex.window import Window
import pytomo3d.window.window as win
import json


def _upper_level(path, nlevel=4):
    """
    Go the nlevel dir up
    """
    for i in range(nlevel):
        path = os.path.dirname(path)
    return path


# Most generic way to get the data folder path.
TESTBASE_DIR = _upper_level(
    os.path.abspath(inspect.getfile(inspect.currentframe())), 4)
print TESTBASE_DIR
DATA_DIR = os.path.join(TESTBASE_DIR, "tests", "data")

obsfile = os.path.join(DATA_DIR, "proc", "IU.KBL.obs.proc.mseed")
synfile = os.path.join(DATA_DIR, "proc", "IU.KBL.syn.proc.mseed")
staxml = os.path.join(DATA_DIR, "stationxml", "IU.KBL.xml")
quakeml = os.path.join(DATA_DIR, "quakeml", "C201009031635A.xml")


def test_read_config():
    config_file = os.path.join(DATA_DIR, "window", "27_60.BHZ.config.yaml")
    config = win.load_window_config_yaml(config_file)
    assert isinstance(config, pyflex.Config)
    assert config.max_period == 60.0
    assert config.min_period == 27.0
    assert config.stalta_waterlevel == 0.10


def test_read_window_json():
    winfile_bm = os.path.join(DATA_DIR, "window",
                              "IU.KBL..BHR.window.json")
    with open(winfile_bm) as fh:
        windows_json = json.load(fh)
    for _win_json_bm in windows_json:
        Window._load_from_json_content(_win_json_bm)


def test_window_on_trace():
    obs_tr = read(obsfile).select(channel="*R")[0]
    syn_tr = read(synfile).select(channel="*R")[0]

    config_file = os.path.join(DATA_DIR, "window", "27_60.BHZ.config.yaml")
    config = win.load_window_config_yaml(config_file)

    cat = readEvents(quakeml)

    inv = read_inventory(staxml)
    windows = win.window_on_trace(obs_tr, syn_tr, config, station=inv,
                                  event=cat, _verbose=False,
                                  figure_mode=False)

    winfile_bm = os.path.join(DATA_DIR, "window",
                              "IU.KBL..BHR.window.json")
    with open(winfile_bm) as fh:
        windows_json = json.load(fh)
    for _win, _win_json_bm in zip(windows, windows_json):
        _win_bm = Window._load_from_json_content(_win_json_bm)
        assert _win == _win_bm
