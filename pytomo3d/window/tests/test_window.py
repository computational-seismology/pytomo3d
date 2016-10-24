import os
import inspect
import matplotlib as mpl
import matplotlib.pyplot as plt
from obspy import read, read_inventory, readEvents
from pyflex.window import Window
import pytomo3d.window.window as win
import pytomo3d.window.io as wio
import json
from pyflex import WindowSelector


def _upper_level(path, nlevel=4):
    """
    Go the nlevel dir up
    """
    for i in range(nlevel):
        path = os.path.dirname(path)
    return path


def reset_matplotlib():
    """
    Reset matplotlib to a common default.
    """
    # Set all default values.
    mpl.rcdefaults()
    # Force agg backend.
    plt.switch_backend('agg')
    # These settings must be hardcoded for running the comparision tests and
    # are not necessarily the default values.
    mpl.rcParams['font.family'] = 'Bitstream Vera Sans'
    mpl.rcParams['text.hinting'] = False
    # Not available for all matplotlib versions.
    try:
        mpl.rcParams['text.hinting_factor'] = 8
    except KeyError:
        pass
    import locale
    locale.setlocale(locale.LC_ALL, str('en_US.UTF-8'))


# Most generic way to get the data folder path.
TESTBASE_DIR = _upper_level(
    os.path.abspath(inspect.getfile(inspect.currentframe())), 4)
print TESTBASE_DIR
DATA_DIR = os.path.join(TESTBASE_DIR, "tests", "data")

obsfile = os.path.join(DATA_DIR, "proc", "IU.KBL.obs.proc.mseed")
synfile = os.path.join(DATA_DIR, "proc", "IU.KBL.syn.proc.mseed")
staxml = os.path.join(DATA_DIR, "stationxml", "IU.KBL.xml")
quakeml = os.path.join(DATA_DIR, "quakeml", "C201009031635A.xml")


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
    config = wio.load_window_config_yaml(config_file)

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


def test_window_on_stream():
    obs_tr = read(obsfile)
    syn_tr = read(synfile)

    config_file = os.path.join(DATA_DIR, "window", "27_60.BHZ.config.yaml")
    config = wio.load_window_config_yaml(config_file)
    config_dict = {"Z": config, "R": config, "T": config}

    config_file = os.path.join(DATA_DIR, "window", "27_60.BHZ.config.yaml")
    config = wio.load_window_config_yaml(config_file)

    cat = readEvents(quakeml)
    inv = read_inventory(staxml)

    windows = win.window_on_stream(obs_tr, syn_tr, config_dict, station=inv,
                                   event=cat, _verbose=False,
                                   figure_mode=False)
    assert len(windows) > 0


def test_plot_window_figure(tmpdir):
    reset_matplotlib()

    obs_tr = read(obsfile).select(channel="*R")[0]
    syn_tr = read(synfile).select(channel="*R")[0]

    config_file = os.path.join(DATA_DIR, "window", "27_60.BHZ.config.yaml")
    config = wio.load_window_config_yaml(config_file)

    cat = readEvents(quakeml)
    inv = read_inventory(staxml)

    ws = WindowSelector(obs_tr, syn_tr, config, event=cat, station=inv)
    windows = ws.select_windows()

    assert len(windows) > 0

    win.plot_window_figure(str(tmpdir), obs_tr.id, ws, True,
                           figure_format="png")
