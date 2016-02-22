import os
import inspect
import pyadjoint
from obspy import read
from pyflex.window import Window
import pytomo3d.adjoint.adjsrc as adj
import json
# from pytomo3d.adjoint.plot_util import plot_adjoint_source


def _upper_level(path, nlevel=4):
    """
    Go the nlevel dir up
    """
    for i in range(nlevel):
        path = os.path.dirname(path)
    return path

# Most generic way to get the data folder path.
TESTBASE_DIR = _upper_level(os.path.abspath(
    inspect.getfile(inspect.currentframe())), 4)
DATA_DIR = os.path.join(TESTBASE_DIR, "tests", "data")

# Most generic way to get the data folder path.
obsfile = os.path.join(DATA_DIR, "proc", "IU.KBL.obs.proc.mseed")
synfile = os.path.join(DATA_DIR, "proc", "IU.KBL.syn.proc.mseed")
winfile = os.path.join(DATA_DIR, "window", "IU.KBL..BHR.window.json")


def test_read_config():
    config_file = os.path.join(DATA_DIR, "adjoint",
                               "waveform.adjoint.config.yaml")
    config = adj.load_adjoint_config_yaml(config_file)
    assert isinstance(config, pyadjoint.Config)
    assert config.max_period == 60.0
    assert config.min_period == 27.0
    assert config.taper_percentage == 0.15
    assert config.taper_type == 'hann'
    assert not config.use_cc_error

    config_file = os.path.join(DATA_DIR, "adjoint",
                               "cc_traveltime.adjoint.config.yaml")
    config = adj.load_adjoint_config_yaml(config_file)
    assert isinstance(config, pyadjoint.Config)
    assert config.max_period == 60.0
    assert config.min_period == 27.0
    assert config.ipower_costaper == 10
    assert config.taper_percentage == 0.15
    assert config.taper_type == 'hann'
    assert config.use_cc_error

    config_file = os.path.join(DATA_DIR, "adjoint",
                               "multitaper.adjoint.config.yaml")
    config = adj.load_adjoint_config_yaml(config_file)
    assert isinstance(config, pyadjoint.Config)
    assert config.max_period == 60.0
    assert config.min_period == 27.0
    assert config.lnpt == 15
    assert config.transfunc_waterlevel == 1.0e-10
    assert config.ipower_costaper == 10
    assert config.min_cycle_in_window == 3
    assert config.taper_percentage == 0.3
    assert config.mt_nw == 4.0
    assert config.num_taper == 5
    assert config.phase_step == 1.5
    assert config.dt_fac == 2.0
    assert config.err_fac == 2.5
    assert config.dt_max_scale == 3.5
    assert config.measure_type == "dt"
    assert config.taper_type == 'hann'
    assert config.use_cc_error
    assert not config.use_mt_error


def test_waveform_adjoint():
    obs = read(obsfile).select(channel="*R")[0]
    syn = read(synfile).select(channel="*R")[0]

    with open(winfile) as fh:
        wins_json = json.load(fh)
    windows = []
    for _win in wins_json:
        windows.append(Window._load_from_json_content(_win))

    config_file = os.path.join(DATA_DIR, "adjoint",
                               "waveform.adjoint.config.yaml")
    config = adj.load_adjoint_config_yaml(config_file)

    win_time, _ = adj._extract_window_time(windows)

    adjsrc = adj.calculate_adjsrc_on_trace(
        obs, syn, win_time, config, adj_src_type="waveform_misfit",
        adjoint_src_flag=True, figure_mode=False)

    # tr_adj = adj._convert_adj_to_trace(adjsrc, syn.stats.starttime, syn.id)
    # tr.write("%s.sac" % syn.id, format="SAC")
    # plot_adjoint_source(adjsrc, win_time, obs, syn)

    assert adjsrc


def test_multitaper_adjoint():
    obs = read(obsfile).select(channel="*R")[0]
    syn = read(synfile).select(channel="*R")[0]

    with open(winfile) as fh:
        wins_json = json.load(fh)
    windows = []
    for _win in wins_json:
        windows.append(Window._load_from_json_content(_win))

    config_file = os.path.join(DATA_DIR, "adjoint",
                               "multitaper.adjoint.config.yaml")
    config = adj.load_adjoint_config_yaml(config_file)

    win_time, _ = adj._extract_window_time(windows)

    adjsrc = adj.calculate_adjsrc_on_trace(
        obs, syn, win_time, config, adj_src_type="multitaper_misfit",
        adjoint_src_flag=True, figure_mode=False)

    # tr_adj = adj._convert_adj_to_trace(adjsrc, syn.stats.starttime, syn.id)
    # tr.write("%s.sac" % syn.id, format="SAC")
    # plot_adjoint_source(adjsrc, win_time, obs, syn, figname="mt.png")
    assert adjsrc


def test_cc_traveltime_adjoint():
    obs = read(obsfile).select(channel="*R")[0]
    syn = read(synfile).select(channel="*R")[0]
    with open(winfile) as fh:
        wins_json = json.load(fh)
    windows = []
    for _win in wins_json:
        windows.append(Window._load_from_json_content(_win))

    config_file = os.path.join(DATA_DIR, "adjoint",
                               "cc_traveltime.adjoint.config.yaml")
    config = adj.load_adjoint_config_yaml(config_file)

    win_time, _ = adj._extract_window_time(windows)

    adjsrc = adj.calculate_adjsrc_on_trace(
        obs, syn, win_time, config, adj_src_type="cc_traveltime_misfit",
        adjoint_src_flag=True, figure_mode=False)

    # tr_adj = adj._convert_adj_to_trace(adjsrc, syn.stats.starttime, syn.id)
    # tr.write("%s.sac" % syn.id, format="SAC")
    # plot_adjoint_source(adjsrc, win_time, obs, syn, figname="cc.png")
    assert adjsrc


if __name__ == "__main__":
    test_waveform_adjoint()
    test_cc_traveltime_adjoint()
    test_multitaper_adjoint()
