import os
import inspect
import json
from obspy import read, Stream
from pyflex.window import Window
import pytomo3d.adjoint.adjsrc as adj
import pytest
import matplotlib.pyplot as plt
import pyadjoint.adjoint_source


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

obsfile = os.path.join(DATA_DIR, "proc", "IU.KBL.obs.proc.mseed")
synfile = os.path.join(DATA_DIR, "proc", "IU.KBL.syn.proc.mseed")
winfile = os.path.join(DATA_DIR, "window", "IU.KBL..BHR.window.json")


@pytest.fixture
def load_config_waveform():
    config_file = os.path.join(DATA_DIR, "adjoint",
                               "waveform.adjoint.config.yaml")
    return adj.load_adjoint_config_yaml(config_file)


@pytest.fixture
def load_config_traveltime():
    config_file = os.path.join(DATA_DIR, "adjoint",
                               "cc_traveltime.adjoint.config.yaml")
    return adj.load_adjoint_config_yaml(config_file)


@pytest.fixture
def load_config_multitaper():
    config_file = os.path.join(DATA_DIR, "adjoint",
                               "multitaper.adjoint.config.yaml")
    return adj.load_adjoint_config_yaml(config_file)


def test_load_adjoint_config_yaml_for_waveform_misfit():
    config = load_config_waveform()
    assert isinstance(config, pyadjoint.Config)
    assert config.max_period == 60.0
    assert config.min_period == 27.0
    assert config.taper_percentage == 0.15
    assert config.taper_type == 'hann'
    assert not config.use_cc_error


def test_load_adjoint_config_yaml_for_traveltime_misfit():
    config = load_config_traveltime()
    assert isinstance(config, pyadjoint.Config)
    assert config.max_period == 60.0
    assert config.min_period == 27.0
    assert config.ipower_costaper == 10
    assert config.taper_percentage == 0.15
    assert config.taper_type == 'hann'
    assert config.use_cc_error


def test_load_adjoint_config_yaml_for_multitaper_misfit():
    config = load_config_multitaper()
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


@pytest.fixture
def setup_calculate_adjsrc_on_trace_args():
    obs = read(obsfile).select(channel="*R")[0]
    syn = read(synfile).select(channel="*R")[0]

    with open(winfile) as fh:
        wins_json = json.load(fh)
    windows = []
    for _win in wins_json:
        windows.append(Window._load_from_json_content(_win))

    return obs, syn, windows


def test_calculate_adjsrc_on_trace_raises_if_obs_is_not_trace():
    obs, syn, win_time = setup_calculate_adjsrc_on_trace_args()
    config = load_config_multitaper()
    obs = []
    with pytest.raises(ValueError):
        adj.calculate_adjsrc_on_trace(obs, syn, win_time, config,
                                      adj_src_type="multitaper_misfit")


def test_calculate_adrjrc_on_trace_raises_if_syn_is_not_trace():
    obs, syn, win_time = setup_calculate_adjsrc_on_trace_args()
    config = load_config_multitaper()
    syn = []
    with pytest.raises(ValueError):
        adj.calculate_adjsrc_on_trace(obs, syn, win_time, config,
                                      adj_src_type="multitaper_misfit")


def test_calculate_adjsrc_on_trace_raises_if_config_is_not_config():
    obs, syn, win_time = setup_calculate_adjsrc_on_trace_args()
    config = []
    with pytest.raises(ValueError):
        adj.calculate_adjsrc_on_trace(obs, syn, win_time, config,
                                      adj_src_type="multitaper_misfit")


def test_calculate_adjsrc_on_trace_raises_bad_windows_shape():
    obs, syn, win_time = setup_calculate_adjsrc_on_trace_args()
    config = load_config_multitaper()
    win_time = []
    with pytest.raises(ValueError):
        adj.calculate_adjsrc_on_trace(obs, syn, win_time, config,
                                      adj_src_type="multitaper_misfit")


def test_calculate_adjsrc_on_trace_figure_mode_none_figure_dir():
    obs, syn, win_time = setup_calculate_adjsrc_on_trace_args()
    config = load_config_multitaper()
    plt.switch_backend('agg')
    adjsrc = adj.calculate_adjsrc_on_trace(
        obs, syn, win_time, config, adj_src_type="multitaper_misfit",
        figure_mode=True)
    assert adjsrc


def test_calculate_adjsrc_on_trace_waveform_misfit_produces_adjsrc():
    obs, syn, win_time = setup_calculate_adjsrc_on_trace_args()
    config = load_config_waveform()

    adjsrc = adj.calculate_adjsrc_on_trace(
        obs, syn, win_time, config, adj_src_type="waveform_misfit",
        adjoint_src_flag=True, figure_mode=False)
    assert adjsrc


def test_calculate_adjsrc_on_trace_multitaper_misfit_produces_adjsrc():
    obs, syn, win_time = setup_calculate_adjsrc_on_trace_args()
    config = load_config_multitaper()

    adjsrc = adj.calculate_adjsrc_on_trace(
        obs, syn, win_time, config, adj_src_type="multitaper_misfit",
        adjoint_src_flag=True, figure_mode=False)
    assert adjsrc


def test_calculate_adjsrc_on_trace_traveltime_misfit_produces_adjsrc():
    obs, syn, win_time = setup_calculate_adjsrc_on_trace_args()
    config = load_config_traveltime()

    adjsrc = adj.calculate_adjsrc_on_trace(
        obs, syn, win_time, config, adj_src_type="cc_traveltime_misfit",
        adjoint_src_flag=True, figure_mode=False)
    assert adjsrc


@pytest.fixture
def setup_calculate_adjsrc_on_stream_args():
    obs = Stream(traces=[read(obsfile).select(channel="*R")[0]])
    syn = Stream(traces=[read(synfile).select(channel="*R")[0]])

    with open(winfile) as fh:
        wins_json = json.load(fh)

    return obs, syn, {obs[0].id: wins_json}


def test_calculate_adjsrc_on_stream_raises_if_obs_is_not_stream():
    _, syn, windows = setup_calculate_adjsrc_on_stream_args()
    config = load_config_multitaper()
    obs = []
    with pytest.raises(ValueError):
        adj.calculate_adjsrc_on_stream(obs, syn, windows, config,
                                       adj_src_type="multitaper_misfit")


def test_calculate_adjsrc_on_stream_raises_if_syn_is_not_stream():
    obs, _, windows = setup_calculate_adjsrc_on_stream_args()
    config = load_config_multitaper()
    syn = []
    with pytest.raises(ValueError):
        adj.calculate_adjsrc_on_stream(obs, syn, windows, config,
                                       adj_src_type="multitaper_misfit")


def test_calculate_adjsrc_on_stream_raises_if_config_is_not_config():
    obs, syn, windows = setup_calculate_adjsrc_on_stream_args()
    config = []
    with pytest.raises(ValueError):
        adj.calculate_adjsrc_on_stream(obs, syn, windows, config,
                                       adj_src_type="multitaper_misfit")


def test_calculate_adjsrc_on_stream_raises_if_windows_is_empty():
    obs, syn, _ = setup_calculate_adjsrc_on_stream_args()
    config = load_config_multitaper()
    windows = None
    ret = adj.calculate_adjsrc_on_stream(obs, syn, windows, config,
                                         adj_src_type="multitaper_misfit")
    assert ret is None
    windows = {}
    ret = adj.calculate_adjsrc_on_stream(obs, syn, windows, config,
                                         adj_src_type="multitaper_misfit")
    assert ret is None


def test_calculate_adjsrc_on_stream_multitaper_misfit_produces_adjsrc():
    obs, syn, windows = setup_calculate_adjsrc_on_stream_args()
    config = load_config_traveltime()

    adjsrc = adj.calculate_adjsrc_on_stream(
        obs, syn, windows, config, adj_src_type="multitaper_misfit",
        adjoint_src_flag=True, figure_mode=False)
    assert adjsrc


def test_calculate_adjsrc_on_stream_waveform_misfit_produces_adjsrc():
    obs, syn, windows = setup_calculate_adjsrc_on_stream_args()
    config = load_config_traveltime()

    adjsrc = adj.calculate_adjsrc_on_stream(
        obs, syn, windows, config, adj_src_type="waveform_misfit",
        adjoint_src_flag=True, figure_mode=False)
    assert adjsrc


def test_calculate_adjsrc_on_stream_traveltime_misfit_produces_adjsrc():
    obs, syn, windows = setup_calculate_adjsrc_on_stream_args()
    config = load_config_traveltime()

    adjsrc = adj.calculate_adjsrc_on_stream(
        obs, syn, windows, config, adj_src_type="cc_traveltime_misfit",
        adjoint_src_flag=True, figure_mode=False)
    assert adjsrc
