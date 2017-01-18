import os
import inspect
import pytest
import copy

import numpy as np
import numpy.testing as npt

import pytomo3d.window.filter_windows as fw
from pytomo3d.utils.io import load_json


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
DATA_DIR = os.path.join(TESTBASE_DIR, "tests", "data")

WINDOWFILE = os.path.join(DATA_DIR, "window", "windows.fake.json")
windows = load_json(WINDOWFILE)
MEASUREFILE = os.path.join(DATA_DIR, "window", "measurements.fake.json")
measures = load_json(MEASUREFILE)
STATIONFILE = os.path.join(DATA_DIR, "stations", "stations.fake.json")
stations = load_json(STATIONFILE)


def test_is_right_sensor():
    pools = ["STS-1", "STS1", "KS54000"]

    assert fw.is_right_sensor("STS1", pools)
    assert fw.is_right_sensor("STS-1", pools)
    assert not fw.is_right_sensor("KS5400", pools)


def test_count_windows():
    assert fw.count_windows({}) == (0, 0)
    wins = {"A": {"R": [], "T": {}, "Z": []}}
    assert fw.count_windows(wins) == (0, 0)
    wins = {"A": {"R": [1, 2, 3], "T": [1], "Z": [1, 2]}}
    assert fw.count_windows(wins) == (3, 6)
    assert fw.count_windows(windows) == (8, 18)


def test_filter_windows_on_sensor():
    sensor_types = ["STS-1"]
    new_wins = fw.filter_windows_on_sensors(windows, stations, sensor_types,
                                            verbose=False)
    assert len(new_wins) == 1
    assert len(new_wins["II.AAK"]) == 3
    assert new_wins["II.AAK"] == windows["II.AAK"]

    sensor_types = ["STS-1", "CMG3ESP"]
    new_wins = fw.filter_windows_on_sensors(windows, stations, sensor_types,
                                            verbose=False)
    assert len(new_wins) == 2
    assert len(new_wins["II.AAK"]) == 3
    assert new_wins["II.AAK"] == windows["II.AAK"]
    assert len(new_wins["IU.BCD"]) == 3
    assert new_wins["IU.BCD"] == windows["IU.BCD"]

    sensor_types = ["STS-1", "STS1"]
    new_wins = fw.filter_windows_on_sensors(windows, stations, sensor_types,
                                            verbose=False)
    assert len(new_wins) == 2
    assert len(new_wins["II.AAK"]) == 3
    assert new_wins["II.AAK"] == windows["II.AAK"]
    assert len(new_wins["II.ABKT"]) == 2
    assert new_wins["II.ABKT"]["II.ABKT..BHR"] == \
        windows["II.ABKT"]["II.ABKT..BHR"]
    assert new_wins["II.ABKT"]["II.ABKT..BHZ"] == \
        windows["II.ABKT"]["II.ABKT..BHZ"]
    assert "II.ABKT..BHT" not in new_wins["II.ABKT"]


def test_get_measurements_std():
    dt_means, dt_stds, dlna_means, dlna_stds = \
        fw.get_measurements_std({})
    assert len(dt_means) == 0
    assert len(dt_stds) == 0
    assert len(dlna_means) == 0
    assert len(dlna_stds) == 0

    dt_means, dt_stds, dlna_means, dlna_stds = \
        fw.get_measurements_std(measures)

    # from tests/data/window/measurements.fake.json
    _true_dt_mean = \
        {"R": np.mean([1, -1, 1, 1, -2]),
         "T": np.mean([1, 1.5, -2.5]),
         "Z": np.mean([1, 2, -1.5, 2, -5, -0.2, 0.8, -1.6, 1.6, 0.9])}
    _true_dt_stds = \
        {"R": np.std([1, -1, 1, 1, -2]),
         "T": np.std([1, 1.5, -2.5]),
         "Z": np.std([1, 2, -1.5, 2, -5, -0.2, 0.8, -1.6, 1.6, 0.9])}

    _true_dlna_mean = \
        {"R": np.mean([0.7, -0.7, 0.6, 1.0, -0.8]),
         "T": np.mean([0.9, 0.3, -0.7]),
         "Z": np.mean([0.6, 0.4, -0.5, 1.2, -1.5, -0.2, 0.8, -0.6, 1.1, 0.9])}
    _true_dlna_stds = \
        {"R": np.std([0.7, -0.7, 0.6, 1.0, -0.8]),
         "T": np.std([0.9, 0.3, -0.7]),
         "Z": np.std([0.6, 0.4, -0.5, 1.2, -1.5, -0.2, 0.8, -0.6, 1.1, 0.9])}

    for comp in dt_means:
        npt.assert_array_almost_equal(dt_means[comp], _true_dt_mean[comp])
        npt.assert_array_almost_equal(dt_stds[comp], _true_dt_stds[comp])
        npt.assert_array_almost_equal(dlna_means[comp], _true_dlna_mean[comp])
        npt.assert_array_almost_equal(dlna_stds[comp], _true_dlna_stds[comp])


def test_get_user_bound():
    info = {"tshift_acceptance_level": 3, "tshift_reference": 0,
            "dlna_acceptance_level": 0.8, "dlna_reference": 0.0}
    v = fw.get_user_bound(info)
    npt.assert_array_almost_equal(v, [-3, 3, -0.8, 0.8])

    info = {"tshift_acceptance_level": 10, "tshift_reference": -1,
            "dlna_acceptance_level": 0.6, "dlna_reference": 0.2}
    v = fw.get_user_bound(info)
    npt.assert_array_almost_equal(v, [-11, 9, -0.4, 0.8])


def test_filter_measurements_on_bounds():
    bounds = {"R": [-1.1, 1.1, -0.8, 0.8],
              "T": [-1.1, 1.1, -0.8, 0.8],
              "Z": [-1.1, 1.1, -0.8, 0.8]}

    v, m = fw.filter_measurements_on_bounds(windows, measures, bounds)
    assert v["II.AAK"]["II.AAK..BHR"] == windows["II.AAK"]["II.AAK..BHR"]
    assert "II.AAK..BHT" not in v["II.AAK"]
    assert len(v["II.AAK"]["II.AAK..BHZ"]) == 1
    assert v["II.AAK"]["II.AAK..BHZ"] == [{"left_index": 1, "right_index": 2}]
    assert len(v["IU.BCD"]["IU.BCD..BHZ"]) == 2
    assert v["IU.BCD"]["IU.BCD..BHZ"] == [{"left_index": 1, "right_index": 2},
                                          {"left_index": 2, "right_index": 3}]

    assert m["II.AAK"] == {
        "II.AAK..BHR": [{"dt": 1.0, "misfit_dt": 1.0,
                         "dlna": 0.7, "misfit_dlna": 1.0},
                        {"dt": -1.0, "misfit_dt": 1.0,
                         "dlna": -0.7, "misfit_dlna": 1.0}],
        "II.AAK..BHZ": [{"dt": 1.0, "misfit_dt": 1.0,
                         "dlna": 0.6, "misfit_dlna": 1.0}]}
    assert m["II.ABKT"] == {
       "II.ABKT..BHR": [{"dt": 1.0, "misfit_dt": 1.5,
                         "dlna": 0.6, "misfit_dlna": 0.5}]}
    assert m["IU.BCD"] == {
           "IU.BCD..BHT": [{"dt": 1.0, "misfit_dt": 2.0,
                            "dlna": 0.3, "misfit_dlna": 0.2}],
           "IU.BCD..BHZ": [{"dt": -0.2, "misfit_dt": 2.0,
                            "dlna": -0.2, "misfit_dlna": 2.0},
                           {"dt": 0.8, "misfit_dt": 3.0,
                            "dlna": 0.8, "misfit_dlna": 0.6}]}

    bounds = {"R": [-0.1, 0.1], "T": [-0.1, 0.1], "Z": [-0.1, 0.1]}
    v, m = fw.filter_measurements_on_bounds(windows, measures, bounds)
    assert len(v) == 0
    assert len(m) == 0


def assert_wins_and_meas_same_length(wins, meas):
    assert len(wins) == len(meas)
    for sta in wins:
        assert len(wins[sta]) == len(meas[sta])
        for chan in wins[sta]:
            assert len(wins[sta][chan]) == len(meas[sta][chan])


def test_filter_windows_on_measurements():
    measure_config = \
        {"component": {
            "R": {"tshift_reference": 0, "tshift_acceptance_level": 10,
                  "dlna_reference": 0, "dlna_acceptance_level": 1.0,
                  "std_ratio": 4.0},
            "T": {"tshift_reference": 0, "tshift_acceptance_level": 10,
                  "dlna_reference": 0, "dlna_acceptance_level": 1.0,
                  "std_ratio": 4.0},
            "Z": {"tshift_reference": 0, "tshift_acceptance_level": 10,
                  "dlna_reference": 0, "dlna_acceptance_level": 1.0,
                  "std_ratio": 4.0}}}

    _wins, _meas = fw.filter_windows_on_measurements(
        windows, measures, measure_config)
    assert _wins["II.AAK"] == windows["II.AAK"]
    assert _wins["IU.BCD"]["IU.BCD..BHR"] == windows["IU.BCD"]["IU.BCD..BHR"]
    assert _wins["IU.BCD"]["IU.BCD..BHT"] == windows["IU.BCD"]["IU.BCD..BHT"]
    assert _wins["II.ABKT"]["II.ABKT..BHR"] == \
        windows["II.ABKT"]["II.ABKT..BHR"]
    assert len(_wins["IU.BCD"]["IU.BCD..BHZ"]) == 4
    assert "II.ABKT..BHT" not in _wins["II.ABKT"]
    assert "II.ABKT..BHZ" not in _wins["II.ABKT"]
    assert_wins_and_meas_same_length(_wins, _meas)

    measure_config = \
        {"component": {
            "R": {"tshift_reference": 0, "tshift_acceptance_level": 10.0,
                  "dlna_reference": 0, "dlna_acceptance_level": 1.0,
                  "std_ratio": 1.0},
            "T": {"tshift_reference": 0, "tshift_acceptance_level": 10.0,
                  "dlna_reference": 0, "dlna_acceptance_level": 1.0,
                  "std_ratio": 1.0},
            "Z": {"tshift_reference": 0, "tshift_acceptance_level": 10.0,
                  "dlna_reference": 0, "dlna_acceptance_level": 1.0,
                  "std_ratio": 1.0}}}

    _wins, _meas = fw.filter_windows_on_measurements(
        windows, measures, measure_config)
    assert _wins["II.AAK"]["II.AAK..BHR"] == \
        [{"left_index": 1, "right_index": 2}]
    assert _wins["II.AAK"]["II.AAK..BHZ"] == \
        [{"left_index": 1, "right_index": 2},
         {"left_index": 2, "right_index": 3},
         {"left_index": 3, "right_index": 4}]
    assert _wins["IU.BCD"]["IU.BCD..BHT"] == \
        [{"left_index": 1, "right_index": 2}]
    assert _wins["IU.BCD"]["IU.BCD..BHZ"] == \
        [{"left_index": 1, "right_index": 2},
         {"left_index": 2, "right_index": 3},
         {"left_index": 3, "right_index": 4},
         {"left_index": 5, "right_index": 6}]
    assert_wins_and_meas_same_length(_wins, _meas)


def test_filter_windows_on_measurements_2():

    measure_config = \
        {"component": {
            "R": {"tshift_reference": 0, "tshift_acceptance_level": 0.1,
                  "dlna_reference": 0, "dlna_acceptance_level": 0.1,
                  "std_ratio": 1.0},
            "T": {"tshift_reference": 0, "tshift_acceptance_level": 0.1,
                  "dlna_reference": 0, "dlna_acceptance_level": 0.1,
                  "std_ratio": 1.0},
            "Z": {"tshift_reference": 0, "tshift_acceptance_level": 0.1,
                  "dlna_reference": 0, "dlna_acceptance_level": 0.1,
                  "std_ratio": 1.0}}
         }

    _wins, _meas = fw.filter_windows_on_measurements(
        windows, measures, measure_config)

    assert len(_wins) == 0
    assert len(_meas) == 0


def test_check_consistency():
    fw.check_consistency(windows, measures)

    _measures = copy.deepcopy(measures)
    _measures["II.AAK"].pop("II.AAK..BHZ")
    with pytest.raises(KeyError) as errmsg:
        fw.check_consistency(windows, _measures)

    assert "Missing" in str(errmsg)


def test_filter_windows():
    sensor_config = {"flag": True, "sensor_types": ["STS-1", "STS1"]}
    measure_config = \
        {"component": {
            "R": {"tshift_reference": 0, "tshift_acceptance_level": 6.0,
                  "dlna_reference": 0, "dlna_acceptance_level": 1.0,
                  "std_ratio": 3.0},
            "T": {"tshift_reference": 0, "tshift_acceptance_level": 6.0,
                  "dlna_reference": 0, "dlna_acceptance_level": 1.0,
                  "std_ratio": 3.0},
            "Z": {"tshift_reference": 0, "tshift_acceptance_level": 6.0,
                  "dlna_reference": 0, "dlna_acceptance_level": 1.0,
                  "std_ratio": 3.0}},
         "flag": True}

    config = {"sensor": sensor_config, "measurement": measure_config}
    _wins, _meas, log = fw.filter_windows(
        windows, stations, measures, config, verbose=False)

    assert _wins["II.AAK"] == windows["II.AAK"]
    assert _wins["II.ABKT"]["II.ABKT..BHR"] == \
        windows["II.ABKT"]["II.ABKT..BHR"]
    assert "II.ABKT..BHZ" not in _wins["II.ABKT"]
    assert "II.ABKT..BHT" not in _wins["II.ABKT"]
    assert "IU.BCD" not in _wins["II.ABKT"]

    assert _meas["II.AAK"] == {
        "II.AAK..BHR": [{"dt": 1.0, "misfit_dt": 1.0,
                         "dlna": 0.7, "misfit_dlna": 1.0},
                        {"dt": -1.0, "misfit_dt": 1.0,
                         "dlna": -0.7, "misfit_dlna": 1.0}],
        "II.AAK..BHT": [{"dt": 1.5, "misfit_dt": 2.5,
                         "dlna": 0.9, "misfit_dlna": 0.5}],
        "II.AAK..BHZ": [{"dt": 1.0, "misfit_dt": 1.0,
                         "dlna": 0.6, "misfit_dlna": 1.0},
                        {"dt": 2.0, "misfit_dt": 4.0,
                         "dlna": 0.4, "misfit_dlna": 0.6},
                        {"dt": -1.5, "misfit_dt": 3.00,
                         "dlna": -0.5, "misfit_dlna": 1.0}]}

    assert _meas["II.ABKT"] == {
        "II.ABKT..BHR": [{"dt": 1.0, "misfit_dt": 1.5,
                          "dlna": 0.6, "misfit_dlna": 0.5}]}
    assert_wins_and_meas_same_length(_wins, _meas)

    print(log)
    _true_log = \
        {'sensor': {
            'window_rejection_percentage': '50.00', 'nwindows_old': 18,
            'channel_rejection_percentage': '37.50', 'nwindows_new': 9,
            'nchannels_new': 5, 'nchannels_old': 8},
         'measurement': {
             'window_rejection_percentage': '22.22', 'nwindows_old': 9,
             'channel_rejection_percentage': '20.00', 'nwindows_new': 7,
             'nchannels_new': 4, 'nchannels_old': 5}}

    assert log == _true_log


def test_filter_windows_2():
    sensor_config = {"flag": True, "sensor_types": ["STS-1", "STS1"]}
    measure_config = \
        {"component": {
            "R": {"tshift_reference": 0, "tshift_acceptance_level": 0.1,
                  "dlna_reference": 0, "dlna_acceptance_level": 1.0,
                  "std_ratio": 3.0},
            "T": {"tshift_reference": 0, "tshift_acceptance_level": 0.1,
                  "dlna_reference": 0, "dlna_acceptance_level": 1.0,
                  "std_ratio": 3.0},
            "Z": {"tshift_reference": 0, "tshift_acceptance_level": 0.1,
                  "dlna_reference": 0, "dlna_acceptance_level": 1.0,
                  "std_ratio": 3.0}},
         "flag": True}

    config = {"sensor": sensor_config, "measurement": measure_config}
    _wins, _meas, log = fw.filter_windows(
        windows, stations, measures, config, verbose=False)

    assert len(_wins) == 0

    assert len(_meas) == 0

    measure_config = \
        {"component": {
            "R": {"tshift_reference": 0, "tshift_acceptance_level": 10.0,
                  "dlna_reference": 0, "dlna_acceptance_level": 1.0,
                  "std_ratio": 0.01},
            "T": {"tshift_reference": 0, "tshift_acceptance_level": 10.0,
                  "dlna_reference": 0, "dlna_acceptance_level": 1.0,
                  "std_ratio": 0.01},
            "Z": {"tshift_reference": 0, "tshift_acceptance_level": 10.0,
                  "dlna_reference": 0, "dlna_acceptance_level": 1.0,
                  "std_ratio": 0.01}},
         "flag": True}
    config = {"sensor": sensor_config, "measurement": measure_config}
    _wins, _meas, log = fw.filter_windows(
        windows, stations, measures, config, verbose=False)

    assert len(_wins) == 0
    assert len(_meas) == 0
