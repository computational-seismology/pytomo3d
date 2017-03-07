import os
import inspect
from copy import deepcopy
import pytest
import numpy.testing as npt
import pytomo3d.station.generate_adjoint_stations as gas
from pytomo3d.utils.io import load_json


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

measure_file = os.path.join(DATA_DIR, "window", "measurements.fake.json")
_measurements = load_json(measure_file)

station_file = os.path.join(DATA_DIR, "stations", "stations.fake.json")
_stations = load_json(station_file)


def test_extract_usable_stations_from_one_period():
    ms = deepcopy(_measurements)
    stations, channels = gas.extract_usable_stations_from_one_period(ms)
    assert set(stations) == set(["II.AAK", "II.ABKT", "IU.BCD"])
    assert set(channels) == set(["II.AAK..BHR", "II.AAK..BHT", "II.AAK..BHZ",
                                 "II.ABKT..BHR", "II.ABKT..BHZ",
                                 "IU.BCD..BHR", "IU.BCD..BHT", "IU.BCD..BHZ"])

    # add a fake station with no measurements
    ms["FK.FAKE"] = {}
    stations, channels = gas.extract_usable_stations_from_one_period(ms)
    assert set(stations) == set(["II.AAK", "II.ABKT", "IU.BCD"])
    assert set(channels) == set(["II.AAK..BHR", "II.AAK..BHT", "II.AAK..BHZ",
                                 "II.ABKT..BHR", "II.ABKT..BHZ",
                                 "IU.BCD..BHR", "IU.BCD..BHT", "IU.BCD..BHZ"])

    stations, channels = gas.extract_usable_stations_from_one_period({})
    assert len(stations) == 0
    assert len(channels) == 0


def test_extract_usable_stations_from_measurements():
    ms = {"17_40": deepcopy(_measurements),
          "40_100": deepcopy(_measurements),
          "90_250": deepcopy(_measurements)}

    stations, channels = gas.extract_usable_stations_from_measurements(ms)
    assert set(stations) == set(["II.AAK", "II.ABKT", "IU.BCD"])
    assert set(channels) == set(["II.AAK..BHR", "II.AAK..BHT", "II.AAK..BHZ",
                                 "II.ABKT..BHR", "II.ABKT..BHZ",
                                 "IU.BCD..BHR", "IU.BCD..BHT", "IU.BCD..BHZ"])

    ms["90_250"]["IU.BCD"]["IU.BCD..BHT"] = []
    stations, channels = gas.extract_usable_stations_from_measurements(ms)
    assert set(stations) == set(["II.AAK", "II.ABKT", "IU.BCD"])
    assert set(channels) == set(["II.AAK..BHR", "II.AAK..BHT", "II.AAK..BHZ",
                                 "II.ABKT..BHR", "II.ABKT..BHZ",
                                 "IU.BCD..BHR", "IU.BCD..BHT", "IU.BCD..BHZ"])

    ms["90_250"]["IU.BCD"] = {}
    stations, channels = gas.extract_usable_stations_from_measurements(ms)
    assert set(stations) == set(["II.AAK", "II.ABKT", "IU.BCD"])
    assert set(channels) == set(["II.AAK..BHR", "II.AAK..BHT", "II.AAK..BHZ",
                                 "II.ABKT..BHR", "II.ABKT..BHZ",
                                 "IU.BCD..BHR", "IU.BCD..BHT", "IU.BCD..BHZ"])


def test_extract_one_station():
    info = gas.extract_one_station("II.ABKT..BHZ", _stations)

    true_info = {"depth": 0.0, "elevation": 2437.8, "latitude": 0.0,
                 "longitude": 120.0,
                 "sensor": "Streckeisen STS1H/VBB Seismometer"}

    assert info == true_info

    info = gas.extract_one_station("II.ABKT..BHR", _stations)
    assert info == true_info

    with pytest.raises(KeyError):
        gas.extract_one_station("II.ABKT..LHZ", _stations)


def test_prepare_adjoint_station_information():
    adjoint_stations = gas.prepare_adjoint_station_information(
        ["II.ABKT..BHR"], _stations)

    _true = [0.0, 120.0, 2437.8, 0.0]
    assert len(adjoint_stations) == 1
    npt.assert_allclose(adjoint_stations["II.ABKT"], _true)

    adjoint_stations = gas.prepare_adjoint_station_information(
        ["II.ABKT..BHR", "II.ABKT..BHZ"], _stations)
    assert len(adjoint_stations) == 1
    npt.assert_allclose(adjoint_stations["II.ABKT"], _true)

    adjoint_stations = gas.prepare_adjoint_station_information(
        ["II.ABKT..BHR", "II.ABKT..BHZ", "IU.BCD..BHZ"], _stations)
    assert len(adjoint_stations) == 2
    npt.assert_allclose(adjoint_stations["II.ABKT"], _true)
    npt.assert_allclose(adjoint_stations["IU.BCD"],
                        [0.0, -120.0, 2437.8, 0.0])


def test_check_adjoint_stations_consistency():
    adjoint_stations = {"II.AAK": [1, 2, 3], "II.ABKT": [1, 2],
                        "IU.BCD": [1, 2]}
    usable_stations = ["II.AAK", "II.ABKT", "IU.BCD"]
    gas.check_adjoint_stations_consistency(adjoint_stations, usable_stations)

    adjoint_stations = {"II.AAK": [1, 2, 3], "II.ABKT": [1, 2],
                        "IU.BCD": [1, 2], "FK.FAKE": [1]}
    with pytest.raises(ValueError):
        gas.check_adjoint_stations_consistency(
            adjoint_stations, usable_stations)

    adjoint_stations = {"II.AAK": [1, 2, 3], "II.ABKT": [1, 2],
                        "FK.FAKE": [1]}
    with pytest.raises(ValueError):
        gas.check_adjoint_stations_consistency(
            adjoint_stations, usable_stations)


def test_benchmark_stations():
    adjoint_stations = {"II.AAK": [42.6375, 74.4942, 1633.10, 30.00],
                        "FK.FAKE": [10.0, 20.0, 30.0, 40.0]}
    npass = gas.benchmark_stations(adjoint_stations)
    assert npass == 1

    adjoint_stations = {"II.AAK": [42.6375, 74.4942, 1633.10, 30.00],
                        "II.ABPO": [-19.0180, 47.2290, 1528.00, 5.30],
                        "G.CAN": [-35.4187, 148.9963, 700.00, 0.00]}
    with pytest.raises(ValueError):
        gas.benchmark_stations(adjoint_stations)


def read_station_txt(fn):
    stations = {}
    with open(fn) as fh:
        for line in fh:
            content = line.split()
            sta = content[0]
            nw = content[1]
            stations["%s.%s" % (nw, sta)] = \
                [float(content[i]) for i in range(2, 6)]

    return stations


def test_generate_adjoint_stations(tmpdir):
    ms_one = deepcopy(_measurements)
    ms_one["IU.BCD"].pop("IU.BCD..BHR")
    ms_one["IU.BCD"].pop("IU.BCD..BHT")

    ms = {"17_40": deepcopy(ms_one),
          "40_100": deepcopy(ms_one),
          "90_250": deepcopy(ms_one)}

    outputfn = os.path.join(str(tmpdir), "STATIONS.tmp")
    gas.generate_adjoint_stations(
        ms, _stations, outputfn, benchmark_flag=False)

    output_station = read_station_txt(outputfn)
    assert len(output_station) == 3
    npt.assert_allclose(output_station["II.AAK"], [0.0, 0.0, 2437.8, 0.0])
    npt.assert_allclose(output_station["II.ABKT"], [0.0, 120.0, 2437.8, 0.0])
    npt.assert_allclose(output_station["IU.BCD"], [0.0, -120.0, 2437.8, 0.0])

    with pytest.raises(ValueError):
        gas.generate_adjoint_stations(
            ms, _stations, outputfn, benchmark_flag=True)
