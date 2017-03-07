import os
import inspect
from copy import deepcopy
from pytomo3d.station import extract_staxml_info
import obspy


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

staxmlfile = os.path.join(DATA_DIR, "stationxml", "IU.KBL.xml")
teststaxml = obspy.read_inventory(staxmlfile)


def test_process_obsd():

    true_type = {
        u'IU.KBL..BHZ': {
            'latitude': 34.5408, 'depth': 7.0, 'elevation': 1913.0,
            'longitude': 69.0432,
            'sensor': 'Streckeisen STS-2/VBB Seismometer'},
        u'IU.KBL..BHN': {
            'latitude': 34.5408, 'depth': 7.0, 'elevation': 1913.0,
            'longitude': 69.0432,
            'sensor': 'Streckeisen STS-2/VBB Seismometer'},
        u'IU.KBL..BHE': {
            'latitude': 34.5408, 'depth': 7.0, 'elevation': 1913.0,
            'longitude': 69.0432,
            'sensor': 'Streckeisen STS-2/VBB Seismometer'}
    }

    inv = deepcopy(teststaxml)
    sensor_type = extract_staxml_info(inv)
    assert sensor_type == true_type

    inv = deepcopy(staxmlfile)
    sensor_type = extract_staxml_info(inv)
    assert sensor_type == true_type
