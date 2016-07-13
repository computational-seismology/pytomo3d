import os
import inspect
from copy import deepcopy
from pytomo3d.station import extract_sensor_type
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

    true_type = \
        {'IU.KBL..BHE': 'Streckeisen STS-2/VBB Seismometer',
         'IU.KBL..BHN': 'Streckeisen STS-2/VBB Seismometer',
         'IU.KBL..BHZ': 'Streckeisen STS-2/VBB Seismometer'}

    inv = deepcopy(teststaxml)
    sensor_type = extract_sensor_type(inv)
    assert sensor_type == true_type

    inv = deepcopy(staxmlfile)
    sensor_type = extract_sensor_type(inv)
    assert sensor_type == true_type
