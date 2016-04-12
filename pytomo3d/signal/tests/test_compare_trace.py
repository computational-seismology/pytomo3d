import os
import inspect
import numpy.testing as npt
import pytomo3d.signal.compare_trace as ct
from obspy import read, read_inventory


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
teststaxml = read_inventory(staxmlfile)
testquakeml = os.path.join(DATA_DIR, "quakeml", "C201009031635A.xml")

# synfile = os.path.join(DATA_DIR, "raw", "IU.KBL.syn.mseed")
# testsyn = read(synfile)
small_mseed = os.path.join(DATA_DIR, "raw", "BW.RJOB.obs.mseed")
smallobs = read(small_mseed)


def test_calculate_misfit():

    tr1 = smallobs[0]
    tr2 = smallobs[0]

    tr1_c, tr2_c, corr, err, _, _ = ct.calculate_misfit(tr1, tr2)

    npt.assert_allclose(tr1_c, 0.90220, rtol=1e-3)
    npt.assert_allclose(tr2_c, 0.90220, rtol=1e-3)
    npt.assert_allclose(corr, 1.0)
    npt.assert_allclose(err, 0.0)
