import os
import inspect
import numpy as np
import numpy.testing as npt
import pytomo3d.signal.rotate_utils as rotate
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

obsfile = os.path.join(DATA_DIR, "raw", "IU.KBL.obs.mseed")
testobs = read(obsfile)
synfile = os.path.join(DATA_DIR, "raw", "IU.KBL.syn.mseed")
testsyn = read(synfile)
small_mseed = os.path.join(DATA_DIR, "raw", "BW.RJOB.obs.mseed")


def test_check_orthogonality():

    azi1 = 1
    azi2 = 91
    assert rotate.check_orthogonality(azi1, azi2) == "left-hand"
    assert rotate.check_orthogonality(azi2, azi1) == "right-hand"

    azi1 = 315
    azi2 = 45
    assert rotate.check_orthogonality(azi1, azi2) == "left-hand"
    assert rotate.check_orthogonality(azi2, azi1) == "right-hand"

    azi1 = 405
    azi2 = 495
    assert rotate.check_orthogonality(azi1, azi2) == "left-hand"
    assert rotate.check_orthogonality(azi2, azi1) == "right-hand"

    azi1 = 46
    azi2 = 137
    assert not rotate.check_orthogonality(azi1, azi2)
    assert not rotate.check_orthogonality(azi2, azi1)

    azi1 = 46
    azi2 = 314
    assert not rotate.check_orthogonality(azi1, azi2)
    assert not rotate.check_orthogonality(azi2, azi1)


def test_check_orthogonality_2():
    azi1 = -180
    azi2 = -90
    assert rotate.check_orthogonality(azi1, azi2) == "left-hand"
    assert rotate.check_orthogonality(azi2, azi1) == "right-hand"

    azi1 = -315
    azi2 = 135
    assert rotate.check_orthogonality(azi1, azi2) == "left-hand"
    assert rotate.check_orthogonality(azi2, azi1) == "right-hand"

    azi1 = -181
    azi2 = -90
    assert not rotate.check_orthogonality(azi1, azi2)
    assert not rotate.check_orthogonality(azi2, azi1)


def test_rotate_certain_angle():

    d1 = np.array([1.0, 0.0])
    d2 = np.array([0.0, 1.0])

    dnew1, dnew2 = rotate.rotate_certain_angle(d1, d2, 30.0)

    dnew1_true = np.array([np.sqrt(3)/2.0, 0.5])
    dnew2_true = np.array([-0.5, np.sqrt(3)/2.0])
    npt.assert_allclose(dnew1, dnew1_true)
    npt.assert_allclose(dnew2, dnew2_true)


def test_rotate_certain_angle_2():

    d1 = np.array([1.0, 0.0])
    d2 = np.array([0.0, 1.0])

    dnew1, dnew2 = rotate.rotate_certain_angle(d1, d2, 90.0)
    npt.assert_array_almost_equal(dnew1, [0.0, 1.0])
    npt.assert_array_almost_equal(dnew2, [-1.0, 0.0])

    dnew1, dnew2 = rotate.rotate_certain_angle(d1, d2, 180.0)
    npt.assert_array_almost_equal(dnew1, [-1.0, 0.0])
    npt.assert_array_almost_equal(dnew2, [0.0, -1.0])

    dnew1, dnew2 = rotate.rotate_certain_angle(d1, d2, 270.0)
    npt.assert_array_almost_equal(dnew1, [0.0, -1.0])
    npt.assert_array_almost_equal(dnew2, [1.0, 0.0])

    dnew1, dnew2 = rotate.rotate_certain_angle(d1, d2, 360.0)
    npt.assert_array_almost_equal(dnew1, [1.0, 0.0])
    npt.assert_array_almost_equal(dnew2, [0.0, 1.0])


def test_rotate_12_ne():

    d1 = np.array([1.0, 0.0])
    d2 = np.array([0.0, 1.0])

    n, e = rotate.rotate_12_ne(d1, d2, 30, 120)

    n_true = np.array([np.sqrt(3)/2.0, -0.5])
    e_true = np.array([0.5, np.sqrt(3)/2.0])
    npt.assert_allclose(n, n_true)
    npt.assert_allclose(e, e_true)


def test_rotate_ne_12():

    n = np.array([1.0, 0.0])
    e = np.array([0.0, 1.0])

    dnew1, dnew2 = rotate.rotate_ne_12(n, e, 30, 120)

    assert rotate.check_orthogonality(30, 120) == "left-hand"

    dnew1_true = np.array([np.sqrt(3)/2.0, 0.5])
    dnew2_true = np.array([-0.5, np.sqrt(3)/2.0])
    npt.assert_allclose(dnew1, dnew1_true)
    npt.assert_allclose(dnew2, dnew2_true)


def test_rotate_ne_and_12():
    # test if rotate_NE_12 and rotate_12_NE are reversable

    n = np.array([1.0, 0.0])
    e = np.array([0.0, 1.0])

    d1, d2 = rotate.rotate_ne_12(n, e, 30, 120)

    n_new, e_new = rotate.rotate_12_ne(d1, d2, 30, 120)

    npt.assert_allclose(n, n_new)
    npt.assert_allclose(e, e_new)


def test_rotate_12_rt():

    d1 = np.array([1.0, 0.0])
    d2 = np.array([0.0, 1.0])
    azi1 = 30
    azi2 = 120
    baz = 240

    r, t = rotate.rotate_12_rt(d1, d2, baz, azi1, azi2)

    n, e = rotate.rotate_12_ne(d1, d2, azi1, azi2)
    r_true, t_true = rotate.rotate_ne_12(n, e, baz - 180, baz - 90)

    npt.assert_allclose(r, r_true)
    npt.assert_allclose(t, t_true)


def test_rotate_rt_12():

    r = np.array([1.0, 0.0])
    t = np.array([0.0, 1.0])
    azi1 = 30
    azi2 = 120
    baz = 240

    d1, d2 = rotate.rotate_rt_12(r, t, baz, azi1, azi2)

    n, e = rotate.rotate_12_ne(r, t, baz - 180, baz - 90)
    d1_true, d2_true = rotate.rotate_ne_12(n, e, azi1, azi2)


def test_rotate_rt_and_12():

    r = np.array([1.0, 0.0])
    t = np.array([0.0, 1.0])
    azi1 = 30
    azi2 = 120
    baz = 240

    d1, d2 = rotate.rotate_rt_12(r, t, baz, azi1, azi2)
    r_new, t_new = rotate.rotate_12_rt(d1, d2, baz, azi1, azi2)

    npt.assert_allclose(r, r_new)
    npt.assert_allclose(t, t_new)
