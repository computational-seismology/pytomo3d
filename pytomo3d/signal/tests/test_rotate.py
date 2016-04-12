import os
import inspect
import numpy as np
import numpy.testing as npt
import pytomo3d.signal.rotate as rotate
from obspy import read, read_inventory
from copy import deepcopy


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


def test_calculate_baz():
    elat = 0.0
    elon = 0.0
    slat = 10.0
    # put a very small value here
    slon = 0.000000001
    npt.assert_allclose(rotate.calculate_baz(elat, elon, slat, slon),
                        180.0)
    assert np.isclose(rotate.calculate_baz(slat, slon, elat, elon), 0.0)

    elat = 0.0
    elon = 0.0
    slat = 0.0
    slon = 10.0
    npt.assert_allclose(rotate.calculate_baz(elat, elon, slat, slon),
                        270.0)
    npt.assert_allclose(rotate.calculate_baz(slat, slon, elat, elon),
                        90.0)


def test_check_orthogonality():

    azi1 = 1
    azi2 = 91
    assert rotate.check_orthogonality(azi1, azi2) == "left-hand"
    assert rotate.check_orthogonality(azi2, azi1) == "right-hand"

    azi1 = 45
    azi2 = 315
    assert rotate.check_orthogonality(azi1, azi2) == "right-hand"
    assert rotate.check_orthogonality(azi2, azi1) == "left-hand"

    azi1 = 46
    azi2 = 137
    assert not rotate.check_orthogonality(azi1, azi2)
    assert not rotate.check_orthogonality(azi2, azi1)

    azi1 = 46
    azi2 = 314
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


def test_extract_channel_orientation_info():
    st = testobs.copy()
    inv = deepcopy(teststaxml)

    tr_z = st.select(channel="*Z")[0]
    dip, azi = rotate.extract_channel_orientation_info(tr_z, inv)
    assert dip == -90.0
    assert azi == 0.0

    tr_e = st.select(channel="*E")[0]
    dip, azi = rotate.extract_channel_orientation_info(tr_e, inv)
    assert dip == 0.0
    assert azi == 100.0

    tr_n = st.select(channel="*N")[0]
    dip, azi = rotate.extract_channel_orientation_info(tr_n, inv)
    assert dip == 0.0
    assert azi == 10.0


def test_check_inventory_orientation():
    inv = deepcopy(teststaxml)
    error = rotate._check_inventory_orientation(inv)
    assert error == 6
    assert bin(error)[2:].zfill(4) == "0110"


def test_check_inventory_sanity():
    inv = deepcopy(teststaxml)
    obs = testobs.copy()
    new_obs, new_inv = rotate._check_inventory_sanity(obs, inv)
    assert len(new_obs) == 1
    assert new_obs[0].stats.channel == "BHZ"


def sort_stream_by_station():

    st = read(small_mseed)
    st += testobs.copy()
    st += testsyn.copy()

    assert len(rotate.sort_stream_by_station) == 3


def test_rotate_one_station_stream_obsd():
    obs = testobs.copy()
    inv = deepcopy(teststaxml)

    obs1 = obs.copy()
    obs_r1 = rotate.rotate_one_station_stream(
        obs1, 0.0, 0.0, inventory=inv, mode="NE->RT",
        sanity_check=False)

    obs2 = obs.copy()
    obs_r2 = rotate.rotate_one_station_stream(
        obs2, 0.0, 0.0, station_latitude=34.5408,
        station_longitude=69.0432, mode="NE->RT",
        sanity_check=False)
    assert obs_r1 == obs_r2

    obs3 = obs.copy()
    obs_r3 = rotate.rotate_one_station_stream(
        obs3, 0.0, 0.0, inventory=inv, mode="NE->RT",
        sanity_check=True)
    assert len(obs_r3) == 1
    assert obs_r3[0].stats.channel == "BHZ"
    assert obs_r3[0] == obs_r1.select(channel="BHZ")[0]


def test_rotate_one_station_stream_synt():
    syn = testsyn.copy()
    inv = deepcopy(teststaxml)

    syn1 = syn.copy()
    syn_r1 = rotate.rotate_one_station_stream(
        syn1, 0.0, 0.0, inventory=inv, mode="NE->RT")
    syn2 = syn.copy()
    syn_r2 = rotate.rotate_one_station_stream(
        syn2, 0.0, 0.0, station_latitude=34.5408,
        station_longitude=69.0432, mode="NE->RT")
    assert syn_r1 == syn_r2

    syn3 = syn.copy()
    syn_r3 = rotate.rotate_one_station_stream(
        syn3, 0.0, 0.0, inventory=inv, mode="NE->RT")
    assert syn_r3 == syn_r1


def test_rotate_stream():

    obs = testobs.copy()
    syn = testsyn.copy()
    inv = deepcopy(teststaxml)
    obs_r = rotate.rotate_stream(obs, 0.0, 0.0, inv, mode="NE->RT")
    syn_r = rotate.rotate_stream(syn, 0.0, 0.0, inv, mode="NE->RT")

    st = testobs.copy() + testsyn.copy()
    st_r = rotate.rotate_stream(st, 0.0, 0.0, inv, mode="NE->RT")
    assert st_r.select(location="") == obs_r
    assert st_r.select(location="S3") == syn_r

    st = testobs.copy() + testsyn.copy()
    st_r = rotate.rotate_stream(st, 0.0, 0.0, inv, mode="NE->RT",
                                sanity_check=True)
    assert len(st_r.select(location="")) == 1
    assert st_r.select(location="")[0].stats.channel == "BHZ"
    assert len(st_r.select(location="S3")) == 0


if __name__ == "__main__":
    test_rotate_stream()
