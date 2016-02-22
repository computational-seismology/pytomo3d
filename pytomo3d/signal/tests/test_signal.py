import os
import inspect
import numpy as np
import pytest
import obspy
import pytomo3d.signal.process as proc


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

teststaxml = os.path.join(DATA_DIR, "stationxml", "IU.KBL.xml")
testquakeml = os.path.join(DATA_DIR, "quakeml", "C201009031635A.xml")
testobs = os.path.join(DATA_DIR, "raw", "IU.KBL.obs.mseed")
testsyn = os.path.join(DATA_DIR, "raw", "IU.KBL.syn.mseed")
small_mseed = os.path.join(DATA_DIR, "raw", "BW.RJOB.obs.mseed")


def test_check_array():
    array = [1, 2, 3, 4]
    assert proc.check_array_order(array, order='ascending')
    array = [-1.0, -2.0, -3, -4]
    assert proc.check_array_order(array, order='descending')
    array = [2.0, 1.0, 3.0, 4.0]
    assert (not proc.check_array_order(array))


def test_flex_cut_trace():
    st = obspy.read(small_mseed)
    tr = st[0]
    tstart = tr.stats.starttime
    tend = tr.stats.endtime
    npts = tr.stats.npts
    dt = tr.stats.delta

    t1 = tstart + int(npts / 4) * dt
    t2 = tend - int(npts / 4) * dt
    tr_cut = proc.flex_cut_trace(tr, t1, t2)
    assert tr_cut.stats.starttime == t1
    assert tr_cut.stats.endtime == t2

    t1 = tstart - int(npts / 4) * dt
    t2 = tend + int(npts / 4) * dt
    tr_cut = proc.flex_cut_trace(tr, t1, t2)
    assert tr_cut.stats.starttime == tstart
    assert tr_cut.stats.endtime == tend

    t1 = tstart + int(npts * 0.8) * dt
    t2 = tend - int(npts * 0.8) * dt
    with pytest.raises(ValueError):
        proc.flex_cut_trace(tr, t1, t2)


def test_flex_cut_stream():
    st = obspy.read(small_mseed)
    tstart = st[0].stats.starttime
    tend = st[0].stats.endtime
    t1 = tstart + 10
    t2 = tend - 10
    dynamic_length = 5
    st = proc.flex_cut_stream(st, t1, t2, dynamic_length=dynamic_length)
    for tr in st:
        assert tr.stats.starttime == t1 - dynamic_length
        assert tr.stats.endtime == t2 + dynamic_length


def test_filter_trace():
    assert True


def compare_stream_kernel(st1, st2):
    if len(st1) != len(st2):
        return False
    for tr1 in st1:
        tr2 = st2.select(id=tr1.id)[0]
        if tr1.stats.starttime != tr2.stats.starttime:
            return False
        if tr1.stats.endtime != tr2.stats.endtime:
            return False
        if tr1.stats.sampling_rate != tr2.stats.sampling_rate:
            return False
        if tr1.stats.npts != tr2.stats.npts:
            return False
        if not np.allclose(tr1.data, tr2.data):
            return False
    return True


def test_process_obsd():

    st = obspy.read(testobs)
    inv = obspy.read_inventory(teststaxml)
    event = obspy.readEvents(testquakeml)[0]
    origin = event.preferred_origin() or event.origins[0]
    event_lat = origin.latitude
    event_lon = origin.longitude
    event_time = origin.time

    pre_filt = [1/90., 1/60., 1/27.0, 1/22.5]
    t1 = event_time
    t2 = event_time + 6000.0
    st_new = proc.process(st, remove_response_flag=True, inventory=inv,
                          filter_flag=True, pre_filt=pre_filt,
                          starttime=t1, endtime=t2, resample_flag=True,
                          sampling_rate=2.0, taper_type="hann",
                          taper_percentage=0.05, rotate_flag=True,
                          event_latitude=event_lat,
                          event_longitude=event_lon)
    bmfile = os.path.join(DATA_DIR, "proc", "IU.KBL.obs.proc.mseed")
    st_compare = obspy.read(bmfile)
    assert compare_stream_kernel(st_new, st_compare)


def test_process_synt():

    st = obspy.read(testsyn)
    inv = obspy.read_inventory(teststaxml)
    event = obspy.readEvents(testquakeml)[0]
    origin = event.preferred_origin() or event.origins[0]
    event_lat = origin.latitude
    event_lon = origin.longitude
    event_time = origin.time

    pre_filt = [1/90., 1/60., 1/27.0, 1/22.5]
    t1 = event_time
    t2 = event_time + 6000.0
    st_new = proc.process(st, remove_response_flag=False, inventory=inv,
                          filter_flag=True, pre_filt=pre_filt,
                          starttime=t1, endtime=t2, resample_flag=True,
                          sampling_rate=2.0, taper_type="hann",
                          taper_percentage=0.05, rotate_flag=True,
                          event_latitude=event_lat,
                          event_longitude=event_lon)
    bmfile = os.path.join(DATA_DIR, "proc", "IU.KBL.syn.proc.mseed")
    st_compare = obspy.read(bmfile)
    assert compare_stream_kernel(st_new, st_compare)
