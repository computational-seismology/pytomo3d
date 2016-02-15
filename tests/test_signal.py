import pytomo3d.signal.process as proc
import obspy
import pytest
import numpy as np
import os
import inspect

# Most generic way to get the data folder path.
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(
    inspect.getfile(inspect.currentframe()))), "data")
testfile = os.path.join(DATA_DIR, "BW.RJOB.obs.mseed")
teststaxml = os.path.join(DATA_DIR, "BW.RJOB.xml")
bmfile = os.path.join(DATA_DIR, "BW.RJOB.obs.proc.mseed")


def test_check_array():
    array = [1, 2, 3, 4]
    assert proc.check_array_order(array, order='ascending')
    array = [-1.0, -2.0, -3, -4]
    assert proc.check_array_order(array, order='descending')
    array = [2.0, 1.0, 3.0, 4.0]
    assert (not proc.check_array_order(array))


def test_flex_cut_trace():
    st = obspy.read(testfile)
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
    st = obspy.read(testfile)
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
        if not np.allclose(tr1.data, tr2.data):
            return False
    return True


def test_process():
    st = obspy.read(testfile)
    inv = obspy.read_inventory(teststaxml)
    t1 = st[0].stats.starttime + 5
    t2 = st[0].stats.endtime - 5
    st_new = proc.process(st, remove_response_flag=True, inv=inv,
                          filter_flag=False, pre_filt=[10, 20, 40, 50],
                          starttime=t1, endtime=t2, resample_flag=True,
                          sampling_rate=10)
    st_compare = obspy.read(bmfile)
    assert compare_stream_kernel(st_new, st_compare)


if __name__ == '__main__':
    test_process()
