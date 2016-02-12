import obspy
from obspy.signal.invsim import c_sac_taper
from obspy.signal.util import _npts2nfft
import numpy as np
from obspy import UTCDateTime
from rotate import rotate_stream


def check_array_order(array, order="ascending"):
    """
    Check whether a array is in ascending order or descending order
    :param array:
    :return:
    """
    array = np.array(array)
    if order == "descending":
        array *= -1.0

    if array == array.sort():
        return True
    else:
        return False


def flex_cut_trace(tr, cut_starttime, cut_endtime):
    """
    not cut strictly, but also based on the original trace length
    """
    if not isinstance(tr, obspy.Trace):
        raise ValueError("cut_trace method only accepts obspy.Trace"
                         "as the first argument")

    starttime = tr.stats.starttime
    endtime = tr.stats.endtime
    cut_starttime = max(starttime, cut_starttime)
    cut_endtime = min(endtime, cut_endtime)
    if cut_starttime > cut_endtime:
        raise ValueError("Cut starttime is larger than cut endtime")
    return tr.slice(cut_starttime, cut_endtime)


def cut_func(st, cut_start, cut_end, dynamic_length=10.0):
    t1 = cut_start - dynamic_length
    t2 = cut_start + dynamic_length

    if isinstance(st, obspy.Trace):
        return flex_cut_trace(st, t1, t2)
    elif isinstance(st, obspy.Stream):
        tr_list = []
        for tr in st:
            tr_list.append(flex_cut_trace(tr, t1, t2))
        return obspy.Stream(traces=tr_list)
    else:
        raise ValueError("cut_func method only accepts obspy.Trace or"
                         "obspy.Stream as the first Argument")


def filter_synt(tr, pre_filt):
    """
    Perform a frequency domain taper like during the response removal
    just without an actual response...
    :param tr:
    :param pre_filt:
    :return:
    """

    data = tr.data.astype(np.float64)

    # smart calculation of nfft dodging large primes
    nfft = _npts2nfft(len(data))

    fy = 1.0 / (tr.stats.delta * 2.0)
    freqs = np.linspace(0, fy, nfft // 2 + 1)

    # Transform data to Frequency domain
    data = np.fft.rfft(data, n=nfft)
    data *= c_sac_taper(freqs, flimit=pre_filt)
    data[-1] = abs(data[-1]) + 0.0j
    # transform data back into the time domain
    data = np.fft.irfft(data)[0:len(data)]
    # assign processed data and store processing information
    tr.data = data


def process(st, inv, remove_response=False, pre_filt=[0.025, 0.05, 0.1, 0.2],
            starttime=UTCDateTime(0), endtime=UTCDateTime(3600),
            resample_flag=True, sampling_rate=1.0,
            taper_type="hann", taper_percentage=0.05,
            rotate_flag=True, event_latitude=0.0, event_longitude=0.0):
    """
    Data processing base funtion
    """

    # cut the stream out before processing
    cut_func(st, starttime, endtime)

    st.detrend("linear")
    st.detrend("demean")
    st.taper(max_percentage=taper_percentage, type=taper_type)

    if remove_response:
        # remove response
        st.attach_response(inv)
        st.remove_response(output="DISP", pre_filt=pre_filt, zero_mean=False,
                           taper=False)
    else:
        # Perform a frequency domain taper like during the response removal
        # just without an actual response...
        for tr in st:
            filter_synt(tr, pre_filt)

    st.detrend("linear")
    st.detrend("demean")
    st.taper(max_percentage=taper_percentage, type=taper_type)

    if resample_flag:
        # interpolation
        npts = (endtime - starttime) * sampling_rate
        st.interpolate(sampling_rate=sampling_rate, starttime=starttime,
                       npts=npts)
    else:
        # cut
        st.trim(starttime, endtime)

    if rotate_flag:
        rotate_stream(st, inv, event_latitude, event_longitude, mode="ALL")

    # Convert to single precision to save space.
    for tr in st:
        tr.data = np.require(tr.data, dtype="float32")

    return st
