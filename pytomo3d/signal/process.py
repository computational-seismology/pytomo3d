import obspy
from obspy.signal.invsim import c_sac_taper
from obspy.signal.util import _npts2nfft
import numpy as np
from rotate import rotate_stream


def check_array_order(array, order="ascending"):
    """
    Check whether a array is in ascending order or descending order
    :param array: the input array
    :param order: "ascending" or "descending"
    :return:
    """
    array = np.array(array)
    if order not in ("descending", "ascending"):
        raise ValueError("Order should be either ascending or descending")

    if order == "descending":
        array *= -1

    return (array == sorted(array)).all()


def flex_cut_trace(tr, cut_starttime, cut_endtime):
    """
    not cut strictly(but also based on the original trace length)

    :param tr: input trace
    :type tr: obspy.Trace
    :param cut_starttime: starttime of cutting
    :type cut_starttime: obspy.UTCDateTime
    :param cut_endtime: endtime of cutting
    :type cut_endtime: obspy.UTCDateTime
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


def flex_cut_stream(st, cut_start, cut_end, dynamic_length=10.0):
    """
    Flexible cut stream

    :param st: input stream
    :param cut_start: cut starttime
    :param cut_end: cut endtime
    :param dynamic_length: the dynamic length before cut_start and after
        cut_end
    :return:
    """
    t1 = cut_start - dynamic_length
    t2 = cut_end + dynamic_length

    if isinstance(st, obspy.Trace):
        return flex_cut_trace(st, t1, t2)
    elif isinstance(st, obspy.Stream):
        tr_list = []
        for tr in st:
            tr_list.append(flex_cut_trace(tr, t1, t2))
        return obspy.Stream(traces=tr_list)
    else:
        raise ValueError("cut_func method only accepts obspy. Trace or"
                         "obspy.Stream as the first Argument")


def filter_trace(tr, pre_filt):
    """
    Perform a frequency domain taper mimicing the behavior during the
    response removal, without a actual response removal.

    :param tr: input trace
    :param pre_filt: frequency array(Hz) in ascending order, to define
        the four corners of filter, for example, [0.01, 0.1, 0.2, 0.5].
    :type pre_filt: Numpy.array or list
    :return: filtered trace
    """
    if type(tr) != obspy.Trace:
        raise ValueError("First Argument should be trace: %s" % type(tr))

    if not check_array_order(pre_filt):
        raise ValueError("Frequency band should be in ascending order: %s"
                         % pre_filt)

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


def process(st, remove_response_flag=False, inv=None,
            filter_flag=False, pre_filt=None,
            starttime=None, endtime=None,
            resample_flag=False, sampling_rate=1.0,
            taper_type="hann", taper_percentage=0.05,
            rotate_flag=False, event_latitude=None, event_longitude=None):
    """
    Stream processing function defined for general purpose of tomography

    :param st: input stream
    :type st: obspy.Stream
    :param remove_response_flag: flag for remove instrument response
    :type remove_response_flag: boolen
    :param inv: inventory information
    :type inv: obspy.Inventory
    :param filter_flag:flag for filter
    :type filter_flag: boolen
    :param pre_filt: list of tuple of 4 corner frequency for filter,
        in ascending order
    :param starttime: starttime of cutting
    :param endtime: endtime of cutting
    :param resample_flag: flag for data resampling
    :type resample_flag: boolen
    :param sampling_rate: resampling rate
    :type sampling_rate: float
    :param taper_type: taper type, options from obspy taper
    :type taper_type: str
    :param taper_percentage: percentage of taper
    :type taper_percentage: float
    :param rotate_flag: rotate flag. If true, both inv and event location
        information should be provided
    :param event_latitude: latitude of event, for rotation usage
    :type event_latitude: float
    :param event_longitude: longitude of event, for rotation usage
    :type event_longitude: float
    :return: processed stream
    """
    # cut the stream out before processing to reduce computation
    if not isinstance(st, obspy.Stream) and not isinstance(st, obspy.Trace):
        raise ValueError("Input seismogram should be either obspy.Stream "
                         "or obspy.Trace")

    if starttime is not None and endtime is not None:
        flex_cut_stream(st, starttime, endtime)

    st.detrend("linear")
    st.detrend("demean")
    st.taper(max_percentage=taper_percentage, type=taper_type)

    if remove_response_flag:
        # remove response
        st.attach_response(inv)
        st.remove_response(output="DISP", pre_filt=pre_filt, zero_mean=False,
                           taper=False)
    elif filter_flag:
        if pre_filt is None or len(pre_filt) != 4:
            raise ValueError("Filter band should be list or tuple with "
                             "length of 4")
        # Perform a frequency domain taper like during the response removal
        # just without an actual response...
        for tr in st:
            filter_trace(tr, pre_filt)

    st.detrend("linear")
    st.detrend("demean")
    st.taper(max_percentage=taper_percentage, type=taper_type)

    if resample_flag:
        # interpolation
        if sampling_rate is None:
            raise ValueError("sampling rate should be provided if you set"
                             "resample_flag=True")
        if endtime is not None and starttime is not None:
            npts = int((endtime - starttime) * sampling_rate) + 1
            st.interpolate(sampling_rate=sampling_rate,
                           starttime=starttime,
                           npts=npts)
        else:
            st.interpolate(sampling_rate=sampling_rate,
                           starttime=starttime)
    else:
        # just cut
        st.trim(starttime, endtime)

    if isinstance(st, obspy.Trace):
        st.data = np.require(st.data, dtype="float32")
    elif isinstance(st, obspy.Stream):
        if rotate_flag:
            rotate_stream(st, inv, event_latitude, event_longitude,
                          mode="ALL")
        # Convert to single precision to save space.
        for tr in st:
            tr.data = np.require(tr.data, dtype="float32")

    return st
