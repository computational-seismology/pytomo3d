#!/usr/bin/env python
import os
import yaml
import numpy as np
from obspy import Stream, Trace
from obspy.core.util.geodetics import gps2DistAzimuth
import pyadjoint
from pyadjoint import AdjointSource


def plot_adjsrc_figure(figdir, obs_tr, adjsrc, _verbose):
    outfn = "%s.pdf" % obs_tr.id
    figfn = os.path.join(figdir, outfn)
    if _verbose:
        print "Output fig:", figfn
    adjsrc.plot(figfn)


def _stats_comp_window(windows):
    """
    Determine number of windows on each channel of each component.
    1) component weight: for example, {"Z":1.0, "R":0.5, "T":0.5}
    2) instrument weight: if there is mulitple instrument, we assign
        different weight to differnt instruments based on the number
        of windows on each instrument. For example, if there is two
        instrument, 'II.AAK.00.BHZ' and 'II.AAK.10.BHZ', and there are
        4 and 6 windows selected on each window, then for instrument
        weight, the values are 0.4 and 0.6
    """
    comp_win_dict = dict()
    for chan_win in windows:
        chan_id = chan_win[0].channel_id
        chan_name = chan_id.split()[-1]
        comp = chan_name[-1]
        nwin = len(chan_win)
        if comp not in comp_win_dict.keys():
            comp_win_dict[comp] = {}
        comp_win_dict[comp][chan_id] = nwin

    return comp_win_dict


def _clean_adj_results(comp_adj_dict, comp_nwins_dict):
    """
    Remove chan from comp_nwins_dict if the key is not in comp_adj_dict,
    for clean purpose. Also, rip off the comlex structure of comp_adj_dict
    and comp_nwins_dict(because they are two layered dict) to make them
    one-layer dict with {chan_id: adjsrc, ...}
    """
    clean_adj_dict = {}
    clean_nwin_dict = {}
    for comp, comp_adj in comp_adj_dict.iteritems():
        for chan_id, chan_adj in comp_adj.iteritems():
            clean_adj_dict[chan_id] = chan_adj
            clean_nwin_dict[chan_id] = comp_nwins_dict[comp][chan_id]

    return clean_adj_dict, clean_nwin_dict


def load_adjoint_config_yaml(filename):
    """
    load yaml and setup pyadjoint.Config object
    """
    with open(filename) as fh:
        data = yaml.load(fh)

    if data["min_period"] > data["max_period"]:
        raise ValueError("min_period is larger than max_period in config "
                         "file: %s" % filename)

    return pyadjoint.Config(**data)


def calculate_adjoint_sources(observed, synthetic, windows, config,
                              adj_src_type):

    comp_adj_dict = {}

    for chan_win in windows:
        if len(chan_win) == 0:
            continue

        obsd_id = chan_win[0].channel_id
        channel_name = obsd_id.split(".")[-1]
        comp_name = channel_name[-1]

        try:
            obs = observed.select(id=obsd_id)[0]
        except:
            raise ValueError("Missing observed trace for window: %s" % obsd_id)

        try:
            syn = synthetic.select(channel="*%s" % obs.stats.channel[-1])[0]
        except:
            raise ValueError("Missing synthetic trace matching obsd id: %s"
                             % obsd_id)

        wins = []
        # read windows for this trace
        for _win in chan_win:
            win_b = _win.relative_starttime
            win_e = _win.relative_endtime
            wins.append([win_b, win_e])

        try:
            adjsrc = pyadjoint.calculate_adjoint_source(
                adj_src_type=adj_src_type, observed=obs, synthetic=syn,
                config=config, window=wins, adjoint_src=True, plot=False)
        except:
            print("No adjoint source calculated for %s" % obsd_id)
            continue

        if comp_name not in comp_adj_dict.keys():
            comp_adj_dict[comp_name] = {}
        comp_adj_dict[comp_name][obsd_id] = adjsrc

    return comp_adj_dict


def adjsrc_function(observed, synthetic, adj_src_type='multitaper_misfit',
                    period=[27, 60], windows=None, figure_mode=False,
                    figure_dir=None, _verbose=False):
    '''
    Calculate adjoint sources using the time windows selected by pyflex

    :param observed: Observed data for one station
    :type observed: An obspy.core.stream.Stream object.
    :param synthetic: Synthetic data for one station
    :type synthetic: An obspy.core.stream.Stream object
    :param selection_mode: measurement type (cc_traveltime_misfit;
        multitaper_misfir; waveform_midfit)
    :type selection_mode: str
    :param config: parameters
    :type config: a class instance with all necessary constants/parameters
    :param window: window files for one station, produced by FLEXWIN/pyflexwin
    :type window: a dictionary instance with all time windows for each
        contained traces in the stream object.
    '''
    if windows is None or len(windows) == 0:
        return

    comp_nwins_dict = _stats_comp_window(windows)
    comp_adj_dict = calculate_adjoint_sources(observed, synthetic, windows,
                                              adj_src_type, period)

    return _clean_adj_results(comp_adj_dict, comp_nwins_dict)


def adjsrc_wrapper(obsd_station_group, synt_station_group,
                   obsd_tag="proc_obsd_27_60", synt_tag="proc_synt_27_60",
                   period=[27, 60], window=None, event=None,
                   selection_mode="multitaper_misfit",
                   figure_mode=False, figure_dir=None,
                   _verbose=False, outputdir="."):
    """
    Wrapper for pyasdf APIs
    """

    # Make sure everything thats required is there.
    if not hasattr(obsd_station_group, obsd_tag) or \
       not hasattr(synt_station_group, synt_tag):
        print "Missing attr, return"
        return 1

    observed = getattr(obsd_station_group, obsd_tag)
    synthetic = getattr(synt_station_group, synt_tag)

    # select associated windows
    ntwk = observed[0].stats.network
    stnm = observed[0].stats.station
    win_key = ntwk + "." + stnm
    st_wins = window[win_key]

    return adjsrc_function(observed, synthetic, selection_mode=selection_mode,
                           period=period, window=st_wins,
                           figure_mode=figure_mode,
                           figure_dir=figure_dir, _verbose=_verbose,
                           outputdir=outputdir)


def calculate_baz(elat, elon, slat, slon):

    _, baz, _ = gps2DistAzimuth(elat, elon, slat, slon)

    return baz


def convert_adj_to_trace(adj, starttime, chan_id):

    tr = Trace()
    tr.data = adj.adjoint_source
    tr.stats.starttime = starttime
    tr.stats.delta = adj.dt

    tr.stats.channel = str(chan_id.split(".")[-1])
    tr.stats.station = adj.station
    tr.stats.network = adj.network
    tr.stats.location = chan_id.split(".")[2]

    return tr


def convert_trace_to_adj(tr, adj):

    adj.dt = tr.stats.delta
    adj.component = tr.stats.channel[-1]
    adj.adjoint_source = tr.data
    adj.station = tr.stats.station
    adj.network = tr.stats.network
    return adj


def zero_padding_stream(stream, starttime, endtime):
    """
    Zero padding the stream to time [starttime, endtime)
    """
    if starttime > endtime:
        raise ValueError("Starttime is larger than endtime: [%f, %f]"
                         % (starttime, endtime))

    for tr in stream:
        dt = tr.stats.delta
        npts = tr.stats.npts
        tr_starttime = tr.stats.starttime
        tr_endtime = tr.stats.endtime

        npts_before = int((tr_starttime - starttime) / dt) + 1
        npts_before = max(npts_before, 0)
        npts_after = int((endtime - tr_endtime) / dt) + 1
        npts_after = max(npts_after, 0)

        # recalculate the time for padding trace
        padding_starttime = tr_starttime - npts_before * dt
        padding_array = np.zeros(npts_before + npts + npts_after)
        padding_array[npts_before:(npts_before + npts)] = \
            tr.data[:]

        tr.data = padding_array
        tr.stats.starttime = padding_starttime


def sum_adj_on_component(adj_stream, weight_dict):

    new_stream = Stream()
    done_comps = []
    for comp, comp_weights in weight_dict.iteritems():
        for chan_id, chan_weight in comp_weights.iteritems():
            if comp not in done_comps:
                done_comps.append(comp)
                comp_tr = adj_stream.select(id=chan_id)[0]
                comp_tr.data *= chan_weight
                comp_tr.stats.location = ""
            else:
                comp_tr.data += \
                    chan_weight * adj_stream.select(id=chan_id)[0].data
        new_stream.append(comp_tr)
    return new_stream


def postprocess_adjsrc(adjsrcs, adj_starttime, raw_synthetic, staxml, event,
                       sum_over_comp=False, weight_dict=None):
    """
    Postprocess adjoint sources to fit SPECFEM input(same as raw_synthetic)
    1) zero padding the adjoint sources
    2) interpolation
    3) add multiple instrument together if there are
    4) rotate from (R, T) to (N, E)

    :param adjsrcs: adjoint sources list(no multiple instruments
        at this stage)
    :param raw_synthetic: raw synthetic from SPECFEM output
    """
    # extract event information
    origin = event.preferred_origin() or event.origins[0]
    elat = origin.latitude
    elon = origin.longitude
    event_time = origin.time

    # extract station information
    slat = float(staxml[0][0].latitude)
    slon = float(staxml[0][0].longitude)

    # transfer adjoint_source type to stream
    adj_stream = Stream()
    for chan_id, adj in adjsrcs.iteritems():
        _tr = convert_adj_to_trace(adj, adj_starttime, chan_id)
        adj_stream.append(_tr)

    interp_starttime = raw_synthetic[0].stats.starttime
    interp_delta = raw_synthetic[0].stats.delta
    interp_npts = raw_synthetic[0].stats.npts
    interp_endtime = interp_starttime + interp_delta * interp_npts
    time_offset = interp_starttime - event_time

    # zero padding
    zero_padding_stream(adj_stream, interp_starttime, interp_endtime)

    # interpolate
    adj_stream.interpolate(sampling_rate=1.0/interp_delta,
                           starttime=interp_starttime,
                           npts=interp_npts)

    # sum multiple instruments
    if sum_over_comp:
        if weight_dict is None:
            raise ValueError("weight_dict should be assigned if you want"
                             "to add")
        adj_stream = sum_adj_on_component(adj_stream, weight_dict)

    # rotate
    baz = calculate_baz(elat, elon, slat, slon)

    components = [tr.stats.channel[-1] for tr in adj_stream]

    # add zero trace for missing components
    missinglist = ["Z", "R", "T"]
    tr_template = adj_stream[0]
    for tr in adj_stream:
        missinglist.remove(tr.stats.channel[-1])
    for component in missinglist:
        zero_adj = tr_template.copy()
        zero_adj.data.fill(0.0)
        zero_adj.stats.channel = "%s%s" % (tr_template.stats.channel[0:2],
                                           component)
        adj_stream.append(zero_adj)

    if "R" in components and "T" in components:
        try:
            adj_stream.rotate(method="RT->NE", back_azimuth=baz)
        except Exception as e:
            print e

    final_adjsrcs = []

    _temp_id = adjsrcs.keys()[0]
    adj_src_type = adjsrcs[_temp_id].adj_src_type
    minp = adjsrcs[_temp_id].min_period
    maxp = adjsrcs[_temp_id].max_period
    for tr in adj_stream:
        _adj = AdjointSource(adj_src_type, 0.0, 0.0, minp, maxp, "")
        final_adjsrcs.append(convert_trace_to_adj(tr, _adj))

    return final_adjsrcs, time_offset
