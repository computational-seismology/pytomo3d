#!/usr/bin/env python
from __future__ import (print_function, division)
import os
import yaml
import numpy as np
from obspy import Stream, Trace
from obspy.core.util.geodetics import gps2DistAzimuth
import pyadjoint
from pyadjoint import AdjointSource
from .plot_util import plot_adjoint_source
from pytomo3d.signal.process import filter_trace, check_array_order


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


def _extract_window_time(windows):
    """
    Extract window time information from a list of windows(pyflex.Window).
    Windows should come from the same channel.

    :param windows: a list of pyflex.Window
    :return: a two dimension numpy.array of time window, with window
        starttime and endtime
    """
    wins = []
    if isinstance(windows[0], dict):
        id_base = windows[0]["channel_id"]
        for _win in windows:
            if _win["channel_id"] != id_base:
                raise ValueError("Windows come from different channel: %s, %s"
                                 % (id_base, _win["channel_id"]))
            win_b = _win["relative_starttime"]
            win_e = _win["relative_endtime"]
            wins.append([win_b, win_e])
    else:
        id_base = windows[0].channel_id
        for _win in windows:
            if _win.channel_id != id_base:
                raise ValueError("Windows come from different channel: %s, %s"
                                 % (id_base, _win["channel_id"]))
            win_b = _win.relative_starttime
            win_e = _win.relative_endtime
            wins.append([win_b, win_e])

    # read windows for this trace
    return np.array(wins), id_base


def calculate_adjsrc_on_trace(obs, syn, window_time, config, adj_src_type,
                              figure_mode=False, figure_dir=None,
                              adjoint_src_flag=True):
    """
    Calculate adjoint source on a pair of trace and windows selected

    :param obs: observed trace
    :type obs: obspy.Trace
    :param syn: synthetic trace
    :type syn: obspy.Trace
    :param window_time: window time information, 2-dimension array, like
        [[win_1_left, win_1_right], [win_2_left, win_2_right], ...]
    :type windows: 2-d list or numpy.array
    :param config: config of pyadjoint
    :type config: pyadjoint.Config
    :param adj_src_type: adjoint source type, options include:
        1) "cc_traveltime_misfit"
        2) "multitaper_misfit"
        3) "waveform_misfit"
    :type adj_src_type: str
    :param adjoint_src_flag: whether calcualte adjoint source or not.
        If False, only make measurements
    :type adjoint_src_flag: bool
    :param plot_flag: whether make plots or not. If True, it will lot
        a adjoint source figure right after calculation
    :type plot_flag:  bool
    :return: adjoint source(pyadjoit.AdjointSource)
    """
    if not isinstance(obs, Trace):
        raise ValueError("Input obs should be obspy.Trace")
    if not isinstance(syn, Trace):
        raise ValueError("Input syn should be obspy.Trace")
    if not isinstance(config, pyadjoint.Config):
        raise ValueError("Input config should be pyadjoint.Config")
    windows = np.array(window_time)
    if len(windows.shape) != 2 or windows.shape[1] != 2:
        raise ValueError("Input windows dimension incorrect, dimention"
                         "(*, 2) expected")

    try:
        adjsrc = pyadjoint.calculate_adjoint_source(
            adj_src_type=adj_src_type, observed=obs, synthetic=syn,
            config=config, window=window_time, adjoint_src=adjoint_src_flag,
            plot=figure_mode)
    except:
        adjsrc = None

    if figure_mode:
        if figure_dir is None:
            figname = None
        else:
            figname = os.path.join(figure_dir, "%s.pdf" % obs.id)
        plot_adjoint_source(adjsrc, win_times=windows, obs_tr=obs,
                            syn_tr=syn, figname=figname)

    return adjsrc


def calculate_adjsrc_on_stream(observed, synthetic, windows, config,
                               adj_src_type, figure_mode=False,
                               figure_dir=None, adjoint_src_flag=True):
    """
    calculate adjoint source on a pair of stream and windows selected

    :param observed: observed stream
    :type observed: obspy.Stream
    :param synthetic: observed stream
    :type synthetic: obspy.Stream
    :param windows: list of pyflex windows, like:
        [[Windows(), Windows(), Windows()], [Windows(), Windows()], ...]
        For each element, it contains windows for one channel
    :type windows: list
    :param config: config for calculating adjoint source
    :type config: pyadjoit.Config
    :param adj_src_type: adjoint source type
    :type adj_src_type: str
    :param figure_mode: plot flag. Leave it to True if you want to see adjoint
        plots for every trace
    :type figure_mode: bool
    :param adjoint_src_flag: adjoint source flag. Set it to True if you want
        to calculate adjoint sources
    :type adjoint_src_flag: bool
    :return:
    """
    if not isinstance(observed, Stream):
        raise ValueError("Input observed should be obspy.Stream")
    if not isinstance(synthetic, Stream):
        raise ValueError("Input synthetic should be obspy.Stream")
    if windows is None or len(windows) == 0:
        return
    if not isinstance(config, pyadjoint.Config):
        raise ValueError("Input config should be pyadjoint.Config")

    adjsrcs_list = []

    for chan_win in windows:
        if len(chan_win) == 0:
            continue

        win_time, obsd_id = _extract_window_time(chan_win)

        try:
            obs = observed.select(id=obsd_id)[0]
        except:
            raise ValueError("Missing observed trace for window: %s" % obsd_id)

        try:
            syn = synthetic.select(channel="*%s" % obs.stats.channel[-1])[0]
        except:
            raise ValueError("Missing synthetic trace matching obsd id: %s"
                             % obsd_id)

        adjsrc = calculate_adjsrc_on_trace(
            obs, syn, win_time, config, adj_src_type,
            adjoint_src_flag=adjoint_src_flag,
            figure_mode=figure_mode, figure_dir=figure_dir)

        if adjsrc is None:
            continue
        adjsrcs_list.append(adjsrc)

    return adjsrcs_list


def calculate_baz(elat, elon, slat, slon):
    """
    Calculate back azimuth

    :param elat: event latitude
    :param elon: event longitude
    :param slat: station latitude
    :param slon: station longitude
    :return: back azimuth
    """

    _, baz, _ = gps2DistAzimuth(elat, elon, slat, slon)

    return baz


def _convert_adj_to_trace(adj):
    """
    Convert AdjointSource to Trace,for internal use only
    """

    tr = Trace()
    tr.data = adj.adjoint_source
    tr.stats.starttime = adj.starttime
    tr.stats.delta = adj.dt

    tr.stats.channel = adj.component
    tr.stats.station = adj.station
    tr.stats.network = adj.network
    tr.stats.location = adj.location

    return tr


def _convert_trace_to_adj(tr, adj_src_type, minp, maxp):
    """
    Convert Trace to AdjointSource, for internal use only
    """
    adj = AdjointSource(adj_src_type, 0.0, 0.0, minp, maxp, "")
    adj.dt = tr.stats.delta
    adj.adjoint_source = tr.data
    adj.station = tr.stats.station
    adj.network = tr.stats.network
    adj.location = tr.stats.location
    adj.component = tr.stats.channel
    adj.starttime = tr.stats.starttime

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


def sum_adj_on_component(adj_stream, weight_flag, weight_dict=None):
    """
    Sum adjoint source on different channels but same component
    together, like "II.AAK.00.BHZ" and "II.AAK.10.BHZ" to form
    "II.AAK.BHZ"

    :param adj_stream: adjoint source stream
    :param weight_dict: weight dictionary, should be something like
        {"Z":{"II.AAK.00.BHZ": 0.5, "II.AAK.10.BHZ": 0.5},
         "R":{"II.AAK.00.BHR": 0.3, "II.AAK.10.BHR": 0.7},
         "T":{"II.AAK..BHT": 1.0}}
    :return: summed adjoint source stream
    """
    if weight_dict is None:
        raise ValueError("weight_dict should be assigned if you want"
                         "to add")

    new_stream = Stream()
    done_comps = []

    if not weight_flag:
        # just add same components without weight
        for tr in adj_stream:
            comp = tr.stats.channel[-1]
            if comp not in done_comps:
                comp_tr = tr
                comp_tr.stats.location = ""
                new_stream.append(comp_tr)
            else:
                comp_tr = new_stream.select("*%s" % comp)
                comp_tr.data += tr.data
    else:
        # sum using components weight
        for comp, comp_weights in weight_dict.iteritems():
            for chan_id, chan_weight in comp_weights.iteritems():
                if comp not in done_comps:
                    done_comps.append(comp)
                    comp_tr = adj_stream.select(id=chan_id)[0]
                    comp_tr.data *= chan_weight
                    comp_tr.stats.location = ""
                    comp_tr.stats.channel = comp
                    new_stream.append(comp_tr)
                else:
                    comp_tr = new_stream.select(channel="*%s" % comp)[0]
                    comp_tr.data += \
                        chan_weight * adj_stream.select(id=chan_id)[0].data

    return new_stream


def rotate_adj(adj_stream, event, inventory):

    if event is None or inventory is None:
        raise ValueError("Event and Station must be provied to rotate the"
                         "adjoint source")
    # extract event information
    origin = event.preferred_origin() or event.origins[0]
    elat = origin.latitude
    elon = origin.longitude

    # extract station information
    slat = float(inventory[0][0].latitude)
    slon = float(inventory[0][0].longitude)

    # rotate
    baz = calculate_baz(elat, elon, slat, slon)
    components = [tr.stats.channel[-1] for tr in adj_stream]

    if "R" in components and "T" in components:
        try:
            adj_stream.rotate(method="RT->NE", back_azimuth=baz)
        except Exception as e:
            print(e)


def postprocess_adjsrc(adjsrcs, interp_starttime, interp_delta,
                       interp_npts, rotate_flag=False, inventory=None,
                       event=None, sum_over_comp_flag=False,
                       weight_flag=False, weight_dict=None,
                       filter_flag=False, pre_filt=None):
    """
    Postprocess adjoint sources to fit SPECFEM input(same as raw_synthetic)
    1) zero padding the adjoint sources
    2) interpolation
    3) add multiple instrument together if there are
    4) rotate from (R, T) to (N, E)

    :param adjsrcs: adjoint sources list from the same station
    :type adjsrcs: list
    :param adj_starttime: starttime of adjoint sources
    :param adj_starttime: obspy.UTCDateTime
    :param raw_synthetic: raw synthetic from SPECFEM output, as reference
    :type raw_synthetic: obspy.Stream or obspy.Trace
    :param inventory: station inventory
    :type inventory: obspy.Inventory
    :param event: event information
    :type event: obspy.Event
    :param sum_over_comp_flag: sum over component flag
    :param weight_dict: weight dictionary
    """

    if not isinstance(adjsrcs, list):
        raise ValueError("Input adjsrcs should be type of list of adjoint "
                         "sources")

    # transfer AdjointSource type to stream for easy processing
    adj_stream = Stream()
    for adj in adjsrcs:
        _tr = _convert_adj_to_trace(adj)
        adj_stream.append(_tr)

    # zero padding
    interp_endtime = interp_starttime + interp_delta * interp_npts
    zero_padding_stream(adj_stream, interp_starttime, interp_endtime)

    # interpolate
    adj_stream.interpolate(sampling_rate=1.0/interp_delta,
                           starttime=interp_starttime,
                           npts=interp_npts)

    # sum multiple instruments
    if sum_over_comp_flag:
        adj_stream = sum_adj_on_component(adj_stream, weight_flag,
                                          weight_dict)

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

    if rotate_flag:
        rotate_adj(adj_stream, event, inventory)

    if filter_flag:
        # filter the adjoint source
        if pre_filt is None or len(pre_filt) != 4:
            raise ValueError("Input pre_filt should be a list or tuple with "
                             "length of 4")
        if not check_array_order(pre_filt, order="ascending"):
            raise ValueError("Input pre_filt must a in ascending order. The "
                             "unit is Hz")
        for tr in adj_stream:
            filter_trace(tr, pre_filt)

    # convert the stream to pyadjoint.AdjointSource
    final_adjsrcs = []
    adj_src_type = adjsrcs[0].adj_src_type
    minp = adjsrcs[0].min_period
    maxp = adjsrcs[0].max_period
    for tr in adj_stream:
        final_adjsrcs.append(_convert_trace_to_adj(tr, adj_src_type,
                                                   minp, maxp))

    return final_adjsrcs
