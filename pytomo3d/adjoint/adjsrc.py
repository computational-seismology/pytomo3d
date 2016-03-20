#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Methods that handles adjoint sources

:copyright:
    Wenjie Lei (lei@princeton.edu), 2016
:license:
    GNU General Public License, Version 3
    (http://www.gnu.org/copyleft/gpl.html)
"""
from __future__ import (print_function, division)
import os
import yaml
import numpy as np
from obspy import Stream, Trace
import pyadjoint
from .plot_util import plot_adjoint_source


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

    adjsrc = pyadjoint.calculate_adjoint_source(
        adj_src_type=adj_src_type, observed=obs, synthetic=syn,
        config=config, window=window_time, adjoint_src=adjoint_src_flag,
        plot=figure_mode)

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
