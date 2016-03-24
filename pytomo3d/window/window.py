#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Methods that handles Window selection

:copyright:
    Wenjie Lei (lei@princeton.edu), 2016
:license:
    GNU General Public License, Version 3
    (http://www.gnu.org/copyleft/gpl.html)
"""
import os
import yaml
import pyflex
import obspy
import copy


def plot_window_figure(figure_dir, figure_id, ws, _verbose=False,
                       figure_format="pdf"):
    """
    Plot window figure out

    :param figure_dir: output figure directory
    :type figure_dir: str
    :param figure_id: figure id to distinguish windows plots, for
        example, trace id could be used, like "II.AAK.00.BHZ"
    :type figure_id: str
    :param ws: window selector object from pyflex
    :type ws: pyflex.WindowSelector
    :param _verbose: verbose output flag
    :type _verbose: bool
    :param figure_format: figure format, could be "pdf", "png" and etc.
    :type figure_format: str
    :return:
    """
    outfn = "%s.%s" % (figure_id, figure_format)
    figfn = os.path.join(figure_dir, outfn)
    if _verbose:
        print "Output window figure:", figfn
    ws.plot(figfn)


def load_window_config_yaml(filename):
    """
    Load yaml and setup pyflex.Config object

    :param filename:
    :return:
    """
    with open(filename) as fh:
        data = yaml.load(fh)

    if data["min_period"] > data["max_period"]:
        raise ValueError("min_period is larger than max_period in config "
                         "file: %s" % filename)

    return pyflex.Config(**data)


def window_on_trace(obs_tr, syn_tr, config, station=None,
                    event=None, _verbose=False,
                    figure_mode=False, figure_dir=None,
                    figure_format="pdf"):
    """
    Window selection on a trace(obspy.Trace)

    :param observed: observed trace
    :type observed: obspy.Trace
    :param synthetic: synthetic trace
    :type synthetic: obspy.Trace
    :param config: window selection config
    :type config_dict: pyflex.Config
    :param station: station information which provids station location to
        calculate the epicenter distance
    :type station: obspy.Inventory or pyflex.Station
    :param event: event information, providing the event information
    :type event: pyflex.Event, obspy.Catalog or obspy.Event
    :param figure_mode: output figure flag
    :type figure_mode: bool
    :param figure_dir: output figure directory
    :type figure_dir: str
    :param _verbose: verbose flag
    :type _verbose: bool
    :return:
    """

    if not isinstance(obs_tr, obspy.Trace):
        raise ValueError("Input obs_tr should be obspy.Trace")
    if not isinstance(syn_tr, obspy.Trace):
        raise ValueError("Input syn_tr should be obspy.Trace")
    if not isinstance(config, pyflex.Config):
        raise ValueError("Input config should be pyflex.Config")

    ws = pyflex.WindowSelector(obs_tr, syn_tr, config,
                               event=event, station=station)
    try:
        windows = ws.select_windows()
    except:
        print("Error on %s" % obs_tr.id)
        windows = []

    if figure_mode:
        plot_window_figure(figure_dir, obs_tr.id, ws, _verbose,
                           figure_format=figure_format)

    if _verbose:
        print("Station %s picked %i windows" % (obs_tr.id, len(windows)))

    return windows


def window_on_stream(observed, synthetic, config_dict, station=None,
                     event=None, figure_mode=False, figure_dir=None,
                     _verbose=False):
    """
    Window selection on a Stream

    :param observed: observed stream
    :type observed: obspy.Stream
    :param synthetic: synthetic stream
    :type synthetic: obspy.Stream
    :param config_dict: window selection config dictionary, for example,
        {"Z": pyflex.Config, "R": pyflex.Config, "T": pyflex.Config}
    :type config_dict: dict
    :param station: station information which provids station location to
        calculate the epicenter distance
    :type station: obspy.Inventory or pyflex.Station
    :param event: event information, providing the event information
    :type event: pyflex.Event, obspy.Catalog or obspy.Event
    :param figure_mode: output figure flag
    :type figure_mode: bool
    :param figure_dir: output figure directory
    :type figure_dir: str
    :param _verbose: verbose flag
    :type _verbose: bool
    :return:
    """
    if not isinstance(observed, obspy.Stream):
        raise ValueError("Input observed should be obspy.Stream")
    if not isinstance(synthetic, obspy.Stream):
        raise ValueError("Input synthetic should be obspy.Stream")
    if not isinstance(config_dict, dict):
        raise ValueError("Input config_dict should be dict")

    all_windows = {}

    for category in config_dict:
        config_base = config_dict[category]
        if len(category) == 1:
            # then it is component
            obs = observed.select(component=category)
        elif len(category) == 3:
            # then it is channel
            obs = observed.select(channel=category)
        else:
            raise ValueError("The length of Config_dict.keys()[%s] should be "
                             "either 1 or 3, for example, ['E', 'N', 'Z'] "
                             "or ['BHE', 'BHN', 'BHZ']" % config_dict.keys())

        for obs_tr in obs:
            component = obs_tr.stats.channel[-1]
            try:
                syn_tr = synthetic.select(station=obs_tr.stats.station,
                                          network=obs_tr.stats.network,
                                          component=component)[0]
            except Exception as err:
                print("Couldn't find corresponding synt for obsd trace(%s):"
                      "%s" % (obs_tr.id, err))
                continue

            config = copy.deepcopy(config_base)
            windows = window_on_trace(
                obs_tr, syn_tr, config, station=station,
                event=event, _verbose=_verbose,
                figure_mode=figure_mode, figure_dir=figure_dir)

            if windows is None:
                continue

            # Notice: Ebru suggests to write out window even its length is
            # zero, which means no windows selected on the traces, in order
            # to keep track of every thing
            all_windows[obs_tr.id] = windows

    return all_windows
