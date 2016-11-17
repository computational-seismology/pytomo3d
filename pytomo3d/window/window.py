#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Methods that handles Window selection

:copyright:
    Wenjie Lei (lei@princeton.edu), 2016
:license:
    GNU Lesser General Public License, version 3 (LGPLv3)
    (http://www.gnu.org/licenses/lgpl-3.0.en.html)
"""
import os
import pyflex
import obspy
import copy
import importlib


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


def update_user_levels(user_module, config, station, event, obsd, synt):
    """Update user levels as an array using user_module

    This function emulates the user_functions feature in FLEXWIN. User
    has to prepare a python module that has a method called
    generate_user_levels. This module has to be passed out as a string
    (It should be formatted as you would in an import statement). And
    then, user_module is dynamically imported and the
    generate_user_levels function is called with the following
    parameters: pyflex config, obspy station data, obspy event data,
    observed and synthetic obspy traces. It is excepted that user to
    use given scalar acceptance levels as base values and generate
    acceptance level arrays as long as trace data to be given to the
    pyflex. Function should return these values in order: stalta water
    level, tshift acceptance level, dlna acceptance level,
    cc_acceptance_level, s2n_limit.

    :param user_module: user module as a string
    :type user_module: str
    :param config: window selection config
    :type config_dict: pyflex.Config
    :param station: station information which provids station location to
        calculate the epicenter distance
    :type station: obspy.Inventory or pyflex.Station
    :param event: event information, providing the event information
    :type event: pyflex.Event, obspy.Catalog or obspy.Event
    :param observed: observed trace
    :type observed: obspy.Trace
    :param synthetic: synthetic trace
    :type synthetic: obspy.Trace
    :return: window selection config with arrays
    :rtype: pyflex.Config
    """
    # Ridvan Orsvuran, 2016
    # If user gives a user_module, allow it to create the user
    # acceptance levels as arrays.
    try:
        # Add current working directory to path. This enables user to
        # use python files in the working directory.
        # import sys
        # sys.path.append(".")

        # Import the user module
        user = importlib.import_module(user_module)
        # Assign generate function to a variable. This enables to
        # catch the AttributeError.
        generate_user_levels = user.generate_user_levels
    except ImportError:
        raise Exception("Could not import the user_function module: %s"
                        % user_module)
    except AttributeError:
        raise Exception("Given user module does not have a "
                        "generate_user_levels method: %s" % user_module)

    # do not give the original config to the user
    config_copy = copy.deepcopy(config)
    stalta_waterlevel, tshift, dlna, cc, s2n = generate_user_levels(
        config_copy, station, event, obsd, synt)

    # Create a new config using new acceptance levels
    new_config = copy.deepcopy(config)
    new_config.stalta_waterlevel = stalta_waterlevel
    new_config.tshift_acceptance_level = tshift
    new_config.dlna_acceptance_level = dlna
    new_config.cc_acceptance_level = cc
    new_config.s2n_limit = s2n

    return new_config


def window_on_trace(obs_tr, syn_tr, config, station=None,
                    event=None, user_module=None, _verbose=False,
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
    :param user_module: user module as a string
    :type user_module: str
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

    # Ridvan Orsvuran, 2016
    # If user gives a user_module, use it to update acceptance levels
    # as arrays.
    if user_module is not None and user_module != "None":
        config = update_user_levels(user_module, config, station,
                                    event, obs_tr, syn_tr)

    ws = pyflex.WindowSelector(obs_tr, syn_tr, config,
                               event=event, station=station)
    try:
        windows = ws.select_windows()
    except Exception as err:
        print("Error(%s): %s" % (obs_tr.id, err))
        windows = []

    if figure_mode:
        plot_window_figure(figure_dir, obs_tr.id, ws, _verbose,
                           figure_format=figure_format)

    if _verbose:
        print("Station %s picked %i windows" % (obs_tr.id, len(windows)))

    return windows


def window_on_stream(observed, synthetic, config_dict, station=None,
                     event=None, user_modules=None,
                     figure_mode=False, figure_dir=None,
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
    :param user_modules: user_module strings in a dict similar to config_dict.
    :type user_modules: dict
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

    # Ridvan Orsvuran, 2016
    # Assign an empty dict to user_modules if it is None to avoid errors.
    if user_modules is None:
        user_modules = {}

    for category in config_dict:
        config_base = config_dict[category]
        user_module = user_modules.get(category, None)
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
                event=event, user_module=user_module, _verbose=_verbose,
                figure_mode=figure_mode, figure_dir=figure_dir)

            if windows is None:
                continue

            # Notice: Ebru suggests to write out window even its length is
            # zero, which means no windows selected on the traces, in order
            # to keep track of every thing
            all_windows[obs_tr.id] = windows

    return all_windows
