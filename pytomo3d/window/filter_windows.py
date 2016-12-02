#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Filter the window based on the sensor type. For example, in long
period band(90-250s), we want to keep only STS-1 instrument windows.
For the input file, it requires 1) sensor type as json file; 2) windows
as json file. For the output, it is going to replace the origin window file
and keep a copy of original windows as "***.origin.json"

:copyright:
    Wenjie Lei (lei@princeton.edu), 2016
:license:
    GNU Lesser General Public License, version 3 (LGPLv3)
    (http://www.gnu.org/licenses/lgpl-3.0.en.html)
"""
from __future__ import (absolute_import, division, print_function)
from pprint import pprint
import numpy as np


def is_right_sensor(sensor, sensor_types):
    """
    Check if sensor in the list of sensor_types
    """
    for stype in sensor_types:
        if stype in sensor:
            return True
    return False


def count_windows(windows):
    """
    Count the number of channels and windows
    :param windows:
    :return:
    """
    nchans = 0
    nwins = 0
    for stainfo in windows.itervalues():
        for chaninfo in stainfo.itervalues():
            _nw = len(chaninfo)
            if _nw == 0:
                continue
            nchans += 1
            nwins += _nw
    return nchans, nwins


def print_window_filter_summary(old_windows, new_windows):
    nchans_old, nwins_old = count_windows(old_windows)
    nchans_new, nwins_new = count_windows(new_windows)
    nchans_rej = (nchans_old - nchans_new) / nchans_old
    nwins_rej = (nwins_old - nwins_new) / nwins_old
    print("Number of channels old and new: %d --> %d(rej: %.2f %%)"
          % (nchans_old, nchans_new, nchans_rej * 100))
    print("Number of windows old and new:  %d --> %d(rej: %.2f %%)"
          % (nwins_old, nwins_new, nwins_rej * 100))
    return {"nchannels_old": nchans_old, "nwindows_old": nwins_old,
            "nchannels_new": nchans_new, "nwindows_new": nwins_new,
            "channel_rejection_percentage": nchans_rej * 100,
            "window_rejection_percentage": nwins_rej * 100}


def filter_windows_on_sensors(windows, stations, sensor_types, verbose=False):
    """
    Filter the windows based on sensor types and station information.
    Only sensor types in 'sensor_types' will be kept.
    """
    new_wins = {}
    print("sensor_types: %s" % sensor_types)

    if verbose:
        print("channel name             |" + " " * 30 +
              "sensor type |   pick flag |   windows | total windows |")

    for sta, sta_info in windows.iteritems():
        sta_wins = {}
        for chan, chan_info in sta_info.iteritems():
            pick_flag = False
            if len(chan_info) == 0:
                continue
            try:
                # since windows are on RTZ component and
                # instruments are on NEZ compoennt, so
                # just use Z component instrument information
                zchan = chan[:-1] + "Z"
                _st = stations[zchan]["sensor"]
            except:
                continue
            if is_right_sensor(_st, sensor_types):
                sta_wins[chan] = chan_info
                pick_flag = True
            if verbose:
                print("channel(%15s) | %40s | %9s | %12d |"
                      % (chan, _st[:40], pick_flag, len(chan_info)))

        if len(sta_wins) > 0:
            new_wins[sta] = sta_wins

    return new_wins


def get_measurements_std(measurements):
    """
    Calculate the mean and standard deviation values from the measurements
    """
    comp_meas = {}

    for sta_info in measurements.itervalues():
        for chan, chan_info in sta_info.iteritems():
            comp = chan.split(".")[-1][-1]
            dts = [v["dt"] for v in chan_info]
            if comp not in comp_meas:
                comp_meas[comp] = []
            comp_meas[comp].extend(dts)

    means = dict((k, np.mean(v)) for (k, v) in comp_meas.iteritems())
    stds = dict((k, np.std(v)) for (k, v) in comp_meas.iteritems())

    print("means of measurements:")
    pprint(means)
    print("std of measurements:")
    pprint(stds)
    return means, stds


def get_user_bound(info):
    """ user specified bound for measurements filter """
    v = [-info["tshift_acceptance_level"] + info["tshift_reference"],
         info["tshift_acceptance_level"] + info["tshift_reference"]]
    if v[0] > v[1]:
        raise ValueError("error on user bound: %s" % v)
    return v


def get_std_bound(mean, std, std_ratio):
    """ bound from mean and std values from measurements """
    b = [mean - std_ratio * std, mean + std_ratio * std]
    if b[0] > b[1]:
        raise ValueError("Error on std bound: %s" % b)
    return b


def filter_measurements_on_bounds(windows, measurements, bounds):
    """
    filter the windows based on its measurements
    """
    def _filter_(chan_wins, chan_meas, bounds):
        new_wins = []
        if len(chan_wins) != len(chan_meas):
            raise ValueError("number of windows is not the same as number"
                             "measurements")
        for win, meas in zip(chan_wins, chan_meas):
            # print("%.2f, %.2f, %.2f" % (meas["dt"], bounds[0], bounds[1]))
            if (meas["dt"] >= bounds[0]) and (meas["dt"] <= bounds[1]):
                new_wins.append(win)
        return new_wins

    new_wins = {}
    for sta, sta_info in windows.iteritems():
        new_sta_wins = {}
        for chan, chan_info in sta_info.iteritems():
            if len(chan_info) == 0:
                continue
            comp = chan.split(".")[-1][-1]
            m = measurements[sta][chan]
            w = _filter_(chan_info, m, bounds[comp])
            # print(chan, ":", len(chan_info), ", ", len(w))
            if len(w) == 0:
                continue
            new_sta_wins[chan] = w

        if len(new_sta_wins) > 0:
            new_wins[sta] = new_sta_wins

    return new_wins


def filter_windows_on_measurements(windows, measurements, measure_config):
    """
    Filter windows based on measurements and threshold specified by
    the user.
    """
    # calculate standard deviation for each component
    pprint("Config:")
    pprint(measure_config)
    comp_config = measure_config['component']

    means, stds = get_measurements_std(measurements)

    final_bounds = {}
    for comp in means:
        print("-" * 20 + "\nComponent: %s" % comp)
        user_bound = get_user_bound(comp_config[comp])
        std_bound = get_std_bound(means[comp], stds[comp],
                                  comp_config[comp]["std_ratio"])

        bound = [max(std_bound[0], user_bound[0]),
                 min(std_bound[1], user_bound[1])]
        final_bounds[comp] = bound
        print("user specified bound: %s" % user_bound)
        print("std bound: %s" % std_bound)
        print("final bound: %s" % bound)

    new_wins = filter_measurements_on_bounds(windows, measurements,
                                             final_bounds)

    return new_wins


def check_consistency(windows, measurements):
    for sta, stainfo in windows.iteritems():
        for chan, chan_info in stainfo.iteritems():
            nwins_chan = len(chan_info)
            if nwins_chan == 0:
                continue
            try:
                nwins_meas = len(measurements[sta][chan])
            except KeyError as errmsg:
                raise KeyError("Missing %s(%s) in measurements: %s"
                               % (chan, sta, errmsg))
            if nwins_chan != nwins_meas:
                raise ValueError(
                    "Inconsistent between windows(%d) and measurements(%d)"
                    "from %s" % (nwins_chan, nwins_meas, chan))
    print("Pass the consistency check...")


def filter_windows(windows, stations, measurements, config, verbose=False):
    """
    Filter windows based on measurements and station information
    """
    check_consistency(windows, measurements)

    log = {}
    sensor_config = config["sensor"]
    if sensor_config["flag"]:
        print("=" * 10 + "  Filter on sensors  " + "=" * 10)
        windows_sensor = filter_windows_on_sensors(
            windows, stations, sensor_config["sensor_types"],
            verbose=verbose)
    else:
        windows_sensor = windows
    log["sensor"] = print_window_filter_summary(windows, windows_sensor)

    measure_config = config["measurement"]
    if measure_config["flag"]:
        print("=" * 10 + "  Filter on measurements  " + "=" * 10)
        windows_measure = filter_windows_on_measurements(
            windows_sensor, measurements, measure_config)
    else:
        windows_measure = windows_sensor

    log["measurement"] = print_window_filter_summary(windows_sensor,
                                                     windows_measure)

    return windows_measure, log
