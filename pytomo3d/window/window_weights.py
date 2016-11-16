#!/usr/bin/env pythoiin
# -*- coding: utf-8 -*-
"""
Calculate the adjoint source weighting based on the station and source
distribution.

:copyright:
    Wenjie Lei (lei@princeton.edu), 2016
:license:
    GNU Lesser General Public License, version 3 (LGPLv3)
    (http://www.gnu.org/licenses/lgpl-3.0.en.html)
"""
from __future__ import print_function, division, absolute_import

from collections import defaultdict
import numpy as np

from spaceweight import SpherePoint
from spaceweight import SphereDistRel


def _receiver_validator(weights, rec_wcounts, cat_wcounts):
    """
    Validate the receiver weights, and make sure it sums to
    category window counts.

    :param weights:
    :param rec_wcounts:
    :param cat_wcounts:
    :return:
    """
    for comp, comp_weights in weights.iteritems():
        wsum = 0
        for chan, chan_weight in comp_weights.iteritems():
            nwin = rec_wcounts[comp][chan]
            wsum += chan_weight * nwin

        if not np.isclose(wsum, cat_wcounts[comp]):
            raise ValueError("receiver validator fails: %f, %f" %
                             (wsum, cat_wcounts[comp]))


def calculate_receiver_window_counts(windows):
    """
    Given windows, count them based on receiver level and category
    (component) level.

    :param windows:
    :return:
    """
    rec_wcounts = defaultdict(dict)
    cat_wcounts = defaultdict(lambda: 0)
    for sta, sta_window in windows.iteritems():
        for chan, chan_win in sta_window.iteritems():
            comp = chan.split(".")[-1]
            nwin = len(chan_win)
            if nwin == 0:
                continue
            rec_wcounts[comp][chan] = nwin
            cat_wcounts[comp] += nwin

    return rec_wcounts, cat_wcounts


def assign_receiver_to_points(channels, stations):
    """
    Assign the receiver information to SpherePoint

    :param rec_wcounts:
    :param stations:
    :return:
    """
    points = []
    for chan in channels:
        component = chan.split(".")[-1][-1]
        if component == "Z":
            point = SpherePoint(stations[chan]["latitude"],
                                stations[chan]["longitude"],
                                tag=chan, weight=1.0)
        else:
            # for R and T component. In station file, there
            # are only `EN` or `12`
            echan = chan[:-1] + "E"
            chan1 = chan[:-1] + "1"
            zchan = chan[:-1] + "Z"

            if echan in stations:
                point = SpherePoint(stations[echan]["latitude"],
                                    stations[echan]["longitude"],
                                    tag=chan, weight=1.0)
            elif chan1 in stations:
                point = SpherePoint(stations[chan1]["latitude"],
                                    stations[chan1]["longitude"],
                                    tag=chan, weight=1.0)
            elif zchan in stations:
                point = SpherePoint(stations[zchan]["latitude"],
                                    stations[zchan]["longitude"],
                                    tag=chan, weight=1.0)
            else:
                raise ValueError("Can't find station information(%s)"
                                 % (chan))
        points.append(point)
    return points


def get_receiver_weights(component, center, points, max_ratio, plot=False,
                         figname_prefix=None):
    """
    Calculate the receiver weights given receiver(points) distribution.

    :param component:
    :param center:
    :param points:
    :param max_ratio:
    :param plot:
    :param figname_prefix:
    :return:
    """
    # calculate weight; otherwise, leave it as default value(1)
    weightobj = SphereDistRel(points, center=center)

    if plot:
        scan_figname = figname_prefix + ".%s.smart_scan.png" % component
    else:
        scan_figname = None

    ref_distance, cond_number = weightobj.smart_scan(
        max_ratio=max_ratio, start=0.5, gap=0.5,
        drop_ratio=0.95, plot=plot,
        figname=scan_figname)

    if plot:
        map_figname = figname_prefix + ".%s.weight.png" % component
        weightobj.plot_global_map(figname=map_figname, lon0=180.0)

    return ref_distance, cond_number


def normalize_receiver_weights(points, wcounts):
    """
    Normalize the receiver weights

    :param points:
    :param rec_wcounts:
    :return:
    """
    wsum = 0
    nwins_total = 0
    for point in points:
        nwin = wcounts[point.tag]
        nwins_total += nwin
        wsum += point.weight * nwin
    norm_factor = nwins_total / wsum

    weights = {}
    for point in points:
        weights[point.tag] = point.weight * norm_factor

    return weights


def determine_receiver_weighting(
        src, stations, windows, search_ratio=0.35, weight_flag=True,
        plot_flag=False, figname_prefix=None):
    """
    Given one station and window information, determine the receiver
    weighting
    In one asdf file, there are still 3 components, for example,
    ["BHR", "BHT", "BHZ"]. These three components should be treated
    indepandently and weights will be calculated independantly.

    :return: dict of weights which contains 3 components. Each components
        contains weights values
    """
    center = SpherePoint(src["latitude"], src["longitude"],
                         tag="source")

    rec_wcounts, cat_wcounts = calculate_receiver_window_counts(windows)

    weights = {}
    # in each components, calculate weight
    ref_dists = {}
    cond_nums = {}
    for comp, comp_info in rec_wcounts.iteritems():
        points = assign_receiver_to_points(comp_info, stations)

        if weight_flag:
            ref_dists[comp], cond_nums[comp] = \
                get_receiver_weights(comp, center, points, search_ratio,
                                     plot=plot_flag,
                                     figname_prefix=figname_prefix)
        else:
            ref_dists[comp] = None
            cond_nums[comp] = None

        weights[comp] = normalize_receiver_weights(points, rec_wcounts[comp])

    _receiver_validator(weights, rec_wcounts, cat_wcounts)

    return {"rec_weights": weights, "rec_wcounts": rec_wcounts,
            "cat_wcounts": cat_wcounts, "rec_ref_dists": ref_dists,
            "rec_cond_nums": cond_nums}


def _category_validator(weights, wcounts):
    """
    Validate the category weights

    :param weights:
    :param counts:
    :return:
    """
    wsum = 0.0
    nwins_total = 0
    for p, pinfo in weights.iteritems():
        for c in pinfo:
            wsum += weights[p][c] * wcounts[p][c]
            nwins_total += wcounts[p][c]

    if not np.isclose(wsum, nwins_total):
        raise ValueError("Category validator fails: %f, %f" %
                         (wsum, nwins_total))


def normalize_category_weights(category_ratio, cat_wcounts):
    sumv = 0
    nwins_total = 0
    for p, pinfo in cat_wcounts.iteritems():
        for c in pinfo:
            sumv += cat_wcounts[p][c] * category_ratio[p][c]
            nwins_total += cat_wcounts[p][c]

    normc = nwins_total / sumv
    weights = {}
    for p, pinfo in cat_wcounts.iteritems():
        weights[p] = {}
        for c in pinfo:
            weights[p][c] = normc * category_ratio[p][c]

    return weights


def determine_category_weighting(category_ratio, cat_wcounts):
    """
    determine the category weighting based on window counts
    in each category. The weighting ratios for different categoies
    are input parameters. So this function only normlizes
    the weights without change the ratio.

    :param weight_param: weight parameter
    :param cat_wcounts: category window counts
    """
    weights = normalize_category_weights(category_ratio, cat_wcounts)

    _category_validator(weights, cat_wcounts)

    return weights
