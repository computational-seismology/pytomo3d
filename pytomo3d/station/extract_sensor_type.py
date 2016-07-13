#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This script extract channel instrument type from obspy.inventory

:copyright:
    Wenjie Lei (lei@princeton.edu), 2016
:license:
    GNU Lesser General Public License, version 3 (LGPLv3)
    (http://www.gnu.org/licenses/lgpl-3.0.en.html)
"""

from __future__ import print_function, division, absolute_import
import os
from obspy import Inventory
from obspy import read_inventory


def safe_load_staxml(staxmlfile):
    try:
        inv = read_inventory(staxmlfile)
    except Exception as exp:
        raise("Failed to parse staxml file(%s) due to: %s"
              % (staxmlfile, exp))
    return inv


def extract_sensor_type(staxml):
    instruments = {}

    if isinstance(staxml, Inventory):
        inv = staxml
    else:
        if os.path.isfile(staxml):
            inv = safe_load_staxml(staxml)
        else:
            raise ValueError("Input staxml is neither obspy.Inventory or "
                             "staxml file")
    for nw in inv:
        nw_code = nw.code
        for sta in nw:
            sta_code = sta.code
            for chan in sta:
                chan_code = chan.code
                loc_code = chan.location_code
                key = "%s.%s.%s.%s" % (nw_code, sta_code, loc_code, chan_code)
                if chan.sensor.description is not None:
                    instruments[key] = chan.sensor.description
                elif chan.sensor.type is not None:
                    instruments[key] = chan.sensor.type
                else:
                    instruments[key] = "None"

    return instruments
