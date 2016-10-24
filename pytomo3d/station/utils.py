#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Methods that contains utils for adjoint sources

:copyright:
    Wenjie Lei (lei@princeton.edu), 2016
:license:
    GNU Lesser General Public License, version 3 (LGPLv3)
    (http://www.gnu.org/licenses/lgpl-3.0.en.html)
"""
from __future__ import (absolute_import, division, print_function)
from obspy import UTCDateTime
from obspy.core.inventory import Channel, Station, Network, Inventory, Site


def create_simple_inventory(network, station, latitude=None, longitude=None,
                            elevation=None, depth=None, start_date=None,
                            end_date=None, location_code="S3",
                            channel_code="MX"):
    """
    Create simple inventory with only location information,
    for ZNE component, especially usefull for synthetic data
    """
    azi_dict = {"MXZ": 0.0,  "MXN": 0.0, "MXE": 90.0}
    dip_dict = {"MXZ": 90.0, "MXN": 0.0, "MXE": 0.0}
    channel_list = []

    if start_date is None:
        start_date = UTCDateTime(0)

    # specfem default channel code is MX
    for _comp in ["Z", "E", "N"]:
        _chan_code = "%s%s" % (channel_code, _comp)
        chan = Channel(_chan_code, location_code, latitude=latitude,
                       longitude=longitude, elevation=elevation,
                       depth=depth, azimuth=azi_dict[_chan_code],
                       dip=dip_dict[_chan_code], start_date=start_date,
                       end_date=end_date)
        channel_list.append(chan)

    site = Site("N/A")
    sta = Station(station, latitude=latitude, longitude=longitude,
                  elevation=elevation, channels=channel_list, site=site,
                  creation_date=start_date, total_number_of_channels=3,
                  selected_number_of_channels=3)

    nw = Network(network, stations=[sta, ], total_number_of_stations=1,
                 selected_number_of_stations=1)

    inv = Inventory([nw, ], source="SPECFEM3D_GLOBE", sender="Princeton",
                    created=UTCDateTime.now())

    return inv
