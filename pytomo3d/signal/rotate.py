#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Methods that handles rotation of seismograms

:copyright:
    Wenjie Lei (lei@princeton.edu), 2016
:license:
    GNU Lesser General Public License, version 3 (LGPLv3)
    (http://www.gnu.org/licenses/lgpl-3.0.en.html)
"""
from __future__ import print_function
from obspy.core.util.geodetics import gps2DistAzimuth
from math import cos, sin
from obspy import Stream
import numpy as np
from numpy import deg2rad


SMALL_DEGREE = 0.01


def calculate_baz(elat, elon, slat, slon):
    _, _, baz = gps2DistAzimuth(elat, elon, slat, slon)
    return baz


def check_orthogonality(azim1, azim2):
    """
    Check if two azimuth are orthogonal, check whether
    (azim1, azim2, vertical) forms a left-hand or right-hand
    coordinate system.
    Remember the defination of azimuth is angle between north
    direction.
    Unit is degree, not radian.
    """
    azim1 = azim1 % 360
    azim2 = azim2 % 360

    if abs(abs(azim1 - azim2) - 90.0) < SMALL_DEGREE:
        if abs(azim1 - azim2 - 90.0) < SMALL_DEGREE:
            return "right-hand"
        elif abs(azim1 - azim2 + 90.0) < SMALL_DEGREE:
            # should be orthogonal; otherwise return
            return "left-hand"
        else:
            return False
    else:
        # cross 360 degree
        if abs(azim1 - azim2 + 270.0) < SMALL_DEGREE:
            return "right-hand"
        elif abs(azim1 - azim2 - 270.0) < SMALL_DEGREE:
            return "left-hand"
        else:
            return False


def rotate_certain_angle(d1, d2, angle, unit="degree"):
    """
    Basic rotating function.

    (d1, d2, Vertical) should form a left-handed coordinate system, i.e.,
                    Azimuth_{d2} = Azimuth_{d1} + 90.0
    For example, (North, East, Vertical) & (Radial, Transverse, Vertical)
    are both left-handed coordinate systems. The return value (dnew1,
    dnew2, vertic) should also form a left-handed coordinate system.
    The angle is azimuth differnce between d1 and dnew1, i.e.,
                    angle = Azimuth_{dnew1} - Azimuth_{d1}

    :type d1: :class:`~numpy.ndarray`
    :param d1: Data of one of the two horizontal components
    :type d2: :class:`~numpy.ndarray`
    :param d2: Data of one of the two horizontal components
    :type angle: float
    :param angle: component azimuth of data2
    :return: two new components after rotation
    """
    if unit == "degree":
        angle = deg2rad(angle)
    elif unit == "radian":
        angle = angle
    else:
        raise ValueError("Unregonized unit(%s): 1) degree; 2) radian"
                         % unit)

    dnew1 = d1 * cos(angle) + d2 * sin(angle)
    dnew2 = -d1 * sin(angle) + d2 * cos(angle)
    return dnew1, dnew2


def rotate_12_rt(d1, d2, baz, azim1, azim2):
    """
    Rotate from any two orthogonal horizontal components to RT components

    :type d1: :class:`~numpy.ndarray`
    :param d1: Data of one of the two horizontal components
    :type d2: :class:`~numpy.ndarray`
    :param d2: Data of the other horizontal components
    :type baz: float
    :param baz: the back azimuth from station to source in degrees
    :type azim1: float
    :param azim1: component azimuth of data1
    :type azim2: float
    :param azim2: component azimuth of data2
    :return: Radial and Transeversal component of seismogram. (None, None)
        returned if input two components are not orthogonal
    """
    status = check_orthogonality(azim1, azim2)
    if not status:
        # raise ValueError("azim1 and azim2 not orthogonal")
        return None, None
    if "right" in status:
        # flip to left-hand
        d1, d2 = d2, d1
        azim1, azim2 = azim2, azim1

    if len(d1) != len(d2):
        # raise ValueError("Component 1 and 2 have different length")
        return None, None
    if baz < 0 or baz > 360:
        raise ValueError("Back Azimuth should be between 0 and 360 degree")

    # caculate the angle of rotation
    angle = baz + 180.0 - azim1
    r, t = rotate_certain_angle(d1, d2, angle)

    return r, t


def rotate_rt_12(r, t, baz, azim1, azim2):
    """
    Rotate from any two orthogonal horizontal components to RT components

    :type data1: :class:`~numpy.ndarray`
    :param data1: Data of one of the two horizontal components
    :type data2: :class:`~numpy.ndarray`
    :param data2: Data of the other horizontal components
    :type baz: float
    :param baz: the back azimuth from station to source in degrees
    :type azim1: float
    :param azim1: component azimuth of data1
    :type azim2: float
    :param azim2: component azimuth of data2
    :return: Radial and Transeversal component of seismogram.
    """
    status = check_orthogonality(azim1, azim2)
    if not status:
        raise ValueError("azim1 and azim2 not orthogonal")
    if "left" in status:
        azim = azim1
    elif "right" in status:
        azim = azim2

    if len(r) != len(t):
        raise TypeError("Component R and T have different length")
    if baz < 0 or baz > 360:
        raise ValueError("Back Azimuth should be between 0 and 360 degree")

    # caculate the angle of rotation
    angle = - (baz + 180.0 - azim)
    d1, d2 = rotate_certain_angle(r, t, angle)

    if "right" in status:
        return d2, d1
    elif "left" in status:
        return d1, d2


def rotate_12_ne(d1, d2, azim1, azim2):
    """
    Rotate from any two orthogonal horizontal components to EN components

    :type d1: :class:`~numpy.ndarray`
    :param d1: Data of one of the two horizontal components
    :type d2: :class:`~numpy.ndarray`
    :param d2: Data of the other horizontal components
    :type azim1: float
    :param azim1: component azimuth of data1
    :type azim2: float
    :param azim2: component azimuth of data2
    :return: East and North component of seismogram.
    """
    status = check_orthogonality(azim1, azim2)
    if not status:
        raise ValueError("azim1 and azim2 not orthogonal")
    if "right" in status:
        # flip to left-hand
        d1, d2 = d2, d1
        azim1, azim2 = azim2, azim1

    if len(d1) != len(d2):
        raise TypeError("Component 1 and 2 have different length")

    # caculate the angle of rotation
    n, e = rotate_certain_angle(d1, d2, -azim1)

    return n, e


def rotate_ne_12(n, e, azim1, azim2):
    """
    Rotate from East and North components to give two orghogonal horizontal
    components. Returned values are (d1, d2) and (d1, d2, Vertical) will
    form a left-handed coordinate system.

    :type data1: :class:`~numpy.ndarray`
    :param data1: Data of one of the two horizontal components
    :type data2: :class:`~numpy.ndarray`
    :param data2: Data of the other horizontal components
    :type azim1: float
    :param azim1: component azimuth of data1
    :type azim2: float
    :param azim2: component azimuth of data2
    :return: two horizontal orthogonal seismogram after rotation.
    """
    status = check_orthogonality(azim1, azim2)
    if not status:
        raise ValueError("azim1 and azim2 not orthogonal")
    if "left" in status:
        azim = azim1
    elif "right" in status:
        azim = azim2

    if len(n) != len(e):
        raise TypeError("Component North and East have different length")

    # caculate the angle of rotation
    d1, d2 = rotate_certain_angle(n, e, azim)

    if "right" in status:
        return d2, d1
    elif "left" in status:
        return d1, d2


def extract_channel_orientation_info(tr, inv):
    try:
        nw = tr.stats.network
        sta = tr.stats.station
        loc = tr.stats.location
        chan = tr.stats.channel
        chan_inv = inv.select(network=nw, station=sta, location=loc,
                              channel=chan)[0][0][0]
        dip = chan_inv.dip
        azi = chan_inv.azimuth
        return dip, azi
    except:
        return None, None


def rotate_12_rt_func(st, inv, method="12->RT", back_azimuth=None):
    """
    Rotate 12 component to RT

    :param st: input stream
    :param inv: station inventory information
    :param method: rotation method
    :param back_azimuth: back azimuth(station to event azimuth)
    :return: rotated stream
    """
    if method != "12->RT":
        raise ValueError("rotate_12_RT only supports method = 12->RT now")
    input_components, output_components = method.split("->")
    if len(input_components) == 2:
        input_1 = st.select(component=input_components[0])
        input_2 = st.select(component=input_components[1])
        for i_1, i_2 in zip(input_1, input_2):
            # check
            dt = 0.5 * i_1.stats.delta
            if (len(i_1) != len(i_2)) or \
                    (abs(i_1.stats.starttime - i_2.stats.starttime) > dt) \
                    or (i_1.stats.sampling_rate != i_2.stats.sampling_rate):
                msg = "All components need to have the same time span."
                raise ValueError(msg)
                continue
            inc1, azi1 = extract_channel_orientation_info(i_1, inv)
            inc2, azi2 = extract_channel_orientation_info(i_2, inv)
            if inc1 is None or inc2 is None \
                    or inc1 != 0.0 or inc2 != 0.0:
                continue

            output_1, output_2 = rotate_12_rt(i_1.data, i_2.data, back_azimuth,
                                              azi1, azi2)
            if output_1 is None or output_2 is None:
                continue

            i_1.data = output_1
            i_2.data = output_2
            # Rename the components
            i_1.stats.channel = i_1.stats.channel[:-1] + output_components[0]
            i_2.stats.channel = i_2.stats.channel[:-1] + output_components[1]
            # Add the azimuth backj to stats object
            for comp in (i_1, i_2):
                comp.stats.back_azimuth = back_azimuth
    return st


def _check_inventory_orientation(inv):
    """
    Check the sanity of inventory from one group of instrument, like
    "II.AAK.00.BH*"
    Error code based on binary calculation
    ZNE = Z * 8 + N * 4 + E * 2 + Ortho * 1
    The first three digits stands for three components, the last digits
    is for checking whether two horizontal components are vertical to
    each other. For example,
    "8(1000)"  -> Z component error
    "6(0110)"  -> N,E component error, but still orthogonal to each other
    "15(1110)" -> Z,N,E component error, but "NE" are still orthogonal
                  to each other
    """
    def __is_subset(dict1, dict2):
        # check if dict1 is subset of dict2
        for key in dict1:
            if key not in dict2:
                return False
            if isinstance(dict1[key], float):
                if not np.isclose(dict1[key], dict2[key]):
                    return False
            else:
                if dict1[key] != dict2[key]:
                    return False
        return True

    error = 0
    # right information of components
    right_info = {"Z": {"dip": 90., "azi": 0.0},
                  "N": {"dip": 0., "azi": 0.0},
                  "E": {"dip": 0., "azi": 90.0}}
    error_code = {"Z": 8, "N": 4, "E": 2}
    # extract real value out from inventory
    real_info = {}
    for comp in right_info.keys():
        try:
            comp_inv = inv.select(channel="*%s" % comp)[0][0][0]
        except IndexError:
            continue
        real_info[comp] = {"dip": abs(comp_inv.dip), "azi": comp_inv.azimuth}

    # compare real_info with error_info
    for comp in real_info:
        if not __is_subset(real_info[comp], right_info[comp]):
            error += error_code[comp]

    # check NE orthogonality
    if "N" in real_info and "E" in real_info:
        if not check_orthogonality(real_info["N"]["azi"],
                                   real_info["E"]["azi"]):
            error += 1

    return error


def _check_inventory_sanity(stream, inventory):
    """
    Checks the inventory sanity based on the stream. The stream should
    contains only one station. If one component doesn't pass the sanity
    check, it will be removed from the stream.
    """
    nw = stream[0].stats.network
    sta = stream[0].stats.station
    loc = stream[0].stats.location
    chan = stream[0].stats.channel[0:2]
    sta_inv = inventory.select(network=nw, station=sta, location=loc,
                               channel="%s*" % chan)
    err_code = _check_inventory_orientation(inventory)

    _format_bin = bin(err_code)[2:].zfill(4)
    if _format_bin[0] == "1":
        for tr in stream.select(component="Z"):
            stream.remove(tr)
    if _format_bin[1] == "1":
        for tr in stream.select(component="N"):
            stream.remove(tr)
    if _format_bin[2] == "1":
        for tr in stream.select(component="E"):
            stream.remove(tr)
    return stream, sta_inv


def rotate_one_station_stream(st, event_latitude, event_longitude,
                              station_latitude=None, station_longitude=None,
                              inventory=None, mode="NE->RT",
                              sanity_check=False):
    """
    Rotate the stream from the same network, station, location and channel
    code, for example the stream should only contains traces whose ids are
    "II.AAK.00.BH*"
    """

    mode = mode.upper()
    mode_options = ["NE->RT", "ALL->RT", "12->RT", "RT->NE"]
    if mode not in mode_options:
        raise ValueError("rotate_stream mode(%s) should be within %s"
                         % (mode, mode_options))
    if mode[0:2] == "RT" and sanity_check:
        raise ValueError("if rotating from RT to NE, then set "
                         "sanity_check to False")

    if inventory is None and \
            (station_latitude is None and station_longitude is None):
        raise ValueError("You need either specify station inventory "
                         "or station location information")

    if mode in ["12->RT", "ALL->RT"] and inventory is None:
        raise ValueError("If you want to rotate 12, you need to specify "
                         "station inventory")

    if len(sort_stream_by_station(st)) != 1:
        raise ValueError("This method only accepts stream that contains "
                         "one station")

    if sanity_check:
        st, inventory = _check_inventory_sanity(st, inventory)

    if station_latitude is None or station_longitude is None:
        nw = st[0].stats.network
        station = st[0].stats.station
        _inv = inventory.select(network=nw, station=station)
        try:
            station_latitude = float(_inv[0][0].latitude)
            station_longitude = float(_inv[0][0].longitude)
        except Exception as err:
            print("Error extracting station('%s.%s') info:%s"
                  % (nw, station, err))
            return

    baz = calculate_baz(event_latitude, event_longitude,
                        station_latitude, station_longitude)

    components = [tr.stats.channel[-1] for tr in st]

    if mode in ["12->RT", "ALL->RT"]:
        if "1" in components and "2" in components:
            try:
                rotate_12_rt_func(st, inventory, back_azimuth=baz)
            except Exception as e:
                print("Error rotating 12->RT:%s" % e)

    if mode in ["NE->RT", "ALL->RT"]:
        if "N" in components and "E" in components:
            try:
                st.rotate(method="NE->RT", back_azimuth=baz)
            except Exception as e:
                print("Error rotating NE->RT:%s" % e)

    if mode in ["RT->NE"]:
        st.rotate(method="RT->NE", back_azimuth=baz)

    return st


def sort_stream_by_station(st):
    """
    Sort the traces in stream. Group the traces from same network,
    station, location, and channel code together
    """
    ntotal = len(st)
    sta_dict = {}
    n_added = 0
    for tr in st:
        nw = tr.stats.network
        station = tr.stats.station
        loc = tr.stats.location
        chan = tr.stats.channel[0:2]
        station_id = "%s.%s.%s.%s" % (nw, station, loc, chan)
        if station_id not in sta_dict:
            sta_dict[station_id] = st.select(network=nw, station=station,
                                             location=loc)
            n_added += len(sta_dict[station_id])

    if n_added != ntotal:
        raise ValueError("Sort stream by station errors: number of traces "
                         "is inconsistent")
    return sta_dict


def rotate_stream(st, event_latitude, event_longitude,
                  inventory, mode="ALL->RT", sanity_check=False):
    """
    Rotate a stream to radial and transverse components based on the
    station information and event information

    :param st: input stream
    :type st: obspy.Stream
    :param event_latitude: event latitude
    :type event_latitude: float
    :param event_longitude: event longitude
    :type event_longitude: float
    :param inv: station inventory information. If you want to rotate
    "12" components, you need to provide inventory since only station
    and station_longitude is not enough.
    :type inv: obspy.Inventory
    :param mode: rotation mode, could be one of:
        1) "NE->RT": rotate only North and East channel to RT
        2) "12->RT": rotate only 1 and 2 channel, like "BH1" and "BH2" to RT
        3) "ALL->RT": rotate all components to RT
        4) "RT->NE": rotate RT to NE
    :param sanity_check: check the sanity of inventory, mianly of the
        orientation of ZNE components.
        1) If rotating observed data from NE(12)to RT, it is recommended
            to set to True; if rotating from RT to NE, then you could set
            it to False;
        2) If rotating synthetic data, you could set it to False since
            I assume for synthetic data, there should not be problem
            associated with orientation
    :return: rotated stream(obspy.Stream)
    """

    rotated_stream = Stream()

    mode = mode.upper()
    mode_options = ["NE->RT", "ALL->RT", "12->RT", "RT->NE"]
    if mode not in mode_options:
        raise ValueError("rotate_stream mode(%s) should be within %s"
                         % (mode, mode_options))
    if mode[0:2] == "RT" and sanity_check:
        raise ValueError("if rotating from RT to NE, then set "
                         "sanity_check to False")

    if mode in ["12->RT", "ALL->RT"] and inventory is None:
        raise ValueError("Mode %s required inventory(stationxml) "
                         "information provided(to rotate '12')" % mode)

    # the stream might contains multiple stations so each station should
    # be rotated indepandantly
    sorted_st_dict = sort_stream_by_station(st)

    for sta_stream in sorted_st_dict.itervalues():
        # loop over stations
        nw = sta_stream[0].stats.network
        station = sta_stream[0].stats.station
        loc = sta_stream[0].stats.location
        chan = sta_stream[0].stats.channel

        if loc == "S3" or chan[0:2] == "MX":
            # SPECFEM TUNE: if the synthetic is generated by SPECFEM
            # and the stationxml is from read data, then the location
            # won't match between synt and staxml
            station_inv = inventory.select(network=nw, station=station)
        else:
            station_inv = inventory.select(network=nw, station=station,
                                           location=loc)

        _st = \
            rotate_one_station_stream(sta_stream, event_latitude,
                                      event_longitude, inventory=station_inv,
                                      mode=mode, sanity_check=sanity_check)
        if _st is not None:
            rotated_stream += _st

    return rotated_stream
