from obspy.core.util.geodetics import gps2DistAzimuth
from math import pi, cos, sin


SMALL_DEGREE = 0.01


def check_orthogonality(azim1, azim2):
    """
    Check if two azimuth are orthogonal
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


def rotate_certain_angle(d1, d2, angle):
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
    dnew1 = d1 * cos(angle) + d2 * sin(angle)
    dnew2 = -d1 * sin(angle) + d2 * cos(angle)
    return dnew1, dnew2


def rotate_12_RT(d1, d2, baz, azim1, azim2):
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
    angle = (baz + 180.0 - azim1) * 2 * pi / 360.
    r, t = rotate_certain_angle(d1, d2, angle)

    return r, t


def rotate_RT_12(r, t, baz, azim1, azim2):
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
    angle = - (baz + 180.0 - azim) * 2 * pi / 360.
    d1, d2 = rotate_certain_angle(r, t, angle)

    if "right" in status:
        return d2, d1
    elif "left" in status:
        return d1, d2


def rotate_12_NE(d1, d2, azim1, azim2):
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
    angle = - azim1 * 2 * pi / 360.
    n, e = rotate_certain_angle(d1, d2, angle)

    return n, e


def rotate_NE_12(n, e, azim1, azim2):
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
    angle = (azim) * 2 * pi / 360.
    d1, d2 = rotate_certain_angle(n, e, angle)

    if "right" in status:
        return d2, d1
    elif "left" in status:
        return d1, d2


def extract_channel_orientation_info(tr, inv):
    try:
        sta_inv = inv[0][0]
        loc = tr.stats.location
        chan = tr.stats.channel
        inv = sta_inv.select(location=loc, channel=chan)[0]
        inc = inv.dip
        azi = inv.azimuth
        return inc, azi
    except:
        return None, None


def rotate_12_RT_func(st, inv, method="12->RT", back_azimuth=None):
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

            output_1, output_2 = rotate_12_RT(i_1.data, i_2.data, back_azimuth,
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


def rotate_stream(st, event_latitude, event_longitude,
                  station_latitude=None, station_longitude=None,
                  inventory=None, mode="ALL"):
    """
    Rotate a stream to radial and transverse components based on the
    station information and event information

    :param st: input stream
    :type st: obspy.Stream
    :param event_latitude: event latitude
    :type event_latitude: float
    :param event_longitude: event longitude
    :type event_longitude: float
    :param station_latitude: station latitude. If not provided, extract
        information from inventory
    :type station_latitude: float
    :param station_longitude: station longitude. If not provided, extrace
        information from inventory
    :type station_longitude: float
    :param inv: station inventory information. If you want to rotate
    "12" components, you need to provide inventory since only station
    and station_longitude is not enough.
    :type inv: obspy.Inventory
    :param mode: rotation mode, could be one of:
        1) "NE": rotate only North and East channel to RT
        2) "12": rotate only 1 and 2 channel, like "BH1" and "BH2" to RT
        3) "all": rotate all components to RT
    :return: rotated stream(obspy.Stream)
    """

    if station_longitude is None or station_latitude is None:
        station_latitude = float(inventory[0][0].latitude)
        station_longitude = float(inventory[0][0].longitude)

    mode = mode.upper()
    if mode not in ["NE", "ALL", "12"]:
        raise ValueError("rotate_stream supports mode: 1) 12; 2) NE; 3) ALL")
    if mode in ["12", "ALL"] and inventory is None:
        raise ValueError("Mode %s required inventory(stationxml) "
                         "information provided" % mode)

    _, baz, _ = gps2DistAzimuth(station_latitude, station_longitude,
                                event_latitude, event_longitude)

    components = [tr.stats.channel[-1] for tr in st]

    if mode in ["NE", "ALL"]:
        if "N" in components and "E" in components:
            try:
                st.rotate(method="NE->RT", back_azimuth=baz)
            except Exception as e:
                print e

    if mode in ["12", "ALL"]:
        if "1" in components and "2" in components:
            try:
                rotate_12_RT_func(st, inventory, back_azimuth=baz)
            except Exception as e:
                print e
