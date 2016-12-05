#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Basic Methods that handles rotation of seismograms as
extension to Obspy. It can rotate `12`, `EN` and `RT`,
forward and backward.

:copyright:
    Wenjie Lei (lei@princeton.edu), 2016
:license:
    GNU Lesser General Public License, version 3 (LGPLv3)
    (http://www.gnu.org/licenses/lgpl-3.0.en.html)
"""
from __future__ import print_function
from math import cos, sin
from numpy import deg2rad


SMALL_DEGREE = 0.01


def check_orthogonality(azim1, azim2):
    """
    Check if two azimuth are orthogonal, check whether
    (azim1, azim2, vertical) forms a left-hand or right-hand
    coordinate system.
    Remember the defination of azimuth is angle between north
    direction.
    Unit is degree, not radian.
    """
    azim1 = (azim1 + 360) % 360
    azim2 = (azim2 + 360) % 360

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
    Basic rotating function which rotate d1 and d2 by angle.
    d1 and d2 should be orthogonal to each other and form a
    'left-handed' coordinate system together with vertical
    component.

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

    if len(d1) != len(d2):
        raise ValueError("Length of d1(%d) and d2(%d) are not the same!"
                         % (len(d1), len(d2)))

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
    Rotate from any two orthogonal horizontal components to EN components.
    The azimuth of the two horizontal components are specified by azim1
    and azim2.

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

    # caculate the angle of rotation
    d1, d2 = rotate_certain_angle(n, e, azim)

    if "right" in status:
        return d2, d1
    elif "left" in status:
        return d1, d2
