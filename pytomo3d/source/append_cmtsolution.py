#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Functions that append CMTSOLUTION information into catalog
:copyright:
    Wenjie Lei (lei@princeton.edu), 2016
:license:
    GNU Lesser General Public License, version 3 (LGPLv3)
    (http://www.gnu.org/licenses/lgpl-3.0.en.html)
"""
from __future__ import print_function, division
import obspy
from obspy.core.event.source import ResourceIdentifier
from obspy.core.event import Catalog
from obspy.core.event import CreationInfo


def _validator(event, cmt_origin, cmt_mag, cmt_focal):
    if event.preferred_origin() != cmt_origin:
        raise ValueError("preferred_origin_id wrong, not the same as the "
                         "new added cmt")
    if event.preferred_magnitude() != cmt_mag:
        raise ValueError("preferred_magnitude_id wrong, not the same as "
                         "the new added cmt")
    if event.preferred_focal_mechanism() != cmt_focal:
        raise ValueError("preferred_focal_mechanism_id wrong, not the same "
                         "as the new added cmt")


def prepare_cmt_origin(cmt, tag, creation_info):
    cmt_origin = None
    for _origin in cmt.origins:
        if str(_origin.resource_id).endswith("origin#cmt"):
            cmt_origin = _origin
            break

    if cmt_origin is None:
        raise ValueError("No cmt origin found")

    new_id = str(cmt_origin.resource_id).strip() + "#%s" % tag
    # new_id = str(cmt_origin.resource_id).replace("origin#cmt",
    #                                             "origin#%s" % tag)
    cmt_origin.resource_id = ResourceIdentifier(new_id)
    cmt_origin.creation_info = creation_info
    return cmt_origin


def prepare_cmt_mag(cmt, tag, origin_id, creation_info):
    cmt_mag = None
    for _mag in cmt.magnitudes:
        if _mag.magnitude_type == "mw":
            cmt_mag = _mag

    if cmt_mag is None:
        raise ValueError("No cmt Mw mag found")

    new_id = str(cmt_mag.resource_id).strip() + "#%s" % tag
    cmt_mag.resource_id = ResourceIdentifier(new_id)
    cmt_mag.origin_id = origin_id
    cmt_mag.creation_info = creation_info
    return cmt_mag


def prepare_cmt_focal(cmt, tag, origin_id, mag_id, creation_info):

    cmt_focal = None
    for _focal in cmt.focal_mechanisms:
        if "cmtsolution" in str(_focal.resource_id):
            cmt_focal = _focal
            break

    if cmt_focal is None:
        raise ValueError("no cmt focal found")

    focal_id = str(cmt_focal.resource_id).strip() + "#%s" % tag
    cmt_focal.resource_id = ResourceIdentifier(focal_id)
    cmt_focal.creation_info = creation_info
    tensor = cmt_focal.moment_tensor
    tensor_id = str(tensor.resource_id).strip() + "#%s" % tag
    tensor.resource_id = ResourceIdentifier(tensor_id)
    tensor.derived_origin_id = origin_id
    tensor.moment_magnitude_id = mag_id
    tensor.creation_info = creation_info
    return cmt_focal


def append_cmt_to_catalog(event, cmt, tag, change_preferred_id=True):
    """
    Add cmt to event. The cmt.resource_id will be appened tag to avoid
    tag duplication problem in event.
    :param event: the event that you want to add cmt in.
    :type event: str, obspy.core.event.Event or obspy.core.event.Catalog
    :param cmt: the cmt that you want to add to event.
    :type event: str, obspy.core.event.Event or obspy.core.event.Catalog
    :param change_preferred_id: change all preferred_id to the new added cmt
    :type change_preferred_id: bool
    """
    if isinstance(event, str):
        event = obspy.read_events(event)[0]
    elif isinstance(event, Catalog):
        event = event[0]

    if isinstance(cmt, str):
        cmt_event = obspy.read_events(cmt)[0]
    elif isinstance(cmt, Catalog):
        cmt_event = cmt_event[0]

    if not isinstance(tag, str):
        raise TypeError("tag(%s) should be type of str" % type(tag))

    creation_info = CreationInfo(author="GATG", version=tag)

    cmt_origin = prepare_cmt_origin(cmt_event, tag, creation_info)
    cmt_mag = prepare_cmt_mag(cmt_event, tag, cmt_origin.resource_id,
                              creation_info)
    cmt_focal = prepare_cmt_focal(cmt_event, tag, cmt_origin.resource_id,
                                  cmt_mag.resource_id, creation_info)
    event.origins.append(cmt_origin)
    event.magnitudes.append(cmt_mag)
    event.focal_mechanisms.append(cmt_focal)

    if change_preferred_id:
        event.preferred_origin_id = str(cmt_origin.resource_id)
        event.preferred_magnitude_id = str(cmt_mag.resource_id)
        event.preferred_focal_mechanism_id = str(cmt_focal.resource_id)
        _validator(event, cmt_origin, cmt_mag, cmt_focal)

    new_cat = Catalog()
    new_cat.append(event)

    return new_cat
