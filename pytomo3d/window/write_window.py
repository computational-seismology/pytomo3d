#!/usr/bin/env python

import json
import obspy
import numpy as np


def write_txtfile(windows, filename):
    """
    Write windows to text file

    :param windows: list of windows(from same observed and synthetic)
    :type windows: list
    :param filename: output filename
    :type filename: str
    :return:
    """
    with open(filename, 'w') as fh:
        fh.write("%s\n" % windows[0].channel_id)
        fh.write("%d\n" % len(windows))
        for win in windows:
            fh.write("%10.2f %10.2f %10.2f %10.3f %10.3f\n"
                     % (win.relative_starttime, win.relative_endtime,
                        win.cc_shift, win.dlnA, win.max_cc_value))


def get_json_content(window):
    """
    Extract information from json to a dict

    :param window:
    :return:
    """
    info = {
        "left_index": window.left,
        "right_index": window.right,
        "center_index": window.center,
        "channel_id": window.channel_id,
        "time_of_first_sample": window.time_of_first_sample,
        "max_cc_value":  window.max_cc_value,
        "cc_shift_in_samples":  window.cc_shift,
        "cc_shift_in_seconds":  window.cc_shift_in_seconds,
        "dlnA":  window.dlnA,
        "dt": window.dt,
        "min_period": window.min_period,
        "phase_arrivals": window.phase_arrivals,
        "absolute_starttime": window.absolute_starttime,
        "absolute_endtime": window.absolute_endtime,
        "relative_starttime": window.relative_starttime,
        "relative_endtime": window.relative_endtime,
        "window_weight": window.weight}

    return info


class WindowEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, obspy.UTCDateTime):
            return str(obj)
        # Numpy objects also require explicit handling.
        elif isinstance(obj, np.int64):
            return int(obj)
        elif isinstance(obj, np.int32):
            return int(obj)
        elif isinstance(obj, np.float64):
            return float(obj)
        elif isinstance(obj, np.float32):
            return float(obj)
        # Let the base class default method raise the TypeError
        return json.JSONEncoder.default(self, obj)


def write_jsonfile(windows, filename):
    """
    Write windows to a json file

    :param windows: list of windows
    :param filename: output filename
    :return:
    """

    win_json = [get_json_content(_i) for _i in windows]
    with open(filename, 'w') as fh:
        j = json.dumps(win_json, cls=WindowEncoder, sort_keys=True,
                       indent=2, separators=(',', ':'))
        try:
            fh.write(j)
        except TypeError:
            fh.write(j.encode())
