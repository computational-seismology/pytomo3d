#!/usr/bin/env python

import os
import json
import obspy
import numpy as np


def write_window_txtfile_single(results, outputdir):
    for key, sta_win in results.iteritems():
        print "window:", key, sta_win
        if sta_win is None:
            continue 
        for comp_win in sta_win:
            fn = comp_win[0].channel_id + ".win"
            fn = os.path.join(outputdir, fn)
            f = open(fn, 'w')
            f.write("%s\n" % comp_win[0].channel_id)
            f.write("%d\n" % len(comp_win))
            for win in comp_win:
                f.write("%10.2f %10.2f %10.2f %10.3f %10.3f\n"
                        % (win.relative_starttime, win.relative_endtime,
                           win.cc_shift, win.dlnA, win.max_cc_value))
            f.close()


def write_window_txtfile_combine(results, outputdir):
    fn = os.path.join(outputdir, "all.win")
    fh = open(fn, 'w')
    for key, sta_win in results.iteritems():
        if sta_win is None:
            continue
        for comp_win in sta_win:
            fh.write("%s\n" % comp_win[0].channel_id)
            fh.write("%d\n" % len(comp_win))
            for win in comp_win:
                fh.write("%10.2f %10.2f %10.2f %10.3f %10.3f\n"
                         % (win.relative_starttime, win.relative_endtime,
                           win.cc_shift, win.dlnA, win.max_cc_value))
    fh.close()


def get_json_content(window):
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
        #"phase_arrivals": window.phase_arrivals,
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


def write_window_jsonfile_single(results, outputdir):

    for key, sta_win in results.iteritems():
        if sta_win is None:
            continue
        _window_all = {}
        for _comp in sta_win:
            _window = [ get_json_content(_i) for _i in _comp]
            _window_all[_window[0]["channel_id"]] = _window
        output_json = os.path.join(outputdir, "%s.json" % key)
        with open(output_json, 'w') as fh:
            j = json.dumps(_window_all, cls=WindowEncoder, sort_keys=True, 
                    indent=2, separators=(',', ':'))
            try:
                fh.write(j)
            except TypeError:
                fh.write(j.encode())


def write_window_jsonfile_combine(results, outputdir):
 
    output_json = os.path.join(outputdir, "windows.json")
    window_all = {}
    for key, sta_win in results.iteritems():
        if sta_win is None:
            continue
        window_all[key] = {}
        _window_comp = {}
        for _comp in sta_win:
            _window = [ get_json_content(_i) for _i in _comp]
            _window_comp[_window[0]["channel_id"]] = _window
        window_all[key] = _window_comp

    with open(output_json, 'w') as fh:
        j = json.dumps(window_all, cls=WindowEncoder, sort_keys=True, 
                indent=2, separators=(',', ':'))
        try:
            fh.write(j)
        except TypeError:
            fh.write(j.encode())


def write_window_file(results, outputdir, fileformat="json", method="combine"):

    print "WRITE OUT WINDOW FILE at dir: %s" % outputdir
    fileformat = fileformat.lower()
    method = method.lower()
    if fileformat not in ['txt', 'json']:
        raise ValueError("format can only be: 1)json; 2)txt")

    if method not in ['combine', 'single']:
        raise ValueError("method can only be: 1)single; 2)combine")

    if fileformat == "txt":
        if method == "single":
            write_window_txtfile_single(results, outputdir)
        else:
            write_window_txtfile_combine(results, outputdir)
    else:
        if method == "single":
            write_window_jsonfile_single(results, outputdir)
        else:
            write_window_jsonfile_combine(results, outputdir)
