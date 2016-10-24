"""
# functions mainly for pypaw.
"""
import os
import json
import numpy as np


def merge_instruments_window(sta_win):
    """
    Merge windows from the same channel, for example, if
    there are windows from 00.BHZ and 10.BHZ, keep only one
    with the most windows
    """
    if len(sta_win) == 0:
        return sta_win

    sort_dict = {}
    for trace_id, trace_win in sta_win.iteritems():
        chan = trace_id.split('.')[-1][0:2]
        loc = trace_id.split('.')[-2]
        if chan not in sort_dict:
            sort_dict[chan] = {}
        if loc not in sort_dict[chan]:
            sort_dict[chan][loc] = {"traces": [], "nwins": 0}
        sort_dict[chan][loc]["traces"].append(trace_id)
        sort_dict[chan][loc]["nwins"] += len(trace_win)

    choosen_wins = {}
    for chan, chan_info in sort_dict.iteritems():
        if len(chan_info.keys()) == 1:
            choosen_loc = chan_info.keys()[0]
        else:
            _locs = []
            _nwins = []
            for loc, loc_info in chan_info.iteritems():
                _locs.append(loc)
                _nwins.append(loc_info["nwins"])
            _max_idx = np.array(_nwins).argmax()
            choosen_loc = _locs[_max_idx]

        choosen_traces = sort_dict[chan][choosen_loc]["traces"]
        for _trace_id in choosen_traces:
            choosen_wins[_trace_id] = sta_win[_trace_id]

    return choosen_wins


def merge_channels_window(sta_win):
    """
    Merge windows from different channels.
    This step should be done after merge instruments windows
    because after that there will only one instrument left
    on one channel
    """
    sort_dict = {}

    if len(sta_win) == 0:
        return sta_win

    for trace_id, trace_win in sta_win.iteritems():
        chan = trace_id.split(".")[-1][0:2]
        if chan not in sort_dict:
            sort_dict[chan] = {"traces": [], "nwins": 0}
        sort_dict[chan]["traces"].append(trace_id)
        sort_dict[chan]["nwins"] += len(trace_win)

    choosen_wins = {}
    if len(sort_dict.keys()) == 1:
        choosen_chan = sort_dict.keys()[0]
    else:
        _chans = []
        _nwins = []
        for chan, chan_info in sort_dict.iteritems():
            _chans.append(chan)
            _nwins.append(chan_info["nwins"])
        _max_idx = np.array(_nwins).argmax()
        choosen_chan = _chans[_max_idx]

    choosen_traces = sort_dict[choosen_chan]["traces"]
    for _trace_id in choosen_traces:
        choosen_wins[_trace_id] = sta_win[_trace_id]

    return choosen_wins


def merge_station_windows(windows):
    """
    Merge windows for one station group.
    1) merge multiple instruments: keep the instrument with the most
    number of windows
    2) merge channels
    """
    w = merge_instruments_window(windows)
    w = merge_channels_window(w)
    return w


def stats_all_windows(windows, obsd_tag, synt_tag,
                      instrument_merge_flag,
                      outputdir):
    """
    Generate window statistic information
    """

    window_stats = {"obsd_tag": obsd_tag, "synt_tag": synt_tag,
                    "instrument_merge_flag": instrument_merge_flag,
                    "stations": 0, "stations_with_windows": 0}
    for sta_name, sta_win in windows.iteritems():
        if sta_win is None:
            continue
        nwin_sta = 0
        for trace_id, trace_win in sta_win.iteritems():
            comp = trace_id.split(".")[-1]
            if comp not in window_stats:
                window_stats[comp] = {"window": 0, "traces": 0,
                                      "traces_with_windows": 0}
            window_stats[comp]["window"] += len(trace_win)
            if len(trace_win) > 0:
                window_stats[comp]["traces_with_windows"] += 1
            window_stats[comp]["traces"] += 1
            nwin_sta += len(trace_win)

        window_stats["stations"] += 1
        if nwin_sta > 0:
            window_stats["stations_with_windows"] += 1

    filename = os.path.join(outputdir, "windows.stats.json")
    with open(filename, "w") as fh:
        json.dump(window_stats, fh, indent=2, sort_keys=True)
