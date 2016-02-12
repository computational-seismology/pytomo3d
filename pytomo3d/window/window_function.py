#!/usr/bin/env python

import pyflex
import os
from win_const import config_repo 


def plot_window_figure(figdir, obs_tr, ws, _verbose):
    """
    Plot and save window figure
    """
    outfn = "%s.pdf" % obs_tr.id
    figfn = os.path.join(figdir, outfn)
    if _verbose:
        print "Output fig:", figfn
    ws.plot(figfn)


def window_function(observed, synthetic, stationxml, period=[27, 60],
                    event=None, selection_mode="body_waves",
                    figure_mode=False, figure_dir=None,
                    _verbose=False):
    """
    Basic window function
    """

    all_windows = []

    for component in ["Z", "R", "T"]:
        obs = observed.select(component=component)
        syn = synthetic.select(component=component)
        if not obs or not syn:
            continue
        for obs_tr in obs:
            config = config_repo(mode=selection_mode, period=period,
                                 component=component)
            ws = pyflex.WindowSelector(obs_tr, syn, config,
                                       event=event, station=stationxml)
            try:
                windows = ws.select_windows()
            except:
                print("Error on %s" % obs_tr.id)
                continue

            if figure_mode:
                plot_window_figure(figure_dir, obs_tr, ws, _verbose)
            if windows is None or len(windows) == 0:
                continue
            if _verbose:
                print("Station %s picked %i windows" % ( 
                    obs_tr.id, len(windows)))
            all_windows.append(windows)
    return all_windows


def window_wrapper(obsd_station_group, synt_station_group, 
                   obsd_tag="proc_obsd_27_60", synt_tag="proc_synt_27_60", 
                   period=[27, 60], event=None, selection_mode="body_waves",
                   figure_mode=False, figure_dir=None,
                   _verbose=False):
    """
    Wrapper for asdf IO
    """

    # Make sure everything thats required is there.
    if not hasattr(synt_station_group, "StationXML") or \
            not hasattr(obsd_station_group, obsd_tag) or \
            not hasattr(synt_station_group, synt_tag):
        print "Missing attr, return"
        return

    stationxml = synt_station_group.StationXML
    observed = getattr(obsd_station_group, obsd_tag)
    synthetic = getattr(synt_station_group, synt_tag)

    return window_function(observed, synthetic, stationxml,
                           period=period, event=event,
                           selection_mode=selection_mode,
                           figure_mode=figure_mode, figure_dir=figure_dir,
                           _verbose=_verbose)

