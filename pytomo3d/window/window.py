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
