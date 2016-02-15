#!/usr/bin/env python
import os
import yaml
import pyflex
import obspy


def plot_window_figure(figure_dir, figure_id, ws, _verbose=False,
                       figure_format="pdf"):
    """
    Plot window figure out

    :param figure_dir: output figure directory
    :type figure_dir: str
    :param figure_id: figure id to distinguish windows plots, for
        example, trace id could be used, like "II.AAK.00.BHZ"
    :type figure_id: str
    :param ws: window selector object from pyflex
    :type ws: pyflex.WindowSelector
    :param _verbose: verbose output flag
    :type _verbose: bool
    :param figure_format: figure format, could be "pdf", "png" and etc.
    :type figure_format: str
    :return:
    """

    outfn = "%s.%s" % (figure_id, figure_format)
    figfn = os.path.join(figure_dir, outfn)
    if _verbose:
        print "Output window figure:", figfn
    ws.plot(figfn)


def check_period_band(period_band, frequency_band):
    if period_band is None and frequency_band is None:
        raise ValueError("Specify either period_band or frequency_band")
    if period_band is not None and frequency_band is not None:
        raise ValueError(
            "Specify only period_band or frequency_band."
            "Because they contain the same piece of information:"
            "(period_band = 1.0 / frequency_band)")

    if frequency_band is None:
        if len(period_band) != 2:
            raise ValueError("length of period_band should be 2")
        if period_band[0] >= period_band[1]:
            raise ValueError("period_band should be in ascending"
                             "order: %s" % period_band)
    else:
        if len(frequency_band) != 2:
            raise ValueError("length of frequency_band should be 2")
        if frequency_band[0] >= frequency_band[1]:
            raise ValueError("frequency_band should be in ascending"
                             "order: %s" % frequency_band)
        period_band = [1.0/frequency_band[1], 1.0/frequency_band[0]]

    return period_band


def load_config_yaml(filename):
    with open(filename) as fh:
        data = yaml.load(fh)

    return pyflex.Config(**data)


def window_trace(obs_tr, syn_tr, config, station=None,
                 event=None, _verbose=False,
                 figure_mode=False, figure_dir=None):

    if not isinstance(obs_tr, obspy.Trace):
        raise ValueError("Input obs_tr should be obspy.Trace")
    if not isinstance(syn_tr, obspy.Trace):
        raise ValueError("Input syn_tr should be obspy.Trace")
    if not isinstance(config, pyflex.Config):
        raise ValueError("Input config should be pyflex.Config")

    ws = pyflex.WindowSelector(obs_tr, syn_tr, config,
                               event=event, station=station)
    try:
        windows = ws.select_windows()
    except:
        print("Error on %s" % obs_tr.id)
        windows = []

    if figure_mode:
        plot_window_figure(figure_dir, obs_tr.id, ws, _verbose)

    if _verbose:
        print("Station %s picked %i windows" % (obs_tr.id, len(windows)))

    return windows


def window_stream(observed, synthetic, config_dict, station=None,
                  event=None, figure_mode=False, figure_dir=None,
                  _verbose=False):
    """
    Window selection function

    :param observed: observed stream
    :type observed: obspy.Stream
    :param synthetic: synthetic stream
    :type synthetic: obspy.Stream
    :param config_dict: window selection config dictionary, for example,
        {"Z": pyflex.Config, "R": pyflex.Config, "T": pyflex.Config}
    :type config_dict: dict
    :param station: station information which provids station location to
        calculate the epicenter distance
    :type station: obspy.Inventory or pyflex.Station
    :param event: event information, providing the event information
    :type event: pyflex.Event, obspy.Catalog or obspy.Event
    :param figure_mode: output figure flag
    :type figure_mode: bool
    :param figure_dir: output figure directory
    :type figure_dir: str
    :param _verbose: verbose flag
    :type _verbose: bool
    :return:
    """
    if not isinstance(observed, obspy.Stream):
        raise ValueError("Input observed should be obspy.Stream")
    if not isinstance(synthetic, obspy.Stream):
        raise ValueError("Input synthetic should be obspy.Stream")
    if not isinstance(config_dict, dict):
        raise ValueError("Input config_dict should be dict")

    all_windows = []

    components = config_dict.keys()

    for component in components:
        obs = observed.select(component=component)
        syn_tr = synthetic.select(component=component)[0]
        if not obs or not syn_tr:
            continue
        for obs_tr in obs:
            config = config_dict[component]

            windows = window_trace(
                obs_tr, syn_tr, config, station=station,
                event=event, _verbose=_verbose,
                figure_mode=figure_mode, figure_dir=figure_dir)

            if _verbose:
                print("Station %s picked %i windows" % (
                    obs_tr.id, len(windows)))

            if windows is None or len(windows) == 0:
                continue
            all_windows.append(windows)

    return all_windows
