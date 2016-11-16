#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Functions that compare two traces, return the measurements metrics.

:copyright:
    Wenjie Lei (lei@princeton.edu), 2016
:license:
    GNU Lesser General Public License, version 3 (LGPLv3)
    (http://www.gnu.org/licenses/lgpl-3.0.en.html)
"""
import numpy as np
import matplotlib.pyplot as plt
from obspy import Trace


def least_squre_error(data1, data2):
    """
    waveform difference between data1 and data2
    :param data1:
    :param data2:
    :return:
    """
    # least square test
    err_max = 0.0
    norm = np.linalg.norm
    err = norm(data1 - data2) / np.sqrt(norm(data1) * norm(data2))
    err_max = max(err, err_max)
    return err_max


def cross_correlation(data1, data2):
    """
    :param data1:
    :param data2:
    :return:
    """
    # correlation test
    corr_min = 1.0
    corr_mat = np.corrcoef(data1, data2)
    corr = np.min(corr_mat)
    corr_min = min(corr, corr_min)
    return corr_min


def trace_length(tr):
    return (tr.stats.npts-1) * tr.stats.delta


def calculate_misfit(_tr1, _tr2, taper_flag=True, taper_percentage=0.05,
                     correlation_flag=True):
    """
    Calculate the misfit between two traces
    :param tr1: trace 1
    :type tr1: Obspy.Trace
    :param tr2: trace 2
    :type tr2: Obspy.Trace
    :param taper_flag: taper the seismogram or not
    :type taper_flag: bool
    :param taper_percentage: the taper percentage
    :type taper_percentage: float
    """
    if not isinstance(_tr1, Trace):
        raise TypeError("Input tr1(type:%s) must be type of obspy.Trace"
                        % type(_tr1))
    if not isinstance(_tr2, Trace):
        raise TypeError("Input tr2(type:%s) must be type of obspy.Trace"
                        % type(_tr2))

    tr1 = _tr1.copy()
    tr2 = _tr2.copy()

    starttime = max(tr1.stats.starttime, tr2.stats.starttime)
    endtime = min(tr1.stats.endtime, tr2.stats.endtime)
    sampling_rate = min(tr1.stats.sampling_rate, tr2.stats.sampling_rate)
    npts = int((endtime - starttime) * sampling_rate)

    tr1.interpolate(sampling_rate, starttime=starttime, npts=npts)
    tr2.interpolate(sampling_rate, starttime=starttime, npts=npts)

    if taper_flag:
        tr1.taper(max_percentage=taper_percentage, type='hann')
        tr2.taper(max_percentage=taper_percentage, type='hann')

    corr_min = cross_correlation(tr1.data, tr2.data)
    err_max = least_squre_error(tr1.data, tr2.data)

    # coverage
    tr1_cover = trace_length(tr1) / trace_length(_tr1)
    tr2_cover = trace_length(tr2) / trace_length(_tr2)

    # amplitude diff
    twdiff = [i / sampling_rate for i in range(npts)]
    amp_ref = np.sum(np.abs(tr1.data) + np.abs(tr2.data)) / (2 * npts)
    wdiff = (tr1.data - tr2.data) / amp_ref

    return {"tr1_coverage": tr1_cover, "tr2_coverage": tr2_cover,
            "correlation": corr_min, "error": err_max,
            "time_array": twdiff, "diff_array": wdiff}


def plot_two_trace(tr1, tr2, trace1_tag="trace 1", trace2_tag="trace 2",
                   figname=None):

    if not isinstance(tr1, Trace):
        raise TypeError("Input tr1(type:%s) must be type of obspy.Trace"
                        % type(tr1))
    if not isinstance(tr2, Trace):
        raise TypeError("Input tr2(type:%s) must be type of obspy.Trace"
                        % type(tr2))

    fig = plt.figure(figsize=(20, 10))

    # subplot 1
    plt.subplot(211)
    t1 = tr1.stats.starttime
    t2 = tr2.stats.starttime
    t_ref = max(t1, t2)

    bt = t1 - t_ref
    times1 = [bt + i * tr1.stats.delta for i in range(tr1.stats.npts)]
    plt.plot(times1, tr1.data, linestyle='-', color='r', marker="*",
             markersize=3, label=trace1_tag, markerfacecolor='r',
             markeredgecolor='none')

    bt = t2 - t_ref
    times2 = [bt + i * tr2.stats.delta for i in range(tr2.stats.npts)]
    plt.plot(times2, tr2.data, '-', color="b", linewidth=0.7,
             label=trace2_tag)

    plt.xlim([min(times1[0], times2[0]), max(times1[-1], times2[-1])])
    plt.legend(loc="upper right")

    xmax = plt.xlim()[1]
    ymin = plt.ylim()[0]
    xpos = 0.7 * xmax
    ypos = 0.4 * ymin
    dypos = abs(0.1 * ymin)

    plt.text(xpos, ypos, "trace id:['%s', '%s']" % (tr1.id, tr2.id))
    ypos -= dypos
    plt.text(xpos, ypos, "reference time: %s" % t_ref)
    ypos -= dypos
    plt.text(xpos, ypos, "detaT: [%6.3f, %6.3f]" % (tr1.stats.delta,
                                                    tr2.stats.delta))

    # calcualte misfit
    res = calculate_misfit(tr1, tr2)

    ypos -= dypos
    plt.text(xpos, ypos, "coverage:[%6.2f%% %6.2f%%]"
             % (res["tr1_coverage"] * 100, res["tr2_coverage"] * 100))
    ypos -= dypos
    plt.text(xpos, ypos, "min correlation: % 6.4f" % res["correlation"])
    ypos -= dypos
    plt.text(xpos, ypos, "max error: % 6.4f" % res["error"])
    plt.grid()

    # subplot 2
    plt.subplot(212)
    plt.plot(res["time_array"], res["diff_array"], 'g',
             label="amplitude difference")
    plt.legend()
    plt.xlim([min(times1[0], times2[0]), max(times1[-1], times2[-1])])
    plt.grid()

    plt.tight_layout()
    if figname is None:
        plt.show()
    else:
        plt.savefig(figname)

    plt.close(fig)
