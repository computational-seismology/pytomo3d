import matplotlib.pyplot as plt
import numpy as np
from obspy import Trace


def calculate_misfit(tr1, tr2, taper_part=0.05):

    tr1 = tr1.copy()
    tr2 = tr2.copy()

    if not isinstance(tr1, Trace):
        raise TypeError("Input tr1(type:%s) must be type of obspy.Trace"
                        % type(tr1))
    if not isinstance(tr2, Trace):
        raise TypeError("Input tr2(type:%s) must be type of obspy.Trace"
                        % type(tr2))

    starttime = max(tr1.stats.starttime, tr2.stats.starttime)
    t_ref = starttime
    endtime = min(tr1.stats.endtime, tr2.stats.endtime)
    starttime += (endtime - starttime) * taper_part
    endtime -= (endtime - starttime) * taper_part
    sampling_rate = min(tr1.stats.sampling_rate, tr2.stats.sampling_rate)
    npts = int(abs(starttime - endtime) * sampling_rate)

    tr1_cover = (endtime - starttime) / (tr1.stats.delta * tr1.stats.npts)
    tr2_cover = (endtime - starttime) / (tr2.stats.delta * tr2.stats.npts)

    tr1.interpolate(sampling_rate, starttime=starttime, npts=npts)
    tr2.interpolate(sampling_rate, starttime=starttime, npts=npts)
    tr1.taper(max_percentage=0.05, type='hann')
    tr2.taper(max_percentage=0.05, type='hann')

    # correlation test
    corr_min = 1.0
    corr_mat = np.corrcoef(tr1.data, tr2.data)
    corr = np.min(corr_mat)
    corr_min = min(corr, corr_min)

    # least square test
    err_max = 0.0
    norm = np.linalg.norm
    sqrt = np.sqrt
    err = norm(tr1.data - tr2.data) / sqrt(norm(tr1.data) * norm(tr2.data))
    err_max = max(err, err_max)

    # amplitude diff
    twdiff = [(starttime - t_ref) + i / sampling_rate for i in range(npts)]
    amp_ref = np.max(np.abs(tr2.data))
    wdiff = (tr1.data - tr2.data) / amp_ref

    return tr1_cover, tr2_cover, corr_min, err_max, twdiff, wdiff


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
    plt.plot(times1, tr1.data, linestyle='', color='r', marker="*",
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
    tr1_cover, tr2_cover, corr_min, err_max, twdiff, wdiff = \
        calculate_misfit(tr1, tr2)

    ypos -= dypos
    plt.text(xpos, ypos, "coverage:[%6.2f%% %6.2f%%]"
             % (tr1_cover * 100, tr2_cover * 100))
    ypos -= dypos
    plt.text(xpos, ypos, "min correlation: % 6.4f" % corr_min)
    ypos -= dypos
    plt.text(xpos, ypos, "max error: % 6.4f" % err_max)
    plt.grid()

    # subplot 2
    plt.subplot(212)
    plt.plot(twdiff, wdiff, 'g', label="amplitude difference")
    plt.legend()
    plt.xlim([min(times1[0], times2[0]), max(times1[-1], times2[-1])])
    plt.grid()

    plt.tight_layout()
    if figname is None:
        plt.show()
    else:
        plt.savefig(figname)

    plt.close(fig)
