import os
import inspect
import pytest
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy.testing as npt
import pytomo3d.signal.compare_trace as ct
from obspy import read


def _upper_level(path, nlevel=4):
    """
    Go the nlevel dir up
    """
    for i in range(nlevel):
        path = os.path.dirname(path)
    return path


# Most generic way to get the data folder path.
TESTBASE_DIR = _upper_level(os.path.abspath(
    inspect.getfile(inspect.currentframe())), 4)
DATA_DIR = os.path.join(TESTBASE_DIR, "tests", "data")

# synfile = os.path.join(DATA_DIR, "raw", "IU.KBL.syn.mseed")
# testsyn = read(synfile)
small_mseed = os.path.join(DATA_DIR, "raw", "BW.RJOB.obs.mseed")
smallobs = read(small_mseed)

obs = read(os.path.join(DATA_DIR, "raw", "IU.KBL.obs.mseed"))
syn = read(os.path.join(DATA_DIR, "raw", "IU.KBL.syn.mseed"))


def reset_matplotlib():
    """
    Reset matplotlib to a common default.
    """
    # Set all default values.
    mpl.rcdefaults()
    # Force agg backend.
    plt.switch_backend('agg')


def test_least_square_error():
    d1 = np.array([1, 2, 3])
    d2 = 2 * d1
    err = ct.least_squre_error(d1, d2)
    npt.assert_almost_equal(err, 1/np.sqrt(2))

    d1 = np.random.random(10)
    d2 = 2 * d1
    err = ct.least_squre_error(d1, d2)
    npt.assert_almost_equal(err, 1/np.sqrt(2))


def test_cross_correlation():
    d1 = np.random.random(10)
    d2 = 2 * d1
    corr = ct.cross_correlation(d1, d2)
    npt.assert_almost_equal(corr, 1.0)

    d1 = np.random.random(10)
    d2 = -2 * d1
    corr = ct.cross_correlation(d1, d2)
    npt.assert_almost_equal(corr, -1.0)

    d1 = np.array([1, 2, 3, 4, 3, 2, 1])
    d2 = np.array([1, 3, 4, -6, 4, 3, 1])
    corr = ct.cross_correlation(d1, d2)
    npt.assert_almost_equal(corr, -0.37849937)


def test_calculate_misfit():

    tr1 = smallobs[0]
    tr2 = smallobs[0]

    res = ct.calculate_misfit(tr1, tr2)

    npt.assert_allclose(res["tr1_coverage"], 1.0, rtol=1e-3)
    npt.assert_allclose(res["tr2_coverage"], 1.0, rtol=1e-3)
    npt.assert_allclose(res["correlation"], 1.0)
    npt.assert_allclose(res["error"], 0.0)


def test_trace_length():
    l1 = ct.trace_length(smallobs[0])
    l2 = smallobs[0].stats.endtime - smallobs[0].stats.starttime
    npt.assert_almost_equal(l1, l2)


def test_plot_two_trace_raise():
    with pytest.raises(TypeError):
        ct.calculate_misfit(obs[0], syn)

    with pytest.raises(TypeError):
        ct.calculate_misfit(obs, syn[0])


def test_plot_two_trace(tmpdir):

    reset_matplotlib()
    figname = os.path.join(str(tmpdir), "trace_compare.png")
    ct.plot_two_trace(smallobs[0], smallobs[0].copy(), figname=figname)


def test_plot_two_traces_raise(tmpdir):

    reset_matplotlib()
    figname = os.path.join(str(tmpdir), "trace_compare.png")

    with pytest.raises(TypeError):
        ct.plot_two_trace(smallobs, smallobs[0].copy(), figname=figname)

    with pytest.raises(TypeError):
        ct.plot_two_trace(smallobs[0], smallobs.copy(), figname=figname)
