import os
import inspect
import pytest
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


def test_calculate_misfit():

    tr1 = smallobs[0]
    tr2 = smallobs[0]

    tr1_c, tr2_c, corr, err, _, _ = ct.calculate_misfit(tr1, tr2)

    npt.assert_allclose(tr1_c, 0.90220, rtol=1e-3)
    npt.assert_allclose(tr2_c, 0.90220, rtol=1e-3)
    npt.assert_allclose(corr, 1.0)
    npt.assert_allclose(err, 0.0)


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
