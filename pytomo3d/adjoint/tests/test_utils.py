import os
import inspect
from obspy import read
import pytomo3d.adjoint.utils as adj_utils
# import pyadjoint.adjoint_source


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

obsfile = os.path.join(DATA_DIR, "proc", "IU.KBL.obs.proc.mseed")
synfile = os.path.join(DATA_DIR, "proc", "IU.KBL.syn.proc.mseed")
winfile = os.path.join(DATA_DIR, "window", "IU.KBL..BHR.window.json")


class TestUtilOne:

    @staticmethod
    def get_fake_adjsrcs():
        obs = read(obsfile)
        adjsrcs = adj_utils.ensemble_fake_adj(obs)
        return adjsrcs

    def test_ensemble_fake_adj(self):
        adjsrcs = self.get_fake_adjsrcs()

        assert len(adjsrcs) == 3
        for adj in adjsrcs:
            assert adj.adj_src_type == "waveform_misfit"
            assert adj.misfit == 0.0
            assert adj.dt == 0.5
            assert adj.min_period == 50.0
            assert adj.max_period == 100.0
            assert adj.network == "IU"
            assert adj.station == "KBL"
            assert adj.component in ["BHR", "BHT", "BHZ"]
            assert adj.location == ""
            assert adj.measurement is None

    def test_change_adjsrc_channel_name(self):
        adjsrcs = self.get_fake_adjsrcs()
        adj_utils.change_adjsrc_channel_name(adjsrcs, "MX")
        for adj in adjsrcs:
            assert adj.component[:2] == "MX"
