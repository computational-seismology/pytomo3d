import os
import inspect
import pytomo3d.source.append_cmtsolution as app_cmt


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

testquakeml = os.path.join(DATA_DIR, "quakeml", "C201009031635A.xml")
testcmt = os.path.join(DATA_DIR, "quakeml", "C201009031635A.inv")


def test_append_cmt_to_catalog():
    tag = "GATG_M15"
    new_cat = app_cmt.append_cmt_to_catalog(
        testquakeml, testcmt, tag, change_preferred_id=True)

    assert new_cat[0].preferred_origin()
    assert new_cat[0].preferred_magnitude()
    assert new_cat[0].preferred_focal_mechanism()
