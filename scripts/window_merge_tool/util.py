import sys
import json
import getopt
import os
from pyasdf import ASDFDataSet


class JSONObject():
    def __init__(self, d):
        self.__dict__ = d


def read_json_file(parfile):
    with open(parfile, 'r') as f:
        data = json.load(f, object_hook=JSONObject)
    return data


def is_mpi_env():
    """
    Test if current environment is MPI or not
    """
    try:
        import mpi4py
    except ImportError:
        return False

    try:
        import mpi4py.MPI
    except ImportError:
        return False

    if mpi4py.MPI.COMM_WORLD.size == 1 and mpi4py.MPI.COMM_WORLD.rank == 0:
        return False

    return True


def get_system_arg(argv, Usage):

    verbose = False
    parfile = None
    eventname = None

    opts, args = getopt.getopt(argv, "vp:h", ["parfile=", 'help', 'verbose'])
    for opt, value in opts:
        if opt in ("-p", "--parfile"):
            parfile = value
        elif opt in ("-v", "--verbose"):
            verbose = True
        elif opt in ("-h", "--help"):
            Usage()
            sys.exit()
        else:
            Usage()
            sys.exit()
    try:
        eventname = args[0]
    except:
        Usage()
        sys.exit()

    return eventname, verbose, parfile

def print_option():
    print "========== Command line option help =========="
    print "-p(--parfile=)  path to the parameter file"
    print "-v(--verbose)   verbose mode"
    print "-h(--help)      print out help information"


def smart_read_json(mpi_mode, json_file):
    """
    read json file under mpi and multi-processing environment
    """
    if not mpi_mode:
        json_obj = read_json_file(json_file)
    else:
        from mpi4py import MPI
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        if rank == 0:
            json_obj = read_json_file(json_file)
        else:
            json_obj = None
        json_obj = comm.bcast(json_obj, root=0)
    return json_obj


def load_asdf_file(asdf_fn):
    if not os.path.exists(asdf_fn):
        print "No asdf file: %s" % asdf_fn
    asdf_ds = ASDFDataSet(asdf_fn)
    return asdf_ds


def clean_memory(asdf_ds):
    del asdf_ds


def isclose(a, b, rel_tol=1.0e-09, abs_tol=0.0):
    return abs(a - b) <= max(rel_tol * max(abs(a), abs(b)), abs_tol)
