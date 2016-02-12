#!/usr/bin/env python

import matplotlib as mpl
mpl.use('Agg')
import pyadjoint
from pyasdf import ASDFDataSet
import os
from procasdf.shared.utils import is_mpi_env, smart_read_json, JSONObject
from procasdf.shared.utils import load_asdf_file, clean_memory
from adj_const import config_setup
from adjsrc_function import adjsrc_wrapper
import argparse
import json
#import obspy
#import numpy as np 
from mpi4py import MPI
from functools import partial


class AdjointSourceASDF(object):

    def __init__(self, paramfile, dirfile, verbose):
        self.paramfile = paramfile
        self.dirfile   = dirfile
        self._verbose  = verbose

        # mpi variable
        self.mpi_mode = False
        self.comm = None
        self.rank = None

    def detect_env(self):
        # detect environment
        self.mpi_mode = is_mpi_env()
        if not self.mpi_mode:
            raise EnvironmentError(
                "mpi environment required for parallel"
                "running window selection")
        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
    
    @staticmethod
    def clean_outputdir(rank, outputdir, figdir):
        if rank != 0:
            return

        if not os.path.exists(outputdir):
            os.makedirs(outputdir)
        else:
            os.system("rm -rf %s" % os.path.join(outputdir, "*"))
        if not os.path.exists(figdir):
            os.makedirs(figdir)
   
    @staticmethod
    def print_info(rank, par_obj, file_obj):
        if rank != 0:
            return
        print "\n\n--- dir infor ---"
        print "obsd asdf file: %s" % file_obj.obsd_asdf
        print "obsd tag:", par_obj.obsd_tag
        print "synt asdf file: %s" % file_obj.synt_asdf
        print "synt tag:", par_obj.synt_tag
        print "window file: %s" % file_obj.window
        print "output dir:", file_obj.outputdir
        print "--- selection param ---"
        print "period band:", par_obj.period_band
        print "selection mode:", par_obj.selection_mode
        print "figure mode:", par_obj.figure_mode
        print "\n"

    def _read_window_json_mpi(self, json_file, mpi_mode=True):
        """
        read window json file under mpi and multi-processing environment
        """
        def _read_json_file(filename):
            with open(filename) as fh:
                return json.load(fh)

        if not mpi_mode:
            json_obj = _read_json_file(json_file)
        else:
            from mpi4py import MPI
            comm = MPI.COMM_WORLD
            rank = comm.Get_rank()
            if rank == 0:
                json_obj = _read_json_file(json_file)
            else:
                json_obj = None
            json_obj = comm.bcast(json_obj, root=0)
        return json_obj


    def _launch(self, par_obj, file_obj):

        # load obsd and synt asdf file
        obsd_ds = load_asdf_file(file_obj.obsd_asdf)
        synt_ds = load_asdf_file(file_obj.synt_asdf)

        # read window file
        window = self._read_window_json_mpi(file_obj.window, mpi_mode=self.mpi_mode)

        if not obsd_ds.mpi:
            raise EnvironmentError("MPI environment is not launched"
                                   "successfully in this running")
      
        self.print_info(self.rank, par_obj, file_obj)

        # output dir
        outputdir = file_obj.outputdir
        figdir = os.path.join(outputdir, "figure")

        self.clean_outputdir(self.rank, outputdir, figdir)

        # adjoint source calculation
        results = self._core(par_obj, obsd_ds, synt_ds, window, 
                             figdir, outputdir, self._verbose)

        # Output all adjoint src in ASDF (feature saved for later).
        # write out adjoint source on master node
        #if self.rank == 0:
        #    write_adjsrc_file(results, outputdir)

        clean_memory(obsd_ds)
        clean_memory(synt_ds)


    def _core(self, parlist, obsd_ds, synt_ds, window, figdir, outputdir, _verbose):                 
                                                                                
        event = obsd_ds.events[0]

        obsd_tag = parlist.obsd_tag                                                 
        synt_tag = parlist.synt_tag                                                 
        period = parlist.period_band                                                
        figure_mode = parlist.figure_mode                                           
        selection_mode = parlist.selection_mode

        adjsrc_func = partial(adjsrc_wrapper, obsd_tag=obsd_tag, synt_tag=synt_tag,     
                          period=period, event=event, window=window,  
                          selection_mode=selection_mode,                            
                          figure_mode=figure_mode, figure_dir=figdir,               
                          _verbose=_verbose,outputdir=outputdir)                                        
                                                                   
        # results: A dictionary for each station with gathered values.
        results = obsd_ds.process_two_files_without_parallel_output(synt_ds, adjsrc_func)     

        return results               

    def smart_run(self):
        self.detect_env()

        # read paramter json file
        parlist  = smart_read_json(self.paramfile, self.mpi_mode)
        filelist = smart_read_json(self.dirfile, self.mpi_mode)

        print parlist

        if isinstance(parlist, list) and isinstance(filelist, list):
            if len(parlist) != len(filelist):
                raise ValueError("Lengths of params and dirs files"
                                 "do not match")
            for _par, _file in zip(parlist, filelist):
                print "_par",  dir(_par)
                print "_file", dir(_file)
                self._launch(_par, _file)

        elif isinstance(parlist, JSONObject) and isinstance(filelist, JSONObject):
            self._launch(parlist, filelist)

        else:
            raise ValueError("Problem in params and dirs files...Check")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', action='store', dest='params', required=True)
    parser.add_argument('-f', action='store', dest='files',  required=True)
    parser.add_argument('-v', action='store_true', dest='verbose')
    args = parser.parse_args()

    adjsrc = AdjointSourceASDF(args.params, args.files, args.verbose)
    adjsrc.smart_run()
