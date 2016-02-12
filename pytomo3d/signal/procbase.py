#!/usr/bin/env python

import os
from procasdf.shared.utils import JSONObject, smart_read_json, is_mpi_env
from proc_util import process
from pyasdf import ASDFDataSet
from functools import partial


class ProcBase(object):

    def __init__(self, paramfile, dirfile, verbose):
        """
        """
        self.mpi_mode = False
        self.rank = None
        self.comm = None

        self._detect_env()

        self.parlist = self.__parse(paramfile)
        self.filelist = self.__parse(dirfile)

        self._verbose = verbose
        
    @staticmethod
    def __parse(info):
        """
        If info is a json file(then read in)
        """
        if type(info) is str:
            info = smart_read_json(info)
        elif type(info) in (list, dict):
            return info
        else:
            raise TypeError("__parse only accepts json file, dictionary or"
                            "list")
        return info
    
    def clean_dir(self, input_asdf, output_asdf):

        def clean_dir_subs(input_asdf, output_asdf):
            if not os.path.exists(input_asdf):
                raise IOError("No input asdf file found:%s" % input_asdf)

            if input_asdf == output_asdf:
                raise ValueError("Current version does not allow "
                                 "input_asdf == output_asdf")

            outputdir = os.path.dirname(output_asdf)
            if not os.path.exists(outputdir):
                os.makedirs(outputdir)
            if os.path.exists(output_asdf):
                print("output_asdf exists and removed:%s" % output_asdf)
                os.remove(output_asdf)

        if self.mpi_mode: 
            if self.rank == 0:
                clean_dir_subs(input_asdf, output_asdf)
            self.comm.Barrier()
        else:
            clean_dir_subs(input_asdf, output_asdf)

    def print_info(self, parlist, file_list):

        def print_info_subs(parlist, file_list):
            print "="*20
            print "Dir info:"
            print "input asdf:", file_list.input_asdf
            print "output asdf:", file_list.output_asdf
            print "Processing info:"
            print "filter band:", parlist.filter_band
            print "tag map: {\"%s\" ==> \"%s\"}" % (
                    parlist.old_tag, parlist.new_tag)
            print "interp relative start and end(to event" \
                  "centroid time): (%-8.2f, %8.2f)" \
                  %(parlist.relative_starttime, parlist.relative_endtime)
            print "interp deltat:", parlist.interp_deltat

        if self.mpi_mode: 
            if self.rank == 0:
                print_info_subs(parlist, file_list)
        else:
            print_info_subs(parlist, file_list) 

    def _detect_env(self):
        # detect the environment
        self.mpi_mode = is_mpi_env()
        if self.mpi_mode:
            from mpi4py import MPI
            self.comm = MPI.COMM_WORLD
            self.rank = self.comm.Get_rank()

    def _core(self, *arg, **keywords):
        pass

    def _launch(self, par_obj, file_obj):
        """
        Process observed asdf file
        """
        input_asdf = file_obj.input_asdf
        output_asdf = file_obj.output_asdf

        filter_band = par_obj.filter_band
        old_tag = par_obj.old_tag
        new_tag = par_obj.new_tag
        relative_starttime = par_obj.relative_starttime
        relative_endtime = par_obj.relative_endtime
        interp_deltat = par_obj.interp_deltat

        if self._verbose:
            self.print_info(par_obj, file_obj)

        self.clean_dir(input_asdf, output_asdf)

        self._core(input_asdf, output_asdf,
                   filter_band=filter_band,
                   old_tag=old_tag, new_tag=new_tag,
                   relative_starttime=relative_starttime,
                   relative_endtime=relative_endtime,
                   interp_deltat=interp_deltat)

    def smart_run(self):
        """
        check whether params_file and dir_file contains mulitple 
        elements(list). If so, process all of them one by one
        """
        parlist = self.parlist
        filelist = self.filelist

        if isinstance(parlist, list) and isinstance(filelist, list):
            if len(parlist) != len(filelist):
                raise ValueError("length of params and dirs are not the same")
            for _par, _file in zip(parlist, filelist):
                self._launch(_par, _file)
        elif isinstance(parlist, JSONObject) and \
                isinstance(filelist, JSONObject):
            self._launch(parlist, filelist)
        else:
            raise ValueError("params and dirs file do not match")


class ProcOBSD(ProcBase):

    def __init__(self, paramfile, dirfile, verbose):
        """
        """
        ProcBase.__init__(self, paramfile, dirfile, verbose)

    def _core(self, asdf_fn, outputfn, filter_band=None, old_tag=None,             
              new_tag=None, relative_starttime=0.0, 
              relative_endtime=3600.0, interp_deltat=1.0):                                            
                                                                                
        ds = ASDFDataSet(asdf_fn, compression=None, debug=False)                    
                                                                                
        [period1, period2, period3, period4] = filter_band                          
        pre_filt = [1.0/period4, 1.0/period3, 1.0/period2, 1.0/period1]             
                                                                                
        # read in event                                                             
        event = ds.events[0]                                                        
        origin = event.preferred_origin() or event.origins[0]                       
        event_latitude = origin.latitude                                            
        event_longitude = origin.longitude                                          
        event_time = origin.time                                                    
                                                                                
        # figure out interpolation parameter                                        
        starttime = event_time + relative_starttime                                 
        endtime = event_time + relative_endtime                                     
        sampling_rate = 1.0 / interp_deltat                                         
                                                                                
        new_tag = "proc_obsd_%i_%i" % (int(period2), int(period3))                  
        tag_map = { old_tag : new_tag }
                                                                                
        process_function = \
            partial(process, remove_response=True, pre_filt=pre_filt,               
                    starttime=starttime, endtime=endtime,                           
                    resample_flag=True, sampling_rate=sampling_rate,                
                    rotate_flag=True, event_latitude=event_latitude,                
                    event_longitude=event_longitude)                                
                                                                                
        ds.process(process_function, outputfn, tag_map=tag_map)                     
        del ds                             


class ProcSYNT(ProcBase):

    def __init__(self, paramfile, dirfile, verbose):
        """
        """
        ProcBase.__init__(self, paramfile, dirfile, verbose)

    def _core(self, asdf_fn, outputfn, filter_band=None, old_tag=None,             
              new_tag=None, relative_starttime=0.0,                          
              relative_endtime=3600.0, interp_deltat=1.0):                   
        # read in dataset                                                           
        ds = ASDFDataSet(asdf_fn, compression=None, debug=False)                    
                                                                                
        [period1, period2, period3, period4] = filter_band                          
        pre_filt = [1.0/period4, 1.0/period3, 1.0/period2, 1.0/period1]             
                                                                                
        # read in event                                                             
        event = ds.events[0]                                                        
        for _origin in event.origins:                                               
            if "cmt" in str(_origin.resource_id):                                   
                origin = _origin                                                    
        event_latitude = origin.latitude                                            
        event_longitude = origin.longitude                                          
        event_time = origin.time                                                    
                                                                                
        # Figure out interpolation parameters                                       
        starttime = event_time + relative_starttime                                 
        endtime = event_time + relative_endtime                                     
        sampling_rate = 1.0 / interp_deltat                                         
        npts = (endtime - starttime) * sampling_rate                                
                                                                                
        tag_map = { old_tag : new_tag }                                             
                                                                                
        process_function = \
            partial(process, remove_response=False, pre_filt=pre_filt,              
                    starttime=starttime, endtime=endtime,                           
                    resample_flag=True, sampling_rate=sampling_rate,                
                    rotate_flag=True, event_latitude=event_latitude,                
                    event_longitude=event_longitude)                                
                                                                                
        ds.process(process_function, outputfn, tag_map=tag_map)
 
