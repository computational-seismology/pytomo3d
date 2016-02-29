Tutorial
========

The data used here should be read in by `obspy`_ as a ``Stream`` or ``Trace``.

.. _obspy: https://github.com/obspy/obspy/wiki 

1. Signal processing
--------------------

Give observed seismograms, you want to apply signal processing operations to remove the instrument response(stationxml file is required), filter to a certain band, re-sampling and rotate from ``NE`` to ``RT``. You can write your script this way::

  from pytomo3d.signal.process import process 
  from obspy import read, read_inventory

  # read in waveform data
  obs = read("II.AAK.obs.mseed")
  # read in stationxml
  inv = read_inventory("II.AAK.xml")
  # set up your filter frequency band
  pre_filt = [1/150., 1/100., 1/50., 1/40.]
  # setup cutting starttime and endtime
  starttime = stream[0].stats.starttime + 10 # second
  endtime = stream[0].stats.starttime + 3610 # second
  new_obs = process(obs, remove_response_flag=True, inventory=inv,
                    filter_flag=True, pre_filt=pre_filt,
                    starttime=starttime, endtime=endtime, 
                    resample_flag=True, sampling_rate=1.0,
                    rotate_flag=True, event_latitude=12.2,
                    event_longitude=-95.6)
  # write out processed stream
  new_obs.write("II.AAK.obs.proc.mseed", format="MSEED")

Given an synthetic stream, you want to filter, re-sample and rotate::

  new_syn = process(syn, remove_response_flag=False, inventory=inv,
                    filter_flag=True, pre_filt=pre_filt,
                    starttime=starttime, endtime=endtime, 
                    resample_flag=True, sampling_rate=1.0,
                    rotate_flag=True, event_latitude=12.2,
                    event_longitude=-95.6)
  # write out processed stream
  new_obs.write("II.AAK.syn.proc.mseed", format="MSEED")

2. Window Selection
-------------------
To make window selections, you need first prepare window selection config dictionary in python::

  {
    "Z": pyflex.Config,
    "R": pyflex.Config,
    "T": pyflex.Config
  }

And the selection script::

  window = window_on_stream(
      obsd, synt, config_dict, station=inv, event=event, figure_mode=True,
      figure_dir="./figure/", verbose=False)

3. Adjoint Sources
------------------
For adjoint source calculate, the script::

  adjsrcs = calcualte_adjsrc_on_stream(
      obsd, synt, windows, config, 'multitaper_misfit',
      figure_mode=True, figure_dir="./figure")
