### Intro

In this directory, there are tools used to process the asdf files

### Usage

There are now two ways to implement the parallel processing.

1. multi-processing

```
python process_observed.py -p /path/to/params.json -f /path/to/dir.json -v
```
2. mpi

```
mpiexec -n 4 process_observed.py -p /path/to.params.json -f /path/to/dir.json -v
```
Same usage for the *process_synthetic.py*.

There are two parameter files it needs.

1. params.json: processing parameter. See example at ./parfile/proc\_obsd.params.json
  * "filter\_band": filter period band, unit is second 
  * "old\_tag": tag in the old asdf file
  * "new\_tag": tag in the new asdf file
  * "relative\_startime": interpolation starttime(relative to event centroid time), unit is second
  * "relative\_endtime": interpolation endtime(relative to event centroid time), unit is second
  * "interp\_deltat": interpolation deltat(time step), unit is second

2. dirs.json: directory and file parameter. See example at ./parfile/proc\_obsd.dirs.json
  * "input\_asdf": intput asdf file, the one waiting to be processed(already exists)
  * "output\_asdf": output asdf file, the one generated(new one to be generated)

A new type of json format are adopted recently that allows you to put a list of params and dirs into one json file. Please check the example at "./example\_run/parfile/proc\_obsd.params.json" and "./example\_run/parfile/proc\_obsd.dirs.json".

### Notes

1. Examples are at "./example\_run"
