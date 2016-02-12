### Intro

Tools used for window selection(mpi required)

### Usage:

```
mpiexec -n numproc python parallel_pyflex.py -p /path/to/parameter/json/file -f /path/to/dir/json -v
```

### Notes
1. paramter json file
  * "obsd_asdf": path to the input observed asdf file
  * "synt_asdf": path to the input synthetic asdf file
  * "outputdir": path to the output window directory

2. dir json file
  * "period_band": [period_min, period_max], same as the orignal flexwin setting
  * "obsd_tag": observed waveform tag in asdf file
  * "synt_tag": synthetic waveform tag in asdf file
  * "selection mode": selection mode, could be ["body_wave", "surface_wave"]
  * "figure mode": flag whether output window figure or not. Figure output could slow down the program a lot since it writes out large(and a lot) figures.

### Required packages

#### pyflex

Should be installed from:
```https://github.com/wjlei1990/pyflex```, development branch
