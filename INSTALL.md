# Installing python dependencies

Pytomo3d has dependancies on the following packages:

1. [obspy](https://github.com/obspy/obspy)
2. [pyflex *devel* branch](https://github.com/wjlei1990/pyflex)
3. [pyadjoint *dev* branch](https://github.com/chukren/pyadjoint)

---

### Manual installation

Wenjie: If you are new to python, [anaconda](https://www.continuum.io/downloads) is recommmended. Please download the newest version( >= Anaconda2 - 2.5.0) since it already contains a lot of useful python packages, like pip, numpy and scipy.  Older versions is not recommended since it usually has compliers inside, like gfortran and gcc. It is always better to use comiplers coming from your system rather than the very old ones embeded in anaconda. If you are expert in python, please choose the way you like.

1. downwnload Anaconda for Python 2.7 and 64 bit Linux and install it (http://continuum.io/downloads)(**optional**)

2. install obspy using anaconda.
  ```
  conda install -c obspy obspy=0.10.2
  ```
  Recently, obspy group has a big upgrade for obspy, which boost the version number from 0.10.x to 1.0.0. A lot of kernel functions has changed its module path. I currently keep the old version. Going to update it very soon.

  Or install from source code:
  ```
  git clone https://github.com/obspy/obspy.git
  cd obspy
  pip install -v -e .
  cd ..
  ```

3. install pyflex.
  ```
  git clone --branch devel https://github.com/wjlei1990/pyflex 
  cd pyflex
  pip install -v -e .
  cd ..
  ```

4. Install pyadjoint
  ```
  git clone --branch dev https://github.com/chukren/pyadjoint 
  cd pyadjoint
  pip install -v -e .
  cd ..
  ```

5. Install pytomo3d.
  ```
  git clone https://github.com/wjlei1990/pytomo3d
  cd pytomo3d
  pip install -v -e .
  cd ..
  ```

### Script installation

Wenjie: recommended for experienced user

  ```
  pip install -r requirements.txt
  ```

---

After installation, you can run `py.test` in pytomo3d directory to see if you installed all the things correctly.

### Notes
1. If you already have some of the packages, please make sure to update them(not including anaconda)
