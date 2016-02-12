from setuptools import setup, find_packages

setup(
    name="pytomo3d",
    version="0.0.1",
    license='GNU General Public License, Version 3 (GPLv3)',
    description="Python toolkits for seismic tomograpy",
    author="Wenjie Lei",
    author_email="lei@princeton.edu",
    url="https://github.com/wjlei1990/pytomo3d",
    packages=find_packages(),
    zip_safe=False,
    classifiers=[
        # complete classifier list:
        # http://pypi.python.org/pypi?%3Aaction=list_classifiers
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Operating System :: Unix",
        "Operating System :: POSIX",
        "Operating System :: Microsoft :: Windows",
        "Programming Language :: Python",
        "Programming Language :: Python :: 2.7",
        "Programming Language :: Python :: Implementation :: CPython",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Physics",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)"
    ],
    keywords=[
        "seismology", "tomography", "adjoint", "signal", "inversion", "window"
    ],
    install_requires=[
        "obspy", "flake8", "pytest", "nose", "future>=0.14.1", "pyflex",
        "pyadjoint"
    ],
    extras_require={
        "docs": ["sphinx", "ipython", "runipy"]
    }
)
