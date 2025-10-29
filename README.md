# TAT-pulsar (Timing Analysis Toolkit for pulsar astrophysics)
Hey there! This is a Python toolbox made for folks interested in messing around with pulsar timing analysis, especially in high-energy scenarios. You know those pesky corrections needed for stuff like the barycentric correction and binary systems? Yeah, we’ve got functions for those. Plus, we’ve included some timing analysis tools and some examples to help you get started with your pulsar timing exploration. Enjoy!

[![DOI:<your number>](https://zenodo.org/badge/DOI/10.5281/zenodo.6784362.svg)](<https://zenodo.org/record/6784362#.Yr5jiC8RqrM>)
[![Upload Python Package](https://github.com/tuoyl/tat-pulsar/actions/workflows/python-publish.yml/badge.svg)](https://github.com/tuoyl/tat-pulsar/actions/workflows/python-publish.yml)
[![codecov](https://codecov.io/github/tuoyl/tat-pulsar/branch/master/graph/badge.svg?token=LShslbL6pg)](https://codecov.io/github/tuoyl/tat-pulsar)
<a href="https://ascl.net/2404.004"><img src="https://img.shields.io/badge/ascl-2404.004-blue.svg?colorB=262255" alt="ascl:2404.004" /></a>

## Install the Package using `pip`
1. Open your terminal or command prompt.
2. Type in the following command:
```pip install tat-pulsar```
  
## Install the Package using conda

## Download the repository

Download the whole repository to your local directory using `git clone` or `git fetch`.

For example, In you local path execute:

```plaintext
git clone https://github.com/tuoyl/tat-pulsar.git
```

And you will get the folder `tat-pulsar`, in the folder you will see a file named `setup.py`. We will install the whole package based on this script.

## Create a conda environment

The most elegant thing to do before installing is to create a new conda environment to avoid conflicts with your existing python environment.

```plaintext
conda create -n pulsar-timing python=3
```

after downloading the dependancies, execute

```plaintext
conda activate pulsar-timing
```

to enter the pulsar-timing environment of conda, you will see `(pulsar-timing)` before the shell prompt.

## Install the repository

Now you are all set to install the repository. In the directory where the setup.py located, execute:

```
python3 -m pip install -e .
```

## Uninstall the repository

if you want to uninstall the package for generating the product.

```
python3 -m pip uninstall tat-pulsar
```

## Command line tools

Installing the project exposes a set of convenience entry points that wrap the utilities in `tatpulsar/launcher`:

- `tatpulsar-fold2d` folds event lists into phase-resolved histograms.
- `tatpulsar-residuals` calculates and plots timing residuals from par/tim pairs.
- `tatpulsar-gti-generator` launches the interactive GTI selection helper.
- `tatpulsar-tempo-gui` opens the PyQt timing analysis interface described below.

## Timing analysis GUI

The package now bundles a PyQt-based application that wraps the TEMPO2-style timing workflow. After installing the package (either via `pip` or from source) you can launch the interface with:

```
tatpulsar-tempo-gui
```

From the main window you can:

- Load `*.par` and `*.tim` files, inspect individual TOAs, and toggle them in or out of the fit.
- Choose which pulsar parameters are free, then run a weighted least-squares fit using PINT.
- Review interactive residual plots and the weighted RMS of the solution.
- Export the curated TOA list and the fitted timing model to new files.
