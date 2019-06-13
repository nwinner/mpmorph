from __future__ import division, unicode_literals

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.signal import savgol_filter as savgol
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import math, os, sys, time

from pymatgen.io.lammps.outputs import parse_lammps_dumps
from pymatgen.io.vasp.outputs import Xdatcar

__author__ = "Nicholas Winner"
__copyright__ = "None"
__version__ = "0.2"
__maintainer__ = "Nicholas Winner"
__email__ = "nwinner@berkeley.edu"
__status__ = "Development"
__date__ = ""


def power_spectrum(v, window_func='Gaussian'):

    """
    This function calculates the power spectral density for 1D array, v. It is very similar to the autocorrelation
    function, which is the more common function to use. The only difference in these is that there is no final inverse
    fast Fourier transform in the calculation, and that this function only supports the FFT method, and no "exact"
    method.

    :param v:
    :return:
    """

    if window_func == 'Gaussian':
        sigma = 2 * math.sqrt(2 * math.log(2))
        window = signal.gaussian(len(v), std=4000.0 / sigma, sym=False)
    elif window_func == 'BH':
        window = signal.blackmanharris(len(v), sym=False)
    elif window_func == 'Hamming':
        window = signal.hamming(len(v), sym=False)
    elif window_func == 'Hann':
        window = signal.hann(len(v), sym=False)

    WE = sum(window) / len(v)
    wf = window / WE
    # convolve the window function.
    if isinstance(v[0], (list, np.ndarray, tuple)):
        sig = v * wf[None, :].T
    else:
        sig = v * wf.T

    # A series of number of zeros will be padded to the end of the DACF \
    # array before FFT.
    N = zero_padding(sig)

    yfft = np.fft.fft(sig, N, axis=0) / len(sig)

    if window_func == 'None':
        yfft = np.fft.fft(v, n=int(N), axis=0) / len(v)
        return np.square(np.absolute(yfft))
    else:
        return np.square(np.absolute(yfft))


def dump_to_df(filename, write_csv=True, output="data.csv"):
    """
    A helper function that takes a lammps dump file and returns a Pandas Dataframe. It is also recommended to write
    a CSV file. This has the benefits:
        (1) CSV files take up less space than dump files.
        (2) It is far more efficient to parse massive amounts of lammps data entirely through Pandas instead of
            using a list of Pandas DataFrames, as Pymatgen currently provides, it just requires a little more knowledge
            of how to use Pandas efficiently. A single CSV can be read as a DF and then processed.
        (3) We can pre-sort the particles. LAMMPS does not retain the order of its particles in the dump file, which
            can be very annoying for post processing. When the csv is written, it sorts the Pd dataframe so that at
            each time step, the particles are listed in order of their id.

    Args:
        filename: (str) file name of the lammps dump.
        write_csv: (bool) Whether or not to write csv file
        output: (str) file name to output the csv. Include ".csv" in your filename.

    Returns:
        Pandas Dataframe of the dump
    """

    dump = parse_lammps_dumps(filename)
    dfs  = []
    for frame in dump:
        dfs.append(frame.data)
        dfs[-1]['Timestep'] = frame.timestep
    df = pd.concat(dfs).sort_values(by=['Timestep', 'id']).reset_index(drop=True)

    if write_csv:
        df.to_csv(output)

    return df


def xdatcar_to_df(xdat="XDATCAR"):

    if isinstance(xdat, str):
        structures = Xdatcar(xdat).structures
    else:
        structures = xdat
    dfs = []

    for i, s in enumerate(structures):
        coords = [site.coords for site in s.sites]
        x = [c[0] for c in coords]
        y = [c[0] for c in coords]
        z = [c[0] for c in coords]
        t = [site.specie for site in s.sites]
        ids = [i+1 for i in range(len(s.sites))]

        dfs.append(pd.DataFrame.from_dict({'x':x,'y':y,'z':z,'type':t,'Id':ids}))
        dfs[-1]['Timestep'] = i+1

    df = pd.concat(dfs)

    return df


def _xdatcar_to_df(filename="XDATCAR", write_csv=True, output="xdatcar.csv"):
    """
    Parses a DATCAR file into a Pandas Dataframe, and by default writes a csv file, which is quicker to read when
    doing further analysis. By default, it is equipped for reading XDATCAR, but the format is general. So, if you
    have a VDATCAR for velocities for example. You could read that as well.

    Args:
        filename: (str) Name of the file to read.
        header: [list] The values that are going to be read from the columns of the DATCAR file. Default='XDATCAR'
        write_csv: (bool) Whether or not to write the pandas dataframe to a csv file. Defaults to True.
        output: (str) Name of the csv file to write. Default='xdatcar.csv'
    Returns:
        Pandas Dataframe of the datcar
    """

    f = open(filename)
    lines = f.readlines()
    f.close()

    spec = lines[5].split()
    num  = lines[6].split()
    num = list(map(int, num))
    N    = 0

    species = []
    for i in range(len(num)):
        for j in range(num[i]):
            species.append(spec[i])

    N = sum(num) # Number of atoms/ions

    dfs = []
    j = 8
    while j+N+8 < len(lines):
        #temp = [ [float(val) for val in lines[i].split()] for i in range(j, j+N)]
        temp = [lines[i].split() for i in range(j, j + N)]
        dfs.append(pd.DataFrame().from_records(temp))
        dfs[-1]['Timestep'] = (lines[j-1].split()[-1])
        dfs[-1]['type']   = species
        dfs[-1]['Id'] = np.arange(1, len(dfs[-1].index)+1)
        j += N+8

    df = pd.concat(dfs).rename(columns={0: 'x', 1: 'y', 2: 'z'})

    if write_csv:
        df.to_csv(output)

    return df


def ensemble_average(df, function, values, types):
    """
    For doing ensemble averaging. Using some function, it will get the ensemble average for all time, and all particle
    types that you specify

    Args:
        df: (Dataframe) Input data, needs to have the standard form with 'Id', 'type', and 'Timestep'
        function: (function) the function to whch we want to pass the ensemble data. I.e. an averaging
                    function, an autocorrelation function, a spectral density function, etc.
        values: The data that you want to evaluate with your function.
                Example: ['x', 'y', 'z']
        types: [array] Which particle types you want to evaluate.
                Example: ['H', 'O']
    :return:
    """
    ids = {}
    for i_type in types:
        ids[i_type] = df[(df.Timestep == 1) & (df.type == i_type)]['Id'].values

    results = []
    for key, value in ids.items():
        for id in value:
            data = df[df['Id'] == id][values].values
            results.append(function(data))
    return np.mean(results, axis=1)


def read_data(path, fname):
    with open(os.path.join(path, fname), "r") as fo:
        dipole = np.loadtxt(fo, dtype=np.float64, usecols=(1, 2, 3))
    return dipole


def calc_derivative(v, delta_t, thr=0.1):
    dy = np.zeros(np.shape(v))
    for i in range(3):
        dy[:, i] = np.gradient(v[:, i], edge_order=2)
    dy = dy[~(np.absolute(dy) > thr).any(1), :]
    return np.divide(dy, delta_t)


def zero_padding(sample_data):
    '''
      A series of Zeros will be padded to the end of the dipole moment array
    (before FFT performed), in order to obtain a array with the length which
    is the "next power of two" of numbers.
    #### Next power of two is calculated as: 2**np.ceil(log2(x))
    #### or Nfft = 2**int(math.log(len(data_array)*2-1, 2))
    '''
    N = 2 ** int(math.log(len(sample_data) * 2 - 1, 2))
    return N


def autocorrelation(v, normalize=True):
    if normalize:
        yunbiased = v - np.mean(v, axis=0)
        ynorm = np.sum(np.power(yunbiased, 2), axis=0)
    else:
        ynorm = np.ones(np.shape(v))
    autocor = np.zeros(np.shape(v))

    autocor = signal.fftconvolve(v, v[::-1], mode='full')[len(v) - 1:] / ynorm

    return autocor