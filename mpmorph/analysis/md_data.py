import numpy as np
import re
import os
from astropy.stats import bootstrap
from statsmodels.tsa.stattools import acf as autocorr

def get_MD_data(outcar_path, search_keys=None, search_data_column=None):
    '''
    Extracts the pressure, kinetic energy and total energy data from
    VASP MD OUTCAR.

    Args:
          outcar_path:
          search_keys:
          search_keys:
          search_data_column:
        - outcar_path = path to OUTCAR to be parsed
    Returns:
        - A nested list of MD steps where each search key value is
          listed.
    '''
    # Initial map of keywords to search for and data to map out from that line in OUTCAR
    # search_keys = ['external', 'kinetic energy EKIN', 'ETOTAL']
    # index of stripped column of data in that line, starts from 0
    # search_data_column = [3, 4, 4]
    if search_data_column is None:
        search_data_column = [3, 4, 4, 4]
    if search_keys is None:
        search_keys = ['external', 'kinetic energy EKIN', '% ion-electron', 'ETOTAL']
    outcar = open(outcar_path)
    print("OUTCAR opened")
    data_list = []
    md_step = 0
    for line in outcar:
        line = line.rstrip()
        for key_index in range(len(search_keys)):
            if re.search(search_keys[key_index],line):
                if key_index == 0:
                    data_list.append([float(line.split()[search_data_column[key_index]])])
                else:
                    try:
                        data_list[md_step].append(float(line.split()[search_data_column[key_index]]))
                    except IndexError:
                        break
                if key_index == len(search_keys)-1:
                    md_step +=1
    print("Requested information parsed.")
    outcar.close()

    data = {}
    for i, k in enumerate(search_keys):
        data[k] = [d[i] for d in data_list]
    return data


def autocorrelation(v, *args):
    """
    T function calculates the autocorrelation for an input vector v. The
    only difference is that this function accounts for v being composed of vectors, e.g: v[0] = [v_x, v_y, v_z]

    Args:
        v: [array] Input vector, can be a 1D array of scalars, or an array of vectors.
        mode: (int) What mode to run the calculation (see __autocorrelation__()). 1 is for the FFT method and
                2 is for the discrete (long) calculation. Default=1
        norm: (Bool) Whether or not to normalize the result. Default=False
        detrend: (Bool) Whether or not to detrend the data, v = v - v_average. Default=False

    Returns:
        [Array] The autocorrelation data.
    """
    if isinstance(v[0], (list, tuple, np.ndarray)):
        transposed_v = list(zip(*v))
        acf = []
        for i in transposed_v:

            acf.append(autocorr(i, args))

        return np.mean(acf, axis=0)

    else:
        return autocorr(v, args)


def get_MD_stats(data_list):
    """
    Args: data_list is the list of MD data returned by get_MD_data
    Returns: means and standard deviations
    """
    data_list = np.array(data_list)
    stats = []
    for col in range(data_list.shape[1]):
        data_col = data_list[:,col]
        stats.append( ( np.mean(data_col), np.std(data_col) ) )
    return stats

def plot_md_data(data_list):
    '''
    Args:
        data_list:

    Returns:
        matplotlib plt object

    '''


def parse_pressure(path, averaging_fraction=0.5):
    os.system("grep external " + path + "/OUTCAR | awk '{print $4}' > "+path +"/pres")
    os.system("grep volume/ion " + path + "/OUTCAR | awk '{print $5}' > "+path +"/vol")
    if os.path.isfile(path+"/OUTCAR"):
        with open(path+"/pres") as f:
            p = [float(line.rstrip()) for line in f]
        with open(path+"/vol") as f:
            vol = [float(line.rstrip()) for line in f][0]
        pressure = np.array(p)
        avg_pres = np.mean( pressure[int(averaging_fraction*(len(pressure)-1)):] )
    else:
        raise ValueError("No OUTCAR found.")
    return avg_pres, vol, pressure


def correlation_time():
    acf = autocorrelation(np.divide(self.data['E'], (self.L ** 2)), nlags=1000)
    t = np.arange(0, len(acf), 1)

    for i in range(len(acf)):
        if acf[i] < 0:
            intercept = i
            break
    try:
        return np.trapz(acf[0:intercept])
    except:
        print("MD didn't run long enough")
        return np.trapz(acf)


def n_trials(self):
    tau = self.correlation_time()
    n_steps = len(self.data['E'])
    if n_steps < 2 * tau:
        return n_steps
    return int(n_steps / (2 * tau))


def uncertainty(self, x):
    nt = self.n_trials()
    x = np.array(x)
    result = bootstrap(x, samples=nt)
    if isprop:
        s = np.std([np.std(i) for i in result])
        return s
    return np.mean([np.std(i) for i in result])
