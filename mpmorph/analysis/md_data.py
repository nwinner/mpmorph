import numpy as np
import re
import os
from astropy.stats import bootstrap
from mpmorph.analysis.utils import autocorrelation
from pymatgen.io.vasp.outputs import Oszicar, Vasprun
from matplotlib import pyplot as plt


def get_MD_data(dir):
    if os.path.isfile(os.path.join(dir, 'vasprun.xml.gz')):
        v = Vasprun(os.path.join(dir, 'vasprun.xml.gz'))
    elif os.path.isfile(os.path.join(dir, 'vasprun.xml')):
        v = Vasprun(os.path.join(dir, 'vasprun.xml'))
    else:
        raise FileNotFoundError

    if os.path.isfile(os.path.join(dir, 'OSZICAR.gz')):
        o = Oszicar(os.path.join(dir, 'OSZICAR.gz'))
    elif os.path.isfile(os.path.join(dir, 'OSZICAR')):
        o = Oszicar(os.path.join(dir, 'OSZICAR'))
    else:
        raise FileNotFoundError

    pressure = []
    etot     = []
    temp     = []
    ekin     = []
    time     = np.arange(0, v.nionic_steps)*v.parameters['POTIM']
    for step in o.ionic_steps:
        ekin.append(step['EK'])
        etot.append(step['E0'])
        temp.append(step['T'])
    for i, step in enumerate(v.ionic_steps):
        stress   = step['stress']
        kinP     = (2/3)*ekin[i]
        pressure.append((1/3)*np.trace(stress)+kinP)

    return {'pressure': {'val': pressure, 'acf':  autocorrelation(pressure)},
            'etot': {'val': etot, 'acf': autocorrelation(etot)},
            'ekin': {'val': ekin, 'acf': autocorrelation(ekin)},
            'temp': {'val': temp, 'acf': autocorrelation(temp)},
            'time': time
            }


def get_MD_stats(data):
    """
    Args: data_list is the list of MD data returned by get_MD_data
    Returns: means and standard deviations
    """
    stats = {}
    time = data['time']
    for k, v in data.items():
        if k != 'time':
            stats[k] = {'Mean': np.mean(v['val']), 'StdDev': np.std(v['val']),
                        'Relaxation time': np.trapz(v['acf'], time)}
    return stats


def plot_md_data(data, show=True, save=False):
    '''
    Args:
        data_list:

    Returns:
        matplotlib plt object

    '''

    time = data['time']
    for k, v in data.items():
        if k != 'time':
            fig, axs = plt.subplots(2)

            axs[0].plot(time, v['val'])
            axs[0].set_ylabel('Simulation {}'.format(k), size=18)

            axs[1].plot(time, v['acf'])
            axs[1].set_ylabel('Autocorrelation Function', size=18)
            axs[1].set_xlabel('Time (fs)', size=18)

            axs[0].minorticks_on()
            axs[0].tick_params(which='major', length=8, width=1, direction='in', top=True, right=True, labelsize=18)
            axs[0].tick_params(which='minor', length=2, width=.5, direction='in', top=True, right=True)
            axs[1].minorticks_on()
            axs[1].tick_params(which='major', length=8, width=1, direction='in', top=True, right=True, labelsize=18)
            axs[1].tick_params(which='minor', length=2, width=.5, direction='in', top=True, right=True)

            if show:
                plt.show()
            if save:
                plt.savefig('{}.{}'.format(k, 'png'), fmt='png')

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
