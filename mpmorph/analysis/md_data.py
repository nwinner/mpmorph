import numpy as np
import re
import os
from astropy.stats import bootstrap
from mpmorph.analysis.utils import autocorrelation
from pymatgen.io.vasp.outputs import Oszicar, Vasprun
from matplotlib import pyplot as plt

boltzman = 8.6173303e-5
eV_to_J = 1.60218e-19

class MD_Data:

    def __init__(self):

        self.nsteps = 0
        self.time = []

        self.md_data = {'pressure': [], 'etot': [], 'ekin': [], 'temp': []}
        self.md_acfs = {'pressure': [], 'etot': [], 'ekin': [], 'temp': []}
        self.md_stats = {}
        self.composition = None

    def parse_md_data(self, input):
        """
        Parses the md run data from a directory.

        Args:
            input: directory for the md run containing a vasprun.xml and OSZICAR file
        Returns:
             None
        """

        if os.path.isfile(os.path.join(input, 'vasprun.xml.gz')):
            v = Vasprun(os.path.join(input, 'vasprun.xml.gz'))
        elif os.path.isfile(os.path.join(input, 'vasprun.xml')):
            v = Vasprun(os.path.join(input, 'vasprun.xml'))
        else:
            raise FileNotFoundError

        if os.path.isfile(os.path.join(input, 'OSZICAR.gz')):
            o = Oszicar(os.path.join(input, 'OSZICAR.gz'))
        elif os.path.isfile(os.path.join(input, 'OSZICAR')):
            o = Oszicar(os.path.join(input, 'OSZICAR'))
        else:
            raise FileNotFoundError

        self.composition = v.structures[0].composition
        print(self.composition.weight)
        nsteps = v.nionic_steps
        self.nsteps += nsteps
        if self.time:
            starttime = self.time[-1]
            self.time.extend(np.add(np.arange(0, nsteps) * v.parameters['POTIM'], starttime))
        else:
            self.time.extend(np.arange(0, nsteps) * v.parameters['POTIM'])

        pressure = []
        etot = []
        ekin = []
        temp = []
        for i, step in enumerate(o.ionic_steps):
            temp.append(step['T'])
        for i, step in enumerate(v.ionic_steps):
            ekin.append(step['kinetic'])
            etot.append(step['total'])
            stress = step['stress']
            kinP = (2 / 3) * ekin[i]
            pressure.append((1 / 3) * np.trace(stress) + kinP)

        self.md_data['pressure'].extend(pressure)
        self.md_data['etot'].extend(etot)
        self.md_data['ekin'].extend(ekin)
        self.md_data['temp'].extend(temp)

        self.md_acfs['pressure'] = autocorrelation(self.md_data['pressure'], normalize=True)
        self.md_acfs['pressure'] = autocorrelation(self.md_data['pressure'], normalize=True)
        self.md_acfs['pressure'] = autocorrelation(self.md_data['pressure'], normalize=True)
        self.md_acfs['pressure'] = autocorrelation(self.md_data['pressure'], normalize=True)

    @property
    def get_md_acfs(self):
        return self.md_acfs

    @property
    def get_temp(self):
        return self.temp

    def get_md_data(self):
        return self.md_data

    def get_md_stats(self):
        """
        Return the basic statistics about the run: the average and standard deviation of the run data.
        Because an individual run can be very brief, we ignore the first 50 steps to help eliminate the
        cage/collision regime while leaving enough data to get good statistics. This is an estimate for
        automated calculations, so use it with caution.

        Args:
            None

        Returns:
            (dict) A dictionary of the means and standard deviations for the run stats.
        """

        stats = {}
        for k, v in self.md_data.items():
            stats[k] = {'Mean': np.mean(v[int(len(v)/2):]), 'StdDev': np.std(v[int(len(v)/2):])}
        return stats

    def plot_md_data(self, show=True, save=False):
        """
        Plots the md run data versus time.

        Args:
            show: (bool) Whether or not to show the figure
            save: (bool) Whether or not to save the figure in png format

        Returns:
            matplotlib plt object

        """

        for k, v in self.md_data.items():
            fig, axs = plt.subplots()

            axs.plot(self.time, v)
            axs.set_ylabel('Simulation {}'.format(k), size=18)

            axs.minorticks_on()
            axs.tick_params(which='major', length=8, width=1, direction='in', top=True, right=True, labelsize=18)
            axs.tick_params(which='minor', length=2, width=.5, direction='in', top=True, right=True)

            if show:
                plt.show()
            if save:
                plt.savefig('{}.{}'.format(k, 'png'), fmt='png')