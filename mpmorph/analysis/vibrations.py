from __future__ import division, unicode_literals

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from pymatgen.core.periodic_table import Element
from pymatgen.io.vasp.outputs import Vasprun
from astropy.stats import bootstrap
import os

__author__ = "Nicholas Winner"
__copyright__ = "None"
__version__ = "0.2"
__maintainer__ = "Nicholas Winner"
__email__ = "nwinner@berkeley.edu"
__status__ = "Development"
__date__ = ""

from mpmorph.analysis.utils import autocorrelation, power_spectrum, calc_derivative, xdatcar_to_df

boltzmann = 8.6173303e-5       # eV/K
boltzmann_SI = 1.38064852e-23
plank     = 4.135667e-15       # eV/s
plank_ps  = plank / 1e12       # eV/ps
hbar      = plank/(2*np.pi)    # eV/s
hbar_ps   = plank_ps/(2*np.pi) # eV/ps

c     = 2.9979245899e10  # speed of light in vacuum in [cm/s], from Wikipedia.
kB    = 0.6950347  # Boltzman constant in [cm^-1/K], from Wikipedia.
h_bar = 6.283185  # Reduced Planck constant in atomic unit, where h = 2*pi

class Vibrations(object):

    def __init__(self, data):
        self.data = data
        self.vdos_spectrum  = None
        self.ir_spectrum    = None
        self.raman_spectrum = None

        self.D_p = None
        self.DACF = None

    def calc_vdos_spectrum(self, filename=None):

        df = pd.read_csv("/Users/nwinner/bin/venv/xdatcar.csv")

        def vel(x):
            return np.gradient(x, axis=0)

        v_li = vel(df[df['Id'] == 1][['x', 'y', 'z']])
        v_f = vel(df[df['Id'] == 2][['x', 'y', 'z']])
        v_be = vel(df[df['Id'] == 3][['x', 'y', 'z']])

        fig, axs = plt.subplots(2)

        axs[0].plot(autocorrelation(v_li), label='Li')
        axs[0].plot(autocorrelation(v_f), label='F')
        axs[0].plot(autocorrelation(v_be), label='Be')

        print(autocorrelation(v_li))

        axs[1].plot(power_spectrum(autocorrelation(v_li))[0:200], label='Li')
        axs[1].plot(power_spectrum(autocorrelation(v_f))[0:200], label='F')
        axs[1].plot(power_spectrum(autocorrelation(v_be))[0:200], label='Be')
        plt.legend()
        plt.show()

    def calc_ir_spectrum(self, delta_t=1e-15):
        """
        Calculate the IR spectrum from the dipole autocorrelation function.

            P(freq) = m * freq**2 * FFT( ACF(x) )
            P(freq) = m * FFT( ACF(derivative(x)) )

        :return:
        """

        data = self.data[['dipole_x', 'dipole_y', 'dipole_z']]

        D_p  = calc_derivative(data.to_numpy(), delta_t=delta_t)  # derivative of dipole
        DACF = autocorrelation(D_p)                    # Dipole derivative autocorrelation function
        yfft = power_spectrum(DACF)                    # power spectrum of the dipole derivative ACF

        wavenumber = np.fft.fftfreq(len(yfft), delta_t * c)[0:int(len(yfft) / 2)]  # Wavenumber in units of cm^-1
        prefactor = 1
        intensity = prefactor*np.sum(yfft, axis=1)[0:int(len(yfft) / 2)]
        intensity = intensity/np.max(intensity)

        # TODO: Create pre-factor

        self.D_p = D_p
        self.DACF = DACF
        self.ir_spectrum = pd.DataFrame.from_dict({'wavenumber': wavenumber, 'intensity': intensity})

    def calc_raman_spectrum(self):
        return

    def calc_spectra(self):

        self.calc_power_spectrum()
        self.calc_ir_spectrum()
        self.calc_raman_spectrum()

    def plot_ir(self):

        D_p = self.D_p
        DACF = self.DACF
        wavenumber = self.ir_spectrum['wavenumber']
        intensity = self.ir_spectrum['intensity']

        fig, axs = plt.subplots(2)

        L2 = np.arange(len(DACF))
        axs[0].plot(L2, DACF[:, 0], color='red', linewidth=1.5)
        axs[0].plot(L2, DACF[:, 1], color='green', linewidth=1.5)
        axs[0].plot(L2, DACF[:, 2], color='blue', linewidth=1.5)
        axs[0].axis([0, len(DACF), 1.1 * np.min(DACF), 1.1 * np.max(DACF)], fontsize=15)
        axs[0].set_xlabel("Data Points", fontsize=15)
        axs[0].set_ylabel("DACF (a.u.)", fontsize=15)

        axs[1].plot(wavenumber, intensity, color='black', linewidth=1.5)
        axs[1].axis([0, 4000,
                  -1.1 * np.min(intensity), 1.1 * np.max(intensity)],
                 fontsize=15)
        axs[1].set_xlabel("Wavenumber (cm$^{-1}$)", fontsize=15)
        axs[1].set_ylabel("Intensity (a.u.)", fontsize=15)
        plt.show()

    def to_csv(self):
        return

class VDOS(object):

    def __init__(self, filename):
        if isinstance(filename, str):
            if '.csv' in filename:
                self.positions = pd.read_csv(filename)
        else:
            self.positions = xdatcar_to_df(filename)

        self.vel  = {}
        self.acfs = {}
        self.vdos = {}
        self.freq = {}

    def calc_vdos_spectrum(self, time_step=1):

        df = self.positions

        def vel(x):
            return np.divide(np.gradient(x, axis=0), time_step)

        ids = {}
        types = df.type.unique()
        for i_type in types:
            ids[i_type] = df[(df.Timestep == 1) & (df.type == i_type)]['Id'].values

        for key, value in ids.items():
            results = []
            for id in value:
                data = df[df['Id'] == id][['x', 'y', 'z']]
                results.append(np.mean(vel(data), axis=1))
            self.vel[key] = results

        for key, value in self.vel.items():
            acf = []
            for v in value:
                acf.append(autocorrelation(v))
            mean_acf = np.mean(acf, axis=0)
            self.acfs[key] = mean_acf

        for key, value in self.acfs.items():
            spectrum  = power_spectrum(value)*Element(key).atomic_mass
            intensity = spectrum / np.max(spectrum)
            self.freq[key] = np.fft.fftfreq(len(spectrum), time_step)[0:int(len(spectrum) /2)]  # freq
            self.vdos[key] = intensity[0:int(len(intensity)/2)]

    def plot_vdos(self, show=True, save=False):
        fig, axs = plt.subplots(2)

        for k,v in self.acfs.items():
            axs[0].plot(v, label=k)
        axs[0].set_xlabel('Steps', size=22)
        axs[0].set_ylabel('Velocity Autocorrelation Function, VACF', size=14)

        for k,v in self.vdos.items():
            p = axs[1].plot(self.freq[k], v, label=k)
            axs[1].fill(self.freq[k], v, alpha=0.3, color=p[-1].get_color())
        axs[1].set_xlabel('Frequency ($fs^{-1}$)', size=22)
        axs[1].set_xscale('log')
        axs[1].set_ylabel('Velocity Density of States (a.u.)', size=14)

        plt.legend()
        if save:
            plt.savefig('vdos.png', fmt='png')
        if show:
            plt.show()
        return plt

class Viscosity(object):

    def __init__(self, vasprun):
        v = Vasprun(vasprun)

        self.volume   = v.structures[0].volume
        self.stresses = []
        for step in v.ionic_steps:
            self.stresses.append(step['stress'])
        self.time_step = v.parameters['POTIM']
        self.temp = v.parameters['TEBEG']
        self.nsteps = v.nionic_steps
        self.acf = []
        self.shear_stresses = []

        self.acfs = []
        self.norm_acfs = []

        self.acf = None
        self.tao = None
        self.ntrials = None

        print(v.structures[0].composition.fractional_composition)

    def calc_viscosity(self, formula_units):
        for i in range(3):
            for j in range(3):
                self.shear_stresses.append([s[i][j] * 100000000 for s in self.stresses])
                self.acfs.append(autocorrelation(self.shear_stresses[-1], normalize=False))
                self.norm_acfs.append(autocorrelation(self.shear_stresses[-1], normalize=True))

        self.acf = np.mean(self.acfs)
        self.tao = np.trapz( np.mean(self.norm_acfs, axis=0), np.arange(0, self.nsteps, self.time_step))
        self.ntrials = int(self.nsteps / (2 * self.tao))

        boots = bootstrap(self.acf, bootnum=self.ntrials)

        visc = []
        std  = []
        for boot in boots:
            visc.append(self.volume*(1e-30)*np.trapz(boot,(1e-15)*np.arange(0,self.nsteps,self.time_step))/(formula_units*self.temp*boltzmann_SI))

        return {'viscosity': np.mean(visc), 'StdDev': np.std(visc)}