from __future__ import division, unicode_literals

import os
import numpy as np
import pandas as pd
from scipy import stats
from matplotlib import pyplot as plt
from pymatgen.core.periodic_table import Element
from pymatgen.io.vasp.outputs import Vasprun, Xdatcar
from astropy.stats import bootstrap

from mpmorph.analysis.utils import autocorrelation, power_spectrum, calc_derivative, xdatcar_to_df


__author__ = "Nicholas Winner"
__copyright__ = "None"
__version__ = "0.2"
__maintainer__ = "Nicholas Winner"
__email__ = "nwinner@berkeley.edu"
__status__ = "Development"
__date__ = ""

boltzmann = 8.6173303e-5       # eV/K
boltzmann_SI = 1.38064852e-23  # J/K
plank     = 4.135667e-15       # eV/s
plank_ps  = plank / 1e12       # eV/ps
hbar      = plank/(2*np.pi)    # eV/s
hbar_ps   = plank_ps/(2*np.pi) # eV/ps

c     = 2.9979245899e10  # speed of light in vacuum in [cm/s], from Wikipedia.
kB    = 0.6950347  # Boltzman constant in [cm^-1/K], from Wikipedia.
h_bar = 6.283185  # Reduced Planck constant in atomic unit, where h = 2*pi


class Spectrum(object):

    def __init__(self, data):
        self.data = data
        self.ir_spectrum    = None
        self.raman_spectrum = None

        self.D_p = None
        self.DACF = None

    def calc_ir_spectrum(self, time_step=1):
        """
        Calculate the IR spectrum from the dipole autocorrelation function.

            P(freq) = m * freq**2 * FFT( ACF(x) )
            P(freq) = m * FFT( ACF(derivative(x)) )

        :return:
        """
        delta_t = time_step*(1e-15)

        def diff(x):
            return np.divide(np.gradient(x, axis=0), time_step)

        data = self.data[['dipole_x', 'dipole_y', 'dipole_z']]

        D_p  = np.mean( diff(data.values) , axis=1)  # derivative of dipole
        DACF = autocorrelation(D_p)                    # Dipole derivative autocorrelation function
        yfft = power_spectrum(DACF)                    # power spectrum of the dipole derivative ACF

        wavenumber = np.fft.fftfreq(len(yfft), delta_t * c)[0:int(len(yfft) / 2)]  # Wavenumber in units of cm^-1
        prefactor = 1
        intensity = prefactor*yfft[0:int(len(yfft) / 2)]
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
        axs[0].plot(L2, DACF, color='red', linewidth=1.5)
        axs[0].axis([0, len(DACF), 1.1 * np.min(DACF), 1.1 * np.max(DACF)], fontsize=15)
        axs[0].set_xlabel("Data Points", fontsize=15)
        axs[0].set_ylabel("Dipole Autocorrelation Function (a.u.)", fontsize=15)

        axs[1].plot(wavenumber, intensity, color='black', linewidth=1.5)
        axs[1].axis([0, 4000,
                  -1.1 * np.min(intensity), 1.1 * np.max(intensity)],
                 fontsize=15)
        axs[1].set_xlabel("Wavenumber (cm$^{-1}$)", fontsize=15)
        axs[1].set_ylabel("Intensity (a.u.)", fontsize=15)
        plt.show()

    def to_csv(self):
        return


class Diffusion(object):
    """
    Robust calculation of diffusion coefficients with different statistical analysis techniques:
        - Block averaging (default)
        - Jackknife (to be implemented)
        - Bootstrap (to be implemented)

    Args:
        structures: (list) list of Structures
        corr_t: (float) correlation time (in terms of # of steps).
            Each time origin will be this many steps apart.
        block_l: (int)  defines length of a block in terms of corr_t. (block_t = block_l * corr_t)
        t_step: (float) time-step in MD simulation. Defaults to 2.0 fs.
        l_lim: (int) this many time-steps are skipped in MSD while fitting D. I.e. approximate length of
            ballistic and cage regions. Defaults to 50.
        skip_first: (int) this many initial time-steps are skipped. Defaults to 0.
        ci: (float) confidence interval desired estimating the mean D of population.
    """

    def __init__(self, structures, sampling_method='bootstrap', block_l=50, corr_t=1, t_step=2.0, l_lim=50, skip_first=0, ci=0.95):
        self.structures = structures
        self.abc = self.structures[0].lattice.abc
        self.natoms = len(self.structures[0])
        self.sampling_method = sampling_method
        self.skip_first = skip_first
        self.total_t = len(self.structures)
        self.corr_t = corr_t
        self.block_l = block_l
        self.l_lim = l_lim
        self.t_step = t_step
        self.ci = ci
        self.msds = None
        self.scaling_factor = 0.1 / self.t_step  # conv. to cm2/s

    @property
    def n_origins(self):
        n = (self.total_t - self.block_t - self.skip_first) / self.corr_t + 1
        if n <= 0:
            raise ValueError("Too many blocks for the correlation time")
        return int(n)

    def n_trials(self, el):
        return int(len(self.structures)/(2*self.get_tao(el)))

    @property
    def block_t(self):
        return self.block_l * self.corr_t

    def _getd(self, el):
        md = [np.zeros(self.structures[0].frac_coords.shape)]
        for i in range(self.skip_first + 1, self.total_t):
            dx = self.structures[i].frac_coords - self.structures[i - 1].frac_coords
            dx -= np.round(dx)
            md.append(dx)

        self.md = np.array(md) * self.abc

        # remove other elements from the rest of the calculations
        s = set(self.structures[0].indices_from_symbol(el))
        self.md = np.delete(self.md, [x for x in list(range(self.natoms)) if x not in s], 1)

        msds = []

        block      = False
        boot_strap = True

        if self.sampling_method == 'block':
            for i in range(self.n_origins):
                su = np.square(np.cumsum(self.md[i * self.corr_t: i * self.corr_t + self.block_t], axis=0))
                msds.append(np.mean(su, axis=1))

        elif self.sampling_method == 'bootstrap':
            boots = bootstrap(self.md, bootnum=self.n_trials(el))
            self.block_l = len(boots[0])/self.corr_t
            for boot in boots:
                su = np.square(np.cumsum(boot, axis=0))
                msds.append(np.mean(su, axis=1))

        self.msds = msds

    def plot_block_msds(self):
        for i in self.msds:
            plt.plot(i)
        plt.show()

    def getD(self, el):
        """
        Method to calculate diffusion coefficient(s) of the given element (el).
        """
        self._getd(el)
        D = [[], [], []]
        for i in self.msds:
            for j in range(3):
                slope, intercept, r_value, p_value, std_err = \
                    stats.linregress(np.arange(self.l_lim, self.block_t), i[:, j][self.l_lim:])
                D[j].append(slope / 2.0)
        D = np.array(D) * self.scaling_factor
        self.D_blocks = D

        alpha = 1.0-self.ci
        tn = stats.t.ppf(1.0-alpha/2.0, len(self.D_blocks) - 1) / np.sqrt(len(self.D_blocks))

        if tn == "nan":
            tn = 1
        self.D_i = np.mean(D, axis=1)
        self.D_i_std = np.std(D, axis=1)*tn
        self.D_avg = np.sum(self.D_i) / 3.0
        self.D_avg_std = np.std(np.sum(D, axis=0) / 3.0)*tn
        return self.D_dict

    @property
    def D_dict(self):
        D_dict = {}
        dirs = ["Dx", "Dy", "Dz"]
        D_dict.update(dict(zip(dirs, self.D_i)))
        D_dict.update(dict(zip([s + "_std" for s in dirs], self.D_i_std)))
        D_dict.update({"D": self.D_avg, "D_std": self.D_avg_std})
        return D_dict

    def get_tao(self, el):
        md = [np.zeros(self.structures[0].frac_coords.shape)]
        for i in range(self.skip_first + 1, self.total_t):
            dx = self.structures[i].frac_coords - self.structures[i - 1].frac_coords
            dx -= np.round(dx)
            md.append(dx)
        self.md = np.array(md) * self.abc

        # remove other elements from the rest of the calculations
        s = set(self.structures[0].indices_from_symbol(el))
        self.md = np.delete(self.md, [x for x in list(range(self.natoms)) if x not in s], 1)

        mean_md = [np.mean(np.mean(x, axis=1), axis=0) for x in self.md]
        acf = autocorrelation(mean_md)
        tao = np.trapz(acf, np.arange(0,len(acf), self.t_step))
        self.corr_t = tao
        return tao

    @property
    def tao(self):
        tao_dict = {}
        for k, v in self.D_dict.items():
            if "_std" not in k:
                tao_dict[k] = 1.0 / v * self.scaling_factor
        return tao_dict

    def autocorrelation(self):
        df = self.positions

        def disp(x):
            return np.gradient(x, axis=0)

        ids = {}
        types = df.type.unique()
        for i_type in types:
            ids[i_type] = df[(df.Timestep == 1) & (df.type == i_type)]['Id'].values

        for key, value in ids.items():
            results = []
            for id in value:
                data = df[df['Id'] == id][['x', 'y', 'z']]
                results.append(np.mean(disp(data), axis=1))
            self.vel[key] = results

        for key, value in self.vel.items():
            acf = []
            for v in value:
                acf.append(autocorrelation(v))
            mean_acf = np.mean(acf, axis=0)
            self.acfs[key] = mean_acf

        for key, value in self.acfs.items():
            spectrum = power_spectrum(value)
            self.vdos[key] = spectrum[0:int(len(spectrum) / 2)][0:200]


class Activation(object):
    def __init__(self, D_t):
        self.D_t = D_t
        self.Q = None
        self.intercept = None
        self.Q_std = None

    def LS(self):
        self.x = np.array([1 / float(t[0]) for t in self.D_t])
        self.y = np.array([np.log(t[1]["D"]) for t in self.D_t])
        self.yerr = np.array([[-np.log((t[1]["D"] - t[1]["D_std"]) / t[1]["D"]),
                               np.log((t[1]["D"] + t[1]["D_std"]) / t[1]["D"])
                               ] for t in self.D_t])
        self.Q, self.intercept, self.r_value, self.p_value, self.std_err = \
            stats.linregress(self.x, self.y)
        self.Q *= -1
        return self.Q

    def ODR(self):
        if not self.Q:
            self.LS()
        import scipy.odr

        def fit_func(p, t):
            return p[0] * t + p[1]

        Model = scipy.odr.Model(fit_func)
        Data = scipy.odr.RealData(self.x, self.y, sy=np.mean(self.yerr, axis=1))
        Odr = scipy.odr.ODR(Data, Model, [-self.Q, self.intercept])
        Odr.set_job(fit_type=2)
        self.output = Odr.run()
        self.Q, self.intercept = -self.output.beta[0], self.output.beta[1]
        self.Q_std = self.output.sd_beta[0]
        self.intercept_std = self.output.sd_beta[1]
        return self.Q, self.Q_std

    def plot(self, title=None, annotate=True, el='', **kwargs):
        #fig = plt.figure()

        line = np.polyval([-self.Q, self.intercept], self.x)
        tx = str(int(np.rint(self.Q)))
        if self.Q_std:
            tx += "$\pm${}".format(str(int(np.rint(self.Q_std))))
        c = kwargs.get('color','')
        plt.plot(self.x * 1000, line, c+'-', )
        plt.errorbar(self.x * 1000, self.y, yerr=self.yerr.T, label="Q[{}]: ".format(el) + tx + " K", **kwargs)
        plt.ylabel("ln(D cm$^2$/s)", fontsize=15)
        plt.xlabel("1000/T K$^{-1}$", fontsize=15)

        if annotate:
            plt.annotate("Q: " + tx + " K", xy=(0.98, 0.95), xycoords='axes fraction', fontsize=14,
                         horizontalalignment='right', verticalalignment='top')
        if title:
            plt.title = title
        #return fig

    @classmethod
    def from_run_paths(cls, p, T, el, corr_t, block_l, t_step=2.0, l_lim=50, skip_first=0):
        D_t = []
        for t in range(len(p)):
            xdatcar = Xdatcar(p[t])
            d = Diffusion(xdatcar.structures, corr_t=corr_t, block_l=block_l,
                          t_step=t_step, l_lim=l_lim, skip_first=skip_first)
            D_t.append([T[t], d.getD(el)])
        return cls(D_t)


class VDOS(object):

    def __init__(self, input):
        if isinstance(input, Xdatcar):
            self.positions = xdatcar_to_df(input)
        elif os.path.isfile(input):
            self.positions = xdatcar_to_df(input)
        elif os.path.isdir(input):
            if os.path.isfile(os.path.join(input, 'XDATCAR.gz')):
                self.positions = xdatcar_to_df(os.path.join(input, 'XDATCAR.gz'))
            elif os.path.isfile(os.path.join(input, 'XDATCAR')):
                self.positions = xdatcar_to_df(os.path.join(input, 'XDATCAR'))
            else:
                raise FileNotFoundError("Could not located XDATCAR in the provided file.")
        else:
            raise TypeError("The provided input is either not and XDATCAR object ",
                            "or is not a path to an XDATCAR file")

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

        if os.path.isfile(vasprun):
            v = Vasprun(vasprun)
        elif os.path.isdir(vasprun):
            if os.path.isfile(os.path.join(vasprun, 'vasprun.xml.gz')):
                v = Vasprun(os.path.join(vasprun, 'vasprun.xml.gz'))
            elif os.path.isfile(os.path.join(vasprun, 'vasprun.xml')):
                v = Vasprun(os.path.join(vasprun, 'vasprun.xml'))
        elif isinstance(vasprun, Vasprun):
            v = vasprun
        else:
            raise TypeError("The argument provided is neither a Vasprun object",
                            "nor a path that leads to a Vasprun file.")

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

        self.formula_units = v.structures[0].composition.get_reduced_composition_and_factor()[1]

    def calc_viscosity(self):
        for i in range(3):
            for j in range(3):
                if i != j:
                    self.shear_stresses.append([s[i][j] * 100000000 for s in self.stresses])
                    self.acfs.append(autocorrelation(self.shear_stresses[-1], normalize=False))
                    self.norm_acfs.append(autocorrelation(self.shear_stresses[-1], normalize=True))

        self.acf = np.mean(self.acfs, axis=0)

        visc = []
        for acf in self.acfs:
            visc.append(self.volume*(1e-30)*np.trapz(self.acf,(1e-15)*self.time_step*np.arange(0,self.nsteps))/(self.formula_units*self.temp*boltzmann_SI))

        return {'viscosity': np.mean(visc), 'StdDev': np.std(visc)}
