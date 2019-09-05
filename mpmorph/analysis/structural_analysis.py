from monty.json import MSONable

import numpy as np
import itertools
from multiprocessing import Pool
from scipy.spatial import Voronoi, Delaunay
from copy import deepcopy
from pymatgen.analysis import structure_analyzer
#from pymatgen.util.coord_utils import get_angle
from pymatgen.io.vasp.outputs import Xdatcar
from pymatgen.core.structure import Structure
from matplotlib import pyplot as plt
from scipy.signal import find_peaks, savgol_filter, find_peaks_cwt
from mpmorph.analysis.utils import autocorrelation
from pymatgen.analysis.structure_analyzer import Voronoi, VoronoiAnalyzer, VoronoiConnectivity, VoronoiNN


def polyhedra_connectivity(structures, pair, cutoff, step_freq=1):
    """
    Args:
        structures:
        pair:
        cutoff:
        step_freq:
        given: Given polyhedra are of this

    Returns:

    """
    n_frames = len(structures)
    center_atom = pair[0]
    shell_atom = pair[1]

    connectivity = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7:0, 8:0}
    connectivity_template = deepcopy(connectivity)
    connectivity_sub_categories={}
    connections = {}

    for s_index in itertools.count(0, step_freq):
        if s_index >= n_frames:
            break
        structure = structures[s_index]

        polyhedra_list=[]

        for i in range(len(structure)):
            current_poly=[]
            if str(structure[i].specie) == center_atom:
                for j in range(len(structure)):
                    if str(structure[j].specie) == shell_atom:
                        d = structure.get_distance(i, j)
                        if d < cutoff:
                            current_poly.append(j)
                polyhedra_list.append(set(current_poly))

        for polypair in itertools.combinations(polyhedra_list,2):

            polyhedra_pair_type = (len(polypair[0]), len(polypair[1]))

            shared_vertices = len(polypair[0].intersection(polypair[1]))
            network = polypair[0].union(polypair[1])

            if shared_vertices in connectivity:
                connectivity[shared_vertices] += 1

            if shared_vertices:
                if polyhedra_pair_type in connectivity_sub_categories:
                    if shared_vertices in connectivity_sub_categories[polyhedra_pair_type]:
                        connectivity_sub_categories[polyhedra_pair_type][shared_vertices] += 1
                elif polyhedra_pair_type[::-1] in connectivity_sub_categories:
                    if shared_vertices in connectivity_sub_categories[polyhedra_pair_type[::-1]]:
                        connectivity_sub_categories[polyhedra_pair_type[::-1]][shared_vertices] += 1
                else:
                    connectivity_sub_categories[polyhedra_pair_type]=deepcopy(connectivity_template)
                    if shared_vertices in connectivity_sub_categories[polyhedra_pair_type]:
                        connectivity_sub_categories[polyhedra_pair_type][shared_vertices] = 1

        """
        connectivity_length = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0}
        pairs = []
        intersections = []
        not_intersections = []
        for i in range(len(polyhedra_list)-1):
            for j in range(1, len(polyhedra_list)):
                intersect = polyhedra_list[i].intersection(polyhedra_list[j])

                if len(intersect) > 0:
                    pairs.append([polyhedra_list[i], polyhedra_list[j]])
                    intersections.append(intersect)
                    not_intersections.append(polyhedra_list[i].symmetric_difference(polyhedra_list[j]))

        for i in range(len(intersections)-1):
            count = 2
            for j in range(1, len(not_intersections)):
                intersect = intersections[i].intersection(not_intersections[j])

                if len(intersect) > 0:
                    count += 1

        def is_connected():
        """
    return connectivity, connectivity_sub_categories


def coordination_number_distribution(structures, pair, cutoff, step_freq=1):
    """
    Calculates coordination number distribution
    Args:
        structures:
        pair:
        cutoff:
        step_freq:

    Returns:

    """

    cn_dist=[]
    n_frames = len(structures)
    for s_index in itertools.count(0, step_freq):
        if s_index >= n_frames:
            break
        structure = structures[s_index]
        for i in range(len(structure)):
            if str(structure[i].specie) == pair[0]:
                cn = 0
                for j in range(len(structure)):
                    if str(structure[j].specie) == pair[1]:
                        d = structure.get_distance(i,j)
                        if d < cutoff:
                            cn+=1
                cn_dist.append(cn)
    return cn_dist


class BondAngleDistribution(object):
    """
    Bond Angle Distribution
    Args:
        structures (list): list of structures
        cutoffs (dict): a dictionary of cutoffs where keys are tuples of pairs ('A','B')
        step_freq: calculate every this many steps
    Attributes:
        bond_angle_distribution (dict)
    """

    def __init__(self, structures, cutoffs, step_freq=1, bin_size=5, triplets=None):

        self.bond_angle_distribution = {}

        self.structures = structures
        self.step_freq = step_freq
        self.bin_size = bin_size

        self.theta = np.arange(0, 180, bin_size)

        if triplets:
            self.triplets = triplets
        else:
            self.triplets = self.get_unique_triplets(structures[0])

        if isinstance(cutoffs,dict):
            self.cutoffs = cutoffs
            self._cutoff_type = 'dict'
            self.max_cutoff = max(cutoffs.values())
        elif isinstance(cutoffs,float) or isinstance(cutoffs, int):
            self.cutoffs = cutoffs
            self.max_cutoff = cutoffs
            self._cutoff_type = 'constant'
        else:
            raise ValueError("Cutoffs must be specified as dict of pairs or globally as a single float.")

    @property
    def n_bins(self):
        _bins = int(np.ceil(180 / self.bin_size))
        if _bins < 2:
            raise ValueError("More bins required!")
        return _bins

    @property
    def n_frames(self):
        return len(self.structures)

    @staticmethod
    def _unit_vector(v):
        return v / np.linalg.norm(v)

    @staticmethod
    def _angle_between(v1, v2, degrees=True):
        v1_u = BondAngleDistribution._unit_vector(v1)
        v2_u = BondAngleDistribution._unit_vector(v2)

        if degrees:
            coeff = (180/np.pi)
        else:
            coeff = 1
        return coeff*np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))

    @staticmethod
    def _get_angle(a, b, c):

        """
        Gets the angle between two vectors defined by a set of coordinates a,b,c
        a is the central atom coordinates, b and c are the adjacent atom coordinates.
        :param a: (array) Cartesian coordinates of origin point (i.e. central atom)
        :param b: (array) Cartesian coordinates of the point defining the first vector AB (i.e. first non-central atom)
        :param c: (array) Cartesian coordinates of the point defining the next vector AB (i.e. next non-central atom)
        :return: (float) angle between the two vectors defined by a,b,c
        """

        ab = np.subtract(b, a)
        ac = np.subtract(c, a)

        return BondAngleDistribution._angle_between(ab, ac)

    def get_angle(self, s_index, i, j, k):
        """
        Returns **Minimum Image** angle specified by three sites.

        Args:
            s_index: Structure index in structures list
            i (int): Index of first site.
            j (int): Index of second site.
            k (int): Index of third site.

        Returns:
            (float) Angle in degrees.
        """
        s = self.structures[s_index]
        return BondAngleDistribution._get_angle(s[i].coords, s[j].coords, s[k].coords)

    @staticmethod
    def get_unique_triplets(s):
        central_atoms = s.symbol_set
        possible_end_members = []
        for i in itertools.combinations_with_replacement(central_atoms, 2):
            possible_end_members.append(i)
        unique_triplets = []
        for i in central_atoms:
            for j in possible_end_members:
                triplet = (j[0],i,j[1])
                unique_triplets.append(triplet)
        return unique_triplets

    def _check_skip_triplet(self,s_index,i,n1,n2):
        """
        Helper method to find if a triplet should be skipped
        Args:
            s_index: index of structure in self.structures
            i: index of the central site
            n1: index of the first neighbor site
            n2: index of the second neighbor site
        Returns:
            True if pair distance is longer than specified in self.cutoffs
        """
        ns = [n1,n2]
        s = self.structures[s_index]
        skip_triplet = False
        for j in ns:
            pair = (s[i].species_string, s[j].species_string)
            if pair not in self.cutoffs:
                pair = pair[::-1]
            if s.get_distance(i,j)>self.cutoffs[pair]:
                skip_triplet = True
                break
        return skip_triplet

    def plot_bond_angle_distribution(self):
        if not self.bond_angle_distribution:
            self.get_bond_angle_distribution()

        triplets = self.bond_angle_distribution.keys()

        fig, ax1 = plt.subplots()

        ax1.minorticks_on()
        ax1.set_ylabel("Frequency (fractional)", size=24)
        ax1.set_xlabel("Angle, $\\theta$ (degrees)", size=24)
        ax1.tick_params(which='major', length=8, width=1, direction='in', top=True, right=True, labelsize=14)
        ax1.tick_params(which='minor', length=2, width=.5, direction='in', top=True, right=True, labelsize=14)
        ax1.set_xlim(left=-5, right=185)
        #ax1.set_ylim(bottom=-.1, top=1.1)

        for triplet in triplets:
            ax1.plot(self.theta, self.bond_angle_distribution[triplet], 'o-', label='-'.join(triplet))

        plt.legend(loc=0, frameon=False)
        plt.show()
        return plt

    def get_binary_angle_dist_plot(self, title=None):
        import matplotlib.pyplot as plt
        fig = plt.figure(figsize=(12, 6))
        c = 0
        maxes = []
        for triplet in self.bond_angle_distribution:
            p = self.bond_angle_distribution[triplet]
            c += 1
            ax = fig.add_subplot(2, 3, c)
            ax.plot(range(len(p)), p)
            maxes.append(max(p))
            ax.annotate('-'.join(triplet), (0.75, 0.88), xycoords='axes fraction', size=16)
            ax.set_yticklabels([])
            ax.xaxis.set_ticks(np.arange(0, 181, 30))
            plt.gca().set_ylim([0, 0.1])
            if c in [1, 2, 3]:
                ax.set_xticklabels([])
            else:
                plt.xlabel('Angle (degrees)', fontsize=16)
            if c in [1, 4]:
                plt.ylabel('Intensity (a.u.)', fontsize=16)
            ax.tick_params(axis='both', which='major', labelsize=16)
            if c == 2:
                if title:
                    plt.title(title)
        for ax in fig.axes:
            ax.set_ylim(0.0, max(maxes))
        fig.subplots_adjust(top=0.75)
        fig.tight_layout()
        return fig

    def get_bond_angle_distribution(self, nproc=4, smooth=0):

        bond_angle_dict = {}

        frames = [(self.structures[i * self.step_freq], self.triplets, self.n_bins, self.cutoffs,
                   self.max_cutoff, self._cutoff_type, self.bin_size)
                  for i in range(int(np.floor(self.n_frames / self.step_freq)))]

        pool = Pool(nproc)
        results = pool.map(BondAngleDistribution._process_frame, frames)
        pool.close()
        pool.join()

        # Collect all BADs
        for triplet in self.triplets:
            bond_angle_dict[triplet] = np.zeros(self.n_bins)
        for i in results:
            for triplet in self.triplets:
                bond_angle_dict[triplet] += i[triplet]

        below_threshold_i = []
        for triplet in bond_angle_dict:
            total = np.sum(bond_angle_dict[triplet])
            if total != 0.0:
                bond_angle_dict[triplet] /= total
            if all(bond_angle_dict[triplet] == 0.0):
                below_threshold_i.append(triplet)
        for i in below_threshold_i:
            del bond_angle_dict[i]

        self.bond_angle_distribution = bond_angle_dict

        if smooth > 0:
            self.smooth(passes=smooth)

        return self.bond_angle_distribution

    @staticmethod
    def _process_frame(data):
        """
        Helper function for parallel rdf computation
        """
        coord_frame, triplets, n_bins, cutoffs, max_cutoff, cutoff_type, bin_size \
            = data[0], data[1], data[2], data[3], data[4], data[5], data[6]

        process_BADs = {}

        for triplet in triplets:
            process_BADs[triplet] = np.zeros(n_bins)

        for site in coord_frame.sites:

            pts = coord_frame.get_neighbors(site, r=max_cutoff, include_image=True)

            central_atom_specie = site.species_string

            for pair in itertools.combinations(pts, 2):
                atom2_specie = pair[0][0].species_string
                atom3_specie = pair[1][0].species_string

                if cutoff_type is 'dict':
                    x = cutoffs.get((central_atom_specie, atom2_specie), None)
                    if x < pair[0][1] or x is None:
                        continue

                    x = cutoffs.get((atom2_specie, central_atom_specie), None)
                    if x < pair[0][1] or x is None:
                        continue

                    x = cutoffs.get((atom3_specie, central_atom_specie), None)
                    if x < pair[1][1] or x is None:
                        continue

                    x = cutoffs((central_atom_specie, atom3_specie), None)
                    if x < pair[1][1] or x is None:
                        continue

                    if (central_atom_specie, atom2_specie) or (atom2_specie, central_atom_specie) not in cutoffs.keys():
                        continue
                    if (central_atom_specie, atom3_specie) or (atom3_specie, central_atom_specie) not in cutoffs.keys():
                        continue

                angle = BondAngleDistribution._get_angle(site.coords, pair[0][0].coords, pair[1][0].coords)

                if angle > 180:
                    print("ANGLE IS MORE THAN 180, FIXING")
                    angle = 360 - angle

                bin_index = int(angle / bin_size)

                key = (atom2_specie, central_atom_specie, atom3_specie)
                if key in process_BADs:
                    process_BADs[key][bin_index] += 1
                if key[::-1] in process_BADs:
                    process_BADs[key[::-1]][bin_index] += 1
        return process_BADs

    def smooth(self, passes=1):
        if passes == 0:
            return
        else:
            for triplet in self.bond_angle_distribution:
                self.bond_angle_distribution[triplet] = \
                    savgol_filter(deepcopy(self.bond_angle_distribution[triplet]), window_length=5, polyorder=3)
            passes -= 1
            return self.smooth(passes=passes)


class VoronoiTetrahedron:

    def __init__(self, vertex, sites):

        self.vertex = vertex
        self.sites = sites

    def __eq__(self, other):
        if len(set(self.sites).intersection(other.sites)) == 0:
            return True
        return False

    def __hash__(self):
        return int(list(self.vertex)[0])

    @property
    def coords(self):
        return [s.coords for s in self.sites]

    @property
    def volume(self):
        c = self.coords
        return abs(np.dot((np.subtract(c[0], c[3])), np.cross((np.subtract(c[1], c[3])), (np.subtract(c[2], c[3])))))/6


class VoronoiAnalysis(object):

    """
    NOTE: This class has also been migrated to pymatgen so will be removed!
    """

    def __init__(self, structures, cutoff=5, bin_size=0.01, step_freq=2, smooth=1,
                 title="Connectivty Motif Radial Distribution Functions"):
        self.vor_ensemble = None
        self.structures = structures
        self.cutoff = cutoff
        self.bin_size = bin_size
        self.step_freq = step_freq
        self.smooth = smooth
        self.title = title

        self.r = np.arange(0, cutoff, bin_size)
        self.n_frames = len(structures)
        self.n_species = structures[0].composition.as_dict()
        ss = structures[0].symbol_set
        self.pairs = [p for p in itertools.product(ss, repeat=2)]

        if self.step_freq > self.n_frames:
            raise ValueError("ERROR: TOO FEW STRUCTURES FOR GIVEN STEP FREQUENCY.")

        self.RDFs = {}

    @property
    def n_bins(self):
        _bins = int(np.ceil(self.cutoff / self.bin_size))
        if _bins < 2:
            raise ValueError("More bins required!")
        return _bins

    @staticmethod
    def voronoi_analysis(structure, n=0, cutoff=5.0, qhull_options="Qbb Qc Qz"):
        """
        Performs voronoi analysis and returns the polyhedra around atom n
        in Schlaefli notation.

        Note: Part of this function is obtained from the pymatgen VoronoiCoordinationFinder
              and should be merged with that function in future.

        Args:
            - structure: pymatgen Structure object
            - n: index of the center atom in structure
            - cutoff: cutoff distance around n to search for neighbors
        Returns:
            - voronoi index of n: <x3,x4,x5,x6,x6,x7,x8,x9,x10>
              where x_i denotes number of facets with i edges
        """

        center = structure[n]
        neighbors = structure.get_sites_in_sphere(center.coords, cutoff)
        neighbors = [i[0] for i in sorted(neighbors, key=lambda s: s[1])]
        qvoronoi_input = np.array([s.coords for s in neighbors])
        voro = Voronoi(qvoronoi_input, qhull_options=qhull_options)
        vor_index = np.array([0,0,0,0,0,0,0,0])

        for key in voro.ridge_dict:
            if 0 in key:
                "This means if the center atom is in key"
                if -1 in key:
                    "This means if an infinity point is in key"
                    print("Cutoff too short. Exiting.")
                    return None
                else:
                    try:
                        vor_index[len(voro.ridge_dict[key])-3]+=1
                    except IndexError:
                        # If a facet has more than 10 edges, it's skipped here
                        pass
        return vor_index

    def from_structures(self, structures, cutoff=4.0, step_freq=10, qhull_options="Qbb Qc Qz"):
        """
        A constructor to perform Voronoi analysis on a list of pymatgen structrue objects

        Args:
            structures (list): list of Structures
            cutoff (float: cutoff distance around an atom to search for neighbors
            step_freq (int): perform analysis every step_freq steps
            qhull_options (str): options to pass to qhull
        Returns:
            A list of [voronoi_tesellation, count]
        """
        print("This might take a while...")
        voro_dict = {}
        step = 0
        for structure in structures:
            step+=1
            if step%step_freq != 0:
                continue

            v = []
            for n in range(len(structure)):
                v.append(str(self.voronoi_analysis(structure,n=n,cutoff=cutoff,
                                                   qhull_options=qhull_options).view()))
            for voro in v:
                if voro in voro_dict:
                    voro_dict[voro]+=1
                else:
                    voro_dict[voro]=1
        self.vor_ensemble = sorted(voro_dict.items(), key=lambda x: (x[1],x[0]), reverse=True)[:15 ]
        return self.vor_ensemble

    @property
    def plot_vor_analysis(self):
        import matplotlib.pyplot as plt
        t = zip(*self.vor_ensemble)
        labels = t[0]
        val = list(t[1])
        tot = np.sum(val)
        val = [float(j)/tot for j in val]
        pos = np.arange(len(val))+.5    # the bar centers on the y axis
        plt.figure(figsize=(4,4))
        plt.barh(pos, val, align='center', alpha=0.5)
        plt.yticks(pos, labels)
        plt.xlabel('Fraction')
        plt.title('Voronoi Spectra')
        plt.grid(True)
        return plt

    @staticmethod
    def voronoi_polyhedra_to_voronoi_tetrahedra(structure, n, **kwargs):

        central_site = structure[n]
        i_verts = []
        voronoi_nn = VoronoiNN(**kwargs)
        voronoi = voronoi_nn.get_voronoi_polyhedra(structure, n)

        for k, v in voronoi.items():
            i_verts.append([k, set(v['verts'])])

        tetras = []
        for i in itertools.combinations(i_verts, 3):
            intersect = i[0][1].intersection(i[1][1], i[2][1])
            if len(intersect) > 0:
                sites = [voronoi[j[0]]['site'] for j in i]
                sites.append(central_site)
                return [VoronoiTetrahedron(intersect, sites)]
                tetras.append(VoronoiTetrahedron(intersect, sites))
        return tetras

    @staticmethod
    def all_voronoi_polyhedra_to_tetrahedra(structure):
        tetrahedra = []
        for i in range(len(structure)):
            tetrahedra.extend(VoronoiAnalysis.voronoi_polyhedra_to_voronoi_tetrahedra(structure, n=i))
        tetra = {tetra for tetra in tetrahedra}
        return list(tetra)

    @staticmethod
    def get_connection_motifs(data):

        structure, pairs, n_bins, cutoff, bin_size = \
            data[0], data[1], data[2], data[3], data[4]
        _process_RDFs = {}

        for pair in pairs:
            _process_RDFs[pair] = np.zeros(n_bins)

        process_RDFs = {1: deepcopy(_process_RDFs),
                        2: deepcopy(_process_RDFs),
                        3: deepcopy(_process_RDFs),
                        4: deepcopy(_process_RDFs)}

        bins = {1: 0, 2: 0, 3: 0, 4: 0}

        tetras = VoronoiAnalysis.all_voronoi_polyhedra_to_tetrahedra(structure)

        for t in tetras:
            print(t.vertex)

        for i in range(len(tetras) - 1):
            for j in range(i + 1, len(tetras)):
                intersect = set(tetras[i].sites).intersection(set(tetras[j].sites))
                motif = len(intersect)
                if motif > 0:
                    connected = set(tetras[i].sites).symmetric_difference(set(tetras[j].sites))
                    left_connected = set(tetras[i].sites).intersection(connected)
                    right_connected = set(tetras[j].sites).intersection(connected)
                    bins[motif] += 1
                    for k in itertools.product(left_connected, right_connected):
                        dist = k[0].distance(k[1])

                        if dist > cutoff:
                            continue

                        atom1_specie = k[0].species_string
                        atom2_specie = k[1].species_string

                        bin_index = int(dist / bin_size)

                        if bin_index == 0:
                            print("SELF")
                            continue

                        key = (atom1_specie, atom2_specie)
                        if key in process_RDFs[motif].keys():
                            process_RDFs[motif][key][bin_index] += 1
                        if key[::-1] in process_RDFs:
                            process_RDFs[motif][key[::-1]][bin_index] += 1

        return process_RDFs

    def get_connective_rdfs(self, n_proc=1):

        RDFs = {1: {}, 2: {}, 3: {}, 4: {}}

        nproc = 4

        frames = [(self.structures[i * self.step_freq], self.pairs, self.n_bins, self.cutoff, self.bin_size) for i in
                  range(int(np.floor(self.n_frames / self.step_freq)))]
        counter = len(frames)
        pool = Pool(nproc)
        results = pool.map(VoronoiAnalysis.get_connection_motifs, frames)
        pool.close()
        pool.join()

        # Collect all rdfs
        for k in RDFs.keys():
            for pair in self.pairs:
                RDFs[k][pair] = np.zeros(self.n_bins)
        for i in results:
            for k in RDFs.keys():
                for pair in self.pairs:
                    RDFs[k][pair] += i[k][pair]

        get_pair_order = []

        for motif in RDFs.keys():
            for i in RDFs[motif].keys():
                get_pair_order.append('-'.join(list(i)))
                density_of_atom2 = self.n_species[i[1]] / self.structures[0].volume
                for j in range(self.n_bins):
                    r = j * self.bin_size
                    if r == 0:
                        continue
                    shell_volume = (4.0 / 3 * np.pi * (r ** 3 - (r - self.bin_size) ** 3))
                    RDFs[motif][i][j] = RDFs[motif][i][j] / self.n_species[i[0]] / density_of_atom2 / shell_volume / counter

                for j in range(self.n_bins):
                    if RDFs[motif][i][j] < 0:
                        RDFs[motif][i][j] = 0

        for motif in RDFs.keys():
            empty_i = []
            for i in RDFs[motif].keys():
                if all(v == 0.0 for v in RDFs[motif][i]):
                    empty_i.append(i)
            for i in empty_i:
                del RDFs[motif][i]

        self.RDFs = RDFs

        return RDFs

    def plot_connective_rdfs(self, pairs_to_include):
        """
        :return: a plot of RDFs
        """
        fig, ax1 = plt.subplots()

        ax1.minorticks_on()
        ax1.set_ylabel("RDF, g(r)", size=24)
        ax1.set_xlabel("Radial Distance, r ($Å^{3}$)", size=24)
        ax1.tick_params(which='major', length=8, width=1, direction='in', top=True, right=True, labelsize=14)
        ax1.tick_params(which='minor', length=2, width=.5, direction='in', top=True, right=True, labelsize=14)
        ax1.set_xlim(left=-.1, right=self.cutoff * 1.05)
        ax1.set_ylim(bottom=-.1, top=12)

        for motif in self.RDFs.keys():
            for key, rdf in self.RDFs[motif].items():
                if key in pairs_to_include:
                    p = ax1.plot(self.r, rdf, label="{}-atom: {}-{}".format(motif, key[0], key[1]))

        plt.legend(bbox_to_anchor=(0.975, 0.975),
                   borderaxespad=0., prop={'family': 'sans-serif', 'size': 13}, frameon=False)
        plt.tight_layout()
        plt.show()
        return plt


class RadialDistributionFunction(MSONable):
    """
    Class to calculate partial radial distribution functions (RDFs) of sites.
    Typically used to analyze the pair correlations in liquids or amorphous structures.
    Supports multiprocessing: see get_radial_distribution_function method

    Args:
        structures (list): a list of Structure objects.
        cutoff (float): maximum distance to search for pairs. (defauly = 5.0)
            Note cutoff should be smaller than or equal to the half of the edge
            length of the box due to periodic boundaires.
        bin_size (float): thickness of each coordination shell in Angstroms (default = 0.1)
        step_freq (int): compute and store RDFs every step_freq steps
            to average later. (default = 2)
        smooth (int): number of smoothing passes (default = 1)
        title (str): title for the RDF plot.
    Returns:
        A dictionary of partial radial distribution functions with pairs as keys and RDFs as values.
        RDFs themselves are arrays of length cutoff/bin_size.
    """

    def __init__(self, structures, cutoff=5.0, bin_size=0.01, step_freq=2,
                 title="Radial distribution functions", rdf_dict={}):
        if structures:
            self.structures = structures
            self.cutoff = cutoff
            self.bin_size = bin_size
            self.step_freq = step_freq

            self.n_frames = len(self.structures)
            self.n_atoms = len(self.structures[0])
            self.n_species = self.structures[0].composition.as_dict()
            self.get_pair_order = None
            self.title = title
            self.RDFs = {}

            self.N = {}  # running coordination number
            self.CN = {}  # first-shell coordination number
            self.peak_locations = {}
            self.first_shell = {}
            ss = self.structures[0].symbol_set
            self.pairs = [p for p in itertools.combinations_with_replacement(ss, 2)]
            self.pairs = [p for p in itertools.product(ss, repeat=2)]
            self.counter = 1
            self.r = np.arange(0, cutoff, bin_size)
        elif rdf_dict:
            self.r = rdf_dict['r']
            self.bin_size = rdf_dict['bin_size']
            self.step_freq = rdf_dict['step_freq']
            self.cutoff = rdf_dict['cutoff']
            self.RDFs = rdf_dict['rdf']
            self.title = rdf_dict['r']
            self.CN = rdf_dict['first_shell_coordination_numbers']
            self.first_shell = rdf_dict['first_shell_coordination_radius']
            del self.RDFs['r']

    @property
    def n_bins(self):
        _bins = int(np.ceil(self.cutoff / self.bin_size))
        if _bins < 2:
            raise ValueError("More bins required!")
        return _bins

    @property
    def r_range(self):
        return self.r

    def get_radial_distribution_functions(self, smooth=0, nproc=1):
        """
        Args:
            nproc: (int) number of processors to utilize (defaults to 1)
            smooth: (int) Number of passes of a Savitzky-Golay filter to smooth the RDF data
        Returns:
            A dictionary of partial radial distribution functions
            with pairs as keys and RDFs as values.
            Each RDF arrays of length cutoff/bin_size.
        """

        frames = [(self.structures[i * self.step_freq], self.pairs, self.n_bins, self.cutoff, self.bin_size) for i in
                  range(int(np.floor(self.n_frames / self.step_freq)))]
        self.counter = len(frames)
        pool = Pool(nproc)
        results = pool.map(RadialDistributionFunction._process_frame, frames)
        pool.close()
        pool.join()

        # Collect all rdfs
        for pair in self.pairs:
            self.RDFs[pair] = np.zeros(self.n_bins)
        for i in results:
            for pair in self.pairs:
                self.RDFs[pair] += i[pair]

        self.get_pair_order = []

        for i in self.RDFs.keys():
            self.get_pair_order.append('-'.join(list(i)))
            density_of_atom2 = self.n_species[i[1]] / self.structures[0].volume
            for j in range(self.n_bins):
                r = j * self.bin_size
                if r == 0:
                    continue
                shell_volume = (4.0/3 * np.pi * (r**3 - (r-self.bin_size)**3))
                self.RDFs[i][j] = self.RDFs[i][j] / self.n_species[i[0]] / shell_volume / density_of_atom2 / self.counter
                #self.RDFs[i][j] = self.RDFs[i][j] / self.n_species[
                #    i[0]] / shell_volume / density_of_atom2 / self.counter / 2
                #self.RDFs[i][j] = self.RDFs[i][j] / self.n_species[i[0]] / (4.0 / np.pi / r / r) / self.bin_size / density_of_atom2 / self.counter

            for j in range(self.n_bins):
                if self.RDFs[i][j] < 0:
                    self.RDFs[i][j] = 0

        empty_i=[]
        for i in self.RDFs.keys():
            if all(v == 0.0 for v in self.RDFs[i]):
                empty_i.append(i)
        for i in empty_i:
            del self.RDFs[i]

        if smooth > 0:
            self.smooth(passes=smooth)

        return self.RDFs

    def plot_radial_distribution_functions(self, pairs_to_include=None, show=True, save=False, include_CN=True):
        """
        :return: a plot of RDFs
        """

        if pairs_to_include is None:
            pairs_to_include = self.RDFs.keys()

        fig, ax1 = plt.subplots()

        ax1.minorticks_on()
        ax1.set_ylabel("RDF, g(r)", size=24)
        ax1.set_xlabel("Radial Distance, r ($Å^{3}$)", size=24)
        ax1.tick_params(which='major', length=8, width=1, direction='in', top=True, right=True, labelsize=14)
        ax1.tick_params(which='minor', length=2, width=.5, direction='in', top=True, right=True, labelsize=14)
        ax1.set_xlim(left=-.1, right=self.cutoff*1.05)
        ax1.set_ylim(bottom=-.1, top=12)

        for key, rdf in self.RDFs.items():
            if key in pairs_to_include:
                p = ax1.plot(self.r, rdf, label=key)

                if include_CN:
                    if key in self.first_shell.keys():
                        ax1.plot(self.r, self.N[key], '--', color=p[-1].get_color())
                if self.peak_locations:
                    if key in self.peak_locations.keys():
                        ax1.plot(self.r[self.peak_locations[key]], rdf[self.peak_locations[key]], 'x')
                if self.first_shell:
                    if key in self.first_shell.keys():
                        ax1.plot(self.r[self.first_shell[key]], rdf[self.first_shell[key]], 'x')

        plt.legend(bbox_to_anchor=(0.975, 0.975),
                   borderaxespad=0., prop={'family': 'sans-serif', 'size': 13}, frameon=False)
        plt.title(self.title)
        plt.tight_layout()
        if save:
            plt.savefig('rdf.png', fmt='png', dpi=500)
        if show:
            plt.show()
        return plt

    def get_rdf_db_dict(self):
        db_rdf = self.RDFs.copy()
        for key, value in self.RDFs.items():
            db_rdf[key[0]+'-'+key[1]] = list(self.RDFs[key])
            del db_rdf[key]
        db_rdf.update({'r': list(np.arange(0, self.cutoff, self.bin_size))})
        return db_rdf

    def get_coordination_numbers(self, composition=None, volume=None):
        if not self.N:
            print("GETTING RUNNING COORDINATION NUMBERS")
            self.get_running_coordination_number()
        if not self.first_shell:
            print("GETTING RADIUS OF FIRST COORDINATION SHELL")
            self.get_first_coordination_shell_radius()
        print("GETTING FIRST COORDINATION SHELL NUMBER")
        self.get_first_coordination_shell()

        return self.CN

    def get_running_coordination_number(self):
        number_densities = {}
        N = {}

        volume = self.structures[0].volume
        composition = self.structures[0].composition.element_composition

        for k, v in composition.items():
            number_densities[k.symbol] = v / volume

        for k, v in self.RDFs.items():

            N[k] = []

            for i in range(len(self.r)):
                N[k].append(
                    4 * np.pi * number_densities[k[1]] * np.trapz(((self.r[0:i]) ** 2) * v[0:i], self.r[0:i]))

        self.N = N
        return N

    def _get_coordination_number(self, rdfs):
        number_densities = {}
        N = {}

        volume = self.structures[0].volume
        composition = self.structures[0].composition.element_composition

        for k, v in composition.items():
            number_densities[k.symbol] = v / volume

        for k, v in rdfs.items():

            N[k] = []

            for i in range(len(self.r)):
                N[k].append(
                    4 * np.pi * number_densities[k[1]] * np.trapz(((self.r[0:i]) ** 2) * v[0:i], self.r[0:i]))
        return N

    def get_first_coordination_shell(self):
        CN = {}
        if not self.first_shell:
            self.get_first_coordination_shell_radius()
        for k, v in self.RDFs.items():
            try:
                CN[k] = self.N[k][self.first_shell[k]]
            except:
                print("No first coordination shell for {}".format(k), "... ignoring it.")
        self.CN = CN
        return CN

    def get_peaks(self):

        peak_locations = {}

        for key, value in self.RDFs.items():

            peaks, properties = find_peaks(value, prominence=1)

            if len(peaks) > 0:
                peak_locations[key] = peaks
            else:
                peaks, properties = find_peaks(value, height=.5)
                peak_locations[key] = sorted(peaks)

        self.peak_locations = peak_locations

        return self.peak_locations

    def get_first_coordination_shell_radius(self):

        if not self.peak_locations:
            self.get_peaks()
        if not self.N:
            self.get_running_coordination_number()

        for key, value in self.peak_locations.items():
            min = self.RDFs[key][value[0]]
            for i in range(value[0], len(self.r)):
                if self.RDFs[key][i] < min:
                    min = self.RDFs[key][i]
                    if np.mean(self.RDFs[key][i:i+5]) > min:
                        self.first_shell[key] = i
                        break
                    elif min < .3:
                        self.first_shell[key] = i
                        break

                elif np.average(self.RDFs[key][i:i+10]) > min:
                    self.first_shell[key] = i
                    break

        return self.first_shell

    @property
    def first_coordination_shell_radius(self):
        r = {}
        for k,v in self.first_shell.items():
            r[k] = self.r[v]
        return r

    # TODO This is extremely inefficient. It calls RDF function repeatedly to get the corr func, must be a better way

    def get_cage_correlation_function(self, pair, nproc=4, c=1):
        """

        :return:
        """

        cage_function = []
        #if isinstance(pair[0], str):
        #    pair = [pair]
        for t in range(1, self.n_frames, self.step_freq):
            try:
                frames = [(self.structures[i], [pair], self.n_bins, self.r[self.first_shell[pair]], self.bin_size) for i in
                          range(t-1, t)]
            except:
                print("PROBLEM")
                continue
            self.counter = len(frames)
            pool = Pool(nproc)
            results = pool.map(RadialDistributionFunction._process_frame, frames)
            pool.close()
            pool.join()

            temp_RDF = {}
            temp_RDF[pair] = np.zeros(self.n_bins)

            for i in results:
                temp_RDF[pair] += i[pair]

            self.get_pair_order = []

            density_of_atom2 = self.n_species[pair[1]] / self.structures[0].volume
            for j in range(self.n_bins):
                r = j * self.bin_size
                if r == 0:
                    continue
                shell_volume = (4.0 / 3 * np.pi * (r ** 3 - (r - self.bin_size) ** 3))
                temp_RDF[pair][j] = temp_RDF[pair][j] / self.n_species[
                    pair[0]] / shell_volume / density_of_atom2 / self.counter
            for j in range(self.n_bins):
                if temp_RDF[pair][j] < 0:
                    temp_RDF[pair][j] = 0

            x = self._get_coordination_number(temp_RDF)[pair][self.first_shell[pair]]
            if x < self.CN[pair] - c + 1:
                cage_function.append(0)
            else:
                cage_function.append(1)

        for i in cage_function:
            if np.isnan(i):
                i = 0

        fig, axs = plt.subplots(2)

        axs[0].minorticks_on()
        axs[0].tick_params(which='major', length=8, width=1, direction='in', top=True, right=True, labelsize=14)
        axs[0].tick_params(which='minor', length=2, width=.5, direction='in', top=True, right=True, labelsize=14)

        axs[0].plot(cage_function)

        axs[0].minorticks_on()
        axs[0].tick_params(which='major', length=8, width=1, direction='in', top=True, right=True, labelsize=14)
        axs[0].tick_params(which='minor', length=2, width=.5, direction='in', top=True, right=True, labelsize=14)

        acf = autocorrelation(cage_function, normalize=True)
        acf = np.add(acf, 1)
        axs[1].plot(acf)
        from scipy.stats import linregress
        t = np.arange(0, self.step_freq*len(acf), self.step_freq)
        slope, intercept, r_value, p_value, std_err = linregress(t[len(t)//2:], np.log(acf)[len(t)//2:])
        print(-1/slope)
        plt.show()

    def as_dict(self):
        d = dict()
        d["@module"] = self.__class__.__module__
        d["@class"] = self.__class__.__name__
        d["rdf"] = self.RDFs
        d['r'] = self.r
        d['cutoff'] = self.cutoff
        d['bin_size'] = self.bin_size
        d['step_freq'] = self.step_freq
        d["first_shell_coordination_numbers"] = self.CN
        d["first_shell_coordination_radius"] = self.first_shell
        d['title'] = self.title
        return d

    @classmethod
    def from_dict(cls, rdf_dict):
        return cls(structures=None, rdf_dict=rdf_dict)

    @staticmethod
    def _process_frame(data):
        """
        Helper function for parallel rdf computation
        """
        coord_frame, pairs, n_bins, cutoff, bin_size = \
            data[0], data[1], data[2], data[3], data[4]
        process_RDFs = {}

        for pair in pairs:
            process_RDFs[pair] = np.zeros(n_bins)

        for site in coord_frame.sites:
            pts = coord_frame.get_neighbors(site, r=cutoff, include_image=True)
            atom1_specie = site.species_string
            for pt in pts:
                atom2_specie = pt[0].species_string
                bin_index = int(pt[1] / bin_size)
                key = (atom1_specie, atom2_specie)
                if key in process_RDFs:
                    process_RDFs[key][bin_index] += 1

        return process_RDFs

    def smooth(self, passes=1):
        if passes == 0:
            return
        else:
            for pair in self.RDFs.keys():
                self.RDFs[pair] = \
                    savgol_filter(deepcopy(self.RDFs[pair]), window_length=5, polyorder=3)
            passes -= 1
            return self.smooth(passes=passes)


class PotentialOfMeanForce(MSONable):

    def __init__(self, r, RDFs, temp):

        self.r    = r
        self.RDFs = RDFs
        self.temp = temp
        self.PMFs  = {}

        #remove zeros from the RDFs, as these will give infinite energy
        self._clean()

    def _clean(self):
        new_RDFs = {}
        for key, value in self.RDFs.items():
            idx = []
            for i in range(len(value)):
                if value[i] <= 0:
                    idx.append(i)
            new_r = []
            new_g = []
            for i in range(len(self.r)):
                if i in idx:
                    continue
                new_r.append(self.r[i])
                new_g.append(self.RDFs[key][i])
            new_RDFs[key] = (new_r, new_g)

    def get_potential_of_mean_force(self):
        for key, value in self.RDFs.items():
            self.PMFs[key] = -8.617333e-5 * self.temp * np.log(value)

    def plot(self, show=True, save=False, pairs_to_include=None):

        if pairs_to_include is None:
            pairs_to_include = self.PMFs.keys()

        fig, ax1 = plt.subplots()

        ax1.minorticks_on()
        ax1.set_ylabel("PMF, W(r)", size=24)
        ax1.set_xlabel("Radial Distance, r ($Å^{3}$)", size=24)
        ax1.tick_params(which='major', length=8, width=1, direction='in', top=True, right=True, labelsize=14)
        ax1.tick_params(which='minor', length=2, width=.5, direction='in', top=True, right=True, labelsize=14)
        ax1.set_xlim(left=-.1 * self.r[0], right=self.r[-1] * 1.05)
        ax1.set_ylim(bottom=-.1, top=12)

        for key, rdf in self.PMFs.items():
            if key in pairs_to_include:
                p = ax1.plot(self.r, rdf, label=key)

        plt.legend(bbox_to_anchor=(0.975, 0.975),
                   borderaxespad=0., prop={'family': 'sans-serif', 'size': 13}, frameon=False)
        plt.title("Potential of Mean Force")
        plt.tight_layout()

        if show:
            plt.show()
        if save:
            plt.savefig('pmf.png', fmt='png', dpi=500)

    @classmethod
    def from_dict(cls, d):
        return PotentialOfMeanForce.__init__(d['r'], d['RDFs'], d['temp'])

    def as_dict(self):
        return {'RDFs': self.RDFs, 'r': self.r, 'temp': self.temp, 'PMFs': self.PMFs}


class NetworkSizeDistribution:

    def __init__(self, structures, coordination_cutoffs, centers=None, connectors=None, step_freq=10):
        """

        :param structures: list of structures
        :param coordination_cutoffs: (dict) containing the first coordination shell radius for all the atoms
                                     you wish to analyze

        """

        self.structures = structures
        self.step_freq = step_freq
        self.coordination_cutoffs = coordination_cutoffs
        if centers is None:
            self.centers = self.structures[0].symbol_set
        else:
            self.centers = centers
        if connectors is None:
            self.connectors = self.structures[0].symbol_set

    def get_network_size_distribution(self, nproc=4):

        frames = [(self.structures[i * self.step_freq], self.coordination_cutoffs, self.centers, self.connectors) for i in
                  range(int(np.floor(self.n_frames / self.step_freq)))]
        self.counter = len(frames)
        pool = Pool(nproc)
        results = pool.map(NetworkSizeDistribution._process_frame, frames)
        pool.close()
        pool.join()


    @staticmethod
    def _process_frame(data):

        coord_frame, coordination_cutoffs, centers, connectors = data[0], data[1], data[2], data[3]

        networks = []

        for site in coord_frame.sites:
            if any(site.species_string != c for c in centers):
                continue

            pts = coord_frame.get_sites_in_sphere(site.coords,
                                                  coordination_cutoffs[site.species_string],
                                                  include_image=True)

            sites = []
            for pt in pts:
                if any(pt[0].species_string == c for c in connectors):
                    sites.append(pt[0])
                elif any(pt[0].species_string == c for c in centers):
                    sites.append(pt[0])

            networks.append(sites)

        num_complexes = len(networks)
        bins = {1: num_complexes}

        # For the first step, establish the connections by finding intersection of complexes, removing
        # unecessary sites to speed up the calculation after
        new_networks = []
        for i in itertools.combinations(networks, 2):
            intersect = set(i[0]).intersection(set(i[1]))
            if len(intersect) > 0:
                new_networks.append(set(i[0]).union(set(i[1])))

        bins[2] = len(new_networks)

        size = 3

        while size < num_complexes:
            new_networks = []
            for i in itertools.combinations(networks, 2):
                intersect = set(i[0]).intersection(set(i[1]))
                if len(intersect) > 0:
                    new_networks.append(set(i[0]).union(set(i[1])))

            bins[size] = len(new_networks)
            bins[size-1] = len(networks) - 2*bins[size]
            networks = new_networks
            size += 1
            print(bins)

        return bins


class Network:

    def __init__(self, sites):
        self.sites = set(sites)

    def __eq__(self, other):
        if len(self.sites.intersection(other.sites)) == 0:
            return True
        return False

    def compare(self, other):
        intersect = self.sites.intersection(other.sites)
        return len(intersect)


class CageCorrelationFunction:

    def __init__(self, structures, first_coordination_shell_radii, step_freq):
        self.structures = structures
        self.radii = first_coordination_shell_radii
        self.step_freq = step_freq
        self.n_frames = len(structures)

    def get_cage_correlation_function(self, pair, c=1, nproc=1):
        """
        Args:
            nproc: (int) number of processors to utilize (defaults to 1)
            smooth: (int) Number of passes of a Savitzky-Golay filter to smooth the RDF data
        Returns:
            A dictionary of partial radial distribution functions
            with pairs as keys and RDFs as values.
            Each RDF arrays of length cutoff/bin_size.
        """

        frames = [(self.structures[i * self.step_freq], pair, self.radii[pair]) for i in
                  range(int(np.floor(self.n_frames / self.step_freq)))]

        num_centers = int(self.structures[0].composition.as_dict()[pair[0]])
        pool = Pool(nproc)
        results = pool.map(CageCorrelationFunction._process_frame, frames)
        pool.close()
        pool.join()
        N = len(frames)

        # TODO: This is where correlation needs to be calculated

        autocorr_nout = np.zeros(len(results))
        autocorr_nin  = np.zeros(len(results))
        N = len(frames)
        for tau in range(1, N-1):
            for i in range(1, N-2*tau, tau):
                data = [results[i-1], results[i -1 + tau], num_centers, c]
                nout, nin = CageCorrelationFunction._compare_neighbor_lists(data)
                autocorr_nout[tau] += nout
            autocorr_nout[tau] = autocorr_nout[tau]/(N-tau)

        autocorr_nout /= np.max(autocorr_nout)

        fig, axs = plt.subplots()

        axs.minorticks_on()
        axs.tick_params(which='major', length=8, width=1, direction='in', top=True, right=True, labelsize=14)
        axs.tick_params(which='minor', length=2, width=.5, direction='in', top=True, right=True, labelsize=14)

        axs.plot(np.arange(0, N)*self.step_freq*.001, autocorr_nout)
        plt.show()

    @staticmethod
    def _process_frame(data):
        coord_frame, pair, cutoff = \
            data[0], data[1], data[2]

        n_atoms = len(coord_frame)

        distance_matrix = coord_frame.distance_matrix
        neighbor_lists = [np.zeros(n_atoms) for i in range(n_atoms)]

        for atom1 in range(n_atoms - 1):
            atom1_specie = coord_frame[atom1].species_string
            for atom2 in range(atom1 + 1, n_atoms):
                atom2_specie = coord_frame[atom2].species_string
                if pair != (atom1_specie, atom2_specie):
                    continue
                if distance_matrix[atom1, atom2] > cutoff:
                    continue
                if distance_matrix[atom1, atom2] == 0:
                    continue
                neighbor_lists[atom1][atom2] = 1

        cleaned = []
        for i in range(len(neighbor_lists)):
            if all([v == 0 for v in neighbor_lists[i]]):
                continue
            cleaned.append(neighbor_lists[i])

        return cleaned

    @staticmethod
    def _compare_neighbor_lists(data):
        neighbor_list1, neighbor_list2, num_centers, c = \
            data[0], data[1], data[2], data[3]

        diff = np.subtract(neighbor_list2, neighbor_list1)
        CN_1 = np.sum(neighbor_list1)
        CN_2 = np.sum(neighbor_list2)

        # for the diff between the neighbor lists of each atom
        n_out = 0
        for atom in diff:
            net = np.sum(atom)
            n_out += np.heaviside(c + net, 0)
        print(n_out)
        return (num_centers-n_out)/num_centers, 0


def compute_mean_coord(structures, freq=100):
    '''
    NOTE: This function will be removed as it has been migrated
    to pymatgen.
    Calculate average coordination numbers
    With voronoi polyhedra construction
    args:
        - structures: list of Structures
        - freq: sampling frequency of coord number [every freq steps]
    returns:
        - a dictionary of elements and corresponding mean coord numbers
    '''
    cn_dict={}
    for el in structures[0].composition.elements:
        cn_dict[el.name]=0.0
    count = 0
    for t in range(len(structures)):
        if t%freq != 0:
            continue
        count += 1
        vor = structure_analyzer.VoronoiCoordFinder(structures[t])
        for atom in range(len(structures[0])):
            CN = vor.get_coordination_number(atom)
            cn_dict[structures[t][atom].species_string] += CN
    el_dict = structures[0].composition.as_dict()
    for el in cn_dict:
        cn_dict[el] = cn_dict[el]/el_dict[el]/count
    return cn_dict


def get_sample_structures(xdatcar_path, n=10, steps_skip_first=1000):
    """
    Helper method to extract n unique structures from an MD output
    Args:
        xdatcar_path: (str) path
    Returns:
        A list of Structure objects
    """
    input_structures = Xdatcar(xdatcar_path).structures
    output_structures = []
    t = len(input_structures)-steps_skip_first
    for i in range(n):
        output_structures.append(input_structures[::-1][i*t//n])
    return output_structures