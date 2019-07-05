from pymatgen.core.structure import Structure, Molecule
from pymatgen.io.xyz import XYZ
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from pymatgen.io.vasp.outputs import Xdatcar


def get_entropies(file='entropy', total=False):
    # Group; Entropies in eV/K [trans, rot, vib]; Total entropy in eV/K; Total entropy in J/K

    with open(file) as f:
        lines = f.readlines()
    ind = []
    data = []
    for i, line in enumerate(lines):
        if "#" in line:
            ind.append(i)

    for i in range(ind[0]+1, ind[1]):
        data.append([float(x) for x in lines[i].split()])

    data = []
    for i in range(ind[0]+1, ind[1]-1):
        data.append([float(x) for x in lines[i].split()])

    df = pd.DataFrame.from_records(data=data, columns=['Group', 'S_trans', 'S_rot', 'S_vib', 'S', 'S_SI'])

    df_tot = pd.DataFrame({'S_trans': np.sum(df['S_trans']), 'S_rot': np.sum(df['S_rot']),
                           "S_vib": np.sum(df['S_vib']), 'S': np.sum(df['S']), 'S_SI': np.sum(df['S_SI'])},
                          index=[0])

    df.to_csv('entropy.csv')
    df_tot.to_csv('total_entropy.csv')

    if total:
        return df_tot
    else:
        return df


def get_dos_sg(file='dos_sg', masses_file='masses'):

    atoms = {}

    with open(masses_file) as f:
        masses = f.readlines()
        for line in masses:
            atoms[line.split()[0]] = []

    with open(file) as f:
        lines = f.readlines()
    ind = []
    for i, line in enumerate(lines):
        if "#" in line:
            ind.append(i)
    ind.append(len(lines))

    j = 1
    for k,v in atoms.items():
        for i in range(ind[j-1]+1, ind[j]-1):
            v.append([float(x) for x in lines[i].split()])
        j += 1

    dfs = {}
    for k,v in atoms.items():
        dfs[k] = pd.DataFrame.from_records(data=v, columns=['Freq', 'DOS_trans', 'DOS_rot', 'DOS_vib'])

    return dfs

def write_traj(structures):
    molecules = []

    for struc in structures:
        molecules.append(Molecule(struc.species, coords=[s.coords for s in struc.sites]))

    XYZ(mol=molecules).write_file('traj.xyz')


def write_masses(structure):
    if isinstance(structure, list):
        structure = structure[0]

    elements = Structure.composition.elements
    masslist = [[e.symbol, e.data['Atomic mass']] for e in elements]

    with open('masses', 'w') as f:
        for item in masslist:
            f.write("{} {}\n".format(item[0], item[1]))


def write_input(points, total_time, lattice, temp):

    points = "# number of points in trajectory\npoints = {}\n".format(points)
    time = "# total time (ps)\ntau = {}\n".format(total_time)
    size = "# Size of the box in nm\ncell = {} {} {}\n".format(lattice.a/10, lattice.b/10, lattice.c/10)
    temp = "# Temperature in K\ntemperature = {}\n# Trajectory info\n".format(temp)
    format = "format = xyz\n"
    vel = "# Estimate velocities from positions\nestimate_velocities =.true."

    string = points+time+size+temp+format+vel
    with open('input', 'w') as f:
        f.writelines(string.strip())


def write_groups(structure):
    num_atoms = structure.num_sites
    num_groups = num_atoms

    with open('groups', 'w') as f:
        f.writelines(str(num_atoms)+' '+str(num_groups)+'\n')

        for i in range(num_groups):
            f.writelines('1 120'+'\n')
            f.writelines(str(i+1)+'\n')


def write_supergroups(structures):

    elements = structures[0].composition.as_dict()

    with open('supergroups', 'w') as f:
        i = 1
        for k,v in elements.items():
            f.writelines(str(i)+'-'+str(i+int(v)-1)+'\n')
            i += int(v)


def write_all(structures, points, total_time, lattice, temp):
    write_traj(structures)
    write_supergroups(structures)
    write_groups(structures[0])
    write_masses(structures)
    write_input(points, total_time, lattice, temp)