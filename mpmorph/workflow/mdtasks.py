from fireworks import FireTaskBase, Firework
from mpmorph.runners.amorphous_maker import AmorphousMaker
from mpmorph.runners.rescale_volume import RescaleVolume
from mpmorph.analysis.md_data import parse_pressure, get_MD_data, get_MD_stats, plot_md_data
from mpmorph.analysis.structural_analysis import RadialDistributionFunction
from mpmorph.analysis.transport import VDOS, Viscosity, Diffusion
from atomate.vasp.firetasks.glue_tasks import CopyVaspOutputs, get_calc_loc
from atomate.vasp.firetasks.run_calc import RunVaspCustodian
from atomate.common.firetasks.glue_tasks import PassCalcLocs
from atomate.vasp.firetasks.write_inputs import WriteVaspFromIOSet
from fireworks.user_objects.firetasks.script_task import ScriptTask
from pymatgen.io.vasp.outputs import Xdatcar, Vasprun
from pymatgen.core.structure import Structure
import shutil
import numpy as np
from atomate.vasp.firetasks.parse_outputs import VaspToDbTask
import os
import json


import random

from pymatgen.io.lammps.data import LammpsData
from pymatgen.io.vasp.sets import MITMDSet
from pymatgen.core.periodic_table import Specie

from fireworks.core.firework import FiretaskBase, FWAction
from fireworks import explicit_serialize

from atomate.utils.utils import get_logger
from atomate.vasp.fireworks.core import MDFW

__authors__ = 'Nicholas Winner, Muratahan Aykol'

logger = get_logger(__name__)

#TODO: 2. Add option to lead to a production run of specified length after density is found
#TODO: 3. Switch to MPRelax Parameters in MD
#TODO: 4. Database insertion?
#TODO: 5. Parser tasks

@explicit_serialize
class AmorphousMakerTask(FireTaskBase):
    """
    Create a constrained-random packed structure from composition and box dimensions.
    Required params:
        composition: (dict) a dict of target composition with integer atom numbers
                        e.g. {"V":22, "Li":10, "O":75, "B":10}
        box_scale: (float) all lattice vectors are multiplied with this scalar value.
                        e.g. edge length of a cubic simulation box.
    Optional params:
        tol (float): tolerance factor for how close the atoms can get (angstroms).
                        e.g. tol = 2.0 angstroms
        packmol_path (str): path to the packmol executable. Defaults to "packmol"
        clean (bool): whether the intermedite files generated are deleted. Defaults to True.
    """

    required_params = ["composition", "box_scale"]
    optional_params = ["packmol_path", "clean", "tol"]

    def run_task(self, fw_spec):
        glass = AmorphousMaker(self.get("composition"), self.get("box_scale"), self.get("tol", 2.0),
                               packmol_path=self.get("packmol_path", "packmol"),
                               clean=self.get("clean", True))
        structure = glass.random_packed_structure.as_dict()
        return FWAction(stored_data=structure)


@explicit_serialize
class GetPressureTask(FireTaskBase):
    required_params = ["outcar_path"]
    optional_params = ["averaging_fraction"]

    def run_task(self, fw_spec):
        p = parse_pressure(self["outcar_path"], self.get("averaging_fraction", 0.5))
        if fw_spec['avg_pres']:
            fw_spec['avg_pres'].append(p[0]*1000)
        else:
            fw_spec['avg_pres'] = [p[0]*1000]
        return FWAction()


@explicit_serialize
class SpawnMDFWTask(FireTaskBase):
    """
    Decides if a new MD calculation should be spawned or if density is found. If so, spawns a new calculation.
    """
    required_params = ["pressure_threshold", "max_rescales", "vasp_cmd", "wall_time",
                       "db_file", "spawn_count", "copy_calcs", "calc_home"]
    optional_params = ["averaging_fraction", 'production']

    def run_task(self, fw_spec):
        calc_dir = get_calc_loc(True, fw_spec['calc_locs'])['path'] or os.getcwd()
        vasp_cmd = self["vasp_cmd"]
        wall_time = self["wall_time"]
        db_file = self["db_file"]
        max_rescales = self["max_rescales"]
        pressure_threshold = self["pressure_threshold"]
        spawn_count = self["spawn_count"]
        calc_home = self["calc_home"]
        copy_calcs = self["copy_calcs"]
        production = self['production'] or False

        if spawn_count > max_rescales:
            logger.info("WARNING: The max number of rescales has been reached... stopping density search.")
            return FWAction(defuse_workflow=True)

        name = ("spawnrun"+str(spawn_count))

        averaging_fraction = self.get("averaging_fraction", 0.5)
        pressure = get_MD_data(calc_dir)['pressure']['val']
        p = np.mean(pressure[int(averaging_fraction*(len(pressure)-1)):])

        logger.info("LOGGER: Current pressure is {}".format(p))

        if np.fabs(p) > pressure_threshold:
            logger.info("LOGGER: Pressure is outside of threshold: Spawning another MD Task")
            t = []
            # Copy the VASP outputs from previous run. Very first run get its from the initial MDWF which
            # uses PassCalcLocs. For the rest we just specify the previous dir.
            if spawn_count==0:
                t.append(CopyVaspOutputs(calc_dir=calc_dir, contcar_to_poscar=False))
            else:
                t.append(CopyVaspOutputs(calc_dir=calc_dir, contcar_to_poscar=True))

            t.append(RescaleVolumeTask(initial_pressure=p*1000.0, initial_temperature=1))
            t.append(RunVaspCustodian(vasp_cmd=vasp_cmd, gamma_vasp_cmd=">>vasp_gam<<",
                                      handler_group="md", wall_time=wall_time))
            # Will implement the database insertion
            # t.append(VaspToDbTask(db_file=db_file,
            #                       additional_fields={"task_label": "density_adjustment"}))
            if copy_calcs:
                t.append(CopyCalsHome(calc_home=calc_home, run_name=name))
            t.append(SpawnMDFWTask(pressure_threshold=pressure_threshold,
                                   max_rescales=max_rescales,
                                   wall_time=wall_time,
                                   vasp_cmd=vasp_cmd,
                                   db_file=db_file,
                                   spawn_count=spawn_count+1,
                                   copy_calcs=copy_calcs,
                                   calc_home=calc_home,
                                   averaging_fraction=averaging_fraction,
                                   production=production))
            t.append(PassCalcLocs(name=name))
            new_fw = Firework(t, name=name)
            return FWAction(stored_data={'pressure': p}, detours=[new_fw])

        else:
            if production:
                logger.info("LOGGER: Pressure is within the threshold: Moving to production runs...")
                t = []
                t.append(ProductionSpawnTask(vasp_cmd=vasp_cmd, wall_time=wall_time, db_file=db_file,
                                             spawn_count=0, production=production))
                new_fw = Firework(t)
                return FWAction(stored_data={'pressure': p, 'density_calculated': True}, detours=[new_fw])
            else:
                logger.info("LOGGER: Pressure is within the threshold: Stopping Spawns.")
            return FWAction(stored_data={'pressure': p, 'density_calculated': True})


@explicit_serialize
class ProductionSpawnTask(FireTaskBase):

    """
    A task for spawning MD calculations in production runs. Only considers whether or not the number of
    production tasks is reached for the stop criteria at the moment. It also stores where all the
    checkpoints of a production run are located. This list of directories is used for assembling the
    checkpoints into a single analysis task.

    Required Params:
        vasp_cmd (str): command to run vasp
        wall_time (int): wall time for each checkpoint in seconds
        db_file (str): path to file with db credentials
        spawn_count (int): The number of MD checkpoints that have been spawned. Used to track when production
                            is completed.
        production (int): The number of MD checkpoints in total for this production run.

    Optional Params:
        checkpoint_dirs (list): A list of all directories where checkpoints exist for this production
                                MD run. Is listed as optional because the first spawn will not have
                                any checkpoint directories



    """

    required_params = ['vasp_cmd', 'wall_time', 'db_file', 'spawn_count', 'production']
    optional_params = ['checkpoint_dirs']

    def run_task(self, fw_spec):

        prev_checkpoint_dirs = fw_spec.get("checkpoint_dirs", [])  # If this is the first spawn, have no prev dirs
        prev_checkpoint_dirs.append(os.getcwd())  # add the current directory to the list of checkpoints

        calc_dir = get_calc_loc(True, fw_spec['calc_locs'])['path'] or os.getcwd()
        vasp_cmd = self["vasp_cmd"]
        wall_time = self["wall_time"]
        db_file = self["db_file"]
        spawn_count = self["spawn_count"]
        production = self['production']

        if spawn_count > production:
            logger.info("LOGGER: Production run completed. Took {} spawns total")
            return FWAction(stored_data={'production_run_completed': True})

        else:
            name = ("ProductionRun" + str(abs(spawn_count)))

            logger.info("LOGGER: Starting spawn {} of production run".format(spawn_count))

            t = []

            t.append(CopyVaspOutputs(calc_dir=calc_dir, contcar_to_poscar=True))

            t.append(RunVaspCustodian(vasp_cmd=vasp_cmd, gamma_vasp_cmd=">>vasp_gam<<",
                                      handler_group="md", wall_time=wall_time))

            t.append(ProductionSpawnTask(wall_time=wall_time,
                                         vasp_cmd=vasp_cmd,
                                         db_file=db_file,
                                         spawn_count=spawn_count + 1,
                                         production=production))
            t.append(PassCalcLocs(name=name))
            new_fw = Firework(t, name=name)

            return FWAction(stored_data={'production_run_completed': False},
                            update_spec={'checkpoint_dirs': prev_checkpoint_dirs}, detours=[new_fw])


@explicit_serialize
class RescaleVolumeTask(FireTaskBase):
    """
    Volume rescaling
    """
    required_params = ["initial_temperature", "initial_pressure"]
    optional_params = ["target_pressure", "target_temperature", "target_pressure", "alpha", "beta"]

    def run_task(self, fw_spec):
        # Initialize volume correction object with last structure from last_run
        initial_temperature = self["initial_temperature"]
        initial_pressure = self["initial_pressure"]
        target_temperature = self.get("target_temperature", initial_temperature)
        target_pressure = self.get("target_pressure", 0.0)
        alpha = self.get("alpha", 10e-6)
        beta = self.get("beta", 10e-7)
        corr_vol = RescaleVolume.of_poscar(poscar_path="./POSCAR", initial_temperature=initial_temperature,
                                           initial_pressure=initial_pressure,
                                           target_pressure=target_pressure,
                                           target_temperature=target_temperature, alpha=alpha, beta=beta)
        # Rescale volume based on temperature difference first. Const T will return no volume change:
        corr_vol.by_thermo(scale='temperature')
        # TO DB ("Rescaled volume due to delta T: ", corr_vol.structure.volume)
        # Rescale volume based on pressure difference:
        corr_vol.by_thermo(scale='pressure')
        # TO DB ("Rescaled volume due to delta P: ", corr_vol.structure.volume)
        corr_vol.poscar.write_file("./POSCAR")
        # Pass the rescaled volume to Poscar
        return FWAction(stored_data=corr_vol.structure.as_dict())


@explicit_serialize
class CopyCalsHome(FireTaskBase):
    required_params = ["calc_home","run_name"]
    optional_params = ["files"]

    def run_task(self, fw_spec):
        default_list = ["INCAR", "POSCAR", "CONTCAR", "OUTCAR", "POTCAR", "vasprun.xml", "XDATCAR", "OSZICAR", "DOSCAR"]
        files = self.get("files", default_list)
        calc_home = self["calc_home"]
        run_name = self["run_name"]
        target_dir = os.path.join(calc_home, run_name)
        if not os.path.exists(target_dir):
            os.mkdir(target_dir)
        for f in files:
            try:
                shutil.copy2(f,target_dir)
            except:
                pass
        return FWAction()


@explicit_serialize
class VaspMdToDbTask(FireTaskBase):
    pass


@explicit_serialize
class VaspMdToDiffusion(FireTaskBase):
    pass


@explicit_serialize
class VaspMdToStructuralAnalysis(FireTaskBase):
    pass


@explicit_serialize
class PackToLammps(FireTaskBase):

    required_params = ['box_size']
    optional_params = ['atom_style', 'packmol_file', 'lammps_data']

    def run_task(self, fw_spec):
        calc_dir = os.getcwd()
        box_size = self.get('box_size')
        atom_style = self.get('atom_style') or 'full'
        packmol_file = self.get('packmol_file') or 'packed_mol.xyz'
        lammps_data = self.get('lammps_data', 'lammps.data')
        charges = self.get('charges', None)

        data = LammpsData.from_xyz(os.path.join(calc_dir, packmol_file), box_size,
                                   atom_style=atom_style, charges=charges)
        data.write_file(lammps_data)

        return FWAction()


@explicit_serialize
class LammpsToVaspMD(FiretaskBase):
    _fw_name = "LammpsToVasp"
    required_params = ["atom_style", "start_temp", "end_temp", "nsteps"]
    optional_params = ['time_step', 'vasp_input_set', 'user_kpoints_settings', 'vasp_cmd',
                       'copy_vasp_outputs', 'db_file', 'name', 'parents', 'spawn']

    def run_task(self, fw_spec):
        atom_style = self.get('atom_style')
        start_temp = self.get('start_temp')
        end_temp = self.get('end_temp')
        nsteps = self.get('nsteps')
        spawn = self.get('spawn') or False
        production = self.get('production') or False

        time_step = self.get('time_step') or 1
        vasp_cmd = self.get('vasp_cmd') or ">>vasp_gam<<"
        copy_vasp_outputs = self.get('copy_vasp_outputs') or False
        user_incar_settings = self.get('user_incar_settings') or {}
        user_kpoints_settings = self.get('user_kpoints_settings') or None
        db_file = self.get('db_file') or None
        transmute = self.get('transmute') or None

        pressure_threshold = self.get('pressure_threshold') or 5
        max_rescales = self.get('max_rescales') or 6
        wall_time = self.get('wall_time') or 19200
        copy_calcs = self.get('copy_calcs') or False
        calc_home = self.get('calc_home') or '~'

        logger.info("PARSING \"lammps.final\" to VASP.")
        data = LammpsData.from_file(os.path.join(os.getcwd(), self.get('final_data')),
                                    atom_style=atom_style, sort_id=True)
        struc = data.structure
        structure = Structure(lattice=struc.lattice, species=[s.specie for s in struc.sites],
                              coords=[s.coords for s in struc.sites], coords_are_cartesian=True)

        if transmute:
            sites = structure.sites
            indices = []
            for i, s in enumerate(sites):
                if s.specie.symbol == transmute[0]:
                    indices.append(i)
            index = random.choice(indices)
            structure.replace(index, species=transmute[1],
                              properties={'charge': Specie(transmute[1]).oxi_state})

        vasp_input_set = MITMDSet(structure, start_temp, end_temp, nsteps, time_step,
                                  force_gamma=True, user_incar_settings=user_incar_settings)

        if user_kpoints_settings:
            v = vasp_input_set.as_dict()
            v.update({"user_kpoints_settings": user_kpoints_settings})
            vasp_input_set = vasp_input_set.from_dict(v)

        fw = MDFW(structure, start_temp, end_temp, nsteps, vasp_input_set=vasp_input_set, vasp_cmd=vasp_cmd,
                  copy_vasp_outputs=copy_vasp_outputs, db_file=db_file, name='MDFW', wall_time=wall_time)

        if spawn:
            t = fw.tasks
            t.append(SpawnMDFWTask(pressure_threshold=pressure_threshold, max_rescales=max_rescales,
                       wall_time=wall_time, vasp_cmd=vasp_cmd, db_file=db_file,
                       copy_calcs=copy_calcs, calc_home=calc_home,
                       spawn_count=1, production=production))
            fw = Firework(tasks=t, name='SpawnMDFW')

        return FWAction(detours=fw, stored_data={'LammpsStructure': structure})


@explicit_serialize
class MDAnalysisTask(FireTaskBase):
    required_params = []
    optional_params = ['time_step', 'get_rdf', 'get_diffusion', 'get_viscosity',
                       'get_vdos', 'get_run_data', 'checkpoint_dirs']

    def run_task(self, fw_spec):

        get_rdf = self.get('get_rdf') or True
        get_diffusion = self.get('get_diffusion') or True
        get_viscosity = self.get('get_viscosity') or False
        get_vdos = self.get('get_vdos') or True
        get_run_data = self.get('get_run_data') or False
        time_step = self.get('time_step') or 1
        checkpoint_dirs = self.get('checkpoint_dirs') or False

        calc_dir = get_calc_loc(True, fw_spec["calc_locs"])["path"]
        calc_loc = os.path.join(calc_dir, 'XDATCAR.gz')

        if checkpoint_dirs:
            structures = []
            for d in checkpoint_dirs:
                structures.extend(Vasprun(os.path.join(d, 'vasprun.xml.gz')).structures)
        else:
            structures = Xdatcar(calc_loc).structures

        if get_rdf:
            rdf = RadialDistributionFunction(structures=structures)
            rdf_dat = rdf.get_radial_distribution_functions(nproc=16)
            rdf_plt = rdf.plot_radial_distribution_functions(show=False, save=True)

        if get_vdos:
            vdos = VDOS(structures)
            vdos.calc_vdos_spectrum(time_step=time_step)
            vdos.plot_vdos(show=False, save=True)

        if get_diffusion:
            diffusion = Diffusion(structures, t_step=time_step, l_lim=0, ci=0.95)
            D = {}
            for s in structures[0].types_of_specie:
                D[s] = diffusion.getD(s.symbol)

        if get_viscosity:
            viscosity = Viscosity(calc_dir).calc_viscosity()

        if get_run_data:
            md_data = get_MD_data(calc_dir)
            md_stats = get_MD_stats(md_data)
            plot_md_data(md_data, show=False, save=True)

        return FWAction()



@explicit_serialize
class TransmuteTask(FireTaskBase):

    required_params = ['structure', 'species']
    optional_params = []

    def run_task(self, fw_spec):

        structure = self.get('structure')
        species = self.get('species')

        sites = structure.sites
        indices = []
        for i, s in enumerate(sites):
            if s.specie.symbol == species[0]:
                indices.append(i)
        index = random.choice(indices)
        structure.replace(index, species=species[1],
                          properties={'charge': Specie(species[1]).oxi_state})