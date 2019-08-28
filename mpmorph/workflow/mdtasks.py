import os
import numpy as np
import random
import shutil
import json
import gzip
from monty.json import MontyEncoder, MontyDecoder
import subprocess

from pymatgen.core.periodic_table import Specie
from pymatgen.core.structure import Structure, Molecule
from pymatgen.io.lammps.data import LammpsData
from pymatgen.io.vasp.outputs import Xdatcar, Vasprun
from pymatgen.io.vasp.sets import MITMDSet
from pymatgen.io.xyz import XYZ

from fireworks import FireTaskBase, Firework, FWAction, explicit_serialize

from atomate.utils.utils import get_logger, env_chk, load_class
from atomate.common.firetasks.glue_tasks import PassCalcLocs
from atomate.vasp.fireworks.core import MDFW
from atomate.vasp.firetasks.write_inputs import ModifyIncar
from atomate.vasp.firetasks.glue_tasks import CopyVaspOutputs, get_calc_loc
from atomate.vasp.firetasks.run_calc import RunVaspCustodian, RunVaspFake
from atomate.vasp.database import VaspCalcDb

from mpmorph.runners.amorphous_maker import AmorphousMaker
from mpmorph.runners.rescale_volume import RescaleVolume
from mpmorph.analysis.md_data import MD_Data
from mpmorph.analysis.structural_analysis import RadialDistributionFunction
from mpmorph.analysis.transport import VDOS, Viscosity, Diffusion

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
                       "db_file", "spawn_count"]
    optional_params = ["averaging_fraction", 'production']

    def run_task(self, fw_spec):
        vasp_cmd = self["vasp_cmd"]
        wall_time = self["wall_time"]
        db_file = self["db_file"]
        max_rescales = self["max_rescales"]
        pressure_threshold = self["pressure_threshold"]
        spawn_count = self["spawn_count"]
        production = self['production'] or False

        if spawn_count > max_rescales:
            logger.info("WARNING: The max number of rescales has been reached... stopping density search.")
            return FWAction(defuse_workflow=True)

        name = ("spawnrun"+str(spawn_count))

        current_dir = os.getcwd()

        averaging_fraction = self.get("averaging_fraction", 0.5)
        data = MD_Data()
        data.parse_md_data(current_dir)
        pressure = data.get_md_data()['pressure']
        p = np.mean(pressure[int(averaging_fraction*(len(pressure)-1)):])

        logger.info("LOGGER: Current pressure is {}".format(p))
        if np.fabs(p) > pressure_threshold:
            logger.info("LOGGER: Pressure is outside of threshold: Spawning another MD Task")
            t = []
            t.append(CopyVaspOutputs(calc_dir=current_dir, contcar_to_poscar=True))
            t.append(RescaleVolumeTask(initial_pressure=p*1000.0, initial_temperature=1))
            t.append(RunVaspCustodian(vasp_cmd=vasp_cmd, gamma_vasp_cmd=">>vasp_gam<<",
                                      handler_group="md", wall_time=wall_time))

            t.append(SpawnMDFWTask(pressure_threshold=pressure_threshold,
                                   max_rescales=max_rescales,
                                   wall_time=wall_time,
                                   vasp_cmd=vasp_cmd,
                                   db_file=db_file,
                                   spawn_count=spawn_count+1,
                                   averaging_fraction=averaging_fraction,
                                   production=production))
            new_fw = Firework(t, name=name)
            return FWAction(stored_data={'pressure': p}, detours=[new_fw])

        elif production:
            logger.info("LOGGER: Pressure is within the threshold: Moving to production runs...")
            t = []
            t.append(CopyVaspOutputs(calc_dir=current_dir, contcar_to_poscar=True))
            t.append(RunVaspCustodian(vasp_cmd=vasp_cmd, gamma_vasp_cmd=">>vasp_gam<<",
                                      handler_group="md", wall_time=wall_time))
            t.append(ProductionSpawnTask(vasp_cmd=vasp_cmd, wall_time=wall_time, db_file=db_file,
                                         spawn_count=0, production=production))
            production_fw = Firework(t, name="ProductionRun1")
            return FWAction(stored_data={'pressure': p, 'density_calculated': True}, detours=[production_fw])

        else:
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

    required_params = ['vasp_cmd', 'wall_time', 'spawn_count', 'production']
    optional_params = ['checkpoint_dirs', 'db_file', 'modify_incar']

    def run_task(self, fw_spec):

        prev_checkpoint_dirs = fw_spec.get("checkpoint_dirs", [])  # If this is the first spawn, have no prev dirs
        prev_checkpoint_dirs.append(os.getcwd())  # add the current directory to the list of checkpoints

        vasp_cmd = self["vasp_cmd"]
        wall_time = self["wall_time"]
        db_file = self.get("db_file", None)
        spawn_count = self["spawn_count"]
        production = self['production']
        num_checkpoints = production.get('num_checkpoints',1)
        incar_update    = production.get('incar_update', None)

        if spawn_count > num_checkpoints:
            logger.info("LOGGER: Production run completed. Took {} spawns total".format(spawn_count))
            return FWAction(stored_data={'production_run_completed': True})

        else:
            name = ("ProductionRun" + str(abs(spawn_count)))

            logger.info("LOGGER: Starting spawn {} of production run".format(spawn_count))

            t = []

            t.append(CopyVaspOutputs(calc_dir=os.getcwd(), contcar_to_poscar=True))

            if incar_update:
                t.append(ModifyIncar(incar_update=incar_update))

            t.append(RunVaspCustodian(vasp_cmd=vasp_cmd, gamma_vasp_cmd=">>vasp_gam<<",
                                      handler_group="md", wall_time=wall_time))
            t.append(ProductionSpawnTask(wall_time=wall_time,
                                         vasp_cmd=vasp_cmd,
                                         db_file=None,
                                         spawn_count=spawn_count + 1,
                                         production=production))
            new_fw = Firework(t, name=name, spec={'checkpoint_dirs': prev_checkpoint_dirs})

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

    required_params = ['time_step']
    optional_params = ['checkpoint_dirs', 'output_file']

    def run_task(self, fw_spec):

        time_step = fw_spec.get('time_step', 2)
        checkpoint_dirs = fw_spec.get('checkpoint_dirs', False)
        output_file = fw_spec.get('output_file', 'diffusion.json')

        calc_dir = get_calc_loc(True, fw_spec["calc_locs"])["path"]
        calc_loc = os.path.join(calc_dir, 'XDATCAR.gz')

        if checkpoint_dirs:
            logger.info("LOGGER: Assimilating checkpoint structures")
            structures = []
            for d in checkpoint_dirs:
                structures.extend(Vasprun(os.path.join(d, 'vasprun.xml.gz')).structures)
        else:
            structures = Xdatcar(calc_loc).structures

        db_dict = {}
        db_dict.update({'density': float(structures[0].density)})
        db_dict.update(structures[0].composition.to_data_dict)

        logger.info("LOGGER: Calculating the diffusion coefficients...")
        diffusion = Diffusion(structures, t_step=time_step, l_lim=50, skip_first=250, block_l=1000, ci=0.95)
        D = {'msd': {}, 'vdos': {}}
        for s in structures[0].types_of_specie:
            D['msd'][s.symbol] = diffusion.getD(s.symbol)
            D['vdos'] = vdos_diff

        vdos = VDOS(structures)
        vdos_dat = vdos.calc_vdos_spectrum(time_step=time_step * ionic_step_skip)
        vdos_diff = vdos.calc_diffusion_coefficient(time_step=time_step * ionic_step_skip)

        db_dict.update({'diffusion': D})

        with open(os.path.join(output_file)) as f:
            json = json.dumps(db_dict)
            f.write(json)

        return FWAction()


@explicit_serialize
class VaspMdToStructuralAnalysis(FireTaskBase):

    required_params = []
    optional_params = ['checkpoint_dirs']

    def run_task(self, fw_spec):

        checkpoint_dirs = fw_spec.get('checkpoint_dirs', False)

        calc_dir = get_calc_loc(True, fw_spec["calc_locs"])["path"]
        calc_loc = os.path.join(calc_dir, 'XDATCAR.gz')

        if checkpoint_dirs:
            logger.info("LOGGER: Assimilating checkpoint structures")
            structures = []
            for d in checkpoint_dirs:
                structures.extend(Vasprun(os.path.join(d, 'vasprun.xml.gz')).structures)
        else:
            structures = Xdatcar(calc_loc).structures

        db_dict = {}
        db_dict.update({'density': float(structures[0].density)})
        db_dict.update(structures[0].composition.to_data_dict)

        logger.info("LOGGER: Calculating radial distribution functions...")
        rdf = RadialDistributionFunction(structures=structures)
        rdf.get_radial_distribution_functions(nproc=4)
        db_dict.update({'rdf': rdf.get_rdf_db_dict()})

        return FWAction()


@explicit_serialize
class ParseCheckpointsTask(FireTaskBase):

    required_params = ['checkpoint_dirs']
    optional_params = []

    def run_task(self, fw_spec):

        checkpoint_dirs = self.get('checkpoint_dirs')

        ionic_steps = []
        for d in checkpoint_dirs:
            ionic_steps.extend(Vasprun(os.path.join(d, "vasprun.xml.gz")).ionic_steps)

        with gzip.open('ionic_steps.json.gz', 'wt', encoding="ascii") as zipfile:
            json.dump(ionic_steps, zipfile, cls=MontyEncoder)


@explicit_serialize
class RunDospt(FireTaskBase):
    """
    Execute a command directly.

    Required params:
        cmd (str): the name of the full executable to run. Supports env_chk.
    Optional params:
        expand_vars (str): Set to true to expand variable names in the cmd.
    """

    required_params = []
    optional_params = []

    def run_task(self, fw_spec):
        cmd = "dospt"
        logger.info("Running command: {}".format(cmd))
        return_code = subprocess.call(cmd, shell=True)
        logger.info("Command {} finished running with returncode: {}".format(cmd, return_code))


@explicit_serialize
class MDAnalysisTask(FireTaskBase):
    required_params = []
    optional_params = ['time_step', 'get_rdf', 'get_diffusion', 'get_viscosity',
                       'get_vdos', 'get_run_data', 'checkpoint_dirs', 'analysis_spec']

    def run_task(self, fw_spec):

        get_rdf = self.get('get_rdf') or True
        get_diffusion = self.get('get_diffusion') or True
        get_viscosity = self.get('get_viscosity') or True
        get_vdos = self.get('get_vdos') or True
        get_run_data = self.get('get_run_data') or True
        time_step = self.get('time_step') or 2
        checkpoint_dirs = fw_spec.get('checkpoint_dirs', False)

        calc_dir = get_calc_loc(True, fw_spec["calc_locs"])["path"]
        calc_loc = os.path.join(calc_dir, 'XDATCAR.gz')

        ionic_step_skip = self.get('ionic_step_skip') or 1
        ionic_step_offset = self.get('ionic_step_offset') or 0

        analysis_spec = self.get('analysis_spec') or {}

        if checkpoint_dirs:
            logger.info("LOGGER: Assimilating checkpoint structures")
            ionic_steps = []
            structures = []
            for d in checkpoint_dirs:
                ionic_steps.extend(Vasprun(os.path.join(d,"vasprun.xml.gz")).ionic_steps)

                structures.extend(Vasprun(os.path.join(d, 'vasprun.xml.gz'),
                                          ionic_step_skip=ionic_step_skip,
                                          ionic_step_offset=ionic_step_offset).structures)

        else:
            structures = Xdatcar(calc_loc).structures

        #write a trajectory file for Dospt
        molecules = []
        for struc in structures:
            molecules.append(Molecule(species=struc.species, coords=[s.coords for s in struc.sites]))
        XYZ(mol=molecules).write_file('traj.xyz')

        db_dict = {}
        db_dict.update({'density': float(structures[0].density)})
        db_dict.update(structures[0].composition.to_data_dict)
        db_dict.update({'checkpoint_dirs': checkpoint_dirs})

        if get_rdf:
            logger.info("LOGGER: Calculating radial distribution functions...")
            rdf = RadialDistributionFunction(structures=structures)
            rdf_dat = rdf.get_radial_distribution_functions(nproc=4)
            db_dict.update({'rdf': rdf.get_rdf_db_dict()})
            del rdf
            del rdf_dat

        if get_vdos:
            logger.info("LOGGER: Calculating vibrational density of states...")
            vdos = VDOS(structures)
            vdos_dat = vdos.calc_vdos_spectrum(time_step=time_step*ionic_step_skip)
            vdos_diff = vdos.calc_diffusion_coefficient(time_step=time_step*ionic_step_skip)
            db_dict.update({'vdos': vdos_dat})
            del vdos
            del vdos_dat

        if get_diffusion:
            logger.info("LOGGER: Calculating the diffusion coefficients...")
            diffusion = Diffusion(structures, t_step=time_step, l_lim=50, skip_first=250, block_l=1000, ci=0.95)
            D = {'msd':{}, 'vdos':{}}
            for s in structures[0].types_of_specie:
                D['msd'][s.symbol] = diffusion.getD(s.symbol)
            if vdos_diff:
                D['vdos'] = vdos_diff
            db_dict.update({'diffusion': D})
            del D

        if get_viscosity:
            logger.info("LOGGER: Calculating the viscosity...")
            viscosities = []
            if checkpoint_dirs:
                for dir in checkpoint_dirs:
                    visc = Viscosity(dir).calc_viscosity()
                    viscosities.append(visc['viscosity'])
            viscosity_dat = {'viscosity': np.mean(viscosities), 'StdDev': np.std(viscosities)}
            db_dict.update({'viscosity': viscosity_dat})
            del viscosity_dat

        if get_run_data:
            if checkpoint_dirs:
                logger.info("LOGGER: Assimilating run stats...")
                data = MD_Data()
                for directory in checkpoint_dirs:
                    data.parse_md_data(directory)
                md_stats = data.get_md_stats()
            else:
                logger.info("LOGGER: Getting run stats...")
                data = MD_Data()
                data.parse_md_data(calc_dir)
                md_stats = data.get_md_stats()
            db_dict.update({'md_data': md_stats})

        if analysis_spec:
            logger.info("LOGGER: Adding user-specified data...")
            db_dict.update(analysis_spec)

        logger.info("LOGGER: Pushing data to database collection...")
        db_file = env_chk(">>db_file<<", fw_spec)
        db = VaspCalcDb.from_db_file(db_file, admin=True)
        db.collection = db.db["md_data"]
        db.collection.insert_one(db_dict)

        return FWAction()


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
class LammpsToVaspMD(FireTaskBase):
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
class WriteVaspFromLammpsAndIOSet(FireTaskBase):

    required_params = ["vasp_input_set", "structure_loc"]
    optional_params = ["vasp_input_params", 'atom_style']

    def run_task(self, fw_spec):
        logger.info("PARSING final lammps positions to VASP.")

        data = LammpsData.from_file(self['structure_loc'], atom_style=self.get('atom_style', 'full'), sort_id=True)

        struc = data.structure
        structure = Structure(lattice=struc.lattice, species=[s.specie for s in struc.sites],
                              coords=[s.coords for s in struc.sites], coords_are_cartesian=True)

        vis_cls = load_class("pymatgen.io.vasp.sets", self["vasp_input_set"])
        vis = vis_cls(structure, **self.get("vasp_input_params", {}))
        vis.write_input(".")


@explicit_serialize
class MultiSpawn(FireTaskBase):

    required_params = ["spawn_type", "spawn_number"]
    optional_params = ["wall_time", "vasp_cmd", "num_checkpoints"]

    def run_task(self, fw_spec):

        spawn_type = self.get('spawn_type')
        spawn_number = self.get('spawn_number')

        wall_time = self.get('wall_time', 19200)
        vasp_cmd = self.get('vasp_cmd', ">>vasp_cmd<<")
        num_checkpoints = self.get('num_checkpoints', 1)

        fws = []
        for i in range(spawn_number):
            t = []
            t.append(ProductionSpawnTask(wall_time=wall_time,
                                         vasp_cmd=vasp_cmd,
                                         db_file=None,
                                         spawn_count=0,
                                         production=num_checkpoints))
            fws.append(Firework(t, name="Multispawn_{}_FW".format(i+1)))

        return FWAction(detours=[fws])


@explicit_serialize
class SpawnFromTrajectory(FireTaskBase):

    required_params = ['wf_name', 'trajectory']
    optional_params = []

    def run_task(self, fw_spec):

        logger.info("Spawning WFs from completed MD Trajectory...")

        new_fw = Firework(t, name=name, spec={'checkpoint_dirs': prev_checkpoint_dirs})

        return FWAction(stored_data={'production_run_completed': False},
                        update_spec={'checkpoint_dirs': prev_checkpoint_dirs}, detours=[new_fw])
