import os
import numpy as np

from fireworks import FireTaskBase, Firework, FWAction, explicit_serialize

from atomate.utils.utils import get_logger, env_chk, load_class
from atomate.vasp.firetasks.write_inputs import ModifyIncar
from atomate.vasp.firetasks.glue_tasks import CopyVaspOutputs, get_calc_loc
from atomate.vasp.firetasks.run_calc import RunVaspCustodian, RunVaspFake

from mpmorph.runners.rescale_volume import RescaleVolume
from mpmorph.analysis.md_data import MD_Data

__authors__ = 'Nicholas Winner, Muratahan Aykol'

logger = get_logger(__name__)


@explicit_serialize
class SpawnMDFWTask(FireTaskBase):
    """
    The spawn task dictating volume searching. Given a VASP run, decide if the MD simulation has operated at the
    equilibrium density. i.e. see if the ensemble average pressure is approximately zero. If it is close enough to
    the equilibrium density, either complete the current sequence of calculations, or alternatively you can take
    the equilibrium density and proceed to a production run (via ProductionSpawnTask).

    Required parameters:

        pressure_threshold: (int or float) The window of acceptable pressures after which to stop.
            i.e: for pressure threshold = 5, if the pressure was -2.4, we'd consider it complete

        max_rescales: (int) The maximum number of times to rescale the volume before canceling the sequence
        vasp_cmd: (str) command to run vasp
        wall_time: (int) wall time, in seconds
        db_file: (str) path to file with database credentials
        spawn_count: (int) The number of spawns/rescales that have been executed prior to the current one

    Optional parameters:

        averaging_fraction: (float) given a VASP run of X steps, get the ensemble averaged pressure only for
            steps X*(1-averaging_fraction) to the end
            Default: .5

        production: (dict) The production run dictionary. If this quantity is not set, then the sequence of
            runs will stop when the density is found. If it is set, it dictates how the production
            run will be carried out. (see SingleMultiSpawn)

        p_v: (list) a list of tuples of the form [(pressure1, volume1),(pressure2, volume2),...] that is updated
            with each successive iteration of the spawn task. It is used to calcualte equilibrium volume via
            equation of state fitting once at least 3 datapoints exist.

    """
    required_params = ["pressure_threshold", "max_rescales", "vasp_cmd", "wall_time",
                       "db_file", "spawn_count"]
    optional_params = ["averaging_fraction", 'production', 'p_v']

    def run_task(self, fw_spec):
        vasp_cmd = self.get("vasp_cmd")
        wall_time = self.get("wall_time")
        db_file = self.get("db_file")
        max_rescales = self.get("max_rescales")
        pressure_threshold = self.get("pressure_threshold")
        spawn_count = self.get("spawn_count")
        production = self.get('production', {})
        p_v = self.get('p_v', [])

        if spawn_count > max_rescales:
            logger.info("WARNING: The max number of rescales has been reached... stopping density search.")
            return FWAction(defuse_workflow=True)

        name = ("spawnrun"+str(spawn_count))

        current_dir = os.getcwd()

        averaging_fraction = self.get("averaging_fraction", 0.5)
        data = MD_Data()
        data.parse_md_data(current_dir)
        pressure = data.get_md_data()['pressure']
        v = data.get_volume
        p = np.mean(pressure[int(averaging_fraction*(len(pressure)-1)):])
        p_v.append([p, v])

        logger.info("LOGGER: Current pressure is {}".format(p))
        if np.fabs(p) > pressure_threshold:
            logger.info("LOGGER: Pressure is outside of threshold: Spawning another MD Task")
            t = []
            t.append(CopyVaspOutputs(calc_dir=current_dir, contcar_to_poscar=True))
            t.append(RescaleVolumeTask(initial_pressure=p*1000.0, initial_temperature=1, p_v=p_v))
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
            new_fw = Firework(t, name=name, spec={'p_v': p_v})
            return FWAction(stored_data={'pressure': p}, detours=[new_fw])

        elif production:
            logger.info("LOGGER: Pressure is within the threshold: Moving to production runs...")
            t = []
            t.append(CopyVaspOutputs(calc_dir=current_dir, contcar_to_poscar=True))
            t.append(RunVaspCustodian(vasp_cmd=vasp_cmd, gamma_vasp_cmd=">>vasp_gam<<",
                                      handler_group="md", wall_time=wall_time))
            t.append(ProductionSpawnTask(vasp_cmd=vasp_cmd, wall_time=wall_time, db_file=db_file, spawn_count=1,
                                         production=production))
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
        incar_update = production.get('incar_update', None)

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
                                         db_file=db_file,
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
    optional_params = ["target_pressure", "target_temperature", "target_pressure", "alpha", "beta", "p_v"]

    def run_task(self, fw_spec):
        # Initialize volume correction object with last structure from last_run
        initial_temperature = self["initial_temperature"]
        initial_pressure = self["initial_pressure"]
        target_temperature = self.get("target_temperature", initial_temperature)
        target_pressure = self.get("target_pressure", 0.0)
        alpha = self.get("alpha", 10e-6)
        beta = self.get("beta", 10e-7)
        p_v = self.get('p_v', [])

        corr_vol = RescaleVolume.of_poscar(poscar_path="./POSCAR", initial_temperature=initial_temperature,
                                           initial_pressure=initial_pressure,
                                           target_pressure=target_pressure,
                                           target_temperature=target_temperature, alpha=alpha, beta=beta)
        # Rescale volume based on temperature difference first. Const T will return no volume change:
        corr_vol.by_thermo(scale='temperature')
        # TO DB ("Rescaled volume due to delta T: ", corr_vol.structure.volume)
        # Rescale volume based on pressure difference:
        if len(p_v) > 2:
            corr_vol.by_EOS(p_v=p_v)
        else:
            corr_vol.by_thermo(scale='pressure')
        # TO DB ("Rescaled volume due to delta P: ", corr_vol.structure.volume)
        corr_vol.poscar.write_file("./POSCAR")
        # Pass the rescaled volume to Poscar
        return FWAction(stored_data=corr_vol.structure.as_dict())